import logging
from flask import request, session
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from bson import ObjectId

load_dotenv()

RESEARCH_EVENTS = [
    'quiz_start', 'quiz_complete', 'question_answer', 'video_play',
    'video_pause', 'content_view', 'flashcard_flip', 'flashcard_view',
    'translation', 'concept_hover', 'difficulty_change', 'session_start',
    'session_end', 'experiment_assignment', 'memory_prediction',
    'adaptive_content_shown', 'concept_forgotten', 'graph_view',
    'summary_view', 'knowledge_graph_view', 'model_api_call'
]

class Database:
    def __init__(self):
        self.uri = os.getenv("MONGO_URI")
        self.client = None
        self.db = None
        self.connect()
        self.research_metrics = self.db.research_metrics
        self.users = self.db.users

    def connect(self):
        # FIX: guard so calling connect() twice doesn't open a second MongoClient
        if self.client is not None:
            return
        try:
            self.client = MongoClient(self.uri)
            self.db = self.client.get_database("quiz_app")
            self._create_indexes()
            print("Connected to MongoDB!")
        except Exception as e:
            print(f"Connection failed: {e}")
            raise

    def _create_indexes(self):
        # TTL Indexes
        self.db.temp_content.create_index(
            "created_at",
            expireAfterSeconds=259200  # 3 days
        )

        self.db.learning_activities.create_index([
            ("user_id", 1),
            ("activity_type", 1),
            ("timestamp", -1)
        ])

        self.db.user_progress.create_index([
            ("user_id", 1),
            ("timestamp", -1)
        ])

        self.db.video_progress.create_index(
            [("attempts.timestamp", 1)],
            name="attempt_expiration",
            expireAfterSeconds=86400  # 24 hours
        )

        self.db.user_progress.create_index(
            [("weak_concepts", 1)]
        )

        # Research Metrics Indexes
        self.db.research_metrics.create_index([
            ("event_type", 1),
            ("timestamp", -1)
        ])
        self.db.research_metrics.create_index([
            ("user_id", 1),
            ("event_type", 1)
        ])
        # FIX: added (user_id, event_type, timestamp) compound index — most queries filter all three
        self.db.research_metrics.create_index([
            ("user_id", 1),
            ("event_type", 1),
            ("timestamp", -1)
        ])

        # Content Indexes
        self.db.temp_content.create_index("type")
        self.db.temp_content.create_index([
            ("type", 1),
            ("created_at", -1)
        ])

        # Learning Analytics Indexes
        self.db.learning_activities.create_index([
            ("user_id", 1),
            ("timestamp", -1)
        ])

        self.db.video_progress.create_index([
            ('user_id', 1),
            ('video_id', 1),
            ('attempts.timestamp', -1)
        ])
        self.db.video_progress.create_index([
            ('user_id', 1),
            ('aggregates.mastery_level', 1)
        ])

        self.db.video_progress.create_index([
            ('attempts.status', 1),
            ('attempts.timestamp', 1)
        ])

        # Concept Mastery Indexes
        self.db.concept_mastery.create_index([
            ("user_id", 1),
            ("concept", 1),
            ("score", -1)
        ], name="user_concept_mastery", partialFilterExpression={"score": {"$exists": True}})

        # User Progress Indexes
        self.db.user_progress.create_index([
            ('user_id', 1),
            ('video_id', 1)
        ], name='user_video_progress')

        self.db.video_progress.create_index([
            ('aggregates.mastery_level', 1),
            ('aggregates.last_attempt', -1)
        ], name='mastery_tracking')

        self.db.user_progress.create_index([("user_id", 1)], unique=True)

        # User Management
        self.db.users.create_index("email", unique=True)
        self.db.research_metrics.create_index([("metadata.predicted_score", 1)])
        self.db.research_metrics.create_index([("metadata.actual_score", 1)])

    def store_temp_content(self, content_type, data):
        document = {
            "type": content_type,
            "data": {
                **data,
                "video_title": data.get('video_title', 'Untitled'),
                "short_summary": data.get('short_summary', '')
            },
            "created_at": datetime.utcnow(),
            "is_saved": False
        }
        return self.db.temp_content.insert_one(document)

    def get_temp_content(self, content_id):
        try:
            if isinstance(content_id, str):
                content_id = ObjectId(content_id)
            return self.db.temp_content.find_one({"_id": content_id})
        except Exception as e:
            print(f"Error retrieving content: {e}")
            return None

    def save_content_permanently(self, content_id):
        content = self.db.temp_content.find_one({"_id": content_id})
        if content:
            return self.db.saved_content.insert_one({
                **content,
                "is_saved": True,
                "saved_at": datetime.utcnow()
            })
        return None

    def track_learning_activity(self, user_id, activity_type, metadata):
        try:
            # FIX: request may not be available outside Flask request context
            try:
                user_agent = request.headers.get('User-Agent', 'unknown')
            except RuntimeError:
                user_agent = 'unknown'

            return self.db.learning_activities.insert_one({
                "user_id": user_id,
                "type": activity_type,
                "metadata": {
                    **metadata,
                    "timestamp": datetime.utcnow(),
                    "platform": "web",
                    "user_agent": user_agent
                },
                "performance_metrics": {
                    "score": metadata.get('score'),
                    "total": metadata.get('total'),
                    "accuracy": metadata.get('percentage') / 100 if metadata.get('percentage') else None
                }
            })
        except Exception as e:
            logging.error(f"LEARNING ACTIVITY TRACKING FAILED: {str(e)}")
            logging.error(f"Failed data: {metadata}")
            raise

    def log_research_event(self, event_type, metadata=None):
        if event_type not in RESEARCH_EVENTS:
            raise ValueError(f"Invalid event type. Allowed: {RESEARCH_EVENTS}")

        return self.db.research_metrics.insert_one({
            "timestamp": datetime.utcnow(),
            "user_id": session.get('user_id'),
            "auth_status": 'authenticated' if 'user' in session else 'anonymous',
            "event_type": event_type,
            "metadata": metadata or {},
            "experiment_group": session.get('experiment_group'),
            "concept_tags": metadata.get('concepts') if metadata else []
        })

    def log_memory_prediction(self, user_id, concept, predicted_date, actual_recall, predicted_score, actual_score):
        return self.db.research_metrics.insert_one({
            "event_type": "memory_prediction",
            "user_id": user_id,
            "concept": concept,
            "metadata": {
                "predicted": predicted_date,
                "actual": actual_recall,
                "error_days": (actual_recall - predicted_date).days,
                "predicted_score": predicted_score,
                "actual_score": actual_score
            },
            "timestamp": datetime.utcnow()
        })

    def get_engagement_metrics(self, start_date, end_date):
        return self.db.research_metrics.aggregate([
            {"$match": {
                "timestamp": {"$gte": start_date, "$lte": end_date},
                "event_type": {"$in": ["session_start", "quiz_start", "quiz_complete"]}
            }},
            {"$group": {
                "_id": "$user_id",
                "total_sessions": {"$sum": 1},
                "session_duration": {"$avg": "$metadata.duration"},
                "quizzes_started": {"$sum": {"$cond": [{"$eq": ["$event_type", "quiz_start"]}, 1, 0]}},
                "quizzes_completed": {"$sum": {"$cond": [{"$eq": ["$event_type", "quiz_complete"]}, 1, 0]}}
            }}
        ])

    def get_video_progress(self, user_id):
        return list(self.db.video_progress.find(
            {'user_id': user_id},
            {'video_id': 1, 'video_title': 1, 'attempts': 1, 'aggregates': 1}
        ))

    def update_concept_mastery(self, user_id, concept, delta):
        return self.db.concept_mastery.update_one(
            {"user_id": user_id, "concept": concept},
            {"$inc": {"score": delta}},
            upsert=True
        )

    def get_weak_concepts(self, user_id, threshold=5):
        return self.db.concept_mastery.distinct(
            "concept",
            {"user_id": user_id, "score": {"$lt": threshold}}
        )

    def get_learning_curve_data(self, days=30):
        return self.db.user_progress.aggregate([
            {"$match": {
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=days)}
            }},
            {"$unwind": "$attempts"},
            {"$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$attempts.timestamp"}},
                "avg_score": {"$avg": "$attempts.score"},
                "user_count": {"$addToSet": "$user_id"}
            }},
            {"$project": {
                "date": "$_id",
                "avg_score": 1,
                "active_users": {"$size": "$user_count"}
            }},
            {"$sort": {"date": 1}}
        ])

    def get_model_performance(self):
        return self.db.temp_content.aggregate([
            {"$group": {
                "_id": "$type",
                "total": {"$sum": 1},
                "errors": {"$sum": {"$cond": [{"$ifNull": ["$error", False]}, 1, 0]}},
                "api_calls": {"$sum": {"$size": {"$ifNull": ["$api_log", []]}}}
            }}
        ])

    def migrate_progress_data(self, old_user_id, new_user_id):
        """Transfer all user data from anonymous to authenticated ID"""
        try:
            collections = [
                'user_progress',
                'research_metrics',
                'concept_mastery',
                'learning_activities',
                'video_progress',
                'temp_content'
            ]

            for collection in collections:
                result = self.db[collection].update_many(
                    {"user_id": old_user_id},
                    {"$set": {"user_id": new_user_id}}
                )
                logging.info(f"Migrated {result.modified_count} entries in {collection}")

            return True
        except Exception as e:
            logging.error(f"Migration failed: {str(e)}")
            return False
db = Database()
