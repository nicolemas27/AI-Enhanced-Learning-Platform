from datetime import datetime, timedelta
import hashlib
import logging
import random
import pandas as pd
import time, requests
from collections import defaultdict
from flask import session
from bson import ObjectId
import spacy
from db import db
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from dtw import dtw  # For LearningPatternAnalyzer
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class LearningAnalyzer:
    def __init__(self):
        self.learning_styles = ['visual', 'verbal', 'sequential', 'global']
        self.user_model = defaultdict(float)
        self.concept_graph = {}
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def track_activity(self, activity_type, metadata=None):
        """Store user learning activities anonymously"""
        if 'user_id' not in session:
            session['user_id'] = str(ObjectId())
            
        activity = {
            'user_id': session['user_id'],
            'type': activity_type,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        
        db.db.learning_activities.insert_one(activity)

    def analyze_learning_style(self, user_id):
        """Improved learning style analysis based on multiple factors"""
        activities = list(db.db.learning_activities.find(
            {'user_id': user_id}
        ).sort('timestamp', -1).limit(100))

        style_weights = defaultdict(float)
        
        # Analyze different activity types
        for act in activities:
            # Quiz behavior analysis
            if act['type'] == 'quiz_attempt':
                time_per_question = act['metadata'].get('time_per_question', 10)
                
                # Fast answers (<10s) indicate active learning style
                if time_per_question < 10:
                    style_weights['active'] += 0.2
                else:
                    style_weights['reflective'] += 0.2
                    
                # Visual preference detection
                if act['metadata'].get('diagrams_used', 0) > 0:
                    style_weights['visual'] += 0.3

            # Content interaction analysis
            elif act['type'] == 'content_view':
                if act['metadata']['content_type'] == 'diagram':
                    style_weights['visual'] += 0.4
                elif act['metadata']['content_type'] == 'text':
                    style_weights['verbal'] += 0.4
                    
                if act['metadata'].get('navigation_pattern') == 'sequential':
                    style_weights['sequential'] += 0.3
                else:
                    style_weights['global'] += 0.3

        # Normalize and format results
        total = sum(style_weights.values()) or 1  # Prevent division by zero
        return {k: v/total for k, v in style_weights.items()}

    # Retention Analysis Methods
    def calculate_retention_score(self, user_id):
        """Calculate retention with fallback for insufficient attempts"""
        try:
            pipeline = [
                {"$match": {
                    "user_id": user_id, 
                    "event_type": "quiz_attempt",
                    "metadata.score": {"$exists": True},
                    "metadata.total": {"$gt": 0}
                }},
                {"$sort": {"timestamp": -1}},
                {"$limit": 5},
                {"$group": {
                    "_id": None,
                    "total_score": {"$sum": "$metadata.score"},
                    "total_possible": {"$sum": "$metadata.total"},
                    "attempt_count": {"$sum": 1} 
                }}
            ]
            result = list(db.db.research_metrics.aggregate(pipeline))
            
            if not result or not result[0].get('total_possible'):
                return 50.0  # Changed from 0.0 to neutral 50%
                
            data = result[0]
            
            if data['attempt_count'] < 3:
                return max(30.0, (data['total_score'] / data['total_possible']) * 100)
                
            return round((data['total_score'] / data['total_possible']) * 100, 1)
            
        except Exception as e:
            logging.error(f"Retention score error: {str(e)}")
            return 50.0  

    def next_optimal_review_days(self, user_id):
        try:
            # Get last 3 attempts
            attempts = list(db.db.research_metrics.find(
                {"user_id": user_id, "event_type": "quiz_attempt"},
                sort=[("timestamp", -1)],
                limit=3
            ))
            
            # Base cases
            if not attempts:
                return 7  
                
            if len(attempts) == 1:
                return 4 if attempts[0]['metadata']['score']/attempts[0]['metadata']['total'] > 0.7 else 2
            
            # Weighted calculation for multiple attempts
            total_weight = 0
            weighted_days = 0
            
            for i in range(1, len(attempts)):
                prev = attempts[i-1]
                curr = attempts[i]
                
                score = prev['metadata']['score']
                total = prev['metadata']['total'] or 1
                days = (prev['timestamp'] - curr['timestamp']).days or 1
                weight = (score / total) ** 2  
                
                weighted_days += days * weight
                total_weight += weight
            
            # Dynamic bounds based on performance
            base_days = round(weighted_days / (total_weight or 1))
            
            # Apply Ebbinghaus-inspired scaling
            last_score = attempts[0]['metadata']['score'] / attempts[0]['metadata']['total']
            if last_score > 0.9:
                return min(base_days * 2, 21)  
            elif last_score > 0.7:
                return min(base_days, 14)
            else:
                return max(1, min(base_days // 2, 7)) 
                
        except Exception:
            return 7  

    def forgetting_risk(self, user_id, days=7):
        """Calculate risk of forgetting using time-weighted decay"""
        try:
            concepts = list(db.db.concept_mastery.find(
                {"user_id": user_id},
                sort=[("timestamp", -1)]
            ))
            
            if not concepts:
                return 0.0
                
            total_risk = 0
            concept_count = 0
            now = datetime.now()
            
            for concept in concepts:
                # Calculate time decay factor (0-1 where 1 = recent)
                hours_old = (now - concept['timestamp']).total_seconds() / 3600
                decay_factor = 1 / (1 + hours_old/24)  # Halves every 24 hours
                
                # Calculate mastery decay
                mastery = concept.get('score', 0)
                risk = (1 - mastery/10) * 100 * decay_factor
                total_risk += min(max(risk, 0), 100)
                concept_count += 1
                
            return round(total_risk / concept_count, 1) if concept_count else 0.0
            
        except Exception as e:
            logging.error(f"Forgetting risk error: {str(e)}")
            return 0.0

    def actual_retention_curve(self, user_id):
        """Create realistic decay even with sparse data"""
        try:
            attempts = list(db.db.research_metrics.find(
                {"user_id": user_id, "event_type": "quiz_attempt"},
                sort=[("timestamp", 1)]
            ))
            
            if not attempts:
                return [100, 85, 70, 50, 30]

            # Create time-based retention scores
            retention_data = []
            reference_date = attempts[0]['timestamp']
            
            for attempt in attempts:
                days_passed = (attempt['timestamp'] - reference_date).days
                score = (attempt['metadata']['score']/attempt['metadata']['total'])*100
                retention_data.append( (days_passed, score) )

            # Fill in gaps using exponential decay
            curve = []
            for target_day in [0, 1, 3, 7, 30]:
                # Find nearest actual attempt
                nearest = min([p for p in retention_data if p[0] <= target_day], 
                            key=lambda x: abs(x[0]-target_day), default=None)
                
                if nearest:
                    curve.append(nearest[1])
                else:
                    # Apply default decay if no data
                    prev_value = curve[-1] if curve else 100
                    decay = 0.15 * (target_day - (curve[-1][0] if curve else 0))
                    curve.append(max(0, prev_value * (1 - decay)))

            return curve[:5]  # Ensure exactly 5 points
        
        except Exception:
            return [100, 85, 70, 50, 30]

    def _normalize_curve(self, scores):
        """Convert variable-length scores to standardized 5-point curve"""
        length = len(scores)
        if length < 5:
            return scores + [30]*(5-length)
            
        return [
            scores[0],
            scores[length//4],
            scores[length//2],
            scores[3*length//4],
            scores[-1]
        ]

    # Content Adaptation Methods
    def generate_adaptive_content(self, content_type, base_content):
        """Adapt content based on user model"""
        style_weights = self.analyze_learning_style(session['user_id'])
        
        if content_type == 'quiz':
            return self._adapt_quiz(base_content, style_weights)
        elif content_type == 'summary':
            return self._adapt_summary(base_content, style_weights)
        
        return base_content

    def _adapt_quiz(self, quiz_data, style_weights):
        """Modify quiz based on learning style"""
        if style_weights['visual'] > 0.6:
            for question in quiz_data['questions']:
                if not question.get('diagram'):
                    question['diagram'] = self._generate_diagram(question['question'])
        return quiz_data

    def _adapt_summary(self, summary, style_weights):
        """Modify summary presentation based on preferences"""
        if style_weights['visual'] > 0.4:
            summary['visual_summary'] = self._generate_visual_summary(summary['text'])
        return summary

    def _generate_diagram(self, concept):
        """Generate visual representation using AI"""
        prompt = f"Create a text-based diagram explaining: {concept}"
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        return response.text

    def _generate_visual_summary(self, text):
        """Generate visual summary using AI"""
        prompt = f"Convert this summary to visual elements:\n{text}"
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        return response.text
    

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        self.mastery_threshold = 0.7
        self.nlp = spacy.load("en_core_web_sm")
        
    def calculate_mastery_from_results(self, results):
        """Calculate mastery directly from quiz results"""
        concept_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for question in results:
            concept = self.extract_question_concept(question['question'])
            concept_stats[concept]['total'] += 1
            if question['is_correct']:
                concept_stats[concept]['correct'] += 1
                
        return {concept: stats['correct']/stats['total'] 
                for concept, stats in concept_stats.items()}

    def extract_question_concept(self, question_text):
        """Improved concept extraction"""
        try:
            # Use both TF-IDF and noun phrases
            features = self.vectorizer.transform([question_text])
            tfidf_concept = max(zip(features.toarray()[0], self.vectorizer.get_feature_names_out()))[1]
            
            # Get noun chunks
            doc = self.nlp(question_text)
            nouns = [chunk.text for chunk in doc.noun_chunks]
            
            return tfidf_concept if tfidf_concept in nouns else nouns[0] if nouns else tfidf_concept
        except Exception as e:
            logging.error(f"Concept extraction error: {str(e)}")
            return "general"

    def calculate_mastery(self, user_id):
        """Calculate concept mastery from quiz history"""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$unwind": "$details"},
            {"$group": {
                "_id": "$details.concept",
                "total": {"$sum": 1},
                "correct": {"$sum": {"$cond": ["$details.is_correct", 1, 0]}}
            }},
            {"$project": {
                "mastery": {"$divide": ["$correct", "$total"]},
                "concept": "$_id",
                "_id": 0
            }}
        ]
        results = db.db.quiz_results.aggregate(pipeline)
        return {res['concept']: res['mastery'] for res in results}
    
    def extract_concepts(self, transcript):
        """Extract key concepts from transcript"""
        # Use TF-IDF to find important terms
        tfidf = self.vectorizer.fit_transform([transcript])
        feature_names = self.vectorizer.get_feature_names_out()
        return sorted(
            [(feature_names[i], tfidf[0, i]) 
            for i in tfidf.nonzero()[1]],
            key=lambda x: -x[1]
        )[:10]

    def estimate_difficulty(self, transcript):
        """Classify content difficulty using Gemini"""
        prompt = f"""Rate the difficulty of this content:
        Options: beginner, intermediate, advanced
        Content: {transcript[:3000]}
        Answer ONLY with the difficulty level."""
        
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        return response.text.strip().lower()
    
    def get_adaptive_content(self, content_id, user_id):
        """Retrieve content with adaptive modifications"""
        content = self.get_temp_content(content_id)
        user_model = self.get_user_model(user_id)
        
        if user_model and content:
            # Apply adaptive modifications
            if user_model.get('prefers_diagrams'):
                content['data'] = self._add_visual_aids(content['data'])
            
            if user_model.get('prefers_easy_first'):
                content['data']['questions'].sort(key=lambda q: q['difficulty'])
                
        return content
    
    def get_concept_evolution(self, user_id):
        """Return concept mastery over time with video context"""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$unwind": "$details"},
            {"$group": {
                "_id": {
                    "concept": "$details.concept",
                    "video_id": "$video_id"
                },
                "timeline": {
                    "$push": {
                        "date": "$timestamp",
                        "score": {"$cond": ["$details.is_correct", 1, 0]},
                        "video_title": "$video_title"
                    }
                },
                "total_attempts": {"$sum": 1},
                "correct_attempts": {"$sum": {"$cond": ["$details.is_correct", 1, 0]}
            }}},
            {"$project": {
                "concept": "$_id.concept",
                "video_id": "$_id.video_id",
                "mastery": {"$divide": ["$correct_attempts", "$total_attempts"]},
                "timeline": 1,
                "_id": 0
            }}
        ]
        return list(db.db.research_metrics.aggregate(pipeline))

    def _add_visual_aids(self, content):
        """Helper method to add visual elements"""
        for question in content.get('questions', []):
            if not question.get('diagram'):
                question['diagram'] = generate_diagram(question['question'])
        return content    
    
from scipy import stats
import numpy as np

class ABTestManager:
    def __init__(self):
        self.experiment_groups = ['control', 'ML_based', 'rule_based']  # Define experiment groups

    def assign_group(self, user_id):
        """Consistent group assignment using hash"""
        hash_obj = hashlib.sha256(user_id.encode() + b'salt')
        hash_int = int(hash_obj.hexdigest(), 16)
        return self.experiment_groups[hash_int % len(self.experiment_groups)]
    
    def analyze_results(self, experiment_name):
        pipeline = [
            {"$match": {
                "event_type": "algorithm_performance",
                "experiment": experiment_name
            }},
            {"$group": {
                "_id": "$group",
                "avg_score": {"$avg": "$normalized_score"},
                "count": {"$sum": 1},
                "scores": {"$push": "$normalized_score"}
            }},
            {"$project": {
                "group": "$_id",
                "avg_score": 1,
                "count": 1,
                "variance": {"$stdDevPop": "$scores"},
                "_id": 0
            }}
        ]
        
        groups = list(db.db.research_metrics.aggregate(pipeline))
        
        # Calculate statistical significance
        results = {}
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                g1 = groups[i]
                g2 = groups[j]
                
                t_stat, p_value = stats.ttest_ind(
                    g1['scores'], g2['scores'], equal_var=False
                )
                
                key = f"{g1['group']}_vs_{g2['group']}"
                results[key] = {
                    'p_value': p_value,
                    'effect_size': abs(g1['avg_score'] - g2['avg_score'])
                }
        
        return {
            'groups': {g['group']: g for g in groups},
            'comparisons': results
        }
        
    def log_model_comparison(self, user_id):
            """Store model comparisons for analysis"""
            comparisons = self.get_model_comparison(user_id)
            for comp in comparisons:
                db.log_research_event('memory_model_prediction', {
                    "concept": comp['concept'],
                    "predictions": {
                        "ebbinghaus": comp['ebbinghaus'],
                        "act_r": comp['act_r'],
                        "ml": comp['ml']
                    },
                    "actual_recall": comp['actual_recall']
                })
    def log_experiment_result(self, user_id, experiment_name, score, total):
        group = self.assign_group(user_id)
        db.log_algorithm_performance(  # Use new method
            user_id=user_id,
            experiment_name=experiment_name,
            group=group,
            score=score,
            total=total
        )

class MemoryModel:
    def schedule_review(self, user_id, concept):
        """Calculate optimal review schedule for a concept"""
        try:
            last_tested = self._get_last_tested_date(user_id, concept)
            mastery = self._get_concept_mastery(user_id, concept)
            
            if mastery < 4:
                return datetime.now() + timedelta(days=1)
            elif mastery < 6:
                return datetime.now() + timedelta(days=3)
            else:
                return datetime.now() + timedelta(days=7)
                
        except Exception as e:
            logging.error(f"Schedule error: {str(e)}")
            return datetime.now() + timedelta(days=1)

    def _get_concept_mastery(self, user_id, concept):
        """Get current mastery score for a concept"""
        doc = db.db.concept_mastery.find_one(
            {"user_id": user_id, "concept": concept},
            {"score": 1}
        )
        return doc.get("score", 0) if doc else 0
    
    def get_review_schedule(self, user_id):
        """Return optimized review schedule using multiple models"""
        concepts = self._get_weak_concepts(user_id)
        schedule = {}
        
        for concept in concepts:
            # Get predictions from all models
            ebbinghaus = self._ebbinghaus_prediction(user_id, concept)
            act_r = self._act_r_prediction(user_id, concept)
            ml = self._ml_prediction(user_id, concept)
            
            # Use earliest recommended review date
            schedule[concept] = min(ebbinghaus, act_r, ml)
        
        return schedule

    def _get_weak_concepts(self, user_id):
        return db.db.concept_mastery.find(
            {"user_id": user_id, "score": {"$lt": 7}},
            {"concept": 1}
        ).distinct("concept")
    
    def get_detailed_predictions(self, user_id):
        """Return predictions from all models with explanations"""
        concepts = self._get_weak_concepts(user_id)
        predictions = {}
        
        for concept in concepts:
            predictions[concept] = {
                'ebbinghaus': self._ebbinghaus_prediction(user_id, concept),
                'act_r': self._act_r_prediction(user_id, concept),
                'ml': self._ml_prediction(user_id, concept),
                'last_tested': self._get_last_tested_date(user_id, concept)
            }
        return predictions

    def _ebbinghaus_prediction(self, user_id, concept):
        """Ebbinghaus forgetting curve implementation"""
        last_tested = self._get_last_tested_date(user_id, concept)
        if not last_tested:
            return timedelta(days=1)  # Default interval for new concepts
        
        days_since = (datetime.now() - last_tested).days
        return timedelta(days=2 ** min(days_since, 5))  # Cap at 32 days max interval

    def _act_r_prediction(self, user_id, concept):
        """ACT-R memory decay model"""
        mastery = db.db.concept_mastery.find_one(
            {"user_id": user_id, "concept": concept}
        )
        if mastery and mastery.get('score', 0) < 7:
            return datetime.now() + timedelta(days=3)
        return datetime.now() + timedelta(days=7)

    def _get_last_tested_date(self, user_id, concept):
        result = db.db.quiz_results.find_one(
            {"user_id": user_id, "details.concept": concept},
            sort=[("timestamp", -1)]
        )
        return result['timestamp'] if result else None
    
    def get_model_comparison(self, user_id):
        """Compare different memory models for research paper"""
        concepts = self._get_weak_concepts(user_id)
        results = []
        
        for concept in concepts:
            entry = {
                "concept": concept,
                "ebbinghaus": self._ebbinghaus_prediction(...),
                "act_r": self._act_r_prediction(...),
                "ml": self._ml_prediction(...),
                "actual_recall": self._get_actual_recall(user_id, concept)
            }
            results.append(entry)
        
        return results

    def _get_actual_recall(self, user_id, concept):
        """Calculate real-world recall accuracy"""
        attempts = db.db.quiz_results.find({
            "user_id": user_id,
            "details.concept": concept
        }).sort("timestamp", 1)
        
        return [a['is_correct'] for a in attempts]
    
    def calculate_retention_score(self, user_id):
        """Calculate actual retention from quiz history"""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$sort": {"timestamp": -1}},
            {"$limit": 5},  # Last 5 quizzes
            {"$group": {
                "_id": None,
                "avg_score": {"$avg": "$score"},
                "total": {"$sum": "$total"}
            }}
        ]
        
        result = db.db.research_metrics.aggregate(pipeline).next()
        return (result['avg_score'] / result['total']) * 100 if result['total'] > 0 else 0

    def next_optimal_review_days(self, user_id):
        """Calculate using Ebbinghaus formula with actual attempt data"""
        last_attempt = db.db.research_metrics.find_one(
            {"user_id": user_id, "type": "quiz_attempt"},
            sort=[("timestamp", -1)]
        )
        
        if not last_attempt:
            return 1  # Default to 1 day if no attempts
            
        days_since = (datetime.now() - last_attempt['timestamp']).days
        return int(2 ** min(days_since, 5))  # Ebbinghaus spacing

    def forgetting_risk(self, user_id, days=7):
        """Calculate risk based on concept mastery decay"""
        concepts = db.get_weak_concepts(user_id)
        if not concepts:
            return 0
            
        pipeline = [
            {"$match": {"user_id": user_id, "concept": {"$in": concepts}}},
            {"$group": {
                "_id": None,
                "avg_decay": {"$avg": "$decay_rate"},
                "count": {"$sum": 1}
            }}
        ]
        
        result = db.db.concept_mastery.aggregate(pipeline).next()
        base_risk = 100 - (result['avg_decay'] * days * 100)
        return max(0, min(100, base_risk))

    def actual_retention_curve(self, user_id):
        """Calculate actual retention at specific time intervals"""
        try:
            attempts = list(db.db.research_metrics.find(
                {"user_id": user_id, "event_type": "quiz_attempt"},
                {"metadata.score": 1, "metadata.total": 1, "timestamp": 1},
                sort=[("timestamp", 1)]
            ))
            
            if not attempts:
                return [100, 80, 65, 50, 30]

            first_attempt_date = attempts[0]['timestamp']
            time_buckets = {
                'Now': 0,
                '1 Day': 1,
                '3 Days': 3,
                '1 Week': 7,
                '1 Month': 30
            }

            bucket_scores = {k: [] for k in time_buckets}
            
            for attempt in attempts:
                days_since_first = (attempt['timestamp'] - first_attempt_date).days
                score_pct = (attempt['metadata']['score'] / attempt['metadata']['total']) * 100
                
                for label, days in time_buckets.items():
                    if days_since_first <= days:
                        bucket_scores[label].append(score_pct)
                        break

            return [
                np.mean(bucket_scores['Now']) if bucket_scores['Now'] else 100,
                np.mean(bucket_scores['1 Day']) if bucket_scores['1 Day'] else 80,
                np.mean(bucket_scores['3 Days']) if bucket_scores['3 Days'] else 65,
                np.mean(bucket_scores['1 Week']) if bucket_scores['1 Week'] else 50,
                np.mean(bucket_scores['1 Month']) if bucket_scores['1 Month'] else 30
            ]
            
        except Exception as e:
            logging.error(f"Retention curve error: {str(e)}")
            return [100, 80, 65, 50, 30]
    
    def _normalize_curve(self, scores):
        """Convert variable-length scores to 5-point curve"""
        if len(scores) < 5:
            return [100, 80, 65, 50, 30]  # Default curve
            
        step = len(scores) // 5
        return [
            scores[0],
            scores[step*1],
            scores[step*2],
            scores[step*3],
            scores[-1]
        ]

    def _calculate_personal_decay(self, user_id):
        """Safer decay calculation with zero handling"""
        try:
            attempts = list(db.db.research_metrics.find(
                {"user_id": user_id, "event_type": "quiz_attempt"},
                sort=[("timestamp", 1)]
            ))
            
            if len(attempts) < 2:
                return 0.15  # Default for new users

            decay_rates = []
            for i in range(1, len(attempts)):
                try:
                    # Prevent zero divisions
                    prev_total = attempts[i-1]['metadata']['total'] or 1
                    curr_total = attempts[i]['metadata']['total'] or 1
                    prev_score = attempts[i-1]['metadata']['score']/prev_total
                    curr_score = attempts[i]['metadata']['score']/curr_total
                    
                    # Ensure minimum 1 day between attempts
                    time_diff = max(1, (attempts[i]['timestamp'] - attempts[i-1]['timestamp']).days)
                    score_diff = abs(prev_score - curr_score)
                    
                    decay_rate = score_diff / time_diff
                    decay_rates.append(min(max(decay_rate, 0.05), 0.5))
                except KeyError:
                    continue

            return np.median(decay_rates) if decay_rates else 0.15
        except Exception:
            return 0.15
       
class EnhancedMemoryModel(MemoryModel):
    def get_video_predictions(self, user_id, video_id):
        """Get predictions for all concepts in a video with proper date handling"""
        try:
            video = db.db.video_progress.find_one({
                'user_id': user_id,
                'video_id': video_id
            })
            
            if not video:
                return {}
                
            concepts = set()
            for attempt in video.get('attempts', []):
                concepts.update(attempt.get('metadata', {}).get('concepts', []))
            
            predictions = {}
            for concept in concepts:
                next_review = self.get_forgetting_prediction(user_id, concept)
                predictions[concept] = {
                    'next_review': next_review,
                    'formatted_date': next_review.strftime('%Y-%m-%d %H:%M')
                }
            
            return predictions
            
        except Exception as e:
            logging.error(f"Video prediction error: {str(e)}")
            return {}
    
    def get_forgetting_prediction(self, user_id, concept):
        """Enhanced Ebbinghaus model with proper datetime handling"""
        try:
            base_interval = self._ebbinghaus_prediction(user_id, concept)
            
            # Get personalization factors
            user_factor = self._get_cognitive_factor(user_id)
            concept_factor = self._get_concept_difficulty(concept)
            
            # Calculate adjusted interval
            adjusted_days = base_interval.days * (0.6 * user_factor + 0.4 * concept_factor)
            final_days = max(1, int(adjusted_days))
            
            return datetime.now() + timedelta(days=final_days)
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return datetime.now() + timedelta(days=1)
                
    def _get_cognitive_factor(self, user_id):
        """Calculate individual memory retention factor"""
        attempts = list(db.db.quiz_results.find(
            {"user_id": user_id},
            sort=[("timestamp", -1)],
            limit=10
        ))
        
        if not attempts:
            return 1.0  # Default
        
        decay_rates = [
            (a['timestamp'] - attempts[i+1]['timestamp']).days / 
            (attempts[i]['metadata']['score'] - attempts[i+1]['metadata']['score'] + 0.1)
            for i in range(len(attempts)-1)
        ]
        return sum(decay_rates)/len(decay_rates)

    def _get_concept_difficulty(self, concept):
        """Get normalized difficulty (0.5-1.5 range)"""
        return db.db.concept_difficulty.find_one(
            {"concept": concept},
            {"score": 1}
        ) or 1.0
    
    def predict_retention_curve(self, user_id):
        """Generate realistic decay even with limited data"""
        base_retention = 100
        try:
            # Get user's actual decay rate or use default
            decay_rate = self._calculate_personal_decay(user_id) or 0.15
            last_activity = self._get_last_activity_date(user_id)
            days_inactive = (datetime.now() - last_activity).days if last_activity else 0
            
            # Increase decay rate for inactive days
            decay_rate *= (1 + days_inactive * 0.05)
            
            # Calculate curve with accelerated decay for inactivity
            return [
                base_retention,
                max(0, base_retention * (1 - decay_rate)**1),
                max(0, base_retention * (1 - decay_rate)**3),
                max(0, base_retention * (1 - decay_rate)**7),
                max(0, base_retention * (1 - decay_rate)**30)
            ]
        except Exception:
            # Default curve for new users
            return [100, 85, 70, 50, 30]

    def _get_last_activity_date(self, user_id):
        """Get last learning activity date"""
        last_attempt = db.db.research_metrics.find_one(
            {"user_id": user_id},
            sort=[("timestamp", -1)]
        )
        return last_attempt['timestamp'] if last_attempt else None

    def _calculate_study_frequency(self, user_id):
        """Calculate days between study sessions for this user, considering multiple quizzes on the same day as separate sessions"""
        sessions = list(db.db.research_metrics.find(
            {"user_id": user_id, "event_type": {"$in": ["quiz_attempt", "flashcard_view"]}},
            sort=[("timestamp", 1)]
        ))
        
        if len(sessions) < 2:
            return 7  # Default to weekly if not enough data
        
        intervals = []
        for i in range(1, len(sessions)):
            delta = (sessions[i]['timestamp'] - sessions[i-1]['timestamp']).days
            intervals.append(delta)
        
        # Calculate the average time difference between sessions
        return np.percentile(intervals, 70)  # 70th percentile to ignore outliers


    def _get_average_difficulty(self, user_id):
        """Get average difficulty of studied content"""
        avg_difficulty = db.db.video_progress.aggregate([
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": None,
                "avg_difficulty": {"$avg": "$metadata.difficulty_score"}
            }}
        ])
        return avg_difficulty.next()['avg_difficulty'] or 50

    def _calculate_personal_decay(self, user_id):
        attempts = list(db.db.research_metrics.find(
            {"user_id": user_id, "event_type": "quiz_attempt"},
            sort=[("timestamp", 1)]
        ))
        
        if len(attempts) < 2:
            return 0.15  # Safe default
        
        decay_rates = []
        for i in range(1, len(attempts)):
            try:
                prev_total = attempts[i-1]['metadata']['total'] or 1
                curr_total = attempts[i]['metadata']['total'] or 1
                
                prev_score = attempts[i-1]['metadata']['score']/prev_total
                curr_score = attempts[i]['metadata']['score']/curr_total
                time_diff = (attempts[i]['timestamp'] - attempts[i-1]['timestamp']).days or 1
                
                score_diff = prev_score - curr_score
                decay_rate = abs(score_diff) / time_diff
                decay_rates.append(min(max(decay_rate, 0.05), 0.5))
            except KeyError:
                continue
        
        return np.median(decay_rates) if decay_rates else 0.15