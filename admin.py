from datetime import datetime, timedelta
from flask import Blueprint, render_template
from flask_login import login_required, current_user
from db import db
import logging

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

@admin_bp.before_request
@login_required
def restrict_admin():
    if not current_user.is_admin:
        return "Unauthorized", 403

def safe_convert(value, default=0.0):
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default

def get_total_users():
    try:
        return db.users.count_documents({})
    except Exception as e:
        logging.error(f"Error counting users: {str(e)}")
        return 0

def get_active_users():
    """Get users active in last 7 days"""
    try:
        active_window = datetime.utcnow() - timedelta(days=7)
        return db.research_metrics.distinct("user_id", {
            'timestamp': {'$gte': active_window},
            'event_type': {'$in': ['page_view', 'quiz_attempt', 'session_heartbeat']}
        })
    except Exception as e:
        logging.error(f"Error getting active users: {str(e)}")
        return []

def get_quiz_completion_rate(days=7):
    try:
        pipeline = [
            {"$match": {
                "event_type": {"$in": ["quiz_start", "quiz_complete"]},
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=days)},
                "metadata.quiz_id": {"$exists": True}  
            }},
            {"$group": {
                "_id": "$metadata.quiz_id",
                "started": {"$sum": {"$cond": [{"$eq": ["$event_type", "quiz_start"]}, 1, 0]}},
                "completed": {"$sum": {"$cond": [{"$eq": ["$event_type", "quiz_complete"]}, 1, 0]}}
            }},
            {"$match": {  
                "started": {"$gte": 1},
                "completed": {"$lte": "$started"} 
            }},
            {"$group": {
                "_id": None,
                "completion_rate": {"$avg": {
                    "$cond": [
                        {"$eq": ["$started", 0]},
                        0,
                        {"$divide": ["$completed", "$started"]}
                    ]
                }}
            }}
        ]
        
        logging.debug(f"Completion rate pipeline: {pipeline}")
        result = list(db.research_metrics.aggregate(pipeline))
        logging.debug(f"Aggregation result: {result}")
        return (result[0]['completion_rate'] * 100) if result else 0
    except Exception as e:
        logging.error(f"Completion rate error: {str(e)}")
        return 0
    
def get_total_sessions(days=7):
    """Get total sessions started in last X days"""
    try:
        active_window = datetime.utcnow() - timedelta(days=days)
        return db.research_metrics.count_documents({
            'event_type': 'session_start',
            'timestamp': {'$gte': active_window}
        })
    except Exception as e:
        logging.error(f"Error counting sessions: {str(e)}")
        return 0

def get_feature_usage(days=7):
    try:
        time_filter = {"timestamp": {"$gte": datetime.utcnow() - timedelta(days=days)}}
        return {
            'quizzes': db.research_metrics.count_documents({'event_type': 'quiz_start', **time_filter}),
            'flashcards': db.research_metrics.count_documents({'event_type': 'flashcard_view', **time_filter}),
            'summaries': db.research_metrics.count_documents({'event_type': 'summary_view', **time_filter}),
            'graphs': db.research_metrics.count_documents({'event_type': 'graph_view', **time_filter})
        }
    except Exception as e:
        logging.error(f"Error getting feature usage: {str(e)}")
        return {'quizzes': 0, 'flashcards': 0, 'summaries': 0, 'graphs': 0}

def get_model_performance():
    try:
        results = db.db.temp_content.aggregate([
            {"$match": {
                "created_at": {"$gte": datetime.utcnow() - timedelta(days=7)}
            }},
            {"$group": {
                "_id": "$type",
                "total": {"$sum": 1},
                "errors": {"$sum": {"$cond": [{"$ifNull": ["$error", False]}, 1, 0]}},
                "api_calls": {"$sum": {"$size": {"$ifNull": ["$api_log", []]}}}
            }},
            {"$project": {
                "success_rate": {"$multiply": [
                    {"$divide": [
                        {"$subtract": ["$total", "$errors"]},
                        {"$cond": [{"$eq": ["$total", 0]}, 1, "$total"]}
                    ]},
                    100
                ]},
                "total": 1,
                "api_calls": 1
            }}
        ])
        return {res['_id']: {
            'success_rate': safe_convert(res.get('success_rate')),
            'total': res.get('total', 0),
            'api_calls': res.get('api_calls', 0)
        } for res in results}
    except Exception as e:
        logging.error(f"Error getting model performance: {str(e)}")
        return {}

@admin_bp.route('/dashboard')
def dashboard():
    
    metrics = {
        'user_stats': {
            'total_users': get_total_users(),
            'weekly_active': len(get_active_users()),
            'total_sessions': get_total_sessions(days=7),
            'feature_usage': get_feature_usage(),
            'model_performance': get_model_performance(),
            'quiz_completion': get_quiz_completion_rate(days=7)
        }
       
    }

    return render_template('admin_dashboard.html', **metrics)