from datetime import datetime, timedelta
from flask import Blueprint, render_template, current_app
from flask_login import login_required, current_user
from db import db
import logging
import math
import numpy as np

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
        # FIX: 'page_view', 'quiz_attempt', 'session_heartbeat' are NOT in RESEARCH_EVENTS
        # and are never logged — this always returned [].
        # Changed to the events that are actually recorded.
        return db.research_metrics.distinct("user_id", {
            'timestamp': {'$gte': active_window},
            'event_type': {'$in': ['quiz_start', 'quiz_complete', 'flashcard_view', 'summary_view']}
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
        result = list(db.research_metrics.aggregate(pipeline))
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
            'quizzes':    db.research_metrics.count_documents({'event_type': 'quiz_start',     **time_filter}),
            'flashcards': db.research_metrics.count_documents({'event_type': 'flashcard_view', **time_filter}),
            'summaries':  db.research_metrics.count_documents({'event_type': 'summary_view',   **time_filter}),
            'graphs':     db.research_metrics.count_documents({'event_type': 'graph_view',     **time_filter})
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


def get_ab_experiments():
    """
    Pull every active experiment from ABTestManager and format it for the template.
    Returns a list of experiment dicts, one per experiment.
    Returns [] on any error or when no experiments exist.
    """
    try:
        ab_manager = current_app.ab_test_manager
        raw = ab_manager.get_all_experiment_results()
    except Exception as e:
        logging.error(f"Error fetching A/B experiments: {str(e)}")
        return []

    experiments = []
    for exp_name, result in raw.items():
        groups_out = []
        for label, gdata in result.get("groups", {}).items():
            groups_out.append({
                "label":            label,
                "n":                gdata.get("n", 0),
                "mean_score":       gdata.get("mean_score", 0.0),
                "p_value_adjusted": gdata.get("p_value_adjusted"),
                "cohens_d":         gdata.get("cohens_d"),
                "effect_magnitude": gdata.get("effect_magnitude", ""),
                "recommended_n":    gdata.get("recommended_n"),
                "underpowered":     gdata.get("underpowered", False),
            })

        experiments.append({
            "name": exp_name,
            "methodology_note": result.get(
                "methodology_note",
                "p-values corrected via Holm–Bonferroni. "
                "Alpha spending uses O'Brien–Fleming boundaries. "
                "Effect sizes calculated as Cohen's d vs control group."
            ),
            "groups": groups_out,
        })

    return experiments


def get_model_comparison():
    """
    Compare Ebbinghaus, ACT-R, and ML predictions against actual retention values
    stored in the 'model_predictions' MongoDB collection.

    Each document must have: model (str), predicted (float), actual (float).
    Returns None when fewer than 5 predictions exist.
    """
    try:
        docs = list(db.db.model_predictions.find({}))
    except Exception as e:
        logging.error(f"Error fetching model predictions: {str(e)}")
        return None

    if len(docs) < 5:
        return None

    model_errors: dict = {}
    for doc in docs:
        name      = doc.get("model", "unknown")
        predicted = doc.get("predicted")
        actual    = doc.get("actual")
        if predicted is None or actual is None:
            continue
        model_errors.setdefault(name, []).append(abs(predicted - actual))

    if not model_errors:
        return None

    display_names = {
        "ebbinghaus": "Ebbinghaus",
        "act_r":      "ACT-R",
        "ml":         "ML (BayesianRidge)"
    }
    model_order = ["ebbinghaus", "act_r", "ml"]

    models_out = []
    best_mae   = float("inf")

    for key in model_order:
        if key not in model_errors:
            continue
        errs = model_errors[key]
        mae  = float(np.mean(errs))
        rmse = float(math.sqrt(np.mean([e ** 2 for e in errs])))
        best_mae = min(best_mae, mae)
        models_out.append({
            "name":          display_names.get(key, key),
            "mae":           round(mae, 2),
            "rmse":          round(rmse, 2),
            "n_predictions": len(errs),
            "is_best":       False,
        })

    # FIX: float == float comparison is unreliable — use tolerance instead
    for m in models_out:
        m["is_best"] = abs(m["mae"] - round(best_mae, 2)) < 0.001

    return {"models": models_out}


def get_irt_distribution():
    """
    Build histogram data from per-user IRT theta estimates stored in 'user_progress'.
    Each document needs an 'irt_theta' field (float).
    Returns None when fewer than 3 users have been assessed.
    """
    try:
        docs = list(db.db.user_progress.find(
            {"irt_theta": {"$exists": True}},
            {"irt_theta": 1}
        ))
    except Exception as e:
        logging.error(f"Error fetching IRT distribution: {str(e)}")
        return None

    thetas = [d["irt_theta"] for d in docs if isinstance(d.get("irt_theta"), (int, float))]

    if len(thetas) < 3:
        return None

    arr = np.array(thetas)
    counts, edges = np.histogram(arr, bins=10, range=(-3.0, 3.0))
    bin_labels = [f"{edges[i]:.1f}" for i in range(len(edges) - 1)]
    n = len(arr)

    return {
        "bins":       bin_labels,
        "counts":     counts.tolist(),
        "mean_theta": round(float(arr.mean()), 3),
        "std_theta":  round(float(arr.std()),  3),
        "n_users":    n,
        "pct_high":   round(float(np.sum(arr >  1) / n * 100), 1),
        "pct_low":    round(float(np.sum(arr < -1) / n * 100), 1),
    }


@admin_bp.route('/dashboard')
def dashboard():
    metrics = {
        'user_stats': {
            'total_users':       get_total_users(),
            'weekly_active':     len(get_active_users()),
            'total_sessions':    get_total_sessions(days=7),
            'feature_usage':     get_feature_usage(),
            'model_performance': get_model_performance(),
            'quiz_completion':   get_quiz_completion_rate(days=7)
        },
        'ab_experiments':   get_ab_experiments(),
        'model_comparison': get_model_comparison(),
        'irt_distribution': get_irt_distribution(),
    }

    return render_template('admin_dashboard.html', **metrics)
