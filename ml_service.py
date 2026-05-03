# ml_service.py
import joblib
import numpy as np
from db import db


class PerformancePredictor:
    def __init__(self):
        # NOTE: trained_model.pkl must be generated and saved separately.
        # Train with sklearn, then: joblib.dump(model, 'trained_model.pkl')
        self.model = joblib.load('trained_model.pkl')

    def preprocess_features(self, user_id):
        """Aggregate user data into a numeric feature vector"""
        # FIX: aggregate returns a cursor — must consume it before use
        result = list(db.db.research_metrics.aggregate([
            {'$match': {'user_id': user_id}},
            {'$group': {
                '_id': None,
                'avg_score':       {'$avg': '$metadata.score'},
                'total_errors':    {'$sum': '$metadata.errors'},
                'concept_variety': {'$addToSet': '$concept_tags'}
            }}
        ]))

        if not result:
            return np.zeros(3)

        r = result[0]
        return np.array([
            r.get('avg_score') or 0.0,
            r.get('total_errors') or 0.0,
            # concept_variety is a list of lists — flatten and count unique tags
            len(set(tag for sublist in (r.get('concept_variety') or []) for tag in (sublist or [])))
        ], dtype=float)

    def predict_performance(self, user_id):
        """Return probability (0–1) that the user will pass their next quiz"""
        features = self.preprocess_features(user_id)
        return float(self.model.predict_proba([features])[0][1])
