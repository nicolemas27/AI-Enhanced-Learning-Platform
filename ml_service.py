# ml_service.py
import joblib
from sklearn.linear_model import LogisticRegression

import db

class PerformancePredictor:
    def __init__(self):
        self.model = joblib.load('trained_model.pkl')
    
    def preprocess_features(self, user_id):
        """Aggregate user data into model features"""
        features = db.db.research_metrics.aggregate([
            {'$match': {'user_id': user_id}},
            {'$group': {
                '_id': None,
                'avg_score': {'$avg': '$metadata.score'},
                'total_errors': {'$sum': '$metadata.errors'},
                'concept_variety': {'$addToSet': '$concept_tags'}
            }}
        ])
        return self._vectorize(features)
    
    def predict_performance(self, user_id):
        features = self.preprocess_features(user_id)
        return self.model.predict_proba([features])[0][1]