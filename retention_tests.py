# retention_tests.py
import unittest
from datetime import datetime, timedelta
from app import app, db
from adaptive_learning import EnhancedMemoryModel, LearningAnalyzer

class TestRetention(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.user_base = "test_user_"
        
    def simulate_quiz(self, user_id, scores, days_ago):
        """Insert quiz attempts into the database
        scores: List of scores (0-5)
        days_ago: List of days before today 
        """
        db.db.research_metrics.delete_many({"user_id": user_id})
        
        for score, day in zip(scores, days_ago):
            db.db.research_metrics.insert_one({
                "user_id": user_id,
                "event_type": "quiz_attempt",
                "timestamp": datetime.now() - timedelta(days=day),
                "metadata": {
                    "score": score,
                    "total": 5,  # 5-mark quizzes
                    "time_spent": 60
                }
            })
    
    def test_new_user(self):
        user_id = self.user_base + "new"
        model = EnhancedMemoryModel()
        
        predicted = model.predict_retention_curve(user_id)
        # Should return default curve
        self.assertTrue(predicted[0] == 100)
        self.assertEqual(len(predicted), 5)
        self.assertTrue(all(isinstance(x, (int, float)) for x in predicted))

    
    def test_perfect_scores(self):
        user_id = self.user_base + "perfect"
        # 3 perfect attempts in last 3 days
        self.simulate_quiz(user_id, [5,5,5], [2,1,0])
        
        analyzer = LearningAnalyzer()
        actual = analyzer.actual_retention_curve(user_id)
        # Last score should be 100%
        self.assertAlmostEqual(actual[0], 100.0, delta=1)
        
        # Next day prediction should be high
        model = EnhancedMemoryModel()
        predicted = model.predict_retention_curve(user_id)
        self.assertGreater(predicted[1], 90)  # >90% retention
    
    def test_poor_performance(self):
        user_id = self.user_base + "poor"
        # Scores 1/5 over 3 days
        self.simulate_quiz(user_id, [1,1,1], [2,1,0])
        
        model = EnhancedMemoryModel()
        predicted = model.predict_retention_curve(user_id)
        # Should show rapid decay
        self.assertLess(predicted[-1], 25)  # <20% after 1 month

if __name__ == '__main__':
    unittest.main()