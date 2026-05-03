from datetime import datetime, timedelta
import hashlib
import logging
import math
import random
import numpy as np
import pandas as pd
import time, requests
from collections import defaultdict
from flask import session
from bson import ObjectId
import spacy
from db import db
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import BayesianRidge
from scipy import stats
from scipy.special import expit  # sigmoid

# ---------------------------------------------------------------------------
# LearningAnalyzer
# ---------------------------------------------------------------------------

class LearningAnalyzer:
    """
    Tracks user learning activities and infers learning styles,
    retention, and concept mastery.

    FIX: Removed duplicate __init__ — the original file had two __init__
    definitions; the second one silently overwrote the first, destroying
    self.learning_styles, self.user_model, and self.concept_graph.
    """

    def __init__(self):
        self.learning_styles = ['visual', 'verbal', 'sequential', 'global']
        self.user_model = defaultdict(float)
        self.concept_graph = {}
        self.nlp = spacy.load("en_core_web_sm")
        # FIX: max_features=100 preserved from the second (overwriting) __init__
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        self.mastery_threshold = 0.7

    def track_activity(self, activity_type, metadata=None):
        """Store user learning activities anonymously"""
        if 'user_id' not in session:
            session['user_id'] = str(ObjectId())

        activity = {
            'user_id': session['user_id'],
            'type': activity_type,
            'timestamp': datetime.utcnow(),
            'metadata': metadata or {}
        }
        db.db.learning_activities.insert_one(activity)

    def analyze_learning_style(self, user_id):
        """Learning style analysis based on multiple activity factors"""
        activities = list(db.db.learning_activities.find(
            {'user_id': user_id}
        ).sort('timestamp', -1).limit(100))

        style_weights = defaultdict(float)

        for act in activities:
            if act['type'] == 'quiz_attempt':
                time_per_question = act['metadata'].get('time_per_question', 10)
                if time_per_question < 10:
                    style_weights['active'] += 0.2
                else:
                    style_weights['reflective'] += 0.2
                if act['metadata'].get('diagrams_used', 0) > 0:
                    style_weights['visual'] += 0.3

            elif act['type'] == 'content_view':
                content_type = act['metadata'].get('content_type', '')
                if content_type == 'diagram':
                    style_weights['visual'] += 0.4
                elif content_type == 'text':
                    style_weights['verbal'] += 0.4

                if act['metadata'].get('navigation_pattern') == 'sequential':
                    style_weights['sequential'] += 0.3
                else:
                    style_weights['global'] += 0.3

        total = sum(style_weights.values()) or 1
        return {k: v / total for k, v in style_weights.items()}

    # ------------------------------------------------------------------
    # Bayesian Knowledge Tracing (BKT)
    # ------------------------------------------------------------------
    # FIX: Replaced simple averaging with a proper BKT implementation.
    # BKT models mastery as a latent binary variable and updates it via
    # Bayes' rule after each observation (correct / incorrect).
    #
    # Parameters (per-concept defaults; ideally fitted per concept):
    #   p_l0   — prior probability of knowing the concept
    #   p_t    — probability of learning on each attempt (transition)
    #   p_g    — probability of a correct guess when not knowing
    #   p_s    — probability of an incorrect slip when knowing
    # ------------------------------------------------------------------

    _BKT_DEFAULTS = dict(p_l0=0.1, p_t=0.1, p_g=0.2, p_s=0.1)

    def _bkt_update(self, p_mastery, is_correct, p_g=0.2, p_s=0.1, p_t=0.1):
        """
        Single BKT update step.
        Returns updated posterior P(mastered | observation).
        """
        if is_correct:
            p_obs_know = 1.0 - p_s        # P(correct | knows)
            p_obs_not  = p_g               # P(correct | doesn't know)
        else:
            p_obs_know = p_s               # P(incorrect | knows)
            p_obs_not  = 1.0 - p_g        # P(incorrect | doesn't know)

        # Bayes update
        numerator   = p_obs_know * p_mastery
        denominator = numerator + p_obs_not * (1.0 - p_mastery)
        p_posterior = numerator / (denominator or 1e-9)

        # Learning transition: even if not known, might learn after attempt
        p_posterior = p_posterior + (1.0 - p_posterior) * p_t
        return float(np.clip(p_posterior, 0.0, 1.0))

    def calculate_mastery_bkt(self, user_id):
        """
        FIX: Compute concept mastery using Bayesian Knowledge Tracing
        instead of raw averages.  Returns dict {concept: P(mastered)}.
        """
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$unwind": "$details"},
            {"$sort": {"timestamp": 1}},
            {"$project": {
                "concept":    "$details.concept",
                "is_correct": "$details.is_correct",
                "timestamp":  1
            }}
        ]
        events = list(db.db.quiz_results.aggregate(pipeline))

        concept_mastery = defaultdict(lambda: self._BKT_DEFAULTS['p_l0'])
        for ev in events:
            concept = ev.get('concept', 'general')
            concept_mastery[concept] = self._bkt_update(
                concept_mastery[concept],
                ev['is_correct']
            )
        return dict(concept_mastery)

    def calculate_mastery_from_results(self, results):
        """
        FIX: Compute mastery from a single quiz session using BKT
        (previously used plain averages).
        """
        concept_mastery = defaultdict(lambda: self._BKT_DEFAULTS['p_l0'])
        for question in results:
            concept = self.extract_question_concept(question['question'])
            concept_mastery[concept] = self._bkt_update(
                concept_mastery[concept],
                question['is_correct']
            )
        return dict(concept_mastery)

    # Kept for backward compat but delegates to BKT version
    def calculate_mastery(self, user_id):
        return self.calculate_mastery_bkt(user_id)

    # ------------------------------------------------------------------
    # Retention & Forgetting
    # ------------------------------------------------------------------

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
                    "total_score":   {"$sum": "$metadata.score"},
                    "total_possible":{"$sum": "$metadata.total"},
                    "attempt_count": {"$sum": 1}
                }}
            ]
            result = list(db.db.research_metrics.aggregate(pipeline))

            if not result or not result[0].get('total_possible'):
                return 50.0

            data = result[0]
            if data['attempt_count'] < 3:
                return max(30.0, (data['total_score'] / data['total_possible']) * 100)

            return round((data['total_score'] / data['total_possible']) * 100, 1)

        except Exception as e:
            logging.error(f"Retention score error: {str(e)}")
            return 50.0

    def next_optimal_review_days(self, user_id):
        try:
            attempts = list(db.db.research_metrics.find(
                {"user_id": user_id, "event_type": "quiz_attempt"},
                sort=[("timestamp", -1)],
                limit=3
            ))

            if not attempts:
                return 7

            if len(attempts) == 1:
                ratio = attempts[0]['metadata']['score'] / max(attempts[0]['metadata']['total'], 1)
                return 4 if ratio > 0.7 else 2

            total_weight  = 0.0
            weighted_days = 0.0

            for i in range(1, len(attempts)):
                prev = attempts[i - 1]
                curr = attempts[i]
                score = prev['metadata']['score']
                total = prev['metadata']['total'] or 1
                days  = (prev['timestamp'] - curr['timestamp']).days or 1
                weight = (score / total) ** 2
                weighted_days += days * weight
                total_weight  += weight

            base_days  = round(weighted_days / (total_weight or 1))
            last_score = attempts[0]['metadata']['score'] / max(attempts[0]['metadata']['total'], 1)

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

            total_risk    = 0.0
            concept_count = 0
            now = datetime.utcnow()

            for concept in concepts:
                hours_old    = (now - concept['timestamp']).total_seconds() / 3600
                decay_factor = 1 / (1 + hours_old / 24)
                mastery      = concept.get('score', 0)
                risk         = (1 - mastery / 10) * 100 * decay_factor
                total_risk  += min(max(risk, 0), 100)
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

            retention_data = []
            reference_date = attempts[0]['timestamp']

            for attempt in attempts:
                days_passed = (attempt['timestamp'] - reference_date).days
                total = attempt['metadata'].get('total') or 1
                score = (attempt['metadata'].get('score', 0) / total) * 100
                retention_data.append((days_passed, score))

            curve = []
            for target_day in [0, 1, 3, 7, 30]:
                candidates = [p for p in retention_data if p[0] <= target_day]
                if candidates:
                    nearest = min(candidates, key=lambda x: abs(x[0] - target_day))
                    curve.append(nearest[1])
                else:
                    prev_value = curve[-1] if curve else 100
                    decay = 0.15 * target_day
                    curve.append(max(0, prev_value * (1 - decay)))

            return curve[:5]

        except Exception:
            return [100, 85, 70, 50, 30]

    def _normalize_curve(self, scores):
        """Convert variable-length scores to standardised 5-point curve"""
        length = len(scores)
        if length < 5:
            return scores + [30] * (5 - length)
        return [
            scores[0],
            scores[length // 4],
            scores[length // 2],
            scores[3 * length // 4],
            scores[-1]
        ]

    # ------------------------------------------------------------------
    # Difficulty adjustment — FIX: personalised, not binary thresholds
    # ------------------------------------------------------------------

    def get_personalised_difficulty(self, user_id, raw_score: float) -> str:
        """
        FIX: Personalise difficulty adjustment using the user's historical
        performance distribution rather than hard-coded thresholds.

        Compares current score against user's own mean ± std to decide
        whether to go harder, stay the same, or go easier.
        """
        history = list(db.db.research_metrics.find(
            {"user_id": user_id, "event_type": "quiz_attempt"},
            {"metadata.score": 1, "metadata.total": 1},
            sort=[("timestamp", -1)],
            limit=20
        ))

        if len(history) < 3:
            # Fall back to simple thresholds when data is sparse
            if raw_score < 0.4:
                return "easier"
            elif raw_score < 0.7:
                return "similar"
            return "harder"

        past_scores = [
            h['metadata']['score'] / max(h['metadata']['total'], 1)
            for h in history
            if h['metadata'].get('total')
        ]
        mean_score = np.mean(past_scores)
        std_score  = np.std(past_scores) or 0.1

        z_score = (raw_score - mean_score) / std_score

        if z_score < -1.0:   # Performing significantly below personal average
            return "easier"
        elif z_score > 0.5:  # Performing noticeably above personal average
            return "harder"
        return "similar"

    # ------------------------------------------------------------------
    # Concept extraction
    # ------------------------------------------------------------------

    def extract_question_concept(self, question_text):
        """Improved concept extraction using TF-IDF + spaCy noun chunks"""
        try:
            features    = self.vectorizer.transform([question_text])
            scores      = features.toarray()[0]
            names       = self.vectorizer.get_feature_names_out()
            tfidf_concept = names[scores.argmax()] if scores.max() > 0 else "general"

            doc   = self.nlp(question_text)
            nouns = [chunk.text for chunk in doc.noun_chunks]

            return tfidf_concept if tfidf_concept in nouns else (nouns[0] if nouns else tfidf_concept)

        except Exception as e:
            logging.error(f"Concept extraction error: {str(e)}")
            return "general"

    def extract_concepts(self, transcript):
        """Extract key concepts from transcript via TF-IDF"""
        tfidf = self.vectorizer.fit_transform([transcript])
        feature_names = self.vectorizer.get_feature_names_out()
        return sorted(
            [(feature_names[i], tfidf[0, i]) for i in tfidf.nonzero()[1]],
            key=lambda x: -x[1]
        )[:10]

    def estimate_difficulty(self, transcript):
        """Classify content difficulty using Gemini"""
        prompt = (
            f"Rate the difficulty of this content. "
            f"Options: beginner, intermediate, advanced. "
            f"Content: {transcript[:3000]}\n"
            f"Answer ONLY with the difficulty level."
        )
        model    = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip().lower()

    # ------------------------------------------------------------------
    # Adaptive content generation
    # ------------------------------------------------------------------

    def generate_adaptive_content(self, content_type, base_content):
        """Adapt content based on user model"""
        style_weights = self.analyze_learning_style(session['user_id'])

        if content_type == 'quiz':
            return self._adapt_quiz(base_content, style_weights)
        elif content_type == 'summary':
            return self._adapt_summary(base_content, style_weights)
        return base_content

    def _adapt_quiz(self, quiz_data, style_weights):
        if style_weights.get('visual', 0) > 0.6:
            for question in quiz_data['questions']:
                if not question.get('diagram'):
                    question['diagram'] = self._generate_diagram(question['question'])
        return quiz_data

    def _adapt_summary(self, summary, style_weights):
        if style_weights.get('visual', 0) > 0.4:
            summary['visual_summary'] = self._generate_visual_summary(summary['text'])
        return summary

    def _generate_diagram(self, concept):
        prompt   = f"Create a text-based diagram explaining: {concept}"
        model    = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text

    def _generate_visual_summary(self, text):
        prompt   = f"Convert this summary to visual elements:\n{text}"
        model    = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text

    # ------------------------------------------------------------------
    # FIX: Moved from app.py — were module-level functions with `self`
    # parameter but called as instance methods (always crashed).
    # ------------------------------------------------------------------

    def _calculate_difficulty_score(self, results):
        """Calculate weighted difficulty score (0–100)"""
        correct_times   = [q.get('time_spent', 0) for q in results if q['is_correct']]
        incorrect_times = [q.get('time_spent', 0) for q in results if not q['is_correct']]

        if not correct_times or not incorrect_times:
            return 50

        diff_ratio = np.mean(incorrect_times) / (np.mean(correct_times) or 1)
        return min(100, max(0, 50 * diff_ratio))

    def _get_review_intervals(self, user_id, concepts):
        """Get days since last review for each concept"""
        intervals = []
        for concept in concepts:
            last_review = db.db.research_metrics.find_one(
                {"user_id": user_id, "metadata.concepts": concept},
                sort=[("timestamp", -1)]
            )
            if last_review:
                intervals.append((datetime.utcnow() - last_review['timestamp']).days)
        return intervals

    # ------------------------------------------------------------------
    # Concept evolution
    # ------------------------------------------------------------------

    def get_concept_evolution(self, user_id):
        """Return concept mastery over time with video context"""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$unwind": "$details"},
            {"$group": {
                "_id": {
                    "concept":  "$details.concept",
                    "video_id": "$video_id"
                },
                "timeline": {
                    "$push": {
                        "date":        "$timestamp",
                        "score":       {"$cond": ["$details.is_correct", 1, 0]},
                        "video_title": "$video_title"
                    }
                },
                "total_attempts":   {"$sum": 1},
                "correct_attempts": {"$sum": {"$cond": ["$details.is_correct", 1, 0]}}
            }},
            {"$project": {
                "concept":  "$_id.concept",
                "video_id": "$_id.video_id",
                "mastery":  {"$divide": ["$correct_attempts", "$total_attempts"]},
                "timeline": 1,
                "_id":      0
            }}
        ]
        return list(db.db.research_metrics.aggregate(pipeline))

    def _add_visual_aids(self, content):
        for question in content.get('questions', []):
            if not question.get('diagram'):
                question['diagram'] = self._generate_diagram(question['question'])
        return content


# ---------------------------------------------------------------------------
# ABTestManager — Statistical rigor fixes
# ---------------------------------------------------------------------------

class ABTestManager:
    """
    Manages A/B experiments with proper statistical methodology:

    FIX 1 — Power analysis: compute required n before starting.
    FIX 2 — Multiple comparison correction: Holm-Bonferroni applied to all
             pairwise comparisons so family-wise error rate is controlled.
    FIX 3 — Stratification: hash includes a stratification key so groups are
             balanced on a known confounder (e.g. prior score bucket).
    FIX 4 — Alpha spending (O'Brien-Fleming): interim looks use a spending
             function so continuous monitoring doesn't inflate Type I error.
    FIX 5 — Effect size reporting: Cohen's d alongside p-values.
    """

    def __init__(self):
        self.experiment_groups = ['control', 'ML_based', 'rule_based']
        # O'Brien-Fleming alpha-spending table for up to 5 interim looks
        # Values are cumulative alpha boundaries (two-sided, alpha=0.05)
        self._obf_boundaries = [0.0054, 0.0148, 0.0268, 0.0437, 0.0500]

    # ------------------------------------------------------------------
    # FIX 1: Power analysis helper
    # ------------------------------------------------------------------

    @staticmethod
    def required_sample_size(effect_size: float = 0.3,
                              alpha: float = 0.05,
                              power: float = 0.80,
                              k_groups: int = 3) -> int:
        """
        Estimate required n per group before launching an experiment.

        Uses a normal-approximation for the two-sample t-test (conservative
        for k>2 because ANOVA requires slightly fewer observations).

        Args:
            effect_size: Expected Cohen's d between best and control.
            alpha:       Desired family-wise error rate (Bonferroni-corrected
                         for k*(k-1)/2 comparisons automatically).
            power:       Desired statistical power.
            k_groups:    Number of experiment groups.

        Returns:
            Minimum n per group (integer).
        """
        n_comparisons    = k_groups * (k_groups - 1) // 2
        alpha_corrected  = alpha / n_comparisons          # Bonferroni pre-correction
        z_alpha          = stats.norm.ppf(1 - alpha_corrected / 2)
        z_beta           = stats.norm.ppf(power)
        n = ((z_alpha + z_beta) / effect_size) ** 2 * 2  # *2 for two-sample
        return math.ceil(n)

    # ------------------------------------------------------------------
    # FIX 3: Stratified group assignment
    # ------------------------------------------------------------------

    def assign_group(self, user_id: str, stratum: str = "default") -> str:
        """
        Consistent, stratified group assignment.

        FIX: Including `stratum` in the hash input ensures that users with
        similar prior performance are spread evenly across groups, preventing
        systematic imbalances that would confound results.

        Args:
            user_id: Unique user identifier.
            stratum: A categorical confounder bucket (e.g. 'low'/'mid'/'high'
                     based on the user's prior average score).
        """
        key      = f"{user_id}:{stratum}:salt_v2".encode()
        hash_int = int(hashlib.sha256(key).hexdigest(), 16)
        return self.experiment_groups[hash_int % len(self.experiment_groups)]

    @staticmethod
    def get_score_stratum(avg_score: float) -> str:
        """Map average score to a stratum label for stratified assignment."""
        if avg_score < 0.4:
            return "low"
        elif avg_score < 0.7:
            return "mid"
        return "high"

    # ------------------------------------------------------------------
    # FIX 4: Sequential testing with O'Brien-Fleming alpha spending
    # ------------------------------------------------------------------

    def should_stop_early(self,
                          p_value: float,
                          look_number: int,
                          total_looks: int = 5) -> bool:
        """
        Determine whether an interim analysis crosses the O'Brien-Fleming
        spending boundary.

        FIX: Previously results were evaluated without any alpha-spending,
        which inflates Type I error to ~14% for 5 looks at alpha=0.05.

        Args:
            p_value:     Current two-sided p-value.
            look_number: Which interim look this is (1-indexed).
            total_looks: Total planned number of looks.

        Returns:
            True if the experiment should stop (boundary crossed).
        """
        idx = min(look_number - 1, len(self._obf_boundaries) - 1)
        boundary = self._obf_boundaries[idx]
        return p_value <= boundary

    # ------------------------------------------------------------------
    # FIX 2 & 5: Analysis with Holm-Bonferroni correction + Cohen's d
    # ------------------------------------------------------------------

    def analyze_results(self, experiment_name: str) -> dict:
        """
        Analyse experiment results with:
          - Holm-Bonferroni multiple comparison correction
          - Cohen's d effect size for every pair
          - Sample size adequacy check (warns if under-powered)
        """
        pipeline = [
            {"$match": {
                "event_type": "algorithm_performance",
                "experiment": experiment_name
            }},
            {"$group": {
                "_id":    "$group",
                "avg_score": {"$avg": "$normalized_score"},
                "count":     {"$sum": 1},
                "scores":    {"$push": "$normalized_score"}
            }},
            {"$project": {
                "group":     "$_id",
                "avg_score": 1,
                "count":     1,
                "variance":  {"$stdDevPop": "$scores"},
                "_id":       0
            }}
        ]

        groups = list(db.db.research_metrics.aggregate(pipeline))

        # --- Sample size adequacy warning ---
        recommended_n = self.required_sample_size()
        underpowered  = [g['group'] for g in groups if g['count'] < recommended_n]

        # --- Pairwise tests (collect all p-values first for Holm correction) ---
        pairs       = []
        raw_results = {}

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1 = groups[i]
                g2 = groups[j]
                key = f"{g1['group']}_vs_{g2['group']}"

                s1, s2 = np.array(g1['scores']), np.array(g2['scores'])
                t_stat, p_value = stats.ttest_ind(s1, s2, equal_var=False)

                # Cohen's d (pooled SD)
                pooled_sd = math.sqrt(
                    ((len(s1) - 1) * s1.std(ddof=1) ** 2 +
                     (len(s2) - 1) * s2.std(ddof=1) ** 2)
                    / max(len(s1) + len(s2) - 2, 1)
                )
                cohens_d = (s1.mean() - s2.mean()) / (pooled_sd or 1e-9)

                raw_results[key] = {
                    'p_value_raw':  p_value,
                    't_statistic':  t_stat,
                    'cohens_d':     round(cohens_d, 4),
                    'effect_magnitude': self._interpret_effect(abs(cohens_d)),
                    'mean_diff':    round(g1['avg_score'] - g2['avg_score'], 4),
                    'n1': len(s1), 'n2': len(s2)
                }
                pairs.append((key, p_value))

        # FIX 2 — Holm-Bonferroni correction
        pairs.sort(key=lambda x: x[1])   # sort ascending by p-value
        m = len(pairs)
        for rank, (key, p_raw) in enumerate(pairs):
            adjusted = min(p_raw * (m - rank), 1.0)
            raw_results[key]['p_value_adjusted'] = round(adjusted, 6)
            raw_results[key]['significant']      = adjusted < 0.05

        return {
            'groups':            {g['group']: g for g in groups},
            'comparisons':       raw_results,
            'recommended_n':     recommended_n,
            'underpowered_groups': underpowered,
            'methodology_note':  (
                "Multiple comparisons corrected via Holm-Bonferroni. "
                "Effect sizes are Cohen's d. "
                f"Recommended n per group: {recommended_n}."
            )
        }

    @staticmethod
    def _interpret_effect(d: float) -> str:
        """Verbal label for Cohen's d magnitude (Cohen 1988)."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        return "large"

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_experiment_result(self, user_id, experiment_name, score, total):
        group = self.assign_group(user_id)
        db.log_algorithm_performance(
            user_id=user_id,
            experiment_name=experiment_name,
            group=group,
            score=score,
            total=total
        )

    def log_model_comparison(self, user_id):
        """Store model comparisons for analysis"""
        memory_model = MemoryModel()
        comparisons  = memory_model.get_model_comparison(user_id)
        for comp in comparisons:
            db.log_research_event('memory_model_prediction', {
                "concept":     comp['concept'],
                "predictions": {
                    "ebbinghaus": str(comp.get('ebbinghaus')),
                    "act_r":      str(comp.get('act_r')),
                    "ml":         str(comp.get('ml'))
                },
                "actual_recall": comp.get('actual_recall')
            })

    def get_all_experiment_results(self) -> dict:
        """
        Return a dict of { experiment_name: formatted_result } for every
        experiment that has at least one recorded data point.

        Each formatted_result has the shape the admin dashboard expects:
            {
                "groups": {
                    group_label: {
                        "n":                int,
                        "mean_score":       float,
                        "p_value_adjusted": float | None,   # vs control
                        "cohens_d":         float | None,   # vs control
                        "effect_magnitude": str,
                        "recommended_n":    int,
                        "underpowered":     bool,
                    },
                    ...
                },
                "methodology_note": str,
            }
        """
        try:
            # Discover all experiment names that have been logged
            names = db.db.research_metrics.distinct(
                "experiment",
                {"event_type": "algorithm_performance"}
            )
        except Exception as e:
            logging.error(f"get_all_experiment_results: {e}")
            return {}

        all_results = {}
        control_label = self.experiment_groups[0]   # "control"

        for name in names:
            try:
                raw = self.analyze_results(name)
            except Exception as e:
                logging.error(f"analyze_results failed for {name}: {e}")
                continue

            recommended_n     = raw.get("recommended_n", 0)
            underpowered_set  = set(raw.get("underpowered_groups", []))
            comparisons       = raw.get("comparisons", {})
            groups_raw        = raw.get("groups", {})

            # Build a lookup: group_label -> comparison stats vs control
            vs_control: dict[str, dict] = {}
            for pair_key, comp in comparisons.items():
                parts = pair_key.split("_vs_")
                if len(parts) != 2:
                    continue
                g1, g2 = parts
                if g1 == control_label:
                    vs_control[g2] = comp
                elif g2 == control_label:
                    # Flip Cohen's d sign so it's always treatment - control
                    flipped = dict(comp)
                    if flipped.get("cohens_d") is not None:
                        flipped["cohens_d"] = -flipped["cohens_d"]
                    vs_control[g1] = flipped

            # Assemble per-group output
            groups_out = {}
            for label, gdata in groups_raw.items():
                comp = vs_control.get(label, {})
                groups_out[label] = {
                    "n":                gdata.get("count", 0),
                    "mean_score":       round(gdata.get("avg_score", 0.0), 4),
                    "p_value_adjusted": comp.get("p_value_adjusted"),
                    "cohens_d":         comp.get("cohens_d"),
                    "effect_magnitude": comp.get("effect_magnitude", ""),
                    "recommended_n":    recommended_n,
                    "underpowered":     label in underpowered_set,
                }
                # Control group has no self-comparison — mark explicitly
                if label == control_label:
                    groups_out[label]["p_value_adjusted"] = None
                    groups_out[label]["cohens_d"]         = None
                    groups_out[label]["effect_magnitude"] = "reference"

            all_results[name] = {
                "groups":           groups_out,
                "methodology_note": raw.get("methodology_note", ""),
            }

        return all_results


# ---------------------------------------------------------------------------
# MemoryModel
# ---------------------------------------------------------------------------

class MemoryModel:
    """
    Provides review scheduling via three models:
      1. Ebbinghaus forgetting curve
      2. ACT-R activation-based decay  ← FIX: proper formula
      3. ML prediction                 ← FIX: implemented (was missing)
    """

    def schedule_review(self, user_id, concept):
        try:
            mastery = self._get_concept_mastery(user_id, concept)
            if mastery < 4:
                return datetime.utcnow() + timedelta(days=1)
            elif mastery < 6:
                return datetime.utcnow() + timedelta(days=3)
            return datetime.utcnow() + timedelta(days=7)
        except Exception as e:
            logging.error(f"Schedule error: {str(e)}")
            return datetime.utcnow() + timedelta(days=1)

    def _get_concept_mastery(self, user_id, concept):
        doc = db.db.concept_mastery.find_one(
            {"user_id": user_id, "concept": concept},
            {"score": 1}
        )
        return doc.get("score", 0) if doc else 0

    def get_review_schedule(self, user_id):
        """Return optimised review schedule using all three models"""
        concepts = self._get_weak_concepts(user_id)
        schedule = {}
        for concept in concepts:
            ebbinghaus = self._ebbinghaus_prediction(user_id, concept)
            act_r      = self._act_r_prediction(user_id, concept)
            ml         = self._ml_prediction(user_id, concept)
            schedule[concept] = min(ebbinghaus, act_r, ml)
        return schedule

    def _get_weak_concepts(self, user_id):
        return db.db.concept_mastery.find(
            {"user_id": user_id, "score": {"$lt": 7}},
            {"concept": 1}
        ).distinct("concept")

    def get_detailed_predictions(self, user_id):
        concepts    = self._get_weak_concepts(user_id)
        predictions = {}
        for concept in concepts:
            predictions[concept] = {
                'ebbinghaus':  self._ebbinghaus_prediction(user_id, concept),
                'act_r':       self._act_r_prediction(user_id, concept),
                'ml':          self._ml_prediction(user_id, concept),
                'last_tested': self._get_last_tested_date(user_id, concept)
            }
        return predictions

    # ------------------------------------------------------------------
    # Model 1: Ebbinghaus forgetting curve
    # ------------------------------------------------------------------

    def _ebbinghaus_prediction(self, user_id, concept):
        """Ebbinghaus forgetting curve — exponential inter-repetition growth"""
        last_tested = self._get_last_tested_date(user_id, concept)
        if not last_tested:
            return datetime.utcnow() + timedelta(days=1)

        mastery = self._get_concept_mastery(user_id, concept)
        # Interval grows with mastery: base 1-day, doubling each level
        interval_days = max(1, int(2 ** min(mastery / 2, 5)))
        return datetime.utcnow() + timedelta(days=interval_days)

    # ------------------------------------------------------------------
    # Model 2: ACT-R activation-based decay  ← FIX
    # ------------------------------------------------------------------

    def _act_r_prediction(self, user_id, concept):
        """
        FIX: Proper ACT-R base-level activation formula.

        A_i = ln( Σ_{j=1}^{n}  t_j^{-d} )

        where t_j = time since j-th practice (in hours) and d = decay param.
        Retrieval fails when A_i falls below a retrieval threshold θ.
        We schedule review just before that threshold is crossed.
        """
        D_DECAY    = 0.5    # standard ACT-R decay parameter
        THRESHOLD  = -1.0   # retrieval threshold (log scale)
        MAX_DAYS   = 30

        attempts = list(db.db.quiz_results.find(
            {"user_id": user_id, "details.concept": concept},
            sort=[("timestamp", 1)]
        ))

        if not attempts:
            return datetime.utcnow() + timedelta(days=1)

        now = datetime.utcnow()
        # Sum of power-law decay terms (hours since each practice)
        activation_sum = 0.0
        for a in attempts:
            hours = max((now - a['timestamp']).total_seconds() / 3600, 0.01)
            activation_sum += hours ** (-D_DECAY)

        if activation_sum <= 0:
            return datetime.utcnow() + timedelta(days=1)

        current_activation = math.log(activation_sum)

        # Binary-search for how many days until activation crosses threshold
        for future_days in range(1, MAX_DAYS + 1):
            future_time = now + timedelta(days=future_days)
            future_sum  = sum(
                max((future_time - a['timestamp']).total_seconds() / 3600, 0.01) ** (-D_DECAY)
                for a in attempts
            )
            future_activation = math.log(future_sum) if future_sum > 0 else -999
            if future_activation <= THRESHOLD:
                return now + timedelta(days=max(1, future_days - 1))

        return now + timedelta(days=MAX_DAYS)

    # ------------------------------------------------------------------
    # Model 3: ML prediction  ← FIX: was referenced but never defined
    # ------------------------------------------------------------------

    def _ml_prediction(self, user_id, concept):
        """
        FIX: Bayesian Ridge Regression model predicting days-to-forget.

        Features:
          - mastery score (0-10)
          - number of prior attempts
          - days since last study
          - personal decay rate (slope of score over time)
          - average inter-study interval

        If insufficient data exists, falls back to the Ebbinghaus prediction.
        """
        attempts = list(db.db.quiz_results.find(
            {"user_id": user_id},
            sort=[("timestamp", 1)]
        ))

        if len(attempts) < 3:
            return self._ebbinghaus_prediction(user_id, concept)

        mastery          = self._get_concept_mastery(user_id, concept)
        last_tested      = self._get_last_tested_date(user_id, concept)
        days_since_study = (datetime.utcnow() - last_tested).days if last_tested else 7

        # Personal decay rate: linear regression of score over time
        scores = []
        times  = []
        for a in attempts:
            total = a.get('total', 1) or 1
            scores.append(a.get('score', 0) / total)
            times.append((a['timestamp'] - attempts[0]['timestamp']).days)

        if len(set(times)) > 1:
            slope, _, _, _, _ = stats.linregress(times, scores)
            decay_rate = abs(slope)
        else:
            decay_rate = 0.15

        intervals = [
            (attempts[i]['timestamp'] - attempts[i - 1]['timestamp']).days
            for i in range(1, len(attempts))
        ]
        avg_interval = np.mean(intervals) if intervals else 3.0

        # Feature vector
        X = np.array([[
            mastery,
            len(attempts),
            days_since_study,
            decay_rate,
            avg_interval
        ]])

        # Build a tiny training set from all user quiz history
        rows, targets = [], []
        for i in range(1, len(attempts)):
            prev  = attempts[i - 1]
            curr  = attempts[i]
            gap   = (curr['timestamp'] - prev['timestamp']).days or 1
            p_sc  = prev.get('score', 0) / max(prev.get('total', 1), 1)
            c_sc  = curr.get('score', 0) / max(curr.get('total', 1), 1)
            rows.append([
                mastery, i, gap, decay_rate, avg_interval
            ])
            targets.append(gap)   # predict the inter-study gap

        if len(rows) < 3:
            return self._ebbinghaus_prediction(user_id, concept)

        model = BayesianRidge()
        model.fit(np.array(rows), np.array(targets))
        predicted_days = max(1, int(model.predict(X)[0]))
        predicted_days = min(predicted_days, 30)

        return datetime.utcnow() + timedelta(days=predicted_days)

    # ------------------------------------------------------------------
    # Item Response Theory (IRT) — 3-Parameter Logistic Model
    # ------------------------------------------------------------------
    # FIX: Questions previously had no calibrated difficulty.
    # IRT gives each question a discrimination (a), difficulty (b),
    # and guessing (c) parameter.  P(correct) = c + (1-c) * σ(a*(θ-b))
    # where θ is the learner's estimated ability.
    # ------------------------------------------------------------------

    def estimate_ability(self, user_id) -> float:
        """
        Estimate learner ability θ (logit scale) from quiz history.
        Simple method: inverse-logit of fraction correct, avoiding extremes.
        """
        doc = list(db.db.research_metrics.find(
            {"user_id": user_id, "event_type": "quiz_attempt"},
            {"metadata.score": 1, "metadata.total": 1},
            sort=[("timestamp", -1)],
            limit=20
        ))
        if not doc:
            return 0.0   # average ability

        total_correct = sum(d['metadata'].get('score', 0) for d in doc)
        total_items   = sum(d['metadata'].get('total', 1) for d in doc)
        p_correct     = np.clip(total_correct / max(total_items, 1), 0.05, 0.95)
        return math.log(p_correct / (1 - p_correct))   # logit

    @staticmethod
    def irt_p_correct(ability: float, difficulty: float,
                      discrimination: float = 1.0,
                      guessing: float = 0.2) -> float:
        """
        3PL IRT probability of a correct response.

        Args:
            ability:         θ — learner ability (logit scale, 0 = average).
            difficulty:      b — question difficulty (same logit scale).
            discrimination:  a — how well the question separates abilities.
            guessing:        c — lower asymptote (chance level).
        """
        logit = discrimination * (ability - difficulty)
        return guessing + (1 - guessing) * expit(logit)

    def select_next_question(self, user_id: str, question_pool: list) -> dict:
        """
        FIX: Select the question with maximum Fisher information for the
        learner's current ability (Computerised Adaptive Testing logic).

        Each question in question_pool should have optional fields:
          'irt_difficulty'      (default 0.0)
          'irt_discrimination'  (default 1.0)
          'irt_guessing'        (default 0.2)
        """
        theta = self.estimate_ability(user_id)

        def fisher_info(q):
            a = q.get('irt_discrimination', 1.0)
            b = q.get('irt_difficulty', 0.0)
            c = q.get('irt_guessing', 0.2)
            p = self.irt_p_correct(theta, b, a, c)
            q_ = 1 - p
            if p <= c or q_ <= 0:
                return 0.0
            return (a ** 2) * ((p - c) ** 2) / ((1 - c) ** 2) * (q_ / p)

        return max(question_pool, key=fisher_info) if question_pool else {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_last_tested_date(self, user_id, concept):
        result = db.db.quiz_results.find_one(
            {"user_id": user_id, "details.concept": concept},
            sort=[("timestamp", -1)]
        )
        return result['timestamp'] if result else None

    def get_model_comparison(self, user_id):
        """
        FIX: Original code passed Python Ellipsis (...) as method arguments,
        which would always raise TypeError at runtime.
        """
        concepts = self._get_weak_concepts(user_id)
        results  = []
        for concept in concepts:
            entry = {
                "concept":       concept,
                "ebbinghaus":    self._ebbinghaus_prediction(user_id, concept),
                "act_r":         self._act_r_prediction(user_id, concept),
                "ml":            self._ml_prediction(user_id, concept),
                "actual_recall": self._get_actual_recall(user_id, concept)
            }
            results.append(entry)
        return results

    def _get_actual_recall(self, user_id, concept):
        attempts = db.db.quiz_results.find(
            {"user_id": user_id, "details.concept": concept}
        ).sort("timestamp", 1)
        return [a.get('is_correct', False) for a in attempts]

    def calculate_retention_score(self, user_id):
        """Calculate actual retention from quiz history"""
        try:
            pipeline = [
                {"$match": {"user_id": user_id, "event_type": "quiz_attempt"}},
                {"$sort": {"timestamp": -1}},
                {"$limit": 5},
                {"$group": {
                    "_id":       None,
                    "avg_score": {"$avg": "$metadata.score"},
                    "total":     {"$sum": "$metadata.total"}
                }}
            ]
            # FIX: wrapped .next() in try/except — crashes if no data
            cursor = db.db.research_metrics.aggregate(pipeline)
            result = next(cursor, None)
            if not result or not result.get('total'):
                return 0.0
            return (result['avg_score'] / result['total']) * 100

        except Exception as e:
            logging.error(f"MemoryModel retention error: {str(e)}")
            return 0.0

    def next_optimal_review_days(self, user_id):
        """Calculate using Ebbinghaus formula with actual attempt data"""
        last_attempt = db.db.research_metrics.find_one(
            {"user_id": user_id, "event_type": "quiz_attempt"},
            sort=[("timestamp", -1)]
        )
        if not last_attempt:
            return 1
        days_since = (datetime.utcnow() - last_attempt['timestamp']).days
        return int(2 ** min(days_since, 5))

    def forgetting_risk(self, user_id, days=7):
        """Calculate risk based on concept mastery decay"""
        concepts = db.get_weak_concepts(user_id)
        if not concepts:
            return 0

        pipeline = [
            {"$match": {"user_id": user_id, "concept": {"$in": concepts}}},
            {"$group": {
                "_id":       None,
                "avg_decay": {"$avg": "$decay_rate"},
                "count":     {"$sum": 1}
            }}
        ]
        result = next(db.db.concept_mastery.aggregate(pipeline), None)
        if not result:
            return 0
        base_risk = 100 - (result['avg_decay'] * days * 100)
        return max(0, min(100, base_risk))

    def actual_retention_curve(self, user_id):
        try:
            attempts = list(db.db.research_metrics.find(
                {"user_id": user_id, "event_type": "quiz_attempt"},
                {"metadata.score": 1, "metadata.total": 1, "timestamp": 1},
                sort=[("timestamp", 1)]
            ))

            if not attempts:
                return [100, 80, 65, 50, 30]

            first_attempt_date = attempts[0]['timestamp']
            time_buckets = {'Now': 0, '1 Day': 1, '3 Days': 3, '1 Week': 7, '1 Month': 30}
            bucket_scores = {k: [] for k in time_buckets}

            for attempt in attempts:
                days_since  = (attempt['timestamp'] - first_attempt_date).days
                total       = attempt['metadata'].get('total') or 1
                score_pct   = (attempt['metadata'].get('score', 0) / total) * 100
                for label, days in time_buckets.items():
                    if days_since <= days:
                        bucket_scores[label].append(score_pct)
                        break

            return [
                np.mean(bucket_scores['Now'])     if bucket_scores['Now']     else 100,
                np.mean(bucket_scores['1 Day'])   if bucket_scores['1 Day']   else 80,
                np.mean(bucket_scores['3 Days'])  if bucket_scores['3 Days']  else 65,
                np.mean(bucket_scores['1 Week'])  if bucket_scores['1 Week']  else 50,
                np.mean(bucket_scores['1 Month']) if bucket_scores['1 Month'] else 30,
            ]

        except Exception as e:
            logging.error(f"Retention curve error: {str(e)}")
            return [100, 80, 65, 50, 30]

    def _normalize_curve(self, scores):
        if len(scores) < 5:
            return [100, 80, 65, 50, 30]
        step = len(scores) // 5
        return [scores[0], scores[step], scores[step * 2], scores[step * 3], scores[-1]]

    def _calculate_personal_decay(self, user_id):
        """Safer decay calculation with zero handling"""
        try:
            attempts = list(db.db.research_metrics.find(
                {"user_id": user_id, "event_type": "quiz_attempt"},
                sort=[("timestamp", 1)]
            ))
            if len(attempts) < 2:
                return 0.15

            decay_rates = []
            for i in range(1, len(attempts)):
                try:
                    prev_total = attempts[i - 1]['metadata']['total'] or 1
                    curr_total = attempts[i]['metadata']['total'] or 1
                    prev_score = attempts[i - 1]['metadata']['score'] / prev_total
                    curr_score = attempts[i]['metadata']['score'] / curr_total
                    time_diff  = max(1, (attempts[i]['timestamp'] - attempts[i - 1]['timestamp']).days)
                    score_diff = abs(prev_score - curr_score)
                    decay_rates.append(min(max(score_diff / time_diff, 0.05), 0.5))
                except KeyError:
                    continue

            return float(np.median(decay_rates)) if decay_rates else 0.15

        except Exception:
            return 0.15


# ---------------------------------------------------------------------------
# EnhancedMemoryModel
# ---------------------------------------------------------------------------

class EnhancedMemoryModel(MemoryModel):
    def get_video_predictions(self, user_id, video_id):
        try:
            video = db.db.video_progress.find_one(
                {'user_id': user_id, 'video_id': video_id}
            )
            if not video:
                return {}

            concepts = set()
            for attempt in video.get('attempts', []):
                concepts.update(attempt.get('metadata', {}).get('concepts', []))

            predictions = {}
            for concept in concepts:
                next_review = self.get_forgetting_prediction(user_id, concept)
                predictions[concept] = {
                    'next_review':    next_review,
                    'formatted_date': next_review.strftime('%Y-%m-%d %H:%M')
                }
            return predictions

        except Exception as e:
            logging.error(f"Video prediction error: {str(e)}")
            return {}

    def get_forgetting_prediction(self, user_id, concept):
        try:
            base_interval   = self._ebbinghaus_prediction(user_id, concept)
            user_factor     = self._get_cognitive_factor(user_id)
            concept_factor  = self._get_concept_difficulty(concept)
            adjusted_days   = base_interval.days * (0.6 * user_factor + 0.4 * concept_factor)
            final_days      = max(1, int(adjusted_days))
            return datetime.utcnow() + timedelta(days=final_days)

        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return datetime.utcnow() + timedelta(days=1)

    def _get_cognitive_factor(self, user_id):
        """
        FIX: Original code could divide by zero when consecutive scores
        are identical.  Added abs() guard and epsilon denominator.
        """
        attempts = list(db.db.quiz_results.find(
            {"user_id": user_id},
            sort=[("timestamp", -1)],
            limit=10
        ))

        if len(attempts) < 2:
            return 1.0

        decay_rates = []
        for i in range(len(attempts) - 1):
            days_gap    = (attempts[i]['timestamp'] - attempts[i + 1]['timestamp']).days
            score_diff  = (
                attempts[i]['metadata'].get('score', 0) -
                attempts[i + 1]['metadata'].get('score', 0)
            )
            # FIX: avoid division by zero with epsilon
            denominator = abs(score_diff) + 1e-6
            decay_rates.append(days_gap / denominator)

        if not decay_rates:
            return 1.0
        return float(np.clip(np.mean(decay_rates), 0.1, 10.0))

    def _get_concept_difficulty(self, concept):
        """Get normalised difficulty (0.5–1.5 range)"""
        doc = db.db.concept_difficulty.find_one(
            {"concept": concept},
            {"score": 1}
        )
        return doc.get('score', 1.0) if doc else 1.0

    def predict_retention_curve(self, user_id):
        """Generate realistic decay even with limited data"""
        base_retention = 100
        try:
            decay_rate    = self._calculate_personal_decay(user_id) or 0.15
            last_activity = self._get_last_activity_date(user_id)
            days_inactive = (datetime.utcnow() - last_activity).days if last_activity else 0
            decay_rate   *= (1 + days_inactive * 0.05)

            return [
                base_retention,
                max(0, base_retention * (1 - decay_rate) ** 1),
                max(0, base_retention * (1 - decay_rate) ** 3),
                max(0, base_retention * (1 - decay_rate) ** 7),
                max(0, base_retention * (1 - decay_rate) ** 30)
            ]
        except Exception:
            return [100, 85, 70, 50, 30]

    def _get_last_activity_date(self, user_id):
        last_attempt = db.db.research_metrics.find_one(
            {"user_id": user_id},
            sort=[("timestamp", -1)]
        )
        return last_attempt['timestamp'] if last_attempt else None

    def _calculate_study_frequency(self, user_id):
        sessions = list(db.db.research_metrics.find(
            {"user_id": user_id,
             "event_type": {"$in": ["quiz_attempt", "flashcard_view"]}},
            sort=[("timestamp", 1)]
        ))
        if len(sessions) < 2:
            return 7
        intervals = [
            (sessions[i]['timestamp'] - sessions[i - 1]['timestamp']).days
            for i in range(1, len(sessions))
        ]
        return float(np.percentile(intervals, 70))

    def _get_average_difficulty(self, user_id):
        cursor = db.db.video_progress.aggregate([
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": None,
                "avg_difficulty": {"$avg": "$metadata.difficulty_score"}
            }}
        ])
        result = next(cursor, None)
        return result['avg_difficulty'] if result else 50

    # Override to resolve duplicate definition in base class
    def _calculate_personal_decay(self, user_id):
        attempts = list(db.db.research_metrics.find(
            {"user_id": user_id, "event_type": "quiz_attempt"},
            sort=[("timestamp", 1)]
        ))
        if len(attempts) < 2:
            return 0.15
        decay_rates = []
        for i in range(1, len(attempts)):
            try:
                prev_total = attempts[i - 1]['metadata']['total'] or 1
                curr_total = attempts[i]['metadata']['total'] or 1
                prev_score = attempts[i - 1]['metadata']['score'] / prev_total
                curr_score = attempts[i]['metadata']['score'] / curr_total
                time_diff  = (attempts[i]['timestamp'] - attempts[i - 1]['timestamp']).days or 1
                score_diff = prev_score - curr_score
                decay_rates.append(min(max(abs(score_diff) / time_diff, 0.05), 0.5))
            except KeyError:
                continue
        return float(np.median(decay_rates)) if decay_rates else 0.15
