"""
Unit tests for the three core algorithms:

  1. Bayesian Knowledge Tracing (BKT)        — adaptive_learning.LearningAnalyzer
  2. Item Response Theory — 3PL model (IRT)  — adaptive_learning.EnhancedMemoryModel
  3. A/B test statistical rigor              — adaptive_learning.ABTestManager

These tests are intentionally isolated from MongoDB so they run anywhere
without a live database.  Integration tests (which need the DB) live in
retention_tests.py.

Run with:
    python -m unittest test_algorithms.py
"""

import math
import unittest
from unittest.mock import MagicMock, patch

import sys

# Stub out every module that touches Flask / MongoDB / spaCy / Gemini
# so the import succeeds in a plain Python environment.
_db_stub = MagicMock()
_flask_stub = MagicMock()
_spacy_stub = MagicMock()
_genai_stub = MagicMock()

sys.modules.setdefault('db',                        _db_stub)
sys.modules.setdefault('flask',                     _flask_stub)
sys.modules.setdefault('spacy',                     _spacy_stub)
sys.modules.setdefault('google',                    MagicMock())
sys.modules.setdefault('google.generativeai',       _genai_stub)
sys.modules.setdefault('bson',                      MagicMock())

from adaptive_learning import LearningAnalyzer, EnhancedMemoryModel, ABTestManager  # noqa: E402


# ===========================================================================
# 1. Bayesian Knowledge Tracing (BKT)
# ===========================================================================

class TestBKTUpdate(unittest.TestCase):
    """
    Tests for LearningAnalyzer._bkt_update.

    The BKT model maintains P(mastered) as a latent probability and updates
    it via Bayes' rule after each observation:

        P(mastered | correct) ∝ P(correct | mastered) × P(mastered)
                               = (1 − p_s) × p_mastery

    After the Bayes update, a learning-transition term is applied:
        p_posterior += (1 − p_posterior) × p_t
    """

    def setUp(self):
        # Patch spaCy model load so __init__ doesn't crash
        with patch('spacy.load', return_value=MagicMock()):
            self.analyzer = LearningAnalyzer()

    # ── Basic direction tests ────────────────────────────────────────────

    def test_correct_answer_increases_mastery(self):
        """A correct answer must raise P(mastered)."""
        initial = 0.1
        updated = self.analyzer._bkt_update(initial, is_correct=True)
        self.assertGreater(updated, initial,
            "Correct answer should increase mastery probability")

    def test_incorrect_answer_decreases_mastery(self):
        """An incorrect answer must lower P(mastered)."""
        initial = 0.9
        updated = self.analyzer._bkt_update(initial, is_correct=False)
        self.assertLess(updated, initial,
            "Incorrect answer should decrease mastery probability")

    # ── Numerical precision tests ────────────────────────────────────────

    def test_correct_from_low_prior_known_value(self):
        """
        Manual calculation:
          p_obs_know = 1 − 0.1  = 0.9   (1 − p_s)
          p_obs_not  = 0.2               (p_g)
          numerator  = 0.9 × 0.1 = 0.09
          denominator= 0.09 + 0.2 × 0.9 = 0.27
          posterior  = 0.09 / 0.27 ≈ 0.333
          with transition: 0.333 + 0.667 × 0.1 ≈ 0.400
        """
        result = self.analyzer._bkt_update(
            p_mastery=0.1, is_correct=True, p_g=0.2, p_s=0.1, p_t=0.1
        )
        self.assertAlmostEqual(result, 0.400, delta=0.01,
            msg="BKT update value differs from hand-calculated result")

    def test_incorrect_from_high_prior_known_value(self):
        """
        Manual calculation:
          p_obs_know = 0.1               (p_s)
          p_obs_not  = 1 − 0.2 = 0.8    (1 − p_g)
          numerator  = 0.1 × 0.9 = 0.09
          denominator= 0.09 + 0.8 × 0.1 = 0.17
          posterior  = 0.09 / 0.17 ≈ 0.529
          with transition: 0.529 + 0.471 × 0.1 ≈ 0.576
        """
        result = self.analyzer._bkt_update(
            p_mastery=0.9, is_correct=False, p_g=0.2, p_s=0.1, p_t=0.1
        )
        self.assertAlmostEqual(result, 0.576, delta=0.01,
            msg="BKT update value differs from hand-calculated result")

    # ── Boundary & convergence tests ─────────────────────────────────────

    def test_output_always_in_0_1(self):
        """P(mastered) must remain a valid probability after any update."""
        for p in [0.0, 0.01, 0.5, 0.99, 1.0]:
            for correct in [True, False]:
                result = self.analyzer._bkt_update(p, is_correct=correct)
                self.assertGreaterEqual(result, 0.0)
                self.assertLessEqual(result, 1.0)

    def test_many_correct_answers_converge_to_high_mastery(self):
        """Repeated correct answers should drive mastery close to 1."""
        p = 0.1
        for _ in range(30):
            p = self.analyzer._bkt_update(p, is_correct=True)
        self.assertGreater(p, 0.9,
            "30 correct answers should yield mastery > 0.9")

    def test_many_incorrect_answers_stay_low(self):
        """Repeated incorrect answers should keep mastery low."""
        p = 0.5
        for _ in range(30):
            p = self.analyzer._bkt_update(p, is_correct=False)
        # p_t ensures mastery never reaches absolute zero — stays in (0, ~0.3)
        self.assertLess(p, 0.35,
            "30 incorrect answers should yield mastery < 0.35")

    def test_calculate_mastery_from_results_returns_dict(self):
        """calculate_mastery_from_results should return a concept → float dict."""
        with patch.object(self.analyzer, 'extract_question_concept',
                          return_value='photosynthesis'):
            results = [
                {'question': 'What is X?', 'is_correct': True},
                {'question': 'What is X?', 'is_correct': False},
                {'question': 'What is X?', 'is_correct': True},
            ]
            mastery = self.analyzer.calculate_mastery_from_results(results)

        self.assertIsInstance(mastery, dict)
        self.assertIn('photosynthesis', mastery)
        p = mastery['photosynthesis']
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_mastery_order_correct_gt_incorrect(self):
        """
        A user who answered only correctly should have higher mastery than
        one who answered only incorrectly on the same concept.
        """
        with patch.object(self.analyzer, 'extract_question_concept',
                          return_value='concept_a'):
            all_correct = self.analyzer.calculate_mastery_from_results([
                {'question': 'Q', 'is_correct': True} for _ in range(5)
            ])
            all_wrong = self.analyzer.calculate_mastery_from_results([
                {'question': 'Q', 'is_correct': False} for _ in range(5)
            ])

        self.assertGreater(
            all_correct['concept_a'],
            all_wrong['concept_a'],
            "All-correct mastery should exceed all-incorrect mastery"
        )


# ===========================================================================
# 2. Item Response Theory — 3-Parameter Logistic Model
# ===========================================================================

class TestIRT(unittest.TestCase):
    """
    Tests for EnhancedMemoryModel.irt_p_correct and select_next_question.

    The 3PL IRT model computes:
        P(correct) = c + (1 − c) × σ(a × (θ − b))

    where θ = ability, b = difficulty, a = discrimination, c = guessing.
    σ is the logistic sigmoid (expit).
    """

    def setUp(self):
        with patch('spacy.load', return_value=MagicMock()):
            self.model = EnhancedMemoryModel()

    # ── Known-value tests ────────────────────────────────────────────────

    def test_average_ability_average_difficulty(self):
        """
        θ = b = 0, a = 1, c = 0.2:
          σ(1 × (0 − 0)) = σ(0) = 0.5
          P = 0.2 + 0.8 × 0.5 = 0.6
        """
        p = self.model.irt_p_correct(
            ability=0.0, difficulty=0.0,
            discrimination=1.0, guessing=0.2
        )
        self.assertAlmostEqual(p, 0.6, delta=0.001,
            msg="3PL IRT at θ=b=0 should give p=0.6")

    def test_high_ability_easy_question(self):
        """High-ability learner on an easy question → near-certain correct."""
        p = self.model.irt_p_correct(
            ability=3.0, difficulty=-1.0,
            discrimination=1.0, guessing=0.2
        )
        self.assertGreater(p, 0.95,
            "Very high ability on easy question should give p > 0.95")

    def test_low_ability_hard_question(self):
        """Low-ability learner on a hard question → near guessing level."""
        p = self.model.irt_p_correct(
            ability=-3.0, difficulty=2.0,
            discrimination=1.0, guessing=0.2
        )
        self.assertAlmostEqual(p, 0.2, delta=0.03,
            msg="Very low ability on hard question should be close to guessing level")

    def test_guessing_is_lower_asymptote(self):
        """P(correct) must never fall below the guessing parameter c."""
        for guessing in [0.1, 0.2, 0.25]:
            p = self.model.irt_p_correct(
                ability=-10.0, difficulty=10.0,
                discrimination=2.0, guessing=guessing
            )
            self.assertGreaterEqual(p, guessing - 0.001,
                f"P(correct) must be ≥ guessing parameter c={guessing}")

    def test_output_is_valid_probability(self):
        """P(correct) must always be in [0, 1]."""
        test_cases = [
            (3.0, 0.0, 1.0, 0.2),
            (-3.0, 0.0, 1.0, 0.2),
            (0.0, 3.0, 2.0, 0.25),
            (0.0, -3.0, 0.5, 0.0),
        ]
        for ability, difficulty, disc, guess in test_cases:
            p = self.model.irt_p_correct(ability, difficulty, disc, guess)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_higher_discrimination_steeper_curve(self):
        """
        A higher discrimination parameter means the ICC (item characteristic
        curve) is steeper — the gap in P(correct) between θ slightly above
        and slightly below b is larger.
        """
        delta_low  = (self.model.irt_p_correct(0.5, 0.0, discrimination=0.5) -
                      self.model.irt_p_correct(-0.5, 0.0, discrimination=0.5))
        delta_high = (self.model.irt_p_correct(0.5, 0.0, discrimination=2.0) -
                      self.model.irt_p_correct(-0.5, 0.0, discrimination=2.0))
        self.assertGreater(delta_high, delta_low,
            "Higher discrimination should produce steeper probability curve")

    # ── Adaptive question selection ──────────────────────────────────────

    def test_select_next_question_returns_from_pool(self):
        """Selected question must be one of the pool items."""
        pool = [
            {'id': 1, 'irt_difficulty': -1.0},
            {'id': 2, 'irt_difficulty':  0.0},
            {'id': 3, 'irt_difficulty':  1.0},
        ]
        with patch.object(self.model, 'estimate_ability', return_value=0.0):
            chosen = self.model.select_next_question('user_x', pool)
        self.assertIn(chosen, pool,
            "Selected question must belong to the pool")

    def test_select_next_question_matches_ability(self):
        """
        Fisher information is maximised when question difficulty ≈ ability.
        For θ = 0, the question with b = 0 should be selected over b = ±2.
        """
        pool = [
            {'id': 'easy',   'irt_difficulty': -2.0},
            {'id': 'medium', 'irt_difficulty':  0.0},
            {'id': 'hard',   'irt_difficulty':  2.0},
        ]
        with patch.object(self.model, 'estimate_ability', return_value=0.0):
            chosen = self.model.select_next_question('user_x', pool)
        self.assertEqual(chosen['id'], 'medium',
            "For θ=0, question with b=0 maximises Fisher information")

    def test_select_next_question_handles_empty_pool(self):
        """An empty pool should return an empty dict, not raise."""
        with patch.object(self.model, 'estimate_ability', return_value=0.0):
            result = self.model.select_next_question('user_x', [])
        self.assertEqual(result, {})


# ===========================================================================
# 3. A/B Test Statistical Rigor
# ===========================================================================

class TestABTestStatistics(unittest.TestCase):
    """
    Tests for ABTestManager.required_sample_size and statistical analysis.

    The manager uses:
      - Power analysis (normal approximation) before starting experiments
      - Holm–Bonferroni correction for multiple comparisons
      - O'Brien–Fleming alpha spending for interim looks
      - Cohen's d for effect size reporting
    """

    def setUp(self):
        self.manager = ABTestManager()

    # ── Power analysis ───────────────────────────────────────────────────

    def test_required_sample_size_returns_positive_int(self):
        """required_sample_size must always return a positive integer."""
        n = ABTestManager.required_sample_size(effect_size=0.5)
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)

    def test_larger_effect_needs_fewer_participants(self):
        """Larger effect size → need fewer participants to detect it."""
        n_small  = ABTestManager.required_sample_size(effect_size=0.2)
        n_medium = ABTestManager.required_sample_size(effect_size=0.5)
        n_large  = ABTestManager.required_sample_size(effect_size=0.8)
        self.assertGreater(n_small, n_medium,
            "Small effect requires more participants than medium effect")
        self.assertGreater(n_medium, n_large,
            "Medium effect requires more participants than large effect")

    def test_higher_power_needs_more_participants(self):
        """Higher desired power → need more participants."""
        n_80 = ABTestManager.required_sample_size(effect_size=0.5, power=0.80)
        n_90 = ABTestManager.required_sample_size(effect_size=0.5, power=0.90)
        self.assertGreater(n_90, n_80,
            "90% power requires more participants than 80% power")

    def test_stricter_alpha_needs_more_participants(self):
        """Stricter (lower) alpha → need more participants."""
        n_05 = ABTestManager.required_sample_size(effect_size=0.5, alpha=0.05)
        n_01 = ABTestManager.required_sample_size(effect_size=0.5, alpha=0.01)
        self.assertGreater(n_01, n_05,
            "α=0.01 requires more participants than α=0.05")

    def test_more_groups_needs_more_participants(self):
        """More experimental groups → Bonferroni correction is stricter → need more."""
        n_2_groups = ABTestManager.required_sample_size(effect_size=0.5, k_groups=2)
        n_4_groups = ABTestManager.required_sample_size(effect_size=0.5, k_groups=4)
        self.assertGreater(n_4_groups, n_2_groups,
            "More comparison groups require larger sample per group")

    def test_sample_size_realistic_range(self):
        """
        For a medium effect (d=0.5), 80% power, α=0.05, 3 groups:
        the answer should be in a plausible ballpark (roughly 60–200 per group).
        """
        n = ABTestManager.required_sample_size(
            effect_size=0.5, alpha=0.05, power=0.80, k_groups=3
        )
        self.assertGreater(n, 40,  "n per group seems unrealistically low")
        self.assertLess(n, 500,    "n per group seems unrealistically high")

    # ── OBrien-Fleming alpha spending boundaries ──────────────────────────

    def test_obf_boundaries_ordered_and_bounded(self):
        """
        O'Brien-Fleming cumulative alpha boundaries must be:
          - strictly increasing (each look allows slightly more alpha)
          - final boundary ≤ overall alpha (0.05)
        """
        boundaries = self.manager._obf_boundaries
        for i in range(1, len(boundaries)):
            self.assertGreater(boundaries[i], boundaries[i - 1],
                "OBF boundaries must be strictly increasing")
        self.assertLessEqual(boundaries[-1], 0.05,
            "Final OBF boundary must not exceed nominal alpha")

    def test_obf_first_boundary_very_small(self):
        """
        The first interim look uses a very conservative threshold so
        we don't stop too early.  It should be well below 0.01.
        """
        self.assertLess(self.manager._obf_boundaries[0], 0.01,
            "First OBF boundary should be very conservative (< 0.01)")

    # ── Group assignment ──────────────────────────────────────────────────

    def test_assign_group_returns_valid_group(self):
        """Every user must be assigned to one of the defined experiment groups."""
        for uid in ['user_001', 'user_002', 'anon_abc', 'test_xyz']:
            group = self.manager.assign_group(uid)
            self.assertIn(group, self.manager.experiment_groups,
                f"User {uid!r} assigned to unknown group {group!r}")

    def test_same_user_same_group(self):
        """Assignment must be deterministic — same user always gets same group."""
        uid = 'stable_user_999'
        group_a = self.manager.assign_group(uid)
        group_b = self.manager.assign_group(uid)
        self.assertEqual(group_a, group_b,
            "Group assignment must be deterministic for the same user ID")

    def test_assignment_distributes_across_groups(self):
        """
        With enough users, all groups should receive at least one assignment.
        This catches regressions where the hash function maps everyone to
        the same bucket.
        """
        assigned = {self.manager.assign_group(f'u_{i}') for i in range(300)}
        self.assertEqual(
            assigned, set(self.manager.experiment_groups),
            "All experiment groups should be reachable via assign_group"
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)
