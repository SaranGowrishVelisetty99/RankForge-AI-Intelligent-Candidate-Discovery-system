import unittest
from unittest.mock import patch

from config.constants import DEFAULT_WEIGHTS
from models.candidate import Candidate, CandidateFeatures
from ranking.final_ranker import final_ranker
from ranking.penalties import compute_penalty_multiplier
from ranking.ranker import ranking_engine


def make_test_features(candidate_id: str, **overrides) -> CandidateFeatures:
    feat = CandidateFeatures(
        candidate_id=candidate_id,
        jd_fit=80.0,
        evidence=70.0,
        credibility=65.0,
        trajectory=75.0,
        recruitability=60.0,
        availability=80.0,
        authenticity=90.0,
        consistency=85.0,
    )
    for key, value in overrides.items():
        setattr(feat, key, value)
    return feat


class TestRankingEngine(unittest.TestCase):
    def test_weighted_scoring(self):
        feat = make_test_features("CAND_0001")
        score = ranking_engine._compute_weighted_score(feat)
        expected = sum(
            getattr(feat, dim) * weight
            for dim, weight in DEFAULT_WEIGHTS.items()
        )
        self.assertAlmostEqual(score, expected)

    def test_ranking_returns_top_k(self):
        from tests.test_agents import make_test_candidate
        candidates = []
        features = []
        for i in range(10):
            candidates.append(make_test_candidate(candidate_id=f"CAND_{i:07d}"))
            candidates[-1].candidate_id = f"CAND_{i:07d}"
            features.append(make_test_features(f"CAND_{i:07d}", jd_fit=float(100 - i)))

        result = ranking_engine.rank(features, candidates)
        self.assertLessEqual(len(result), 300)
        self.assertEqual(len(result), 10)

    def test_ranking_order(self):
        from tests.test_agents import make_test_candidate
        candidates = []
        features = []
        for i in range(5):
            candidates.append(make_test_candidate(candidate_id=f"CAND_{i:07d}"))
            candidates[-1].candidate_id = f"CAND_{i:07d}"
            features.append(make_test_features(f"CAND_{i:07d}", jd_fit=float(80 - i * 10)))

        result = ranking_engine.rank(features, candidates)
        for i in range(len(result) - 1):
            self.assertGreaterEqual(
                result[i].ranking_score, result[i + 1].ranking_score
            )


class TestFinalRanking(unittest.TestCase):
    def test_final_rerank(self):
        features = [
            make_test_features("CAND_0001", ranking_score=90.0),
            make_test_features("CAND_0002", ranking_score=80.0),
            make_test_features("CAND_0003", ranking_score=70.0),
        ]
        reviews = [
            {"review_score": 85.0},
            {"review_score": 75.0},
            {"review_score": 65.0},
        ]

        result = final_ranker.rerank(features, reviews)
        self.assertLessEqual(len(result), 100)
        self.assertEqual(result[0].rank, 1)
        self.assertEqual(result[-1].rank, len(result))


class TestPenalties(unittest.TestCase):
    def test_no_penalty_normal(self):
        from tests.test_agents import make_test_candidate
        candidate = make_test_candidate()
        multiplier = compute_penalty_multiplier(candidate)
        self.assertAlmostEqual(multiplier, 1.0)

    def test_honeypot_penalty(self):
        candidate = Candidate(
            candidate_id="CAND_HONEY",
            skills=[],
            career_history=[],
            redrob_signals__github_activity_score=95,
            redrob_signals__profile_completeness_score=20,
        )
        # Should not crash
        multiplier = compute_penalty_multiplier(candidate)
        self.assertGreaterEqual(multiplier, 0)


if __name__ == "__main__":
    unittest.main()
