import unittest
from unittest.mock import patch

from agents.authenticity import AuthenticityAgent
from agents.availability import AvailabilityAgent
from agents.contradiction import ContradictionAgent
from agents.credibility import CredibilityAgent
from agents.evidence import EvidenceAgent
from agents.jd_fit import JDFitAgent
from agents.recruitability import RecruitabilityAgent
from agents.trajectory import TrajectoryAgent
from models.candidate import Candidate, CareerEntry, Profile, RedrobSignals, Skill
from models.jd import JDRequirements


def make_test_candidate(**overrides) -> Candidate:
    candidate = Candidate(
        candidate_id="CAND_0000001",
        profile=Profile(
            anonymized_name="Test Candidate",
            headline="Senior ML Engineer with NLP expertise",
            summary="Experienced ML engineer building ranking and retrieval systems.",
            location="Bangalore",
            country="India",
            years_of_experience=6.0,
            current_title="Senior Machine Learning Engineer",
            current_company="TechCorp",
            current_company_size="1001-5000",
            current_industry="Technology",
        ),
        career_history=[
            CareerEntry(
                company="TechCorp",
                title="Senior ML Engineer",
                start_date="2021-01-01",
                end_date=None,
                duration_months=36,
                is_current=True,
                industry="Technology",
                company_size="1001-5000",
                description="Built ranking models and retrieval systems using PyTorch and FAISS.",
            ),
            CareerEntry(
                company="DataCo",
                title="Machine Learning Engineer",
                start_date="2018-06-01",
                end_date="2020-12-31",
                duration_months=30,
                is_current=False,
                industry="Technology",
                company_size="501-1000",
                description="Developed NLP pipelines and search algorithms.",
            ),
        ],
        skills=[
            Skill(name="Python", proficiency="expert", endorsements=30, duration_months=72),
            Skill(name="PyTorch", proficiency="advanced", endorsements=15, duration_months=48),
            Skill(name="FAISS", proficiency="advanced", endorsements=8, duration_months=24),
            Skill(name="NLP", proficiency="advanced", endorsements=20, duration_months=36),
        ],
        redrob_signals=RedrobSignals(
            profile_completeness_score=90.0,
            last_active_date="2025-06-15",
            open_to_work_flag=True,
            profile_views_received_30d=40,
            recruiter_response_rate=0.8,
            interview_completion_rate=0.85,
            offer_acceptance_rate=0.7,
            saved_by_recruiters_30d=8,
            search_appearance_30d=50,
            skill_assessment_scores={"Python": 95, "PyTorch": 82, "NLP": 75},
            notice_period_days=30,
            willing_to_relocate=True,
            preferred_work_mode="remote",
            github_activity_score=80.0,
            connection_count=500,
            endorsements_received=150,
            verified_email=True,
            verified_phone=True,
            linkedin_connected=True,
            applications_submitted_30d=3,
            avg_response_time_hours=5.0,
            expected_salary_range_inr_lpa={"min": 40, "max": 60},
        ),
    )

    for key, value in overrides.items():
        parts = key.split(".")
        obj = candidate
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    return candidate


def make_test_jd() -> JDRequirements:
    return JDRequirements(
        must_have=["python", "machine learning", "nlp", "retrieval", "ranking", "pytorch"],
        preferred=["fastapi", "kubernetes", "spark"],
        seniority="senior",
        experience_required=5,
        domain_requirements=["information retrieval", "search"],
        behavioral_traits=["analytical", "collaborative"],
        weighting_strategy={"technical": 60, "experience": 25, "culture": 15},
    )


class TestJDFitAgent(unittest.TestCase):
    def setUp(self):
        self.agent = JDFitAgent()
        self.candidate = make_test_candidate()
        self.jd = make_test_jd()
        import numpy as np
        self.jd.jd_embedding = np.zeros(1463, dtype=np.float32).tolist()

    @patch("agents.jd_fit.embedding_service.encode")
    def test_jd_fit_score_range(self, mock_encode):
        import numpy as np
        mock_encode.return_value = np.random.randn(1, 1463).astype(np.float32)

        result = self.agent.score(self.candidate, self.jd)
        self.assertIn("jd_fit_score", result)
        score = result["jd_fit_score"]
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)


class TestEvidenceAgent(unittest.TestCase):
    def setUp(self):
        self.agent = EvidenceAgent()
        self.candidate = make_test_candidate()
        self.jd = make_test_jd()

    def test_evidence_validation(self):
        result = self.agent.score(self.candidate, self.jd)
        self.assertIn("evidence_score", result)
        self.assertIn("validated_skills", result)
        self.assertIn("unsupported_skills", result)
        self.assertGreaterEqual(result["evidence_score"], 0)
        self.assertLessEqual(result["evidence_score"], 100)


class TestCredibilityAgent(unittest.TestCase):
    def setUp(self):
        self.agent = CredibilityAgent()
        self.candidate = make_test_candidate()
        self.jd = make_test_jd()

    def test_credibility_score_range(self):
        result = self.agent.score(self.candidate, self.jd)
        self.assertIn("credibility_score", result)
        self.assertGreaterEqual(result["credibility_score"], 0)
        self.assertLessEqual(result["credibility_score"], 100)

    def test_empty_skills(self):
        empty_candidate = make_test_candidate()
        empty_candidate.skills = []
        result = self.agent.score(empty_candidate, self.jd)
        self.assertEqual(result["credibility_score"], 0)


class TestTrajectoryAgent(unittest.TestCase):
    def setUp(self):
        self.agent = TrajectoryAgent()
        self.candidate = make_test_candidate()
        self.jd = make_test_jd()

    def test_trajectory_score_range(self):
        result = self.agent.score(self.candidate, self.jd)
        self.assertIn("trajectory_score", result)
        self.assertGreaterEqual(result["trajectory_score"], 0)
        self.assertLessEqual(result["trajectory_score"], 100)


class TestRecruitabilityAgent(unittest.TestCase):
    def setUp(self):
        self.agent = RecruitabilityAgent()
        self.candidate = make_test_candidate()
        self.jd = make_test_jd()

    def test_recruitability_score_range(self):
        result = self.agent.score(self.candidate, self.jd)
        self.assertIn("recruitability_score", result)
        self.assertGreaterEqual(result["recruitability_score"], 0)
        self.assertLessEqual(result["recruitability_score"], 100)


class TestAvailabilityAgent(unittest.TestCase):
    def setUp(self):
        self.agent = AvailabilityAgent()
        self.jd = make_test_jd()

    def test_availability_open_to_work(self):
        candidate = make_test_candidate()
        result = self.agent.score(candidate, self.jd)
        self.assertIn("availability_score", result)
        self.assertGreater(result["availability_score"], 50)

    def test_availability_not_open_to_work(self):
        candidate = make_test_candidate()
        candidate.redrob_signals.open_to_work_flag = False
        candidate.redrob_signals.notice_period_days = 90
        result = self.agent.score(candidate, self.jd)
        self.assertLess(result["availability_score"], 50)


class TestAuthenticityAgent(unittest.TestCase):
    def setUp(self):
        self.agent = AuthenticityAgent()
        self.jd = make_test_jd()

    def test_normal_candidate_low_risk(self):
        candidate = make_test_candidate()
        result = self.agent.score(candidate, self.jd)
        self.assertEqual(result["risk_level"], "low")

    def test_suspicious_candidate_high_risk(self):
        candidate = make_test_candidate()
        candidate.skills = [
            Skill(name=f"Skill_{i}", proficiency="beginner", endorsements=0, duration_months=0)
            for i in range(30)
        ]
        candidate.redrob_signals.github_activity_score = -1
        candidate.redrob_signals.profile_completeness_score = 25
        result = self.agent.score(candidate, self.jd)
        self.assertIn(result["risk_level"], ["medium", "high"])


class TestContradictionAgent(unittest.TestCase):
    def setUp(self):
        self.agent = ContradictionAgent()
        self.jd = make_test_jd()

    def test_consistent_candidate(self):
        candidate = make_test_candidate()
        result = self.agent.score(candidate, self.jd)
        self.assertIn("consistency_score", result)
        self.assertGreaterEqual(result["consistency_score"], 50)

    def test_inconsistent_candidate(self):
        candidate = make_test_candidate()
        candidate.skills.append(
            Skill(name="Speech Recognition", proficiency="advanced", endorsements=0, duration_months=0)
        )
        candidate.career_history[0].description = "Built Spark and Kafka data pipelines"
        result = self.agent.score(candidate, self.jd)
        self.assertIn("consistency_score", result)


if __name__ == "__main__":
    unittest.main()
