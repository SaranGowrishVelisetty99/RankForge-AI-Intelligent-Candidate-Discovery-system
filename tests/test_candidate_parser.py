import json
import os
import tempfile
import unittest

from core.candidate_parser import load_candidates, parse_candidate


class TestCandidateParser(unittest.TestCase):
    def setUp(self):
        self.sample_line = {
            "candidate_id": "CAND_0000001",
            "profile": {
                "anonymized_name": "Candidate 1",
                "headline": "Senior ML Engineer at TechCorp",
                "summary": "Experienced ML engineer with 5+ years in NLP and ranking systems.",
                "location": "Bangalore",
                "country": "India",
                "years_of_experience": 5.0,
                "current_title": "Senior Machine Learning Engineer",
                "current_company": "TechCorp",
                "current_company_size": "1001-5000",
                "current_industry": "Technology",
            },
            "career_history": [
                {
                    "company": "TechCorp",
                    "title": "Senior ML Engineer",
                    "start_date": "2021-01-01",
                    "end_date": None,
                    "duration_months": 42,
                    "is_current": True,
                    "industry": "Technology",
                    "company_size": "1001-5000",
                    "description": "Built ranking models and search infrastructure.",
                }
            ],
            "education": [
                {
                    "institution": "IIT Bombay",
                    "degree": "B.Tech",
                    "field_of_study": "Computer Science",
                    "start_year": 2012,
                    "end_year": 2016,
                    "grade": "8.5",
                    "tier": "tier_1",
                }
            ],
            "skills": [
                {"name": "Python", "proficiency": "expert", "endorsements": 25, "duration_months": 60},
                {"name": "PyTorch", "proficiency": "advanced", "endorsements": 15, "duration_months": 36},
            ],
            "certifications": [],
            "languages": [],
            "redrob_signals": {
                "profile_completeness_score": 85.0,
                "signup_date": "2023-01-01",
                "last_active_date": "2025-06-01",
                "open_to_work_flag": True,
                "profile_views_received_30d": 45,
                "applications_submitted_30d": 2,
                "recruiter_response_rate": 0.75,
                "avg_response_time_hours": 4.5,
                "skill_assessment_scores": {"Python": 92, "PyTorch": 78},
                "connection_count": 350,
                "endorsements_received": 120,
                "notice_period_days": 30,
                "expected_salary_range_inr_lpa": {"min": 35, "max": 50},
                "preferred_work_mode": "remote",
                "willing_to_relocate": True,
                "github_activity_score": 75.0,
                "search_appearance_30d": 25,
                "saved_by_recruiters_30d": 5,
                "interview_completion_rate": 0.9,
                "offer_acceptance_rate": 0.8,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
            },
        }

    def test_parse_valid_candidate(self):
        candidate = parse_candidate(self.sample_line)
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.candidate_id, "CAND_0000001")
        self.assertEqual(candidate.profile.anonymized_name, "Candidate 1")
        self.assertEqual(candidate.profile.years_of_experience, 5.0)
        self.assertEqual(len(candidate.career_history), 1)
        self.assertEqual(candidate.career_history[0].company, "TechCorp")
        self.assertEqual(len(candidate.skills), 2)
        self.assertEqual(candidate.skills[0].name, "Python")
        self.assertTrue(candidate.redrob_signals.open_to_work_flag)

    def test_parse_malformed_candidate(self):
        result = parse_candidate({})
        self.assertIsNotNone(result)
        self.assertEqual(result.candidate_id, "")

    def test_load_candidates_from_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            f.write(json.dumps(self.sample_line) + "\n")
            f.write("invalid json line\n")
            f.write(json.dumps(self.sample_line) + "\n")
            fname = f.name

        try:
            candidates = list(load_candidates(fname))
            self.assertEqual(len(candidates), 2)
            self.assertEqual(candidates[0].candidate_id, "CAND_0000001")
            self.assertEqual(candidates[1].candidate_id, "CAND_0000001")
        finally:
            os.unlink(fname)

    def test_load_empty_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            fname = f.name

        try:
            candidates = list(load_candidates(fname))
            self.assertEqual(len(candidates), 0)
        finally:
            os.unlink(fname)


if __name__ == "__main__":
    unittest.main()
