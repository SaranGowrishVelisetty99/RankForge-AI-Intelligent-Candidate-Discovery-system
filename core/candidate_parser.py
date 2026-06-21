import json
import logging
from typing import Dict, Generator, List, Optional

from models.candidate import (
    Candidate,
    CareerEntry,
    Certification,
    Education,
    Profile,
    RedrobSignals,
    Skill,
)

logger = logging.getLogger(__name__)


def parse_candidate(raw: Dict) -> Optional[Candidate]:
    try:
        profile_data = raw.get("profile", {})
        profile = Profile(
            anonymized_name=profile_data.get("anonymized_name", ""),
            headline=profile_data.get("headline", ""),
            summary=profile_data.get("summary", ""),
            location=profile_data.get("location", ""),
            country=profile_data.get("country", ""),
            years_of_experience=float(profile_data.get("years_of_experience", 0)),
            current_title=profile_data.get("current_title", ""),
            current_company=profile_data.get("current_company", ""),
            current_company_size=profile_data.get("current_company_size", ""),
            current_industry=profile_data.get("current_industry", ""),
        )

        career_history = []
        for entry in raw.get("career_history", []):
            career_history.append(
                CareerEntry(
                    company=entry.get("company", ""),
                    title=entry.get("title", ""),
                    start_date=entry.get("start_date", ""),
                    end_date=entry.get("end_date"),
                    duration_months=int(entry.get("duration_months", 0)),
                    is_current=bool(entry.get("is_current", False)),
                    industry=entry.get("industry", ""),
                    company_size=entry.get("company_size", ""),
                    description=entry.get("description", ""),
                )
            )

        education = []
        for edu in raw.get("education", []):
            education.append(
                Education(
                    institution=edu.get("institution", ""),
                    degree=edu.get("degree", ""),
                    field_of_study=edu.get("field_of_study", ""),
                    start_year=int(edu.get("start_year", 0)),
                    end_year=int(edu.get("end_year", 0)),
                    grade=edu.get("grade"),
                    tier=edu.get("tier", "unknown"),
                )
            )

        skills = []
        for skill in raw.get("skills", []):
            skills.append(
                Skill(
                    name=skill.get("name", ""),
                    proficiency=skill.get("proficiency", "beginner"),
                    endorsements=int(skill.get("endorsements", 0)),
                    duration_months=int(skill.get("duration_months", 0)),
                )
            )

        certifications = []
        for cert in raw.get("certifications", []):
            certifications.append(
                Certification(
                    name=cert.get("name", ""),
                    issuer=cert.get("issuer", ""),
                    year=int(cert.get("year", 0)),
                )
            )

        signals_data = raw.get("redrob_signals", {})
        redrob_signals = RedrobSignals(
            profile_completeness_score=float(signals_data.get("profile_completeness_score", 0)),
            signup_date=signals_data.get("signup_date", ""),
            last_active_date=signals_data.get("last_active_date", ""),
            open_to_work_flag=bool(signals_data.get("open_to_work_flag", False)),
            profile_views_received_30d=int(signals_data.get("profile_views_received_30d", 0)),
            applications_submitted_30d=int(signals_data.get("applications_submitted_30d", 0)),
            recruiter_response_rate=float(signals_data.get("recruiter_response_rate", 0)),
            avg_response_time_hours=float(signals_data.get("avg_response_time_hours", 0)),
            skill_assessment_scores=signals_data.get("skill_assessment_scores", {}),
            connection_count=int(signals_data.get("connection_count", 0)),
            endorsements_received=int(signals_data.get("endorsements_received", 0)),
            notice_period_days=int(signals_data.get("notice_period_days", 0)),
            expected_salary_range_inr_lpa=signals_data.get("expected_salary_range_inr_lpa", {"min": 0, "max": 0}),
            preferred_work_mode=signals_data.get("preferred_work_mode", "flexible"),
            willing_to_relocate=bool(signals_data.get("willing_to_relocate", False)),
            github_activity_score=float(signals_data.get("github_activity_score", -1)),
            search_appearance_30d=int(signals_data.get("search_appearance_30d", 0)),
            saved_by_recruiters_30d=int(signals_data.get("saved_by_recruiters_30d", 0)),
            interview_completion_rate=float(signals_data.get("interview_completion_rate", 0)),
            offer_acceptance_rate=float(signals_data.get("offer_acceptance_rate", -1)),
            verified_email=bool(signals_data.get("verified_email", False)),
            verified_phone=bool(signals_data.get("verified_phone", False)),
            linkedin_connected=bool(signals_data.get("linkedin_connected", False)),
        )

        return Candidate(
            candidate_id=raw.get("candidate_id", ""),
            profile=profile,
            career_history=career_history,
            education=education,
            skills=skills,
            certifications=certifications,
            languages=raw.get("languages", []),
            redrob_signals=redrob_signals,
        )
    except Exception as e:
        logger.warning(f"Failed to parse candidate {raw.get('candidate_id', 'unknown')}: {e}")
        return None


def load_candidates(filepath: str) -> Generator[Candidate, None, None]:
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                candidate = parse_candidate(raw)
                if candidate is not None:
                    yield candidate
                else:
                    logger.warning(f"Skipping malformed candidate at line {line_num}")
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON at line {line_num}")
            except Exception as e:
                logger.warning(f"Error at line {line_num}: {e}")
