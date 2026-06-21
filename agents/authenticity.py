import logging
from typing import Any, Dict, List

from agents.base import BaseAgent
from models.candidate import Candidate
from models.jd import JDRequirements

logger = logging.getLogger(__name__)


class AuthenticityAgent(BaseAgent):
    def __init__(self):
        super().__init__("AuthenticityAgent")

    def score(self, candidate: Candidate, jd: JDRequirements) -> Dict[str, Any]:
        flags = self._detect_red_flags(candidate)
        total_flags = len(flags)
        risk_level = self._compute_risk_level(total_flags)

        authenticity_score = max(100 - total_flags * 15, 20)

        return {
            "authenticity_score": round(authenticity_score, 2),
            "risk_level": risk_level,
        }

    def _detect_red_flags(self, candidate: Candidate) -> List[str]:
        flags = []

        if len(candidate.skills) > 20:
            flags.append("unrealistic_skill_breadth")

        beginner_count = sum(
            1 for s in candidate.skills if s.proficiency == "beginner"
        )
        if candidate.skills and beginner_count / len(candidate.skills) > 0.5:
            flags.append("mostly_beginner_skills")

        if candidate.redrob_signals.github_activity_score == -1:
            flags.append("no_github")

        if candidate.redrob_signals.profile_completeness_score < 40:
            flags.append("incomplete_profile")

        short_descriptions = sum(
            1 for c in candidate.career_history
            if len(c.description.strip()) < 50
        )
        if candidate.career_history and short_descriptions / len(candidate.career_history) > 0.5:
            flags.append("sparse_career_descriptions")

        assessment_count = len(candidate.redrob_signals.skill_assessment_scores)
        if assessment_count > 0:
            low_assessments = sum(
                1 for v in candidate.redrob_signals.skill_assessment_scores.values()
                if v < 30
            )
            if low_assessments / assessment_count > 0.5:
                flags.append("weak_assessments")

        if candidate.redrob_signals.endorsements_received > 500:
            flags.append("suspicious_endorsements")

        advanced_skills = sum(
            1 for s in candidate.skills if s.proficiency in ("advanced", "expert")
        )
        if (
            advanced_skills > 10
            and candidate.profile.years_of_experience < 3
        ):
            flags.append("inflated_profile")

        return flags

    def _compute_risk_level(self, num_flags: int) -> str:
        if num_flags == 0:
            return "low"
        if num_flags <= 2:
            return "medium"
        return "high"
