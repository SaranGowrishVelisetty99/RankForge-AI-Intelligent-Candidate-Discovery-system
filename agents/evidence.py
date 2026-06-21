import logging
from typing import Any, Dict, List

from agents.base import BaseAgent
from models.candidate import Candidate
from models.jd import JDRequirements

logger = logging.getLogger(__name__)


class EvidenceAgent(BaseAgent):
    def __init__(self):
        super().__init__("EvidenceAgent")

    def score(self, candidate: Candidate, jd: JDRequirements) -> Dict[str, Any]:
        career_text = self._build_career_text(candidate)
        summary_text = candidate.profile.summary.lower()

        assessment_lookup = {
            s.lower() for s in candidate.redrob_signals.skill_assessment_scores
            if candidate.redrob_signals.skill_assessment_scores[s] >= 50
        }

        validated = []
        unsupported = []

        for skill in candidate.skills:
            skill_name = skill.name.lower()
            if self._has_evidence(skill_name, career_text, summary_text, assessment_lookup, candidate):
                validated.append(skill.name)
            else:
                unsupported.append(skill.name)

        total_skills = len(candidate.skills)
        if total_skills == 0:
            evidence_score = 0.0
        else:
            evidence_score = (len(validated) / total_skills) * 100

        return {
            "evidence_score": round(evidence_score, 2),
            "validated_skills": validated,
            "unsupported_skills": unsupported,
        }

    def _build_career_text(self, candidate: Candidate) -> str:
        parts = []
        for career in candidate.career_history:
            parts.append(career.description.lower())
        return " ".join(parts)

    def _has_evidence(
        self, skill_name: str, career_text: str, summary_text: str,
        assessment_lookup: set, candidate: Candidate,
    ) -> bool:
        if skill_name in career_text or skill_name in summary_text:
            return True
        if skill_name in assessment_lookup:
            return True
        for skill in candidate.skills:
            if skill.name.lower() == skill_name and skill.endorsements >= 5:
                return True
        return False
