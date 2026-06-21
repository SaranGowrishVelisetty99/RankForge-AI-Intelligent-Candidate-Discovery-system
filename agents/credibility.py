import logging
from typing import Any, Dict, List

import numpy as np

from agents.base import BaseAgent
from models.candidate import Candidate
from models.jd import JDRequirements

logger = logging.getLogger(__name__)

PROFICIENCY_MAP = {
    "beginner": 0.25,
    "intermediate": 0.50,
    "advanced": 0.75,
    "expert": 1.0,
}


class CredibilityAgent(BaseAgent):
    def __init__(self):
        super().__init__("CredibilityAgent")

    def score(self, candidate: Candidate, jd: JDRequirements) -> Dict[str, Any]:
        if not candidate.skills:
            return {"credibility_score": 0.0}

        skill_scores = []
        for skill in candidate.skills:
            score = self._compute_skill_credibility(skill, candidate)
            skill_scores.append(score)

        credibility_score = float(np.mean(skill_scores))

        return {"credibility_score": round(credibility_score, 2)}

    def _compute_skill_credibility(
        self, skill, candidate: Candidate
    ) -> float:
        assessment_base = self._get_assessment_score(skill.name, candidate)
        endorsement_boost = min(skill.endorsements / 20, 1.0) * 25
        duration_boost = min(skill.duration_months / 36, 1.0) * 15
        proficiency_factor = PROFICIENCY_MAP.get(skill.proficiency, 0.5) * 20

        score = assessment_base + endorsement_boost + duration_boost + proficiency_factor
        return max(0, min(100, score))

    def _get_assessment_score(self, skill_name: str, candidate: Candidate) -> float:
        scores = candidate.redrob_signals.skill_assessment_scores
        for assessed_skill, score in scores.items():
            if skill_name.lower() in assessed_skill.lower():
                return float(score)
        return 40.0
