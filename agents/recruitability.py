import logging
from typing import Any, Dict

from agents.base import BaseAgent
from models.candidate import Candidate
from models.jd import JDRequirements

logger = logging.getLogger(__name__)


class RecruitabilityAgent(BaseAgent):
    def __init__(self):
        super().__init__("RecruitabilityAgent")

    def score(self, candidate: Candidate, jd: JDRequirements) -> Dict[str, Any]:
        signals = candidate.redrob_signals

        response_rate_score = signals.recruiter_response_rate * 100

        profile_views = min(signals.profile_views_received_30d / 50, 1.0) * 100

        interview_score = signals.interview_completion_rate * 100

        if signals.offer_acceptance_rate >= 0:
            offer_score = signals.offer_acceptance_rate * 100
        else:
            offer_score = 50.0

        saved_score = min(signals.saved_by_recruiters_30d / 10, 1.0) * 100

        search_score = min(signals.search_appearance_30d / 100, 1.0) * 100

        completeness_bonus = min(signals.profile_completeness_score / 100, 1.0) * 10

        recruitability_score = (
            0.25 * response_rate_score
            + 0.20 * profile_views
            + 0.20 * interview_score
            + 0.15 * offer_score
            + 0.10 * saved_score
            + 0.10 * search_score
            + completeness_bonus
        )

        return {"recruitability_score": round(recruitability_score, 2)}
