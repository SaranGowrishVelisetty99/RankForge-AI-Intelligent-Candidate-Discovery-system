import logging
import re
from typing import Any, Dict, List

from agents.base import BaseAgent
from config.jd_config import JD_SENIORITY_KEYWORDS
from models.candidate import Candidate, CareerEntry
from models.jd import JDRequirements

logger = logging.getLogger(__name__)

SENIORITY_LEVELS = ["junior", "mid", "senior", "lead", "principal", "staff", "architect", "manager", "director", "vp", "head"]


class TrajectoryAgent(BaseAgent):
    def __init__(self):
        super().__init__("TrajectoryAgent")

    def score(self, candidate: Candidate, jd: JDRequirements) -> Dict[str, Any]:
        history = candidate.career_history
        if not history:
            return {"trajectory_score": 50.0}

        promotion_score = self._compute_promotion_score(history)
        seniority_growth = self._compute_seniority_growth(history)
        domain_consistency = self._compute_domain_consistency(history)
        technical_evolution = self._compute_technical_evolution(history)
        stability = self._compute_stability(history)

        trajectory_score = (
            0.25 * promotion_score
            + 0.20 * seniority_growth
            + 0.20 * domain_consistency
            + 0.15 * technical_evolution
            + 0.20 * stability
        )

        return {"trajectory_score": round(trajectory_score, 2)}

    def _compute_promotion_score(self, history: List[CareerEntry]) -> float:
        if len(history) < 2:
            return 50.0

        promotions = 0
        for i in range(1, len(history)):
            prev_level = self._extract_seniority_level(history[i - 1].title)
            curr_level = self._extract_seniority_level(history[i].title)
            if curr_level > prev_level:
                promotions += 1

        max_possible = len(history) - 1
        if max_possible == 0:
            return 50.0
        return min(100, (promotions / max_possible) * 100)

    def _compute_seniority_growth(self, history: List[CareerEntry]) -> float:
        if len(history) < 2:
            return 50.0

        total_months = sum(c.duration_months for c in history)
        if total_months == 0:
            return 50.0

        first_level = self._extract_seniority_level(history[0].title)
        last_level = self._extract_seniority_level(history[-1].title)
        level_gain = last_level - first_level
        years = total_months / 12

        if years == 0:
            return 50.0

        growth_rate = level_gain / years
        if growth_rate >= 0.5:
            return 100.0
        elif growth_rate >= 0.25:
            return 75.0
        elif growth_rate >= 0.0:
            return 50.0
        else:
            return 25.0

    def _compute_domain_consistency(self, history: List[CareerEntry]) -> float:
        if len(history) < 2:
            return 75.0

        industries = [c.industry.lower() for c in history if c.industry]
        if not industries:
            return 50.0

        consistent = sum(
            1 for i in range(1, len(industries))
            if industries[i] == industries[i - 1]
        )
        return (consistent / (len(industries) - 1)) * 100

    def _compute_technical_evolution(self, history: List[CareerEntry]) -> float:
        if len(history) < 2:
            return 50.0

        descriptions = [c.description.lower() for c in history if c.description]
        if not descriptions:
            return 50.0

        modern_terms = ["ml", "ai", "deep learning", "pytorch", "tensorflow",
                        "kubernetes", "docker", "cloud", "aws", "gcp", "azure",
                        "microservice", "api", "data pipeline"]

        recent_descriptions = descriptions[-min(3, len(descriptions)):]
        modern_count = sum(
            1 for desc in recent_descriptions
            for term in modern_terms
            if term in desc
        )
        return min(100, modern_count * 20)

    def _compute_stability(self, history: List[CareerEntry]) -> float:
        if not history:
            return 50.0

        durations = [c.duration_months for c in history]
        if not durations:
            return 50.0

        short_stints = sum(1 for d in durations if d < 12)
        job_hopping_penalty = min(short_stints / len(durations), 1.0) * 50

        avg_tenure = sum(durations) / len(durations)
        tenure_bonus = min(avg_tenure / 24, 1.0) * 30

        current_tenure = durations[-1] if durations else 0
        current_bonus = min(current_tenure / 24, 1.0) * 20

        return tenure_bonus + current_bonus - job_hopping_penalty + 50

    def _extract_seniority_level(self, title: str) -> int:
        title_lower = title.lower()
        for level, keywords in JD_SENIORITY_KEYWORDS.items():
            for kw in keywords:
                if kw in title_lower:
                    return SENIORITY_LEVELS.index(level) + 1 if level in SENIORITY_LEVELS else 1
        for i, level in enumerate(SENIORITY_LEVELS):
            if level in title_lower:
                return i + 1
        return 1
