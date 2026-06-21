import logging
from typing import Any, Dict
from datetime import datetime, timedelta

from agents.base import BaseAgent
from models.candidate import Candidate
from models.jd import JDRequirements

logger = logging.getLogger(__name__)


class AvailabilityAgent(BaseAgent):
    def __init__(self):
        super().__init__("AvailabilityAgent")

    def score(self, candidate: Candidate, jd: JDRequirements) -> Dict[str, Any]:
        signals = candidate.redrob_signals

        open_to_work = self._compute_open_to_work(signals.open_to_work_flag)
        notice_period = self._compute_notice_period(signals.notice_period_days)
        recency = self._compute_recency(signals.last_active_date)
        relocation = self._compute_relocation(
            signals.willing_to_relocate, candidate.profile.country
        )
        work_mode = self._compute_work_mode(
            signals.preferred_work_mode, jd
        )

        availability_score = (
            0.30 * open_to_work
            + 0.30 * notice_period
            + 0.20 * recency
            + 0.10 * relocation
            + 0.10 * work_mode
        )

        return {"availability_score": round(availability_score, 2)}

    def _compute_open_to_work(self, flag: bool) -> float:
        return 100.0 if flag else 20.0

    def _compute_notice_period(self, days: int) -> float:
        if days <= 0:
            return 100.0
        if days < 30:
            return 100.0
        if days < 60:
            return 70.0
        if days < 90:
            return 50.0
        if days < 120:
            return 30.0
        return 15.0

    def _compute_recency(self, last_active: str) -> float:
        if not last_active:
            return 50.0
        try:
            last_date = datetime.strptime(last_active, "%Y-%m-%d")
            delta = datetime.now() - last_date
            days_since = delta.days
            if days_since < 7:
                return 100.0
            if days_since < 30:
                return 90.0
            if days_since < 90:
                return 70.0
            if days_since < 180:
                return 40.0
            return 10.0
        except ValueError:
            return 50.0

    def _compute_relocation(self, willing: bool, country: str) -> float:
        base = 100.0 if willing else 30.0
        country_lower = country.lower()
        if country_lower in ("india", "in"):
            base += 10
        else:
            base += 5
        return min(100, base)

    def _compute_work_mode(self, preferred: str, jd: JDRequirements) -> float:
        flexible_modes = ["flexible", "remote", "hybrid"]
        if preferred in flexible_modes:
            return 100.0
        return 50.0
