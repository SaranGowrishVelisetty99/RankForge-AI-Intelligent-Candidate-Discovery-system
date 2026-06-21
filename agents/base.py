import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from models.candidate import Candidate
from models.jd import JDRequirements

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def score(self, candidate: Candidate, jd: JDRequirements) -> Dict[str, Any]:
        ...

    def __call__(self, candidate: Candidate, jd: JDRequirements) -> Dict[str, Any]:
        result = self.score(candidate, jd)
        logger.debug(f"{self.name}: candidate={candidate.candidate_id} => {result}")
        return result
