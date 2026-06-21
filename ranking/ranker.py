import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.constants import DEFAULT_WEIGHTS, TOP_K_INITIAL
from models.candidate import CandidateFeatures
from ranking.penalties import compute_penalty_multiplier

logger = logging.getLogger(__name__)


class RankingEngine:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS
        self._ranker = None
        logger.info(f"RankingEngine initialized with weights={self.weights}")

    def rank(
        self,
        features: List[CandidateFeatures],
        candidates: List,
        use_ml: bool = False,
    ) -> List[CandidateFeatures]:
        scored_features = []
        for feat, candidate in zip(features, candidates):
            try:
                penalty = compute_penalty_multiplier(candidate)
            except Exception as e:
                logger.warning(f"Penalty computation failed for {candidate.candidate_id}: {e}")
                penalty = 1.0

            score = self._compute_weighted_score(feat)
            score = score * penalty
            feat.ranking_score = round(score, 2)
            scored_features.append(feat)

        scored_features.sort(key=lambda x: (-x.ranking_score, x.candidate_id))

        for rank, feat in enumerate(scored_features, 1):
            feat.rank = rank

        return scored_features[:TOP_K_INITIAL]

    def _compute_weighted_score(self, features: CandidateFeatures) -> float:
        score = 0.0
        for dim, weight in self.weights.items():
            value = getattr(features, dim, 0.0)
            score += weight * value
        return score

    def _normalize_scores(self, features: List[CandidateFeatures]) -> None:
        scores = np.array([f.ranking_score for f in features])
        if scores.max() - scores.min() > 1e-6:
            normalized = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            normalized = np.ones_like(scores)
        for feat, norm in zip(features, normalized):
            feat.ranking_score = round(norm * 100, 2)


ranking_engine = RankingEngine()
