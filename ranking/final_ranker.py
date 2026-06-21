import logging
from typing import Dict, List

import numpy as np

from config.constants import TOP_K_FINAL
from models.candidate import CandidateFeatures

logger = logging.getLogger(__name__)


class FinalRankingEngine:
    def __init__(self, ranking_weight: float = 0.7, review_weight: float = 0.3):
        self.ranking_weight = ranking_weight
        self.review_weight = review_weight
        logger.info(f"FinalRankingEngine: ranking_weight={ranking_weight}, review_weight={review_weight}")

    def rerank(
        self,
        features: List[CandidateFeatures],
        reviews: List[Dict],
    ) -> List[CandidateFeatures]:
        if len(features) != len(reviews):
            logger.warning(f"Features ({len(features)}) / reviews ({len(reviews)}) mismatch")

        ranking_scores = np.array([f.ranking_score for f in features])
        review_scores = np.array([r.get("review_score", 50.0) for r in reviews])

        ranking_norm = self._normalize(ranking_scores)
        review_norm = self._normalize(review_scores)

        for feat, rank_n, rev_n in zip(features, ranking_norm, review_norm):
            final = (self.ranking_weight * rank_n + self.review_weight * rev_n)
            feat.final_score = round(final, 4)
            feat.review_score = round(rev_n * 100, 2)

        features.sort(key=lambda x: (-x.final_score, x.candidate_id))

        for rank, feat in enumerate(features, 1):
            feat.rank = rank

        top_100 = features[:TOP_K_FINAL]

        for rank, feat in enumerate(top_100, 1):
            feat.rank = rank

        logger.info(f"Final ranking complete: {len(top_100)} candidates selected")
        return top_100

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return np.array([])
        if scores.max() - scores.min() > 1e-6:
            return (scores - scores.min()) / (scores.max() - scores.min())
        return np.full_like(scores, 0.5)


final_ranker = FinalRankingEngine()
