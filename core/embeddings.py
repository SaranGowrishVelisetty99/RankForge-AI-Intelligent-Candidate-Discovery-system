import logging
import pickle
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._dimension = 0
        self._fitted = False
        logger.info("EmbeddingService initialized (TF-IDF)")

    def fit(self, texts: List[str]) -> None:
        cleaned = [t if t else "" for t in texts]
        self._vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
            norm="l2",
            dtype=np.float32,
        )
        self._vectorizer.fit(cleaned)
        self._dimension = len(self._vectorizer.get_feature_names_out())
        self._fitted = True
        logger.info(f"TF-IDF vectorizer fitted: vocabulary={self._dimension}")

    def encode(self, texts: List[str], batch_size: int = 4096) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EmbeddingService must be fitted before encoding")
        cleaned = [t if t else "" for t in texts]
        return self._vectorizer.transform(cleaned).toarray().astype(np.float32)

    def cosine_similarity(
        self, embedding_a: np.ndarray, embedding_b: np.ndarray
    ) -> float:
        a_2d = embedding_a.reshape(1, -1)
        b_2d = embedding_b.reshape(1, -1)
        return float(sk_cosine_similarity(a_2d, b_2d)[0, 0])

    def cosine_similarity_matrix(
        self, embeddings_a: np.ndarray, embeddings_b: np.ndarray
    ) -> np.ndarray:
        return sk_cosine_similarity(embeddings_a, embeddings_b)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_fitted(self) -> bool:
        return self._fitted


embedding_service = EmbeddingService()
