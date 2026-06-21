import logging
from typing import Any, Dict, List

import numpy as np

from agents.base import BaseAgent
from core.embeddings import embedding_service
from core.knowledge_graph import knowledge_graph
from models.candidate import Candidate
from models.jd import JDRequirements

logger = logging.getLogger(__name__)


class JDFitAgent(BaseAgent):
    def __init__(self):
        super().__init__("JDFitAgent")

    def score(self, candidate: Candidate, jd: JDRequirements) -> Dict[str, Any]:
        candidate_skill_names = [s.name.lower() for s in candidate.skills]

        must_have_coverage = self._compute_coverage(
            candidate_skill_names, [s.lower() for s in jd.must_have]
        )
        preferred_coverage = self._compute_coverage(
            candidate_skill_names, [s.lower() for s in jd.preferred]
        )
        ontology_overlap = knowledge_graph.compute_ontology_overlap(
            candidate_skill_names,
            [s.lower() for s in jd.must_have + jd.preferred],
        )
        headline_fit = self._compute_headline_fit(candidate, jd)

        candidate_text = self._build_candidate_text(candidate)
        candidate_emb = embedding_service.encode([candidate_text])[0]
        jd_emb = np.array(jd.jd_embedding)
        semantic_similarity = embedding_service.cosine_similarity(candidate_emb, jd_emb)

        jd_fit_score = (
            0.20 * must_have_coverage * 100
            + 0.15 * preferred_coverage * 100
            + 0.15 * ontology_overlap * 100
            + 0.10 * headline_fit
            + 0.40 * semantic_similarity * 100
        )

        jd_fit_score = max(0, min(100, jd_fit_score))
        return {"jd_fit_score": round(jd_fit_score, 2)}

    def score_batch(self, candidates: List[Candidate], jd: JDRequirements) -> List[Dict[str, Any]]:
        texts = [self._build_candidate_text(c) for c in candidates]
        embeddings = embedding_service.encode(texts, batch_size=4096)
        jd_emb = np.array(jd.jd_embedding).reshape(1, -1)

        expanded_jd_must = knowledge_graph.expand_skills([s.lower() for s in jd.must_have])
        expanded_jd_pref = knowledge_graph.expand_skills([s.lower() for s in jd.preferred])
        expanded_jd_all = knowledge_graph.expand_skills(
            [s.lower() for s in jd.must_have + jd.preferred]
        )

        semantic_similarities = embedding_service.cosine_similarity_matrix(embeddings, jd_emb)[:, 0]

        jd_keywords_lower = [s.lower() for s in jd.must_have + jd.preferred]

        results = []
        for i, candidate in enumerate(candidates):
            candidate_skill_names = [s.name.lower() for s in candidate.skills]
            expanded_candidate = knowledge_graph.expand_skills(candidate_skill_names)

            must_matched = sum(1 for s in expanded_jd_must if s in expanded_candidate)
            must_have_coverage = must_matched / len(expanded_jd_must) if expanded_jd_must else 1.0

            pref_matched = sum(1 for s in expanded_jd_pref if s in expanded_candidate)
            preferred_coverage = pref_matched / len(expanded_jd_pref) if expanded_jd_pref else 1.0

            ontology_overlap = knowledge_graph.compute_ontology_overlap(
                candidate_skill_names, [s.lower() for s in jd.must_have + jd.preferred]
            )

            combined = (candidate.profile.headline + " " + candidate.profile.current_title).lower()
            headline_fit = min(100, (sum(1 for kw in jd_keywords_lower if kw in combined) / max(len(jd_keywords_lower), 1)) * 100)

            jd_fit_score = (
                0.20 * must_have_coverage * 100
                + 0.15 * preferred_coverage * 100
                + 0.15 * ontology_overlap * 100
                + 0.10 * headline_fit
                + 0.40 * semantic_similarities[i] * 100
            )

            jd_fit_score = max(0, min(100, jd_fit_score))
            results.append({"jd_fit_score": round(jd_fit_score, 2)})

        return results

    def _build_candidate_text(self, candidate: Candidate) -> str:
        parts = [
            candidate.profile.headline,
            candidate.profile.summary,
        ]
        for career in candidate.career_history:
            parts.append(career.description)
        return " ".join(parts)

    def _compute_coverage(self, candidate_skills, required_skills):
        if not required_skills:
            return 1.0
        expanded_candidate = knowledge_graph.expand_skills(candidate_skills)
        expanded_required = knowledge_graph.expand_skills(required_skills)
        matched = sum(1 for s in expanded_required if s in expanded_candidate)
        return matched / len(expanded_required)

    def _compute_headline_fit(self, candidate, jd):
        combined = (candidate.profile.headline + " " + candidate.profile.current_title).lower()
        jd_keywords = [s.lower() for s in jd.must_have + jd.preferred]
        if not jd_keywords:
            return 50.0
        matches = sum(1 for kw in jd_keywords if kw in combined)
        return min(100, (matches / len(jd_keywords)) * 100)
