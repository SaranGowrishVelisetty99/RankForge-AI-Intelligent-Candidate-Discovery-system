import logging
from typing import List

from agents.authenticity import AuthenticityAgent
from agents.availability import AvailabilityAgent
from agents.contradiction import ContradictionAgent
from agents.credibility import CredibilityAgent
from agents.evidence import EvidenceAgent
from agents.jd_fit import JDFitAgent
from agents.recruitability import RecruitabilityAgent
from agents.trajectory import TrajectoryAgent
from core.embeddings import embedding_service
from models.candidate import Candidate, CandidateFeatures
from models.jd import JDRequirements

logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(self):
        self.jd_fit_agent = JDFitAgent()
        self.agents = {
            "evidence": EvidenceAgent(),
            "credibility": CredibilityAgent(),
            "trajectory": TrajectoryAgent(),
            "recruitability": RecruitabilityAgent(),
            "availability": AvailabilityAgent(),
            "authenticity": AuthenticityAgent(),
            "consistency": ContradictionAgent(),
        }
        logger.info(f"FeatureExtractor initialized with {len(self.agents) + 1} agents")

    def extract_batch(
        self,
        candidates: List[Candidate],
        jd: JDRequirements,
        batch_size: int = 5000,
    ) -> List[CandidateFeatures]:
        all_features = []
        total = len(candidates)
        logger.info(f"Feature extraction on {total} candidates (TF-IDF + {len(self.agents)} agents)")

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = candidates[start:end]

            jd_fit_results = self.jd_fit_agent.score_batch(batch, jd)

            for i, candidate in enumerate(batch):
                try:
                    features = CandidateFeatures(candidate_id=candidate.candidate_id)
                    features.jd_fit = jd_fit_results[i].get("jd_fit_score", 50.0)

                    for name, agent in self.agents.items():
                        try:
                            result = agent(candidate, jd)
                            score_key = self._get_score_key(name)
                            score = result.get(score_key, 0)
                            setattr(features, name, score)
                        except Exception as e:
                            logger.error(f"Agent {name} failed for {candidate.candidate_id}: {e}")
                            setattr(features, name, 0.0)

                    all_features.append(features)
                except Exception as e:
                    logger.error(f"Features failed for {candidate.candidate_id}: {e}")
                    all_features.append(CandidateFeatures(candidate_id=candidate.candidate_id))

            logger.info(f"  Batch [{start}:{end}] ({len(batch)} candidates)")

        logger.info(f"Feature extraction complete for {len(all_features)} candidates")
        return all_features

    def _get_score_key(self, feature_name: str) -> str:
        mapping = {
            "evidence": "evidence_score",
            "credibility": "credibility_score",
            "trajectory": "trajectory_score",
            "recruitability": "recruitability_score",
            "availability": "availability_score",
            "authenticity": "authenticity_score",
            "consistency": "consistency_score",
        }
        return mapping.get(feature_name, f"{feature_name}_score")


feature_extractor = FeatureExtractor()
