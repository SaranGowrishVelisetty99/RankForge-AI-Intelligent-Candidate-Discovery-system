import json
import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

SKILL_TAXONOMY: Dict[str, List[str]] = {
    "vector_database": ["pinecone", "milvus", "weaviate", "qdrant", "chroma", "vespa"],
    "retrieval_systems": ["faiss", "ann", "hnsw", "ivf", "pq", "scann", "voyager"],
    "information_retrieval": ["bm25", "tf-idf", "dense retrieval", "sparse retrieval", "hybrid search", "learning to rank", "reranking"],
    "embedding_models": ["sentence transformer", "openai embedding", "bert embedding", "cross-encoder", "bi-encoder"],
    "ranking_systems": ["lambdamart", "xgboost rank", "lightgbm rank", "ndcg", "mrr", "map"],
    "fine_tuning": ["lora", "peft", "qlora", "adapter", "prompt tuning"],
    "llm_frameworks": ["langchain", "llamaindex", "haystack", "semantic kernel"],
    "ml_systems": ["pytorch", "tensorflow", "jax", "mlflow", "kubeflow", "triton"],
    "data_engineering": ["spark", "beam", "kafka", "flink", "hadoop", "hive"],
    "orchestration": ["airflow", "dagster", "prefect", "luigi", "argo"],
    "nlp": ["transformers", "spacy", "nltk", "huggingface", "tokenization"],
    "computer_vision": ["opencv", "yolo", "detectron", "image segmentation", "object detection"],
    "speech_audio": ["speech recognition", "wav2vec", "whisper", "text to speech", "audio processing"],
    "mle_tools": ["docker", "kubernetes", "terraform", "ci/cd", "git", "linux"],
    "databases": ["postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra"],
    "cloud_platforms": ["aws", "gcp", "azure", "sagemaker", "vertex ai"],
}

DOMAIN_ALIASES: Dict[str, List[str]] = {
    "vector database": ["vector db", "vector store", "ann index"],
    "retrieval": ["search", "information retrieval", "semantic search", "hybrid search"],
    "ml": ["machine learning", "deep learning", "ai", "artificial intelligence"],
    "nlp": ["natural language processing", "text processing", "language model"],
    "data engineering": ["data pipeline", "etl", "big data", "data infrastructure"],
    "mle": ["ml engineering", "ml infrastructure", "mlep", "ml platform"],
}

PARENT_DOMAIN: Dict[str, str] = {}
for domain, skills in SKILL_TAXONOMY.items():
    for skill in skills:
        PARENT_DOMAIN[skill] = domain


class KnowledgeGraph:
    def __init__(self):
        self.skill_domain: Dict[str, str] = PARENT_DOMAIN.copy()
        self.domain_skills: Dict[str, List[str]] = {
            d: list(s) for d, s in SKILL_TAXONOMY.items()
        }
        self.domain_aliases: Dict[str, List[str]] = DOMAIN_ALIASES

    def get_domain(self, skill: str) -> Optional[str]:
        skill_lower = skill.strip().lower()
        if skill_lower in self.skill_domain:
            return self.skill_domain[skill_lower]
        for domain, skills in self.skill_domain.items():
            if skill_lower in skills:
                return domain
        for domain, aliases in self.domain_aliases.items():
            if skill_lower in aliases:
                return domain
            if domain in aliases:
                return domain
        return None

    def get_related_skills(self, skill: str, depth: int = 1) -> Set[str]:
        domain = self.get_domain(skill)
        if domain is None:
            return set()
        related = set(self.domain_skills.get(domain, []))
        related.discard(skill.lower())
        if depth > 1:
            for domain_name, skills in self.domain_skills.items():
                if skill.lower() in skills:
                    for s in skills:
                        related.update(self.get_related_skills(s, depth - 1))
        return related

    def compute_ontology_overlap(
        self, candidate_skills: List[str], jd_skills: List[str]
    ) -> float:
        if not jd_skills:
            return 0.0
        jd_domains = set()
        for s in jd_skills:
            d = self.get_domain(s)
            if d:
                jd_domains.add(d)

        candidate_domains = set()
        for s in candidate_skills:
            d = self.get_domain(s)
            if d:
                candidate_domains.add(d)

        if not jd_domains:
            return 0.0

        overlap = len(jd_domains & candidate_domains)
        return overlap / len(jd_domains)

    def expand_skills(self, skills: List[str]) -> Set[str]:
        expanded = set(s.lower() for s in skills)
        for skill in skills:
            related = self.get_related_skills(skill, depth=1)
            expanded.update(related)
        return expanded

    def to_json(self) -> str:
        return json.dumps(
            {
                "taxonomy": {
                    domain: list(skills)
                    for domain, skills in self.domain_skills.items()
                },
                "aliases": self.domain_aliases,
            },
            indent=2,
        )


knowledge_graph = KnowledgeGraph()
