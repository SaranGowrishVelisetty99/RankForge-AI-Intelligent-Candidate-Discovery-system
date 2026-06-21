from typing import Dict, List

IR_RANKING_KEYWORDS: List[str] = [
    "retrieval", "ranking", "search", "information retrieval",
    "vector search", "semantic search", "reranking", "BM25",
    "dense retrieval", "sparse retrieval", "hybrid search",
    "learning to rank", "LTR", "NDCG", "MRR", "MAP",
    "relevance", "relevance scoring", "A/B testing",
    "evaluation", "offline evaluation", "online evaluation",
    "feature engineering", "ranking model", "Lambdamart",
    "LightGBM ranker", "XGBoost ranker", "neural ranker",
    "embedding", "sentence transformer", "cross-encoder",
    "bi-encoder", "late interaction", "ColBERT",
    "RAG", "retrieval augmented generation",
    "faiss", "pinecone", "milvus", "weaviate", "qdrant",
    "ann", "approximate nearest neighbor", "hnsw",
    "inverted index", "indexing", "sharding",
]

PRODUCTION_KEYWORDS: List[str] = [
    "production", "deployment", "A/B test", "online",
    "scalable", "low latency", "high throughput",
    "distributed system", "pipeline", "real-time",
    "serving", "inference", "model serving",
    "monitoring", "observability", "SLI", "SLO",
    "data pipeline", "ETL", "streaming",
]

JD_SENIORITY_KEYWORDS: Dict[str, List[str]] = {
    "junior": ["junior", "associate", "trainee", "fresher", "entry level"],
    "mid": ["software engineer", "developer", "engineer ii", "mid-level"],
    "senior": ["senior", "staff", "lead", "principal", "architect", "manager"],
}

CONSULTING_FIRM_NAMES: List[str] = [
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra",
]
