import logging
from typing import List

from config.jd_config import IR_RANKING_KEYWORDS
from models.candidate import Candidate, CandidateFeatures

logger = logging.getLogger(__name__)

AI_CORE_SKILLS = [
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "pytorch", "tensorflow", "keras", "scikit-learn",
    "xgboost", "lightgbm", "transformers", "huggingface", "bert",
    "gpt", "llm", "large language model", "rag", "retrieval augmented generation",
    "embedding", "sentence transformer", "vector search", "faiss",
    "pinecone", "milvus", "weaviate", "qdrant", "chroma",
    "ranking", "learning to rank", "recommendation system",
    "information retrieval", "semantic search", "hybrid search",
    "fine-tuning", "lora", "peft", "qlora", "prompt engineering",
    "langchain", "llamaindex", "haystack", "openai",
    "neural network", "cnn", "rnn", "lstm", "attention",
    "mlflow", "kubeflow", "ml engineering", "mlops",
    "data science", "ai", "artificial intelligence",
    "anomaly detection", "forecasting", "classification",
    "regression", "clustering", "dimensionality reduction",
    "feature engineering", "model deployment", "model serving",
    "a/b testing", "experimentation", "evaluation",
    "spark", "kafka", "airflow", "data pipeline",
]


def generate_reasoning(
    feature: CandidateFeatures,
    candidate: Candidate,
) -> str:
    title = candidate.profile.current_title or "Professional"
    yoe = candidate.profile.years_of_experience

    all_skill_names = [s.name.lower() for s in candidate.skills]
    all_skill_text = " ".join(all_skill_names)

    career_text = " ".join(c.description.lower() for c in candidate.career_history)
    combined_text = all_skill_text + " " + career_text

    ai_core_count = sum(
        1 for kw in AI_CORE_SKILLS if kw in combined_text
    )

    response_rate = candidate.redrob_signals.recruiter_response_rate

    reasoning = (
        f"{title} with {yoe:.1f} yrs; "
        f"{ai_core_count} AI core skills; "
        f"response rate {response_rate:.2f}."
    )

    return reasoning
