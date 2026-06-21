import logging
from typing import Any, Dict

from llm.openrouter_client import openrouter_client
from llm.prompts import JD_ANALYSIS_PROMPT
from llm.schemas import JDAnalysisOutput
from models.jd import JDRequirements

logger = logging.getLogger(__name__)


async def analyze_job_description(jd_text: str) -> JDRequirements:
    logger.info("Analyzing job description with Nemotron")

    try:
        result = await openrouter_client.extract_structured(
            system_prompt="You are a Senior Technical Recruiter. Extract structured requirements from job descriptions.",
            user_prompt=JD_ANALYSIS_PROMPT.format(jd_text=jd_text),
            response_model=JDAnalysisOutput,
        )

        jd_reqs = JDRequirements(
            must_have=result.get("must_have", []),
            preferred=result.get("preferred", []),
            seniority=result.get("seniority", ""),
            experience_required=result.get("experience_required", 0),
            domain_requirements=result.get("domain_requirements", []),
            behavioral_traits=result.get("behavioral_traits", []),
            weighting_strategy=result.get("weighting_strategy", {}),
            hidden_expectations=result.get("hidden_expectations", []),
        )

        logger.info(f"JD analysis complete: {len(jd_reqs.must_have)} must-have, "
                     f"{len(jd_reqs.preferred)} preferred, "
                     f"seniority={jd_reqs.seniority}")

    except Exception as e:
        logger.warning(f"LLM JD analysis failed: {e}. Using fallback extraction.")
        jd_reqs = _fallback_extract_jd(jd_text)

    return jd_reqs


def _fallback_extract_jd(jd_text: str) -> JDRequirements:
    jd_lower = jd_text.lower()

    must_have = []
    preferred = []

    key_skills = [
        "python", "machine learning", "deep learning", "nlp",
        "vector search", "semantic search", "retrieval", "ranking",
        "pytorch", "tensorflow", "transformers", "rag",
        "data engineering", "spark", "kafka", "airflow",
        "sql", "kubernetes", "docker", "aws", "gcp",
    ]

    for skill in key_skills:
        if skill in jd_lower:
            must_have.append(skill)

    pref_skills = [
        "fastapi", "flask", "mongodb", "redis", "elasticsearch",
        "bert", "gpt", "lora", "peft", "mlflow",
    ]
    for skill in pref_skills:
        if skill in jd_lower:
            preferred.append(skill)

    return JDRequirements(
        must_have=must_have,
        preferred=preferred,
        seniority="senior" if "senior" in jd_lower else "mid",
        experience_required=5,
        domain_requirements=[],
        behavioral_traits=[],
        weighting_strategy={},
        hidden_expectations=[],
    )
