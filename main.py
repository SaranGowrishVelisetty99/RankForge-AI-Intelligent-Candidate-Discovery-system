#!/usr/bin/env python3
import argparse
import asyncio
import logging
import time
from typing import Dict, List

from config.settings import settings
from core.candidate_parser import load_candidates
from core.embeddings import embedding_service
from core.feature_extraction import feature_extractor
from core.reasoning_generator import generate_reasoning
from models.candidate import Candidate, CandidateFeatures
from models.jd import JDRequirements
from output.csv_generator import generate_csv, validate_csv
from ranking.final_ranker import final_ranker
from ranking.ranker import ranking_engine

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def load_default_jd() -> str:
    jd_path = "data/job_description.txt"
    try:
        with open(jd_path, "r", encoding="utf-8") as f:
            content = f.read()
            if content.strip():
                logger.info(f"Loaded JD from {jd_path}")
                return content
    except (FileNotFoundError, IOError):
        pass
    logger.info("Using built-in default JD")
    return """Senior AI Engineer — Founding Team at Redrob AI.
We need production experience with embeddings-based retrieval systems, vector databases or hybrid search infrastructure, strong Python, and evaluation frameworks for ranking systems (NDCG, MRR, MAP).
Disqualifiers: pure research without production, recent LangChain-only AI experience, consulting-firm-only careers, CV/speech without NLP/IR, title-chasers.
Location: Pune/Noida, hybrid. 5-9 years experience preferred."""


def build_jd_text(jd: JDRequirements) -> str:
    parts = [
        f"Seniority: {jd.seniority}",
        f"Experience Required: {jd.experience_required} years",
        f"Must-have: {', '.join(jd.must_have)}",
        f"Preferred: {', '.join(jd.preferred)}",
        f"Domain: {', '.join(jd.domain_requirements)}",
    ]
    return " ".join(parts)


async def main():
    parser = argparse.ArgumentParser(description="RankForge AI - Candidate Intelligence Engine")
    parser.add_argument(
        "--candidates", "-c",
        default="data/candidates.jsonl",
        help="Path to candidates.jsonl file",
    )
    parser.add_argument(
        "--out", "-o",
        default="submission.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--jd", "-j",
        default="",
        help="Path to job description file (optional)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM calls (use fallbacks)",
    )
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=" * 60)
    logger.info("RankForge AI - Candidate Intelligence Engine")
    logger.info("=" * 60)

    logger.info("Loading job description...")
    if args.jd:
        with open(args.jd, "r", encoding="utf-8") as f:
            jd_text = f.read()
    else:
        jd_text = load_default_jd()

    if not args.skip_llm:
        from agents.jd_intelligence import analyze_job_description
        logger.info("Running JD Intelligence Agent (Nemotron)...")
        jd_requirements = await analyze_job_description(jd_text)
    else:
        logger.info("Skipping LLM - using basic JD analysis")
        jd_requirements = JDRequirements(
            must_have=[
                "python", "machine learning", "embeddings", "retrieval",
                "vector search", "ranking", "nlp", "production"
            ],
            preferred=[
                "fine-tuning", "lora", "learning to rank", "xgboost",
                "hr-tech", "distributed systems", "open source"
            ],
            seniority="senior",
            experience_required=5,
            domain_requirements=["information retrieval", "search", "recruiting tech"],
            behavioral_traits=["shipper mentality", "product-aware", "async communicator"],
            weighting_strategy={"technical": 50, "production_experience": 25, "domain": 15, "behavioral": 10},
            hidden_expectations=[
                "not consulting-only career",
                "not research-only",
                "not langchain-only with no prior ML",
                "pune or noida or willing to relocate",
                "sub-30 day notice preferred",
            ],
        )

    logger.info(f"JD Analysis: {len(jd_requirements.must_have)} must-have, "
                f"{len(jd_requirements.preferred)} preferred, "
                f"seniority={jd_requirements.seniority}")

    logger.info("Loading candidates...")
    all_candidates: List[Candidate] = list(load_candidates(args.candidates))
    logger.info(f"Loaded {len(all_candidates)} candidates")

    logger.info("Fitting TF-IDF embedding service on candidate corpus...")
    fit_texts = [
        f"{c.profile.headline} {c.profile.summary}"
        for c in all_candidates[:20000]
    ]
    embedding_service.fit(fit_texts)
    logger.info(f"TF-IDF fitted: vocabulary={embedding_service.dimension}")

    jd_text_combined = build_jd_text(jd_requirements)
    jd_embedding = embedding_service.encode([jd_text_combined])[0]
    jd_requirements.jd_embedding = jd_embedding.tolist()
    logger.info("JD embedding computed")

    logger.info("Extracting features with TF-IDF semantic embeddings...")
    all_features = feature_extractor.extract_batch(all_candidates, jd_requirements)
    logger.info(f"Features extracted for {len(all_features)} candidates")

    logger.info("Ranking to get Top 300...")
    top_300_features = ranking_engine.rank(all_features, all_candidates)
    logger.info(f"Top 300 selected")

    top_300_id_map: Dict[str, Candidate] = {}
    feat_lookup = {f.candidate_id: f for f in top_300_features}
    for c in all_candidates:
        if c.candidate_id in feat_lookup:
            top_300_id_map[c.candidate_id] = c

    if not args.skip_llm:
        from agents.recruiter_review import review_candidates
        logger.info("Running Recruiter Review Agent (Nemotron) on Top 300...")

        top_300_candidates = [top_300_id_map[f.candidate_id] for f in top_300_features]
        top_300_feature_dicts = [
            {
                "candidate_id": f.candidate_id,
                "jd_fit": f.jd_fit,
                "evidence": f.evidence,
                "credibility": f.credibility,
                "trajectory": f.trajectory,
                "recruitability": f.recruitability,
                "availability": f.availability,
                "authenticity": f.authenticity,
                "consistency": f.consistency,
                "ranking_score": f.ranking_score,
            }
            for f in top_300_features
        ]

        reviews = await review_candidates(
            top_300_candidates,
            top_300_feature_dicts,
            jd_requirements,
            batch_size=10,
        )
    else:
        logger.info("Skipping LLM - using fallback reviews")
        reviews = []
        for feat in top_300_features:
            avg = (feat.jd_fit + feat.evidence + feat.credibility + feat.trajectory) / 4
            reviews.append({
                "review_score": round(avg, 1),
                "summary": "Fallback review.",
                "recommendation": "consider" if avg >= 60 else "reject",
                "strengths": [],
                "weaknesses": [],
                "hiring_risk": "medium",
                "career_quality": "fair",
            })

    logger.info("Running final ranking...")
    top_100_features = final_ranker.rerank(top_300_features, reviews)

    logger.info("Generating candidate reasoning...")
    for feat in top_100_features:
        candidate = top_300_id_map.get(feat.candidate_id)
        if candidate:
            feat.reasoning = generate_reasoning(feat, candidate)

    logger.info("Generating CSV output...")
    generate_csv(top_100_features, args.out)

    valid = validate_csv(args.out)
    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info(f"RankForge AI Complete!")
    logger.info(f"  Candidates processed: {len(all_candidates)}")
    logger.info(f"  Top 100 output: {args.out}")
    logger.info(f"  CSV valid: {valid}")
    logger.info(f"  Time elapsed: {elapsed:.2f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
