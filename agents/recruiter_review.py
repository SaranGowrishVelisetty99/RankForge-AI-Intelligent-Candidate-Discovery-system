import asyncio
import logging
from typing import Any, Dict, List

from llm.openrouter_client import openrouter_client
from llm.prompts import RECRUITER_REVIEW_PROMPT
from llm.schemas import RecruiterReviewOutput
from models.candidate import Candidate
from models.jd import JDRequirements

logger = logging.getLogger(__name__)


def _build_profile_summary(candidate: Candidate) -> str:
    profile = candidate.profile
    lines = [
        f"Name: {profile.anonymized_name}",
        f"Headline: {profile.headline}",
        f"Summary: {profile.summary[:300]}",
        f"Location: {profile.location}, {profile.country}",
        f"Years of Experience: {profile.years_of_experience}",
        f"Current Title: {profile.current_title}",
        f"Current Company: {profile.current_company}",
        f"Skills: {', '.join(s.name for s in candidate.skills[:15])}",
    ]

    if candidate.career_history:
        lines.append("Career History:")
        for role in candidate.career_history[:5]:
            lines.append(
                f"  - {role.title} @ {role.company} ({role.duration_months} months)"
            )

    return "\n".join(lines)


def _build_agent_scores(features: Dict[str, float]) -> str:
    return "\n".join(
        f"  {k}: {v:.1f}"
        for k, v in features.items()
        if k != "candidate_id"
    )


def _build_role_requirements(jd: JDRequirements) -> str:
    lines = [
        f"Role: {jd.seniority} level",
        f"Experience Required: {jd.experience_required} years",
        f"Must-have skills: {', '.join(jd.must_have)}",
        f"Preferred skills: {', '.join(jd.preferred)}",
        f"Domain: {', '.join(jd.domain_requirements)}",
    ]
    return "\n".join(lines)


async def review_candidates(
    candidates: List[Candidate],
    features: List[Dict[str, Any]],
    jd: JDRequirements,
    batch_size: int = 10,
) -> List[Dict[str, Any]]:
    logger.info(f"Reviewing {len(candidates)} candidates with Nemotron (batch_size={batch_size})")

    semaphore = asyncio.Semaphore(3)
    results = [None] * len(candidates)

    async def review_single(idx: int, candidate: Candidate, feature: Dict[str, Any]):
        async with semaphore:
            profile_summary = _build_profile_summary(candidate)
            agent_scores = _build_agent_scores(feature)
            role_requirements = _build_role_requirements(jd)

            prompt = RECRUITER_REVIEW_PROMPT.format(
                profile_summary=profile_summary,
                agent_scores=agent_scores,
                role_requirements=role_requirements,
            )

            try:
                result = await openrouter_client.extract_structured(
                    system_prompt="You are a Chief Recruiting Officer evaluating candidates.",
                    user_prompt=prompt,
                    response_model=RecruiterReviewOutput,
                )
                return result
            except Exception as e:
                logger.warning(f"Review failed for {candidate.candidate_id}: {e}")
                return _fallback_review(feature)

    tasks = []
    for i, (candidate, feature) in enumerate(zip(candidates, features)):
        task = asyncio.create_task(review_single(i, candidate, feature))
        tasks.append(task)

    completed = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(completed):
        if isinstance(result, dict):
            results[i] = result
        elif isinstance(result, Exception):
            logger.error(f"Review task {i} failed: {result}")
            results[i] = _fallback_review(features[i] if i < len(features) else {})
        else:
            results[i] = _fallback_review(features[i] if i < len(features) else {})

    logger.info(f"Completed {sum(1 for r in results if r is not None)} reviews")
    return results


def _fallback_review(features: Dict[str, Any]) -> Dict[str, Any]:
    avg_score = sum(
        features.get(k, 0) for k in
        ["jd_fit", "evidence", "credibility", "trajectory"]
    ) / 4.0

    return {
        "review_score": round(avg_score, 1),
        "summary": "Automated review based on agent scores.",
        "recommendation": "consider" if avg_score >= 60 else "reject",
        "strengths": [],
        "weaknesses": [],
        "hiring_risk": "medium",
        "career_quality": "fair",
    }
