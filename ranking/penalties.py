import logging
from typing import Dict, List, Optional

from config.constants import CONSULTING_FIRMS, PENALTIES
from config.jd_config import IR_RANKING_KEYWORDS, PRODUCTION_KEYWORDS
from models.candidate import Candidate

logger = logging.getLogger(__name__)


def compute_penalty_multiplier(candidate: Candidate) -> float:
    multiplier = 1.0

    if _is_consulting_only(candidate):
        multiplier *= PENALTIES["consulting_only"]
        logger.debug(f"  -> consulting-only penalty (x{multiplier:.2f})")

    if _is_research_only(candidate):
        multiplier *= PENALTIES["research_only"]
        logger.debug(f"  -> research-only penalty (x{multiplier:.2f})")

    llm_penalty = _compute_llm_no_ir_penalty(candidate)
    if llm_penalty < 1.0:
        multiplier *= llm_penalty
        logger.debug(f"  -> LLM-only no IR penalty (x{multiplier:.2f})")

    cv_penalty = _compute_cv_no_nlp_penalty(candidate)
    if cv_penalty < 1.0:
        multiplier *= cv_penalty
        logger.debug(f"  -> CV/Speech no NLP penalty (x{multiplier:.2f})")

    if _has_title_inflation(candidate):
        multiplier *= PENALTIES["title_inflation"]
        logger.debug(f"  -> title inflation penalty (x{multiplier:.2f})")

    if _is_honeypot(candidate):
        multiplier *= PENALTIES["honeypot_detected"]
        logger.debug(f"  -> HONEYPOT detected (x{multiplier:.2f})")

    return multiplier


def _is_consulting_only(candidate: Candidate) -> bool:
    if not candidate.career_history:
        return False
    companies = [c.company.lower() for c in candidate.career_history]
    consulting_roles = sum(
        1 for company in companies
        for firm in CONSULTING_FIRMS
        if firm in company
    )
    return consulting_roles == len(companies)


def _is_research_only(candidate: Candidate) -> bool:
    if not candidate.career_history:
        return False
    career_text = " ".join(c.description.lower() for c in candidate.career_history)
    has_production = any(kw in career_text for kw in PRODUCTION_KEYWORDS)
    research_roles = sum(
        1 for c in candidate.career_history
        if any(kw in c.title.lower() for kw in ["research", "scientist", "phd"])
    )
    return not has_production and research_roles > 0 and len(candidate.career_history) > 1


def _compute_llm_no_ir_penalty(candidate: Candidate) -> float:
    yoe = candidate.profile.years_of_experience

    skill_text = " ".join(s.name.lower() for s in candidate.skills)
    career_text = " ".join(c.description.lower() for c in candidate.career_history)
    combined = skill_text + " " + career_text

    has_llm_focus = any(kw in combined for kw in ["langchain", "llamaindex", "gpt", "rag"])
    has_ir_focus = any(kw in combined for kw in IR_RANKING_KEYWORDS[:10])

    if has_llm_focus and not has_ir_focus and yoe < 3.5:
        return PENALTIES["llm_only_no_ir"]

    return 1.0


def _compute_cv_no_nlp_penalty(candidate: Candidate) -> float:
    skill_text = " ".join(s.name.lower() for s in candidate.skills)
    career_text = " ".join(c.description.lower() for c in candidate.career_history)
    combined = skill_text + " " + career_text

    has_cv = any(kw in combined for kw in ["opencv", "yolo", "image", "object detection"])
    has_nlp = any(kw in combined for kw in ["nlp", "text", "language", "bert", "transformer"])

    if has_cv and not has_nlp:
        return PENALTIES["cv_speech_no_nlp"]

    return 1.0


def _has_title_inflation(candidate: Candidate) -> bool:
    if len(candidate.career_history) < 3:
        return False

    titles = [c.title.lower() for c in candidate.career_history]
    senior_titles = sum(
        1 for t in titles
        if any(kw in t for kw in ["senior", "lead", "staff", "principal", "head"])
    )

    avg_tenure = sum(c.duration_months for c in candidate.career_history) / len(candidate.career_history)

    return senior_titles > 2 and avg_tenure < 15


def _is_honeypot(candidate: Candidate) -> bool:
    contradictions_found = 0

    timeline_overlap = False
    if len(candidate.career_history) > 1:
        history = candidate.career_history
        for i in range(len(history)):
            for j in range(i + 1, len(history)):
                if (history[i].start_date and history[j].start_date
                        and history[i].end_date and history[j].end_date):
                    if (history[i].start_date < history[j].end_date
                            and history[j].start_date < history[i].end_date):
                        timeline_overlap = True
                        if contradictions_found >= 2:
                            return True

    if candidate.redrob_signals.github_activity_score > 90 and candidate.redrob_signals.profile_completeness_score < 30:
        contradictions_found += 1

    if len(candidate.skills) > 25 and candidate.profile.years_of_experience < 2:
        contradictions_found += 1

    return False
