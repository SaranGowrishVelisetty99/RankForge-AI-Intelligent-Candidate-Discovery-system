import logging
from typing import Any, Dict, List

from agents.base import BaseAgent
from core.knowledge_graph import knowledge_graph
from models.candidate import Candidate
from models.jd import JDRequirements

logger = logging.getLogger(__name__)

AI_SKILLS = [
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "speech recognition", "gan", "generative adversarial",
    "reinforcement learning", "transformer", "bert", "gpt", "lora", "peft",
    "pytorch", "tensorflow", "keras", "scikit-learn", "xgboost", "lightgbm",
    "neural network", "cnn", "rnn", "lstm", "attention",
]

DATA_ENG_SKILLS = [
    "spark", "kafka", "airflow", "hadoop", "hive", "flink",
    "beam", "hbase", "cassandra", "mongodb", "elasticsearch",
]


class ContradictionAgent(BaseAgent):
    def __init__(self):
        super().__init__("ContradictionAgent")

    def score(self, candidate: Candidate, jd: JDRequirements) -> Dict[str, Any]:
        contradictions = []

        ai_skill_mismatch = self._detect_ai_skill_mismatch(candidate)
        contradictions.extend(ai_skill_mismatch)

        timeline_issues = self._detect_timeline_overlaps(candidate)
        contradictions.extend(timeline_issues)

        seniority_gap = self._detect_seniority_skill_gap(candidate)
        if seniority_gap:
            contradictions.append(seniority_gap)

        consistency_score = max(100 - len(contradictions) * 20, 0)

        return {
            "consistency_score": round(consistency_score, 2),
            "contradictions": contradictions,
        }

    def _detect_ai_skill_mismatch(self, candidate: Candidate) -> List[str]:
        contradictions = []
        claimed_ai = set()

        for skill in candidate.skills:
            skill_lower = skill.name.lower()
            domain = knowledge_graph.get_domain(skill_lower)
            if domain in ("nlp", "ml_systems", "computer_vision", "speech_audio"):
                claimed_ai.add(skill_lower)
            if skill_lower in AI_SKILLS:
                claimed_ai.add(skill_lower)

        if not claimed_ai:
            return contradictions

        career_text = " ".join(c.description.lower() for c in candidate.career_history)
        found_evidence = any(skill in career_text for skill in claimed_ai)

        data_eng_dominant = sum(
            1 for s in candidate.skills
            if s.name.lower() in DATA_ENG_SKILLS
        )
        data_eng_career = sum(
            1 for s in DATA_ENG_SKILLS if s in career_text
        )

        if not found_evidence and (data_eng_dominant > 3 or data_eng_career > 2):
            contradictions.append(
                f"Claims AI/ML skills ({', '.join(list(claimed_ai)[:3])}) "
                f"but career evidence shows data engineering focus"
            )

        return contradictions

    def _detect_timeline_overlaps(self, candidate: Candidate) -> List[str]:
        contradictions = []
        history = candidate.career_history

        for i in range(len(history)):
            for j in range(i + 1, len(history)):
                if history[i].start_date and history[j].start_date:
                    if history[i].end_date and history[j].end_date:
                        if (
                            history[i].start_date < history[j].end_date
                            and history[j].start_date < history[i].end_date
                        ):
                            contradictions.append(
                                f"Timeline overlap between "
                                f"{history[i].company} ({history[i].title}) and "
                                f"{history[j].company} ({history[j].title})"
                            )

        return contradictions

    def _detect_seniority_skill_gap(self, candidate: Candidate) -> List[str]:
        if not candidate.career_history:
            return None

        last_role = candidate.career_history[-1]
        has_senior_title = any(
            kw in last_role.title.lower()
            for kw in ["senior", "lead", "staff", "principal", "architect", "head"]
        )

        if has_senior_title:
            advanced_skills = sum(
                1 for s in candidate.skills if s.proficiency in ("advanced", "expert")
            )
            if advanced_skills < 3 and len(candidate.skills) < 5:
                return (
                    f"Senior title '{last_role.title}' but only {advanced_skills} "
                    f"advanced skills and {len(candidate.skills)} total skills"
                )

        return None
