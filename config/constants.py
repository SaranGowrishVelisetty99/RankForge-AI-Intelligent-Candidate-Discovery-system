from typing import Dict, List

CONSULTING_FIRMS: List[str] = [
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "ltimindtree",
    "mphasis", "hexaware", "cyient", "persistent systems",
]

PENALTIES: Dict[str, float] = {
    "consulting_only": 0.08,
    "research_only": 0.12,
    "llm_only_no_ir": 0.40,
    "cv_speech_no_nlp": 0.35,
    "title_inflation": 0.60,
    "honeypot_detected": 0.00,
}

DEFAULT_WEIGHTS: Dict[str, float] = {
    "jd_fit": 0.30,
    "evidence": 0.20,
    "credibility": 0.15,
    "trajectory": 0.10,
    "recruitability": 0.10,
    "availability": 0.05,
    "authenticity": 0.05,
    "consistency": 0.05,
}

SCORE_RANGE = (0, 100)

TOP_K_INITIAL = 300
TOP_K_FINAL = 100
