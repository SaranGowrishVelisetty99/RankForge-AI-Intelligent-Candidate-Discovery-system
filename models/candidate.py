from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class Profile(BaseModel):
    anonymized_name: str = ""
    headline: str = ""
    summary: str = ""
    location: str = ""
    country: str = ""
    years_of_experience: float = 0.0
    current_title: str = ""
    current_company: str = ""
    current_company_size: str = ""
    current_industry: str = ""


class CareerEntry(BaseModel):
    company: str = ""
    title: str = ""
    start_date: str = ""
    end_date: Optional[str] = None
    duration_months: int = 0
    is_current: bool = False
    industry: str = ""
    company_size: str = ""
    description: str = ""


class Education(BaseModel):
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    start_year: int = 0
    end_year: int = 0
    grade: Optional[str] = None
    tier: str = "unknown"


class Skill(BaseModel):
    name: str = ""
    proficiency: str = "beginner"
    endorsements: int = 0
    duration_months: int = 0


class Certification(BaseModel):
    name: str = ""
    issuer: str = ""
    year: int = 0


class RedrobSignals(BaseModel):
    profile_completeness_score: float = 0.0
    signup_date: str = ""
    last_active_date: str = ""
    open_to_work_flag: bool = False
    profile_views_received_30d: int = 0
    applications_submitted_30d: int = 0
    recruiter_response_rate: float = 0.0
    avg_response_time_hours: float = 0.0
    skill_assessment_scores: Dict[str, float] = Field(default_factory=dict)
    connection_count: int = 0
    endorsements_received: int = 0
    notice_period_days: int = 0
    expected_salary_range_inr_lpa: Dict[str, float] = Field(default_factory=lambda: {"min": 0.0, "max": 0.0})
    preferred_work_mode: str = "flexible"
    willing_to_relocate: bool = False
    github_activity_score: float = -1.0
    search_appearance_30d: int = 0
    saved_by_recruiters_30d: int = 0
    interview_completion_rate: float = 0.0
    offer_acceptance_rate: float = -1.0
    verified_email: bool = False
    verified_phone: bool = False
    linkedin_connected: bool = False


class Candidate(BaseModel):
    candidate_id: str = ""
    profile: Profile = Field(default_factory=Profile)
    career_history: List[CareerEntry] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills: List[Skill] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    languages: List[Dict] = Field(default_factory=list)
    redrob_signals: RedrobSignals = Field(default_factory=RedrobSignals)


class CandidateFeatures(BaseModel):
    candidate_id: str = ""
    jd_fit: float = 0.0
    evidence: float = 0.0
    credibility: float = 0.0
    trajectory: float = 0.0
    recruitability: float = 0.0
    availability: float = 0.0
    authenticity: float = 0.0
    consistency: float = 0.0
    ranking_score: float = 0.0
    review_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0
    reasoning: str = ""


class SkillTaxonomyNode(BaseModel):
    name: str
    domain: str
    synonyms: List[str] = Field(default_factory=list)
    children: List[str] = Field(default_factory=list)
    parents: List[str] = Field(default_factory=list)
