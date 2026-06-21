from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class JDAnalysisOutput(BaseModel):
    must_have: List[str] = Field(description="Critical skills/requirements for the role")
    preferred: List[str] = Field(description="Nice-to-have skills/requirements")
    seniority: str = Field(description="Expected seniority level (junior/mid/senior/lead)")
    experience_required: int = Field(description="Minimum years of experience required")
    domain_requirements: List[str] = Field(description="Specific domain or industry knowledge needed")
    behavioral_traits: List[str] = Field(description="Desired behavioral/cultural traits")
    weighting_strategy: Dict[str, float] = Field(
        description="Relative importance weights for requirement categories"
    )
    hidden_expectations: List[str] = Field(
        description="Unofficial or implicit expectations not stated in JD"
    )


class RecruiterReviewOutput(BaseModel):
    review_score: float = Field(
        ge=0, le=100,
        description="Holistic recruiter assessment score 0-100"
    )
    summary: str = Field(description="1-2 sentence candidate summary")
    recommendation: str = Field(description="hire / consider / reject decision")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    hiring_risk: str = Field(default="low", description="low / medium / high")
    career_quality: str = Field(default="good", description="poor / fair / good / excellent")


class SkillAssessmentOutput(BaseModel):
    validated_skills: List[str] = Field(default_factory=list)
    unsupported_skills: List[str] = Field(default_factory=list)
    evidence_score: float = Field(ge=0, le=100)
