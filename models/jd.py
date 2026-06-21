from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class JDRequirements(BaseModel):
    must_have: List[str] = Field(default_factory=list)
    preferred: List[str] = Field(default_factory=list)
    seniority: str = ""
    experience_required: int = 0
    domain_requirements: List[str] = Field(default_factory=list)
    behavioral_traits: List[str] = Field(default_factory=list)
    weighting_strategy: Dict[str, float] = Field(default_factory=dict)
    hidden_expectations: List[str] = Field(default_factory=list)
    jd_embedding: Optional[List[float]] = None
