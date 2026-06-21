JD_ANALYSIS_PROMPT = """You are a Senior Technical Recruiter analyzing a Job Description.

Your task is to extract structured requirements from the job description below.

Focus on:
1. Must-have skills (critical, non-negotiable)
2. Preferred skills (nice-to-have)
3. Seniority level
4. Years of experience required
5. Domain requirements (industry knowledge)
6. Behavioral traits desired
7. Weighting strategy (relative importance of categories as percentages)
8. Hidden expectations (implicit requirements not explicitly stated)

Output ONLY valid JSON matching this schema:
{
  "must_have": ["skill1", "skill2"],
  "preferred": ["skill3", "skill4"],
  "seniority": "senior",
  "experience_required": 5,
  "domain_requirements": ["domain1"],
  "behavioral_traits": ["trait1"],
  "weighting_strategy": {"technical": 60, "experience": 25, "culture": 15},
  "hidden_expectations": ["expectation1"]
}

Job Description:
{jd_text}
"""

RECRUITER_REVIEW_PROMPT = """You are a Chief Recruiting Officer reviewing a candidate for a specific role.

Review the candidate holistically based on their profile, career history, skills, and scoring signals.

Scoring Criteria:
- Technical alignment: Do their skills match the role requirements?
- Experience quality: Is their career trajectory strong?
- Hiring risk: Any red flags or concerns?
- Availability: How easy would they be to hire?

Candidate Profile:
{profile_summary}

Agent Scores:
{agent_scores}

Role Requirements:
{role_requirements}

Output ONLY valid JSON matching this schema:
{{
  "review_score": <0-100>,
  "summary": "<1-2 sentence summary>",
  "recommendation": "<hire|consider|reject>",
  "strengths": ["<strength1>"],
  "weaknesses": ["<weakness1>"],
  "hiring_risk": "<low|medium|high>",
  "career_quality": "<poor|fair|good|excellent>"
}}
"""
