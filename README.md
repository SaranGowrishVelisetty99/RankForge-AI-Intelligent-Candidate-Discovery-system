# RankForge AI

### Explainable Multi-Agent Candidate Intelligence Engine

RankForge AI is an AI-powered recruiting intelligence system for the **Redrob Intelligent Candidate Discovery Challenge**. It processes 100,000 candidate profiles against a job description, evaluates them across 8 dimensions using a hybrid of LLM agents and deterministic scoring, and produces a ranked Top 100 with explainable reasoning.

## Why RankForge AI is Different from Traditional ATS Screening

Traditional ATS systems rank candidates by keyword-match percentage against a JD. RankForge AI was intentionally designed to avoid this trap:

| Traditional ATS | RankForge AI |
|----------------|-------------|
| Counts keyword matches in resumes | Uses TF-IDF cosine similarity to measure semantic document-level alignment with the JD |
| Treats all skills equally | Knowledge graph expands skills into ontology domains (e.g. "PyTorch" → deep_learning, ml_systems) and measures overlap |
| No career trajectory awareness | Trajectory agent analyzes progression quality, promotion patterns, and role seniority |
| Cannot detect inflated claims | Evidence agent verifies each skill appears in actual career history; contradictions agent flags title-skill gaps |
| Ignores platform signals | Recruitability and availability agents use response rates, notice periods, relocation willingness |
| No penalty for disqualifying patterns | Consulting-only careers, research-only profiles, LangChain-only AI experience, title inflation — all multiplies score down |
| Only reads the resume | 10 independent agents each analyze a different dimension, then combine into a weighted score |
| Black-box ranking | Every output row includes human-readable reasoning in a standardized format |

**The result:** A candidate with strong semantic alignment, verified skills, clear career progression, and high hiring signal scores higher — even if their resume doesn't contain every keyword from the JD.

## Architecture

### Agent Architecture

```
                                    ┌─────────────────────┐
                                    │   Job Description    │
                                    └──────────┬──────────┘
                                               │
                               ┌───────────────▼───────────────┐
                               │   JD Intelligence Agent       │
                               │   (Nemotron 3 Super LLM)      │
                               │   Extracts structured reqs    │
                               └───────────────┬───────────────┘
                                               │
                               ┌───────────────▼───────────────┐
                               │   Structured JDRequirements   │
                               └───────────────┬───────────────┘
                                               │
             ┌─────────────────────────────────┼─────────────────────────────────┐
             │                                 │                                 │
             ▼                                 ▼                                 ▼
 ┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
 │   8 Deterministic     │       │     TF-IDF Embedding   │       │     Penalty Engine     │
 │       Agents          │       │     Service (Sklearn)   │       │                        │
 │                       │       │                        │       │  consulting_only  → ×0.08│
 │  ┌─────────────────┐  │       │  Fits on 20K samples   │       │  research_only    → ×0.12│
 │  │ JD Fit Agent    │  │       │  Encodes all 100K + JD │       │  llm_only_no_ir   → ×0.40│
 │  │ (TF-IDF + rules) │  │       │  → 1463-dim vectors    │       │  title_inflation  → ×0.60│
 │  └─────────────────┘  │       └───────────────────────┘       │  honeypot         → ×0.00│
 │                       │                                        └───────────────────────┘
 │  ┌─────────────────┐  │
 │  │ Evidence Agent  │  │
 │  │ (claim check)   │  │
 │  └─────────────────┘  │
 │                       │
 │  ┌─────────────────┐  │
 │  │ Credibility     │  │
 │  │ Agent (skills)  │  │
 │  └─────────────────┘  │
 │                       │
 │  ┌─────────────────┐  │
 │  │ Trajectory      │  │
 │  │ Agent (career)  │  │
 │  └─────────────────┘  │
 │                       │
 │  ┌─────────────────┐  │
 │  │ Recruitability  │  │
 │  │ Agent (signals) │  │
 │  └─────────────────┘  │
 │                       │
 │  ┌─────────────────┐  │
 │  │ Availability    │  │
 │  │ Agent (readiness)│  │
 │  └─────────────────┘  │
 │                       │
 │  ┌─────────────────┐  │
 │  │ Authenticity    │  │
 │  │ Agent (red flags)│  │
 │  └─────────────────┘  │
 │                       │
 │  ┌─────────────────┐  │
 │  │ Contradiction   │  │
 │  │ Agent (mismatch)│  │
 │  └─────────────────┘  │
 └───────────────────────┘
             │
             └─────────────────────────────────┬─────────────────────────────────┘
                                               │
                               ┌───────────────▼───────────────┐
                               │   Recruiter Review Agent      │
                               │   (Nemotron 3 Super LLM)      │
                               │   Holistic review of Top 300  │
                               └───────────────┬───────────────┘
                                               │
                                           ┌───▼───┐
                                           │ Top   │
                                           │ 100   │
                                           │ CSV   │
                                           └───────┘
```

### Agent Summary

| Agent | Type | Purpose |
|-------|------|---------|
| **JD Intelligence** | Nemotron LLM | Extract structured requirements from JD text |
| **JD Fit** | TF-IDF + Rules | Semantic alignment via cosine similarity + skill coverage + ontology overlap |
| **Evidence** | Rule-based | Verify skill claims appear in career history or summary |
| **Credibility** | Rule-based | Score confidence based on endorsements, proficiency, assessment scores |
| **Trajectory** | Rule-based | Analyze career progression quality and growth |
| **Recruitability** | Rule-based | Measure successful-hire likelihood from platform signals |
| **Availability** | Rule-based | Assess readiness (notice period, relocation, open-to-work) |
| **Authenticity** | Rule-based | Detect suspicious/inflated profiles |
| **Contradiction** | Rule-based | Find timeline overlaps, title-skill gaps, AI-vs-data mismatches |
| **Recruiter Review** | Nemotron LLM | Holistic review of Top 300, outputs final adjusted score |

### Data Flow

```
                                 DATA FLOW
                                 ═════════

  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────┐
  │ candidates   │───▸│ Candidate Parser │───▸│ List[Candidate]      │
  │ .jsonl       │    │ (streaming JSONL)│    │ (100,000 objects)    │
  └──────────────┘    └──────────────────┘    └──────────┬───────────┘
                                                         │
  ┌──────────────┐    ┌──────────────────┐               │
  │ job_descri-  │───▸│ JD Intelligence  │               │
  │ ption.txt    │    │ or fallback      │               │
  └──────────────┘    └──────┬───────────┘               │
                             │                           │
                             ▼                           ▼
                    ┌──────────────────┐    ┌──────────────────────┐
                    │ JDRequirements   │    │ TF-IDF Vectorizer   │
                    │ (structured)     │    │ fitted on 20K       │
                    └──────┬───────────┘    └──────────┬───────────┘
                           │                           │
                           └──────────┬────────────────┘
                                      │
                          ┌───────────▼───────────┐
                          │ Feature Extraction    │
                          │ (20 batches × 5000)   │
                          │                      │
                          │ For each candidate:  │
                          │  1. JD Fit Agent     │
                          │     → TF-IDF vec     │
                          │     → cosine sim     │
                          │     → skill coverage │
                          │     → ontology %     │
                          │  2. Evidence Agent   │
                          │  3. Credibility Ag.  │
                          │  4. Trajectory Ag.   │
                          │  5. Recruitability   │
                          │  6. Availability     │
                          │  7. Authenticity     │
                          │  8. Contradiction    │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │ 8 features × weight   │
                          │ + penalty multiplier  │
                          │ = ranking_score       │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │ Sort by score → Top 300│
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │ Recruiter Review      │
                          │ (Nemotron batched)    │
                          │ → adjusted scores     │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │ Final Ranker          │
                          │ → 0-1 normalize       │
                          │ → tiebreak by ID      │
                          │ → Top 100             │
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │ Reasoning Generator   │
                          │ → "Title with X.X yrs;│
                          │    N AI core skills;  │
                          │    response rate X.XX."│
                          └───────────┬───────────┘
                                      │
                          ┌───────────▼───────────┐
                          │ CSV Generator         │
                          │ → submission.csv      │
                          │ → validate_submission │
                          └───────────────────────┘
```

## Setup

### Prerequisites

- Python 3.10+
- OpenRouter API key (free, for Nemotron LLM calls)

### Installation

```bash
git clone <repo-url>
cd RankForge-AI
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your OpenRouter API key:
# OPENROUTER_API_KEY=sk-or-v1-...
```

The default model is **NVIDIA Nemotron 3 Super** (`nvidia/nemotron-3-super-120b-a12b:free`), which is free on OpenRouter — **$0 cost** for all LLM calls.

### Dataset

Place the challenge dataset in `data/`:
- `data/candidates.jsonl` — 100,000 candidate profiles (JSONL, ~487 MB)
- `data/job_description.txt` — Job description for the target role

The file `job_description.docx` can be extracted using `python -c "import docx; doc = docx.Document('data/job_description.docx'); open('data/job_description.txt','w').write('\n'.join(p.text for p in doc.paragraphs))"`.

## Usage

### Full Pipeline (with LLM)

```bash
python main.py --candidates data/candidates.jsonl --out submission.csv
```

### With Custom JD

```bash
python main.py --candidates data/candidates.jsonl --jd path/to/jd.txt --out submission.csv
```

### Skip LLM (fallback mode, no API key needed)

```bash
python main.py --candidates data/candidates.jsonl --skip-llm --out submission.csv
```

This uses built-in JD analysis and fallback reviews. Produces valid output.

### Validate Submission

```bash
python data/validate_submission.py submission.csv
```

## Output Format

```csv
candidate_id,rank,score,reasoning
CAND_0078492,1,0.9709,Recommendation Systems Engineer with 5.1 yrs; 13 AI core skills; response rate 0.70.
CAND_0095619,2,0.9474,NLP Engineer with 15.6 yrs; 14 AI core skills; response rate 0.90.
CAND_0030031,3,0.9300,AI Engineer with 5.7 yrs; 20 AI core skills; response rate 0.94.
...
```

- **Exactly 100 rows**
- **Unique candidate IDs**, no duplicates
- **Ranks 1–100**, strictly increasing
- **Scores 0.0000–1.0000**, monotonically decreasing
- **Reasoning format**: `Current Title with X.X yrs; N AI core skills; response rate X.XX.`

### Running Tests

```bash
python -m pytest tests/ -v
```

26 tests covering all agents, ranking, penalties, CSV validation, and candidate parsing.

## Design Philosophy

- **LLM where reasoning is required** — Only 2 agents use Nemotron (JD analysis + recruiter review). 8 agents use rule-based algorithms, keeping LLM costs at $0.
- **TF-IDF for semantic matching** — Lightweight, CPU-friendly approach that measures document-to-JD similarity via weighted term vectors. Fits and transforms 100K candidates without requiring a GPU.
- **Not keyword filtering** — Every candidate is scored using a blend of TF-IDF cosine similarity (40%), knowledge graph ontology overlap, skill coverage analysis, and headline matching. Consulting-only and research-only profiles are penalized automatically.
- **Memory efficient** — Streaming JSONL parser + batch processing handles 100K candidates in under 2 GB RAM.
- **Explainable** — Every ranking decision has human-readable reasoning matching the challenge format.
- **Graceful degradation** — LLM calls have fallbacks; rate limits or API errors don't prevent valid output.

## LLM Usage

| Agent | Calls | Model | Cost |
|-------|-------|-------|------|
| JD Intelligence | 1 | Nemotron 3 Super | Free |
| Recruiter Review | 30 (batched) | Nemotron 3 Super | Free |
| **Total** | **31** | — | **$0.00** |

Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b:free`) is available on OpenRouter's free tier — no API charges for this model. Without an API key, `--skip-llm` uses built-in fallbacks that produce equally valid submissions.

## Penalties

The system applies multiplicative penalty multipliers to disqualify candidates with problematic career patterns:

| Condition | Multiplier | Effect |
|-----------|-----------|--------|
| Consulting-only career (TCS, Infosys, Wipro, etc.) | 0.08 | 92% score reduction |
| Research-only (no production experience) | 0.12 | 88% score reduction |
| LangChain-only without prior ML | 0.40 | 60% score reduction |
| CV/Speech without NLP/IR | 0.35 | 65% score reduction |
| Title inflation (senior title, no advanced skills) | 0.60 | 40% score reduction |
| Honeypot skills detected | 0.00 | Automatic disqualification |

## Project Structure

```
config/         — Pydantic settings, constants, JD config
models/         — Pydantic models for Candidate, JDRequirements
llm/            — OpenRouter client, prompt templates, schemas
core/           — TF-IDF embeddings, knowledge graph, feature extraction, reasoning
agents/         — 10 agents: jd_intelligence, jd_fit, evidence, credibility,
                  trajectory, recruitability, availability, authenticity,
                  contradiction, recruiter_review
ranking/        — Weighted ranking engine, penalty multipliers, final 0-1 normalization
output/         — CSV generation with validation
tests/          — 26 unit tests
main.py         — Async entry point
```
