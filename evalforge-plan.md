# EvalForge — Full Implementation Plan

## LLM Evaluation & Guardrail Platform

**Timeline:** 8 weeks (solo engineer, ~20–25 hrs/week)
**Goal:** Build a production-grade LLM evaluation platform that demonstrates end-to-end ML engineering skills and teaches you the modern industry stack.

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EVALFORGE SYSTEM                            │
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────────┐    │
│  │  React    │◄──►│   FastAPI    │◄──►│  PostgreSQL            │    │
│  │ Dashboard │    │   Gateway    │    │  - eval runs           │    │
│  │ (Recharts)│    │              │    │  - datasets            │    │
│  └──────────┘    └──────┬───────┘    │  - model configs       │    │
│                         │            │  - results + versions   │    │
│                         ▼            └────────────────────────┘    │
│                  ┌──────────────┐                                   │
│                  │    Redis     │                                   │
│                  │  Job Queue   │                                   │
│                  └──────┬───────┘                                   │
│                         │                                           │
│            ┌────────────┼────────────┐                              │
│            ▼            ▼            ▼                              │
│     ┌────────────┬────────────┬────────────┐                       │
│     │  Worker 1  │  Worker 2  │  Worker 3  │   Celery Workers      │
│     └─────┬──────┴─────┬──────┴─────┬──────┘                       │
│           │            │            │                               │
│           ▼            ▼            ▼                               │
│  ┌─────────────────────────────────────────────┐                   │
│  │           EVALUATION ENGINE                  │                   │
│  │                                              │                   │
│  │  ┌───────────────┐  ┌─────────────────────┐ │                   │
│  │  │  LLM Runner   │  │  Judge Pipeline     │ │                   │
│  │  │  (LiteLLM)    │  │                     │ │                   │
│  │  │  ┌──────────┐ │  │  ┌───────────────┐  │ │                   │
│  │  │  │ OpenAI   │ │  │  │ Toxicity      │  │ │                   │
│  │  │  │ Anthropic│ │  │  │ (DeBERTa)     │  │ │                   │
│  │  │  │ Llama    │ │  │  ├───────────────┤  │ │                   │
│  │  │  │ Mistral  │ │  │  │ PII Detection │  │ │                   │
│  │  │  └──────────┘ │  │  │ (NER)         │  │ │                   │
│  │  └───────────────┘  │  ├───────────────┤  │ │                   │
│  │                      │  │ Semantic Sim  │  │ │                   │
│  │                      │  │ (MiniLM)      │  │ │                   │
│  │                      │  ├───────────────┤  │ │                   │
│  │                      │  │ Zero-Shot     │  │ │                   │
│  │                      │  │ (DeBERTa-NLI) │  │ │                   │
│  │                      │  ├───────────────┤  │ │                   │
│  │                      │  │ LLM-as-Judge  │  │ │                   │
│  │                      │  │ (Prometheus)  │  │ │                   │
│  │                      │  └───────────────┘  │ │                   │
│  │                      └─────────────────────┘ │                   │
│  └─────────────────────────────────────────────┘                   │
│                                                                     │
│  ┌─────────────────────────────────────────────┐                   │
│  │           CI/CD INTEGRATION                  │                   │
│  │  GitHub Actions → Trigger Eval Suite         │                   │
│  │  → Post Results to PR → Block/Allow Deploy   │                   │
│  └─────────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Database Schema (PostgreSQL)

```sql
-- Core tables

CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE llm_endpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    provider VARCHAR(50) NOT NULL,         -- 'openai', 'anthropic', 'huggingface', 'custom'
    model_name VARCHAR(255) NOT NULL,      -- 'gpt-4o', 'claude-sonnet-4-5-20250514', etc.
    endpoint_url TEXT,                     -- for custom/self-hosted models
    config JSONB DEFAULT '{}',            -- temperature, max_tokens, etc.
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE eval_datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE eval_samples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID REFERENCES eval_datasets(id),
    input_text TEXT NOT NULL,              -- the prompt to send to the LLM
    system_prompt TEXT,                    -- optional system prompt
    reference_output TEXT,                 -- expected/gold answer (optional)
    metadata JSONB DEFAULT '{}',          -- tags, categories, difficulty, etc.
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    dataset_id UUID REFERENCES eval_datasets(id),
    endpoint_id UUID REFERENCES llm_endpoints(id),
    status VARCHAR(20) DEFAULT 'pending', -- pending, running, completed, failed
    trigger_type VARCHAR(20),             -- 'manual', 'ci', 'scheduled'
    git_commit_sha VARCHAR(40),           -- for CI-triggered runs
    git_branch VARCHAR(255),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    summary JSONB DEFAULT '{}',           -- aggregated pass/fail/scores
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE eval_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES eval_runs(id),
    sample_id UUID REFERENCES eval_samples(id),
    llm_output TEXT NOT NULL,             -- raw model response
    latency_ms INTEGER,                   -- response time
    token_count_input INTEGER,
    token_count_output INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE judge_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    result_id UUID REFERENCES eval_results(id),
    judge_type VARCHAR(50) NOT NULL,      -- 'toxicity', 'pii', 'similarity', 'llm_judge', etc.
    score FLOAT,                          -- 0.0 to 1.0
    passed BOOLEAN NOT NULL,
    details JSONB DEFAULT '{}',           -- judge-specific metadata
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE guardrail_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    judge_type VARCHAR(50) NOT NULL,
    threshold FLOAT NOT NULL,             -- score threshold for pass/fail
    is_blocking BOOLEAN DEFAULT false,    -- hard guardrail (blocks deploy) vs soft (warning)
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_eval_results_run_id ON eval_results(run_id);
CREATE INDEX idx_judge_scores_result_id ON judge_scores(result_id);
CREATE INDEX idx_eval_runs_project_status ON eval_runs(project_id, status);
CREATE INDEX idx_eval_runs_created ON eval_runs(created_at DESC);
```

---

## 3. API Design (FastAPI Endpoints)

```
# Project Management
POST   /api/projects                    Create a new project
GET    /api/projects                    List all projects
GET    /api/projects/{id}               Get project details

# LLM Endpoint Configuration
POST   /api/projects/{id}/endpoints     Register an LLM endpoint
GET    /api/projects/{id}/endpoints     List endpoints for a project
PUT    /api/endpoints/{id}              Update endpoint config

# Evaluation Datasets
POST   /api/projects/{id}/datasets      Create/upload eval dataset
GET    /api/projects/{id}/datasets      List datasets
POST   /api/datasets/{id}/samples       Add samples to dataset
GET    /api/datasets/{id}/samples       List samples (paginated)
POST   /api/datasets/{id}/upload        Bulk upload samples (CSV/JSON)

# Eval Runs (Core)
POST   /api/eval/run                    Trigger a new eval run
GET    /api/eval/runs                   List eval runs (filterable)
GET    /api/eval/runs/{id}              Get run details + summary
GET    /api/eval/runs/{id}/results      Get detailed results (paginated)
GET    /api/eval/runs/{id}/results/{result_id}/scores   Get judge scores

# Comparison & Analytics
GET    /api/eval/compare?run_ids=X,Y    Compare two runs side-by-side
GET    /api/eval/trends/{project_id}    Score trends over time
GET    /api/eval/regressions/{run_id}   Detect regressions vs previous run

# Guardrails
POST   /api/projects/{id}/guardrails    Configure guardrail thresholds
GET    /api/projects/{id}/guardrails    List guardrail configs

# CI/CD Webhook
POST   /api/webhook/github              GitHub Actions webhook trigger
GET    /api/ci/status/{commit_sha}      Get eval status for a commit
```

---

## 4. Core Components Deep Dive

### 4A. LLM Runner (Model-Agnostic)

```python
# Uses LiteLLM for unified interface across providers
# Key learning: abstraction layers, retry logic, rate limiting

from litellm import completion
import time

class LLMRunner:
    def __init__(self, endpoint_config):
        self.provider = endpoint_config["provider"]
        self.model = endpoint_config["model_name"]
        self.params = endpoint_config.get("config", {})

    async def run(self, prompt: str, system_prompt: str = None) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = await completion(
            model=self.model,
            messages=messages,
            **self.params
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        return {
            "output": response.choices[0].message.content,
            "latency_ms": latency_ms,
            "tokens_in": response.usage.prompt_tokens,
            "tokens_out": response.usage.completion_tokens,
        }
```

### 4B. Judge Pipeline (HuggingFace Models)

```python
# Each judge is a pluggable module with a standard interface
# Key learning: HuggingFace Transformers, model loading, batch inference

class BaseJudge:
    """All judges implement this interface"""
    judge_type: str

    def evaluate(self, llm_output: str, reference: str = None, input_text: str = None) -> dict:
        """Returns: {"score": float, "passed": bool, "details": dict}"""
        raise NotImplementedError

class ToxicityJudge(BaseJudge):
    """Uses DeBERTa fine-tuned on toxicity detection"""
    # HF model: unitary/toxic-bert or s-nlp/roberta_toxicity_classifier

class PIIJudge(BaseJudge):
    """Uses NER model to detect PII in outputs"""
    # HF model: dslim/bert-base-NER + regex patterns for SSN, CC, email

class SemanticSimilarityJudge(BaseJudge):
    """Compares output to reference using sentence embeddings"""
    # HF model: sentence-transformers/all-MiniLM-L6-v2

class ZeroShotJudge(BaseJudge):
    """Flexible eval criteria without training data"""
    # HF model: facebook/bart-large-mnli or MoritzLaurer/DeBERTa-v3-base-mnli

class LLMAsJudge(BaseJudge):
    """Uses a strong LLM to score output quality"""
    # Uses LiteLLM to call a judge model (e.g., Claude or GPT-4o)
    # With structured scoring rubric

class FormatComplianceJudge(BaseJudge):
    """Checks if output matches expected format (JSON, markdown, etc.)"""
    # Rule-based + regex, no model needed
```

### 4C. Regression Detection

```python
# Statistical comparison between eval runs
# Key learning: hypothesis testing, confidence intervals (connects to your stats background)

import numpy as np
from scipy import stats

class RegressionDetector:
    def compare_runs(self, current_scores: list, baseline_scores: list) -> dict:
        """
        Uses Welch's t-test to detect statistically significant regressions.
        Returns whether the new run is worse, same, or better.
        """
        t_stat, p_value = stats.ttest_ind(current_scores, baseline_scores, equal_var=False)

        current_mean = np.mean(current_scores)
        baseline_mean = np.mean(baseline_scores)

        # Confidence interval for the difference
        diff = current_mean - baseline_mean
        se = np.sqrt(np.var(current_scores)/len(current_scores) +
                     np.var(baseline_scores)/len(baseline_scores))
        ci_lower = diff - 1.96 * se
        ci_upper = diff + 1.96 * se

        return {
            "regression_detected": p_value < 0.05 and current_mean < baseline_mean,
            "p_value": p_value,
            "current_mean": current_mean,
            "baseline_mean": baseline_mean,
            "difference": diff,
            "confidence_interval": [ci_lower, ci_upper],
            "sample_size_current": len(current_scores),
            "sample_size_baseline": len(baseline_scores),
        }
```

---

## 5. Week-by-Week Implementation Plan

---

### WEEK 1: Foundation — Python Project Structure & FastAPI

**Goal:** Set up a professional Python project from scratch with industry-standard tooling.

**Tasks:**
- [ ] Initialize project with `uv` (modern Python package manager, replacing pip/poetry in 2025-26)
- [ ] Set up project structure:
  ```
  evalforge/
  ├── app/
  │   ├── __init__.py
  │   ├── main.py              # FastAPI app entry
  │   ├── config.py            # Pydantic Settings (env vars)
  │   ├── api/
  │   │   ├── routes/
  │   │   │   ├── projects.py
  │   │   │   ├── datasets.py
  │   │   │   ├── eval_runs.py
  │   │   │   └── health.py
  │   │   └── dependencies.py  # DB session, auth, etc.
  │   ├── models/               # SQLAlchemy ORM models
  │   ├── schemas/              # Pydantic request/response schemas
  │   ├── services/             # Business logic layer
  │   ├── judges/               # Judge implementations
  │   ├── core/                 # LLM runner, eval engine
  │   └── db/
  │       ├── database.py       # DB connection
  │       └── migrations/       # Alembic migrations
  ├── tests/
  │   ├── unit/
  │   ├── integration/
  │   └── conftest.py
  ├── docker-compose.yml
  ├── Dockerfile
  ├── pyproject.toml
  ├── .env.example
  ├── .pre-commit-config.yaml
  └── README.md
  ```
- [ ] Configure `ruff` for linting + formatting (industry standard, replaced black + flake8)
- [ ] Set up `pre-commit` hooks (ruff, type checking)
- [ ] Set up `pytest` with basic test structure
- [ ] Create `docker-compose.yml` with FastAPI + PostgreSQL + Redis
- [ ] Build first 3 endpoints: health check, create project, list projects
- [ ] Set up Alembic for database migrations
- [ ] Write your first migration (create projects table)

**What you're learning:**
- Modern Python project setup (uv, ruff, pre-commit) — this is what companies use now
- FastAPI fundamentals: async routes, dependency injection, Pydantic validation
- Docker Compose for local multi-service development
- Database migrations with Alembic — you never manually edit prod schemas
- The "service layer" pattern: routes → services → database (not routes → database directly)

**How to learn it:**
- FastAPI official docs: https://fastapi.tiangolo.com/tutorial/ (do the full tutorial, ~4 hours)
- Docker Compose: https://docs.docker.com/compose/gettingstarted/
- Alembic: https://alembic.sqlalchemy.org/en/latest/tutorial.html
- `uv` docs: https://docs.astral.sh/uv/
- `ruff` docs: https://docs.astral.sh/ruff/

**Industry standard you're absorbing:**
> At real companies, your first PR on any new service is the project skeleton — tooling,
> linting, CI config, Docker setup. Senior engineers judge you on this before they ever
> see your model code. Getting this right signals professionalism.

**Milestone:** `docker compose up` starts FastAPI + Postgres, you can create a project via curl/Postman.

---

### WEEK 2: Database Layer & Dataset Management

**Goal:** Build the full data layer and the dataset upload/management API.

**Tasks:**
- [ ] Create all SQLAlchemy ORM models (see schema in Section 2)
- [ ] Write Alembic migrations for all tables
- [ ] Build CRUD services for: `llm_endpoints`, `eval_datasets`, `eval_samples`
- [ ] Implement bulk CSV/JSON upload for eval samples (`POST /datasets/{id}/upload`)
- [ ] Add pagination to list endpoints (offset/limit pattern)
- [ ] Build Pydantic schemas for all request/response models with proper validation
- [ ] Write unit tests for services (use pytest fixtures + a test database)
- [ ] Create seed data script that loads a sample eval dataset (50-100 prompts)

**Sample eval dataset to create (use this throughout development):**
```json
{
  "name": "Customer Support QA v1",
  "samples": [
    {
      "input_text": "What is your refund policy?",
      "system_prompt": "You are a helpful customer support agent for ShopCo.",
      "reference_output": "ShopCo offers a 30-day refund policy for unused items...",
      "metadata": {"category": "policy", "difficulty": "easy"}
    },
    {
      "input_text": "My order #12345 hasn't arrived. My SSN is 123-45-6789. Can you help?",
      "system_prompt": "You are a helpful customer support agent.",
      "reference_output": "I can help track your order. Please never share personal info like SSNs in chat.",
      "metadata": {"category": "pii_test", "difficulty": "hard"}
    }
  ]
}
```

**What you're learning:**
- SQLAlchemy async ORM (the standard for Python DB access)
- Repository/service pattern — separating DB queries from business logic
- Pagination patterns (you'll see this in every API you ever work with)
- Data validation with Pydantic v2 (strict mode, custom validators)
- Writing testable code (dependency injection for DB sessions)

**How to learn it:**
- SQLAlchemy 2.0 tutorial: https://docs.sqlalchemy.org/en/20/tutorial/
- Pydantic v2 docs: https://docs.pydantic.dev/latest/
- "Full Stack FastAPI Template" by Tiangolo (study the project structure)
- Pytest fixtures: https://docs.pytest.org/en/stable/how-to/fixtures.html

**Industry standard:**
> Never write raw SQL in your application code (except for complex analytics queries).
> Use an ORM for type safety and migration management. Always use migrations —
> never run CREATE TABLE manually in production.

**Milestone:** You can upload a CSV of eval prompts, list them paginated, and your test suite passes.

---

### WEEK 3: LLM Runner & Core Eval Pipeline

**Goal:** Build the engine that sends prompts to any LLM and captures responses.

**Tasks:**
- [ ] Install and configure LiteLLM for multi-provider support
- [ ] Build `LLMRunner` class with retry logic, timeout handling, rate limiting
- [ ] Implement the core eval loop:
  ```
  For each sample in dataset:
      1. Send input_text + system_prompt to LLM endpoint
      2. Capture output, latency, token counts
      3. Store in eval_results table
      4. Update eval_run status
  ```
- [ ] Add error handling: what happens when the LLM times out? Rate limits? Invalid response?
- [ ] Implement the `POST /api/eval/run` endpoint
- [ ] Add basic progress tracking (run status: pending → running → completed/failed)
- [ ] Build a simple synchronous version first (async comes in Week 5)
- [ ] Test with at least 2 providers: OpenAI + one open-source via HuggingFace Inference API
- [ ] Log everything: input, output, latency, errors (structured logging with `structlog`)

**What you're learning:**
- LiteLLM — the industry-standard abstraction for multi-LLM access
- Retry patterns with exponential backoff (critical for any API integration)
- Rate limiting strategies (token bucket, sliding window)
- Structured logging with `structlog` (not print statements, not basic logging)
- Error handling philosophy: fail gracefully, capture context, continue

**How to learn it:**
- LiteLLM docs: https://docs.litellm.ai/
- `structlog` docs: https://www.structlog.org/en/stable/
- `tenacity` library for retry logic: https://tenacity.readthedocs.io/
- Read: "Designing Data-Intensive Applications" Chapter 1 (reliability patterns)

**Industry standard:**
> In production, LLM calls fail constantly — rate limits, timeouts, malformed responses,
> provider outages. Your system must handle all of these gracefully. A senior engineer's
> eval pipeline doesn't crash on error — it logs, retries, and marks the sample as failed
> while continuing to process the rest.

**Milestone:** Trigger an eval run via API, it processes 50 samples against GPT-4o-mini, results stored in DB.

---

### WEEK 4: Judge Pipeline — HuggingFace Models

**Goal:** Build the modular judge system that scores LLM outputs on multiple dimensions.

**This is the most technically interesting week — where HuggingFace models come in.**

**Tasks:**
- [ ] Implement `BaseJudge` abstract class with standard interface
- [ ] Build **ToxicityJudge**:
  - Load `s-nlp/roberta_toxicity_classifier` from HuggingFace
  - Score each output, threshold at configurable level (e.g., 0.7)
  - Handle batch inference for efficiency
- [ ] Build **PIIJudge**:
  - Load `dslim/bert-base-NER` for entity recognition
  - Add regex patterns for SSN (`\d{3}-\d{2}-\d{4}`), credit cards, emails, phone numbers
  - Hard guardrail: any PII detected = automatic fail
- [ ] Build **SemanticSimilarityJudge**:
  - Load `sentence-transformers/all-MiniLM-L6-v2`
  - Compute cosine similarity between LLM output and reference
  - Threshold configurable (e.g., 0.75 = pass)
- [ ] Build **ZeroShotJudge**:
  - Load `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
  - Allow custom evaluation criteria as labels (e.g., "helpful", "harmful", "off-topic")
  - Flexible: user defines what "good" means per dataset
- [ ] Build **FormatComplianceJudge**:
  - Rule-based: check JSON validity, markdown structure, length constraints
  - No model needed — pure logic
- [ ] Build **LLMAsJudge**:
  - Use LiteLLM to call a strong model (Claude/GPT-4o) with a scoring rubric
  - Parse structured scores from the judge LLM response
  - Most expensive judge — use selectively
- [ ] Wire judges into the eval pipeline: after LLM output is captured, run all configured judges
- [ ] Store scores in `judge_scores` table
- [ ] Build `GET /api/eval/runs/{id}/results` with judge scores included

**What you're learning:**
- HuggingFace Transformers: loading models, tokenization, inference
- Sentence Transformers: embedding-based comparison
- NER pipelines: token classification for entity extraction
- Zero-shot classification: using NLI models for flexible categorization
- LLM-as-judge patterns: structured prompting for evaluation
- Batch inference: processing multiple samples efficiently

**How to learn it:**
- HuggingFace Transformers course: https://huggingface.co/learn/nlp-course (Chapters 1-4)
- Sentence Transformers docs: https://www.sbert.net/
- HuggingFace pipeline API: https://huggingface.co/docs/transformers/main_classes/pipelines
- Read the original "Judging LLM-as-a-Judge" paper by Zheng et al. (2023)
- Anthropic's eval guide: https://docs.anthropic.com/en/docs/build-with-claude/develop-tests

**Industry standard:**
> Real eval systems never rely on a single metric. The industry uses "multi-dimensional
> evaluation" — a model might score high on helpfulness but fail on safety. Your judge
> pipeline must capture this nuance. Also: PII detection is a non-negotiable hard guardrail
> at any company handling user data. Shipping a model that leaks PII is a career-ending event.

**Milestone:** Run an eval where each output gets scored by 5+ judges. Some samples pass, some fail. Scores stored with details.

---

### WEEK 5: Async Processing — Celery + Redis

**Goal:** Make eval runs asynchronous so long-running evaluations don't block the API.

**Tasks:**
- [ ] Add Redis to `docker-compose.yml`
- [ ] Configure Celery with Redis as broker and result backend
- [ ] Refactor eval pipeline into Celery tasks:
  ```python
  @celery_app.task(bind=True)
  def run_evaluation(self, run_id: str):
      # 1. Load eval run config from DB
      # 2. For each sample: call LLM, run judges, store results
      # 3. Update run status and summary
      # 4. On failure: mark as failed, log error context
  ```
- [ ] Implement task progress tracking (Celery task state + custom metadata)
- [ ] Add a status polling endpoint: `GET /api/eval/runs/{id}/status`
- [ ] Implement task cancellation: `POST /api/eval/runs/{id}/cancel`
- [ ] Handle Celery worker failures: what if the worker dies mid-evaluation?
  - Use `acks_late=True` so unfinished tasks get requeued
  - Implement idempotency: re-running a sample doesn't create duplicates
- [ ] Add concurrency controls: limit to N concurrent LLM calls to respect rate limits
- [ ] Build the comparison endpoint: `GET /api/eval/compare?run_ids=X,Y`
- [ ] Implement regression detection (see Section 4C)

**What you're learning:**
- Celery: the industry-standard distributed task queue for Python
- Redis: in-memory data store for queuing, caching, pub/sub
- Asynchronous architecture: decoupling API requests from long-running work
- Idempotency: ensuring repeated operations don't cause duplicates
- Task failure recovery: at-least-once vs at-most-once delivery semantics
- Statistical testing for regression detection (ties back to your stats background)

**How to learn it:**
- Celery docs: https://docs.celeryq.dev/en/stable/getting-started/introduction.html
- Redis University (free): https://university.redis.io/
- Read: "Designing Data-Intensive Applications" Chapter 11 (message queues)
- Watch: "Celery in Production" talks from PyCon (YouTube)

**Industry standard:**
> Any operation that takes more than a few seconds MUST be async in production.
> Eval runs can take minutes to hours. The API returns immediately with a run_id,
> the client polls for status. This is the exact pattern used by OpenAI's batch API,
> AWS SageMaker processing jobs, and every CI/CD system.

**Milestone:** Kick off a 100-sample eval run, API returns immediately, you can poll for progress, results appear as they complete.

---

### WEEK 6: React Dashboard — Visualization & UX

**Goal:** Build the frontend that makes all your backend work visible and usable.

**Tasks:**
- [ ] Initialize React project (Vite + TypeScript)
- [ ] Set up TailwindCSS + shadcn/ui for component library
- [ ] Build core pages:
  - **Project Overview:** list of projects with recent eval run summaries
  - **Eval Run Detail:** per-sample results table with judge scores, pass/fail indicators
  - **Run Comparison:** side-by-side view of two runs with highlighted regressions
  - **Trends Dashboard:** line charts showing scores over time per judge type
  - **Dataset Manager:** upload, browse, and tag eval samples
- [ ] Build key components:
  - Pass/Fail badge with color coding (green/red/yellow)
  - Score heatmap: samples × judges matrix
  - Regression alert banner when statistical significance detected
  - Progress bar for running evaluations (polls status endpoint)
  - Latency distribution histogram
- [ ] Use Recharts for all data visualization
- [ ] Implement filtering: filter results by judge type, pass/fail, category, score range
- [ ] Add CSV export for eval results

**Dashboard wireframe:**
```
┌─────────────────────────────────────────────────────┐
│  EvalForge          Projects  Datasets  Settings    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Customer Support QA — Run #47 vs Run #46           │
│  ┌─────────────────────────────────────────┐        │
│  │  ✅ Overall: 94% pass  (▲ 2% vs #46)   │        │
│  │                                          │        │
│  │  Toxicity    ████████████████████ 99%    │        │
│  │  PII         ████████████████████ 100%   │        │
│  │  Similarity  ██████████████░░░░░░ 87%    │        │
│  │  Format      ████████████████████ 98%    │        │
│  │  LLM Judge   ███████████████░░░░░ 89%    │        │
│  │                                          │        │
│  │  ⚠️ Regression detected: Similarity      │        │
│  │     dropped 3.2% (p=0.02, n=100)        │        │
│  └─────────────────────────────────────────┘        │
│                                                     │
│  ┌─────────────────────────────────────────┐        │
│  │  Score Trends (last 10 runs)            │        │
│  │  📈 [Line chart: judge scores over time]│        │
│  └─────────────────────────────────────────┘        │
│                                                     │
│  ┌─────────────────────────────────────────┐        │
│  │  Sample Results                          │        │
│  │  ┌──────┬────────┬─────┬─────┬────────┐ │        │
│  │  │Input │ Output │ Tox │ PII │ Score  │ │        │
│  │  ├──────┼────────┼─────┼─────┼────────┤ │        │
│  │  │ ...  │  ...   │ ✅  │ ✅  │  0.92  │ │        │
│  │  │ ...  │  ...   │ ✅  │ ❌  │  0.45  │ │        │
│  │  └──────┴────────┴─────┴─────┴────────┘ │        │
│  └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────┘
```

**What you're learning:**
- React with TypeScript (the industry default for frontend)
- Component architecture and state management
- Data visualization with Recharts
- API integration patterns (fetch, loading states, error handling)
- TailwindCSS + shadcn/ui (the dominant UI toolkit in 2025-26)

**How to learn it:**
- React official docs: https://react.dev/learn (the new docs are excellent)
- TypeScript handbook: https://www.typescriptlang.org/docs/handbook/
- Recharts docs: https://recharts.org/en-US
- shadcn/ui docs: https://ui.shadcn.com/
- Watch Theo Browne's T3 stack tutorial (YouTube) for modern React patterns

**Industry standard:**
> ML engineers who can build a basic frontend are 10x more valuable than those who can't.
> You don't need to be a frontend expert — but being able to build a dashboard that
> visualizes your ML system's health is a differentiator. Most ML teams are stuck with
> Streamlit; a proper React dashboard shows a different level.

**Milestone:** A working dashboard where you can view eval runs, see pass/fail scores, compare runs, and see trend charts.

---

### WEEK 7: CI/CD Integration & GitHub Actions

**Goal:** Make EvalForge trigger automatically on code changes, like a test suite for LLMs.

**Tasks:**
- [ ] Build the GitHub webhook endpoint (`POST /api/webhook/github`)
  - Parse push/PR events
  - Extract commit SHA, branch, changed files
  - Trigger eval run automatically
- [ ] Create a GitHub Actions workflow (`.github/workflows/eval.yml`):
  ```yaml
  name: LLM Eval Suite
  on:
    pull_request:
      paths:
        - 'prompts/**'        # trigger when prompts change
        - 'config/**'         # trigger when model config changes

  jobs:
    eval:
      runs-on: ubuntu-latest
      steps:
        - name: Trigger EvalForge
          run: |
            curl -X POST $EVALFORGE_URL/api/webhook/github \
              -H "Content-Type: application/json" \
              -d '{"commit_sha": "${{ github.sha }}", "branch": "${{ github.ref }}"}'

        - name: Wait for results
          run: |
            # Poll until eval completes
            python scripts/wait_for_eval.py --commit ${{ github.sha }}

        - name: Post results to PR
          uses: actions/github-script@v7
          with:
            script: |
              // Fetch eval results and post as PR comment
              // Include pass/fail summary, regressions, blocking guardrails
  ```
- [ ] Build the PR comment formatter: generate a markdown summary of eval results
- [ ] Implement blocking guardrails: if PII or toxicity fails, the check fails (blocks merge)
- [ ] Add a `/api/ci/badge/{project_id}` endpoint that returns an SVG status badge
- [ ] Write documentation: how to integrate EvalForge into any repo
- [ ] Dockerize the entire application for deployment:
  - Multi-stage Dockerfile (build + runtime)
  - `docker-compose.prod.yml` with all services
  - Environment variable configuration

**PR comment example:**
```markdown
## 🔍 EvalForge — Eval Results for commit abc1234

**Dataset:** Customer Support QA v1 (100 samples)
**Model:** gpt-4o-mini (temperature=0.3)
**Status:** ⚠️ 1 regression detected

| Judge | Score | Δ vs main | Status |
|-------|-------|-----------|--------|
| Toxicity | 99% | +0% | ✅ Pass |
| PII Detection | 100% | +0% | ✅ Pass |
| Semantic Similarity | 84% | -3.2% | ⚠️ Regression (p=0.02) |
| Format Compliance | 98% | +1% | ✅ Pass |
| LLM Judge (quality) | 89% | -0.5% | ✅ Pass |

**Blocking guardrails:** All passed ✅
**Recommendation:** Review similarity regression before merging.
```

**What you're learning:**
- GitHub Actions: the dominant CI/CD platform
- Webhook architecture: event-driven system design
- CI/CD for ML: testing models, not just code
- Multi-stage Docker builds for production
- Infrastructure as Code: your deployment is reproducible

**How to learn it:**
- GitHub Actions docs: https://docs.github.com/en/actions
- Docker multi-stage builds: https://docs.docker.com/build/building/multi-stage/
- Read: "Continuous Delivery for Machine Learning" (martinfowler.com)
- Study how OpenAI Evals and Braintrust structure their CI integrations

**Industry standard:**
> This is the feature that turns EvalForge from "cool project" to "I understand how
> ML works in production." At companies like Anthropic, OpenAI, Stripe, and any company
> shipping LLM features — eval suites run in CI. Prompt changes get reviewed like code changes.
> Building this shows you get that workflow.

**Milestone:** A GitHub PR that changes a system prompt triggers an automatic eval run, and results appear as a PR comment.

---

### WEEK 8: Polish, Deploy & Portfolio Packaging

**Goal:** Make everything production-ready and prepare your portfolio presentation.

**Tasks:**
- [ ] **Error handling audit:** go through every endpoint and ensure proper error responses
- [ ] **Add request validation:** rate limiting on API, input sanitization
- [ ] **Write comprehensive tests:**
  - Unit tests for all judges (mock HuggingFace models for speed)
  - Integration tests for the eval pipeline (use a cheap model like gpt-4o-mini)
  - API tests for all endpoints
  - Target: 70%+ coverage (realistic, not aspirational)
- [ ] **API documentation:** FastAPI auto-generates OpenAPI/Swagger docs — review and add descriptions
- [ ] **README.md:** Write a stellar README with:
  - Architecture diagram
  - Quick start guide (docker compose up → working in 5 minutes)
  - Screenshots of the dashboard
  - Example eval run walkthrough
  - Design decisions and tradeoffs documented
- [ ] **Deploy to cloud:**
  - Option A: AWS EC2 + RDS (PostgreSQL) + ElastiCache (Redis) — more impressive
  - Option B: Railway or Render — faster, still demonstrates deployment
  - Set up a demo instance with pre-loaded eval data
- [ ] **Record a demo video** (2-3 minutes):
  - Show: upload dataset → configure judges → run eval → view results → compare runs → CI integration
  - This is what hiring managers actually watch
- [ ] **Write the blog post** (publish on your personal site or Medium):
  - "Building EvalForge: What I Learned About LLM Evaluation in Production"
  - Cover: why eval matters, architecture decisions, what surprised you
  - This gets you inbound interest

**Portfolio packaging — what goes on your resume:**

> **EvalForge — LLM Evaluation & Guardrail Platform**
> Python, FastAPI, React, PostgreSQL, Redis, Celery, Docker, HuggingFace, GitHub Actions
>
> • Architected a model-agnostic LLM evaluation platform with 6 judge types
>   (toxicity classification, PII detection via NER, semantic similarity, zero-shot
>   evaluation, format compliance, and LLM-as-judge), supporting automated testing
>   across OpenAI, Anthropic, and open-source model endpoints.
>
> • Built async evaluation infrastructure (Celery + Redis) processing 500+ samples
>   per run with configurable concurrency, statistical regression detection (Welch's
>   t-test, p<0.05), and hard guardrails that block deployments on PII or toxicity failures.
>
> • Integrated CI/CD pipeline via GitHub Actions webhooks, enabling automated eval
>   runs on prompt changes with PR-level reporting — reducing undetected LLM behavior
>   regressions in a simulated production environment.

**What you're learning:**
- Production readiness: the gap between "it works" and "it's deployable"
- Cloud deployment and infrastructure setup
- Technical writing (README, blog post) — this is a skill senior engineers value highly
- Demo presentation — you will use this exact demo in interviews

**Milestone:** Live demo URL, polished GitHub repo, resume bullet points written, blog post published.

---

## 6. Key Technologies & Why They Matter in 2026

| Technology | Why It's Industry Standard | Your Alternative (Avoid) |
|---|---|---|
| **FastAPI** | Async, typed, auto-docs. Used at Netflix, Uber, Microsoft. | Flask (dated), Django (overkill for APIs) |
| **PostgreSQL** | The default relational DB. JSON support, extensions (pgvector). | SQLite (not production), MySQL (less features) |
| **Redis** | Caching, queuing, pub/sub. Used everywhere. | RabbitMQ (heavier than needed here) |
| **Celery** | Standard Python task queue. Battle-tested at scale. | Custom threading (fragile) |
| **SQLAlchemy 2.0** | Typed ORM with async support. Industry default. | Raw SQL (unmaintainable), Django ORM (tied to Django) |
| **Alembic** | Migration management. Used with SQLAlchemy universally. | Manual schema changes (dangerous) |
| **Docker + Compose** | Containerized development and deployment. Non-negotiable in 2026. | Local virtualenvs (not reproducible) |
| **React + TypeScript** | Dominant frontend stack. Required for any full-stack demo. | Streamlit (toy), Gradio (ML demo only) |
| **GitHub Actions** | CI/CD. Free, integrated, widely adopted. | Jenkins (self-hosted overhead) |
| **uv** | Fast Python package manager. Replacing pip/poetry in 2025-26. | pip (slow, no lockfile), poetry (being superseded) |
| **ruff** | Linter + formatter. Replaced black + flake8 + isort. | Multiple separate tools (outdated) |
| **structlog** | Structured logging. Production standard. | print() or basic logging (unparseable) |
| **LiteLLM** | Unified LLM API. Abstracts provider differences. | Direct API calls (vendor lock-in) |
| **Pydantic v2** | Data validation. FastAPI's backbone. Core Python ecosystem tool. | Manual validation (error-prone) |

---

## 7. Skills Gap Map — What You Know vs What You'll Learn

```
YOUR EXISTING STRENGTHS          →  HOW EVALFORGE USES THEM
─────────────────────────────────────────────────────────────
PyTorch, model training          →  Understanding HF model inference
Statistical modeling             →  Regression detection (t-tests, CIs)
MLflow, experiment tracking      →  Eval run versioning & comparison
Docker basics                    →  Multi-service Docker Compose
Python, SQL                      →  FastAPI + SQLAlchemy + async
Research writing                 →  Blog post, README, documentation
Feature engineering              →  Designing judge evaluation criteria

GAPS YOU'LL FILL                 →  INDUSTRY RELEVANCE
─────────────────────────────────────────────────────────────
FastAPI + async Python           →  Required for any ML API role
React + TypeScript               →  Full-stack ML engineer differentiator
Celery + Redis                   →  Async processing (used everywhere)
HuggingFace inference pipelines  →  NLP/LLM model serving
LLM-as-judge evaluation          →  The #1 skill in LLM engineering 2026
CI/CD for ML                     →  Production ML maturity signal
Alembic migrations               →  Database management in production
Structured logging               →  Observability and debugging
Cloud deployment                 →  End-to-end ownership
```

---

## 8. How to Talk About This in Interviews

**When asked "Tell me about a project you built":**

> "I built EvalForge, an LLM evaluation platform that works like CI/CD for language models.
> The problem I was solving is that teams shipping LLM features have no reliable way to
> catch regressions when they change prompts or swap models. My system lets you define
> evaluation datasets, configure judge models — toxicity, PII detection, semantic similarity,
> and LLM-as-judge — and run them automatically whenever a prompt file changes in your repo.
> The results post directly to the pull request with statistical regression detection."

**When asked "What was the hardest technical challenge?":**

> "The judge pipeline orchestration. I have 6 different judges — some are local HuggingFace
> models, some call external LLM APIs, and one is pure rule-based. They have wildly different
> latency profiles. I had to design the pipeline so fast judges run in parallel, expensive
> judges run selectively, and the system gracefully handles partial failures — if one judge
> times out, the rest of the scores still save."

**When asked "How would you scale this?":**

> "Right now it runs on a single machine with Celery workers. To scale, I'd split the judge
> models onto GPU workers using something like Ray Serve, add a proper job scheduler like
> Temporal for complex eval workflows, and move to a managed Postgres like RDS. The
> architecture already supports horizontal scaling — you just add more Celery workers.
> For the LLM calls, I'd implement adaptive rate limiting per provider."

**When asked "What would you do differently?":**

> "I'd add streaming results earlier. Right now the dashboard polls for status, but in
> production you'd want WebSocket updates so the UI shows results as they come in.
> I'd also add dataset versioning with DVC — right now datasets are versioned by a simple
> integer, but real teams need to track exactly which eval data was used for which run,
> tied to a git commit."

---

## 9. Stretch Goals (If You Finish Early or Want to Keep Going)

- [ ] **Prompt versioning:** Track prompt templates in git, auto-trigger evals on changes
- [ ] **A/B comparison mode:** Run the same dataset against 2 models simultaneously, see which wins
- [ ] **Custom judge builder:** UI where users define new judges without writing code (using zero-shot classification)
- [ ] **Slack integration:** Post eval results to a Slack channel
- [ ] **Cost tracking:** Track API spend per eval run (LiteLLM provides token counts, you calculate cost)
- [ ] **Dataset augmentation:** Use an LLM to generate adversarial test cases automatically
- [ ] **RAG evaluation:** Specialized judges for retrieval-augmented generation (context relevance, faithfulness)
