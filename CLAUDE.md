# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EvalForge is an LLM evaluation and guardrail platform — essentially CI/CD for language models. It lets teams define evaluation datasets, run multi-dimensional judge pipelines against any LLM, detect regressions statistically, and block deployments on guardrail failures. The full design is in `evalforge-plan.md`.

## Package Manager & Tooling

This project uses `uv` (not pip or poetry):

```bash
uv sync                  # install dependencies
uv add <package>         # add a dependency
uv run <command>         # run a command in the venv
```

Linting and formatting uses `ruff` (replaces black + flake8):

```bash
uv run ruff check .          # lint
uv run ruff check . --fix    # lint and auto-fix
uv run ruff format .         # format
```

## Running the Application

```bash
docker compose up            # start FastAPI + PostgreSQL + Redis
uv run python main.py        # run standalone (dev/testing only)
```

## Testing

```bash
uv run pytest                         # run all tests
uv run pytest tests/unit/             # unit tests only
uv run pytest tests/integration/      # integration tests only
uv run pytest tests/path/test_file.py::test_name   # single test
uv run pytest --cov                   # with coverage
```

## Architecture

The system follows a layered architecture: `routes → services → database`, never routes directly to database.

```
app/
├── api/routes/       # FastAPI route handlers (thin — delegate to services)
├── services/         # Business logic layer
├── models/           # SQLAlchemy ORM models (PostgreSQL)
├── schemas/          # Pydantic v2 request/response schemas
├── judges/           # Pluggable judge implementations (BaseJudge interface)
├── core/             # LLMRunner (LiteLLM), eval engine, regression detection
└── db/               # SQLAlchemy async engine, Alembic migrations
```

**Async processing:** Long-running eval runs are dispatched as Celery tasks (Redis broker). The API returns a `run_id` immediately; clients poll `GET /api/eval/runs/{id}/status`.

**Judge pipeline:** Each judge implements `BaseJudge` with a standard `evaluate(llm_output, reference, input_text) -> {score, passed, details}` interface. Judges are pluggable and run after each LLM response is captured. Judge types: `ToxicityJudge` (DeBERTa), `PIIJudge` (BERT NER + regex), `SemanticSimilarityJudge` (MiniLM), `ZeroShotJudge` (DeBERTa-NLI), `LLMAsJudge` (via LiteLLM), `FormatComplianceJudge` (rule-based).

**LLM abstraction:** All model calls go through LiteLLM (`app/core/`), providing a unified interface across OpenAI, Anthropic, HuggingFace, and custom endpoints.

**Regression detection:** Uses Welch's t-test (`scipy.stats.ttest_ind`) to compare judge score distributions between eval runs. A regression is flagged when `p < 0.05` and the current mean is lower than baseline.

## Database

PostgreSQL with SQLAlchemy 2.0 async ORM and Alembic migrations. Never run raw DDL — always use migrations:

```bash
uv run alembic revision --autogenerate -m "description"
uv run alembic upgrade head
uv run alembic downgrade -1
```

Core tables: `projects`, `llm_endpoints`, `eval_datasets`, `eval_samples`, `eval_runs`, `eval_results`, `judge_scores`, `guardrail_configs`. See `evalforge-plan.md` §2 for full schema.

## Key Design Decisions

- **PII detection is a hard guardrail** — `PIIJudge` failures always block; `is_blocking=true` in `guardrail_configs`.
- **Celery tasks use `acks_late=True`** — ensures unfinished tasks requeue if a worker dies mid-evaluation.
- **Idempotent eval steps** — re-running a sample should not create duplicate `eval_results` rows.
- **Structured logging** with `structlog` throughout — never use `print()` or basic `logging`.
