# APH-IF Backend Alpha Version 0.1.0

## Overview

The `backend` package houses the FastAPI service, utility scripts, and supporting modules that power Advanced Parallel HybridRAG – Intelligent Fusion (APH-IF). It orchestrates semantic (vector) and structural (graph) retrieval, intelligent fusion, schema management, monitoring, and interactive console demos.

Key capabilities include:

- `app/main.py`: FastAPI entrypoint that configures the service, registers lifecycle hooks, and wires domain routers.
- `app/api/`: API-facing package grouping routers, request/response models, lifecycle helpers, and shared utilities.
- `console_*.py`: Console demos illustrating semantic search, traversal, and full fusion pipelines.
- `app/*`: Modular subpackages handling configuration, model definitions, monitoring utilities, processing helpers, search engines, and schema management.
- `schema_cli.py`: Command-line tools for schema inspection and refresh operations.

## Directory Structure

```
backend/
├── app/                           # FastAPI application package
│   ├── api/                       # Routers, API-specific models, lifecycle helpers, shared state
│   ├── core/                      # Configuration, async LLM client, CPU pool utilities
│   ├── models/                    # Pydantic response models and structured payloads
│   ├── monitoring/                # Performance/timing collectors, circuit breakers
│   ├── processing/                # Citation processor and fusion post-processing helpers
│   ├── schema/                    # Schema manager, exporters, and cache utilities
│   ├── search/                    # Parallel retrieval engines and fusion logic
│   ├── __init__.py
│   └── main.py                    # FastAPI app entrypoint wiring routers and middleware
├── console_semantic_demo.py       # VectorRAG console demonstration
├── console_traversal_demo.py      # LLM Structural Cypher demo
├── console_fusion_demo.py         # Parallel + fusion workflow demo
├── console_test_neo4j_connection.py # Neo4j connectivity tester
├── schema_cli.py                  # Schema management CLI utilities
└── README.md                     # (this file)
```

## Key Components

### FastAPI Application (`app/main.py` & `app/api/`)

- `main.py` constructs the FastAPI app, configures CORS, registers lifecycle hooks, and wires routers.
- Domain routers under `app/api/routers/` implement query, schema, performance, timing, session, and graph Cypher endpoints.
- `app/api/models/` centralizes API request/response models while `app/api/utils/` hosts shared helpers.
- Lifecycle helpers in `app/api/lifecycle.py` manage startup/shutdown initialization for async clients and processors.

### Search & Fusion (`app/search/`)

- `parallel_hybrid.py`: Implements concurrent semantic + traversal retrieval.
- `context_fusion.py`: Performs intelligent context fusion using LLMs with citation preservation.
- `tools/vector.py`, `tools/cypher.py`, `tools/llm_structural_cypher.py`: Semantic search, Cypher traversal, and LLM-powered Cypher generation helpers.
- `utils/normalization.py`: Normalizes raw retrieval outputs into structured payloads.

### Monitoring (`app/monitoring/`)

- `performance_monitor.py`: Collects hybrid system metrics.
- `timing_collector.py`: Hierarchical timing instrumentation for requests.
- `circuit_breaker.py`: Safeguards for external service failures.

### Schema Management (`app/schema/`)

- `schema_manager.py`: Handles cached schema retrieval and refresh workflows.
- `schema_acquirer.py` / `schema_exporter.py`: Interfaces for schema acquisition and exporting.
- `schema_models.py`: Strongly typed schema representations.

### Configuration & Infrastructure (`app/core/`)

- `config.py`: Central configuration, environment management, and feature flags.
- `async_llm_client.py`: Async LLM client with resource pooling.
- `cpu_pool.py`: CPU-bound task executor utilities.

### Console Demonstrations

- `console_semantic_demo.py`: Interactive and scripted semantic search showcase.
- `console_traversal_demo.py`: Demonstrates LLM Structural Cypher graph traversal.
- `console_fusion_demo.py`: End-to-end parallel retrieval plus intelligent fusion demo.
- `console_test_neo4j_connection.py`: Quick connectivity check for Neo4j.

## Getting Started

1. Navigate to the repository root (`P:\Projects\APH-IF\APH-IF-Dev`).
2. Ensure dependencies are managed with `uv` (per project rules).
3. Switch to `backend/` for API and testing commands.

### Running the API (Development)

```powershell
cd backend
uv run uvicorn app.main:app --reload
```

### Running Console Demos

```powershell
cd backend
uv run python console_semantic_demo.py
uv run python console_traversal_demo.py
uv run python console_fusion_demo.py --demo
```

### Testing & Linting

```powershell
cd backend
uv run pytest -q                     # Run tests
uv run ruff check app                 # Lint the app package
uv run mypy app                       # Optional static type checking
```

## Notes

- Use `schema_cli.py` for schema inspection and refresh operations.
- Environment variables are centrally managed via `app/core/config.py` and project-wide tooling (`set_environment.py`).
- Follow project conventions: Python 3.12+, 100-character lines, double quotes, `uv` dependency management, and docstring/comment templates from `docs-for-AI/`.

For more details, refer to per-submodule READMEs inside `app/*` and the project-wide documentation under `docs-for-AI/` and `documentation/`.
