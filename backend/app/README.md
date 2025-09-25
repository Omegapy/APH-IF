# APH-IF Backend Application Package

## Overview

The `app` package houses the FastAPI service for the Advanced Parallel HybridRAG – Intelligent Fusion (APH-IF) platform. It exposes HTTP endpoints for hybrid retrieval, schema exploration, monitoring, and lifecycle orchestration while coordinating shared infrastructure such as the asynchronous LLM client, CPU pools, and schema cache.

Key responsibilities include:

- Serving REST endpoints grouped under `api/`, with `main.py` wiring the routers into the FastAPI app.
- Orchestrating semantic/vector, graph/LLM, and hybrid retrieval workflows.
- Managing application startup/shutdown hooks and shared background services.
- Providing schema-aware utilities for Neo4j AuraDB integration.
- Collecting performance, timing, and circuit-breaker telemetry.

## Directory Structure

```
app/
├── api/                 # Domain routers, API models, lifecycle hooks, and shared state
├── core/                # Configuration, async LLM client, CPU pool, and shared infrastructure
├── models/              # Pydantic response payloads and structured response helpers
├── monitoring/          # Performance monitor, timing collector, circuit breaker utilities
├── processing/          # Citation processing and related data post-processing utilities
├── search/              # Vector, graph, and hybrid retrieval engines plus normalization helpers
├── schema/              # Schema manager, cache adapters, and exporters for Neo4j metadata
├── __init__.py          # Package marker (kept minimal by design)
└── main.py              # FastAPI application entrypoint and router wiring
```

## Primary Entry Points

- `main.py`: Constructs the FastAPI application, registers routes, wires monitoring, and coordinates lifecycle hooks.
- `core/config.py`: Loads environment-aware settings governing Neo4j, LLMs, and feature flags.
- `search/parallel_hybrid.py`: Provides the parallel hybrid engine combining semantic and traversal retrieval.
- `search/context_fusion.py`: Performs intelligent fusion of retrieval legs into a unified response.
- `schema/schema_manager.py`: Manages cached schema metadata and exposes assistants for traversal features.

## Working With the Backend App

1. **Environment & Dependencies**
   - From the repository root (`P:\Projects\APH-IF\APH-IF-Dev`), navigate to `backend/`.
   - Use the `uv` toolchain for all dependency operations: `uv sync`, `uv add <package>`, etc.
   - Configure environment variables via `set_environment.py` or the shared `.env` file.

2. **Running the API (development)**
   ```powershell
   cd backend
   uv run uvicorn app.main:app --reload
   ```

3. **Testing & Tooling**
   ```powershell
   cd backend
   uv run pytest -q          # FastAPI & retrieval tests
   uv run ruff check app     # Linting for the app package
   uv run mypy app           # Optional static type checking
   ```

4. **Schema Utilities**
   - `schema_cli.py` offers command-line helpers for inspecting and refreshing schema caches.
   - API endpoints under `/schema` expose JSON/text summaries suitable for UI or debugging.

5. **Monitoring Endpoints**
   - `/performance/*` routes deliver real-time statistics captured by the monitoring subsystem.
   - `/timing/*` routes surface detailed timing breakdowns collected during query execution.

## Conventions & Standards

- **Python Version**: 3.12+
- **Formatting**: PEP 8, 100-character lines, double-quoted strings, Ruff for linting.
- **Async First**: Network- and I/O-bound tasks should use `async/await` patterns.
- **Documentation**: Follow the APH-IF docstring/comment templates (see `docs-for-AI/`).
- **Safety**: Always confirm environment (`python set_environment.py --status`) before performing graph mutations or running database-affecting tests.

## Related Documentation

- `../README.md`: Backend-level overview and setup instructions.
- `../../docs-for-AI/`: Comment and documentation templates targeted at AI-assisted edits.
- `../../documentation/`: Human-facing project documentation.

For questions about service boundaries or architectural decisions, consult the project documentation and the architecture guides within `documentation/`.

