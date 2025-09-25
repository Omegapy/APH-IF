# APH-IF Core Infrastructure

## Overview

The `app/core` package houses the infrastructure primitives that underpin the APH-IF backend. These modules manage configuration, database connectivity, language-model clients, and CPU-bound execution pools. They are consumed by the API layer, search services, schema management, and processing pipelines to orchestrate retrieval and fusion workflows.

## Directory Structure

```
app/core/
├── README.md                 # This document
├── __init__.py               # Package exports and helper imports
├── async_llm_client.py       # Async LLM client management
├── config.py                 # Centralised settings and environment flags
├── cpu_pool.py               # Thread/process pool for CPU-bound tasks
├── database.py               # Neo4j/AuraDB connectivity helpers
└── llm_client.py             # Base LLM client abstractions
```

## Key Modules

### `config.py`
- Single source of truth for application settings (`settings`).
- Handles environment detection, feature flags, safety guardrails, and startup logging.

### `database.py`
- Provides Neo4j driver acquisition, health checks, and safe shutdown routines.
- Integrates with schema caching helpers and enforces retry/backoff policies.

### `cpu_pool.py`
- Manages a shared process pool for CPU-intensive workloads (e.g., citation processing, embedding enrichment).
- Exposes lifecycle-aware helpers to start and tear down workers.

### `llm_client.py` & `async_llm_client.py`
- Define sync/async client abstractions that wrap provider-specific LLM calls.
- Offer pooling, timeout, and graceful shutdown management to keep request handling responsive.

## Design Considerations
- Modules are dependency-injection friendly: they rely on `settings` but avoid importing FastAPI components.
- Shutdown paths are idempotent so integration tests can safely reinitialise state.
- Logging is structured but defers configuration to the application entry point.

## Usage Guidelines
- Use `from app.core.config import settings` as the single source of configuration.
- Coordinate lifecycle-sensitive resources (LLM clients, CPU pools, database connections) via `app/api/lifecycle.py` to ensure consistent startup/shutdown.
- When introducing new core primitives, follow the project’s comment template and expose them through `app/core/__init__.py` only when stable.

## Related Modules
- `app/api/lifecycle.py` – attaches startup/shutdown handlers that initialise core services.
- `app/api/routers/*` – routers that depend on the core infrastructure for search, metrics, and diagnostics.

For broader architectural context, refer to `backend/app/api/README.md`, `docs/backend-main-refactor-plan.md`, and the project README.
