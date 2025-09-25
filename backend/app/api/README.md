# APH-IF API Layer

## Overview

The `app/api` package wires the outward-facing FastAPI surface of the APH-IF backend.
It provides lifecycle hooks, shared runtime state, request/response models, router
implementations, and utility helpers that standardise parameter handling and text
processing across the service.

The module is organised to keep high-level API concerns separate from core search,
schema, and monitoring subsystems. Each router focuses on a single responsibility
area (queries, schema, health, performance, timing, etc.) while common behaviour is
centralised in lifecycle utilities and shared state.

## Directory Structure

```
app/api/
├── README.md                  # This document
├── __init__.py                # Package marker
├── lifecycle.py               # Startup/shutdown registration
├── state.py                   # Shared mutable state for routers
├── models/
│   └── api.py                 # Pydantic request/response contracts
├── routers/
│   ├── graph_cypher.py        # LLM structural Cypher endpoints
│   ├── health.py              # Health and hybrid diagnostics
│   ├── performance.py         # Performance dashboards and metrics
│   ├── query.py               # Primary RAG query interfaces
│   ├── root.py                # Root metadata endpoint
│   ├── schema.py              # Schema inspection and refresh APIs
│   ├── sessions.py            # Session diagnostics and clearing
│   └── timing.py              # Timing analytics and recommendations
└── utils/
    ├── parameters.py          # Result-limit validation helpers
    └── text.py                # Text heuristics and timing recommendations
```

## Key Components

- **Lifecycle (`lifecycle.py`)**
  - `register_startup(app)` attaches asynchronous startup handling that warms caches,
    prepares infrastructure clients, and records environment safety checks.
  - `register_shutdown(app)` cleans up shared clients, CPU pools, and database
    connections during service termination.

- **Shared State (`state.py`)**
  - Exposes `HYBRID_AVAILABLE`, `active_sessions`, and `startup_time` so routers can
    coordinate hybrid feature availability and session tracking without circular
    imports.

- **Models (`models/api.py`)**
  - Defines the Pydantic request/response contracts (e.g., `QueryRequest`,
    `QueryResponse`, `HealthResponse`) shared by routers and other services.

- **Routers (`routers/*`)**
  - `query.py`: Runs vector, traversal, and hybrid retrieval pipelines; exposes both
    free-form and structured responses.
  - `schema.py`: Provides endpoints to inspect cached graph metadata, trigger refreshes,
    and retrieve structural summaries.
  - `health.py`: Supplies `/healthz` and hybrid health diagnostics for observability.
  - `performance.py`: Surfaces dashboard metrics, circuit breaker status, and health
    summaries.
  - `timing.py`: Delivers detailed timing breakdowns, bottleneck analysis, and reset
    operations for the timing collector.
  - `sessions.py`: Offers session inventory and clearing utilities.
  - `graph_cypher.py`: Generates and executes LLM-powered structural Cypher queries and
    exposes engine metrics.
  - `root.py`: Presents the root metadata endpoint used by tooling and documentation.

- **Utilities (`utils/*`)**
  - `parameters.py`: Validates semantic/traversal result limits and applies safe defaults.
  - `text.py`: Detects unknown responses and generates human-readable timing
    recommendations from monitoring data.

## Extension Guidelines

- **Adding Routers**: Place new routers under `routers/`, follow the comment template,
  and register them in `backend/app/main.py`. Reuse shared state and utilities rather
  than duplicating logic.

- **Lifecycle Hooks**: Extend startup/shutdown handlers via helper functions inside
  `lifecycle.py`. Keep heavy work asynchronous and guard optional components with
  feature flags.

- **Shared State**: Prefer encapsulating new runtime flags or caches in purpose-built
  modules. Only extend `state.py` for cross-router data that must remain globally
  accessible.

- **Models and Utilities**: Centralise request/response schemas in `models/api.py` and
  isolate parameter or text helpers within `utils/`. This keeps routers light and easy
  to test.

## Operational Notes

- The API layer assumes environment configuration is loaded through
  `app.core.config.settings` before lifecycle hooks execute.
- Routers rely on underlying search, schema, and monitoring subsystems; ensure those
  services are available when running integration tests.
- Most routers honour the `HYBRID_AVAILABLE` flag to degrade gracefully when hybrid
  modules are not initialised (e.g., in limited environments).

---

For broader system architecture, refer to the backend README and service-specific
documentation within `backend/app/search`, `backend/app/schema`, and `docs/`.

