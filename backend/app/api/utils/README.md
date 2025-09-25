# APH-IF API Utilities

## Overview

The `app/api/utils` package contains helper functions that keep API routers concise. These modules standardise common logic such as validating request parameters and analysing response text, enabling routers to focus on orchestration rather than repeated boilerplate.

## Directory Structure

```
app/api/utils/
├── README.md             # This document
├── __init__.py           # Convenience exports for routers
├── parameters.py         # Result-limit validation helpers
└── text.py               # Text heuristics and timing recommendations
```

## Modules

### `parameters.py`
- Validates caller-provided result limits for semantic and traversal searches.
- Applies safe defaults (`SEMANTIC_DEFAULT_K`, `TRAVERSAL_DEFAULT_K`) and bounds values using `MAX_K`.
- Exposes `get_semantic_k()` and `get_traversal_k()` for reuse across routers.

### `text.py`
- Provides heuristics (`is_unknown_text`) to recognise fallback responses returned by retrieval engines.
- Generates human-readable timing recommendations via `generate_timing_recommendations()` based on monitoring data.

## Usage Guidelines
- Import helpers through `app.api.utils` or the specific module to keep router implementations declarative.
- Maintain stateless utilities here; mutable shared state belongs in `app/api/state.py` or higher-level services.
- Follow the project’s comment template and add Google-style docstrings when extending utilities.

## Related Modules
- `app/api/routers/*` – routers that call these utilities.
- `app/api/state.py` – shared mutable state referenced by timing/text helpers.
- `app/api/lifecycle.py` – orchestrates startup/shutdown of resources that may feed into recommendations.

Refer to `backend/app/api/README.md` for a wider overview of the API layer and its supporting components.
