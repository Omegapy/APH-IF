# APH-IF Search Module

## Purpose

The `search/` module implements APH‑IF’s retrieval layer:

- Semantic VectorRAG over a Neo4j vector index (OpenAI embeddings).
- Graph traversal via LLM‑generated Cypher with structural‑schema prompts and validation.
- Parallel Hybrid execution that runs both searches concurrently and fuses results.

It provides production‑ready building blocks with timing, circuit breakers, and consistent
response normalization used by the FastAPI service.

## Directory Map

```
backend/app/search/
├── __init__.py                          # Package exports
├── README.md                            # This documentation
├── parallel_hybrid.py                   # Parallel engine: Vector + Graph in asyncio parallel
├── context_fusion.py                    # LLM‑powered context fusion with citation preservation
├── tools/                               # Search tools
│   ├── __init__.py
│   ├── vector.py                        # VectorRAG semantic search (OpenAI + Neo4j vector)
│   ├── cypher.py                        # Public wrapper for LLM Structural traversal
│   ├── llm_structural_cypher.py         # LLM Structural Cypher engine (gen/validate/execute)
│   ├── cypher_validator.py              # Safety/structure/schema validation for Cypher
│   └── prompts/
│       ├── __init__.py
│       └── structural_cypher.py         # Token‑aware prompt builders
└── utils/
    ├── __init__.py
    └── normalization.py                 # Normalizers for semantic/traversal/fusion payloads

backend/app/processing/process_parallel.py  # Optional: process‑isolation parallel execution
```

## Components

- Parallel Hybrid Engine (`parallel_hybrid.py`)
  - Executes semantic and traversal searches concurrently with `asyncio.gather(...)`.
  - Computes fusion readiness, primary method, and complementarity metrics.
  - Circuit breakers around semantic and traversal legs; comprehensive timing.

- Context Fusion Engine (`context_fusion.py`)
  - Intelligent LLM fusion to synthesize a single answer from both results.
  - Strict citation preservation, renumbering, and unified References section.
  - Async LLM client, optional parallel preprocessing, and health metrics.

- Semantic VectorRAG (`tools/vector.py`)
  - OpenAI embeddings + Neo4j vector retrieval via the schema gateway.
  - Schema‑aware prompt augmentation (optional) and robust citation handling.
  - Public APIs: `search_semantic`, `search_semantic_detailed`, `get_vector_engine`.

- LLM Structural Cypher (`tools/llm_structural_cypher.py` + `tools/cypher.py`)
  - NL→Cypher generation using structural schema summaries.
  - Validation (safety, structure, schema) and guarded read‑only execution.
  - Narrative summarization with citation validation and domain markers.
  - Public API: `query_knowledge_graph_llm_structural_detailed`.

- Normalization (`utils/normalization.py`)
  - Consistent models across engines: confidence capping, engine metadata, and payload shapes
    consumed by API responses and clients.

## Data Flow

1) Semantic search (VectorRAG)
   - Embed query → Neo4j vector retriever → context → async LLM answer → validate/renumber citations →
     build structured result.

2) Traversal search (LLM Structural)
   - Build schema‑aware prompt → generate Cypher → validate/fix → execute read‑only via schema manager →
     narrative with validated citations → structured result.

3) Hybrid
   - Run semantic + traversal concurrently → analyze results → if fusion‑ready, call fusion engine →
     produce fused answer with preserved citations and unified references.

## Public APIs (Python)

Semantic (VectorRAG):
```python
from app.search.tools.vector import search_semantic_detailed
result = await search_semantic_detailed("What safety measures are required?", k=10)
print(result["answer"])  # includes inline citations and References section
```

Traversal (LLM Structural):
```python
from app.search.tools.cypher import query_knowledge_graph_llm_structural_detailed
result = await query_knowledge_graph_llm_structural_detailed(
    "Show ventilation requirements", max_results=50
)
print(result["answer"])  # includes generated Cypher and validation metadata
```

Parallel Hybrid + Fusion:
```python
from app.search.parallel_hybrid import get_parallel_engine
from app.search.context_fusion import get_fusion_engine

parallel = await get_parallel_engine().retrieve_parallel(
    "What are equipment and safety requirements?", semantic_k=10, traversal_max_results=50
)

if parallel.fusion_ready:
    fusion = await get_fusion_engine().fuse_contexts(parallel)
    print(fusion.fused_content)
else:
    # fallback to the stronger of the two legs
    print((parallel.semantic_result if parallel.semantic_result.confidence >= parallel.traversal_result.confidence
           else parallel.traversal_result).content)
```

## FastAPI Integration

The top‑level API wires these tools in `app/main.py`:

- `POST /query` (free‑form text response)
  - `search_type`: `vector` | `graph_llm_structural` | `hybrid` (default)
  - Hybrid: runs both legs in parallel and fuses when ready.

- `POST /query/structured` (normalized JSON response)
  - Returns `semantic_result`, `traversal_result`, and optionally `fusion_result` using
    `utils/normalization.py` helpers to standardize payloads.

- Health:
  - `GET /healthz` (overall) and `GET /health/hybrid` (hybrid engines detail)

## Configuration (from `backend/.env` via `app/core/config.py`)

Key flags relevant to `search/`:

- Hybrid and traversal
  - `USE_LLM_STRUCTURAL_CYPHER=true` — enable LLM Structural traversal.
  - `LLM_CYPHER_MAX_HOPS=3` — cap traversal hops.
  - `LLM_CYPHER_FORCE_LIMIT=50` — inject `LIMIT` when missing.
  - `LLM_CYPHER_ALLOW_CALL=false` — forbid CALL procedures (recommended).

- Semantic
  - `SEMANTIC_USE_STRUCTURAL_SCHEMA=true` — include structural schema context in prompts.
  - `SEMANTIC_SCHEMA_TOKEN_BUDGET=2500`, `SEMANTIC_SCHEMA_MAX_ITEMS=15` — prompt shaping.
  - `SEMANTIC_CITATION_VALIDATION=true` — validate inline citations.
  - `SEMANTIC_DOMAIN_ENFORCEMENT=true` — inject domain markers (e.g., §, Part, ISO) near [n].

- Confidence caps
  - `SEMANTIC_CONFIDENCE_CAP=1.0`, `TRAVERSAL_CONFIDENCE_CAP=1.0`, `FUSION_CONFIDENCE_CAP=1.0`

- Fusion
  - `FUSION_PARALLEL_PREPROCESSING=true` — parallelize citation/reference extraction before LLM fusion.

All secrets (Neo4j/OpenAI) must reside only in `backend/.env`. See `backend/env.example`.

## Timing, Health, and Resilience

- Timing: the engines use `monitoring/timing_collector` to measure end‑to‑end and nested phases
  (e.g., task creation, parallel execution, citation validation, LLM calls).
- Circuit breakers: semantic, traversal, and fusion paths are wrapped by circuit breakers where
  available to handle transient failures gracefully.
- Health: `get_parallel_engine().health_check()` and `get_fusion_engine().health_check()`
  expose component statuses; the API aggregates them under `GET /health/hybrid`.

## Citations & References

- Semantic and traversal legs extract `[n]` and domain markers (e.g., `§57.4361(a)`, `Part 75.1714`).
- Fusion preserves existing inline citations, renumbers chronologically across both sources, and
  generates a unified References section.
- Validation filters unmatched/invented citations (configurable).

## Demos & Development

- Console demos:
  - `backend/console_semantic_demo.py` — VectorRAG semantics.
  - `backend/console_fusion_demo.py` — Parallel + fusion walkthrough.

- Process‑isolation parallel (optional):
  - `app/processing/process_parallel.py` provides an execution mode with separate processes to
    avoid shared interpreter state. Use via `get_parallel_engine(use_process_isolation=True)`.

## Testing

- Run tests from `backend/`:
  - `uv run pytest -q` (filter with `-k`)
  - Coverage: `uv run pytest --cov=app --cov-report=term-missing`

- Typical checks:
  - Vector tool smoke: `python backend/console_semantic_demo.py`
  - Hybrid health: call `GET /health/hybrid` while API is running.

## Troubleshooting

- Import‑time configuration errors:
  - Ensure `backend/.env` is present and contains non‑placeholder values for `NEO4J_*` and `OPENAI_API_KEY`.

- Traversal disabled:
  - Set `USE_LLM_STRUCTURAL_CYPHER=true` to enable LLM Structural traversal path.

- Citation issues:
  - Keep `SEMANTIC_CITATION_VALIDATION` and `SEMANTIC_DOMAIN_ENFORCEMENT` enabled for best results.

- Parallel execution:
  - Use default asyncio parallel mode; enable process isolation only if you need hard isolation across
    libraries/clients and accept higher overhead.

## Notes

- Follow repository guidelines: do not add backward‑compatibility, deprecated, or legacy paths.
- Keep imports sorted; use type hints; prefer small, focused changes.

(Readme generated by AI)

