# APH-IF Schema Module

## Purpose

Centralize knowledge-graph schema acquisition, caching, structural export, and the Neo4j access boundary used by the rest of the backend. Only this module (and `app/core/database.py`) talk to Neo4j directly; other modules must use the gateway methods here.

## Module Structure

```
backend/app/schema/
├── __init__.py                    # Package exports and get_schema_manager()
├── README.md                      # This documentation
├── schema_manager.py              # Main orchestrator with gateway APIs
├── schema_acquirer.py             # Database introspection and schema acquisition
├── schema_exporter.py             # Structural summary export for LLM prompts
└── schema_models.py               # Dataclasses for schema objects

backend/
└── schema_cli.py                  # CLI tool for schema management
```

## Components

- **SchemaManager**: Orchestrates acquisition, caching, disk persistence, structural export, and gateway APIs. Path: `app/schema/schema_manager.py`.
- **SchemaAcquirer**: Runs comprehensive, operator-driven introspection to build a full schema snapshot. Path: `app/schema/schema_acquirer.py`.
- **SchemaExporter**: Writes a lightweight JSON "structural summary" optimized for LLM prompts, using atomic writes and safe wrappers. Path: `app/schema/schema_exporter.py`.
- **Schema Models**: Dataclasses for complete schema, structural summary, and cache wrapper. Path: `app/schema/schema_models.py`.
- **Package Exports**: `get_schema_manager()` and models available from `app.schema`.
- **[CLI Tool](../../schema_cli.py)**: Command-line interface for developers/administrators to manage schema outside request handling.

## Gateway APIs (Use These From Other Modules)

- `get_schema_manager()`: Returns the process-wide singleton.
- `SchemaManager.execute_read(cypher: str, params: dict | None = None) -> list[dict]`
  - Read-only guard: rejects write ops (`CREATE`, `MERGE`, `DELETE`, `SET`, `REMOVE`, `DROP`, `ALTER`, `LOAD CSV`, `IMPORT`).
  - Allows whitelisted `CALL db.*` introspection procedures.
- `SchemaManager.get_neo4j_vector(embeddings) -> Neo4jVector`
  - Creates a vector retriever via `Neo4jVector.from_existing_index(...)` with index/node field conventions used by VectorRAG.
- `SchemaManager.database_health_check() -> dict`
- `SchemaManager.start_db_background_tasks() -> None`
  - Warms the pool and starts periodic health checks.
- `SchemaManager.shutdown_database_connections() -> None`
- High-level getters for schema artifacts:
  - `get_schema() / get_schema_async()` → `CompleteKGSchema`
  - `get_structural_summary()` → `StructuralSummary`
  - `get_structural_summary_for_llm()` → compact text
  - `get_structural_summary_dict()` → dict for prompt builders
  - `get_schema_summary()` / `get_cache_info()` → status and metrics

## Caching & Files

- **Cache dir**: `backend/schema_cache/` (default). Managed by `SchemaManager`.
- **Files**:
  - `kg_schema.json` — Full `CompleteKGSchema` snapshot for operators.
  - `kg_schema_structural_summary.json` — Lightweight structural export for LLM prompts (filename configurable via `SCHEMA_EXPORT_FILENAME`).
- **Modes**:
  - Static mode (default): Cache does not expire automatically; refresh manually after data ingestion.
  - Dynamic mode: Optional TTL if created with `static_mode=False`.

## Environmental Requirements

- Config comes from `backend/.env` via `app/core/config.py` and is strictly validated.
  - **Required**: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` (defaults to `neo4j`).
  - **Note**: `OPENAI_API_KEY` is also validated globally; set it even if only running schema tasks.

## Typical Code Usage

### Get schema and structural summary:
```python
from app.schema import get_schema_manager

sm = get_schema_manager()
schema = sm.get_schema()  # full snapshot for analysis/tools
summary_text = sm.get_structural_summary_for_llm()  # compact prompt context
summary_dict = sm.get_structural_summary_dict()
```

### Execute safe read queries through the boundary:
```python
rows = sm.execute_read("MATCH (n) RETURN n LIMIT 5")
```

### Obtain Neo4j vector retriever (used by VectorRAG):
```python
vector = sm.get_neo4j_vector(embeddings)
```

## CLI (backend/schema_cli.py)

Intended for developers/administrators; runs outside request handling.

From `backend/` or repo root:
- `python backend/schema_cli.py refresh` — Acquire and cache full schema, export structural summary.
- `python backend/schema_cli.py info` — Show schema/cache status and acquisition timings.
- `python backend/schema_cli.py clear` — Clear cached schema.
- `python backend/schema_cli.py export my_schema.json` — Export full schema to JSON/YAML.
- `python backend/schema_cli.py analyze` — Top-level counts, top labels/relationships, common properties.
- `python backend/schema_cli.py data-refresh` — Static-KG workflow after data ingestion.

## How Acquisition Works

`SchemaAcquirer` assembles `CompleteKGSchema` by:
- Counting nodes/relationships and collecting per-label/per-relationship properties.
- Gathering property statistics, sample nodes/edges (configurable), and patterns.
- Reading constraints/indexes (`SHOW CONSTRAINTS`, `SHOW INDEXES`).

`SchemaManager` persists the snapshot, triggers structural export (best-effort), and updates in-memory cache.

## Structural Summary Export

`schema_exporter.export_structural_summary_safe(...)` composes a compact JSON with:
- Node labels, relationship types.
- Node/relationship property type rows (uses `db.schema.*` procedures; falls back when unavailable).
- Metadata with masked Neo4j URI and counts.

Used by LLM Structural Cypher generation and schema-aware prompts for VectorRAG.

## Security & Boundary

- Do not import Neo4j drivers or `Neo4jVector/Neo4jGraph` outside `app/schema/` and `app/core/database.py`.
- Use the gateway methods above; tests in `backend/tests/test_db_boundary.py` enforce this.

## Troubleshooting

- **Config errors at import**: ensure `backend/.env` uses real credentials (placeholders trigger validation errors).
- **Forbidden query errors**: `execute_read` rejects writes and non-whitelisted procedures by design.
- **Structural export issues**: export is best-effort; failures don't block caching. Check `backend/schema_cache/` and logs.

## Notes

- Keep changes minimal and use existing patterns in the codebase.
- Do not add deprecated or legacy compatibility paths per repository guidelines.

(Generate by AI)

