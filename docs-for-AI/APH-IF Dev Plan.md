# APH-IF (Advanced Parallel HybridRAG – Intelligent Fusion)

### End-to-End Development Plan (Microservice BFF Architecture • Python 3.12 • Docker • FastAPI • Streamlit • Neo4j AuraDB)

Below is a concrete, implementation-ready plan organized by phases, with deliverables, acceptance criteria, and the technical scaffolding you’ll need as a solo developer. It assumes a **mono-repo** with three Python microservices—`frontend`, `backend`, `data_processing`—plus optional local Neo4j for dev. AuraDB is your primary graph/vector store in cloud.

---

## 0) Guiding Principles

* **Separation of concerns:** UI (Streamlit), orchestration (FastAPI), and ingestion/indexing (data\_processing) are cleanly isolated and independently testable.
* **Parallelism by default:** Graph traversal (Cypher) and semantic vector search (cosine) run concurrently; **fusion** then reconciles results (LLM/LRM).
* **Deterministic rails around LLMs:** Use strict I/O schemas, score thresholds, provenance, and templates to keep generations grounded.
* **Observability + testability:** Every service ships with health checks, metrics hooks, and a test harness (unit + integration + E2E).

---

## 1) High-Level Architecture

```mermaid
flowchart LR
    subgraph Frontend (Streamlit)
      UI[bot.py Chat UI]
    end

    subgraph Backend (FastAPI)
      QAPI[/POST /query/]
      Orchestrator[[Parallel Orchestrator]]
      Vector[Vector Retriever<br/>Neo4jVector (k=5, score≥0.7)]
      Graph[Traversal Retriever<br/>LLM→Cypher→Neo4j]
      Fusion[Intelligent Fusion<br/>(GPT-5 or LRM)]
    end

    subgraph Data_Processing (FastAPI/Worker)
      Ingest[/POST /ingest/ PDFs/]
      Chunk[Chunker+Entities]
      Embed[OpenAI Embeddings]
      Upsert[Upsert Nodes+Rels+Vectors→Neo4j AuraDB]
    end

    subgraph Storage (Neo4j AuraDB)
      KG[(Nodes, Rels, Chunk Nodes with Embeddings)]
    end

    UI -->|prompts| QAPI
    QAPI --> Orchestrator
    Orchestrator -->|async| Vector
    Orchestrator -->|async| Graph
    Vector --> KG
    Graph --> KG
    Orchestrator --> Fusion
    Fusion --> QAPI --> UI

    Ingest --> Chunk --> Embed --> Upsert --> KG
```

---

## 2) Repository & Scaffolding

### 2.1 Mono-repo structure

```
aph-if/
├─ README.md
├─ .gitignore
├─ docker-compose.yml
├─ render.yaml                 # Render Blueprint (Phase 7)
├─ documents/                  # project docs
├─ doc_for_ai/                 # prompt rules, system prompts, schemas
├─ pdf_data/                   # PDFs to ingest (dev/local)
├─ test/                       # E2E + integration tests
│  ├─ e2e_test_query.py
│  └─ conftest.py
├─ common/                     # shared utils (pydantic models, logging)
│  ├─ __init__.py
│  ├─ config.py                # reads env vars
│  ├─ models.py                # Pydantic schemas
│  ├─ prompts/                 # fusion, traversal, etc.
│  └─ logging.py
├─ frontend/
│  ├─ __init__.py
│  ├─ app/                     # streamlit app
│  │  └─ bot.py
│  ├─ tests/
│  ├─ requirements.txt
│  ├─ .env.example
│  └─ Dockerfile
├─ backend/
│  ├─ __init__.py
│  ├─ app/
│  │  ├─ main.py               # FastAPI app
│  │  ├─ routers/
│  │  │  └─ query.py           # /query
│  │  ├─ services/
│  │  │  ├─ orchestrator.py    # asyncio.gather, timeouts
│  │  │  ├─ vector_retriever.py
│  │  │  ├─ graph_retriever.py
│  │  │  └─ fusion.py          # IF templates + scoring
│  │  └─ deps.py               # clients (neo4j, openai)
│  ├─ tests/
│  ├─ requirements.txt
│  ├─ .env.example
│  └─ Dockerfile
├─ data_processing/
│  ├─ __init__.py
│  ├─ app/
│  │  ├─ main.py               # FastAPI control plane
│  │  ├─ pipelines/
│  │  │  ├─ pdf_ingest.py
│  │  │  ├─ chunker.py
│  │  │  ├─ entity_extract.py
│  │  │  ├─ embedder.py
│  │  │  └─ upsert_neo4j.py
│  │  └─ routers/
│  │     └─ ingest.py          # /ingest
│  ├─ tests/
│  ├─ requirements.txt
│  ├─ .env.example
│  └─ Dockerfile
└─ ops/
   ├─ Makefile
   └─ ci/                      # GitHub Actions workflows
```

### 2.2 Docker Compose (dev)

```yaml
version: "3.9"
services:
  aph_if_backend:
    build: ./backend
    env_file: ./backend/.env
    ports: ["8000:8000"]
    depends_on: ["aph_if_neo4j"]
  aph_if_frontend:
    build: ./frontend
    env_file: ./frontend/.env
    ports: ["8501:8501"]
    depends_on: ["aph_if_backend"]
  aph_if_data_processing:
    build: ./data_processing
    env_file: ./data_processing/.env
    ports: ["8010:8010"]
    depends_on: ["aph_if_neo4j"]
  # Optional local dev Neo4j (AuraDB for prod)
  aph_if_neo4j:
    image: neo4j:5.20
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_dbms_security_procedures_unrestricted: "apoc.*"
    ports: ["7474:7474", "7687:7687"]
```

> **Note:** For production, point services to **AuraDB** via env vars; don’t run the local container.

### 2.3 Environment variables (each service has its own `.env`)

```
# Common patterns
OPENAI_API_KEY=...
OPENAI_MODEL_GPT5=gpt-5 # or the specific model name you use
OPENAI_EMBED_MODEL=text-embedding-3-large

# Neo4j (AuraDB)
NEO4J_URI=neo4j+s://<your-aura-id>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=...

# Retrieval config
VECTOR_TOP_K=5
VECTOR_SCORE_THRESHOLD=0.7
GRAPH_TIMEOUT_SECS=6
VECTOR_TIMEOUT_SECS=6
FUSION_TIMEOUT_SECS=10
```

---

## PHASE 1 — Environment & Baseline

**Goals**

* Stand up Dockerized microservices with health checks.
* Establish testing harness (pytest), linting (ruff), typing (mypy), and pre-commit.

**Steps**

1. **Bootstrap each service**

   * `__init__.py`, `requirements.txt`, minimal `main.py`.
   * Add `/healthz` route in backend & data\_processing; Streamlit “status pill”.
2. **Common package** for Pydantic schemas:

   * `QueryRequest`, `RetrievedChunk`, `GraphHit`, `FusionInput`, `FusionOutput`, `Citation`.
3. **Logging & tracing**

   * `structlog` JSON logs; add request IDs; plumb simple timing metrics.
4. **Docker compose up** and verify:

   * `curl :8000/healthz`, `curl :8010/healthz`, open `:8501` UI.

**Deliverables**

* Running containers; `make dev-up` & `make dev-down`.
* Basic E2E smoke test (backend returns stub response).

**Definition of Done**

* All services build & pass `pytest -q`.
* `docker-compose up` → health checks green.

---

## PHASE 2 — AI Rules (Guardrails & Templates)

**Goals**

* Author precise prompts & JSON schemas to constrain outputs.
* Define **three agents**: Traversal, Vector, Fusion (IF).

**Artifacts (in `doc_for_ai/`)**

* `traversal.md`

  * Role: convert user prompt → **Cypher** query for KG traversal.
  * Rules: only produce JSON: `{ "cypher": "...", "entities": [...] }`.
  * Few-shot examples matching your schema.
* `vector.md`

  * Role: consume retrieved chunk metadata; **no hallucination**; do not fabricate citations.
* `fusion.md`

  * Role: merge graph + vector hits; produce **grounded** final answer with **ranked citations**.
  * Requires: use per-passage evidence with IDs; abstain if evidence insufficient.

**Technical Settings**

* `temperature=0.2` (fusion 0.1–0.3), `top_p=0.9`, max tokens sized to response + citations.
* Enforce JSON via Pydantic validation + retry if malformed.

**Definition of Done**

* Round-trip tests that validate the agent outputs against Pydantic models.

---

## PHASE 3 — `data_processing` Service (Ingestion → KG)

**Goals**

* Ingest PDFs, chunk, extract entities, embed, and upsert **chunk nodes** + rels into AuraDB.

**Data model (Neo4j)**

* `(:Document {doc_id, title, source, created_at})`
* `(:Chunk {chunk_id, text, tokens, embedding, page, score_meta})`
* `(:Entity {name, type})`
* Relationships: `(:Document)-[:HAS_CHUNK]->(:Chunk)`, `(:Chunk)-[:MENTIONS]->(:Entity)`, domain rels if extracted.

**Pipeline**

1. `POST /ingest` with file or S3/GCS URL.
2. **Chunking** (semantic + page-aware), e.g., \~800–1200 tokens with overlap.
3. **Entity extraction** (LLM or heuristics) to seed KG semantics.
4. **Embeddings** with OpenAI; attach to chunk node (`embedding` array).
5. **Upsert** nodes/rels; build **Neo4jVector** index for cosine search.

**Example endpoints**

* `POST /ingest` → `{job_id}`
* `GET /ingest/{job_id}/status`
* `POST /reindex` (rebuild vectors on demand)

**Tests**

* Unit: chunk sizes, entity extraction contracts.
* Integration: upsert idempotency (re-ingest doesn’t duplicate).
* Performance: embedding concurrency controls (rate-limit friendly).

**Definition of Done**

* A sample PDF ingested; chunks reachable via Cypher & vector retriever.

---

## PHASE 4 — `backend` Service (Orchestration & Parallel Retrieval)

**Goals**

* Implement **parallel** vector + traversal retrieval, then IF fusion.

**API**

* `POST /query`
  **Request**

  ```json
  {
    "query": "string",
    "conversation_id": "optional",
    "top_k": 5,
    "min_score": 0.7
  }
  ```

  **Response**

  ```json
  {
    "answer": "string",
    "citations": [
      {"doc_id":"...", "chunk_id":"...", "page":3, "score":0.83, "source":"..."}
    ],
    "retrieval": {
      "vector": {"hits": 5, "latency_ms": 120},
      "graph": {"hits": 7, "latency_ms": 160}
    },
    "meta": {"orchestrator_latency_ms": 420}
  }
  ```

**Orchestrator (pseudo)**

```python
async def handle_query(q: QueryRequest):
    tasks = [
        asyncio.wait_for(vector_retriever(q), timeout=VECTOR_TIMEOUT),
        asyncio.wait_for(graph_retriever(q),  timeout=GRAPH_TIMEOUT),
    ]
    vector, graph = await asyncio.gather(*tasks, return_exceptions=True)

    # Normalize hits into a common schema with scores & provenance
    fusion_in = normalize(vector, graph)

    # Pre-fusion scoring (optional):
    # - de-dup by doc+span
    # - hybrid score = w1*vec_score + w2*graph_rank_norm (+ coverage bonus)
    fused = await fusion_llm(fusion_in, timeout=FUSION_TIMEOUT)

    return fused
```

**Retrievers**

* **Vector**: `langchain_neo4j.Neo4jVector.as_retriever(k=5, score_threshold=0.7)`
* **Graph**: LLM-to-Cypher (from Phase 2) → run query → map results to chunks/entities.

**Fusion (IF)**

* Inputs: top-N passages (merged), evidence table, entity highlights, and **explicit citation map**.
* Output: final answer with inline citation markers (e.g., `[^1]`) mapped in response JSON.

**Controls**

* Timeouts + circuit breaker (open on persistent upstream failures).
* Cache `query → fused_answer` for short TTL (optional).
* Guardrails: refuse answers if **evidence < threshold**; return clarification ask.

**Tests**

* Integration: identical answer on repeated runs (given same inputs).
* Contract: Fusion always returns citations present in retrieval sets.
* Load: p95 latency budget (≤ 700 ms retrieval, ≤ 2.5 s end-to-end initial target).

**Definition of Done**

* `/query` returns grounded answers against ingested sample PDFs with citations.

---

## PHASE 5 — `frontend` (Streamlit UI)

**Goals**

* Usable chat UI with session state, source viewer, and latency telemetry.

**Features**

* Prompt box, streaming answer, collapsible **“Evidence”** panel with sources (doc title, page, confidence).
* Filters: top-k slider, min score slider (power user mode).
* Session persistence (in memory / simple file for dev).
* Health indicators for backend & AuraDB.

**Tests**

* UI smoke (Streamlit script runner).
* Contract tests against backend (mock API).

**Definition of Done**

* Interactive chat that surfaces citations per answer and basic metrics.

---

## PHASE 6 — Testing Strategy (Unit → Integration → E2E)

**Test Layers & Targets**

* **Unit**

  * Chunker splits within size bounds; entity extractor returns typed entities.
  * Fusion template obeys schema (Pydantic) and includes only known citations.
* **Integration**

  * Ingest → Neo4j upsert idempotency; vector retriever returns ≥`k` when available.
  * Traversal agent produces valid, executable Cypher (no destructive ops).
* **E2E**

  * “Golden set” of \~25 labeled questions per domain PDF:

    * **Grounding**: >92% answer grounding precision (citations actually contain answer span).
    * **Coverage**: at least one gold chunk retrieved in top-5 for ≥85% of cases.
  * **Latency budgets**: p95 ≤ 2.5s; p99 ≤ 4.5s (first answer).

**Tooling**

* `pytest`, `pytest-asyncio`, `httpx` for API tests, `tox` or `uv` for multi-env.
* Quality gates: ruff, mypy, pre-commit.
* Optional: small **labeling harness** to mark correct spans in chunks for evaluation.

**Definition of Done**

* CI green; E2E metrics meet thresholds on the golden set.

---

## PHASE 7 — Deployment (Render)

**Goals**

* One-click deployment of backend + frontend; data\_processing triggered on demand.
* Use **Render Blueprint** (`render.yaml`) with Docker builds.

**Steps**

1. **Secrets**: Store `OPENAI_API_KEY`, `NEO4J_URI/USER/PASSWORD` in Render secrets.
2. **Services**

   * `backend` → Web Service (exposes `:8000`, health `/healthz`)
   * `frontend` → Web Service (exposes `:8501`, env: `BACKEND_URL`)
   * `data_processing` → Background Worker or Web if you want ingestion over HTTP.
3. **Networking**

   * Allow outbound to AuraDB.
   * (Optional) Private service-to-service if available; else use service URLs.
4. **Monitoring**

   * Health checks, CPU/RAM alerts, request logs.
   * Configure autoscaling minimums appropriate for cost.

**Definition of Done**

* Public frontend URL serving chat; answers cite AuraDB-backed content.

---

## Fusion Details (Intelligent Content Fusion, IF)

* **Pre-Fusion Heuristics (deterministic)**

  * Merge & deduplicate by `(doc_id, page, span_hash)`.
  * Hybrid rank: `S = 0.6 * vector_score_norm + 0.4 * graph_rank_norm (+0.05 if cross-supported)`.
  * Keep top-N (e.g., 8–10) to feed the LLM/LRM.
* **Fusion Prompt Contract**

  * Provide **Answer**, **Reasoning summary (brief & non-speculative)**, and **Citations** referencing IDs from inputs.
  * Require “INSUFFICIENT\_EVIDENCE” if sources don’t support a claim.
* **Fallback**

  * If fusion times out, return best-ranked snippet summary with explicit warning & citations.

---

## Example: Backend `/query` Handler Skeleton

```python
# backend/app/services/orchestrator.py
from .vector_retriever import vector_retrieve
from .graph_retriever import graph_retrieve
from .fusion import fuse_answer

async def answer_query(req):
    vec_task = asyncio.create_task(vector_retrieve(req))
    graph_task = asyncio.create_task(graph_retrieve(req))
    vec_res, graph_res = await asyncio.gather(vec_task, graph_task, return_exceptions=True)

    fusion_input = normalize_hits(vec_res, graph_res)  # common schema w/ provenance
    fused = await fuse_answer(fusion_input)
    return fused
```

---

## Agile Solo-Dev Cadence (6–8 Weeks, 5 Workstreams)

| Week | Workstream | Key Deliverables                              | DoD                                        |
| ---- | ---------- | --------------------------------------------- | ------------------------------------------ |
| 1    | Phase 1    | Repo, Docker, health checks, CI, test harness | `docker-compose up` green; smoke test      |
| 2    | Phase 2    | Prompt rules, schemas, validation             | Pydantic validation & retry logic in place |
| 3–4  | Phase 3    | Ingestion pipeline to AuraDB                  | Ingest sample PDFs; vectors searchable     |
| 4–5  | Phase 4    | Orchestrator + retrievers + fusion            | `/query` returns grounded answers          |
| 5    | Phase 5    | Streamlit UI                                  | Evidence panel + latency telemetry         |
| 6    | Phase 6    | Golden set + E2E tests                        | Precision/coverage & latency met           |
| 7–8  | Phase 7    | Render deploy + hardening                     | HTTPS, secrets, alerts, autoscale          |

---

## Risks & Mitigations

* **LLM drift / cost:** Pin model versions; cache fusion for short TTL; add local LRM option for fusion/rerank if needed.
* **Schema creep:** Centralize Pydantic models in `common/`; contract tests on every PR.
* **Neo4j write hot-spots:** Batch upserts; use idempotent MERGE patterns; backoff on rate limits.
* **Timeouts under load:** Independent timeouts per retriever; degrade gracefully (single-retriever answer + banner).

---

## Acceptance Checklist (Project-Level)

* [ ] Parallel retrieval and deterministic pre-fusion implemented.
* [ ] Fusion returns **only** claims supported by citations; abstains when weak.
* [ ] E2E tests pass on golden set with targets met.
* [ ] Render deployment operational with secrets, alerts, and health checks.
* [ ] Documentation: `README.md` (root + each service), runbooks, and prompt rulebook.

---

If you want, I can generate **starter files** (Dockerfiles, minimal FastAPI apps, Streamlit `bot.py`, and a `docker-compose.yml`) tailored to your exact env vars and preferred tooling (pip/uv/poetry).
