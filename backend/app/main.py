"""
APH-IF Backend Service (Phase 1)

Minimal FastAPI app exposing a health check and a stub /query endpoint
for Phase 1 smoke testing and dockerized bring-up.
"""

from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="APH-IF Backend", version="0.1.0")


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    """Liveness/readiness probe."""
    return {"status": "ok", "service": "backend"}


class QueryRequest(BaseModel):
    """Request model for the stub query endpoint.

    Attributes:
        query: User question string
        conversation_id: Optional conversation/session identifier
        top_k: Requested number of passages
        min_score: Minimum score threshold
    """

    query: str
    conversation_id: Optional[str] = None
    top_k: int = 5
    min_score: float = 0.7


@app.post("/query")
async def query_endpoint(req: QueryRequest) -> Dict[str, Any]:
    """Stubbed query endpoint for Phase 1 E2E smoke test."""
    return {
        "answer": f"Stub answer to: {req.query}",
        "citations": [],
        "retrieval": {
            "vector": {"hits": 0, "latency_ms": 0},
            "graph": {"hits": 0, "latency_ms": 0},
        },
        "meta": {"orchestrator_latency_ms": 1},
    }


