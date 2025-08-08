"""
APH-IF Data Processing Service (Phase 1)

Minimal FastAPI app exposing a health check for Phase 1 dockerized bring-up.
"""

from typing import Dict

from fastapi import FastAPI


app = FastAPI(title="APH-IF Data Processing", version="0.1.0")


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    """Liveness/readiness probe."""
    return {"status": "ok", "service": "data_processing"}


