# -------------------------------------------------------------------------
# File: graph_cypher.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/app/api/routers/graph_cypher.py
# -------------------------------------------------------------------------
# Project: APH-IF
#
# Project description:
# Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF)
# is a novel Retrieval Augmented Generation (RAG) system that differs from
# traditional RAG approaches by performing semantic and traversal searches
# concurrently, rather than sequentially, and fusing the results using an LLM
# or an LRM to generate the final response.
# -------------------------------------------------------------------------
#
# --- Module Functionality ---
#   Provides endpoints for generating and executing structural Cypher queries
#   using LLM-based tooling, exposing metrics and reset controls for the
#   structural engine.
# -------------------------------------------------------------------------
#
# --- Module Contents Overview ---
# - Router: router (FastAPI APIRouter instance)
# - Endpoint: generate_cypher_query
# - Endpoint: query_with_cypher_generation
# - Endpoint: get_cypher_engine_metrics
# - Endpoint: reset_cypher_engine_metrics
# - Constant: __all__ (module exports)
# -------------------------------------------------------------------------
#
# --- Dependencies / Imports ---
# - Standard Library: logging, datetime (datetime), typing (Any)
# - Third-Party: fastapi (APIRouter), fastapi.responses (JSONResponse)
# - Local Project Modules: app.api.models, app.api.utils,
#   app.search.tools.llm_structural_cypher, app.search.tools.cypher
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
#
# --- Usage / Integration ---
# Included by `backend/app/main.py` to expose advanced graph Cypher generation
# and evaluation endpoints used by the frontend and monitoring tools.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""Graph Cypher router delivering LLM structural generation capabilities."""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.models import QueryRequest
from app.api.utils import get_traversal_k

logger = logging.getLogger("app.main")

# __________________________________________________________________________
# Router Configuration
#
router = APIRouter(prefix="/graph/cypher")

# __________________________________________________________________________
# Endpoint Definitions
#
# -------------------------------------------------------------- generate_cypher_query()
@router.post("/generate")
async def generate_cypher_query(request: QueryRequest) -> JSONResponse:
    """Generate a Cypher query from natural language without executing it.

    Args:
        request: Validated query payload containing the natural language prompt
            and traversal parameters.

    Returns:
        JSONResponse: Generated Cypher, validation details, and metadata, or an
        error payload when generation fails.
    """

    try:
        from app.search.tools.llm_structural_cypher import (
            get_llm_structural_cypher_engine,
        )

        engine = get_llm_structural_cypher_engine()
        max_results = get_traversal_k(request.max_results_traversal_search)
        generated = await engine.generate_cypher(request.user_input, max_results)

        if generated.confidence == 0.0:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Cypher generation failed",
                    "reason": generated.reasoning,
                    "cypher": generated.cypher,
                },
            )

        validation_report = engine.validate_cypher(generated.cypher)

        response_data: dict[str, Any] = {
            "cypher": generated.cypher,
            "confidence": generated.confidence,
            "generation_time_ms": generated.generation_time_ms,
            "tokens_used": generated.tokens_used,
            "model_used": generated.model_used,
            "reasoning": generated.reasoning,
            "validation": {
                "is_valid": validation_report.is_valid,
                "safe_cypher": validation_report.safe_cypher,
                "issues": [
                    {
                        "severity": issue.severity.value,
                        "code": issue.code,
                        "message": issue.message,
                        "element": issue.element,
                        "suggestion": issue.suggestion,
                    }
                    for issue in validation_report.issues
                ],
                "fixes_applied": [
                    {
                        "description": fix.description,
                        "original": fix.original_element,
                        "fixed": fix.fixed_element,
                        "type": fix.fix_type,
                    }
                    for fix in validation_report.fixes_applied
                ],
                "fallback_recommended": validation_report.fallback_recommended,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        return JSONResponse(content=response_data)
    except ImportError:
        return JSONResponse(
            status_code=503,
            content={"error": "LLM Structural Cypher engine not available"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Cypher generation endpoint error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Cypher generation failed: {exc}"},
        )


# -------------------------------------------------------------- end generate_cypher_query()

# -------------------------------------------------------------- query_with_cypher_generation()
@router.post("/query")
async def query_with_cypher_generation(request: QueryRequest) -> JSONResponse:
    """Complete LLM structural Cypher workflow.

    Args:
        request: Validated query payload containing the prompt and traversal
            parameters.

    Returns:
        JSONResponse: Full retrieval response produced by the structural Cypher
        pipeline or an error payload when execution fails.
    """

    try:
        from app.search.tools.cypher import query_knowledge_graph_llm_structural_detailed

        max_results = get_traversal_k(request.max_results_traversal_search)
        result = await query_knowledge_graph_llm_structural_detailed(
            request.user_input,
            max_results,
        )
        return JSONResponse(content=result)
    except ImportError:
        return JSONResponse(
            status_code=503,
            content={"error": "LLM Structural Cypher engine not available"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Cypher query endpoint error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Cypher query failed: {exc}"},
        )


# -------------------------------------------------------------- end query_with_cypher_generation()

# -------------------------------------------------------------- get_cypher_engine_metrics()
@router.get("/metrics")
async def get_cypher_engine_metrics() -> JSONResponse:
    """Return metrics for the structural Cypher engine.

    Returns:
        JSONResponse: Metrics payload with timestamp or error details if the
        structural engine is unavailable.
    """

    try:
        from app.search.tools.llm_structural_cypher import (
            get_llm_structural_cypher_engine,
        )

        engine = get_llm_structural_cypher_engine()
        metrics = engine.get_metrics()

        return JSONResponse(
            content={
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics,
            }
        )
    except ImportError:
        return JSONResponse(
            status_code=503,
            content={"error": "LLM Structural Cypher engine not available"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Cypher metrics endpoint error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get metrics: {exc}"},
        )


# -------------------------------------------------------------- end get_cypher_engine_metrics()

# -------------------------------------------------------------- reset_cypher_engine_metrics()
@router.post("/metrics/reset")
async def reset_cypher_engine_metrics() -> JSONResponse:
    """Reset structural Cypher engine metrics.

    Returns:
        JSONResponse: Confirmation payload with timestamp or error details if
        the reset fails.
    """

    try:
        from app.search.tools.llm_structural_cypher import (
            get_llm_structural_cypher_engine,
        )

        engine = get_llm_structural_cypher_engine()
        engine.reset_metrics()

        return JSONResponse(
            content={
                "message": "Engine metrics reset successfully",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    except ImportError:
        return JSONResponse(
            status_code=503,
            content={"error": "LLM Structural Cypher engine not available"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Cypher metrics reset endpoint error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to reset metrics: {exc}"},
        )


# -------------------------------------------------------------- end reset_cypher_engine_metrics()

# __________________________________________________________________________
# Module Exports
#
__all__ = ["router"]

# __________________________________________________________________________
# End of File