# -------------------------------------------------------------------------
# File: health.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/app/api/routers/health.py
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
#   Exposes FastAPI health endpoints that report backend status, component
#   readiness, and hybrid system diagnostics for the APH-IF platform.
# -------------------------------------------------------------------------
#
# --- Module Contents Overview ---
# - Router: router (FastAPI APIRouter instance)
# - Endpoint: health_check
# - Endpoint: hybrid_health_check
# - Constant: __all__ (module exports)
# -------------------------------------------------------------------------
#
# --- Dependencies / Imports ---
# - Standard Library: logging, time, datetime (datetime), typing (Any)
# - Third-Party: fastapi (APIRouter), fastapi.responses (JSONResponse)
# - Local Project Modules: app.api.state, app.api.models, app.core.config,
#   app.schema, app.search tools (vector, llm_structural_cypher, context fusion,
#   parallel hybrid)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
#
# --- Usage / Integration ---
# Included by `backend/app/main.py` to provide `/healthz` and hybrid health
# endpoints consumed by monitoring systems and frontend status indicators.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""Health router exposing backend and hybrid diagnostic endpoints."""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api import state as api_state
from app.api.models import HealthResponse
from app.core.config import settings

logger = logging.getLogger("app.main")

# __________________________________________________________________________
# Router Configuration
#
router = APIRouter()

# __________________________________________________________________________
# Endpoint Definitions
#
# -------------------------------------------------------------- health_check()
@router.get("/healthz", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Assess component-level health for the backend service.

    Returns:
        HealthResponse: Aggregated status including environment details,
        engine health, and schema cache information.
    """

    try:
        from app.schema import get_schema_manager

        schema_manager = get_schema_manager()
        cache_info = schema_manager.get_cache_info()
        db_health: dict[str, Any] = {
            "database": "disabled",
            "reason": "database health removed; cache only",
            "cache_info": cache_info,
        }
    except ImportError:
        db_health = {"database": "unavailable", "error": "Schema module not available"}
    except Exception as exc:  # noqa: BLE001
        db_health = {"database": "error", "error": str(exc)}

    overall_status = "healthy"
    db_status = db_health.get("database", "unknown")
    if db_status == "error":
        overall_status = "unhealthy"
    elif db_status == "unavailable":
        overall_status = "degraded"

    semantic_engine_health: dict[str, Any] = {"status": "unavailable"}
    if api_state.HYBRID_AVAILABLE:
        try:
            from app.search.tools.vector import get_engine_stats

            stats = get_engine_stats()
            engine_health = stats.get("engine_health", "unknown")

            semantic_engine_health = {
                "status": engine_health,
                "total_requests": stats.get("total_requests", 0),
                "current_engine_mode": stats.get(
                    "current_engine_mode", "semantic_cached"
                ),
            }

            if engine_health in {"error", "uninitialized"} and overall_status == "healthy":
                overall_status = "degraded"
        except Exception as exc:  # noqa: BLE001
            semantic_engine_health = {"status": "error", "error": str(exc)}

    traversal_engine_health: dict[str, Any] = {"status": "unavailable"}
    if api_state.HYBRID_AVAILABLE and settings.use_llm_structural_cypher:
        try:
            from app.search.tools.llm_structural_cypher import (
                get_llm_structural_cypher_engine,
            )

            engine = get_llm_structural_cypher_engine()
            metrics = engine.get_metrics()

            total_generations = metrics.get("total_generations", 0)

            narrative_success_rate = 0.0
            narrative_attempts = metrics.get("narrative_attempts", 0)
            if narrative_attempts > 0:
                narrative_success_rate = (
                    metrics.get("narrative_successes", 0) / narrative_attempts
                )

            status = "healthy"
            successful_generations = metrics.get("successful_generations", 0)
            validation_failures = metrics.get("validation_failures", 0)

            if validation_failures > successful_generations * 0.3:
                status = "degraded"
            elif not total_generations:
                status = "unused"

            traversal_engine_health = {
                "status": status,
                "total_generations": total_generations,
                "successful_generations": successful_generations,
                "validation_failures": validation_failures,
                "execution_failures": metrics.get("execution_failures", 0),
                "narrative_attempts": narrative_attempts,
                "narrative_successes": metrics.get("narrative_successes", 0),
                "narrative_success_rate": round(narrative_success_rate, 3),
                "citations_validated": metrics.get("citations_validated", 0),
                "citations_dropped": metrics.get("citations_dropped", 0),
                "avg_execution_time_ms": round(
                    metrics.get("total_execution_time_ms", 0)
                    / max(1, total_generations),
                    1,
                ),
            }

            if status == "degraded" and overall_status == "healthy":
                overall_status = "degraded"
        except ImportError:
            traversal_engine_health = {
                "status": "module_unavailable",
                "reason": "LLM Structural Cypher engine not available",
            }
        except Exception as exc:  # noqa: BLE001
            traversal_engine_health = {"status": "error", "error": str(exc)}
    elif api_state.HYBRID_AVAILABLE:
        traversal_engine_health = {
            "status": "disabled",
            "reason": "LLM Structural Cypher disabled via configuration",
            "config_flag": settings.use_llm_structural_cypher,
        }

    components = {
        "api": {
            "status": "healthy",
            "uptime_seconds": int(time.time() - api_state.startup_time),
        },
        "database": {
            "status": db_health.get("database", "unknown"),
            "response_time": db_health.get("response_time"),
            "error": db_health.get("error"),
        },
        "semantic_engine": semantic_engine_health,
        "traversal_engine": traversal_engine_health,
        "parallel_hybrid": {
            "status": "available" if api_state.HYBRID_AVAILABLE else "unavailable"
        },
        "environment": {
            "mode": settings.environment_mode.value,
            "neo4j_instance": settings.get_neo4j_mode_name(),
            "debug": settings.debug,
            "verbose": settings.verbose,
        },
    }

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        environment=settings.environment_mode.value,
        components=components,
    )

# -------------------------------------------------------------- end health_check()

# -------------------------------------------------------------- hybrid_health_check()
@router.get("/health/hybrid")
async def hybrid_health_check() -> JSONResponse:
    """Return health information for the hybrid retrieval system.

    Returns:
        JSONResponse: Status payload describing parallel retrieval and context
        fusion components or an error payload when unavailable.
    """

    if not api_state.HYBRID_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"error": "Parallel hybrid system not available"},
        )

    try:
        from app.search.context_fusion import get_fusion_engine
        from app.search.parallel_hybrid import get_parallel_engine

        parallel_engine = get_parallel_engine()
        fusion_engine = get_fusion_engine()

        parallel_health = await parallel_engine.health_check()
        fusion_health = await fusion_engine.health_check()

        overall_status = "healthy"
        if (
            parallel_health.get("status") != "healthy"
            or fusion_health.get("status") != "healthy"
        ):
            overall_status = "degraded"

        hybrid_health: dict[str, Any] = {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "parallel_retrieval": parallel_health,
                "context_fusion": fusion_health,
            },
            "fusion_stats": fusion_engine.get_fusion_stats(),
            "capabilities": {
                "concurrent_search": True,
                "intelligent_fusion": True,
                "multiple_strategies": True,
                "adaptive_weighting": True,
            },
        }

        return JSONResponse(content=hybrid_health)
    except Exception as exc:  # noqa: BLE001
        logger.error("Hybrid health check failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Hybrid health check failed: {exc}"},
        )

# -------------------------------------------------------------- end hybrid_health_check()

# __________________________________________________________________________
# Module Exports
#
__all__ = ["router"]

# __________________________________________________________________________
# End of File