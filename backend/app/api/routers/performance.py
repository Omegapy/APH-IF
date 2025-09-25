# -------------------------------------------------------------------------
# File: performance.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/app/api/routers/performance.py
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
#   Defines FastAPI endpoints that expose performance dashboards, metrics,
#   circuit breaker health, and aggregated status for the APH-IF backend
#   service.
# -------------------------------------------------------------------------
#
# --- Module Contents Overview ---
# - Router: router (FastAPI APIRouter instance)
# - Endpoint: get_performance_dashboard
# - Endpoint: get_performance_metrics
# - Endpoint: get_circuit_breaker_status
# - Endpoint: get_performance_health
# - Constant: __all__ (module exports)
# -------------------------------------------------------------------------
#
# --- Dependencies / Imports ---
# - Standard Library: logging, datetime (datetime), typing (Any)
# - Third-Party: fastapi (APIRouter)
# - Local Project Modules: app.api.state, app.core.config,
#   app.monitoring.performance_monitor, app.monitoring.circuit_breaker,
#   app.search.tools.llm_structural_cypher
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
#
# --- Usage / Integration ---
# Routers in this module are included by `backend/app/main.py` to provide
# observability endpoints for monitoring APH-IF performance in real time.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""Performance router exposing dashboards and health summaries."""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter

from app.api import state as api_state
from app.core.config import settings

logger = logging.getLogger("app.main")

# __________________________________________________________________________
# Router Configuration
#
router = APIRouter(prefix="/performance")

# __________________________________________________________________________
# Endpoint Definitions
#
# -------------------------------------------------------------- get_performance_dashboard()
@router.get("/dashboard")
async def get_performance_dashboard() -> dict[str, Any]:
    """Collect performance dashboard metrics for the backend.

    Returns:
        dict[str, Any]: Dashboard metrics summarizing hybrid engine performance
        or an error payload when unavailable.
    """

    if not api_state.HYBRID_AVAILABLE:
        return {"error": "Performance monitoring not available - hybrid modules not loaded"}

    try:
        from app.monitoring.performance_monitor import get_performance_monitor

        monitor = get_performance_monitor()
        dashboard = monitor.get_performance_dashboard()

        llm_structural_metrics: dict[str, Any] = {}
        if settings.use_llm_structural_cypher:
            try:
                from app.search.tools.llm_structural_cypher import (
                    get_llm_structural_cypher_engine,
                )

                engine = get_llm_structural_cypher_engine()
                metrics = engine.get_metrics()

                total_requests = metrics.total_generations
                success_rate = 0.0
                validation_failure_rate = 0.0
                narrative_success_rate = 0.0

                if total_requests > 0:
                    success_rate = metrics.successful_generations / total_requests
                    validation_failure_rate = metrics.validation_failures / total_requests

                if metrics.narrative_attempts > 0:
                    narrative_success_rate = (
                        metrics.narrative_successes / metrics.narrative_attempts
                    )

                avg_execution_time_ms = 0.0
                if metrics.total_generations > 0:
                    avg_execution_time_ms = (
                        metrics.total_execution_time_ms / metrics.total_generations
                    )

                llm_structural_metrics = {
                    "total_generations": metrics.total_generations,
                    "successful_generations": metrics.successful_generations,
                    "success_rate": round(success_rate, 3),
                    "validation_failures": metrics.validation_failures,
                    "validation_failure_rate": round(validation_failure_rate, 3),
                    "execution_failures": metrics.execution_failures,
                    "narrative_attempts": metrics.narrative_attempts,
                    "narrative_successes": metrics.narrative_successes,
                    "narrative_failures": metrics.narrative_failures,
                    "narrative_success_rate": round(narrative_success_rate, 3),
                    "citations_validated": metrics.citations_validated,
                    "citations_dropped": metrics.citations_dropped,
                    "fields_injected_total": metrics.fields_injected_total,
                    "injection_skipped_aggregation": metrics.injection_skipped_aggregation,
                    "injection_skipped_scope": metrics.injection_skipped_scope,
                    "avg_execution_time_ms": round(avg_execution_time_ms, 1),
                    "total_execution_time_ms": metrics.total_execution_time_ms,
                    "alerts": [],
                }

                if validation_failure_rate > 0.2:
                    llm_structural_metrics["alerts"].append(
                        {
                            "type": "critical",
                            "message": (
                                "High validation failure rate: "
                                f"{validation_failure_rate:.1%}"
                            ),
                            "threshold": "20%",
                            "current": f"{validation_failure_rate:.1%}",
                        }
                    )

                if narrative_success_rate < 0.8 and metrics.narrative_attempts > 5:
                    llm_structural_metrics["alerts"].append(
                        {
                            "type": "warning",
                            "message": (
                                "Low narrative success rate: "
                                f"{narrative_success_rate:.1%}"
                            ),
                            "threshold": "80%",
                            "current": f"{narrative_success_rate:.1%}",
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                llm_structural_metrics = {
                    "error": f"Failed to get LLM Structural metrics: {exc}"
                }

        dashboard["llm_structural_metrics"] = llm_structural_metrics
        dashboard["timestamp"] = datetime.utcnow().isoformat()

        return dashboard
    except Exception as exc:  # noqa: BLE001
        logger.error("Error getting performance dashboard: %s", exc)
        return {"error": str(exc)}
# -------------------------------------------------------------- end get_performance_dashboard()

# -------------------------------------------------------------- get_performance_metrics()
@router.get("/metrics")
async def get_performance_metrics() -> dict[str, Any]:
    """Return the current performance metrics snapshot.

    Returns:
        dict[str, Any]: Metrics snapshot including timestamps, counters, and raw
        monitor stats or an error payload when unavailable.
    """

    if not api_state.HYBRID_AVAILABLE:
        return {"error": "Performance monitoring not available"}

    try:
        from app.monitoring.performance_monitor import get_performance_monitor

        monitor = get_performance_monitor()
        stats = monitor.get_current_stats()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": stats,
            "counters": monitor._counters.copy(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Error getting performance metrics: %s", exc)
        return {"error": str(exc)}
# -------------------------------------------------------------- end get_performance_metrics()

# -------------------------------------------------------------- get_circuit_breaker_status()
@router.get("/circuit-breakers")
async def get_circuit_breaker_status() -> dict[str, Any]:
    """Return status information for circuit breakers.

    Returns:
        dict[str, Any]: Circuit breaker health data or an error payload when the
        monitor cannot be accessed.
    """

    if not api_state.HYBRID_AVAILABLE:
        return {"error": "Circuit breakers not available"}

    try:
        from app.monitoring.circuit_breaker import circuit_breaker_health_check

        breaker_health = await circuit_breaker_health_check()
        return breaker_health
    except Exception as exc:  # noqa: BLE001
        logger.error("Error getting circuit breaker status: %s", exc)
        return {"error": str(exc)}
# -------------------------------------------------------------- end get_circuit_breaker_status()

# -------------------------------------------------------------- get_performance_health()
@router.get("/health")
async def get_performance_health() -> dict[str, Any]:
    """Return an aggregated performance health summary.

    Returns:
        dict[str, Any]: Health status summary combining performance and circuit
        breaker assessments or an error payload when unavailable.
    """

    if not api_state.HYBRID_AVAILABLE:
        return {"error": "Performance monitoring not available"}

    try:
        from app.monitoring.circuit_breaker import circuit_breaker_health_check
        from app.monitoring.performance_monitor import performance_health_check

        perf_health = await performance_health_check()
        breaker_health = await circuit_breaker_health_check()

        components = {
            "performance_monitor": perf_health["status"],
            "circuit_breakers": breaker_health["status"],
        }

        overall_status = "healthy"
        if any(status in {"poor", "critical"} for status in components.values()):
            overall_status = "critical"
        elif any(status in {"degraded", "fair"} for status in components.values()):
            overall_status = "degraded"

        return {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": components,
            "detailed_status": {
                "performance": perf_health,
                "circuit_breakers": breaker_health,
            },
            "summary": {
                "total_queries": perf_health.get("health_metrics", {}).get(
                    "total_queries", 0
                ),
                "success_rate": perf_health.get("health_metrics", {}).get(
                    "success_rate", 0
                ),
                "avg_response_time_ms": perf_health.get("health_metrics", {}).get(
                    "avg_response_time_ms", 0
                ),
                "processing_mode": "direct_no_cache",
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Error getting performance health: %s", exc)
        return {"error": str(exc)}

# __________________________________________________________________________
# Module Exports
#
__all__ = ["router"]

# __________________________________________________________________________
# End of File