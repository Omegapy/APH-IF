# -------------------------------------------------------------------------
# File: timing.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/app/api/routers/timing.py
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
#   Exposes FastAPI endpoints for timing analytics, detailed breakdowns,
#   performance analysis, and collector administration for APH-IF.
# -------------------------------------------------------------------------
#
# --- Module Contents Overview ---
# - Router: router (FastAPI APIRouter instance)
# - Endpoint: get_timing_breakdown
# - Endpoint: get_detailed_timing_breakdown
# - Endpoint: get_database_timing_metrics
# - Endpoint: get_active_timing_operations
# - Endpoint: get_timing_performance_analysis
# - Endpoint: clear_timing_data
# - Constant: __all__ (module exports)
# -------------------------------------------------------------------------
#
# --- Dependencies / Imports ---
# - Standard Library: logging, datetime (datetime), typing (Any)
# - Third-Party: fastapi (APIRouter)
# - Local Project Modules: app.monitoring.timing_collector,
#   app.api.utils.generate_timing_recommendations
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
#
# --- Usage / Integration ---
# Included by `backend/app/main.py` to expose timing analytics and fine-grained
# instrumentation data to monitoring dashboards and operations tooling.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""Timing analytics router delivering breakdowns and performance insights."""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from app.api.utils import generate_timing_recommendations

logger = logging.getLogger("app.main")

# __________________________________________________________________________
# Router Configuration
#
router = APIRouter(prefix="/timing")


# __________________________________________________________________________
# Endpoint Definitions
#
# -------------------------------------------------------------- get_timing_breakdown()
@router.get("/breakdown")
async def get_timing_breakdown(
    operation_id: Optional[str] = None,
    recent_count: int = 100,
) -> Dict[str, Any]:
    """Return hierarchical timing breakdown for recent requests.

    Args:
        operation_id: Optional operation identifier used to retrieve a specific
            timing breakdown.
        recent_count: Limit on the number of recent operations included when no
            operation identifier is provided.

    Returns:
        dict[str, Any]: Timing breakdown payload for a specific operation or a
        list of recent operations with summary statistics. Returns an error
        payload when any timing collection errors occur.
    """

    try:
        from app.monitoring.timing_collector import get_timing_collector

        collector = get_timing_collector()

        if operation_id:
            breakdown = collector.get_timing_breakdown(operation_id)
            if not breakdown:
                return {"error": f"Operation {operation_id} not found"}
            return breakdown

        recent_timings = collector.get_recent_timings(recent_count)
        return {
            "recent_operations": [
                {
                    "operation_id": timing.operation_id,
                    "operation_name": timing.operation_name,
                    "duration_ms": timing.duration_ms,
                    "success": timing.success,
                    "start_time": timing.start_time,
                    "parent_id": timing.parent_id,
                    "children_count": len(timing.children),
                }
                for timing in recent_timings
            ],
            "total_operations": len(recent_timings),
            "collector_stats": collector.get_statistics(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Error getting timing breakdown: %s", exc)
        return {"error": str(exc)}
# -------------------------------------------------------------- end get_timing_breakdown()

# -------------------------------------------------------------- get_detailed_timing_breakdown()
@router.get("/detailed-breakdown")
async def get_detailed_timing_breakdown() -> Dict[str, Any]:
    """Get comprehensive detailed timing breakdown.

    Returns:
        dict[str, Any]: Detailed timing analytics including request lifecycle,
        database access, LLM interaction, parallel processing, cache, and
        quality metrics or an error payload when unavailable.
    """

    try:
        from app.monitoring.timing_collector import get_timing_collector

        collector = get_timing_collector()
        detailed_breakdown = collector.generate_detailed_breakdown()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "detailed_timing_breakdown": {
                "total_request_response_time_ms": (
                    detailed_breakdown.total_request_response_time_ms
                ),
                "user_perceived_response_time_ms": (
                    detailed_breakdown.user_perceived_response_time_ms
                ),
                "time_to_first_result_ms": detailed_breakdown.time_to_first_result_ms,
                "request_lifecycle": {
                    "parsing_time_ms": detailed_breakdown.request_parsing_time_ms,
                    "routing_time_ms": detailed_breakdown.request_routing_time_ms,
                    "session_handling_time_ms": (
                        detailed_breakdown.session_handling_time_ms
                    ),
                    "parameter_validation_time_ms": (
                        detailed_breakdown.parameter_validation_time_ms
                    ),
                    "queue_time_ms": detailed_breakdown.request_queue_time_ms,
                },
                "database_access": {
                    "connection_acquisition_ms": (
                        detailed_breakdown.neo4j_connection_acquisition_time_ms
                    ),
                    "connection_establishment_ms": (
                        detailed_breakdown.neo4j_connection_establishment_time_ms
                    ),
                    "query_compilation_ms": (
                        detailed_breakdown.cypher_query_compilation_time_ms
                    ),
                    "query_execution_ms": (
                        detailed_breakdown.cypher_query_execution_time_ms
                    ),
                    "result_processing_ms": (
                        detailed_breakdown.graph_result_processing_time_ms
                    ),
                    "network_latency_ms": detailed_breakdown.neo4j_network_latency_ms,
                },
                "llm_api": {
                    "total_request_time_ms": (
                        detailed_breakdown.llm_api_request_time_ms
                    ),
                    "network_latency_ms": detailed_breakdown.llm_network_latency_ms,
                    "processing_time_ms": detailed_breakdown.llm_processing_time_ms,
                    "token_encoding_ms": detailed_breakdown.token_encoding_time_ms,
                    "token_decoding_ms": detailed_breakdown.token_decoding_time_ms,
                    "rate_limit_wait_ms": detailed_breakdown.rate_limit_wait_time_ms,
                },
                "parallel_processing": {
                    "coordination_overhead_ms": (
                        detailed_breakdown.parallel_coordination_overhead_ms
                    ),
                    "task_synchronization_ms": (
                        detailed_breakdown.parallel_task_synchronization_ms
                    ),
                    "semantic_execution_ms": (
                        detailed_breakdown.semantic_search_execution_time_ms
                    ),
                    "traversal_execution_ms": (
                        detailed_breakdown.traversal_search_execution_time_ms
                    ),
                    "speedup_achieved_ms": (
                        detailed_breakdown.parallel_speedup_achieved_ms
                    ),
                },
                "cache_operations": {
                    "lookup_time_ms": detailed_breakdown.cache_lookup_time_ms,
                    "hit_retrieval_ms": detailed_breakdown.cache_hit_retrieval_time_ms,
                    "miss_processing_ms": detailed_breakdown.cache_miss_processing_time_ms,
                    "storage_time_ms": detailed_breakdown.cache_storage_time_ms,
                    "similarity_matching_ms": (
                        detailed_breakdown.cache_similarity_matching_time_ms
                    ),
                },
                "quality_processing": {
                    "entity_extraction_ms": detailed_breakdown.entity_extraction_time_ms,
                    "result_validation_ms": detailed_breakdown.result_validation_time_ms,
                    "source_attribution_ms": detailed_breakdown.source_attribution_time_ms,
                    "content_formatting_ms": detailed_breakdown.content_formatting_time_ms,
                },
            },
            "collector_statistics": collector.get_statistics(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Error generating detailed timing breakdown: %s", exc)
        return {"error": str(exc)}

# -------------------------------------------------------------- end get_detailed_timing_breakdown()

# -------------------------------------------------------------- get_database_timing_metrics()
@router.get("/database")
async def get_database_timing_metrics() -> Dict[str, Any]:
    """Return placeholder database timing metrics.

    Returns:
        dict[str, Any]: Timestamped payload noting that database metrics are
        disabled.
    """

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "database_metrics": "disabled",
    }

# -------------------------------------------------------------- end get_database_timing_metrics()


# -------------------------------------------------------------- get_active_timing_operations()
@router.get("/active-operations")
async def get_active_timing_operations() -> Dict[str, Any]:
    """Get currently active timing operations.

    Returns:
        dict[str, Any]: Snapshot of active timing operations including metadata
        for each active context or an error payload when unavailable.
    """

    try:
        from app.monitoring.timing_collector import get_timing_collector

        collector = get_timing_collector()
        active_contexts = collector.get_active_contexts()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "active_operations_count": len(active_contexts),
            "active_operations": [
                {
                    "operation_id": ctx.operation_id,
                    "operation_name": ctx.operation_name,
                    "start_time": ctx.start_time,
                    "duration_so_far_ms": ctx.duration_ms,
                    "parent_id": ctx.parent_id,
                    "children_count": len(ctx.children),
                    "metadata": ctx.metadata,
                }
                for ctx in active_contexts.values()
            ],
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Error getting active timing operations: %s", exc)
        return {"error": str(exc)}

# -------------------------------------------------------------- end get_active_timing_operations()

# -------------------------------------------------------------- get_timing_performance_analysis()
@router.get("/performance-analysis")
async def get_timing_performance_analysis() -> Dict[str, Any]:
    """Get performance analysis based on timing data.

    Returns:
        dict[str, Any]: Performance analysis summary including bottlenecks,
        collector health, and recommended optimizations or an error payload when
        unavailable.
    """

    try:
        from app.monitoring.timing_collector import get_timing_collector

        timing_collector = get_timing_collector()
        recent_timings = timing_collector.get_recent_timings(500)
        db_stats = {"disabled": "database metrics not available"}
        collector_stats = timing_collector.get_statistics()

        operation_analysis: Dict[str, Dict[str, Any]] = {}
        for timing in recent_timings:
            op_name = timing.operation_name
            if op_name not in operation_analysis:
                operation_analysis[op_name] = {
                    "count": 0,
                    "total_time_ms": 0,
                    "success_count": 0,
                    "avg_time_ms": 0,
                    "success_rate": 0,
                }

            op_stats = operation_analysis[op_name]
            op_stats["count"] += 1
            op_stats["total_time_ms"] += timing.duration_ms
            if timing.success:
                op_stats["success_count"] += 1

        for stats in operation_analysis.values():
            if stats["count"] > 0:
                stats["avg_time_ms"] = stats["total_time_ms"] / stats["count"]
                stats["success_rate"] = (
                    stats["success_count"] / stats["count"]
                ) * 100

        bottlenecks: List[Dict[str, Any]] = []
        for op_name, stats in operation_analysis.items():
            if stats["avg_time_ms"] > 1000:
                bottlenecks.append(
                    {
                        "operation": op_name,
                        "avg_time_ms": stats["avg_time_ms"],
                        "count": stats["count"],
                        "success_rate": stats["success_rate"],
                    }
                )

        bottlenecks.sort(key=lambda item: item["avg_time_ms"], reverse=True)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "analysis_window": "30 minutes",
            "operation_analysis": operation_analysis,
            "performance_bottlenecks": bottlenecks[:10],
            "database_performance": db_stats,
            "collector_health": {
                "total_timings_collected": collector_stats["total_timings"],
                "active_contexts": collector_stats["active_contexts"],
                "collection_overhead_ms": collector_stats["overhead_ms"],
                "collection_uptime_seconds": collector_stats.get(
                    "collection_uptime_seconds", 0
                ),
            },
            "recommendations": generate_timing_recommendations(
                operation_analysis, bottlenecks, db_stats
            ),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Error generating timing performance analysis: %s", exc)
        return {"error": str(exc)}

# -------------------------------------------------------------- end get_timing_performance_analysis()

# -------------------------------------------------------------- clear_timing_data()
@router.post("/clear")
async def clear_timing_data() -> Dict[str, Any]:
    """Clear all timing data and reset collectors.

    Returns:
        dict[str, Any]: Confirmation payload with cleared counts and timestamp
        or an error payload when the operation fails.
    """

    try:
        from app.monitoring.timing_collector import get_timing_collector

        timing_collector = get_timing_collector()
        cleared_timings = timing_collector.clear_completed_timings()

        return {
            "message": "All timing data cleared successfully",
            "cleared_timings": cleared_timings,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Error clearing timing data: %s", exc)
        return {"error": str(exc)}


# -------------------------------------------------------------- end clear_timing_data()

# __________________________________________________________________________
# Module Exports
#
__all__ = ["router"]

# __________________________________________________________________________
# End of File