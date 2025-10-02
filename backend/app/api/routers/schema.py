# -------------------------------------------------------------------------
# File: schema.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/app/api/routers/schema.py
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
#   Exposes FastAPI routers that surface schema management capabilities to
#   clients. Includes endpoints for cached schema inspection, refresh actions,
#   and human-readable summaries sourced from the schema manager subsystem.
# -------------------------------------------------------------------------
#
# --- Module Contents Overview ---
# - Router: router (FastAPI APIRouter instance)
# - Endpoint: get_schema
# - Endpoint: refresh_schema
# - Endpoint: get_schema_info
# - Endpoint: get_node_info
# - Endpoint: get_relationship_info
# - Endpoint: get_structural_schema
# - Endpoint: refresh_structural_schema
# - Helper Function: _format_schema_text_summary
# - Constant: __all__ (module exports)
# -------------------------------------------------------------------------
#
# --- Dependencies / Imports ---
# - Standard Library: logging, typing (Any)
# - Third-Party: fastapi (APIRouter, JSONResponse)
# - Local Project Modules: app.schema (schema manager factory)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
#
# --- Usage / Integration ---
# Routers in this module are included by `backend/app/main.py` to provide
# HTTP endpoints for interacting with schema metadata and caches during RAG
# operations.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""Backend schema router exposing cached graph metadata endpoints.

Provides read and refresh capabilities for the knowledge graph schema. The
endpoints are consumed by the frontend and other services to inspect schema
state, request refreshes, and generate summaries for LLM consumption.
"""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

# __________________________________________________________________________
# Router Configuration
#
logger = logging.getLogger("app.main")
router = APIRouter(prefix="/schema")

# __________________________________________________________________________
# Endpoint Definitions
#
# -------------------------------------------------------------- get_schema()
@router.get("")
async def get_schema(
    detailed: bool = False,
    force_refresh: bool = False,
    format_type: str = "json",
) -> JSONResponse:
    """Retrieve knowledge graph schema details.

    Args:
        detailed: When True, return detailed schema structures instead of the
            cached summary.
        force_refresh: When True, bypass cached data and request a fresh schema
            acquisition.
        format_type: Output format selection. Supports "json" and "text".

    Returns:
        JSONResponse: API response containing schema data or error details.
    """

    try:
        from app.schema import get_schema_manager

        schema_manager = get_schema_manager()

        if detailed:
            schema = await schema_manager.get_schema_async(
                force_refresh=force_refresh
            )
            if not schema:
                return JSONResponse(
                    status_code=503,
                    content={"error": "Schema not available"},
                )

            if format_type == "text":
                summary = _format_schema_text_summary(schema)
                return JSONResponse(
                    content={"schema_summary": summary, "format": "text"}
                )

            return JSONResponse(content=schema.to_dict())

        summary = schema_manager.get_schema_summary()
        if not summary.get("available"):
            return JSONResponse(
                status_code=503,
                content={"error": "Schema not available"},
            )

        schema_info = {
            "node_labels": schema_manager.get_node_labels(),
            "relationship_types": schema_manager.get_relationship_types(),
            "property_keys": schema_manager.get_all_properties(),
            "total_labels": summary["node_labels_count"],
            "total_relationship_types": summary[
                "relationship_types_count"
            ],
            "total_property_keys": summary["global_properties_count"],
            "total_nodes": summary["total_nodes"],
            "total_relationships": summary["total_relationships"],
            "constraints_count": summary["constraints_count"],
            "indexes_count": summary["indexes_count"],
            "timestamp": summary["acquisition_timestamp"],
            "cache_age_seconds": summary["cache_age_seconds"],
            "cache_valid": summary["cache_valid"],
        }

        return JSONResponse(content=schema_info)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error getting schema: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to retrieve schema: {exc}"},
        )
# -------------------------------------------------------------- end get_schema()

# -------------------------------------------------------------- refresh_schema()
@router.post("/refresh")
async def refresh_schema() -> JSONResponse:
    """Force a schema refresh via the schema manager.

    Returns:
        JSONResponse: API response indicating refresh status and summary
            details when successful.
    """

    try:
        from app.schema import get_schema_manager

        schema_manager = get_schema_manager()

        logger.info("Schema refresh requested via API")
        success = await schema_manager.refresh_schema_async()

        if success:
            summary = schema_manager.get_schema_summary()
            return JSONResponse(
                content={
                    "message": "Schema refreshed successfully",
                    "summary": summary,
                }
            )
        return JSONResponse(
            status_code=500,
            content={"error": "Schema refresh failed"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error refreshing schema: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to refresh schema: {exc}"},
        )
# -------------------------------------------------------------- end refresh_schema()

# -------------------------------------------------------------- get_schema_info()
@router.get("/info")
async def get_schema_info() -> JSONResponse:
    """Return cached schema metadata and cache details.

    Returns:
        JSONResponse: Mapping containing schema summary and cache metadata, or
        an error payload when retrieval fails.
    """

    try:
        from app.schema import get_schema_manager

        schema_manager = get_schema_manager()
        info = {
            "schema_summary": schema_manager.get_schema_summary(),
            "cache_info": schema_manager.get_cache_info(),
        }
        return JSONResponse(content=info)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error getting schema info: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get schema info: {exc}"},
        )
# -------------------------------------------------------------- end get_schema_info()

# -------------------------------------------------------------- get_node_info()
@router.get("/nodes/{label}")
async def get_node_info(label: str) -> JSONResponse:
    """Return information for a specific node label.

    Args:
        label: Neo4j node label to inspect.

    Returns:
        JSONResponse: Node metadata or error payload if the label is missing.
    """

    try:
        from app.schema import get_schema_manager

        schema_manager = get_schema_manager()
        node_info = schema_manager.get_node_info(label)

        if not node_info:
            return JSONResponse(
                status_code=404,
                content={"error": f"Node label '{label}' not found"},
            )

        return JSONResponse(content=node_info)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error getting node info for %s: %s", label, exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get node info: {exc}"},
        )
# -------------------------------------------------------------- end get_node_info()

# -------------------------------------------------------------- get_relationship_info()
@router.get("/relationships/{rel_type}")
async def get_relationship_info(rel_type: str) -> JSONResponse:
    """Return information for a specific relationship type.

    Args:
        rel_type: Relationship type to inspect.

    Returns:
        JSONResponse: Relationship metadata or error payload if missing.
    """

    try:
        from app.schema import get_schema_manager

        schema_manager = get_schema_manager()
        rel_info = schema_manager.get_relationship_info(rel_type)

        if not rel_info:
            return JSONResponse(
                status_code=404,
                content={"error": f"Relationship type '{rel_type}' not found"},
            )

        return JSONResponse(content=rel_info)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error getting relationship info for %s: %s", rel_type, exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get relationship info: {exc}"},
        )
# -------------------------------------------------------------- end get_relationship_info()

# -------------------------------------------------------------- get_structural_schema()
@router.get("/structural")
async def get_structural_schema(format_type: str = "json") -> JSONResponse:
    """Retrieve lightweight structural schema optimized for LLM consumption.

    Args:
        format_type: Output representation. Supports "json" and "text".

    Returns:
        JSONResponse: Structural schema payload tailored to the requested
            format or an error payload when unavailable.
    """

    try:
        from app.schema import get_schema_manager

        schema_manager = get_schema_manager()

        if format_type == "text":
            text_summary = schema_manager.get_structural_summary_for_llm()
            if not text_summary:
                return JSONResponse(
                    status_code=404,
                    content={"error": "Structural summary not available"},
                )

            return JSONResponse(
                content={
                    "structural_summary": text_summary,
                    "format": "text",
                }
            )

        summary_dict = schema_manager.get_structural_summary_dict()
        if not summary_dict:
            return JSONResponse(
                status_code=404,
                content={"error": "Structural summary not available"},
            )

        return JSONResponse(content=summary_dict)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error getting structural schema: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to get structural schema: {exc}"},
        )
# -------------------------------------------------------------- end get_structural_schema()

# -------------------------------------------------------------- refresh_structural_schema()
@router.post("/structural/refresh")
async def refresh_structural_schema() -> JSONResponse:
    """Force reload of structural schema from disk.

    Returns:
        JSONResponse: API response summarizing refreshed structural schema
            counts or an error payload when refresh fails.
    """

    try:
        from app.schema import get_schema_manager

        schema_manager = get_schema_manager()

        logger.info("Structural schema refresh requested via API")
        summary = schema_manager.get_structural_summary(force_refresh=True)

        if summary:
            return JSONResponse(
                content={
                    "message": "Structural schema refreshed successfully",
                    "summary": {
                        "node_labels": len(summary.get_node_labels_list()),
                        "relationship_types": len(summary.get_relationship_types_list()),
                        "last_loaded": (
                            summary.last_loaded.isoformat()
                            if summary.last_loaded
                            else None
                        ),
                    },
                }
            )
        return JSONResponse(
            status_code=404,
            content={"error": "Structural summary file not found"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error refreshing structural schema: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to refresh structural schema: {exc}"},
        )
# -------------------------------------------------------------- end refresh_structural_schema()

# -------------------------------------------------------------- _format_schema_text_summary()
def _format_schema_text_summary(schema: Any) -> str:
    """Format schema information into a human-readable summary.

    Args:
        schema: Schema data object returned by the schema manager.

    Returns:
        str: Multi-line textual summary suited for UI rendering or logs.
    """

    lines = [
        "=== Knowledge Graph Schema Summary ===",
        f"Retrieved at: {schema.acquisition_timestamp}",
        f"Acquisition time: {schema.acquisition_duration_seconds:.2f}s",
        "",
        "📊 Overview:",
        f"  • Total Nodes: {schema.total_nodes:,}",
        f"  • Total Relationships: {schema.total_relationships:,}",
        f"  • Node Labels: {len(schema.nodes)}",
        f"  • Relationship Types: {len(schema.relationships)}",
        f"  • Property Keys: {len(schema.global_property_keys)}",
        f"  • Constraints: {len(schema.all_constraints)}",
        f"  • Indexes: {len(schema.all_indexes)}",
        "",
        "🏷️ Node Labels:",
    ]

    for label, node_info in list(schema.nodes.items())[:10]:
        lines.append(f"  • {label}: {node_info.total_count:,} nodes")
        if node_info.all_properties:
            props = sorted(list(node_info.all_properties))[:5]
            props_str = ", ".join(props)
            if len(node_info.all_properties) > 5:
                props_str += f" ... (+{len(node_info.all_properties) - 5} more)"
            lines.append(f"    Properties: {props_str}")

    if len(schema.nodes) > 10:
        lines.append(f"  ... and {len(schema.nodes) - 10} more labels")

    lines.extend(["", "🔗 Relationship Types:"])

    for rel_type, rel_info in list(schema.relationships.items())[:10]:
        lines.append(f"  • {rel_type}: {rel_info.total_count:,} relationships")
        if rel_info.all_patterns:
            patterns = []
            for pattern in rel_info.all_patterns[:3]:
                patterns.append(
                    f"({pattern['source_label']})-[:{rel_type}]->({pattern['target_label']})"
                )
            patterns_str = ", ".join(patterns)
            if len(rel_info.all_patterns) > 3:
                patterns_str += f" ... (+{len(rel_info.all_patterns) - 3} more)"
            lines.append(f"    Patterns: {patterns_str}")

    if len(schema.relationships) > 10:
        lines.append(f"  ... and {len(schema.relationships) - 10} more types")

    return "\n".join(lines)
# -------------------------------------------------------------- end _format_schema_text_summary()

# __________________________________________________________________________
# Module Exports
#
__all__ = ["router"]

# __________________________________________________________________________
# End of File