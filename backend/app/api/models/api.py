# -------------------------------------------------------------------------
# File: api.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/app/api/models/api.py
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
#   Defines Pydantic request and response models shared by FastAPI routers.
#   These models provide the validated data contracts for query and health
#   endpoints exposed by the APH-IF backend service.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Class (BaseModel): QueryRequest
# - Class (BaseModel): QueryResponse
# - Class (BaseModel): HealthResponse
# - Constant: __all__ (module exports)
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: typing (Any)
# - Third-Party: pydantic (BaseModel)
# - Local Project Modules: None
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Imported by routers under `backend/app/api/routers` to validate request and
# response payloads for query and health endpoints exposed through FastAPI.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""Typed API contracts for the APH-IF backend service.

Collects Pydantic models that describe the payloads exchanged by query and
health endpoints. Imported by FastAPI routers within `backend/app/api` to keep
serialization schemas consistent across the service.
"""

# __________________________________________________________________________
# Imports

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


# ____________________________________________________________________________
# Class Definitions
#
# TODO: BaseModel inheritance already provides validation; @dataclass conversion is unnecessary.
# ------------------------------------------------------------------------- class QueryRequest
class QueryRequest(BaseModel):
    """Request payload accepted by the `/query` endpoint.

    Attributes:
        user_input: Natural language prompt submitted by the user.
        session_id: Optional conversation identifier that keeps context between
            invocations.
        search_type: Retrieval strategy requested by the caller. Defaults to
            "hybrid" to enable semantic and traversal execution.
        max_results_semantic_search: Limit for vector search result count when
            provided.
        max_results_traversal_search: Limit for graph traversal result count
            when provided.
    """

    # ______________________
    #  Instance Fields
    #
    user_input: str
    session_id: str | None = None
    search_type: str | None = "hybrid"
    max_results_semantic_search: int | None = None
    max_results_traversal_search: int | None = None
# ------------------------------------------------------------------------- end class QueryRequest


# TODO: BaseModel inheritance already provides validation; @dataclass conversion is unnecessary.
# ------------------------------------------------------------------------- class QueryResponse
class QueryResponse(BaseModel):
    """Response payload delivered by the `/query` endpoint.

    Attributes:
        response: Generated answer produced by the hybrid RAG pipeline.
        session_id: Conversation identifier associated with the processed
            request.
        processing_time: Total elapsed time for the query pipeline, expressed
            in seconds.
        timestamp: ISO 8601 timestamp representing when the response was
            finalized.
        metadata: Auxiliary diagnostic details (scores, ranking, source map)
            returned to callers.
    """

    # ______________________
    #  Instance Fields
    #
    response: str
    session_id: str
    processing_time: float
    timestamp: str
    metadata: dict[str, Any]
# ------------------------------------------------------------------------- end class QueryResponse


# TODO: BaseModel inheritance already provides validation; @dataclass conversion is unnecessary.
# ------------------------------------------------------------------------- class HealthResponse
class HealthResponse(BaseModel):
    """Response payload returned by the `/healthz` endpoint.

    Attributes:
        status: Overall service status indicator, typically "ok" or
            "degraded".
        timestamp: ISO 8601 timestamp denoting when the health snapshot was
            captured.
        version: Deployed backend application version string.
        environment: Runtime environment label (e.g., development, production,
            testing).
        components: Mapping of component names to their individual health
            details or metrics.
    """

    # ______________________
    #  Instance Fields
    #
    status: str
    timestamp: str
    version: str
    environment: str
    components: dict[str, Any]
# ------------------------------------------------------------------------- end class HealthResponse

# __________________________________________________________________________
# Module Exports
#
__all__ = [
    "HealthResponse",
    "QueryRequest",
    "QueryResponse",
]

# __________________________________________________________________________
# End of File
#
