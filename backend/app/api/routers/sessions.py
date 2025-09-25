# -------------------------------------------------------------------------
# File: sessions.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/app/api/routers/sessions.py
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
#   Defines FastAPI session management endpoints that expose active session
#   metadata and support explicit session clearing for the APH-IF backend
#   service.
# -------------------------------------------------------------------------
#
# --- Module Contents Overview ---
# - Router: router (FastAPI APIRouter instance)
# - Endpoint: get_active_sessions
# - Endpoint: clear_session
# - Constant: __all__ (module exports)
# -------------------------------------------------------------------------
#
# --- Dependencies / Imports ---
# - Standard Library: datetime (datetime), typing (Any)
# - Third-Party: fastapi (APIRouter, JSONResponse)
# - Local Project Modules: app.api.state (shared API state)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
#
# --- Usage / Integration ---
# Routers in this module are included by `backend/app/main.py` to provide
# HTTP surfaces for session diagnostics and manual clearing operations.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""Session router exposing active session diagnostics and clear operations."""

# __________________________________________________________________________
# Imports

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api import state as api_state

# __________________________________________________________________________
# Router Configuration
#
router = APIRouter(prefix="/sessions")


# __________________________________________________________________________
# Endpoint Definitions
#
# -------------------------------------------------------------- get_active_sessions()
@router.get("")
async def get_active_sessions() -> JSONResponse:
    """Return active session metadata.

    Returns:
        JSONResponse: Summary containing total count, per-session statistics,
        and a timestamp indicating when the snapshot was generated.
    """

    session_info: dict[str, Any] = {
        "total_sessions": len(api_state.active_sessions),
        "sessions": {
            session_id[:8]: {
                "query_count": info.get("query_count", 0),
                "last_activity": info.get("timestamp", "unknown"),
            }
            for session_id, info in api_state.active_sessions.items()
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
    return JSONResponse(content=session_info)
# -------------------------------------------------------------- end get_active_sessions()


# -------------------------------------------------------------- clear_session()
@router.delete("/{session_id}")
async def clear_session(session_id: str) -> JSONResponse:
    """Clear a specific session if present.

    Args:
        session_id: Unique session identifier to remove from the active
            session store.

    Returns:
        JSONResponse: Confirmation message when the session is removed, or a
        404 error payload when the session does not exist.
    """

    if session_id in api_state.active_sessions:
        del api_state.active_sessions[session_id]
        return JSONResponse(content={"message": f"Session {session_id[:8]} cleared"})
    return JSONResponse(status_code=404, content={"error": "Session not found"})
# -------------------------------------------------------------- end clear_session()


# __________________________________________________________________________
# Module Exports
#
__all__ = ["router"]

# __________________________________________________________________________
# End of File