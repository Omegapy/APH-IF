# -------------------------------------------------------------------------
# File: root.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/app/api/routers/root.py
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
#   Hosts the root FastAPI endpoint that exposes backend metadata for client
#   discovery and service introspection.
# -------------------------------------------------------------------------
#
# --- Module Contents Overview ---
# - Router: router (FastAPI APIRouter instance)
# - Endpoint: root
# - Constant: __all__ (module exports)
# -------------------------------------------------------------------------
#
# --- Dependencies / Imports ---
# - Third-Party: fastapi (APIRouter)
# - Local Project Modules: app.core.config (settings)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
#
# --- Usage / Integration ---
# Included by `backend/app/main.py` to surface the root index endpoint with
# service metadata for external clients and health tooling.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""Root router exposing backend metadata for client discovery."""

# __________________________________________________________________________
# Imports

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings

# __________________________________________________________________________
# Router Configuration
#
router = APIRouter()


# __________________________________________________________________________
# Endpoint Definitions
#
# -------------------------------------------------------------- root()
@router.get("/", response_model=dict[str, str])
async def root() -> dict[str, str]:
    """Return metadata describing the backend service.

    Returns:
        dict[str, str]: Mapping of descriptive fields for the backend API,
        including version, status, and helpful navigation links.
    """

    return {
        "service": "APH-IF Backend API",
        "version": "1.0.0",
        "description": "Advanced Parallel HybridRAG - Intelligent Fusion System",
        "status": "running",
        "environment": settings.environment_mode.value,
        "docs": "/docs",
        "health": "/healthz",
    }
# -------------------------------------------------------------- end root()


# __________________________________________________________________________
# Module Exports
#
__all__ = ["router"]

# __________________________________________________________________________
# End of File