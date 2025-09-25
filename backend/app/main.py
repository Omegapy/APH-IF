# -------------------------------------------------------------------------
# File: main.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/app/main.py
# ------------------------------------------------------------------------
# Project: APH-IF
#
# Project description:
# Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF)
# is a novel Retrieval Augmented Generation (RAG) system that differs from
# traditional RAG approaches by performing semantic and traversal searches
# concurrently, rather than sequentially, and fusing the results using an LLM
# or an LRM to generate the final response.
# -------------------------------------------------------------------------

# --- Module Functionality ---
#   Defines the FastAPI application entrypoint responsible for orchestrating
#   hybrid retrieval, schema access, monitoring, and lifecycle hooks within the
#   backend service of APH-IF.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - FastAPI application instance and middleware configuration
# - Domain router registration for backend API surfaces
# - Lifecycle hook registration for startup and shutdown orchestration
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: logging
# - Third-Party: fastapi, fastapi.middleware.cors
# - Local Project Modules:
#   - .core.config (settings)
#   - .api.lifecycle (startup/shutdown hooks)
#   - .api.routers (domain route modules)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Exposed endpoints are consumed by the Streamlit frontend and other services to
# perform hybrid RAG queries, retrieve schema metadata, monitor performance, and
# manage application lifecycle orchestration within the backend module.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent
# Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""FastAPI entrypoint for the APH-IF backend service.

Initializes the FastAPI app, wires hybrid retrieval engines, exposes monitoring
and schema endpoints, and manages startup/shutdown orchestration for backend
resources in the APH-IF platform.
"""

# __________________________________________________________________________
# Imports
from __future__ import annotations

import importlib
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import state as api_state
from .api.lifecycle import register_shutdown, register_startup
from .api.routers import (
    graph_cypher,
    health,
    performance,
    query,
    root,
    schema,
    sessions,
    timing,
)
from .core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Try to import hybrid modules
try:
    importlib.import_module("app.search.parallel_hybrid")
    importlib.import_module("app.search.context_fusion")
    importlib.import_module("app.monitoring.performance_monitor")
    importlib.import_module("app.monitoring.circuit_breaker")
    api_state.HYBRID_AVAILABLE = True
    logger.info("✅ Advanced Parallel Hybrid modules loaded successfully")
except ImportError as exc:
    api_state.HYBRID_AVAILABLE = False
    logger.warning("⚠️ Advanced Parallel Hybrid modules not available: %s", exc)
    logger.info("   Backend will run in basic mode")

# FastAPI app initialization
app = FastAPI(
    title="APH-IF Backend API",
    description="Advanced Parallel HybridRAG - Intelligent Fusion System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should be tightened in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_startup(app)
register_shutdown(app)

app.include_router(root.router)
app.include_router(health.router)
app.include_router(performance.router)
app.include_router(timing.router)
app.include_router(schema.router)
app.include_router(sessions.router)
app.include_router(graph_cypher.router)
app.include_router(query.router)
 
# =========================================================================
# Server
# =========================================================================

# __________________________________________________________________________
# Module Initialization / Main Execution Guard
# This code runs only when the file is executed
# -------------------------------------------------------------- main_guard
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=settings.host, 
        port=settings.port,
        reload=settings.debug
    )
# -------------------------------------------------------------- end main_guard

# __________________________________________________________________________
# End of File
#
