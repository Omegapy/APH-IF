# -------------------------------------------------------------------------
# File: state.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/app/api/state.py
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
#   Provides mutable shared state for API routers and lifecycle hooks,
#   tracking hybrid availability, active sessions, and service startup time.
# -------------------------------------------------------------------------
#
# --- Module Contents Overview ---
# - Constant: HYBRID_AVAILABLE
# - Constant: active_sessions
# - Constant: startup_time
# - Constant: __all__ (module exports)
# -------------------------------------------------------------------------
#
# --- Dependencies / Imports ---
# - Standard Library: time, typing (Any, Dict)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
#
# --- Usage / Integration ---
# Imported by routers and lifecycle handlers to share session state and feature
# availability flags across the backend service.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""Shared mutable API state controlling session tracking and hybrid flags."""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import time
from typing import Any, Dict

# __________________________________________________________________________
# Shared State
#
HYBRID_AVAILABLE: bool = False
active_sessions: Dict[str, Dict[str, Any]] = {}
startup_time: float = time.time()

# __________________________________________________________________________
# Module Exports
#
__all__ = [
    "HYBRID_AVAILABLE",
    "active_sessions",
    "startup_time",
]

# __________________________________________________________________________
# End of File
