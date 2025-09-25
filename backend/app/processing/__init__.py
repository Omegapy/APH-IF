# -------------------------------------------------------------------------
# File: __init__.py
# Author: Alexander Ricciardi
# Date: 2025-09-18
# [File Path] backend/app/processing/__init__.py
# ------------------------------------------------------------------------
# Project:
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
#   Public entrypoints for the processing package, re-exporting citation and
#   process-level parallel execution utilities used by the backend service.
# -------------------------------------------------------------------------
# --- Dependencies / Imports ---
# - Local modules: citation_processor, process_parallel
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
# --- Usage / Integration ---
# Import these utilities from app.processing for clarity and stable API:
#   from app.processing import get_citation_processor, get_process_parallel_engine
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Processing utilities module for APH-IF backend.

This module provides citation processing and process-level parallel utilities and
exposes a small, stable public API for import convenience.
"""

# __________________________________________________________________________
# Imports
from __future__ import annotations

from .citation_processor import get_citation_processor, shutdown_citation_processor
from .process_parallel import get_process_parallel_engine

# __________________________________________________________________________
# Public API Exports

__all__: list[str] = [
    "get_citation_processor",
    "shutdown_citation_processor",
    "get_process_parallel_engine",
]

# __________________________________________________________________________
# End of File
#