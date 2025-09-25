# -------------------------------------------------------------------------
# File: parameters.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/app/api/utils/parameters.py
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
#   Provides helper functions to validate and normalize request parameters
#   controlling semantic and traversal result limits for API calls.
# -------------------------------------------------------------------------
#
# --- Module Contents Overview ---
# - Constant: SEMANTIC_DEFAULT_K
# - Constant: TRAVERSAL_DEFAULT_K
# - Constant: MAX_K
# - Function: _validate_k
# - Function: get_semantic_k
# - Function: get_traversal_k
# - Constant: __all__ (module exports)
# -------------------------------------------------------------------------
#
# --- Dependencies / Imports ---
# - Standard Library: typing (Optional)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
#
# --- Usage / Integration ---
# Imported by API routers to enforce consistent result limits for semantic and
# traversal searches across the APH-IF backend.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""Utility helpers that normalize semantic and traversal result limits."""

# __________________________________________________________________________
# Imports

from __future__ import annotations

from typing import Optional

# __________________________________________________________________________
# Constants
#
SEMANTIC_DEFAULT_K = 15
TRAVERSAL_DEFAULT_K = 50
MAX_K = 1000


# __________________________________________________________________________
# Internal Helpers
#
# -------------------------------------------------------------- _validate_k()
def _validate_k(value: Optional[int], default: int) -> int:
    """Return a bounded positive integer with sensible defaults.

    Args:
        value: Caller-provided maximum results value to validate.
        default: Fallback value used when validation fails or the input is not
            provided.

    Returns:
        int: Validated maximum results bounded by `MAX_K`.
    """

    if value is not None and value > 0:
        return min(value, MAX_K)
    return default

# -------------------------------------------------------------- end _validate_k()

# __________________________________________________________________________
# Public API
#
# -------------------------------------------------------------- get_semantic_k()
def get_semantic_k(max_results_semantic: Optional[int]) -> int:
    """Return the semantic search result limit with validation.

    Args:
        max_results_semantic: Caller-provided maximum semantic results.

    Returns:
        int: Validated semantic results limit.
    """

    return _validate_k(max_results_semantic, SEMANTIC_DEFAULT_K)

# -------------------------------------------------------------- end get_semantic_k()

# -------------------------------------------------------------- get_traversal_k()
def get_traversal_k(max_results_traversal: Optional[int]) -> int:
    """Return the traversal search result limit with validation.

    Args:
        max_results_traversal: Caller-provided maximum traversal results.

    Returns:
        int: Validated traversal results limit.
    """

    return _validate_k(max_results_traversal, TRAVERSAL_DEFAULT_K)

# -------------------------------------------------------------- end get_traversal_k()

# __________________________________________________________________________
# Module Exports
#
__all__ = [
    "get_semantic_k",
    "get_traversal_k",
]

# __________________________________________________________________________
# End of File