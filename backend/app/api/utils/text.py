# -------------------------------------------------------------------------
# File: text.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/app/api/utils/text.py
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
#   Provides text normalization utilities for API responses, including
#   heuristics for identifying unknown responses and generating timing
#   recommendations from monitoring data.
# -------------------------------------------------------------------------
#
# --- Module Contents Overview ---
# - Constant: _UNKNOWN_PATTERNS
# - Function: is_unknown_text
# - Function: generate_timing_recommendations
# - Constant: __all__ (module exports)
# -------------------------------------------------------------------------
#
# --- Dependencies / Imports ---
# - Standard Library: typing (Any, Dict, Iterable, List)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
#
# --- Usage / Integration ---
# Imported by routers and utils to evaluate fallback responses and provide
# human-readable timing guidance for operational tooling.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""Text utilities supporting response normalization and timing analytics."""

# __________________________________________________________________________
# Imports

from __future__ import annotations

from typing import Any, Dict, Iterable, List

# __________________________________________________________________________
# Constants
#
_UNKNOWN_PATTERNS = {
    "i don't know",
    "no documents or sources",
    "no relevant information",
    "no matching",
}


# __________________________________________________________________________
# Public API
#
# -------------------------------------------------------------- is_unknown_text()
def is_unknown_text(text: str) -> bool:
    """Return True when text represents an unknown or empty response.

    Args:
        text: Response text to evaluate.

    Returns:
        bool: True when the text matches an unknown-response heuristic.
    """

    if not text:
        return True
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in _UNKNOWN_PATTERNS)


# -------------------------------------------------------------- end is_unknown_text()

# -------------------------------------------------------------- generate_timing_recommendations()
def generate_timing_recommendations(
    operation_analysis: Dict[str, Any],
    bottlenecks: Iterable[Dict[str, Any]],
    db_stats: Dict[str, Any],
) -> List[str]:
    """Build human readable timing recommendations from monitoring data.

    Args:
        operation_analysis: Aggregated per-operation statistics.
        bottlenecks: Collection of operations identified as bottlenecks.
        db_stats: Database-related timing statistics.

    Returns:
        list[str]: Ordered list of recommendation messages.
    """

    recommendations: List[str] = []

    if db_stats.get("avg_total_time_ms", 0) > 500:
        recommendations.append(
            "Database queries are slow - consider query optimization or connection pooling"
        )

    if db_stats.get("connection_pool", {}).get("avg_acquisition_time_ms", 0) > 100:
        recommendations.append(
            "Connection acquisition is slow - consider increasing connection pool size"
        )

    bottlenecks_list = list(bottlenecks)
    if bottlenecks_list:
        slowest_op = bottlenecks_list[0]
        recommendations.append(
            f"'{slowest_op['operation']}' is the slowest operation at "
            f"{slowest_op['avg_time_ms']:.1f}ms average"
        )

    for op_name, stats in operation_analysis.items():
        if stats.get("success_rate", 100) < 90 and stats.get("count", 0) > 10:
            recommendations.append(
                (
                    f"'{op_name}' has low success rate "
                    f"({stats['success_rate']:.1f}%) - investigate error patterns"
                )
            )

    if not recommendations:
        recommendations.append("Overall timing performance is within acceptable ranges")

    return recommendations

# -------------------------------------------------------------- end generate_timing_recommendations()

# __________________________________________________________________________
# Module Exports
#
__all__ = [
    "generate_timing_recommendations",
    "is_unknown_text",
]

# __________________________________________________________________________
# End of File