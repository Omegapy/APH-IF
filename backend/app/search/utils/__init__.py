"""
Search utilities package for APH-IF Backend.

Contains helper functions for metadata normalization, response formatting,
and common search operations.
"""

from .normalization import (
    create_engine_summary,
    extract_citations_from_content,
    normalize_confidence_metadata,
    normalize_engine_metadata,
    normalize_fusion_result,
    normalize_semantic_result,
    normalize_traversal_result,
)

__all__ = [
    "normalize_confidence_metadata",
    "normalize_engine_metadata",
    "normalize_semantic_result",
    "normalize_traversal_result",
    "normalize_fusion_result",
    "extract_citations_from_content",
    "create_engine_summary"
]