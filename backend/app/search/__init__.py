"""
Search and retrieval module for APH-IF backend.

This module provides parallel hybrid search capabilities including semantic search,
graph traversal search, and intelligent context fusion.
"""

from .context_fusion import (
    FusionStrategy,
    IntelligentFusionEngine,
    combine_source_references_sync,
    create_combined_references_section_sync,
    extract_source_references_sync,
    get_fusion_engine,
    renumber_fused_citations_sync,
)
from .parallel_hybrid import (
    FusionResult,
    ParallelRetrievalEngine,
    ParallelRetrievalResponse,
    RetrievalResult,
    create_error_result,
    get_parallel_engine,
    get_process_parallel_engine,
    validate_parallel_response,
)

__all__ = [
    # Parallel hybrid search
    "RetrievalResult",
    "ParallelRetrievalResponse",
    "FusionResult",
    "ParallelRetrievalEngine",
    "get_parallel_engine",
    "get_process_parallel_engine",
    "create_error_result",
    "validate_parallel_response",
    # Context fusion
    "FusionStrategy",
    "IntelligentFusionEngine",
    "get_fusion_engine",
    "extract_source_references_sync",
    "combine_source_references_sync",
    "renumber_fused_citations_sync",
    "create_combined_references_section_sync"
]