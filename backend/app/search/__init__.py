"""
Search and retrieval module for APH-IF backend.

This module provides parallel hybrid search capabilities including semantic search,
graph traversal search, and intelligent context fusion.
"""

from .parallel_hybrid import (
    RetrievalResult,
    ParallelRetrievalResponse,
    FusionResult,
    ParallelRetrievalEngine,
    get_parallel_engine,
    get_process_parallel_engine,
    create_error_result,
    validate_parallel_response
)
from .context_fusion import (
    FusionStrategy,
    IntelligentFusionEngine,
    get_fusion_engine,
    extract_source_references_sync,
    combine_source_references_sync,
    renumber_fused_citations_sync,
    create_combined_references_section_sync
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