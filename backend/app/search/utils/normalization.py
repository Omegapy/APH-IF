# -------------------------------------------------------------------------
# File: normalization.py
# Author: Alexander Ricciardi
# Date: 2025-09-15
# [File Path] backend/app/search/utils/normalization.py
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
#   Normalization helpers that standardize engine metadata, confidence values, and
#   structured payloads for semantic, traversal, and fusion results used by the
#   backend tools and API layers.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Function: normalize_confidence_metadata
# - Function: normalize_engine_metadata
# - Function: normalize_semantic_result
# - Function: normalize_traversal_result
# - Function: normalize_fusion_result
# - Function: extract_citations_from_content
# - Function: create_engine_summary
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: typing (type hints)
# - Third-Party: (none)
# - Local Project Modules: models.structured_responses, core.config.settings
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# These helpers are consumed by search tools and API endpoints to return consistent
# JSON responses and metadata across engines, enabling stable client contracts.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Metadata normalization utilities for APH-IF backend.

Provides helpers that standardize confidence, engine metadata, and payload shapes across
semantic, traversal, and fusion paths so API responses remain consistent regardless of
which engines contributed to a result.
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports

from typing import Any, Dict, List, Optional
from ...models.structured_responses import (
    EngineMetadata, 
    ConfidenceMetadata,
    SemanticResultPayload,
    TraversalResultPayload, 
    FusionResultPayload
)
from ...core.config import settings


# __________________________________________________________________________
# Standalone Function Definitions
#
# ______________________
# Utility Functions
#

# --------------------------------------------------------------------------------- normalize_confidence_metadata()
def normalize_confidence_metadata(
    original_confidence: float,
    cap_value: float,
    search_type: str = "unknown",
) -> ConfidenceMetadata:
    """Normalize confidence values and record capping.

    Ensures a provided confidence value is clamped to the range [0.0, 1.0] and optionally capped
    to a configured ceiling. Returns a structured object that captures original, capped, and
    whether capping occurred, which downstream callers can surface in metadata.

    Args:
        original_confidence: Raw confidence value from the engine.
        cap_value: Maximum allowed confidence (inclusive).
        search_type: Optional label for the calling engine (semantic, traversal, fusion).

    Returns:
        ConfidenceMetadata: Structured confidence information including capping details.
    """
    # Ensure original confidence is in valid range
    original_confidence = max(0.0, min(1.0, original_confidence))
    
    # Apply cap
    capped_confidence = min(original_confidence, cap_value)
    was_capped = capped_confidence < original_confidence
    
    return ConfidenceMetadata(
        original=original_confidence,
        capped=capped_confidence,
        cap_value=cap_value,
        was_capped=was_capped
    )
# --------------------------------------------------------------------------------- end normalize_confidence_metadata()

# --------------------------------------------------------------------------------- normalize_engine_metadata()
def normalize_engine_metadata(
    search_method: str,
    response_time_ms: int,
    raw_metadata: Dict[str, Any],
) -> EngineMetadata:
    """Normalize engine metadata for consistent keys.

    Extracts common LLM execution details and validation info from engine-specific metadata
    payloads and maps them into a shared schema used by API responses.

    Args:
        search_method: Canonical engine identifier (e.g., "semantic_vector_search").
        response_time_ms: End-to-end time spent by the engine, in milliseconds.
        raw_metadata: Unstructured metadata as returned by the engine.

    Returns:
        EngineMetadata: Normalized engine metadata with consistent key names.
    """
    # Extract LLM usage information
    tokens_used = raw_metadata.get("tokens_used") or raw_metadata.get("llm_tokens_used")
    model_used = raw_metadata.get("model_used") or raw_metadata.get("llm_model_used")
    
    # Extract validation information
    validation_issues = raw_metadata.get("validation_issues")
    fixes_applied = raw_metadata.get("fixes_applied")
    
    # Convert to list if needed
    if validation_issues and not isinstance(validation_issues, list):
        validation_issues = [str(validation_issues)]
    if fixes_applied and not isinstance(fixes_applied, list):
        fixes_applied = [str(fixes_applied)]
    
    return EngineMetadata(
        search_method=search_method,
        tokens_used=tokens_used,
        model_used=model_used,
        validation_issues=validation_issues,
        fixes_applied=fixes_applied,
        response_time_ms=response_time_ms,
    )
# --------------------------------------------------------------------------------- end normalize_engine_metadata()

# --------------------------------------------------------------------------------- normalize_semantic_result()
def normalize_semantic_result(
    search_result: Dict[str, Any],
    response_time_ms: int,
) -> SemanticResultPayload:
    """Normalize semantic search results into the API schema.

    Args:
        search_result: Raw result dict from the semantic search engine.
        response_time_ms: Time spent generating the result, in milliseconds.

    Returns:
        SemanticResultPayload: Structured semantic payload including confidence, engine, and
        extracted citations/entities.
    """
    # Extract basic result data
    content = search_result.get("answer", "No semantic results found")
    original_confidence = search_result.get("confidence", 0.5)
    sources = search_result.get("sources", [])
    entities = search_result.get("entities_found", [])
    citations = extract_citations_from_content(content)
    
    # Normalize confidence
    confidence_meta = normalize_confidence_metadata(
        original_confidence,
        settings.validated_semantic_confidence_cap,
        "semantic"
    )
    
    # Normalize engine metadata
    engine_meta = normalize_engine_metadata(
        search_method="semantic_vector_search",
        response_time_ms=response_time_ms,
        raw_metadata=search_result.get("metadata", {})
    )
    
    # Extract semantic-specific metadata
    semantic_metadata = {
        "search_type": "semantic_vector_search",
        "k": search_result.get("k", 10),
        "num_sources": search_result.get("num_sources", len(sources)),
        "search_time_ms": search_result.get("search_time_ms", response_time_ms),
        "engine_metadata": search_result.get("metadata", {}),
        "parallel_execution": search_result.get("parallel_execution", False)
    }
    
    return SemanticResultPayload(
        content=content,
        confidence=confidence_meta,
        sources=sources,
        entities=entities,
        citations=citations,
        engine=engine_meta,
        metadata=semantic_metadata
    )
# --------------------------------------------------------------------------------- end normalize_semantic_result()

# --------------------------------------------------------------------------------- normalize_traversal_result()
def normalize_traversal_result(
    search_result: Dict[str, Any],
    response_time_ms: int
) -> TraversalResultPayload:
    """
    Normalize traversal search result to structured format.
    
    Args:
        search_result: Raw traversal search result
        response_time_ms: Search execution time
        
    Returns:
        TraversalResultPayload with normalized structure
    """
    # Extract basic result data
    content = search_result.get("answer", "No traversal results found")
    original_confidence = search_result.get("confidence", 0.0)
    cypher_query = search_result.get("cypher_query", "")
    entities = []
    citations = extract_citations_from_content(content)
    
    # Extract entities from metadata if available
    metadata = search_result.get("metadata", {})
    if metadata.get("entities_found"):
        entities = metadata["entities_found"]
    
    # Normalize confidence
    confidence_meta = normalize_confidence_metadata(
        original_confidence,
        settings.validated_traversal_confidence_cap,
        "traversal"
    )
    
    # Determine search method from metadata
    search_method = metadata.get("search_method", "unknown")
    if not search_method or search_method == "unknown":
        # Try to infer from other metadata
        if metadata.get("llm_tokens_used") or metadata.get("tokens_used"):
            search_method = "llm_structural_cypher"
        else:
            search_method = "langchain_cypher_chain"
    
    # Normalize engine metadata
    engine_meta = normalize_engine_metadata(
        search_method=search_method,
        response_time_ms=response_time_ms,
        raw_metadata=metadata
    )
    
    # Extract traversal-specific metadata
    traversal_metadata = {
        "search_type": "graph_traversal_search",
        "search_method": search_method,
        "max_results": metadata.get("max_results", 50),
        "cypher_query": cypher_query,
        "response_time_ms": search_result.get("response_time_ms", response_time_ms),
        "confidence_capped": confidence_meta.was_capped,
        "original_confidence": original_confidence,
        "parallel_execution": metadata.get("parallel_execution", False)
    }
    
    # Add LLM-specific metadata if available
    if search_method == "llm_structural_cypher":
        for key in ["generation_time_ms", "execution_time_ms", "validation_issues", "fixes_applied"]:
            if key in metadata:
                traversal_metadata[f"llm_{key}"] = metadata[key]
    
    return TraversalResultPayload(
        content=content,
        confidence=confidence_meta,
        cypher_query=cypher_query,
        execution_results=[],  # TODO: Extract if available
        entities=entities,
        citations=citations,
        engine=engine_meta,
        metadata=traversal_metadata
    )
# --------------------------------------------------------------------------------- end normalize_traversal_result()

# --------------------------------------------------------------------------------- normalize_fusion_result()
def normalize_fusion_result(
    fusion_result: Any,
    processing_time_ms: int,
) -> FusionResultPayload:
    """Normalize fusion results into the API schema.

    Handles both object and dictionary representations of fusion results, producing a stable
    payload that includes confidence, strategy, contributions, and preserved citations.

    Args:
        fusion_result: Result object or dict produced by the fusion engine.
        processing_time_ms: Total time spent in fusion processing, in milliseconds.

    Returns:
        FusionResultPayload: Structured fusion payload suitable for API responses.
    """
    # Handle both object and dict formats
    if hasattr(fusion_result, "fused_content"):
        # Object format
        content = fusion_result.fused_content
        original_confidence = fusion_result.final_confidence
        vector_contribution = fusion_result.vector_contribution
        graph_contribution = fusion_result.graph_contribution
        fusion_strategy = fusion_result.fusion_strategy
        complementarity_score = fusion_result.complementarity_score
        citations_preserved = fusion_result.citations_preserved
        sources_combined = fusion_result.sources_combined
        entities_combined = fusion_result.entities_combined
        metadata = getattr(fusion_result, "metadata", {})
    else:
        # Dict format
        content = fusion_result.get("fused_content", "")
        original_confidence = fusion_result.get("final_confidence", 0.0)
        vector_contribution = fusion_result.get("vector_contribution", 0.5)
        graph_contribution = fusion_result.get("graph_contribution", 0.5)
        fusion_strategy = fusion_result.get("fusion_strategy", "unknown")
        complementarity_score = fusion_result.get("complementarity_score", 0.0)
        citations_preserved = fusion_result.get("citations_preserved", [])
        sources_combined = fusion_result.get("sources_combined", [])
        entities_combined = fusion_result.get("entities_combined", [])
        metadata = fusion_result.get("metadata", {})
    
    # Normalize confidence
    confidence_meta = normalize_confidence_metadata(
        original_confidence,
        settings.validated_fusion_confidence_cap,
        "fusion"
    )
    
    # Create fusion-specific metadata
    fusion_metadata = {
        "fusion_strategy": fusion_strategy,
        "citation_accuracy": getattr(fusion_result, "citation_accuracy", 1.0),
        "domain_adaptation": getattr(fusion_result, "domain_adaptation", "unknown"),
        "processing_time_ms": processing_time_ms,
    }
    fusion_metadata.update(metadata)
    
    return FusionResultPayload(
        content=content,
        confidence=confidence_meta,
        vector_contribution=vector_contribution,
        graph_contribution=graph_contribution,
        fusion_strategy=fusion_strategy,
        complementarity_score=complementarity_score,
        citations_preserved=citations_preserved,
        sources_combined=sources_combined,
        entities_combined=entities_combined,
        processing_time_ms=processing_time_ms,
        metadata=fusion_metadata,
    )

# --------------------------------------------------------------------------------- end normalize_fusion_result()

# --------------------------------------------------------------------------------- extract_citations_from_content()
def extract_citations_from_content(content: str) -> List[str]:
    """Extract inline citation markers from content.

    Finds numbered citations like "[1]" and simple domain‑specific markers (e.g., "§57.4361(a)",
    "Part 75.1714"). This is a best‑effort extractor used to enrich metadata and does not attempt
    full normalization.

    Args:
        content: Free‑form text content to scan for citations.

    Returns:
        A de‑duplicated list of citation strings discovered in the content.
    """
    import re
    
    citations = []
    
    try:
        # Extract numbered citations [1], [2], etc.
        numbered_citations = re.findall(r"\[(\d+)\]", content)
        citations.extend([f"[{num}]" for num in numbered_citations])
        
        # Extract regulatory citations like §57.4361(a)
        regulatory_citations = re.findall(r"§[\d.]+(?:\([a-z]\))?", content)
        citations.extend(regulatory_citations)
        
        # Extract Part citations
        part_citations = re.findall(r"Part\s+[\d.]+", content)
        citations.extend(part_citations)
        
        return list(set(citations))  # Remove duplicates
        
    except Exception:
        return []

# --------------------------------------------------------------------------------- end extract_citations_from_content()

# --------------------------------------------------------------------------------- create_engine_summary()
def create_engine_summary(
    semantic_result: Optional[SemanticResultPayload] = None,
    traversal_result: Optional[TraversalResultPayload] = None,
    fusion_result: Optional[FusionResultPayload] = None,
) -> Dict[str, Any]:
    """Aggregate a compact summary of engine activity.

    Collates engine usage, token totals, models used, and basic validation counts across the
    semantic, traversal, and fusion contributors to a response. Intended for lightweight
    diagnostics and UI metadata.

    Args:
        semantic_result: Optional structured semantic payload.
        traversal_result: Optional structured traversal payload.
        fusion_result: Optional structured fusion payload.

    Returns:
        dict: Summary including engines used, total_tokens, models_used, and validation counts.
    """
    summary = {
        "engines_used": [],
        "total_tokens": 0,
        "models_used": set(),
        "validation_issues_count": 0,
        "fixes_applied_count": 0,
    }
    
    # Process semantic engine data
    if semantic_result and semantic_result.engine:
        summary["engines_used"].append(semantic_result.engine.search_method)
        if semantic_result.engine.tokens_used:
            summary["total_tokens"] += semantic_result.engine.tokens_used
        if semantic_result.engine.model_used:
            summary["models_used"].add(semantic_result.engine.model_used)
        if semantic_result.engine.validation_issues:
            summary["validation_issues_count"] += len(semantic_result.engine.validation_issues)
        if semantic_result.engine.fixes_applied:
            summary["fixes_applied_count"] += len(semantic_result.engine.fixes_applied)
    
    # Process traversal engine data  
    if traversal_result and traversal_result.engine:
        summary["engines_used"].append(traversal_result.engine.search_method)
        if traversal_result.engine.tokens_used:
            summary["total_tokens"] += traversal_result.engine.tokens_used
        if traversal_result.engine.model_used:
            summary["models_used"].add(traversal_result.engine.model_used)
        if traversal_result.engine.validation_issues:
            summary["validation_issues_count"] += len(traversal_result.engine.validation_issues)
        if traversal_result.engine.fixes_applied:
            summary["fixes_applied_count"] += len(traversal_result.engine.fixes_applied)
    
    # Convert set to list for JSON serialization
    summary["models_used"] = list(summary["models_used"])
    
    # Add fusion information
    if fusion_result:
        summary["fusion_used"] = True
        summary["fusion_strategy"] = fusion_result.fusion_strategy
        summary["complementarity_score"] = fusion_result.complementarity_score
    else:
        summary["fusion_used"] = False

    return summary

# --------------------------------------------------------------------------------- end create_engine_summary()

# __________________________________________________________________________
# End of File
#
