# -------------------------------------------------------------------------
# File: structured_responses.py
# Author: Alexander Ricciardi
# Date: 2025-09-16
# [File Path] backend/app/models/structured_responses.py
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
#   Pydantic models that provide consistent, typed JSON envelopes across search modes (vector,
#   graph, hybrid), including normalized engine metadata and confidence details used by the
#   backend service and clients.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Class: EngineMetadata
# - Class: ConfidenceMetadata
# - Class: SemanticResultPayload
# - Class: TraversalResultPayload
# - Class: FusionResultPayload
# - Class: StructuredQueryResponse
# - Class: StructuredHealthResponse
# - Class: PerformanceDashboardResponse
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Third-Party: pydantic (BaseModel, Field)
# - Standard Library: typing (hints)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# These models shape the backend API responses and are consumed by the frontend and clients.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Structured Response Models for APH-IF Backend API.

Pydantic models that provide consistent, typed JSON envelopes across search modes (vector,
graph, hybrid), including normalized engine metadata and confidence details used by the
backend service and clients.
"""

# __________________________________________________________________________
# Imports

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ____________________________________________________________________________
# Class Definitions


# ------------------------------------------------------------------------- class EngineMetadata
class EngineMetadata(BaseModel):
    """Normalized engine metadata for all search types.

    Attributes:
        search_method: Search method used (e.g., llm_structural_cypher | langchain_cypher_chain |
            semantic_vector_search).
        tokens_used: LLM tokens consumed (if available).
        model_used: Name of the LLM model used (if available).
        validation_issues: Any validation issues detected by the engine.
        fixes_applied: Automatic fixes applied by the engine.
        response_time_ms: Engine response time in milliseconds.
    """
    search_method: str = Field(
        description="Search method: llm_structural_cypher | langchain_cypher_chain | semantic_vector_search",
    )
    tokens_used: Optional[int] = Field(default=None, description="LLM tokens consumed")
    model_used: Optional[str] = Field(default=None, description="LLM model used")
    validation_issues: Optional[List[str]] = Field(default=None, description="Validation issues found")
    fixes_applied: Optional[List[str]] = Field(default=None, description="Automatic fixes applied")
    response_time_ms: int = Field(description="Engine response time in milliseconds")
# ------------------------------------------------------------------------- end class EngineMetadata


# ------------------------------------------------------------------------- class ConfidenceMetadata
class ConfidenceMetadata(BaseModel):
    """Normalized confidence scoring metadata.

    Attributes:
        original: Original confidence score before capping.
        capped: Final confidence score after capping.
        cap_value: Configuration cap value applied.
        was_capped: Whether capping reduced the score.
    """
    original: float = Field(description="Original confidence score before capping", ge=0.0, le=1.0)
    capped: float = Field(description="Final confidence score after capping", ge=0.0, le=1.0)
    cap_value: float = Field(description="Configuration cap value applied", ge=0.0, le=1.0)
    was_capped: bool = Field(description="Whether confidence was reduced by capping")
# ------------------------------------------------------------------------- end class ConfidenceMetadata


# ------------------------------------------------------------------------- class SemanticResultPayload
class SemanticResultPayload(BaseModel):
    """Semantic (vector) search result payload.

    Attributes:
        content: Semantic search response content.
        confidence: Confidence scoring details for the semantic result.
        sources: Source documents and chunks supporting the content.
        entities: Extracted entities.
        citations: Extracted citations.
        engine: Engine execution metadata.
        metadata: Additional semantic-specific metadata.
    """
    content: str = Field(description="Semantic search response content")
    confidence: ConfidenceMetadata = Field(description="Confidence scoring details")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents and chunks")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    citations: List[str] = Field(default_factory=list, description="Extracted citations")
    engine: EngineMetadata = Field(description="Engine execution metadata")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional semantic-specific metadata")
# ------------------------------------------------------------------------- end class SemanticResultPayload


# ------------------------------------------------------------------------- class TraversalResultPayload
class TraversalResultPayload(BaseModel):
    """Traversal (graph) search result payload.

    Attributes:
        content: Traversal search response content.
        confidence: Confidence scoring details for traversal.
        cypher_query: Generated Cypher query, if applicable.
        execution_results: Raw Cypher execution results.
        entities: Graph entities found.
        citations: Extracted citations.
        engine: Engine execution metadata.
        metadata: Additional traversal-specific metadata.
    """
    content: str = Field(description="Traversal search response content")
    confidence: ConfidenceMetadata = Field(description="Confidence scoring details")
    cypher_query: Optional[str] = Field(default=None, description="Generated Cypher query")
    execution_results: List[Dict[str, Any]] = Field(default_factory=list, description="Raw Cypher execution results")
    entities: List[str] = Field(default_factory=list, description="Graph entities found")
    citations: List[str] = Field(default_factory=list, description="Extracted citations")
    engine: EngineMetadata = Field(description="Engine execution metadata")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional traversal-specific metadata")
# ------------------------------------------------------------------------- end class TraversalResultPayload


# ------------------------------------------------------------------------- class FusionResultPayload
class FusionResultPayload(BaseModel):
    """Fusion result payload for hybrid searches.

    Attributes:
        content: Fused response content.
        confidence: Final fused confidence details.
        vector_contribution: Semantic search contribution weight.
        graph_contribution: Traversal search contribution weight.
        fusion_strategy: Strategy used during fusion.
        complementarity_score: How well sources complemented.
        citations_preserved: Citations preserved in the fused content.
        sources_combined: All sources from both searches.
        entities_combined: All entities from both searches.
        processing_time_ms: Fusion processing time in milliseconds.
        metadata: Additional fusion-specific metadata.
    """
    content: str = Field(description="Fused response content")
    confidence: ConfidenceMetadata = Field(description="Final fused confidence details")
    vector_contribution: float = Field(description="Semantic search contribution weight", ge=0.0, le=1.0)
    graph_contribution: float = Field(description="Traversal search contribution weight", ge=0.0, le=1.0)
    fusion_strategy: str = Field(description="Strategy used for fusion")
    complementarity_score: float = Field(description="How well sources complemented", ge=0.0, le=1.0)
    citations_preserved: List[str] = Field(default_factory=list, description="Citations preserved in fusion")
    sources_combined: List[Dict[str, Any]] = Field(default_factory=list, description="All sources from both searches")
    entities_combined: List[str] = Field(default_factory=list, description="All entities from both searches")
    processing_time_ms: int = Field(description="Fusion processing time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional fusion-specific metadata")
# ------------------------------------------------------------------------- end class FusionResultPayload


# ------------------------------------------------------------------------- class StructuredQueryResponse
class StructuredQueryResponse(BaseModel):
    """Structured response envelope for all query types.

    Attributes:
        query: Original user query.
        search_type: Requested search type.
        session_id: Session identifier.
        success: Whether the query was successful.
        processing_time_ms: Total processing time in milliseconds.
        timestamp: Response timestamp.
        semantic_result: Semantic search results for vector/hybrid.
        traversal_result: Traversal search results for graph/hybrid.
        fusion_result: Fusion results for hybrid.
        engine_metadata: System-level metadata about engines and performance.
        error: Error message if the query failed.
        warnings: Non-fatal warnings encountered during processing.
    """
    # Request metadata
    query: str = Field(description="Original user query")
    search_type: str = Field(description="Search type requested: vector | graph | graph_llm_structural | hybrid")
    session_id: str = Field(description="Session identifier")
    
    # Execution metadata
    success: bool = Field(description="Whether the query was successful")
    processing_time_ms: int = Field(description="Total processing time in milliseconds")
    timestamp: str = Field(description="Response timestamp")
    
    # Search results (conditional based on search_type)
    semantic_result: Optional[SemanticResultPayload] = Field(
        default=None,
        description="Semantic search results (present for vector/hybrid modes)",
    )
    traversal_result: Optional[TraversalResultPayload] = Field(
        default=None,
        description="Traversal search results (present for graph/hybrid modes)",
    )
    fusion_result: Optional[FusionResultPayload] = Field(
        default=None,
        description="Fusion results (present for hybrid mode when both searches succeed)",
    )
    
    # System metadata
    engine_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="System-level metadata about engines used and performance",
    )
    
    # Error information
    error: Optional[str] = Field(default=None, description="Error message if query failed")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings during processing")
# ------------------------------------------------------------------------- end class StructuredQueryResponse


# ------------------------------------------------------------------------- class StructuredHealthResponse
class StructuredHealthResponse(BaseModel):
    """Enhanced health response with traversal engine details.

    Attributes:
        status: Overall system health status.
        timestamp: Health check timestamp.
        version: API version string.
        environment: Active environment mode.
        components: Component health details map.
        traversal_engine: Traversal engine health and metrics, if available.
        production_safety: Production safety check results.
    """
    status: str = Field(description="Overall system health status")
    timestamp: str = Field(description="Health check timestamp")
    version: str = Field(description="API version")
    environment: str = Field(description="Environment mode")
    
    components: Dict[str, Any] = Field(description="Component health details")
    traversal_engine: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Traversal engine health and metrics",
    )
    
    # Safety information
    production_safety: Dict[str, Any] = Field(
        default_factory=dict,
        description="Production safety check results",
    )
# ------------------------------------------------------------------------- end class StructuredHealthResponse

# ------------------------------------------------------------------------- class PerformanceDashboardResponse
class PerformanceDashboardResponse(BaseModel):
    """Enhanced performance dashboard with LLM Structural metrics.

    Attributes:
        timestamp: Dashboard generation timestamp.
        system_metrics: Aggregated system performance metrics.
        query_metrics: Aggregated query performance metrics.
        llm_structural_metrics: LLM Structural engine metrics and performance.
        alerts: Active alerts and thresholds exceeded.
    """
    timestamp: str = Field(description="Dashboard generation timestamp")
    
    # Existing metrics
    system_metrics: Dict[str, Any] = Field(description="System performance metrics")
    query_metrics: Dict[str, Any] = Field(description="Query performance metrics")
    
    # New LLM Structural metrics  
    llm_structural_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="LLM Structural engine metrics and performance",
    )
    
    # Alert status
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active alerts and thresholds exceeded",
    )
# __________________________________________________________________________
# End of File
#
# ------------------------------------------------------------------------- end class PerformanceDashboardResponse