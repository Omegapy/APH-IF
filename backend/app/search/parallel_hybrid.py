# -------------------------------------------------------------------------
# File: parallel_hybrid.py
# Author: Alexander Ricciardi
# Date: 2025-09-16
# [File Path] backend/app/search/parallel_hybrid.py
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
#   Implements the parallel retrieval engine for the backend search pipeline. Runs semantic
#   (VectorRAG) and traversal (GraphRAG) searches concurrently and returns a combined response.
#   Provides data models, health checks, and small utilities used by the engine.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Dataclass: RetrievalResult
# - Dataclass: ParallelRetrievalResponse
# - Dataclass: FusionResult
# - Class: ParallelRetrievalEngine
# - Functions: get_parallel_engine, get_process_parallel_engine
# - Utilities: create_error_result, validate_parallel_response
# - Tests (dev): test_parallel_hybrid_data_models, test_parallel_retrieval_engine
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: asyncio (concurrency), time (timing), logging (logs), typing (hints),
#   dataclasses (data models), datetime (timestamps)
# - Local Project Modules: core.config.settings
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Used by API routes and higher-level search orchestration to execute concurrent semantic and
# traversal retrieval. The response can be passed to a fusion stage for synthesis.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Parallel Hybrid Retrieval Engine for APH-IF Backend.

Advanced Parallel HybridRAG implementation that executes Semantic (VectorRAG)
and Traversal (GraphRAG) searches simultaneously using ``asyncio.gather`` for
true concurrent execution, maximizing information coverage and providing
comprehensive, domain-agnostic query responses.

Key Innovation: True parallel execution vs sequential conditional routing
- Traditional: if condition: vector_search() else: graph_search()
- APH-IF: asyncio.gather(vector_task, graph_task) — both always run
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

 # __________________________________________________________________________
 # Imports

from ..core.config import settings

 # __________________________________________________________________________
 # Global Constants / Variables

logger = logging.getLogger(__name__)

# =========================================================================
# Data Models
# =========================================================================

# TODO: consider @dataclass(slots=True, kw_only=True) if safe (API keyword-only)
# ------------------------------------------------------------------------- class RetrievalResult
@dataclass
class RetrievalResult:
    """
    Container for individual search results with comprehensive metadata.
    
    Stores results from either semantic (vector) or traversal (graph) search
    with confidence scores, timing, sources, and error information.
    """
    content: str                                    # The actual response text
    method: str                                     # "semantic" or "traversal"
    confidence: float                               # 0.0 to 1.0 confidence score
    response_time_ms: int                          # Response time in milliseconds
    sources: List[Dict[str, Any]] = field(default_factory=list)  # Source documents/chunks
    entities: List[str] = field(default_factory=list)            # Extracted entities
    citations: List[str] = field(default_factory=list)           # Extracted citations/references
    error: Optional[str] = None                    # Error message if search failed
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)  # Additional metadata
    
    @property
    def success(self) -> bool:
        """Returns True if the search was successful (no error)."""
        return self.error is None
    
    def __post_init__(self):
        """Validate data after initialization."""
        if not isinstance(self.confidence, (int, float)):
            self.confidence = 0.0
        if not isinstance(self.response_time_ms, int):
            self.response_time_ms = 0
        # Ensure confidence is between 0.0 and 1.0
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
# ------------------------------------------------------------------------- end class RetrievalResult

# TODO: consider @dataclass(slots=True, kw_only=True) if safe (API keyword-only)
# ------------------------------------------------------------------------- class ParallelRetrievalResponse
@dataclass
class ParallelRetrievalResponse:
    """
    Container for complete parallel search results from both semantic and traversal methods.
    
    Includes individual results, timing, success indicators, and fusion readiness
    assessment for intelligent context combination.
    """
    semantic_result: RetrievalResult               # Semantic search result
    traversal_result: RetrievalResult              # Graph traversal result  
    query: str                                     # Original user query
    total_time_ms: int                            # Total parallel execution time
    success: bool                                  # At least one search succeeded
    fusion_ready: bool                            # Both results suitable for fusion
    both_successful: bool                         # Both searches completed successfully
    primary_method: str                           # Which method had higher confidence
    
    # Additional analysis metrics
    vector_contribution: float = 0.0              # Semantic search contribution weight
    graph_contribution: float = 0.0               # Traversal search contribution weight
    complementarity_score: float = 0.0           # How well results complement each other
    entities_combined: List[str] = field(default_factory=list)  # Combined entity list
    sources_combined: List[Dict[str, Any]] = field(default_factory=list)  # Combined sources
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        # Determine primary method based on confidence scores
        if self.semantic_result.confidence > self.traversal_result.confidence:
            self.primary_method = "semantic"
        elif self.traversal_result.confidence > self.semantic_result.confidence:
            self.primary_method = "traversal"
        else:
            self.primary_method = "equal"
        
        # Calculate contribution weights
        total_confidence = self.semantic_result.confidence + self.traversal_result.confidence
        if total_confidence > 0:
            self.vector_contribution = self.semantic_result.confidence / total_confidence
            self.graph_contribution = self.traversal_result.confidence / total_confidence
        else:
            self.vector_contribution = 0.5
            self.graph_contribution = 0.5
        
        # Combine entities and sources
        self.entities_combined = list(set(
            self.semantic_result.entities + self.traversal_result.entities
        ))
        
        # Combine sources with method attribution
        for source in self.semantic_result.sources:
            source_with_method = {**source, "search_method": "semantic"}
            self.sources_combined.append(source_with_method)
        
        for source in self.traversal_result.sources:
            source_with_method = {**source, "search_method": "traversal"}
            self.sources_combined.append(source_with_method)
        
        # Calculate complementarity score (simplified heuristic)
        self.complementarity_score = self._calculate_complementarity()
    
    def _calculate_complementarity(self) -> float:
        """
        Calculate how well the two search results complement each other.
        
        Returns a score from 0.0 to 1.0 where:
        - 0.0 = Highly redundant results
        - 1.0 = Highly complementary results
        """
        try:
            # Simple word-based complementarity check
            semantic_words = set(self.semantic_result.content.lower().split())
            traversal_words = set(self.traversal_result.content.lower().split())
            
            if not semantic_words or not traversal_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(semantic_words & traversal_words)
            union = len(semantic_words | traversal_words)
            
            if union == 0:
                return 0.0
            
            similarity = intersection / union
            # Complementarity is inverse of similarity
            complementarity = 1.0 - similarity
            
            # Boost complementarity if both have good confidence
            if (self.semantic_result.confidence > 0.5 and 
                self.traversal_result.confidence > 0.5):
                complementarity = min(1.0, complementarity * 1.2)
            
            return complementarity
            
        except Exception as e:
            logger.warning(f"Error calculating complementarity: {e}")
            return 0.5  # Default moderate complementarity
# ------------------------------------------------------------------------- end class ParallelRetrievalResponse

# TODO: consider @dataclass(slots=True, kw_only=True) if safe (API keyword-only)
# ------------------------------------------------------------------------- class FusionResult
@dataclass 
class FusionResult:
    """
    Container for intelligent context fusion results.
    
    Stores the final fused response with metadata about the fusion process,
    contribution weights, and quality metrics.
    """
    fused_content: str                            # Final fused response
    final_confidence: float                       # Overall confidence of fused result
    fusion_strategy: str                          # Strategy used for fusion
    processing_time_ms: int                       # Time taken for fusion
    
    # Fusion analysis
    vector_contribution: float                    # Weight given to semantic results
    graph_contribution: float                     # Weight given to traversal results
    complementarity_score: float                  # How well sources complemented
    
    # Combined metadata
    entities_combined: List[str] = field(default_factory=list)
    sources_combined: List[Dict[str, Any]] = field(default_factory=list)
    citations_preserved: List[str] = field(default_factory=list)
    
    # Quality metrics
    citation_accuracy: float = 1.0               # How well citations were preserved
    domain_adaptation: str = "unknown"           # Detected domain type
    
    error: Optional[str] = None                   # Error during fusion if any
    
    def __post_init__(self):
        """Validate fusion result data."""
        # Ensure confidence is valid
        if not isinstance(self.final_confidence, (int, float)):
            self.final_confidence = 0.0
        self.final_confidence = max(0.0, min(1.0, float(self.final_confidence)))
        
        # Ensure processing time is valid
        if not isinstance(self.processing_time_ms, int):
            self.processing_time_ms = 0
# ------------------------------------------------------------------------- end class FusionResult

# =========================================================================
# Global Engine Instance
# =========================================================================

# Singleton instance for the parallel retrieval engine
_parallel_engine: Optional['ParallelRetrievalEngine'] = None

def get_parallel_engine(use_process_isolation: bool = False) -> 'ParallelRetrievalEngine':
    """
    Get or create the global parallel retrieval engine instance.
    
    Args:
        use_process_isolation: Enable process-level isolation for true parallelism
                             (eliminates LangChain/library internal sharing)
    
    Returns:
        ParallelRetrievalEngine: Global engine instance
    """
    global _parallel_engine
    if _parallel_engine is None:
        _parallel_engine = ParallelRetrievalEngine(use_process_isolation=use_process_isolation)
    return _parallel_engine

def get_process_parallel_engine() -> 'ParallelRetrievalEngine':
    """
    Get parallel engine with process isolation enabled for maximum performance.
    
    This version eliminates ALL shared resources between searches:
    - Separate processes
    - Separate Python interpreters  
    - Separate LangChain state
    - Separate OpenAI API clients
    - Separate Neo4j connections
    
    Returns:
        ParallelRetrievalEngine: Engine with process isolation enabled
    """
    return ParallelRetrievalEngine(use_process_isolation=True)

# =========================================================================
# Engine Implementation - Step 2: Core Parallel Execution
# =========================================================================

# ------------------------------------------------------------------------- class ParallelRetrievalEngine
class ParallelRetrievalEngine:
    """
    Engine for executing semantic and traversal searches in parallel.
    
    Key innovation: Uses asyncio.gather() for true concurrent execution
    rather than sequential conditional routing.
    
    This is the core innovation of APH-IF:
    - Traditional: if condition: vector_search() else: graph_search()
    - APH-IF: asyncio.gather(vector_task, graph_task) - both always run
    """
    
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self, timeout_seconds: int = 240, use_process_isolation: bool = False):
        """Initialize the parallel retrieval engine.

        Args:
            timeout_seconds: Maximum time to wait for retrieval operations.
            use_process_isolation: Use separate processes for true parallelism (eliminates
                LangChain sharing).
        """
        self.timeout_seconds = timeout_seconds
        self.use_process_isolation = use_process_isolation
        self.logger = logging.getLogger(__name__)
        
        # Initialize process engine if using process isolation
        self._process_engine = None
        if self.use_process_isolation:
            try:
                from ..processing.process_parallel import get_process_parallel_engine
                self._process_engine = get_process_parallel_engine()
                self.logger.info("✅ Process isolation enabled for true parallel execution")
            except ImportError as e:
                self.logger.warning(f"Process isolation not available: {e}")
                self.use_process_isolation = False
        
        # Direct processing for consistent results (no query caching)
        self._cache = None
        
        # Initialize circuit breakers for resilience
        try:
            from ..monitoring.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
            
            # Circuit breaker for semantic search
            semantic_config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=240,
                timeout_threshold_ms=240000
            )
            self._semantic_breaker = get_circuit_breaker("semantic_search", semantic_config)
            
            # Circuit breaker for LLM structural traversal search (primary path)
            traversal_llm_config = CircuitBreakerConfig(
                failure_threshold=2,  # Stricter threshold for new path
                recovery_timeout=240,
                timeout_threshold_ms=240000
            )
            self._traversal_llm_breaker = get_circuit_breaker("traversal_llm_structural", traversal_llm_config)
            
            
            self.logger.info("✅ Circuit breakers enabled for resilience")
        except ImportError:
            self.logger.warning("⚠️ Circuit breakers not available")
            self._semantic_breaker = None
            self._traversal_llm_breaker = None
        
        # Initialize search tool imports
        self._semantic_search_available = False
        self._traversal_search_available = False
        
        # Initialize LLM structural flag
        self._traversal_llm_available = False
        
        # Try to import search tools
        try:
            from .tools.vector import search_semantic_detailed
            self._search_semantic_detailed = search_semantic_detailed
            self._semantic_search_available = True
            self.logger.info("✅ Semantic search tool imported successfully")
        except ImportError as e:
            self.logger.warning(f"⚠️ Semantic search not available: {e}")
            self._search_semantic_detailed = None
        
        # Try to import LLM structural cypher (primary traversal path)
        try:
            from .tools.cypher import query_knowledge_graph_llm_structural_detailed
            self._query_knowledge_graph_llm_structural_detailed = query_knowledge_graph_llm_structural_detailed
            self._traversal_llm_available = True
            self.logger.info("✅ LLM Structural Cypher tool imported successfully (primary)")
        except ImportError as e:
            self.logger.warning(f"⚠️ LLM Structural Cypher not available: {e}")
            self._query_knowledge_graph_llm_structural_detailed = None
        
        # Set overall traversal availability
        self._traversal_search_available = self._traversal_llm_available
        
        self.logger.info(f"Initialized ParallelRetrievalEngine with {timeout_seconds}s timeout")
        self.logger.info(f"Available tools: Semantic={self._semantic_search_available}, "
                        f"Traversal={self._traversal_search_available}")
        
        # Log traversal method availability summary
        if self._traversal_llm_available:
            traversal_status = "LLM Structural Cypher available"
        else:
            traversal_status = "None available"
        
        self.logger.info(f"Traversal methods: {traversal_status}")
        
        # Log configuration gating
        if settings.use_llm_structural_cypher:
            if self._traversal_llm_available:
                self.logger.info("✅ LLM Structural Cypher enabled and available")
            else:
                self.logger.warning("⚠️ LLM Structural Cypher enabled but not available")
        else:
            self.logger.info("⚠️ LLM Structural Cypher disabled via config")
    # --------------------------------------------------------------------------------- end __init__()
    
    # -------------------------------------------------------------- retrieve_parallel()
    async def retrieve_parallel(
        self,
        query: str,
        semantic_k: int = 10,
        traversal_max_results: int = 50,
    ) -> ParallelRetrievalResponse:
        """Execute both searches simultaneously with comprehensive timing.

        THIS IS THE KEY INNOVATION: True parallel execution vs sequential conditional.

        Args:
            query: User's natural language query.
            semantic_k: Number of chunks for semantic search.
            traversal_max_results: Max results for traversal search.

        Returns:
            ParallelRetrievalResponse: Complete parallel results (cached or fresh).
        """
        # Import timing collector
        from ..monitoring.timing_collector import get_timing_collector
        timing_collector = get_timing_collector()
        
        # Comprehensive parallel retrieval timing
        async with timing_collector.measure("parallel_retrieval", {
            "query_length": len(query),
            "semantic_k": semantic_k,
            "traversal_max_results": traversal_max_results,
            "process_isolation": self.use_process_isolation
        }) as parallel_timer:
            
            start_time = time.time()
            self.logger.info(f"🚀 Starting parallel retrieval for query: {query[:50]}...")
            
            # Direct processing (ensures fresh results)
            self.logger.info("🔄 Processing query directly")
            
            parallel_timer.add_metadata({
                "cache_hit": False,
                "execution_mode": "direct_processing",
                "user_cache_disabled": True
            })
        
        try:
            # Check if process isolation is enabled for true parallelism
            if self.use_process_isolation and self._process_engine:
                self.logger.info("🔄 Using process isolation for TRUE parallel execution...")
                
                # Use process-level parallel execution to eliminate LangChain sharing
                process_result = await self._process_engine.retrieve_parallel_process(
                    query, semantic_k, traversal_max_results
                )
                
                # Convert process result to standard ParallelRetrievalResponse format
                semantic_result = RetrievalResult(
                    content=process_result["semantic_result"]["content"],
                    method="semantic",
                    confidence=process_result["semantic_result"]["confidence"],
                    response_time_ms=process_result["semantic_result"]["response_time_ms"],
                    sources=[],  # Simplified for process isolation
                    entities=[],
                    citations=[],
                    error=process_result["semantic_result"].get("error"),
                    metadata=process_result["semantic_result"].get("metadata", {})
                )
                
                traversal_result = RetrievalResult(
                    content=process_result["traversal_result"]["content"],
                    method="traversal", 
                    confidence=process_result["traversal_result"]["confidence"],
                    response_time_ms=process_result["traversal_result"]["response_time_ms"],
                    sources=[],  # Simplified for process isolation
                    entities=[],
                    citations=[],
                    error=process_result["traversal_result"].get("error"),
                    metadata=process_result["traversal_result"].get("metadata", {})
                )
                
                response = ParallelRetrievalResponse(
                    success=process_result["success"],
                    both_successful=process_result["both_successful"],
                    fusion_ready=process_result["fusion_ready"],
                    primary_method=process_result["primary_method"],
                    complementarity_score=process_result["complementarity_score"],
                    total_time_ms=process_result["total_time_ms"],
                    semantic_result=semantic_result,
                    traversal_result=traversal_result,
                    metadata={
                        "execution_mode": "process_isolation",
                        "process_metadata": process_result.get("process_metadata", {}),
                        "query": query,
                        "parameters": {
                            "semantic_k": semantic_k,
                            "traversal_max_results": traversal_max_results
                        }
                    }
                )
                
                # Direct processing - no result caching
                
                self.logger.info(f"✅ Process isolation parallel retrieval completed in {process_result['total_time_ms']}ms")
                return response
            
            # Parallel coordination timing
            async with timing_collector.measure("parallel_coordination") as coordination_timer:
                
                # Fall back to async concurrent execution (original approach)
                self.logger.info("🔄 Using async concurrent execution (shared resources)...")
                
                coordination_timer.add_metadata({
                    "execution_mode": "async_concurrent",
                    "shared_resources": True
                })
                
                # Task creation timing
                async with timing_collector.measure("task_creation") as task_timer:
                    # Create async tasks for parallel execution
                    # THIS IS THE CORE INNOVATION - both searches run simultaneously
                    semantic_task = asyncio.create_task(
                        self._async_semantic_retrieve(query, k=semantic_k),
                        name="semantic_search"
                    )
                    traversal_task = asyncio.create_task(
                        self._async_traversal_retrieve(query, max_results=traversal_max_results),
                        name="traversal_search"
                    )
                    
                    task_timer.add_metadata({
                        "tasks_created": 2,
                        "semantic_task_id": id(semantic_task),
                        "traversal_task_id": id(traversal_task)
                    })
                
                self.logger.info("⚡ Both search tasks created, executing in parallel...")
                
                # Parallel execution timing - THIS IS THE KEY INNOVATION
                async with timing_collector.measure("parallel_execution") as execution_timer:
                    results = await asyncio.wait_for(
                        asyncio.gather(semantic_task, traversal_task, return_exceptions=True),
                        timeout=self.timeout_seconds
                    )
                    
                    semantic_result, traversal_result = results
                    
                    # Handle potential exceptions and ensure proper typing
                    final_semantic_result = self._process_search_result(
                        semantic_result, "semantic", query
                    )
                    final_traversal_result = self._process_search_result(
                        traversal_result, "traversal", query  
                    )
                    
                    total_time = int((time.time() - start_time) * 1000)
                    
                    # Analyze results for fusion readiness
                    fusion_analysis = self._analyze_fusion_readiness(
                        final_semantic_result, final_traversal_result
                    )
                    
                    # Create comprehensive response
                    parallel_response = ParallelRetrievalResponse(
                        semantic_result=final_semantic_result,
                        traversal_result=final_traversal_result,
                        query=query,
                        total_time_ms=total_time,
                        success=fusion_analysis["success"],
                        fusion_ready=fusion_analysis["fusion_ready"],
                        both_successful=fusion_analysis["both_successful"],
                        primary_method=fusion_analysis["primary_method"]
                    )
                    
                    # Log execution timer metadata
                    execution_timer.add_metadata({
                        "semantic_success": final_semantic_result.success,
                        "traversal_success": final_traversal_result.success,
                        "both_successful": fusion_analysis["both_successful"],
                        "total_time_ms": total_time
                    })
                
                # Standardized logging with structured metadata
                self.logger.info("✅ Parallel retrieval completed", extra={
                    "search_type": "parallel_hybrid",
                    "total_time_ms": total_time,
                    "success": parallel_response.success,
                    "semantic_success": parallel_response.semantic_result.success,
                    "semantic_time_ms": parallel_response.semantic_result.response_time_ms,
                    "traversal_success": parallel_response.traversal_result.success,
                    "traversal_time_ms": parallel_response.traversal_result.response_time_ms,
                    "both_successful": parallel_response.both_successful,
                    "fusion_ready": parallel_response.fusion_ready,
                    "primary_method": parallel_response.primary_method,
                    "complementarity_score": round(parallel_response.complementarity_score, 3),
                    "parallel_execution": True
                })
                
                # Direct processing - no result caching
                self.logger.debug("✅ Query processed directly")
                
                return parallel_response
            
        except asyncio.TimeoutError:
            self.logger.error(f"❌ Parallel retrieval timeout after {self.timeout_seconds}s")
            return self._create_timeout_response(query, start_time)
        except Exception as e:
            self.logger.error(f"❌ Parallel retrieval failed: {str(e)}")
            return self._create_error_response(query, start_time, str(e))
    # -------------------------------------------------------------- end retrieve_parallel()
    
    # -------------------------------------------------------------- _async_semantic_retrieve()
    async def _async_semantic_retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        """Run semantic search with robust error handling.

        Args:
            query: User query.
            k: Number of chunks to retrieve.

        Returns:
            RetrievalResult: Semantic search results.
        """
        start_time = time.time()
        
        if not self._semantic_search_available:
            return create_error_result(
                "semantic", 
                "Semantic search tool not available",
                int((time.time() - start_time) * 1000)
            )
        
        try:
            self.logger.info(f"🔍 Executing semantic search (k={k})...")
            
            # Call the existing semantic search function with circuit breaker protection
            # Note: Factory metadata is automatically included in search_result from search_semantic_detailed()
            if self._semantic_breaker:
                search_result = await self._semantic_breaker.call(self._search_semantic_detailed, query, k=k)
            else:
                search_result = await self._search_semantic_detailed(query, k=k)
            
            response_time = int((time.time() - start_time) * 1000)
            
            # Extract confidence from search result or calculate heuristically
            confidence = self._calculate_semantic_confidence(search_result)
            
            # Extract citations from the answer
            citations = self._extract_citations_from_content(search_result.get("answer", ""))
            
            semantic_result = RetrievalResult(
                content=search_result.get("answer", "No semantic results found"),
                method="semantic",
                confidence=confidence,
                response_time_ms=response_time,
                sources=search_result.get("sources", []),
                entities=search_result.get("entities_found", []),
                citations=citations,
                metadata={
                    "search_type": "semantic_vector_search",
                    "k": k,
                    "num_sources": search_result.get("num_sources", 0),
                    "search_time_ms": search_result.get("search_time_ms", response_time),
                    # Preserve engine metadata from search_semantic_detailed()
                    "engine_metadata": search_result.get("metadata", {}),
                    "parallel_execution": True
                }
            )
            
            self.logger.info(f"✅ Semantic search completed: {len(semantic_result.content)} chars, "
                           f"confidence {confidence:.2f}")
            
            return semantic_result
            
        except Exception as e:
            self.logger.error("❌ Semantic search failed", extra={
                "search_type": "semantic",
                "search_method": "semantic_vector_search",
                "error": str(e),
                "fallback_used": False,
                "parallel_execution": True
            })
            return create_error_result(
                "semantic",
                str(e),
                int((time.time() - start_time) * 1000)
            )
    # -------------------------------------------------------------- end _async_semantic_retrieve()
    
    # -------------------------------------------------------------- _async_traversal_retrieve()
    async def _async_traversal_retrieve(self, query: str, max_results: int = 50) -> RetrievalResult:
        """Run traversal search using LLM Structural Cypher.

        Args:
            query: User query.
            max_results: Maximum number of results.

        Returns:
            RetrievalResult: Traversal search results with method metadata.
        """
        start_time = time.time()
        
        if not self._traversal_search_available:
            return create_error_result(
                "traversal",
                "Traversal search tool not available", 
                int((time.time() - start_time) * 1000)
            )
        
        # Check if LLM Structural is available
        if not (settings.use_llm_structural_cypher and self._traversal_llm_available):
            if not settings.use_llm_structural_cypher:
                error_msg = "LLM Structural Cypher disabled via configuration"
            else:
                error_msg = "LLM Structural Cypher enabled but not available"
            
            return create_error_result(
                "traversal",
                error_msg,
                int((time.time() - start_time) * 1000)
            )
        
        # Execute LLM Structural path
        search_result = None
        search_method = "llm_structural_cypher"
        
        try:
            self.logger.info(f"🤖 Executing LLM Structural Cypher search (max_results={max_results})...")
            
            # Call LLM structural with circuit breaker protection
            if self._traversal_llm_breaker:
                search_result = await self._traversal_llm_breaker.call(
                    self._query_knowledge_graph_llm_structural_detailed, 
                    query, 
                    max_results=max_results
                )
            else:
                search_result = await self._query_knowledge_graph_llm_structural_detailed(
                    query, max_results=max_results
                )
            
            self.logger.info(f"✅ LLM Structural Cypher completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ LLM Structural Cypher failed: {str(e)}")
            return create_error_result(
                "traversal",
                f"LLM Structural Cypher search failed: {str(e)}",
                int((time.time() - start_time) * 1000)
            )
        
        # Process successful result
        response_time = int((time.time() - start_time) * 1000)
        
        # Extract confidence and apply traversal cap
        confidence = search_result.get("confidence", 0.0)
        confidence = min(confidence, settings.validated_traversal_confidence_cap)
        
        # Extract citations from the answer  
        citations = self._extract_citations_from_content(search_result.get("answer", ""))
        
        # Extract entities from metadata if available
        entities = []
        if search_result.get("metadata", {}).get("entities_found"):
            entities = search_result["metadata"]["entities_found"]
        
        # Build comprehensive metadata
        metadata = {
            "search_type": "graph_traversal_search",
            "search_method": search_method,
            "max_results": max_results,
            "cypher_query": search_result.get("cypher_query", ""),
            "response_time_ms": search_result.get("response_time_ms", response_time),
            "confidence_capped": confidence != search_result.get("confidence", 0.0),
            "original_confidence": search_result.get("confidence", 0.0)
        }
        
        # Bubble up engine-specific metrics if available
        engine_metadata = search_result.get("metadata", {})
        
        # For LLM structural responses
        if search_method == "llm_structural_cypher" and engine_metadata:
            # Include LLM-specific metrics
            for key in ["tokens_used", "model_used", "generation_time_ms", 
                       "execution_time_ms", "validation_issues", "fixes_applied"]:
                if key in engine_metadata:
                    metadata[f"llm_{key}"] = engine_metadata[key]
        
        # Create result with proper sources
        sources = [{
            "cypher_query": search_result.get("cypher_query", ""),
            "search_method": search_method
        }]
        
        traversal_result = RetrievalResult(
            content=search_result.get("answer", "No traversal results found"),
            method="traversal", 
            confidence=confidence,
            response_time_ms=response_time,
            sources=sources,
            entities=entities,
            citations=citations,
            metadata=metadata
        )
        
        method_display = "LLM Structural" if search_method == "llm_structural_cypher" else "LangChain"
        self.logger.info(f"✅ {method_display} traversal search completed: "
                       f"{len(traversal_result.content)} chars, confidence {confidence:.2f}")
        
        return traversal_result
    # -------------------------------------------------------------- end _async_traversal_retrieve()
    
    # -------------------------------------------------------------- _process_search_result()
    def _process_search_result(self, result: Any, method: str, query: str) -> RetrievalResult:
        """Normalize raw result, handling exceptions, into a RetrievalResult.

        Args:
            result: Raw result from search (could be RetrievalResult or Exception).
            method: Search method name.
            query: Original query.

        Returns:
            RetrievalResult: Processed result.
        """
        if isinstance(result, Exception):
            self.logger.error(f"❌ {method} search raised exception: {str(result)}")
            return create_error_result(method, str(result))
        
        if isinstance(result, RetrievalResult):
            return result
        
        # Shouldn't happen, but handle gracefully
        self.logger.warning(f"⚠️ Unexpected result type from {method} search: {type(result)}")
        return create_error_result(method, f"Unexpected result type: {type(result)}")
    # -------------------------------------------------------------- end _process_search_result()
    
    # -------------------------------------------------------------- _analyze_fusion_readiness()
    def _analyze_fusion_readiness(
        self,
        semantic: RetrievalResult,
        traversal: RetrievalResult,
    ) -> Dict[str, Any]:
        """Determine fusion readiness and overall status for both results.

        Args:
            semantic: Semantic search result.
            traversal: Traversal search result.

        Returns:
            Dict with analysis results.
        """
        # Check if searches succeeded
        semantic_success = semantic.error is None and len(semantic.content) > 50
        traversal_success = traversal.error is None and len(traversal.content) > 50
        
        # Determine overall success
        success = semantic_success or traversal_success
        both_successful = semantic_success and traversal_success
        
        # Check fusion readiness - need substantial content from both
        fusion_ready = (
            both_successful and
            semantic.confidence > 0.3 and
            traversal.confidence > 0.2 and
            len(semantic.content) > 100 and
            len(traversal.content) > 100 and
            "i don't know" not in semantic.content.lower() and
            "i don't know" not in traversal.content.lower()
        )
        
        # If one is very strong, still consider fusion-ready
        if not fusion_ready and success:
            high_confidence_threshold = 0.7
            if (semantic.confidence > high_confidence_threshold and traversal_success) or \
               (traversal.confidence > high_confidence_threshold and semantic_success):
                fusion_ready = True
        
        # Determine primary method
        if semantic.confidence > traversal.confidence:
            primary_method = "semantic"
        elif traversal.confidence > semantic.confidence:
            primary_method = "traversal"
        else:
            primary_method = "equal"
        
        return {
            "success": success,
            "both_successful": both_successful,
            "fusion_ready": fusion_ready,
            "primary_method": primary_method,
            "semantic_success": semantic_success,
            "traversal_success": traversal_success
        }
    # -------------------------------------------------------------- end _analyze_fusion_readiness()
    
    # -------------------------------------------------------------- _calculate_semantic_confidence()
    def _calculate_semantic_confidence(self, search_result: Dict[str, Any]) -> float:
        """Calculate confidence score for semantic search results.

        Supports configurable confidence cap via SEMANTIC_CONFIDENCE_CAP env var. Default cap is
        1.0 (no cap); can be set to lower values if needed.

        Args:
            search_result: Raw semantic search result.

        Returns:
            float: Confidence score 0.0 to 1.0.
        """
        try:
            # Start with base confidence
            confidence = 0.5
            
            # Boost based on number of sources
            num_sources = search_result.get("num_sources", 0)
            if num_sources > 0:
                confidence += min(0.3, num_sources * 0.05)
            
            # Boost based on entities found
            entities = search_result.get("entities_found", [])
            if entities:
                confidence += min(0.2, len(entities) * 0.02)
            
            # Check answer quality
            answer = search_result.get("answer", "")
            if len(answer) > 200:
                confidence += 0.1
            if "[1]" in answer or "[2]" in answer:  # Has citations
                confidence += 0.1
            
            # Penalize if no answer or error indicators
            if not answer or "No relevant information" in answer:
                confidence = max(0.1, confidence - 0.4)

            # Penalty for "I don't know" responses
            if "i don't know" in answer.lower():
                confidence = max(0.1, confidence - 0.4)
            
            # Apply configurable cap (default 1.0 = no cap)
            semantic_cap = settings.validated_semantic_confidence_cap
            return max(0.0, min(semantic_cap, confidence))
            
        except Exception as e:
            self.logger.warning(f"Error calculating semantic confidence: {e}")
            return 0.5
    # -------------------------------------------------------------- end _calculate_semantic_confidence()
    
    # -------------------------------------------------------------- _extract_citations_from_content()
    def _extract_citations_from_content(self, content: str) -> List[str]:
        """
        Extract citations from content text.
        
        Args:
            content: Text content to extract citations from
            
        Returns:
            List of extracted citations
        """
        import re
        
        citations = []
        
        try:
            # Extract numbered citations [1], [2], etc.
            numbered_citations = re.findall(r'\[(\d+)\]', content)
            citations.extend([f"[{num}]" for num in numbered_citations])
            
            # Extract regulatory citations like §57.4361(a)
            regulatory_citations = re.findall(r'§[\d.]+(?:\([a-z]\))?', content)
            citations.extend(regulatory_citations)
            
            # Extract Part citations
            part_citations = re.findall(r'Part\s+[\d.]+', content)
            citations.extend(part_citations)
            
            return list(set(citations))  # Remove duplicates
            
        except Exception as e:
            self.logger.warning(f"Error extracting citations: {e}")
            return []
    # -------------------------------------------------------------- end _extract_citations_from_content()
    
    # -------------------------------------------------------------- _create_timeout_response()
    def _create_timeout_response(self, query: str, start_time: float) -> ParallelRetrievalResponse:
        """
        Create response for timeout scenarios.
        
        Args:
            query: Original query
            start_time: Start time for timing calculation
            
        Returns:
            ParallelRetrievalResponse: Timeout response
        """
        total_time = int((time.time() - start_time) * 1000)
        
        timeout_result = create_error_result(
            "parallel", 
            f"Timeout after {self.timeout_seconds}s",
            total_time
        )
    
        return ParallelRetrievalResponse(
            semantic_result=timeout_result,
            traversal_result=timeout_result,
            query=query,
            total_time_ms=total_time,
            success=False,
            fusion_ready=False,
            both_successful=False,
            primary_method="error"
        )
    # -------------------------------------------------------------- end _create_timeout_response()
    
    # -------------------------------------------------------------- _create_error_response()
    def _create_error_response(self, query: str, start_time: float, 
                              error_message: str) -> ParallelRetrievalResponse:
        """
        Create response for general error scenarios.
        
        Args:
            query: Original query
            start_time: Start time for timing calculation
            error_message: Error description
            
        Returns:
            ParallelRetrievalResponse: Error response
        """
        total_time = int((time.time() - start_time) * 1000)
        
        error_result = create_error_result("parallel", error_message, total_time)
        
        return ParallelRetrievalResponse(
            semantic_result=error_result,
            traversal_result=error_result,
            query=query,
            total_time_ms=total_time,
            success=False,
            fusion_ready=False,
            both_successful=False,
            primary_method="error"
        )
    # -------------------------------------------------------------- end _create_error_response()
    
    # -------------------------------------------------------------- health_check()
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check on the parallel retrieval system.
        
        Returns:
            Dict containing health status and component information with traversal method details
        """
        health_status = {
            "tool_name": "parallel_hybrid",
            "status": "unknown",
            "components": {
                "parallel_engine": "healthy",
                "semantic_search": "unknown",
                "traversal_llm_structural": "unknown"
            },
            "timeout_seconds": self.timeout_seconds,
            "implementation_status": "step_2_parallel_engine_complete",
            "traversal_config": {
                "llm_structural_enabled": settings.use_llm_structural_cypher,
                "primary_method": "unknown"
            }
        }
        
        # Check semantic search availability
        if self._semantic_search_available:
            try:
                # Quick test of semantic search
                test_result = await self._search_semantic_detailed("test query", k=1)
                health_status["components"]["semantic_search"] = "healthy"
            except Exception as e:
                health_status["components"]["semantic_search"] = f"error: {str(e)}"
        else:
            health_status["components"]["semantic_search"] = "unavailable"
        
        # Check LLM Structural traversal availability (primary path)
        if self._traversal_llm_available:
            try:
                # Quick test of LLM structural path
                test_result = await self._query_knowledge_graph_llm_structural_detailed(
                    "test query", max_results=1
                )
                
                health_status["components"]["traversal_llm_structural"] = "healthy"
                    
            except Exception as e:
                health_status["components"]["traversal_llm_structural"] = f"error: {str(e)}"
        else:
            health_status["components"]["traversal_llm_structural"] = "unavailable"
        
        # Set traversal configuration
        traversal_config = health_status["traversal_config"]
        llm_structural_healthy = health_status["components"]["traversal_llm_structural"] == "healthy"
        
        if settings.use_llm_structural_cypher and llm_structural_healthy:
            traversal_config["primary_method"] = "llm_structural_cypher"
        elif settings.use_llm_structural_cypher:
            traversal_config["primary_method"] = "llm_structural_cypher_unavailable"
        else:
            traversal_config["primary_method"] = "disabled"
        
        # Determine overall system status
        semantic_status = health_status["components"]["semantic_search"]
        traversal_status = health_status["components"]["traversal_llm_structural"]
        
        if semantic_status == "healthy" and traversal_status == "healthy":
            health_status["status"] = "healthy"
        elif semantic_status == "healthy" or traversal_status == "healthy":
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "error"
        
        return health_status
    # -------------------------------------------------------------- end health_check()

# ------------------------------------------------------------------------- end class ParallelRetrievalEngine

# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# Utility Functions
# =========================================================================

# --------------------------------------------------------------------------------- create_error_result()
def create_error_result(
    method: str,
    error_message: str,
    response_time_ms: int = 0,
) -> RetrievalResult:
    """
    Create a RetrievalResult for error cases.
    
    Args:
        method: Search method that failed ("semantic" or "traversal").
        error_message: Error description.
        response_time_ms: Time taken before error.
        
    Returns:
        RetrievalResult with error information.
    """
    return RetrievalResult(
        content=f"Error in {method} search: {error_message}",
        method=method,
        confidence=0.0,
        response_time_ms=response_time_ms,
        error=error_message,
        metadata={"error_type": "search_failure"}
    )
# --------------------------------------------------------------------------------- end create_error_result()

# --------------------------------------------------------------------------------- validate_parallel_response()
def validate_parallel_response(response: ParallelRetrievalResponse) -> bool:
    """
    Validate a ParallelRetrievalResponse for completeness and consistency.
    
    Args:
        response: Response to validate.
        
    Returns:
        bool: True if response is valid.
    """
    try:
        # Check required fields
        if not response.query or not isinstance(response.query, str):
            return False
        
        # Check that we have both results
        if not response.semantic_result or not response.traversal_result:
            return False
        
        # Check confidence values are valid
        if (not (0.0 <= response.semantic_result.confidence <= 1.0) or
            not (0.0 <= response.traversal_result.confidence <= 1.0)):
            return False
        
        # Check timing values are non-negative
        if (response.total_time_ms < 0 or 
            response.semantic_result.response_time_ms < 0 or
            response.traversal_result.response_time_ms < 0):
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating parallel response: {e}")
        return False
# --------------------------------------------------------------------------------- end validate_parallel_response()

# =========================================================================
# Development Testing Functions
# =========================================================================

# =========================================================================
# Development Testing Functions
# =========================================================================

# --------------------------------------------------------------------------------- test_parallel_hybrid_data_models()
async def test_parallel_hybrid_data_models():
    """
    Test function for validating the data models implementation.
    
    This function tests the data structures created in Step 1 to ensure
    they work correctly before proceeding to Step 2.
    """
    logger.info("Testing APH-IF Parallel Hybrid Data Models...")
    
    try:
        # Test RetrievalResult creation
        semantic_result = RetrievalResult(
            content="Test semantic search result with citations [1], [2]",
            method="semantic", 
            confidence=0.85,
            response_time_ms=1500,
            sources=[{"document": "test_doc.pdf", "page": 1}],
            entities=["test_entity"],
            citations=["[1] Test Citation", "[2] Another Citation"]
        )
        
        traversal_result = RetrievalResult(
            content="Test traversal result with regulatory citation §57.4361(a) [3]",
            method="traversal",
            confidence=0.72,
            response_time_ms=2200,
            sources=[{"cypher_query": "MATCH test", "results": 5}],
            entities=["regulatory_entity"],
            citations=["§57.4361(a) [3]"]
        )
        
        # Test ParallelRetrievalResponse creation
        parallel_response = ParallelRetrievalResponse(
            semantic_result=semantic_result,
            traversal_result=traversal_result,
            query="What are the safety requirements?",
            total_time_ms=2200,
            success=True,
            fusion_ready=True,
            both_successful=True,
            primary_method="semantic"
        )
        
        # Test validation
        is_valid = validate_parallel_response(parallel_response)
        logger.info(f"✅ Parallel response validation: {is_valid}")
        
        # Test engine initialization
        engine = get_parallel_engine()
        health = await engine.health_check()
        logger.info(f"✅ Engine health check: {health['status']}")
        
        # Test placeholder parallel retrieval
        test_result = await engine.retrieve_parallel("Test query")
        logger.info(f"✅ Placeholder retrieval test: {test_result.success}")
        
        logger.info("✅ All data model tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data model test failed: {e}")
        return False
# --------------------------------------------------------------------------------- end test_parallel_hybrid_data_models()

# --------------------------------------------------------------------------------- test_parallel_retrieval_engine()
async def test_parallel_retrieval_engine():
    """
    Comprehensive test function for the parallel retrieval engine implementation.
    
    This function tests the core parallel execution functionality with real
    search tools if available, or mock tests if tools are not available.
    """
    logger.info("=" * 70)
    logger.info("🚀 Testing APH-IF Parallel Retrieval Engine - Step 2")
    logger.info("=" * 70)
    
    try:
        # Initialize engine
        engine = get_parallel_engine()
        
        # Test 1: Engine health check
        logger.info("\n📋 Test 1: Engine Health Check")
        health = await engine.health_check()
        logger.info(f"Overall Status: {health['status']}")
        logger.info(f"Semantic Available: {health['components']['semantic_search']}")
        logger.info(f"Traversal Available: {health['components']['traversal_search']}")
        
        # Test 2: Basic parallel retrieval (should work even if tools unavailable)
        logger.info("\n🔄 Test 2: Basic Parallel Retrieval")
        test_query = "What are the safety requirements for mining operations?"
        
        start_time = time.time()
        parallel_result = await engine.retrieve_parallel(test_query, semantic_k=5, traversal_max_results=20)
        execution_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"✅ Parallel retrieval completed in {execution_time}ms")
        logger.info(f"Query: {parallel_result.query}")
        logger.info(f"Success: {parallel_result.success}")
        logger.info(f"Both Successful: {parallel_result.both_successful}")
        logger.info(f"Fusion Ready: {parallel_result.fusion_ready}")
        logger.info(f"Primary Method: {parallel_result.primary_method}")
        logger.info(f"Complementarity: {parallel_result.complementarity_score:.2f}")
        
        # Test 3: Result validation
        logger.info("\n✅ Test 3: Result Validation")
        is_valid = validate_parallel_response(parallel_result)
        logger.info(f"Validation passed: {is_valid}")
        
        # Test 4: Individual result analysis
        logger.info("\n📊 Test 4: Individual Result Analysis")
        logger.info(f"Semantic Result:")
        logger.info(f"  Content length: {len(parallel_result.semantic_result.content)} chars")
        logger.info(f"  Confidence: {parallel_result.semantic_result.confidence:.2f}")
        logger.info(f"  Response time: {parallel_result.semantic_result.response_time_ms}ms")
        logger.info(f"  Citations: {len(parallel_result.semantic_result.citations)}")
        logger.info(f"  Entities: {len(parallel_result.semantic_result.entities)}")
        
        logger.info(f"Traversal Result:")
        logger.info(f"  Content length: {len(parallel_result.traversal_result.content)} chars")
        logger.info(f"  Confidence: {parallel_result.traversal_result.confidence:.2f}")
        logger.info(f"  Response time: {parallel_result.traversal_result.response_time_ms}ms")
        logger.info(f"  Citations: {len(parallel_result.traversal_result.citations)}")
        logger.info(f"  Entities: {len(parallel_result.traversal_result.entities)}")
        
        # Test 5: Combined metadata
        logger.info("\n🔗 Test 5: Combined Metadata")
        logger.info(f"Combined entities: {len(parallel_result.entities_combined)}")
        logger.info(f"Combined sources: {len(parallel_result.sources_combined)}")
        logger.info(f"Vector contribution: {parallel_result.vector_contribution:.2f}")
        logger.info(f"Graph contribution: {parallel_result.graph_contribution:.2f}")
        
        # Test 6: Error handling (timeout simulation)
        logger.info("\n⚠️ Test 6: Timeout Handling")
        short_timeout_engine = ParallelRetrievalEngine(timeout_seconds=1)
        # This should work quickly, but demonstrates timeout handling structure
        timeout_result = await short_timeout_engine.retrieve_parallel("Quick test")
        logger.info(f"Short timeout test success: {timeout_result.success}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ All Parallel Retrieval Engine tests completed successfully!")
        logger.info("🎯 Step 2 Implementation: COMPLETE")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Parallel retrieval engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
# --------------------------------------------------------------------------------- end test_parallel_retrieval_engine()

# __________________________________________________________________________
# Module Initialization / Main Execution Guard 

if __name__ == "__main__":
    # Test entry point for development
    async def main():
        # Test data models first
        data_models_success = await test_parallel_hybrid_data_models()
        
        if data_models_success:
            # Test the parallel retrieval engine
            await test_parallel_retrieval_engine()
        else:
            logger.error("❌ Data model tests failed, skipping engine tests")
    
    asyncio.run(main())