# =========================================================================
# File: parallel_hybrid.py
# Project: APH-IF Technology Framework
#          Advanced Parallel HybridRAG - Intelligent Fusion System
# Author: Alexander Ricciardi
# Date: 2025-08-05
# File Path: backend/parallel_hybrid.py
# =========================================================================

"""
Parallel HybridRAG Processing Engine for APH-IF

Implements the core Advanced Parallel HybridRAG technology that executes
VectorRAG and GraphRAG searches concurrently, then fuses results using
intelligent fusion algorithms.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pydantic import BaseModel

# Local imports
from .config import get_config
from .circuit_breaker import get_circuit_breaker_manager
from .tools.vector import VectorSearchTool
from .tools.cypher import CypherSearchTool
from .context_fusion import get_fusion_engine

# =========================================================================
# Data Models
# =========================================================================
@dataclass
class SearchResult:
    """Individual search result"""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    search_type: str  # "vector" or "graph"

class ParallelRetrievalResponse(BaseModel):
    """Response from parallel retrieval operation"""
    vector_results: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    processing_time: float
    vector_search_time: float
    graph_search_time: float
    total_results: int
    query: str
    timestamp: datetime

class ParallelHybridResult(BaseModel):
    """Final result from parallel hybrid processing"""
    fused_results: List[Dict[str, Any]]
    original_query: str
    processing_time: float
    vector_results_count: int
    graph_results_count: int
    fusion_strategy: str
    confidence_score: float
    metadata: Dict[str, Any]

# =========================================================================
# Parallel HybridRAG Engine
# =========================================================================
class ParallelHybridRAGEngine:
    """
    Core engine for Advanced Parallel HybridRAG processing

    Coordinates concurrent execution of VectorRAG and GraphRAG searches,
    then applies intelligent fusion to combine results.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger("parallel_hybridrag_engine")
        
        # Initialize circuit breakers
        self.cb_manager = get_circuit_breaker_manager()
        self.vector_cb = self.cb_manager.create_circuit_breaker(
            "vector_search",
            failure_threshold=3,
            recovery_timeout=30,
            timeout=30
        )
        self.graph_cb = self.cb_manager.create_circuit_breaker(
            "graph_search", 
            failure_threshold=3,
            recovery_timeout=30,
            timeout=30
        )
        
        # Initialize search tools
        self.vector_tool = None
        self.cypher_tool = None
        self.fusion_engine = None
        
        # Performance tracking
        self.total_queries = 0
        self.total_processing_time = 0.0
        self.average_processing_time = 0.0
    
    async def initialize(self):
        """Initialize search tools and fusion engine"""
        try:
            self.vector_tool = VectorSearchTool()
            await self.vector_tool.initialize()
            
            self.cypher_tool = CypherSearchTool()
            await self.cypher_tool.initialize()
            
            self.fusion_engine = get_fusion_engine()
            
            self.logger.info("Parallel hybrid engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize parallel hybrid engine: {e}")
            raise
    
    async def process_query(
        self,
        query: str,
        max_vector_results: Optional[int] = None,
        max_graph_results: Optional[int] = None
    ) -> ParallelHybridResult:
        """
        Process query using parallel HybridRAG approach

        Args:
            query: User query to process
            max_vector_results: Maximum VectorRAG search results
            max_graph_results: Maximum GraphRAG search results

        Returns:
            ParallelHybridResult with intelligently fused results
        """
        start_time = time.time()
        self.total_queries += 1
        
        try:
            # Execute parallel retrieval
            retrieval_response = await self._parallel_retrieval(
                query, max_vector_results, max_graph_results
            )
            
            # Apply intelligent fusion
            fused_results = await self._apply_fusion(retrieval_response)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.average_processing_time = self.total_processing_time / self.total_queries
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(
                retrieval_response.vector_results,
                retrieval_response.graph_results,
                fused_results
            )
            
            result = ParallelHybridResult(
                fused_results=fused_results,
                original_query=query,
                processing_time=processing_time,
                vector_results_count=len(retrieval_response.vector_results),
                graph_results_count=len(retrieval_response.graph_results),
                fusion_strategy="intelligent",
                confidence_score=confidence_score,
                metadata={
                    "vector_search_time": retrieval_response.vector_search_time,
                    "graph_search_time": retrieval_response.graph_search_time,
                    "total_results": retrieval_response.total_results,
                    "timestamp": datetime.now().isoformat(),
                    "engine_stats": self.get_stats()
                }
            )
            
            self.logger.info(
                f"Processed query in {processing_time:.2f}s: "
                f"{len(retrieval_response.vector_results)} vector + "
                f"{len(retrieval_response.graph_results)} graph results"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise
    
    async def _parallel_retrieval(
        self,
        query: str,
        max_vector_results: Optional[int] = None,
        max_graph_results: Optional[int] = None
    ) -> ParallelRetrievalResponse:
        """Execute vector and graph searches in parallel"""
        
        # Set defaults from config
        max_vector_results = max_vector_results or self.config.parallel_hybrid.max_vector_results
        max_graph_results = max_graph_results or self.config.parallel_hybrid.max_graph_results
        
        # Create search tasks
        vector_task = self._vector_search_task(query, max_vector_results)
        graph_task = self._graph_search_task(query, max_graph_results)
        
        # Execute searches concurrently
        start_time = time.time()
        
        try:
            # Use asyncio.gather for true parallel execution
            (vector_results, vector_time), (graph_results, graph_time) = await asyncio.gather(
                vector_task,
                graph_task,
                return_exceptions=False
            )
            
            total_time = time.time() - start_time
            
            return ParallelRetrievalResponse(
                vector_results=vector_results,
                graph_results=graph_results,
                processing_time=total_time,
                vector_search_time=vector_time,
                graph_search_time=graph_time,
                total_results=len(vector_results) + len(graph_results),
                query=query,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error in parallel retrieval: {e}")
            # Return partial results if one search fails
            return await self._handle_partial_failure(query, e)
    
    async def _vector_search_task(self, query: str, max_results: int) -> Tuple[List[Dict], float]:
        """Execute vector search with circuit breaker protection"""
        start_time = time.time()
        
        try:
            if self.vector_tool is None:
                raise RuntimeError("Vector search tool not initialized")
            
            # Use circuit breaker for fault tolerance
            results = await self.vector_cb.call_async(
                self.vector_tool.search,
                query=query,
                limit=max_results,
                threshold=self.config.parallel_hybrid.vector_similarity_threshold
            )
            
            search_time = time.time() - start_time
            return results, search_time
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return [], time.time() - start_time
    
    async def _graph_search_task(self, query: str, max_results: int) -> Tuple[List[Dict], float]:
        """Execute graph search with circuit breaker protection"""
        start_time = time.time()
        
        try:
            if self.cypher_tool is None:
                raise RuntimeError("Cypher search tool not initialized")
            
            # Use circuit breaker for fault tolerance
            results = await self.graph_cb.call_async(
                self.cypher_tool.search,
                query=query,
                limit=max_results,
                max_depth=self.config.parallel_hybrid.graph_traversal_depth
            )
            
            search_time = time.time() - start_time
            return results, search_time
            
        except Exception as e:
            self.logger.error(f"Graph search failed: {e}")
            return [], time.time() - start_time
    
    async def _apply_fusion(
        self,
        retrieval_response: ParallelRetrievalResponse
    ) -> List[Dict[str, Any]]:
        """Apply intelligent fusion to combine search results"""
        
        if self.fusion_engine is None:
            self.logger.warning("Fusion engine not available, returning concatenated results")
            return retrieval_response.vector_results + retrieval_response.graph_results
        
        try:
            fused_results = await self.fusion_engine.fuse_results(
                vector_results=retrieval_response.vector_results,
                graph_results=retrieval_response.graph_results,
                original_query=retrieval_response.query
            )
            
            return fused_results
            
        except Exception as e:
            self.logger.error(f"Context fusion failed: {e}")
            # Fallback to simple concatenation
            return retrieval_response.vector_results + retrieval_response.graph_results
    
    def _calculate_confidence(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict],
        fused_results: List[Dict]
    ) -> float:
        """Calculate confidence score for the results"""
        # Use fused_results length as a factor in confidence calculation
        
        # Simple confidence calculation based on result counts and scores
        total_results = len(vector_results) + len(graph_results)
        
        if total_results == 0:
            return 0.0
        
        # Factor in result diversity and quality
        vector_score = sum(r.get('score', 0) for r in vector_results) / max(len(vector_results), 1)
        graph_score = sum(r.get('score', 0) for r in graph_results) / max(len(graph_results), 1)
        
        # Weighted average with bonus for having both types of results
        if len(vector_results) > 0 and len(graph_results) > 0:
            diversity_bonus = 0.1
        else:
            diversity_bonus = 0.0
        
        confidence = min(1.0, (vector_score + graph_score) / 2 + diversity_bonus)
        return round(confidence, 3)
    
    async def _handle_partial_failure(self, query: str, error: Exception) -> ParallelRetrievalResponse:
        """Handle cases where one or both searches fail"""
        self.logger.warning(f"Handling partial failure for query: {query}, error: {error}")
        
        return ParallelRetrievalResponse(
            vector_results=[],
            graph_results=[],
            processing_time=0.0,
            vector_search_time=0.0,
            graph_search_time=0.0,
            total_results=0,
            query=query,
            timestamp=datetime.now()
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        return {
            "total_queries": self.total_queries,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.average_processing_time,
            "circuit_breaker_stats": self.cb_manager.get_all_stats()
        }
    
    def get_health_status(self) -> Dict[str, str]:
        """Get health status of engine components"""
        return {
            "engine": "healthy",
            "vector_tool": "healthy" if self.vector_tool else "not_initialized",
            "cypher_tool": "healthy" if self.cypher_tool else "not_initialized",
            "fusion_engine": "healthy" if self.fusion_engine else "not_initialized",
            **self.cb_manager.get_health_status()
        }

# =========================================================================
# Global HybridRAG Engine Instance
# =========================================================================
_parallel_engine: Optional[ParallelHybridRAGEngine] = None

async def get_parallel_engine() -> ParallelHybridRAGEngine:
    """Get the global parallel HybridRAG engine instance"""
    global _parallel_engine

    if _parallel_engine is None:
        _parallel_engine = ParallelHybridRAGEngine()
        await _parallel_engine.initialize()

    return _parallel_engine
