# -------------------------------------------------------------------------
# File: process_parallel.py
# Author: Alexander Ricciardi
# Date: 2025-09-16
# [File Path] backend/app/processing/process_parallel.py
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
#   Process-level parallel search utilities and engine. Executes semantic and traversal
#   searches in separate OS processes to achieve true parallelism and eliminate shared
#   resources across clients, drivers, and in-memory state.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Dataclass: ProcessSearchResult
# - Function: run_semantic_search_process
# - Function: run_traversal_search_process
# - Class: ProcessParallelEngine
# - Function: get_process_parallel_engine
# - Function: test_process_parallel (dev)
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: asyncio, time, sys, multiprocessing, logging, pathlib, concurrent.futures
# - Third-Party: (none)
# - Local Project Modules: search.tools (vector, cypher)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Used by the backend parallel retrieval flow to optionally run searches with process isolation,
# improving concurrency by avoiding shared client and driver state.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Process-Level Parallel Search Implementation for APH-IF

This module implements true parallel execution using separate processes
to eliminate LangChain/library internal sharing and achieve maximum
concurrent performance for semantic and traversal searches.

Key Innovation: Process isolation eliminates ALL shared resources:
- Separate OpenAI API clients
- Separate Neo4j connections
- Separate LangChain state
- Separate memory spaces
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports
import asyncio
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# __________________________________________________________________________
# Global Constants / Variables

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------- class ProcessSearchResult
@dataclass
class ProcessSearchResult:
    """Result from a process-isolated search operation."""
    method: str
    content: str
    confidence: float
    response_time_ms: int
    sources: int
    entities: int
    citations: int
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
# ------------------------------------------------------------------------- end class ProcessSearchResult

# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# Process-Isolated Search Functions
# =========================================================================

# --------------------------------------------------------------------------------- run_semantic_search_process()
def run_semantic_search_process(query: str, k: int = 5) -> Dict[str, Any]:
    """Run semantic search in a completely isolated process.

    This function executes in a separate OS process with its own interpreter, LangChain state,
    OpenAI client, Neo4j connection, and memory space to guarantee isolation.

    Args:
        query: Search query text.
        k: Number of semantic chunks to retrieve.

    Returns:
        Dict[str, Any]: A serializable dictionary with search results and process metadata.
    """
    import asyncio
    import time
    
    # Re-add path in subprocess (each process needs its own path setup)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    async def _isolated_semantic_search():
        """Async semantic search in isolated process."""
        try:
            # Import inside process to avoid shared state
            from ..search.tools.vector import search_semantic_detailed
            
            start_time = time.time()
            result = await search_semantic_detailed(query, k=k)
            end_time = time.time()
            
            # Extract citations from result
            citations_count = 0
            if "answer" in result:
                # Count citation patterns [1], [2], etc.
                import re
                citations = re.findall(r'\[\d+\]', result["answer"])
                citations_count = len(set(citations))
            
            return {
                "method": "semantic",
                "content": result.get("answer", ""),
                "confidence": 1.0,  # Vector search always high confidence if successful
                "response_time_ms": int((end_time - start_time) * 1000),
                "sources": len(result.get("sources", [])),
                "entities": len(result.get("entities_found", [])),
                "citations": citations_count,
                "success": "error" not in result,
                "error": result.get("error"),
                "metadata": {
                    "process_id": mp.current_process().pid,
                    "search_type": "semantic_vector_search",
                    "k_parameter": k
                }
            }
        except Exception as e:
            return {
                "method": "semantic",
                "content": f"Error in semantic search: {str(e)}",
                "confidence": 0.0,
                "response_time_ms": 0,
                "sources": 0,
                "entities": 0,
                "citations": 0,
                "success": False,
                "error": str(e),
                "metadata": {"process_id": mp.current_process().pid}
            }
    
    # Run the async search in this process
    return asyncio.run(_isolated_semantic_search())
# --------------------------------------------------------------------------------- end run_semantic_search_process()

# --------------------------------------------------------------------------------- run_traversal_search_process()
def run_traversal_search_process(query: str, max_results: int = 20) -> Dict[str, Any]:
    """Run traversal search in a completely isolated process.

    This function executes in a separate OS process with its own interpreter, LangChain state,
    Neo4j connection, and memory space to guarantee isolation.

    Args:
        query: Search query text.
        max_results: Maximum number of graph traversal results to return.

    Returns:
        Dict[str, Any]: A serializable dictionary with search results and process metadata.
    """
    import asyncio
    import time
    
    # Re-add path in subprocess (each process needs its own path setup)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    async def _isolated_traversal_search():
        """Async traversal search in isolated process."""
        try:
            # Import inside process to avoid shared state
            from ..search.tools.cypher import query_knowledge_graph_llm_structural_detailed
            
            start_time = time.time()
            result = await query_knowledge_graph_llm_structural_detailed(query, max_results=max_results)
            end_time = time.time()
            
            # Extract citations from result
            citations_count = 0
            if "answer" in result:
                # Count various citation patterns
                import re
                citations = re.findall(r'(?:\[\d+\]|§\d+|Part \d+|CFR \d+)', result["answer"])
                citations_count = len(set(citations))
            
            return {
                "method": "traversal",
                "content": result.get("answer", ""),
                "confidence": result.get("confidence", 0.0),
                "response_time_ms": int((end_time - start_time) * 1000),
                "sources": 1,  # Graph search typically returns aggregated results
                "entities": 0,  # Simplified for now
                "citations": citations_count,
                "success": "error" not in result,
                "error": result.get("error"),
                "metadata": {
                    "process_id": mp.current_process().pid,
                    "search_type": "graph_traversal_search",
                    "max_results_parameter": max_results,
                    "cypher_query": result.get("cypher_query")
                }
            }
        except Exception as e:
            return {
                "method": "traversal",
                "content": f"Error in traversal search: {str(e)}",
                "confidence": 0.0,
                "response_time_ms": 0,
                "sources": 0,
                "entities": 0,
                "citations": 0,
                "success": False,
                "error": str(e),
                "metadata": {"process_id": mp.current_process().pid}
            }
    
    # Run the async search in this process
    return asyncio.run(_isolated_traversal_search())
# --------------------------------------------------------------------------------- end run_traversal_search_process()

# ____________________________________________________________________________
# Class Definitions

# ------------------------------------------------------------------------- class ProcessParallelEngine
class ProcessParallelEngine:
    """Process-level parallel search engine for true concurrency.

    Uses separate processes to eliminate shared resources and achieve true parallel execution
    of semantic and traversal searches.
    """
    
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self, max_workers: int = 2):
        """Initialize the process parallel engine.

        Args:
            max_workers: Number of worker processes to use.
        """
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Set multiprocessing start method
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
        
        self.logger.info(f"Process Parallel Engine initialized with {max_workers} workers")
    # --------------------------------------------------------------------------------- end __init__()
    
    # -------------------------------------------------------------- retrieve_parallel_process()
    async def retrieve_parallel_process(
        self,
        query: str,
        semantic_k: int = 5,
        traversal_max_results: int = 20,
    ) -> Dict[str, Any]:
        """Execute parallel searches using separate processes for true concurrency.

        Args:
            query: User query text.
            semantic_k: Number of semantic chunks to retrieve.
            traversal_max_results: Maximum graph traversal results to return.

        Returns:
            Dict[str, Any]: Combined results from both searches with process metadata.
        """
        start_time = time.time()
        self.logger.info(f"Starting process-level parallel searches for: {query[:50]}...")
        
        try:
            # Use ProcessPoolExecutor for true process isolation
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                loop = asyncio.get_event_loop()
                
                self.logger.info("⚡ Submitting searches to separate processes...")
                
                # Submit both searches to separate processes
                semantic_future = loop.run_in_executor(
                    executor,
                    run_semantic_search_process,
                    query,
                    semantic_k
                )
                
                traversal_future = loop.run_in_executor(
                    executor,
                    run_traversal_search_process,
                    query,
                    traversal_max_results
                )
                
                # Wait for both processes to complete
                semantic_result, traversal_result = await asyncio.gather(
                    semantic_future,
                    traversal_future,
                    return_exceptions=True
                )
                
                # Handle any exceptions
                if isinstance(semantic_result, Exception):
                    semantic_result = {
                        "method": "semantic",
                        "content": f"Process error: {str(semantic_result)}",
                        "confidence": 0.0,
                        "response_time_ms": 0,
                        "sources": 0,
                        "entities": 0,
                        "citations": 0,
                        "success": False,
                        "error": str(semantic_result)
                    }
                
                if isinstance(traversal_result, Exception):
                    traversal_result = {
                        "method": "traversal",
                        "content": f"Process error: {str(traversal_result)}",
                        "confidence": 0.0,
                        "response_time_ms": 0,
                        "sources": 0,
                        "entities": 0,
                        "citations": 0,
                        "success": False,
                        "error": str(traversal_result)
                    }
            
            total_time_ms = int((time.time() - start_time) * 1000)
            
            # Calculate metrics
            both_successful = semantic_result["success"] and traversal_result["success"]
            success = semantic_result["success"] or traversal_result["success"]
            
            # Determine primary method
            primary_method = "semantic"
            if traversal_result["confidence"] > semantic_result["confidence"]:
                primary_method = "traversal"
            elif semantic_result["confidence"] == traversal_result["confidence"]:
                primary_method = "balanced"
            
            # Calculate complementarity (simplified)
            complementarity_score = 1.0 if both_successful else 0.5
            
            # Fusion readiness
            fusion_ready = (
                both_successful and 
                semantic_result["confidence"] > 0.6 and 
                traversal_result["confidence"] > 0.6
            )
            
            return {
                "success": success,
                "both_successful": both_successful,
                "fusion_ready": fusion_ready,
                "primary_method": primary_method,
                "complementarity_score": complementarity_score,
                "total_time_ms": total_time_ms,
                "semantic_result": semantic_result,
                "traversal_result": traversal_result,
                "process_isolation": True,
                "process_metadata": {
                    "semantic_process_id": semantic_result.get("metadata", {}).get("process_id"),
                    "traversal_process_id": traversal_result.get("metadata", {}).get("process_id"),
                    "processes_different": (
                        semantic_result.get("metadata", {}).get("process_id") != 
                        traversal_result.get("metadata", {}).get("process_id")
                    )
                }
            }
            
        except Exception as e:
            self.logger.error(f"Process parallel execution failed: {e}")
            return {
                "success": False,
                "both_successful": False,
                "fusion_ready": False,
                "primary_method": "error",
                "complementarity_score": 0.0,
                "total_time_ms": int((time.time() - start_time) * 1000),
                "semantic_result": {"method": "semantic", "success": False, "error": str(e)},
                "traversal_result": {"method": "traversal", "success": False, "error": str(e)},
                "process_isolation": False,
                "error": str(e)
            }
    # -------------------------------------------------------------- end retrieve_parallel_process()
    
    # -------------------------------------------------------------- health_check()
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the process parallel engine.

        Returns:
            Dict[str, Any]: Health status, components, and probing result details.
        """
        try:
            # Test process pool creation
            with ProcessPoolExecutor(max_workers=1) as executor:
                loop = asyncio.get_event_loop()
                test_result = await loop.run_in_executor(
                    executor,
                    lambda: {"test": "success", "pid": mp.current_process().pid}
                )
            
            return {
                "status": "healthy",
                "components": {
                    "process_pool": "healthy",
                    "multiprocessing": "available",
                    "isolation": "enabled"
                },
                "test_result": test_result,
                "max_workers": self.max_workers
            }
        except Exception as e:
            return {
                "status": "error",
                "components": {
                    "process_pool": "error",
                    "multiprocessing": "unavailable"
                },
                "error": str(e)
            }
    # -------------------------------------------------------------- end health_check()
 
# ------------------------------------------------------------------------- end class ProcessParallelEngine

# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# Process Parallel Engine Functions
# =========================================================================

# Global process engine instance
_process_engine: Optional[ProcessParallelEngine] = None

 # --------------------------------------------------------------------------------- get_process_parallel_engine()
def get_process_parallel_engine() -> ProcessParallelEngine:
    """Get or create the global process parallel engine.

    Returns:
        ProcessParallelEngine: The singleton engine instance.
    """
    global _process_engine
    if _process_engine is None:
        _process_engine = ProcessParallelEngine()
    return _process_engine
 # --------------------------------------------------------------------------------- end get_process_parallel_engine()

 # --------------------------------------------------------------------------------- test_process_parallel()
async def test_process_parallel() -> bool:
    """Test the process parallel engine.

    Returns:
        bool: True when the engine test succeeds, otherwise False.
    """
    print("🚀 Testing Process-Level Parallel Search Engine...")
    
    engine = get_process_parallel_engine()
    
    # Health check
    health = await engine.health_check()
    print(f"Health Status: {health['status']}")
    
    if health["status"] != "healthy":
        print(f"❌ Engine not healthy: {health.get('error')}")
        return False
    
    # Test query
    test_query = "What are the safety equipment requirements for mining operations?"
    print(f"\nTesting with query: {test_query}")
    
    start_time = time.time()
    result = await engine.retrieve_parallel_process(test_query)
    total_time = time.time() - start_time
    
    print("\n📊 PROCESS PARALLEL RESULTS:")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Success: {result['success']}")
    print(f"Both Successful: {result['both_successful']}")
    print(f"Process Isolation: {result['process_isolation']}")
    
    if result.get("process_metadata"):
        meta = result["process_metadata"]
        print(f"Semantic Process ID: {meta.get('semantic_process_id')}")
        print(f"Traversal Process ID: {meta.get('traversal_process_id')}")
        print(f"Different Processes: {meta.get('processes_different')}")
    
    # Calculate parallel efficiency
    semantic_time = result["semantic_result"]["response_time_ms"] / 1000
    traversal_time = result["traversal_result"]["response_time_ms"] / 1000
    max_individual = max(semantic_time, traversal_time)
    
    efficiency = (max_individual / total_time) * 100 if total_time > 0 else 0
    print(f"Parallel Efficiency: {efficiency:.1f}%")
    
    return result["success"]
 # --------------------------------------------------------------------------------- end test_process_parallel()

# __________________________________________________________________________
# Module Initialization / Main Execution Guard

if __name__ == "__main__":
    async def main():
        success = await test_process_parallel()
        return success
    
    try:
        result = asyncio.run(main())
        print(f"\nTest {'PASSED' if result else 'FAILED'}")
    except Exception as e:
        print(f"Test failed: {e}")

# __________________________________________________________________________
# End of File
#