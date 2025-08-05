# =========================================================================
# File: context_fusion.py
# Project: APH-IF Technology Framework
#          Advanced Parallel HybridRAG - Intelligent Fusion System
# Author: Alexander Ricciardi
# Date: 2025-08-05
# File Path: backend/context_fusion.py
# =========================================================================

"""
Intelligent Fusion Engine for APH-IF

Implements advanced algorithms for intelligently fusing results from parallel VectorRAG
and GraphRAG searches using LLM-based intelligent fusion for optimal result synthesis.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Local imports
from .config import get_config
from .llm import get_llm_client

# =========================================================================
# Intelligent Fusion Implementation
# =========================================================================
# This engine uses only intelligent LLM-based fusion for optimal results

# =========================================================================
# Data Models
# =========================================================================
@dataclass
class FusionResult:
    """Result from context fusion operation"""
    fused_content: str
    confidence_score: float
    source_breakdown: Dict[str, int]
    fusion_strategy: str
    processing_time: float
    metadata: Dict[str, Any]

# =========================================================================
# Intelligent Fusion Engine
# =========================================================================
class IntelligentFusionEngine:
    """
    Intelligent Fusion Engine for Advanced Parallel HybridRAG

    Combines results from parallel VectorRAG and GraphRAG searches using
    intelligent LLM-based fusion to produce coherent, comprehensive responses.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger("intelligent_fusion_engine")
        self.llm_client = None
        
        # Fusion weights and parameters
        self.vector_weight = 0.6
        self.graph_weight = 0.4
        self.relevance_threshold = 0.5
        self.max_fusion_length = 4000
        
        # Performance tracking
        self.total_fusions = 0
        self.total_processing_time = 0.0
    
    async def initialize(self):
        """Initialize the fusion engine"""
        try:
            self.llm_client = await get_llm_client()
            self.logger.info("Context fusion engine initialized successfully")
        except Exception as e:
            self.logger.warning(f"LLM client initialization failed: {e}")
            self.llm_client = None
    
    async def fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        original_query: str = "",
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fuse VectorRAG and GraphRAG search results using intelligent fusion

        Args:
            vector_results: Results from VectorRAG search
            graph_results: Results from GraphRAG search
            original_query: Original user query for context
            max_results: Maximum number of results to return

        Returns:
            List of intelligently fused results
        """
        start_time = asyncio.get_event_loop().time()
        self.total_fusions += 1

        try:
            # Validate inputs
            if not vector_results and not graph_results:
                return []

            # Apply intelligent fusion using LLM
            if self.llm_client:
                fused_results = await self._intelligent_fusion(
                    vector_results, graph_results, original_query, max_results
                )
            else:
                # Fallback to simple concatenation if LLM not available
                fused_results = await self._simple_concatenation(
                    vector_results, graph_results, max_results
                )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            self.total_processing_time += processing_time
            
            self.logger.info(
                f"Fused {len(vector_results)} VectorRAG + {len(graph_results)} GraphRAG results "
                f"into {len(fused_results)} results using intelligent fusion "
                f"in {processing_time:.3f}s"
            )
            
            return fused_results
            
        except Exception as e:
            self.logger.error(f"Error in intelligent fusion: {e}")
            # Fallback to simple concatenation
            return await self._simple_concatenation(vector_results, graph_results, max_results)
    
    async def _intelligent_fusion(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict],
        query: str,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """LLM-based intelligent fusion of results"""
        
        if not self.llm_client:
            self.logger.warning("LLM client not available, falling back to weighted fusion")
            return await self._weighted_fusion(vector_results, graph_results, max_results)
        
        try:
            # Prepare context for LLM
            context = self._prepare_fusion_context(vector_results, graph_results, query)
            
            # Create fusion prompt
            fusion_prompt = self._create_fusion_prompt(context, query, max_results)
            
            # Get LLM response
            response = await self.llm_client.generate_response(
                prompt=fusion_prompt,
                max_tokens=self.max_fusion_length,
                temperature=0.1
            )
            
            # Parse and structure the response
            fused_results = self._parse_llm_fusion_response(
                response, vector_results, graph_results
            )
            
            return fused_results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Intelligent fusion failed: {e}")
            return await self._simple_concatenation(vector_results, graph_results, max_results)
    
    async def _simple_concatenation(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Simple concatenation fallback when LLM is not available"""

        all_results = []

        # Add VectorRAG results
        for result in vector_results:
            result_copy = result.copy()
            result_copy['source_type'] = 'vector'
            all_results.append(result_copy)

        # Add GraphRAG results
        for result in graph_results:
            result_copy = result.copy()
            result_copy['source_type'] = 'graph'
            all_results.append(result_copy)

        # Remove duplicates and limit results
        deduplicated_results = self._remove_duplicates(all_results)

        return deduplicated_results[:max_results]

    
    def _prepare_fusion_context(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict],
        query: str
    ) -> Dict[str, Any]:
        """Prepare context for LLM-based fusion"""
        
        return {
            "query": query,
            "vector_results": vector_results[:5],  # Limit for context size
            "graph_results": graph_results[:5],
            "total_vector_count": len(vector_results),
            "total_graph_count": len(graph_results)
        }
    
    def _create_fusion_prompt(self, context: Dict, query: str, max_results: int) -> str:
        """Create prompt for LLM-based fusion"""
        
        prompt = f"""
You are an expert at fusing search results from different sources to provide comprehensive answers.

Query: {query}

Vector Search Results ({context['total_vector_count']} total):
{self._format_results_for_prompt(context['vector_results'])}

Graph Search Results ({context['total_graph_count']} total):
{self._format_results_for_prompt(context['graph_results'])}

Please fuse these results to create a comprehensive response that:
1. Combines the most relevant information from both sources
2. Eliminates redundancy while preserving important details
3. Maintains factual accuracy
4. Provides a coherent, well-structured answer

Return up to {max_results} key points that best address the query.
"""
        return prompt
    
    def _format_results_for_prompt(self, results: List[Dict]) -> str:
        """Format results for inclusion in LLM prompt"""
        
        formatted = []
        for i, result in enumerate(results, 1):
            content = result.get('content', result.get('text', 'No content'))
            score = result.get('score', 0.0)
            formatted.append(f"{i}. (Score: {score:.2f}) {content[:200]}...")
        
        return "\n".join(formatted)
    
    def _parse_llm_fusion_response(
        self,
        response: str,
        vector_results: List[Dict],
        graph_results: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Parse LLM response into structured results"""
        
        # Simple parsing - in production, this would be more sophisticated
        lines = response.strip().split('\n')
        parsed_results = []
        
        for line in lines:
            if line.strip() and not line.startswith('#'):
                parsed_results.append({
                    'content': line.strip(),
                    'source_type': 'fused',
                    'fusion_method': 'intelligent',
                    'confidence': 0.8,  # Default confidence
                    'metadata': {
                        'original_vector_count': len(vector_results),
                        'original_graph_count': len(graph_results),
                        'fusion_timestamp': datetime.now().isoformat()
                    }
                })
        
        return parsed_results
    
    def _remove_duplicates(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on content similarity"""
        
        unique_results = []
        seen_content = set()
        
        for result in results:
            content = result.get('content', result.get('text', ''))
            content_key = content[:100].lower().strip()  # Simple deduplication
            
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append(result)
        
        return unique_results
    
    def _get_result_key(self, result: Dict) -> str:
        """Generate unique key for result"""
        content = result.get('content', result.get('text', ''))
        return content[:50].lower().strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fusion engine statistics"""
        avg_time = self.total_processing_time / max(self.total_fusions, 1)
        
        return {
            "total_fusions": self.total_fusions,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_time,
            "llm_available": self.llm_client is not None
        }

# =========================================================================
# Global Intelligent Fusion Engine Instance
# =========================================================================
_fusion_engine: Optional[IntelligentFusionEngine] = None

async def get_fusion_engine() -> IntelligentFusionEngine:
    """Get the global intelligent fusion engine instance"""
    global _fusion_engine

    if _fusion_engine is None:
        _fusion_engine = IntelligentFusionEngine()
        await _fusion_engine.initialize()

    return _fusion_engine
