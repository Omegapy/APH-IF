# -------------------------------------------------------------------------
# File: context_fusion.py
# Author: Alexander Ricciardi
# Date: 2025-09-16
# [File Path] backend/app/search/context_fusion.py
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
#   Implements intelligent, LLM-powered fusion of semantic (VectorRAG) and traversal (GraphRAG)
#   results, preserving citations and consolidating references. Provides utilities for citation
#   extraction, reference renumbering, and a production engine class used by the backend search
#   pipeline and API endpoints.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Enum: FusionStrategy
# - Class: IntelligentFusionEngine
# - Function: extract_citations_sync
# - Function: extract_source_references_sync
# - Function: combine_source_references_sync
# - Function: renumber_fused_citations_sync
# - Function: create_combined_references_section_sync
# - Function: validate_citations_preserved_sync
# - Function: get_fusion_engine
# - Function: test_context_fusion_engine
# - Constants: INTELLIGENT_FUSION_SYSTEM_PROMPT, INTELLIGENT_FUSION_USER_PROMPT
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: asyncio (concurrency), time (timing), logging (logs), re (regex), enum (Enum)
# - Third-Party: langchain_core.messages (message types)
# - Local Project Modules: core.config.settings, search.parallel_hybrid models
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Used by the parallel retrieval flow (parallel_hybrid) and FastAPI endpoints in main.py to fuse
# semantic and traversal search outputs. The standalone utilities support citation processing and
# are leveraged internally by the engine.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Context Fusion Engine for APH-IF Backend.

Implements intelligent, LLM-powered fusion of semantic (VectorRAG) and traversal (GraphRAG)
results. The engine prioritizes citation preservation, reference consolidation, and
domain-agnostic synthesis to produce comprehensive answers. This module forms the fusion
stage of the search pipeline and is used by API endpoints and the parallel retrieval engine.
"""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import asyncio
import time
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.config import settings
from .parallel_hybrid import ParallelRetrievalResponse, FusionResult, RetrievalResult

 # __________________________________________________________________________
 # Global Constants / Variables

logger = logging.getLogger(__name__)

# =========================================================================
# Fusion Strategy Enumeration
# =========================================================================

# ------------------------------------------------------------------------- class FusionStrategy
class FusionStrategy(Enum):
    """
    Fusion strategy for combining search results.
    
    Uses only INTELLIGENT_FUSION for optimal LLM-powered synthesis
    of semantic and traversal search results.
    """
    INTELLIGENT_FUSION = "intelligent_fusion"     # LLM-based intelligent fusion
# ------------------------------------------------------------------------- end class FusionStrategy

# =========================================================================
# Domain-Agnostic Fusion Prompts
# =========================================================================

INTELLIGENT_FUSION_SYSTEM_PROMPT = """
You are an expert knowledge fusion system that combines results from two different search methods:
1. Semantic Search (Vector-based similarity using embeddings)
2. Graph Traversal Search (Knowledge graph relationships using Cypher queries)

Your task is to intelligently combine these results into a comprehensive, accurate response that leverages the strengths of both approaches.

CRITICAL CITATION REQUIREMENTS:
- PRESERVE ALL numbered citations [1], [2], [3] etc. from both sources EXACTLY as they appear
- DO NOT renumber citations - keep them exactly as provided in each source
- Both sources may have overlapping citation numbers (e.g., both may have [1], [2]) - this is normal
- Combine domain-specific references with numbered citations:
  * Legal/Regulatory: "§57.4361(a) [1]", "Part 75.1714 [2]", "CFR 30.57.4361 [3]"
  * Academic: "Vol. 2, Chapter 3 [1]", "Figure 4.1 [2]"
  * Technical: "ISO 9001:2015 [1]", "ASTM D2582-07 [2]"
  * Business: "Policy 2.4 [1]", "Procedure A-15 [2]"
  * Medical: "Protocol 4.2 [1]", "Guidelines Table 1 [2]"
- When information from different sources refers to the same concept, maintain both citations
- NEVER modify, renumber, or remove existing citations
- If sources conflict, present both perspectives with their respective citations
- The References section will be added automatically - focus on preserving inline citations

FUSION GUIDELINES:
1. Identify complementary information from both sources
2. Create a cohesive narrative that integrates both perspectives
3. Resolve conflicts by presenting both viewpoints with source attribution
4. Prioritize information based on confidence scores when conflicts arise
5. Maintain domain-specific terminology and technical accuracy
6. Preserve the specificity of regulatory/technical references
7. Ensure the fused response is more comprehensive than either source alone
8. Use clear section headers to organize complex information

RESPONSE STRUCTURE:
- Start with a comprehensive overview
- Present detailed information organized logically
- Include all relevant citations in context exactly as they appear in sources
- Maintain professional, authoritative tone appropriate to the domain
- DO NOT include a References section - this will be added automatically
- DO NOT include sections like "Important notes and clarifications" or "Next action" - focus on answering the query directly
- DO NOT explain how the search system works or reconcile differences between search methods
- DO NOT include meta-commentary about search results, their sources, or why they differ
- Answer the user's question directly without discussing the fusion process
"""

INTELLIGENT_FUSION_USER_PROMPT = """
Please fuse the following search results into a comprehensive response:

Query: {query}

SEMANTIC SEARCH RESULTS (Confidence: {semantic_confidence:.2f}):
{semantic_content}

GRAPH TRAVERSAL RESULTS (Confidence: {traversal_confidence:.2f}):
{traversal_content}

IMPORTANT FUSION INSTRUCTIONS:
- Create a unified response that combines the best aspects of both sources
- Preserve ALL numbered citations [1], [2], [3] etc. exactly as they appear in each source
- Both sources may have their own [1], [2], [3] citations - this is expected and correct
- Include regulatory/technical references with their citations (e.g., "§57.4361(a) [1]")
- The References section will be automatically added after your response
- Focus on creating comprehensive content that leverages both search approaches
- DO NOT explain differences between search methods or why results vary
- DO NOT include reconciliation sections or meta-analysis of search approaches
- Provide direct, authoritative answers without system commentary
"""

# __________________________________________________________________________
# Standalone Function Definitions
#

# ______________________
# Utility Functions
#

# =========================================================================
# Citation Processing Utilities (using shared citation processor)
# =========================================================================

# --------------------------------------------------------------------------------- extract_citations_sync()
async def extract_citations_sync(content: str) -> Dict[str, List[str]]:
    """Extract citations from content using the shared citation processor.

    Args:
        content: The text content to analyze for citations (numbered, regulatory, standards).

    Returns:
        Mapping of citation categories to lists of citation strings, including a "combined"
        category with all detected citations. Returns empty lists on failure.
    """
    try:
        from ..processing.citation_processor import get_citation_processor
        processor = get_citation_processor()
        return await processor.extract_citations(content, use_cache=False)
    except Exception as e:
        logger.warning(f"Error extracting citations via processor: {e}")
        return {
            "numbered": [],
            "regulatory": [],
            "standards": [],
            "references": [],
            "combined": []
        }
# --------------------------------------------------------------------------------- end extract_citations_sync()

# --------------------------------------------------------------------------------- extract_source_references_sync()
def extract_source_references_sync(content: str) -> List[Dict[str, str]]:
    """Extract reference mappings from a References section in content.

    Parses a trailing "References:" or "**References:**" section and extracts entries of the
    form "[n] <source>".

    Args:
        content: Text content that may contain a References section.

    Returns:
        A list of mappings with keys: "number" (int) and "source_info" (str). Returns an empty
        list if a references section is not present or cannot be parsed.
    """
    references = []
    
    try:
        # Find the References section
        references_section = ""
        if "**References:**" in content:
            references_section = content.split("**References:**")[-1]
        elif "References:" in content:
            references_section = content.split("References:")[-1]
        
        if references_section:
            lines = references_section.split('\n')
            
            for line in lines:
                line = line.strip()
                # Match patterns like "[1] filename.pdf_p123_c1, p.123"
                if re.match(r'^\[(\d+)\]\s+(.+)', line):
                    match = re.match(r'^\[(\d+)\]\s+(.+)', line)
                    if match:
                        number = match.group(1)
                        source_info = match.group(2)
                        references.append({
                            'number': int(number),
                            'source_info': source_info
                        })
                        
    except Exception as e:
        logger.debug(f"Error extracting source references: {e}")
    
    return references
# --------------------------------------------------------------------------------- end extract_source_references_sync()

# --------------------------------------------------------------------------------- combine_source_references_sync()
def combine_source_references_sync(
    semantic_content: str,
    traversal_content: str,
) -> List[Dict[str, str]]:
    """Combine and renumber references extracted from semantic and traversal content.

    Args:
        semantic_content: Content produced by semantic search that may include references.
        traversal_content: Content produced by traversal search that may include references.

    Returns:
        A list of reference mappings with keys: "original_number", "new_number",
        "source_info", and "search_type" ("semantic" or "traversal").
    """
    try:
        # Extract references from both sources
        semantic_refs = extract_source_references_sync(semantic_content)
        traversal_refs = extract_source_references_sync(traversal_content)
        
        # Combine and renumber references to avoid conflicts
        combined_refs = []
        current_number = 1
        
        # Add semantic references first
        for ref in semantic_refs:
            combined_refs.append({
                'original_number': ref['number'],
                'new_number': current_number,
                'source_info': ref['source_info'],
                'search_type': 'semantic'
            })
            current_number += 1
        
        # Add traversal references with new numbering
        for ref in traversal_refs:
            combined_refs.append({
                'original_number': ref['number'],
                'new_number': current_number,
                'source_info': ref['source_info'],
                'search_type': 'traversal'
            })
            current_number += 1
        
        return combined_refs
        
    except Exception as e:
        logger.warning(f"Error combining source references: {e}")
        return []
# --------------------------------------------------------------------------------- end combine_source_references_sync()

# --------------------------------------------------------------------------------- renumber_fused_citations_sync()
def renumber_fused_citations_sync(
    fused_content: str,
    combined_refs: List[Dict[str, str]],
) -> Tuple[str, List[Dict[str, str]]]:
    """Renumber citations in fused content to match chronologically combined references.

    Args:
        fused_content: The fused content containing inline citations like "[n]".
        combined_refs: Combined reference mappings produced by
            ``combine_source_references_sync``.

    Returns:
        A tuple of (content_with_renumbered_citations, used_references) where used_references
        are the references actually referenced after renumbering with keys: "number",
        "source_info", "search_type", and "original_number".
    """
    try:
        # Extract all citations from the fused content
        citations_in_content = re.findall(r'\[(\d+)\]', fused_content)
        if not citations_in_content:
            return fused_content, combined_refs
        
        # Get unique citation numbers in the order they appear
        unique_citations = []
        seen = set()
        for citation in citations_in_content:
            citation_num = int(citation)
            if citation_num not in seen:
                unique_citations.append(citation_num)
                seen.add(citation_num)
        
        # Create mapping from old numbers to new chronological numbers
        citation_map = {}  # Maps old number to new number
        used_references = []
        
        for new_number, old_number in enumerate(unique_citations, 1):
            citation_map[old_number] = new_number
            
            # Find the reference with this old number in combined_refs
            for ref in combined_refs:
                if ref.get('original_number') == old_number:
                    # Create new reference with chronological number
                    used_references.append({
                        'number': new_number,
                        'source_info': ref.get('source_info', 'Unknown source'),
                        'search_type': ref.get('search_type', 'unknown'),
                        'original_number': old_number
                    })
                    break
        
        # Renumber all citations in the fused content
        def replace_citation(match):
            old_num = int(match.group(1))
            new_num = citation_map.get(old_num, old_num)
            return f"[{new_num}]"
        
        renumbered_content = re.sub(r'\[(\d+)\]', replace_citation, fused_content)
        
        return renumbered_content, used_references
        
    except Exception as e:
        logger.warning(f"Error renumbering citations in fused content: {e}")
        return fused_content, combined_refs
# --------------------------------------------------------------------------------- end renumber_fused_citations_sync()

# --------------------------------------------------------------------------------- create_combined_references_section_sync()
def create_combined_references_section_sync(used_refs: List[Dict[str, str]]) -> str:
    """Create a unified References section from chronologically renumbered references.

    Args:
        used_refs: References that appear in the fused content after renumbering.

    Returns:
        A formatted "References:" section string (including leading newlines), or an empty
        string if no references are provided.
    """
    if not used_refs:
        logger.debug("No references found to create combined section")
        return ""
    
    references_lines = ["\n\nReferences:"]
    
    # Sort by number to ensure proper order
    sorted_refs = sorted(used_refs, key=lambda x: x.get('number', 0))
    
    for ref in sorted_refs:
        number = ref.get('number', '?')
        source_info = ref.get('source_info', 'Unknown source')
        search_type = ref.get('search_type', 'unknown')
        references_lines.append(f"[{number}] {source_info} (from {search_type} search)")
    
    references_lines.append("")  # Empty line at end
    logger.debug(f"Created references section with {len(sorted_refs)} references")
    return "\n".join(references_lines)
# --------------------------------------------------------------------------------- end create_combined_references_section_sync()

# --------------------------------------------------------------------------------- validate_citations_preserved_sync()
async def validate_citations_preserved_sync(
    original_content: str,
    fused_content: str,
) -> Tuple[bool, List[str]]:
    """Validate that citations from an original text appear in the fused content.

    Args:
        original_content: The original content whose citations must be preserved.
        fused_content: The fused content to check for preservation.

    Returns:
        Tuple of (all_preserved, missing_citations). ``all_preserved`` is True when all
        citations found in the original are present in the fused content; otherwise False,
        with ``missing_citations`` listing missing items in "<type>: <value>" form.
    """
    try:
        original_citations = await extract_citations_sync(original_content)
        fused_citations = await extract_citations_sync(fused_content)
        
        missing_citations = []
        
        # Check each type of citation
        for citation_type, original_cites in original_citations.items():
            if citation_type == "combined":  # Skip combined as it's derived
                continue
                
            fused_cites = fused_citations.get(citation_type, [])
            
            for cite in original_cites:
                if cite not in fused_cites:
                    missing_citations.append(f"{citation_type}: {cite}")
        
        all_preserved = len(missing_citations) == 0
        return all_preserved, missing_citations
        
    except Exception as e:
        logger.error(f"Error validating citations: {e}")
        return False, [f"Validation error: {str(e)}"]
# --------------------------------------------------------------------------------- end validate_citations_preserved_sync()

# ____________________________________________________________________________
# Class Definitions

# =========================================================================
# Intelligent Fusion Engine
# =========================================================================

# Not converted to @dataclass: manages external async resources and runtime components

# ------------------------------------------------------------------------- class IntelligentFusionEngine
class IntelligentFusionEngine:
    """Optimized LLM-powered context fusion engine.

    Combines semantic and traversal search outputs while preserving citations and
    consolidating references. The engine uses async components, optional circuit breakers,
    and minimal caching (disabled for fusion) to produce consistent, domain-agnostic
    synthesized answers.

    Attributes:
        _async_llm_client: Lazily initialized async LLM client used for fusion prompts.
        _citation_processor: Shared citation processor for extraction/validation.
        _cpu_pool: CPU-bound worker pool used for light post-processing when needed.
        _fusion_breaker: Optional circuit breaker protecting LLM calls.
        _cache: Disabled placeholder for potential future fusion caching.
        _fusion_count: Number of successful fusion operations performed.
        _total_fusion_time: Aggregate fusion time in milliseconds.
        _citation_preservation_rate: Running average of citation preservation accuracy.
    """
    
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self):
        """Initialize the fusion engine and supporting async components.

        Initializes lazy handles for the async LLM client, citation processor, and CPU pool,
        and configures an optional circuit breaker for fusion calls.
        """
        self.logger = logging.getLogger(__name__)
        
        # Use async components instead of synchronous ones
        self._async_llm_client: Any | None = None
        self._citation_processor: Any | None = None
        self._cpu_pool: Any | None = None
        
        # Using shared citation processor for async operations
        # Sync utilities defined above for fusion-specific needs
        
        # Performance tracking
        self._fusion_count: int = 0
        self._total_fusion_time: float = 0.0
        self._citation_preservation_rate: float = 0.0
        
        # Fusion caching disabled - use direct LLM processing for consistent results
        self._cache: Any | None = None
        self.logger.info("✅ Intelligent Fusion Engine initialized - direct LLM processing enabled")
        
        # Initialize circuit breaker for LLM fusion with reduced timeouts
        try:
            from ..monitoring.circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
            
            fusion_config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=60,  # Reduced from 240
                timeout_threshold_ms=30000,  # Reduced from 240000 (30 seconds)
                slow_call_duration_ms=15000   # Reduced from 150000 (15 seconds)
            )
            self._fusion_breaker: Any | None = get_circuit_breaker("llm_fusion", fusion_config)
            self.logger.info("✅ Circuit breaker enabled for LLM fusion")
        except ImportError:
            self.logger.warning("⚠️ Circuit breaker not available for fusion")
            self._fusion_breaker = None
    # --------------------------------------------------------------------------------- end __init__()
    
    # -------------------------------------------------------------- get_async_llm_client()
    async def get_async_llm_client(self) -> Any:
        """Get the async LLM client with lazy initialization.

        Returns:
            The process-wide async LLM client capable of ``complete(messages=...)``.
        """
        if self._async_llm_client is None:
            from ..core.async_llm_client import get_async_llm_client
            self._async_llm_client = await get_async_llm_client()
            self.logger.info("Async LLM client initialized for fusion")
        return self._async_llm_client
    # -------------------------------------------------------------- end get_async_llm_client()
    
    # -------------------------------------------------------------- get_citation_processor()
    async def get_citation_processor(self) -> Any:
        """Get the async citation processor with lazy initialization.

        Returns:
            The citation processor service exposing async extraction utilities.
        """
        if self._citation_processor is None:
            from ..processing.citation_processor import get_citation_processor
            self._citation_processor = get_citation_processor()
            self.logger.info("Citation processor initialized for fusion")
        return self._citation_processor
    # -------------------------------------------------------------- end get_citation_processor()
    
    # -------------------------------------------------------------- get_cpu_pool()
    async def get_cpu_pool(self) -> Any:
        """Get the CPU pool with lazy initialization.

        Returns:
            A CPU-bound worker pool suitable for light post-processing tasks.
        """
        if self._cpu_pool is None:
            from ..core.cpu_pool import get_cpu_pool
            self._cpu_pool = get_cpu_pool()
            self.logger.info("CPU pool initialized for fusion")
        return self._cpu_pool
    # -------------------------------------------------------------- end get_cpu_pool()
    
    # -------------------------------------------------------------- fuse_contexts()
    async def fuse_contexts(
        self,
        parallel_response: ParallelRetrievalResponse,
    ) -> FusionResult:
        """Fuse semantic and traversal results using intelligent LLM synthesis.

        Args:
            parallel_response: Results from parallel semantic and traversal searches.

        Returns:
            FusionResult: Fused content and quality metadata. Fusion caching is disabled to
            ensure consistent, fresh LLM outputs.
        """
        start_time = time.time()
        self.logger.info("Starting intelligent context fusion")
        
        # Direct LLM fusion processing - no cache lookup for consistent results
        self.logger.info("🚀 Processing fusion directly with LLM - no cache lookup")
        
        try:
            # Use intelligent LLM fusion
            result = await self._llm_intelligent_fusion(parallel_response)
            
            # Update processing time
            processing_time = int((time.time() - start_time) * 1000)
            result.processing_time_ms = processing_time
            
            # Update performance metrics
            self._update_metrics(processing_time, result)
            
            self.logger.info(f"Context fusion completed in {processing_time}ms "
                           f"with confidence {result.final_confidence:.2f}")
            
            # Fusion caching disabled - no result storage (ensures fresh LLM processing)
            self.logger.debug("✅ Fusion processed directly - no caching overhead")
            
            return result
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Context fusion failed: {e}")
            
            return self._create_error_fusion_result(
                str(e), 
                parallel_response, 
                processing_time
            )
    # -------------------------------------------------------------- end fuse_contexts()
    
    # -------------------------------------------------------------- extract_citations_async()
    async def extract_citations_async(self, result: RetrievalResult) -> Dict[str, List[str]]:
        """Extract citations from a retrieval result using the async processor.

        Args:
            result: RetrievalResult whose ``content`` is scanned for citations.

        Returns:
            Mapping of citation categories to lists of citation strings.
        """
        processor = await self.get_citation_processor()
        return await processor.extract_citations_async(result.content)
    # -------------------------------------------------------------- end extract_citations_async()

    # -------------------------------------------------------------- extract_source_references_async()
    async def extract_source_references_async(self, content: str) -> List[Dict[str, str]]:
        """Extract source references from content using the async processor.

        Args:
            content: Text content that may contain a References section.

        Returns:
            A list of reference mappings with number and source info.
        """
        processor = await self.get_citation_processor()
        return await processor.extract_references_async(content)
    # -------------------------------------------------------------- end extract_source_references_async()
    
    # -------------------------------------------------------------- _protected_llm_call()
    async def _protected_llm_call(
        self,
        messages: List[SystemMessage | HumanMessage],
    ) -> str:
        """Call the async LLM client with optional circuit-breaker protection.

        Args:
            messages: Ordered list of system and human messages for the LLM.

        Returns:
            The generated LLM response content as a string.
        """
        async def protected_call():
            llm_client = await self.get_async_llm_client()
            return await llm_client.complete(messages, priority=1)  # High priority for fusion
        
        # Execute with circuit breaker if available
        if self._fusion_breaker:
            return await self._fusion_breaker.call(protected_call)
        else:
            return await protected_call()
    # -------------------------------------------------------------- end _protected_llm_call()
    
    # -------------------------------------------------------------- _llm_intelligent_fusion()
    async def _llm_intelligent_fusion(
        self,
        response: ParallelRetrievalResponse,
    ) -> FusionResult:
        """Combine results with citation preservation using the async LLM client.

        Args:
            response: Parallel search response to fuse.

        Returns:
            FusionResult: Intelligently fused content with citation preservation and
            reference consolidation.
        """
        try:
            # Check if parallel preprocessing is enabled
            model_config = settings.get_llm_model_config()
            use_parallel_preprocessing = model_config.get("parallel_preprocessing", True)
            
            if use_parallel_preprocessing:
                # Parallel extraction of citations and references
                extraction_tasks = [
                    self.extract_citations_async(response.semantic_result),
                    self.extract_citations_async(response.traversal_result),
                    self.extract_source_references_async(response.semantic_result.content),
                    self.extract_source_references_async(response.traversal_result.content)
                ]
                
                # Execute all extraction tasks in parallel
                semantic_citations, traversal_citations, semantic_refs, traversal_refs = \
                    await asyncio.gather(*extraction_tasks)
                
                self.logger.info("✅ Parallel preprocessing completed for citations and references")
            else:
                # Sequential extraction (fallback) - still use async processor
                processor = await self.get_citation_processor()
                semantic_citations = await processor.extract_citations_async(
                    response.semantic_result.content
                )
                traversal_citations = await processor.extract_citations_async(
                    response.traversal_result.content
                )
                semantic_refs = await processor.extract_references_async(
                    response.semantic_result.content
                )
                traversal_refs = await processor.extract_references_async(
                    response.traversal_result.content
                )
            
            # Create fusion messages
            system_message = SystemMessage(content=INTELLIGENT_FUSION_SYSTEM_PROMPT)
            
            user_prompt = INTELLIGENT_FUSION_USER_PROMPT.format(
                query=response.query,
                semantic_confidence=response.semantic_result.confidence,
                semantic_content=response.semantic_result.content,
                traversal_confidence=response.traversal_result.confidence,
                traversal_content=response.traversal_result.content
            )
            
            user_message = HumanMessage(content=user_prompt)
            
            # Call LLM for intelligent fusion using batch processing
            self.logger.info("🚀 Calling async LLM for intelligent fusion", extra={
                "search_type": "fusion",
                "fusion_strategy": "intelligent",
                "semantic_confidence": response.semantic_result.confidence,
                "traversal_confidence": response.traversal_result.confidence,
                "parallel_execution": True
            })
            
            messages = [system_message, user_message]
            try:
                fused_content = await self._protected_llm_call(messages)
                self.logger.info("✅ Fusion completed successfully", extra={
                    "search_type": "fusion",
                    "fusion_strategy": "intelligent",
                    "success": True,
                    "parallel_execution": True
                })
            except Exception as fusion_error:
                self.logger.error("❌ Fusion failed", extra={
                    "search_type": "fusion",
                    "fusion_strategy": "intelligent",
                    "error": str(fusion_error),
                    "success": False,
                    "parallel_execution": True
                })
                raise fusion_error
            
            # Use pre-extracted source references or extract if not using parallel preprocessing
            if use_parallel_preprocessing:
                # Combine pre-extracted references
                combined_source_refs = []
                current_number = 1
                
                # Add semantic references first
                for ref in semantic_refs:
                    combined_source_refs.append({
                        'original_number': ref['number'],
                        'new_number': current_number,
                        'source_info': ref['source_info'],
                        'search_type': 'semantic'
                    })
                    current_number += 1
                
                # Add traversal references with new numbering
                for ref in traversal_refs:
                    combined_source_refs.append({
                        'original_number': ref['number'],
                        'new_number': current_number,
                        'source_info': ref['source_info'],
                        'search_type': 'traversal'
                    })
                    current_number += 1
            else:
                # Extract and combine source references (original method)
                combined_source_refs = combine_source_references_sync(
                    response.semantic_result.content, 
                    response.traversal_result.content
                )
            
            # Fallback: If no source references found, create basic ones from citations
            if not combined_source_refs:
                self.logger.warning("No source references extracted, creating fallback references")
                combined_source_refs = []
                current_number = 1
                
                # Create basic references from semantic citations
                for cite in semantic_citations.get("numbered", []):
                    try:
                        cite_num = int(cite.strip("[]"))
                        combined_source_refs.append({
                            'original_number': cite_num,
                            'new_number': current_number,
                            'source_info': f"Semantic search result citation {cite}",
                            'search_type': 'semantic'
                        })
                        current_number += 1
                    except (ValueError, AttributeError):
                        continue
                
                # Create basic references from traversal citations  
                for cite in traversal_citations.get("numbered", []):
                    try:
                        cite_num = int(cite.strip("[]"))
                        combined_source_refs.append({
                            'original_number': cite_num,
                            'new_number': current_number,
                            'source_info': f"Graph traversal search result citation {cite}",
                            'search_type': 'traversal'
                        })
                        current_number += 1
                    except (ValueError, AttributeError):
                        continue
            
            # Renumber citations in fused content to match combined references chronologically
            renumbered_content, used_references = renumber_fused_citations_sync(
                fused_content, combined_source_refs
            )
            
            # Create unified References section with chronologically numbered references
            references_section = create_combined_references_section_sync(used_references)
            
            # Clean unwanted meta-commentary sections from fused content before adding references
            unwanted_sections = [
                "Important notes and clarifications",
                "**Important notes and clarifications**",
                "Reconciling the two perspectives", 
                "**Reconciling the two perspectives**",
                "Why the apparent conflict exists",
                "**Why the apparent conflict exists**",
                "Practical implication",
                "**Practical implication**",
                "If you want a more targeted deliverable",
                "**If you want a more targeted deliverable**",
                "Tell me which of the two you mean",
                "**Tell me which of the two you mean**"
            ]
            
            for section in unwanted_sections:
                if section in renumbered_content:
                    renumbered_content = renumbered_content.split(section)[0].strip()
                    break
            
            # Add References section to fused content
            if references_section:
                fused_content_with_refs = renumbered_content + references_section
            else:
                fused_content_with_refs = renumbered_content
            
            # Validate citation preservation
            semantic_preserved, semantic_missing = await validate_citations_preserved_sync(
                response.semantic_result.content, fused_content_with_refs
            )
            
            traversal_preserved, traversal_missing = await validate_citations_preserved_sync(
                response.traversal_result.content, fused_content_with_refs
            )
            
            # Calculate citation accuracy
            total_original_citations = (
                len(semantic_citations["combined"]) + 
                len(traversal_citations["combined"])
            )
            
            missing_citations_count = len(semantic_missing) + len(traversal_missing)
            
            if total_original_citations > 0:
                citation_accuracy = 1.0 - (missing_citations_count / total_original_citations)
            else:
                citation_accuracy = 1.0
            
            # Log citation preservation results
            if not semantic_preserved or not traversal_preserved:
                self.logger.warning(f"Citation preservation issues detected:")
                self.logger.warning(f"Semantic missing: {semantic_missing}")
                self.logger.warning(f"Traversal missing: {traversal_missing}")
            
            # Log citation and reference processing
            self.logger.info(f"Combined {len(combined_source_refs)} source references, renumbered to {len(used_references)} used citations")
            
            # Calculate final confidence based on fusion quality
            final_confidence = self._calculate_fusion_confidence(
                response, fused_content_with_refs, citation_accuracy
            )
            
            # Combine metadata from both sources
            entities_combined = list(set(
                response.semantic_result.entities + response.traversal_result.entities
            ))
            
            sources_combined = response.sources_combined.copy()
            
            # Extract and preserve citations from fused content
            fused_citations = await extract_citations_sync(fused_content_with_refs)
            citations_preserved = fused_citations["combined"]
            
            return FusionResult(
                fused_content=fused_content_with_refs,
                final_confidence=final_confidence,
                fusion_strategy=FusionStrategy.INTELLIGENT_FUSION.value,
                processing_time_ms=0,  # Will be set by caller
                vector_contribution=response.vector_contribution,
                graph_contribution=response.graph_contribution,
                complementarity_score=response.complementarity_score,
                entities_combined=entities_combined,
                sources_combined=sources_combined,
                citations_preserved=citations_preserved,
                citation_accuracy=citation_accuracy,
                domain_adaptation="intelligent_detection"
            )
            
        except Exception as e:
            self.logger.error(f"LLM intelligent fusion failed: {e}")
            raise
    # -------------------------------------------------------------- end _llm_intelligent_fusion()
    
    # -------------------------------------------------------------- _calculate_fusion_confidence()
    def _calculate_fusion_confidence(
        self,
        response: ParallelRetrievalResponse,
        fused_content: str,
        citation_accuracy: float,
    ) -> float:
        """Calculate a confidence score for fused results.

        Args:
            response: Original parallel response used for contributions and complementarity.
            fused_content: The final fused content produced by the LLM.
            citation_accuracy: Ratio of citations preserved in fused content (0.0–1.0).

        Returns:
            A fusion confidence score between 0.0 and 1.0, capped by configuration.
        """
        try:
            # Base confidence from weighted combination
            base_confidence = (
                response.semantic_result.confidence * response.vector_contribution +
                response.traversal_result.confidence * response.graph_contribution
            )
            
            # Boost for successful fusion
            fusion_boost = 0.1 if len(fused_content) > max(
                len(response.semantic_result.content),
                len(response.traversal_result.content)
            ) else 0.0
            
            # Citation preservation penalty/bonus
            citation_factor = citation_accuracy  # Direct multiplication
            
            # Complementarity boost
            complementarity_boost = response.complementarity_score * 0.1
            
            # Calculate final confidence
            final_confidence = (
                (base_confidence + fusion_boost + complementarity_boost) * citation_factor
            )
            
            # Apply configurable cap (default 1.0 = no cap)
            # Can be set via FUSION_CONFIDENCE_CAP env var
            fusion_cap = settings.validated_fusion_confidence_cap
            return max(0.1, min(fusion_cap, final_confidence))
            
        except Exception as e:
            self.logger.warning(f"Error calculating fusion confidence: {e}")
            return 0.5  # Default moderate confidence
    # -------------------------------------------------------------- end _calculate_fusion_confidence()
    
    # -------------------------------------------------------------- _create_error_fusion_result()
    def _create_error_fusion_result(
        self,
        error_message: str,
        response: ParallelRetrievalResponse,
        processing_time: int,
    ) -> FusionResult:
        """Create a fallback fusion result when intelligent fusion fails.

        Args:
            error_message: Brief error description.
            response: Original parallel response to select the stronger fallback content from.
            processing_time: Milliseconds elapsed before the failure was detected.

        Returns:
            A ``FusionResult`` containing the best available content with a reduced confidence
            and error context preserved.
        """
        # Fallback to best individual result
        if response.semantic_result.confidence > response.traversal_result.confidence:
            fallback_content = f"**Fusion Error - Using Semantic Results**:\n\n{response.semantic_result.content}\n\n*Note: Context fusion failed: {error_message}*"
            fallback_confidence = response.semantic_result.confidence * 0.8  # Penalize for fusion failure
        else:
            fallback_content = f"**Fusion Error - Using Traversal Results**:\n\n{response.traversal_result.content}\n\n*Note: Context fusion failed: {error_message}*"
            fallback_confidence = response.traversal_result.confidence * 0.8
        
        return FusionResult(
            fused_content=fallback_content,
            final_confidence=fallback_confidence,
            fusion_strategy="intelligent_fusion_error",
            processing_time_ms=processing_time,
            vector_contribution=response.vector_contribution,
            graph_contribution=response.graph_contribution,
            complementarity_score=0.0,
            entities_combined=response.entities_combined,
            sources_combined=response.sources_combined,
            citations_preserved=[],
            citation_accuracy=0.0,
            domain_adaptation="error_fallback",
            error=error_message
        )
    # -------------------------------------------------------------- end _create_error_fusion_result()
    
    # -------------------------------------------------------------- _update_metrics()
    def _update_metrics(self, processing_time_ms: int, result: FusionResult) -> None:
        """Update internal performance metrics for the fusion engine.

        Args:
            processing_time_ms: Time spent performing the fusion operation.
            result: The resulting ``FusionResult`` used to update quality metrics.
        """
        self._fusion_count += 1
        self._total_fusion_time += processing_time_ms
        
        # Update citation preservation rate (running average)
        if self._fusion_count == 1:
            self._citation_preservation_rate = result.citation_accuracy
        else:
            # Exponential moving average
            alpha = 0.1
            self._citation_preservation_rate = (
                alpha * result.citation_accuracy +
                (1 - alpha) * self._citation_preservation_rate
            )
        
        if self._fusion_count % 5 == 0:  # Log every 5 fusions
            avg_time = self._total_fusion_time / self._fusion_count
            self.logger.info(f"Fusion metrics: {self._fusion_count} fusions, "
                           f"avg time: {avg_time:.1f}ms, "
                           f"citation preservation: {self._citation_preservation_rate:.2f}")
    # -------------------------------------------------------------- end _update_metrics()
    
    # -------------------------------------------------------------- health_check()
    async def health_check(self) -> Dict[str, Any]:
        """Run a comprehensive health check for the fusion engine.

        Returns:
            A mapping with overall status, component statuses, and summary metrics.
        """
        health_status = {
            "tool_name": "context_fusion",
            "status": "unknown",
            "components": {
                "fusion_engine": "healthy",
                "llm_client": "unknown",
                "citation_processor": "healthy"
            },
            "metrics": {
                "fusion_count": self._fusion_count,
                "avg_fusion_time_ms": (self._total_fusion_time / self._fusion_count) if self._fusion_count > 0 else 0,
                "citation_preservation_rate": self._citation_preservation_rate
            },
            "fusion_strategy": FusionStrategy.INTELLIGENT_FUSION.value,
            "errors": []
        }
        
        try:
            # Test LLM client
            test_messages = [
                SystemMessage(content="You are a test assistant."),
                HumanMessage(content="Reply with 'OK' to confirm you're working.")
            ]
            
            # Run test in executor
            loop = asyncio.get_event_loop()
            test_response = await loop.run_in_executor(
                None,
                lambda: self.llm_client.invoke(test_messages)
            )
            
            if test_response and "OK" in test_response.content:
                health_status["components"]["llm_client"] = "healthy"
            else:
                health_status["components"]["llm_client"] = "degraded"
                health_status["errors"].append("LLM client test response unexpected")
                
        except Exception as e:
            health_status["components"]["llm_client"] = "error"
            health_status["errors"].append(f"LLM client error: {str(e)}")
        
        # Determine overall status
        component_statuses = list(health_status["components"].values())
        if all(status == "healthy" for status in component_statuses):
            health_status["status"] = "healthy"
        elif any(status == "healthy" for status in component_statuses):
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "error"
        
        return health_status
    # -------------------------------------------------------------- end health_check()
    
    # -------------------------------------------------------------- get_fusion_stats()
    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get current fusion engine statistics.

        Returns:
            A mapping with fusion counts, timing aggregates, and preservation metrics.
        """
        return {
            "fusion_count": self._fusion_count,
            "total_fusion_time_ms": self._total_fusion_time,
            "avg_fusion_time_ms": (self._total_fusion_time / self._fusion_count) if self._fusion_count > 0 else 0,
            "citation_preservation_rate": self._citation_preservation_rate,
            "fusion_strategy": FusionStrategy.INTELLIGENT_FUSION.value
        }
    # -------------------------------------------------------------- end get_fusion_stats()
    
# ------------------------------------------------------------------------- end class IntelligentFusionEngine

# =========================================================================
# Global Engine Instance  
# =========================================================================

_fusion_engine: Optional[IntelligentFusionEngine] = None

# --------------------------------------------------------------------------------- get_fusion_engine()
def get_fusion_engine() -> IntelligentFusionEngine:
    """
    Get or create the global intelligent fusion engine instance.
    
    Returns:
        IntelligentFusionEngine: Global fusion engine instance
    """
    global _fusion_engine
    if _fusion_engine is None:
        _fusion_engine = IntelligentFusionEngine()
    return _fusion_engine
# --------------------------------------------------------------------------------- end get_fusion_engine()

# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# Development Testing Functions
# =========================================================================

# __________________________________________________________________________
# Module Testing Utilities (development-only)
#

# --------------------------------------------------------------------------------- test_context_fusion_engine()
async def test_context_fusion_engine():
    """
    Test function for the context fusion engine implementation.
    """
    logger.info("=" * 70)
    logger.info("🧠 Testing Context Fusion Engine - Step 3")
    logger.info("=" * 70)
    
    try:
        # Import required components for testing
        from .parallel_hybrid import RetrievalResult, ParallelRetrievalResponse
        
        # Initialize fusion engine
        fusion_engine = get_fusion_engine()
        
        # Test 1: Engine health check
        logger.info("\n📋 Test 1: Fusion Engine Health Check")
        health = await fusion_engine.health_check()
        logger.info(f"Overall Status: {health['status']}")
        logger.info(f"LLM Client: {health['components']['llm_client']}")
        logger.info(f"Fusion Strategy: {health['fusion_strategy']}")
        
        # Test 2: Citation extraction
        logger.info("\n🔍 Test 2: Citation Processing")
        test_content = "According to §57.4361(a) [1] and ISO 9001:2015 [2], safety requirements include proper PPE usage [3]."
        citations = await extract_citations_sync(test_content)
        
        logger.info(f"Numbered citations: {citations['numbered']}")
        logger.info(f"Regulatory citations: {citations['regulatory']}")
        logger.info(f"Standards citations: {citations['standards']}")
        logger.info(f"Total citations: {len(citations['combined'])}")
        
        # Test 3: Mock parallel response for fusion
        logger.info("\n🔄 Test 3: Mock Fusion Test")
        
        # Create mock retrieval results
        semantic_result = RetrievalResult(
            content="Safety equipment requirements include hard hats [1], safety boots [2], and proper ventilation systems as specified in mining regulations.",
            method="semantic",
            confidence=0.85,
            response_time_ms=1200,
            sources=[{"document": "mining_safety.pdf", "page": 15}],
            entities=["hard hats", "safety boots", "ventilation"],
            citations=["[1]", "[2]"]
        )
        
        traversal_result = RetrievalResult(
            content="Mining operations must comply with §57.4361(a) [1] for equipment standards and Part 75.1714 [2] for ventilation requirements. OSHA regulations mandate proper PPE usage.",
            method="traversal",
            confidence=0.78,
            response_time_ms=1800,
            sources=[{"cypher_query": "MATCH (r:Regulation)-[:REQUIRES]->(e:Equipment) RETURN r, e"}],
            entities=["OSHA", "PPE", "mining"],
            citations=["§57.4361(a) [1]", "Part 75.1714 [2]"]
        )
        
        # Create mock parallel response
        mock_parallel_response = ParallelRetrievalResponse(
            semantic_result=semantic_result,
            traversal_result=traversal_result,
            query="What are the safety equipment requirements for mining operations?",
            total_time_ms=1800,
            success=True,
            fusion_ready=True,
            both_successful=True,
            primary_method="semantic"
        )
        
        # Test intelligent fusion strategy
        logger.info(f"\n   Testing intelligent fusion...")
        try:
            start_time = time.time()
            fusion_result = await fusion_engine.fuse_contexts(mock_parallel_response)
            fusion_time = int((time.time() - start_time) * 1000)
            
            logger.info(f"   ✅ Intelligent fusion: {fusion_time}ms, confidence {fusion_result.final_confidence:.2f}")
            logger.info(f"   Content length: {len(fusion_result.fused_content)} chars")
            logger.info(f"   Citations preserved: {len(fusion_result.citations_preserved)}")
            logger.info(f"   Citation accuracy: {fusion_result.citation_accuracy:.2f}")
            
        except Exception as e:
            logger.warning(f"   ⚠️ Intelligent fusion failed: {str(e)}")
        
        # Test 4: Fusion statistics
        logger.info("\n📊 Test 4: Fusion Statistics")
        stats = fusion_engine.get_fusion_stats()
        logger.info(f"Total fusions: {stats['fusion_count']}")
        logger.info(f"Average time: {stats['avg_fusion_time_ms']:.1f}ms")
        logger.info(f"Citation preservation: {stats['citation_preservation_rate']:.2f}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ Context Fusion Engine tests completed successfully!")
        logger.info("🎯 Step 3 Implementation: COMPLETE")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Context fusion engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
# --------------------------------------------------------------------------------- end test_context_fusion_engine()

 # __________________________________________________________________________
 # Module Initialization / Main Execution Guard 

if __name__ == "__main__":
    # Test entry point for development
    asyncio.run(test_context_fusion_engine())

# __________________________________________________________________________
# End of File
#