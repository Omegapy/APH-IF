# -------------------------------------------------------------------------
# File: llm_structural_cypher.py
# Author: Alexander Ricciardi
# Date: 2025-09-15
# [File Path] backend/app/search/tools/llm_structural_cypher.py
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
#   Core engine for LLM-powered NL→Cypher generation using cached structural schema summaries,
#   comprehensive validation against the full schema, and read-only execution via the schema
#   manager gateway. Provides narrative generation with citation validation, domain markers, and
#   rich metrics for observability.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Dataclass: GeneratedCypher
# - Dataclass: ExecutionResult
# - Dataclass: EngineMetrics
# - Class: LLMStructuralCypherEngine (generation/validation/execution/narrative)
# - Function: get_llm_structural_cypher_engine (singleton factory)
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: asyncio, time, logging, re, dataclasses, datetime, threading, typing
# - Third-Party: langchain_core.messages (SystemMessage, HumanMessage)
# - Local Project Modules: settings, async_llm_client, schema_manager, schema_models,
#   prompts.structural_cypher, cypher_validator, monitoring (optional)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# - Higher-level tools and API endpoints call into `LLMStructuralCypherEngine` via the factory
#   `get_llm_structural_cypher_engine()` to perform NL→Cypher traversal with validation and
#   narrative summarization that enforces citation correctness.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
LLM Structural Cypher Engine for APH-IF Backend

Core engine that orchestrates LLM-powered natural language to Cypher generation
using cached structural schema summaries for token-efficient prompting and
comprehensive validation against full schema for safety and correctness.
"""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from ...core.async_llm_client import AsyncLLMClient, get_async_llm_client
from ...core.config import settings
from ...schema.schema_manager import get_schema_manager
from ...schema.schema_models import CompleteKGSchema
from .cypher_validator import ValidationReport, get_cypher_validator
from .prompts.structural_cypher import (
    PromptComponents,
    get_structural_cypher_prompt_builder,
    get_structural_narrative_prompt_builder,
)

# Import observability components
try:
    from ...monitoring.timing_collector import get_timing_collector
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False


logger = logging.getLogger(__name__)

# __________________________________________________________________________
# Global Constants / Variables

# Global engine instance
_engine_instance: Optional['LLMStructuralCypherEngine'] = None
_engine_lock = Lock()

# ____________________________________________________________________________
# Class Definitions

# =========================================================================
# Data Models
# =========================================================================

# ------------------------------------------------------------------------- class GeneratedCypher
@dataclass
class GeneratedCypher:
    """Container for LLM-generated Cypher with metadata."""
    cypher: str
    confidence: float
    reasoning: Optional[str] = None
    generation_time_ms: int = 0
    tokens_used: int = 0
    model_used: str = ""
# ------------------------------------------------------------------------- end class GeneratedCypher

# ------------------------------------------------------------------------- class ExecutionResult
@dataclass
class ExecutionResult:
    """Container for Cypher execution results."""
    success: bool
    results: List[Dict[str, Any]]
    result_count: int
    execution_time_ms: int
    error_message: Optional[str] = None
    cypher_executed: str = ""
# ------------------------------------------------------------------------- end class ExecutionResult

# ------------------------------------------------------------------------- class EngineMetrics
@dataclass
class EngineMetrics:
    """Enhanced metrics collection for engine performance and citation tracking."""
    total_generations: int = 0
    successful_generations: int = 0
    validation_failures: int = 0
    execution_failures: int = 0
    total_execution_time_ms: int = 0
    
    # NEW: Citation and injection tracking metrics
    narrative_attempts: int = 0
    narrative_successes: int = 0
    narrative_failures: int = 0
    citations_validated: int = 0
    citations_dropped: int = 0  # Invented citations removed
    references_sections_stripped: int = 0  # LLM added unwanted sections
    injection_skipped_aggregation: int = 0
    injection_skipped_scope: int = 0
    fields_injected_total: int = 0
    avg_sources_per_narrative: float = 0.0
    avg_citations_per_answer: float = 0.0
    
    # Domain marker tracking metrics
    domain_markers_found_legal: int = 0
    domain_markers_found_academic: int = 0
    domain_markers_found_technical: int = 0
    domain_markers_found_business: int = 0
    domain_markers_found_medical: int = 0
    answers_with_domain_citations: int = 0  # Answers where domain text appears near [n]
    domain_marker_extraction_errors: int = 0
    
    
    # --------------------------------------------------------------------------------- __post_init__()
    def __post_init__(self) -> None:
        self.reset_timestamp = time.time()
    # --------------------------------------------------------------------------------- end __post_init__()
    
    @property
    # --------------------------------------------------------------------------------- success_rate()
    def success_rate(self) -> float:
        """Calculate generation success rate."""
        if self.total_generations == 0:
            return 0.0
        return self.successful_generations / self.total_generations
    # --------------------------------------------------------------------------------- end success_rate()
    
    @property
    # --------------------------------------------------------------------------------- narrative_success_rate()
    def narrative_success_rate(self) -> float:
        """Calculate narrative generation success rate."""
        if self.narrative_attempts == 0:
            return 0.0
        return self.narrative_successes / self.narrative_attempts
    # --------------------------------------------------------------------------------- end narrative_success_rate()
    
    @property
    # --------------------------------------------------------------------------------- avg_execution_time_ms()
    def avg_execution_time_ms(self) -> float:
        """Calculate average execution time."""
        if self.successful_generations == 0:
            return 0.0
        return self.total_execution_time_ms / self.successful_generations
    # --------------------------------------------------------------------------------- end avg_execution_time_ms()
    
    @property
    # --------------------------------------------------------------------------------- injection_skip_rate()
    def injection_skip_rate(self) -> float:
        """Calculate field injection skip rate."""
        total_skips = self.injection_skipped_aggregation + self.injection_skipped_scope
        total_attempts = self.total_generations
        if total_attempts == 0:
            return 0.0
        return total_skips / total_attempts
    # --------------------------------------------------------------------------------- end injection_skip_rate()

# ------------------------------------------------------------------------- end class EngineMetrics

# =========================================================================
# Core LLM Structural Cypher Engine
# =========================================================================

# ------------------------------------------------------------------------- class LLMStructuralCypherEngine
class LLMStructuralCypherEngine:
    """
    Core engine for LLM-powered NL→Cypher generation with structural schema optimization.
    
    Features:
    - Token-efficient prompting using structural schema summaries
    - Comprehensive validation against full schema
    - Smart rewrites and fallback mechanisms
    - Performance monitoring and metrics collection
    - Integration with existing APH-IF infrastructure
    """
    
    # ______________________
    # Constructor 
    # 
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self) -> None:
        """Initialize the LLM Structural Cypher Engine."""
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.schema_manager = get_schema_manager()
        self.prompt_builder = get_structural_cypher_prompt_builder(
            token_budget=getattr(settings, 'llm_cypher_prompt_token_budget', 3500),
            examples_enabled=getattr(settings, 'llm_cypher_examples_enabled', True)
        )
        self.validator = get_cypher_validator(
            max_hops=getattr(settings, 'llm_cypher_max_hops', 3),
            force_limit=getattr(settings, 'llm_cypher_force_limit', 50),
            allow_call=getattr(settings, 'llm_cypher_allow_call', False)
        )
        
        # LLM client (initialized lazily)
        self._llm_client: Optional[AsyncLLMClient] = None
        
        # Performance metrics
        self.metrics = EngineMetrics()
        
        # Configuration
        self.enabled = getattr(settings, 'use_llm_structural_cypher', True)
        
        self.logger.info("LLM Structural Cypher Engine initialized")
    # --------------------------------------------------------------------------------- end __init__()
    
    # -------------------------------------------------------------- _get_llm_client()
    async def _get_llm_client(self) -> AsyncLLMClient:
        """Get async LLM client (lazy initialization)."""
        client = self._llm_client
        if client is None:
            client = await get_async_llm_client()
            self._llm_client = client
        return client
    # -------------------------------------------------------------- end _get_llm_client()
    
    # -------------------------------------------------------------- _execute_read_query()
    def _execute_read_query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute read-only query via schema manager gateway."""
        return self.schema_manager.execute_read(cypher, params or {})
    # -------------------------------------------------------------- end _execute_read_query()
    
    # -------------------------------------------------------------- generate_cypher()
    async def generate_cypher(
        self,
        user_query: str,
        max_results: int = 50,
        hop_cap: Optional[int] = None
    ) -> GeneratedCypher:
        """
        Generate Cypher query from natural language using LLM and structural schema.
        
        Args:
            user_query: Natural language query from user
            max_results: Maximum number of results to return
            hop_cap: Maximum relationship hops (overrides config default)
            
        Returns:
            GeneratedCypher with query and metadata
        """
        start_time = time.time()
        
        # Initialize timing collector if available
        timing_collector = None
        if OBSERVABILITY_AVAILABLE:
            try:
                timing_collector = get_timing_collector()
            except Exception:
                pass  # Fallback to no timing if collector unavailable
        
        try:
            self.metrics.total_generations += 1
            
            if not self.enabled:
                raise ValueError("LLM Structural Cypher generation is disabled")
            
            # Schema acquisition with timing
            if timing_collector:
                async with timing_collector.measure("llm_cypher_schema_acquisition", {
                    "user_query_length": len(user_query),
                    "max_results": max_results
                }):
                    structural_summary = self.schema_manager.get_structural_summary_dict()
            else:
                structural_summary = self.schema_manager.get_structural_summary_dict()
            
            if not structural_summary:
                raise ValueError("Structural schema summary not available")
            
            # Prompt building with timing
            effective_hop_cap = hop_cap or getattr(settings, 'llm_cypher_max_hops', 3)
            if timing_collector:
                async with timing_collector.measure("llm_cypher_prompt_building", {
                    "schema_elements": len(structural_summary.get("node_labels", []) + 
                                         structural_summary.get("relationship_types", [])),
                    "hop_cap": effective_hop_cap
                }):
                    prompt_components = self.prompt_builder.build_prompt(
                        user_query=user_query,
                        structural_summary=structural_summary,
                        max_results=max_results,
                        max_hops=effective_hop_cap
                    )
            else:
                prompt_components = self.prompt_builder.build_prompt(
                    user_query=user_query,
                    structural_summary=structural_summary,
                    max_results=max_results,
                    max_hops=effective_hop_cap
                )
            
            self.logger.info(f"Built prompt: {prompt_components.total_estimated_tokens} tokens, "
                           f"truncation: {prompt_components.truncation_applied}")
            
            # LLM generation with enhanced timing
            llm_client = await self._get_llm_client()
            messages = [
                SystemMessage(content=prompt_components.system_prompt),
                HumanMessage(content=prompt_components.user_prompt)
            ]
            
            if timing_collector:
                async with timing_collector.measure("llm_cypher_generation", {
                    "tokens_estimated": prompt_components.total_estimated_tokens,
                    "model": settings.openai_model_mini,
                    "truncation_applied": prompt_components.truncation_applied
                }) as llm_timer:
                    llm_start = time.time()
                    response = await llm_client.complete(messages)
                    llm_time_ms = int((time.time() - llm_start) * 1000)
                    
                    llm_timer.add_metadata({
                        "llm_response_time_ms": llm_time_ms,
                        "response_length": len(response or "")
                    })
            else:
                llm_start = time.time()
                response = await llm_client.complete(messages)
                llm_time_ms = int((time.time() - llm_start) * 1000)
            
            # Guard against empty/None response
            if not response:
                return GeneratedCypher(
                    cypher="",
                    confidence=0.0,
                    reasoning="LLM returned empty response",
                    generation_time_ms=llm_time_ms,
                    tokens_used=0,
                    model_used=settings.openai_model_mini
                )
            
            # Extract Cypher from response
            cypher = self._extract_cypher_from_response(response)
            if not cypher:
                return GeneratedCypher(
                    cypher="",
                    confidence=0.0,
                    reasoning="No Cypher code block found in LLM response",
                    generation_time_ms=llm_time_ms,
                    tokens_used=len(response.split()) if isinstance(response, str) else 0,
                    model_used=settings.openai_model_mini
                )
            
            # Inject LIMIT if missing (prefer injection over rejection)
            cypher = self._ensure_limit_clause(cypher, max_results)

            # Inject chunk reference fields for citations (with safety guards)
            cypher = self._inject_chunk_return_fields(cypher)
            
            # Calculate confidence based on generation quality
            confidence = self._calculate_generation_confidence(
                cypher, 
                prompt_components, 
                response
            )
            
            generation_time_ms = int((time.time() - start_time) * 1000)
            
            self.metrics.successful_generations += 1
            self.metrics.total_execution_time_ms += generation_time_ms
            
            self.logger.info(f"Generated Cypher in {generation_time_ms}ms: {cypher[:100]}...")
            
            return GeneratedCypher(
                cypher=cypher,
                confidence=confidence,
                reasoning=f"Generated using structural schema with {len(prompt_components.schema_elements_used)} schema elements",
                generation_time_ms=generation_time_ms,
                tokens_used=prompt_components.total_estimated_tokens,
                model_used=settings.openai_model_mini
            )
            
        except Exception as e:
            generation_time_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"Cypher generation failed in {generation_time_ms}ms: {e}")
            
            # Return fallback with low confidence
            return GeneratedCypher(
                cypher=f"// Generation failed: {str(e)}",
                confidence=0.0,
                reasoning=f"Generation error: {str(e)}",
                generation_time_ms=generation_time_ms,
                tokens_used=0,
                model_used=settings.openai_model_mini
            )
    # -------------------------------------------------------------- end generate_cypher()
    
    # -------------------------------------------------------------- validate_cypher()
    def validate_cypher(
        self,
        cypher: str,
        full_schema: Optional[CompleteKGSchema] = None
    ) -> ValidationReport:
        """
        Validate generated Cypher against safety rules and schema.
        
        Args:
            cypher: Cypher query to validate
            full_schema: Complete schema for validation (fetched if None)
            
        Returns:
            Comprehensive validation report
        """
        try:
            # Get full schema if not provided
            if full_schema is None:
                full_schema = self.schema_manager.get_schema()
            
            # Perform comprehensive validation
            report = self.validator.validate(cypher, full_schema)
            
            # Update metrics
            if not report.is_valid or report.fallback_recommended:
                self.metrics.validation_failures += 1
            
            self.logger.info(f"Validation: {report.is_valid}, issues: {len(report.issues)}, "
                           f"fallback: {report.fallback_recommended}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Cypher validation failed: {e}")
            self.metrics.validation_failures += 1
            
            return ValidationReport(
                is_valid=False,
                original_cypher=cypher,
                safe_cypher=None,
                issues=[{
                    "severity": "error",
                    "code": "VALIDATION_EXCEPTION",
                    "message": f"Validation failed: {str(e)}"
                }],
                fallback_recommended=True
            )
    # -------------------------------------------------------------- end validate_cypher()
    
    # -------------------------------------------------------------- execute_cypher()
    async def execute_cypher(
        self,
        cypher: str,
        max_results: int = 50
    ) -> ExecutionResult:
        """
        Execute validated Cypher query against the knowledge graph.
        
        Args:
            cypher: Validated Cypher query to execute
            max_results: Maximum number of results
            
        Returns:
            Execution results with performance metrics
        """
        start_time = time.time()
        
        # Initialize timing collector if available
        timing_collector = None
        if OBSERVABILITY_AVAILABLE:
            try:
                timing_collector = get_timing_collector()
            except Exception:
                pass
        
        try:
            # Execute query with timing via schema manager gateway
            if timing_collector:
                async with timing_collector.measure("llm_cypher_execution", {
                    "cypher_length": len(cypher),
                    "max_results": max_results,
                    "contains_limit": "LIMIT" in cypher.upper()
                }) as exec_timer:
                    self.logger.debug("Executing Cypher via schema manager gateway")
                    results = await asyncio.to_thread(self._execute_read_query, cypher, {})
                    
                    # Defensive check: ensure results is a list
                    if results is not None and not isinstance(results, list):
                        results = list(results)
                    
                    exec_timer.add_metadata({
                        "database_response_time_ms": int((time.time() - start_time) * 1000),
                        "raw_result_count": len(results) if results else 0
                    })
            else:
                self.logger.debug("Executing Cypher via schema manager gateway")
                results = await asyncio.to_thread(self._execute_read_query, cypher, {})
            
            # Defensive check: ensure results is a list
            if results is not None and not isinstance(results, list):
                results = list(results)
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Process results with enhanced null safety
            result_list = []
            result_count = 0

            if results:
                result_count = len(results)
                for record in results[:max_results]:  # Additional safety limit
                    result_dict = {}
                    for key in record.keys():
                        value = record[key]
                        # Convert Neo4j objects to serializable format
                        if hasattr(value, '_properties'):
                            result_dict[key] = dict(value._properties)
                            result_dict[key]['_labels'] = list(value.labels) if hasattr(value, 'labels') else []
                        elif hasattr(value, '_start_node'):
                            result_dict[key] = {
                                'type': str(value.type),
                                'start_node': dict(value.start_node._properties),
                                'end_node': dict(value.end_node._properties)
                            }
                        else:
                            result_dict[key] = value
                    result_list.append(result_dict)
            else:
                # Ensure result_count is 0 for None/empty results
                result_count = 0
            
            self.logger.info(f"Executed Cypher in {execution_time_ms}ms, {len(result_list)} results")
            
            return ExecutionResult(
                success=True,
                results=result_list,
                result_count=result_count,
                execution_time_ms=execution_time_ms,
                cypher_executed=cypher
            )
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"Cypher execution failed in {execution_time_ms}ms: {e}")
            
            self.metrics.execution_failures += 1
            
            return ExecutionResult(
                success=False,
                results=[],
                result_count=0,
                execution_time_ms=execution_time_ms,
                error_message=str(e),
                cypher_executed=cypher
            )
    # -------------------------------------------------------------- end execute_cypher()
    
    # -------------------------------------------------------------- query_knowledge_graph_llm_structural()
    async def query_knowledge_graph_llm_structural(
        self,
        user_query: str,
        max_results: int = 50
    ) -> Dict[str, Any]:
        """
        Complete orchestration: generate → validate → execute Cypher query.
        
        Uses LLM Structural Cypher generation with comprehensive validation.
        
        Args:
            user_query: Natural language query from user
            max_results: Maximum number of results to return
            
        Returns:
            Standardized response dict matching other traversal methods.
        """
        start_time = time.time()
        
        # Initialize timing collector for end-to-end tracking
        timing_collector = None
        if OBSERVABILITY_AVAILABLE:
            try:
                timing_collector = get_timing_collector()
            except Exception:
                pass
        
        if timing_collector:
            async with timing_collector.measure("llm_structural_cypher_complete", {
                "user_query_length": len(user_query),
                "max_results": max_results,
                "engine_enabled": self.enabled
            }) as complete_timer:
                return await self._execute_complete_query_with_timing(
                    user_query, max_results, start_time, complete_timer
                )
        else:
            return await self._execute_complete_query_with_timing(
                user_query, max_results, start_time, None
            )
    # -------------------------------------------------------------- end query_knowledge_graph_llm_structural()
    
    # -------------------------------------------------------------- _execute_complete_query_with_timing()
    async def _execute_complete_query_with_timing(
        self,
        user_query: str,
        max_results: int,
        start_time: float,
        timing_context=None
    ) -> Dict[str, Any]:
        """Execute complete query with optional timing context."""
        
        
        try:
            # Step 1: Generate Cypher
            try:
                generated = await self.generate_cypher(user_query, max_results)
                
                if generated.confidence == 0.0:
                    return self._build_error_response(
                        "Cypher generation failed",
                        generated.reasoning or "Unknown generation error",
                        int((time.time() - start_time) * 1000)
                    )
            except Exception:
                raise
            
            # Step 2: Validate generated Cypher
            try:
                validation_report = self.validate_cypher(generated.cypher)

                if not validation_report.is_valid:
                    self.logger.error("Validation failed")
                    return self._build_error_response(
                        "Cypher validation failed",
                        f"Generated query failed validation with {len(validation_report.issues)} issues.",
                        int((time.time() - start_time) * 1000)
                    )
            except Exception:
                raise
            
            # Step 3: Execute validated Cypher
            try:
                safe_cypher = validation_report.safe_cypher or generated.cypher
                execution_result = await self.execute_cypher(safe_cypher, max_results)
                
                if not execution_result.success:
                    return self._build_error_response(
                        "Cypher execution failed",
                        execution_result.error_message or "Unknown execution error",
                        int((time.time() - start_time) * 1000)
                    )
            except Exception:
                raise
            
            # Step 4: Format response
            total_time_ms = int((time.time() - start_time) * 1000)
            
            # Add final timing metadata if timing context available
            if timing_context:
                timing_context.add_metadata({
                    "final_confidence": generated.confidence,
                    "validation_successful": validation_report.is_valid,
                    "execution_successful": execution_result.success,
                    "result_count": execution_result.result_count,
                    "total_time_ms": total_time_ms,
                    "generation_to_execution_ratio": (
                        generated.generation_time_ms / max(execution_result.execution_time_ms, 1)
                    )
                })
            
            # If no results returned, respond with explicit unknown message
            if execution_result.result_count == 0:
                total_time_ms = int((time.time() - start_time) * 1000)
                return self._build_idk_response(
                    response_time_ms=total_time_ms,
                    reason="no_results",
                    extra_metadata={
                        "search_method": "llm_structural_cypher",
                        "result_count": 0,
                        "generation_time_ms": generated.generation_time_ms,
                        "execution_time_ms": execution_result.execution_time_ms,
                        "validation_issues": len(validation_report.issues),
                        "fixes_applied": len(validation_report.fixes_applied),
                        "tokens_used": generated.tokens_used,
                        "model_used": generated.model_used,
                    }
                )

            # If results exist but appear unrelated to the user query, return unknown
            if not self._is_result_relevant(user_query, execution_result.results):
                total_time_ms = int((time.time() - start_time) * 1000)
                return self._build_idk_response(
                    response_time_ms=total_time_ms,
                    reason="low_relevance",
                    extra_metadata={
                        "search_method": "llm_structural_cypher",
                        "result_count": execution_result.result_count,
                        "generation_time_ms": generated.generation_time_ms,
                        "execution_time_ms": execution_result.execution_time_ms,
                        "validation_issues": len(validation_report.issues),
                        "fixes_applied": len(validation_report.fixes_applied),
                        "tokens_used": generated.tokens_used,
                        "model_used": generated.model_used,
                    }
                )

            # NEW: Narrative summarization path
            if execution_result.result_count > 0:
                narrative_used = False
                citation_count = 0
                
                if (settings.llm_structural_narrative_enabled and 
                    execution_result.result_count > 0):
                    
                    try:
                        answer, citation_count = await self._generate_narrative_answer(
                            user_query, execution_result, timing_context
                        )
                        narrative_used = True
                        self._update_narrative_metrics(True, min(execution_result.result_count, settings.llm_structural_source_max), citation_count)
                        
                    except Exception as narrative_error:
                        self.logger.warning(f"Narrative generation failed: {narrative_error}")
                        self._update_narrative_metrics(False, 0, 0)
                        
                        # Fallback to existing deterministic rendering
                        answer = await self._generate_fallback_answer(
                            execution_result, timing_context
                        )
                else:
                    # Disabled or no results - use existing path
                    answer = await self._generate_fallback_answer(
                        execution_result, timing_context
                    )
                
                # Extract references for metadata (from fallback if narrative failed)
                visible_results = execution_result.results[:settings.llm_structural_source_max]
                document_refs = self._extract_document_references_from_results(visible_results)
                
                # Determine inline citation count (body only, not References section)
                if narrative_used:
                    # Use citation_count from narrative generation (already computed correctly)
                    inline_citation_count = citation_count
                else:
                    # For fallback path, strip References section before counting
                    answer_body = self._strip_references_section(answer)
                    citations_in_body = re.findall(r'\[(\d+)\]', answer_body)
                    inline_citation_count = len(set(citations_in_body))

                # Keep full answer stats for metadata
                citations_in_answer = re.findall(r'\[(\d+)\]', answer)
                occurrences_in_answer = len(citations_in_answer)
                unique_in_answer = len(set(citations_in_answer))

                # Guard: If references exist but no inline citations in body, treat as unrelated
                if len(document_refs) > 0 and inline_citation_count == 0:
                    self.logger.info(
                        f"No inline citations in body with {len(document_refs)} references found; returning IDK response"
                    )
                    return self._build_idk_response(
                        response_time_ms=total_time_ms,
                        reason="no_inline_citations",
                        extra_metadata={
                            "search_method": "llm_structural_cypher",
                            "result_count": execution_result.result_count,
                            "references_found": len(document_refs),
                            "narrative_attempted": narrative_used,
                            "generation_time_ms": generated.generation_time_ms,
                            "execution_time_ms": execution_result.execution_time_ms,
                            "tokens_used": generated.tokens_used,
                            "model_used": generated.model_used,
                        }
                    )

            else:
                # This branch is no longer reachable due to early return above
                answer = ""
                document_refs = []
                narrative_used = False
                citation_count = 0
                occurrences_in_answer = 0
                unique_in_answer = 0
            
            # Calculate final confidence
            final_confidence = min(
                generated.confidence * (1.0 if validation_report.is_valid else 0.8),
                settings.validated_traversal_confidence_cap
            )
            
            return {
                "answer": answer,
                "cypher_query": safe_cypher,
                "confidence": final_confidence,
                "response_time_ms": total_time_ms,
                "metadata": {
                    "search_method": "llm_structural_cypher",
                    "result_count": execution_result.result_count,
                    "generation_time_ms": generated.generation_time_ms,
                    "execution_time_ms": execution_result.execution_time_ms,
                    "validation_issues": len(validation_report.issues),
                    "fixes_applied": len(validation_report.fixes_applied),
                    "tokens_used": generated.tokens_used,
                    "model_used": generated.model_used,
                    # NEW: Narrative summarization metadata
                    "narrative_summarization": {
                        "enabled": settings.llm_structural_narrative_enabled,
                        "used": narrative_used,
                        "fallback_reason": None if narrative_used else "disabled_or_failed"
                    },
                    "citations": {
                        "available": len(document_refs),
                        "included_in_answer_unique": citation_count if narrative_used else unique_in_answer,
                        "included_in_answer_occurrences": occurrences_in_answer
                    },
                    "references": {
                        "used": citation_count if narrative_used else unique_in_answer,
                        "available": len(document_refs)
                    },
                    "chunk_references": len(document_refs),  # LangChain parity
                    "engine_metrics": {
                        "success_rate": self.metrics.success_rate,
                        "narrative_success_rate": self.metrics.narrative_success_rate,
                        "total_generations": self.metrics.total_generations
                    },
                    "observability": {
                        "timing_available": OBSERVABILITY_AVAILABLE,
                        "circuit_breaker_available": OBSERVABILITY_AVAILABLE
                    }
                }
            }
            
        except Exception as e:
            total_time_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"LLM structural query failed: {e}")
            
            # On failure, present standardized unknown response to the user
            return self._build_idk_response(
                response_time_ms=total_time_ms,
                reason="error",
                extra_metadata={
                    "search_method": "llm_structural_cypher",
                    "error_detail": str(e)
                }
            )
    # -------------------------------------------------------------- end _execute_complete_query_with_timing()
    
    # =========================================================================
    # Prompting, Extraction, and Field Injection
    # =========================================================================
    # -------------------------------------------------------------- _extract_cypher_from_response()
    def _extract_cypher_from_response(self, response: str) -> Optional[str]:
        """Extract Cypher query from LLM response, looking for code fences."""
        # Look for cypher code blocks
        cypher_pattern = r'```cypher\s*\n?(.*?)\n?```'
        match = re.search(cypher_pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: look for any code blocks
        code_pattern = r'```\s*\n?(.*?)\n?```'
        match = re.search(code_pattern, response, re.DOTALL)
        
        if match:
            potential_cypher = match.group(1).strip()
            # Basic check if it looks like Cypher
            if any(keyword in potential_cypher.upper() for keyword in ['MATCH', 'RETURN', 'WHERE']):
                return potential_cypher
        
        return None
    # -------------------------------------------------------------- end _extract_cypher_from_response()
    
    # -------------------------------------------------------------- _ensure_limit_clause()
    def _ensure_limit_clause(self, cypher: str, max_results: int) -> str:
        """Ensure query has LIMIT clause, inject if missing."""
        if re.search(r'\bLIMIT\s+\d+', cypher, re.IGNORECASE):
            return cypher
        
        # Simple injection at the end
        cypher = cypher.strip()
        if cypher.endswith(';'):
            cypher = cypher[:-1] + f" LIMIT {max_results};"
        else:
            cypher = cypher + f" LIMIT {max_results}"
        
        return cypher
    # -------------------------------------------------------------- end _ensure_limit_clause()
    
    # -------------------------------------------------------------- _inject_chunk_return_fields()
    def _inject_chunk_return_fields(self, cypher: str) -> str:
        """
        Enhanced field injection with comprehensive safeguards.
        
        Critical improvements:
        - Aggregation detection (count|sum|avg|max|min|collect)
        - WITH pipeline scope validation
        - Backticked identifier support
        - Multiple :Chunk alias handling
        - Support for DISTINCT, ORDER BY, LIMIT, and trailing semicolons
        
        Args:
            cypher: Generated Cypher query
            
        Returns:
            Modified Cypher with chunk_id and page fields injected if safe and needed
        """
        # Find all :Chunk node aliases (including backticked)
        chunk_pattern = r'\((`?[\w]+`?):\s*Chunk\b'
        chunk_matches = re.findall(chunk_pattern, cypher, re.IGNORECASE)
        
        if not chunk_matches:
            return cypher  # No Chunk nodes, return as-is
        
        # Extract RETURN clause with support for various formats
        return_match = re.search(
            r'\bRETURN\s+(DISTINCT\s+)?(.+?)(?:\s+ORDER\s+BY|\s+LIMIT|\s*;|\s*$)', 
            cypher, re.IGNORECASE | re.DOTALL
        )
        
        if not return_match:
            return cypher  # No RETURN clause found
        
        return_fields = return_match.group(2).strip()
        
        # CRITICAL: Aggregation guard - skip injection if aggregates detected
        aggregate_pattern = r'\b(count|sum|avg|max|min|collect)\s*\('
        if re.search(aggregate_pattern, return_fields, re.IGNORECASE):
            self.logger.debug("Skipping field injection: aggregation detected")
            self._update_injection_metrics(skipped=True, skip_reason="aggregation")
            return cypher
        
        # CRITICAL: WITH pipeline guard - check scope validity
        with_clauses = re.findall(r'\bWITH\b[^;]*?(?=\bMATCH\b|\bWHERE\b|\bRETURN\b|$)', 
                                 cypher, re.IGNORECASE | re.DOTALL)
        if with_clauses:
            # Validate chunk alias is still in scope at RETURN
            last_with = with_clauses[-1]
            chunk_aliases_in_with = [alias.strip('`') for alias in chunk_matches 
                                    if alias.strip('`') in last_with]
            if not chunk_aliases_in_with:
                self.logger.debug("Skipping field injection: chunk alias out of scope after WITH")
                self._update_injection_metrics(skipped=True, skip_reason="scope")
                return cypher
        
        # Select best chunk alias - prefer one already in RETURN, then first match
        chunk_alias = None
        for alias in chunk_matches:
            alias_clean = alias.strip('`')  # Remove backticks for comparison
            if re.search(rf'\b{re.escape(alias_clean)}\b', return_fields, re.IGNORECASE):
                chunk_alias = alias_clean
                break
        
        if not chunk_alias:
            chunk_alias = chunk_matches[0].strip('`')  # Use first match, remove backticks
        
        # Check existing fields with enhanced patterns
        chunk_id_patterns = [
            r'\bchunk_id\b',
            rf'\b{re.escape(chunk_alias)}\.chunk_id\b',
            r'\bAS\s+chunk_id\b'  # Already aliased
        ]
        page_patterns = [
            r'\b(page|page_number|page_num)\b',
            rf'\b{re.escape(chunk_alias)}\.(page|page_number|page_num)\b',
            r'\bAS\s+(page|page_number|page_num)\b'  # Already aliased
        ]
        
        has_chunk_id = any(re.search(pattern, return_fields, re.IGNORECASE) 
                          for pattern in chunk_id_patterns)
        has_page = any(re.search(pattern, return_fields, re.IGNORECASE) 
                      for pattern in page_patterns)
        
        # Inject missing fields
        additions = []
        if not has_chunk_id:
            additions.append(f"{chunk_alias}.chunk_id AS chunk_id")
        if not has_page:
            additions.append(f"{chunk_alias}.page AS page")
        
        if additions:
            new_return_fields = return_fields.rstrip() + ", " + ", ".join(additions)
            
            # Replace RETURN clause preserving structure
            cypher_modified = re.sub(
                r'(\bRETURN\s+(?:DISTINCT\s+)?)(.+?)(?=\s+ORDER\s+BY|\s+LIMIT|\s*;|\s*$)',
                rf'\1{new_return_fields}',
                cypher,
                flags=re.IGNORECASE | re.DOTALL
            )
            
            self.logger.debug(f"Injected citation fields: {', '.join(additions)}")
            self._update_injection_metrics(skipped=False, fields_added=len(additions))
            return cypher_modified
        
        return cypher
    # -------------------------------------------------------------- end _inject_chunk_return_fields()
    
    # -------------------------------------------------------------- _update_injection_metrics()
    def _update_injection_metrics(
        self, 
        skipped: bool = False, 
        skip_reason: Optional[str] = None,
        fields_added: int = 0
    ) -> None:
        """Update field injection metrics for observability."""
        if skipped:
            if skip_reason == "aggregation":
                self.metrics.injection_skipped_aggregation += 1
            elif skip_reason == "scope":
                self.metrics.injection_skipped_scope += 1
        else:
            self.metrics.fields_injected_total += fields_added
    # -------------------------------------------------------------- end _update_injection_metrics()
    
    # -------------------------------------------------------------- _update_domain_metrics()
    def _update_domain_metrics(self, domain_markers: Dict[int, Dict[str, List[str]]], final_answer: str = "") -> None:
        """
        Update domain citation metrics for observability.
        
        Args:
            domain_markers: Dictionary of domain markers found in sources
            final_answer: Final answer text to check for domain citation usage
        """
        if not settings.llm_domain_citation_enabled:
            return
            
        try:
            # Count markers found per domain
            for source_markers in domain_markers.values():
                for domain, markers in source_markers.items():
                    if markers:  # If any markers found for this domain
                        if domain == 'legal':
                            self.metrics.domain_markers_found_legal += len(markers)
                        elif domain == 'academic':
                            self.metrics.domain_markers_found_academic += len(markers)
                        elif domain == 'technical':
                            self.metrics.domain_markers_found_technical += len(markers)
                        elif domain == 'business':
                            self.metrics.domain_markers_found_business += len(markers)
                        elif domain == 'medical':
                            self.metrics.domain_markers_found_medical += len(markers)
            
            # Check if final answer contains domain citations (domain text near [n])
            if final_answer and self._has_domain_citations_in_answer(final_answer, domain_markers):
                self.metrics.answers_with_domain_citations += 1
                
        except Exception as e:
            self.logger.debug(f"Error updating domain metrics: {e}")
            self.metrics.domain_marker_extraction_errors += 1
    # -------------------------------------------------------------- end _update_domain_metrics()
    
    # -------------------------------------------------------------- _has_domain_citations_in_answer()
    def _has_domain_citations_in_answer(
        self, 
        answer: str, 
        domain_markers: Dict[int, Dict[str, List[str]]]
    ) -> bool:
        """
        Check if the answer contains domain identifiers near citations.
        
        This checks if domain-specific markers appear in proximity to [n] citations,
        indicating successful domain-aware citation rendering.
        """
        if not domain_markers:
            return False
            
        # Extract all domain markers from sources
        all_markers = set()
        for source_markers in domain_markers.values():
            for markers_list in source_markers.values():
                all_markers.update(markers_list)
        
        if not all_markers:
            return False
        
        # Check if any domain marker appears near a citation in the answer
        # Look for pattern: marker_text [n] OR [n] marker_text within reasonable distance
        citation_pattern = r'\[(\d+)\]'
        citations = list(re.finditer(citation_pattern, answer))
        
        for marker in all_markers:
            # Escape special regex characters in the marker
            escaped_marker = re.escape(marker)
            
            for citation_match in citations:
                citation_start = citation_match.start()
                citation_end = citation_match.end()
                
                # Check 100 characters before and after citation
                context_start = max(0, citation_start - 100)
                context_end = min(len(answer), citation_end + 100)
                context = answer[context_start:context_end]
                
                if re.search(escaped_marker, context, re.IGNORECASE):
                    return True
        
        return False
    # -------------------------------------------------------------- end _has_domain_citations_in_answer()
    
    # -------------------------------------------------------------- _calculate_generation_confidence()
    def _calculate_generation_confidence(
        self,
        cypher: str,
        prompt_components: PromptComponents,
        response: Any
    ) -> float:
        """Calculate confidence score for generated Cypher."""
        confidence = 0.5  # Base confidence
        
        # Bonus for valid Cypher structure
        if any(keyword in cypher.upper() for keyword in ['MATCH', 'RETURN']):
            confidence += 0.2
        
        # Bonus for including LIMIT
        if 'LIMIT' in cypher.upper():
            confidence += 0.1
        
        # Bonus for not being truncated
        if not prompt_components.truncation_applied:
            confidence += 0.1
        
        # Bonus for using schema elements
        schema_elements_count = sum(prompt_components.schema_elements_used.values())
        if schema_elements_count > 0:
            confidence += 0.1
        
        return min(confidence, 1.0)
    # -------------------------------------------------------------- end _calculate_generation_confidence()
    
    # =========================================================================
    # References and Formatting
    # =========================================================================
    # -------------------------------------------------------------- _extract_document_references_from_results()
    def _extract_document_references_from_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract unique chunk references from query results with dual source support.
        
        Extracts chunk_id and page from:
        1. Top-level row fields (e.g., if Cypher returns c.chunk_id AS chunk_id)
        2. Node dicts (converted from Neo4j objects) with nested properties
        
        Args:
            results: List of result rows from Cypher execution
            
        Returns:
            List of unique references sorted by (chunk_id, page)
        """
        references = []
        seen_refs = set()
        
        try:
            for result in results:
                chunk_id = None
                page_number = None
                
                # Extract from top-level row fields first
                chunk_id = result.get('chunk_id')
                page_number = result.get('page') or result.get('page_number') or result.get('page_num')
                
                # If not found at top level, check node dicts
                if not chunk_id or page_number is None:
                    for key, value in result.items():
                        if isinstance(value, dict):
                            # Check if this is a converted Neo4j node (has _labels or other properties)
                            if not chunk_id:
                                chunk_id = value.get('chunk_id')
                            if page_number is None:
                                page_number = value.get('page') or value.get('page_number') or value.get('page_num')
                            
                            # Break early if we found both
                            if chunk_id and page_number is not None:
                                break
                
                # Fallback construction only if both document/title AND page are present
                if not chunk_id and page_number is not None:
                    document = None
                    # Check top-level fields
                    document = result.get('document') or result.get('title') or result.get('doc_title')
                    
                    # Check node dicts if not found
                    if not document:
                        for value in result.values():
                            if isinstance(value, dict):
                                document = value.get('document') or value.get('title') or value.get('doc_title')
                                if document:
                                    break
                    
                    # Construct chunk_id if we have both document and page
                    if document:
                        chunk_id = f"{document}_p{page_number}_c1"
                
                # Store reference if we have valid chunk_id and page
                if chunk_id and page_number is not None:
                    try:
                        # Coerce page to string, handle both string and numeric pages
                        page_str = str(page_number)
                        ref_key = f"{chunk_id}::{page_str}"
                        
                        if ref_key not in seen_refs:
                            references.append({
                                'chunk_id': chunk_id,
                                'page': page_str,
                                'reference_key': ref_key
                            })
                            seen_refs.add(ref_key)
                    except (ValueError, TypeError) as e:
                        self.logger.debug(f"Invalid page number {page_number}: {e}")
                        continue
        
        except Exception as e:
            self.logger.error(f"Error extracting document references: {e}")
        
        # Sort by chunk_id, then numeric page for stable ordering
        references.sort(key=lambda x: (
            x['chunk_id'], 
            int(x['page']) if x['page'].isdigit() else 0
        ))
        
        return references
    # -------------------------------------------------------------- end _extract_document_references_from_results()
    
    # -------------------------------------------------------------- _number_references()
    def _number_references(self, document_refs: List[Dict[str, str]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Assign consecutive numbers to references and build mapping for inline citations.
        
        Args:
            document_refs: List of reference dicts with chunk_id and page
            
        Returns:
            Tuple of (used_refs with numbers, ref_map for inline placement)
        """
        used_refs = []
        ref_map = {}
        
        for i, ref in enumerate(document_refs, 1):
            # Build numbered reference for References section
            used_ref = {
                'number': i,
                'chunk_id': ref['chunk_id'],
                'page': ref['page']
            }
            used_refs.append(used_ref)
            
            # Build mapping for inline citation placement
            ref_map[ref['reference_key']] = i
        
        return used_refs, ref_map
    # -------------------------------------------------------------- end _number_references()
    
    # -------------------------------------------------------------- _format_references_section_numbered_llm()
    def _format_references_section_numbered_llm(self, used_refs: List[Dict[str, Any]]) -> str:
        """
        Format References section in exact LangChain format.
        
        Args:
            used_refs: List of numbered references
            
        Returns:
            Formatted References section or empty string if no references
        """
        if not used_refs:
            return ""
        
        references_lines = ["\n\n**References:**"]
        
        for ref in used_refs:
            number = ref['number']
            chunk_id = ref['chunk_id']
            page = ref['page']
            references_lines.append(f"[{number}] {chunk_id}, p.{page}")
        
        references_lines.append("")  # Trailing newline
        return "\n".join(references_lines)
    # -------------------------------------------------------------- end _format_references_section_numbered_llm()
    
    # -------------------------------------------------------------- _format_execution_results()
    def _format_execution_results(self, results: List[Dict[str, Any]], ref_map: Optional[Dict[str, int]] = None) -> str:
        """
        Format execution results into readable text response with optional inline citations.
        
        Args:
            results: List of result rows from Cypher execution
            ref_map: Optional mapping from reference_key to citation number for inline citations
            
        Returns:
            Formatted text with inline [n] citations when ref_map provided
        """
        if not results:
            return "No results found."
        
        # Use shared reference key extraction method
        
        # Handle simple single result case
        if len(results) == 1:
            result = results[0]
            if len(result) == 1:
                key, value = next(iter(result.items()))
                base_text = f"Result: {value}"
                
                # Add citation if ref_map provided
                if ref_map:
                    ref_key = self._get_reference_key_for_result(result)
                    if ref_key and ref_key in ref_map:
                        citation_num = ref_map[ref_key]
                        base_text += f" [{citation_num}]"
                
                return base_text
        
        # Multiple results or complex structure - prioritize citation fields
        formatted_lines = []
        for i, result in enumerate(results[:settings.llm_structural_source_max], 1):  # Show first N results
            line_parts = []
            
            # Priority order: citation fields first, then content
            priority_fields = ['chunk_id', 'page', 'page_number', 'page_num', 'document', 'title', 'doc_title']
            processed_keys = set()
            
            # Process priority fields first
            for priority_field in priority_fields:
                if priority_field in result:
                    value = result[priority_field]
                    if isinstance(value, dict) and '_labels' in value:
                        labels = value.get('_labels', [])
                        label_str = f"({':'.join(labels)})" if labels else "()"
                        line_parts.append(f"{priority_field}: {label_str}")
                    elif isinstance(value, dict):
                        line_parts.append(f"{priority_field}: {str(value)[:50]}...")
                    else:
                        line_parts.append(f"{priority_field}: {value}")
                    processed_keys.add(priority_field)
            
            # Process remaining fields (NO TRUNCATION - full content display)
            for key, value in result.items():
                if key not in processed_keys:
                    if isinstance(value, dict) and '_labels' in value:
                        labels = value.get('_labels', [])
                        label_str = f"({':'.join(labels)})" if labels else "()"
                        line_parts.append(f"{key}: {label_str}")
                    elif isinstance(value, dict):
                        line_parts.append(f"{key}: {str(value)[:50]}...")
                    else:
                        # CRITICAL: No truncation - display full content as requested
                        line_parts.append(f"{key}: {value}")
            
            base_line = f"{i}. {', '.join(line_parts)}"
            
            # Add citation if ref_map provided
            if ref_map:
                ref_key = self._get_reference_key_for_result(result)
                if ref_key and ref_key in ref_map:
                    citation_num = ref_map[ref_key]
                    base_line += f" [{citation_num}]"
            
            formatted_lines.append(base_line)
        
        if len(results) > 10:
            formatted_lines.append(f"... and {len(results) - 10} more results")
        
        return "\n".join(formatted_lines)
    # -------------------------------------------------------------- end _format_execution_results()
    
    # =========================================================================
    # Citation Validation and LLM Response Cleaning
    # =========================================================================
    
    # -------------------------------------------------------------- _validate_and_clean_narrative()
    def _validate_and_clean_narrative(
        self, 
        narrative: str, 
        provided_numbers: set
    ) -> str:
        """
        Validate citations and strip accidental References sections.
        
        Critical safeguards:
        - Remove any References section from LLM output
        - Validate citation numbers against provided sources
        - Drop or warn about invented citations
        
        Args:
            narrative: Raw narrative text from LLM
            provided_numbers: Set of valid citation numbers
            
        Returns:
            Cleaned narrative text
        """
        # CRITICAL: Strip any References section that LLM might have added
        cleaned_narrative = self._strip_references_section(narrative)
        
        # CRITICAL: Validate citation numbers if enabled
        if settings.llm_structural_citation_validation:
            cleaned_narrative = self._validate_citation_numbers(
                cleaned_narrative, provided_numbers
            )
        
        return cleaned_narrative
    # -------------------------------------------------------------- end _validate_and_clean_narrative()

    # =========================================================================
    # Domain Marker Extraction Functions
    # =========================================================================

    # -------------------------------------------------------------- _extract_regulatory_markers()
    def _extract_regulatory_markers(self, text: str) -> List[str]:
        """
        Extract legal/regulatory markers from text.
        
        Patterns include CFR sections, Parts, and regulatory identifiers.
        
        Args:
            text: Source text to extract markers from
            
        Returns:
            List of unique regulatory markers in order of appearance
        """
        if not settings.llm_domain_citation_enabled:
            return []
            
        markers = []
        seen = set()
        
        # CFR sections: §57.4361, §57.4361(a), §75.380(a)(1)
        cfr_pattern = r'§\s*\d+(?:\.\d+)*(?:\([a-z0-9]+\))*'
        for match in re.finditer(cfr_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        # Parts: Part 75.1714, Part 30.1
        parts_pattern = r'\bPart\s+\d+(?:\.\d+)*'
        for match in re.finditer(parts_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        # CFR/Title references: 30 CFR, Title 30 CFR
        cfr_title_pattern = r'(?:\bTitle\s+)?\d+\s*CFR\b'
        for match in re.finditer(cfr_title_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        return markers
    # -------------------------------------------------------------- end _extract_regulatory_markers()

    # -------------------------------------------------------------- _extract_academic_markers()
    def _extract_academic_markers(self, text: str) -> List[str]:
        """
        Extract academic/scholarly markers from text.
        
        Patterns include volumes, chapters, figures, tables.
        
        Args:
            text: Source text to extract markers from
            
        Returns:
            List of unique academic markers in order of appearance
        """
        if not settings.llm_domain_citation_enabled:
            return []
            
        markers = []
        seen = set()
        
        # Volume/Chapter: Vol. 2, Chapter 3, Vol. 2, Chapter 3
        vol_chapter_pattern = r'\b(?:Vol\.?\s*\d+(?:,\s*)?)?(?:Chapter|Ch\.?)\s+\d+(?:\.\d+)?'
        for match in re.finditer(vol_chapter_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        # Figures: Figure 4.1, Fig. 2.3
        figure_pattern = r'\b(?:Figure|Fig\.?)\s+\d+(?:\.\d+)?'
        for match in re.finditer(figure_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        # Tables: Table 2, Table 4.1
        table_pattern = r'\bTable\s+\d+(?:\.\d+)?'
        for match in re.finditer(table_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        # Sections: Section 2.1, Sec. 4.5
        section_pattern = r'\b(?:Section|Sec\.?)\s+\d+(?:\.\d+)*'
        for match in re.finditer(section_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        return markers
    # -------------------------------------------------------------- end _extract_academic_markers()

    # -------------------------------------------------------------- _extract_technical_markers()
    def _extract_technical_markers(self, text: str) -> List[str]:
        """
        Extract technical/standards markers from text.
        
        Patterns include ISO, ASTM, IEEE standards.
        
        Args:
            text: Source text to extract markers from
            
        Returns:
            List of unique technical markers in order of appearance
        """
        if not settings.llm_domain_citation_enabled:
            return []
            
        markers = []
        seen = set()
        
        # ISO standards: ISO 9001:2015, ISO/IEC 27001:2013
        iso_pattern = r'\bISO(?:/[A-Z]+)?\s+\d{4,5}:\d{4}'
        for match in re.finditer(iso_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        # ASTM standards: ASTM D2582-07, ASTM E119-20
        astm_pattern = r'\bASTM\s+[A-Z]?\d{1,4}-\d{2}'
        for match in re.finditer(astm_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        # IEEE standards: IEEE 802.11, IEEE 830-1998
        ieee_pattern = r'\bIEEE\s+\d+(?:\.\d+)*(?:-\d{4})?'
        for match in re.finditer(ieee_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        # ANSI standards: ANSI/NIST SP 800-53
        ansi_pattern = r'\b(?:ANSI(?:/[A-Z]+)*\s+)?[A-Z]+\s+\d+(?:-\d+)*'
        for match in re.finditer(ansi_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen and len(marker) > 6:  # Avoid false positives
                markers.append(marker)
                seen.add(marker)
        
        return markers
    # -------------------------------------------------------------- end _extract_technical_markers()

    # -------------------------------------------------------------- _extract_business_markers()
    def _extract_business_markers(self, text: str) -> List[str]:
        """
        Extract business/policy markers from text.
        
        Patterns include policies, procedures, SOPs.
        
        Args:
            text: Source text to extract markers from
            
        Returns:
            List of unique business markers in order of appearance
        """
        if not settings.llm_domain_citation_enabled:
            return []
            
        markers = []
        seen = set()
        
        # Policies: Policy 2.4, Policy HR-001
        policy_pattern = r'\bPolicy\s+[A-Z0-9-]+(?:\.\d+)?'
        for match in re.finditer(policy_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        # Procedures: Procedure A-15, Procedure SOP-17
        procedure_pattern = r'\bProcedure\s+[A-Z0-9-]+(?:\.\d+)?'
        for match in re.finditer(procedure_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        # SOPs: SOP-17, SOP 4.2
        sop_pattern = r'\bSOP[-\s]?\d+(?:\.\d+)?'
        for match in re.finditer(sop_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        # Guidelines: Guideline 3.1, Guidelines Document A
        guideline_pattern = r'\bGuidelines?\s+(?:Document\s+)?[A-Z0-9-]+(?:\.\d+)?'
        for match in re.finditer(guideline_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        return markers
    # -------------------------------------------------------------- end _extract_business_markers()

    # -------------------------------------------------------------- _extract_medical_markers()
    def _extract_medical_markers(self, text: str) -> List[str]:
        """
        Extract medical/clinical markers from text.
        
        Patterns include protocols, guidelines, clinical references.
        
        Args:
            text: Source text to extract markers from
            
        Returns:
            List of unique medical markers in order of appearance
        """
        if not settings.llm_domain_citation_enabled:
            return []
            
        markers = []
        seen = set()
        
        # Protocols: Protocol 4.2, Clinical Protocol A
        protocol_pattern = r'\b(?:Clinical\s+)?Protocol\s+[A-Z0-9-]+(?:\.\d+)?'
        for match in re.finditer(protocol_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        # NICE guidelines: NICE NG12, NICE CG180
        nice_pattern = r'\bNICE\s+[A-Z]+\d+'
        for match in re.finditer(nice_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        # Clinical guidelines: Guidelines Table 1, Clinical Guidelines 2.3
        clinical_guidelines_pattern = r'\b(?:Clinical\s+)?Guidelines\s+(?:Table\s+)?\d+(?:\.\d+)?'
        for match in re.finditer(clinical_guidelines_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        # ICD codes: ICD-10 Z51.1, ICD-9 V57.1
        icd_pattern = r'\bICD[-\s]?\d+\s+[A-Z]\d+(?:\.\d+)?'
        for match in re.finditer(icd_pattern, text, re.IGNORECASE):
            marker = match.group().strip()
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
        
        return markers
    # -------------------------------------------------------------- end _extract_medical_markers()
    
    # -------------------------------------------------------------- _collect_source_domain_markers()
    def _collect_source_domain_markers(self, numbered_sources: List[Dict[str, Any]]) -> Dict[int, Dict[str, List[str]]]:
        """
        Extract domain markers from all numbered sources.
        
        Args:
            numbered_sources: List of numbered source dictionaries with excerpts
            
        Returns:
            Dictionary mapping source numbers to domain markers
        """
        if not settings.llm_domain_citation_enabled:
            return {}
            
        domain_markers = {}
        
        for source in numbered_sources:
            n = source.get('n', 0)
            excerpt = source.get('excerpt', '')
            
            if not excerpt:
                continue
                
            markers = {
                'legal': self._extract_regulatory_markers(excerpt),
                'academic': self._extract_academic_markers(excerpt),
                'technical': self._extract_technical_markers(excerpt),
                'business': self._extract_business_markers(excerpt),
                'medical': self._extract_medical_markers(excerpt)
            }
            
            # Only include sources with at least one marker
            has_markers = any(markers.values())
            if has_markers:
                domain_markers[n] = markers
        
        return domain_markers
    # -------------------------------------------------------------- end _collect_source_domain_markers()

    # -------------------------------------------------------------- _strip_references_section()
    def _strip_references_section(self, text: str) -> str:
        """Strip References section from LLM output (mirrors fusion logic)."""
        # Look for References section patterns
        references_patterns = [
            r'\n\n\*\*References:\*\*.*$',
            r'\n\nReferences:.*$',
            r'\n\n## References.*$',
            r'\n\n# References.*$',
            r'\n\n\*\*Sources:\*\*.*$',
            r'\n\nSources:.*$'
        ]
        
        original_length = len(text)
        
        for pattern in references_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        if len(text) < original_length:
            self.metrics.references_sections_stripped += 1
            self.logger.debug("Stripped References section from LLM output")
        
        return text.rstrip()
    # -------------------------------------------------------------- end _strip_references_section()

    # -------------------------------------------------------------- _validate_citation_numbers()
    def _validate_citation_numbers(
        self, 
        text: str, 
        provided_numbers: set
    ) -> str:
        """Validate and clean citation numbers in narrative."""
        citations_found = re.findall(r'\[(\d+)\]', text)
        invented_citations = []
        
        def replace_citation(match):
            cite_num = int(match.group(1))
            if cite_num in provided_numbers:
                return match.group(0)  # Keep valid citation
            else:
                invented_citations.append(cite_num)
                self.logger.warning(f"LLM invented citation [{cite_num}] - removing")
                return ""  # Remove invalid citation
        
        cleaned_text = re.sub(r'\[(\d+)\]', replace_citation, text)
        
        # Update metrics
        if invented_citations:
            self.metrics.citations_dropped += len(set(invented_citations))
        
        self.metrics.citations_validated += len(set(citations_found))
        
        return cleaned_text
    # -------------------------------------------------------------- end _validate_citation_numbers()
    
    # -------------------------------------------------------------- _renumber_citations_langchain_compatible()
    def _renumber_citations_langchain_compatible(
        self, 
        answer: str, 
        used_refs: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Port of LangChain's renumbering logic for exact compatibility.
        
        Args:
            answer: Narrative text with inline citations
            used_refs: List of reference dicts with 'number' field
            
        Returns:
            Tuple of (renumbered_answer, chronologically_ordered_refs)
        """
        # Extract citation order from answer (matching LangChain regex)
        citations_in_answer = re.findall(r'\[(\d+)\]', answer)
        if not citations_in_answer:
            return answer, used_refs
        
        # Get unique citations in order of first appearance
        unique_citations = []
        seen = set()
        for citation in citations_in_answer:
            citation_num = int(citation)
            if citation_num not in seen:
                unique_citations.append(citation_num)
                seen.add(citation_num)
        
        # Create mapping from old to new numbers
        citation_map = {}
        chronological_references = []
        
        for new_number, old_number in enumerate(unique_citations, 1):
            citation_map[old_number] = new_number
            
            # Find corresponding reference
            for ref in used_refs:
                if ref.get('number') == old_number:
                    new_ref = ref.copy()
                    new_ref['number'] = new_number
                    chronological_references.append(new_ref)
                    break
        
        # Renumber all citations in answer (exact LangChain logic)
        def replace_citation(match):
            old_num = int(match.group(1))
            new_num = citation_map.get(old_num, old_num)
            return f"[{new_num}]"
        
        renumbered_answer = re.sub(r'\[(\d+)\]', replace_citation, answer)
        
        self.logger.debug(f"Renumbered {len(unique_citations)} citations chronologically")
        
        return renumbered_answer, chronological_references
    # -------------------------------------------------------------- end _renumber_citations_langchain_compatible()
    
    # =========================================================================
    # Shared Reference Key Utility and Source Selection
    # =========================================================================
    
    # -------------------------------------------------------------- _get_reference_key_for_result()
    def _get_reference_key_for_result(self, result: Dict[str, Any]) -> Optional[str]:
        """
        Shared utility for reference key extraction.
        Used by both _build_numbered_sources and _format_execution_results.
        
        Args:
            result: Result row from Cypher execution
            
        Returns:
            Reference key in format "chunk_id::page" or None
        """
        chunk_id = None
        page_number = None
        
        # Top-level extraction
        chunk_id = result.get('chunk_id')
        page_number = result.get('page') or result.get('page_number') or result.get('page_num')
        
        # Nested object extraction
        if not chunk_id or page_number is None:
            for value in result.values():
                if isinstance(value, dict):
                    if not chunk_id:
                        chunk_id = value.get('chunk_id')
                    if page_number is None:
                        page_number = value.get('page') or value.get('page_number') or value.get('page_num')
                    
                    if chunk_id and page_number is not None:
                        break
        
        # Fallback construction
        if not chunk_id and page_number is not None:
            document = result.get('document') or result.get('title')
            if not document:
                for value in result.values():
                    if isinstance(value, dict):
                        document = value.get('document') or value.get('title')
                        if document:
                            break
            
            if document:
                chunk_id = f"{document}_p{page_number}_c1"
        
        return f"{chunk_id}::{str(page_number)}" if chunk_id and page_number is not None else None
    # -------------------------------------------------------------- end _get_reference_key_for_result()
    
    # -------------------------------------------------------------- _select_best_sources()
    def _select_best_sources(
        self, 
        results: List[Dict[str, Any]], 
        max_sources: int
    ) -> List[Dict[str, Any]]:
        """
        Select best sources using heuristics instead of truncation.
        
        Selection criteria:
        - Presence of citation fields (chunk_id, page)
        - Content length (longer = more informative)
        - Unique chunk_id values
        
        Args:
            results: All query results
            max_sources: Maximum sources to select
            
        Returns:
            Selected results ranked by quality
        """
        scored_results = []
        seen_chunks = set()
        
        for result in results:
            score = 0
            
            # Boost for citation fields
            if self._get_reference_key_for_result(result):
                score += 10
            
            # Boost for content length
            content = self._extract_content_excerpt(result)
            if len(content) > 500:
                score += 8
            elif len(content) > 100:
                score += 5
            
            # Boost for unique chunks
            chunk_id = result.get('chunk_id')
            if chunk_id and chunk_id not in seen_chunks:
                score += 3
                seen_chunks.add(chunk_id)
            
            scored_results.append((score, result))
        
        # Sort by score and take top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        selected = [result for _, result in scored_results[:max_sources]]
        
        self.logger.debug(f"Selected {len(selected)}/{len(results)} sources based on quality scores")
        return selected
    # -------------------------------------------------------------- end _select_best_sources()
    
    # -------------------------------------------------------------- _extract_content_excerpt()
    def _extract_content_excerpt(self, result: Dict[str, Any]) -> str:
        """Extract the best content field from a result row."""
        # Priority order for content fields
        content_fields = ['text', 'content', 'chunk_text', 'description', 'summary']
        
        # Check top-level fields first
        for field in content_fields:
            if field in result and result[field]:
                return str(result[field])
        
        # Check nested objects
        for value in result.values():
            if isinstance(value, dict):
                for field in content_fields:
                    if field in value and value[field]:
                        return str(value[field])
        
        # Fallback to string representation
        return str(result)
    # -------------------------------------------------------------- end _extract_content_excerpt()

    # -------------------------------------------------------------- _extract_chunk_info()
    def _extract_chunk_info(self, result: Dict[str, Any]) -> Tuple[str, str]:
        """Extract chunk_id and page from result."""
        chunk_id = result.get('chunk_id')
        page = result.get('page') or result.get('page_number') or result.get('page_num')
        
        # Check nested objects if not found
        if not chunk_id or page is None:
            for value in result.values():
                if isinstance(value, dict):
                    if not chunk_id:
                        chunk_id = value.get('chunk_id')
                    if page is None:
                        page = value.get('page') or value.get('page_number') or value.get('page_num')
        
        # Fallback construction
        if not chunk_id and page is not None:
            document = result.get('document') or result.get('title')
            if document:
                chunk_id = f"{document}_p{page}_c1"
        
        return chunk_id or "unknown", str(page or "")
    # -------------------------------------------------------------- end _extract_chunk_info()
    
    # -------------------------------------------------------------- _build_error_response()
    # =========================================================================
    # Error Handling
    # =========================================================================
    # -------------------------------------------------------------- _build_error_response()
    def _build_error_response(
        self,
        error_title: str,
        error_message: str,
        response_time_ms: int,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build standardized error response (maps to user-facing unknown message)."""
        # Delegate to IDK response while preserving diagnostic metadata
        meta = {
            "error": True,
            "error_type": error_title,
            "error_detail": error_message,
        }
        if extra_metadata:
            meta.update(extra_metadata)
        return self._build_idk_response(response_time_ms=response_time_ms, reason="error", extra_metadata=meta)
    # -------------------------------------------------------------- end _build_error_response()

    # -------------------------------------------------------------- _build_idk_response()
    def _build_idk_response(
        self,
        response_time_ms: int,
        reason: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build the mandated unknown-response payload for unrelated/failed traversal.

        The user-facing message is fixed by product requirements.
        """
        metadata: Dict[str, Any] = {
            "search_method": "llm_structural_cypher",
            "reason": reason,
            "response_time_ms": response_time_ms,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return {
            "answer": "I don't know — there are no documents or sources in the provided context matching your prompt",
            "cypher_query": "",
            "confidence": 0.0,
            "response_time_ms": response_time_ms,
            "metadata": metadata,
        }
    # -------------------------------------------------------------- end _build_idk_response()

    # -------------------------------------------------------------- _is_result_relevant()
    def _is_result_relevant(self, user_query: str, results: List[Dict[str, Any]], sample: int = 10) -> bool:
        """Lightweight relevance check between query keywords and result content.

        Returns False when there is negligible overlap, indicating likely unrelated content.
        """
        try:
            # Basic stopword set to filter common words
            stopwords = {
                "what", "is", "are", "the", "a", "an", "of", "for", "to", "in", "on",
                "and", "or", "with", "by", "about", "tell", "me", "how", "now"
            }
            # Extract query keywords
            q_words = [w.strip(".,!?;:()[]{}\"'\n\t").lower() for w in user_query.split()]
            query_terms = {w for w in q_words if len(w) > 2 and w not in stopwords}
            if not query_terms:
                return True  # Nothing meaningful to compare

            # Aggregate content from a sample of results
            content_tokens: set[str] = set()
            for row in results[: max(1, sample)]:
                text = self._extract_content_excerpt(row).lower()
                if not text:
                    continue
                tokens = [t.strip(".,!?;:()[]{}\"'\n\t") for t in text.split()]
                content_tokens.update(t for t in tokens if len(t) > 2)

            if not content_tokens:
                return False

            overlap = query_terms & content_tokens
            # Require minimal overlap ratio to treat as related
            overlap_ratio = len(overlap) / max(1, len(query_terms))
            return overlap_ratio >= settings.llm_structural_relevance_threshold
        except Exception:
            # Be permissive on any error
            return True
    
    # ______________________
    # Class Information Methods
    #
    # -------------------------------------------------------------- get_metrics()
    def get_metrics(self) -> Dict[str, Any]:
        """Get current engine performance metrics.

        Returns:
            Mapping of counters and rates for generation, validation, execution, and narrative
            summarization with citation safety, including optional domain-marker metrics when
            enabled by settings. Keys include totals, success rates, average execution time, and
            citation-related statistics.
        """
        base_metrics = {
            "total_generations": self.metrics.total_generations,
            "successful_generations": self.metrics.successful_generations,
            "validation_failures": self.metrics.validation_failures,
            "execution_failures": self.metrics.execution_failures,
            "success_rate": self.metrics.success_rate,
            "avg_execution_time_ms": self.metrics.avg_execution_time_ms,
            "uptime_seconds": int(time.time() - self.metrics.reset_timestamp)
        }
        
        # Add narrative and citation metrics
        narrative_metrics = {
            "narrative_attempts": self.metrics.narrative_attempts,
            "narrative_successes": self.metrics.narrative_successes,
            "narrative_failures": self.metrics.narrative_failures,
            "narrative_success_rate": self.metrics.narrative_success_rate,
            "citations_validated": self.metrics.citations_validated,
            "citations_dropped": self.metrics.citations_dropped,
            "references_sections_stripped": self.metrics.references_sections_stripped,
            "avg_sources_per_narrative": self.metrics.avg_sources_per_narrative,
            "avg_citations_per_answer": self.metrics.avg_citations_per_answer
        }
        
        # Add injection metrics
        injection_metrics = {
            "injection_skipped_aggregation": self.metrics.injection_skipped_aggregation,
            "injection_skipped_scope": self.metrics.injection_skipped_scope,
            "fields_injected_total": self.metrics.fields_injected_total
        }
        
        # Add domain citation metrics if enabled
        domain_metrics = {}
        if settings.llm_domain_citation_enabled:
            domain_metrics = {
                "domain_markers_found_legal": self.metrics.domain_markers_found_legal,
                "domain_markers_found_academic": self.metrics.domain_markers_found_academic,
                "domain_markers_found_technical": self.metrics.domain_markers_found_technical,
                "domain_markers_found_business": self.metrics.domain_markers_found_business,
                "domain_markers_found_medical": self.metrics.domain_markers_found_medical,
                "answers_with_domain_citations": self.metrics.answers_with_domain_citations,
                "domain_marker_extraction_errors": self.metrics.domain_marker_extraction_errors,
                "total_domain_markers_found": (
                    self.metrics.domain_markers_found_legal +
                    self.metrics.domain_markers_found_academic +
                    self.metrics.domain_markers_found_technical +
                    self.metrics.domain_markers_found_business +
                    self.metrics.domain_markers_found_medical
                ),
                "domain_citation_success_rate": (
                    self.metrics.answers_with_domain_citations / max(self.metrics.narrative_successes, 1)
                ) if self.metrics.narrative_successes > 0 else 0.0
            }
        
        # Combine all metrics
        return {
            **base_metrics,
            **narrative_metrics,
            **injection_metrics,
            **domain_metrics
        }
    # -------------------------------------------------------------- end get_metrics()
    
    # =========================================================================
    # Narrative Generation Pipeline with Validation
    # =========================================================================
    
    # -------------------------------------------------------------- _build_numbered_sources()
    def _build_numbered_sources(
        self, 
        results: List[Dict[str, Any]], 
        ref_map: Dict[str, int]
    ) -> List[Dict[str, Any]]:
        """
        Build pre-numbered sources for LLM summarization with domain markers.
        
        Args:
            results: Raw query results (limited to visible window)
            ref_map: Reference key to citation number mapping
            
        Returns:
            List of numbered sources with full content excerpts and domain markers
        """
        numbered_sources = []
        
        for result in results:
            # Extract reference components
            ref_key = self._get_reference_key_for_result(result)
            if not ref_key or ref_key not in ref_map:
                continue
                
            citation_num = ref_map[ref_key]
            
            # Extract content - prioritize text fields
            excerpt = self._extract_content_excerpt(result)
            
            # Extract chunk_id and page
            chunk_id, page = self._extract_chunk_info(result)
            
            numbered_sources.append({
                'n': citation_num,
                'chunk_id': chunk_id,
                'page': page,
                'excerpt': excerpt,
                'reference_key': ref_key
            })
        
        # Extract domain markers for all sources if enabled
        domain_markers = self._collect_source_domain_markers(numbered_sources)
        
        # Add domain markers to each source
        for source in numbered_sources:
            source['domain_markers'] = domain_markers.get(source['n'], {})
        
        # Sort by citation number for consistent ordering
        numbered_sources.sort(key=lambda x: x['n'])
        return numbered_sources
    # -------------------------------------------------------------- end _build_numbered_sources()
    
    # -------------------------------------------------------------- _compose_narrative_with_llm()
    async def _compose_narrative_with_llm(
        self,
        user_query: str,
        numbered_sources: List[Dict[str, Any]],
        schema_text: Optional[str] = None
    ) -> str:
        """
        Compose narrative answer using LLM with pre-numbered sources.
        
        Args:
            user_query: Original user query
            numbered_sources: Pre-numbered sources from _build_numbered_sources
            schema_text: Optional schema summary for domain context
            
        Returns:
            Raw narrative text with inline [n] citations
        """
        # Get narrative prompt builder
        prompt_builder = get_structural_narrative_prompt_builder(
            token_budget=settings.llm_structural_narrative_token_budget,
            preserve_numbers=settings.llm_structural_preserve_numbers
        )
        
        # Build prompt components
        prompt_components = prompt_builder.build_narrative_prompt(
            user_query=user_query,
            numbered_sources=numbered_sources,
            schema_text=schema_text
        )
        
        # Prepare LLM messages
        messages = [
            SystemMessage(content=prompt_components.system_prompt),
            HumanMessage(content=prompt_components.user_prompt)
        ]
        
        # Call LLM with timing
        timing_collector = None
        if OBSERVABILITY_AVAILABLE:
            try:
                timing_collector = get_timing_collector()
            except Exception:
                pass
        
        if timing_collector:
            async with timing_collector.measure("llm_structural_summarization_generation", {
                "sources_count": len(numbered_sources),
                "estimated_tokens": prompt_components.total_estimated_tokens,
                "schema_included": schema_text is not None
            }):
                llm_client = await self._get_llm_client()
                narrative = await llm_client.complete(messages)
        else:
            llm_client = await self._get_llm_client()
            narrative = await llm_client.complete(messages)
        
        return narrative or ""
    
    # -------------------------------------------------------------- _generate_narrative_answer()
    async def _generate_narrative_answer(
        self,
        user_query: str,
        execution_result: ExecutionResult,
        timing_context=None
    ) -> Tuple[str, int]:
        """
        Generate narrative with enhanced validation and citation tracking.
        
        Args:
            user_query: Original user query
            execution_result: Results from Cypher execution
            timing_context: Optional timing context for observability
            
        Returns:
            Tuple of (final_answer, actual_citations_count)
        """
        # Source selection (not truncation) based on heuristics
        visible_results = self._select_best_sources(
            execution_result.results, 
            max_sources=settings.llm_structural_source_max
        )
        
        document_refs = self._extract_document_references_from_results(visible_results)
        if not document_refs:
            raise ValueError("No document references for narrative generation")
        
        used_refs, ref_map = self._number_references(document_refs)
        numbered_sources = self._build_numbered_sources(visible_results, ref_map)
        
        # Track provided citation numbers for validation
        provided_numbers = {source['n'] for source in numbered_sources}
        
        # Get schema with token budget management
        schema_text = self.schema_manager.get_structural_summary_for_llm() if hasattr(self.schema_manager, 'get_structural_summary_for_llm') else None
        
        # Generate narrative
        raw_narrative = await self._compose_narrative_with_llm(
            user_query, numbered_sources, schema_text
        )
        
        # CRITICAL: Validate and clean narrative
        clean_narrative = self._validate_and_clean_narrative(raw_narrative, provided_numbers)
        
        if not clean_narrative.strip():
            raise ValueError("Empty narrative after validation")
        
        # Renumber chronologically (using LangChain-compatible logic)
        renumbered_answer, chronological_refs = self._renumber_citations_langchain_compatible(
            clean_narrative, used_refs
        )
        
        # Count actual citations in final answer for accurate metadata
        actual_citations_count = len(re.findall(r'\[(\d+)\]', renumbered_answer))
        
        # Append References section (always when available)
        final_answer = self._format_references_section_numbered_llm(chronological_refs)
        if final_answer:
            final_answer = renumbered_answer + final_answer
        else:
            final_answer = renumbered_answer
        
        # Extract domain markers from numbered sources for metrics
        domain_markers = {}
        for source in numbered_sources:
            if source.get('domain_markers'):
                domain_markers[source['n']] = source['domain_markers']
        
        # Update domain citation metrics
        self._update_domain_metrics(domain_markers, final_answer)
        
        self.logger.info(f"Generated narrative answer with {len(chronological_refs)} citations")
        return final_answer, actual_citations_count
    # -------------------------------------------------------------- end _generate_narrative_answer()

    # -------------------------------------------------------------- _generate_fallback_answer()
    async def _generate_fallback_answer(
        self,
        execution_result: ExecutionResult,
        timing_context=None
    ) -> str:
        """Generate answer using existing deterministic rendering."""
        # Use existing _format_execution_results logic
        visible_results = execution_result.results[:settings.llm_structural_source_max]
        document_refs = self._extract_document_references_from_results(visible_results)
        
        if document_refs:
            used_refs, ref_map = self._number_references(document_refs)
            answer = self._format_execution_results(execution_result.results, ref_map)
            references_section = self._format_references_section_numbered_llm(used_refs)
            if references_section:
                answer += references_section
        else:
            answer = self._format_execution_results(execution_result.results)
            self.logger.warning("Fallback: No reference fields for citations")

        return answer
    # -------------------------------------------------------------- end _generate_fallback_answer()

    # -------------------------------------------------------------- _update_narrative_metrics()
    def _update_narrative_metrics(
        self, 
        success: bool, 
        sources_count: int, 
        citations_count: int
    ):
        """Update narrative-specific metrics."""
        self.metrics.narrative_attempts += 1
        
        if success:
            self.metrics.narrative_successes += 1
            
            # Update running averages
            total_narratives = self.metrics.narrative_successes
            self.metrics.avg_sources_per_narrative = (
                (self.metrics.avg_sources_per_narrative * (total_narratives - 1) + sources_count) / total_narratives
            )
            self.metrics.avg_citations_per_answer = (
                (self.metrics.avg_citations_per_answer * (total_narratives - 1) + citations_count) / total_narratives
            )
        else:
            self.metrics.narrative_failures += 1
    # -------------------------------------------------------------- end _update_narrative_metrics()
    
    # -------------------------------------------------------------- reset_metrics()
    def reset_metrics(self) -> None:
        """Reset engine metrics to initial values.

        Returns:
            None
        """
        self.metrics = EngineMetrics()
        self.logger.info("Engine metrics reset")
    # -------------------------------------------------------------- end reset_metrics()

# ------------------------------------------------------------------------- end class LLMStructuralCypherEngine

# __________________________________________________________________________
# Standalone Function Definitions
#
# --------------------------------------------------------------------------------- get_llm_structural_cypher_engine()
def get_llm_structural_cypher_engine() -> LLMStructuralCypherEngine:
    """
    Get singleton instance of LLM Structural Cypher Engine.
    
    Returns:
        Singleton LLMStructuralCypherEngine instance
    """
    global _engine_instance
    
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = LLMStructuralCypherEngine()
        
        return _engine_instance
# --------------------------------------------------------------------------------- end get_llm_structural_cypher_engine()

# __________________________________________________________________________
# End of File
