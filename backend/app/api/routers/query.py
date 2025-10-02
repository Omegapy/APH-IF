# -------------------------------------------------------------------------
# File: query.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/app/api/routers/query.py
# -------------------------------------------------------------------------
# Project: APH-IF
#
# Project description:
# Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF)
# is a novel Retrieval Augmented Generation (RAG) system that differs from
# traditional RAG approaches by performing semantic and traversal searches
# concurrently, rather than sequentially, and fusing the results using an LLM
# or an LRM to generate the final response.
# -------------------------------------------------------------------------
#
# --- Module Functionality ---
#   Implements FastAPI query endpoints responsible for orchestrating semantic,
#   traversal, and hybrid retrieval pipelines with detailed monitoring,
#   structured responses, and timing instrumentation.
# -------------------------------------------------------------------------
#
# --- Module Contents Overview ---
# - Router: router (FastAPI APIRouter instance)
# - Endpoint: process_query (textual response)
# - Endpoint: process_query_structured (structured JSON response)
# - Constant: __all__ (module exports)
# -------------------------------------------------------------------------
#
# --- Dependencies / Imports ---
# - Standard Library: logging, time, uuid, datetime, typing (Any)
# - Third-Party: fastapi (APIRouter, Header, HTTPException)
# - Local Project Modules: app.api.state, app.api.models, app.api.utils,
#   app.core.config, app.models.structured_responses, app.monitoring (timing,
#   performance), app.search (tools, engines)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
#
# --- Usage / Integration ---
# Included by `backend/app/main.py` to expose the primary query interfaces for
# the APH-IF platform, providing both human-readable and structured responses
# for frontend and service consumers.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""FastAPI query router powering APH-IF retrieval and fusion workflows.

Provides endpoints that execute vector, traversal, and hybrid retrieval flows,
apply intelligent fusion, collect detailed timing metadata, and return either
textual or structured responses for downstream consumers.
"""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException

from app.api import state as api_state
from app.api.models import QueryRequest, QueryResponse
from app.api.utils import get_semantic_k, get_traversal_k, is_unknown_text
from app.core.config import settings
from app.models.structured_responses import (
    StructuredQueryResponse,
    create_engine_summary,
    normalize_fusion_result,
    normalize_semantic_result,
    normalize_traversal_result,
)

logger = logging.getLogger("app.main")

router = APIRouter(prefix="/query")


# __________________________________________________________________________
# Endpoint Definitions
#
# -------------------------------------------------------------- process_query()
# __________________________________________________________________________
# Endpoint Definitions
#
# -------------------------------------------------------------- process_query()
@router.post("", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
) -> QueryResponse:
    """Execute a user query through the configured retrieval pipeline.

    Args:
        request: Validated query payload containing the prompt and search
            parameters.
        session_id: Optional session header used to maintain conversational
            context. When absent, a new session identifier is generated.

    Returns:
        QueryResponse: Textual response enriched with timing metadata,
        structured diagnostics, and the effective session identifier.

    Raises:
        HTTPException: If the caller requests an unsupported search type.
    """

    from app.monitoring.timing_collector import get_timing_collector

    timing_collector = get_timing_collector()

    async with timing_collector.measure(
        "total_request",
        {
            "query_length": len(request.user_input),
            "search_type": request.search_type,
            "session_id": session_id or "anonymous",
        },
    ) as total_timer:
        async with timing_collector.measure("request_preprocessing") as prep_timer:
            start_time = time.time()
            current_session_id = (
                session_id or request.session_id or str(uuid.uuid4())
            )

            prep_timer.add_metadata(
                {
                    "session_creation": current_session_id
                    != (session_id or request.session_id),
                    "parameter_validation": True,
                }
            )

            logger.info(
                "Processing query for session %s", current_session_id[:8]
            )

        async with timing_collector.measure("cache_lookup") as cache_timer:
            cache_timer.add_metadata({"cache_enabled": True})

        try:
            if request.search_type == "graph":
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Search type 'graph' removed. Use 'graph_llm_structural' "
                        "or 'hybrid'"
                    ),
                )

            if request.search_type == "graph_llm_structural":
                try:
                    from app.search.tools.cypher import (
                        query_knowledge_graph_llm_structural_detailed,
                    )

                    detailed_result = (
                        await query_knowledge_graph_llm_structural_detailed(
                            request.user_input,
                            max_results=get_traversal_k(
                                request.max_results_traversal_search
                            ),
                        )
                    )

                    search_method = detailed_result.get("metadata", {}).get(
                        "search_method",
                        "unknown",
                    )

                    response_text = (
                        "LLM Structural Cypher Search Results (via "
                        f"{search_method}):\n\n"
                    )
                    response_text += detailed_result["answer"]

                    if detailed_result.get("cypher_query"):
                        response_text += "\n\n--- Query Details ---\n"
                        response_text += (
                            "Generated Cypher: "
                            f"{detailed_result['cypher_query']}\n"
                        )
                        response_text += (
                            "Confidence: "
                            f"{detailed_result['confidence']:.2f}\n"
                        )
                        response_text += (
                            "Response Time: "
                            f"{detailed_result['response_time_ms']}ms\n"
                        )
                        response_text += f"Search Method: {search_method}"

                        metadata = detailed_result.get("metadata", {})
                        if metadata.get("tokens_used"):
                            response_text += (
                                "\nTokens Used: "
                                f"{metadata['tokens_used']}\n"
                            )
                            response_text += (
                                "Model Used: "
                                f"{metadata.get('model_used', 'unknown')}"
                            )
                        if metadata.get("validation_issues"):
                            response_text += (
                                "\nValidation Issues: "
                                f"{metadata['validation_issues']}"
                            )
                        if metadata.get("fixes_applied"):
                            response_text += (
                                "\nFixes Applied: "
                                f"{metadata['fixes_applied']}"
                            )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "LLM structural Cypher search error: %s", exc
                    )
                    response_text = (
                        "LLM structural Cypher search failed: "
                        f"{exc}"
                    )

            elif request.search_type == "vector":
                try:
                    from app.search.tools.vector import search_semantic

                    vector_result = await search_semantic(
                        request.user_input,
                        k=get_semantic_k(
                            request.max_results_semantic_search
                        ),
                    )
                    response_text = vector_result.get(
                        "answer", "No results returned"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Vector search error: %s", exc)
                    response_text = f"Vector search failed: {exc}"

            elif request.search_type == "hybrid" and api_state.HYBRID_AVAILABLE:
                try:
                    from app.monitoring.performance_monitor import (
                        get_performance_monitor,
                    )
                    from app.search.context_fusion import get_fusion_engine
                    from app.search.parallel_hybrid import get_parallel_engine

                    parallel_engine = get_parallel_engine()
                    fusion_engine = get_fusion_engine()
                    monitor = get_performance_monitor()

                    async with monitor.track_operation(
                        "parallel_retrieval"
                    ) as tracker:
                        parallel_result = await parallel_engine.retrieve_parallel(
                            request.user_input,
                            semantic_k=get_semantic_k(
                                request.max_results_semantic_search
                            ),
                            traversal_max_results=get_traversal_k(
                                request.max_results_traversal_search
                            ),
                        )

                        tracker.add_metadata(
                            {
                                "success": parallel_result.success,
                                "fusion_ready": parallel_result.fusion_ready,
                                "semantic_success": parallel_result.
                                semantic_result.success,
                                "traversal_success": parallel_result.
                                traversal_result.success,
                                "total_time_ms": parallel_result.total_time_ms,
                            }
                        )

                    if not parallel_result.success:
                        response_text = (
                            "Parallel search failed: Both vector and graph "
                            "searches encountered errors.\n\n"
                        )
                        response_text += (
                            "Vector: "
                            f"{parallel_result.semantic_result.error or 'Low confidence'}\n"
                        )
                        response_text += (
                            "Graph: "
                            f"{parallel_result.traversal_result.error or 'Low confidence'}"
                        )
                    elif not parallel_result.fusion_ready:
                        if (
                            is_unknown_text(
                                parallel_result.semantic_result.content
                            )
                            and is_unknown_text(
                                parallel_result.traversal_result.content
                            )
                        ):
                            response_text = (
                                "I don't know - there are no documents or "
                                "sources in the provided context matching "
                                "your prompt"
                            )
                            response_text += (
                                "\n\n--- Parallel Search Details ---\n"
                            )
                            response_text += (
                                "Total Processing Time: "
                                f"{parallel_result.total_time_ms}ms\n"
                            )
                            response_text += (
                                "Both Successful: "
                                f"{parallel_result.both_successful}"
                            )
                        else:
                            if (
                                parallel_result.semantic_result.confidence
                                >= parallel_result.traversal_result.confidence
                            ):
                                response_text = (
                                    "**Vector Search Results** (Confidence: "
                                    f"{parallel_result.semantic_result.confidence:.2f}):\n\n"
                                )
                                response_text += (
                                    parallel_result.semantic_result.content
                                )
                            else:
                                response_text = (
                                    "**Graph Search Results** (Confidence: "
                                    f"{parallel_result.traversal_result.confidence:.2f}):\n\n"
                                )
                                response_text += (
                                    parallel_result.traversal_result.content
                                )

                            response_text += (
                                "\n\n--- Parallel Search Details ---\n"
                            )
                            response_text += (
                                "Total Processing Time: "
                                f"{parallel_result.total_time_ms}ms\n"
                            )
                            response_text += (
                                "Primary Method: "
                                f"{parallel_result.primary_method}\n"
                            )
                            response_text += (
                                "Both Successful: "
                                f"{parallel_result.both_successful}"
                            )
                    else:
                        async with monitor.track_operation(
                            "context_fusion"
                        ) as fusion_tracker:
                            fusion_result = await fusion_engine.fuse_contexts(
                                parallel_result
                            )

                            fusion_tracker.add_metadata(
                                {
                                    "confidence": fusion_result.final_confidence,
                                    "fusion_strategy": (
                                        fusion_result.fusion_strategy
                                    ),
                                    "citation_accuracy": (
                                        fusion_result.citation_accuracy
                                    ),
                                    "vector_contribution": (
                                        fusion_result.vector_contribution
                                    ),
                                    "graph_contribution": (
                                        fusion_result.graph_contribution
                                    ),
                                    "citations_preserved": len(
                                        fusion_result.citations_preserved
                                    ),
                                    "domain_adaptation": (
                                        fusion_result.domain_adaptation
                                    ),
                                }
                            )

                            response_text = (
                                "**APH-IF Hybrid Search Results** (Confidence: "
                                f"{fusion_result.final_confidence:.2f}):\n\n"
                            )
                            response_text += fusion_result.response_text

                            if fusion_result.citations_preserved:
                                response_text += "\n\n--- Citations ---\n"
                                for citation in fusion_result.citations_preserved:
                                    response_text += f"• {citation}\n"

                            response_text += (
                                "\n--- Fusion Details ---\n"
                                f"Strategy: {fusion_result.fusion_strategy}\n"
                                f"Vector Contribution: {fusion_result.vector_contribution:.2f}\n"
                                f"Graph Contribution: {fusion_result.graph_contribution:.2f}\n"
                                f"Citation Accuracy: {fusion_result.citation_accuracy:.2f}\n"
                            )

                except Exception as exc:  # noqa: BLE001
                    logger.error("Parallel hybrid search error: %s", exc)
                    response_text = (
                        "Parallel hybrid search failed due to an error: "
                        f"{exc}"
                    )

            elif request.search_type == "hybrid":
                response_text = (
                    "APH-IF Parallel Hybrid Search System Ready\n\n"
                    f"Query: '{request.user_input}'\n\n"
                    "Available search modes:\n"
                    "• `hybrid` - Parallel Vector + Graph search with "
                    "intelligent fusion\n"
                    "• `vector` - Semantic similarity search only\n"
                    "• `graph_llm_structural` - LLM-powered natural language to "
                    "Cypher generation\n\n"
                    f"Current mode: {request.search_type}\n"
                    "Use search_type='hybrid' for full APH-IF processing."
                )
            else:
                response_text = (
                    f"Query received: '{request.user_input}'\n\n"
                    f"Search type: {request.search_type}\n\n"
                    "Available search types: 'graph_llm_structural' (LLM "
                    "Cypher), 'vector' (VectorRAG), 'hybrid' (both)"
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Query processing error: %s", exc)
            response_text = f"Query processing failed: {exc}"

        async with timing_collector.measure("response_generation") as resp_timer:
            processing_time = time.time() - start_time

            api_state.active_sessions[current_session_id] = {
                "last_query": request.user_input,
                "timestamp": datetime.now().isoformat(),
                "query_count": api_state.active_sessions.get(
                    current_session_id,
                    {},
                ).get("query_count", 0)
                + 1,
            }

            detailed_breakdown = timing_collector.generate_detailed_breakdown()

            metadata: dict[str, Any] = {
                "search_type": request.search_type,
                "processing_time_ms": int(processing_time * 1000),
                "session_info": api_state.active_sessions[current_session_id],
                "hybrid_available": api_state.HYBRID_AVAILABLE,
                "environment": settings.environment_mode.value,
                "parameters_used": {
                    "semantic_k": (
                        get_semantic_k(request.max_results_semantic_search)
                        if request.search_type in {"vector", "hybrid"}
                        else None
                    ),
                    "traversal_max_results": (
                        get_traversal_k(
                            request.max_results_traversal_search
                        )
                        if request.search_type
                        in {"graph_llm_structural", "hybrid"}
                        else None
                    ),
                    "semantic_requested": (
                        request.max_results_semantic_search
                    ),
                    "traversal_requested": (
                        request.max_results_traversal_search
                    ),
                },
                "detailed_timing": {
                    "total_request_response_time_ms": (
                        detailed_breakdown.total_request_response_time_ms
                    ),
                    "request_preprocessing_time_ms": (
                        detailed_breakdown.request_parsing_time_ms
                        + detailed_breakdown.parameter_validation_time_ms
                    ),
                    "database_access_time_ms": (
                        detailed_breakdown.cypher_query_execution_time_ms
                        + detailed_breakdown.neo4j_connection_acquisition_time_ms
                    ),
                    "cache_operation_time_ms": (
                        detailed_breakdown.cache_lookup_time_ms
                        + detailed_breakdown.cache_storage_time_ms
                    ),
                    "parallel_coordination_time_ms": (
                        detailed_breakdown.parallel_coordination_overhead_ms
                    ),
                    "response_generation_time_ms": (
                        detailed_breakdown.content_formatting_time_ms
                    ),
                },
                "database_metrics": "disabled",
            }

            resp_timer.add_metadata(
                {
                    "response_length": len(response_text),
                    "metadata_size": len(str(metadata)),
                }
            )

            total_timer.add_metadata(
                {
                    "final_processing_time_ms": int(processing_time * 1000),
                    "response_generated": True,
                    "total_operations": len(
                        timing_collector.get_recent_timings(10)
                    ),
                }
            )

        return QueryResponse(
            response=response_text,
            session_id=current_session_id,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            metadata=metadata,
        )


# -------------------------------------------------------------- end process_query()


# -------------------------------------------------------------- process_query_structured()
@router.post("/structured", response_model=StructuredQueryResponse)
async def process_query_structured(
    request: QueryRequest,
    session_id: Optional[str] = Header(None, alias="X-Session-ID"),
) -> StructuredQueryResponse:
    """Execute a query and return normalized structured results.

    Args:
        request: Validated query payload containing the prompt and search
            parameters.
        session_id: Optional session header used to maintain conversational
            context. When absent, a new session identifier is generated.

    Returns:
        StructuredQueryResponse: Structured response including per-engine
        outputs, timing metrics, and diagnostic metadata.

    Raises:
        HTTPException: If the caller requests an unsupported search type.
    """

    from app.monitoring.timing_collector import get_timing_collector

    timing_collector = get_timing_collector()

    async with timing_collector.measure(
        "structured_query",
        {
            "query_length": len(request.user_input),
            "search_type": request.search_type,
            "session_id": session_id or "anonymous",
        },
    ) as total_timer:
        start_time = time.time()
        current_session_id = (
            session_id or request.session_id or str(uuid.uuid4())
        )

        logger.info(
            "Processing structured query for session %s",
            current_session_id[:8],
        )

        response = StructuredQueryResponse(
            query=request.user_input,
            search_type=request.search_type,
            session_id=current_session_id,
            success=False,
            processing_time_ms=0,
            timestamp=datetime.now().isoformat(),
            engine_metadata={},
        )

        try:
            if request.search_type == "vector":
                try:
                    from app.search.tools.vector import search_semantic_detailed

                    vector_result = await search_semantic_detailed(
                        request.user_input,
                        k=get_semantic_k(
                            request.max_results_semantic_search
                        ),
                    )

                    response.semantic_result = normalize_semantic_result(
                        vector_result,
                        int((time.time() - start_time) * 1000),
                    )
                    response.success = True
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Structured vector search failed: %s", exc
                    )
                    response.error = f"Vector search failed: {exc}"

            elif request.search_type == "graph":
                response.error = (
                    "Search type 'graph' removed. Use 'graph_llm_structural' "
                    "or 'hybrid'"
                )

            elif request.search_type == "graph_llm_structural":
                try:
                    from app.search.tools.cypher import (
                        query_knowledge_graph_llm_structural_detailed,
                    )

                    traversal_result = (
                        await query_knowledge_graph_llm_structural_detailed(
                            request.user_input,
                            max_results=get_traversal_k(
                                request.max_results_traversal_search
                            ),
                        )
                    )

                    response.traversal_result = normalize_traversal_result(
                        traversal_result,
                        int((time.time() - start_time) * 1000),
                    )
                    response.success = True
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Structured traversal search failed: %s", exc
                    )
                    response.error = f"Traversal search failed: {exc}"

            elif request.search_type == "hybrid" and api_state.HYBRID_AVAILABLE:
                try:
                    from app.monitoring.performance_monitor import (
                        get_performance_monitor,
                    )
                    from app.search.context_fusion import get_fusion_engine
                    from app.search.parallel_hybrid import get_parallel_engine

                    parallel_engine = get_parallel_engine()
                    fusion_engine = get_fusion_engine()
                    monitor = get_performance_monitor()

                    async with monitor.track_operation(
                        "structured_parallel_retrieval"
                    ) as tracker:
                        parallel_result = await parallel_engine.retrieve_parallel(
                            request.user_input,
                            semantic_k=get_semantic_k(
                                request.max_results_semantic_search
                            ),
                            traversal_max_results=get_traversal_k(
                                request.max_results_traversal_search
                            ),
                        )

                        tracker.add_metadata(
                            {
                                "success": parallel_result.success,
                                "fusion_ready": parallel_result.fusion_ready,
                            }
                        )

                    if parallel_result.success:
                        if parallel_result.semantic_result.success:
                            semantic_dict = {
                                "answer": parallel_result.semantic_result.content,
                                "confidence": parallel_result.semantic_result.confidence,
                                "sources": parallel_result.semantic_result.sources,
                                "entities_found": parallel_result.semantic_result.entities,
                                "metadata": parallel_result.semantic_result.metadata,
                            }
                            response.semantic_result = normalize_semantic_result(
                                semantic_dict,
                                parallel_result.semantic_result.response_time_ms,
                            )

                        if parallel_result.traversal_result.success:
                            traversal_dict = {
                                "answer": parallel_result.traversal_result.content,
                                "confidence": (
                                    parallel_result.traversal_result.confidence
                                ),
                                "cypher_query": (
                                    parallel_result.traversal_result.sources[0].get(
                                        "cypher_query",
                                        "",
                                    )
                                    if parallel_result.traversal_result.sources
                                    else ""
                                ),
                                "metadata": (
                                    parallel_result.traversal_result.metadata
                                ),
                            }
                            response.traversal_result = normalize_traversal_result(
                                traversal_dict,
                                parallel_result.traversal_result.response_time_ms,
                            )

                        if not parallel_result.fusion_ready:
                            if (
                                is_unknown_text(
                                    parallel_result.semantic_result.content
                                )
                                and is_unknown_text(
                                    parallel_result.traversal_result.content
                                )
                            ):
                                standard_unknown = (
                                    "I don't know - there are no documents or "
                                    "sources in the provided context matching "
                                    "your prompt"
                                )

                                if response.semantic_result:
                                    response.semantic_result.content = (
                                        standard_unknown
                                    )
                                    response.semantic_result.confidence.original = 0.1
                                    response.semantic_result.confidence.capped = 0.1
                                if response.traversal_result:
                                    response.traversal_result.content = (
                                        standard_unknown
                                    )
                                    response.traversal_result.confidence.original = 0.1
                                    response.traversal_result.confidence.capped = 0.1

                        if parallel_result.fusion_ready:
                            async with monitor.track_operation(
                                "structured_fusion"
                            ) as fusion_tracker:
                                fusion_result = await fusion_engine.fuse_contexts(
                                    parallel_result
                                )

                                response.fusion_result = normalize_fusion_result(
                                    fusion_result,
                                    fusion_result.processing_time_ms,
                                )

                                fusion_tracker.add_metadata(
                                    {
                                        "confidence": (
                                            fusion_result.final_confidence
                                        ),
                                        "strategy": fusion_result.fusion_strategy,
                                    }
                                )

                        response.success = True
                    else:
                        response.error = (
                            "Both vector and graph searches failed"
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Structured hybrid search failed: %s", exc
                    )
                    response.error = f"Hybrid search failed: {exc}"
            else:
                response.error = (
                    f"Search type '{request.search_type}' not supported or "
                    "hybrid modules unavailable"
                )

            processing_time = time.time() - start_time
            response.processing_time_ms = int(processing_time * 1000)

            detailed_breakdown = timing_collector.generate_detailed_breakdown()
            timing_dict = (
                detailed_breakdown.to_dict()
                if hasattr(detailed_breakdown, "to_dict")
                else {}
            )

            response.engine_metadata = create_engine_summary(
                semantic_result=response.semantic_result,
                traversal_result=response.traversal_result,
                fusion_result=response.fusion_result,
            )
            response.engine_metadata.update(
                {
                    "timing_breakdown": timing_dict,
                    "database_metrics": "disabled",
                    "session_info": {
                        "session_id": current_session_id,
                        "query_count": api_state.active_sessions.get(
                            current_session_id,
                            {},
                        ).get("query_count", 0)
                        + 1,
                    },
                }
            )

            api_state.active_sessions[current_session_id] = {
                "last_query": request.user_input,
                "timestamp": datetime.now().isoformat(),
                "query_count": api_state.active_sessions.get(
                    current_session_id,
                    {},
                ).get("query_count", 0)
                + 1,
            }

            total_timer.add_metadata(
                {
                    "final_processing_time_ms": response.processing_time_ms,
                    "response_generated": True,
                    "success": response.success,
                }
            )

            logger.info(
                "✅ Structured query completed: success=%s, time=%sms, engines=%s",
                response.success,
                response.processing_time_ms,
                len(response.engine_metadata.get("engines_used", [])),
            )

            return response
        except Exception as exc:  # noqa: BLE001
            logger.error("Structured query processing failed: %s", exc)
            processing_time = time.time() - start_time
            response.processing_time_ms = int(processing_time * 1000)
            response.error = f"Query processing failed: {exc}"
            return response


# -------------------------------------------------------------- end process_query_structured()

# __________________________________________________________________________
# Module Exports
#
__all__ = ["router"]

# __________________________________________________________________________
# End of File