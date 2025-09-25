# -------------------------------------------------------------------------
# File: vector.py
# Author: Alexander Ricciardi
# Date: 2025-09-15
# [File Path] backend/app/search/tools/vector.py
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
#   VectorRAG utilities and engine for semantic similarity search over the
#   knowledge graph. Provides provider adapters (embeddings, Neo4j vector
#   retriever, async LLM client, citation processor), the semantic search
#   engine, citation/domain-marker helpers, and public API facades used by the
#   backend service and the hybrid retrieval pipeline.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Class: SemanticSearchProviders (provider adapters)
# - Class: SemanticSearchEngine (semantic VectorRAG engine)
# - Functions (utility): extract_citations_from_answer, renumber_citations_chronologically,
#   format_references_section, extract_document_title_from_chunk_id
# - Functions (helpers): _detect_domain_markers, _validate_inline_citations,
#   _apply_domain_markers, process_result_with_enhanced_citations,
#   format_enhanced_references_section
# - Public API: get_vector_engine, get_engine_stats, search_semantic,
#   search_semantic_detailed, get_vector_tool, test_vector_search
# - Globals: _engine_instance, _engine_lock, _engine_stats
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: asyncio, logging, threading, time, datetime, typing
# - Third-Party: langchain_openai (OpenAIEmbeddings), langchain.tools (Tool)
# - Local Project Modules: app.core.config (settings, EMBEDDING_CONFIG),
#   app.core.llm_client, app.schema.schema_manager
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# The semantic engine is used by API endpoints and the parallel hybrid engine
# to perform vector-based retrieval with schema-aware prompting and robust
# citation handling. Public facades (e.g., search_semantic_detailed) are
# consumed by higher layers and tools.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Vector Search Tool for APH-IF Backend

VectorRAG implementation for semantic similarity search using OpenAI embeddings
and Neo4j vector indexing for high-performance semantic retrieval in domain-agnostic
knowledge graphs.

Key Features:
- OpenAI text-embedding-3-large (3072 dimensions)
- Neo4j vector indexing for similarity search
- Domain-agnostic knowledge graph compatibility
- Comprehensive entity and document metadata
- Advanced error handling and fallback mechanisms
- Health monitoring and performance metrics
"""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import logging
import time
import asyncio
import threading
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain.tools import Tool
except ImportError as e:
    logging.warning(f"LangChain imports failed: {e}. Vector search will use fallback mode.")
    # Fallback imports will be handled in function implementations

from ...core.config import settings, EMBEDDING_CONFIG
from ...core.llm_client import get_openai_client
from ...schema.schema_manager import get_schema_manager

logger = logging.getLogger(__name__)

# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# Utility Functions
# =========================================================================
# Citation Processing Utilities (Extracted for Better Testing)
# =========================================================================

# --------------------------------------------------------------------------------- extract_citations_from_answer()
def extract_citations_from_answer(answer: str) -> List[int]:
    """Extract numeric citation markers from answer text.

    Args:
        answer: LLM response text containing citations in the form "[n]".

    Returns:
        List of citation numbers (e.g., [1, 2, 3]) found in order of appearance.
    """
    import re
    citations = re.findall(r'\[(\d+)\]', answer)
    return [int(c) for c in citations]
# --------------------------------------------------------------------------------- end extract_citations_from_answer()

# --------------------------------------------------------------------------------- renumber_citations_chronologically()
def renumber_citations_chronologically(
    answer: str,
    reference_list: List[Dict[str, Any]],
    citations_used: List[int],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Renumber citations to be chronological (1, 2, 3...) in order of appearance.
    
    Args:
        answer: The LLM response text with citations
        reference_list: List of reference dictionaries with metadata
        citations_used: List of citation numbers that were used
        
    Returns:
        Tuple of (renumbered_answer, used_references)
    """
    import re
    
    # Get unique citation numbers in the order they appear in the answer
    citations_in_order = []
    seen = set()
    for citation in re.findall(r'\[(\d+)\]', answer):
        citation_num = int(citation)
        if citation_num not in seen and citation_num in citations_used:
            citations_in_order.append(citation_num)
            seen.add(citation_num)
    
    # Create mapping from old numbers to new chronological numbers
    citation_map = {}
    used_references = []
    
    for new_number, old_number in enumerate(citations_in_order, 1):
        citation_map[old_number] = new_number
        
        # Find the reference with this old number
        for ref in reference_list:
            if ref.get('number') == old_number:
                # Create new reference with chronological number
                used_references.append({
                    'number': new_number,
                    'chunk_id': ref.get('chunk_id', 'Unknown'),
                    'page': ref.get('page', ''),
                    'document': ref.get('document', '')
                })
                break
    
    # Renumber citations in the answer
    def replace_citation(match):
        old_num = int(match.group(1))
        new_num = citation_map.get(old_num, old_num)
        return f"[{new_num}]"
    
    renumbered_answer = re.sub(r'\[(\d+)\]', replace_citation, answer)
    return renumbered_answer, used_references
# --------------------------------------------------------------------------------- end renumber_citations_chronologically()

# --------------------------------------------------------------------------------- format_references_section()
def format_references_section(document_references: List[Dict[str, Any]]) -> str:
    """Format numbered chunk references into a terminal References section.

    Args:
        document_references: Reference dicts with 'number', 'chunk_id', and optional 'page'.

    Returns:
        A string with a "References" section suitable for appending to an answer.
    """
    if not document_references:
        return ""
    
    references = []
    references.append("\n\n**References:**")
    
    # Use the pre-assigned numbers from references
    for ref in document_references:
        number = ref.get('number', '?')
        chunk_id = ref.get('chunk_id', 'Unknown')
        page = ref.get('page', '')
        
        if page:
            references.append(f"[{number}] {chunk_id}, p.{page}")
        else:
            references.append(f"[{number}] {chunk_id}")
    
    references.append("")
    return "\n".join(references)
# --------------------------------------------------------------------------------- end format_references_section()

# --------------------------------------------------------------------------------- process_result_with_citations()
def process_result_with_citations(raw_answer: str, reference_list: List[Dict[str, Any]]) -> str:
    """Attach a numbered References section based on inline citations.

    Args:
        raw_answer: Answer text produced by the LLM.
        reference_list: Reference metadata corresponding to sources in context.

    Returns:
        Answer string with citations renumbered chronologically and a References section
        appended when applicable.
    """
    # Extract citations from the answer
    citations_used = extract_citations_from_answer(raw_answer)
    
    if not citations_used or not reference_list:
        return raw_answer
    
    # Renumber citations chronologically
    processed_answer, used_references = renumber_citations_chronologically(raw_answer, reference_list, citations_used)
    
    # Add references section if we have used citations
    if used_references:
        references_section = format_references_section(used_references)
        return processed_answer + references_section
    
    return processed_answer
# --------------------------------------------------------------------------------- end process_result_with_citations()

# --------------------------------------------------------------------------------- extract_document_title_from_chunk_id()
def extract_document_title_from_chunk_id(
    chunk_id: str,
    document_title: Optional[str] = None,
) -> str:
    """Derive a document name from a chunk_id when no title is present.

    Args:
        chunk_id: Identifier like "filename_pN_cM" (e.g., "CFR-2024-title30-vol1.pdf_p372_c2").
        document_title: An existing title to prefer if available.

    Returns:
        A document title derived from the chunk_id, or the provided title when given.
    """
    if document_title:
        return document_title
        
    if not chunk_id or chunk_id == "Unknown":
        return "Unknown"
# --------------------------------------------------------------------------------- end extract_document_title_from_chunk_id()
    
    # Extract document name from chunk_id (format: filename_pN_cN)
    if "_p" in chunk_id:
        return chunk_id.split("_p")[0]
    else:
        return "Unknown"

# __________________________________________________________________________
# Global Constants / Variables
#
# =========================================================================
# Simple Engine Stats + Singleton
# =========================================================================

_engine_instance = None
_engine_lock = threading.RLock()
_engine_stats = {
    "total_requests": 0,
    "last_request_ts": None,
    "current_engine_mode": "semantic_cached",
}

# --------------------------------------------------------------------------------- get_vector_engine()
def get_vector_engine() -> "SemanticSearchEngine":
    """Get or create the process-wide semantic search engine singleton.

    Returns:
        SemanticSearchEngine: Cached engine instance ready for semantic queries.
    """
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = SemanticSearchEngine()
        return _engine_instance
# --------------------------------------------------------------------------------- end get_vector_engine()

# --------------------------------------------------------------------------------- get_engine_stats()
def get_engine_stats() -> Dict[str, Any]:
    """Return lightweight health and usage statistics for the vector engine.

    Returns:
        dict: Engine health, total requests, current mode, and last request timestamp.
    """
    status = "uninitialized" if _engine_stats["total_requests"] == 0 else "healthy"
    return {
        "engine_health": status,
        "total_requests": _engine_stats["total_requests"],
        "current_engine_mode": _engine_stats["current_engine_mode"],
        "last_request_timestamp": _engine_stats["last_request_ts"],
    }
# --------------------------------------------------------------------------------- end get_engine_stats()

# --------------------------------------------------------------------------------- _build_simple_metadata()
def _build_simple_metadata(k: int, score_threshold: float) -> Dict[str, Any]:
    """Compose compact, stable metadata for semantic responses.

    Args:
        k: Number of chunks retrieved from the vector index.
        score_threshold: Minimum similarity score for candidate chunks.

    Returns:
        dict: Engine, retriever, embedding, index, and feature flags snapshot.
    """
    from ...core.config import EMBEDDING_CONFIG, settings

    return {
        "engine_mode": "semantic_cached",
        "retriever": {"k": k, "score_threshold": score_threshold},
        "embeddings": {
            "model": EMBEDDING_CONFIG["model"],
            "dimensions": EMBEDDING_CONFIG["dimensions"],
        },
        "vector_index": {"name": "chunk_embedding_index", "node_label": "Chunk"},
        "features": {
            "schema_aware": bool(getattr(settings, "semantic_use_structural_schema", False)),
            "citation_validation": bool(getattr(settings, "semantic_citation_validation", True)),
            "domain_enforcement": bool(getattr(settings, "semantic_domain_enforcement", True)),
        },
    }
# --------------------------------------------------------------------------------- end _build_simple_metadata()

# =========================================================================
# Service Provider Adapters (Reuse Existing Backend Services)
# =========================================================================

# __________________________________________________________________________
# Class Definitions
#
# ------------------------------------------------------------------------- class SemanticSearchProviders
class SemanticSearchProviders:
    """Provider adapters for embeddings, vector index, LLM client, and citations.

    The engine uses this adapter to reuse existing backend services (embedding client,
    Neo4j vector retriever, async LLM client, and the citation processor) without
    re-implementing them. All resources are initialized lazily and cached per-process.
    """
    
    # ______________________
    # Constructor
    #
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self) -> None:
        """Initialize providers with lazy loading for better performance."""
        self._embedding_client: Any | None = None
        self._neo4j_vector: Any | None = None
        self._llm_client: Any | None = None
        self._citation_processor: Any | None = None
        self.schema_manager: Any = get_schema_manager()
    # --------------------------------------------------------------------------------- end __init__()
        
    # -------------------------------------------------------------- get_embedding_client()
    def get_embedding_client(self) -> Any:
        """Get the cached OpenAI embeddings client.

        Returns:
            Any: Instance compatible with LangChain `OpenAIEmbeddings`.
        """
        if self._embedding_client is None:
            try:
                from langchain_openai import OpenAIEmbeddings
                self._embedding_client = OpenAIEmbeddings(
                    model=EMBEDDING_CONFIG["model"],
                    dimensions=EMBEDDING_CONFIG["dimensions"],
                    api_key=settings.openai_api_key,  # Use 'api_key' for consistency
                    timeout=EMBEDDING_CONFIG["timeout"],
                    max_retries=EMBEDDING_CONFIG["max_retries"]
                )
                logger.info(f"Provider: Initialized OpenAI embeddings client")
            except Exception as e:
                logger.error(f"Provider: Failed to initialize embeddings client: {e}")
                raise
        return self._embedding_client
    # -------------------------------------------------------------- end get_embedding_client()
    
    # -------------------------------------------------------------- get_neo4j_vector()
    def get_neo4j_vector(self) -> Any:
        """Get the cached Neo4j vector retriever via schema manager gateway.

        Returns:
            Any: Instance compatible with `Neo4jVector` from LangChain.
        """
        if self._neo4j_vector is None:
            try:
                embeddings = self.get_embedding_client()
                self._neo4j_vector = self.schema_manager.get_neo4j_vector(embeddings)
                logger.info("Provider: Initialized Neo4j vector via schema manager gateway")
            except Exception as e:
                logger.error(f"Provider: Failed to initialize Neo4j vector: {e}")
                raise
        return self._neo4j_vector
    # -------------------------------------------------------------- end get_neo4j_vector()
    
    # -------------------------------------------------------------- get_llm_client()
    async def get_llm_client(self) -> Any:
        """Get the async LLM client provided by the backend.

        Returns:
            Any: Async LLM client exposing `complete(messages=...)`.
        """
        if self._llm_client is None:
            from ...core.async_llm_client import get_async_llm_client
            self._llm_client = await get_async_llm_client()
            logger.info("Provider: Using existing async LLM client")
        return self._llm_client
    # -------------------------------------------------------------- end get_llm_client()
    
    # -------------------------------------------------------------- get_citation_processor()
    def get_citation_processor(self) -> Any:
        """Get the citation processor service.

        Returns:
            Any: Processor with citation extraction/validation helpers.
        """
        if self._citation_processor is None:
            from ...processing.citation_processor import get_citation_processor
            self._citation_processor = get_citation_processor()
            logger.info("Provider: Using existing citation processor")
        return self._citation_processor
    # -------------------------------------------------------------- end get_citation_processor()
    
# ------------------------------------------------------------------------- end class SemanticSearchProviders

# Vector search instructions for domain-agnostic knowledge graphs
VECTOR_SEARCH_INSTRUCTIONS = """
You are an expert knowledge assistant providing information from a comprehensive knowledge graph.

Use the given context from documents to answer questions accurately and comprehensively.
CRITICAL: You must cite information using numbered references [1], [2], [3] etc. that correspond to the [Source N] sections in the context.

When referencing information from the context:
- Use inline citations like [1], [2], [3] throughout your response
- Each piece of information should be cited with the appropriate source number
- MANDATORY: Always combine numbered citations with domain-specific references:
  • Legal/Regulatory: Extract and include section numbers like "§57.4361(a) [1]" or "Part 75.1714 [2]" - NEVER use just "[1]" for regulatory content
  • Academic: "Vol. 2, Chapter 3 [1]" or "Figure 4.1 [3]"  
  • Technical: "ISO 9001:2015 [1]" or "ASTM D2582-07 [2]"
  • Business: "Policy 2.4 [1]" or "Procedure A-15 [2]"
  • Medical: "Protocol 4.2 [1]" or "Guidelines Table 1 [2]"
- CRITICAL: For regulatory content, scan each source for section numbers (§, Part) and include them: "§75.1502 [3]"
- Be comprehensive but ensure every claim is properly cited with specific regulatory sections

If you don't know the answer based on the provided context, say you don't know.
Focus on providing accurate information based on the knowledge graph content with proper citations.

Context from knowledge graph:
{context}
"""

# ______________________
# Helper Functions
#
# =========================================================================
# Citation and Domain Marker Utilities
# =========================================================================

# --------------------------------------------------------------------------------- _detect_domain_markers()
def _detect_domain_markers(text: str) -> Dict[str, List[str]]:
    """Extract domain-specific markers (legal, academic, etc.) from source text.

    Args:
        text: Source text where markers may be found (e.g., "§57.4361(a)").

    Returns:
        Mapping of domain name to list of detected markers in the text.
    """
    import re
    from ...core.config import settings
    
    patterns = {
        'legal': [
            r'§\s*\d+[\w.()]*',           # §57.4361(a)
            r'Part\s+\d+(?:\.\d+)?'       # Part 75.1714
        ],
        'academic': [
            r'Vol\.\s*\d+',               # Vol. 2
            r'Chapter\s*\d+',             # Chapter 3
            r'Figure\s*\d+(?:\.\d+)?'     # Figure 4.1
        ],
        'technical': [
            r'ISO\s+\d{4,}[:\-\d]*',      # ISO 9001:2015
            r'ASTM\s+[A-Z0-9\-:]+'        # ASTM D2582-07
        ],
        'business': [
            r'Policy\s+[\w.]+',           # Policy 2.4
            r'Procedure\s+[\w\-]+'        # Procedure A-15
        ],
        'medical': [
            r'Protocol\s+\d+(?:\.\d+)?',  # Protocol 4.2
            r'Guidelines(?:\s+Table)?\s*\d+' # Guidelines Table 1
        ]
    }
    
    markers = {}
    for domain, domain_patterns in patterns.items():
        domain_markers = []
        for pattern in domain_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            domain_markers.extend(matches)
        if domain_markers:
            # Limit markers per reference
            markers[domain] = domain_markers[:settings.semantic_domain_markers_per_ref]
    
    return markers
# --------------------------------------------------------------------------------- end _detect_domain_markers()

# --------------------------------------------------------------------------------- _validate_inline_citation()
def _validate_inline_citations(answer: str, ref_count: int, mode: str) -> Dict[str, Any]:
    """Validate inline citations against available references.

    Args:
        answer: Answer text with inline [n] citations.
        ref_count: Number of references available in the context.
        mode: "strict" drops unmatched citations; "permissive" leaves them.

    Returns:
        Dict with basic counts and which citations were matched/unmatched.
    """
    import re
    
    inline_citations = re.findall(r'\[(\d+)\]', answer)
    inline_numbers_used = set(int(c) for c in inline_citations)
    valid_numbers = set(range(1, ref_count + 1))
    
    unmatched = inline_numbers_used - valid_numbers
    matched = inline_numbers_used & valid_numbers
    
    return {
        'total_in_text': len(inline_citations),
        'unique_in_text': len(inline_numbers_used),
        'unmatched_in_text': list(unmatched),
        'matched_citations': list(matched),
        'validation_mode': mode
    }
# --------------------------------------------------------------------------------- end _validate_inline_citation()

# --------------------------------------------------------------------------------- _apply_domain_markers()
def _apply_domain_markers(answer: str, ref_markers: Dict[int, List[str]]) -> Tuple[str, Dict[str, Any]]:
    """Apply domain markers near citations with idempotent insertion.

    Args:
        answer: LLM-produced answer text.
        ref_markers: Mapping from original ref number -> markers to inject.

    Returns:
        Tuple of (enhanced_answer, metadata) including counts and placement info.
    """
    import re
    
    stats = {
        'markers_added': 0,
        'markers_per_ref': {},
        'enforcement': 'not_available'
    }
    
    if not ref_markers:
        return answer, stats
    
    #_______________
    # Embedded Function
    #
    # ----------------------------------------------------------------- replace_citation()
    def replace_citation(match):
        citation_num = int(match.group(1))
        citation_text = match.group(0)  # "[N]"
        
        if citation_num in ref_markers and ref_markers[citation_num]:
            marker = ref_markers[citation_num][0]  # Use first marker
            
            # Check if marker already exists in the same sentence (idempotent)
            # Find sentence boundaries around the citation
            start = max(0, match.start() - 200)  # Look back 200 chars
            end = min(len(answer), match.end() + 200)  # Look forward 200 chars
            sentence_context = answer[start:end]
            
            # If marker already present in context, don't add again
            if marker in sentence_context:
                return citation_text
            
            # Insert marker before citation
            stats['markers_added'] += 1
            if citation_num not in stats['markers_per_ref']:
                stats['markers_per_ref'][citation_num] = []
            stats['markers_per_ref'][citation_num].append(marker)
            
            return f"{marker} {citation_text}"
        #---- If

        return citation_text
    # ----------------------------------------------------------------- end replace_citation()
    
    enhanced_answer = re.sub(r'\[(\d+)\]', replace_citation, answer)
    
    if stats['markers_added'] > 0:
        stats['enforcement'] = 'inline_added'
    elif any(ref_markers.values()):
        stats['enforcement'] = 'already_present'
    
    return enhanced_answer, stats
# --------------------------------------------------------------------------------- end _apply_domain_markers()

# =========================================================================
# Cross-Module Processing
# =========================================================================

# --------------------------------------------------------------------------------- process_result_with_enhanced_citations()
def process_result_with_enhanced_citations(raw_answer: str, reference_list: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """Enhanced citation processing with validation and domain enforcement.

    Args:
        raw_answer: Answer text produced by the LLM.
        reference_list: Reference metadata corresponding to numbered sources.

    Returns:
        Tuple of (final_answer, citation_metadata) including validation results and
        domain marker statistics.
    """
    from ...core.config import settings
    
    # Step 1: Validate inline citations
    validation_mode = "strict" if settings.semantic_citation_validation else "permissive"
    validation_result = _validate_inline_citations(raw_answer, len(reference_list), validation_mode)
    
    # Step 2: Remove unmatched citations if in strict mode
    processed_answer = raw_answer
    if settings.semantic_citation_validation and validation_result['unmatched_in_text']:
        import re
        # Replace unmatched citations with [?]
        for unmatched_num in validation_result['unmatched_in_text']:
            processed_answer = re.sub(f'\\[{unmatched_num}\\]', '[?]', processed_answer)
    
    # Step 3: Get citations that will actually be used (before renumbering)
    citations_used = extract_citations_from_answer(processed_answer)
    citations_used = [c for c in citations_used if 1 <= c <= len(reference_list)]  # Filter valid only
    
    # Step 4: Extract domain markers ONLY for references that will be used
    ref_markers = {}  # Maps old reference numbers to markers
    domain_detected = set()
    
    if settings.semantic_domain_enforcement and citations_used:
        for ref_num in citations_used:
            if 1 <= ref_num <= len(reference_list):
                ref = reference_list[ref_num - 1]  # Convert to 0-based index
                # Use content_preview or chunk_id for marker detection
                source_content = ref.get('content_preview', '') or ref.get('chunk_id', '')
                markers = _detect_domain_markers(source_content)
                
                if markers:
                    domain_detected.update(markers.keys())
                    # Flatten markers for this reference
                    ref_markers[ref_num] = []
                    for domain_markers in markers.values():
                        ref_markers[ref_num].extend(domain_markers[:settings.semantic_domain_markers_per_ref])
    
    # Step 5: Apply domain markers using old reference numbers
    enhanced_answer, marker_stats = _apply_domain_markers(processed_answer, ref_markers)
    
    # Step 6: Renumber citations chronologically (existing logic)
    final_answer, used_references = renumber_citations_chronologically(
        enhanced_answer, reference_list, citations_used
    )
    
    # Step 7: CRITICAL - Remap ref_markers from old numbers to new numbers
    if used_references and ref_markers:
        # Build old -> new number mapping
        old_to_new_map = {}
        for new_ref in used_references:
            new_num = new_ref.get('number')
            chunk_id = new_ref.get('chunk_id')
            # Find corresponding old reference by chunk_id
            for old_num, ref in enumerate(reference_list, 1):
                if ref.get('chunk_id') == chunk_id:
                    old_to_new_map[old_num] = new_num
                    break
        
        # Remap ref_markers keys
        remapped_ref_markers = {}
        for old_num, markers in ref_markers.items():
            if old_num in old_to_new_map:
                new_num = old_to_new_map[old_num]
                remapped_ref_markers[new_num] = markers
        
        ref_markers = remapped_ref_markers
    
    # Step 8: Build enhanced references section with remapped markers
    if used_references:
        references_section = format_enhanced_references_section(used_references, ref_markers)
        final_answer += references_section
    
    # Build comprehensive metadata
    citation_metadata = {
        'citations': validation_result,
        'domain_citations': {
            'domain_detected': list(domain_detected) if domain_detected else [],
            **marker_stats
        }
    }
    
    return final_answer, citation_metadata
# --------------------------------------------------------------------------------- end process_result_with_enhanced_citations()

# --------------------------------------------------------------------------------- format_enhanced_references_section()
def format_enhanced_references_section(document_references: List[Dict[str, Any]], 
                                     ref_markers: Dict[int, List[str]]) -> str:
    """Format references with domain markers using new reference numbers."""
    if not document_references:
        return ""
    
    references = ["\n\n**References:**"]
    
    for ref in document_references:
        number = ref.get('number', '?')
        chunk_id = ref.get('chunk_id', 'Unknown')
        page = ref.get('page', '')
        
        # Add domain markers if available (using new reference number)
        markers_text = ""
        if number in ref_markers and ref_markers[number]:
            markers_text = f", {', '.join(ref_markers[number])}"
        
        if page:
            references.append(f"[{number}] {chunk_id}{markers_text}, p.{page}")
        else:
            references.append(f"[{number}] {chunk_id}{markers_text}")
    
    references.append("")
    return "\n".join(references)
# --------------------------------------------------------------------------------- end format_enhanced_references_section()

# ------------------------------------------------------------------------- class SemanticSearchEngine
class SemanticSearchEngine:
    """
    Semantic VectorRAG search engine with schema awareness and citation validation.
    
    Uses provider caches and supports schema-aware prompts, citation validation,
    and domain marker enforcement based on configuration flags.
    """
    
    # ______________________
    # Constructor
    #
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self) -> None:
        """Initialize semantic engine with cached providers."""
        # Initialize provider adapters (lazy loading)
        self._providers: SemanticSearchProviders | None = None

        # Components for enhanced features
        self._schema_manager: Any | None = None
        self._schema_summary_cache: Any | None = None
        self._schema_last_loaded_ts: float = 0.0
        self._schema_refresh_interval: int = 1800  # 30 minutes TTL
        self._timing_collector: Any | None = None

        # Feature availability flags
        self._schema_feature_available: bool = True
        self._citation_feature_available: bool = True
    # --------------------------------------------------------------------------------- end __init__()
    
    # -------------------------------------------------------------- _get_providers()
    def _get_providers(self) -> SemanticSearchProviders:
        """Get cached provider adapters."""
        if self._providers is None:
            self._providers = SemanticSearchProviders()
        return self._providers
    # -------------------------------------------------------------- end _get_providers()
    
    # -------------------------------------------------------------- _get_timing_collector()
    def _get_timing_collector(self) -> Any:
        """Get timing collector with lazy loading."""
        if self._timing_collector is None:
            try:
                from ...monitoring.timing_collector import get_timing_collector
                self._timing_collector = get_timing_collector()
            except ImportError:
                logger.warning("Timing collector not available")
                return None
        return self._timing_collector
    # -------------------------------------------------------------- end _get_timing_collector()
    
    # -------------------------------------------------------------- _get_schema_manager()
    def _get_schema_manager(self) -> Any:
        """Lazy load schema manager."""
        if self._schema_manager is None:
            try:
                from ...schema import get_schema_manager
                self._schema_manager = get_schema_manager()
            except ImportError:
                logger.debug("Schema manager not available")
                return None
        return self._schema_manager
    # -------------------------------------------------------------- end _get_schema_manager()
    
    # -------------------------------------------------------------- _load_structural_summary()
    def _load_structural_summary(self, force: bool = False) -> Any:
        """Load structural summary with caching and TTL."""
        current_time = time.time()
        if (not force and self._schema_summary_cache and 
            (current_time - self._schema_last_loaded_ts) < self._schema_refresh_interval):
            return self._schema_summary_cache
        
        try:
            schema_manager = self._get_schema_manager()
            if not schema_manager:
                return None
                
            summary = schema_manager.get_structural_summary()
            if summary:
                self._schema_summary_cache = summary
                self._schema_last_loaded_ts = current_time
                logger.debug("Schema structural summary loaded and cached")
            return summary
        except Exception as e:
            logger.debug(f"Schema summary load failed: {e}")
        return None
    # -------------------------------------------------------------- end _load_structural_summary()
    
    # -------------------------------------------------------------- _build_schema_prompt_segment()
    def _build_schema_prompt_segment(
        self,
        query: str,
        context_docs: List,
        token_budget: int = 300,
    ) -> tuple:
        """Build compact schema context for prompt augmentation.

        Returns:
            tuple: A pair of (schema_segment_text, metadata_dict).
        """
        summary = self._load_structural_summary()
        if not summary:
            return "", {"used": False, "labels_included": 0, "relationships_included": 0, "truncated": False}
        
        # Extract query keywords (case-insensitive)
        query_keywords = set(word.lower().strip('.,!?') for word in query.split() if len(word) > 2)
        
        # Collect context entities from document metadata
        context_entities = set()
        for doc in context_docs:
            metadata = getattr(doc, 'metadata', {})
            entities = metadata.get('entities', [])
            for entity in entities:
                if isinstance(entity, dict) and entity.get('name'):
                    context_entities.add(entity['name'].lower())
                elif isinstance(entity, str):
                    context_entities.add(entity.lower())
        
        # Get schema elements using correct API
        try:
            if hasattr(summary, 'get_node_labels_list'):
                all_node_labels = summary.get_node_labels_list()
                all_rel_types = summary.get_relationship_types_list()
            else:
                # Fallback to dict access
                summary_dict = summary.to_dict() if hasattr(summary, 'to_dict') else summary
                all_node_labels = summary_dict.get("node_labels", [])
                all_rel_types = summary_dict.get("relationship_types", [])
        except Exception as e:
            logger.debug(f"Error accessing schema elements: {e}")
            return "", {"used": False, "labels_included": 0, "relationships_included": 0, "truncated": False}
        
        #______________________
        # Embedded Function
        #
        # ----------------------------------------------------------------- relevance_score()
        # Prioritize schema elements by relevance to query and context
        def relevance_score(item_name: str) -> int:
            item_lower = item_name.lower()
            score = 0
            # Boost if matches query keywords
            for keyword in query_keywords:
                if keyword in item_lower:
                    score += 2
            # Boost if matches context entities
            for entity in context_entities:
                if entity in item_lower or item_lower in entity:
                    score += 1
            return score
        # ----------------------------------------------------------------- relevance_score()

        # Sort and select top elements
        from ...core.config import settings
        max_items = min(settings.semantic_schema_max_items, len(all_node_labels))
        sorted_labels = sorted(all_node_labels, key=relevance_score, reverse=True)[:max_items]
        sorted_rels = sorted(all_rel_types, key=relevance_score, reverse=True)[:max_items]
        
        # Build compact segment
        schema_lines = [
            "Schema Context:",
            f"Node Types: {', '.join(sorted_labels)}",
            f"Relationships: {', '.join(sorted_rels)}",
            ""
        ]
        
        segment_text = "\n".join(schema_lines)
        
        # Check token budget (rough estimation: ~4 chars per token)
        truncated = False
        if len(segment_text) > token_budget * 4:
            # Truncate by reducing items, not text cutting
            reduced_labels = sorted_labels[:max_items // 2]
            reduced_rels = sorted_rels[:max_items // 2]
            
            schema_lines = [
                "Schema Context:",
                f"Node Types: {', '.join(reduced_labels)}",
                f"Relationships: {', '.join(reduced_rels)}",
                ""
            ]
            segment_text = "\n".join(schema_lines)
            truncated = True
            sorted_labels, sorted_rels = reduced_labels, reduced_rels
        
        metadata = {
            "used": True,
            "labels_included": len(sorted_labels),
            "relationships_included": len(sorted_rels),
            "token_budget": token_budget,
            "truncated": truncated
        }
        
        return segment_text, metadata
    # -------------------------------------------------------------- end _build_schema_prompt_segment()
    
    # -------------------------------------------------------------- _build_search_response()
    async def _build_search_response(
        self,
        final_answer: str,
        query: str,
        context_docs: List,
        reference_list: List[Dict[str, Any]],
        start_time: float,
        engine_used: str,
    ) -> Dict[str, Any]:
        """Build standardized search response maintaining API compatibility."""
        try:
            # Process sources and entities for backward compatibility
            sources = []
            entities_found = set()
            
            for i, doc in enumerate(context_docs):
                metadata = getattr(doc, 'metadata', {})
                content = getattr(doc, 'page_content', str(doc))
                
                # Truncate content to 6000 characters for better context
                truncated_content = content[:6000] + "..." if len(content) > 6000 else content
                
                # Use reference list data if available, otherwise fallback to metadata
                ref_info = reference_list[i] if i < len(reference_list) else {}
                
                # Get document title from reference list or extract using pure function
                chunk_id = ref_info.get("chunk_id") or metadata.get("chunk_id", "Unknown")
                document_title = ref_info.get("document_title") or extract_document_title_from_chunk_id(
                    chunk_id, metadata.get("document_title") or metadata.get("title")
                )
                
                source_info = {
                    "chunk_id": chunk_id,
                    "document_id": metadata.get("document_id", "Unknown"),
                    "document_title": document_title,
                    "page": ref_info.get("page") or metadata.get("page"),
                    "tokens": metadata.get("tokens"),
                    "content_preview": truncated_content,
                    "entities": ref_info.get("entities") or metadata.get("entities", [])
                }
                sources.append(source_info)
                
                # Collect entities
                entities = ref_info.get("entities") or metadata.get("entities", [])
                for entity in entities:
                    if isinstance(entity, dict) and entity.get("name"):
                        entities_found.add(entity["name"])
            
            search_time = time.time() - start_time
            
            return {
                "answer": final_answer,
                "query": query,
                "sources": sources,
                "entities_found": list(entities_found)[:15],  # Limit to top 15
                "num_sources": len(sources),
                "search_time_ms": int(search_time * 1000),
                "search_type": "semantic_vector_search",
                "engine_used": engine_used,  # Track which engine was used
                "parameters": {
                    "k": len(context_docs),
                    "score_threshold": "dynamic"  # Since we might adjust this
                },
                "metadata": {}
            }

        except Exception as e:
            logger.error(f"Error building search response: {e}")
            search_time = time.time() - start_time
            return {
                "answer": f"Error building response: {str(e)}",
                "query": query,
                "sources": [],
                "entities_found": [],
                "num_sources": 0,
                "search_time_ms": int(search_time * 1000),
                "search_type": "semantic_vector_search",
                "engine_used": engine_used,
                "error": str(e),
                "metadata": {}
            }
    # -------------------------------------------------------------- end _build_search_response()

    # -------------------------------------------------------------- search()
    async def search(
        self,
        query: str,
        k: int = 15,
        score_threshold: float = 0.65,
    ) -> Dict[str, Any]:
        """Execute semantic search with schema awareness and citation validation."""
        try:
            start_time = time.time()
            logger.info(f"Processing semantic search query: {query[:50]}...")
            
            # Get timing collector
            timing_collector = self._get_timing_collector()
            
            providers = self._get_providers()
            
            # Get retriever using provider adapters
            neo4j_vector = providers.get_neo4j_vector()
            retriever = neo4j_vector.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold}
            )
            
            # Get documents
            loop = asyncio.get_event_loop()
            context_docs = await loop.run_in_executor(
                None, retriever.get_relevant_documents, query
            )
            
            # Use async LLM client
            llm_client = await providers.get_llm_client()
            
            # Build reference list
            reference_list = []
            context_lines = []
            
            for i, doc in enumerate(context_docs, 1):
                metadata = getattr(doc, 'metadata', {})
                content = getattr(doc, 'page_content', str(doc))
                
                context_lines.append(f"[Source {i}]:")
                context_lines.append(content[:6000])
                
                chunk_id = metadata.get("chunk_id", "Unknown")
                page = metadata.get("page")
                document_title = extract_document_title_from_chunk_id(
                    chunk_id, metadata.get("document_title") or metadata.get("title")
                )
                
                reference_list.append({
                    'number': i,
                    'chunk_id': chunk_id,
                    'page': str(page) if page else None,
                    'document_title': document_title,
                    'entities': metadata.get("entities", [])
                })
            
            numbered_context = "\n\n".join(context_lines)
            
            # Schema-aware prompt augmentation
            from ...core.config import settings
            schema_segment = ""
            schema_metadata = {"used": False, "labels_included": 0, "relationships_included": 0, "truncated": False}
            
            if settings.semantic_use_structural_schema:
                try:
                    if timing_collector:
                        async with timing_collector.measure("semantic_schema_augmentation") as schema_timer:
                            schema_segment, schema_metadata = self._build_schema_prompt_segment(
                                query, context_docs, settings.semantic_schema_token_budget
                            )
                            schema_timer.add_metadata(schema_metadata)
                    else:
                        schema_segment, schema_metadata = self._build_schema_prompt_segment(
                            query, context_docs, settings.semantic_schema_token_budget
                        )
                except Exception as e:
                    logger.warning(f"Schema augmentation failed, continuing without: {e}")
                    self._schema_feature_available = False
            
            # Build messages with optional schema augmentation
            base_instructions = VECTOR_SEARCH_INSTRUCTIONS.format(context=numbered_context)
            
            if schema_segment:
                system_content = f"{schema_segment}\n\n{base_instructions}"
            else:
                system_content = base_instructions
            
            from langchain_core.messages import SystemMessage, HumanMessage
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=query)
            ]
            
            # Call async LLM client
            raw_answer = await llm_client.complete(messages=messages)
            if not raw_answer:
                raw_answer = "No response generated."
            
            # Citation processing
            citation_metadata = {}
            try:
                if timing_collector:
                    async with timing_collector.measure("semantic_citation_validation") as citation_timer:
                        if settings.semantic_citation_validation or settings.semantic_domain_enforcement:
                            final_answer, citation_metadata = process_result_with_enhanced_citations(
                                raw_answer, reference_list
                            )
                            citation_timer.add_metadata(citation_metadata)
                        else:
                            # Use existing processing
                            final_answer = process_result_with_citations(raw_answer, reference_list)
                else:
                    if settings.semantic_citation_validation or settings.semantic_domain_enforcement:
                        final_answer, citation_metadata = process_result_with_enhanced_citations(
                            raw_answer, reference_list
                        )
                    else:
                        # Use existing processing
                        final_answer = process_result_with_citations(raw_answer, reference_list)
            except Exception as e:
                logger.warning(f"Citation processing failed, using basic: {e}")
                self._citation_feature_available = False
                final_answer = process_result_with_citations(raw_answer, reference_list)
            
            # Update simple stats
            _engine_stats["total_requests"] += 1
            _engine_stats["last_request_ts"] = datetime.utcnow().isoformat()
            
            # Build response
            response = await self._build_search_response(
                final_answer, query, context_docs, reference_list, start_time, "semantic_engine"
            )
            
            # Add metadata
            response_metadata = response.get('metadata', {})
            response_metadata.update({
                'engine_mode': 'semantic_cached',
                'schema_aware_semantic': schema_metadata
            })
            
            # Merge citation metadata
            if citation_metadata:
                response_metadata.update(citation_metadata)
            else:
                response_metadata.update({
                    'citations': {'validation_mode': 'basic'},
                    'domain_citations': {'enforcement': 'not_available'}
                })
            
            response['metadata'] = response_metadata
            
            return response
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            search_time = time.time() - start_time if 'start_time' in locals() else 0
            return {
                "answer": f"Error during semantic search: {str(e)}",
                "query": query,
                "sources": [],
                "entities_found": [],
                "num_sources": 0,
                "search_time_ms": int(search_time * 1000),
                "search_type": "semantic_vector_search",
                "engine_used": "semantic_engine",
                "error": str(e),
                "metadata": {
                    "engine_mode": "semantic_cached",
                    "error_context": str(e)
                }
            }
    # -------------------------------------------------------------- end search()

# ------------------------------------------------------------------------- end class SemanticSearchEngine

# =========================================================================
# Public API Facades
# =========================================================================

# --------------------------------------------------------------------------------- search_semantic()
async def search_semantic(
    query: str,
    k: int = 15,
    score_threshold: float = 0.65,
) -> str:
    """Perform a semantic search and return only the answer text.

    Args:
        query: Natural language query.
        k: Number of chunks to retrieve from the vector index.
        score_threshold: Minimum similarity score for candidate chunks.

    Returns:
        str: Answer text produced by the semantic engine.
    """
    try:
        engine = get_vector_engine()
        result = await engine.search(query, k=k, score_threshold=score_threshold)
        return result["answer"]
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return f"Error during semantic search: {str(e)}"
# --------------------------------------------------------------------------------- end search_semantic()

# --------------------------------------------------------------------------------- search_semantic_detailed()
async def search_semantic_detailed(
    query: str,
    k: int = 15,
    score_threshold: float = 0.65,
) -> Dict[str, Any]:
    """Perform semantic search and return a structured result with metadata.

    Args:
        query: Natural language query.
        k: Number of chunks to retrieve from the vector index.
        score_threshold: Minimum similarity score for candidate chunks.

    Returns:
        dict: Normalized result including `answer`, `sources`, `entities_found`, timing,
            and compact engine metadata.
    """
    try:
        engine = get_vector_engine()
        result = await engine.search(query, k=k, score_threshold=score_threshold)
        # Attach simplified metadata
        result["metadata"] = {**result.get("metadata", {}), **_build_simple_metadata(k, score_threshold)}
        return result
    except Exception as e:
        logger.error(f"Detailed semantic search failed: {e}")
        return {
            "answer": f"Error during semantic search: {str(e)}",
            "query": query,
            "sources": [],
            "entities_found": [],
            "num_sources": 0,
            "search_time_ms": 0,
            "search_type": "semantic_vector_search",
            "error": str(e),
            "metadata": {**_build_simple_metadata(k, score_threshold), "error_context": str(e)},
        }
# --------------------------------------------------------------------------------- end search_semantic_detailed()

# --------------------------------------------------------------------------------- get_vector_tool()
def get_vector_tool() -> Any:
    """Create a LangChain Tool wrapper for semantic vector search.

    Returns:
        Any: A LangChain `Tool` instance named `search_semantic`, or None on error.
    """
    try:
        def search_tool(query: str) -> str:
            """Tool function for semantic search."""
            import asyncio
            
            # Handle event loop for sync tool interface
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If already in async context, create task
                    task = loop.create_task(search_semantic(query))
                    return task.result() if hasattr(task, 'result') else "Search initiated"
                else:
                    return loop.run_until_complete(search_semantic(query))
            except RuntimeError:
                # Create new event loop if needed
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(search_semantic(query))
                finally:
                    loop.close()
        
        return Tool(
            name="search_semantic",
            description="""
            Searches the knowledge graph using semantic vector similarity search.
            
            This tool performs true similarity search using OpenAI embeddings to find
            the most relevant content based on semantic meaning across all documents
            and entities in the knowledge graph.
            
            Use this tool when users ask about:
            - Any topic that requires finding semantically related information
            - Concepts that may be expressed differently across documents
            - Questions requiring comprehensive context from multiple sources
            - Information that benefits from semantic understanding over keyword matching
            
            Input: A natural language question or topic
            Output: Comprehensive information with sources and entity context
            """,
            func=search_tool
        )
    except Exception as e:
        logger.error(f"Error creating vector tool: {e}")
        return None
# --------------------------------------------------------------------------------- end get_vector_tool()

# --------------------------------------------------------------------------------- test_vector_search()
async def test_vector_search() -> None:
    """Run a small set of smoke tests against the vector search engine.

    Logs the status, number of sources, entity count, response time, and an answer preview
    for a few predefined queries. This is intended for developer diagnostics only.
    """
    logger.info("Testing APH-IF VectorRAG System...")
    
    test_queries = [
        "What safety measures are required?",
        "Tell me about equipment requirements",
        "What are the compliance procedures?"
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting: {query}")
        try:
            result = await search_semantic_detailed(query)
            logger.info(f"Status: Success")
            logger.info(f"Sources found: {result['num_sources']}")
            logger.info(f"Entities: {len(result['entities_found'])}")
            logger.info(f"Response time: {result['search_time_ms']}ms")
            logger.info(f"Answer preview: {result['answer'][:100]}...")
        except Exception as e:
            logger.error(f"Error: {e}")
    
    logger.info("\nVectorRAG testing complete!")
# --------------------------------------------------------------------------------- end test_vector_search()

# __________________________________________________________________________
# Module Initialization / Main Execution Guard (if applicable)
# Test entry point
if __name__ == "__main__":
    async def main():
        await test_vector_search()
    
    asyncio.run(main())

# __________________________________________________________________________
# End of File
