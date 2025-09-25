# -------------------------------------------------------------------------
# File: citation_processor.py
# Author: Alexander Ricciardi
# Date: 2025-09-18
# [File Path] backend/app/processing/citation_processor.py
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
#   Asynchronous citation extraction and processing with TTL caching and
#   background task support. Provides APIs used by the backend search/fusion
#   pipeline to detect inline citations and parse References sections.
# -------------------------------------------------------------------------
# --- Module Contents Overview ---
# - Dataclass: CitationCache
# - Class: AsyncCitationProcessor
# - Function: get_citation_processor
# - Function: shutdown_citation_processor
# -------------------------------------------------------------------------
# --- Dependencies / Imports ---
# - Standard Library: asyncio, re, time, logging, dataclasses, typing, hashlib
# - Local Project Modules: (used by other modules; this file is standalone)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
# --- Usage / Integration ---
# Used by the backend search/fusion pipeline:
#   - Imported by search/context_fusion via get_citation_processor()
#   - Long-lived module-level singleton to avoid reinitialization overhead
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Async Citation Processor Service for APH-IF Backend

Background citation extraction and processing with caching and parallel execution for
optimal performance. Provides async APIs to extract citations and references, with TTL cache
and background tasks.

Usage:
    - Used by search/context_fusion via get_citation_processor()
    - Library module; not a program entry point
"""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# __________________________________________________________________________
# Global Constants / Variables

logger = logging.getLogger(__name__)


# __________________________________________________________________________
# Class Definitions

# ------------------------------------------------------------------------- class CitationCache
@dataclass
class CitationCache:
    """Cache entry for citation results."""
    content_hash: str
    citations: Dict[str, List[str]]
    references: List[Dict[str, str]]
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = 1800  # 30 minutes
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired.

        Returns:
            True if this cache entry has passed its TTL; otherwise False.
        """
        return time.time() > (self.created_at + self.ttl_seconds)
# ------------------------------------------------------------------------- end class CitationCache


# TODO: Not a candidate for @dataclass — manages background tasks, cache state, and async I/O.
# ------------------------------------------------------------------------- class AsyncCitationProcessor
class AsyncCitationProcessor:
    """
    Asynchronous citation extraction and processing service.
    
    Features:
    - Background citation extraction
    - Result caching with TTL
    - Parallel processing of multiple contents
    - Fire-and-forget task execution
    
    Attributes:
        cache_ttl: Cache time-to-live in seconds for results stored in-memory.
        max_cache_size: Maximum number of entries permitted in the cache.
        _cache: Mapping of content hash to cached citation/reference results.
        _cache_hits: Number of cache hits recorded.
        _cache_misses: Number of cache misses recorded.
        _background_tasks: Set of in-flight background asyncio Tasks.
        _patterns: Precompiled regular expressions for citation/reference detection.
    """
    
    # -------------------------------------------------------------- __init__()
    def __init__(self, cache_ttl: int = 1800, max_cache_size: int = 500):
        """
        Initialize the async citation processor.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
            max_cache_size: Maximum number of cache entries
        """
        self.cache_ttl = cache_ttl
        self.max_cache_size = max_cache_size
        
        # Citation cache
        self._cache: Dict[str, CitationCache] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        
        # Regex patterns (compiled once for efficiency)
        self._patterns = {
            "numbered": re.compile(r"\[(\d+)\]"),
            "regulatory": re.compile(r"§[\d.]+(?:\([a-z]\))?(?:\(\d+\))?", re.IGNORECASE),
            "part": re.compile(r"Part\s+[\d.]+", re.IGNORECASE),
            "cfr": re.compile(r"CFR\s+\d+\.[\d.]+", re.IGNORECASE),
            "iso": re.compile(r"ISO\s+\d+(?::\d+)?", re.IGNORECASE),
            "astm": re.compile(r"ASTM\s+[A-Z]\d+(?:-\d+)?", re.IGNORECASE),
            "ansi": re.compile(r"ANSI\s+[A-Z]\d+(?:\.\d+)?", re.IGNORECASE),
            "osha": re.compile(r"OSHA\s+[\d.]+", re.IGNORECASE),
            "nfpa": re.compile(r"NFPA\s+\d+", re.IGNORECASE),
            "volume": re.compile(r"Vol\.\s+\d+,?\s*Chapter\s+\d+", re.IGNORECASE),
            "policy": re.compile(r"Policy\s+[\d.]+", re.IGNORECASE),
            "procedure": re.compile(r"Procedure\s+[A-Z]-\d+", re.IGNORECASE),
            "protocol": re.compile(r"Protocol\s+[\d.]+", re.IGNORECASE),
            "guidelines": re.compile(r"Guidelines\s+Table\s+\d+", re.IGNORECASE),
            "figure": re.compile(r"Figure\s+[\d.]+", re.IGNORECASE),
            "section": re.compile(r"Section\s+[\d.]+", re.IGNORECASE),
        }
        
        logger.info(f"Async citation processor initialized with {cache_ttl}s TTL")
    # -------------------------------------------------------------- end __init__()
    
    # -------------------------------------------------------------- get_content_hash()
    def _get_content_hash(self, content: str) -> str:
        """Generate hash for content caching."""
        return hashlib.md5(content.encode()).hexdigest()
    # -------------------------------------------------------------- get_content_hash()
    
    # -------------------------------------------------------------- extract_citations_async()
    async def extract_citations_async(
        self,
        content: str,
        use_cache: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Extract citations from content asynchronously.
        
        Args:
            content: Text content to extract citations from
            use_cache: Whether to use cache
            
        Returns:
            Dictionary of citation types and their values
        """
        # Check cache first
        if use_cache:
            content_hash = self._get_content_hash(content)
            if content_hash in self._cache:
                cache_entry = self._cache[content_hash]
                if not cache_entry.is_expired():
                    self._cache_hits += 1
                    logger.debug(
                        f"Citation cache hit (rate: {self._get_hit_rate():.2%})",
                    )
                    return cache_entry.citations
                else:
                    # Remove expired entry
                    del self._cache[content_hash]
        
        self._cache_misses += 1
        
        # Extract citations in executor to avoid blocking
        loop = asyncio.get_event_loop()
        citations = await loop.run_in_executor(
            None,
            self._extract_citations_sync,
            content,
        )
        
        # Cache result
        if use_cache:
            self._cache_result(content, citations, [])
        
        return citations
    # -------------------------------------------------------------- end extract_citations_async()
    
    # -------------------------------------------------------------- _extract_citations_sync()
    def _extract_citations_sync(self, content: str) -> Dict[str, List[str]]:
        """
        Synchronous citation extraction (runs in executor).
        
        Args:
            content: Text content to extract citations from
            
        Returns:
            Dictionary of citation types and their values
        """
        citations = {
            "numbered": [],
            "regulatory": [],
            "standards": [],
            "references": [],
            "combined": [],
        }
        
        try:
            # Extract numbered citations
            numbered = self._patterns["numbered"].findall(content)
            citations["numbered"] = [f"[{num}]" for num in numbered]
            
            # Extract regulatory citations
            for pattern_name in ["regulatory", "part", "cfr"]:
                matches = self._patterns[pattern_name].findall(content)
                citations["regulatory"].extend(matches)
            
            # Extract standards
            for pattern_name in ["iso", "astm", "ansi", "osha", "nfpa"]:
                matches = self._patterns[pattern_name].findall(content)
                citations["standards"].extend(matches)
            
            # Extract references
            for pattern_name in [
                "volume",
                "policy",
                "procedure",
                "protocol",
                "guidelines",
                "figure",
                "section",
            ]:
                matches = self._patterns[pattern_name].findall(content)
                citations["references"].extend(matches)
            
            # Combine all unique citations
            all_citations = (
                citations["numbered"] +
                citations["regulatory"] +
                citations["standards"] +
                citations["references"]
            )
            citations["combined"] = list(set(all_citations))
            
        except Exception as e:
            logger.warning(f"Citation extraction error: {e}")
        
        return citations
    # -------------------------------------------------------------- end _extract_citations_sync()
    
    # -------------------------------------------------------------- extract_references_async()
    async def extract_references_async(
        self,
        content: str,
        use_cache: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Extract source references from content asynchronously.
        
        Args:
            content: Text content with References section
            use_cache: Whether to use cache
            
        Returns:
            List of source reference dictionaries
        """
        # Check cache
        if use_cache:
            content_hash = self._get_content_hash(content)
            if content_hash in self._cache:
                cache_entry = self._cache[content_hash]
                if not cache_entry.is_expired():
                    return cache_entry.references
        
        # Extract in executor
        loop = asyncio.get_event_loop()
        references = await loop.run_in_executor(
            None,
            self._extract_references_sync,
            content,
        )
        
        return references
    # -------------------------------------------------------------- end extract_references_async()
    
    # -------------------------------------------------------------- _extract_references_sync()
    def _extract_references_sync(self, content: str) -> List[Dict[str, str]]:
        """
        Synchronous reference extraction (runs in executor).
        
        Args:
            content: Text content with References section
            
        Returns:
            List of source reference dictionaries
        """
        references = []
        
        try:
            # Find References section
            references_section = ""
            if "**References:**" in content:
                references_section = content.split("**References:**")[-1]
            elif "References:" in content:
                references_section = content.split("References:")[-1]
            
            if references_section:
                lines = references_section.split("\n")
                
                for line in lines:
                    line = line.strip()
                    # Match patterns like "[1] filename.pdf_p123_c1, p.123"
                    match = re.match(r"^\[(\d+)\]\s+(.+)", line)
                    if match:
                        number = int(match.group(1))
                        source_info = match.group(2)
                        references.append({
                            "number": number,
                            "source_info": source_info,
                        })
        
        except Exception as e:
            logger.debug(f"Reference extraction error: {e}")
        
        return references
    # -------------------------------------------------------------- end _extract_references_sync()
    
    # -------------------------------------------------------------- process_parallel()
    async def process_parallel(
        self,
        contents: List[str],
        extract_references: bool = True,
    ) -> List[Tuple[Dict[str, List[str]], List[Dict[str, str]]]]:
        """
        Process multiple contents in parallel.
        
        Args:
            contents: List of content strings to process
            extract_references: Whether to also extract references
            
        Returns:
            List of (citations, references) tuples
        """
        tasks = []
        
        for content in contents:
            if extract_references:
                # Extract both citations and references
                task = asyncio.create_task(
                    self._extract_both(content),
                )
            else:
                # Extract only citations
                task = asyncio.create_task(
                    self.extract_citations_async(content),
                )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Format results
        if extract_references:
            return results
        else:
            return [(r, []) for r in results]
    # -------------------------------------------------------------- end process_parallel()
    
    # -------------------------------------------------------------- _extract_both()
    async def _extract_both(
        self,
        content: str,
    ) -> Tuple[Dict[str, List[str]], List[Dict[str, str]]]:
        """Extract both citations and references.

        Args:
            content: Text content containing inline citations and optional References section.

        Returns:
            Tuple of (citations mapping, references list).
        """
        citations_task = asyncio.create_task(
            self.extract_citations_async(content),
        )
        references_task = asyncio.create_task(
            self.extract_references_async(content),
        )
        
        citations, references = await asyncio.gather(
            citations_task,
            references_task,
        )
        
        return citations, references
    # -------------------------------------------------------------- end _extract_both()
    
    # -------------------------------------------------------------- create_background_task()
    def create_background_task(
        self,
        content: str,
        callback: Optional[asyncio.Future] = None,
    ) -> asyncio.Task:
        """
        Create a fire-and-forget background task for citation extraction.
        
        Args:
            content: Content to process
            callback: Optional future to set with result
            
        Returns:
            Background task
        """
        task = asyncio.create_task(
            self._background_extract(content, callback),
        )
        
        # Track task
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
        return task
    
    # -------------------------------------------------------------- _background_extract()
    async def _background_extract(
        self,
        content: str,
        callback: Optional[asyncio.Future],
    ) -> None:
        """Background extraction task.

        Args:
            content: Content to extract citations and references from.
            callback: Optional future to receive the (citations, references) tuple.
        """
        try:
            citations = await self.extract_citations_async(content)
            references = await self.extract_references_async(content)
            
            if callback:
                callback.set_result((citations, references))
                
        except Exception as e:
            logger.error(f"Background extraction error: {e}")
            if callback:
                callback.set_exception(e)
    # -------------------------------------------------------------- end _background_extract()
    
    # -------------------------------------------------------------- _cache_result()
    def _cache_result(
        self,
        content: str,
        citations: Dict[str, List[str]],
        references: List[Dict[str, str]],
    ) -> None:
        """Cache extraction results."""
        # Clean cache if too large
        if len(self._cache) >= self.max_cache_size:
            self._clean_cache()
        
        content_hash = self._get_content_hash(content)
        self._cache[content_hash] = CitationCache(
            content_hash=content_hash,
            citations=citations,
            references=references,
            ttl_seconds=self.cache_ttl,
        )
    # -------------------------------------------------------------- end _cache_result()
    
    # -------------------------------------------------------------- _clean_cache()
    def _clean_cache(self) -> None:
        """Remove expired and oldest entries from cache."""
        # Remove expired entries
        expired = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        for key in expired:
            del self._cache[key]
        
        # If still too large, remove oldest entries
        if len(self._cache) >= self.max_cache_size:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1].created_at,
            )
            # Remove oldest 20%
            to_remove = len(self._cache) // 5
            for key, _ in sorted_entries[:to_remove]:
                del self._cache[key]
    # -------------------------------------------------------------- end _clean_cache()
    
    # -------------------------------------------------------------- _get_hit_rate()
    def _get_hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            The cache hit ratio as a float in [0.0, 1.0].
        """
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total
    # -------------------------------------------------------------- end _get_hit_rate()
    
    # -------------------------------------------------------------- get_metrics()
    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics.

        Returns:
            Mapping of cache and background task statistics.
        """
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._get_hit_rate(),
            "background_tasks": len(self._background_tasks),
        }
    # -------------------------------------------------------------- end get_metrics()
    
    # -------------------------------------------------------------- cleanup()
    async def cleanup(self) -> None:
        """Cleanup background tasks."""
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Clear cache
        self._cache.clear()
        
        logger.info("Citation processor cleaned up")
    # -------------------------------------------------------------- end cleanup()
    
# ------------------------------------------------------------------------- end class AsyncCitationProcessor

# __________________________________________________________________________
# Standalone Function Definitions

# Global processor instance
_citation_processor: Optional[AsyncCitationProcessor] = None

# --------------------------------------------------------------------------------- get_citation_processor()
def get_citation_processor() -> AsyncCitationProcessor:
    """Get or create the global citation processor.

    Returns:
        A singleton instance of AsyncCitationProcessor.
    """
    global _citation_processor
    
    if _citation_processor is None:
        _citation_processor = AsyncCitationProcessor()
    
    return _citation_processor
# --------------------------------------------------------------------------------- end get_citation_processor()

# --------------------------------------------------------------------------------- shutdown_citation_processor()
async def shutdown_citation_processor() -> None:
    """Shutdown the global citation processor.

    Returns:
        None
    """
    global _citation_processor
    
    if _citation_processor:
        await _citation_processor.cleanup()
        _citation_processor = None
# --------------------------------------------------------------------------------- end shutdown_citation_processor()

# __________________________________________________________________________
# End of File