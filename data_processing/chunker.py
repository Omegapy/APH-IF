# -------------------------------------------------------------------------
# File: data_processing/chunker.py
# Author: Alexander Ricciardi
# Date: 2025-10-02
# [File Path] data_processing/chunker.py
# ------------------------------------------------------------------------
# Project: APH-IF — Advanced Parallel HybridRAG – Intelligent Fusion
#
# Module Functionality:
#   Page-aware chunking utilities that split Docling page records into text chunks and
#   compute basic metadata used downstream (section hints, token estimates).
#
# Module Contents Overview:
# - Class (dataclass): ChunkRecord
# - Function: chunk_pages
# - Function: _split_text_into_chunks
# - Function: _split_into_sentences
# - Function: generate_document_id
#
# Dependencies / Imports:
# - Standard Library: logging, dataclasses, typing
# - Local Project Modules: docling_adapter.PageRecord
#
# Usage / Integration:
#   Used by Phase 1 ingestion within data_processing to create chunk nodes for Neo4j.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""Page-aware chunking for document processing.

Splits document pages into chunks while maintaining page and section metadata.
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import logging
from dataclasses import dataclass
from typing import Optional

from docling_adapter import PageRecord

logger = logging.getLogger(__name__)

# ____________________________________________________________________________
# Class Definitions
# ------------------------------------------------------------------------- class ChunkRecord
@dataclass
class ChunkRecord:
    """Record for a text chunk with metadata."""

    chunk_id: str
    document_id: str
    page: int
    text: str
    section: Optional[str] = None
    tokens: int = 0
    entities: list[str] = None
    embedding: list[float] = None

    def __post_init__(self) -> None:
        """Initialize empty lists for mutable fields."""
        if self.entities is None:
            self.entities = []

# ------------------------------------------------------------------------- end class ChunkRecord

# __________________________________________________________________________
# Standalone Function Definitions
#
# --------------------------------------------------------------------------------- chunk_pages()
def chunk_pages(
    pages: list[PageRecord],
    document_id: str,
    chunk_size_chars: int,
) -> list[ChunkRecord]:
    """Chunk pages into fixed-size text chunks with metadata.

    Args:
        pages: List of PageRecord objects from Docling
        document_id: Unique identifier for the document
        chunk_size_chars: Target size for each chunk in characters

    Returns:
        List of ChunkRecord objects with page and section metadata
    """
    if not pages:
        logger.warning(f"No pages provided for chunking document {document_id}")
        return []

    chunks: list[ChunkRecord] = []

    for page_record in pages:
        page_num = page_record.page
        page_text = page_record.text
        sections = page_record.sections
        headings = page_record.headings

        # Determine section name for this page
        section_name = None
        if headings:
            section_name = headings[0]  # Use first heading as section name
        elif sections:
            section_name = sections[0]  # Use first section ID

        # Split page text into chunks
        page_chunks = _split_text_into_chunks(page_text, chunk_size_chars)

        # Create ChunkRecord for each chunk
        for chunk_ordinal, chunk_text in enumerate(page_chunks):
            chunk_id = f"{document_id}_p{page_num}_c{chunk_ordinal}"

            # Estimate token count (rough approximation: ~4 chars per token)
            tokens = len(chunk_text) // 4

            chunk = ChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                page=page_num,
                text=chunk_text,
                section=section_name,
                tokens=tokens,
            )

            chunks.append(chunk)

    logger.info(f"Created {len(chunks)} chunks for document {document_id}")
    return chunks
# --------------------------------------------------------------------------------- end chunk_pages()

# --------------------------------------------------------------------------------- _split_text_into_chunks()
def _split_text_into_chunks(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks of approximately chunk_size characters.

    Tries to split on sentence boundaries when possible to maintain semantic coherence.

    Args:
        text: Text to split
        chunk_size: Target size for each chunk in characters

    Returns:
        List of text chunks
    """
    if not text:
        return []

    # If text is shorter than chunk size, return as single chunk
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    current_chunk = ""

    # Split on sentence boundaries (rough heuristic)
    sentences = _split_into_sentences(text)

    for sentence in sentences:
        # If adding this sentence would exceed chunk size, save current chunk
        if current_chunk and len(current_chunk) + len(sentence) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    # Add remaining text as final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
# --------------------------------------------------------------------------------- end _split_text_into_chunks()

# --------------------------------------------------------------------------------- _split_into_sentences()
def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    # Simple sentence splitting on common terminators
    # This is a basic implementation; could be improved with more sophisticated NLP
    sentences = []
    current_sentence = ""

    for char in text:
        current_sentence += char

        # Check for sentence terminators
        if char in ".!?":
            # Look ahead to see if this is really the end of a sentence
            # (avoid splitting on abbreviations, decimals, etc.)
            sentences.append(current_sentence.strip())
            current_sentence = ""

    # Add any remaining text
    if current_sentence.strip():
        sentences.append(current_sentence.strip())

    return [s for s in sentences if s]  # Filter out empty strings
# --------------------------------------------------------------------------------- end _split_into_sentences()

# --------------------------------------------------------------------------------- generate_document_id()
def generate_document_id(filename: str) -> str:
    """Generate a stable document ID from filename.

    Args:
        filename: PDF filename (with or without extension)

    Returns:
        Document ID (filename without extension, lowercased with underscores)
    """
    # Remove extension if present
    if filename.endswith(".pdf"):
        filename = filename[:-4]

    # Convert to lowercase and replace spaces/special chars with underscores
    doc_id = filename.lower().replace(" ", "_").replace("-", "_")

    # Remove any non-alphanumeric characters except underscores
    doc_id = "".join(c for c in doc_id if c.isalnum() or c == "_")

    return doc_id
# --------------------------------------------------------------------------------- end generate_document_id()

# __________________________________________________________________________
# End of File
#
