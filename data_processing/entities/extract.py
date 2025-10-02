# -------------------------------------------------------------------------
# File: data_processing/entities/extract.py
# Author: Alexander Ricciardi
# Date: 2025-10-02
# [File Path] data_processing/entities/extract.py
# ------------------------------------------------------------------------
# Project: APH-IF — Advanced Parallel HybridRAG – Intelligent Fusion
#
# Module Functionality:
#   Orchestrates entity extraction from text/chunks/pages using a hybrid
#   approach (spaCy pipeline + regex rules), followed by normalization and
#   type filtering. Exposes a top-level helper to process chunk lists.
#
# Module Contents Overview:
# - Class: EntityExtractor
# - Function: extract_entities
#
# Dependencies / Imports:
# - Standard Library: logging, pathlib, typing
# - Local Modules: chunker.ChunkRecord, docling_adapter.PageRecord
# - Entities Package: normalizer, pipeline, rules
#
# Usage / Integration:
#   Used by data_processing Phase 2 to extract structured entities per chunk
#   for Neo4j upserts.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""Main entity extraction orchestrator combining all methods."""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import logging
from pathlib import Path
from typing import Optional

from chunker import ChunkRecord
from docling_adapter import PageRecord

from .normalizer import Entity, EntityNormalizer
from .pipeline import EntityExtractionPipeline
from .rules import extract_with_regex

logger = logging.getLogger(__name__)

# ____________________________________________________________________________
# Class Definitions
# ------------------------------------------------------------------------- class EntityExtractor
class EntityExtractor:
    """Orchestrates entity extraction using multiple methods.

    Responsibilities:
        - Build a spaCy pipeline (EntityRuler + PhraseMatcher) and keep it ready.
        - Run hybrid extraction on text (spaCy + regex) and deduplicate results.
        - Provide helpers to extract from chunks and pages and filter by allowed types.

    Attributes:
        allowed_types: List of entity labels to keep.
        pipeline_builder: Configured ``EntityExtractionPipeline`` instance.
        nlp: Built spaCy ``Language`` object.
        normalizer: ``EntityNormalizer`` used for deduplication and filtering.
    """

    # -------------------------------------------------------------- __init__()
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        patterns_dir: Optional[Path] = None,
        lexicons_dir: Optional[Path] = None,
        allowed_types: Optional[list[str]] = None,
    ):
        self.allowed_types = allowed_types or [
            "LEGAL_SECTION",
            "CFR_PART",
            "SUBPART",
            "APPENDIX",
            "CFR_TITLE",
            "STANDARD",
        ]

        # Build spaCy pipeline
        self.pipeline_builder = EntityExtractionPipeline(
            model_name=spacy_model,
            patterns_dir=patterns_dir,
            lexicons_dir=lexicons_dir,
        )
        self.nlp = self.pipeline_builder.build()
        self.normalizer = EntityNormalizer()
    # -------------------------------------------------------------- end __init__()

    # -------------------------------------------------------------- extract_from_text()
    def extract_from_text(self, text: str) -> list[Entity]:
        """Extract entities from raw text.

        Args:
            text: Input text

        Returns:
            List of deduplicated Entity objects
        """
        # Method 1: spaCy (EntityRuler + PhraseMatcher)
        spacy_entities = self.pipeline_builder.extract_entities_spacy(text)

        # Method 2: Regex
        regex_entities = extract_with_regex(text)

        # Combine all entities
        all_entities = spacy_entities + regex_entities

        # Deduplicate and normalize
        deduplicated = self.normalizer.deduplicate(all_entities)

        # Filter by allowed types
        filtered = self.normalizer.filter_by_type(
            deduplicated,
            self.allowed_types
        )

        return filtered
    # -------------------------------------------------------------- end extract_from_text()

    # -------------------------------------------------------------- extract_from_chunks()
    def extract_from_chunks(
        self,
        chunks: list[ChunkRecord]
    ) -> dict[str, list[Entity]]:
        """Extract entities from chunk records.

        Args:
            chunks: List of ChunkRecord objects

        Returns:
            Dict mapping chunk_id -> list of entities
        """
        chunk_entities = {}

        for chunk in chunks:
            entities = self.extract_from_text(chunk.text)
            chunk_entities[chunk.chunk_id] = entities

            logger.debug(
                f"Extracted {len(entities)} entities from {chunk.chunk_id}"
            )

        total_entities = sum(len(ents) for ents in chunk_entities.values())
        logger.info(
            f"Extracted {total_entities} total entities from "
            f"{len(chunks)} chunks"
        )

        return chunk_entities
    # -------------------------------------------------------------- end extract_from_chunks()

    # -------------------------------------------------------------- extract_from_pages()
    def extract_from_pages(
        self,
        pages: list[PageRecord],
        use_structure: bool = True,
    ) -> list[Entity]:
        """Extract entities from page records with structural hints.

        Args:
            pages: List of PageRecord objects from Docling
            use_structure: Use headings/sections to constrain extraction

        Returns:
            List of deduplicated entities across all pages
        """
        all_entities = []

        for page in pages:
            # Extract from main text
            page_entities = self.extract_from_text(page.text)

            # If using structure, prioritize entities from headings
            if use_structure and page.headings:
                for heading in page.headings:
                    heading_entities = self.extract_from_text(heading)
                    # Boost priority (could add a weight field)
                    all_entities.extend(heading_entities)

            all_entities.extend(page_entities)

        # Final deduplication across all pages
        # Convert back to raw format for deduplication
        raw_entities = [
            {
                "text": ent.name,
                "type": ent.type,
                "method": ent.method,
            }
            for ent in all_entities
        ]

        deduplicated = self.normalizer.deduplicate(raw_entities)

        return deduplicated
    # -------------------------------------------------------------- end extract_from_pages()

# ------------------------------------------------------------------------- end class EntityExtractor

# __________________________________________________________________________
# Standalone Function Definitions
#
# --------------------------------------------------------------------------------- extract_entities()
def extract_entities(
    chunks: list[ChunkRecord],
    config,
) -> dict[str, list[Entity]]:
    """Main entry point for entity extraction.

    Args:
        chunks: List of ChunkRecord objects
        config: ProcessingConfig instance

    Returns:
        Dict mapping chunk_id -> list of entities
    """
    extractor = EntityExtractor(
        spacy_model=config.spacy_model,
        patterns_dir=config.entity_patterns_dir,
        lexicons_dir=config.entity_lexicons_dir,
        allowed_types=config.entity_types,
    )

    return extractor.extract_from_chunks(chunks)

# --------------------------------------------------------------------------------- end extract_entities()

# __________________________________________________________________________
# End of File
#
