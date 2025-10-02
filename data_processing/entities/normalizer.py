# -------------------------------------------------------------------------
# File: data_processing/entities/normalizer.py
# Author: Alexander Ricciardi
# Date: 2025-10-02
# [File Path] data_processing/entities/normalizer.py
# ------------------------------------------------------------------------
# Project: APH-IF — Advanced Parallel HybridRAG – Intelligent Fusion
#
# Module Functionality:
#   Normalization and deduplication utilities for extracted entities. Provides
#   canonicalization, frequency aggregation, and type filtering for downstream use.
#
# Module Contents Overview:
# - Dataclass: Entity
# - Class: EntityNormalizer
#
# Dependencies / Imports:
# - Standard Library: logging, re, collections
# - Third-Party: (none)
#
# Usage / Integration:
#   Used by data_processing entity extraction to normalize and deduplicate raw
#   matches (regex/spaCy) into a canonical entity inventory.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""Entity normalization and deduplication."""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import logging
import re
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ____________________________________________________________________________
# Class Definitions
# TODO: Consider @dataclass(slots=True, kw_only=True) for performance in a refactor PR.
# ------------------------------------------------------------------------- class Entity
@dataclass
class Entity:
    """Normalized entity record.

    Attributes:
        name: Original (normalized) surface form for display.
        type: Entity label (e.g., "LEGAL_SECTION", "CFR_PART").
        canonical_name: Canonical key used for grouping/deduplication.
        occurrences: Number of raw occurrences aggregated into this entity.
        method: Comma-delimited provenance of extraction methods (e.g., "regex,spacy").
    """

    name: str
    type: str
    canonical_name: str
    occurrences: int = 1
    method: str = "unknown"
# ------------------------------------------------------------------------- end class Entity

# ------------------------------------------------------------------------- class EntityNormalizer
class EntityNormalizer:
    """Normalizes and deduplicates extracted entities.

    Responsibilities:
        - Normalize raw surface text (whitespace, symbols, punctuation).
        - Compute canonical names for grouping by type-specific rules.
        - Aggregate occurrences across methods and return `Entity` objects.
        - Filter entities by an allowed type list.
    """

    # -------------------------------------------------------------- normalize_text()
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize entity text to canonical form.

        Args:
            text: Raw entity text

        Returns:
            Normalized text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Standardize section symbol
        text = text.replace('§', '§ ')
        text = re.sub(r'§\s+', '§', text)

        # Normalize CFR references
        text = re.sub(r'\s+CFR\s+', ' CFR ', text)

        # Remove trailing punctuation
        text = text.rstrip('.,;:')

        return text
    # -------------------------------------------------------------- end normalize_text()

    # -------------------------------------------------------------- get_canonical_name()
    @staticmethod
    def get_canonical_name(text: str, entity_type: str) -> str:
        """Generate canonical name for grouping.

        Args:
            text: Normalized entity text
            entity_type: Entity type

        Returns:
            Canonical name (usually uppercased, stripped)
        """
        canonical = text.upper()

        # Type-specific normalization
        if entity_type == "LEGAL_SECTION":
            # Ensure consistent § symbol
            canonical = canonical.replace('§', '§')
            canonical = re.sub(r'\s+', '', canonical)

        elif entity_type == "CFR_PART":
            # Normalize "Part 75" vs "part 75"
            canonical = re.sub(r'\bPART\s+', 'PART', canonical)

        elif entity_type == "SUBPART":
            canonical = re.sub(r'\bSUBPART\s+', 'SUBPART', canonical)

        elif entity_type == "APPENDIX":
            canonical = re.sub(r'\bAPPENDIX\s+', 'APPENDIX', canonical)

        return canonical
    # -------------------------------------------------------------- end get_canonical_name()

    # -------------------------------------------------------------- deduplicate()
    @classmethod
    def deduplicate(cls, entities: list[dict]) -> list[Entity]:
        """Deduplicate and count entity occurrences.

        Args:
            entities: List of raw entities from extraction

        Returns:
            List of deduplicated Entity objects
        """
        # Group by (canonical_name, type)
        entity_map = defaultdict(lambda: {"count": 0, "methods": set()})

        for ent in entities:
            text = ent["text"]
            ent_type = ent["type"]
            method = ent.get("method", "unknown")

            # Normalize
            normalized = cls.normalize_text(text)
            canonical = cls.get_canonical_name(normalized, ent_type)

            key = (canonical, ent_type)
            entity_map[key]["count"] += 1
            entity_map[key]["methods"].add(method)
            entity_map[key]["normalized"] = normalized

        # Convert to Entity objects
        deduplicated = []
        for (canonical, ent_type), data in entity_map.items():
            deduplicated.append(
                Entity(
                    name=data["normalized"],
                    type=ent_type,
                    canonical_name=canonical,
                    occurrences=data["count"],
                    method=",".join(sorted(data["methods"])),
                )
            )

        logger.info(
            f"Deduplicated {len(entities)} raw entities to "
            f"{len(deduplicated)} unique entities"
        )

        return deduplicated
    # -------------------------------------------------------------- end deduplicate()

    # -------------------------------------------------------------- filter_by_type()
    @classmethod
    def filter_by_type(
        cls,
        entities: list[Entity],
        allowed_types: list[str]
    ) -> list[Entity]:
        """Filter entities by allowed types.

        Args:
            entities: List of entities
            allowed_types: List of allowed entity types

        Returns:
            Filtered list
        """
        filtered = [ent for ent in entities if ent.type in allowed_types]
        logger.info(
            f"Filtered {len(entities)} entities to {len(filtered)} "
            f"by types: {allowed_types}"
        )
        return filtered
    # -------------------------------------------------------------- end filter_by_type()

# ------------------------------------------------------------------------- end class EntityNormalizer

# __________________________________________________________________________
# End of File
#
