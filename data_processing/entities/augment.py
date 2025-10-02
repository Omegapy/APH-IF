# -------------------------------------------------------------------------
# File: data_processing/entities/augment.py
# Author: Alexander Ricciardi
# Date: 2025-10-02
# [File Path] data_processing/entities/augment.py
# ------------------------------------------------------------------------
# Project: APH-IF — Advanced Parallel HybridRAG – Intelligent Fusion
#
# Module Functionality:
#   LLM-powered relationship augmentation between extracted entities. Detects
#   semantic relationships (e.g., REFERENCES, PART_OF) and returns edge data
#   suitable for Neo4j upserts.
#
# Module Contents Overview:
# - Dataclass: EntityRelationship
# - Class: RelationshipAugmenter
# - Function: augment_relationships
#
# Dependencies / Imports:
# - Standard Library: logging, dataclasses, typing
# - Third-Party: openai
# - Local Modules: entities.normalizer.Entity
#
# Usage / Integration:
#   Used by data_processing Phase 3 to augment entity graphs with RELATED_TO edges.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""LLM-based entity relationship augmentation using GPT-5-mini.

Detects semantic relationships between legal entities and creates RELATED_TO edges.
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import logging
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from .normalizer import Entity

logger = logging.getLogger(__name__)


# ____________________________________________________________________________
# Class Definitions
# ------------------------------------------------------------------------- class EntityRelationship
@dataclass
class EntityRelationship:
    """Represents a relationship between two entities.

    Attributes:
        source_entity: Canonical name of the source entity.
        target_entity: Canonical name of the target entity.
        relationship_type: Relationship label (e.g., "RELATED_TO", "PART_OF").
        confidence: Detector confidence in [0.0, 1.0].
        source: Provenance tag (e.g., "llm").
        reason: Short natural-language rationale from the detector.
    """

    source_entity: str
    target_entity: str
    relationship_type: str = "RELATED_TO"
    confidence: float = 0.0
    source: str = "llm"
    reason: str = ""


# ------------------------------------------------------------------------- end class EntityRelationship

# ------------------------------------------------------------------------- class RelationshipAugmenter
class RelationshipAugmenter:
    """Augments entities with LLM-detected relationships.

    Responsibilities:
        - Call an LLM with a structured prompt to detect relationships.
        - Parse LLM output and enforce a confidence threshold.
        - Batch over multiple chunks with optional processing limits.

    Attributes:
        client: OpenAI client bound to the provided API key.
        model: Chat model identifier used for augmentation.
        max_tokens: Cap for completion tokens when calling the LLM.
        temperature: Generation temperature (kept low for consistency).
    """

    SYSTEM_PROMPT = """You are a legal document relationship analyzer.
Your task is to identify meaningful relationships between legal references (CFR sections, parts, subparts, appendices).

Rules:
1. Only identify direct, explicit relationships
2. Common relationships:
   - REFERENCES: One entity explicitly references another
   - PART_OF: Entity is a subdivision of another (e.g., section is part of a part)
   - SUPERSEDES: One entity replaces or updates another
   - RELATES_TO: General semantic relationship

3. Return ONLY valid JSON array, no explanation:
[
  {
    "source": "entity_name_1",
    "target": "entity_name_2",
    "type": "PART_OF",
    "confidence": 0.85,
    "reason": "Brief explanation"
  }
]

4. If no relationships found, return: []
5. Confidence: 0.0-1.0 (only include if >= 0.6)
"""

    # -------------------------------------------------------------- __init__()
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini",
        max_tokens: int = 500,
        temperature: float = 0.1,
    ):
        """Initialize relationship augmenter.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-5-mini)
            max_tokens: Maximum tokens for response
            temperature: Temperature for generation (low for consistency)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    # -------------------------------------------------------------- end __init__()

    # -------------------------------------------------------------- detect_relationships_in_chunk()
    def detect_relationships_in_chunk(
        self,
        entities: list[Entity],
        chunk_text: Optional[str] = None,
    ) -> list[EntityRelationship]:
        """Detect relationships between entities in a chunk.

        Args:
            entities: List of entities from the chunk
            chunk_text: Optional chunk text for context

        Returns:
            List of detected relationships
        """
        if len(entities) < 2:
            # Need at least 2 entities to form a relationship
            return []

        # Build entity list for prompt
        entity_list = [
            f"- {ent.name} ({ent.type})" for ent in entities
        ]

        user_prompt = f"""Analyze these legal entities and identify relationships:

Entities:
{chr(10).join(entity_list)}

Context (optional):
{chunk_text[:500] if chunk_text else 'N/A'}

Return relationships as JSON array."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            import json

            try:
                relationships_data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                relationships_data = json.loads(content)

            # Convert to EntityRelationship objects
            relationships = []
            for rel in relationships_data:
                if isinstance(rel, dict):
                    # Validate confidence threshold
                    confidence = float(rel.get("confidence", 0.0))
                    if confidence >= 0.6:
                        relationships.append(
                            EntityRelationship(
                                source_entity=rel["source"],
                                target_entity=rel["target"],
                                relationship_type=rel.get("type", "RELATED_TO"),
                                confidence=confidence,
                                source="llm",
                                reason=rel.get("reason", ""),
                            )
                        )

            logger.debug(
                f"Detected {len(relationships)} relationships from "
                f"{len(entities)} entities"
            )

            return relationships

        except Exception as e:
            logger.error(f"Failed to detect relationships: {e}")
            return []
    # -------------------------------------------------------------- end detect_relationships_in_chunk()

    # -------------------------------------------------------------- augment_chunk_entities()
    def augment_chunk_entities(
        self,
        chunk_entities: dict[str, list[Entity]],
        chunk_texts: Optional[dict[str, str]] = None,
        max_chunks: int = 0,
    ) -> dict[str, list[EntityRelationship]]:
        """Augment multiple chunks with relationship detection.

        Args:
            chunk_entities: Dict mapping chunk_id -> list of entities
            chunk_texts: Optional dict mapping chunk_id -> chunk text
            max_chunks: Maximum chunks to process (0 = no limit)

        Returns:
            Dict mapping chunk_id -> list of relationships
        """
        chunk_relationships = {}
        chunks_processed = 0

        for chunk_id, entities in chunk_entities.items():
            if max_chunks > 0 and chunks_processed >= max_chunks:
                logger.info(
                    f"Reached max chunks limit ({max_chunks}), stopping"
                )
                break

            chunk_text = chunk_texts.get(chunk_id) if chunk_texts else None

            relationships = self.detect_relationships_in_chunk(
                entities, chunk_text
            )

            if relationships:
                chunk_relationships[chunk_id] = relationships

            chunks_processed += 1

            if chunks_processed % 10 == 0:
                logger.info(
                    f"Processed {chunks_processed}/{len(chunk_entities)} chunks"
                )

        total_relationships = sum(
            len(rels) for rels in chunk_relationships.values()
        )
        logger.info(
            f"Augmentation complete: {total_relationships} relationships "
            f"detected across {len(chunk_relationships)} chunks"
        )

        return chunk_relationships
    # -------------------------------------------------------------- end augment_chunk_entities()

# ------------------------------------------------------------------------- end class RelationshipAugmenter

# __________________________________________________________________________
# Standalone Function Definitions
#
# --------------------------------------------------------------------------------- augment_relationships()
def augment_relationships(
    chunk_entities: dict[str, list[Entity]],
    config,
    chunk_texts: Optional[dict[str, str]] = None,
) -> dict[str, list[EntityRelationship]]:
    """Main entry point for relationship augmentation.

    Args:
        chunk_entities: Dict mapping chunk_id -> list of entities
        config: ProcessingConfig instance
        chunk_texts: Optional dict mapping chunk_id -> chunk text

    Returns:
        Dict mapping chunk_id -> list of relationships
    """
    augmenter = RelationshipAugmenter(
        api_key=config.openai_api_key,
        model=config.openai_model_mini,
        max_tokens=config.openai_max_tokens,
    )

    # Limit processing to avoid excessive API costs
    max_chunks = getattr(config, "max_augmentation_chunks", 100)

    return augmenter.augment_chunk_entities(
        chunk_entities,
        chunk_texts=chunk_texts,
        max_chunks=max_chunks,
    )

# --------------------------------------------------------------------------------- end augment_relationships()

# __________________________________________________________________________
# End of File
#
