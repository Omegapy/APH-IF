# -------------------------------------------------------------------------
# File: data_processing/entities/pipeline.py
# Author: Alexander Ricciardi
# Date: 2025-10-02
# [File Path] data_processing/entities/pipeline.py
# ------------------------------------------------------------------------
# Project: APH-IF — Advanced Parallel HybridRAG – Intelligent Fusion
#
# Module Functionality:
#   Builds and manages a spaCy pipeline (EntityRuler + PhraseMatcher) for
#   extracting legal/regulatory references used during entity extraction.
#
# Module Contents Overview:
# - Class: EntityExtractionPipeline
#
# Dependencies / Imports:
# - Standard Library: json, logging, pathlib, typing
# - Third-Party: spacy
#
# Usage / Integration:
#   Consumed by data_processing entity extraction to produce structured entity
#   candidates complementing regex-based rules.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""spaCy NLP pipeline builder for entity extraction."""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import json
import logging
from pathlib import Path
from typing import Optional

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler

logger = logging.getLogger(__name__)

# ____________________________________________________________________________
# Class Definitions
# ------------------------------------------------------------------------- class EntityExtractionPipeline
class EntityExtractionPipeline:
    """Builds and manages spaCy pipeline for legal entity extraction.

    Responsibilities:
        - Load and configure a spaCy Language model.
        - Attach an EntityRuler with JSONL patterns and a PhraseMatcher with lexicons.
        - Provide a method to extract entities from text using the configured pipeline.

    Attributes:
        model_name: Name of the spaCy model to load.
        patterns_dir: Directory containing EntityRuler JSONL patterns.
        lexicons_dir: Directory containing phrase lexicons (JSON).
        nlp: The configured spaCy Language instance once built.
        phrase_matcher: PhraseMatcher instance attached to the pipeline.
    """

    # -------------------------------------------------------------- __init__()
    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        patterns_dir: Optional[Path] = None,
        lexicons_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self.patterns_dir = patterns_dir or Path(__file__).parent / "patterns"
        self.lexicons_dir = lexicons_dir or Path(__file__).parent / "lexicons"
        self.nlp: Optional[Language] = None
        self.phrase_matcher: Optional[PhraseMatcher] = None

    # -------------------------------------------------------------- build()
    def build(self) -> Language:
        """Build the complete spaCy pipeline.

        Returns:
            Configured spaCy Language object
        """
        logger.info(f"Loading spaCy model: {self.model_name}")

        # Load base model
        try:
            self.nlp = spacy.load(self.model_name)
        except OSError:
            logger.error(
                f"spaCy model '{self.model_name}' not found. "
                f"Run: python -m spacy download {self.model_name}"
            )
            raise

        # Add EntityRuler
        self._add_entity_ruler()

        # Add PhraseMatcher
        self._add_phrase_matcher()

        logger.info("Entity extraction pipeline built successfully")
        return self.nlp
    # -------------------------------------------------------------- end build()

    # -------------------------------------------------------------- _add_entity_ruler()
    def _add_entity_ruler(self) -> None:
        """Add EntityRuler with patterns from JSONL files."""
        patterns_file = self.patterns_dir / "legal_patterns.jsonl"

        if not patterns_file.exists():
            logger.warning(f"Patterns file not found: {patterns_file}")
            return

        # Load patterns
        patterns = []
        with open(patterns_file, "r") as f:
            for line in f:
                if line.strip():
                    patterns.append(json.loads(line))

        logger.info(f"Loaded {len(patterns)} EntityRuler patterns")

        # Add ruler to pipeline (before NER to have priority)
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(patterns)
    # -------------------------------------------------------------- end _add_entity_ruler()

    # -------------------------------------------------------------- _add_phrase_matcher()
    def _add_phrase_matcher(self) -> None:
        """Add PhraseMatcher with lexicon terms."""
        lexicon_file = self.lexicons_dir / "legal_terms.json"

        if not lexicon_file.exists():
            logger.warning(f"Lexicon file not found: {lexicon_file}")
            return

        # Load lexicons
        with open(lexicon_file, "r") as f:
            lexicons = json.load(f)

        # Create PhraseMatcher
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        # Add patterns for each category
        for category, terms in lexicons.items():
            patterns = [self.nlp.make_doc(term) for term in terms]
            self.phrase_matcher.add(category.upper(), patterns)
            logger.info(f"Added {len(terms)} phrases for {category}")
    # -------------------------------------------------------------- end _add_phrase_matcher()

    # -------------------------------------------------------------- extract_entities_spacy()
    def extract_entities_spacy(self, text: str) -> list[dict]:
        """Extract entities using spaCy pipeline.

        Args:
            text: Input text

        Returns:
            List of {"text": str, "type": str, "start": int, "end": int}
        """
        if self.nlp is None:
            raise RuntimeError("Pipeline not built. Call build() first.")

        doc = self.nlp(text)
        entities = []

        # Extract from EntityRuler + NER
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "method": "spacy",
            })

        # Extract from PhraseMatcher
        if self.phrase_matcher:
            matches = self.phrase_matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                entities.append({
                    "text": span.text,
                    "type": self.nlp.vocab.strings[match_id],
                    "start": span.start_char,
                    "end": span.end_char,
                    "method": "phrase_matcher",
                })

        return entities
    # -------------------------------------------------------------- end extract_entities_spacy()

# ------------------------------------------------------------------------- end class EntityExtractionPipeline

# __________________________________________________________________________
# End of File
#
