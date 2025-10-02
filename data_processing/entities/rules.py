# -------------------------------------------------------------------------
# File: data_processing/entities/rules.py
# Author: Alexander Ricciardi
# Date: 2025-10-02
# [File Path] data_processing/entities/rules.py
# ------------------------------------------------------------------------
# Project: APH-IF — Advanced Parallel HybridRAG – Intelligent Fusion
#
# Module Functionality:
#   Compiled regex patterns and helpers for extracting legal/regulatory
#   references (CFR sections, parts, subparts, appendices, titles, standards).
#
# Module Contents Overview:
# - Class (dataclass): RegexPattern
# - Class: LegalPatterns
# - Function: extract_with_regex
#
# Dependencies / Imports:
# - Standard Library: re, dataclasses, typing
#
# Usage / Integration:
#   Used by Phase 2 entity extraction pipeline to augment spaCy-based matches.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""Compiled regex patterns for legal entity extraction."""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import re
from dataclasses import dataclass
from typing import Pattern

# ____________________________________________________________________________
# Class Definitions
# TODO: Consider @dataclass(slots=True, kw_only=True) in a focused refactor PR.
# ------------------------------------------------------------------------- class RegexPattern
@dataclass
class RegexPattern:
    """Regex pattern with metadata.

    Attributes:
        name: Human-readable identifier for the pattern.
        pattern: Compiled regex object.
        entity_type: Target entity label to emit when matched.
        priority: Ordering weight; higher values are matched/emitted first.
        description: Short, human-readable explanation of the pattern.
    """

    name: str
    pattern: Pattern
    entity_type: str
    priority: int = 0
    description: str = ""
# ------------------------------------------------------------------------- end class RegexPattern

# ------------------------------------------------------------------------- class LegalPatterns
class LegalPatterns:
    """Compiled regex patterns for legal citations.

    Provides curated expressions for common CFR references and industry standards.
    Patterns are exposed via class variables and enumerated by ``get_all_patterns``.
    """

    # CFR Section with symbol: §75.1714-1, § 57.4361(a)
    CFR_SECTION_SYMBOL = re.compile(
        r'§\s*(\d+(?:\.\d+)+(?:-\d+)?(?:\([a-z0-9]+\))?)',
        re.IGNORECASE
    )

    # Part reference: Part 75.1714, Part 75
    CFR_PART = re.compile(
        r'\bPart\s+(\d+(?:\.\d+)?)\b',
        re.IGNORECASE
    )

    # Subpart: Subpart D, Subpart AC
    SUBPART = re.compile(
        r'\bSubpart\s+([A-Z]{1,2})\b'
    )

    # Appendix: Appendix A, Appendix B-1, Appendix A to Part 75
    APPENDIX = re.compile(
        r'\bAppendix\s+([A-Z](?:\s*[-–]\s*\d+)?(?:\s+to\s+Part\s+\d+)?)\b'
    )

    # Title: Title 30
    CFR_TITLE = re.compile(
        r'\bTitle\s+(\d+)\b'
    )

    # CFR shorthand: 30 CFR 75.1714(a)
    CFR_SHORTHAND = re.compile(
        r'\b(\d+\s*CFR\s+\d+(?:\.\d+)*(?:\([a-z0-9]+\))?)\b',
        re.IGNORECASE
    )

    # Standards: ISO 9001, ASTM D4318
    STANDARD = re.compile(
        r'\b((?:ISO|ASTM|ANSI|NIST)\s+[A-Z]?\d+(?:[-.]\d+)?)\b'
    )

    @classmethod
    # -------------------------------------------------------------- get_all_patterns()
    def get_all_patterns(cls) -> list[RegexPattern]:
        """Return all patterns with metadata."""
        return [
            RegexPattern(
                name="CFR_SECTION_SYMBOL",
                pattern=cls.CFR_SECTION_SYMBOL,
                entity_type="LEGAL_SECTION",
                priority=10,
                description="CFR section with § symbol"
            ),
            RegexPattern(
                name="CFR_SHORTHAND",
                pattern=cls.CFR_SHORTHAND,
                entity_type="LEGAL_SECTION",
                priority=9,
                description="CFR shorthand notation"
            ),
            RegexPattern(
                name="CFR_PART",
                pattern=cls.CFR_PART,
                entity_type="CFR_PART",
                priority=8,
                description="Part reference"
            ),
            RegexPattern(
                name="SUBPART",
                pattern=cls.SUBPART,
                entity_type="SUBPART",
                priority=7,
                description="Subpart identifier"
            ),
            RegexPattern(
                name="APPENDIX",
                pattern=cls.APPENDIX,
                entity_type="APPENDIX",
                priority=7,
                description="Appendix reference"
            ),
            RegexPattern(
                name="CFR_TITLE",
                pattern=cls.CFR_TITLE,
                entity_type="CFR_TITLE",
                priority=6,
                description="Title number"
            ),
            RegexPattern(
                name="STANDARD",
                pattern=cls.STANDARD,
                entity_type="STANDARD",
                priority=5,
                description="Industry standard"
            ),
        ]
    # -------------------------------------------------------------- end get_all_patterns()
# ------------------------------------------------------------------------- end class LegalPatterns

# __________________________________________________________________________
# Standalone Function Definitions
#
# --------------------------------------------------------------------------------- extract_with_regex()
def extract_with_regex(text: str) -> list[dict]:
    """Extract entities using regex patterns.

    Args:
        text: Input text to extract entities from

    Returns:
        List of {"text": str, "type": str, "start": int, "end": int}
    """
    entities = []

    for pattern_def in sorted(
        LegalPatterns.get_all_patterns(),
        key=lambda p: p.priority,
        reverse=True
    ):
        for match in pattern_def.pattern.finditer(text):
            entities.append({
                "text": match.group(0),
                "type": pattern_def.entity_type,
                "start": match.start(),
                "end": match.end(),
                "method": "regex",
                "pattern_name": pattern_def.name,
            })

    return entities
# --------------------------------------------------------------------------------- end extract_with_regex()

# __________________________________________________________________________
# End of File
#
