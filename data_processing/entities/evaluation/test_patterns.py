"""Unit tests for entity extraction patterns."""

import pytest
from entities.rules import extract_with_regex, LegalPatterns
from entities.normalizer import EntityNormalizer


class TestRegexPatterns:
    """Test regex pattern matching."""

    def test_cfr_section_symbol(self):
        """Test §75.1714-1 style."""
        text = "See §75.1714-1 for details."
        entities = extract_with_regex(text)

        assert len(entities) >= 1
        assert any(
            ent["type"] == "LEGAL_SECTION" and "75.1714" in ent["text"]
            for ent in entities
        )

    def test_cfr_shorthand(self):
        """Test 30 CFR 75.1714(a) style."""
        text = "Refer to 30 CFR 75.1714(a)."
        entities = extract_with_regex(text)

        assert len(entities) >= 1
        assert any(
            ent["type"] == "LEGAL_SECTION" and "CFR" in ent["text"]
            for ent in entities
        )

    def test_part_reference(self):
        """Test Part 75 style."""
        text = "Part 75 applies to underground mines."
        entities = extract_with_regex(text)

        assert len(entities) >= 1
        assert any(
            ent["type"] == "CFR_PART" and "Part 75" in ent["text"]
            for ent in entities
        )

    def test_subpart(self):
        """Test Subpart D style."""
        text = "Requirements in Subpart D must be followed."
        entities = extract_with_regex(text)

        assert len(entities) >= 1
        assert any(
            ent["type"] == "SUBPART" and "Subpart D" in ent["text"]
            for ent in entities
        )

    def test_appendix(self):
        """Test Appendix A style."""
        text = "See Appendix A to Part 75."
        entities = extract_with_regex(text)

        assert len(entities) >= 1
        assert any(
            ent["type"] == "APPENDIX" and "Appendix A" in ent["text"]
            for ent in entities
        )

    def test_title(self):
        """Test Title 30 style."""
        text = "Title 30 regulations apply."
        entities = extract_with_regex(text)

        assert len(entities) >= 1
        assert any(
            ent["type"] == "CFR_TITLE" and "Title 30" in ent["text"]
            for ent in entities
        )

    def test_standard(self):
        """Test ISO 9001 style."""
        text = "Equipment must meet ISO 9001 standards."
        entities = extract_with_regex(text)

        assert len(entities) >= 1
        assert any(
            ent["type"] == "STANDARD" and "ISO 9001" in ent["text"]
            for ent in entities
        )


class TestNormalizer:
    """Test entity normalization."""

    def test_normalize_text(self):
        """Test text normalization."""
        assert EntityNormalizer.normalize_text("  § 75.1714  ") == "§75.1714"
        assert EntityNormalizer.normalize_text("Part  75.") == "Part 75"

    def test_canonical_name(self):
        """Test canonical name generation."""
        canonical = EntityNormalizer.get_canonical_name(
            "§75.1714", "LEGAL_SECTION"
        )
        assert canonical == "§75.1714"

    def test_deduplication(self):
        """Test entity deduplication."""
        entities = [
            {"text": "§75.1714", "type": "LEGAL_SECTION", "method": "regex"},
            {"text": "§ 75.1714", "type": "LEGAL_SECTION", "method": "spacy"},
            {"text": "Part 75", "type": "CFR_PART", "method": "regex"},
        ]

        deduplicated = EntityNormalizer.deduplicate(entities)

        # §75.1714 should be deduplicated to 1
        legal_sections = [e for e in deduplicated if e.type == "LEGAL_SECTION"]
        assert len(legal_sections) == 1
        assert legal_sections[0].occurrences == 2
