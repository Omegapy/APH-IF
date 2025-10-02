"""Entity extraction package for legal/regulatory documents."""

from .augment import EntityRelationship, RelationshipAugmenter, augment_relationships
from .extract import EntityExtractor, extract_entities
from .normalizer import Entity, EntityNormalizer
from .pipeline import EntityExtractionPipeline
from .rules import LegalPatterns, extract_with_regex

__all__ = [
    "extract_entities",
    "EntityExtractor",
    "Entity",
    "EntityNormalizer",
    "EntityExtractionPipeline",
    "LegalPatterns",
    "extract_with_regex",
    "EntityRelationship",
    "RelationshipAugmenter",
    "augment_relationships",
]
