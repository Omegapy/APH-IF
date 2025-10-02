"""
APH-IF Schema Management Module

This module provides comprehensive knowledge graph schema acquisition,
storage, and management for the APH-IF backend system.
"""

from .schema_acquirer import SchemaAcquirer
from .schema_manager import SchemaManager, get_schema_manager
from .schema_models import (
    CompleteKGSchema,
    ComprehensiveNodeInfo,
    ComprehensiveRelationshipInfo,
    SchemaCache,
)

__all__ = [
    'SchemaManager',
    'get_schema_manager',
    'ComprehensiveNodeInfo',
    'ComprehensiveRelationshipInfo',
    'CompleteKGSchema', 
    'SchemaCache',
    'SchemaAcquirer'
]