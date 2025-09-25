# -------------------------------------------------------------------------
# File: schema_models.py
# Author: Alexander Ricciardi
# Date:
# [File Path] backend/app/schema/schema_models.py
# -------------------------------------------------------------------------
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
#   Data models for the knowledge-graph schema: comprehensive per-label and
#   per-relationship information, a full schema aggregate, a lightweight
#   structural summary for LLMs, and an in-memory cache wrapper.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Dataclass: ComprehensiveNodeInfo
# - Dataclass: ComprehensiveRelationshipInfo
# - Dataclass: CompleteKGSchema
# - Dataclass: StructuralSummary
# - Dataclass: SchemaCache
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: json, dataclasses (dataclass/asdict), datetime, typing
# - Third-Party: (none)
# - Local Project Modules: (none)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# The SchemaManager composes and returns these models for read-only access via
# API endpoints; they are not instantiated directly by request handlers.
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent
# Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""
Schema models for APH-IF knowledge-graph management.

This module defines immutable-friendly, type-hinted dataclasses that capture:
- Comprehensive per-label and per-relationship details needed for deep analysis
- A complete schema snapshot with global properties/indices/constraints
- A compact structural summary used to augment LLM prompts efficiently
- A simple in-memory cache for schema and structural summaries
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


# ------------------------------------------------------------------------- class ComprehensiveNodeInfo
@dataclass
class ComprehensiveNodeInfo:
    """Complete information about a node label in the knowledge graph.

    Attributes:
        label: Node label name.
        total_count: Total number of nodes with this label.
        all_properties: Set of all property keys observed on nodes with this label.
        property_statistics: Per-property metrics (e.g., counts, distinct counts, data types).
        sample_nodes: Optional sample node payloads for inspection.
        constraints: Constraints related to this label (filtered from SHOW CONSTRAINTS).
        indexes: Indexes related to this label (filtered from SHOW INDEXES).
    """

    # ______________________
    #  Class Variable (excluded from dataclass constructor/compare via ClassVar)
    #

    # ______________________
    #  Instance Fields
    #
    label: str
    total_count: int
    all_properties: Set[str]
    property_statistics: Dict[str, Dict[str, Any]]
    sample_nodes: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    indexes: List[Dict[str, Any]]
    
    # ______________________
    # Post-Initialization (validation + derived fields)
    #
    # -------------------------------------------------------------- __post_init__()
    def __post_init__(self) -> None:
        """Normalize and initialize internal collections.

        Ensures set-based properties are coerced from lists and initializes
        optional collections to empty instances when missing.
        """
        # Ensure all_properties is a set
        if isinstance(self.all_properties, list):
            self.all_properties = set(self.all_properties)
        
        # Initialize empty collections if None
        if self.property_statistics is None:
            self.property_statistics = {}
        if self.sample_nodes is None:
            self.sample_nodes = []
        if self.constraints is None:
            self.constraints = []
        if self.indexes is None:
            self.indexes = []
    # -------------------------------------------------------------- end __post_init__()

    # ______________________
    # Constructor 
    # 

    # ______________________
    # Destructor (Use only if absolutely necessary for external resource cleanup)
    #

    # ______________________
    # -- Embedded Classes --
    #

    # ______________________
    # Getters (Property decorators are often preferred for simple getters)
    #

    # ______________________
    # Setters / Mutators
    #

    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #

    # =========================================================================
    # Functionality: e.g. Data Processing Utilities
    # =========================================================================

    # ______________________
    # Class Information Methods (Optional, but highly recommended) 
    #

    # -------------------------------------------------------------- to_dict()
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            A dict representation with sets converted to lists for JSON safety.
        """
        data = asdict(self)
        data["all_properties"] = list(self.all_properties)
        return data
    # -------------------------------------------------------------- end to_dict()

    # -------------------------------------------------------------- from_dict()
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComprehensiveNodeInfo":
        """Deserialize from a dictionary.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            A populated ComprehensiveNodeInfo instance.
        """
        if "all_properties" in data:
            data["all_properties"] = set(data["all_properties"])
        return cls(**data)
    # -------------------------------------------------------------- end from_dict()

# ------------------------------------------------------------------------- end class ComprehensiveNodeInfo


# ------------------------------------------------------------------------- class ComprehensiveRelationshipInfo
@dataclass
class ComprehensiveRelationshipInfo:
    """Complete information about a relationship type in the knowledge graph.

    Attributes:
        relationship_type: Relationship type name.
        total_count: Total number of relationships of this type.
        all_patterns: Source/target label patterns with counts.
        pattern_statistics: Mapping of pattern-string to frequency.
        all_properties: Set of relationship property keys observed.
        property_statistics: Per-property metrics (e.g., counts, distinct counts, data types).
        sample_relationships: Optional sample relationship payloads for inspection.
        constraints: Constraints related to this relationship type.
        indexes: Indexes related to this relationship type.
    """

    # ______________________
    #  Class Variable (excluded from dataclass constructor/compare via ClassVar)
    #

    # ______________________
    #  Instance Fields
    #
    relationship_type: str
    total_count: int
    all_patterns: List[Dict[str, Any]]
    pattern_statistics: Dict[str, int]
    all_properties: Set[str]
    property_statistics: Dict[str, Dict[str, Any]]
    sample_relationships: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    indexes: List[Dict[str, Any]]
    
    # ______________________
    # Post-Initialization (validation + derived fields)
    #
    # -------------------------------------------------------------- __post_init__()
    def __post_init__(self) -> None:
        """Normalize and initialize internal collections.

        Ensures set-based properties are coerced from lists and initializes
        optional collections to empty instances when missing.
        """
        # Ensure all_properties is a set
        if isinstance(self.all_properties, list):
            self.all_properties = set(self.all_properties)
        
        # Initialize empty collections if None
        if self.all_patterns is None:
            self.all_patterns = []
        if self.pattern_statistics is None:
            self.pattern_statistics = {}
        if self.property_statistics is None:
            self.property_statistics = {}
        if self.sample_relationships is None:
            self.sample_relationships = []
        if self.constraints is None:
            self.constraints = []
        if self.indexes is None:
            self.indexes = []
    # -------------------------------------------------------------- end __post_init__()

    # ______________________
    # Constructor 
    # 

    # ______________________
    # Destructor (Use only if absolutely necessary for external resource cleanup)
    #

    # ______________________
    # -- Embedded Classes --
    #

    # ______________________
    # Getters (Property decorators are often preferred for simple getters)
    #

    # ______________________
    # Setters / Mutators
    #

    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #

    # =========================================================================
    # Functionality: e.g. Data Processing Utilities
    # =========================================================================

    # ______________________
    # Class Information Methods (Optional, but highly recommended) 
    #

    # -------------------------------------------------------------- to_dict()
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            A dict representation with sets converted to lists for JSON safety.
        """
        data = asdict(self)
        data["all_properties"] = list(self.all_properties)
        return data
    # -------------------------------------------------------------- end to_dict()

    # -------------------------------------------------------------- from_dict()
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComprehensiveRelationshipInfo":
        """Deserialize from a dictionary.

        Args:
            data: Dictionary produced by :meth:`to_dict`.

        Returns:
            A populated ComprehensiveRelationshipInfo instance.
        """
        if "all_properties" in data:
            data["all_properties"] = set(data["all_properties"])
        return cls(**data)
    # -------------------------------------------------------------- end from_dict()

# ------------------------------------------------------------------------- end class ComprehensiveRelationshipInfo


# ------------------------------------------------------------------------- class CompleteKGSchema
@dataclass 
class CompleteKGSchema:
    """Complete knowledge graph schema with comprehensive information.

    Attributes:
        nodes: Mapping from node label to comprehensive node information.
        relationships: Mapping from relationship type to comprehensive information.
        global_property_keys: Set of all property keys observed globally.
        all_constraints: All constraints present in the database.
        all_indexes: All indexes present in the database.
        total_nodes: Total number of nodes at acquisition time.
        total_relationships: Total number of relationships at acquisition time.
        database_info: Raw database information and metadata.
        acquisition_timestamp: ISO timestamp for schema acquisition time.
        acquisition_duration_seconds: Duration to acquire the schema.
    """

    # ______________________
    #  Class Variable (excluded from dataclass constructor/compare via ClassVar)
    #

    # ______________________
    #  Instance Fields
    #
    nodes: Dict[str, ComprehensiveNodeInfo]
    relationships: Dict[str, ComprehensiveRelationshipInfo]
    global_property_keys: Set[str]
    all_constraints: List[Dict[str, Any]]
    all_indexes: List[Dict[str, Any]]
    total_nodes: int
    total_relationships: int
    database_info: Dict[str, Any]
    acquisition_timestamp: str
    acquisition_duration_seconds: float
    
    # ______________________
    # Post-Initialization (validation + derived fields)
    #
    # -------------------------------------------------------------- __post_init__()
    def __post_init__(self) -> None:
        """Normalize and initialize internal collections and metadata."""
        # Ensure global_property_keys is a set
        if isinstance(self.global_property_keys, list):
            self.global_property_keys = set(self.global_property_keys)
        
        # Initialize empty collections if None
        if self.nodes is None:
            self.nodes = {}
        if self.relationships is None:
            self.relationships = {}
        if self.all_constraints is None:
            self.all_constraints = []
        if self.all_indexes is None:
            self.all_indexes = []
        if self.database_info is None:
            self.database_info = {}
        if self.acquisition_timestamp is None:
            self.acquisition_timestamp = datetime.utcnow().isoformat()
    # -------------------------------------------------------------- end __post_init__()

    # ______________________
    # Constructor 
    # 

    # ______________________
    # Destructor (Use only if absolutely necessary for external resource cleanup)
    #

    # ______________________
    # -- Embedded Classes --
    #

    # ______________________
    # Getters (Property decorators are often preferred for simple getters)
    #
    # -------------------------------------------------------------- get_node_labels()
    def get_node_labels(self) -> List[str]:
        """Get a list of all node labels.

        Returns:
            A list of label names present in this schema snapshot.
        """
        return list(self.nodes.keys())

    # -------------------------------------------------------------- get_relationship_types()
    def get_relationship_types(self) -> List[str]:
        """Get a list of all relationship types.

        Returns:
            A list of relationship type names present in this schema snapshot.
        """
        return list(self.relationships.keys())

    # -------------------------------------------------------------- get_all_properties()
    def get_all_properties(self) -> List[str]:
        """Get a sorted list of all property keys (global).

        Returns:
            Sorted property key names aggregated across the graph.
        """
        return sorted(list(self.global_property_keys))

    # -------------------------------------------------------------- get_node_info()
    def get_node_info(self, label: str) -> Optional[ComprehensiveNodeInfo]:
        """Get comprehensive information about a specific node label.

        Args:
            label: Node label name.

        Returns:
            A ComprehensiveNodeInfo instance if the label exists, otherwise None.
        """
        return self.nodes.get(label)

    # -------------------------------------------------------------- get_relationship_info()
    def get_relationship_info(self, rel_type: str) -> Optional[ComprehensiveRelationshipInfo]:
        """Get comprehensive information about a specific relationship type.

        Args:
            rel_type: Relationship type name.

        Returns:
            A ComprehensiveRelationshipInfo instance if the type exists, otherwise None.
        """
        return self.relationships.get(rel_type)

    # ______________________
    # Setters / Mutators
    #

    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #

    # =========================================================================
    # Functionality: e.g. Data Processing Utilities
    # =========================================================================

    # ______________________
    # Class Information Methods (Optional, but highly recommended) 
    #

    # -------------------------------------------------------------- to_dict()
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary suitable for JSON encoding."""
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "relationships": {k: v.to_dict() for k, v in self.relationships.items()},
            "global_property_keys": list(self.global_property_keys),
            "all_constraints": self.all_constraints,
            "all_indexes": self.all_indexes,
            "total_nodes": self.total_nodes,
            "total_relationships": self.total_relationships,
            "database_info": self.database_info,
            "acquisition_timestamp": self.acquisition_timestamp,
            "acquisition_duration_seconds": self.acquisition_duration_seconds,
        }

    # -------------------------------------------------------------- end to_dict()

    # -------------------------------------------------------------- from_dict()
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompleteKGSchema":
        """Deserialize from a dictionary into a complete schema object."""
        nodes = {k: ComprehensiveNodeInfo.from_dict(v) for k, v in data.get('nodes', {}).items()}
        relationships = {
            k: ComprehensiveRelationshipInfo.from_dict(v)
            for k, v in data.get("relationships", {}).items()
        }
        
        return cls(
            nodes=nodes,
            relationships=relationships,
            global_property_keys=set(data.get("global_property_keys", [])),
            all_constraints=data.get("all_constraints", []),
            all_indexes=data.get("all_indexes", []),
            total_nodes=data.get("total_nodes", 0),
            total_relationships=data.get("total_relationships", 0),
            database_info=data.get("database_info", {}),
            acquisition_timestamp=data.get(
                "acquisition_timestamp", datetime.utcnow().isoformat()
            ),
            acquisition_duration_seconds=data.get("acquisition_duration_seconds", 0.0),
        )

    # -------------------------------------------------------------- end from_dict()

    # -------------------------------------------------------------- to_json()
    def to_json(self) -> str:
        """Convert to a pretty-printed JSON string representation."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    # -------------------------------------------------------------- end to_json()

    # -------------------------------------------------------------- from_json()
    @classmethod
    def from_json(cls, json_str: str) -> "CompleteKGSchema":
        """Create a schema object from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    # -------------------------------------------------------------- end from_json()

# ------------------------------------------------------------------------- end class CompleteKGSchema

# ------------------------------------------------------------------------- class StructuralSummary
@dataclass
class StructuralSummary:
    """Lightweight structural summary optimized for LLM consumption.

    Attributes:
        metadata: Generation metadata (timestamps, counts, versions, etc.).
        node_labels: List of dicts describing node labels.
        relationship_types: List of dicts describing relationship types.
        node_property_types: Node property type rows (label, property, types, mandatory).
        relationship_property_types: Relationship property type rows.
        last_loaded: Timestamp when loaded into memory (set if absent on init).
    """

    # ______________________
    #  Class Variable (excluded from dataclass constructor/compare via ClassVar)
    #

    # ______________________
    #  Instance Fields
    #
    metadata: Dict[str, Any]
    node_labels: List[Dict[str, Any]]
    relationship_types: List[Dict[str, Any]]
    node_property_types: List[Dict[str, Any]]
    relationship_property_types: List[Dict[str, Any]]
    last_loaded: Optional[datetime] = None
    
    # ______________________
    # Post-Initialization (validation + derived fields)
    #
    # -------------------------------------------------------------- __post_init__()
    def __post_init__(self) -> None:
        """Set load-time metadata if not already provided."""
        if self.last_loaded is None:
            self.last_loaded = datetime.utcnow()
    # -------------------------------------------------------------- end __post_init__()

    # ______________________
    # Constructor 
    # 

    # ______________________
    # Destructor (Use only if absolutely necessary for external resource cleanup)
    #

    # ______________________
    # -- Embedded Classes --
    #

    # ______________________
    # Getters (Property decorators are often preferred for simple getters)
    #

    # ______________________
    # Setters / Mutators
    #

    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #

    # =========================================================================
    # Functionality: e.g. Data Processing Utilities
    # =========================================================================

    # ______________________
    # Class Information Methods (Optional, but highly recommended) 
    #

    # -------------------------------------------------------------- from_dict()
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StructuralSummary":
        """Create from dictionary (typically loaded from JSON)."""
        return cls(
            metadata=data.get("metadata", {}),
            node_labels=data.get("node_labels", []),
            relationship_types=data.get("relationship_types", []),
            node_property_types=data.get("node_property_types", []),
            relationship_property_types=data.get("relationship_property_types", []),
        )
    # -------------------------------------------------------------- end from_dict()

    # -------------------------------------------------------------- to_dict()
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metadata": self.metadata,
            "node_labels": self.node_labels,
            "relationship_types": self.relationship_types,
            "node_property_types": self.node_property_types,
            "relationship_property_types": self.relationship_property_types,
        }
    # -------------------------------------------------------------- end to_dict()

    # -------------------------------------------------------------- get_node_labels_list()
    def get_node_labels_list(self) -> List[str]:
        """Get a simple list of node label names."""
        return [node["label"] for node in self.node_labels if "label" in node]

    # -------------------------------------------------------------- get_relationship_types_list()
    def get_relationship_types_list(self) -> List[str]:
        """Get a simple list of relationship type names."""
        return [rel["type"] for rel in self.relationship_types if "type" in rel]

    # -------------------------------------------------------------- get_llm_friendly_summary()
    def get_llm_friendly_summary(self) -> str:
        """Generate an LLM-friendly, human-readable text summary of the schema."""
        lines = ["=== Knowledge Graph Schema Summary ==="]
        
        # Metadata
        if self.metadata:
            lines.append(f"Generated: {self.metadata.get('generated_at', 'Unknown')}")
            counts = self.metadata.get('counts', {})
            lines.append(f"Node Labels: {counts.get('node_labels', 0)}")
            lines.append(f"Relationship Types: {counts.get('relationship_types', 0)}")
            lines.append("")
        
        # Node types
        lines.append("Node Types:")
        for node in self.node_labels:
            lines.append(f"  • {node.get('label', 'Unknown')}")
        
        lines.append("")

        # Relationship types
        lines.append("Relationship Types:")
        for rel in self.relationship_types:
            lines.append(f"  • {rel.get('type', 'Unknown')}")

        lines.append("")
        
        # Key properties by node type
        lines.append("Key Properties by Node:")
        node_props = {}
        for prop in self.node_property_types:
            labels = prop.get('labels', [])
            prop_name = prop.get('property', 'unknown')
            mandatory = prop.get('mandatory', False)
            prop_types = prop.get('types', [])
            
            for label in labels:
                if label not in node_props:
                    node_props[label] = []
                node_props[label].append(f"{prop_name} ({'/'.join(prop_types)}){'*' if mandatory else ''}")
        
        for label, props in node_props.items():
            lines.append(f"  {label}: {', '.join(props)}")
        
        return "\n".join(lines)
    # -------------------------------------------------------------- end get_llm_friendly_summary()

# ------------------------------------------------------------------------- end class StructuralSummary


# ------------------------------------------------------------------------- class SchemaCache
@dataclass
class SchemaCache:
    """In-memory cache for schema information optimized for static KG databases.

    Attributes:
        schema: Current comprehensive schema snapshot (if available).
        structural_summary: Current structural summary (if available).
        last_updated: Timestamp indicating last update (schema or summary).
        cache_ttl_seconds: TTL for dynamic mode; ignored in static mode.
        static_mode: If True, treat cache as always valid once populated.
    """

    # ______________________
    #  Class Variable (excluded from dataclass constructor/compare via ClassVar)
    #

    # ______________________
    #  Instance Fields
    #
    schema: Optional[CompleteKGSchema] = None
    structural_summary: Optional[StructuralSummary] = None
    last_updated: Optional[datetime] = None
    cache_ttl_seconds: int = 86400 * 365  # 1 year (effectively infinite for static KG)
    static_mode: bool = True  # Optimized for static knowledge graphs
    
    # ______________________
    # Post-Initialization (validation + derived fields)
    #
    # -------------------------------------------------------------- is_valid()
    def is_valid(self) -> bool:
        """Check whether the cached schema is still valid.

        Returns:
            True if valid. In static mode, valid once populated; in dynamic mode,
            compares against the configured TTL.
        """
        if self.schema is None or self.last_updated is None:
            return False
        
        # For static mode, cache is always valid once set (until manually cleared)
        if self.static_mode:
            return True
            
        # Fallback to TTL check for dynamic mode
        age_seconds = (datetime.utcnow() - self.last_updated).total_seconds()
        return age_seconds < self.cache_ttl_seconds
    
    # -------------------------------------------------------------- update()
    def update(self, schema: CompleteKGSchema) -> None:
        """Update cache with a new comprehensive schema snapshot."""
        self.schema = schema
        self.last_updated = datetime.utcnow()
    
    # -------------------------------------------------------------- update_structural_summary()
    def update_structural_summary(self, structural_summary: StructuralSummary) -> None:
        """Update cache with a new structural summary."""
        self.structural_summary = structural_summary
        if self.last_updated is None:
            self.last_updated = datetime.utcnow()
    
    # -------------------------------------------------------------- clear()
    def clear(self) -> None:
        """Clear both schema and structural summary from the cache."""
        self.schema = None
        self.structural_summary = None
        self.last_updated = None
    
    # -------------------------------------------------------------- clear_structural_summary()
    def clear_structural_summary(self) -> None:
        """Clear only the structural summary portion of the cache."""
        self.structural_summary = None
    
    # -------------------------------------------------------------- get_age_seconds()
    def get_age_seconds(self) -> Optional[float]:
        """Get age of cached data in seconds.

        Returns:
            Age in seconds since last update, or None if never updated.
        """
        if self.last_updated is None:
            return None
        return (datetime.utcnow() - self.last_updated).total_seconds()

# ------------------------------------------------------------------------- end class SchemaCache

# __________________________________________________________________________
# End of File
