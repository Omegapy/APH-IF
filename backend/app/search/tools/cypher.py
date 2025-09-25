# -------------------------------------------------------------------------
# File: cypher.py
# Author: Alexander Ricciardi
# Date: 2025-09-15
# [File Path] backend/app/search/tools/cypher.py
# ------------------------------------------------------------------------
# Project:
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
#   LLM structural Cypher tools that expose schema data models and a thin public wrapper to
#   delegate NL→Cypher generation, validation, and execution to the structural engine. Retains a
#   deprecated schema acquirer for historical context and utilities to format schema summaries.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Class (dataclass): NodeLabelInfo
# - Class (dataclass): RelationshipTypeInfo
# - Class (dataclass): KnowledgeGraphSchema
# - Class: KGSchemaAcquirer (deprecated direct DB access)
# - Function: format_schema_summary
# - Function: get_kg_schema_acquirer
# - Function: query_knowledge_graph_llm_structural_detailed
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: logging, time, asyncio, dataclasses, datetime, typing
# - Third-Party: (none)
# - Local Project Modules: llm_structural_cypher (loaded at call time)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# - The public async function `query_knowledge_graph_llm_structural_detailed` is used by higher-level
#   tools/services to perform LLM-guided traversal queries with schema-aware prompting and
#   validation, returning standardized responses for the backend API.
# - Data models and formatting helpers support diagnostics and developer tooling.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
LLM structural Cypher tools for APH-IF backend.

Provides schema data models and a thin public wrapper that delegates NL→Cypher generation and
execution to the LLM Structural Cypher engine. Includes a deprecated schema acquirer retained for
historical context, plus helpers for formatting schema summaries.
"""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import logging
import time
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# ____________________________________________________________________________
# Class Definitions

# =========================================================================
# Schema Data Structures
# =========================================================================

# ------------------------------------------------------------------------- class NodeLabelInfo
@dataclass
class NodeLabelInfo:
    """Information about a node label in the knowledge graph.

    Attributes:
        label: Node label name.
        count: Optional number of nodes with this label.
        sample_properties: Example property keys observed on nodes with this label.
    """
    label: str
    count: Optional[int] = None
    sample_properties: List[str] = None

    # --------------------------------------------------------------------------------- __post_init__()
    def __post_init__(self) -> None:
        if self.sample_properties is None:
            self.sample_properties = []
    # --------------------------------------------------------------------------------- end __post_init__()
# ------------------------------------------------------------------------- end class NodeLabelInfo

# ------------------------------------------------------------------------- class RelationshipTypeInfo
@dataclass
class RelationshipTypeInfo:
    """Information about a relationship type in the knowledge graph.

    Attributes:
        relationship_type: Relationship type name.
        count: Optional number of relationships of this type.
        common_patterns: Common source→target label patterns for this relationship type.
    """
    relationship_type: str
    count: Optional[int] = None
    common_patterns: List[str] = None

    # --------------------------------------------------------------------------------- __post_init__()
    def __post_init__(self) -> None:
        if self.common_patterns is None:
            self.common_patterns = []
    # --------------------------------------------------------------------------------- end __post_init__()
# ------------------------------------------------------------------------- end class RelationshipTypeInfo

# ------------------------------------------------------------------------- class KnowledgeGraphSchema
@dataclass
class KnowledgeGraphSchema:
    """Complete knowledge graph schema information.

    Attributes:
        node_labels: Label metadata including counts and sample properties.
        relationship_types: Relationship type metadata and common patterns.
        property_keys: All known property keys across nodes/relationships.
        constraints: Constraint metadata as returned by the database layer.
        indexes: Index metadata as returned by the database layer.
        total_nodes: Optional total node count in the graph.
        total_relationships: Optional total relationship count in the graph.
        retrieved_at: ISO timestamp indicating when the snapshot was taken.
    """
    node_labels: List[NodeLabelInfo]
    relationship_types: List[RelationshipTypeInfo]
    property_keys: List[str]
    constraints: List[Dict[str, Any]]
    indexes: List[Dict[str, Any]]
    total_nodes: Optional[int] = None
    total_relationships: Optional[int] = None
    retrieved_at: Optional[str] = None

    # --------------------------------------------------------------------------------- __post_init__()
    def __post_init__(self) -> None:
        if self.retrieved_at is None:
            self.retrieved_at = datetime.utcnow().isoformat()
    # --------------------------------------------------------------------------------- end __post_init__()
# ------------------------------------------------------------------------- end class KnowledgeGraphSchema

# =========================================================================
# Schema Acquisition Functions
# =========================================================================

# ------------------------------------------------------------------------- class KGSchemaAcquirer
class KGSchemaAcquirer:
    """Knowledge Graph Schema Acquisition tool for APH-IF.

    Note:
        Direct database access in this class is retained for historical reasons and is marked
        deprecated in this codebase. Prefer using the schema manager gateway (`get_schema_manager`)
        elsewhere in the backend. This class remains to document expected shapes and for
        compatibility of local tooling/tests that may reference it.
    """
    
    # ______________________
    # Constructor 
    # 
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self, database: Any | None = None) -> None:
        """Initialize the schema acquirer.

        Args:
            database: Optional database/gateway object exposing query helpers. This argument is
                deprecated here; the recommended path is via the schema manager gateway.
        """
        self.database = database
        self.logger = logging.getLogger(__name__)
    # --------------------------------------------------------------------------------- end __init__()
        
    # -------------------------------------------------------------- get_comprehensive_schema()
    def get_comprehensive_schema(self) -> KnowledgeGraphSchema:
        """Get comprehensive schema information from the knowledge graph.

        Returns:
            A snapshot of node labels, relationship types, keys, constraints, indexes, and
            aggregate counts where available.

        Raises:
            ImportError: When the acquirer is used without a database (deprecated path).
            Exception: For errors encountered during query execution or processing.
        """
        try:
            if not self.database:
                # DEPRECATED: Direct database access removed to enforce schema boundary
                raise ImportError("KGSchemaAcquirer is deprecated. Use get_schema_manager() instead.")
            
            self.logger.info("Acquiring comprehensive KG schema information...")
            
            # Get basic schema info from database layer
            basic_schema = self.database.get_schema_info()
            
            # Process node labels with counts and sample properties
            node_labels = []
            for label in basic_schema.get("node_labels", []):
                node_info = self._get_node_label_details(label)
                node_labels.append(node_info)
            
            # Process relationship types with patterns
            relationship_types = []
            for rel_type in basic_schema.get("relationship_types", []):
                rel_info = self._get_relationship_type_details(rel_type)
                relationship_types.append(rel_info)
            
            # Get total counts
            total_nodes = self._get_total_node_count()
            total_relationships = self._get_total_relationship_count()
            
            schema = KnowledgeGraphSchema(
                node_labels=node_labels,
                relationship_types=relationship_types,
                property_keys=basic_schema.get("property_keys", []),
                constraints=basic_schema.get("constraints", []),
                indexes=basic_schema.get("indexes", []),
                total_nodes=total_nodes,
                total_relationships=total_relationships
            )
            
            self.logger.info(f"Schema acquired: {len(schema.node_labels)} labels, "
                           f"{len(schema.relationship_types)} relationship types")
            
            return schema
            
        except Exception as e:
            self.logger.error(f"Error acquiring KG schema: {e}")
            raise
    # -------------------------------------------------------------- end get_comprehensive_schema()
    
    # -------------------------------------------------------------- _get_node_label_details()
    def _get_node_label_details(self, label: str) -> NodeLabelInfo:
        """Get details for a specific node label.

        Args:
            label: Node label name.

        Returns:
            NodeLabelInfo with count and sample property keys (best-effort).
        """
        try:
            # Get node count for this label
            count_query = f"MATCH (n:{label}) RETURN count(n) as count"
            count_result = self.database.execute_query(count_query)
            count = count_result[0]["count"] if count_result else 0
            
            # Get sample properties (limit to avoid performance issues)
            if count > 0:
                props_query = f"MATCH (n:{label}) RETURN keys(n) as props LIMIT 10"
                props_result = self.database.execute_query(props_query)
                
                # Collect unique properties
                all_props = set()
                for record in props_result:
                    all_props.update(record["props"])
                
                sample_properties = sorted(list(all_props))
            else:
                sample_properties = []
            
            return NodeLabelInfo(
                label=label,
                count=count,
                sample_properties=sample_properties
            )
            
        except Exception as e:
            self.logger.warning(f"Error getting details for node label {label}: {e}")
            return NodeLabelInfo(label=label, count=0, sample_properties=[])
    # -------------------------------------------------------------- end _get_node_label_details()
    
    # -------------------------------------------------------------- _get_relationship_type_details()
    def _get_relationship_type_details(self, rel_type: str) -> RelationshipTypeInfo:
        """Get details for a specific relationship type.

        Args:
            rel_type: Relationship type name.

        Returns:
            RelationshipTypeInfo with counts and frequent source→target patterns (best-effort).
        """
        try:
            # Get relationship count for this type
            count_query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
            count_result = self.database.execute_query(count_query)
            count = count_result[0]["count"] if count_result else 0
            
            # Get common patterns (source->target label combinations)
            common_patterns = []
            if count > 0:
                patterns_query = f"""
                MATCH (source)-[r:{rel_type}]->(target)
                RETURN DISTINCT labels(source)[0] as source_label, 
                       labels(target)[0] as target_label,
                       count(*) as pattern_count
                ORDER BY pattern_count DESC
                LIMIT 5
                """
                patterns_result = self.database.execute_query(patterns_query)
                
                for record in patterns_result:
                    pattern = f"({record['source_label']})-[:{rel_type}]->({record['target_label']})"
                    common_patterns.append(pattern)
            
            return RelationshipTypeInfo(
                relationship_type=rel_type,
                count=count,
                common_patterns=common_patterns
            )
            
        except Exception as e:
            self.logger.warning(f"Error getting details for relationship type {rel_type}: {e}")
            return RelationshipTypeInfo(relationship_type=rel_type, count=0, common_patterns=[])
    # -------------------------------------------------------------- end _get_relationship_type_details()
    
    # -------------------------------------------------------------- _get_total_node_count()
    def _get_total_node_count(self) -> Optional[int]:
        """Get total number of nodes in the graph.

        Returns:
            Total node count if available; otherwise None.
        """
        try:
            result = self.database.execute_query("MATCH (n) RETURN count(n) as total")
            return result[0]["total"] if result else None
        except Exception as e:
            self.logger.warning(f"Error getting total node count: {e}")
            return None
    # -------------------------------------------------------------- end _get_total_node_count()
    
    # -------------------------------------------------------------- _get_total_relationship_count()
    def _get_total_relationship_count(self) -> Optional[int]:
        """Get total number of relationships in the graph.

        Returns:
            Total relationship count if available; otherwise None.
        """
        try:
            result = self.database.execute_query("MATCH ()-[r]->() RETURN count(r) as total")
            return result[0]["total"] if result else None
        except Exception as e:
            self.logger.warning(f"Error getting total relationship count: {e}")
            return None
    # -------------------------------------------------------------- end _get_total_relationship_count()
    
    # -------------------------------------------------------------- get_apoc_schema_info()
    def get_apoc_schema_info(self) -> Dict[str, Any]:
        """Get schema information using APOC procedures if available.

        Returns:
            Mapping with fields: `available` flag, optional `meta_schema`, optional
            `node_constraints`, optional `relationship_constraints`, and `error` text when
            collection fails or APOC is unavailable.
        """
        apoc_info = {
            "available": False,
            "meta_schema": None,
            "node_constraints": None,
            "relationship_constraints": None,
            "error": None
        }
        
        try:
            # Test if APOC is available and get meta schema
            result = self.database.execute_query("CALL apoc.meta.schema()")
            apoc_info["available"] = True
            apoc_info["meta_schema"] = [dict(record) for record in result]
            
            # Get node constraints/indexes
            try:
                nodes_result = self.database.execute_query("CALL apoc.schema.nodes()")
                apoc_info["node_constraints"] = [dict(record) for record in nodes_result]
            except Exception:
                pass
            
            # Get relationship constraints/indexes
            try:
                rels_result = self.database.execute_query("CALL apoc.schema.relationships()")
                apoc_info["relationship_constraints"] = [dict(record) for record in rels_result]
            except Exception:
                pass
                
        except Exception as e:
            apoc_info["error"] = str(e)
            self.logger.debug(f"APOC not available or error: {e}")
        
        return apoc_info
# ------------------------------------------------------------------------- end class KGSchemaAcquirer

# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# Utility Functions for Schema Information
# =========================================================================

# --------------------------------------------------------------------------------- format_schema_summary()
def format_schema_summary(schema: KnowledgeGraphSchema) -> str:
    """Format a knowledge graph schema into a readable summary.

    Args:
        schema: Complete schema snapshot to render.

    Returns:
        Multiline text summarizing counts, label/type highlights, and properties.
    """
    summary_lines = [
        f"=== Knowledge Graph Schema Summary ===",
        f"Retrieved at: {schema.retrieved_at}",
        f"",
        f"📊 Overview:",
        f"  • Total Nodes: {schema.total_nodes or 'Unknown'}",
        f"  • Total Relationships: {schema.total_relationships or 'Unknown'}",
        f"  • Node Labels: {len(schema.node_labels)}",
        f"  • Relationship Types: {len(schema.relationship_types)}",
        f"  • Property Keys: {len(schema.property_keys)}",
        f"  • Constraints: {len(schema.constraints)}",
        f"  • Indexes: {len(schema.indexes)}",
        f"",
        f"🏷️ Node Labels:",
    ]
    
    for node in schema.node_labels[:10]:  # Limit to first 10
        summary_lines.append(f"  • {node.label}: {node.count or 'Unknown'} nodes")
        if node.sample_properties:
            props = ", ".join(node.sample_properties[:5])
            if len(node.sample_properties) > 5:
                props += f" ... (+{len(node.sample_properties) - 5} more)"
            summary_lines.append(f"    Properties: {props}")
    
    if len(schema.node_labels) > 10:
        summary_lines.append(f"  ... and {len(schema.node_labels) - 10} more labels")
    
    summary_lines.extend([
        f"",
        f"🔗 Relationship Types:",
    ])
    
    for rel in schema.relationship_types[:10]:  # Limit to first 10
        summary_lines.append(f"  • {rel.relationship_type}: {rel.count or 'Unknown'} relationships")
        if rel.common_patterns:
            patterns = ", ".join(rel.common_patterns[:3])
            if len(rel.common_patterns) > 3:
                patterns += f" ... (+{len(rel.common_patterns) - 3} more)"
            summary_lines.append(f"    Common patterns: {patterns}")
    
    if len(schema.relationship_types) > 10:
        summary_lines.append(f"  ... and {len(schema.relationship_types) - 10} more types")
    
    return "\n".join(summary_lines)
# --------------------------------------------------------------------------------- end format_schema_summary()

# --------------------------------------------------------------------------------- get_kg_schema_acquirer()
def get_kg_schema_acquirer(database: Any | None = None) -> KGSchemaAcquirer:
    """Create a `KGSchemaAcquirer` instance.

    Args:
        database: Optional database/gateway object for direct queries. Deprecated in this
            codebase in favor of the schema manager gateway.

    Returns:
        A `KGSchemaAcquirer` configured with the provided database object.
    """
    return KGSchemaAcquirer(database=database)
# --------------------------------------------------------------------------------- end get_kg_schema_acquirer()

# =========================================================================
# LLM Structural Cypher Integration
# =========================================================================

# --------------------------------------------------------------------------------- query_knowledge_graph_llm_structural_detailed()
async def query_knowledge_graph_llm_structural_detailed(
    user_query: str,
    max_results: int = 50,
) -> Dict[str, Any]:
    """Run NL→Cypher generation, validation, and read-only execution.

    Uses structural schema summaries for token-aware prompting and validates generated Cypher
    against the full cached schema for safety and correctness. Delegates to the LLM Structural
    Cypher engine and returns a standardized response structure.

    Args:
        user_query: Natural language query provided by the caller.
        max_results: Maximum number of rows to return from the traversal.

    Returns:
        A standardized response with fields such as `answer`, `cypher_query`, `confidence`,
        `response_time_ms`, and `metadata`. On errors, returns a response with `error` context
        rather than raising.
    """
    try:
        # Import LLM structural engine
        from .llm_structural_cypher import get_llm_structural_cypher_engine
        
        engine = get_llm_structural_cypher_engine()
        
        # Execute complete LLM structural query flow
        result = await engine.query_knowledge_graph_llm_structural(
            user_query=user_query,
            max_results=max_results
        )
        
        return result
        
    except ImportError as e:
        logger.error(f"LLM structural Cypher engine not available: {e}")
        return {
            "answer": "LLM structural Cypher generation not available - missing dependencies",
            "cypher_query": "",
            "confidence": 0.0,
            "response_time_ms": 0,
            "error": f"Import error: {str(e)}",
            "metadata": {"search_method": "llm_structural_cypher_unavailable", "error": str(e)}
        }
    except Exception as e:
        logger.error(f"LLM structural Cypher query failed: {e}")
        return {
            "answer": f"LLM structural Cypher query failed: {str(e)}",
            "cypher_query": "",
            "confidence": 0.0,
            "response_time_ms": 0,
            "error": str(e),
            "metadata": {"search_method": "llm_structural_cypher_error", "primary_error": str(e)}
        }
    # --------------------------------------------------------------------------------- end query_knowledge_graph_llm_structural_detailed()

# __________________________________________________________________________
# End of File


