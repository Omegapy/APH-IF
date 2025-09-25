# -------------------------------------------------------------------------
# File: schema_acquirer.py
# Author: Alexander Ricciardi
# Date:
# [File Path] backend/app/schema/schema_acquirer.py
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
#   Comprehensive schema acquisition utilities used by the schema manager to
#   build a full knowledge-graph schema snapshot. Intended for operator- or
#   admin-driven runs; not invoked per-request in normal API paths.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Class: SchemaAcquirer
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: logging, time, typing
# - Third-Party: (none)
# - Local Project Modules: schema_models, core.database (get_database)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Created by SchemaManager when refreshing schema; not used directly by
# endpoints. Runs multi-query introspection routines against Neo4j.
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent
# Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""
Comprehensive Schema Acquisition for APH-IF Knowledge Graph

This module provides complete schema acquisition without performance limitations.
It's designed to be run by developers/administrators when needed, not during
regular API operations.
"""

import logging
import time
from typing import Any, Dict, List

from .schema_models import CompleteKGSchema, ComprehensiveNodeInfo, ComprehensiveRelationshipInfo

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------- class SchemaAcquirer
class SchemaAcquirer:
    """Comprehensive knowledge graph schema acquisition tool.

    Responsibilities:
        - Query database for labels, relationship types, global properties
        - Derive per-label and per-relationship statistics and samples
        - Aggregate results into a CompleteKGSchema instance

    Attributes:
        database: Database adapter providing an `execute_query` method.
        logger: Per-instance logger.
    """
    
    # ______________________
    # Constructor 
    # 
    # -------------------------------------------------------------- __init__()
    def __init__(self, database=None):
        """Initialize with database connection."""
        self.database = database
        self.logger = logging.getLogger(__name__)
        
        if not self.database:
            from ..core.database import get_database
            self.database = get_database()
    # -------------------------------------------------------------- end __init__()
    
    # ______________________
    # Getters (Property decorators are often preferred for simple getters)
    #

    # -------------------------------------------------------------- acquire_complete_schema()
    def acquire_complete_schema(
        self,
        include_samples: bool = True,
        sample_size: int = 100,
    ) -> CompleteKGSchema:
        """
        Acquire complete schema information without performance limitations.
        
        Args:
            include_samples: Whether to include sample nodes/relationships.
            sample_size: Number of samples to collect for each type.
            
        Returns:
            Complete knowledge graph schema.
        """
        start_time = time.time()
        self.logger.info("🔍 Starting comprehensive schema acquisition...")
        
        # Get basic schema information
        basic_schema = self.database.get_schema_info()
        
        # Get comprehensive node information
        nodes = {}
        for label in basic_schema.get("node_labels", []):
            self.logger.info(f"  📊 Analyzing node label: {label}")
            nodes[label] = self._acquire_complete_node_info(label, include_samples, sample_size)
        
        # Get comprehensive relationship information
        relationships = {}
        for rel_type in basic_schema.get("relationship_types", []):
            self.logger.info(f"  🔗 Analyzing relationship type: {rel_type}")
            relationships[rel_type] = self._acquire_complete_relationship_info(rel_type, include_samples, sample_size)
        
        # Get global statistics
        total_nodes = self._get_total_node_count()
        total_relationships = self._get_total_relationship_count()
        
        # Collect all property keys
        global_property_keys = set(basic_schema.get("property_keys", []))
        
        # Get database information
        database_info = self._get_database_info()
        
        acquisition_duration = time.time() - start_time
        
        schema = CompleteKGSchema(
            nodes=nodes,
            relationships=relationships,
            global_property_keys=global_property_keys,
            all_constraints=basic_schema.get("constraints", []),
            all_indexes=basic_schema.get("indexes", []),
            total_nodes=total_nodes,
            total_relationships=total_relationships,
            database_info=database_info,
            acquisition_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            acquisition_duration_seconds=acquisition_duration
        )
        
        self.logger.info(f"✅ Schema acquisition completed in {acquisition_duration:.2f}s")
        self.logger.info(f"   📈 Found {len(nodes)} node types, {len(relationships)} relationship types")
        self.logger.info(f"   🔢 Total: {total_nodes:,} nodes, {total_relationships:,} relationships")
        
        return schema
    # -------------------------------------------------------------- end acquire_complete_schema()
    
    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #

    # -------------------------------------------------------------- _acquire_complete_node_info()
    def _acquire_complete_node_info(self, label: str, include_samples: bool, sample_size: int) -> ComprehensiveNodeInfo:
        """Get comprehensive information about a node label."""
        
        # Get total count
        count_query = f"MATCH (n:{label}) RETURN count(n) as count"
        count_result = self.database.execute_query(count_query)
        total_count = count_result[0]["count"] if count_result else 0
        
        if total_count == 0:
            return ComprehensiveNodeInfo(
                label=label,
                total_count=0,
                all_properties=set(),
                property_statistics={},
                sample_nodes=[],
                constraints=[],
                indexes=[]
            )
        
        # Get ALL properties (no limits)
        all_props_query = f"""
        MATCH (n:{label})
        UNWIND keys(n) as prop
        RETURN DISTINCT prop
        ORDER BY prop
        """
        props_result = self.database.execute_query(all_props_query)
        all_properties = set(record["prop"] for record in props_result)
        
        # Get property statistics
        property_statistics = {}
        for prop in all_properties:
            prop_stats = self._get_property_statistics(label, prop)
            property_statistics[prop] = prop_stats
        
        # Get sample nodes if requested
        sample_nodes = []
        if include_samples and total_count > 0:
            sample_query = f"""
            MATCH (n:{label})
            RETURN n
            LIMIT {sample_size}
            """
            sample_result = self.database.execute_query(sample_query)
            sample_nodes = [dict(record["n"]) for record in sample_result]
        
        # Get constraints specific to this label
        constraints = self._get_node_constraints(label)
        
        # Get indexes specific to this label  
        indexes = self._get_node_indexes(label)
        
        return ComprehensiveNodeInfo(
            label=label,
            total_count=total_count,
            all_properties=all_properties,
            property_statistics=property_statistics,
            sample_nodes=sample_nodes,
            constraints=constraints,
            indexes=indexes
        )
    # -------------------------------------------------------------- end _acquire_complete_node_info()

    # -------------------------------------------------------------- _acquire_complete_relationship_info()
    def _acquire_complete_relationship_info(self, rel_type: str, include_samples: bool, sample_size: int) -> ComprehensiveRelationshipInfo:
        """Get comprehensive information about a relationship type."""
        
        # Get total count
        count_query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
        count_result = self.database.execute_query(count_query)
        total_count = count_result[0]["count"] if count_result else 0
        
        if total_count == 0:
            return ComprehensiveRelationshipInfo(
                relationship_type=rel_type,
                total_count=0,
                all_patterns=[],
                pattern_statistics={},
                all_properties=set(),
                property_statistics={},
                sample_relationships=[],
                constraints=[],
                indexes=[]
            )
        
        # Get ALL relationship patterns (no limits)
        patterns_query = f"""
        MATCH (source)-[r:{rel_type}]->(target)
        WITH labels(source) as source_labels, labels(target) as target_labels, count(*) as count
        RETURN source_labels[0] as source_label, 
               target_labels[0] as target_label,
               count
        ORDER BY count DESC
        """
        patterns_result = self.database.execute_query(patterns_query)
        
        all_patterns = []
        pattern_statistics = {}
        for record in patterns_result:
            pattern_info = {
                "source_label": record["source_label"],
                "target_label": record["target_label"],
                "count": record["count"]
            }
            pattern_key = f"({record['source_label']})-[:{rel_type}]->({record['target_label']})"
            all_patterns.append(pattern_info)
            pattern_statistics[pattern_key] = record["count"]
        
        # Get ALL relationship properties (no limits)
        rel_props_query = f"""
        MATCH ()-[r:{rel_type}]->()
        WHERE size(keys(r)) > 0
        UNWIND keys(r) as prop
        RETURN DISTINCT prop
        ORDER BY prop
        """
        props_result = self.database.execute_query(rel_props_query)
        all_properties = set(record["prop"] for record in props_result)
        
        # Get property statistics for relationships
        property_statistics = {}
        for prop in all_properties:
            prop_stats = self._get_relationship_property_statistics(rel_type, prop)
            property_statistics[prop] = prop_stats
        
        # Get sample relationships if requested
        sample_relationships = []
        if include_samples and total_count > 0:
            sample_query = f"""
            MATCH (source)-[r:{rel_type}]->(target)
            RETURN {{
                source: source,
                relationship: r,
                target: target,
                source_labels: labels(source),
                target_labels: labels(target)
            }} as sample
            LIMIT {sample_size}
            """
            sample_result = self.database.execute_query(sample_query)
            
            for record in sample_result:
                sample = record["sample"]
                sample_relationships.append({
                    "source": dict(sample["source"]),
                    "relationship": dict(sample["relationship"]),
                    "target": dict(sample["target"]),
                    "source_labels": sample["source_labels"],
                    "target_labels": sample["target_labels"]
                })
        
        # Get constraints and indexes for relationships
        constraints = self._get_relationship_constraints(rel_type)
        indexes = self._get_relationship_indexes(rel_type)
        
        return ComprehensiveRelationshipInfo(
            relationship_type=rel_type,
            total_count=total_count,
            all_patterns=all_patterns,
            pattern_statistics=pattern_statistics,
            all_properties=all_properties,
            property_statistics=property_statistics,
            sample_relationships=sample_relationships,
            constraints=constraints,
            indexes=indexes
        )
    # -------------------------------------------------------------- end _acquire_complete_relationship_info()
    
    # -------------------------------------------------------------- _get_property_statistics()
    def _get_property_statistics(self, label: str, property_name: str) -> Dict[str, Any]:
        """Get statistics for a specific property on a node label."""
        try:
            stats_query = f"""
            MATCH (n:{label})
            WHERE n.{property_name} IS NOT NULL
            WITH n.{property_name} as value
            RETURN 
                count(*) as count,
                count(DISTINCT value) as distinct_count,
                collect(DISTINCT type(value))[0] as data_type
            """
            result = self.database.execute_query(stats_query)
            
            if result:
                record = result[0]
                return {
                    "count": record["count"],
                    "distinct_count": record["distinct_count"], 
                    "data_type": record["data_type"],
                    "null_count": None  # Would require additional query
                }
            
        except Exception as e:
            self.logger.debug(f"Could not get property statistics for {label}.{property_name}: {e}")
        
        return {"count": 0, "distinct_count": 0, "data_type": "unknown", "null_count": 0}
    
    # -------------------------------------------------------------- _get_relationship_property_statistics()
    def _get_relationship_property_statistics(self, rel_type: str, property_name: str) -> Dict[str, Any]:
        """Get statistics for a specific property on a relationship type."""
        try:
            stats_query = f"""
            MATCH ()-[r:{rel_type}]->()
            WHERE r.{property_name} IS NOT NULL
            WITH r.{property_name} as value
            RETURN 
                count(*) as count,
                count(DISTINCT value) as distinct_count,
                collect(DISTINCT type(value))[0] as data_type
            """
            result = self.database.execute_query(stats_query)
            
            if result:
                record = result[0]
                return {
                    "count": record["count"],
                    "distinct_count": record["distinct_count"],
                    "data_type": record["data_type"]
                }
            
        except Exception as e:
            self.logger.debug(f"Could not get relationship property statistics for {rel_type}.{property_name}: {e}")
        
        return {"count": 0, "distinct_count": 0, "data_type": "unknown"}
    
    # -------------------------------------------------------------- _get_node_constraints()
    def _get_node_constraints(self, label: str) -> List[Dict[str, Any]]:
        """Get constraints specific to a node label."""
        try:
            result = self.database.execute_query("SHOW CONSTRAINTS")
            return [
                dict(record) for record in result 
                if label in record.get("labelsOrTypes", [])
            ]
        except Exception:
            return []
    
    # -------------------------------------------------------------- _get_node_indexes()
    def _get_node_indexes(self, label: str) -> List[Dict[str, Any]]:
        """Get indexes specific to a node label."""
        try:
            result = self.database.execute_query("SHOW INDEXES")
            return [
                dict(record) for record in result 
                if label in record.get("labelsOrTypes", [])
            ]
        except Exception:
            return []
    
    # -------------------------------------------------------------- _get_relationship_constraints()
    def _get_relationship_constraints(self, rel_type: str) -> List[Dict[str, Any]]:
        """Get constraints specific to a relationship type."""
        try:
            result = self.database.execute_query("SHOW CONSTRAINTS")
            return [
                dict(record) for record in result 
                if rel_type in record.get("labelsOrTypes", [])
            ]
        except Exception:
            return []
    
    # -------------------------------------------------------------- _get_relationship_indexes()
    def _get_relationship_indexes(self, rel_type: str) -> List[Dict[str, Any]]:
        """Get indexes specific to a relationship type.""" 
        try:
            result = self.database.execute_query("SHOW INDEXES")
            return [
                dict(record) for record in result 
                if rel_type in record.get("labelsOrTypes", [])
            ]
        except Exception:
            return []
    
    # -------------------------------------------------------------- _get_total_node_count()
    def _get_total_node_count(self) -> int:
        """Get total number of nodes in the graph."""
        try:
            result = self.database.execute_query("MATCH (n) RETURN count(n) as total")
            return result[0]["total"] if result else 0
        except Exception as e:
            self.logger.warning(f"Could not get total node count: {e}")
            return 0
    
    # -------------------------------------------------------------- _get_total_relationship_count()
    def _get_total_relationship_count(self) -> int:
        """Get total number of relationships in the graph."""
        try:
            result = self.database.execute_query("MATCH ()-[r]->() RETURN count(r) as total")
            return result[0]["total"] if result else 0
        except Exception as e:
            self.logger.warning(f"Could not get total relationship count: {e}")
            return 0
    
    # -------------------------------------------------------------- _get_database_info()
    def _get_database_info(self) -> Dict[str, Any]:
        """Get database information."""
        try:
            # Try to get database version and info
            info_queries = [
                "CALL dbms.components()",
                "CALL db.info()"
            ]
            
            database_info = {}
            
            for query in info_queries:
                try:
                    result = self.database.execute_query(query)
                    if "components" in query:
                        database_info["components"] = [dict(record) for record in result]
                    elif "db.info" in query:
                        database_info["database_info"] = [dict(record) for record in result]
                except Exception:
                    continue
            
            return database_info
            
        except Exception as e:
            self.logger.debug(f"Could not get database info: {e}")
            return {"error": str(e)}

# ------------------------------------------------------------------------- end class SchemaAcquirer

# __________________________________________________________________________
# End of File
