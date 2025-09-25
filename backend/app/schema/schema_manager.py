# -------------------------------------------------------------------------
# File: schema_manager.py
# Author: Alexander Ricciardi
# Date:
# [File Path] backend/app/schema/schema_manager.py
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
#   Central management for schema acquisition, caching, persistence, and
#   structural-summary export/loads. Provides a strict DB access boundary via
#   gateway methods used elsewhere in the backend.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Class: SchemaManager
# - Function: get_schema_manager
# - Function: reset_schema_manager
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: json, os, asyncio, pathlib (Path), threading (Lock), typing
# - Third-Party: (none)
# - Local Project Modules: schema_models, schema_acquirer, schema_exporter,
#   core.database (via gateway calls), core.config (settings)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Used by API endpoints to read schema and by CLI tooling to manage lifecycle.
# It is the sole component (besides core.database) permitted to reach Neo4j.
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent
# Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""
Schema Manager for APH-IF Knowledge Graph

Central management system for knowledge-graph schema with caching, on-disk
persistence, and structural-summary export, including DB gateway methods that
enforce the Neo4j access boundary outside this module.
"""

import asyncio
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from .schema_acquirer import SchemaAcquirer
from .schema_exporter import export_structural_summary_safe
from .schema_models import CompleteKGSchema, SchemaCache, StructuralSummary

logger = logging.getLogger(__name__)

# Global schema manager instance
_schema_manager_instance: Optional['SchemaManager'] = None
_schema_manager_lock = Lock()

# ------------------------------------------------------------------------- class SchemaManager
class SchemaManager:
    """Central manager for knowledge graph schema information.

    Responsibilities:
        - Acquire and cache a comprehensive schema snapshot (in-memory + disk)
        - Export and reload a lightweight structural summary for LLMs
        - Provide read-only schema accessors and summaries
        - Act as the sole DB gateway for non-core modules

    Attributes:
        cache_dir: Directory where cache artifacts are stored.
        cache_file: Path to JSON cache for the full schema.
        structural_summary_file: Path to structural summary JSON.
        cache: In-memory cache wrapper for schema/summary.
        acquirer: Acquisition helper that talks to the database layer.
        static_mode: If True, cache is effectively non-expiring until cleared.
        logger: Per-instance logger.
    """

    # ______________________
    #  Class Variable (excluded from dataclass constructor/compare via ClassVar)
    #

    # ______________________
    #  Instance Fields
    #
    
    # ______________________
    # Constructor 
    # 
    # -------------------------------------------------------------- __init__()
    def __init__(self, cache_dir: Optional[str] = None, static_mode: bool = True, cache_ttl_seconds: Optional[int] = None):
        """
        Initialize schema manager optimized for static knowledge graphs.
        
        Args:
            cache_dir: Directory to store schema cache files
            static_mode: True for static KG (cache never expires), False for dynamic KG
            cache_ttl_seconds: TTL in seconds (only used when static_mode=False)
        """
        self.cache_dir = Path(cache_dir or "schema_cache")
        self.cache_file = self.cache_dir / "kg_schema.json"
        
        # Use settings for structural summary filename to avoid drift
        try:
            from ..core.config import settings
            structural_filename = settings.schema_export_filename
        except ImportError:
            structural_filename = "kg_schema_structural_summary.json"
        
        self.structural_summary_file = self.cache_dir / structural_filename
        
        # Configure cache based on usage pattern
        if static_mode:
            self.cache = SchemaCache(static_mode=True)
            self.logger = logging.getLogger(__name__)
            self.logger.info("Schema manager initialized in STATIC mode - cache never expires automatically")
        else:
            ttl = cache_ttl_seconds or 3600  # Default 1 hour for dynamic mode
            self.cache = SchemaCache(cache_ttl_seconds=ttl, static_mode=False)
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Schema manager initialized in DYNAMIC mode - TTL: {ttl}s")
            
        self.acquirer = SchemaAcquirer()
        self._lock = Lock()
        self.static_mode = static_mode
        self._db_health_check_task = None
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(exist_ok=True)
        
        # Try to load existing cached schema and structural summary
        self._load_cached_schema()
        self._load_structural_summary()
    # -------------------------------------------------------------- end __init__()
    
    # ______________________
    # Getters (Property decorators are often preferred for simple getters)
    #
    # -------------------------------------------------------------- get_schema()
    def get_schema(self, force_refresh: bool = False) -> Optional[CompleteKGSchema]:
        """
        Get knowledge graph schema, using cache if available and valid.
        
        Args:
            force_refresh: Force refresh from database even if cache is valid
            
        Returns:
            Complete knowledge graph schema or None if unavailable
        """
        with self._lock:
            # Return cached schema if valid and not forcing refresh
            if not force_refresh and self.cache.is_valid():
                self.logger.debug("Returning cached schema")
                return self.cache.schema
            
            # Cache is invalid or refresh forced, need to acquire fresh schema
            self.logger.info("Acquiring fresh schema from database...")
            
            try:
                new_schema = self.acquirer.acquire_complete_schema(
                    include_samples=True,
                    sample_size=50  # Reasonable sample size
                )
                
                # Update cache
                self.cache.update(new_schema)
                
                # Persist to disk
                self._save_cached_schema()
                
                self.logger.info("Schema refreshed and cached successfully")
                return new_schema
                
            except Exception as e:
                self.logger.error(f"Failed to acquire schema: {e}")
                
                # If we have an old cached schema, return it as fallback
                if self.cache.schema:
                    self.logger.warning("Returning stale cached schema as fallback")
                    return self.cache.schema
                
                return None
    # -------------------------------------------------------------- end get_schema()
    
    # -------------------------------------------------------------- get_schema_async()
    async def get_schema_async(self, force_refresh: bool = False) -> Optional[CompleteKGSchema]:
        """Async version of get_schema."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_schema, force_refresh)
    # -------------------------------------------------------------- end get_schema_async()
    
    # ______________________
    # Setters / Mutators
    #
    # -------------------------------------------------------------- refresh_schema()
    def refresh_schema(self) -> bool:
        """Force refresh schema from database."""
        try:
            self.get_schema(force_refresh=True)
            return True
        except Exception as e:
            self.logger.error(f"Schema refresh failed: {e}")
            return False
    # -------------------------------------------------------------- end refresh_schema()
    
    # -------------------------------------------------------------- refresh_after_data_processing()
    def refresh_after_data_processing(self) -> bool:
        """
        Refresh schema after running data processing module.
        Convenience method for static KG workflow.
        """
        self.logger.info("🔄 Refreshing schema after data processing...")
        if self.static_mode:
            self.logger.info("Static mode: Clearing old cache and acquiring fresh schema")
            self.cache.clear()  # Clear old cache first
        
        success = self.refresh_schema()
        if success:
            self.logger.info("✅ Schema refreshed successfully after data processing")
        else:
            self.logger.error("❌ Schema refresh failed after data processing")
        return success
    # -------------------------------------------------------------- end refresh_after_data_processing()
    
    # -------------------------------------------------------------- refresh_schema_async()
    async def refresh_schema_async(self) -> bool:
        """Async version of refresh_schema."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.refresh_schema)
    # -------------------------------------------------------------- end refresh_schema_async()
    
    # -------------------------------------------------------------- get_node_labels()
    def get_node_labels(self) -> List[str]:
        """Get list of all node labels."""
        schema = self.get_schema()
        return schema.get_node_labels() if schema else []
    
    # -------------------------------------------------------------- get_relationship_types()
    def get_relationship_types(self) -> List[str]:
        """Get list of all relationship types."""
        schema = self.get_schema()
        return schema.get_relationship_types() if schema else []
    
    # -------------------------------------------------------------- get_all_properties()
    def get_all_properties(self) -> List[str]:
        """Get list of all property keys."""
        schema = self.get_schema()
        return schema.get_all_properties() if schema else []
    
    # -------------------------------------------------------------- get_node_info()
    def get_node_info(self, label: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a specific node label."""
        schema = self.get_schema()
        if not schema:
            return None
        
        node_info = schema.get_node_info(label)
        return node_info.to_dict() if node_info else None
    
    # -------------------------------------------------------------- get_relationship_info()
    def get_relationship_info(self, rel_type: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a specific relationship type."""
        schema = self.get_schema()
        if not schema:
            return None
        
        rel_info = schema.get_relationship_info(rel_type)
        return rel_info.to_dict() if rel_info else None
    
    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #

    # -------------------------------------------------------------- get_structural_summary()
    def get_structural_summary(self, force_refresh: bool = False) -> Optional[StructuralSummary]:
        """
        Get lightweight structural summary optimized for LLM consumption.
        
        Args:
            force_refresh: Force reload from disk even if cached
            
        Returns:
            Structural summary or None if unavailable
        """
        with self._lock:
            # Return cached summary if valid and not forcing refresh
            if not force_refresh and self.cache.structural_summary is not None:
                self.logger.debug("Returning cached structural summary")
                return self.cache.structural_summary
            
            # Try to load from disk
            if self.structural_summary_file.exists():
                self.logger.info("Loading structural summary from disk...")
                return self._load_structural_summary()
            else:
                self.logger.debug("No structural summary file found")
                return None
    
    # -------------------------------------------------------------- get_structural_summary_for_llm()
    def get_structural_summary_for_llm(self) -> Optional[str]:
        """
        Get structural summary formatted for LLM consumption.
        
        Returns:
            LLM-friendly text summary or None if unavailable
        """
        summary = self.get_structural_summary()
        if summary:
            return summary.get_llm_friendly_summary()
        return None
    
    # -------------------------------------------------------------- get_structural_summary_dict()
    def get_structural_summary_dict(self) -> Optional[Dict[str, Any]]:
        """
        Get structural summary as dictionary.
        
        Returns:
            Structural summary as dict or None if unavailable
        """
        summary = self.get_structural_summary()
        if summary:
            return summary.to_dict()
        return None
    
    # -------------------------------------------------------------- get_schema_summary()
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the schema."""
        schema = self.get_schema()
        if not schema:
            return {
                "available": False,
                "error": "No schema available"
            }
        
        cache_age = self.cache.get_age_seconds()
        
        return {
            "available": True,
            "node_labels_count": len(schema.nodes),
            "relationship_types_count": len(schema.relationships),
            "total_nodes": schema.total_nodes,
            "total_relationships": schema.total_relationships,
            "global_properties_count": len(schema.global_property_keys),
            "constraints_count": len(schema.all_constraints),
            "indexes_count": len(schema.all_indexes),
            "acquisition_timestamp": schema.acquisition_timestamp,
            "acquisition_duration_seconds": schema.acquisition_duration_seconds,
            "cache_age_seconds": cache_age,
            "cache_valid": self.cache.is_valid()
        }
    
    # -------------------------------------------------------------- clear_cache()
    def clear_cache(self) -> None:
        """Clear cached schema and structural summary."""
        with self._lock:
            self.cache.clear()
            if self.cache_file.exists():
                self.cache_file.unlink()
            if self.structural_summary_file.exists():
                self.structural_summary_file.unlink()
            self.logger.info("Schema cache and structural summary cleared")
    
    # -------------------------------------------------------------- clear_structural_summary_cache()
    def clear_structural_summary_cache(self) -> None:
        """Clear only the structural summary cache."""
        with self._lock:
            self.cache.clear_structural_summary()
            if self.structural_summary_file.exists():
                self.structural_summary_file.unlink()
            self.logger.info("Structural summary cache cleared")
    
    # -------------------------------------------------------------- _load_cached_schema()
    def _load_cached_schema(self) -> None:
        """Load schema from disk cache if available."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                
                # Load schema
                schema = CompleteKGSchema.from_dict(data['schema'])
                self.cache.schema = schema
                
                # Load cache metadata
                if 'cache_metadata' in data:
                    from datetime import datetime
                    timestamp_str = data['cache_metadata']['last_updated']
                    self.cache.last_updated = datetime.fromisoformat(timestamp_str)
                
                self.logger.info("Loaded cached schema from disk")
                
        except Exception as e:
            self.logger.debug(f"Could not load cached schema: {e}")
    
    # -------------------------------------------------------------- _load_structural_summary()
    def _load_structural_summary(self) -> Optional[StructuralSummary]:
        """Load structural summary from disk cache if available."""
        try:
            if self.structural_summary_file.exists():
                with open(self.structural_summary_file, 'r') as f:
                    data = json.load(f)
                
                # Create structural summary from loaded data
                structural_summary = StructuralSummary.from_dict(data)
                
                # Update cache
                self.cache.update_structural_summary(structural_summary)
                
                self.logger.info("Loaded structural summary from disk")
                return structural_summary
                
        except Exception as e:
            self.logger.debug(f"Could not load structural summary: {e}")
        
        return None
    
    # -------------------------------------------------------------- _save_cached_schema()
    def _save_cached_schema(self) -> None:
        """Save schema to disk cache."""
        try:
            if not self.cache.schema:
                return
            
            data = {
                'schema': self.cache.schema.to_dict(),
                'cache_metadata': {
                    'last_updated': self.cache.last_updated.isoformat() if self.cache.last_updated else None,
                    'cache_ttl_seconds': self.cache.cache_ttl_seconds
                }
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.debug("Saved schema to disk cache")
            
            # Export structural summary after successful schema save
            self._export_structural_summary()
            
        except Exception as e:
            self.logger.warning(f"Could not save schema cache: {e}")
    
    # =========================================================================
    # Functionality: Export helpers
    # =========================================================================

    # -------------------------------------------------------------- export_schema()
    def export_schema(self, file_path: str, format: str = 'json') -> bool:
        """
        Export schema to file.
        
        Args:
            file_path: Path to save the schema
            format: Export format ('json' or 'yaml')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            schema = self.get_schema()
            if not schema:
                return False
            
            if format.lower() == 'json':
                with open(file_path, 'w') as f:
                    f.write(schema.to_json())
                    
            elif format.lower() == 'yaml':
                try:
                    import yaml
                    with open(file_path, 'w') as f:
                        yaml.dump(schema.to_dict(), f, default_flow_style=False, default=str)
                except ImportError:
                    self.logger.error("PyYAML not available for YAML export")
                    return False
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Schema exported to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export schema: {e}")
            return False
    
    # -------------------------------------------------------------- get_cache_info()
    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache status information.

        Returns:
            Dict with file presence, cache validity, age/ttl, paths, and mode flags.
        """
        return {
            # Presence/validity
            "has_cached_schema": self.cache.schema is not None,
            "has_cached_structural_summary": self.cache.structural_summary is not None,
            "cache_valid": self.cache.is_valid(),
            # Timing and TTL
            "cache_age_seconds": self.cache.get_age_seconds(),
            "cache_ttl_seconds": self.cache.cache_ttl_seconds,
            # Filesystem details
            "cache_file_exists": self.cache_file.exists(),
            "cache_file_path": str(self.cache_file),
            "structural_summary_file_exists": self.structural_summary_file.exists(),
            "structural_summary_file_path": str(self.structural_summary_file),
            # Mode and timestamps
            "static_mode": self.static_mode,
            "last_updated": self.cache.last_updated.isoformat() if self.cache.last_updated else None,
        }
    
    # -------------------------------------------------------------- _export_structural_summary()
    def _export_structural_summary(self) -> None:
        """
        Export structural schema summary to JSON file.
        
        This method is called after successful schema acquisition and caching.
        It creates a lightweight JSON export for LLM consumption.
        Best-effort operation - failures are logged but don't break the main workflow.
        """
        try:
            # Check if export is enabled
            from ..core.config import settings
            if not settings.schema_export_enabled:
                self.logger.debug("Schema export disabled via configuration")
                return
            
            if not self.cache.schema:
                self.logger.debug("No cached schema available for structural export")
                return
            
            # Get database instance for executing introspection queries
            from ..core.database import get_database
            database = get_database()
            
            # Prepare metadata
            metadata = {
                "cache_timestamp": self.cache.last_updated.isoformat() if self.cache.last_updated else None,
                "schema_acquisition_timestamp": self.cache.schema.acquisition_timestamp,
                "static_mode": self.static_mode
            }
            
            # Export structural summary (best-effort, never fails)
            export_path = export_structural_summary_safe(
                database=database,
                cache_dir=self.cache_dir,
                metadata=metadata,
                output_filename=settings.schema_export_filename
            )
            
            if export_path:
                self.logger.info(f"📄 Structural schema summary exported to {export_path}")
                
                # Load the newly exported structural summary into cache
                self._load_structural_summary()
            else:
                self.logger.debug("Structural schema export was not successful (non-fatal)")
                
        except Exception as e:
            # This should never happen due to safe wrapper, but extra safety
            self.logger.warning(f"Unexpected error in structural schema export: {e}")
    
    # -------------------------------------------------------------- get_cache_info()
    # (Removed duplicate get_cache_info; consolidated above)
    
    # =========================================================================
    # Database Gateway Methods (Neo4j Access Boundary)
    # =========================================================================
    
    # -------------------------------------------------------------- database_health_check()
    def database_health_check(self) -> Dict[str, Any]:
        """Wrap database health check - gateway method for Neo4j access."""
        from ..core.database import database_health_check
        return database_health_check()
    
    # -------------------------------------------------------------- start_db_background_tasks()
    async def start_db_background_tasks(self) -> None:
        """Warm pool and start periodic health checks - gateway method for Neo4j access."""
        from ..core.database import get_database
        db = get_database()
        await db.warm_up_connections()
        
        # Start periodic health check task and keep a reference to avoid GC
        import asyncio
        self._db_health_check_task = asyncio.create_task(
            db.periodic_health_check(),
            name="db_health_check"
        )
        self.logger.info("✅ Database connection pool warmed up and health checks started via schema gateway")
    
    # -------------------------------------------------------------- execute_read()
    def execute_read(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute read-only Cypher query with validation - gateway method for Neo4j access."""
        # Simple read-only guard
        write_ops = ['CREATE', 'MERGE', 'DELETE', 'SET', 'REMOVE', 'DROP', 'ALTER', 'LOAD CSV', 'IMPORT']
        cypher_upper = cypher.upper()
        
        for op in write_ops:
            if op in cypher_upper:
                raise ValueError(f"Write operation '{op}' not allowed in read-only query")
        
        # Allow whitelisted CALL procedures
        if 'CALL' in cypher_upper:
            allowed_calls = ['db.labels', 'db.relationshipTypes', 'db.schema.nodeTypeProperties', 'db.schema.relTypeProperties']
            if not any(call in cypher for call in allowed_calls):
                raise ValueError("Only whitelisted CALL procedures allowed")
        
        from ..core.database import get_database
        db = get_database()
        return db.execute_query(cypher, params or {})
    
    # -------------------------------------------------------------- get_neo4j_vector()
    def get_neo4j_vector(self, embeddings):
        """Create Neo4jVector instance - gateway method for Neo4j access."""
        from langchain_community.graphs import Neo4jGraph
        from langchain_neo4j import Neo4jVector

        from ..core.config import settings
        
        # Create Neo4jGraph first (as was done in vector.py)
        neo4j_graph = Neo4jGraph(
            url=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,  # Plain string, no .get_secret_value()
            database=settings.neo4j_database
        )
        
        # Create vector index using the graph
        return Neo4jVector.from_existing_index(
            embedding=embeddings,
            graph=neo4j_graph,
            index_name="chunk_embedding_index",     # Hardcoded from vector.py
            node_label="Chunk",                     # Hardcoded from vector.py
            text_node_property="text",              # Hardcoded from vector.py
            embedding_node_property="embedding"    # Hardcoded from vector.py
        )
    
    # -------------------------------------------------------------- shutdown_database_connections()
    def shutdown_database_connections(self) -> None:
        """Shutdown database connections - gateway method for Neo4j access."""
        from ..core.database import reset_database_connection
        reset_database_connection()

# ------------------------------------------------------------------------- end class SchemaManager


# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# Cross-Module Processing
# =========================================================================

# --------------------------------------------------------------------------------- get_schema_manager()
def get_schema_manager(cache_dir: Optional[str] = None, static_mode: bool = True, cache_ttl_seconds: Optional[int] = None) -> SchemaManager:
    """Get or create the global schema manager instance optimized for static KG."""
    global _schema_manager_instance
    
    with _schema_manager_lock:
        if _schema_manager_instance is None:
            _schema_manager_instance = SchemaManager(cache_dir, static_mode, cache_ttl_seconds)
            logger.info("Global schema manager instance created")
        return _schema_manager_instance

# --------------------------------------------------------------------------------- end get_schema_manager()

# --------------------------------------------------------------------------------- reset_schema_manager()
def reset_schema_manager() -> None:
    """Reset the global schema manager instance (useful for testing)."""
    global _schema_manager_instance
    
    with _schema_manager_lock:
        _schema_manager_instance = None
        logger.info("Schema manager instance reset")

# --------------------------------------------------------------------------------- end reset_schema_manager()

# __________________________________________________________________________
# End of File
