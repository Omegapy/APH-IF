# -------------------------------------------------------------------------
# File: schema_exporter.py
# Author: Alexander Ricciardi
# Date: 2025-09-18
# [File Path] backend/app/schema/schema_exporter.py
# ------------------------------------------------------------------------
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
#   Export and validate a lightweight structural JSON summary of the knowledge
#   graph. Includes URI masking and atomic write helpers used by the schema
#   manager during acquisition/refresh.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Function: mask_neo4j_uri
# - Function: write_atomic_json
# - Function: export_structural_summary
# - Function: export_structural_summary_safe
# - Function: validate_export_file
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: json, logging, tempfile, datetime, pathlib, typing, urllib.parse
# - Local Project Modules: ..core.database.APHIFDatabase
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Used by SchemaManager to export a compact structural summary to cache files.
# Safe wrapper ensures schema refresh flows are resilient to export failures.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""
Schema structural JSON export utilities for APH-IF knowledge graphs.

This module provides helper utilities to:
- Mask sensitive parts of Neo4j URIs for logs and metadata
- Write JSON exports atomically to avoid partial files on failure
- Export a compact structural summary suitable for LLMs
- Validate that exported files are structurally sound and self-consistent
"""

# __________________________________________________________________________
# Imports

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from ..core.database import APHIFDatabase

# __________________________________________________________________________
# Global Constants / Variables

logger = logging.getLogger(__name__)


# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# Utility Functions
# =========================================================================

# --------------------------------------------------------------------------------- mask_neo4j_uri()
def mask_neo4j_uri(uri: str) -> str:
    """
    Mask sensitive information in Neo4j URI while keeping useful parts.
    
    Args:
        uri: Original Neo4j URI
        
    Returns:
        Masked URI with credentials redacted
        
    Examples:
        neo4j+s://user:pass@host:port -> neo4j+s://***:***@host:port
        neo4j://localhost:7687 -> neo4j://localhost:7687
    """
    try:
        parsed = urlparse(uri)
        
        # Check if parsing was successful (valid scheme and netloc)
        if not parsed.scheme or not parsed.netloc:
            return "***masked***"
        
        # Reconstruct with masked credentials
        if parsed.username and parsed.password:
            masked_netloc = f"***:***@{parsed.hostname}"
            if parsed.port:
                masked_netloc += f":{parsed.port}"
        else:
            masked_netloc = parsed.netloc
            
        masked_uri = f"{parsed.scheme}://{masked_netloc}{parsed.path}"
        
        # Add query parameters if present (usually safe)
        if parsed.query:
            masked_uri += f"?{parsed.query}"
            
        return masked_uri
        
    except Exception as e:
        logger.debug(f"Error masking URI: {e}")
        return "***masked***"

# --------------------------------------------------------------------------------- end mask_neo4j_uri()

# --------------------------------------------------------------------------------- write_atomic_json()
def write_atomic_json(file_path: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON data to file atomically using temporary file + rename.
    
    Args:
        file_path: Target file path
        data: JSON-serializable data to write
        
    Raises:
        Exception: If write fails
    """
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(
        mode='w',
        dir=file_path.parent,
        prefix=f"{file_path.stem}_tmp_",
        suffix=".json",
        delete=False
    ) as tmp_file:
        json.dump(data, tmp_file, indent=2, default=str, ensure_ascii=False)
        tmp_file_path = Path(tmp_file.name)
    
    # Atomic rename
    tmp_file_path.replace(file_path)
    logger.debug(f"Atomically wrote JSON to {file_path}")

# --------------------------------------------------------------------------------- end write_atomic_json()

# =========================================================================
# Export API
# =========================================================================

# --------------------------------------------------------------------------------- export_structural_summary()
def export_structural_summary(
    database: APHIFDatabase,
    cache_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
    output_filename: str = "kg_schema_structural_summary.json"
) -> Path:
    """
    Export structural schema summary to JSON file.
    
    Args:
        database: Database instance for executing queries
        cache_dir: Directory to write the export file
        metadata: Additional metadata to include (e.g., cache timestamp)
        output_filename: Name of output file
        
    Returns:
        Path to the created export file
        
    Raises:
        Exception: If export fails
    """
    from ..core.config import settings
    
    start_time = datetime.utcnow()
    logger.info("Starting structural schema export...")
    
    try:
        # Execute Neo4j introspection procedures
        logger.debug("Querying node labels...")
        labels_records = database.execute_query("CALL db.labels()")
        node_labels = [{"label": record["label"]} for record in labels_records]
        
        logger.debug("Querying relationship types...")
        rel_types_records = database.execute_query("CALL db.relationshipTypes()")
        relationship_types = [{"type": record["relationshipType"]} for record in rel_types_records]
        
        # Get node property types (with fallback)
        node_properties = []
        try:
            logger.debug("Querying node property types...")
            node_props_records = database.execute_query("CALL db.schema.nodeTypeProperties()")
            
            for record in node_props_records:
                node_properties.append({
                    "labels": record.get("nodeLabels", []),
                    "property": record.get("propertyName"),
                    "types": record.get("propertyTypes", []),
                    "mandatory": record.get("mandatory", False)
                })
                
        except Exception as e:
            logger.warning(f"Advanced node property query failed, using fallback: {e}")
            # Fallback: basic property keys without types
            try:
                props_records = database.execute_query("CALL db.propertyKeys()")
                for record in props_records:
                    node_properties.append({
                        "labels": [],  # Cannot determine specific labels without advanced query
                        "property": record.get("propertyKey"),
                        "types": ["UNKNOWN"],  # Cannot determine types without advanced query
                        "mandatory": False
                    })
            except Exception as fallback_error:
                logger.warning(f"Property keys fallback also failed: {fallback_error}")
        
        # Get relationship property types (with fallback)
        relationship_properties = []
        try:
            logger.debug("Querying relationship property types...")
            rel_props_records = database.execute_query("CALL db.schema.relTypeProperties()")
            
            for record in rel_props_records:
                relationship_properties.append({
                    "relationship_type": record.get("relationshipType"),
                    "property": record.get("propertyName"),
                    "types": record.get("propertyTypes", []),
                    "mandatory": record.get("mandatory", False)
                })
                
        except Exception as e:
            logger.warning(f"Advanced relationship property query failed: {e}")
            # For relationships, we don't have a good fallback, so leave empty
        
        # Build export payload
        generation_time = datetime.utcnow()
        
        payload = {
            "metadata": {
                "generated_at": generation_time.isoformat() + "Z",
                "neo4j_uri": mask_neo4j_uri(settings.neo4j_uri),
                "database": settings.neo4j_database,
                "counts": {
                    "node_labels": len(node_labels),
                    "relationship_types": len(relationship_types),
                    "node_property_rows": len(node_properties),
                    "relationship_property_rows": len(relationship_properties)
                },
                "schema_manager_cache_timestamp": metadata.get("cache_timestamp") if metadata else None,
                "version": 1
            },
            "node_labels": node_labels,
            "relationship_types": relationship_types,
            "node_property_types": node_properties,
            "relationship_property_types": relationship_properties
        }
        
        # Write to file atomically
        output_path = cache_dir / output_filename
        write_atomic_json(output_path, payload)
        
        export_time = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(
            f"✅ Structural schema export completed in {export_time:.3f}s: "
            f"{len(node_labels)} labels, {len(relationship_types)} rel types, "
            f"{len(node_properties)}/{len(relationship_properties)} property rows"
        )
        
        return output_path
        
    except Exception as e:
        export_time = (datetime.utcnow() - start_time).total_seconds()
        logger.error(f"❌ Structural schema export failed after {export_time:.3f}s: {e}")
        raise

# --------------------------------------------------------------------------------- end export_structural_summary()

# --------------------------------------------------------------------------------- export_structural_summary_safe()
def export_structural_summary_safe(
    database: APHIFDatabase,
    cache_dir: Path,
    metadata: Optional[Dict[str, Any]] = None,
    output_filename: str = "kg_schema_structural_summary.json"
) -> Optional[Path]:
    """
    Safe wrapper for export_structural_summary that never raises exceptions.
    
    This function is designed for integration with SchemaManager where export
    failures should not break the main schema refresh process.
    
    Args:
        database: Database instance for executing queries
        cache_dir: Directory to write the export file
        metadata: Additional metadata to include
        output_filename: Name of output file
        
    Returns:
        Path to created file if successful, None if failed
    """
    try:
        return export_structural_summary(database, cache_dir, metadata, output_filename)
    except Exception as e:
        logger.warning(f"Structural schema export failed (non-fatal): {e}")
        return None

# --------------------------------------------------------------------------------- end export_structural_summary_safe()

# =========================================================================
# Validation
# =========================================================================

# --------------------------------------------------------------------------------- validate_export_file()
def validate_export_file(file_path: Path) -> Dict[str, Any]:
    """
    Validate the structure and content of an exported schema file.
    
    Args:
        file_path: Path to the export file
        
    Returns:
        Validation results with status and details
    """
    validation_result = {
        "valid": False,
        "file_exists": False,
        "file_size_bytes": 0,
        "json_parseable": False,
        "required_fields_present": False,
        "metadata_valid": False,
        "counts_consistent": False,
        "errors": []
    }
    
    try:
        # Check file existence
        if not file_path.exists():
            validation_result["errors"].append("Export file does not exist")
            return validation_result
        
        validation_result["file_exists"] = True
        validation_result["file_size_bytes"] = file_path.stat().st_size
        
        # Try to parse JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
        validation_result["json_parseable"] = True
        
        # Check required fields
        required_fields = ["metadata", "node_labels", "relationship_types", 
                          "node_property_types", "relationship_property_types"]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            validation_result["errors"].append(f"Missing required fields: {missing_fields}")
        else:
            validation_result["required_fields_present"] = True
        
        # Validate metadata
        metadata = data.get("metadata", {})
        required_metadata = ["generated_at", "version", "counts"]
        missing_metadata = [field for field in required_metadata if field not in metadata]
        if missing_metadata:
            validation_result["errors"].append(f"Missing metadata fields: {missing_metadata}")
        else:
            validation_result["metadata_valid"] = True
        
        # Check count consistency
        if validation_result["metadata_valid"]:
            counts = metadata["counts"]
            actual_counts = {
                "node_labels": len(data.get("node_labels", [])),
                "relationship_types": len(data.get("relationship_types", [])),
                "node_property_rows": len(data.get("node_property_types", [])),
                "relationship_property_rows": len(data.get("relationship_property_types", []))
            }
            
            consistent = all(counts.get(key) == actual_counts[key] for key in actual_counts)
            if not consistent:
                validation_result["errors"].append(
                    f"Count mismatch: declared {counts}, actual {actual_counts}"
                )
            else:
                validation_result["counts_consistent"] = True
        
        # Overall validation
        validation_result["valid"] = (
            validation_result["json_parseable"] and
            validation_result["required_fields_present"] and
            validation_result["metadata_valid"] and
            validation_result["counts_consistent"]
        )
        
    except json.JSONDecodeError as e:
        validation_result["errors"].append(f"JSON parse error: {e}")
    except Exception as e:
        validation_result["errors"].append(f"Validation error: {e}")
    
    return validation_result

# --------------------------------------------------------------------------------- end validate_export_file()

# __________________________________________________________________________
# End of File