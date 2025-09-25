#!/usr/bin/env python3
# -------------------------------------------------------------------------
# File: schema_cli.py
# Author: Alexander Ricciardi
# Date:
# [File Path] backend/schema_cli.py
# -------------------------------------------------------------------------
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
#   Command-line interface for managing knowledge-graph schema acquisition,
#   caching, export, and analysis. Intended for developers/administrators to
#   operate schema workflows outside of the request path.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Function: refresh_schema
# - Function: refresh_after_data_processing
# - Function: show_info
# - Function: clear_cache
# - Function: export_schema
# - Function: analyze_schema
# - Function: main (CLI entry)
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: sys, json, time, argparse, pathlib (Path), typing
# - Third-Party: (none)
# - Local Project Modules: app.schema.get_schema_manager
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Executed from the repository root or backend/ to manage schema lifecycle
# tasks. Not imported by application code; invoked as a CLI entry.
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent
# Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""
APH-IF Schema Management CLI Tool

Command-line interface for managing knowledge graph schema acquisition,
caching, export, and maintenance. This tool is designed for developers and
administrators to manage schema without performance constraints.

Usage:
    python schema_cli.py refresh               # Refresh schema from database  
    python schema_cli.py info                  # Show schema and cache info
    python schema_cli.py clear                 # Clear cached schema
    python schema_cli.py export schema.json    # Export schema to file
    python schema_cli.py analyze               # Show detailed schema analysis
    python schema_cli.py data-refresh          # Refresh after data processing (static KG)
"""

import argparse
import sys
import time
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).parent / "app"))

from app.schema import get_schema_manager

# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# Utility Functions
# =========================================================================

# --------------------------------------------------------------------------------- refresh_schema()
def refresh_schema():
    """Refresh schema from database."""
    print("🔄 Refreshing knowledge graph schema from database...")
    
    schema_manager = get_schema_manager()
    
    start_time = time.time()
    success = schema_manager.refresh_schema()
    duration = time.time() - start_time
    
    if success:
        summary = schema_manager.get_schema_summary()
        print(f"✅ Schema refresh completed in {duration:.2f}s")
        print()
        print("📊 Schema Summary:")
        print(f"  • Node Labels: {summary['node_labels_count']}")
        print(f"  • Relationship Types: {summary['relationship_types_count']}")
        print(f"  • Total Nodes: {summary['total_nodes']:,}")
        print(f"  • Total Relationships: {summary['total_relationships']:,}")
        print(f"  • Property Keys: {summary['global_properties_count']}")
        print(f"  • Constraints: {summary['constraints_count']}")
        print(f"  • Indexes: {summary['indexes_count']}")
        print()
        print("💾 Cache Info:")
        cache_info = schema_manager.get_cache_info()
        print(f"  • Cache File: {cache_info['cache_file_path']}")
        print(f"  • Cache Valid: {cache_info['cache_valid']}")
        
    else:
        print("❌ Schema refresh failed")
        sys.exit(1)

# --------------------------------------------------------------------------------- end refresh_schema()

# --------------------------------------------------------------------------------- refresh_after_data_processing()
def refresh_after_data_processing():
    """Refresh schema after data processing module execution."""
    print("🔄 Refreshing schema after data processing...")
    print("📊 This command is optimized for static knowledge graphs")
    
    schema_manager = get_schema_manager()
    
    start_time = time.time()
    success = schema_manager.refresh_after_data_processing()
    duration = time.time() - start_time
    
    if success:
        summary = schema_manager.get_schema_summary()
        print(f"✅ Schema refreshed after data processing in {duration:.2f}s")
        print()
        print("📊 Updated Schema Summary:")
        print(f"  • Node Labels: {summary['node_labels_count']}")
        print(f"  • Relationship Types: {summary['relationship_types_count']}")
        print(f"  • Total Nodes: {summary['total_nodes']:,}")
        print(f"  • Total Relationships: {summary['total_relationships']:,}")
        print(f"  • Property Keys: {summary['global_properties_count']}")
        print()
        print("💡 Schema is now cached indefinitely (static mode)")
        print("   Run this command again only after next data processing")
    else:
        print("❌ Schema refresh after data processing failed")
        sys.exit(1)

# --------------------------------------------------------------------------------- end refresh_after_data_processing()

# --------------------------------------------------------------------------------- show_info()
def show_info():
    """Show schema and cache information."""
    schema_manager = get_schema_manager()
    
    print("📋 APH-IF Schema Information")
    print("=" * 50)
    
    summary = schema_manager.get_schema_summary()
    
    if summary.get("available"):
        print("✅ Schema Available")
        print()
        print("📊 Schema Statistics:")
        print(f"  • Node Labels: {summary['node_labels_count']}")
        print(f"  • Relationship Types: {summary['relationship_types_count']}")
        print(f"  • Total Nodes: {summary['total_nodes']:,}")
        print(f"  • Total Relationships: {summary['total_relationships']:,}")
        print(f"  • Property Keys: {summary['global_properties_count']}")
        print(f"  • Constraints: {summary['constraints_count']}")
        print(f"  • Indexes: {summary['indexes_count']}")
        print()
        print("⏱️  Timing Information:")
        print(f"  • Acquisition Time: {summary['acquisition_duration_seconds']:.2f}s")
        print(f"  • Acquired At: {summary['acquisition_timestamp']}")
        if summary['cache_age_seconds']:
            print(f"  • Cache Age: {summary['cache_age_seconds']:.0f}s")
        print(f"  • Cache Valid: {summary['cache_valid']}")
        
    else:
        print("❌ Schema Not Available")
    
    print()
    print("💾 Cache Information:")
    cache_info = schema_manager.get_cache_info()
    print(f"  • Has Cached Schema: {cache_info['has_cached_schema']}")
    print(f"  • Cache Valid: {cache_info['cache_valid']}")
    print(f"  • Cache TTL: {cache_info['cache_ttl_seconds']}s")
    print(f"  • Cache File Exists: {cache_info['cache_file_exists']}")
    print(f"  • Cache File Path: {cache_info['cache_file_path']}")
    
    if cache_info['cache_age_seconds']:
        age_hours = cache_info['cache_age_seconds'] / 3600
        print(f"  • Cache Age: {age_hours:.1f} hours")

# --------------------------------------------------------------------------------- end show_info()

# --------------------------------------------------------------------------------- clear_cache()
def clear_cache():
    """Clear cached schema."""
    print("🧹 Clearing schema cache...")
    
    schema_manager = get_schema_manager()
    schema_manager.clear_cache()
    
    print("✅ Schema cache cleared")

# --------------------------------------------------------------------------------- end clear_cache()

# --------------------------------------------------------------------------------- export_schema()
def export_schema(file_path: str, format_type: str = "json"):
    """Export schema to file."""
    print(f"📤 Exporting schema to {file_path}...")
    
    schema_manager = get_schema_manager()
    
    # Determine format from file extension if not specified
    if not format_type:
        ext = Path(file_path).suffix.lower()
        if ext == ".yaml" or ext == ".yml":
            format_type = "yaml"
        else:
            format_type = "json"
    
    success = schema_manager.export_schema(file_path, format_type)
    
    if success:
        file_size = Path(file_path).stat().st_size
        print(f"✅ Schema exported successfully ({file_size:,} bytes)")
    else:
        print("❌ Schema export failed")
        sys.exit(1)

# --------------------------------------------------------------------------------- end export_schema()

# --------------------------------------------------------------------------------- analyze_schema()
def analyze_schema():
    """Show detailed schema analysis."""
    schema_manager = get_schema_manager()
    schema = schema_manager.get_schema()
    
    if not schema:
        print("❌ No schema available")
        sys.exit(1)
    
    print("🔍 Detailed Schema Analysis")
    print("=" * 50)
    
    print("📊 Overview:")
    print(f"  • Total Nodes: {schema.total_nodes:,}")
    print(f"  • Total Relationships: {schema.total_relationships:,}")
    print(f"  • Node Labels: {len(schema.nodes)}")
    print(f"  • Relationship Types: {len(schema.relationships)}")
    print(f"  • Global Properties: {len(schema.global_property_keys)}")
    
    print()
    print("🏷️  Node Labels (Top 10):")
    sorted_nodes = sorted(schema.nodes.items(), key=lambda x: x[1].total_count, reverse=True)
    for label, node_info in sorted_nodes[:10]:
        print(f"  • {label}: {node_info.total_count:,} nodes, {len(node_info.all_properties)} properties")
    
    if len(schema.nodes) > 10:
        print(f"  ... and {len(schema.nodes) - 10} more labels")
    
    print()
    print("🔗 Relationship Types (Top 10):")
    sorted_rels = sorted(schema.relationships.items(), key=lambda x: x[1].total_count, reverse=True)
    for rel_type, rel_info in sorted_rels[:10]:
        print(f"  • {rel_type}: {rel_info.total_count:,} relationships, {len(rel_info.all_patterns)} patterns")
    
    if len(schema.relationships) > 10:
        print(f"  ... and {len(schema.relationships) - 10} more types")
    
    print()
    print("🔑 Most Common Properties:")
    # Count property usage across all node types
    prop_usage = {}
    for node_info in schema.nodes.values():
        for prop in node_info.all_properties:
            prop_usage[prop] = prop_usage.get(prop, 0) + 1
    
    sorted_props = sorted(prop_usage.items(), key=lambda x: x[1], reverse=True)
    for prop, count in sorted_props[:15]:
        print(f"  • {prop}: used in {count} node type(s)")
    
    print()
    print("⚡ Performance Metrics:")
    print(f"  • Schema Acquisition: {schema.acquisition_duration_seconds:.2f}s")
    print(f"  • Database Queries: ~{len(schema.nodes) * 3 + len(schema.relationships) * 2} executed")

# --------------------------------------------------------------------------------- end analyze_schema()

# =========================================================================
# CLI Entry
# =========================================================================

# --------------------------------------------------------------------------------- main()
def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="APH-IF Knowledge Graph Schema Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python schema_cli.py refresh                    # Refresh schema from database
  python schema_cli.py info                       # Show current schema info
  python schema_cli.py clear                      # Clear cached schema
  python schema_cli.py export my_schema.json      # Export schema to JSON
  python schema_cli.py export my_schema.yaml      # Export schema to YAML
  python schema_cli.py analyze                    # Show detailed analysis
        """
    )
    
    parser.add_argument('command', 
                       choices=['refresh', 'info', 'clear', 'export', 'analyze', 'data-refresh'],
                       help='Command to execute')
    
    parser.add_argument('file_path', nargs='?',
                       help='File path for export command')
    
    parser.add_argument('--format', choices=['json', 'yaml'],
                       help='Export format (auto-detected from file extension if not specified)')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'refresh':
            refresh_schema()
            
        elif args.command == 'info':
            show_info()
            
        elif args.command == 'clear':
            clear_cache()
            
        elif args.command == 'export':
            if not args.file_path:
                print("❌ File path required for export command")
                sys.exit(1)
            export_schema(args.file_path, args.format)
            
        elif args.command == 'analyze':
            analyze_schema()
            
        elif args.command == 'data-refresh':
            refresh_after_data_processing()
            
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------- end main()

# __________________________________________________________________________
# End of File
