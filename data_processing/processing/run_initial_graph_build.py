#!/usr/bin/env python3
# -------------------------------------------------------------------------
# File: run_initial_graph_build.py
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 2025-08-12
# File Path: data_processing/processing/run_initial_graph_build.py
# ------------------------------------------------------------------------

# --- Module Objective ---
#   Configurable wrapper for initial graph build with environment validation,
#   test-mode safety, and progress reporting.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Function: get_processing_config() -> Dict[str, str]
# - Function: validate_environment() -> bool
# - Function: run_initial_graph_build() -> int
# - Function: main() -> int
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: os, sys, pathlib.Path, typing
# - Local Project Modules: env_manager, processing.initial_graph_build
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - APH-IF  
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG – Intelligent Fusion (APH-IF)
# -------------------------------------------------------------------------


# =========================================================================
# Imports
# =========================================================================
# Standard library imports
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the parent directories to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # APH-IF-Dev root

from processing.initial_graph_build import HybridStoreBuilder
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent.parent / '.env')


def get_neo4j_config():
    """Get Neo4j configuration from environment variables set by set_environment.py"""
    return {
        'uri': os.getenv('NEO4J_URI'),
        'username': os.getenv('NEO4J_USERNAME'),
        'password': os.getenv('NEO4J_PASSWORD')
    }


# =========================================================================
# Global Constants / Variables
# =========================================================================
# (No module-level constants; configuration is derived from environment)


# =========================================================================
# Standalone Function Definitions
# =========================================================================
# --------------------------------------------------------------------------------- get_processing_config()
def get_processing_config() -> Dict[str, str]:
    """Get processing configuration from environment variables.

    Returns:
        Dict[str, str]: Flat config for the initial build runner, including
                        environment, chunking, extraction, and limits.
    """
    
    # Get current environment info
    app_env = os.getenv('APP_ENV', 'development')
    force_test_db = os.getenv('FORCE_TEST_DB', 'false').lower() == 'true'
    verbose = os.getenv('VERBOSE', 'false').lower() == 'true'
    
    config = {
        # Core settings
        'APP_ENV': app_env,
        'FORCE_TEST_DB': str(force_test_db).lower(),
        'VERBOSE': str(verbose).lower(),
        
        # Processing settings with environment-specific defaults
        'PDF_DIR': os.getenv('PDF_DIR', 'processing/data_pdf' if not force_test_db else 'processing/data_pdf_test'),
        'CLEAR_DB': os.getenv('CLEAR_DB', 'false'),
        'MAX_DOCS': os.getenv('MAX_DOCS', ''),  # Empty = unlimited
        'MAX_PAGES': os.getenv('MAX_PAGES', ''),  # Empty = unlimited
        
        # Chunking settings
        'CHUNK_SIZE_CHARS': os.getenv('CHUNK_SIZE_CHARS', '1000'),
        'CHUNK_OVERLAP_CHARS': os.getenv('CHUNK_OVERLAP_CHARS', '200'),
        
        # Entity extraction settings
        'EXTRACT_EVERY_N_CHUNKS': os.getenv('EXTRACT_EVERY_N_CHUNKS', '2'),
        'OPENAI_MODEL': os.getenv('OPENAI_MODEL_NANO', os.getenv('OPENAI_MODEL', 'gpt-5-nano')),
    }
    
    # Environment-specific adjustments
    if force_test_db:
        # Test environment - use smaller, faster settings
        config.update({
            'CHUNK_SIZE_CHARS': os.getenv('CHUNK_SIZE_CHARS', '3000'),
            'EXTRACT_EVERY_N_CHUNKS': os.getenv('EXTRACT_EVERY_N_CHUNKS', '5'),
            'MAX_PAGES': os.getenv('MAX_PAGES', '10'),
            'CLEAR_DB': os.getenv('CLEAR_DB', 'true'),  # Default to clearing in test
        })
    
    return config
# --------------------------------------------------------------------------------- end get_processing_config()


# --------------------------------------------------------------------------------- validate_environment()
def validate_environment() -> bool:
    """Validate that the environment is properly configured.

    Returns:
        bool: True if the runner can proceed; False otherwise.
    """
    try:
        # Check Neo4j configuration
        config = get_neo4j_config()
        if not all([config['uri'], config['username'], config['password']]):
            print("❌ Neo4j configuration incomplete")
            return False
        
        # Check OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ OPENAI_API_KEY not found in environment")
            return False
        
        # Check PDF directory exists
        # Use environment-specific default
        force_test_db = os.getenv('FORCE_TEST_DB', 'false').lower() == 'true'
        default_pdf_dir = 'processing/data_pdf' if not force_test_db else 'processing/data_pdf_test'
        pdf_dir = os.getenv('PDF_DIR', default_pdf_dir)
        data_processing_dir = Path(__file__).parent.parent  # data_processing directory
        full_pdf_path = data_processing_dir / pdf_dir
        
        if not full_pdf_path.exists():
            print(f"❌ PDF directory not found: {full_pdf_path}")
            return False
        
        # Check for PDF files
        pdf_files = list(full_pdf_path.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in: {full_pdf_path}")
            return False
        
        print(f"✅ Found {len(pdf_files)} PDF file(s) in {pdf_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return False
# --------------------------------------------------------------------------------- end validate_environment()


# --------------------------------------------------------------------------------- run_initial_graph_build()
def run_initial_graph_build() -> int:
    """Run the initial graph build process.

    Returns:
        int: 0 on success; non-zero on failure conditions.
    """
    
    print("APH-IF Initial Graph Build Runner")
    print("=" * 80)
    
    # Validate environment
    if not validate_environment():
        return 1
    
    # Get configuration
    config = get_processing_config()
    
    # Display current configuration
    print("\nCurrent Configuration:")
    print(f"  Environment: {config['APP_ENV']}")
    print(f"  Test Database: {config['FORCE_TEST_DB']}")
    print(f"  PDF Directory: {config['PDF_DIR']}")
    print(f"  Clear Database: {config['CLEAR_DB']}")
    print(f"  Max Documents: {config['MAX_DOCS'] or 'unlimited'}")
    print(f"  Max Pages: {config['MAX_PAGES'] or 'unlimited'}")
    print(f"  Chunk Size: {config['CHUNK_SIZE_CHARS']} chars")
    print(f"  Extract Every: {config['EXTRACT_EVERY_N_CHUNKS']} chunks")
    print(f"  Model: {config['OPENAI_MODEL']}")
    print(f"  Verbose: {config['VERBOSE']}")
    
    # Get Neo4j info for display
    neo4j_config = get_neo4j_config()
    print(f"  Database: {neo4j_config['uri']}")
    
    print("\n" + "=" * 80)
    print("Starting Initial Graph Build")
    print("=" * 80)
    
    try:
        # Set environment variables for the process
        for key, value in config.items():
            os.environ[key] = value
        
        # Create and run the graph builder directly
        builder = HybridStoreBuilder()
        builder.print_progress_header()
        
        # Get data path with environment-specific default
        force_test_db = os.getenv('FORCE_TEST_DB', 'false').lower() == 'true'
        default_pdf_dir = 'processing/data_pdf' if not force_test_db else 'processing/data_pdf_test'
        data_path = os.getenv("PDF_DIR", default_pdf_dir)
        
        # Optional clearing via env flag
        if os.getenv("CLEAR_DB", "false").lower() == "true":
            print("Clearing existing database...")
            builder.graph.query("MATCH (n) DETACH DELETE n")
            print("[SUCCESS] Database cleared - starting fresh build")
        
        # Process all PDFs to build the graph
        builder.process_directory(data_path)
        builder.create_vector_index()
        
        # Final summary
        builder.print_final_summary()
        
        print("\nINITIAL GRAPH BUILD COMPLETED SUCCESSFULLY!")
        print("✅ Graph database has been populated with documents, chunks, and entities")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Initial graph build failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
# --------------------------------------------------------------------------------- end run_initial_graph_build()


# --------------------------------------------------------------------------------- main()
def main():
    """Main entry point for the initial build runner."""
    try:
        return run_initial_graph_build()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
# --------------------------------------------------------------------------------- end main()


# =========================================================================
# Module Initialization / Main Execution Guard
# =========================================================================
# This block runs only when the file is executed directly, not when imported.
# It launches the initial graph build with the current environment settings.

# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
# ---------------------------------------------------------------------------------

# =========================================================================
# End of File
# =========================================================================
