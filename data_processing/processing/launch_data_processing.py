#!/usr/bin/env python3
# -------------------------------------------------------------------------
# File: add_relationships_to_graph.py
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 2025-08-17
# File Path: data_processing/processing/add_relationships_to_graph.py
# ------------------------------------------------------------------------

# --- Module Objective ---
#   Augment the base knowledge graph with higher-level relationships
#   between Entities and Documents (similarity, citations, supports, etc.).
#   Provides preconditions, taxonomy, candidate discovery, evidence hooks,
#   and safe MERGE upserts back into Neo4j.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Class: GraphRelationshipAdder
# - Class: Direction
# - Class: ValueType
# - Class: EdgePropSpec
# - Class: EdgeTypeSpec
# - Class: RelationInput
# - Class: RelationOutput
# - Class: Neo4jConfig
# - Function: info
# - Function: warn
# - Function: error
# - Function: debug
# - Function: print_taxonomy
# - Function: _coerce_type
# - Function: validate_edge_payload
# - Function: check_document_embedding_presence
# - Function: stream_entity_similarity
# - Function: stream_doc_similarity
# - Function: fetch_entity_evidence
# - Function: fetch_document_evidence
# - Function: label_with_llm
# - Function: label_with_llm_stub
# - Function: merge_entity_relations
# - Function: merge_document_relations
# - Function: main
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: os, sys, dataclasses, typing, enum, datetime
# - Third-Party: neo4j
# - Local Project Modules: common.monitored_openai, common.api_monitor
# -------------------------------------------------------------------------

# --- Usage / Integration ---
#   Imported by Phase 3 launchers or run helpers to discover and write
#   relationships. See companion usage guide: add_relationships_to_graph_usage.md
#   IMPORTANT: For experiments, set FORCE_TEST_DB=true to protect production data.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF)
# -------------------------------------------------------------------------

# =========================================================================
# Imports
# =========================================================================
# Standard library imports
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add the parent directories to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # APH-IF-Dev root

from processing.run_initial_graph_build import run_initial_graph_build
from processing.run_relationship_augmentation import main as run_augmentation
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
# (No module-level constants required; configuration is derived from env)


# =========================================================================
# Standalone Function Definitions
# =========================================================================
# --------------------------------------------------------------------------------- parse_arguments()
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Flags:
        --initial-only / --augmentation-only: select exactly one step.
        --skip-initial / --skip-augmentation: run the other step only.
        --test-mode: temporarily enable FORCE_TEST_DB for safe execution.
        --dry-run: preview augmentation writes without committing to Neo4j.

    Returns:
        argparse.Namespace: Parsed flags controlling pipeline behavior.
    """
    parser = argparse.ArgumentParser(
        description="APH-IF Complete Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete pipeline with current environment
    python -m processing.launch_data_processing
    
    # Run only initial graph build in test mode
    python -m processing.launch_data_processing --initial-only --test-mode
    
    # Run only relationship augmentation with dry run
    python -m processing.launch_data_processing --augmentation-only --dry-run
    
    # Run complete pipeline skipping initial build
    python -m processing.launch_data_processing --skip-initial
        """
    )
    
    # Execution control
    parser.add_argument('--initial-only', action='store_true',
                       help='Run only initial graph build')
    parser.add_argument('--augmentation-only', action='store_true', 
                       help='Run only relationship augmentation')
    parser.add_argument('--skip-initial', action='store_true',
                       help='Skip initial graph build')
    parser.add_argument('--skip-augmentation', action='store_true',
                       help='Skip relationship augmentation')
    
    # Environment control
    parser.add_argument('--test-mode', action='store_true',
                       help='Force test database mode')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without writing to database')
    
    return parser.parse_args()
# --------------------------------------------------------------------------------- end parse_arguments()


# --------------------------------------------------------------------------------- get_pipeline_config()
def get_pipeline_config(args: argparse.Namespace) -> Dict[str, str]:
    """Get complete pipeline configuration.

    Precedence:
        1) Current environment settings (via set_environment.py and process env)
        2) Test-mode overrides (when --test-mode is passed or FORCE_TEST_DB is true)

    Returns:
        Dict[str, str]: A flat config map for both steps (initial + augmentation).
    """

    # Get current environment info
    app_env = os.getenv('APP_ENV', 'development')
    force_test_db = os.getenv('FORCE_TEST_DB', 'false').lower() == 'true'
    verbose = os.getenv('VERBOSE', 'false').lower() == 'true'

    # Override test mode if requested
    if args.test_mode:
        force_test_db = True
        # Set test mode in environment
        os.environ['FORCE_TEST_DB'] = 'true'
    
    config = {
        # Core environment
        'APP_ENV': app_env,
        'FORCE_TEST_DB': str(force_test_db).lower(),
        'VERBOSE': str(verbose).lower(),
        
        # Initial graph build settings - environment-specific defaults
        'PDF_DIR': os.getenv('PDF_DIR', 'processing/data_pdf' if not force_test_db else 'processing/data_pdf_test'),
        'CLEAR_DB': os.getenv('CLEAR_DB', 'false'),
        'MAX_DOCS': os.getenv('MAX_DOCS', '1'),
        'CHUNK_SIZE_CHARS': os.getenv('CHUNK_SIZE_CHARS', '1000'),
        'EXTRACT_EVERY_N_CHUNKS': os.getenv('EXTRACT_EVERY_N_CHUNKS', '2'),
        'OPENAI_MODEL': os.getenv('OPENAI_MODEL_NANO', os.getenv('OPENAI_MODEL', 'gpt-5-nano')),
        
        # Relationship augmentation settings
        'AUGMENT_ENTITY': os.getenv('AUGMENT_ENTITY', 'true'),
        'AUGMENT_DOCUMENT': os.getenv('AUGMENT_DOCUMENT', 'false'),
        'ENTITY_SIM_CUTOFF': os.getenv('ENTITY_SIM_CUTOFF', '0.25'),
        'ENTITY_LIMIT': os.getenv('ENTITY_LIMIT', '1000'),
        'DOC_SIM_CUTOFF': os.getenv('DOC_SIM_CUTOFF', '0.8'),
        'DRY_RUN': str(args.dry_run).lower(),
    }
    
    # Environment-specific adjustments
    if force_test_db:
        # Test environment - optimized for speed and safety
        config.update({
            'CHUNK_SIZE_CHARS': os.getenv('CHUNK_SIZE_CHARS', '3000'),
            'EXTRACT_EVERY_N_CHUNKS': os.getenv('EXTRACT_EVERY_N_CHUNKS', '5'),
            'MAX_PAGES': os.getenv('MAX_PAGES', '10'),
            'CLEAR_DB': os.getenv('CLEAR_DB', 'true'),
            'ENTITY_SIM_CUTOFF': os.getenv('ENTITY_SIM_CUTOFF', '0.1'),
            'ENTITY_LIMIT': os.getenv('ENTITY_LIMIT', '10'),
        })
    
    return config
# --------------------------------------------------------------------------------- end get_pipeline_config()


# --------------------------------------------------------------------------------- validate_pipeline_environment()
def validate_pipeline_environment() -> bool:
    """Validate that the environment is ready for pipeline execution.

    Validates:
        - Neo4j credentials (uri/username/password) are available
        - OPENAI_API_KEY is set for model access

    Returns:
        bool: True if ready, False otherwise (with printed diagnostics).
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
        
        print("✅ Environment validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Environment validation failed: {e}")
        return False
# --------------------------------------------------------------------------------- end validate_pipeline_environment()


# --------------------------------------------------------------------------------- run_initial_step()
def run_initial_step(config: Dict[str, str]) -> bool:
    """Run the initial graph build step.

    Behavior:
        - Exports the provided configuration into the process env
        - Invokes run_initial_graph_build() which performs the full build

    Returns:
        bool: True on success, False on failure.
    """
    print("\n" + "=" * 80)
    print("🏗️  STEP 1: Initial Graph Build")
    print("=" * 80)
    
    try:
        # Set environment variables
        for key, value in config.items():
            os.environ[key] = value
        
        # Run initial graph build
        result = run_initial_graph_build()
        
        if result == 0:
            print("✅ Initial graph build completed successfully")
            return True
        else:
            print(f"❌ Initial graph build failed with return code: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Initial graph build failed: {e}")
        return False
# --------------------------------------------------------------------------------- end run_initial_step()


# --------------------------------------------------------------------------------- run_augmentation_step()
def run_augmentation_step(config: Dict[str, str]) -> bool:
    """Run the relationship augmentation step.

    Behavior:
        - Exports the provided configuration into the process env
        - Invokes the augmentation runner which handles similarity + labeling

    Returns:
        bool: True on success, False on failure.
    """
    print("\n" + "=" * 80)
    print("STEP 2: Relationship Augmentation")
    print("=" * 80)
    
    try:
        # Set environment variables
        for key, value in config.items():
            os.environ[key] = value
        
        # Run relationship augmentation
        run_augmentation()
        
        print("✅ Relationship augmentation completed successfully")
        return True
            
    except Exception as e:
        print(f"❌ Relationship augmentation failed: {e}")
        return False
# --------------------------------------------------------------------------------- end run_augmentation_step()


# --------------------------------------------------------------------------------- main()
def main():
    """Main pipeline execution.

    Exit codes:
        0: All requested steps succeeded
        1: Validation failed or at least one requested step failed
    """
    args = parse_arguments()
    
    print("APH-IF Complete Data Processing Pipeline")
    print("=" * 80)
    
    # Validate arguments
    if args.initial_only and args.augmentation_only:
        print("❌ Cannot specify both --initial-only and --augmentation-only")
        return 1
    
    if args.skip_initial and args.initial_only:
        print("❌ Cannot specify both --skip-initial and --initial-only")
        return 1
    
    if args.skip_augmentation and args.augmentation_only:
        print("❌ Cannot specify both --skip-augmentation and --augmentation-only")
        return 1
    
    # Store original test mode for restoration
    original_force_test_db = os.getenv('FORCE_TEST_DB', 'false').lower() == 'true'

    try:
        # Validate environment
        if not validate_pipeline_environment():
            return 1
        
        # Get configuration
        config = get_pipeline_config(args)
        
        # Display configuration
        print("\n📋 Pipeline Configuration:")
        print(f"  Environment: {config['APP_ENV']}")
        print(f"  Test Database: {config['FORCE_TEST_DB']}")
        print(f"  Dry Run: {config['DRY_RUN']}")
        
        neo4j_config = get_neo4j_config()
        print(f"  Database: {neo4j_config['uri']}")
        
        # Determine what to run
        run_initial = not (args.skip_initial or args.augmentation_only)
        run_augmentation = not (args.skip_augmentation or args.initial_only)
        
        print(f"  Run Initial Build: {run_initial}")
        print(f"  Run Augmentation: {run_augmentation}")
        
        # Execute pipeline
        success = True
        
        if run_initial:
            success = run_initial_step(config)
            if not success:
                print("\n❌ Pipeline failed at initial graph build step")
                return 1
        
        if run_augmentation and success:
            success = run_augmentation_step(config)
            if not success:
                print("\n❌ Pipeline failed at relationship augmentation step")
                return 1
        
        # Final status
        if success:
            print("\n" + "=" * 80)
            print("🎉 COMPLETE DATA PROCESSING PIPELINE SUCCESSFUL!")
            print("=" * 80)
            print("✅ Knowledge graph has been built and augmented")
            print("✅ Ready for querying and retrieval operations")
            return 0
        else:
            print("\n❌ Pipeline execution failed")
            return 1
    
    finally:
        # Restore original test mode if it was changed
        if args.test_mode:
            os.environ['FORCE_TEST_DB'] = str(original_force_test_db).lower()
            print(f"✅ Restored test database mode to: {original_force_test_db}")
# --------------------------------------------------------------------------------- end main()


 # =========================================================================
 # Module Initialization / Main Execution Guard
 # =========================================================================
 # This block runs only when the file is executed directly, not when imported.
 # It invokes the complete pipeline launcher with current environment settings.

 # ---------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
 # ---------------------------------------------------------------------------------

# =========================================================================
# End of File
# =========================================================================
