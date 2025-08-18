#!/usr/bin/env python3
# -------------------------------------------------------------------------
# File: test_complete_graph_pipeline.py
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 2025-08-12
# File Path: data_processing/tests/test_complete_graph_pipeline.py
# ------------------------------------------------------------------------

# --- Module Objective ---
#   End-to-end tests validating initial graph build and relationship
#   augmentation with environment-safe patterns.
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - APH-IF  
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG – Intelligent Fusion (APH-IF)
# -------------------------------------------------------------------------

"""
APH-IF Complete Graph Pipeline Test

Purpose
-------
Validate the complete knowledge graph pipeline:
1. Initial graph build (documents, chunks, entities)
2. Relationship augmentation (entity/document relationships)
3. End-to-end validation of graph structure and counts

Test Safety
-----------
- Forces FORCE_TEST_DB=true during testing
- Restores FORCE_TEST_DB=false after completion
- Uses isolated test database to prevent data corruption
- Processes exactly one document from test data

Requirements
------------
- Test document in data_processing/processing/data_pdf_test/
- OpenAI API key configured
- Test database credentials in .env

How it works
------------
1) setUpClass enables test DB mode via `EnvManager`, verifies the test
   database URI, prepares a temporary directory with a single PDF copied
   from `processing/data_pdf_test/`, and configures environment variables to
   speed up processing (e.g., larger chunk size, less frequent extraction).
2) test_01 runs the initial build using `run_initial_graph_build()` and then
   validates the presence of Documents, Chunks, and Entities.
3) test_02 runs relationship augmentation and performs a lightweight check
   that LLM-labeled relationships can be created (without asserting counts).
4) test_03 collects graph statistics and asserts minimum content exists.
5) test_04 exercises the unified launcher helpers and validates generated
   configuration in test mode.
6) tearDownClass deletes the temporary directory and restores test DB state.

Notes
-----
- Relationship creation volume may vary by content and thresholds. Tests
  avoid brittle assertions on exact counts and focus on safety and sanity.
- End-to-end run can be time-consuming because it processes a real PDF; see
  prints for progress. Adjust env knobs to iterate faster locally.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List

# Add the parent directories to the path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))  # data_processing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # APH-IF-Dev

from processing.run_initial_graph_build import run_initial_graph_build
from processing.launch_data_processing import main as launch_pipeline
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


# --------------------------------------------------------------------------------- TestCompleteGraphPipeline
class TestCompleteGraphPipeline(unittest.TestCase):
    """End-to-end tests for the APH-IF data processing pipeline.

    Purpose:
        Validate that the initial graph build, relationship augmentation,
        and overall graph structure function correctly under test settings.

    Class Attributes:
        original_force_test_db (str): Original FORCE_TEST_DB env value
            captured in setUpClass() for later restoration.
        test_pdf_dir (str): Temporary directory path containing a single
            test PDF used during the run.

    Methods:
        setUpClass(): Configure test environment (enable test DB, verbose).
        tearDownClass(): Cleanup env and restore FORCE_TEST_DB.
        _setup_test_document() -> str: Prepare a temp directory with one PDF.
        setUp(): Per-test verification that test DB is active.
        test_01_initial_graph_build(): Run and validate initial build.
        test_02_relationship_augmentation(): Run and validate augmentation.
        test_03_complete_graph_validation(): Validate graph counts and types.
        test_04_pipeline_launcher_integration(): Validate launcher helpers.
        _validate_initial_graph(): Assert minimal structure exists.
        _validate_relationships(): Inspect created relationships.
        _get_graph_statistics() -> Dict[str, int]: Aggregate graph stats.
    
    Additional Notes:
        - These tests intentionally prioritize environment safety and
          minimal correctness checks over strict performance.
        - Relationship augmentation relies on similarity thresholds and the
          content of the test PDF; counts can vary.
    """ 
   
    
    # --------------------------------------------------------------------------------- setUpClass()
    @classmethod
    def setUpClass(cls):
        """Set up test environment before all tests.

        Steps:
            1) Capture original FORCE_TEST_DB and enable test DB mode.
            2) Turn on verbose logging for better visibility during CI.
            3) Verify the Neo4j URI corresponds to a test instance.
            4) Prepare a temp directory containing one PDF copied from
               `processing/data_pdf_test/` for faster, deterministic runs.
            5) Export environment knobs optimized for speed.
        """
        print("\nSetting up test environment...")
        
        # Store original FORCE_TEST_DB state
        cls.original_force_test_db = os.getenv('FORCE_TEST_DB', 'false')
        print(f"  Original FORCE_TEST_DB: {cls.original_force_test_db}")
        
        # Force test database mode
        os.environ['FORCE_TEST_DB'] = 'true'

        # Enable verbose logging for debugging
        os.environ['VERBOSE'] = 'true'

        # Verify we're in test mode
        config = get_neo4j_config()
        if "test" not in config['uri'].lower() and "7a25197a" not in config['uri']:
            raise RuntimeError(f"Not using test database! URI: {config['uri']}")
        
        print(f"✅ Test database confirmed: {config.uri}")
        
        # Set up test document
        cls.test_pdf_dir = cls._setup_test_document()
        print(f"✅ Test document ready: {cls.test_pdf_dir}")
        
        # Configure environment for testing (matches launch_data_processing.py test defaults)
        os.environ.update({
            'PDF_DIR': cls.test_pdf_dir,
            'CLEAR_DB': 'true',
            'MAX_DOCS': '1',
            'CHUNK_SIZE_CHARS': '3000',  # Larger chunks = fewer total chunks
            'EXTRACT_EVERY_N_CHUNKS': '5',  # Extract from every 5th chunk for speed
            'MAX_PAGES': '10',  # Process only first 10 pages for testing
            'OPENAI_MODEL': 'gpt-5-nano',
            'VERBOSE': 'true',
            # Relationship augmentation settings
            'AUGMENT_ENTITY': 'true',
            'AUGMENT_DOCUMENT': 'false',
            'ENTITY_SIM_CUTOFF': '0.1',  # Lower threshold for testing
            'ENTITY_LIMIT': '10',  # Limit for faster testing
            'DRY_RUN': 'false'
        })

        print("✅ Test environment setup complete")
    # --------------------------------------------------------------------------------- end setUpClass()


    # --------------------------------------------------------------------------------- tearDownClass()
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests.

        Steps:
            1) Remove the temporary directory with the test PDF.
            2) Restore the original FORCE_TEST_DB value and disable verbose.
            3) Print final cleanup status for traceability.
        """
        print("\n🧹 Cleaning up test environment...")
        
        # Remove temporary test directory
        if hasattr(cls, 'test_pdf_dir') and cls.test_pdf_dir:
            try:
                shutil.rmtree(cls.test_pdf_dir)
                print(f"  Removed temporary directory: {cls.test_pdf_dir}")
            except Exception as e:
                print(f"  Warning: Could not remove temp dir: {e}")
        
        # Restore original FORCE_TEST_DB state
        os.environ['FORCE_TEST_DB'] = cls.original_force_test_db
        print(f"✅ Restored FORCE_TEST_DB to: {cls.original_force_test_db}")

        # Disable verbose logging
        os.environ['VERBOSE'] = 'false'
        
        print("✅ Test environment cleanup complete")
    # --------------------------------------------------------------------------------- end tearDownClass()
    
    # --------------------------------------------------------------------------------- _setup_test_document()
    @classmethod
    def _setup_test_document(cls) -> str:
        """Set up a temporary directory with one test document.

        Returns:
            str: Filesystem path to the temporary directory housing a single
                 PDF file for the test run.
        """
        # Find test documents
        test_data_dir = Path(__file__).parent.parent / "processing" / "data_pdf_test"
        
        if not test_data_dir.exists():
            raise FileNotFoundError(f"Test data directory not found: {test_data_dir}")
        
        # Get the first PDF file
        pdf_files = list(test_data_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {test_data_dir}")
        
        test_pdf = pdf_files[0]  # Use first PDF
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="aph_if_pipeline_test_")
        temp_pdf_path = Path(temp_dir) / test_pdf.name
        
        # Copy test document to temporary directory
        shutil.copy2(test_pdf, temp_pdf_path)
        
        return temp_dir
    # --------------------------------------------------------------------------------- end _setup_test_document()
    
    # --------------------------------------------------------------------------------- setUp()
    def setUp(self):
        """Set up before each test.

        Verifies that test DB mode is still active by inspecting the
        configured Neo4j URI. Fails fast if not in test mode.
        """
        # Verify we're still in test mode
        config = get_neo4j_config()
        if "test" not in config['uri'].lower() and "7a25197a" not in config['uri']:
            self.fail(f"Test database not active! URI: {config['uri']}")
    # --------------------------------------------------------------------------------- end setUp()
    
    # --------------------------------------------------------------------------------- test_01_initial_graph_build()
    def test_01_initial_graph_build(self):
        """Test the initial graph building process using the new runner.

        Behavior:
            - Invokes `run_initial_graph_build()` which reads config from env.
            - Asserts success (return code 0) and validates minimal graph
              structure via `_validate_initial_graph()`.
        """
        print("\nTesting initial graph build...")
        print("Note: This test processes the complete document and may take 30-60 minutes")
        print("Progress will be shown in real-time...")

        try:
            # Use the new run_initial_graph_build function
            result = run_initial_graph_build()

            if result != 0:
                self.fail(f"Initial graph build failed with return code: {result}")

            print("✅ Initial graph build completed successfully")

            # Validate the graph was created
            self._validate_initial_graph()

        except Exception as e:
            self.fail(f"Initial graph build failed: {e}")
    # --------------------------------------------------------------------------------- end test_01_initial_graph_build()
    
    # --------------------------------------------------------------------------------- test_02_relationship_augmentation()
    def test_02_relationship_augmentation(self):
        """Test the relationship augmentation process.

        Behavior:
            - Configures entity-only augmentation with a lower similarity
              cutoff and a small limit for speed.
            - Runs the augmentation orchestrator and performs a lightweight
              sanity check via `_validate_relationships()`.
        """
        print("\n🔗 Testing relationship augmentation...")
        
        try:
            # Set up environment for augmentation
            os.environ.update({
                'AUGMENT_ENTITY': 'true',
                'AUGMENT_DOCUMENT': 'false',  # Focus on entities for testing
                'ENTITY_SIM_CUTOFF': '0.1',  # Lower threshold for testing
                'ENTITY_LIMIT': '10',  # Limit for faster testing
                'DRY_RUN': 'false'
            })
            
            # Run relationship augmentation
            run_augmentation()
            
            print("✅ Relationship augmentation completed successfully")
            
            # Validate relationships were created
            self._validate_relationships()
            
        except Exception as e:
            self.fail(f"Relationship augmentation failed: {e}")
    # --------------------------------------------------------------------------------- end test_02_relationship_augmentation()
    
    # --------------------------------------------------------------------------------- test_03_complete_graph_validation()
    def test_03_complete_graph_validation(self):
        """Test the complete graph structure and content.

        Asserts minimum viable content exists across major node and
        relationship categories, then prints summary counts for visibility.
        """
        print("\n✅ Testing complete graph validation...")

        try:
            stats = self._get_graph_statistics()

            # Validate minimum expected content
            self.assertGreater(stats['documents'], 0, "No documents found in graph")
            self.assertGreater(stats['chunks'], 0, "No chunks found in graph")
            self.assertGreater(stats['entities'], 0, "No entities found in graph")
            self.assertGreater(stats['total_relationships'], 0, "No relationships found in graph")

            # Validate specific relationship types
            self.assertGreater(stats['has_chunk_rels'], 0, "No HAS_CHUNK relationships found")
            self.assertGreater(stats['has_entity_rels'], 0, "No HAS_ENTITY relationships found")

            print(f"✅ Graph validation passed:")
            print(f"  Documents: {stats['documents']}")
            print(f"  Chunks: {stats['chunks']}")
            print(f"  Entities: {stats['entities']}")
            print(f"  Total Relationships: {stats['total_relationships']}")
            print(f"  HAS_CHUNK: {stats['has_chunk_rels']}")
            print(f"  HAS_ENTITY: {stats['has_entity_rels']}")
            print(f"  Entity Relationships: {stats['entity_relationships']}")

        except Exception as e:
            self.fail(f"Complete graph validation failed: {e}")
    # --------------------------------------------------------------------------------- end test_03_complete_graph_validation()
    
    # --------------------------------------------------------------------------------- test_04_pipeline_launcher_integration()
    def test_04_pipeline_launcher_integration(self):
        """Test that the new pipeline launcher works correctly.

        Validates that helper functions in `processing.launch_data_processing`
        generate an appropriate test-mode configuration and environment
        validation passes when prerequisites are present.
        """
        print("\nTesting pipeline launcher integration...")

        try:
            # Test that we can import and validate the launcher
            from processing.launch_data_processing import (
                get_pipeline_config, validate_pipeline_environment
            )

            # Create a mock args object for testing
            class MockArgs:
                def __init__(self):
                    self.test_mode = True
                    self.dry_run = False
                    self.initial_only = False
                    self.augmentation_only = False
                    self.skip_initial = False
                    self.skip_augmentation = False

            mock_args = MockArgs()

            # Test configuration generation
            config = get_pipeline_config(mock_args)

            # Validate expected configuration keys
            expected_keys = [
                'APP_ENV', 'FORCE_TEST_DB', 'VERBOSE',
                'PDF_DIR', 'CLEAR_DB', 'MAX_DOCS',
                'CHUNK_SIZE_CHARS', 'EXTRACT_EVERY_N_CHUNKS',
                'AUGMENT_ENTITY', 'ENTITY_SIM_CUTOFF', 'ENTITY_LIMIT'
            ]

            for key in expected_keys:
                self.assertIn(key, config, f"Missing configuration key: {key}")

            # Validate test mode settings
            self.assertEqual(config['FORCE_TEST_DB'], 'true', "Test mode not properly set")
            self.assertEqual(config['CHUNK_SIZE_CHARS'], '3000', "Test chunk size not set")
            self.assertEqual(config['EXTRACT_EVERY_N_CHUNKS'], '5', "Test extraction frequency not set")

            # Test environment validation
            is_valid = validate_pipeline_environment()
            self.assertTrue(is_valid, "Pipeline environment validation failed")

            print("✅ Pipeline launcher integration test passed")

        except Exception as e:
            self.fail(f"Pipeline launcher integration test failed: {e}")
    # --------------------------------------------------------------------------------- end test_04_pipeline_launcher_integration()
    
    # --------------------------------------------------------------------------------- _validate_initial_graph()
    def _validate_initial_graph(self):
        """Validate the initial graph structure.

        Checks that Documents, Chunks, and Entities exist after the initial
        build. Prints a concise summary for debugging.
        """
        config = get_neo4j_config()

        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(config['uri'], auth=(config['username'], config['password']))
        
        try:
            with driver.session() as session:
                # Check for documents
                doc_result = session.run("MATCH (d:Document) RETURN count(d) as count")
                doc_count = doc_result.single()["count"]
                self.assertGreater(doc_count, 0, "No documents created")
                
                # Check for chunks
                chunk_result = session.run("MATCH (c:Chunk) RETURN count(c) as count")
                chunk_count = chunk_result.single()["count"]
                self.assertGreater(chunk_count, 0, "No chunks created")
                
                # Check for entities
                entity_result = session.run("MATCH (e:Entity) RETURN count(e) as count")
                entity_count = entity_result.single()["count"]
                self.assertGreater(entity_count, 0, "No entities created")
                
                print(f"  Initial graph: {doc_count} docs, {chunk_count} chunks, {entity_count} entities")
                
        finally:
            driver.close()
    # --------------------------------------------------------------------------------- end _validate_initial_graph()
    
    # --------------------------------------------------------------------------------- _validate_relationships()
    def _validate_relationships(self):
        """Validate that relationships were created.

        Notes:
            - Relationship counts are not asserted due to variability across
              documents and thresholds. This function prints counts to aid
              debugging without introducing brittle expectations.
        """
        config = get_neo4j_config()

        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(config['uri'], auth=(config['username'], config['password']))
        
        try:
            with driver.session() as session:
                # Check for entity relationships
                entity_rel_result = session.run(
                    "MATCH ()-[r:RELATED_TO]->() WHERE r.source = 'llm' RETURN count(r) as count"
                )
                entity_rel_count = entity_rel_result.single()["count"]
                
                print(f"  Entity relationships created: {entity_rel_count}")
                
                # Note: We don't require relationships to be created as it depends on
                # the similarity threshold and content of the test document
                
        finally:
            driver.close()
    # --------------------------------------------------------------------------------- end _validate_relationships()
    
    # --------------------------------------------------------------------------------- _get_graph_statistics()
    def _get_graph_statistics(self) -> Dict[str, int]:
        """Get comprehensive graph statistics.

        Returns:
            Dict[str, int]: A dictionary with counts for:
                - documents: number of `Document` nodes
                - chunks: number of `Chunk` nodes
                - entities: number of `Entity` nodes
                - total_relationships: total number of relationships
                - has_chunk_rels: count of `HAS_CHUNK` relationships
                - has_entity_rels: count of `HAS_ENTITY` relationships
                - entity_relationships: count of `RELATED_TO` relationships
        """
        config = get_neo4j_config()

        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(config['uri'], auth=(config['username'], config['password']))
        
        try:
            with driver.session() as session:
                stats = {}
                
                # Count nodes
                stats['documents'] = session.run("MATCH (d:Document) RETURN count(d) as count").single()["count"]
                stats['chunks'] = session.run("MATCH (c:Chunk) RETURN count(c) as count").single()["count"]
                stats['entities'] = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
                
                # Count relationships
                stats['total_relationships'] = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
                stats['has_chunk_rels'] = session.run("MATCH ()-[r:HAS_CHUNK]->() RETURN count(r) as count").single()["count"]
                stats['has_entity_rels'] = session.run("MATCH ()-[r:HAS_ENTITY]->() RETURN count(r) as count").single()["count"]
                stats['entity_relationships'] = session.run("MATCH ()-[r:RELATED_TO]->() RETURN count(r) as count").single()["count"]
                
                return stats
                
        finally:
            driver.close()
    # --------------------------------------------------------------------------------- end _get_graph_statistics()
       
# --------------------------------------------------------------------------------- end TestCompleteGraphPipeline   

# --------------------------------------------------------------------------------- run_tests()
def run_tests():
    """Run the complete pipeline tests.

    Behavior:
        - Optionally loads a project `.env` if available for local runs.
        - Verifies the presence of `OPENAI_API_KEY` before executing tests.
        - Runs the `TestCompleteGraphPipeline` suite at verbosity 2 and
          returns a conventional exit code (0=pass, 1=fail).

    Returns:
        int: 0 when all tests pass; 1 when any test fails.
    """
    print("APH-IF Complete Graph Pipeline Test Suite")
    print("=" * 80)

    # Load environment variables first
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            print(f"✅ Loaded environment from {env_path}")
        else:
            print(f"⚠️  No .env file found at {env_path}")
    except ImportError:
        print("⚠️  python-dotenv not available, using system environment")

    # Verify OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not found. Please set it in your environment.")
        return 1
    
    # Run the test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCompleteGraphPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\nALL PIPELINE TESTS PASSED!")
        print("✅ Complete graph creation pipeline is working correctly")
        return 0
    else:
        print(f"\n❌ {len(result.failures + result.errors)} test(s) failed")
        return 1

# --------------------------------------------------------------------------------- end run_tests()


# =========================================================================
# Module Initialization / Main Execution Guard
# =========================================================================
# This block runs only when the file is executed directly, not when imported.
# It launches the unittest suite for the complete pipeline.

# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(run_tests())
# ---------------------------------------------------------------------------------
