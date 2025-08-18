# -------------------------------------------------------------------------
# File: check_processed_documents.py
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 08-03-2025
# File Path: data_processing/check_processed_documents.py
# ------------------------------------------------------------------------

# --- Module Objective ---
#   This module provides functionality to verify and analyze which documents
#   have been processed in the Neo4j database. It serves as a diagnostic tool
#   to check the current state of document processing, including document counts,
#   chunk statistics, entity extraction results, and overall database health.
#   The module helps users understand what has been processed and what remains
#   to be processed in the APH-IF data processing pipeline.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Function: get_neo4j_config()
# - Function: check_processed_documents()
# - Function: main()
# - Constants: None
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: os (environment variables), sys (system operations), pathlib (path handling)
# - Third-Party: neo4j (GraphDatabase for Neo4j connectivity), dotenv (environment variable loading)
# - Local Project Modules: None (standalone utility script)
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# This module is designed to be run as a standalone diagnostic script from the
# command line or integrated into monitoring workflows. It can be executed directly
# to check the current state of document processing in the Neo4j database.
# Other modules in the data processing pipeline may import functions from this
# module for status checking and validation purposes.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""

Document Processing Status Checker for APH-IF

Provides comprehensive verification and analysis of document processing status
in the Neo4j database, including document counts, chunk statistics, entity
extraction results, and processing recommendations.

"""

# =========================================================================
# Imports
# =========================================================================
# Standard library imports
#!/usr/bin/env python3
import os  # Environment variable access and system operations
import sys  # System-specific parameters and functions
from pathlib import Path  # Object-oriented filesystem path handling

# Add project root to path for module imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Third-party library imports
try:
    from neo4j import GraphDatabase  # Neo4j database connectivity and operations
    from dotenv import load_dotenv  # Environment variable loading from .env files
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

# Local application/library specific imports
# None - this is a standalone utility script

# Load environment variables from .env file
load_dotenv(project_root / '.env')


# =========================================================================
# Global Constants / Variables
# =========================================================================
# No global constants defined in this module


# =========================================================================
# Standalone Function Definitions
# =========================================================================
# These are functions that are not methods of any specific class within this module.

# --------------------------
# --- Utility Functions ---
# --------------------------

# --------------------------------------------------------------------------------- get_neo4j_config()
def get_neo4j_config():
    """Retrieves Neo4j database configuration from environment variables.

    Extracts the Neo4j connection parameters that were set by the set_environment.py
    module, providing a centralized way to access database configuration across
    the application.

    Returns:
        dict: A dictionary containing Neo4j connection parameters with keys:
            - 'uri' (str): Neo4j database URI
            - 'username' (str): Neo4j database username
            - 'password' (str): Neo4j database password

    Examples:
        >>> config = get_neo4j_config()
        >>> print(config['uri'])
        bolt://localhost:7687
    """
    return {
        'uri': os.getenv('NEO4J_URI'),
        'username': os.getenv('NEO4J_USERNAME'),
        'password': os.getenv('NEO4J_PASSWORD')
    }
# --------------------------------------------------------------------------------- end get_neo4j_config()

# ---------------------------------------------
# --- Callable Functions from other modules ---
# ---------------------------------------------

# --------------------------------------------------------------------------------- check_processed_documents()
def check_processed_documents():
    """Performs comprehensive analysis of processed documents in Neo4j database.

    Connects to the Neo4j database and analyzes the current state of document
    processing, including document counts, chunk statistics, entity extraction
    results, and relationship counts. Also compares processed documents against
    available PDF files to identify unprocessed documents.

    The function provides detailed output including:
    - List of all processed documents with metadata
    - Chunk counts per document
    - Entity counts per document
    - Overall database statistics
    - Comparison with available PDF files
    - Processing recommendations

    Returns:
        bool: True if the analysis completed successfully, False if errors occurred.

    Raises:
        Exception: If database connection fails or query execution encounters errors.

    Examples:
        >>> success = check_processed_documents()
        Checking Processed Documents in Neo4j Database
        ============================================================
        Documents in Database:
        ----------------------------------------
        1. sample_document.pdf
           Path: /path/to/sample_document.pdf
           Pages: 10
           Created: 2025-01-15T10:30:00Z
    """
    print("Checking Processed Documents in Neo4j Database")
    print("=" * 60)
    
    try:
        # Get Neo4j configuration
        config = get_neo4j_config()
        app_env = os.getenv('APP_ENV', 'development')
        force_test_db = os.getenv('FORCE_TEST_DB', 'false').lower() == 'true'

        print(f"Environment: {app_env}")
        print(f"Test DB: {force_test_db}")
        print(f"Neo4j URI: {config['uri']}")
        print()
        
        # Connect to Neo4j
        driver = GraphDatabase.driver(config['uri'], auth=(config['username'], config['password']))
        
        with driver.session() as session:
            # Check documents
            print("Documents in Database:")
            print("-" * 40)
            
            result = session.run("""
                MATCH (d:Document)
                RETURN d.name as document_name, 
                       d.file_path as file_path,
                       d.total_pages as total_pages,
                       d.created_at as created_at
                ORDER BY d.name
            """)
            
            documents = list(result)
            
            if documents:
                for i, record in enumerate(documents, 1):
                    print(f"{i}. {record['document_name']}")
                    print(f"   Path: {record['file_path']}")
                    print(f"   Pages: {record['total_pages']}")
                    print(f"   Created: {record['created_at']}")
                    print()
            else:
                print("❌ No documents found in database!")
                return
            
            # Check chunks per document
            print("Chunks per Document:")
            print("-" * 40)
            
            result = session.run("""
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                RETURN d.name as document_name, 
                       count(c) as chunk_count
                ORDER BY d.name
            """)
            
            chunk_counts = list(result)
            for record in chunk_counts:
                print(f"{record['document_name']}: {record['chunk_count']:,} chunks")
            
            # Check entities per document
            print("\nEntities per Document:")
            print("-" * 40)
            
            result = session.run("""
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:HAS_ENTITY]->(e:Entity)
                RETURN d.name as document_name, 
                       count(DISTINCT e) as entity_count
                ORDER BY d.name
            """)
            
            entity_counts = list(result)
            for record in entity_counts:
                print(f"{record['document_name']}: {record['entity_count']:,} entities")
            
            # Overall statistics
            print("\nOverall Database Statistics:")
            print("-" * 40)
            
            stats_query = """
                MATCH (d:Document) WITH count(d) as doc_count
                MATCH (c:Chunk) WITH doc_count, count(c) as chunk_count
                MATCH (e:Entity) WITH doc_count, chunk_count, count(e) as entity_count
                MATCH ()-[r:RELATED_TO]->() WITH doc_count, chunk_count, entity_count, count(r) as relationship_count
                RETURN doc_count, chunk_count, entity_count, relationship_count
            """
            
            result = session.run(stats_query)
            stats = result.single()
            
            if stats:
                print(f"Documents: {stats['doc_count']:,}")
                print(f"Chunks: {stats['chunk_count']:,}")
                print(f"Entities: {stats['entity_count']:,}")
                print(f"Relationships: {stats['relationship_count']:,}")
            
        driver.close()
        
        # Check available PDF files
        print("\nAvailable PDF Files:")
        print("-" * 40)
        
        pdf_dir = Path("processing/data_pdf")
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        processed_docs = [doc['document_name'] for doc in documents]
        
        for pdf_file in sorted(pdf_files):
            status = "PROCESSED" if pdf_file.name in processed_docs else "❌ NOT PROCESSED"
            print(f"{status}: {pdf_file.name}")
        
        print(f"\nTotal PDF files: {len(pdf_files)}")
        print(f"Processed: {len(processed_docs)}")
        print(f"Remaining: {len(pdf_files) - len(processed_docs)}")
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        return False

    return True
# --------------------------------------------------------------------------------- end check_processed_documents()

# --------------------------------------------------------------------------------- main()
def main():
    """Main entry point for the document processing status checker.

    Orchestrates the document processing analysis and provides user guidance
    based on the results. Executes the check_processed_documents function and
    displays recommendations for further processing if the analysis succeeds.

    Returns:
        int: Exit code (0 for success, 1 for failure) suitable for command-line usage.

    Examples:
        >>> exit_code = main()
        Checking Processed Documents in Neo4j Database
        ============================================================
        ...
        To process all documents, update your .env file:
           MAX_DOCS=0  # 0 = no limit, process all documents
           MAX_PAGES=0 # 0 = no limit, process all pages
    """
    success = check_processed_documents()
    
    if success:
        print("\nTo process all documents, update your .env file:")
        print("   MAX_DOCS=0  # 0 = no limit, process all documents")
        print("   MAX_PAGES=0 # 0 = no limit, process all pages")
        print("\nThen run:")
        print("   uv run python run_data_processing_with_monitoring.py")
    
    return 0 if success else 1
# --------------------------------------------------------------------------------- end main()


# =========================================================================
# Module Initialization / Main Execution Guard
# =========================================================================
# This block runs only when the file is executed directly, not when imported.
# It serves as the entry point for command-line execution of the document
# processing status checker.

if __name__ == "__main__":
    # --- Direct Execution Entry Point ---
    # Execute the main function and exit with appropriate status code
    # This allows the script to be used in shell scripts and automation
    sys.exit(main())
# ---------------------------------------------------------------------------------

# =========================================================================
# End of File
# =========================================================================
