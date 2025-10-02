# -------------------------------------------------------------------------
# File: console_semantic_demo.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/console_semantic_demo.py
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
#   Provides a console-driven demonstration of the semantic VectorRAG workflow
#   by interacting with the backend search stack, showcasing detailed output and
#   optional interactive querying for exploratory testing.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Function: print_banner
# - Function: interactive_query_mode
# - Function: print_header
# - Function: print_subheader
# - Function: demonstrate_semantic_search
# - Function: demonstrate_health_check
# - Function: interactive_mode
# - Function: main (entrypoint for the demo)
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: asyncio, json, logging, sys, time, datetime, pathlib
# - Local Project Modules: app.search.tools.vector, app.core.config
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Execute this script from the backend directory using `uv run python
# console_semantic_demo.py` to explore VectorRAG capabilities through automated
# demonstrations or interactive querying.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------



"""Console demonstration for APH-IF semantic VectorRAG capabilities.

Runs automated and interactive scenarios leveraging the backend semantic search
engine, highlighting embedding usage, Neo4j integrations, and response
metadata.
"""
# __________________________________________________________________________
# Imports
#

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add backend app to path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.search.tools.vector import get_vector_engine, search_semantic_detailed, test_vector_search
from app.core.config import settings

# __________________________________________________________________________
# Global Constants / Variables

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# __________________________________________________________________________
# Standalone Function Definitions
#

# ______________________
# Console Output Helpers
#

# --------------------------------------------------------------------------------- print_banner()
def print_banner() -> None:
    """Display a formatted banner describing the semantic demo."""
    print("=" * 70)
    print("🔍 APH-IF Phase 7: Semantic Search (VectorRAG) Console Demo")
    print("=" * 70)
    print("This demo shows the VectorRAG semantic similarity search capabilities")
    print("using OpenAI embeddings and Neo4j vector indexing.")
    print()
    print("✨ Features:")
    print("   • OpenAI text-embedding-3-large (3072 dimensions)")
    print("   • Neo4j vector indexing for similarity search")
    print("   • Domain-agnostic knowledge graph querying")
    print("   • Comprehensive metadata and entity extraction")
    print("=" * 70)
# --------------------------------------------------------------------------------- end print_banner()

# --------------------------------------------------------------------------------- print_header()
def print_header(title: str) -> None:
    """Print a top-level section header for console output."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


# --------------------------------------------------------------------------------- end print_header()

# --------------------------------------------------------------------------------- print_subheader()
def print_subheader(title: str) -> None:
    """Print a subsection header dividing console sections."""
    print(f"\n{'-' * 60}")
    print(f" {title}")
    print(f"{'-' * 60}")


# --------------------------------------------------------------------------------- end print_subheader()

# ______________________
# Interactive Demo Utilities
#

# --------------------------------------------------------------------------------- interactive_query_mode()
async def interactive_query_mode() -> None:
    """Run an interactive prompt allowing users to issue semantic queries.

    Continuously reads user input from the console, executes semantic searches
    using the backend VectorRAG pipeline, and renders the resulting metadata
    until the user exits.
    """
    print("\n🤔 Interactive Query Mode")
    print("=" * 40)
    print("Enter your questions to search the knowledge graph semantically.")
    print("Type 'demo' for predefined examples, 'quit' or 'exit' to end.")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n🔍 Your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'demo':
                print("Running demonstration queries...")
                await demonstrate_semantic_search()
                continue
            
            if not user_input:
                continue
            
            print(f"\n🔄 Searching for: {user_input}")
            print("-" * 50)
            
            try:
                start_time = time.time()
                result = await search_semantic_detailed(user_input, k=8, score_threshold=0.65)
                search_time = time.time() - start_time
                
                print(f"⏱️  Search Time: {int(search_time * 1000)}ms")
                print(f"📊 Status: {'✅ Success' if 'error' not in result else '❌ Error'}")
                
                if "error" not in result:
                    print(f"📄 Sources Found: {result.get('num_sources', 0)}")
                    print(f"🏷️  Entities Found: {len(result.get('entities_found', []))}")
                    
                    # Show entities if available
                    if result.get('entities_found'):
                        entities_preview = ', '.join(result['entities_found'][:5])
                        if len(result['entities_found']) > 5:
                            entities_preview += f" (+{len(result['entities_found']) - 5} more)"
                        print(f"🔗 Key Entities: {entities_preview}")
                    
                    # Show sources if available
                    if result.get('sources'):
                        print(f"\n📚 Source Details:")
                        for i, source in enumerate(result['sources'][:3], 1):
                            print(f"   {i}. {source.get('document_title', 'Unknown Document')}")
                            if source.get('page'):
                                print(f"      Page: {source['page']}")
                            print(f"      Preview: {source.get('content_preview', '')}")
                    
                    # Main response
                    if result.get('answer'):
                        print(f"\n🎯 Response:")
                        print("-" * 30)
                        print(result['answer'])
                    else:
                        print(f"\n💡 No specific answer generated, but found {result.get('num_sources', 0)} relevant sources.")
                
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")
                logger.error(f"Interactive query '{user_input}' failed: {e}")
        
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            logger.error(f"Interactive mode error: {e}")

# --------------------------------------------------------------------------------- end interactive_query_mode()

# --------------------------------------------------------------------------------- demonstrate_semantic_search()
async def demonstrate_semantic_search() -> None:
    """Demonstrate VectorRAG behavior across several representative queries.

    Issues a curated set of semantic prompts, records timing, and prints
    entities, sources, and generated answers to showcase system capabilities.
    """
    print_header("APH-IF Phase 7: Semantic Search (VectorRAG) Demonstration")
    
    print(f"🔍 Semantic Search System Status")
    print(f"   • Embedding Model: {settings.openai_embedding_model}")
    print(f"   • Embedding Dimensions: {settings.openai_embedding_dimensions}")
    print(f"   • Neo4j Instance: {settings.get_neo4j_mode_name()}")
    print(f"   • Environment: {settings.environment_mode.value}")
    
    # Test queries demonstrating different aspects of semantic search
    test_queries = [
        {
            "query": "What safety equipment is required?",
            "description": "Equipment and safety requirements search",
            "category": "Safety & Equipment"
        },
        {
            "query": "Tell me about regulations and compliance procedures",
            "description": "Regulatory compliance and procedures",
            "category": "Compliance & Regulations"
        },
        {
            "query": "What are the training and certification requirements?",
            "description": "Training and certification topics",
            "category": "Training & Certification"
        },
        {
            "query": "Explain environmental monitoring procedures",
            "description": "Environmental and monitoring systems",
            "category": "Environmental Monitoring"
        },
        {
            "query": "What emergency response procedures are in place?",
            "description": "Emergency procedures and protocols",
            "category": "Emergency Response"
        }
    ]
    
    total_queries = len(test_queries)
    successful_queries = 0
    total_search_time = 0
    all_entities = set()
    
    for i, test_case in enumerate(test_queries, 1):
        print_subheader(f"Query {i}/{total_queries}: {test_case['category']}")
        print(f"📝 Query: {test_case['query']}")
        print(f"🎯 Purpose: {test_case['description']}")
        
        try:
            # Perform semantic search
            start_time = time.time()
            result = await search_semantic_detailed(
                test_case["query"],
                k=8,
                score_threshold=0.65  # Balanced threshold for good coverage
            )
            search_time = time.time() - start_time
            
            if "error" not in result:
                successful_queries += 1
                total_search_time += result.get("search_time_ms", 0)
                
                # Display results
                print(f"✅ Status: Success")
                print(f"⏱️  Response Time: {result['search_time_ms']}ms")
                print(f"📄 Sources Found: {result['num_sources']}")
                print(f"🏷️  Entities Found: {len(result['entities_found'])}")
                print(f"🔍 Search Parameters: k={result['parameters']['k']}, threshold={result['parameters']['score_threshold']}")
                
                # Show entities found
                if result['entities_found']:
                    entities_sample = result['entities_found'][:5]
                    print(f"🔗 Key Entities: {', '.join(entities_sample)}")
                    all_entities.update(result['entities_found'])
                
                # Show source preview
                if result['sources']:
                    print(f"\n📚 Source Examples:")
                    for j, source in enumerate(result['sources'][:2], 1):
                        print(f"   {j}. Document: {source['document_title']}")
                        if source.get('page'):
                            print(f"      Page: {source['page']}")
                        print(f"      Preview: {source['content_preview'][:150]}...")
                
                # Show full response
                if result['answer']:
                    print(f"\n🎯 Response:")
                    print("-" * 40)
                    print(result['answer'])
                else:
                    print(f"\n💡 No specific answer generated, but found {result['num_sources']} relevant sources.")
                
            else:
                print(f"❌ Status: Error")
                print(f"🚨 Error: {result['error']}")
            
        except Exception as e:
            print(f"❌ Status: Exception")
            print(f"🚨 Exception: {str(e)}")
            logger.error(f"Query failed: {e}")
        
        # Brief pause between queries
        await asyncio.sleep(1)
    
    # Summary statistics
    print_subheader("Search Performance Summary")
    print(f"✅ Successful Queries: {successful_queries}/{total_queries}")
    print(f"⏱️  Average Response Time: {int(total_search_time / max(successful_queries, 1))}ms")
    print(f"🏷️  Total Unique Entities: {len(all_entities)}")
    print(f"📊 Success Rate: {int((successful_queries / total_queries) * 100)}%")
    
    if all_entities:
        print(f"🔗 Entity Sample: {', '.join(list(all_entities)[:10])}")


# --------------------------------------------------------------------------------- end demonstrate_semantic_search()

# --------------------------------------------------------------------------------- demonstrate_health_check()
async def demonstrate_health_check() -> None:
    """Run and display semantic engine health check results."""
    print_subheader("System Health Check")
    
    try:
        engine = get_vector_engine()
        health = await engine.health_check()
        
        print(f"🏥 Overall Status: {health['status'].upper()}")
        print(f"⏱️  Health Check Time: {health['metrics']['response_time_ms']}ms")
        print(f"📊 Index Node Count: {health['metrics']['index_node_count']:,}")
        
        print("\n🔧 Component Status:")
        for component, status in health['components'].items():
            status_icon = "✅" if status == "healthy" else "⚠️" if status == "degraded" else "❌"
            print(f"   {status_icon} {component.replace('_', ' ').title()}: {status}")
        
        if health['errors']:
            print("\n🚨 Errors Detected:")
            for error in health['errors']:
                print(f"   • {error}")
    
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        logger.error(f"Health check error: {e}")


# --------------------------------------------------------------------------------- end demonstrate_health_check()

# --------------------------------------------------------------------------------- interactive_mode()
async def interactive_mode() -> None:
    """Provide a minimal interactive semantic query loop."""
    print_subheader("Interactive Query Mode")
    print("Enter your queries to test semantic search (type 'exit' to quit)")
    
    while True:
        try:
            query = input("\n🔍 Enter your query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Exiting interactive mode...")
                break
            
            if not query:
                continue
            
            print(f"🔄 Searching for: {query}")
            
            result = await search_semantic_detailed(query, k=5, score_threshold=0.65)
            
            if "error" not in result:
                print(f"✅ Found {result['num_sources']} sources in {result['search_time_ms']}ms")
                print(f"🏷️  Entities: {', '.join(result['entities_found'][:5])}")
                print(f"💡 Answer: {result['answer'][:200]}...")
            else:
                print(f"❌ Error: {result['error']}")
        
        except KeyboardInterrupt:
            print("\n👋 Exiting interactive mode...")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


# --------------------------------------------------------------------------------- end interactive_mode()

# --------------------------------------------------------------------------------- main()
async def main() -> None:
    """Entrypoint orchestrating either interactive or scripted demo runs."""
    try:
        # Print banner
        print_banner()
        
        # Ask if user wants interactive mode
        response = input("\n🎮 Would you like to try interactive mode? (y/N): ").strip().lower()
        
        if response in ['y', 'yes']:
            await interactive_query_mode()
        else:
            print("\n🤖 Running automatic demonstration...")
            await demonstrate_semantic_search()
        
    except KeyboardInterrupt:
        print("\n\n👋 Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {str(e)}")
        logger.error(f"Demo error: {e}", exc_info=True)
    finally:
        print("\n🏁 Demo session ended")

# --------------------------------------------------------------------------------- end main()

# __________________________________________________________________________
# Module Initialization / Main Execution Guard
# This code runs only when the file is executed
# --------------------------------------------------------------------------------- main_guard
if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())
# --------------------------------------------------------------------------------- end main_guard
# __________________________________________________________________________
# End of File
#

# __________________________________________________________________________
# End of File
#