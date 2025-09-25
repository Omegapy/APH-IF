# -------------------------------------------------------------------------
# File: console_traversal_demo.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/console_traversal_demo.py
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
#   Provides a console-driven demonstration for the LLM Structural Cypher graph
#   traversal workflow, highlighting automated demos, interactive querying, and
#   diagnostic output for APH-IF backend capabilities.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Function: extract_error_message
# - Function: print_banner
# - Function: get_demo_queries
# - Function: get_max_results
# - Function: check_llm_engine_availability
# - Function: run_traversal_query
# - Function: format_result_output
# - Function: demo_traversal_search
# - Function: interactive_mode
# - Function: main (entrypoint orchestrating the demo)
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: asyncio, os, sys, logging, pathlib
# - Typing: Any, Dict, List, Optional, Tuple
# - Local Project Modules: app.core.config, app.schema, app.search.tools.cypher,
#   app.search.tools.llm_structural_cypher
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Execute from `backend/` via `uv run python console_traversal_demo.py`. Supports
# non-interactive mode by setting `MAX_RESULTS` environment variable for CI or
# scripted runs.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""Console demo showcasing APH-IF LLM Structural Cypher traversal capabilities.

Provides automated and interactive flows that transform natural-language prompts
into validated Cypher queries, surfacing detailed metadata and performance
diagnostics for the APH-IF backend traversal stack.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple

# __________________________________________________________________________
# Imports
#

# Add backend to path for imports
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Set up logging to show the process
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# __________________________________________________________________________
# Global Constants / Variables
#

# --------------------------------------------------------------------------------- extract_error_message()
def extract_error_message(result: Dict[str, Any]) -> Optional[str]:
    """Extract an informative error message from traversal results.

    Args:
        result: The response payload returned by the traversal engine.

    Returns:
        A string describing the encountered error, or ``None`` when the result
        does not represent an error condition.
    """

    error_msg = result.get("error")
    if error_msg:
        return error_msg

    metadata = result.get("metadata", {})
    if metadata.get("error_message"):
        return metadata["error_message"]
    if metadata.get("error_type"):
        return metadata["error_type"]
    metadata_error = metadata.get("error")
    if isinstance(metadata_error, str):
        return metadata_error

    answer = result.get("answer", "")
    if isinstance(answer, str) and answer.startswith("Error:"):
        return answer

    return None
# --------------------------------------------------------------------------------- end extract_error_message()

# __________________________________________________________________________
# Standalone Function Definitions
#

# ______________________
# Console Output Helpers
#

# --------------------------------------------------------------------------------- print_banner()
def print_banner() -> None:
    """Display an overview banner describing the traversal demo."""
    print("=" * 70)
    print("🚀 APH-IF LLM Structural Cypher Console Demo")
    print("=" * 70)
    print("This demo showcases APH-IF's advanced graph traversal engine:")
    print("🧠 LLM Structural Cypher - Natural language to Cypher generation")
    print()
    print("✨ Features:")
    print("   • Token-efficient prompting using structural schema summaries")
    print("   • Comprehensive validation with automatic query fixes")
    print("   • Detailed metadata display (tokens, model, validation, fixes)")
    print("   • Interactive and non-interactive modes")
    print("   • Domain-agnostic knowledge graph traversal capabilities")
    print("   • Comprehensive error handling and performance monitoring")
    print("=" * 70)
# --------------------------------------------------------------------------------- end print_banner()

# --------------------------------------------------------------------------------- get_demo_queries()
def get_demo_queries() -> list[str]:
    """Return natural-language prompts executed during the demo run."""
    return [
        "What are the training and certification requirements?",
        "Tell me about regulations and compliance procedures",
        "Explain environmental monitoring procedures",
        "What safety equipment is required for operations?",
        "Find information about standards and specifications",
        "What are the operational requirements and guidelines?",
    ]
# --------------------------------------------------------------------------------- end get_demo_queries()

# ______________________
# Input Validation Utilities
#

# --------------------------------------------------------------------------------- get_max_results()
def get_max_results() -> int:
    """Request a result limit from stdin, enforcing safe bounds.

    Returns:
        int: Maximum number of traversal results per query, clamped between 1
        and 1000.
    """
    max_results_input = input("Max results per query (1-1000, default 25): ").strip()
    try:
        max_results = int(max_results_input) if max_results_input else 25
        max_results = max(1, min(1000, max_results))
    except ValueError:
        print("⚠️ Invalid number, using default (25)")
        max_results = 25

    print(f"✅ Max results set to: {max_results}")
    return max_results
# --------------------------------------------------------------------------------- end get_max_results()

# ______________________
# Engine Availability
#

# --------------------------------------------------------------------------------- check_llm_engine_availability()
def check_llm_engine_availability() -> tuple[bool, Optional[str]]:
    """Determine whether the LLM Structural Cypher engine is ready for use.

    Returns:
        tuple[bool, Optional[str]]: A flag indicating availability and an
        optional diagnostic message explaining failures.
    """
    try:
        from app.core.config import settings

        if not settings.use_llm_structural_cypher:
            return False, "Feature flag USE_LLM_STRUCTURAL_CYPHER is disabled"

        from app.schema import get_schema_manager

        schema_manager = get_schema_manager()
        if not schema_manager.get_structural_summary_dict():
            return False, "LLM engine unavailable: structural schema not loaded"

        from app.search.tools.llm_structural_cypher import get_llm_structural_cypher_engine

        get_llm_structural_cypher_engine  # noqa: F401
        return True, None

    except ImportError as e:
        return False, f"Import error: {str(e)}"
    except Exception as e:
        return False, f"Availability check failed: {str(e)}"
# --------------------------------------------------------------------------------- end check_llm_engine_availability()

# ______________________
# Traversal Execution Helpers
#

# --------------------------------------------------------------------------------- run_traversal_query()
async def run_traversal_query(user_query: str, max_results: int) -> Dict[str, Any]:
    """Execute a traversal request using the LLM Structural Cypher engine.

    Args:
        user_query: Natural-language prompt supplied by the caller.
        max_results: Maximum number of graph results to retrieve.

    Returns:
        dict[str, Any]: Detailed engine response including generated Cypher,
        metadata, and answer content.
    """
    from app.search.tools.cypher import query_knowledge_graph_llm_structural_detailed

    return await query_knowledge_graph_llm_structural_detailed(user_query, max_results)
# --------------------------------------------------------------------------------- end run_traversal_query()

# --------------------------------------------------------------------------------- format_result_output()
def format_result_output(result: Dict[str, Any], query_num: Optional[int] = None) -> None:
    """Render traversal results alongside human-readable metadata.

    Args:
        result: Response payload produced by the traversal engine.
        query_num: Optional index used for labeling demo queries.
    """
    metadata = result.get("metadata", {})
    search_method = metadata.get("search_method", "llm_structural_cypher")

    method_labels = {
        "llm_structural_cypher": "🧠 LLM Structural Cypher",
    }
    method_display = method_labels.get(search_method, f"❓ {search_method.replace('_', ' ').title()}")

    if query_num:
        print(f"\n📝 Demo Query {query_num}")
    print(f"Search Method: {method_display}")

    if result.get("cypher_query"):
        print(f"Generated Cypher: {result['cypher_query']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Response Time: {result['response_time_ms']}ms")

    if metadata.get("tokens_used"):
        print(f"Tokens Used: {metadata['tokens_used']}")
    if metadata.get("model_used"):
        print(f"Model Used: {metadata['model_used']}")
    if "validation_issues" in metadata:
        issues = metadata["validation_issues"]
        issues_count = issues if isinstance(issues, int) else len(metadata.get("validation_issues", []))
        print(f"Validation Issues: {issues_count}")
    if "fixes_applied" in metadata:
        fixes = metadata["fixes_applied"]
        fixes_count = fixes if isinstance(fixes, int) else len(metadata.get("fixes_applied", []))
        print(f"Fixes Applied: {fixes_count}")
    if metadata.get("result_count"):
        print(f"Result Count: {metadata['result_count']}")

    error_msg = extract_error_message(result)
    if error_msg:
        print(f"❌ Query failed: {error_msg}")
    else:
        print("\n🎯 Response:")
        print("-" * 30)
        print(result["answer"])
# --------------------------------------------------------------------------------- end format_result_output()

# --------------------------------------------------------------------------------- demo_traversal_search()
async def demo_traversal_search(max_results: int) -> bool:
    """Execute the scripted demonstration across predefined traversal queries.

    Args:
        max_results: Maximum number of graph results requested per query.

    Returns:
        bool: ``True`` when the demo completes successfully; ``False`` if the
        engine is unavailable or an error interrupts execution.
    """
    
    try:
        # LLM engine availability check
        print("\n🔍 System Health Check...")
        llm_available, llm_message = check_llm_engine_availability()
        llm_icon = "✅" if llm_available else "❌"
        print(f"🧠 LLM Structural Cypher: {llm_icon} {'Available' if llm_available else f'Unavailable - {llm_message}'}")
        
        if not llm_available:
            print("❌ LLM engine is not available. Cannot proceed with demo.")
            print("Please check your configuration and try again.")
            return False
        
        print(f"\n🚀 Running demo with LLM Structural Cypher engine (max {max_results} results)")
        
        print("\n" + "=" * 70)
        print("🎯 Starting LLM Structural Cypher Demo")
        print("=" * 70)
        
        # Get demo queries
        demo_queries = get_demo_queries()
        
        for i, query in enumerate(demo_queries, 1):
            print(f"User Input: \"{query}\"")
            print("-" * 50)
            
            try:
                print("🔄 Processing query...")
                result = await run_traversal_query(query, max_results)
                format_result_output(result, i)
                
            except Exception as e:
                print(f"❌ Query {i} failed: {str(e)}")
                logger.error(f"Query '{query}' failed: {e}")
            
            print("\n" + "-" * 70)
            
            # Add a small delay between queries for readability
            if i < len(demo_queries):
                await asyncio.sleep(1)
        
        print("\n" + "=" * 70)
        print("✅ LLM Structural Cypher Demo Completed Successfully!")
        print("=" * 70)
        print("\n💡 Key Observations:")
        print("  1. 🧠 LLM Structural Cypher provides natural language understanding")
        print("  2. 🔍 Token-efficient prompting using structural schema summaries")
        print("  3. ✅ Comprehensive validation with automatic fixes")
        print("  4. 📊 Detailed metadata including tokens, model, and validation info")
        print("  5. 🛡️ Robust error handling with graceful degradation")
        print("\n🚀 LLM Structural Cypher Engine is fully operational!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you're running this from the backend directory.")
        return False
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True
# --------------------------------------------------------------------------------- end demo_traversal_search()

# --------------------------------------------------------------------------------- interactive_mode()
async def interactive_mode(max_results: int) -> None:
    """Run an interactive loop issuing traversal queries from console input.

    Args:
        max_results: Maximum number of graph results requested per query.
    """
    print("🧠 LLM Structural Cypher Interactive Mode")
    print("Type 'quit' or 'exit' to end.")
    print()
    
    while True:
        try:
            user_input = input("\n🤔 Your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
            
            print("🔄 Processing query...")
            
            try:
                result = await run_traversal_query(user_input, max_results)
                format_result_output(result)
                
            except Exception as e:
                print(f"❌ Query failed: {str(e)}")
                            
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
# --------------------------------------------------------------------------------- end interactive_mode()

# --------------------------------------------------------------------------------- main()
async def main() -> bool:
    """Coordinate demo execution, supporting interactive and CI workflows.

    Returns:
        bool: ``True`` if the demo or interactive session completes without
        fatal errors, otherwise ``False``.
    """
    print_banner()
    
    # Check for non-interactive mode (CI support)
    max_results_env = os.getenv('MAX_RESULTS', '')
    
    if max_results_env:
        # Non-interactive mode
        try:
            max_results = max(1, min(1000, int(max_results_env)))
        except ValueError:
            max_results = 25
        print(f"🤖 Non-interactive mode: {max_results} max results")
        interactive = False
    else:
        # Interactive selection
        max_results = get_max_results()
        response = input("\n🎮 Interactive mode? (y/N): ").strip().lower()
        interactive = response in ['y', 'yes']
    
    # LLM availability check
    available, message = check_llm_engine_availability()
    if not available:
        print(f"❌ LLM engine unavailable: {message}")
        print("Please check your configuration and try again.")
        return False
    
    try:
        if interactive:
            print("\n🎮 Starting Interactive Mode with LLM Structural Cypher...")
            print("=" * 70)
            await interactive_mode(max_results)
        else:
            print("\n🤖 Running Automated Demo with LLM Structural Cypher...")
            print("=" * 70)
            success = await demo_traversal_search(max_results)
            return success
            
        return True
            
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Goodbye!")
        return False
# --------------------------------------------------------------------------------- end main()

# __________________________________________________________________________
# Module Initialization / Main Execution Guard
# This code runs only when the file is executed
# --------------------------------------------------------------------------------- main_guard
if __name__ == "__main__":
    print("Starting APH-IF LLM Structural Cypher Console Demo...")

    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user. Goodbye!")
        sys.exit(0)
# --------------------------------------------------------------------------------- end main_guard

# __________________________________________________________________________
# End of File
#