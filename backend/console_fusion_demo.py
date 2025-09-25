# -------------------------------------------------------------------------
# File: console_fusion_demo.py
# Author: Alexander Ricciardi
# Date: 2025-09-25
# [File Path] backend/console_fusion_demo.py
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
#   Provides a console demonstration of the APH-IF intelligent fusion pipeline,
#   including parallel retrieval, fusion orchestration, metrics reporting, and
#   interactive exploration of hybrid results.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Argument parsing helpers
# - Logging configuration utilities
# - Banner and demo query helpers
# - Fusion metric reporting and post-processing utilities
# - Async flows for automated demos and interactive sessions
# - Main entrypoint coordinating mode selection and execution
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: asyncio, argparse, json, logging, sys, time, datetime,
#   pathlib
# - Typing: Any, Dict, List
# - Local Project Modules: app.search.parallel_hybrid, app.search.context_fusion,
#   app.core.config
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Execute from `backend/` via `uv run python console_fusion_demo.py`. Supports
# CLI flags for interactive mode, automated demo mode, verbosity, and debugging.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""Console demo showcasing the APH-IF intelligent context fusion workflow.

Runs parallel semantic and traversal retrieval, performs LLM-powered fusion, and
prints detailed metrics for demonstration and debugging purposes.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add backend app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.search.parallel_hybrid import get_parallel_engine, ParallelRetrievalEngine
from app.search.context_fusion import get_fusion_engine, IntelligentFusionEngine
from app.core.config import settings

# __________________________________________________________________________
# Imports
#

# __________________________________________________________________________
# Global Constants / Variables
#


# __________________________________________________________________________
# Standalone Function Definitions
#

# ______________________
# CLI Argument Utilities
#

# --------------------------------------------------------------------------------- parse_arguments()
def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments controlling demo behavior."""
    parser = argparse.ArgumentParser(
        description="APH-IF Phase 9: Intelligent Context Fusion Console Demo"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (INFO level logging)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Enable debug output (DEBUG level logging)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start directly in interactive mode",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run automated demonstration without prompting",
    )
    return parser.parse_args()
# --------------------------------------------------------------------------------- end parse_arguments()

# --------------------------------------------------------------------------------- configure_logging()
def configure_logging(args: argparse.Namespace) -> logging.Logger:
    """Configure logging level and filtering based on CLI arguments."""
    log_level = logging.WARNING
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    demo_logger = logging.getLogger("console_fusion_demo")
    demo_logger.setLevel(logging.INFO)

    if not args.debug:
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("app.parallel_hybrid").setLevel(logging.WARNING)
        logging.getLogger("app.context_fusion").setLevel(logging.WARNING)
        logging.getLogger("app.tools").setLevel(logging.WARNING)

    return demo_logger
# --------------------------------------------------------------------------------- end configure_logging()

# ______________________
# Presentation Helpers
#

# --------------------------------------------------------------------------------- print_banner()
def print_banner() -> None:
    """Print banner for the fusion demo."""
    print("=" * 80)
    print("🧠 APH-IF Phase 9: Intelligent Context Fusion Console Demo")
    print("=" * 80)
    print("This demo shows the complete fusion system: Parallel Retrieval + Intelligent Fusion")
    print()
    print("🚀 Core Innovation:")
    print("   Traditional: if condition: vector_search() else: graph_search()")
    print("   APH-IF:      asyncio.gather(vector_task, graph_task) + LLM fusion")
    print()
    print("✨ Features:")
    print("   • True parallel execution of semantic and traversal searches")
    print("   • GPT-5-mini powered intelligent context fusion")
    print("   • Advanced citation preservation (regulatory, technical, academic)")
    print("   • Domain-agnostic processing for any knowledge graph")
    print("   • Comprehensive performance metrics and health monitoring")
    print("   • Complementarity analysis and quality scoring")
    print("=" * 80)
# --------------------------------------------------------------------------------- end print_banner()

# --------------------------------------------------------------------------------- get_demo_queries()
def get_demo_queries() -> List[str]:
    """Get demonstration queries showcasing fusion capabilities."""
    return [
        # Regulatory and compliance
        "What are the safety equipment requirements for mining operations?",
        "Explain training and certification requirements for underground work", 
        "What are the ventilation standards and emergency procedures?",
        
        # Technical and standards
        "Describe equipment specifications and technical standards",
        "What are the regulatory compliance procedures and documentation requirements?",
        "Find information about monitoring systems and safety protocols",
        
        # Domain-agnostic
        "What are the key requirements for workplace safety?",
        "Explain the relationship between regulations and equipment standards",
        "Describe the compliance framework and implementation procedures"
    ]
# --------------------------------------------------------------------------------- end get_demo_queries()

# --------------------------------------------------------------------------------- print_fusion_metrics()
def print_fusion_metrics(
    parallel_result: ParallelRetrievalEngine.ParallelRetrievalResponse,
    fusion_result: IntelligentFusionEngine.FusionResult,
    parallel_total_time_ms: int,
    total_response_time: int,
) -> None:
    """Print comprehensive metrics from the complete fusion pipeline."""
    print("\n📊 COMPREHENSIVE FUSION METRICS")
    print("=" * 80)
    
    # =========================================================================
    # END-TO-END EXECUTION METRICS
    # =========================================================================
    print("🔄 End-to-End Execution Timing:")
    print(f"   ⏱️  Total Response Time: {total_response_time}ms")
    print(f"      └─ Complete end-to-end response including all processing phases")
    print(f"   🔄 Parallel Retrieval Time: {parallel_total_time_ms}ms")
    print(f"      └─ Time for concurrent semantic and traversal search execution")
    print(f"   ✅ Overall Success: {parallel_result.success}")
    print(f"      └─ At least one search method succeeded")
    print(f"   🎯 Both Methods Successful: {parallel_result.both_successful}")
    print(f"      └─ Both semantic and traversal searches completed successfully")
    print(f"   🔀 Fusion Ready: {parallel_result.fusion_ready}")
    print(f"      └─ Results suitable for intelligent context fusion")
    print(f"   🏆 Primary Method: {parallel_result.primary_method}")
    print(f"      └─ Search method with highest confidence score")
    
    # =========================================================================
    # PARALLEL EXECUTION ANALYSIS
    # =========================================================================
    print("\n🚀 Parallel Execution Analysis:")
    semantic_time = parallel_result.semantic_result.response_time_ms
    traversal_time = parallel_result.traversal_result.response_time_ms
    max_individual_time = max(semantic_time, traversal_time)
    sequential_time = semantic_time + traversal_time
    time_saved = sequential_time - max_individual_time
    efficiency = ((sequential_time - max_individual_time) / sequential_time * 100) if sequential_time > 0 else 0
    
    print(f"   🔗 Complementarity Score: {parallel_result.complementarity_score:.3f}")
    print(f"      └─ How well results complement each other (0.0=redundant, 1.0=highly complementary)")
    print(f"   ⚡ Parallelism Efficiency: {efficiency:.1f}%")
    print(f"      └─ Time saved by parallel execution vs sequential")
    print(f"   💾 Time Saved: {time_saved}ms")
    print(f"      └─ Sequential ({sequential_time}ms) vs Parallel ({max_individual_time}ms)")
    print(f"   🔄 Parallel Coordination Overhead: {parallel_total_time_ms - max_individual_time}ms")
    print(f"      └─ Additional time for task coordination and result processing")
    print(f"   🧠 Fusion Processing Time: {fusion_result.processing_time_ms}ms")
    print(f"      └─ Time for intelligent context fusion and synthesis")
    print(f"   📄 Additional Processing Time: {total_response_time - parallel_total_time_ms - fusion_result.processing_time_ms}ms")
    print(f"      └─ Time for result formatting and final response preparation")
    
    # =========================================================================
    # DETAILED SEARCH PERFORMANCE
    # =========================================================================
    print("\n📈 Individual Search Performance Analysis:")
    
    # Semantic Search Metrics
    print(f"   🔍 Semantic Search (Vector-based Similarity):")
    print(f"      🎯 Confidence Score: {parallel_result.semantic_result.confidence:.3f}")
    print(f"         └─ Quality assessment of semantic search results (0.0-1.0)")
    print(f"      ⏱️  Execution Time: {semantic_time}ms")
    print(f"         └─ Vector similarity search + embedding processing time")
    print(f"      📄 Source Documents: {len(parallel_result.semantic_result.sources)}")
    print(f"         └─ Number of document chunks retrieved from vector database")
    print(f"      🏷️  Entities Extracted: {len(parallel_result.semantic_result.entities)}")
    print(f"         └─ Named entities identified in search results")
    print(f"      📖 Citations Found: {len(parallel_result.semantic_result.citations)}")
    print(f"         └─ References and citations extracted from content")
    if parallel_result.semantic_result.error:
        print(f"      ❌ Error: {parallel_result.semantic_result.error}")
    
    # Traversal Search Metrics
    print(f"\n   🕸️  Traversal Search (Graph-based Relationships):")
    print(f"      🎯 Confidence Score: {parallel_result.traversal_result.confidence:.3f}")
    print(f"         └─ Quality assessment of graph traversal results (0.0-1.0)")
    print(f"      ⏱️  Execution Time: {traversal_time}ms")
    print(f"         └─ Cypher query execution + Neo4j network latency")
    print(f"      📄 Source Queries: {len(parallel_result.traversal_result.sources)}")
    print(f"         └─ Number of Cypher queries executed against knowledge graph")
    print(f"      🏷️  Entities Extracted: {len(parallel_result.traversal_result.entities)}")
    print(f"         └─ Graph entities found through relationship traversal")
    print(f"      📖 Citations Found: {len(parallel_result.traversal_result.citations)}")
    print(f"         └─ Regulatory and technical references from graph data")
    if parallel_result.traversal_result.error:
        print(f"      ❌ Error: {parallel_result.traversal_result.error}")
    
    # =========================================================================
    # INTELLIGENT FUSION ANALYSIS
    # =========================================================================
    print("\n🧠 Intelligent Fusion Results (LLM-Powered Synthesis):")
    print(f"   🎯 Final Confidence Score: {fusion_result.final_confidence:.3f}")
    print(f"      └─ Overall confidence in fused result quality (0.0-1.0)")
    print(f"   ⏱️  Fusion Processing Time: {fusion_result.processing_time_ms}ms")
    print(f"      └─ GPT-5-mini intelligent context fusion execution time")
    print(f"   📝 Citation Preservation Accuracy: {fusion_result.citation_accuracy:.3f}")
    print(f"      └─ How well citations were preserved during fusion (0.0-1.0)")
    print(f"   📖 Total Citations Preserved: {len(fusion_result.citations_preserved)}")
    print(f"      └─ Number of citations successfully maintained in fused content")
    print(f"   🔀 Fusion Strategy: {fusion_result.fusion_strategy}")
    print(f"      └─ LLM strategy used for intelligent context combination")
    print(f"   🌐 Domain Adaptation: {fusion_result.domain_adaptation}")
    print(f"      └─ Detected domain type for specialized processing")
    
    # =========================================================================
    # CONTRIBUTION WEIGHT ANALYSIS
    # =========================================================================
    print(f"\n⚖️  Search Method Contribution Analysis:")
    print(f"   📊 Vector Search Weight: {fusion_result.vector_contribution:.3f}")
    print(f"      └─ Relative contribution of semantic search to final result")
    print(f"   🕸️  Graph Search Weight: {fusion_result.graph_contribution:.3f}")
    print(f"      └─ Relative contribution of traversal search to final result")
    print(f"   🔗 Result Complementarity: {fusion_result.complementarity_score:.3f}")
    print(f"      └─ Degree of non-overlapping information between methods")
    
    # Weight validation
    total_weight = fusion_result.vector_contribution + fusion_result.graph_contribution
    print(f"   ✅ Weight Sum Validation: {total_weight:.3f}")
    print(f"      └─ Should sum to 1.0 for balanced contribution analysis")
    
    # =========================================================================
    # COMBINED RESULT STATISTICS
    # =========================================================================
    print(f"\n📚 Combined Knowledge Extraction:")
    print(f"   🏷️  Total Unique Entities: {len(fusion_result.entities_combined)}")
    print(f"      └─ Merged entities from both semantic and traversal searches")
    print(f"   📄 Total Information Sources: {len(fusion_result.sources_combined)}")
    print(f"      └─ Combined document chunks and graph queries with method attribution")
    
    # Entity breakdown
    semantic_entities = len(parallel_result.semantic_result.entities)
    traversal_entities = len(parallel_result.traversal_result.entities)
    total_before_merge = semantic_entities + traversal_entities
    unique_entities = len(fusion_result.entities_combined)
    entity_overlap = total_before_merge - unique_entities if total_before_merge > unique_entities else 0
    
    print(f"   🔄 Entity Deduplication:")
    print(f"      └─ Before merge: {total_before_merge} ({semantic_entities} semantic + {traversal_entities} traversal)")
    print(f"      └─ After merge: {unique_entities} unique entities")
    print(f"      └─ Overlap removed: {entity_overlap} duplicate entities")
    
    # =========================================================================
    # QUALITY AND PERFORMANCE INDICATORS
    # =========================================================================
    print(f"\n📊 Quality & Performance Indicators:")
    
    # Response completeness
    semantic_length = len(parallel_result.semantic_result.content)
    traversal_length = len(parallel_result.traversal_result.content)
    fusion_length = len(fusion_result.fused_content)
    
    print(f"   📏 Content Length Analysis:")
    print(f"      └─ Semantic result: {semantic_length:,} characters")
    print(f"      └─ Traversal result: {traversal_length:,} characters")
    print(f"      └─ Fused result: {fusion_length:,} characters")
    
    # Information density
    content_expansion_ratio = fusion_length / max(semantic_length, traversal_length) if max(semantic_length, traversal_length) > 0 else 0
    print(f"   📈 Information Synthesis Ratio: {content_expansion_ratio:.2f}x")
    print(f"      └─ How much the fusion expanded/synthesized the primary result")
    
    # Citation density
    citation_density = len(fusion_result.citations_preserved) / (fusion_length / 1000) if fusion_length > 0 else 0
    print(f"   📖 Citation Density: {citation_density:.2f} citations/1000 chars")
    print(f"      └─ Density of preserved references in final content")
    
    # =========================================================================
    # DEEP RESEARCH SYSTEM BENCHMARKS
    # =========================================================================
    print(f"\n🔬 Deep Research System Performance:")
    print(f"   🚀 First Search Completion: {min(semantic_time, traversal_time)}ms")
    print(f"      └─ Time until first research method completes (parallel advantage)")
    print(f"   🎯 Complete Research Cycle: {total_response_time}ms")
    print(f"      └─ Full end-to-end response time including parallel search + intelligent synthesis")
    
    # Research-oriented performance classification (based on actual system performance)
    if total_response_time < 120000:  # < 2 min
        perf_class = "⚪ Excellent response time (<2min)."
        perf_desc = "Excellent comprehensive response with optimal performance."
    elif total_response_time < 180000:  # 2-3 min
        perf_class = "🟢 Good response time (2-3min)."
        perf_desc = "Good comprehensive response within acceptable timeframe."
    elif total_response_time < 240000:  # 3-4 min Note the timeout are set to 240000
        perf_class = "🟡 Slow response time (3-4min)."
        perf_desc = "Slower than optimal - may indicate latency issues with parallel processing or fusion."
    elif total_response_time < 300000:  # 4-5 min
        perf_class = "🟠 Poor response time (4-5min)."
        perf_desc = "Significant latency or connection issues with the processing pipeline."
    else:  # > 5 min
        perf_class = "🔴 Critical performance issue (>5min)."
        perf_desc = "Critical performance issue - investigate system health and connectivity."
    
    print(f"   📊 Responce Performance Category: {perf_class}")
    print(f"      └─ {perf_desc}")
    print(f"   🎓 Responce Quality Priority: Accuracy > Speed")
    print(f"      └─ System optimized for comprehensive, citation-rich results")
    
    # =========================================================================
    # ERROR REPORTING
    # =========================================================================
    if fusion_result.error:
        print(f"\n⚠️  Fusion Processing Errors:")
        print(f"   ❌ Error Details: {fusion_result.error}")
        print(f"      └─ Issues encountered during intelligent context fusion")
    
    print("=" * 80)
# --------------------------------------------------------------------------------- end print_fusion_metrics()

# --------------------------------------------------------------------------------- create_synthetic_unknown_fusion()
def create_synthetic_unknown_fusion() -> IntelligentFusionEngine.FusionResult:
    """Construct a fallback fusion result when both retrieval legs are unknown."""
    from app.search.parallel_hybrid import FusionResult

    return FusionResult(
        fused_content="I don't know - there are no documents or sources in the provided context matching your prompt",
        final_confidence=0.0,
        fusion_strategy="skipped_both_unknown",
        processing_time_ms=0,
        vector_contribution=0.0,
        graph_contribution=0.0,
        complementarity_score=0.0,
        citation_accuracy=0.0,
        citations_preserved=[],
        entities_combined=[],
        sources_combined=[],
        domain_adaptation="none",
        error=None,
    )
# --------------------------------------------------------------------------------- end create_synthetic_unknown_fusion()

# --------------------------------------------------------------------------------- print_search_content()
def print_search_content(title: str, content: str, max_length: int | None = None) -> None:
    """Print content with optional truncation for lengthy sections.

    Args:
        title: Section header to display before the content.
        content: Text content to render.
        max_length: Optional maximum number of characters to display.
    """
    print(f"\n{title}")
    print("-" * len(title))

    if max_length is not None and len(content) > max_length:
        print(f"{content[:max_length]}...")
        print(f"\n[Content truncated - showing first {max_length} chars of {len(content)} total]")
    else:
        print(content)
# --------------------------------------------------------------------------------- end print_search_content()

# ______________________
# Fusion Engine Helpers
#

# --------------------------------------------------------------------------------- get_fusion_result()
async def get_fusion_result(
    fusion_engine: IntelligentFusionEngine,
    parallel_result: ParallelRetrievalEngine.ParallelRetrievalResponse,
) -> IntelligentFusionEngine.FusionResult:
    """Get fusion results without streaming."""
    fusion_start = time.time()
    
    try:
        
        # Get fusion result without streaming
        fusion_result = await fusion_engine.fuse_contexts(parallel_result)
        fusion_time = int((time.time() - fusion_start) * 1000)
        
        # IntelligentFusionEngine already handles all citation processing and References sections
        # Use the fusion result content as-is without any duplicate processing
        fusion_result.processing_time_ms = fusion_time
        
        return fusion_result
        
    except Exception as e:
        print(f"❌ Error in streaming: {e}")
        # Ultimate fallback
        try:
            fusion_result = await fusion_engine.fuse_contexts(parallel_result)
            print(fusion_result.fused_content)
            return fusion_result
        except Exception as fallback_error:
            print(f"❌ Fallback also failed: {fallback_error}")
            from app.search.parallel_hybrid import FusionResult
            return FusionResult(
                fused_content=f"Fusion failed: {str(e)}",
                final_confidence=0.0,
                fusion_strategy="error",
                processing_time_ms=0,
                vector_contribution=0.5,
                graph_contribution=0.5,
                complementarity_score=0.0,
                error=str(e)
            )
# --------------------------------------------------------------------------------- end get_fusion_result()

# --------------------------------------------------------------------------------- demonstrate_fusion_system()
async def demonstrate_fusion_system() -> bool:
    """Demonstrate the complete fusion system with predefined queries."""
    
    try:
        # Initialize engines
        print("\n🔧 Initializing APH-IF Fusion System...")
        parallel_engine = get_parallel_engine()
        fusion_engine = get_fusion_engine()
        
        # Health check
        print("\n🏥 System Health Check...")
        print("-" * 30)
        
        parallel_health = await parallel_engine.health_check()
        fusion_health = await fusion_engine.health_check()
        
        print(f"🔄 Parallel Engine: {parallel_health['status'].upper()}")
        print(f"🧠 Fusion Engine: {fusion_health['status'].upper()}")
        
        if parallel_health['status'] != 'healthy' or fusion_health['status'] != 'healthy':
            print("\n⚠️ System health issues detected. Continuing with demo...")
        
        # Get demo queries
        demo_queries = get_demo_queries()
        
        print(f"\n🚀 Running {len(demo_queries)} Fusion Demonstrations...")
        print("=" * 60)
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n🎯 DEMO {i}/{len(demo_queries)}")
            print(f"Query: {query}")
            print("=" * 60)
            
            try:
                # Execute complete fusion pipeline
                overall_start = time.time()
                
                # Step 1: Parallel Retrieval
                print("🔄 Step 1: Executing parallel retrieval...")
                parallel_result = await parallel_engine.retrieve_parallel(
                    query, semantic_k=5, traversal_max_results=20
                )
                
                if not parallel_result.success:
                    print("❌ Parallel retrieval failed, skipping fusion...")
                    continue
                
                # Step 2: Intelligent Context Fusion
                print("\n🧠 Step 2: Performing intelligent context fusion...")
                
                fusion_result = await get_fusion_result(fusion_engine, parallel_result)
                
                overall_time = int((time.time() - overall_start) * 1000)
                
                print(f"✅ Context fusion completed")
                
                # Display COMPLETE FUSION RESULTS SUMMARY - Primary display
                print("\n📋 COMPLETE FUSION RESULTS SUMMARY")
                print("=" * 50)
                
                # Show all three results WITHOUT truncation
                print("\n🔍 Semantic Result:")
                print("-" * 20)
                print(parallel_result.semantic_result.content)
                
                print("\n🕸️  Traversal Result:")
                print("-" * 20)
                print(parallel_result.traversal_result.content)
                
                print("\n🧠 Fusion Result:")
                print("-" * 20)
                print(fusion_result.fused_content)
                
                # Calculate comprehensive timing metrics
                parallel_total_time_ms = parallel_result.total_time_ms
                total_response_time = overall_time  # This is the complete end-to-end time
                
                # Show comprehensive metrics AFTER fusion response and references
                print_fusion_metrics(parallel_result, fusion_result, parallel_total_time_ms, total_response_time)
                
                # Show citation preservation details
                if fusion_result.citations_preserved:
                    print(f"\n📖 Citations Preserved:")
                    for cite in fusion_result.citations_preserved[:8]:  # Show first 8
                        print(f"   • {cite}")
                    if len(fusion_result.citations_preserved) > 8:
                        print(f"   ... and {len(fusion_result.citations_preserved) - 8} more")
                
                print("\n" + "=" * 60)
                
                # Small delay between demos
                if i < len(demo_queries):
                    print("⏳ (3 second pause between demos...)")
                    await asyncio.sleep(3)
                
            except Exception as e:
                print(f"❌ Demo {i} failed: {str(e)}")
                logger.error(f"Demo query failed: {e}", exc_info=True)
                continue
        
        # Final statistics
        fusion_stats = fusion_engine.get_fusion_stats()
        print(f"\n📊 SESSION STATISTICS")
        print("=" * 40)
        print(f"🧠 Total Fusions: {fusion_stats['fusion_count']}")
        print(f"⏱️  Avg Fusion Time: {fusion_stats['avg_fusion_time_ms']:.1f}ms")
        print(f"📖 Citation Rate: {fusion_stats['citation_preservation_rate']:.3f}")
        print(f"🎯 Fusion Strategy: {fusion_stats['fusion_strategy']}")
        
        print("\n✅ All fusion demonstrations completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Fusion demonstration failed: {e}")
        print(f"❌ System error: {str(e)}")
        return False
# --------------------------------------------------------------------------------- end demonstrate_fusion_system()

# --------------------------------------------------------------------------------- interactive_fusion_mode()
async def interactive_fusion_mode() -> None:
    """Interactive mode for custom fusion queries."""
    print("\n🤔 Interactive Fusion Mode")
    print("=" * 50)
    print("Enter your questions to test the complete fusion system.")
    print("The system will run both searches in parallel, then intelligently fuse results.")
    print()
    print("Commands:")
    print("  'demo' - Run predefined demonstration queries")
    print("  'health' - Check system health")
    print("  'stats' - Show current fusion statistics")
    print("  'quit' or 'exit' - End interactive mode")
    print("-" * 50)
    
    # Initialize engines
    parallel_engine = get_parallel_engine()
    fusion_engine = get_fusion_engine()
    
    query_count = 0
    
    while True:
        try:
            user_input = input(f"\n🔍 Fusion Query #{query_count + 1}: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Exiting interactive fusion mode...")
                break
            
            if user_input.lower() == 'demo':
                print("🤖 Running demonstration queries...")
                await demonstrate_fusion_system()
                continue
            
            if user_input.lower() == 'health':
                print("\n🏥 System Health Check:")
                parallel_health = await parallel_engine.health_check()
                fusion_health = await fusion_engine.health_check()
                print(f"   🔄 Parallel Engine: {parallel_health['status'].upper()}")
                print(f"   🧠 Fusion Engine: {fusion_health['status'].upper()}")
                continue
            
            if user_input.lower() == 'stats':
                print("\n📊 Current Statistics:")
                stats = fusion_engine.get_fusion_stats()
                print(f"   🧠 Total Fusions: {stats['fusion_count']}")
                print(f"   ⏱️  Average Time: {stats['avg_fusion_time_ms']:.1f}ms")
                print(f"   📖 Citation Rate: {stats['citation_preservation_rate']:.3f}")
                continue
            
            if not user_input:
                continue
            
            print(f"\n🚀 Processing: {user_input}")
            print("=" * 60)
            
            # Execute complete fusion pipeline
            overall_start = time.time()
            
            # Step 1: Parallel Retrieval
            print("🔄 Step 1: Executing parallel searches...")
            parallel_start = time.time()
            
            parallel_result = await parallel_engine.retrieve_parallel(
                user_input, semantic_k=8, traversal_max_results=30
            )
            
            parallel_time = int((time.time() - parallel_start) * 1000)
            
            if not parallel_result.success:
                print("❌ Parallel retrieval failed!")
                if parallel_result.semantic_result.error:
                    print(f"   Semantic Error: {parallel_result.semantic_result.error}")
                if parallel_result.traversal_result.error:
                    print(f"   Traversal Error: {parallel_result.traversal_result.error}")
                continue
            
            print(f"✅ Parallel retrieval completed in {parallel_time}ms")
            print(f"   🔍 Semantic: {parallel_result.semantic_result.confidence:.3f} confidence")
            print(f"   🕸️  Traversal: {parallel_result.traversal_result.confidence:.3f} confidence")
            print(f"   🔀 Fusion Ready: {parallel_result.fusion_ready}")
            
            # Step 2: Intelligent Context Fusion (always attempted)
            print("\n🧠 Step 2: Processing fusion...")
            fusion_start = time.time()

            # Unknown detection (moved earlier for smart skipping)
            semantic_lower = parallel_result.semantic_result.content.lower()
            traversal_lower = parallel_result.traversal_result.content.lower()
            unknown_patterns = ["i don't know", "no documents or sources", "no relevant information", "no matching"]
            semantic_unknown = any(p in semantic_lower for p in unknown_patterns)
            traversal_unknown = any(p in traversal_lower for p in unknown_patterns)

            # Smart fusion skipping when both unknown
            if semantic_unknown and traversal_unknown:
                # Create synthetic FusionResult to avoid LLM call
                print("⏭️ Skipping LLM fusion (both results unknown)")
                fusion_result = create_synthetic_unknown_fusion()
                fusion_time = 0
            else:
                # At least one leg has content - attempt real fusion
                fusion_result = await get_fusion_result(fusion_engine, parallel_result)
                fusion_time = int((time.time() - fusion_start) * 1000)

            overall_time = int((time.time() - overall_start) * 1000)

            # Display logic based on fusion readiness
            if parallel_result.fusion_ready:
                # Fusion was suitable - show full results
                print(f"✅ Context fusion completed in {fusion_time}ms")

                print("\n📋 COMPLETE FUSION RESULTS SUMMARY")
                print("=" * 50)

                # Show all three results
                print("\n🔍 Semantic Result:")
                print("-" * 20)
                print(parallel_result.semantic_result.content)

                print("\n🕸️  Traversal Result:")
                print("-" * 20)
                print(parallel_result.traversal_result.content)

                print("\n🧠 Fusion Result:")
                print("-" * 20)
                print(fusion_result.fused_content)

                # Full metrics
                parallel_total_time_ms = parallel_result.total_time_ms
                total_response_time = overall_time
                print_fusion_metrics(parallel_result, fusion_result, parallel_total_time_ms, total_response_time)

                # Citation analysis (keep existing)
                if fusion_result.citations_preserved:
                    print(f"\n📖 Citation Preservation Analysis:")
                    print(f"   Total preserved: {len(fusion_result.citations_preserved)}")
                    print(f"   Accuracy rate: {fusion_result.citation_accuracy:.1%}")

                    # Show sample citations
                    for i, cite in enumerate(fusion_result.citations_preserved[:5], 1):
                        print(f"   {i}. {cite}")

                    if len(fusion_result.citations_preserved) > 5:
                        print(f"   ... and {len(fusion_result.citations_preserved) - 5} more")
            else:
                # Not fusion ready - but still show everything
                print("\n⚠️ Results not suitable for fusion")

                # Show individual results based on unknown status
                if semantic_unknown and traversal_unknown:
                    print("\n📭 No relevant information found:")
                    print("I don't know - there are no documents or sources in the provided context matching your prompt")
                else:
                    print("Showing individual results:")
                    print_search_content("🔍 Semantic Result:", parallel_result.semantic_result.content)
                    print_search_content("🕸️  Traversal Result:", parallel_result.traversal_result.content)

                # ALWAYS show fusion result
                print("\n🧠 Fusion Result:")
                print("-" * 20)
                print(fusion_result.fused_content)

                # Show error if present
                if fusion_result.error:
                    print(f"\n⚠️ Fusion Error: {fusion_result.error}")

                # Condensed metrics for not-ready case
                print(f"\n📊 Processing Summary:")
                print(f"   ⏱️  Total Time: {overall_time}ms")
                print(f"   🔄 Parallel Retrieval: {parallel_result.total_time_ms}ms")
                print(f"   🧠 Fusion Processing: {fusion_time}ms")
                print(f"   🎯 Fusion Strategy: {fusion_result.fusion_strategy}")
                print(f"   📊 Final Confidence: {fusion_result.final_confidence:.3f}")
            
            query_count += 1
            
        except KeyboardInterrupt:
            print("\n👋 Interactive mode interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
            logger.error(f"Interactive query error: {e}", exc_info=True)
# --------------------------------------------------------------------------------- end interactive_fusion_mode()

# --------------------------------------------------------------------------------- system_health_and_performance()
async def system_health_and_performance() -> bool:
    """Comprehensive system health and performance check."""
    print("\n🏥 COMPREHENSIVE SYSTEM HEALTH CHECK")
    print("=" * 50)
    
    try:
        # Initialize engines
        parallel_engine = get_parallel_engine()
        fusion_engine = get_fusion_engine()
        
        # Health checks
        print("🔄 Checking Parallel Engine...")
        parallel_health = await parallel_engine.health_check()
        
        print("🧠 Checking Fusion Engine...")
        fusion_health = await fusion_engine.health_check()
        
        # Display health results
        print(f"\n📊 Health Status:")
        print(f"   🔄 Parallel Engine: {parallel_health['status'].upper()}")
        print(f"      - Semantic Search: {parallel_health['components']['semantic_search']}")
        print(f"      - Traversal Search: {parallel_health['components']['traversal_search']}")
        print(f"      - Timeout: {parallel_health['timeout_seconds']}s")
        
        print(f"   🧠 Fusion Engine: {fusion_health['status'].upper()}")
        print(f"      - LLM Client: {fusion_health['components']['llm_client']}")
        print(f"      - Citation Processor: {fusion_health['components']['citation_processor']}")
        print(f"      - Strategy: {fusion_health['fusion_strategy']}")
        
        # Performance metrics
        print(f"\n⚡ Performance Metrics:")
        fusion_stats = fusion_engine.get_fusion_stats()
        print(f"   🧠 Fusion Count: {fusion_stats['fusion_count']}")
        print(f"   ⏱️  Avg Fusion Time: {fusion_stats['avg_fusion_time_ms']:.1f}ms")
        print(f"   📖 Citation Rate: {fusion_stats['citation_preservation_rate']:.3f}")
        
        # Environment info
        print(f"\n🔧 Environment Configuration:")
        print(f"   🎯 OpenAI Model: {settings.openai_model_mini}")
        print(f"   🗃️  Neo4j Database: {settings.neo4j_database}")
        print(f"   🏃 Development Mode: {settings.is_development}")
        print(f"   📝 Verbose Logging: {settings.verbose}")
        
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return False
# --------------------------------------------------------------------------------- end system_health_and_performance()

# --------------------------------------------------------------------------------- quick_fusion_test()
async def quick_fusion_test() -> bool:
    """Quick test of the fusion system with a simple query."""
    print("\n⚡ QUICK FUSION TEST")
    print("=" * 30)
    
    test_query = "What are the main safety requirements?"
    print(f"Query: {test_query}")
    
    try:
        # Initialize engines
        parallel_engine = get_parallel_engine()
        fusion_engine = get_fusion_engine()
        
        # Execute fusion pipeline
        start_time = time.time()
        
        # Parallel retrieval
        print("\n🔄 Running parallel searches...")
        parallel_result = await parallel_engine.retrieve_parallel(test_query, semantic_k=3, traversal_max_results=10)
        
        if parallel_result.success:
            # Check for both unknown to potentially skip LLM
            semantic_content = parallel_result.semantic_result.content.lower()
            traversal_content = parallel_result.traversal_result.content.lower()
            unknown_patterns = ["i don't know", "no documents or sources", "no relevant information", "no matching"]
            both_unknown = (any(p in semantic_content for p in unknown_patterns) and
                           any(p in traversal_content for p in unknown_patterns))

            if both_unknown:
                # Skip LLM for efficiency in quick test
                print("⏭️ Skipping fusion (both unknown)")
                fusion_result = create_synthetic_unknown_fusion()
            else:
                # Attempt real fusion
                print("🧠 Running intelligent fusion...")
                fusion_result = await get_fusion_result(fusion_engine, parallel_result)

            total_time = int((time.time() - start_time) * 1000)

            print(f"\n✅ Quick test completed in {total_time}ms!")
            print(f"   🔀 Fusion Ready: {parallel_result.fusion_ready}")
            print(f"   🎯 Final confidence: {fusion_result.final_confidence:.3f}")
            print(f"   📖 Citations preserved: {len(fusion_result.citations_preserved)}")
            print(f"   🔗 Complementarity: {fusion_result.complementarity_score:.3f}")

            # Always show fusion result preview
            print("\n🧠 Fusion Result Preview:")
            fusion_preview = fusion_result.fused_content[:200] + "..." if len(fusion_result.fused_content) > 200 else fusion_result.fused_content
            print(fusion_preview)

            if fusion_result.error:
                print(f"\n⚠️ Fusion error: {fusion_result.error}")

            return parallel_result.fusion_ready
        else:
            print("⚠️ Parallel retrieval failed")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed: {str(e)}")
        return False
# --------------------------------------------------------------------------------- end quick_fusion_test()

# --------------------------------------------------------------------------------- main()
async def main(args: argparse.Namespace | None = None) -> bool:
    """Main function to run the fusion demo.

    Args:
        args: Parsed command-line arguments. If None, the user will be prompted
            to choose between interactive and automated demo modes.

    Returns:
        bool: ``True`` when the selected demo completes successfully; otherwise
        ``False``.
    """
    print_banner()
    
    # Determine mode based on command-line arguments or prompt
    try:
        if args and args.interactive:
            # Start interactive mode directly when --interactive flag is used
            print("\n🎮 Starting Interactive Fusion Mode...")
            print("=" * 70)
            await interactive_fusion_mode()
            return True
        elif args and args.demo:
            # Start demo mode directly when --demo flag is used
            print("\n🤖 Running Automated Fusion Demonstration...")
            print("=" * 70)
            success = await demonstrate_fusion_system()
            return success
        else:
            # No flags provided, prompt for mode
            response = input("\n🎮 Would you like to try interactive mode? (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                print("\n🎮 Starting Interactive Fusion Mode...")
                print("=" * 70)
                await interactive_fusion_mode()
                return True
            else:
                print("\n🤖 Running Automated Fusion Demonstration...")
                print("=" * 70)
                # Run the automated demo
                success = await demonstrate_fusion_system()
                return success
            
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Goodbye!")
        return False
# --------------------------------------------------------------------------------- end main()

# __________________________________________________________________________
# Module Initialization / Main Execution Guard
# This code runs only when the file is executed
# --------------------------------------------------------------------------------- main_guard
if __name__ == "__main__":
    cli_args = parse_arguments()
    demo_logger = configure_logging(cli_args)

    print("🚀 Starting APH-IF Phase 9: Intelligent Context Fusion Console Demo")
    print("   This demo showcases the complete parallel + fusion pipeline...")

    try:
        result = asyncio.run(main(cli_args))
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        sys.exit(1)
# --------------------------------------------------------------------------------- end main_guard

# __________________________________________________________________________
# End of File
#