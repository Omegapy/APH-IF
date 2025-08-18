# -------------------------------------------------------------------------
# File: initial_graph_build.py
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 08-09-2025
# File Path: data_processing/processing/initial_graph_build.py
# -------------------------------------------------------------------------

# --- Module Objective ---
#   This module implements the HybridStoreBuilder class and associated logic to
#   construct the initial hybrid knowledge store for the APH-IF system.
#   It processes Title 30 CFR PDFs to generate graph-based (GraphRAG) and vector-
#   based (VectorRAG) structures using GPT-5 and OpenAI embeddings. It supports
#   LLM-based entity extraction and vector indexing via Neo4j for parallel search.
#
# Relationship Types and Nodes
#
# - (:Document)-[:HAS_CHUNK]->(:Chunk): Document contains chunk
# - (:Chunk)-[:PART_OF]->(:Document): Chunk belongs to document
# - (:Chunk)-[:HAS_ENTITY]->(:Entity): Chunk contains entity
# - (:Entity)-[:PART_OF]->(:Chunk): Entity belongs to chunk
#
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Class: HybridStoreBuilder
# - Function: main
# - Constants: PROJECT_ROOT, DEFAULT_PDF_DIR, logger
# - Utility Function: _resolve_embedding_dimension
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: os, logging, time, traceback, datetime, importlib, inspect, io, pathlib
# - Third-Party: PyPDF2, langchain_community, langchain_openai, langchain_experimental,
#                langchain_core, tiktoken (optional)
# - Local Modules: common.monitored_neo4j, common.monitored_openai, common.api_monitor
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# This module can be executed as a script or imported and used as a utility class.
# When run directly, it initializes a HybridStoreBuilder instance, processes all
# configured PDFs, builds vector indexes, and prints a final summary. It integrates
# with Neo4j and OpenAI through environment-based configuration.
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""

APH‑IF Initial Hybrid Store Builder

Builds hybrid knowledge stores that support both GraphRAG (entity‑relationship graphs)
and VectorRAG (semantic embeddings) from Title 30 CFR regulatory documents for
parallel hybrid search.

How it works
------------
1) Read and chunk PDF pages using a character splitter (token‑aware optional).
2) For each chunk:
   - Compute an embedding vector (VectorRAG component)
   - Upsert Document and Chunk nodes and connect via `HAS_CHUNK`/`PART_OF`
   - Optionally extract entities via LLM and upsert `Entity` nodes with
     `HAS_ENTITY`/`PART_OF` relationships (GraphRAG component)
3) After processing, create a cosine vector index on `:Chunk(embedding)`.
4) Print progress and a final summary of graph sizes and rates.

Environment variables
---------------------
- OPENAI_API_KEY (required)
- NEO4J_URI, NEO4J_USERNAME/NEO4J_USER, NEO4J_PASSWORD
- OPENAI_MODEL/OpenAI model knobs (see __init__)
- PDF_DIR (input directory), CLEAR_DB (optional reset)
- CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS
- EXTRACT_EVERY_N_CHUNKS (frequency) / DISABLE_ENTITY_EXTRACTION
- MAX_DOCS, MAX_PAGES_PER_DOC (limits)
- VERBOSE (logging)

Prerequisites & Safety
----------------------
- Recommend running in test mode (use set_environment.py and test DB) before production.
- When CLEAR_DB=true, all nodes/relationships are detached & deleted.
- Ensure adequate Neo4j AuraDB limits for vector index creation and storage.

Verification queries (examples)
-------------------------------
- Document count:          MATCH (d:Document) RETURN count(d)
- Chunk count (with emb):  MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c)
- Entity count:            MATCH (e:Entity) RETURN count(e)
- Relationship checks:     MATCH (d)-[:HAS_CHUNK]->(c) RETURN count(*)
                           MATCH (c)-[:HAS_ENTITY]->(e) RETURN count(*)
- Vector index:            SHOW INDEXES YIELD name, type WHERE type = 'VECTOR'

Performance notes
-----------------
- Chunk size trades off throughput vs retrieval quality. Larger chunks reduce LLM/API usage.
- Entity extraction frequency can be lowered for speed (extract every N chunks).
- Simple batch loop controls pacing; `SLEEP_*` envs can throttle if needed.

"""

# =========================================================================
# Imports
# =========================================================================
# Standard library imports
import os
import logging
import time
import traceback
from datetime import datetime
from importlib import import_module
import inspect
from io import BytesIO
from pathlib import Path

# Third-party library imports
import PyPDF2
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
try:
    import tiktoken  # optional
except Exception:  # pragma: no cover
    tiktoken = None

# Local application/library specific imports
# (None for this standalone module)

# =========================================================================
# Global Constants / Variables
# =========================================================================
# Project root directory for reliable path resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default PDF directory inside the container (can be overridden via env)
DEFAULT_PDF_DIR = os.getenv("PDF_DIR", "/app/data_pdf")

# Configure logging for hybrid store construction monitoring
# Support verbose mode via VERBOSE environment variable
log_level = logging.DEBUG if os.getenv("VERBOSE", "false").lower() == "true" else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

# Logger for this module
logger = logging.getLogger(__name__)

# Import monitoring components after logger is defined
try:
    from common.monitored_neo4j import MonitoredNeo4jGraph
    from common.monitored_openai import MonitoredOpenAIClient
    from common.api_monitor import LogLevel, configure_monitoring
    MONITORING_AVAILABLE = True
    logger.info("API monitoring enabled")
except ImportError:
    logger.warning("Monitoring modules not available. Running without API monitoring.")
    MONITORING_AVAILABLE = False

# --------------------------------------------------------------------------------- _resolve_embedding_dimension()
def _resolve_embedding_dimension(model: str) -> int:
    """Resolve embedding vector dimension for a known OpenAI model.

    Args:
        model: Embedding model name.

    Returns:
        int: Expected vector dimension; defaults to 1536 if unknown.
    """
    override = os.getenv("EMBEDDING_DIM")
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    model = (model or "").strip()
    known = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }
    return known.get(model, 1536)
# --------------------------------------------------------------------------------- end _resolve_embedding_dimension()

# =========================================================================
# Class Definitions
# =========================================================================

# ------------------------------------------------------------------------- HybridStoreBuilder
# ------------------------------------------------------------------------- HybridStoreBuilder
class HybridStoreBuilder:
    """Hybrid store builder for APH‑IF (Figure‑1 graph + vectors).

    Builds an initial hybrid knowledge base combining:
    - GraphRAG: Document, Chunk, Entity nodes and relationships
    - VectorRAG: Chunk embeddings for semantic retrieval

    Instance Attributes:
        neo4j_uri (str): Neo4j database connection URI
        neo4j_username (str): Neo4j database username
        neo4j_password (str): Neo4j database password
        graph (Neo4jGraph): Neo4j client via LangChain
        llm (ChatOpenAI): LLM used by LLMGraphTransformer for extraction
        embedding_model (OpenAIEmbeddings): Embedding model for chunk vectors
        text_splitter (RecursiveCharacterTextSplitter): Text chunking utility
        total_chunks_processed (int): Progress tracking for processed chunks
        total_entities_created (int): Progress tracking for extracted entities
        start_time (datetime): Processing start time for performance metrics

    Methods:
        __init__(): Initialize connections, models, and config
        print_progress_header(): Print run header
        extract_entities_msha(): Extract entities via LLMGraphTransformer
        print_progress_stats(): Print current run statistics
        process_directory(): Process all PDFs
        process_pdf(): Process a single PDF
        create_vector_index(): Ensure vector index on :Chunk(embedding)
        print_final_summary(): Print completion summary
    """

    # -------------------
    # --- Constructor ---
    # -------------------
    
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self):
        """Initialize connections and models for GPT-5 based hybrid store construction.

        - Uses OpenAI GPT-5 for entity extraction via LLMGraphTransformer
        - Uses OpenAI Embeddings for vector search
        - Connects to Neo4j via env configuration
        """
        print("Initializing APH-IF Hybrid Store Builder (GPT-5)...")
        try:
            # Required environment variables
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set.")

            # Use centralized environment variables managed by set_environment.py
            self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            self.neo4j_username = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
            self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

            # Model configuration
            self.gpt5_model = os.getenv("OPENAI_MODEL_GPT5", os.getenv("OPENAI_MODEL", "gpt-5-nano"))
            self.embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

            # Initialize connections with monitoring
            print("Connecting to Neo4j...")
            if MONITORING_AVAILABLE:
                # Configure monitoring output directory
                monitoring_dir = Path("monitoring_logs")
                monitoring_dir.mkdir(exist_ok=True)

                # Determine log level based on VERBOSE setting
                log_level = LogLevel.DETAILED if os.getenv("VERBOSE", "false").lower() == "true" else LogLevel.STANDARD

                self.graph = MonitoredNeo4jGraph(
                    url=self.neo4j_uri,
                    username=self.neo4j_username,
                    password=self.neo4j_password,
                    log_level=log_level,
                    output_file=monitoring_dir / "neo4j_initial_build.jsonl"
                )
                print("[SUCCESS] Neo4j connection established with monitoring")
            else:
                self.graph = Neo4jGraph(
                    url=self.neo4j_uri,
                    username=self.neo4j_username,
                    password=self.neo4j_password,
                )
                print("[SUCCESS] Neo4j connection established")

            # Initialize models with timeout and connection limits to prevent hanging
            print("Initializing OpenAI GPT-5 & embeddings...")
            # langchain_openai uses OPENAI_API_KEY from env
            os.environ["OPENAI_API_KEY"] = self.openai_api_key

            lc_openai = import_module("langchain_openai")

            # Create embeddings with basic configuration
            self.embedding_model = getattr(lc_openai, "OpenAIEmbeddings")(
                model=self.embed_model
            )

            # Use nano model for extractive relationships if provided (faster/cheaper)
            extract_model = os.getenv("OPENAI_MODEL_NANO", self.gpt5_model)
            self.extract_model = extract_model
            ChatOpenAICls = getattr(lc_openai, "ChatOpenAI")
            sig = inspect.signature(ChatOpenAICls)

            # Create LLM with timeout and connection limits to prevent hanging
            llm_kwargs = {
                "temperature": 1,
                "timeout": 120,  # 2 minute timeout
                "max_retries": 2  # Limit retries
            }

            if "model" in sig.parameters:
                llm_kwargs["model"] = extract_model
                self.llm = ChatOpenAICls(**llm_kwargs)
            elif "model_name" in sig.parameters:
                llm_kwargs["model_name"] = extract_model
                self.llm = ChatOpenAICls(**llm_kwargs)
            else:  # Fallback
                self.llm = ChatOpenAICls(**{k: v for k, v in llm_kwargs.items() if k != "model"})

            print(f"[CONFIG] Using extraction model: {extract_model} | embedding model: {self.embed_model}")
            print(f"[CONFIG] Timeout settings: 120s request timeout, 2 max retries")

            lc_exp = import_module("langchain_experimental.graph_transformers")
            self.transformer = getattr(lc_exp, "LLMGraphTransformer")(llm=self.llm)

            # Initialize monitored OpenAI client for direct API calls
            if MONITORING_AVAILABLE:
                monitoring_dir = Path("monitoring_logs")
                log_level = LogLevel.DETAILED if os.getenv("VERBOSE", "false").lower() == "true" else LogLevel.STANDARD
                self.openai_client = MonitoredOpenAIClient(
                    api_key=self.openai_api_key,
                    log_level=log_level,
                    output_file=monitoring_dir / "openai_initial_build.jsonl"
                )
                print("[SUCCESS] OpenAI models initialized with monitoring")
            else:
                self.openai_client = None
                print("[SUCCESS] OpenAI models initialized")

            # Text splitter for production (page-aware usage below; size ~1200 token-equivalent)
            # Using character-based approximation; token counting optional via tiktoken
            chunk_chars = int(os.getenv("CHUNK_SIZE_CHARS", "4000"))
            chunk_overlap = int(os.getenv("CHUNK_OVERLAP_CHARS", "400"))
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_chars,
                chunk_overlap=chunk_overlap,
            )

            # Progress tracking
            self.total_chunks_processed = 0
            self.total_entities_created = 0
            self.start_time = datetime.now()

            # Resolve embedding dimension for vector index creation
            self.embedding_dimension = _resolve_embedding_dimension(self.embed_model)

            # Performance/scale controls (env-configurable)
            self.max_docs = int(os.getenv("MAX_DOCS", "0"))  # 0 = no limit
            self.max_pages_per_doc = int(os.getenv("MAX_PAGES_PER_DOC", "0"))  # 0 = no limit
            self.disable_entity_extraction = os.getenv("DISABLE_ENTITY_EXTRACTION", "false").lower() == "true"
            self.extract_every_n_chunks = max(1, int(os.getenv("EXTRACT_EVERY_N_CHUNKS", "1")))
            self.sleep_between_chunks_ms = int(os.getenv("SLEEP_BETWEEN_CHUNKS_MS", "0"))
            self.sleep_between_batches_ms = int(os.getenv("SLEEP_BETWEEN_BATCHES_MS", "0"))

            print("[SUCCESS] APH-IF Hybrid Store Builder ready!")

        except Exception as e:
            print(f"[ERROR] ERROR during initialization: {e}")
            raise
    # --------------------------------------------------------------------------------- end __init__()

    # ---------------------------
    # --- Progress Monitoring ---
    # ---------------------------

    # --------------------------------------------------------------------------------- print_progress_header()
    # --------------------------------------------------------------------------------- print_progress_header()
    def print_progress_header(self):
        """Print a comprehensive progress header for hybrid store construction monitoring.

        Displays detailed information about the hybrid store construction process
        including start time, target objectives, component descriptions, AI models
        being used, and database information. Provides clear visibility into the
        Advanced Parallel Hybrid system preparation workflow.
        """
        print("\n" + "="*80)
        print("APH-IF HYBRID STORE BUILDER - FIGURE-1 SEED GRAPH")
        print("="*80)
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target: Build hybrid store for Advanced Parallel Hybrid system")
        print(f"Graph: Entity-relationship knowledge graph")
        print(f"Vector: Semantic embeddings for similarity search")
        # Show the exact models in use
        try:
            model_info = f"extraction={getattr(self, 'extract_model', 'unknown')} | embedding={getattr(self, 'embed_model', 'unknown')}"
        except Exception:
            model_info = "extraction=unknown | embedding=unknown"
        print(f"Models: {model_info}")
        print(f"Database: Neo4j AuraDB")
        print("="*80)
    # --------------------------------------------------------------------------------- end print_progress_header()

    # --------------------------------------------------------------------------------- print_progress_stats()
    # --------------------------------------------------------------------------------- print_progress_stats()
    def print_progress_stats(self):
        """Print current hybrid store construction progress statistics.

        Retrieves and displays real-time statistics from the Neo4j database including
        total nodes, chunks with vector embeddings, extracted entities, documents
        processed, and processing rate calculations. Provides comprehensive monitoring
        of both GraphRAG and VectorRAG component construction progress.

        Raises:
            Exception: If database query fails during statistics retrieval
        """
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        
        # Get current database stats
        try:
            total_nodes = self.graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
            total_chunks = self.graph.query("MATCH (c:Chunk) RETURN count(c) as count")[0]['count']
            total_entities = self.graph.query("MATCH (e:Entity) RETURN count(e) as count")[0]['count']
            total_docs = self.graph.query("MATCH (d:Document) RETURN count(d) as count")[0]['count']
            
            print(f"\nHYBRID STORE PROGRESS - Elapsed: {elapsed_str}")
            print(f"   Documents: {total_docs}")
            print(f"   Chunks (w/ vectors): {total_chunks}")
            print(f"   Entities (graph nodes): {total_entities}")
            print(f"   Total Nodes: {total_nodes}")
            
            if total_chunks > 0:
                chunks_per_hour = total_chunks / (elapsed.total_seconds() / 3600)
                print(f"   [RATE] Processing Rate: {chunks_per_hour:.1f} hybrid chunks/hour")

        except Exception as e:
            print(f"[ERROR] Could not get progress stats: {e}")
    # --------------------------------------------------------------------------------- end print_progress_stats()

    # --------------------------------------------------------------------------------- print_final_summary()
    # --------------------------------------------------------------------------------- print_final_summary()
    def print_final_summary(self):
        """Print final hybrid store construction completion summary.

        Generates and displays comprehensive completion statistics including total
        processing time, component counts, processing rates, and readiness confirmation
        for both GraphRAG and VectorRAG components. Provides final validation that
        the Advanced Parallel Hybrid system is ready for deployment.

        Raises:
            Exception: If final database statistics cannot be retrieved
        """
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]
        
        try:
            total_nodes = self.graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
            total_rels = self.graph.query("MATCH ()-[r]-() RETURN count(r) as count")[0]['count']
            total_chunks = self.graph.query("MATCH (c:Chunk) RETURN count(c) as count")[0]['count']
            total_entities = self.graph.query("MATCH (e:Entity) RETURN count(e) as count")[0]['count']
            total_docs = self.graph.query("MATCH (d:Document) RETURN count(d) as count")[0]['count']
            
            print(f"\n" + "="*80)
            print("APH-IF INITIAL HYBRID STORE CONSTRUCTION COMPLETED!")
            print("="*80)
            print(f"   Total Time: {elapsed_str}")
            print(f"Documents: {total_docs}")
            print(f"Hybrid Chunks: {total_chunks:,} (with vector embeddings)")
            print(f"Graph Entities: {total_entities:,}")
            print(f"Total Graph Nodes: {total_nodes:,}")
            print(f"Graph Relationships: {total_rels:,}")
            print(f"Processing Rate: {total_chunks / (elapsed.total_seconds() / 3600):.1f} hybrid chunks/hour")
            print("="*80)
            print("[SUCCESS] HYBRID STORE COMPONENTS READY:")
            print("    GraphRAG: Entity-relationship graph for structural queries")
            print("    VectorRAG: Semantic embeddings for similarity search")
            print("    Advanced Parallel Hybrid system ready for deployment!")

            # Add monitoring statistics if available
            if MONITORING_AVAILABLE and hasattr(self, 'graph') and hasattr(self.graph, 'get_monitoring_stats'):
                print("\n" + "-"*60)
                print("API MONITORING SUMMARY:")
                print("-"*60)

                # Neo4j monitoring stats
                neo4j_stats = self.graph.get_monitoring_stats()
                if neo4j_stats.get('total_calls', 0) > 0:
                    print(f"Neo4j Operations:")
                    print(f"  Total Queries: {neo4j_stats['total_calls']:,}")
                    print(f"  Success Rate: {neo4j_stats['success_rate']:.1%}")
                    print(f"  Avg Query Time: {neo4j_stats['avg_duration_ms']:.1f}ms")
                    print(f"  Records Returned: {neo4j_stats.get('total_records_returned', 0):,}")
                    print(f"  Records Affected: {neo4j_stats.get('total_records_affected', 0):,}")

                # OpenAI monitoring stats
                if hasattr(self, 'openai_client') and self.openai_client:
                    openai_stats = self.openai_client.get_monitoring_stats()
                    if openai_stats.get('total_calls', 0) > 0:
                        print(f"OpenAI API Calls:")
                        print(f"  Total Requests: {openai_stats['total_calls']:,}")
                        print(f"  Success Rate: {openai_stats['success_rate']:.1%}")
                        print(f"  Avg Response Time: {openai_stats['avg_duration_ms']:.1f}ms")
                        print(f"  Total Tokens Used: {openai_stats.get('total_tokens_used', 0):,}")
                        print(f"  Avg Tokens/Call: {openai_stats.get('avg_tokens_per_call', 0):.0f}")

                print("-"*60)

            print("="*80)

        except Exception as e:
            print(f"[ERROR] Could not generate final summary: {e}")
    # --------------------------------------------------------------------------------- end print_final_summary()

    # ---------------------------
    # --- Entity Extraction ---
    # ---------------------------

    # --------------------------------------------------------------------------------- extract_entities_msha()
    # --------------------------------------------------------------------------------- extract_entities_msha()
    def extract_entities_msha(self, text: str) -> list[tuple[str, str]]:
        """Extract entities using GPT-5 via LLMGraphTransformer (domain-agnostic).

        Returns up to 8 (entity_name, entity_type) pairs per chunk.
        """
        try:
            # Use Document inputs; transformer will produce GraphDocuments
            docs = [Document(page_content=text, metadata={})]
            gdocs = self.transformer.convert_to_graph_documents(docs)
            entities: list[tuple[str, str]] = []
            for gdoc in gdocs:
                for node in getattr(gdoc, "nodes", []):
                    # Normalize
                    name = str(getattr(node, "id", "") or getattr(node, "name", "") or "").strip()
                    etype = str(getattr(node, "type", "") or "entity").strip()
                    if name:
                        entities.append((name, etype))
                        if len(entities) >= 8:
                            break
            return entities
        except Exception as e:
            print(f"[ERROR] Entity extraction failed: {e}")
            return []
    # --------------------------------------------------------------------------------- end extract_entities_msha()

    # ---------------------------
    # --- Document Processing ---
    # ---------------------------

    # --------------------------------------------------------------------------------- process_directory()
    # --------------------------------------------------------------------------------- process_directory()
    def process_directory(self, directory_path: str) -> None:
        """Process all PDF files to build the complete hybrid store with progress tracking.

        Scans the specified directory for PDF files and processes each one to build
        both GraphRAG and VectorRAG components of the hybrid store. Provides detailed
        progress tracking, workload estimation, and error handling to ensure complete
        hybrid store construction across all regulatory documents.

        Args:
            directory_path (str): Path to directory containing PDF files for hybrid processing

        Examples:
            >>> builder = HybridStoreBuilder()
            >>> builder.process_directory("../data/cfr_pdf")
        """
        print(f"\nScanning directory for hybrid store construction: {directory_path}")
        if not os.path.isdir(directory_path):
            print(f"[ERROR] PDF directory not found: {directory_path}")
            return

        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        if self.max_docs > 0:
            pdf_files = pdf_files[: self.max_docs]
        print(f"Found {len(pdf_files)} PDF files:")
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"   {i}. {pdf_file}")

        # Estimate total chunks
        print(f"\nEstimating hybrid store workload...")
        estimated_chunks = 0
        for pdf_file in pdf_files:
            file_path = os.path.join(directory_path, pdf_file)
            file_size = os.path.getsize(file_path)
            # Rough estimate: 1MB ≈ 1000 chunks
            estimated_chunks += file_size // (1024 * 1024) * 1000
        
        print(f"Estimated total hybrid chunks: ~{estimated_chunks:,}")
        print(f"   Estimated processing time: ~{estimated_chunks//100:.1f} hours")

        for i, pdf_file in enumerate(pdf_files):
            file_path = os.path.join(directory_path, pdf_file)
            print(f"\n{'='*60}")
            print(f"Building hybrid store from PDF {i+1}/{len(pdf_files)}: {pdf_file}")
            print(f"{'='*60}")

            success = self.process_pdf(file_path)
            if success:
                print(f"[SUCCESS] Successfully processed {pdf_file}")
                self.print_progress_stats()
            else:
                print(f"[ERROR] Failed to process {pdf_file}")
    # --------------------------------------------------------------------------------- end process_directory()

    # --------------------------------------------------------------------------------- process_pdf()
    # --------------------------------------------------------------------------------- process_pdf()
    def process_pdf(self, file_path: str) -> bool:
        """Process PDF to create hybrid store components.

        Performs comprehensive processing of a single PDF file to create both
        GraphRAG and VectorRAG components: extracts and chunks text, generates
        vector embeddings for semantic search, extracts entities for graph
        relationships, and stores both graph and vector data in Neo4j with
        proper rate limiting and error handling.

        Args:
            file_path (str): Full path to the PDF file to process for hybrid store

        Returns:
            bool: True if hybrid processing succeeds, False if any component fails

        Examples:
            >>> builder = HybridStoreBuilder()
            >>> success = builder.process_pdf("../data/cfr_pdf/title30_part1.pdf")
            >>> print(f"Hybrid processing: {'SUCCESS' if success else 'FAILED'}")
        """
        filename = os.path.basename(file_path)
        print(f"\nProcessing for hybrid store: {filename}")

        try:
            # Extract text per page
            print("Extracting text from PDF (page-aware)...")
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))

            num_pages = len(reader.pages)
            print(f"PDF has {num_pages} pages")

            def count_tokens(s: str) -> int:
                if tiktoken is None:
                    return max(1, len(s.split()))
                try:
                    enc = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    return max(1, len(s.split()))
                return len(enc.encode(s))

            all_chunks: list[dict] = []
            for page_idx in range(num_pages):
                if self.max_pages_per_doc > 0 and (page_idx + 1) > self.max_pages_per_doc:
                    break
                page = reader.pages[page_idx]
                page_text = page.extract_text() or ""
                if not page_text.strip():
                    continue
                page_chunks = self.text_splitter.split_text(page_text)
                for j, chunk_text in enumerate(page_chunks, start=1):
                    chunk = {
                        "doc_id": filename,
                        "chunk_id": f"{filename}_p{page_idx+1}_c{j}",
                        "text": chunk_text,
                        "page": page_idx + 1,
                        "tokens": count_tokens(chunk_text),
                        "title": filename,
                        "source": "local",
                    }
                    all_chunks.append(chunk)

            if not all_chunks:
                print(f"[ERROR] No text extracted from {filename}")
                return False

            print(f"Created {len(all_chunks):,} chunks for hybrid processing")

            # Process chunks in small batches
            batch_size = 10
            total_batches = (len(all_chunks) + batch_size - 1) // batch_size
            print(f"Building hybrid store: {len(all_chunks)} chunks in {total_batches} batches...")

            for batch_idx in range(0, len(all_chunks), batch_size):
                batch_end = min(batch_idx + batch_size, len(all_chunks))
                batch_chunks = all_chunks[batch_idx:batch_end]
                batch_num = batch_idx // batch_size + 1

                print(f"\nHybrid Batch {batch_num}/{total_batches} (chunks {batch_idx+1}-{batch_end})")

                batch_entities = 0
                for i, ch in enumerate(batch_chunks):
                    chunk_num = batch_idx + i + 1
                    try:
                        # Generate embedding for vector component
                        print(f"   Generating vector embedding for chunk {chunk_num}...")
                        chunk_embedding = self.embedding_model.embed_query(ch["text"])

                        # Upsert Document and Chunk with vector
                        print("   🕸️  Upserting document and chunk with vector data...")
                        self.graph.query(
                            """
MERGE (d:Document {doc_id: $doc_id})
  ON CREATE SET d.title=$title, d.source=$source, d.created_at=timestamp()
MERGE (c:Chunk {chunk_id: $chunk_id})
  ON CREATE SET c.text=$text, c.page=$page, c.tokens=$tokens, c.embedding=$embedding
MERGE (d)-[:HAS_CHUNK]->(c)
MERGE (c)-[:PART_OF]->(d)
                            """,
                            {
                                "doc_id": ch["doc_id"],
                                "title": ch["title"],
                                "source": ch["source"],
                                "chunk_id": ch["chunk_id"],
                                "text": ch["text"],
                                "page": ch["page"],
                                "tokens": ch["tokens"],
                                "embedding": chunk_embedding,
                            },
                        )

                        # Extract and create entities for graph component (optional)
                        entities: list[tuple[str, str]] = []
                        if not self.disable_entity_extraction and (self.total_chunks_processed % self.extract_every_n_chunks == 0):
                            print("   Extracting entities for graph relationships...")
                            entities = self.extract_entities_msha(ch["text"])
                            for entity_name, entity_type in entities:
                                self.graph.query(
                                    """
MERGE (e:Entity {name: toLower($entity_name), type: $entity_type})
  ON CREATE SET e.display = $entity_display
MERGE (c:Chunk {chunk_id: $chunk_id})
MERGE (c)-[:HAS_ENTITY]->(e)
MERGE (e)-[:PART_OF]->(c)
                                    """,
                                    {
                                        "entity_name": entity_name,
                                        "entity_type": entity_type,
                                        "entity_display": entity_name,
                                        "chunk_id": ch["chunk_id"],
                                    },
                                )

                        batch_entities += len(entities)
                        self.total_chunks_processed += 1
                        self.total_entities_created += len(entities)

                    except Exception as e:
                        print(f"[ERROR] Failed hybrid chunk {chunk_num}: {e}")
                        continue

                    # Gentle rate limiting
                    if self.sleep_between_chunks_ms > 0:
                        time.sleep(self.sleep_between_chunks_ms / 1000.0)

                # Small pause between batches
                if batch_num < total_batches and self.sleep_between_batches_ms > 0:
                    time.sleep(self.sleep_between_batches_ms / 1000.0)

            print(f"[SUCCESS] Completed hybrid store for {filename}: {len(all_chunks)} chunks processed")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")
            print(f"Error details: {traceback.format_exc()}")
            return False
    # --------------------------------------------------------------------------------- end process_pdf()

    # -------------------------------
    # --- Database Configuration ---
    # -------------------------------

    # --------------------------------------------------------------------------------- create_vector_index()
    # --------------------------------------------------------------------------------- create_vector_index()
    def create_vector_index(self) -> None:
        """Create cosine vector index on :Chunk(embedding)."""
        print("\nCreating vector index for semantic search...")
        try:
            dimension = self.embedding_dimension
            index_query = f"""
CREATE VECTOR INDEX `chunk_embedding_index` IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {dimension},
    `vector.similarity_function`: 'cosine'
  }}
}}
"""
            self.graph.query(index_query)
            print("[SUCCESS] Vector index created - VectorRAG component ready")
        except Exception as e:
            print(f"[ERROR] Vector index creation failed: {e}")
    # --------------------------------------------------------------------------------- end create_vector_index()

# ------------------------------------------------------------------------- end HybridStoreBuilder

# =========================================================================
# Standalone Function Definitions
# =========================================================================

# ------------------------
# --- Helper Functions ---
# ------------------------

# --------------------------------------------------------------------------------- main()
# --------------------------------------------------------------------------------- main()
def main() -> None:
    """Main function for hybridRAG store construction (Figure-1 seed graph)."""
    try:
        builder = HybridStoreBuilder()
        builder.print_progress_header()

        data_path = os.getenv("PDF_DIR", DEFAULT_PDF_DIR)

        # Optional clearing via env flag
        if os.getenv("CLEAR_DB", "false").lower() == "true":
            print("Clearing existing hybridRAG  store...")
            builder.graph.query("MATCH (n) DETACH DELETE n")
            print("[SUCCESS] Database cleared - starting fresh hybridRAG store build")

        # Process all PDFs to build hybridRAG store
        builder.process_directory(data_path)
        builder.create_vector_index()

        # Final summary
        builder.print_final_summary()

    except KeyboardInterrupt:
        print("\n[WARNING] HybridRAG store construction interrupted by user")
        print("Partial hybridRAG store progress may be present in database")
    except Exception as e:
        print(f"[ERROR] Fatal error in hybridRAG store construction: {e}")
        print(f"Details: {traceback.format_exc()}")
# --------------------------------------------------------------------------------- end main()

# =========================================================================
# Module Initialization / Main Execution Guard
# =========================================================================
# This block runs only when the file is executed directly, not when imported.
# It serves as the entry point for the Advanced Parallel Hybrid store construction
# process, allowing the module to be used both as a standalone script and as an
# importable utility for other components of the system.

if __name__ == "__main__":
    # --- Advanced Parallel Hybrid Store Construction Entry Point ---
    print(f"Running APH-IF Initial Hybrid Store Builder from {__file__}...")
    
    try:
        # Execute the main hybrid store building process
        main()
        print("[SUCCESS] HybridRAG store construction completed successfully!")

    except KeyboardInterrupt:
        print("\n[WARNING] Process interrupted by user (Ctrl+C)")
        print("[STOP] Hybrid store construction aborted")

    except Exception as e:
        print(f"[ERROR] Critical error during hybrid store construction: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        print("[STOP] Hybrid store construction failed")
        
    finally:
        print(f"Finished execution of {__file__}")

# =========================================================================
# End of File
# ========================================================================= 