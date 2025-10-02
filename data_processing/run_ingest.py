# -------------------------------------------------------------------------
# File: data_processing/run_ingest.py
# Author: Alexander Ricciardi
# Date: 2025-10-02
# [File Path] data_processing/run_ingest.py
# ------------------------------------------------------------------------
# Project: APH-IF — Advanced Parallel HybridRAG – Intelligent Fusion
#
# Module Functionality:
#   Unified, phase-based CLI runner for the data_processing service. Orchestrates
#   Phase 1 (PDF → chunks + embeddings), Phase 2 (entity extraction), and Phase 3
#   (relationship augmentation), with optional database clear and dry-run modes.
#
# Module Contents Overview:
# - Function: discover_pdfs
# - Function: clear_database
# - Function: fetch_all_chunks_from_neo4j
# - Function: fetch_chunk_entities_from_neo4j
# - Function: process_single_pdf
# - Function: phase1_init_chunks
# - Function: phase2_extract_entities
# - Function: phase3_augment_relationships
# - Function: main
#
# Dependencies / Imports:
# - Standard Library: argparse, logging, sys, datetime, pathlib
# - Third-Party: rich
# - Local Project Modules: config, chunker, docling_adapter, embeddings, neo4j_writer, utils.logging
#
# Usage / Integration:
#   This script is executed via `uv run python run_ingest.py` and is not intended
#   to be imported as a library entry point. Use flags to select phases.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""CLI runner for PDF ingestion and graph building.

Unified pipeline with clear phase separation:
- Phase 1 (--init-chunks): PDF → Docling → Chunks → Embeddings → Neo4j
- Phase 2 (--extract-entities): Extract entities from existing chunks
- Phase 3 (--augment): Augment relationships from existing entities
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

from config import load_config
from chunker import chunk_pages, generate_document_id, ChunkRecord
from docling_adapter import convert_pdf_to_pages, get_document_title
from embeddings import embed_chunks
from neo4j_writer import Neo4jWriter
from utils.logging import setup_logger, log_processing_summary, log_file_progress


# __________________________________________________________________________
# Global Constants / Variables

console = Console()

# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# File Discovery Utilities
# =========================================================================
#
# --------------------------------------------------------------------------------- discover_pdfs()
def discover_pdfs(data_pdf_dir: Path, max_docs: int = 0, specific_file: str = None) -> list[Path]:
    """Discover PDF files in the data directory.

    Args:
        data_pdf_dir: Directory containing PDFs
        max_docs: Maximum number of PDFs to process (0 = no limit)
        specific_file: Optional specific filename to process

    Returns:
        List of PDF file paths (sorted alphabetically)

    Raises:
        FileNotFoundError: If the PDF directory does not exist or the specific file is missing.
    """
    if not data_pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {data_pdf_dir}")

    if specific_file:
        # Process only the specified file
        pdf_path = data_pdf_dir / specific_file
        if not pdf_path.exists():
            raise FileNotFoundError(f"Specified PDF not found: {pdf_path}")
        return [pdf_path]

    pdf_files = sorted(data_pdf_dir.glob("*.pdf"))

    if max_docs > 0:
        pdf_files = pdf_files[:max_docs]

    return pdf_files
# --------------------------------------------------------------------------------- end discover_pdfs()

# =========================================================================
# Database Utilities (Clearing and Fetching)
# =========================================================================
#
# --------------------------------------------------------------------------------- clear_database()
def clear_database(writer: Neo4jWriter, confirm: bool = False) -> bool:
    """Clear database with confirmation prompt.

    Args:
        writer: Neo4jWriter instance
        confirm: Skip confirmation if True

    Returns:
        True if database was cleared, False otherwise
    """
    if not confirm:
        console.print("\n[bold yellow]⚠️  WARNING: Database Clear Operation[/bold yellow]")
        console.print("This will [bold red]DELETE ALL DATA[/bold red] in the database:")
        console.print(f"  • Database: {writer.database}")
        console.print(f"  • URI: {writer.driver._pool.address}")
        console.print("\nThis action [bold red]CANNOT BE UNDONE[/bold red]!\n")

        response = input("Are you sure you want to continue? Type 'yes' to confirm: ")
        if response.lower() != "yes":
            console.print("[yellow]Database clear cancelled[/yellow]")
            return False

    console.print("[yellow]Clearing database...[/yellow]")
    writer.clear_database(confirm=True)
    console.print("[green]✓ Database cleared successfully[/green]\n")
    return True
# --------------------------------------------------------------------------------- end clear_database()

# --------------------------------------------------------------------------------- fetch_all_chunks_from_neo4j()
def fetch_all_chunks_from_neo4j(writer: Neo4jWriter) -> list[ChunkRecord]:
    """Fetch all chunk data from Neo4j.

    Args:
        writer: Neo4jWriter instance

    Returns:
        List of ChunkRecord objects
    """
    query = """
    MATCH (c:Chunk)
    RETURN c.chunk_id as chunk_id,
           c.document_id as document_id,
           c.page as page,
           c.text as text,
           c.section as section,
           c.tokens as tokens,
           c.embedding as embedding
    ORDER BY c.chunk_id
    """

    chunks = []

    with writer.driver.session(database=writer.database) as session:
        result = session.run(query)

        for record in result:
            chunk = ChunkRecord(
                chunk_id=record["chunk_id"],
                document_id=record["document_id"],
                page=record["page"],
                text=record["text"],
                section=record["section"],
                tokens=record["tokens"] or 0,
                embedding=record["embedding"],
            )
            chunks.append(chunk)

    return chunks
# --------------------------------------------------------------------------------- end fetch_all_chunks_from_neo4j()

# --------------------------------------------------------------------------------- fetch_chunk_entities_from_neo4j()
def fetch_chunk_entities_from_neo4j(
    writer: Neo4jWriter,
    max_chunks: int = 0
) -> tuple[dict[str, list], dict[str, str]]:
    """Fetch chunk entities and texts from Neo4j.

    Args:
        writer: Neo4jWriter instance
        max_chunks: Maximum chunks to process (0 = no limit)

    Returns:
        Tuple of (chunk_entities, chunk_texts) dicts
    """
    from entities.normalizer import Entity

    query = """
    MATCH (c:Chunk)-[:HAS_ENTITY]->(e:Entity)
    WITH c, collect(DISTINCT e) as entities
    RETURN c.chunk_id as chunk_id,
           c.text as text,
           entities
    ORDER BY c.chunk_id
    """

    if max_chunks > 0:
        query += f"\nLIMIT {max_chunks}"

    chunk_entities = {}
    chunk_texts = {}

    with writer.driver.session(database=writer.database) as session:
        result = session.run(query)

        for record in result:
            chunk_id = record["chunk_id"]
            chunk_texts[chunk_id] = record["text"]

            # Convert Neo4j entity nodes to Entity objects
            entities = []
            for entity_node in record["entities"]:
                entity = Entity(
                    name=entity_node["name"],
                    type=entity_node["type"],
                    canonical_name=entity_node.get("canonical_name", entity_node["name"]),
                    occurrences=entity_node.get("occurrences", 1),
                    method="existing",
                )
                entities.append(entity)

            chunk_entities[chunk_id] = entities

    return chunk_entities, chunk_texts
# --------------------------------------------------------------------------------- end fetch_chunk_entities_from_neo4j()

# =========================================================================
# Phase Orchestration
# =========================================================================
#
# --------------------------------------------------------------------------------- process_single_pdf()
def process_single_pdf(
    pdf_path: Path,
    config,
    writer: Neo4jWriter,
    logger: logging.Logger,
) -> tuple[int, int]:
    """Process a single PDF file (Phase 1 only: PDF → chunks).

    Args:
        pdf_path: Path to PDF file
        config: ProcessingConfig instance
        writer: Neo4jWriter instance
        logger: Logger instance

    Returns:
        Tuple of (chunks_created, errors)

    Notes:
        Exceptions are caught and logged; on failure the function returns (0, 1).
    """
    try:
        # 1. Convert PDF to pages
        pages = convert_pdf_to_pages(pdf_path, max_pages=config.max_pages)

        if not pages:
            logger.warning(f"No pages extracted from {pdf_path.name}")
            return 0, 1

        # 2. Generate document ID and metadata
        document_id = generate_document_id(pdf_path.name)
        document_title = get_document_title(pdf_path)

        # 3. Upsert document
        writer.upsert_document(
            document_id=document_id,
            title=document_title,
            filename=pdf_path.name,
        )

        # 4. Chunk pages
        chunks = chunk_pages(
            pages=pages,
            document_id=document_id,
            chunk_size_chars=config.chunk_size_chars,
        )

        if not chunks:
            logger.warning(f"No chunks created for {pdf_path.name}")
            return 0, 1

        # 5. Generate embeddings
        chunks = embed_chunks(
            chunks=chunks,
            api_key=config.openai_api_key,
            batch_size=config.embedding_batch_size,
            show_progress=False,  # Disable per-file progress
        )

        # 6. Upsert chunks
        writer.upsert_chunks_batch(chunks, show_progress=False)

        logger.info(
            f"Successfully processed {pdf_path.name}: {len(pages)} pages, {len(chunks)} chunks"
        )

        return len(chunks), 0

    except Exception as e:
        logger.error(f"Failed to process {pdf_path.name}: {e}")
        return 0, 1
# --------------------------------------------------------------------------------- end process_single_pdf()

# --------------------------------------------------------------------------------- phase1_init_chunks()
def phase1_init_chunks(config, writer: Neo4jWriter, logger: logging.Logger) -> tuple[int, int]:
    """Phase 1: PDF ingestion to chunks with embeddings.

    Args:
        config: ProcessingConfig instance
        writer: Neo4jWriter instance
        logger: Logger instance

    Returns:
        Tuple of (total_chunks, total_errors)
    """
    console.print("\n[bold cyan]Phase 1: PDF → Chunks + Embeddings[/bold cyan]\n")

    # Discover PDFs
    try:
        pdf_files = discover_pdfs(config.data_pdf_dir, config.max_docs, getattr(config, 'specific_file', None))
    except Exception as e:
        console.print(f"[red]Failed to discover PDFs: {e}[/red]")
        return 0, 1

    if not pdf_files:
        console.print(f"[yellow]No PDF files found in {config.data_pdf_dir}[/yellow]")
        return 0, 0

    logger.info(f"Found {len(pdf_files)} PDF files to process")
    console.print(f"📄 Processing {len(pdf_files)} PDF files...\n")

    # Ensure vector index
    try:
        writer.ensure_vector_index(
            index_name=config.vector_index_name,
            node_label=config.chunk_label,
            dimensions=config.embedding_dimensions,
        )
    except Exception as e:
        console.print(f"[red]Failed to ensure vector index: {e}[/red]")
        return 0, 1

    # Process PDFs
    start_time = datetime.now()
    total_chunks = 0
    total_errors = 0

    for i, pdf_path in enumerate(pdf_files, start=1):
        log_file_progress(logger, pdf_path.name, i, len(pdf_files))
        console.print(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")

        chunks, errors = process_single_pdf(pdf_path, config, writer, logger)
        total_chunks += chunks
        total_errors += errors

    end_time = datetime.now()

    # Summary
    console.print("\n" + "=" * 80)
    console.print("[bold green]✓ Phase 1 Complete![/bold green]")
    console.print("=" * 80)
    console.print(f"Documents processed: [bold]{len(pdf_files)}[/bold]")
    console.print(f"Chunks created: [bold]{total_chunks}[/bold]")
    console.print(f"Errors: [bold]{total_errors}[/bold]")
    console.print(f"Duration: [bold]{end_time - start_time}[/bold]")
    console.print("=" * 80 + "\n")

    return total_chunks, total_errors
# --------------------------------------------------------------------------------- end phase1_init_chunks()

# --------------------------------------------------------------------------------- phase2_extract_entities()
def phase2_extract_entities(config, writer: Neo4jWriter, logger: logging.Logger) -> int:
    """Phase 2: Extract entities from existing chunks.

    Args:
        config: ProcessingConfig instance
        writer: Neo4jWriter instance
        logger: Logger instance

    Returns:
        Total entities extracted
    """
    console.print("\n[bold cyan]Phase 2: Entity Extraction[/bold cyan]\n")

    # Fetch all chunks
    console.print("📥 Fetching chunks from Neo4j...")
    start_time = datetime.now()

    try:
        chunks = fetch_all_chunks_from_neo4j(writer)
    except Exception as e:
        console.print(f"[red]Failed to fetch chunks: {e}[/red]")
        return 0

    if not chunks:
        console.print("[yellow]No chunks found in database[/yellow]")
        console.print("Tip: Run Phase 1 first: uv run python run_ingest.py --init-chunks\n")
        return 0

    console.print(f"✓ Found [bold]{len(chunks)}[/bold] chunks\n")

    # Extract entities
    console.print("🔍 Extracting entities from chunks...")

    from entities import extract_entities

    try:
        chunk_entities = extract_entities(chunks, config)
    except Exception as e:
        console.print(f"[red]Entity extraction failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 0

    total_entities = sum(len(ents) for ents in chunk_entities.values())
    console.print(f"✓ Extracted [bold]{total_entities}[/bold] unique entities\n")

    # Upsert entities to Neo4j
    console.print("💾 Upserting entities to Neo4j...")

    try:
        writer.upsert_entities_batch(chunk_entities, show_progress=True)
    except Exception as e:
        console.print(f"[red]Failed to upsert entities: {e}[/red]")
        return 0

    end_time = datetime.now()

    # Summary
    console.print("\n" + "=" * 80)
    console.print("[bold green]✓ Phase 2 Complete![/bold green]")
    console.print("=" * 80)
    console.print(f"Chunks processed: [bold]{len(chunks)}[/bold]")
    console.print(f"Entities extracted: [bold]{total_entities}[/bold]")
    console.print(f"Duration: [bold]{end_time - start_time}[/bold]")
    console.print("=" * 80 + "\n")

    return total_entities
# --------------------------------------------------------------------------------- end phase2_extract_entities()

# --------------------------------------------------------------------------------- phase3_augment_relationships()
def phase3_augment_relationships(config, writer: Neo4jWriter, logger: logging.Logger) -> int:
    """Phase 3: Augment relationships from existing entities.

    Args:
        config: ProcessingConfig instance
        writer: Neo4jWriter instance
        logger: Logger instance

    Returns:
        Total relationships detected
    """
    console.print("\n[bold cyan]Phase 3: Relationship Augmentation[/bold cyan]\n")

    # Fetch chunk entities
    console.print("📥 Fetching entities from Neo4j...")
    start_time = datetime.now()

    try:
        chunk_entities, chunk_texts = fetch_chunk_entities_from_neo4j(
            writer,
            max_chunks=config.max_augmentation_chunks
        )
    except Exception as e:
        console.print(f"[red]Failed to fetch entities: {e}[/red]")
        return 0

    if not chunk_entities:
        console.print("[yellow]No chunk entities found in database[/yellow]")
        console.print("Tip: Run Phase 2 first: uv run python run_ingest.py --extract-entities\n")
        return 0

    total_entities = sum(len(ents) for ents in chunk_entities.values())
    console.print(
        f"✓ Found [bold]{total_entities}[/bold] entities across "
        f"[bold]{len(chunk_entities)}[/bold] chunks\n"
    )

    # Augment relationships
    console.print("🔗 Detecting relationships using GPT-5-mini...")
    console.print(f"[yellow]Note: Processing {len(chunk_entities)} chunks (max: {config.max_augmentation_chunks})[/yellow]")
    console.print("[yellow]This may take time and incur API costs[/yellow]\n")

    from entities import augment_relationships

    try:
        chunk_relationships = augment_relationships(
            chunk_entities,
            config,
            chunk_texts=chunk_texts
        )
    except Exception as e:
        console.print(f"[red]Relationship augmentation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 0

    total_relationships = sum(len(rels) for rels in chunk_relationships.values())

    if total_relationships == 0:
        console.print("[yellow]No relationships detected[/yellow]\n")
        return 0

    console.print(f"✓ Detected [bold]{total_relationships}[/bold] relationships\n")

    # Upsert relationships to Neo4j
    console.print("💾 Upserting relationships to Neo4j...")

    try:
        writer.upsert_relationships_batch(chunk_relationships, show_progress=True)
    except Exception as e:
        console.print(f"[red]Failed to upsert relationships: {e}[/red]")
        return 0

    end_time = datetime.now()

    # Summary
    console.print("\n" + "=" * 80)
    console.print("[bold green]✓ Phase 3 Complete![/bold green]")
    console.print("=" * 80)
    console.print(f"Chunks processed: [bold]{len(chunk_entities)}[/bold]")
    console.print(f"Relationships detected: [bold]{total_relationships}[/bold]")
    console.print(f"Duration: [bold]{end_time - start_time}[/bold]")
    console.print("=" * 80 + "\n")

    return total_relationships
# --------------------------------------------------------------------------------- end phase3_augment_relationships()


# --------------------------------------------------------------------------------- main()
def main() -> int:
    """Main entry point for ingestion pipeline.

    Parses CLI arguments, loads configuration, connects to Neo4j (unless dry run), and
    executes the selected phases in order. Provides a final duration summary.

    Returns:
        Process exit code (0 on success, non-zero on errors or user cancel).
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="APH-IF Data Processing Pipeline - Phase-based ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clear database only
  python run_ingest.py --clear

  # Phase 1: Ingest PDFs to chunks
  python run_ingest.py --init-chunks

  # Phase 2: Extract entities from existing chunks
  python run_ingest.py --extract-entities

  # Phase 3: Augment relationships from existing entities
  python run_ingest.py --augment

  # Run all phases sequentially
  python run_ingest.py --all

  # Custom combinations
  python run_ingest.py --clear --init-chunks --extract-entities
  python run_ingest.py --init-chunks --extract-entities --augment
  python run_ingest.py --extract-entities --augment

  # Phase 1 with options
  python run_ingest.py --init-chunks --max-docs 1 --file "document.pdf"
        """
    )

    # Phase selection
    phase_group = parser.add_argument_group('Phase Selection')
    phase_group.add_argument(
        "--clear",
        action="store_true",
        help="Clear database (requires confirmation)",
    )
    phase_group.add_argument(
        "--init-chunks",
        action="store_true",
        help="Phase 1: Ingest PDFs → chunks + embeddings",
    )
    phase_group.add_argument(
        "--extract-entities",
        action="store_true",
        help="Phase 2: Extract entities from existing chunks",
    )
    phase_group.add_argument(
        "--augment",
        action="store_true",
        help="Phase 3: Augment relationships from existing entities",
    )
    phase_group.add_argument(
        "--all",
        action="store_true",
        help="Run all phases: clear → init-chunks → extract-entities → augment",
    )

    # Phase 1 options
    phase1_group = parser.add_argument_group('Phase 1 Options')
    phase1_group.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Maximum number of documents to process (0 = no limit)",
    )
    phase1_group.add_argument(
        "--file",
        type=str,
        help="Process a specific PDF file by name",
    )
    phase1_group.add_argument(
        "--chunk-size-chars",
        type=int,
        help="Override chunk size in characters",
    )

    # General options
    general_group = parser.add_argument_group('General Options')
    general_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run (no database writes)",
    )
    general_group.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompts",
    )

    args = parser.parse_args()

    # Expand --all flag
    if args.all:
        args.clear = True
        args.init_chunks = True
        args.extract_entities = True
        args.augment = True

    # Validate: at least one phase must be selected
    if not any([args.clear, args.init_chunks, args.extract_entities, args.augment]):
        parser.print_help()
        console.print("\n[red]Error: No phase selected. Use --help for usage examples.[/red]")
        return 1

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        console.print(f"[red]Failed to load configuration: {e}[/red]")
        return 1

    # Override config with CLI args
    if args.max_docs:
        config.max_docs = args.max_docs
    if args.file:
        config.specific_file = args.file
    if args.chunk_size_chars:
        config.chunk_size_chars = args.chunk_size_chars

    # Setup logging
    log_file = None
    if config.monitoring_enabled:
        config.monitoring_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = config.monitoring_dir / f"ingestion_{timestamp}.log"

    logger = setup_logger(
        name="data_processing",
        log_level="DEBUG" if config.verbose else "INFO",
        log_file=log_file,
        verbose=config.verbose,
    )

    # Print header
    console.print("\n" + "=" * 80)
    console.print("[bold blue]APH-IF Data Processing v2.0 - Unified Pipeline[/bold blue]")
    console.print("=" * 80)
    console.print(f"Neo4j database: {config.neo4j_database}")
    console.print(f"Embedding model: {config.embedding_model} ({config.embedding_dimensions}d)")
    console.print(f"Entity model: {config.spacy_model}")
    console.print(f"LLM model: {config.openai_model_mini}")

    if args.dry_run:
        console.print("\n[yellow]DRY RUN MODE - No database writes[/yellow]")
        logger.info("DRY RUN MODE enabled")

    console.print("=" * 80)

    # Initialize Neo4j writer
    if not args.dry_run:
        try:
            writer = Neo4jWriter(
                uri=config.neo4j_uri,
                username=config.neo4j_username,
                password=config.neo4j_password,
                database=config.neo4j_database,
                batch_size=config.neo4j_batch_size,
            )
        except Exception as e:
            console.print(f"[red]Failed to connect to Neo4j: {e}[/red]")
            return 1
    else:
        console.print("[yellow]Skipping Neo4j connection (dry run)[/yellow]")
        return 0

    # Execute phases in order
    overall_start = datetime.now()

    try:
        # Phase 0: Clear database
        if args.clear:
            if not clear_database(writer, confirm=args.yes):
                writer.close()
                return 0

        # Phase 1: Init chunks
        if args.init_chunks:
            total_chunks, total_errors = phase1_init_chunks(config, writer, logger)
            if total_errors > 0:
                console.print(f"[yellow]Phase 1 completed with {total_errors} errors[/yellow]")

        # Phase 2: Extract entities
        if args.extract_entities:
            total_entities = phase2_extract_entities(config, writer, logger)

        # Phase 3: Augment relationships
        if args.augment:
            total_relationships = phase3_augment_relationships(config, writer, logger)

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        writer.close()
        return 1
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        import traceback
        traceback.print_exc()
        writer.close()
        return 1

    # Final summary
    overall_end = datetime.now()
    overall_duration = overall_end - overall_start

    console.print("\n" + "=" * 80)
    console.print("[bold green]✓ All Selected Phases Complete![/bold green]")
    console.print("=" * 80)
    console.print(f"Total duration: [bold]{overall_duration}[/bold]")
    console.print("=" * 80 + "\n")

    console.print("[cyan]Next steps:[/cyan]")
    console.print("1. Verify data in Neo4j Browser")
    console.print("2. Refresh backend schema: cd backend && uv run python -m app.schema.refresh_schema")
    console.print("3. Test queries via backend API or Streamlit frontend\n")

    writer.close()
    return 0
# --------------------------------------------------------------------------------- end main()


# __________________________________________________________________________
# Module Initialization / Main Execution Guard (if applicable)
# This block runs only when the file is executed directly, not when imported. For a file that is
# part of a larger program, it typically contains module-specific tests or example usage. It should
# not contain the main application logic, which belongs in the project's entry point (e.g., main.py).
# --------------------------------------------------------------------------------- main_guard
if __name__ == "__main__":
    sys.exit(main())
# --------------------------------------------------------------------------------- end main_guard

# __________________________________________________________________________
# End of File
#
