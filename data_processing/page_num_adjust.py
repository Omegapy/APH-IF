# -------------------------------------------------------------------------
# File: data_processing/page_num_adjust.py
# Author: Alexander Ricciardi
# Date: 2025-10-02
# [File Path] data_processing/page_num_adjust.py
# ------------------------------------------------------------------------
# Project: APH-IF — Advanced Parallel HybridRAG – Intelligent Fusion
#
# Module Functionality:
#   One-time migration utility to adjust existing Chunk.page values to match
#   CFR-style pagination (roman numerals for preliminary pages, then numeric
#   pages starting from 1). Provides preview, update, and verification steps.
#
# Module Contents Overview:
# - Constants: ROMAN_NUMERALS
# - Function: get_preview_counts
# - Function: get_sample_mappings
# - Function: display_preview
# - Function: execute_update
# - Function: verify_changes
# - Function: display_verification
# - Function: main
#
# Dependencies / Imports:
# - Standard Library: argparse, logging, os, sys, datetime, pathlib, typing
# - Third-Party: rich
# - Local Modules: config.load_config, neo4j_writer.Neo4jWriter, utils.logging.setup_logger
#
# Usage / Integration:
#   Run via `uv run python page_num_adjust.py` to preview and/or execute the
#   page-number transformation. Intended as a controlled migration step.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""One-time migration script to adjust Chunk.page numbering.

Adjusts page numbers to match document pagination:
- Pages 1-10 → roman numerals (i, ii, iii, iv, v, vi, vii, viii, ix, x)
- Pages >10 → subtract 10, display as "1", "2", "3", ...

This is a corrective migration for existing data. Future ingests should apply
the same logic during write if needed.
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from config import load_config
from neo4j_writer import Neo4jWriter
from utils.logging import setup_logger

console = Console()

# Roman numeral mapping for pages 1-10
# __________________________________________________________________________
# Global Constants / Variables
ROMAN_NUMERALS = {
    1: "i",
    2: "ii",
    3: "iii",
    4: "iv",
    5: "v",
    6: "vi",
    7: "vii",
    8: "viii",
    9: "ix",
    10: "x",
}

# __________________________________________________________________________
# Standalone Function Definitions
#
# --------------------------------------------------------------------------------- get_preview_counts()
def get_preview_counts(
    writer: Neo4jWriter,
    document_id: Optional[str] = None
) -> dict:
    """Get counts of chunks by page number range.

    Args:
        writer: Neo4jWriter instance
        document_id: Optional document ID to filter by

    Returns:
        Dict with counts: {roman: int, numeric: int, already_string: int, total: int}
    """
    base_query = """
    MATCH (c:Chunk)
    {doc_filter}
    WITH c,
         CASE
             WHEN c.page IS NOT NULL AND toString(c.page) = c.page THEN 'already_string'
             WHEN c.page >= 1 AND c.page <= 10 THEN 'roman'
             WHEN c.page > 10 THEN 'numeric'
             ELSE 'other'
         END as category
    RETURN category, count(*) as count
    """

    doc_filter = "WHERE c.document_id = $document_id" if document_id else ""
    query = base_query.format(doc_filter=doc_filter)

    params = {"document_id": document_id} if document_id else {}

    with writer.driver.session(database=writer.database) as session:
        result = session.run(query, params)
        counts = {
            "roman": 0,
            "numeric": 0,
            "already_string": 0,
            "other": 0,
            "total": 0,
        }

        for record in result:
            category = record["category"]
            count = record["count"]
            counts[category] = count
            counts["total"] += count

        return counts
# --------------------------------------------------------------------------------- end get_preview_counts()

# --------------------------------------------------------------------------------- get_sample_mappings()
def get_sample_mappings(
    writer: Neo4jWriter,
    document_id: Optional[str] = None,
    limit: int = 20
) -> list[dict]:
    """Get sample chunks showing current → new page mappings.

    Args:
        writer: Neo4jWriter instance
        document_id: Optional document ID to filter by
        limit: Number of samples per document

    Returns:
        List of dicts with {document_id, chunk_id, current_page, new_page}
    """
    if document_id:
        query = """
        MATCH (c:Chunk)
        WHERE c.document_id = $document_id AND c.page IS NOT NULL
        WITH c,
             CASE
                 WHEN toString(c.page) = c.page THEN c.page
                 WHEN c.page = 1 THEN 'i'
                 WHEN c.page = 2 THEN 'ii'
                 WHEN c.page = 3 THEN 'iii'
                 WHEN c.page = 4 THEN 'iv'
                 WHEN c.page = 5 THEN 'v'
                 WHEN c.page = 6 THEN 'vi'
                 WHEN c.page = 7 THEN 'vii'
                 WHEN c.page = 8 THEN 'viii'
                 WHEN c.page = 9 THEN 'ix'
                 WHEN c.page = 10 THEN 'x'
                 WHEN c.page > 10 THEN toString(c.page - 10)
                 ELSE toString(c.page)
             END as new_page
        RETURN c.document_id as document_id,
               c.chunk_id as chunk_id,
               c.page as current_page,
               new_page
        ORDER BY c.document_id, c.page
        LIMIT $limit
        """
    else:
        query = """
        MATCH (c:Chunk)
        WHERE c.page IS NOT NULL
        WITH c,
             CASE
                 WHEN toString(c.page) = c.page THEN c.page
                 WHEN c.page = 1 THEN 'i'
                 WHEN c.page = 2 THEN 'ii'
                 WHEN c.page = 3 THEN 'iii'
                 WHEN c.page = 4 THEN 'iv'
                 WHEN c.page = 5 THEN 'v'
                 WHEN c.page = 6 THEN 'vi'
                 WHEN c.page = 7 THEN 'vii'
                 WHEN c.page = 8 THEN 'viii'
                 WHEN c.page = 9 THEN 'ix'
                 WHEN c.page = 10 THEN 'x'
                 WHEN c.page > 10 THEN toString(c.page - 10)
                 ELSE toString(c.page)
             END as new_page
        RETURN c.document_id as document_id,
               c.chunk_id as chunk_id,
               c.page as current_page,
               new_page
        ORDER BY c.document_id, c.page
        LIMIT $limit
        """

    params = {"limit": limit}
    if document_id:
        params["document_id"] = document_id

    samples = []
    with writer.driver.session(database=writer.database) as session:
        result = session.run(query, params)
        for record in result:
            samples.append({
                "document_id": record["document_id"],
                "chunk_id": record["chunk_id"],
                "current_page": record["current_page"],
                "new_page": record["new_page"],
            })

    return samples
# --------------------------------------------------------------------------------- end get_sample_mappings()

# --------------------------------------------------------------------------------- display_preview()
def display_preview(
    counts: dict,
    samples: list[dict],
    document_id: Optional[str] = None
) -> None:
    """Display preview of changes.

    Args:
        counts: Count dict from get_preview_counts()
        samples: Sample list from get_sample_mappings()
        document_id: Optional document ID being processed
    """
    console.print("\n[bold cyan]Preview: Page Number Transformation[/bold cyan]")

    if document_id:
        console.print(f"Scope: Document [bold]{document_id}[/bold]")
    else:
        console.print("Scope: [bold]All documents[/bold]")

    console.print()

    # Counts table
    table = Table(title="Chunk Counts by Category")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="magenta")
    table.add_column("Description", style="dim")

    table.add_row(
        "Roman (1-10)",
        str(counts["roman"]),
        "Will be converted to i, ii, iii, ..., x"
    )
    table.add_row(
        "Numeric (>10)",
        str(counts["numeric"]),
        "Will be converted to page-10 (1, 2, 3, ...)"
    )
    table.add_row(
        "Already String",
        str(counts["already_string"]),
        "Already converted, will be skipped"
    )
    if counts["other"] > 0:
        table.add_row(
            "Other",
            str(counts["other"]),
            "Pages outside expected range"
        )
    table.add_row("[bold]Total", f"[bold]{counts['total']}", "")

    console.print(table)
    console.print()

    # Sample mappings
    if samples:
        console.print("[bold]Sample Mappings:[/bold]")
        sample_table = Table()
        sample_table.add_column("Document ID", style="cyan", no_wrap=True)
        sample_table.add_column("Current Page", justify="right", style="yellow")
        sample_table.add_column("→", justify="center")
        sample_table.add_column("New Page", justify="right", style="green")

        for sample in samples[:15]:  # Show first 15
            current = str(sample["current_page"])
            new = sample["new_page"]

            # Highlight changes
            if current != new:
                sample_table.add_row(
                    sample["document_id"],
                    current,
                    "→",
                    f"[bold green]{new}[/bold green]"
                )
            else:
                sample_table.add_row(
                    sample["document_id"],
                    current,
                    "=",
                    f"[dim]{new}[/dim]"
                )

        console.print(sample_table)
        console.print()
# --------------------------------------------------------------------------------- end display_preview()

# --------------------------------------------------------------------------------- execute_update()
def execute_update(
    writer: Neo4jWriter,
    document_id: Optional[str] = None,
    dry_run: bool = False
) -> int:
    """Execute the page number update.

    Args:
        writer: Neo4jWriter instance
        document_id: Optional document ID to filter by
        dry_run: If True, don't actually update

    Returns:
        Number of chunks updated
    """
    if dry_run:
        console.print("[yellow]DRY RUN: No changes will be made[/yellow]\n")
        return 0

    # Build update query with proper WHERE clause handling
    if document_id:
        query = """
        MATCH (c:Chunk)
        WHERE c.document_id = $document_id
          AND c.page IS NOT NULL
          AND toString(c.page) <> c.page
        SET c.page = CASE
            WHEN c.page = 1 THEN 'i'
            WHEN c.page = 2 THEN 'ii'
            WHEN c.page = 3 THEN 'iii'
            WHEN c.page = 4 THEN 'iv'
            WHEN c.page = 5 THEN 'v'
            WHEN c.page = 6 THEN 'vi'
            WHEN c.page = 7 THEN 'vii'
            WHEN c.page = 8 THEN 'viii'
            WHEN c.page = 9 THEN 'ix'
            WHEN c.page = 10 THEN 'x'
            WHEN c.page > 10 THEN toString(c.page - 10)
            ELSE toString(c.page)
        END,
        c.updated_at = datetime()
        RETURN count(c) as updated_count
        """
    else:
        query = """
        MATCH (c:Chunk)
        WHERE c.page IS NOT NULL
          AND toString(c.page) <> c.page
        SET c.page = CASE
            WHEN c.page = 1 THEN 'i'
            WHEN c.page = 2 THEN 'ii'
            WHEN c.page = 3 THEN 'iii'
            WHEN c.page = 4 THEN 'iv'
            WHEN c.page = 5 THEN 'v'
            WHEN c.page = 6 THEN 'vi'
            WHEN c.page = 7 THEN 'vii'
            WHEN c.page = 8 THEN 'viii'
            WHEN c.page = 9 THEN 'ix'
            WHEN c.page = 10 THEN 'x'
            WHEN c.page > 10 THEN toString(c.page - 10)
            ELSE toString(c.page)
        END,
        c.updated_at = datetime()
        RETURN count(c) as updated_count
        """

    params = {"document_id": document_id} if document_id else {}

    console.print("[yellow]Executing update query...[/yellow]")

    with writer.driver.session(database=writer.database) as session:
        result = session.run(query, params)
        updated_count = result.single()["updated_count"]

    console.print(f"[green]✓ Updated {updated_count} chunks[/green]\n")

    return updated_count
# --------------------------------------------------------------------------------- end execute_update()

# --------------------------------------------------------------------------------- verify_changes()
def verify_changes(
    writer: Neo4jWriter,
    document_id: Optional[str] = None
) -> dict:
    """Verify the changes were applied correctly.

    Args:
        writer: Neo4jWriter instance
        document_id: Optional document ID to filter by

    Returns:
        Dict with verification results
    """
    # Build queries with proper WHERE clause handling
    if document_id:
        roman_query = """
        MATCH (c:Chunk)
        WHERE c.document_id = $document_id
          AND c.page IN ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
        RETURN count(c) as count
        """

        numeric_query = """
        MATCH (c:Chunk)
        WHERE c.document_id = $document_id
          AND c.page IS NOT NULL
          AND toString(c.page) = c.page
          AND NOT c.page IN ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
        RETURN count(c) as count
        """

        sample_query = """
        MATCH (c:Chunk)
        WHERE c.document_id = $document_id
        RETURN c.document_id as document_id,
               c.chunk_id as chunk_id,
               c.page as page
        ORDER BY c.document_id, c.chunk_id
        LIMIT 20
        """
    else:
        roman_query = """
        MATCH (c:Chunk)
        WHERE c.page IN ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
        RETURN count(c) as count
        """

        numeric_query = """
        MATCH (c:Chunk)
        WHERE c.page IS NOT NULL
          AND toString(c.page) = c.page
          AND NOT c.page IN ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
        RETURN count(c) as count
        """

        sample_query = """
        MATCH (c:Chunk)
        RETURN c.document_id as document_id,
               c.chunk_id as chunk_id,
               c.page as page
        ORDER BY c.document_id, c.chunk_id
        LIMIT 20
        """

    params = {"document_id": document_id} if document_id else {}

    results = {
        "roman_count": 0,
        "numeric_count": 0,
        "samples": [],
    }

    with writer.driver.session(database=writer.database) as session:
        # Count roman pages
        result = session.run(roman_query, params)
        results["roman_count"] = result.single()["count"]

        # Count numeric pages
        result = session.run(numeric_query, params)
        results["numeric_count"] = result.single()["count"]

        # Get samples
        result = session.run(sample_query, params)
        for record in result:
            results["samples"].append({
                "document_id": record["document_id"],
                "chunk_id": record["chunk_id"],
                "page": record["page"],
            })

    return results
# --------------------------------------------------------------------------------- end verify_changes()

# --------------------------------------------------------------------------------- display_verification()
def display_verification(results: dict) -> None:
    """Display verification results.

    Args:
        results: Results dict from verify_changes()
    """
    console.print("\n[bold cyan]Post-Update Verification[/bold cyan]\n")

    # Counts
    table = Table(title="Updated Page Counts")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="magenta")

    table.add_row("Roman pages (i-x)", str(results["roman_count"]))
    table.add_row("Numeric pages (1, 2, ...)", str(results["numeric_count"]))
    table.add_row(
        "[bold]Total",
        f"[bold]{results['roman_count'] + results['numeric_count']}"
    )

    console.print(table)
    console.print()

    # Samples
    if results["samples"]:
        console.print("[bold]Sample Pages After Update:[/bold]")
        sample_table = Table()
        sample_table.add_column("Document ID", style="cyan", no_wrap=True)
        sample_table.add_column("Chunk ID", style="dim")
        sample_table.add_column("Page", justify="right", style="green")

        for sample in results["samples"][:10]:
            sample_table.add_row(
                sample["document_id"],
                sample["chunk_id"],
                str(sample["page"])
            )

        console.print(sample_table)
        console.print()
# --------------------------------------------------------------------------------- end display_verification()

# --------------------------------------------------------------------------------- main()
def main() -> int:
    """Main entry point for page number adjustment script."""
    parser = argparse.ArgumentParser(
        description="Adjust Chunk.page numbering to match document pagination",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview changes)
  uv run python page_num_adjust.py --dry-run

  # Execute with confirmation prompt
  uv run python page_num_adjust.py

  # Execute without prompt
  uv run python page_num_adjust.py --yes

  # Scope to a single document
  uv run python page_num_adjust.py --document-id cfr_2024_title30_vol1 --yes

  # Force execution on production
  uv run python page_num_adjust.py --force --yes
        """
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing updates"
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--document-id",
        type=str,
        help="Limit scope to a single document ID"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow execution on production environment"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        console.print(f"[red]Failed to load configuration: {e}[/red]")
        return 1

    # Setup logging
    config.monitoring_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.monitoring_dir / f"page_num_adjust_{timestamp}.log"

    logger = setup_logger(
        name="page_num_adjust",
        log_level="DEBUG" if config.verbose else "INFO",
        log_file=log_file,
        verbose=config.verbose,
    )

    # Print header
    console.print("\n" + "=" * 80)
    console.print("[bold blue]Chunk Page Number Adjustment Script[/bold blue]")
    console.print("=" * 80)

    # Pre-flight: Environment check
    app_env = os.getenv("APP_ENV", "development").lower()
    console.print(f"Environment: [bold]{app_env}[/bold]")
    console.print(f"Neo4j URI: {config.neo4j_uri}")
    console.print(f"Neo4j Database: [bold]{config.neo4j_database}[/bold]")

    if args.document_id:
        console.print(f"Scope: Document [bold]{args.document_id}[/bold]")
    else:
        console.print("Scope: [bold yellow]All documents[/bold yellow]")

    if args.dry_run:
        console.print("\n[yellow]DRY RUN MODE - No changes will be made[/yellow]")

    console.print("=" * 80 + "\n")

    # Safety check: refuse production without --force
    if app_env == "production" and not args.force:
        console.print(
            "[bold red]ERROR: Running on production environment![/bold red]\n"
            "This script will modify data in the production database.\n"
            "Use --force to proceed if you're sure.\n"
        )
        logger.error("Refused to run on production without --force flag")
        return 1

    # Initialize Neo4j writer
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
        logger.error(f"Neo4j connection failed: {e}")
        return 1

    try:
        # Step 1: Get preview
        logger.info("Getting preview counts...")
        counts = get_preview_counts(writer, args.document_id)
        samples = get_sample_mappings(writer, args.document_id)

        display_preview(counts, samples, args.document_id)

        # Check if there's anything to update
        chunks_to_update = counts["roman"] + counts["numeric"]
        if chunks_to_update == 0:
            console.print(
                "[green]No chunks need updating. "
                "All pages are already in the correct format.[/green]\n"
            )
            logger.info("No chunks to update, exiting")
            writer.close()
            return 0

        logger.info(
            f"Preview: {chunks_to_update} chunks to update "
            f"({counts['roman']} roman, {counts['numeric']} numeric)"
        )

        # Step 2: Confirmation (unless --yes or --dry-run)
        if not args.dry_run and not args.yes:
            console.print("[bold yellow]⚠️  Confirmation Required[/bold yellow]")
            console.print(
                f"This will update [bold]{chunks_to_update}[/bold] chunks in the database."
            )
            console.print("This operation modifies data and cannot be undone easily.\n")

            response = input("Proceed with update? Type 'yes' to confirm: ")
            if response.lower() != "yes":
                console.print("[yellow]Operation cancelled[/yellow]")
                logger.info("Operation cancelled by user")
                writer.close()
                return 0

        # Step 3: Execute update
        logger.info(f"Starting update (dry_run={args.dry_run})...")
        start_time = datetime.now()

        updated_count = execute_update(writer, args.document_id, args.dry_run)

        end_time = datetime.now()
        duration = end_time - start_time

        logger.info(f"Update completed: {updated_count} chunks updated in {duration}")

        # Step 4: Verify changes (only if not dry-run)
        if not args.dry_run and updated_count > 0:
            logger.info("Verifying changes...")
            verification = verify_changes(writer, args.document_id)
            display_verification(verification)

            logger.info(
                f"Verification: {verification['roman_count']} roman pages, "
                f"{verification['numeric_count']} numeric pages"
            )

        # Final summary
        console.print("=" * 80)
        console.print("[bold green]✓ Script Complete![/bold green]")
        console.print("=" * 80)
        if not args.dry_run:
            console.print(f"Chunks updated: [bold]{updated_count}[/bold]")
        console.print(f"Duration: [bold]{duration}[/bold]")
        console.print(f"Log file: [dim]{log_file}[/dim]")
        console.print("=" * 80 + "\n")

        if not args.dry_run:
            console.print("[cyan]Next steps:[/cyan]")
            console.print("1. Verify data in Neo4j Browser")
            console.print(
                "2. Check backend citation formatting displays correctly (p.i, p.v, p.1, etc.)"
            )
            console.print("3. Test queries that reference page numbers\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        logger.info("Operation cancelled by user (KeyboardInterrupt)")
        writer.close()
        return 1
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {e}[/red]")
        logger.error(f"Unexpected error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        writer.close()
        return 1

    writer.close()
    return 0
# --------------------------------------------------------------------------------- end main()

# __________________________________________________________________________
# Module Initialization / Main Execution Guard (if applicable)
#
if __name__ == "__main__":
    sys.exit(main())

# __________________________________________________________________________
# End of File
#
