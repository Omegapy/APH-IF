# -------------------------------------------------------------------------
# File: data_processing/utils/logging.py
# Author: Alexander Ricciardi
# Date: 2025-10-02
# [File Path] data_processing/utils/logging.py
# ------------------------------------------------------------------------
# Project: APH-IF — Advanced Parallel HybridRAG – Intelligent Fusion
#
# Module Functionality:
#   Logging utilities providing consistent, structured console/file logging for
#   the data_processing service, including rich console output.
#
# Module Contents Overview:
# - Function: setup_logger
# - Function: log_processing_summary
# - Function: log_file_progress
#
# Dependencies / Imports:
# - Standard Library: logging, sys, datetime, pathlib, typing
# - Third-Party: rich
#
# Usage / Integration:
#   Imported by Phase 1–3 ingestion flows to configure loggers and print
#   progress/summary information to console and optional log files.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""Logging utilities for data processing module.

Provides consistent, structured logging with support for both console output and file logging.
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler

# Global console instance for rich output
console = Console()

# __________________________________________________________________________
# Standalone Function Definitions
#
# --------------------------------------------------------------------------------- setup_logger()
def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    verbose: bool = True,
) -> logging.Logger:
    """Set up a logger with console and optional file handlers.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        verbose: If True, use rich console output; if False, use basic formatting

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Console handler
    if verbose:
        # Use rich handler for pretty output
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        console_handler.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(console_handler)
    else:
        # Use basic console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
# --------------------------------------------------------------------------------- end setup_logger()

# --------------------------------------------------------------------------------- log_processing_summary()
def log_processing_summary(
    logger: logging.Logger,
    docs_processed: int,
    chunks_created: int,
    entities_created: int = 0,
    relationships_created: int = 0,
    errors: int = 0,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
    """Log a formatted summary of processing results.

    Args:
        logger: Logger instance to use
        docs_processed: Number of documents processed
        chunks_created: Number of chunks created
        entities_created: Number of entities extracted
        relationships_created: Number of relationships created
        errors: Number of errors encountered
        start_time: Processing start time
        end_time: Processing end time
    """
    console.print("\n" + "=" * 80)
    console.print("[bold green]Processing Summary[/bold green]")
    console.print("=" * 80)

    console.print(f"Documents processed: [cyan]{docs_processed}[/cyan]")
    console.print(f"Chunks created: [cyan]{chunks_created}[/cyan]")

    if entities_created > 0:
        console.print(f"Entities extracted: [cyan]{entities_created}[/cyan]")

    if relationships_created > 0:
        console.print(f"Relationships created: [cyan]{relationships_created}[/cyan]")

    if errors > 0:
        console.print(f"Errors encountered: [red]{errors}[/red]")

    if start_time and end_time:
        duration = end_time - start_time
        console.print(f"Total time: [cyan]{duration}[/cyan]")

    console.print("=" * 80 + "\n")

    logger.info(
        f"Processing complete: {docs_processed} docs, "
        f"{chunks_created} chunks, {entities_created} entities, "
        f"{relationships_created} relationships, {errors} errors"
    )
# --------------------------------------------------------------------------------- end log_processing_summary()

# --------------------------------------------------------------------------------- log_file_progress()
def log_file_progress(
    logger: logging.Logger,
    filename: str,
    current: int,
    total: int,
    chunks: int = 0,
) -> None:
    """Log progress for a file being processed.

    Args:
        logger: Logger instance to use
        filename: Name of file being processed
        current: Current file number
        total: Total number of files
        chunks: Number of chunks created (optional)
    """
    status = f"[{current}/{total}] Processing: {filename}"
    if chunks > 0:
        status += f" ({chunks} chunks)"

    console.print(f"[yellow]{status}[/yellow]")
    logger.info(status)
# --------------------------------------------------------------------------------- end log_file_progress()

# __________________________________________________________________________
# End of File
#
