# -------------------------------------------------------------------------
# File: data_processing/docling_adapter.py
# Author: Alexander Ricciardi
# Date: 2025-10-02
# [File Path] data_processing/docling_adapter.py
# ------------------------------------------------------------------------
# Project: APH-IF — Advanced Parallel HybridRAG – Intelligent Fusion
#
# Module Functionality:
#   Adapter utilities for converting PDFs to structured page records using Docling.
#   Preserves page numbers and extracts heading metadata where available.
#
# Module Contents Overview:
# - Class (dataclass): PageRecord
# - Function: convert_pdf_to_pages
# - Function: get_document_title
#
# Dependencies / Imports:
# - Standard Library: logging, dataclasses, pathlib, typing
# - Third-Party: docling
#
# Usage / Integration:
#   Used by Phase 1 of data_processing pipeline to transform PDFs into page records
#   for downstream chunking and embeddings. Not an execution entry point.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""Docling adapter for PDF to structured page conversion.

Converts PDF documents to structured page records with text, page numbers,
and optional section/heading metadata using Docling.
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

# ____________________________________________________________________________
# Class Definitions
# TODO: Consider @dataclass(slots=True, kw_only=True) in a dedicated refactor PR.
# ------------------------------------------------------------------------- class PageRecord
@dataclass
class PageRecord:
    """Record for a single page from a PDF document.

    Attributes:
        page: Page number as detected by Docling provenance.
        text: Concatenated text content for the page.
        sections: Optional structural section identifiers for the page.
        headings: Headings/titles detected for the page, if available.
    """

    page: int
    text: str
    sections: list[str]
    headings: list[str]
# ------------------------------------------------------------------------- end class PageRecord

# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# PDF Conversion Utilities
# =========================================================================
#
# --------------------------------------------------------------------------------- convert_pdf_to_pages()
def convert_pdf_to_pages(source_path: Path, max_pages: int = 0) -> list[PageRecord]:
    """Convert a PDF to a list of PageRecord objects using Docling.

    Args:
        source_path: Path to the PDF file
        max_pages: Maximum number of pages to process (0 = no limit)

    Returns:
        List of PageRecord objects, one per page

    Raises:
        FileNotFoundError: If the source PDF does not exist
        ValueError: If the PDF cannot be processed or returns empty content
    """
    if not source_path.exists():
        raise FileNotFoundError(f"PDF file not found: {source_path}")

    logger.info(f"Converting PDF with Docling: {source_path.name}")

    try:
        # Initialize Docling converter
        converter = DocumentConverter()

        # Convert the PDF
        result = converter.convert(str(source_path))

        if not result or not hasattr(result, "document"):
            logger.warning(f"Docling returned empty result for {source_path.name}")
            raise ValueError(f"Docling conversion returned empty result for {source_path.name}")

        doc = result.document
        page_texts = {}  # page_no -> list of text items
        page_headings = {}  # page_no -> list of headings

        # Use doc.texts collection which is more efficient than iterate_items()
        # According to GitHub discussion, texts preserve provenance metadata
        if hasattr(doc, "texts"):
            for text_item in doc.texts:
                # Extract page numbers from provenance
                page_nos = set()
                if hasattr(text_item, "prov") and text_item.prov:
                    for prov_ref in text_item.prov:
                        if hasattr(prov_ref, "page_no"):
                            page_nos.add(prov_ref.page_no)

                # Add text to all pages it appears on
                if page_nos and hasattr(text_item, "text") and text_item.text:
                    text = text_item.text.strip()
                    if text:
                        for page_no in page_nos:
                            if page_no not in page_texts:
                                page_texts[page_no] = []
                                page_headings[page_no] = []

                            page_texts[page_no].append(text)

                            # Check for headings
                            if hasattr(text_item, "label") and text_item.label:
                                label_str = str(text_item.label).lower()
                                if "heading" in label_str or "title" in label_str:
                                    page_headings[page_no].append(text)

        # Convert to PageRecord objects
        pages: list[PageRecord] = []

        if page_texts:
            for page_no in sorted(page_texts.keys()):
                if max_pages > 0 and page_no > max_pages:
                    break

                page_text = "\n".join(page_texts[page_no])

                if page_text:
                    pages.append(
                        PageRecord(
                            page=page_no,
                            text=page_text,
                            sections=[],
                            headings=page_headings.get(page_no, []),
                        )
                    )

            logger.info(
                f"Extracted {len(pages)} pages using provenance metadata "
                f"(page range: {min(page_texts.keys())}-{max(page_texts.keys())})"
            )
        else:
            # Fallback: Extract text without page info
            logger.warning(
                f"No page provenance found for {source_path.name}. "
                "This may be a DOCX file or unsupported format. "
                "Using text export as single page."
            )
            # Use export_to_text() for a cleaner fallback (not markdown which adds formatting)
            full_text = doc.export_to_text() if hasattr(doc, "export_to_text") else doc.export_to_markdown()
            if full_text:
                pages = [
                    PageRecord(
                        page=1,
                        text=full_text,
                        sections=[],
                        headings=[],
                    )
                ]

        if not pages:
            logger.warning(
                f"No pages extracted from {source_path.name}. "
                "PDF may be empty or unsupported format."
            )
            raise ValueError(f"No pages extracted from {source_path.name}")

        logger.info(
            f"Successfully converted {source_path.name}: "
            f"{len(pages)} pages, "
            f"{sum(len(p.text) for p in pages)} total chars"
        )

        return pages

    except Exception as e:
        logger.error(f"Failed to convert PDF {source_path.name}: {e}")
        raise

# --------------------------------------------------------------------------------- end convert_pdf_to_pages()

# =========================================================================
# Metadata Utilities
# =========================================================================
#
def get_document_title(source_path: Path) -> str:
    """Extract document title from PDF metadata or filename.

    Args:
        source_path: Path to the PDF file

    Returns:
        Document title (falls back to filename without extension)
    """
    # For now, use filename as title
    # TODO: Could extract from PDF metadata if Docling provides it
    return source_path.stem

# --------------------------------------------------------------------------------- end get_document_title()

# __________________________________________________________________________
# End of File
#
