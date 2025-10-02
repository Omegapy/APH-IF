# -------------------------------------------------------------------------
# File: data_processing/embeddings.py
# Author: Alexander Ricciardi
# Date: 2025-10-02
# [File Path] data_processing/embeddings.py
# ------------------------------------------------------------------------
# Project: APH-IF — Advanced Parallel HybridRAG – Intelligent Fusion
#
# Module Functionality:
#   Embedding utilities built on OpenAI text-embedding-3-large with batched calls
#   and retry logic. Provides a simple helper to attach embeddings to chunks.
#
# Module Contents Overview:
# - Class: EmbeddingGenerator
# - Function: embed_chunks
#
# Dependencies / Imports:
# - Standard Library: logging, time, typing
# - Third-Party: openai, tqdm
#
# Usage / Integration:
#   Used in Phase 1 ingestion to generate embeddings for chunk records prior to
#   Neo4j upserts and vector index usage.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""OpenAI embeddings wrapper for text-embedding-3-large.

Provides batched, retry-enabled embedding generation for text chunks.
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import logging
import time
from typing import Optional

from openai import OpenAI, RateLimitError
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ____________________________________________________________________________
# Class Definitions
# ------------------------------------------------------------------------- class EmbeddingGenerator
class EmbeddingGenerator:
    """Wrapper for OpenAI text-embedding-3-large with batching and retry logic.

    Attributes:
        client: OpenAI client bound to the provided API key.
        model: Embedding model identifier (default: "text-embedding-3-large").
        dimensions: Expected embedding vector size (default: 3072).
        batch_size: Number of texts per API call.
        max_retries: Maximum retry attempts on transient failures.
        initial_retry_delay: Initial backoff delay in seconds.
    """

    # -------------------------------------------------------------- __init__()
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int = 3072,
        batch_size: int = 64,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
    ) -> None:
        """Initialize the embedding generator.

        Args:
            api_key: OpenAI API key
            model: Embedding model name
            dimensions: Vector dimensions (3072 for text-embedding-3-large)
            batch_size: Number of texts to embed per API call
            max_retries: Maximum number of retry attempts
            initial_retry_delay: Initial delay between retries (seconds)
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

        logger.info(
            f"Initialized EmbeddingGenerator: model={model}, "
            f"dimensions={dimensions}, batch_size={batch_size}"
        )
    # -------------------------------------------------------------- end __init__()

    # -------------------------------------------------------------- generate_embeddings()
    def generate_embeddings(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors (same order as input texts)

        Raises:
            ValueError: If any embedding has incorrect dimensions
            RuntimeError: If API calls fail after all retries
        """
        if not texts:
            logger.warning("No texts provided for embedding generation")
            return []

        logger.info(f"Generating embeddings for {len(texts)} texts")

        all_embeddings: list[list[float]] = []

        # Process in batches
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=num_batches,
                desc="Generating embeddings",
                unit="batch",
            )

        for start_idx in iterator:
            end_idx = min(start_idx + self.batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]

            # Generate embeddings for batch with retry logic
            batch_embeddings = self._generate_batch_with_retry(batch_texts)

            # Validate dimensions
            for i, emb in enumerate(batch_embeddings):
                if len(emb) != self.dimensions:
                    raise ValueError(
                        f"Embedding dimension mismatch at index {start_idx + i}: "
                        f"expected {self.dimensions}, got {len(emb)}"
                    )

            all_embeddings.extend(batch_embeddings)

        logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings
    # -------------------------------------------------------------- end generate_embeddings()

    # -------------------------------------------------------------- _generate_batch_with_retry()
    def _generate_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch with exponential backoff retry.

        Args:
            texts: Batch of text strings

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If all retries fail
        """
        retry_delay = self.initial_retry_delay

        for attempt in range(self.max_retries):
            try:
                # Call OpenAI API
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model,
                    dimensions=self.dimensions,
                )

                # Extract embeddings in order
                embeddings = [item.embedding for item in response.data]

                return embeddings

            except RateLimitError as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Rate limit hit on attempt {attempt + 1}/{self.max_retries}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error("Max retries exceeded for rate limit error")
                    raise RuntimeError(f"Failed after {self.max_retries} retries: {e}")

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"API error on attempt {attempt + 1}/{self.max_retries}: {e}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Max retries exceeded: {e}")
                    raise RuntimeError(f"Failed after {self.max_retries} retries: {e}")

        # Should never reach here
        raise RuntimeError("Unexpected state in retry logic")
    # -------------------------------------------------------------- end _generate_batch_with_retry()

# ------------------------------------------------------------------------- end class EmbeddingGenerator

# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# Embedding Utilities
# =========================================================================
#
# --------------------------------------------------------------------------------- embed_chunks()
def embed_chunks(
    chunks: list,
    api_key: str,
    batch_size: int = 64,
    show_progress: bool = True,
) -> list:
    """Generate embeddings for chunk records and update them in place.

    Args:
        chunks: List of ChunkRecord objects
        api_key: OpenAI API key
        batch_size: Batch size for API calls
        show_progress: Whether to show progress bar

    Returns:
        The same list of chunks with embeddings added
    """
    if not chunks:
        return chunks

    # Extract texts
    texts = [chunk.text for chunk in chunks]

    # Generate embeddings
    generator = EmbeddingGenerator(api_key=api_key, batch_size=batch_size)
    embeddings = generator.generate_embeddings(texts, show_progress=show_progress)

    # Update chunks with embeddings
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    return chunks
# --------------------------------------------------------------------------------- end embed_chunks()

# __________________________________________________________________________
# End of File
#
