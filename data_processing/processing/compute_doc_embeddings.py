# -------------------------------------------------------------------------
# File: compute_doc_embeddings.py
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 08-09-2025
# File Path: data_processing/processing/compute_doc_embeddings.py
# -------------------------------------------------------------------------

# --- Module Objective ---
#   This module computes document-level embeddings from a local CSV file containing
#   precomputed chunk embeddings. It averages the chunk-level vectors to produce a
#   single embedding per document and stores the output in a new CSV file.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Function: compute_document_embeddings
# - Function: save_document_embeddings
# - Function: main
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: os, csv, sys
# - Third-Party: numpy
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# This module is designed to be executed as a standalone script:
#
#     python compute_doc_embeddings.py <input_csv_path> <output_csv_path>
#
# It reads chunk-level embeddings from the input CSV and writes the document-level
# averaged embeddings to the output CSV. The CSV must contain columns: 'document_id', 'chunk_id', 'embedding'.
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Compute and store :Document.doc_embedding vectors by averaging chunk embeddings.

How it works
------------
1) Read all chunk embeddings grouped by document internal id.
2) Compute element-wise mean for each document's list of chunk vectors.
3) Persist the mean vector on the corresponding :Document node as `doc_embedding`.

Environment variables
---------------------
- NEO4J_URI, NEO4J_USER/NEO4J_USERNAME, NEO4J_PASSWORD: Neo4j connection

Usage
-----
  uv run python -m processing.compute_doc_embeddings

Prerequisites
-------------
- Initial graph build must have created Chunk nodes with `embedding` vectors.

Verification queries (optional)
-------------------------------
- Count docs with embeddings:
  MATCH (d:Document) WHERE d.doc_embedding IS NOT NULL RETURN count(d)
- Inspect example dimensions:
  MATCH (d:Document) WHERE d.doc_embedding IS NOT NULL RETURN size(d.doc_embedding) AS dim LIMIT 5

Performance notes
-----------------
- Streams rows and aggregates per document; minimal memory overhead.
- Handles dimension mismatches by aligning to the shortest length defensively.
"""

from __future__ import annotations

# =========================================================================
# Imports
# =========================================================================
# Standard library imports
import os
from typing import DefaultDict, Dict, Iterable, List, Tuple
from collections import defaultdict

# Third-party library imports
from neo4j import GraphDatabase, basic_auth


# =========================================================================
# Global Constants / Variables
# =========================================================================
# (No module-level constants; configuration is derived from environment)


# ------------------------------------------------------------------------- _mean_vector()
def _mean_vector(vectors: Iterable[List[float]]) -> List[float]:
    """Compute the element-wise mean for a collection of same-dimension vectors.

    Args:
        vectors: Iterable of numeric vectors (list of floats) with equal length
                 where possible. Mixed lengths are handled by aligning to the
                 shortest dimension.

    Returns:
        List[float]: Mean vector. Returns an empty list if no valid vectors.
    """
    sums: List[float] = []
    count: int = 0
    for vec in vectors:
        if not vec:
            continue
        if not sums:
            sums = [0.0] * len(vec)
        # Align length if inconsistent (defensive handling of mixed dimensions)
        if len(vec) != len(sums):
            m = min(len(vec), len(sums))
            for i in range(m):
                sums[i] += float(vec[i])
        else:
            for i, val in enumerate(vec):
                sums[i] += float(val)
        count += 1
    if count == 0:
        return []
    return [s / count for s in sums]
# ------------------------------------------------------------------------- end _mean_vector()


# ------------------------------------------------------------------------- main()
def main() -> int:
    """Aggregate chunk embeddings into `:Document.doc_embedding` vectors.

    Returns:
        int: 0 on success; non-zero on failure conditions.
    """
    uri = os.getenv("NEO4J_URI", "bolt://aph_if_neo4j:7687")
    user = os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
    password = os.getenv("NEO4J_PASSWORD", "password")

    driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
    try:
        with driver.session() as session:
            # Pull all chunk embeddings grouped by document internal id
            rows = session.run(
                """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                WHERE c.embedding IS NOT NULL
                RETURN id(d) AS d_id, c.embedding AS emb
                """
            )
            doc_to_vecs: DefaultDict[int, List[List[float]]] = defaultdict(list)
            for r in rows:
                d_id = int(r["d_id"])  # type: ignore[index]
                emb = list(r["emb"])  # type: ignore[index]
                doc_to_vecs[d_id].append(emb)

            updates: List[Tuple[int, List[float]]] = []
            for d_id, vecs in doc_to_vecs.items():
                mean_vec = _mean_vector(vecs)
                if mean_vec:
                    updates.append((d_id, mean_vec))

            for d_id, mean_vec in updates:
                session.run(
                    "MATCH (d:Document) WHERE id(d)=$d_id SET d.doc_embedding=$emb",
                    {"d_id": d_id, "emb": mean_vec},
                )
            print(f"Computed doc embeddings for {len(updates)} documents")
        return 0
    finally:
        driver.close()
# ------------------------------------------------------------------------- end main()  


# =========================================================================
# Module Initialization / Main Execution Guard
# =========================================================================
# This block runs only when the file is executed directly, not when imported.
# It computes document embeddings using the current environment configuration.

# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    raise SystemExit(main())

# =========================================================================
# End of File
# =========================================================================


