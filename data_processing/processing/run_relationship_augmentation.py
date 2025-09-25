# -------------------------------------------------------------------------
# File: run_relationship_augmentation.py
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 2025-08-12
# File Path: data_processing/processing/run_relationship_augmentation.py
# ------------------------------------------------------------------------

# --- Module Objective ---
#   Orchestrate relationship discovery and augmentation using the utilities
#   in `add_relationships_to_graph.py`, with environment-aware configuration
#   and safety.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Function: env_bool(name, default) -> bool
# - Function: main() -> int
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: os
# - Local Project Modules: processing.add_relationships_to_graph (utilities)
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - APH-IF  
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG – Intelligent Fusion (APH-IF)
# -------------------------------------------------------------------------

from __future__ import annotations

# =========================================================================
# Imports
# =========================================================================
# Standard library imports
import os
from typing import List

# Local application/library specific imports
from processing.add_relationships_to_graph import (
    Neo4jConfig,
    GraphRelationshipAdder,
    stream_entity_similarity,
    stream_doc_similarity,
    fetch_entity_evidence,
    fetch_document_evidence,
    RelationInput,
    label_with_llm,
    merge_entity_relations,
    merge_document_relations,
)


# =========================================================================
# Global Constants / Variables
# =========================================================================
# (No module-level constants required for this orchestrator module)


# =========================================================================
# Standalone Function Definitions
# =========================================================================
# --------------------------------------------------------------------------------- env_bool()
def env_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with common truthy values.

    Truthy values: {"1", "true", "yes", "on"} (case-insensitive)

    Args:
        name: Environment variable name to read.
        default: Fallback value if the variable is not set.

    Returns:
        bool: Parsed boolean value.
    """
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}
# --------------------------------------------------------------------------------- end env_bool()


# --------------------------------------------------------------------------------- main()
def main() -> int:
    """Run relationship augmentation based on current environment settings.

    Flow:
        - Initialize `GraphRelationshipAdder` and run preconditions.
        - If `AUGMENT_ENTITY=true`, generate entity candidates via Jaccard over
          shared chunk references, collect evidence, label via LLM, and MERGE
          unless `DRY_RUN=true`.
        - If `AUGMENT_DOCUMENT=true`, generate document candidates via cosine
          similarity over document embeddings, collect evidence, label via LLM,
          and MERGE unless `DRY_RUN=true`.

    Environment variables:
        - AUGMENT_ENTITY, AUGMENT_DOCUMENT: feature toggles
        - ENTITY_SIM_CUTOFF, ENTITY_LIMIT: control entity candidate volume
        - DOC_SIM_CUTOFF: controls document candidate generation
        - DRY_RUN: when true, performs discovery + labeling without writes

    Returns:
        int: 0 on success.
    """
    cfg = Neo4jConfig.from_env()
    adder = GraphRelationshipAdder(cfg)
    try:
        adder.run_preconditions()

        augment_entity = env_bool("AUGMENT_ENTITY", True)
        augment_document = env_bool("AUGMENT_DOCUMENT", False)
        dry_run = env_bool("DRY_RUN", False)

        if augment_entity:
            cutoff = float(os.getenv("ENTITY_SIM_CUTOFF", "0.25"))
            limit = int(os.getenv("ENTITY_LIMIT", "1000"))
            # Calculate entity similarity using Jaccard over shared chunks
            pairs = stream_entity_similarity(adder.run_cypher, similarity_cutoff=cutoff, limit=limit)

            batch: List[RelationInput] = []
            for row in pairs:
                e1, e2, sim = int(row["e1_id"]), int(row["e2_id"]), float(row["similarity"])
                ev = fetch_entity_evidence(adder.run_cypher, e1, e2, top_n=6)
                batch.append(RelationInput(domain="ENTITY", subject_id=e1, object_id=e2, evidence_chunks=ev, seed_score=sim))

            outputs = label_with_llm(batch)  # LLM relationship labeling (gpt-5-nano default)
            if not dry_run:
                merged = merge_entity_relations(adder.run_cypher, outputs)
                print(f"Merged {merged} ENTITY relations")
            else:
                print(f"Dry-run: prepared {len(outputs)} ENTITY relations (not merged)")

        if augment_document:
            cutoff = float(os.getenv("DOC_SIM_CUTOFF", "0.80"))
            # Calculate document similarity using cosine similarity over embeddings
            pairs = stream_doc_similarity(adder.run_cypher, similarity_cutoff=cutoff)

            batch = []
            for row in pairs:
                d1, d2, sim = int(row["d1_id"]), int(row["d2_id"]), float(row["similarity"])
                ev = fetch_document_evidence(adder.run_cypher, d1, d2, per_doc=3)
                batch.append(RelationInput(domain="DOCUMENT", subject_id=d1, object_id=d2, evidence_chunks=ev, seed_score=sim))

            outputs = label_with_llm(batch)  # LLM relationship labeling (gpt-5-nano default)
            if not dry_run:
                merged = merge_document_relations(adder.run_cypher, outputs)
                print(f"Merged {merged} DOCUMENT relations")
            else:
                print(f"Dry-run: prepared {len(outputs)} DOCUMENT relations (not merged)")

        return 0
    finally:
        adder.close()
# --------------------------------------------------------------------------------- end main()
    
# =========================================================================
# Module Initialization / Main Execution Guard
# =========================================================================
# This block runs only when the file is executed directly, not when imported.
# It invokes the orchestrator using the current environment configuration.

# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    raise SystemExit(main())
# ---------------------------------------------------------------------------------

# =========================================================================
# End of File
# =========================================================================