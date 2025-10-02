# -------------------------------------------------------------------------
# File: data_processing/entities/evaluator.py
# Author: Alexander Ricciardi
# Date: 2025-10-02
# [File Path] data_processing/entities/evaluator.py
# ------------------------------------------------------------------------
# Project: APH-IF — Advanced Parallel HybridRAG – Intelligent Fusion
#
# Module Functionality:
#   Precision/recall evaluation harness for the entity extraction pipeline.
#   Loads labeled samples and computes precision, recall, and F1 for an
#   EntityExtractor instance.
#
# Module Contents Overview:
# - Class: EntityEvaluator
#
# Dependencies / Imports:
# - Standard Library: json, logging, pathlib, typing
# - Local Modules: entities.extract.EntityExtractor
#
# Usage / Integration:
#   Provides a simple evaluation utility for Phase 2 entity extraction in
#   data_processing.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""Precision/recall evaluation harness."""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import json
import logging
from pathlib import Path
from typing import Optional

from .extract import EntityExtractor

logger = logging.getLogger(__name__)

# ____________________________________________________________________________
# Class Definitions
# ------------------------------------------------------------------------- class EntityEvaluator
class EntityEvaluator:
    """Evaluates entity extraction against labeled data.

    Responsibilities:
        - Load labeled samples from a JSONL file.
        - Run an ``EntityExtractor`` over samples and compute metrics.
        - Aggregate precision, recall, and F1 across all samples.

    Attributes:
        labeled_samples_path: Path to the JSONL file with labeled examples.
    """

    # -------------------------------------------------------------- __init__()
    def __init__(self, labeled_samples_path: Optional[Path] = None):
        self.labeled_samples_path = (
            labeled_samples_path or
            Path(__file__).parent / "evaluation" / "labeled_samples.jsonl"
        )
    # -------------------------------------------------------------- end __init__()

    # -------------------------------------------------------------- load_labeled_samples()
    def load_labeled_samples(self) -> list[dict]:
        """Load labeled samples.

        Format per line:
        {
            "text": "...",
            "entities": [{"text": "§75.1714", "type": "LEGAL_SECTION"}, ...]
        }
        """
        if not self.labeled_samples_path.exists():
            logger.warning(f"Labeled samples not found: {self.labeled_samples_path}")
            return []

        samples = []
        with open(self.labeled_samples_path, "r") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))

        return samples
    # -------------------------------------------------------------- end load_labeled_samples()

    # -------------------------------------------------------------- evaluate()
    def evaluate(self, extractor: EntityExtractor) -> dict:
        """Evaluate extractor on labeled samples.

        Args:
            extractor: EntityExtractor instance

        Returns:
            Dict with precision, recall, f1 scores
        """
        samples = self.load_labeled_samples()

        if not samples:
            return {"error": "No labeled samples available"}

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for sample in samples:
            text = sample["text"]
            true_entities = {
                (ent["text"].lower(), ent["type"])
                for ent in sample["entities"]
            }

            # Extract
            predicted = extractor.extract_from_text(text)
            predicted_entities = {
                (ent.name.lower(), ent.type)
                for ent in predicted
            }

            # Calculate metrics
            tp = len(true_entities & predicted_entities)
            fp = len(predicted_entities - true_entities)
            fn = len(true_entities - predicted_entities)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Compute scores
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        results = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "samples_evaluated": len(samples),
        }

        logger.info("Evaluation results:")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1: {f1:.3f}")

        return results
    # -------------------------------------------------------------- end evaluate()

# ------------------------------------------------------------------------- end class EntityEvaluator

# __________________________________________________________________________
# End of File
#
