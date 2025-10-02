# -------------------------------------------------------------------------
# File: data_processing/config.py
# Author: Alexander Ricciardi
# Date: 2025-10-02
# [File Path] data_processing/config.py
# ------------------------------------------------------------------------
# Project: APH-IF — Advanced Parallel HybridRAG – Intelligent Fusion
#
# Module Functionality:
#   Centralized configuration loader for the data_processing service. Reads
#   environment variables from data_processing/.env and exposes a typed
#   configuration object consumed by the ingestion CLI and entity pipeline.
#
# Module Contents Overview:
# - Class (dataclass): ProcessingConfig
# - Function: load_config
#
# Dependencies / Imports:
# - Standard Library: os, dataclasses, pathlib, typing
# - Third-Party: python-dotenv
#
# Usage / Integration:
#   Used by data_processing/run_ingest.py and downstream modules to configure
#   Neo4j, OpenAI, batching, and feature toggles. Not an execution entry point.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------
"""Configuration management for data processing module.

Loads settings from data_processing/.env and provides typed access to all configuration values.
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


# ____________________________________________________________________________
# Class Definitions
#
# TODO: Consider enabling @dataclass(slots=True, kw_only=True) in a dedicated refactor PR
#       after validating compatibility with serialization and dynamic attributes.
# ------------------------------------------------------------------------- class ProcessingConfig
@dataclass
class ProcessingConfig:
    """Configuration for data processing operations.

    Responsibilities:
        - Provide a single, typed container for all module settings.
        - Encapsulate defaults and computed properties used by Neo4j/backend.

    Attributes:
        data_pdf_dir: Directory containing input PDF files.
        monitoring_dir: Directory for processing and audit logs.
        chunk_size_chars: Target chunk size in characters for Phase 1.
        max_docs: Maximum number of PDFs to process (0 = no limit).
        max_pages: Maximum pages per document to read (0 = no limit).
        neo4j_uri: Neo4j connection URI selected per environment.
        neo4j_username: Neo4j username.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.
        openai_api_key: API key used for embeddings/LLM calls.
        openai_model: Default LLM model identifier.
        openai_model_mini: Lightweight LLM model identifier.
        openai_requests_per_minute: Client-side request rate cap.
        openai_tokens_per_minute: Client-side token rate cap.
        openai_max_tokens: Max completion tokens for augmentation.
        embedding_model: Embedding model identifier.
        embedding_dimensions: Embedding vector dimensionality.
        embedding_batch_size: Batch size for embedding requests.
        neo4j_batch_size: Batch size for Neo4j upserts.
        extract_entities: Feature toggle for Phase 2.
        augment_relationships: Feature toggle for Phase 3.
        verbose: Enable verbose logging to console.
        monitoring_enabled: Enable file logging under monitoring_dir.
        monitoring_log_level: Log granularity for monitoring files.
        llm_timeout_seconds: Timeout for LLM requests in seconds.
        spacy_model: spaCy model to load for entity extraction.
        entity_patterns_dir: Directory for EntityRuler patterns.
        entity_lexicons_dir: Directory for PhraseMatcher lexicons.
        entity_types: Allowed entity labels to persist.
        run_entity_evaluation: Toggle for pattern evaluation harness.
        max_augmentation_chunks: Cap for augmentation cost control.
    """

    # ______________________
    #  Instance Fields
    #
    # Paths
    data_pdf_dir: Path
    monitoring_dir: Path

    # Chunking
    chunk_size_chars: int
    max_docs: int
    max_pages: int

    # Neo4j connection
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str

    # OpenAI
    openai_api_key: str
    openai_model: str
    openai_model_mini: str

    # Rate limiting
    openai_requests_per_minute: int
    openai_tokens_per_minute: int
    openai_max_tokens: int

    # Fields with defaults (must come after non-default fields)
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    embedding_batch_size: int = 64
    neo4j_batch_size: int = 500
    extract_entities: bool = False
    augment_relationships: bool = False
    verbose: bool = True
    monitoring_enabled: bool = True
    monitoring_log_level: str = "standard"
    llm_timeout_seconds: int = 240

    # Entity extraction (Phase 2)
    spacy_model: str = "en_core_web_sm"
    entity_patterns_dir: Optional[Path] = None
    entity_lexicons_dir: Optional[Path] = None
    entity_types: list[str] = None
    run_entity_evaluation: bool = False

    # Relationship augmentation (Phase 3)
    max_augmentation_chunks: int = 100

    # ______________________
    #  Properties
    #
    # --------------------------------------------------------------------------------- vector_index_name()
    @property
    def vector_index_name(self) -> str:
        """Backend-compatible vector index name.

        Returns:
            The canonical vector index name used by the backend: "chunk_embedding_index".
        """
        return "chunk_embedding_index"
    # --------------------------------------------------------------------------------- end vector_index_name()

    # --------------------------------------------------------------------------------- chunk_label()
    @property
    def chunk_label(self) -> str:
        """Neo4j node label for chunks.

        Returns:
            The label applied to chunk nodes in Neo4j ("Chunk").
        """
        return "Chunk"
    # --------------------------------------------------------------------------------- end chunk_label()

    # --------------------------------------------------------------------------------- document_label()
    @property
    def document_label(self) -> str:
        """Neo4j node label for documents.

        Returns:
            The label applied to document nodes in Neo4j ("Document").
        """
        return "Document"
    # --------------------------------------------------------------------------------- end document_label()

    # --------------------------------------------------------------------------------- entity_label()
    @property
    def entity_label(self) -> str:
        """Neo4j node label for entities.

        Returns:
            The label applied to entity nodes in Neo4j ("Entity").
        """
        return "Entity"
    # --------------------------------------------------------------------------------- end entity_label()

    # ______________________
    # Post-Initialization (validation + derived fields)
    #
    # --------------------------------------------------------------------------------- __post_init__()
    def __post_init__(self) -> None:
        """Initialize optional fields with default values.

        Ensures directories for patterns/lexicons are set and that allowed entity
        types default to a stable list if not provided.
        """
        if self.entity_patterns_dir is None:
            self.entity_patterns_dir = Path(__file__).parent / "entities" / "patterns"
        if self.entity_lexicons_dir is None:
            self.entity_lexicons_dir = Path(__file__).parent / "entities" / "lexicons"
        if self.entity_types is None:
            self.entity_types = [
                "LEGAL_SECTION",
                "CFR_PART",
                "SUBPART",
                "APPENDIX",
                "CFR_TITLE",
                "STANDARD",
            ]
    # --------------------------------------------------------------------------------- end __post_init__()
# ------------------------------------------------------------------------- end class ProcessingConfig

# __________________________________________________________________________
# Standalone Function Definitions
#
# --------------------------------------------------------------------------------- load_config()
def load_config(env_path: Optional[Path] = None) -> ProcessingConfig:
    """Load configuration from .env file.

    Reads environment variables from ``data_processing/.env`` (or the provided path),
    selects Neo4j credentials based on ``APP_ENV`` and ``FORCE_TEST_DB``, and returns a
    fully-populated ``ProcessingConfig`` instance used by the ingestion pipeline.

    Args:
        env_path: Optional explicit path to a ``.env`` file. If omitted, defaults to
            ``data_processing/.env`` next to this module.

    Returns:
        A populated ``ProcessingConfig`` with paths, database, OpenAI, and batching
        parameters ready for use by callers.

    Raises:
        FileNotFoundError: If the ``.env`` file cannot be found.
        ValueError: If required environment variables are missing.

    Example:
        >>> from pathlib import Path
        >>> cfg = load_config()  # doctest: +SKIP
        >>> isinstance(cfg.data_pdf_dir, Path)
        True
    """
    if env_path is None:
        # Assume we're in data_processing/ directory or running from project root
        current_dir = Path(__file__).parent
        env_path = current_dir / ".env"

    if not env_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {env_path}. "
            "Ensure data_processing/.env exists with required settings."
        )

    # Load environment variables
    load_dotenv(env_path)

    # ______________________
    # Embedded Function
    #
    # --------------------------------------------------------------------------------- get_env()
    def get_env(key: str, default: Optional[str] = None) -> str:
        """Read a required environment variable with an optional default.

        Args:
            key: Environment variable name to read.
            default: Fallback value when the variable is unset.

        Returns:
            The resolved environment variable value as a string.

        Raises:
            ValueError: If the variable is missing and no default is provided.
        """
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Required environment variable {key} not set in {env_path}")
        return value
    # --------------------------------------------------------------------------------- end get_env()

    # ______________________
    # Embedded Function
    #
    # --------------------------------------------------------------------------------- get_int()
    def get_int(key: str, default: int) -> int:
        """Read an integer environment variable with a default fallback.

        Args:
            key: Environment variable name to read.
            default: Default integer value if unset or empty.

        Returns:
            Parsed integer value, or the provided default when unset.
        """
        value = os.getenv(key)
        return int(value) if value else default
    # --------------------------------------------------------------------------------- end get_int()

    # ______________________
    # Embedded Function
    #
    # --------------------------------------------------------------------------------- get_bool()
    def get_bool(key: str, default: bool = False) -> bool:
        """Read a boolean environment variable with common truthy/falsey parsing.

        Args:
            key: Environment variable name to read.
            default: Default boolean when unset or unrecognized.

        Returns:
            True when the value is one of {"true", "1", "yes"}; False for
            {"false", "0", "no", ""}; otherwise the provided default.
        """
        value = os.getenv(key, "").lower()
        if value in ("true", "1", "yes"):
            return True
        if value in ("false", "0", "no", ""):
            return default
        return default
    # --------------------------------------------------------------------------------- end get_bool()

    # Determine Neo4j credentials based on APP_ENV
    app_env = os.getenv("APP_ENV", "development").lower()
    force_test_db = get_bool("FORCE_TEST_DB", False)

    if force_test_db or app_env == "test":
        neo4j_uri = get_env("NEO4J_URI_TEST")
        neo4j_username = get_env("NEO4J_USERNAME_TEST")
        neo4j_password = get_env("NEO4J_PASSWORD_TEST")
        neo4j_database = get_env("NEO4J_DATABASE_TEST")
    else:  # development
        neo4j_uri = get_env("NEO4J_URI_DEV")
        neo4j_username = get_env("NEO4J_USERNAME_DEV")
        neo4j_password = get_env("NEO4J_PASSWORD_DEV")
        neo4j_database = get_env("NEO4J_DATABASE_DEV")

    # =========================================================================
    # Title Functionality  (Configuration Loading Utilities)
    # =========================================================================

    # Build configuration
    config = ProcessingConfig(
        # Paths
        data_pdf_dir=Path(__file__).parent / get_env("PDF_DIR", "data_pdf"),
        monitoring_dir=Path(__file__).parent / get_env("MONITORING_DIR", "monitoring_logs"),
        # Chunking
        chunk_size_chars=get_int("CHUNK_SIZE_CHARS", 3000),
        max_docs=get_int("MAX_DOCS", 0),
        max_pages=get_int("MAX_PAGES", 0),
        # Neo4j
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password,
        neo4j_database=neo4j_database,
        # OpenAI
        openai_api_key=get_env("OPENAI_API_KEY"),
        openai_model=get_env("OPENAI_MODEL", "gpt-5"),
        openai_model_mini=get_env("OPENAI_MODEL_MINI", "gpt-5-mini"),
        # Rate limiting
        openai_requests_per_minute=get_int("OPENAI_REQUESTS_PER_MINUTE", 4500),
        openai_tokens_per_minute=get_int("OPENAI_TOKENS_PER_MINUTE", 3600000),
        openai_max_tokens=get_int("OPENAI_MAX_TOKENS", 6000),
        # Batch sizes
        embedding_batch_size=get_int("EMBEDDING_BATCH_SIZE", 64),
        neo4j_batch_size=get_int("NEO4J_BATCH_SIZE", 500),
        # Feature toggles
        extract_entities=get_bool("EXTRACT_ENTITIES", False),
        augment_relationships=get_bool("AUGMENT_RELATIONSHIPS", False),
        # Logging
        verbose=get_bool("VERBOSE", True),
        monitoring_enabled=get_bool("MONITORING_ENABLED", True),
        monitoring_log_level=get_env("MONITORING_LOG_LEVEL", "standard"),
        # Timeouts
        llm_timeout_seconds=get_int("LLM_TIMEOUT_SECONDS", 240),
        # Entity extraction (Phase 2)
        spacy_model=get_env("SPACY_MODEL", "en_core_web_sm"),
        run_entity_evaluation=get_bool("RUN_ENTITY_EVALUATION", False),
        # Relationship augmentation (Phase 3)
        max_augmentation_chunks=get_int("MAX_AUGMENTATION_CHUNKS", 0),
    )

    return config

# --------------------------------------------------------------------------------- end load_config()

# __________________________________________________________________________
# End of File
