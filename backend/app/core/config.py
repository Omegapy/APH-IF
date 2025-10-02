# -------------------------------------------------------------------------
# File: core/config.py
# Author: Alexander Ricciardi
# Date: 2025-09-24
# [File Path] backend/app/core/config.py
# ------------------------------------------------------------------------
# Project: APH-IF
#
# Project description:
# Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF)
# is a novel Retrieval Augmented Generation (RAG) system that differs from
# traditional RAG approaches by performing semantic and traversal searches
# concurrently, rather than sequentially, and fusing the results using an LLM
# or an LRM to generate the final response.
# -------------------------------------------------------------------------

# --- Module Functionality ---
#   Centralizes backend configuration loading, environment detection, and
#   credential validation for Neo4j, OpenAI, and LLM integration helpers used
#   across the APH-IF backend service.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Enum: EnvironmentMode
# - Class: Settings
# - Constants: APP_ENV, DEBUG, VERBOSE, NEO4J_URI, NEO4J_USERNAME,
#              NEO4J_PASSWORD, NEO4J_DATABASE, OPENAI_API_KEY,
#              OPENAI_MODEL_MINI, OPENAI_REQUESTS_PER_MINUTE,
#              OPENAI_TOKENS_PER_MINUTE, LLM_MODEL_CONFIG,
#              EMBEDDING_CONFIG, EFFECTIVE_RPM, EFFECTIVE_TPM,
#              IS_GPT5_MINI, SUPPORTS_GPT5_ADVANCED
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: logging, os, pathlib.Path, enum.Enum
# - Third-Party: dotenv.load_dotenv, pydantic.BaseSettings, pydantic.Field,
#                pydantic.model_validator
# - Local Project Modules: None (module is self-contained)
# --- Requirements ---
# - Python 3.12+
# -------------------------------------------------------------------------

# --- Usage / Integration ---
#   Imported by backend services to access strongly validated settings and
#   shared configuration constants for external clients and feature flags.
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Backend configuration utilities.

Provides environment-aware settings management for the APH-IF backend service,
including LLM client options and Neo4j connection parameters consumed by other
core modules.
"""

# __________________________________________________________________________
# Imports
from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

# Load environment variables from backend/.env
# __________________________________________________________________________
# Global Constants / Variables
BACKEND_ROOT = Path(__file__).parent.parent.parent
DOTENV_PATH = BACKEND_ROOT / ".env"
load_dotenv(DOTENV_PATH, override=True)

logging.basicConfig(
    level=logging.DEBUG if os.getenv("VERBOSE", "true").lower() == "true" else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# __________________________________________________________________________
# Class Definitions
# ------------------------------------------------------------------------- class EnvironmentMode
class EnvironmentMode(str, Enum):
    """Supported environment modes for the backend service."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
# ------------------------------------------------------------------------- end class EnvironmentMode


# ------------------------------------------------------------------------- class Settings
class Settings(BaseSettings):
    """Application settings container for the APH-IF backend service.

    The settings model normalizes environment configuration for Neo4j,
    OpenAI-based language models, and feature flags that control hybrid
    retrieval behavior. Instances source their values from environment
    variables with development-friendly defaults.

    Attributes:
        app_env: Requested application environment label.
        force_test_db: Flag enabling the safe Neo4j testing instance.
        verbose: Toggles verbose logging output on startup.
        debug: Enables FastAPI debug instrumentation.
        host: Bind host for the FastAPI server.
        port: Bind port for the FastAPI server.
        backend_url: URL used for backend callbacks and health checks.
        data_processing_url: URL for the data processing service.
        frontend_url: URL for the Streamlit frontend.
        neo4j_uri: AuraDB connection URI.
        neo4j_username: AuraDB username credential.
        neo4j_password: AuraDB password credential.
        neo4j_database: Target AuraDB database name.
        openai_api_key: API key for OpenAI-compatible providers.
        openai_model_mini: Default LLM model identifier.
        openai_max_tokens: Maximum completion tokens for generation.
        openai_temperature: Temperature value for LLM sampling.
        openai_requests_per_minute: Rate limit requests per minute.
        openai_tokens_per_minute: Rate limit tokens per minute.
        openai_embedding_model: Embedding model identifier.
        openai_embedding_dimensions: Dimensionality of embedding vectors.
        llm_timeout_seconds: Timeout for LLM API requests in seconds.
        llm_max_retries: Maximum retry attempts for LLM calls.
        llm_rate_limit_buffer_percent: Safety buffer used for rate limiting.
        fusion_parallel_preprocessing: Enables parallel preprocessing steps.
        traversal_confidence_cap: Upper bound confidence for traversal results.
        semantic_confidence_cap: Upper bound confidence for semantic results.
        fusion_confidence_cap: Upper bound confidence for fused results.
        semantic_use_structural_schema: Enables schema-aware semantic prompts.
        semantic_schema_token_budget: Token budget for schema context.
        semantic_schema_max_items: Maximum schema snippets injected.
        semantic_citation_validation: Enables citation validation for semantic results.
        semantic_domain_enforcement: Enables domain marker injection.
        semantic_domain_markers_per_ref: Maximum markers per reference.
        schema_export_enabled: Enables schema export on startup.
        schema_export_filename: Filename used for schema export artifacts.
        use_llm_structural_cypher: Enables LLM-to-Cypher generation.
        llm_cypher_max_hops: Maximum relationship hops allowed in generated Cypher.
        llm_cypher_force_limit: Default LIMIT appended to generated Cypher queries.
        llm_cypher_allow_call: Allows CALL statements in generated Cypher.
        llm_cypher_examples_enabled: Includes few-shot examples in prompts.
        llm_cypher_prompt_token_budget: Token budget for Cypher prompts.
        llm_structural_narrative_enabled: Enables narrative synthesis from KG data.
        llm_structural_narrative_token_budget: Token budget for narrative prompts.
        llm_structural_source_max: Maximum number of KG source rows used.
        llm_structural_preserve_numbers: Enforces manual citation numbering.
        llm_structural_citation_validation: Validates citations in narratives.
        llm_structural_relevance_threshold: Threshold for narrative relevance checks.
        llm_domain_citation_enabled: Enables domain-aware citation formatting.
    """

    # ______________________
    #  Instance Fields
    #
    
    # Environment Configuration
    app_env: str = Field(default="development", validation_alias="APP_ENV")
    force_test_db: bool = Field(default=False, validation_alias="FORCE_TEST_DB")
    verbose: bool = Field(default=True, validation_alias="VERBOSE")
    debug: bool = Field(default=True, validation_alias="DEBUG")
    
    # Service Configuration
    host: str = Field(default="127.0.0.1", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    backend_url: str = Field(default="http://localhost:8000", validation_alias="BACKEND_URL")
    data_processing_url: str = Field(default="http://localhost:8010", validation_alias="DATA_PROCESSING_URL")
    frontend_url: str = Field(default="http://localhost:8501", validation_alias="FRONTEND_URL")
    
    # Neo4j Database Configuration
    neo4j_uri: str = Field(..., validation_alias="NEO4J_URI")
    neo4j_username: str = Field(..., validation_alias="NEO4J_USERNAME")
    neo4j_password: str = Field(..., validation_alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", validation_alias="NEO4J_DATABASE")
    
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., validation_alias="OPENAI_API_KEY")
    openai_model_mini: str = Field(default="gpt-5-mini", validation_alias="OPENAI_MODEL_MINI")
    openai_max_tokens: int = Field(default=6000, validation_alias="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=1.0, validation_alias="OPENAI_TEMPERATURE")
    
    # OpenAI Rate Limiting (Tier 3)
    openai_requests_per_minute: int = Field(default=4500, validation_alias="OPENAI_REQUESTS_PER_MINUTE")
    openai_tokens_per_minute: int = Field(default=3600000, validation_alias="OPENAI_TOKENS_PER_MINUTE")
    
    # OpenAI Embedding Configuration
    openai_embedding_model: str = Field(default="text-embedding-3-large", validation_alias="OPENAI_EMBEDDING_MODEL")
    openai_embedding_dimensions: int = Field(default=3072, validation_alias="OPENAI_EMBEDDING_DIMENSIONS")
    
    # LLM Client Configuration
    llm_timeout_seconds: int = Field(default=240, validation_alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=3, validation_alias="LLM_MAX_RETRIES")
    llm_rate_limit_buffer_percent: float = Field(default=0.9, validation_alias="LLM_RATE_LIMIT_BUFFER_PERCENT")
    fusion_parallel_preprocessing: bool = Field(default=True, validation_alias="FUSION_PARALLEL_PREPROCESSING")
    
    
    # Confidence Scoring Configuration
    traversal_confidence_cap: float = Field(
        default=1.0, validation_alias="TRAVERSAL_CONFIDENCE_CAP",
        description="Maximum confidence score for graph traversal searches (0.0-1.0)"
    )
    semantic_confidence_cap: float = Field(
        default=1.0, validation_alias="SEMANTIC_CONFIDENCE_CAP",
        description="Maximum confidence score for semantic/vector searches (0.0-1.0)"
    )
    fusion_confidence_cap: float = Field(
        default=1.0, validation_alias="FUSION_CONFIDENCE_CAP",
        description="Maximum confidence score for fused results (0.0-1.0)"
    )
    
    
    # Schema-aware Semantic Search Configuration
    semantic_use_structural_schema: bool = Field(
        default=True, validation_alias="SEMANTIC_USE_STRUCTURAL_SCHEMA",
        description="Enable schema-aware mode that augments semantic search prompts with structural schema context"
    )
    semantic_schema_token_budget: int = Field(
        default=2500, validation_alias="SEMANTIC_SCHEMA_TOKEN_BUDGET",
        description="Token budget for schema context segment in semantic search prompts"
    )
    semantic_schema_max_items: int = Field(
        default=15, validation_alias="SEMANTIC_SCHEMA_MAX_ITEMS",
        description="Maximum number of node labels/relationships to include in schema context"
    )
    
    # Citation Validation and Domain Enforcement Configuration
    semantic_citation_validation: bool = Field(
        default=True, validation_alias="SEMANTIC_CITATION_VALIDATION",
        description="Enable validation and cleanup of unmatched inline citations in semantic search"
    )
    semantic_domain_enforcement: bool = Field(
        default=True, validation_alias="SEMANTIC_DOMAIN_ENFORCEMENT",
        description="Enable automatic insertion of domain-specific markers (legal, academic, etc.) from source content"
    )
    semantic_domain_markers_per_ref: int = Field(
        default=2, validation_alias="SEMANTIC_DOMAIN_MARKERS_PER_REF",
        description="Maximum number of domain markers to extract and use per reference"
    )
    
    # Schema Export Configuration
    schema_export_enabled: bool = Field(
        default=True, validation_alias="SCHEMA_EXPORT_ENABLED",
        description="Enable automatic structural schema export to JSON"
    )
    schema_export_filename: str = Field(
        default="kg_schema_structural_summary.json", validation_alias="SCHEMA_EXPORT_FILENAME",
        description="Filename for structural schema export"
    )
    
    # LLM Structural Cypher Configuration
    use_llm_structural_cypher: bool = Field(
        default=True, validation_alias="USE_LLM_STRUCTURAL_CYPHER",
        description="Enable LLM-powered natural language to Cypher generation using structural schema"
    )
    llm_cypher_max_hops: int = Field(
        default=3, validation_alias="LLM_CYPHER_MAX_HOPS",
        description="Maximum allowed relationship hops in generated Cypher queries"
    )
    llm_cypher_force_limit: int = Field(
        default=50, validation_alias="LLM_CYPHER_FORCE_LIMIT",
        description="Default LIMIT value to inject if missing in generated queries"
    )
    llm_cypher_allow_call: bool = Field(
        default=False, validation_alias="LLM_CYPHER_ALLOW_CALL",
        description="Whether to allow CALL procedures in generated queries (dangerous)"
    )
    llm_cypher_examples_enabled: bool = Field(
        default=True, validation_alias="LLM_CYPHER_EXAMPLES_ENABLED",
        description="Include few-shot examples in Cypher generation prompts"
    )
    llm_cypher_prompt_token_budget: int = Field(
        default=4500, validation_alias="LLM_CYPHER_PROMPT_TOKEN_BUDGET",
        description="Maximum tokens allocated for Cypher generation prompts"
    )
    
    # LLM Structural Narrative Configuration
    llm_structural_narrative_enabled: bool = Field(
        default=True, validation_alias="LLM_STRUCTURAL_NARRATIVE_ENABLED",
        description="Enable LLM-powered narrative summarization from KG results"
    )
    llm_structural_narrative_token_budget: int = Field(
        default=3500, validation_alias="LLM_STRUCTURAL_NARRATIVE_TOKEN_BUDGET",
        description="Token budget for narrative summarization prompts (input only)"
    )
    llm_structural_source_max: int = Field(
        default=15, validation_alias="LLM_STRUCTURAL_SOURCE_MAX",
        description="Maximum result rows for numbered sources (selection, not truncation)"
    )
    llm_structural_preserve_numbers: bool = Field(
        default=True, validation_alias="LLM_STRUCTURAL_PRESERVE_NUMBERS",
        description="Require LLM to preserve provided citation numbers exactly"
    )
    llm_structural_citation_validation: bool = Field(
        default=True, validation_alias="LLM_STRUCTURAL_CITATION_VALIDATION",
        description="Validate and filter invented citations from LLM output"
    )
    llm_structural_relevance_threshold: float = Field(
        default=0.2, validation_alias="LLM_STRUCTURAL_RELEVANCE_THRESHOLD",
        description="Minimum keyword overlap ratio for relevance check (0.0-1.0)"
    )

    # Domain-Aware Citations Configuration
    llm_domain_citation_enabled: bool = Field(
        default=True, validation_alias="LLM_DOMAIN_CITATION_ENABLED",
        description="Enable domain-aware citation extraction and rendering (legal, academic, technical, business, medical)"
    )
    
    
    # ______________________
    # Getters (Property decorators are often preferred for simple getters)
    #

    # -------------------------------------------------------------- environment_mode()
    @property
    def environment_mode(self) -> EnvironmentMode:
        """Get the effective environment mode.

        Returns:
            EnvironmentMode: Active environment derived from flags and APP_ENV.
        """
        if self.force_test_db:
            return EnvironmentMode.TESTING
        elif self.app_env == "production":
            return EnvironmentMode.PRODUCTION
        else:
            return EnvironmentMode.DEVELOPMENT
    # -------------------------------------------------------------- end environment_mode()
    
    # -------------------------------------------------------------- is_development()
    @property
    def is_development(self) -> bool:
        """Check whether the service runs in development mode.

        Returns:
            bool: True when the resolved environment mode is development.
        """
        return self.environment_mode == EnvironmentMode.DEVELOPMENT
    # -------------------------------------------------------------- end is_development()
    
    # -------------------------------------------------------------- is_production()
    @property
    def is_production(self) -> bool:
        """Check whether the service runs in production mode.

        Returns:
            bool: True when the resolved environment mode is production.
        """
        return self.environment_mode == EnvironmentMode.PRODUCTION
    # -------------------------------------------------------------- end is_production()
    
    # -------------------------------------------------------------- is_testing()
    @property
    def is_testing(self) -> bool:
        """Check whether the service runs in testing mode.

        Returns:
            bool: True when the resolved environment mode is testing.
        """
        return self.environment_mode == EnvironmentMode.TESTING
    # -------------------------------------------------------------- end is_testing()
    
    # ______________________
    # Helper Functions
    #

    # -------------------------------------------------------------- get_neo4j_mode_name()
    def get_neo4j_mode_name(self) -> str:
        """Get a human-readable description of the Neo4j deployment target.

        Returns:
            str: Label describing whether the database is development, testing, or production.
        """
        mode = self.environment_mode
        if mode == EnvironmentMode.TESTING:
            return "TEST DATABASE (Safe for experimentation)"
        elif mode == EnvironmentMode.PRODUCTION:
            return "PRODUCTION DATABASE (Live data - use with caution!)"
        else:
            return "DEVELOPMENT DATABASE (Safe for development)"
    # -------------------------------------------------------------- end get_neo4j_mode_name()
    
    # -------------------------------------------------------------- effective_openai_rpm()
    @property
    def effective_openai_rpm(self) -> int:
        """Compute the effective requests-per-minute budget.

        Returns:
            int: Requests-per-minute value after applying the configured buffer.
        """
        return int(self.openai_requests_per_minute * self.llm_rate_limit_buffer_percent)
    # -------------------------------------------------------------- end effective_openai_rpm()
    
    # -------------------------------------------------------------- effective_openai_tpm()
    @property
    def effective_openai_tpm(self) -> int:
        """Compute the effective tokens-per-minute budget.

        Returns:
            int: Tokens-per-minute value after applying the configured buffer.
        """
        return int(self.openai_tokens_per_minute * self.llm_rate_limit_buffer_percent)
    # -------------------------------------------------------------- end effective_openai_tpm()
    
    # -------------------------------------------------------------- supports_gpt5_advanced_features()
    @property
    def supports_gpt5_advanced_features(self) -> bool:
        """Check whether the configured model supports GPT-5 advanced features.

        Returns:
            bool: True when the selected model exposes advanced GPT-5 capabilities.
        """
        return "gpt-5" in self.openai_model_mini and "mini" not in self.openai_model_mini
    # -------------------------------------------------------------- end supports_gpt5_advanced_features()
    
    # -------------------------------------------------------------- is_gpt5_mini()
    @property
    def is_gpt5_mini(self) -> bool:
        """Check whether the configured model is the GPT-5 mini variant.

        Returns:
            bool: True when the configured model string targets GPT-5 mini.
        """
        return "gpt-5-mini" in self.openai_model_mini
    # -------------------------------------------------------------- end is_gpt5_mini()
    
    # -------------------------------------------------------------- validated_traversal_confidence_cap()
    @property
    def validated_traversal_confidence_cap(self) -> float:
        """Get the traversal confidence cap constrained to the 0.0–1.0 range.

        Returns:
            float: Normalized traversal confidence cap between 0.0 and 1.0.
        """
        return max(0.0, min(1.0, self.traversal_confidence_cap))
    # -------------------------------------------------------------- end validated_traversal_confidence_cap()
    
    # -------------------------------------------------------------- validated_semantic_confidence_cap()
    @property
    def validated_semantic_confidence_cap(self) -> float:
        """Get the semantic confidence cap constrained to the 0.0–1.0 range.

        Returns:
            float: Normalized semantic confidence cap between 0.0 and 1.0.
        """
        return max(0.0, min(1.0, self.semantic_confidence_cap))
    # -------------------------------------------------------------- end validated_semantic_confidence_cap()
    
    # -------------------------------------------------------------- validated_fusion_confidence_cap()
    @property
    def validated_fusion_confidence_cap(self) -> float:
        """Get the fusion confidence cap constrained to the 0.0–1.0 range.

        Returns:
            float: Normalized fusion confidence cap between 0.0 and 1.0.
        """
        return max(0.0, min(1.0, self.fusion_confidence_cap))
    # -------------------------------------------------------------- end validated_fusion_confidence_cap()
    
    # -------------------------------------------------------------- get_llm_model_config()
    def get_llm_model_config(self) -> dict[str, Any]:
        """Build the LLM configuration tailored to the active model.

        Returns:
            dict[str, Any]: Mapping of LLM parameters adjusted for rate limiting
            and model-specific feature flags.
        """
        config = {
            "model": self.openai_model_mini,
            "max_tokens": self.openai_max_tokens,
            "temperature": self.openai_temperature,
            "timeout": self.llm_timeout_seconds,
            "max_retries": self.llm_max_retries,
            "rpm": self.effective_openai_rpm,
            "tpm": self.effective_openai_tpm,
            "parallel_preprocessing": self.fusion_parallel_preprocessing,
        }
        
        # Model-specific adjustments
        if self.is_gpt5_mini:
            config.update({
                "supports_temperature_control": False,
                "supports_advanced_features": False,
                "parameter_name_tokens": "max_completion_tokens",
                "temperature": 1.0  # Force default for GPT-5-mini
            })
        else:
            config.update({
                "supports_temperature_control": True,
                "supports_advanced_features": self.supports_gpt5_advanced_features,
                "parameter_name_tokens": "max_completion_tokens" if "gpt-5" in self.openai_model_mini else "max_tokens"
            })
        
        return config
    # -------------------------------------------------------------- end get_llm_model_config()
    
    # -------------------------------------------------------------- get_embedding_config()
    def get_embedding_config(self) -> dict[str, Any]:
        """Build the embedding model configuration.

        Returns:
            dict[str, Any]: Mapping of embedding model parameters including timeout
            and retry policies.
        """
        return {
            "model": self.openai_embedding_model,
            "dimensions": self.openai_embedding_dimensions,
            "timeout": self.llm_timeout_seconds,
            "max_retries": self.llm_max_retries
        }
    # -------------------------------------------------------------- end get_embedding_config()
    
    # -------------------------------------------------------------- log_configuration()
    def log_configuration(self) -> None:
        """Log the current configuration with sensitive values masked.

        Returns:
            None
        """
        logger.info("=" * 70)
        logger.info("APH-IF Backend Configuration")
        logger.info("=" * 70)
        
        # Environment
        logger.info(f"Environment Mode: {self.environment_mode.value}")
        logger.info(f"Neo4j Instance: {self.get_neo4j_mode_name()}")
        logger.info(f"Debug Mode: {self.debug}")
        logger.info(f"Verbose Logging: {self.verbose}")
        
        # Service URLs
        logger.info(f"Backend URL: {self.backend_url}")
        logger.info(f"Data Processing URL: {self.data_processing_url}")
        logger.info(f"Frontend URL: {self.frontend_url}")
        
        
        # OpenAI (masked)
        logger.info(f"OpenAI Model: {self.openai_model_mini}")
        logger.info(f"OpenAI API Key: {self.openai_api_key[:8]}...")
        logger.info(f"OpenAI Rate Limits: {self.openai_requests_per_minute} RPM, {self.openai_tokens_per_minute} TPM")
        
        # LLM Configuration
        llm_config = self.get_llm_model_config()
        logger.info(f"LLM Configuration: {llm_config['model']} (Effective: {llm_config['rpm']} RPM, {llm_config['tpm']} TPM)")
        logger.info(f"LLM Features: Temperature control: {llm_config.get('supports_temperature_control', 'Unknown')}, "
                   f"Advanced features: {llm_config.get('supports_advanced_features', 'Unknown')}")
        
        # Embedding Configuration  
        embedding_config = self.get_embedding_config()
        logger.info(f"Embedding Model: {embedding_config['model']} ({embedding_config['dimensions']} dimensions)")
        
        # Semantic Search Configuration
        logger.info(f"Semantic Schema-Aware: {self.semantic_use_structural_schema}")
        if self.semantic_use_structural_schema:
            logger.info(f"  Schema Token Budget: {self.semantic_schema_token_budget}")
            logger.info(f"  Max Schema Items: {self.semantic_schema_max_items}")
        logger.info(f"Semantic Citation Validation: {self.semantic_citation_validation}")
        logger.info(f"Semantic Domain Enforcement: {self.semantic_domain_enforcement}")
        if self.semantic_domain_enforcement:
            logger.info(f"  Domain Markers Per Ref: {self.semantic_domain_markers_per_ref}")
        
        logger.info("=" * 70)
    # -------------------------------------------------------------- end log_configuration()
    
    # -------------------------------------------------------------- validate_configuration()
    @model_validator(mode='after')
    def validate_configuration(self) -> Settings:
        """Validate the configuration after initialization.

        Returns:
            Settings: The validated settings instance.

        Raises:
            ValueError: If critical environment variables are missing or contain placeholder values.
        """
        # Check for critical environment variables
        critical_vars = []
        if not self.neo4j_uri or self.neo4j_uri in ['', 'neo4j+s://your-instance.databases.neo4j.io']:
            critical_vars.append('NEO4J_URI')
        # Allow 'neo4j' if it's paired with a real password (not placeholder)
        if not self.neo4j_username or self.neo4j_username == '':
            critical_vars.append('NEO4J_USERNAME')
        elif (self.neo4j_username == 'neo4j' and 
              self.neo4j_password in ['your-password-here', 'password', '123456', '']):
            critical_vars.append('NEO4J_USERNAME (using default username with placeholder password)')
        if not self.neo4j_password or self.neo4j_password in ['', 'your-password-here']:
            critical_vars.append('NEO4J_PASSWORD')
        if not self.openai_api_key or self.openai_api_key in ['', 'sk-your-api-key-here']:
            critical_vars.append('OPENAI_API_KEY')
        
        if critical_vars:
            raise ValueError(
                f"Missing or placeholder values for critical environment variables: {', '.join(critical_vars)}. "
                f"Please copy .env.example to .env and update with your actual values."
            )
        
        # Enforce production safety settings
        if self.environment_mode == EnvironmentMode.PRODUCTION:
            logger.info("🛡️ Enforcing production safety settings...")
            
            # Force safe LLM Cypher settings in production
            if self.llm_cypher_allow_call:
                logger.warning("⚠️ Overriding LLM_CYPHER_ALLOW_CALL=false in production for security")
                object.__setattr__(self, 'llm_cypher_allow_call', False)
            
            # Ensure reasonable limits
            if self.llm_cypher_max_hops > 5:
                logger.warning(f"⚠️ Capping LLM_CYPHER_MAX_HOPS to 5 in production (was {self.llm_cypher_max_hops})")
                object.__setattr__(self, 'llm_cypher_max_hops', 5)
            
            if self.llm_cypher_force_limit > 100:
                logger.warning(f"⚠️ Capping LLM_CYPHER_FORCE_LIMIT to 100 in production (was {self.llm_cypher_force_limit})")
                object.__setattr__(self, 'llm_cypher_force_limit', 100)
            
            # Log production safety confirmation
            logger.info(f"✅ Production safety enforced: ALLOW_CALL={self.llm_cypher_allow_call}, "
                       f"MAX_HOPS={self.llm_cypher_max_hops}, FORCE_LIMIT={self.llm_cypher_force_limit}")
        
        # Validate confidence caps are in valid range
        confidence_caps = [
            ("SEMANTIC_CONFIDENCE_CAP", self.semantic_confidence_cap),
            ("TRAVERSAL_CONFIDENCE_CAP", self.traversal_confidence_cap),
            ("FUSION_CONFIDENCE_CAP", self.fusion_confidence_cap)
        ]
        
        for cap_name, cap_value in confidence_caps:
            if not (0.0 <= cap_value <= 1.0):
                raise ValueError(f"{cap_name} must be between 0.0 and 1.0, got {cap_value}")
        
        return self
    # -------------------------------------------------------------- end validate_configuration()

    # ______________________
    # -- Embedded Classes --
    #
    # --------------------------------------------------------------------------------- Config
    class Config:
        """Pydantic configuration for the settings model."""
        env_file = str(DOTENV_PATH)
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env file
    # --------------------------------------------------------------------------------- end Config

# ------------------------------------------------------------------------- end class Settings

# __________________________________________________________________________
# Module-Level Singletons and Exports
#

# Create global settings instance
settings = Settings()

# Log configuration on import if verbose
if settings.verbose:
    settings.log_configuration()

# Export commonly used settings
APP_ENV = settings.app_env
DEBUG = settings.debug
VERBOSE = settings.verbose

# Neo4j settings
NEO4J_URI = settings.neo4j_uri
NEO4J_USERNAME = settings.neo4j_username
NEO4J_PASSWORD = settings.neo4j_password
NEO4J_DATABASE = settings.neo4j_database

# OpenAI settings
OPENAI_API_KEY = settings.openai_api_key
OPENAI_MODEL_MINI = settings.openai_model_mini
OPENAI_REQUESTS_PER_MINUTE = settings.openai_requests_per_minute
OPENAI_TOKENS_PER_MINUTE = settings.openai_tokens_per_minute

# LLM Client settings
LLM_MODEL_CONFIG = settings.get_llm_model_config()
EMBEDDING_CONFIG = settings.get_embedding_config()
EFFECTIVE_RPM = settings.effective_openai_rpm
EFFECTIVE_TPM = settings.effective_openai_tpm

# LLM Feature flags
IS_GPT5_MINI = settings.is_gpt5_mini
SUPPORTS_GPT5_ADVANCED = settings.supports_gpt5_advanced_features
# __________________________________________________________________________
# Module Initialization / Main Execution Guard (if applicable)
#
# --------------------------------------------------------------------------------- main()
def _main() -> None:
    """Execute a simple smoke test when running this module directly."""
    print(f"Running configuration smoke test for {__file__}...")
    print(f"Environment mode: {settings.environment_mode}")
    print("Settings loaded successfully.")


if __name__ == "__main__":
    _main()
# --------------------------------------------------------------------------------- end main()

# __________________________________________________________________________
# End of File
#
