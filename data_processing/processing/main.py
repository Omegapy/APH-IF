# -------------------------------------------------------------------------
# File: main.py
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 2025-08-12
# File Path: data_processing/processing/main.py
# ------------------------------------------------------------------------

# --- Module Objective ---
#   Minimal FastAPI service exposing `/healthz` with environment-aware Neo4j
#   configuration for development and service discovery.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Enum: EnvironmentMode
# - Class: Settings (BaseSettings)
# - FastAPI app and GET /healthz endpoint
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Third-Party: fastapi, pydantic-settings
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - APH-IF  
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG – Intelligent Fusion (APH-IF)
# -------------------------------------------------------------------------

"""
APH-IF Data Processing Service (Phase 1)

Purpose
-------
Provide a minimal FastAPI service exposing `/healthz` with environment-aware
Neo4j configuration details to support native Windows development and service
discovery.

How it works
------------
- Settings resolve the active environment and map it to Neo4j credentials.
- `/healthz` returns the resolved instance type (DEVELOPMENT/PRODUCTION/TEST),
  URI, and a clear safety message.

Environment variables
---------------------
- APP_ENV (development/production), FORCE_TEST_DB (true/false)
- Neo4j credentials per environment (DEV/PROD/TEST triples)
- Backend and frontend URLs for visibility

Usage
-----
  uv run uvicorn processing.main:app --reload --port 8010
"""

# =========================================================================
# Imports
# =========================================================================
# Standard library imports
# (None required for this module)

# Third-party library imports
from enum import Enum
from typing import Any, Dict, Tuple

from fastapi import FastAPI
from pydantic_settings import BaseSettings


# =========================================================================
# Class Definitions
# =========================================================================

# ------------------------------------------------------------------------- class EnvironmentMode
class EnvironmentMode(str, Enum):
    """Supported environment modes."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
# ------------------------------------------------------------------------- end class EnvironmentMode


# ------------------------------------------------------------------------- class Settings
class Settings(BaseSettings):
    """Application settings with environment-aware Neo4j selection."""

    # Environment Configuration
    app_env: EnvironmentMode = EnvironmentMode.DEVELOPMENT
    force_test_db: bool = False

    # Service URLs
    backend_url: str = "http://localhost:8000"
    data_processing_url: str = "http://localhost:8010"
    frontend_url: str = "http://localhost:8501"

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-5-2025-08-07"
    openai_model_mini: str = "gpt-5-mini-2025-08-07"
    openai_model_nano: str = "gpt-5-nano-2025-08-07"

    # Gemini Configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"

    # Neo4j Production Instance
    neo4j_uri_prod: str = ""
    neo4j_username_prod: str = "neo4j"
    neo4j_password_prod: str = ""

    # Neo4j Development Instance
    neo4j_uri_dev: str = ""
    neo4j_username_dev: str = "neo4j"
    neo4j_password_dev: str = ""

    # Neo4j Testing Instance
    neo4j_uri_test: str = ""
    neo4j_username_test: str = "neo4j"
    neo4j_password_test: str = ""

    class Config:
        env_file = "../.env"
        case_sensitive = False

    # --------------------------------------------------------------------------------- get_neo4j_credentials()
    def get_neo4j_credentials(self) -> Tuple[str, str, str]:
        """Get Neo4j credentials based on current environment mode."""
        if self.force_test_db:
            if not all([self.neo4j_uri_test, self.neo4j_username_test, self.neo4j_password_test]):
                raise ValueError("Test Neo4j credentials are incomplete")
            return (self.neo4j_uri_test, self.neo4j_username_test, self.neo4j_password_test)

        if self.app_env == EnvironmentMode.DEVELOPMENT:
            if not all([self.neo4j_uri_dev, self.neo4j_username_dev, self.neo4j_password_dev]):
                raise ValueError("Development Neo4j credentials are incomplete")
            return (self.neo4j_uri_dev, self.neo4j_username_dev, self.neo4j_password_dev)

        elif self.app_env == EnvironmentMode.PRODUCTION:
            if not all([self.neo4j_uri_prod, self.neo4j_username_prod, self.neo4j_password_prod]):
                raise ValueError("Production Neo4j credentials are incomplete")
            return (self.neo4j_uri_prod, self.neo4j_username_prod, self.neo4j_password_prod)

        else:
            raise ValueError(f"Invalid environment mode: {self.app_env}")

    # --------------------------------------------------------------------------------- get_current_neo4j_info()
    def get_current_neo4j_info(self) -> dict:
        """Get information about the currently selected Neo4j instance.

        Policy:
            - FORCE_TEST_DB has highest precedence and selects TEST instance.
            - Otherwise APP_ENV determines DEVELOPMENT vs PRODUCTION.
            - Includes a human-focused `warning` string for UI display.
        """
        try:
            uri, username, _ = self.get_neo4j_credentials()

            if self.force_test_db:
                instance_type = "TEST"
                warning = "⚠️  TESTING DATABASE - Only use for explicit testing scenarios"
            elif self.app_env == EnvironmentMode.DEVELOPMENT:
                instance_type = "DEVELOPMENT"
                warning = "✅ Safe for development work"
            elif self.app_env == EnvironmentMode.PRODUCTION:
                instance_type = "PRODUCTION"
                warning = "🚨 LIVE DATA - Use with extreme caution"
            else:
                instance_type = "UNKNOWN"
                warning = "❌ Unknown environment mode"

            return {
                "instance_type": instance_type,
                "environment": self.app_env,
                "uri": uri,
                "username": username,
                "warning": warning,
                "force_test_db": self.force_test_db
            }
    # --------------------------------------------------------------------------------- end get_current_neo4j_info()
# ------------------------------------------------------------------------- end class Settings
        except ValueError as e:
            return {
                "instance_type": "ERROR",
                "environment": self.app_env,
                "uri": "N/A",
                "username": "N/A",
                "warning": f"❌ Configuration Error: {str(e)}",
                "force_test_db": self.force_test_db
            }


# =========================================================================
# FastAPI App and Endpoints
# =========================================================================
settings = Settings()
app = FastAPI(title="APH-IF Data Processing", version="0.1.0")


# =========================================================================
# Standalone Function Definitions
# =========================================================================

# --------------------------------------------------------------------------------- healthz()
@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    """Liveness/readiness probe with Neo4j instance information."""
    neo4j_info = settings.get_current_neo4j_info()

    return {
        "status": "ok",
        "service": "data_processing",
        "environment": settings.app_env,
        "backend_url": settings.backend_url,
        "neo4j_instance": {
            "type": neo4j_info["instance_type"],
            "uri": neo4j_info["uri"],
            "warning": neo4j_info["warning"],
            "force_test_db": neo4j_info["force_test_db"]
        }
    }
# --------------------------------------------------------------------------------- end healthz()


