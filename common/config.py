"""
Common configuration module for APH-IF services.

This module provides centralized environment management with automatic Neo4j instance selection
based on the current environment mode (development, production, testing).

Environment Modes:
- DEVELOPMENT: Uses development Neo4j instance (safe for development work)
- PRODUCTION: Uses production Neo4j instance (live data, use with caution)
- TESTING: Uses testing Neo4j instance (only for explicit testing scenarios)

The Neo4j instance is automatically selected based on the APP_ENV variable, except for
testing mode which must be explicitly activated via FORCE_TEST_DB=true.
"""

import os
from enum import Enum
from typing import Optional, Tuple
from pathlib import Path

from pydantic_settings import BaseSettings


class EnvironmentMode(str, Enum):
    """Supported environment modes."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class Neo4jConfig:
    """Neo4j configuration selector based on environment mode."""

    def __init__(self, settings: 'Settings'):
        self.settings = settings

    def get_neo4j_credentials(self) -> Tuple[str, str, str]:
        """
        Get Neo4j credentials based on current environment mode.

        Returns:
            Tuple of (uri, username, password)

        Raises:
            ValueError: If environment mode is invalid or credentials are missing
        """
        # Check for explicit test mode override
        if self.settings.force_test_db:
            if not all([self.settings.neo4j_uri_test, self.settings.neo4j_username_test, self.settings.neo4j_password_test]):
                raise ValueError("Test Neo4j credentials are incomplete")
            return (
                self.settings.neo4j_uri_test,
                self.settings.neo4j_username_test,
                self.settings.neo4j_password_test
            )

        # Select based on environment mode
        if self.settings.app_env == EnvironmentMode.DEVELOPMENT:
            if not all([self.settings.neo4j_uri_dev, self.settings.neo4j_username_dev, self.settings.neo4j_password_dev]):
                raise ValueError("Development Neo4j credentials are incomplete")
            return (
                self.settings.neo4j_uri_dev,
                self.settings.neo4j_username_dev,
                self.settings.neo4j_password_dev
            )

        elif self.settings.app_env == EnvironmentMode.PRODUCTION:
            if not all([self.settings.neo4j_uri_prod, self.settings.neo4j_username_prod, self.settings.neo4j_password_prod]):
                raise ValueError("Production Neo4j credentials are incomplete")
            return (
                self.settings.neo4j_uri_prod,
                self.settings.neo4j_username_prod,
                self.settings.neo4j_password_prod
            )

        else:
            raise ValueError(f"Invalid environment mode: {self.settings.app_env}")


class Settings(BaseSettings):
    """Shared settings for all APH-IF services with environment-aware Neo4j selection."""

    # Environment Configuration
    app_env: EnvironmentMode = EnvironmentMode.DEVELOPMENT
    force_test_db: bool = False  # Explicit flag to use test database

    # Service URLs
    backend_url: str = "http://localhost:8000"
    data_processing_url: str = "http://localhost:8010"
    frontend_url: str = "http://localhost:8501"

    # OpenAI Configuration
    openai_api_key: str = ""
    openai_model: str = "gpt-5-2025-08-07"
    openai_model_mini: str = "gpt-5-mini-2025-08-07"
    openai_model_nano: str = "gpt-5-nano-2025-08-07"

    # Google Gemini Configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"

    # Neo4j Production Instance (Live Data - Use with Caution)
    neo4j_uri_prod: str = ""
    neo4j_username_prod: str = "neo4j"
    neo4j_password_prod: str = ""

    # Neo4j Development Instance (Safe for Development)
    neo4j_uri_dev: str = ""
    neo4j_username_dev: str = "neo4j"
    neo4j_password_dev: str = ""

    # Neo4j Testing Instance (Only for Explicit Testing)
    neo4j_uri_test: str = ""
    neo4j_username_test: str = "neo4j"
    neo4j_password_test: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def neo4j(self) -> Neo4jConfig:
        """Get Neo4j configuration manager."""
        return Neo4jConfig(self)

    def get_current_neo4j_info(self) -> dict:
        """Get information about the currently selected Neo4j instance."""
        try:
            uri, username, _ = self.neo4j.get_neo4j_credentials()

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
        except ValueError as e:
            return {
                "instance_type": "ERROR",
                "environment": self.app_env,
                "uri": "N/A",
                "username": "N/A",
                "warning": f"❌ Configuration Error: {str(e)}",
                "force_test_db": self.force_test_db
            }


def get_settings() -> Settings:
    """Get application settings with environment-aware configuration."""
    return Settings()
