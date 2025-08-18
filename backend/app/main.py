"""
APH-IF Backend Service (Phase 1)

Minimal FastAPI app exposing a health check and a stub /query endpoint
for Phase 1 smoke testing and native Windows development.
"""

from enum import Enum
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class EnvironmentMode(str, Enum):
    """Supported environment modes."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


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
    openai_model: str = "gpt-5"
    openai_model_mini: str = "gpt-5-mini"
    openai_model_nano: str = "gpt-5-nano"

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

    def get_current_neo4j_info(self) -> dict:
        """Get information about the currently selected Neo4j instance."""
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
        except ValueError as e:
            return {
                "instance_type": "ERROR",
                "environment": self.app_env,
                "uri": "N/A",
                "username": "N/A",
                "warning": f"❌ Configuration Error: {str(e)}",
                "force_test_db": self.force_test_db
            }


settings = Settings()
app = FastAPI(title="APH-IF Backend", version="0.1.0")


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    """Liveness/readiness probe with Neo4j instance information."""
    try:
        neo4j_info = settings.get_current_neo4j_info()

        return {
            "status": "ok",
            "service": "backend",
            "environment": str(settings.app_env),
            "data_processing_url": settings.data_processing_url,
            "neo4j_instance": {
                "type": neo4j_info["instance_type"],
                "uri": neo4j_info["uri"],
                "warning": neo4j_info["warning"]
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "service": "backend",
            "error": str(e),
            "environment": str(settings.app_env)
        }


@app.get("/test-env")
async def test_env() -> Dict[str, Any]:
    """Test endpoint to verify environment configuration."""
    return {
        "app_env": str(settings.app_env),
        "force_test_db": settings.force_test_db,
        "neo4j_info": settings.get_current_neo4j_info()
    }


class QueryRequest(BaseModel):
    """Request model for the stub query endpoint.

    Attributes:
        query: User question string
        conversation_id: Optional conversation/session identifier
        top_k: Requested number of passages
        min_score: Minimum score threshold
    """

    query: str
    conversation_id: Optional[str] = None
    top_k: int = 5
    min_score: float = 0.7


@app.post("/query")
async def query_endpoint(req: QueryRequest) -> Dict[str, Any]:
    """Stubbed query endpoint for Phase 1 E2E smoke test."""
    return {
        "answer": f"Stub answer to: {req.query}",
        "citations": [],
        "retrieval": {
            "vector": {"hits": 0, "latency_ms": 0},
            "graph": {"hits": 0, "latency_ms": 0},
        },
        "meta": {"orchestrator_latency_ms": 1},
    }


