import os
from enum import Enum
from typing import Any, Tuple

import requests
import streamlit as st
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
BACKEND_URL = settings.backend_url

st.set_page_config(page_title="APH-IF", layout="wide")
st.title("APH-IF • Frontend")

# Display environment and Neo4j instance information
neo4j_info = settings.get_current_neo4j_info()
col1, col2 = st.columns([3, 1])

with col1:
    st.caption(f"Environment: {settings.app_env}")

with col2:
    # Color-code the Neo4j instance type
    if neo4j_info["instance_type"] == "DEVELOPMENT":
        st.success(f"🔧 {neo4j_info['instance_type']}")
    elif neo4j_info["instance_type"] == "PRODUCTION":
        st.error(f"🚨 {neo4j_info['instance_type']}")
    elif neo4j_info["instance_type"] == "TEST":
        st.warning(f"⚠️ {neo4j_info['instance_type']}")
    else:
        st.info(f"❓ {neo4j_info['instance_type']}")

# Show warning message if needed
if neo4j_info["instance_type"] in ["PRODUCTION", "TEST"]:
    st.warning(neo4j_info["warning"])


def backend_health() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/healthz", timeout=2)
        return r.ok
    except Exception:
        return False


ok = backend_health()
pill = "🟢 Backend OK" if ok else "🔴 Backend Down"
st.sidebar.markdown(f"**Status:** {pill}")

query_text = st.text_input("Your question")
if st.button("Ask") and query_text:
    with st.spinner("Querying backend…"):
        try:
            resp = requests.post(f"{BACKEND_URL}/query", json={"query": query_text}, timeout=10)
            if resp.ok:
                data: Any = resp.json()
                st.write(data.get("answer", "No answer"))
            else:
                st.error(f"Backend error: {resp.status_code}")
        except Exception as exc:
            st.error(f"Request failed: {exc}")


