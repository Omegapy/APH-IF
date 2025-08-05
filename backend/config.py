# =========================================================================
# File: config.py
# Project: APH-IF Technology Framework
#          Advanced Parallel HybridRAG - Intelligent Fusion System
# Author: Alexander Ricciardi
# Date: 2025-08-05
# File Path: backend/config.py
# =========================================================================

"""
Configuration Management for APH-IF Backend

Handles loading and validation of configuration settings from environment variables,
secrets files, and default values. Provides centralized configuration access
for all backend components.
"""

import os
import logging
import toml
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings

# =========================================================================
# Configuration Classes
# =========================================================================
class DatabaseConfig(BaseSettings):
    """Database configuration settings"""
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", env="NEO4J_USERNAME") 
    neo4j_password: str = Field(default="YourStrongPassword", env="NEO4J_PASSWORD")
    connection_timeout: int = Field(default=30, env="NEO4J_CONNECTION_TIMEOUT")
    max_connection_pool_size: int = Field(default=50, env="NEO4J_MAX_POOL_SIZE")

class LLMConfig(BaseSettings):
    """LLM configuration settings"""
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    gemini_api_key: Optional[str] = Field(default=None, env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-pro", env="GEMINI_MODEL")
    default_provider: str = Field(default="openai", env="LLM_DEFAULT_PROVIDER")
    max_tokens: int = Field(default=4000, env="LLM_MAX_TOKENS")
    temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    timeout: int = Field(default=60, env="LLM_TIMEOUT")

class ParallelHybridRAGConfig(BaseSettings):
    """Parallel HybridRAG processing configuration"""
    max_vector_results: int = Field(default=10, env="PH_MAX_VECTOR_RESULTS")
    max_graph_results: int = Field(default=10, env="PH_MAX_GRAPH_RESULTS")
    vector_similarity_threshold: float = Field(default=0.7, env="PH_VECTOR_THRESHOLD")
    graph_traversal_depth: int = Field(default=3, env="PH_GRAPH_DEPTH")
    enable_caching: bool = Field(default=True, env="PH_ENABLE_CACHING")
    cache_ttl: int = Field(default=3600, env="PH_CACHE_TTL")  # seconds

class CircuitBreakerConfig(BaseSettings):
    """Circuit breaker configuration"""
    failure_threshold: int = Field(default=5, env="CB_FAILURE_THRESHOLD")
    recovery_timeout: int = Field(default=60, env="CB_RECOVERY_TIMEOUT")
    expected_exception: str = Field(default="Exception", env="CB_EXPECTED_EXCEPTION")
    
class APIConfig(BaseSettings):
    """API server configuration"""
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    log_level: str = Field(default="INFO", env="API_LOG_LEVEL")
    cors_origins: str = Field(default="*", env="API_CORS_ORIGINS")
    request_timeout: int = Field(default=120, env="API_REQUEST_TIMEOUT")
    max_request_size: int = Field(default=16777216, env="API_MAX_REQUEST_SIZE")  # 16MB

class APHIFConfig(BaseSettings):
    """Main APH-IF configuration container"""
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    llm: LLMConfig = LLMConfig()
    parallel_hybrid: ParallelHybridRAGConfig = ParallelHybridRAGConfig()
    circuit_breaker: CircuitBreakerConfig = CircuitBreakerConfig()
    api: APIConfig = APIConfig()
    
    # General settings
    environment: str = Field(default="development", env="ENVIRONMENT")
    service_name: str = Field(default="aph-if-backend", env="SERVICE_NAME")
    version: str = Field(default="1.0.0", env="SERVICE_VERSION")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# =========================================================================
# Configuration Loading Functions
# =========================================================================
def load_secrets_from_toml(secrets_path: str = ".streamlit/secrets.toml") -> Dict[str, Any]:
    """Load secrets from Streamlit secrets.toml file"""
    secrets = {}
    
    try:
        if Path(secrets_path).exists():
            with open(secrets_path, 'r') as f:
                secrets = toml.load(f)
            logging.info(f"Loaded secrets from {secrets_path}")
        else:
            logging.warning(f"Secrets file not found: {secrets_path}")
    except Exception as e:
        logging.error(f"Error loading secrets from {secrets_path}: {e}")
    
    return secrets

def update_config_from_secrets(config: APHIFConfig, secrets: Dict[str, Any]) -> APHIFConfig:
    """Update configuration with values from secrets"""
    
    # Update database config
    if "NEO4J_URI" in secrets:
        config.database.neo4j_uri = secrets["NEO4J_URI"]
    if "NEO4J_USERNAME" in secrets:
        config.database.neo4j_username = secrets["NEO4J_USERNAME"]
    if "NEO4J_PASSWORD" in secrets:
        config.database.neo4j_password = secrets["NEO4J_PASSWORD"]
    
    # Update LLM config
    if "OPENAI_API_KEY" in secrets:
        config.llm.openai_api_key = secrets["OPENAI_API_KEY"]
    if "OPENAI_MODEL" in secrets:
        config.llm.openai_model = secrets["OPENAI_MODEL"]
    if "GEMINI_API_KEY" in secrets:
        config.llm.gemini_api_key = secrets["GEMINI_API_KEY"]
    
    return config

def validate_config(config: APHIFConfig) -> bool:
    """Validate configuration settings"""
    errors = []
    
    # Check required LLM API keys
    if not config.llm.openai_api_key and not config.llm.gemini_api_key:
        errors.append("At least one LLM API key (OpenAI or Gemini) must be provided")
    
    # Check database connection
    if not config.database.neo4j_uri:
        errors.append("Neo4j URI must be provided")
    
    # Intelligent fusion is always used - no strategy validation needed
    
    if errors:
        for error in errors:
            logging.error(f"Configuration validation error: {error}")
        return False
    
    return True

# =========================================================================
# Global Configuration Instance
# =========================================================================
_config_instance: Optional[APHIFConfig] = None

def get_config(reload: bool = False) -> APHIFConfig:
    """Get the global configuration instance"""
    global _config_instance
    
    if _config_instance is None or reload:
        # Load base configuration
        _config_instance = APHIFConfig()
        
        # Load and apply secrets
        secrets = load_secrets_from_toml()
        _config_instance = update_config_from_secrets(_config_instance, secrets)
        
        # Validate configuration
        if not validate_config(_config_instance):
            logging.warning("Configuration validation failed, proceeding with warnings")
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, _config_instance.api.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logging.info("Configuration loaded and validated successfully")
    
    return _config_instance

def reload_config() -> APHIFConfig:
    """Reload configuration from sources"""
    return get_config(reload=True)

# =========================================================================
# Configuration Utilities
# =========================================================================
def get_database_url() -> str:
    """Get formatted database URL"""
    config = get_config()
    return config.database.neo4j_uri

def get_llm_config(provider: Optional[str] = None) -> Dict[str, Any]:
    """Get LLM configuration for specified provider"""
    config = get_config()
    provider = provider or config.llm.default_provider
    
    if provider == "openai":
        return {
            "api_key": config.llm.openai_api_key,
            "model": config.llm.openai_model,
            "max_tokens": config.llm.max_tokens,
            "temperature": config.llm.temperature,
            "timeout": config.llm.timeout
        }
    elif provider == "gemini":
        return {
            "api_key": config.llm.gemini_api_key,
            "model": config.llm.gemini_model,
            "max_tokens": config.llm.max_tokens,
            "temperature": config.llm.temperature,
            "timeout": config.llm.timeout
        }
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def is_development() -> bool:
    """Check if running in development environment"""
    config = get_config()
    return config.environment.lower() in ["development", "dev", "local"]

def is_production() -> bool:
    """Check if running in production environment"""
    config = get_config()
    return config.environment.lower() in ["production", "prod"]
