"""
Core infrastructure module for APH-IF backend.

This module provides core functionality including configuration, database access,
LLM clients, and CPU pooling infrastructure.
"""

from .config import settings
from .database import (
    get_database,
    get_database_for_traversal_search,
    get_database_for_semantic_search,
    database_health_check,
    reset_database_connection,
)
from .async_llm_client import get_async_llm_client, shutdown_async_llm_client
from .llm_client import get_openai_client
from .cpu_pool import get_cpu_pool, shutdown_cpu_pool

__all__ = [
    "settings",
    "get_database",
    "get_database_for_traversal_search",
    "get_database_for_semantic_search",
    "database_health_check",
    "reset_database_connection",
    "get_async_llm_client",
    "shutdown_async_llm_client",
    "get_openai_client",
    "get_cpu_pool",
    "shutdown_cpu_pool",
]