# =========================================================================
# File: __init__.py
# Project: APH-IF Technology Framework
#          Advanced Parallel Hybrid - Intelligent Fusion System
# Author: Alexander Ricciardi
# Date: 2025-08-05
# File Path: backend/__init__.py
# =========================================================================

"""
APH-IF Backend Package

Advanced Parallel Hybrid - Intelligent Fusion Backend Service
Provides microservices architecture for parallel hybrid processing,
context fusion, and intelligent query handling.
"""

__version__ = "1.0.0"
__author__ = "Alexander Ricciardi"
__description__ = "APH-IF Backend Service - Advanced Parallel Hybrid Intelligent Fusion"

# Core modules
from .main import app
from .config import get_config
from .parallel_hybrid import get_parallel_engine
from .context_fusion import get_fusion_engine
from .circuit_breaker import get_circuit_breaker_manager
from .llm import get_llm_client

__all__ = [
    "app",
    "get_config", 
    "get_parallel_engine",
    "get_fusion_engine",
    "get_circuit_breaker_manager",
    "get_llm_client"
]
