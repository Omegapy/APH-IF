# -------------------------------------------------------------------------
# File: __init__.py
# Author: Alexander Ricciardi
# Date: 2025-09-18
# [File Path] backend/app/monitoring/__init__.py
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
#
# --- Module Functionality ---
#   Package initializer for monitoring. Re-exports core monitoring utilities
#   (performance monitor, timing collector, database metrics, circuit breaker)
#   to provide a convenient import surface for backend modules and routes.
# -------------------------------------------------------------------------
#
# --- Module Contents Overview ---
# - Functions: get_performance_monitor, performance_health_check, initialize_monitor
# - Functions: get_timing_collector, initialize_timing_collector, measure_operation,
#              get_timing_stats
# - Functions: get_database_metrics, initialize_database_metrics
# - Functions: get_circuit_breaker, circuit_breaker, circuit_breaker_health_check
# -------------------------------------------------------------------------
#
# --- Dependencies / Imports ---
# - Local Project Modules: performance_monitor, timing_collector, database_metrics,
#   circuit_breaker
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

"""
Monitoring and performance module for APH-IF backend.

This module provides performance monitoring, timing collection, database metrics,
and circuit breaker functionality.
"""

# __________________________________________________________________________
# Imports

from .performance_monitor import (
    get_performance_monitor,
    performance_health_check,
    initialize_monitor,
)
from .timing_collector import (
    get_timing_collector,
    initialize_timing_collector,
    measure_operation,
    get_timing_stats,
)
from .database_metrics import (
    get_database_metrics,
    initialize_database_metrics,
)
from .circuit_breaker import (
    get_circuit_breaker,
    circuit_breaker,
    circuit_breaker_health_check,
)

__all__ = [
    "get_performance_monitor",
    "performance_health_check",
    "initialize_monitor",
    "get_timing_collector",
    "initialize_timing_collector",
    "measure_operation",
    "get_timing_stats",
    "get_database_metrics",
    "initialize_database_metrics",
    "get_circuit_breaker",
    "circuit_breaker",
    "circuit_breaker_health_check",
]

# __________________________________________________________________________
# End of File
#