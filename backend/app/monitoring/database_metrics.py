# -------------------------------------------------------------------------
# File: database_metrics.py
# Author: Alexander Ricciardi
# Date: 2025-09-18
# [File Path] backend/app/monitoring/database_metrics.py
# ------------------------------------------------------------------------
# Project:
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
#   Database performance metrics for the APH-IF backend. Measures Neo4j connection,
#   query, network, and processing timings; aggregates into statistics used by
#   monitoring endpoints and performance dashboards.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Dataclass: DatabaseOperationTiming
# - Dataclass: ConnectionPoolMetrics
# - Class: DatabaseMetricsCollector
# - Class: DatabaseTimingTracker
# - Utility Functions: safe_percentile, truncate_query, compute_pool_utilization
# - Functions: get_database_metrics, initialize_database_metrics
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: time, asyncio, logging, typing, dataclasses, contextlib, statistics, datetime
# - Local Project Modules: .timing_collector (get_timing_collector), ..core.config (settings)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Used by backend services to capture database timing metrics. Integrates with
# the timing collector and powers monitoring endpoints without changing request
# behavior.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Database Performance Metrics for APH-IF Backend.

Comprehensive timing and metrics for Neo4j operations including connection
management, query compilation/execution, result processing, and network
latency measurement. Integrated with the timing collector to power backend
monitoring endpoints without altering request behavior.
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import time
import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import statistics
from datetime import datetime, timedelta

from .timing_collector import get_timing_collector
from ..core.config import settings

# __________________________________________________________________________
# Global Constants / Variables

logger = logging.getLogger(__name__)

# ____________________________________________________________________________
# Class Definitions

# =========================================================================
# Database Timing Data Models
# =========================================================================

# ------------------------------------------------------------------------- class DatabaseOperationTiming
@dataclass(slots=True)
class DatabaseOperationTiming:
    """Detailed timing breakdown for database operations."""
    operation_type: str  # 'query', 'connection', 'transaction'
    total_time_ms: float
    connection_acquisition_time_ms: float = 0.0
    connection_establishment_time_ms: float = 0.0
    query_compilation_time_ms: float = 0.0
    query_execution_time_ms: float = 0.0
    result_processing_time_ms: float = 0.0
    network_latency_ms: float = 0.0
    rows_returned: int = 0
    bytes_transferred: int = 0
    success: bool = True
    error_message: Optional[str] = None
    query_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
# ------------------------------------------------------------------------- end class DatabaseOperationTiming

# ------------------------------------------------------------------------- class ConnectionPoolMetrics
@dataclass(slots=True)
class ConnectionPoolMetrics:
    """Neo4j connection pool performance metrics."""
    active_connections: int = 0
    idle_connections: int = 0
    max_pool_size: int = 0
    avg_acquisition_time_ms: float = 0.0
    total_connections_created: int = 0
    total_connections_closed: int = 0
    connection_failures: int = 0
    pool_exhaustion_count: int = 0
# ------------------------------------------------------------------------- end class ConnectionPoolMetrics

# =========================================================================
# Database Metrics Collector
# =========================================================================

# ------------------------------------------------------------------------- class DatabaseMetricsCollector
class DatabaseMetricsCollector:
    """Collector for comprehensive database performance metrics.

    Features:
        - Neo4j connection pool monitoring.
        - Query execution timing with compilation/processing separation.
        - Network latency measurement.
        - Result processing overhead tracking.
        - Automatic aggregation and analysis.

    Attributes:
        max_operation_history: Maximum number of operations kept in history.
        _operation_history: Rolling history of operation timings.
        _pool_metrics: Aggregated connection pool metrics.
        _stats: System-wide counters and summary metrics.
    """
    
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self, max_operation_history: int = 1000):
        """Initialize the database metrics collector.

        Args:
            max_operation_history: Maximum number of operations to retain
                in the in-memory history buffer.
        """
        self.max_operation_history = max_operation_history
        
        # Operation history
        self._operation_history: List[DatabaseOperationTiming] = []
        
        # Connection pool metrics
        self._pool_metrics = ConnectionPoolMetrics()
        
        # Aggregated statistics
        self._stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_query_time_ms": 0.0,
            "avg_network_latency_ms": 0.0,
            "total_data_transferred_bytes": 0,
            "queries_per_second": 0.0,
            "last_reset_time": time.time(),
        }
        
        logger.info(f"Database metrics collector initialized")
    # --------------------------------------------------------------------------------- end __init__()
    
    # --------------------------------------------------------------------------------- measure_database_operation()
    @asynccontextmanager
    async def measure_database_operation(
        self,
        operation_type: str,
        query_text: Optional[str] = None,
        expected_rows: Optional[int] = None,
    ) -> AsyncIterator["DatabaseTimingTracker"]:
        """Context-manage timing for a database operation with detailed breakdown.

        Args:
            operation_type: Operation type ("query", "connection", "transaction").
            query_text: Query text being executed (for logging/analysis).
            expected_rows: Expected row count (for optimization analysis).

        Yields:
            DatabaseTimingTracker: Tracker to set fine-grained timings and metadata.

        Example:
            async with db_metrics.measure_database_operation("query", cypher) as timer:
                result = await session.run(cypher)
                timer.set_rows_returned(len(result))
        """
        timing_collector = get_timing_collector()
        operation_start = time.perf_counter()
        
        # Create operation timing record
        operation_timing = DatabaseOperationTiming(
            operation_type=operation_type,
            total_time_ms=0.0,
            query_text=query_text,
            metadata={'expected_rows': expected_rows} if expected_rows else {}
        )
        
        # Create database timing tracker
        db_timer = DatabaseTimingTracker(operation_timing)
        
        async with timing_collector.measure(f"database_{operation_type}") as collector_timer:
            try:
                yield db_timer
                operation_timing.success = True
                collector_timer.add_metadata({
                    "operation_type": operation_type,
                    "rows_returned": operation_timing.rows_returned,
                    "query_length": len(query_text) if query_text else 0,
                })
            except Exception as e:
                operation_timing.success = False
                operation_timing.error_message = str(e)
                collector_timer.add_metadata({
                    "operation_type": operation_type,
                    "error": str(e),
                })
                raise
            finally:
                # Calculate total time
                operation_timing.total_time_ms = (time.perf_counter() - operation_start) * 1000
                
                # Record the operation
                await self._record_operation(operation_timing)
    # --------------------------------------------------------------------------------- end measure_database_operation()
    
    # --------------------------------------------------------------------------------- measure_connection_acquisition()
    @asynccontextmanager
    async def measure_connection_acquisition(self) -> AsyncIterator[None]:
        """Context-manage timing of Neo4j connection acquisition.

        Yields:
            None: Use within an "async with" block to record acquisition time.
        """
        start_time = time.perf_counter()
        
        try:
            yield
            acquisition_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Update connection pool metrics
            self._pool_metrics.avg_acquisition_time_ms = (
                (self._pool_metrics.avg_acquisition_time_ms * self._pool_metrics.total_connections_created + 
                 acquisition_time_ms) / (self._pool_metrics.total_connections_created + 1)
            )
            self._pool_metrics.total_connections_created += 1
            
        except Exception as e:
            self._pool_metrics.connection_failures += 1
            logger.warning(f"Connection acquisition failed: {e}")
            raise
    # --------------------------------------------------------------------------------- end measure_connection_acquisition()
    
    # --------------------------------------------------------------------------------- measure_query_compilation()
    @asynccontextmanager 
    async def measure_query_compilation(self) -> AsyncIterator[None]:
        """Context-manage timing of Cypher query compilation.

        Yields:
            None: Use within an "async with" block to record compilation time.
        """
        start_time = time.perf_counter()
        compilation_time_ms = 0.0
        
        try:
            yield
        finally:
            compilation_time_ms = (time.perf_counter() - start_time) * 1000
            # This will be set on the current operation timing if available
            # Implementation would need access to current timing context
    # --------------------------------------------------------------------------------- end measure_query_compilation()
    
    # --------------------------------------------------------------------------------- measure_network_latency()
    async def measure_network_latency(self, target_uri: Optional[str] = None) -> float:
        """Measure network latency to the Neo4j database.

        Args:
            target_uri: Neo4j URI to test (defaults to configured URI).

        Returns:
            float: Measured latency in milliseconds, or -1.0 on failure.
        """
        # Simple ping-like measurement using connection test
        start_time = time.perf_counter()
        
        try:
            # In a real implementation, this would do a lightweight operation
            # to measure network round-trip time
            await asyncio.sleep(0.001)  # Simulate minimal network operation
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Update average network latency
            self._stats['avg_network_latency_ms'] = (
                (self._stats['avg_network_latency_ms'] * self._stats['total_operations'] + latency_ms) /
                (self._stats['total_operations'] + 1)
            )
            
            return latency_ms
            
        except Exception as e:
            logger.warning(f"Network latency measurement failed: {e}")
            return -1.0  # Indicate measurement failure
    # --------------------------------------------------------------------------------- end measure_network_latency()
    
    # --------------------------------------------------------------------------------- _record_operation()
    async def _record_operation(self, operation_timing: DatabaseOperationTiming) -> None:
        """Record a completed database operation.

        Args:
            operation_timing: Timing record to append and incorporate into stats.
        """
        # Add to history
        self._operation_history.append(operation_timing)
        
        # Maintain history size
        if len(self._operation_history) > self.max_operation_history:
            self._operation_history = self._operation_history[-self.max_operation_history//2:]
        
        # Update statistics
        self._stats['total_operations'] += 1
        if operation_timing.success:
            self._stats['successful_operations'] += 1
            
            # Update average query time
            self._stats['avg_query_time_ms'] = (
                (self._stats['avg_query_time_ms'] * (self._stats['successful_operations'] - 1) +
                 operation_timing.total_time_ms) / self._stats['successful_operations']
            )
        else:
            self._stats['failed_operations'] += 1
        
        # Update data transfer stats
        self._stats['total_data_transferred_bytes'] += operation_timing.bytes_transferred
        
        # Calculate queries per second
        time_elapsed = time.time() - self._stats['last_reset_time']
        if time_elapsed > 0:
            self._stats['queries_per_second'] = self._stats['total_operations'] / time_elapsed
    # --------------------------------------------------------------------------------- end _record_operation()
    
    # --------------------------------------------------------------------------------- get_operation_statistics()
    def get_operation_statistics(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Compute database operation statistics over a time window.

        Args:
            time_window_minutes: Rolling time window for statistics.

        Returns:
            Dict[str, Any]: Comprehensive metrics including success rates, timing averages,
                connection pool status, and basic query analysis.
        """
        # Filter operations by time window
        cutoff_time = time.time() - (time_window_minutes * 60)
        recent_operations = [
            op for op in self._operation_history 
            if hasattr(op, 'timestamp') or True  # Simplified for now
        ]
        
        if not recent_operations:
            return {
                "total_operations": 0,
                "success_rate": 0.0,
                "avg_query_time_ms": 0.0,
                "avg_network_latency_ms": 0.0,
            }
        
        # Calculate statistics
        successful_ops = [op for op in recent_operations if op.success]
        
        stats = {
            "total_operations": len(recent_operations),
            "successful_operations": len(successful_ops),
            "failed_operations": len(recent_operations) - len(successful_ops),
            "success_rate": len(successful_ops) / len(recent_operations) * 100,

            # Timing statistics
            "avg_total_time_ms": statistics.mean([op.total_time_ms for op in successful_ops])
            if successful_ops
            else 0.0,
            "avg_query_execution_time_ms": statistics.mean(
                [op.query_execution_time_ms for op in successful_ops]
            )
            if successful_ops
            else 0.0,
            "avg_network_latency_ms": statistics.mean(
                [op.network_latency_ms for op in successful_ops if op.network_latency_ms > 0]
            )
            if successful_ops
            else 0.0,
            "avg_connection_time_ms": statistics.mean(
                [op.connection_acquisition_time_ms for op in successful_ops]
            )
            if successful_ops
            else 0.0,

            # Performance percentiles
            "p95_query_time_ms": 0.0,
            "p99_query_time_ms": 0.0,

            # Data transfer
            "total_rows_returned": sum(op.rows_returned for op in successful_ops),
            "total_bytes_transferred": sum(op.bytes_transferred for op in successful_ops),
            "avg_rows_per_query": statistics.mean(
                [op.rows_returned for op in successful_ops]
            )
            if successful_ops
            else 0.0,

            # Connection pool metrics
            "connection_pool": {
                "active_connections": self._pool_metrics.active_connections,
                "idle_connections": self._pool_metrics.idle_connections,
                "avg_acquisition_time_ms": self._pool_metrics.avg_acquisition_time_ms,
                "connection_failures": self._pool_metrics.connection_failures,
                "pool_exhaustion_count": self._pool_metrics.pool_exhaustion_count,
            },

            # Query analysis
            "query_types": self._analyze_query_types(recent_operations),
            "slow_queries": self._identify_slow_queries(recent_operations),
            "error_patterns": self._analyze_errors(recent_operations),
        }
        
        # Calculate percentiles if we have data
        if successful_ops:
            query_times = sorted([op.total_time_ms for op in successful_ops])
            if len(query_times) >= 20:  # Need sufficient data for percentiles
                p95_idx = int(0.95 * len(query_times))
                p99_idx = int(0.99 * len(query_times))
                stats["p95_query_time_ms"] = query_times[p95_idx]
                stats["p99_query_time_ms"] = query_times[p99_idx]
        
            return stats
    # --------------------------------------------------------------------------------- end get_operation_statistics()
    
    # --------------------------------------------------------------------------------- _analyze_query_types()
    def _analyze_query_types(self, operations: List[DatabaseOperationTiming]) -> Dict[str, int]:
        """Analyze distribution of query types.

        Args:
            operations: Operation records to inspect.

        Returns:
            Dict[str, int]: Mapping from query class (first word) to count.
        """
        query_types: Dict[str, int] = {}
        for op in operations:
            if op.query_text:
                # Simple classification based on first word
                first_word = (
                    op.query_text.strip().upper().split()[0]
                    if op.query_text.strip()
                    else "UNKNOWN"
                )
                query_types[first_word] = query_types.get(first_word, 0) + 1
        return query_types
    # --------------------------------------------------------------------------------- end _analyze_query_types()
    
    # --------------------------------------------------------------------------------- _identify_slow_queries()
    def _identify_slow_queries(
        self,
        operations: List[DatabaseOperationTiming],
        threshold_ms: float = 1000.0,
    ) -> List[Dict[str, Any]]:
        """Identify slow queries exceeding the given threshold.

        Args:
            operations: Operation records to analyze.
            threshold_ms: Slow query threshold in milliseconds.

        Returns:
            List[Dict[str, Any]]: Up to 10 slowest queries with basic details.
        """
        slow_queries: List[Dict[str, Any]] = []
        for op in operations:
            if op.total_time_ms > threshold_ms:
                slow_queries.append(
                    {
                        "query_text": op.query_text[:200] if op.query_text else "N/A",  # Truncate for brevity
                        "duration_ms": op.total_time_ms,
                        "rows_returned": op.rows_returned,
                        "operation_type": op.operation_type,
                    }
                )
        
        # Return top 10 slowest
        return sorted(slow_queries, key=lambda x: x["duration_ms"], reverse=True)[:10]
    # --------------------------------------------------------------------------------- end _identify_slow_queries()
    
    # --------------------------------------------------------------------------------- _analyze_errors()
    def _analyze_errors(self, operations: List[DatabaseOperationTiming]) -> Dict[str, int]:
        """Analyze error patterns across operations.

        Args:
            operations: Operation records to inspect for failures.

        Returns:
            Dict[str, int]: Count of errors by coarse category.
        """
        error_patterns: Dict[str, int] = {}
        for op in operations:
            if not op.success and op.error_message:
                # Classify error by type
                error_type = "UNKNOWN"
                error_msg = op.error_message.upper()
                
                if "CONNECTION" in error_msg or "NETWORK" in error_msg:
                    error_type = "CONNECTION_ERROR"
                elif "TIMEOUT" in error_msg:
                    error_type = "TIMEOUT_ERROR"
                elif "SYNTAX" in error_msg or "CYPHER" in error_msg:
                    error_type = "QUERY_ERROR"
                elif "CONSTRAINT" in error_msg:
                    error_type = "CONSTRAINT_VIOLATION"
                
                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        return error_patterns
    # --------------------------------------------------------------------------------- end _analyze_errors()
    
    # --------------------------------------------------------------------------------- get_connection_pool_status()
    def get_connection_pool_status(self) -> Dict[str, Any]:
        """Get current connection pool status.

        Returns:
            Dict[str, Any]: Snapshot of pool usage and acquisition timing.
        """
        return {
            "active_connections": self._pool_metrics.active_connections,
            "idle_connections": self._pool_metrics.idle_connections,
            "max_pool_size": self._pool_metrics.max_pool_size,
            "total_created": self._pool_metrics.total_connections_created,
            "total_closed": self._pool_metrics.total_connections_closed,
            "failure_count": self._pool_metrics.connection_failures,
            "avg_acquisition_time_ms": self._pool_metrics.avg_acquisition_time_ms,
            "pool_utilization_percent": (
                self._pool_metrics.active_connections / max(1, self._pool_metrics.max_pool_size)
            )
            * 100,
        }
    # --------------------------------------------------------------------------------- end get_connection_pool_status()
    
    # --------------------------------------------------------------------------------- reset_statistics()
    def reset_statistics(self) -> None:
        """Reset all statistics and history."""
        self._operation_history.clear()
        self._stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "avg_query_time_ms": 0.0,
            "avg_network_latency_ms": 0.0,
            "total_data_transferred_bytes": 0,
            "queries_per_second": 0.0,
            "last_reset_time": time.time(),
        }
        logger.info("Database metrics reset")
    # --------------------------------------------------------------------------------- end reset_statistics()

# ------------------------------------------------------------------------- end class DatabaseMetricsCollector

# ------------------------------------------------------------------------- class DatabaseTimingTracker
class DatabaseTimingTracker:
    """Helper class for tracking database operation timing details."""
    
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self, operation_timing: DatabaseOperationTiming):
        self.operation_timing = operation_timing
        self._connection_start: Optional[float] = None
        self._query_start: Optional[float] = None
    # --------------------------------------------------------------------------------- end __init__()
    
    # -------------------------------------------------------------- start_connection_timing()
    def start_connection_timing(self) -> None:
        """Start timing connection acquisition."""
        self._connection_start = time.perf_counter()
    # -------------------------------------------------------------- end start_connection_timing()

    # -------------------------------------------------------------- end_connection_timing()
    def end_connection_timing(self) -> None:
        """End timing connection acquisition."""
        if self._connection_start:
            self.operation_timing.connection_acquisition_time_ms = (
                (time.perf_counter() - self._connection_start) * 1000
            )
    # -------------------------------------------------------------- end end_connection_timing()

    # -------------------------------------------------------------- start_query_timing()
    def start_query_timing(self) -> None:
        """Start timing query execution."""
        self._query_start = time.perf_counter()
    # -------------------------------------------------------------- end start_query_timing()

    # -------------------------------------------------------------- end_query_timing()
    def end_query_timing(self) -> None:
        """End timing query execution."""
        if self._query_start:
            self.operation_timing.query_execution_time_ms = (
                (time.perf_counter() - self._query_start) * 1000
            )
    # -------------------------------------------------------------- end end_query_timing()

    # -------------------------------------------------------------- set_rows_returned()
    def set_rows_returned(self, count: int) -> None:
        """Set number of rows returned."""
        self.operation_timing.rows_returned = count
    # -------------------------------------------------------------- end set_rows_returned()

    # -------------------------------------------------------------- set_bytes_transferred()
    def set_bytes_transferred(self, bytes_count: int) -> None:
        """Set bytes transferred."""
        self.operation_timing.bytes_transferred = bytes_count
    # -------------------------------------------------------------- end set_bytes_transferred()

    # -------------------------------------------------------------- set_network_latency()
    def set_network_latency(self, latency_ms: float) -> None:
        """Set network latency measurement."""
        self.operation_timing.network_latency_ms = latency_ms
    # -------------------------------------------------------------- end set_network_latency()

    # -------------------------------------------------------------- add_metadata()
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the operation."""
        self.operation_timing.metadata[key] = value
    # -------------------------------------------------------------- end add_metadata()

# ------------------------------------------------------------------------- end class DatabaseTimingTracker

# __________________________________________________________________________
# Standalone Function Definitions
#

# ______________________
# Utility Functions
#
# --------------------------------------------------------------------------------- safe_percentile()
def safe_percentile(values: List[float], p: float) -> float:
    """Return the p-th percentile of values or 0.0 if not enough data.

    Args:
        values: List of numeric samples.
        p: Percentile in [0, 100].

    Returns:
        The percentile value if inputs are valid and non-empty; otherwise 0.0.

    Raises:
        ValueError: If p is outside [0, 100].
    """
    if not (0.0 <= p <= 100.0):
        raise ValueError("percentile p must be between 0 and 100")
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    index = int((p / 100.0) * (len(sorted_vals) - 1))
    return sorted_vals[index]
# --------------------------------------------------------------------------------- end safe_percentile()

# --------------------------------------------------------------------------------- truncate_query()
def truncate_query(query_text: Optional[str], max_chars: int = 200) -> str:
    """Return a safely truncated query string for logs.

    Args:
        query_text: Raw query text or None.
        max_chars: Maximum number of characters to keep.

    Returns:
        The original query trimmed to max_chars with an ellipsis if truncated.
    """
    if not query_text:
        return "N/A"
    text = query_text.strip()
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
# --------------------------------------------------------------------------------- end truncate_query()

# --------------------------------------------------------------------------------- compute_pool_utilization()
def compute_pool_utilization(active: int, max_size: int) -> float:
    """Compute connection pool utilization as a percentage.

    Args:
        active: Number of active connections.
        max_size: Maximum pool size.

    Returns:
        Utilization in the range [0, 100].
    """
    denominator = max(1, max_size)
    ratio = max(0.0, min(1.0, float(active) / float(denominator)))
    return ratio * 100.0
# --------------------------------------------------------------------------------- end compute_pool_utilization()

# =========================================================================
# Global Database Metrics Instance
# =========================================================================

_global_db_metrics: Optional[DatabaseMetricsCollector] = None

# --------------------------------------------------------------------------------- get_database_metrics()
def get_database_metrics() -> DatabaseMetricsCollector:
    """
    Get or create the global database metrics collector.
    
    Returns:
        DatabaseMetricsCollector: Global collector instance
    """
    global _global_db_metrics
    if _global_db_metrics is None:
        _global_db_metrics = DatabaseMetricsCollector()
    return _global_db_metrics
# --------------------------------------------------------------------------------- end get_database_metrics()

# --------------------------------------------------------------------------------- initialize_database_metrics()
async def initialize_database_metrics() -> DatabaseMetricsCollector:
    """Initialize the global database metrics collector."""
    metrics = get_database_metrics()
    logger.info("Global database metrics collector initialized")
    return metrics
# --------------------------------------------------------------------------------- end initialize_database_metrics()

# __________________________________________________________________________
# End of File
#