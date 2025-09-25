# -------------------------------------------------------------------------
# File: core/database.py
# Author: Alexander Ricciardi
# Date: 2025-09-24
# [File Path] backend/app/core/database.py
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
#   Provides a resilient Neo4j database interface with connection pooling,
#   retry logic, and health monitoring utilities for the APH-IF backend.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Dataclass: DatabaseConfig
# - Dataclass: DatabaseMetrics
# - Class: APHIFDatabase
# - Exceptions: APHIFDatabaseError, DatabaseConnectionError, DatabaseQueryError
# - Functions: get_database, get_database_for_semantic_search,
#              get_database_for_traversal_search, database_health_check,
#              reset_database_connection
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: logging, threading.Lock, time, dataclasses.dataclass,
#                     typing (cast, LiteralString, Optional, Dict, Any, List)
# - Third-Party: neo4j.GraphDatabase, neo4j.records.Record, neo4j.Driver,
#                neo4j.exceptions
# - Local Project Modules:
#   - backend.app.core.config.settings
# --- Requirements ---
# - Python 3.12+
# -------------------------------------------------------------------------

# --- Usage / Integration ---
#   Imported by backend services to obtain a singleton database client and
#   execute safe Neo4j operations with monitoring support.
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Neo4j database access layer with resilience features for the APH-IF backend.

Exposes configuration dataclasses, a singleton-friendly database client, and
health monitoring helpers used by retrieval and monitoring subsystems.
"""

# __________________________________________________________________________
# Imports
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, cast

from neo4j import Driver, GraphDatabase, Record
from neo4j.exceptions import DatabaseError, ServiceUnavailable, TransientError

from .config import settings

# __________________________________________________________________________
# Global Constants / Variables
#

logger = logging.getLogger(__name__)

# Global database instance and thread lock for singleton pattern
_database_instance: Optional['APHIFDatabase'] = None
_database_lock = Lock()

# ____________________________________________________________________________
# Class Definitions

# =========================================================================
# Configuration and Metrics Classes
# =========================================================================

# ------------------------------------------------------------------------- class DatabaseConfig
# TODO: Consider further refinement of DatabaseConfig for stricter validation in future updates.
@dataclass
class DatabaseConfig:
    """Configuration options for establishing Neo4j connections.

    Attributes:
        pool_size: Initial number of pooled connections to maintain.
        max_connection_lifetime: Maximum lifetime for a connection in seconds.
        max_connection_pool_size: Upper bound on pooled connections.
        connection_timeout: Seconds to wait when acquiring a connection.
        max_retry_attempts: Number of retries for transient failures.
        health_check_timeout: Timeout for health check operations in seconds.
        connection_acquisition_timeout: Seconds to wait for connection acquisition.
        enable_connection_warmup: Flag enabling proactive connection warmup.
        warmup_connections: Number of connections to establish during warmup.
        health_check_interval: Interval between periodic health checks in seconds.
    """

    pool_size: int = 25  # Increased from 10 for better concurrency
    max_connection_lifetime: int = 300
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0  # Reduced from 240.0
    max_retry_attempts: int = 3
    health_check_timeout: float = 30.0  # Reduced from 240.0
    connection_acquisition_timeout: float = 5.0  # New parameter
    enable_connection_warmup: bool = True  # New parameter
    warmup_connections: int = 10  # New parameter
    health_check_interval: int = 30  # New parameter (seconds)
# ------------------------------------------------------------------------- end class DatabaseConfig

# ------------------------------------------------------------------------- class DatabaseMetrics
@dataclass(slots=True, kw_only=True)
class DatabaseMetrics:
    """Collect metrics for database operations and connection usage.

    Attributes:
        total_queries: Count of queries attempted.
        successful_queries: Count of queries completed successfully.
        failed_queries: Count of queries raising errors.
        average_response_time: Average query time in seconds.
        total_response_time: Aggregate response time across queries.
        query_types: Mapping of query type names to counts.
        connection_pool_usage: Latest snapshot of pool usage metrics.
        query_time_histogram: List of individual query response times.
        slow_queries: Rolling list of slow query metadata dictionaries.
    """

    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    average_response_time: float = 0.0
    total_response_time: float = 0.0
    query_types: Dict[str, int] = None
    connection_pool_usage: Dict[str, Any] = None
    query_time_histogram: List[float] = None
    slow_queries: List[Dict[str, Any]] = None

    # ______________________
    # Post-Initialization (validation + derived fields)
    #
    # --------------------------------------------------------------------------------- __post_init__()
    def __post_init__(self) -> None:
        """Initialize default metric containers if they were not provided."""
        if self.query_types is None:
            self.query_types = {}
        if self.connection_pool_usage is None:
            self.connection_pool_usage = {}
        if self.query_time_histogram is None:
            self.query_time_histogram = []
        if self.slow_queries is None:
            self.slow_queries = []
    # --------------------------------------------------------------------------------- end __post_init__()

    # ______________________
    # Setters / Mutators
    #
    # --------------------------------------------------------------------------------- update_response_time()
    def update_response_time(self, response_time: float, query_type: str = "unknown") -> None:
        """Update response metrics after executing a query.

        Args:
            response_time: Duration of the query execution in seconds.
            query_type: Descriptive label for the type of query executed.
        """
        self.total_response_time += response_time
        if self.total_queries > 0:
            self.average_response_time = self.total_response_time / self.total_queries

        # Track query type
        self.query_types[query_type] = self.query_types.get(query_type, 0) + 1

        # Add to histogram
        self.query_time_histogram.append(response_time)

        # Track slow queries (> 1 second)
        if response_time > 1.0:
            self.slow_queries.append({
                "query_type": query_type,
                "response_time": response_time,
                "timestamp": time.time(),
            })
            # Keep only last 100 slow queries
            if len(self.slow_queries) > 100:
                self.slow_queries = self.slow_queries[-100:]
    # --------------------------------------------------------------------------------- end update_response_time()

# ------------------------------------------------------------------------- end class DatabaseMetrics

# =========================================================================
# Exception Classes
# =========================================================================

# ------------------------------------------------------------------------- class APHIFDatabaseError
class APHIFDatabaseError(Exception):
    """Base exception for APH-IF database errors."""

# ------------------------------------------------------------------------- end class APHIFDatabaseError

# ------------------------------------------------------------------------- class DatabaseConnectionError
class DatabaseConnectionError(APHIFDatabaseError):
    """Exception for database connection failures."""

# ------------------------------------------------------------------------- end class DatabaseConnectionError

# ------------------------------------------------------------------------- class DatabaseQueryError
class DatabaseQueryError(APHIFDatabaseError):
    """Exception for database query failures."""

# ------------------------------------------------------------------------- end class DatabaseQueryError

# =========================================================================
# Enhanced Database Class
# =========================================================================

# ------------------------------------------------------------------------- class APHIFDatabase
class APHIFDatabase:
    """Enhanced Neo4j database connection with resilience features.

    Attributes:
        config: Connection configuration parameters.
        metrics: Metrics tracker for database operations.
        _driver: Lazily created Neo4j driver instance.
        _lock: Synchronization primitive for thread-safe operations.
        _is_connected: Flag indicating whether the driver is active.
        _health_check_task: Async task handle for periodic health checks.
        _last_health_check: Timestamp of the most recent health check.
    """

    # ______________________
    # Constructor
    #
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self, config: Optional[DatabaseConfig] = None) -> None:
        """Initialize the APH-IF database instance."""
        self.config = config or DatabaseConfig()
        self.metrics = DatabaseMetrics()
        self._driver: Optional[Driver] = None
        self._lock = Lock()
        self._is_connected = False
        self._health_check_task = None
        self._last_health_check = 0

        logger.info("APH-IF Neo4j database initialized")
    # --------------------------------------------------------------------------------- end __init__()

    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #
    # --------------------------------------------------------------------------------- _create_driver()
    def _create_driver(self) -> Driver:
        """Create Neo4j driver with enhanced configuration."""
        if not settings.neo4j_uri:
            raise DatabaseConnectionError("Neo4j URI not configured")

        if not settings.neo4j_username or not settings.neo4j_password:
            raise DatabaseConnectionError("Neo4j credentials not configured")

        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
            max_connection_lifetime=self.config.max_connection_lifetime,
            max_connection_pool_size=self.config.max_connection_pool_size,
            connection_timeout=self.config.connection_timeout,
        )

        logger.info(f"Neo4j driver created for {settings.get_neo4j_mode_name()}")
        return driver
    # --------------------------------------------------------------------------------- end _create_driver()

    # --------------------------------------------------------------------------------- connect()
    def connect(self) -> Driver:
        """Establish database connection."""
        try:
            with self._lock:
                if self._driver is None:
                    self._driver = self._create_driver()
                    self._is_connected = True
                    logger.info("Neo4j database connection established")
                return self._driver

        except Exception as exc:
            logger.error(f"Failed to connect to Neo4j database: {exc}")
            self._is_connected = False
            raise DatabaseConnectionError(f"Connection failed: {exc}")
    # --------------------------------------------------------------------------------- end connect()

    # --------------------------------------------------------------------------------- disconnect()
    def disconnect(self) -> None:
        """Close database connection and cleanup resources."""
        with self._lock:
            if self._driver:
                try:
                    self._driver.close()
                    logger.info("Neo4j database connection closed")
                except Exception as exc:
                    logger.warning(f"Error closing database connection: {exc}")
                finally:
                    self._driver = None
                    self._is_connected = False
    # --------------------------------------------------------------------------------- end disconnect()

    # --------------------------------------------------------------------------------- execute_query()
    def execute_query(self, query: str, parameters: Optional[Dict] = None) -> List[Record]:
        """Execute a Cypher query with retry logic and metrics tracking."""
        start_time = time.time()
        parameters = parameters or {}

        # Determine query type from query string
        query_type = self._determine_query_type(query)

        retry_count = 0
        last_error = None

        while retry_count < self.config.max_retry_attempts:
            try:
                # Validate connection before use
                if not self._driver or not self.validate_connection():
                    self.connect()

                if self._driver is None:
                    raise DatabaseConnectionError("Failed to establish database connection")

                with self._driver.session(database=settings.neo4j_database) as session:
                    result = session.run(query, parameters)
                    records = list(result)

                # Update metrics with query type
                response_time = time.time() - start_time
                self.metrics.total_queries += 1
                self.metrics.successful_queries += 1
                self.metrics.update_response_time(response_time, query_type)

                logger.debug(f"Query ({query_type}) executed successfully in {response_time:.3f}s")
                return records

            except (ServiceUnavailable, TransientError) as exc:
                last_error = exc
                retry_count += 1
                if retry_count < self.config.max_retry_attempts:
                    wait_time = retry_count * 2  # Simple backoff
                    logger.warning(
                        f"Query failed, retrying in {wait_time}s (attempt {retry_count}): {exc}"
                    )
                    time.sleep(wait_time)
                else:
                    break
            except Exception as exc:
                self.metrics.total_queries += 1
                self.metrics.failed_queries += 1
                logger.error(f"Query execution failed: {exc}")
                raise DatabaseQueryError(f"Query failed: {exc}")

        # All retries exhausted
        self.metrics.total_queries += 1
        self.metrics.failed_queries += 1
        logger.error(f"Query failed after {retry_count} attempts: {last_error}")
        raise DatabaseQueryError(f"Query failed after retries: {last_error}")
    # --------------------------------------------------------------------------------- end execute_query()

    # --------------------------------------------------------------------------------- health_check()
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive database health check."""
        health_status = {
            "database": "unknown",
            "connection": "unknown",
            "response_time": None,
            "metrics": self._get_metrics(),
            "error": None,
            "environment": settings.get_neo4j_mode_name(),
        }

        start_time = time.time()

        try:
            if not self._driver:
                self.connect()

            if self._driver is None:
                raise DatabaseConnectionError("Failed to establish database connection")

            # Execute simple health check query
            with self._driver.session(database=settings.neo4j_database) as session:
                result = session.run("RETURN 1 as health_check")
                record = result.single()

                response_time = time.time() - start_time

                if record and record["health_check"] == 1:
                    health_status.update({
                        "database": "healthy",
                        "connection": "active",
                        "response_time": response_time,
                    })
                else:
                    health_status.update({
                        "database": "unhealthy",
                        "connection": "failed",
                        "error": "Health check query failed",
                    })

        except Exception as exc:
            response_time = time.time() - start_time
            health_status.update({
                "database": "unhealthy",
                "connection": "failed",
                "response_time": response_time,
                "error": str(exc),
            })
            logger.warning(f"Database health check failed: {exc}")

        return health_status
    # --------------------------------------------------------------------------------- end health_check()

    # --------------------------------------------------------------------------------- warm_up_connections()
    async def warm_up_connections(self) -> None:
        """Warm up connection pool by pre-establishing connections."""
        if not self.config.enable_connection_warmup:
            return

        logger.info(
            f"Starting connection warm-up with {self.config.warmup_connections} connections..."
        )

        try:
            if not self._driver:
                self.connect()

            for _ in range(self.config.warmup_connections):
                with self._driver.session(database=settings.neo4j_database) as session:
                    session.run("RETURN 1 as warmup")

            logger.info(
                f"Successfully warmed up {self.config.warmup_connections} connections"
            )

        except Exception as exc:
            logger.warning(f"Connection warm-up failed: {exc}")
    # --------------------------------------------------------------------------------- end warm_up_connections()

    # --------------------------------------------------------------------------------- periodic_health_check()
    async def periodic_health_check(self) -> None:
        """Perform periodic health checks on database connections."""
        import asyncio

        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                current_time = time.time()
                if current_time - self._last_health_check < self.config.health_check_interval:
                    continue

                self._last_health_check = current_time

                # Perform health check
                health_status = self.health_check()

                # Update connection pool metrics
                if self._driver:
                    self.metrics.connection_pool_usage = {
                        "status": health_status.get("connection"),
                        "response_time": health_status.get("response_time"),
                        "last_check": current_time,
                    }

                # Log if unhealthy
                if health_status.get("database") != "healthy":
                    logger.warning(f"Database health check failed: {health_status}")

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Error in periodic health check: {exc}")
    # --------------------------------------------------------------------------------- end periodic_health_check()

    # --------------------------------------------------------------------------------- validate_connection()
    def validate_connection(self) -> bool:
        """Validate current connection before use."""
        try:
            if not self._driver:
                return False

            with self._driver.session(database=settings.neo4j_database) as session:
                result = session.run("RETURN 1 as validation")
                return result.single() is not None

        except Exception:
            return False
    # --------------------------------------------------------------------------------- end validate_connection()

    # --------------------------------------------------------------------------------- get_schema_info()
    def get_schema_info(self) -> Dict[str, Any]:
        """Get comprehensive schema information from the knowledge graph."""
        schema_info = {
            "node_labels": [],
            "relationship_types": [],
            "property_keys": [],
            "constraints": [],
            "indexes": [],
        }

        try:
            labels_result = self.execute_query("CALL db.labels()")
            schema_info["node_labels"] = [record["label"] for record in labels_result]

            rel_types_result = self.execute_query("CALL db.relationshipTypes()")
            schema_info["relationship_types"] = [
                record["relationshipType"] for record in rel_types_result
            ]

            try:
                props_result = self.execute_query("CALL db.propertyKeys()")
                schema_info["property_keys"] = [
                    record["propertyKey"] for record in props_result
                ]
            except Exception as exc:
                logger.debug(f"Could not get property keys: {exc}")

            try:
                constraints_result = self.execute_query("SHOW CONSTRAINTS")
                schema_info["constraints"] = [dict(record) for record in constraints_result]
            except Exception as exc:
                logger.debug(f"Could not get constraints: {exc}")

            try:
                indexes_result = self.execute_query("SHOW INDEXES")
                schema_info["indexes"] = [dict(record) for record in indexes_result]
            except Exception as exc:
                logger.debug(f"Could not get indexes: {exc}")

        except Exception as exc:
            logger.error(f"Error getting schema info: {exc}")
            schema_info["error"] = str(exc)

        return schema_info
    # --------------------------------------------------------------------------------- end get_schema_info()

    # --------------------------------------------------------------------------------- _determine_query_type()
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query from the query string."""
        query_upper = query.upper()

        if "MATCH" in query_upper:
            if "CREATE" in query_upper or "MERGE" in query_upper:
                return "write"
            return "read"
        if "CREATE" in query_upper or "MERGE" in query_upper:
            return "write"
        if "DELETE" in query_upper or "DETACH" in query_upper:
            return "delete"
        if "CALL" in query_upper:
            if "db.labels" in query or "db.relationshipTypes" in query:
                return "schema"
            return "procedure"
        if "SHOW" in query_upper:
            return "schema"
        if "RETURN" in query_upper:
            return "simple"
        return "unknown"
    # --------------------------------------------------------------------------------- end _determine_query_type()

    # --------------------------------------------------------------------------------- _get_metrics()
    def _get_metrics(self) -> Dict[str, Any]:
        """Get enhanced database metrics."""
        success_rate = 0.0
        if self.metrics.total_queries > 0:
            success_rate = self.metrics.successful_queries / self.metrics.total_queries

        percentiles = {}
        if self.metrics.query_time_histogram:
            sorted_times = sorted(self.metrics.query_time_histogram)
            percentiles = {
                "p50": sorted_times[len(sorted_times) // 2] if sorted_times else 0,
                "p90": sorted_times[int(len(sorted_times) * 0.9)] if sorted_times else 0,
                "p99": sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0,
            }

        return {
            "total_queries": self.metrics.total_queries,
            "successful_queries": self.metrics.successful_queries,
            "failed_queries": self.metrics.failed_queries,
            "success_rate": success_rate,
            "average_response_time": self.metrics.average_response_time,
            "response_time_percentiles": percentiles,
            "query_types": self.metrics.query_types,
            "slow_queries_count": len(self.metrics.slow_queries),
            "connection_pool_usage": self.metrics.connection_pool_usage,
            "is_connected": self._is_connected,
        }
    # --------------------------------------------------------------------------------- end _get_metrics()

# ------------------------------------------------------------------------- end class APHIFDatabase

# =========================================================================
# Global Functions
# =========================================================================

# __________________________________________________________________________
# Standalone Function Definitions
#

# ______________________
# Utility Functions
#

# --------------------------------------------------------------------------------- get_database()
def get_database() -> APHIFDatabase:
    """Get or create the global database instance (singleton pattern)."""
    global _database_instance

    with _database_lock:
        if _database_instance is None:
            _database_instance = APHIFDatabase()
            logger.info("Global APH-IF database instance created")
        return _database_instance
# --------------------------------------------------------------------------------- end get_database()

# --------------------------------------------------------------------------------- get_database_for_semantic_search()
def get_database_for_semantic_search() -> APHIFDatabase:
    """
    Get database instance for semantic search operations.

    Now uses shared connection pool for better resource utilization.
    The increased pool size (25) handles concurrent operations efficiently.
    """
    return get_database()
# --------------------------------------------------------------------------------- end get_database_for_semantic_search()

# --------------------------------------------------------------------------------- get_database_for_traversal_search()
def get_database_for_traversal_search() -> APHIFDatabase:
    """
    Get database instance for traversal search operations.

    Now uses shared connection pool for better resource utilization.
    The increased pool size (25) handles concurrent operations efficiently.
    """
    return get_database()
# --------------------------------------------------------------------------------- end get_database_for_traversal_search()

# --------------------------------------------------------------------------------- database_health_check()
def database_health_check() -> Dict[str, Any]:
    """Perform database health check (standalone function)."""
    database = get_database()
    return database.health_check()
# --------------------------------------------------------------------------------- end database_health_check()

# --------------------------------------------------------------------------------- reset_database_connection()
def reset_database_connection() -> None:
    """Reset database connection (useful for testing and shutdown)."""
    global _database_instance

    with _database_lock:
        if _database_instance:
            _database_instance.disconnect()
            _database_instance = None
        logger.info("Database connection reset")
# --------------------------------------------------------------------------------- end reset_database_connection()

# __________________________________________________________________________
# Module Initialization / Main Execution Guard (if applicable)
#

# __________________________________________________________________________
# End of File
#

