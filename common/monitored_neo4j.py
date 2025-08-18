"""
Monitored Neo4j Client

This module provides wrappers around Neo4j database operations that automatically
monitor all queries, track timing, result counts, and errors.

Usage:
    from common.monitored_neo4j import MonitoredNeo4jGraph, MonitoredDriver
    
    # Create monitored graph (for LangChain)
    graph = MonitoredNeo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )
    
    # Create monitored driver (for direct Neo4j operations)
    driver = MonitoredDriver("bolt://localhost:7687", auth=("neo4j", "password"))
    
    # Use exactly like regular Neo4j - monitoring is automatic
    result = graph.query("MATCH (n) RETURN count(n) as total")
    
    # Get monitoring statistics
    stats = graph.get_monitoring_stats()
    print(f"Total queries: {stats['total_calls']}")
"""

import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

try:
    from neo4j import GraphDatabase, Driver, Session, Result
    from neo4j.exceptions import Neo4jError
except ImportError:
    raise ImportError("Neo4j library not installed. Install with: pip install neo4j")

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    Neo4jGraph = None

from .api_monitor import get_neo4j_monitor, LogLevel, APICallTracker


class MonitoredSession:
    """Monitored wrapper for Neo4j session."""
    
    def __init__(self, original_session: Session, monitor):
        self._original = original_session
        self._monitor = monitor
    
    def run(self, query: str, parameters: Optional[Dict] = None, **kwargs) -> Result:
        """Run query with monitoring."""
        operation = self._extract_operation_from_query(query)
        
        with self._monitor.track_cypher_query(operation, query, parameters) as tracker:
            try:
                result = self._original.run(query, parameters, **kwargs)
                
                # Convert to list to get record count and consume result
                records = list(result)
                tracker.metrics.records_returned = len(records)
                
                # Create a new result-like object that can be consumed
                class MonitoredResult:
                    def __init__(self, records, original_result):
                        self.records = records
                        self._original = original_result
                        self._consumed = False
                    
                    def __iter__(self):
                        return iter(self.records)
                    
                    def __len__(self):
                        return len(self.records)
                    
                    def consume(self):
                        if not self._consumed:
                            summary = self._original.consume()
                            self._consumed = True
                            return summary
                        return None
                    
                    def data(self):
                        return [record.data() for record in self.records]
                    
                    def values(self):
                        return [record.values() for record in self.records]
                
                monitored_result = MonitoredResult(records, result)
                tracker.record_result(monitored_result)
                return monitored_result
                
            except Exception as e:
                # Error will be automatically recorded by the tracker context manager
                raise
    
    def _extract_operation_from_query(self, query: str) -> str:
        """Extract operation type from Cypher query."""
        query_upper = query.strip().upper()
        
        if query_upper.startswith('MATCH'):
            return 'match_query'
        elif query_upper.startswith('CREATE'):
            return 'create_query'
        elif query_upper.startswith('MERGE'):
            return 'merge_query'
        elif query_upper.startswith('DELETE'):
            return 'delete_query'
        elif query_upper.startswith('SET'):
            return 'set_query'
        elif query_upper.startswith('REMOVE'):
            return 'remove_query'
        elif query_upper.startswith('CALL'):
            return 'procedure_call'
        elif query_upper.startswith('SHOW'):
            return 'show_query'
        else:
            return 'unknown_query'
    
    def close(self):
        """Close the session."""
        return self._original.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # Pass through other session methods
    def __getattr__(self, name):
        return getattr(self._original, name)


class MonitoredDriver:
    """Monitored wrapper for Neo4j driver."""
    
    def __init__(self, uri: str, *, auth=None, log_level: LogLevel = LogLevel.STANDARD,
                 output_file: Optional[Path] = None, **kwargs):
        """
        Initialize monitored Neo4j driver.
        
        Args:
            uri: Neo4j connection URI
            auth: Authentication tuple (username, password)
            log_level: Monitoring detail level
            output_file: Optional file to write monitoring data
            **kwargs: Additional arguments passed to Neo4j driver
        """
        self._driver = GraphDatabase.driver(uri, auth=auth, **kwargs)
        self._monitor = get_neo4j_monitor(log_level, output_file)
        self._uri = uri
    
    def session(self, **kwargs) -> MonitoredSession:
        """Create a monitored session."""
        original_session = self._driver.session(**kwargs)
        return MonitoredSession(original_session, self._monitor)
    
    def verify_connectivity(self):
        """Verify database connectivity with monitoring."""
        with self._monitor.track_query("connectivity_check") as tracker:
            try:
                result = self._driver.verify_connectivity()
                tracker.record_result({"connectivity": "verified"})
                return result
            except Exception as e:
                # Error will be automatically recorded by the tracker context manager
                raise
    
    def close(self):
        """Close the driver."""
        return self._driver.close()
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        return self._monitor.get_metrics_summary()
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        return self._monitor.get_metrics_summary(operation=operation)
    
    def export_monitoring_data(self, output_path: Path, format: str = "json"):
        """Export monitoring data to file."""
        self._monitor.export_metrics(output_path, format)
    
    def clear_monitoring_data(self):
        """Clear all monitoring data."""
        self._monitor.clear_metrics()
    
    # Pass through other driver methods
    def __getattr__(self, name):
        return getattr(self._driver, name)


class MonitoredNeo4jGraph:
    """
    Monitored wrapper for LangChain Neo4jGraph that automatically tracks all queries.
    
    This class provides the same interface as Neo4jGraph but adds comprehensive
    monitoring of all database interactions.
    """
    
    def __init__(self, url: str, username: str, password: str,
                 log_level: LogLevel = LogLevel.STANDARD,
                 output_file: Optional[Path] = None,
                 **kwargs):
        """
        Initialize monitored Neo4j graph.
        
        Args:
            url: Neo4j connection URL
            username: Neo4j username
            password: Neo4j password
            log_level: Monitoring detail level
            output_file: Optional file to write monitoring data
            **kwargs: Additional arguments passed to Neo4jGraph
        """
        if Neo4jGraph is None:
            raise ImportError("langchain-neo4j not installed. Install with: pip install langchain-neo4j")
        
        self._graph = Neo4jGraph(url=url, username=username, password=password, **kwargs)
        self._monitor = get_neo4j_monitor(log_level, output_file)
        self._url = url
        self._username = username
    
    def query(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Execute query with monitoring."""
        operation = self._extract_operation_from_query(query)
        
        with self._monitor.track_cypher_query(operation, query, params) as tracker:
            try:
                result = self._graph.query(query, params)
                
                # Record result information
                if isinstance(result, list):
                    tracker.metrics.records_returned = len(result)
                
                tracker.record_result(result)
                return result
                
            except Exception as e:
                # Error will be automatically recorded by the tracker context manager
                raise
    
    def _extract_operation_from_query(self, query: str) -> str:
        """Extract operation type from Cypher query."""
        query_upper = query.strip().upper()
        
        if query_upper.startswith('MATCH'):
            return 'match_query'
        elif query_upper.startswith('CREATE'):
            return 'create_query'
        elif query_upper.startswith('MERGE'):
            return 'merge_query'
        elif query_upper.startswith('DELETE'):
            return 'delete_query'
        elif query_upper.startswith('SET'):
            return 'set_query'
        elif query_upper.startswith('REMOVE'):
            return 'remove_query'
        elif query_upper.startswith('CALL'):
            return 'procedure_call'
        elif query_upper.startswith('SHOW'):
            return 'show_query'
        else:
            return 'unknown_query'
    
    def refresh_schema(self):
        """Refresh schema with monitoring."""
        with self._monitor.track_query("refresh_schema") as tracker:
            try:
                result = self._graph.refresh_schema()
                tracker.record_result({"schema_refreshed": True})
                return result
            except Exception as e:
                # Error will be automatically recorded by the tracker context manager
                raise
    
    def get_schema(self) -> str:
        """Get schema with monitoring."""
        with self._monitor.track_query("get_schema") as tracker:
            try:
                schema = self._graph.get_schema()
                tracker.record_result({"schema_length": len(schema)})
                return schema
            except Exception as e:
                # Error will be automatically recorded by the tracker context manager
                raise
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        return self._monitor.get_metrics_summary()
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        return self._monitor.get_metrics_summary(operation=operation)
    
    def export_monitoring_data(self, output_path: Path, format: str = "json"):
        """Export monitoring data to file."""
        self._monitor.export_metrics(output_path, format)
    
    def clear_monitoring_data(self):
        """Clear all monitoring data."""
        self._monitor.clear_metrics()
    
    def set_log_level(self, log_level: LogLevel):
        """Change monitoring log level."""
        self._monitor.log_level = log_level
    
    # Pass through other graph methods
    def __getattr__(self, name):
        return getattr(self._graph, name)


def create_monitored_driver(uri: str, auth=None, 
                          log_level: LogLevel = LogLevel.STANDARD,
                          output_dir: Optional[Path] = None) -> MonitoredDriver:
    """
    Create a monitored Neo4j driver with optional output directory.
    
    Args:
        uri: Neo4j connection URI
        auth: Authentication tuple (username, password)
        log_level: Monitoring detail level
        output_dir: Directory to write monitoring files
    
    Returns:
        MonitoredDriver instance
    """
    output_file = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "neo4j_operations.jsonl"
    
    return MonitoredDriver(uri, auth=auth, log_level=log_level, output_file=output_file)


def create_monitored_graph(url: str, username: str, password: str,
                         log_level: LogLevel = LogLevel.STANDARD,
                         output_dir: Optional[Path] = None) -> MonitoredNeo4jGraph:
    """
    Create a monitored Neo4j graph with optional output directory.
    
    Args:
        url: Neo4j connection URL
        username: Neo4j username
        password: Neo4j password
        log_level: Monitoring detail level
        output_dir: Directory to write monitoring files
    
    Returns:
        MonitoredNeo4jGraph instance
    """
    output_file = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "neo4j_operations.jsonl"
    
    return MonitoredNeo4jGraph(url, username, password, log_level=log_level, output_file=output_file)
