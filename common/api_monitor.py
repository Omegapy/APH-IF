"""
API Monitoring System for APH-IF

This module provides comprehensive monitoring for API calls to LLM providers (OpenAI)
and Neo4j database operations. It tracks timing, errors, token usage, and provides
detailed logging for debugging and performance analysis.

Features:
- Request/response logging with configurable detail levels
- Performance metrics (timing, token usage, query counts)
- Error tracking and categorization
- Thread-safe operation for concurrent processing
- Configurable output formats (JSON, structured logs)
- Integration with existing logging infrastructure

Usage:
    from common.api_monitor import APIMonitor, LLMMonitor, Neo4jMonitor
    
    # Initialize monitors
    llm_monitor = LLMMonitor()
    neo4j_monitor = Neo4jMonitor()
    
    # Monitor LLM calls
    with llm_monitor.track_request("entity_extraction") as tracker:
        response = openai_client.chat.completions.create(...)
        tracker.record_response(response)
    
    # Monitor Neo4j operations
    with neo4j_monitor.track_query("create_entity") as tracker:
        result = session.run(query, parameters)
        tracker.record_result(result)
"""

import json
import time
import threading
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, ContextManager
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIType(Enum):
    """API types for monitoring."""
    LLM = "llm"
    NEO4J = "neo4j"
    GENERAL = "general"


class LogLevel(Enum):
    """Monitoring log levels."""
    MINIMAL = "minimal"      # Only errors and basic metrics
    STANDARD = "standard"    # Standard request/response info
    DETAILED = "detailed"    # Full request/response bodies
    DEBUG = "debug"         # Everything including internal state


@dataclass
class APICallMetrics:
    """Metrics for a single API call."""
    call_id: str
    api_type: APIType
    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    request_size: Optional[int] = None
    response_size: Optional[int] = None
    
    # LLM-specific metrics
    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Neo4j-specific metrics
    query: Optional[str] = None
    parameters: Optional[Dict] = None
    records_returned: Optional[int] = None
    records_affected: Optional[int] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        # Convert enum to string
        data['api_type'] = self.api_type.value
        return data


class APICallTracker:
    """Context manager for tracking individual API calls."""
    
    def __init__(self, monitor: 'APIMonitor', call_id: str, api_type: APIType, 
                 operation: str, log_level: LogLevel = LogLevel.STANDARD):
        self.monitor = monitor
        self.call_id = call_id
        self.metrics = APICallMetrics(
            call_id=call_id,
            api_type=api_type,
            operation=operation,
            start_time=datetime.now(timezone.utc)
        )
        self.log_level = log_level
        
    def __enter__(self):
        """Start tracking the API call."""
        if self.log_level.value in ['standard', 'detailed', 'debug']:
            logger.info(f"[{self.api_type.value.upper()}] Starting {self.operation} (ID: {self.call_id})")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finish tracking and record metrics."""
        self.metrics.end_time = datetime.now(timezone.utc)
        self.metrics.duration_ms = (
            (self.metrics.end_time - self.metrics.start_time).total_seconds() * 1000
        )
        
        if exc_type is not None:
            self.metrics.success = False
            self.metrics.error_type = exc_type.__name__
            self.metrics.error_message = str(exc_val)
            
            if self.log_level.value in ['minimal', 'standard', 'detailed', 'debug']:
                logger.error(f"[{self.api_type.value.upper()}] {self.operation} failed "
                           f"(ID: {self.call_id}, Duration: {self.metrics.duration_ms:.1f}ms): "
                           f"{self.metrics.error_type}: {self.metrics.error_message}")
        else:
            if self.log_level.value in ['standard', 'detailed', 'debug']:
                logger.info(f"[{self.api_type.value.upper()}] {self.operation} completed "
                          f"(ID: {self.call_id}, Duration: {self.metrics.duration_ms:.1f}ms)")
        
        # Record the metrics
        self.monitor._record_metrics(self.metrics)
        
    def record_request(self, request_data: Any, model: str = None):
        """Record request details."""
        if isinstance(request_data, str):
            self.metrics.request_size = len(request_data.encode('utf-8'))
        elif hasattr(request_data, '__len__'):
            self.metrics.request_size = len(str(request_data).encode('utf-8'))
            
        if model:
            self.metrics.model = model
            
        if self.log_level == LogLevel.DETAILED:
            logger.debug(f"[{self.api_type.value.upper()}] Request (ID: {self.call_id}): "
                        f"Model: {model}, Size: {self.metrics.request_size} bytes")
        elif self.log_level == LogLevel.DEBUG:
            logger.debug(f"[{self.api_type.value.upper()}] Full Request (ID: {self.call_id}): "
                        f"{json.dumps(request_data, indent=2, default=str)}")
    
    def record_response(self, response_data: Any):
        """Record response details."""
        if hasattr(response_data, 'usage'):
            # OpenAI response with usage info
            usage = response_data.usage
            self.metrics.prompt_tokens = getattr(usage, 'prompt_tokens', None)
            self.metrics.completion_tokens = getattr(usage, 'completion_tokens', None)
            self.metrics.total_tokens = getattr(usage, 'total_tokens', None)
            
        if isinstance(response_data, str):
            self.metrics.response_size = len(response_data.encode('utf-8'))
        elif hasattr(response_data, '__len__'):
            self.metrics.response_size = len(str(response_data).encode('utf-8'))
            
        if self.log_level == LogLevel.DETAILED:
            logger.debug(f"[{self.api_type.value.upper()}] Response (ID: {self.call_id}): "
                        f"Tokens: {self.metrics.total_tokens}, Size: {self.metrics.response_size} bytes")
        elif self.log_level == LogLevel.DEBUG:
            logger.debug(f"[{self.api_type.value.upper()}] Full Response (ID: {self.call_id}): "
                        f"{json.dumps(response_data, indent=2, default=str)}")
    
    def record_query(self, query: str, parameters: Dict = None):
        """Record Neo4j query details."""
        self.metrics.query = query
        self.metrics.parameters = parameters or {}
        
        if self.log_level == LogLevel.DETAILED:
            logger.debug(f"[NEO4J] Query (ID: {self.call_id}): {query[:100]}...")
        elif self.log_level == LogLevel.DEBUG:
            logger.debug(f"[NEO4J] Full Query (ID: {self.call_id}): {query}")
            if parameters:
                logger.debug(f"[NEO4J] Parameters (ID: {self.call_id}): {parameters}")
    
    def record_result(self, result: Any):
        """Record Neo4j result details."""
        if hasattr(result, 'consume'):
            # Neo4j Result object
            summary = result.consume()
            self.metrics.records_returned = len(list(result))
            if hasattr(summary, 'counters'):
                counters = summary.counters
                self.metrics.records_affected = (
                    counters.nodes_created + counters.nodes_deleted +
                    counters.relationships_created + counters.relationships_deleted
                )
        elif isinstance(result, list):
            self.metrics.records_returned = len(result)
            
        if self.log_level.value in ['standard', 'detailed', 'debug']:
            logger.debug(f"[NEO4J] Result (ID: {self.call_id}): "
                        f"Records: {self.metrics.records_returned}, "
                        f"Affected: {self.metrics.records_affected}")


class APIMonitor:
    """Base API monitoring class."""
    
    def __init__(self, log_level: LogLevel = LogLevel.STANDARD, 
                 output_file: Optional[Path] = None):
        self.log_level = log_level
        self.output_file = output_file
        self.metrics_history: List[APICallMetrics] = []
        self._lock = threading.Lock()
        self._call_counter = 0
        
        # Create output directory if specified
        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _generate_call_id(self) -> str:
        """Generate unique call ID."""
        with self._lock:
            self._call_counter += 1
            timestamp = int(time.time() * 1000)
            return f"{timestamp}_{self._call_counter:06d}"
    
    def _record_metrics(self, metrics: APICallMetrics):
        """Record metrics to history and optionally to file."""
        with self._lock:
            self.metrics_history.append(metrics)
            
            # Write to file if configured
            if self.output_file:
                try:
                    with open(self.output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(metrics.to_dict()) + '\n')
                except Exception as e:
                    logger.error(f"Failed to write metrics to file: {e}")
    
    @contextmanager
    def track_call(self, api_type: APIType, operation: str) -> ContextManager[APICallTracker]:
        """Create a context manager for tracking an API call."""
        call_id = self._generate_call_id()
        tracker = APICallTracker(self, call_id, api_type, operation, self.log_level)
        yield tracker
    
    def get_metrics_summary(self, api_type: Optional[APIType] = None, 
                          operation: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for recorded metrics."""
        with self._lock:
            filtered_metrics = self.metrics_history
            
            if api_type:
                filtered_metrics = [m for m in filtered_metrics if m.api_type == api_type]
            if operation:
                filtered_metrics = [m for m in filtered_metrics if m.operation == operation]
            
            if not filtered_metrics:
                return {"total_calls": 0}
            
            successful_calls = [m for m in filtered_metrics if m.success]
            failed_calls = [m for m in filtered_metrics if not m.success]
            
            durations = [m.duration_ms for m in filtered_metrics if m.duration_ms is not None]
            
            summary = {
                "total_calls": len(filtered_metrics),
                "successful_calls": len(successful_calls),
                "failed_calls": len(failed_calls),
                "success_rate": len(successful_calls) / len(filtered_metrics) if filtered_metrics else 0,
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                "min_duration_ms": min(durations) if durations else 0,
                "max_duration_ms": max(durations) if durations else 0,
            }
            
            # Add LLM-specific metrics
            llm_metrics = [m for m in filtered_metrics if m.api_type == APIType.LLM]
            if llm_metrics:
                total_tokens = [m.total_tokens for m in llm_metrics if m.total_tokens is not None]
                summary.update({
                    "total_tokens_used": sum(total_tokens),
                    "avg_tokens_per_call": sum(total_tokens) / len(total_tokens) if total_tokens else 0,
                })
            
            # Add Neo4j-specific metrics
            neo4j_metrics = [m for m in filtered_metrics if m.api_type == APIType.NEO4J]
            if neo4j_metrics:
                records_returned = [m.records_returned for m in neo4j_metrics if m.records_returned is not None]
                records_affected = [m.records_affected for m in neo4j_metrics if m.records_affected is not None]
                summary.update({
                    "total_records_returned": sum(records_returned),
                    "total_records_affected": sum(records_affected),
                    "avg_records_per_query": sum(records_returned) / len(records_returned) if records_returned else 0,
                })
            
            return summary
    
    def clear_metrics(self):
        """Clear all recorded metrics."""
        with self._lock:
            self.metrics_history.clear()
    
    def export_metrics(self, output_path: Path, format: str = "json"):
        """Export metrics to file."""
        with self._lock:
            if format.lower() == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump([m.to_dict() for m in self.metrics_history], f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")


class LLMMonitor(APIMonitor):
    """Specialized monitor for LLM API calls."""

    def __init__(self, log_level: LogLevel = LogLevel.STANDARD,
                 output_file: Optional[Path] = None):
        super().__init__(log_level, output_file)
        self.api_type = APIType.LLM

    @contextmanager
    def track_request(self, operation: str) -> ContextManager[APICallTracker]:
        """Track an LLM API request."""
        with self.track_call(APIType.LLM, operation) as tracker:
            yield tracker

    def track_openai_completion(self, operation: str, messages: List[Dict],
                               model: str, **kwargs) -> ContextManager[APICallTracker]:
        """Track OpenAI chat completion with automatic request recording."""
        @contextmanager
        def _tracker():
            with self.track_request(operation) as tracker:
                request_data = {
                    "messages": messages,
                    "model": model,
                    **kwargs
                }
                tracker.record_request(request_data, model)
                yield tracker
        return _tracker()


class Neo4jMonitor(APIMonitor):
    """Specialized monitor for Neo4j database operations."""

    def __init__(self, log_level: LogLevel = LogLevel.STANDARD,
                 output_file: Optional[Path] = None):
        super().__init__(log_level, output_file)
        self.api_type = APIType.NEO4J

    @contextmanager
    def track_query(self, operation: str) -> ContextManager[APICallTracker]:
        """Track a Neo4j query operation."""
        with self.track_call(APIType.NEO4J, operation) as tracker:
            yield tracker

    def track_cypher_query(self, operation: str, query: str,
                          parameters: Optional[Dict] = None) -> ContextManager[APICallTracker]:
        """Track Cypher query with automatic query recording."""
        @contextmanager
        def _tracker():
            with self.track_query(operation) as tracker:
                tracker.record_query(query, parameters)
                yield tracker
        return _tracker()


# Global monitor instances for easy access
_global_llm_monitor: Optional[LLMMonitor] = None
_global_neo4j_monitor: Optional[Neo4jMonitor] = None
_monitor_lock = threading.Lock()


def get_llm_monitor(log_level: LogLevel = LogLevel.STANDARD,
                   output_file: Optional[Path] = None) -> LLMMonitor:
    """Get or create global LLM monitor instance."""
    global _global_llm_monitor
    with _monitor_lock:
        if _global_llm_monitor is None:
            _global_llm_monitor = LLMMonitor(log_level, output_file)
        return _global_llm_monitor


def get_neo4j_monitor(log_level: LogLevel = LogLevel.STANDARD,
                     output_file: Optional[Path] = None) -> Neo4jMonitor:
    """Get or create global Neo4j monitor instance."""
    global _global_neo4j_monitor
    with _monitor_lock:
        if _global_neo4j_monitor is None:
            _global_neo4j_monitor = Neo4jMonitor(log_level, output_file)
        return _global_neo4j_monitor


def configure_monitoring(log_level: LogLevel = LogLevel.STANDARD,
                        output_dir: Optional[Path] = None) -> tuple[LLMMonitor, Neo4jMonitor]:
    """Configure global monitoring with specified settings."""
    llm_output = None
    neo4j_output = None

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        llm_output = output_dir / "llm_api_calls.jsonl"
        neo4j_output = output_dir / "neo4j_operations.jsonl"

    global _global_llm_monitor, _global_neo4j_monitor
    with _monitor_lock:
        _global_llm_monitor = LLMMonitor(log_level, llm_output)
        _global_neo4j_monitor = Neo4jMonitor(log_level, neo4j_output)

    return _global_llm_monitor, _global_neo4j_monitor


def get_monitoring_summary() -> Dict[str, Any]:
    """Get comprehensive monitoring summary for all APIs."""
    summary = {}

    if _global_llm_monitor:
        summary["llm"] = _global_llm_monitor.get_metrics_summary(APIType.LLM)

    if _global_neo4j_monitor:
        summary["neo4j"] = _global_neo4j_monitor.get_metrics_summary(APIType.NEO4J)

    return summary
