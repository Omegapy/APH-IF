# -------------------------------------------------------------------------
# File: performance_monitor.py
# Author: Alexander Ricciardi
# Date: 2025-09-18
# [File Path] backend/app/monitoring/performance_monitor.py
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
#   Performance monitoring system for the APH-IF backend. Collects operation timings,
#   aggregates metrics, and produces dashboards/health data for parallel retrieval and
#   fusion stages. Provides lightweight APIs used by FastAPI routes without altering
#   request behavior.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Dataclass: PerformanceMetric
# - Dataclass: DetailedTimingBreakdown
# - Dataclass: TimingContext
# - Dataclass: SystemMetrics
# - Dataclass: AlertThreshold
# - Class: PerformanceMonitor
# - Class: OperationTracker
# - Utility Functions: safe_mean, format_duration_ms, mask_sensitive_metadata
# - Functions: get_performance_monitor, initialize_monitor, performance_health_check
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: json, logging, statistics, time, contextlib, dataclasses, datetime, typing,
#   collections
# - Local Project Modules: Integrated by app.main and monitoring endpoints
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Imported by FastAPI app routes in app.main to power /performance and /timing endpoints.
# Retrieve the monitor with get_performance_monitor() and use track_operation(...) around
# parallel retrieval and fusion to record metrics.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Performance monitoring for the APH-IF backend.

Tracks operation timings, aggregates metrics, raises alerts, and powers the
backend "performance" and "timing" endpoints that observe the parallel hybrid
RAG pipeline. This module is imported by `app.main` to provide real-time
analytics for retrieval and fusion stages without changing request behavior.
"""

from __future__ import annotations

# __________________________________________________________________________
# Imports

import json
import logging
import statistics
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, DefaultDict, Dict, List, Optional

# __________________________________________________________________________
# Global Constants / Variables

logger = logging.getLogger(__name__)

# ____________________________________________________________________________
# Class Definitions

# =========================================================================
# Performance Data Models
# =========================================================================

# ------------------------------------------------------------------------- class PerformanceMetric
@dataclass(slots=True)
class PerformanceMetric:
    """Individual performance measurement."""
    timestamp: float
    operation: str
    duration_ms: int
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
# ------------------------------------------------------------------------- end class PerformanceMetric

# ------------------------------------------------------------------------- class DetailedTimingBreakdown
@dataclass(slots=True)
class DetailedTimingBreakdown:
    """Hierarchical timing breakdown for end-to-end request processing."""
    # End-to-end metrics
    total_request_response_time_ms: float = 0.0
    user_perceived_response_time_ms: float = 0.0
    time_to_first_result_ms: float = 0.0
    
    # Request lifecycle breakdown
    request_parsing_time_ms: float = 0.0
    request_routing_time_ms: float = 0.0
    session_handling_time_ms: float = 0.0
    parameter_validation_time_ms: float = 0.0
    request_queue_time_ms: float = 0.0
    
    # Database access timing
    neo4j_connection_acquisition_time_ms: float = 0.0
    neo4j_connection_establishment_time_ms: float = 0.0
    cypher_query_compilation_time_ms: float = 0.0
    cypher_query_execution_time_ms: float = 0.0
    graph_data_retrieval_time_ms: float = 0.0
    graph_result_processing_time_ms: float = 0.0
    neo4j_network_latency_ms: float = 0.0
    
    # Vector database timing
    vector_index_query_time_ms: float = 0.0
    embedding_lookup_time_ms: float = 0.0
    similarity_calculation_time_ms: float = 0.0
    vector_result_ranking_time_ms: float = 0.0
    
    # LLM API timing
    llm_api_request_time_ms: float = 0.0
    llm_network_latency_ms: float = 0.0
    llm_processing_time_ms: float = 0.0
    token_encoding_time_ms: float = 0.0
    token_decoding_time_ms: float = 0.0
    rate_limit_wait_time_ms: float = 0.0
    llm_streaming_chunk_time_ms: float = 0.0
    
    # Parallel processing timing
    parallel_coordination_overhead_ms: float = 0.0
    parallel_task_synchronization_ms: float = 0.0
    semantic_search_execution_time_ms: float = 0.0
    traversal_search_execution_time_ms: float = 0.0
    parallel_speedup_achieved_ms: float = 0.0
    
    # Context fusion timing
    fusion_preprocessing_time_ms: float = 0.0
    fusion_strategy_selection_time_ms: float = 0.0
    result_comparison_time_ms: float = 0.0
    content_merging_time_ms: float = 0.0
    citation_extraction_time_ms: float = 0.0
    confidence_calculation_time_ms: float = 0.0
    fusion_post_processing_time_ms: float = 0.0
    
    # Cache operation timing
    cache_lookup_time_ms: float = 0.0
    cache_hit_retrieval_time_ms: float = 0.0
    cache_miss_processing_time_ms: float = 0.0
    cache_storage_time_ms: float = 0.0
    cache_similarity_matching_time_ms: float = 0.0
    cache_cleanup_time_ms: float = 0.0
    cache_key_generation_time_ms: float = 0.0
    
    # Quality processing timing
    entity_extraction_time_ms: float = 0.0
    result_validation_time_ms: float = 0.0
    source_attribution_time_ms: float = 0.0
    content_formatting_time_ms: float = 0.0
    metadata_enrichment_time_ms: float = 0.0
    
    # Error handling timing
    circuit_breaker_decision_time_ms: float = 0.0
    fallback_execution_time_ms: float = 0.0
    error_recovery_time_ms: float = 0.0
    retry_attempt_overhead_ms: float = 0.0
# ------------------------------------------------------------------------- end class DetailedTimingBreakdown

# ------------------------------------------------------------------------- class TimingContext
@dataclass(slots=True)
class TimingContext:
    """Context for nested timing measurements with parent-child relationships."""
    operation_id: str
    operation_name: str
    start_time: float
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child_id: str) -> None:
        """Add a child timing context."""
        if child_id not in self.children:
            self.children.append(child_id)
    
    @property
    def duration_ms(self) -> int:
        """Calculate duration from start time to now."""
        return int((time.time() - self.start_time) * 1000)
# ------------------------------------------------------------------------- end class TimingContext

# ------------------------------------------------------------------------- class SystemMetrics
@dataclass(slots=True)
class SystemMetrics:
    """System-wide performance metrics with enhanced timing capabilities."""
    # Legacy timing metrics (maintained for compatibility) - no defaults
    avg_parallel_time_ms: float
    avg_fusion_time_ms: float
    avg_total_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    
    # Success rates - no defaults
    parallel_success_rate: float
    fusion_success_rate: float
    overall_success_rate: float
    
    # Cache metrics - no defaults
    cache_hit_rate: float
    cache_time_saved_ms: float
    
    # Volume metrics - no defaults
    total_queries: int
    queries_per_minute: float
    
    # Quality metrics - no defaults
    avg_confidence: float
    avg_complementarity: float
    
    # Resource utilization - no defaults
    avg_semantic_time_ms: float
    avg_traversal_time_ms: float
    parallelism_efficiency: float  # How much time saved by parallelism
    
    # Enhanced timing metrics - with defaults (must be at end)
    avg_end_to_end_time_ms: float = 0.0
    avg_database_access_time_ms: float = 0.0
    avg_llm_api_time_ms: float = 0.0
    p99_9_response_time_ms: float = 0.0
    processing_efficiency_ratio: float = 0.0  # Processing time / total time
    parallel_vs_sequential_ratio: float = 0.0  # Parallel time / sequential time
    cache_efficiency_impact_ms: float = 0.0  # Average time saved per request by caching
# ------------------------------------------------------------------------- end class SystemMetrics

# ------------------------------------------------------------------------- class AlertThreshold
@dataclass(slots=True)
class AlertThreshold:
    """Performance alert threshold configuration."""
    metric_name: str
    threshold_value: float
    comparison: str  # "above", "below"
    severity: str    # "warning", "critical"
    description: str
# ------------------------------------------------------------------------- end class AlertThreshold

# =========================================================================
# Performance Monitor
# =========================================================================

# ------------------------------------------------------------------------- class PerformanceMonitor
class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for APH-IF operations.
    
    Features:
    - Real-time metrics collection
    - Statistical analysis and alerting
    - Performance trend tracking
    - Resource utilization monitoring
    - Automatic performance optimization suggestions
    """
    
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self, max_history: int = 10000, alert_enabled: bool = True):
        """Initialize the performance monitor.

        Args:
            max_history: Maximum number of metrics retained in memory for rolling analysis.
            alert_enabled: Whether to evaluate metrics against alert thresholds.
        """
        self.max_history = max_history
        self.alert_enabled = alert_enabled
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self._metrics: deque[PerformanceMetric] = deque(maxlen=max_history)
        self._operation_stats: DefaultDict[str, List[float]] = defaultdict(list)
        
        # Real-time tracking
        self._current_operations: Dict[str, float] = {}  # operation_id -> start_time
        self._alert_history: List[Dict[str, Any]] = []
        
        # Performance counters
        self._counters: Dict[str, int] = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "parallel_operations": 0,
            "fusion_operations": 0,
            "alerts_triggered": 0,
        }
        
        # Alert thresholds
        self._alert_thresholds: List[AlertThreshold] = [
            AlertThreshold(
                "avg_response_time_ms",
                10000,
                "above",
                "warning",
                "Average response time exceeding 10 seconds",
            ),
            AlertThreshold(
                "success_rate",
                0.8,
                "below",
                "critical",
                "Success rate below 80%",
            ),
            # Cache hit rate threshold removed
            AlertThreshold(
                "p99_response_time_ms",
                30000,
                "above",
                "critical",
                "99th percentile response time exceeding 30 seconds",
            ),
        ]
        
        self.logger.info(f"Performance monitor initialized with {max_history} max history")
    # --------------------------------------------------------------------------------- end __init__()
    
    # --------------------------------------------------------------------------------- track_operation()
    @asynccontextmanager
    async def track_operation(
        self,
        operation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[OperationTracker]:
        """Track an operation's timing within an async context.

        This yields an `OperationTracker` that allows callers to add metadata
        during execution. Timing is recorded on exit, and failures are captured
        as unsuccessful metrics. Typical usage is inside API routes and engine
        orchestration to measure retrieval, fusion, or other stages.

        Args:
            operation: Logical operation name (e.g., "parallel_retrieval").
            metadata: Optional initial metadata for the operation.

        Yields:
            OperationTracker: Tracker to enrich metric metadata before the
            context exits.
        """
        operation_id = f"{operation}_{time.time()}_{id(self)}"
        start_time = time.time()
        self._current_operations[operation_id] = start_time
        
        # Create tracker object
        tracker = OperationTracker(metadata or {})
        
        try:
            yield tracker
            # Operation completed successfully
            duration_ms = int((time.time() - start_time) * 1000)
            await self.record_metric(
                operation=operation,
                duration_ms=duration_ms,
                success=True,
                metadata=tracker.metadata,
            )
            
        except Exception as e:
            # Operation failed
            duration_ms = int((time.time() - start_time) * 1000)
            await self.record_metric(
                operation=operation,
                duration_ms=duration_ms,
                success=False,
                metadata={**tracker.metadata, "error": str(e)},
            )
            raise
        
        finally:
            self._current_operations.pop(operation_id, None)
    # --------------------------------------------------------------------------------- end track_operation()
    
    # --------------------------------------------------------------------------------- record_metric()
    async def record_metric(
        self,
        operation: str,
        duration_ms: int,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a performance metric.

        Args:
            operation: Operation name associated with the measurement.
            duration_ms: Duration of the operation in milliseconds.
            success: Whether the operation completed successfully.
            metadata: Optional metadata to attach (e.g., counts, component times).
        """
        metric = PerformanceMetric(
            timestamp=time.time(),
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            metadata=metadata or {},
        )
        
        # Store metric
        self._metrics.append(metric)
        self._operation_stats[operation].append(duration_ms)
        
        # Maintain operation stats size
        if len(self._operation_stats[operation]) > 1000:
            self._operation_stats[operation] = self._operation_stats[operation][-500:]
        
        # Update counters
        self._counters['total_operations'] += 1
        if success:
            self._counters["successful_operations"] += 1
        else:
            self._counters["failed_operations"] += 1
        
        # Track specific operation types
        if operation == "parallel_retrieval":
            self._counters["parallel_operations"] += 1
        elif operation == "context_fusion":
            self._counters["fusion_operations"] += 1
        
        # Check for alerts
        if self.alert_enabled:
            await self._check_alerts()
        
        self.logger.debug(
            f"Recorded metric: {operation} - {duration_ms}ms - {'✅' if success else '❌'}",
        )
    # --------------------------------------------------------------------------------- end record_metric()
    
    # --------------------------------------------------------------------------------- record_cache_event()
    async def record_cache_event(
        self,
        event_type: str,
        operation: str,
        time_saved_ms: Optional[int] = None,
    ) -> None:
        """Record a cache hit or miss event.

        Args:
            event_type: Either "hit" or "miss".
            operation: Operation name the cache event is associated with.
            time_saved_ms: Optional estimated time saved for cache hits.
        """
        if event_type == "hit":
            self._counters["cache_hits"] += 1
            if time_saved_ms:
                # Record the cache hit as a very fast operation
                await self.record_metric(
                    operation=f"cache_{operation}",
                    duration_ms=1,  # Cache access is nearly instantaneous
                    success=True,
                    metadata={
                        "time_saved_ms": time_saved_ms,
                        "cache_event": "hit",
                    },
                )
        elif event_type == "miss":
            self._counters["cache_misses"] += 1
    # --------------------------------------------------------------------------------- end record_cache_event()
    
    # --------------------------------------------------------------------------------- get_current_stats()
    def get_current_stats(self, time_window_minutes: int = 60) -> SystemMetrics:
        """Compute current system performance statistics.

        Args:
            time_window_minutes: Rolling time window, in minutes, to include in calculations.

        Returns:
            SystemMetrics: Aggregated performance metrics for the specified window.
        """
        cutoff_time = time.time() - (time_window_minutes * 60)
        recent_metrics = [m for m in self._metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            # Return default metrics if no data
            return SystemMetrics(
                avg_parallel_time_ms=0.0,
                avg_fusion_time_ms=0.0,
                avg_total_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                parallel_success_rate=0.0,
                fusion_success_rate=0.0,
                overall_success_rate=0.0,
                cache_hit_rate=0.0,
                cache_time_saved_ms=0.0,
                total_queries=0,
                queries_per_minute=0.0,
                avg_confidence=0.0,
                avg_complementarity=0.0,
                avg_semantic_time_ms=0.0,
                avg_traversal_time_ms=0.0,
                parallelism_efficiency=0.0,
            )
        
        # Calculate timing metrics
        all_durations = [m.duration_ms for m in recent_metrics if m.success]
        parallel_durations = [
            m.duration_ms
            for m in recent_metrics
            if m.operation == "parallel_retrieval" and m.success
        ]
        fusion_durations = [
            m.duration_ms
            for m in recent_metrics
            if m.operation == "context_fusion" and m.success
        ]
        
        avg_parallel_time = statistics.mean(parallel_durations) if parallel_durations else 0.0
        avg_fusion_time = statistics.mean(fusion_durations) if fusion_durations else 0.0
        avg_total_time = statistics.mean(all_durations) if all_durations else 0.0
        
        # Calculate percentiles
        if all_durations:
            sorted_durations = sorted(all_durations)
            p95_index = int(0.95 * len(sorted_durations))
            p99_index = int(0.99 * len(sorted_durations))
            p95_response_time = sorted_durations[p95_index] if p95_index < len(sorted_durations) else sorted_durations[-1]
            p99_response_time = sorted_durations[p99_index] if p99_index < len(sorted_durations) else sorted_durations[-1]
        else:
            p95_response_time = 0.0
            p99_response_time = 0.0
        
        # Calculate success rates
        total_ops = len(recent_metrics)
        successful_ops = len([m for m in recent_metrics if m.success])
        parallel_ops = [m for m in recent_metrics if m.operation == "parallel_retrieval"]
        fusion_ops = [m for m in recent_metrics if m.operation == "context_fusion"]
        
        overall_success_rate = (successful_ops / total_ops * 100) if total_ops > 0 else 0.0
        parallel_success_rate = (len([m for m in parallel_ops if m.success]) / len(parallel_ops) * 100) if parallel_ops else 0.0
        fusion_success_rate = (len([m for m in fusion_ops if m.success]) / len(fusion_ops) * 100) if fusion_ops else 0.0
        
        # Calculate cache metrics
        total_cache_ops = self._counters["cache_hits"] + self._counters["cache_misses"]
        cache_hit_rate = (
            self._counters["cache_hits"] / total_cache_ops * 100
            if total_cache_ops > 0
            else 0.0
        )
        
        # Calculate time saved by caching
        cache_metrics = [
            m for m in recent_metrics if m.metadata.get("cache_event") == "hit"
        ]
        cache_time_saved = sum(m.metadata.get("time_saved_ms", 0) for m in cache_metrics)
        
        # Calculate query volume
        queries_per_minute = len(recent_metrics) / time_window_minutes if time_window_minutes > 0 else 0.0
        
        # Calculate quality metrics
        confidence_values = []
        complementarity_values = []
        for m in recent_metrics:
            if m.metadata.get("confidence"):
                confidence_values.append(m.metadata["confidence"])
            if m.metadata.get("complementarity_score"):
                complementarity_values.append(m.metadata["complementarity_score"])
        
        avg_confidence = statistics.mean(confidence_values) if confidence_values else 0.0
        avg_complementarity = statistics.mean(complementarity_values) if complementarity_values else 0.0
        
        # Calculate component timing
        semantic_times = []
        traversal_times = []
        for m in recent_metrics:
            if m.metadata.get("semantic_time_ms"):
                semantic_times.append(m.metadata["semantic_time_ms"])
            if m.metadata.get("traversal_time_ms"):
                traversal_times.append(m.metadata["traversal_time_ms"])
        
        avg_semantic_time = statistics.mean(semantic_times) if semantic_times else 0.0
        avg_traversal_time = statistics.mean(traversal_times) if traversal_times else 0.0
        
        # Calculate parallelism efficiency
        if avg_semantic_time > 0 and avg_traversal_time > 0:
            sequential_time = avg_semantic_time + avg_traversal_time
            parallel_time = max(avg_semantic_time, avg_traversal_time)
            parallelism_efficiency = ((sequential_time - parallel_time) / sequential_time * 100) if sequential_time > 0 else 0.0
        else:
            parallelism_efficiency = 0.0
        
        return SystemMetrics(
            avg_parallel_time_ms=avg_parallel_time,
            avg_fusion_time_ms=avg_fusion_time,
            avg_total_time_ms=avg_total_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            parallel_success_rate=parallel_success_rate,
            fusion_success_rate=fusion_success_rate,
            overall_success_rate=overall_success_rate,
            cache_hit_rate=cache_hit_rate,
            cache_time_saved_ms=cache_time_saved,
            total_queries=total_ops,
            queries_per_minute=queries_per_minute,
            avg_confidence=avg_confidence,
            avg_complementarity=avg_complementarity,
            avg_semantic_time_ms=avg_semantic_time,
            avg_traversal_time_ms=avg_traversal_time,
            parallelism_efficiency=parallelism_efficiency,
        )
    # --------------------------------------------------------------------------------- end get_current_stats()
    
    # --------------------------------------------------------------------------------- _check_alerts()
    async def _check_alerts(self) -> None:
        """Check current metrics against alert thresholds and record alerts.

        This method computes a short-window snapshot and compares selected
        aggregates (e.g., success rate, p99) against configured thresholds.
        When an alert is triggered, it is appended to the internal alert
        history and counters are updated.
        """
        current_stats = self.get_current_stats(time_window_minutes=10)  # Check last 10 minutes
        
        for threshold in self._alert_thresholds:
            try:
                current_value = getattr(current_stats, threshold.metric_name, None)
                if current_value is None:
                    continue
                
                should_alert = False
                if threshold.comparison == "above" and current_value > threshold.threshold_value:
                    should_alert = True
                elif threshold.comparison == "below" and current_value < threshold.threshold_value:
                    should_alert = True
                
                if should_alert:
                    alert = {
                        "timestamp": time.time(),
                        "metric": threshold.metric_name,
                        "current_value": current_value,
                        "threshold": threshold.threshold_value,
                        "severity": threshold.severity,
                        "description": threshold.description,
                    }
                    
                    self._alert_history.append(alert)
                    self._counters["alerts_triggered"] += 1
                    
                    # Limit alert history
                    if len(self._alert_history) > 100:
                        self._alert_history = self._alert_history[-50:]
                    
                    self.logger.warning(f"Performance Alert [{threshold.severity.upper()}]: "
                                      f"{threshold.description} - Current: {current_value:.2f}, "
                                      f"Threshold: {threshold.threshold_value}")
                    
            except Exception as e:
                self.logger.error(f"Error checking alert threshold {threshold.metric_name}: {e}")
    # --------------------------------------------------------------------------------- end _check_alerts()
    
    # --------------------------------------------------------------------------------- get_performance_dashboard()
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Generate a structured performance dashboard snapshot.

        Returns:
            Dict[str, Any]: Current metrics, counters, recent alerts, trends,
            and high-level recommendations for the last hour.
        """
        current_stats = self.get_current_stats()
        recent_alerts = [
            alert for alert in self._alert_history if alert["timestamp"] > time.time() - 3600
        ]  # Last hour
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": asdict(current_stats),
            "operational_counters": self._counters.copy(),
            "recent_alerts": recent_alerts,
            "alert_summary": {
                "total_alerts": len(self._alert_history),
                "recent_alerts": len(recent_alerts),
                "critical_alerts": len(
                    [a for a in recent_alerts if a["severity"] == "critical"]
                ),
                "warning_alerts": len(
                    [a for a in recent_alerts if a["severity"] == "warning"]
                ),
            },
            "performance_trends": {
                "avg_response_time_trend": self._calculate_trend("avg_total_time_ms"),
                "success_rate_trend": self._calculate_trend("overall_success_rate"),
                # Cache hit rate trend removed
            },
            "recommendations": self._generate_performance_recommendations(current_stats),
        }
    # --------------------------------------------------------------------------------- end get_performance_dashboard()
    
    # --------------------------------------------------------------------------------- _calculate_trend()
    def _calculate_trend(self, metric_name: str, periods: int = 5) -> List[float]:
        """Calculate a coarse trend for a metric across fixed time periods.

        Args:
            metric_name: Name of the metric attribute inside `SystemMetrics`.
            periods: Number of backward periods (10 minutes each) to compute.

        Returns:
            List[float]: Oldest-to-newest sequence of metric values.
        """
        try:
            trend_values = []
            current_time = time.time()
            period_duration = 600  # 10 minutes per period
            
            for i in range(periods):
                period_end = current_time - (i * period_duration)
                period_start = period_end - period_duration
                
                period_metrics = [
                    m for m in self._metrics if period_start <= m.timestamp < period_end
                ]
                
                if period_metrics:
                    period_stats = self.get_current_stats(time_window_minutes=10)
                    value = getattr(period_stats, metric_name, 0.0)
                    trend_values.append(value)
                else:
                    trend_values.append(0.0)
            
            return list(reversed(trend_values))  # Oldest to newest
            
        except Exception as e:
            self.logger.warning(f"Error calculating trend for {metric_name}: {e}")
            return [0.0] * periods
    # --------------------------------------------------------------------------------- end _calculate_trend()
    
    # --------------------------------------------------------------------------------- _generate_performance_recommendations()
    def _generate_performance_recommendations(self, stats: SystemMetrics) -> List[str]:
        """Generate performance optimization recommendations.

        Args:
            stats: Aggregated `SystemMetrics` used to derive suggestions.

        Returns:
            List[str]: Human-readable optimization recommendations.
        """
        recommendations = []
        
        # Response time recommendations
        if stats.avg_total_time_ms > 8000:
            recommendations.append("Consider optimizing search query processing")
        
        if stats.p99_response_time_ms > 20000:
            recommendations.append("Investigate queries causing high P99 response times")
        
        # Cache recommendations removed - using direct processing mode
        
        # Parallel efficiency recommendations
        if stats.parallelism_efficiency < 30:
            recommendations.append("Parallelism efficiency is low - investigate search timing balance")
        
        # Success rate recommendations
        if stats.overall_success_rate < 90:
            recommendations.append("Success rate is below optimal - review error patterns")
        
        # Quality recommendations
        if stats.avg_confidence < 0.6:
            recommendations.append("Average confidence is low - review search algorithms and data quality")
        
        if stats.avg_complementarity > 0.8:
            recommendations.append("High complementarity indicates good search diversity")
        
        if not recommendations:
            recommendations.append("System performance is within optimal ranges")
        
        return recommendations
    # --------------------------------------------------------------------------------- end _generate_performance_recommendations()
    
    # --------------------------------------------------------------------------------- export_metrics()
    async def export_metrics(self, format: str = "json", time_window_hours: int = 24) -> str:
        """Export metrics data for external analysis.

        Args:
            format: Output format; currently only "json" is supported.
            time_window_hours: Number of hours of history to include.

        Returns:
            str: Serialized metrics and a summary block for the requested window.
        """
        cutoff_time = time.time() - (time_window_hours * 3600)
        export_metrics = [
            {
                "timestamp": m.timestamp,
                "datetime": datetime.fromtimestamp(m.timestamp).isoformat(),
                "operation": m.operation,
                "duration_ms": m.duration_ms,
                "success": m.success,
                "metadata": m.metadata,
            }
            for m in self._metrics
            if m.timestamp >= cutoff_time
        ]
        
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "time_window_hours": time_window_hours,
                "total_metrics": len(export_metrics),
            },
            "metrics": export_metrics,
            "summary": asdict(
                self.get_current_stats(time_window_minutes=time_window_hours * 60),
            ),
        }
        
        if format.lower() == "json":
            return json.dumps(export_data, indent=2, default=str)
        else:
            return str(export_data)
    # --------------------------------------------------------------------------------- end export_metrics()

# ------------------------------------------------------------------------- end class PerformanceMonitor

# ------------------------------------------------------------------------- class OperationTracker
class OperationTracker:
    """Operation-scoped metadata tracker used by performance monitoring.

    Instances of this class are yielded by `PerformanceMonitor.track_operation(...)` and allow
    callers to attach structured metadata to the metric that will be recorded when the
    context exits. Use `add_metadata` to merge multiple key/value pairs, or `set_metadata`
    to overwrite a single key.

    Attributes:
        metadata: Mutable dictionary of operation-scoped metadata that will be attached to the
            recorded performance metric when the surrounding context manager completes.

    Examples:
        async with monitor.track_operation("parallel_retrieval") as op:
            op.add_metadata({"semantic_time_ms": 125, "traversal_time_ms": 210})
            op.set_metadata("result_count", 10)
    """
    
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self, initial_metadata: Dict[str, Any]):
        self.metadata = initial_metadata.copy()
    # --------------------------------------------------------------------------------- end __init__()
    
    # --------------------------------------------------------------------------------- add_metadata()
    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add additional metadata to the operation.

        Args:
            metadata: Key-value pairs to merge into the current operation metadata.
        """
        self.metadata.update(metadata)
    # --------------------------------------------------------------------------------- end add_metadata()
    
    # --------------------------------------------------------------------------------- set_metadata()
    def set_metadata(self, key: str, value: Any) -> None:
        """Set a specific metadata key-value pair.

        Args:
            key: Metadata key to set.
            value: Corresponding value to assign.
        """
        self.metadata[key] = value
    # --------------------------------------------------------------------------------- end set_metadata()

# --------------------------------------------------------------------------------- end class OperationTracker

# __________________________________________________________________________
# Standalone Function Definitions
#

# ______________________
# Utility Functions
#
# --------------------------------------------------------------------------------- safe_mean()
def safe_mean(values: List[float]) -> float:
    """Return the arithmetic mean of values, or 0.0 when the list is empty.

    Args:
        values: Sequence of numeric values.

    Returns:
        Mean of the input values, or 0.0 if no values are provided.
    """
    return statistics.mean(values) if values else 0.0
# --------------------------------------------------------------------------------- end safe_mean()

# --------------------------------------------------------------------------------- format_duration_ms()
def format_duration_ms(duration_ms: int) -> str:
    """Format a duration in milliseconds into a human-readable string.

    Args:
        duration_ms: Duration in milliseconds.

    Returns:
        A compact string such as "850 ms" or "1.24 s".
    """
    if duration_ms < 1000:
        return f"{duration_ms} ms"
    seconds = duration_ms / 1000.0
    return f"{seconds:.2f} s"
# --------------------------------------------------------------------------------- end format_duration_ms()

# --------------------------------------------------------------------------------- mask_sensitive_metadata()
def mask_sensitive_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow-copied metadata dict with common secrets masked.

    This masks values for keys commonly associated with secrets (e.g., "token",
    "password", "api_key"). Only top-level keys are considered.

    Args:
        metadata: Metadata dictionary to sanitize.

    Returns:
        A sanitized copy with sensitive values replaced by "***".
    """
    sensitive_keys = {"token", "password", "api_key", "authorization", "secret"}
    sanitized = metadata.copy()
    for key in list(sanitized.keys()):
        if key.lower() in sensitive_keys:
            sanitized[key] = "***"
    return sanitized
# --------------------------------------------------------------------------------- end mask_sensitive_metadata()

# =========================================================================
# Global Monitor Instance
# =========================================================================

_performance_monitor: Optional[PerformanceMonitor] = None

# --------------------------------------------------------------------------------- get_performance_monitor()
def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor instance.

    Returns:
        PerformanceMonitor: Lazily-initialized singleton monitor instance.
    """
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
# --------------------------------------------------------------------------------- end get_performance_monitor()

# --------------------------------------------------------------------------------- initialize_monitor()
async def initialize_monitor() -> PerformanceMonitor:
    """Initialize the global performance monitor.

    Returns:
        PerformanceMonitor: The initialized monitor instance.
    """
    monitor = get_performance_monitor()
    return monitor
# --------------------------------------------------------------------------------- end initialize_monitor()

# =========================================================================
# Health Check Integration
# =========================================================================

# --------------------------------------------------------------------------------- performance_health_check()
async def performance_health_check() -> Dict[str, Any]:
    """Perform health check on the performance monitoring system.

    Returns:
        Dict[str, Any]: Tool health, selected metrics, and recommendations for the UI.
    """
    monitor = get_performance_monitor()
    stats = monitor.get_current_stats(time_window_minutes=60)
    dashboard = monitor.get_performance_dashboard()
    
    # Determine health status
    if stats.overall_success_rate > 90 and stats.avg_total_time_ms < 8000:
        status = "excellent"
    elif stats.overall_success_rate > 80 and stats.avg_total_time_ms < 15000:
        status = "good"
    elif stats.overall_success_rate > 70:
        status = "degraded"
    else:
        status = "poor"
    
    return {
        "tool_name": "performance_monitor",
        "status": status,
        "health_metrics": {
            "success_rate": stats.overall_success_rate,
            "avg_response_time_ms": stats.avg_total_time_ms,
            # Cache hit rate removed - direct processing mode
            "active_alerts": len(
                [a for a in dashboard["recent_alerts"] if a["timestamp"] > time.time() - 1800]
            ),  # Last 30 min
        },
        "recommendations": dashboard["recommendations"][:3],  # Top 3 recommendations
    }
# --------------------------------------------------------------------------------- end performance_health_check()

# __________________________________________________________________________
# End of File