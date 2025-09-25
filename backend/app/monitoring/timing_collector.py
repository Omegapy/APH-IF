# -------------------------------------------------------------------------
# File: timing_collector.py
# Author: Alexander Ricciardi
# Date: 2025-09-18
# [File Path] backend/app/monitoring/timing_collector.py
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
#   Advanced timing collection utilities with hierarchical async contexts and
#   lightweight overhead tracking. Produces detailed timing breakdowns used by
#   performance dashboards and health endpoints.
# -------------------------------------------------------------------------
# --- Module Contents Overview ---
# - Dataclass: CompletedTiming
# - Class: TimingCollector
# - Class: TimingTracker
# - Functions: get_timing_collector, initialize_timing_collector,
#              measure_operation, get_timing_stats
# -------------------------------------------------------------------------
# --- Dependencies / Imports ---
# - Standard Library: asyncio, time, uuid, logging, typing, contextlib,
#   dataclasses, collections, threading, concurrent.futures
# - Local Project Modules: performance_monitor (TimingContext, DetailedTimingBreakdown)
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# Used by backend performance endpoints and monitoring dashboards:
#   - Imported by FastAPI routes (e.g., /timing/* endpoints in app.main)
#   - Provides hierarchical timing via get_timing_collector().measure(...)
#   - Exposes stats and detailed breakdowns for health and analysis
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Advanced Timing Collection System for APH-IF Backend

Comprehensive timing measurement system with hierarchical context support,
minimal overhead, and automatic timing breakdown for performance optimization.
"""

# __________________________________________________________________________
# Imports

import asyncio
import time
import uuid
import logging
from typing import Dict, List, Any, Optional, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

from .performance_monitor import TimingContext, DetailedTimingBreakdown

logger = logging.getLogger(__name__)

# __________________________________________________________________________
# Timing Collection Infrastructure

# ------------------------------------------------------------------------- class CompletedTiming
@dataclass
class CompletedTiming:
    """Completed timing measurement with full context.

    Attributes:
        operation_id: Unique id for this timing context.
        operation_name: Logical operation name being measured.
        start_time: Start timestamp (perf counter).
        end_time: End timestamp (perf counter).
        duration_ms: Total duration in milliseconds.
        parent_id: Optional parent operation id (for hierarchy).
        children: Child operation ids.
        metadata: Arbitrary metadata captured during timing.
        thread_id: OS thread id where the timing was recorded.
        success: Whether the operation completed successfully.
        error: Error string if operation failed.
    """
    operation_id: str
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    parent_id: Optional[str]
    children: List[str]
    metadata: Dict[str, Any]
    thread_id: int
    success: bool = True
    error: Optional[str] = None
    
    @property
    def hierarchy_level(self) -> int:
        """Calculate the depth level in timing hierarchy."""
        level = 0
        if self.parent_id:
            # This would need to be calculated by the collector
            level = 1  # Simplified for now
        return level
# ------------------------------------------------------------------------- end class CompletedTiming

# TODO: Not a candidate for @dataclass — manages active contexts, locks, and internal state.
# ------------------------------------------------------------------------- class TimingCollector
class TimingCollector:
    """Advanced timing collector with hierarchical context support and low overhead.

    Features:
    - Nested timing contexts with parent-child relationships.
    - Thread-safe operation tracking using re-entrant locks and thread-local stacks.
    - Periodic overhead sampling to quantify measurement cost.
    - Hierarchical timing breakdowns for dashboards and health endpoints.
    - Memory-efficient rolling storage of completed timings.

    Attributes:
        max_active_contexts: Upper bound on concurrently tracked contexts.
        overhead_sampling_rate: Fraction of operations sampled for overhead (0.0–1.0).
        _active_contexts: Map of operation id to active ``TimingContext``.
        _completed_timings: Recent list of ``CompletedTiming`` records.
        _overhead_samples: Rolling list of overhead measurements (milliseconds).
        _stats: Aggregate counters and gauges for the collector.
    """
    
    # ______________________
    # Constructor 
    # 
    # -------------------------------------------------------------- __init__()
    def __init__(self, max_active_contexts: int = 10000, overhead_sampling_rate: float = 0.01):
        """
        Initialize timing collector.
        
        Args:
            max_active_contexts: Maximum number of concurrent timing contexts
            overhead_sampling_rate: Rate at which to sample timing overhead (0.0-1.0)
        """
        self.max_active_contexts = max_active_contexts
        self.overhead_sampling_rate = overhead_sampling_rate
        
        # Active timing contexts
        self._active_contexts: Dict[str, TimingContext] = {}
        self._context_lock = threading.RLock()
        
        # Completed timings storage
        self._completed_timings: List[CompletedTiming] = []
        self._completed_lock = threading.RLock()
        
        # Overhead measurement
        self._overhead_samples: List[float] = []
        self._collection_start_time = time.perf_counter()
        
        # Thread-local storage for context stack
        self._local = threading.local()
        
        # Statistics
        self._stats = {
            'total_timings': 0,
            'active_contexts': 0,
            'completed_timings': 0,
            'overhead_ms': 0.0,
            'memory_usage_bytes': 0
        }
        
        logger.info(f"TimingCollector initialized with {max_active_contexts} max contexts")
    # -------------------------------------------------------------- end __init__()
    
    # ______________________
    # Internal/Private Methods (single leading underscore convention)
    #
    # -------------------------------------------------------------- _get_context_stack()
    def _get_context_stack(self) -> List[str]:
        """Return the current thread's timing context stack.

        Returns:
            List[str]: Stack of active operation ids for the current thread.
        """
        if not hasattr(self._local, 'context_stack'):
            self._local.context_stack = []
        return self._local.context_stack
    # -------------------------------------------------------------- end _get_context_stack()
    
    # -------------------------------------------------------------- _generate_operation_id()
    def _generate_operation_id(self) -> str:
        """Generate a unique, sortable operation identifier.

        Returns:
            str: Unique id combining a short UUID and timestamp component.
        """
        return f"op_{uuid.uuid4().hex[:8]}_{int(time.time() * 1000000)}"
    # -------------------------------------------------------------- end _generate_operation_id()
    
    # -------------------------------------------------------------- _measure_overhead()
    def _measure_overhead(self) -> float:
        """Measure the overhead incurred by timing collection itself.

        Returns:
            float: Estimated collection overhead in milliseconds.
        """
        if len(self._overhead_samples) == 0:
            return 0.0
        
        # Simple overhead calculation - can be enhanced
        start = time.perf_counter()
        # Simulate minimal timing work
        _ = time.perf_counter()
        overhead_ns = (time.perf_counter() - start) * 1_000_000  # Convert to microseconds
        return overhead_ns / 1000  # Convert to milliseconds
    # -------------------------------------------------------------- end _measure_overhead()
    #
    # ______________________
    # Functionality: Measurement API
    # =========================================================================
    # -------------------------------------------------------------- measure()
    @asynccontextmanager
    async def measure(self, 
                      operation_name: str, 
                      metadata: Optional[Dict[str, Any]] = None,
                      track_children: bool = True):
        """Async context manager for measuring operation timing.

        Args:
            operation_name: Name of the operation being timed.
            metadata: Additional metadata to attach to this timing.
            track_children: Whether to track child operations (reserved; currently
                informational only).

        Yields:
            TimingTracker: Tracker for adding or setting metadata during execution.

        Example:
            async with timing_collector.measure("database_query") as timer:
                rows = await database.query(...)
                timer.add_metadata({"rows_returned": len(rows)})
        """
        operation_id = self._generate_operation_id()
        start_time = time.perf_counter()
        
        # Measure overhead if sampling
        overhead_start = None
        if self.overhead_sampling_rate > 0 and len(self._overhead_samples) % int(1/self.overhead_sampling_rate) == 0:
            overhead_start = time.perf_counter()
        
        # Get parent context
        context_stack = self._get_context_stack()
        parent_id = context_stack[-1] if context_stack else None
        
        # Create timing context
        context = TimingContext(
            operation_id=operation_id,
            operation_name=operation_name,
            start_time=start_time,
            parent_id=parent_id,
            metadata=metadata or {}
        )
        
        # Add to active contexts
        with self._context_lock:
            if len(self._active_contexts) >= self.max_active_contexts:
                # Clean up oldest contexts to prevent memory issues
                oldest_contexts = sorted(
                    self._active_contexts.items(),
                    key=lambda x: x[1].start_time
                )[:100]  # Remove oldest 100
                for old_id, _ in oldest_contexts:
                    self._active_contexts.pop(old_id, None)
            
            self._active_contexts[operation_id] = context
            self._stats['active_contexts'] = len(self._active_contexts)
        
        # Add to parent's children
        if parent_id and parent_id in self._active_contexts:
            self._active_contexts[parent_id].add_child(operation_id)
        
        # Push to context stack
        context_stack.append(operation_id)
        
        # Create timing tracker for user interaction
        timing_tracker = TimingTracker(context, self)
        
        try:
            yield timing_tracker
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Pop from context stack
            context_stack.pop()
            
            # Create completed timing
            completed_timing = CompletedTiming(
                operation_id=operation_id,
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                parent_id=parent_id,
                children=context.children.copy(),
                metadata=context.metadata.copy(),
                thread_id=threading.get_ident(),
                success=success,
                error=error
            )
            
            # Store completed timing
            with self._completed_lock:
                self._completed_timings.append(completed_timing)
                # Limit memory usage
                if len(self._completed_timings) > 10000:
                    self._completed_timings = self._completed_timings[-5000:]
            
            # Remove from active contexts
            with self._context_lock:
                self._active_contexts.pop(operation_id, None)
                self._stats['active_contexts'] = len(self._active_contexts)
            
            # Update statistics
            self._stats['total_timings'] += 1
            self._stats['completed_timings'] = len(self._completed_timings)
            
            # Record overhead if measuring
            if overhead_start:
                overhead_ms = (time.perf_counter() - overhead_start) * 1000
                self._overhead_samples.append(overhead_ms)
                if len(self._overhead_samples) > 1000:
                    self._overhead_samples = self._overhead_samples[-500:]
                self._stats['overhead_ms'] = sum(self._overhead_samples) / len(self._overhead_samples)
    # -------------------------------------------------------------- end measure()
    
    # -------------------------------------------------------------- get_active_contexts()
    def get_active_contexts(self) -> Dict[str, TimingContext]:
        """Return currently active timing contexts.

        Returns:
            Dict[str, TimingContext]: Copy of the active context map.
        """
        with self._context_lock:
            return self._active_contexts.copy()
    # -------------------------------------------------------------- end get_active_contexts()
    
    # -------------------------------------------------------------- get_recent_timings()
    def get_recent_timings(self, count: int = 100) -> List[CompletedTiming]:
        """Return the most recent completed timings.

        Args:
            count: Maximum number of records to return (most recent first).

        Returns:
            List[CompletedTiming]: Slice of recent completed timing records.
        """
        with self._completed_lock:
            return self._completed_timings[-count:] if self._completed_timings else []
    # -------------------------------------------------------------- end get_recent_timings()
    
    # -------------------------------------------------------------- get_timing_breakdown()
    def get_timing_breakdown(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Build a hierarchical timing breakdown for a specific operation.

        Args:
            operation_id: Target operation id to expand into a tree.

        Returns:
            Optional[Dict[str, Any]]: Tree of timing data or None if not found.
        """
        recent_timings = self.get_recent_timings(1000)
        
        # Find the target operation
        target_timing = None
        for timing in recent_timings:
            if timing.operation_id == operation_id:
                target_timing = timing
                break
        
        if not target_timing:
            return None
        
        # Build hierarchy
        timing_map = {t.operation_id: t for t in recent_timings}
        
        # ______________________
        # Embeded Functions
        #
        # --------------------------------------------------------------------------------- build_hierarchy()
        def build_hierarchy(timing: CompletedTiming) -> Dict[str, Any]:
            """Recursively construct a breakdown node for the given timing.

            Args:
                timing: The timing record to transform.

            Returns:
                Dict[str, Any]: Node with timing metrics and nested children.
            """
            children_data = []
            for child_id in timing.children:
                if child_id in timing_map:
                    children_data.append(build_hierarchy(timing_map[child_id]))
            
            return {
                'operation_id': timing.operation_id,
                'operation_name': timing.operation_name,
                'duration_ms': timing.duration_ms,
                'start_time': timing.start_time,
                'success': timing.success,
                'metadata': timing.metadata,
                'children': children_data,
                'children_total_ms': sum(child['duration_ms'] for child in children_data),
                'overhead_ms': max(0, timing.duration_ms - sum(child['duration_ms'] for child in children_data))
            }
        # --------------------------------------------------------------------------------- end build_hierarchy()
        
        return build_hierarchy(target_timing)
    # -------------------------------------------------------------- end get_timing_breakdown()
    
    # -------------------------------------------------------------- generate_detailed_breakdown()
    def generate_detailed_breakdown(self) -> DetailedTimingBreakdown:
        """Aggregate recent timings into a structured breakdown model.

        Returns:
            DetailedTimingBreakdown: Populated breakdown with averaged fields.
        """
        recent_timings = self.get_recent_timings(1000)
        
        if not recent_timings:
            return DetailedTimingBreakdown()
        
        # Aggregate timings by operation name
        timing_aggregates = defaultdict(list)
        for timing in recent_timings:
            timing_aggregates[timing.operation_name].append(timing.duration_ms)
        
        # Calculate averages
        averages = {}
        for operation, durations in timing_aggregates.items():
            averages[operation] = sum(durations) / len(durations)
        
        # Create breakdown (mapping operation names to breakdown fields)
        breakdown = DetailedTimingBreakdown()
        
        # Map known operations to breakdown fields
        operation_mapping = {
            'total_request': 'total_request_response_time_ms',
            'request_parsing': 'request_parsing_time_ms',
            'database_query': 'cypher_query_execution_time_ms',
            'neo4j_connection': 'neo4j_connection_acquisition_time_ms',
            'llm_request': 'llm_api_request_time_ms',
            'parallel_coordination': 'parallel_coordination_overhead_ms',
            'cache_lookup': 'cache_lookup_time_ms',
            'fusion_processing': 'fusion_preprocessing_time_ms',
        }
        
        # Set values from aggregated timings
        for operation, field_name in operation_mapping.items():
            if operation in averages:
                setattr(breakdown, field_name, averages[operation])
        
        return breakdown
    # -------------------------------------------------------------- end generate_detailed_breakdown()
    
    # -------------------------------------------------------------- get_statistics()
    def get_statistics(self) -> Dict[str, Any]:
        """Return basic statistics and gauges for the timing collector.

        Returns:
            Dict[str, Any]: Counts, overhead, and uptime seconds.
        """
        return {
            **self._stats,
            'overhead_samples': len(self._overhead_samples),
            'avg_overhead_ms': self._stats['overhead_ms'],
            'collection_uptime_seconds': time.perf_counter() - self._collection_start_time
        }
    # -------------------------------------------------------------- end get_statistics()
    
    # -------------------------------------------------------------- clear_completed_timings()
    def clear_completed_timings(self) -> int:
        """Clear stored completed timings and return the number removed.

        Returns:
            int: Number of records cleared from the buffer.
        """
        with self._completed_lock:
            count = len(self._completed_timings)
            self._completed_timings.clear()
            self._stats['completed_timings'] = 0
            return count
    # -------------------------------------------------------------- end clear_completed_timings()

# ------------------------------------------------------------------------- end class TimingCollector

# ------------------------------------------------------------------------- class TimingTracker
class TimingTracker:
    """Helper class for user interaction during timing measurement."""
    
    def __init__(self, context: TimingContext, collector: TimingCollector):
        self.context = context
        self.collector = collector
    
    # -------------------------------------------------------------- add_metadata()
    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata to the current timing context."""
        self.context.metadata.update(metadata)
    # -------------------------------------------------------------- end add_metadata()
    
    # -------------------------------------------------------------- set_metadata()
    def set_metadata(self, key: str, value: Any) -> None:
        """Set a specific metadata key-value pair."""
        self.context.metadata[key] = value
    # -------------------------------------------------------------- end set_metadata()
    
    @property
    def operation_id(self) -> str:
        """Get the operation ID for this timing context."""
        return self.context.operation_id
    
    @property
    def duration_ms(self) -> int:
        """Get current duration of the timing context."""
        return self.context.duration_ms
    
# ------------------------------------------------------------------------- end class TimingTracker

# __________________________________________________________________________
# Standalone Function Definitions

# __________________________________________________________________________
# Global Collector Instance

_global_timing_collector: Optional[TimingCollector] = None

# --------------------------------------------------------------------------------- get_timing_collector()
def get_timing_collector() -> TimingCollector:
    """
    Get or create the global timing collector instance.
    
    Returns:
        TimingCollector: Global collector instance
    """
    global _global_timing_collector
    if _global_timing_collector is None:
        _global_timing_collector = TimingCollector()
    return _global_timing_collector
# --------------------------------------------------------------------------------- end get_timing_collector()

# --------------------------------------------------------------------------------- initialize_timing_collector()
async def initialize_timing_collector() -> TimingCollector:
    """Initialize the global timing collector.

    Returns:
        Initialized global TimingCollector instance.
    """
    collector = get_timing_collector()
    logger.info("Global timing collector initialized")
    return collector
# --------------------------------------------------------------------------------- end initialize_timing_collector()

# __________________________________________________________________________
# Convenience Functions

# --------------------------------------------------------------------------------- measure_operation()
async def measure_operation(operation_name: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> Any:
    """Convenience function for measuring operations.

    Args:
        operation_name: Name of the operation being measured.
        metadata: Optional initial metadata to attach to the timing context.

    Returns:
        An async context manager suitable for `async with` to measure the operation.
    """
    collector = get_timing_collector()
    return collector.measure(operation_name, metadata)
# --------------------------------------------------------------------------------- end measure_operation()

# --------------------------------------------------------------------------------- get_timing_stats()
def get_timing_stats() -> Dict[str, Any]:
    """Get timing statistics from global collector.

    Returns:
        Mapping of collector statistics including overhead and uptime.
    """
    collector = get_timing_collector()
    return collector.get_statistics()
# --------------------------------------------------------------------------------- end get_timing_stats()

# __________________________________________________________________________
# End of File