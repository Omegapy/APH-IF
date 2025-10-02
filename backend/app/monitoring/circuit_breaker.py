# -------------------------------------------------------------------------
# File: circuit_breaker.py
# Author: Alexander Ricciardi
# Date: 2025-09-18
# [File Path] backend/app/monitoring/circuit_breaker.py
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
#   Circuit breaker implementation for backend resilience. Provides configurable
#   failure detection, slow-call tracking, half-open recovery testing, a global
#   registry, and health-check integration for monitoring endpoints.
# -------------------------------------------------------------------------
#
# --- Module Contents Overview ---
# - Enum: CircuitState
# - Dataclass: CircuitBreakerConfig
# - Dataclass: CircuitEvent
# - Dataclass: CallResult
# - Class: CircuitBreaker
# - Class: CircuitBreakerRegistry
# - Exceptions: CircuitBreakerOpen, CircuitBreakerTimeout
# - Functions: get_circuit_breaker, circuit_breaker, circuit_breaker_health_check
# -------------------------------------------------------------------------
#
# --- Dependencies / Imports ---
# - Standard Library: asyncio, time, logging, datetime, enum, collections.deque, statistics,
#   typing, dataclasses
# - Local Project Modules: Integrated by monitoring package and backend health endpoints
# --- Requirements ---
# - Python 3.12
# -------------------------------------------------------------------------
#
# --- Usage / Integration ---
# Use the decorator `@circuit_breaker("name")` on async operations that call
# external services. The global registry surfaces metrics to monitoring routes
# and the health-check function aggregates status and recommendations.
#
# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""
Circuit Breaker Implementation for APH-IF Backend

Advanced circuit breaker pattern implementation for resilience in the
parallel hybrid RAG system with intelligent failure detection and recovery.
"""

# __________________________________________________________________________
# Imports

from __future__ import annotations

import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional

# __________________________________________________________________________
# Global Constants / Variables

logger = logging.getLogger(__name__)

# ____________________________________________________________________________
# Class Definitions

# =========================================================================
# Circuit Breaker States
# =========================================================================

# ------------------------------------------------------------------------- class CircuitState
class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open" # Testing recovery
# ------------------------------------------------------------------------- end class CircuitState

# =========================================================================
# Circuit Breaker Configuration
# =========================================================================

# ------------------------------------------------------------------------- class CircuitBreakerConfig
@dataclass(slots=True)
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    # Failure thresholds
    failure_threshold: int = 5              # Failures before opening
    success_threshold: int = 3              # Successes to close from half-open
    timeout_threshold_ms: int = 240000      # Timeout considered as failure
    
    # Timing configuration
    recovery_timeout: int = 240             # Seconds to wait before half-open
    half_open_max_calls: int = 3            # Max calls in half-open state
    
    # Monitoring window
    monitoring_period: int = 300            # 5 minutes rolling window
    min_calls_to_evaluate: int = 10         # Minimum calls before evaluation
    
    # Advanced configuration  
    slow_call_rate_threshold: float = 0.5   # Rate of slow calls to trigger
    slow_call_duration_ms: int = 10000      # Duration considered "slow"
    
    # Custom failure detection
    custom_failure_conditions: Optional[Callable[[Any], bool]] = None
# ------------------------------------------------------------------------- end class CircuitBreakerConfig

# =========================================================================
# Circuit Breaker Events
# =========================================================================

# ------------------------------------------------------------------------- class CircuitEvent
@dataclass(slots=True)
class CircuitEvent:
    """Circuit breaker state change event."""
    timestamp: float
    old_state: CircuitState
    new_state: CircuitState
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
# ------------------------------------------------------------------------- end class CircuitEvent

# ------------------------------------------------------------------------- class CallResult
@dataclass(slots=True)
class CallResult:
    """Result of a circuit breaker protected call."""
    success: bool
    duration_ms: int
    timestamp: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
# ------------------------------------------------------------------------- end class CallResult

# =========================================================================
# Circuit Breaker Implementation
# =========================================================================

# ------------------------------------------------------------------------- class CircuitBreaker
class CircuitBreaker:
    """
    Advanced circuit breaker implementation for fault tolerance.
    
    Features:
    - Configurable failure detection
    - Slow call rate monitoring
    - Half-open state recovery testing
    - Comprehensive event logging
    - Custom failure condition support
    """
    
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.

        Args:
            name: Human-readable identifier for this breaker instance.
            config: Optional configuration; defaults to sensible thresholds and timings.
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logging.getLogger(__name__)
        
        # State management
        self._state = CircuitState.CLOSED
        self._last_failure_time = 0.0
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._half_open_calls = 0
        
        # Call history (sliding window)
        self._call_history: deque[CallResult] = deque(maxlen=1000)
        self._event_history: deque[CircuitEvent] = deque(maxlen=100)
        
        # Statistics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        
        self.logger.info(f"Circuit breaker '{name}' initialized: {self._state.value}")
    # --------------------------------------------------------------------------------- end __init__()
    
    # --------------------------------------------------------------------------------- state()
    @property
    def state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._state
    # --------------------------------------------------------------------------------- end state()
    
    # --------------------------------------------------------------------------------- is_closed()
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    # --------------------------------------------------------------------------------- end is_closed()
    
    # --------------------------------------------------------------------------------- is_open()
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN
    # --------------------------------------------------------------------------------- end is_open()
    
    # --------------------------------------------------------------------------------- is_half_open()
    @property  
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (recovery testing)."""
        return self._state == CircuitState.HALF_OPEN
    # --------------------------------------------------------------------------------- end is_half_open()
    
    # --------------------------------------------------------------------------------- _transition_to()
    def _transition_to(
        self,
        new_state: CircuitState,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Transition circuit breaker to a new state.

        Args:
            new_state: Target `CircuitState` to transition into.
            reason: Short description of why the transition occurs.
            metadata: Optional contextual metadata to attach to the transition event.
        """
        if self._state == new_state:
            return
        
        old_state = self._state
        self._state = new_state
        
        # Log state transition
        event = CircuitEvent(
            timestamp=time.time(),
            old_state=old_state,
            new_state=new_state,
            reason=reason,
            metadata=metadata or {}
        )
        self._event_history.append(event)
        
        # Reset counters based on new state
        if new_state == CircuitState.OPEN:
            self._last_failure_time = time.time()
            self._consecutive_successes = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        elif new_state == CircuitState.CLOSED:
            self._consecutive_failures = 0
            self._consecutive_successes = 0
        
        self.logger.info(f"Circuit '{self.name}' transitioned: {old_state.value} -> {new_state.value} ({reason})")
    # --------------------------------------------------------------------------------- end _transition_to()
    
    # --------------------------------------------------------------------------------- _should_allow_call()
    def _should_allow_call(self) -> bool:
        """Check if call should be allowed based on current state."""
        current_time = time.time()
        
        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if current_time - self._last_failure_time >= self.config.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN, "recovery timeout elapsed")
                return True
            return False
        elif self._state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            return self._half_open_calls < self.config.half_open_max_calls
        
        return False
    # --------------------------------------------------------------------------------- end _should_allow_call()
    
    # --------------------------------------------------------------------------------- _evaluate_call_result()
    def _evaluate_call_result(self, result: CallResult) -> None:
        """Evaluate call result and update circuit state if needed."""
        self._call_history.append(result)
        self._total_calls += 1
        
        if result.success:
            self._total_successes += 1
            self._consecutive_successes += 1
            self._consecutive_failures = 0
            
            # Handle success in different states
            if self._state == CircuitState.HALF_OPEN:
                if self._consecutive_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED, "success threshold reached")
                    
        else:
            self._total_failures += 1
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            
            # Handle failure in different states
            if self._state == CircuitState.CLOSED:
                if self._should_open_circuit():
                    self._transition_to(CircuitState.OPEN, "failure threshold exceeded")
            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN, "failure in half-open state")
    
    # --------------------------------------------------------------------------------- end _evaluate_call_result()
    
    # --------------------------------------------------------------------------------- _should_open_circuit()
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened based on recent call history."""
        # Check consecutive failures
        if self._consecutive_failures >= self.config.failure_threshold:
            return True
        
        # Check recent call pattern (sliding window)
        recent_calls = self._get_recent_calls()
        if len(recent_calls) < self.config.min_calls_to_evaluate:
            return False
        
        # Calculate failure rate
        recent_failures = [call for call in recent_calls if not call.success]
        failure_rate = len(recent_failures) / len(recent_calls)
        
        if failure_rate >= 0.5:  # 50% failure rate
            return True
        
        # Check slow call rate
        slow_calls = [call for call in recent_calls 
                     if call.duration_ms > self.config.slow_call_duration_ms]
        slow_call_rate = len(slow_calls) / len(recent_calls)
        
        if slow_call_rate >= self.config.slow_call_rate_threshold:
            return True
        
        return False
    # --------------------------------------------------------------------------------- end _should_open_circuit()
   
    # --------------------------------------------------------------------------------- _get_recent_calls()
    def _get_recent_calls(self) -> list[CallResult]:
        """Get calls within the monitoring period."""
        cutoff_time = time.time() - self.config.monitoring_period
        return [call for call in self._call_history if call.timestamp >= cutoff_time]
    
    # --------------------------------------------------------------------------------- end _get_recent_calls()
    
    # --------------------------------------------------------------------------------- _is_failure()
    def _is_failure(self, result: Any, error: Optional[Exception], duration_ms: int) -> tuple[bool, str]:
        """Determine if a call result should be considered a failure."""
        # Timeout failure
        if duration_ms >= self.config.timeout_threshold_ms:
            return True, f"timeout ({duration_ms}ms >= {self.config.timeout_threshold_ms}ms)"
        
        # Exception failure
        if error is not None:
            return True, f"exception: {type(error).__name__}: {str(error)}"
        
        # Custom failure conditions
        if self.config.custom_failure_conditions:
            try:
                if self.config.custom_failure_conditions(result):
                    return True, "custom failure condition"
            except Exception as e:
                self.logger.warning(f"Error in custom failure condition: {e}")
        
        return False, "success"
    # --------------------------------------------------------------------------------- end _is_failure()
    
    # --------------------------------------------------------------------------------- __call__()
    async def __call__(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute an async function with circuit breaker protection.

        Args:
            func: Async function to protect.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Any: The function's result if execution is allowed and succeeds.

        Raises:
            CircuitBreakerOpen: If the circuit is open and calls are blocked.
            Exception: Re-raises the original exception from the protected call.
        """
        return await self.call(func, *args, **kwargs)
    # --------------------------------------------------------------------------------- end __call__()
    
    # --------------------------------------------------------------------------------- call()
    async def call(self, func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Execute an async function with circuit breaker protection.

        Args:
            func: Async function to protect.
            *args: Positional arguments passed to the function.
            **kwargs: Keyword arguments passed to the function.

        Returns:
            Any: The function's result if execution is allowed and succeeds.

        Raises:
            CircuitBreakerOpen: If the circuit is open and calls are blocked.
            Exception: Re-raises the original exception from the protected call.
        """
        # Check if call should be allowed
        if not self._should_allow_call():
            raise CircuitBreakerOpen(f"Circuit breaker '{self.name}' is {self._state.value}")
        
        # Track call in half-open state
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
        
        # Execute the function with timing
        start_time = time.time()
        error = None
        result = None
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            # Record call result
            duration_ms = int((time.time() - start_time) * 1000)
            is_failure, failure_reason = self._is_failure(result, error, duration_ms)
            
            call_result = CallResult(
                success=not is_failure,
                duration_ms=duration_ms,
                timestamp=time.time(),
                error=failure_reason if is_failure else None,
                metadata={
                    "function": func.__name__,
                    "state_during_call": self._state.value,
                },
            )
            
            self._evaluate_call_result(call_result)
    # --------------------------------------------------------------------------------- end call()
    
    # --------------------------------------------------------------------------------- get_metrics()
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics and statistics.

        Returns:
            Dict[str, Any]: Snapshot including state, counts, recent rates and durations,
            configuration, and time since last state change.
        """
        recent_calls = self._get_recent_calls()
        
        # Calculate recent statistics
        if recent_calls:
            recent_success_rate = len([c for c in recent_calls if c.success]) / len(recent_calls) * 100
            recent_avg_duration = statistics.mean([c.duration_ms for c in recent_calls])
            recent_failure_rate = len([c for c in recent_calls if not c.success]) / len(recent_calls) * 100
        else:
            recent_success_rate = 0.0
            recent_avg_duration = 0.0
            recent_failure_rate = 0.0
        
        return {
            "name": self.name,
            "state": self._state.value,
            "state_duration_seconds": time.time()
            - (self._event_history[-1].timestamp if self._event_history else time.time()),
            "total_calls": self._total_calls,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "consecutive_failures": self._consecutive_failures,
            "consecutive_successes": self._consecutive_successes,
            "overall_success_rate": (self._total_successes / self._total_calls * 100)
            if self._total_calls > 0
            else 0.0,
            "recent_metrics": {
                "calls_in_period": len(recent_calls),
                "success_rate": recent_success_rate,
                "failure_rate": recent_failure_rate,
                "avg_duration_ms": recent_avg_duration,
            },
            "configuration": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "monitoring_period": self.config.monitoring_period,
            },
            "last_state_change": self._event_history[-1].timestamp if self._event_history else None,
        }
    # --------------------------------------------------------------------------------- end get_metrics()
    
    # --------------------------------------------------------------------------------- get_recent_events()
    def get_recent_events(self, count: int = 10) -> list[CircuitEvent]:
        """Get recent circuit breaker events.

        Args:
            count: Maximum number of most-recent events to return.

        Returns:
            list[CircuitEvent]: Slice of the most recent circuit breaker events.
        """
        return list(self._event_history)[-count:]
    # --------------------------------------------------------------------------------- end get_recent_events()
    
    # --------------------------------------------------------------------------------- reset()
    def reset(self) -> None:
        """Reset circuit breaker to closed state and clear history.

        Returns:
            None
        """
        self._transition_to(CircuitState.CLOSED, "manual reset")
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._half_open_calls = 0
        self._call_history.clear()
        
        self.logger.info(f"Circuit breaker '{self.name}' manually reset")
    # --------------------------------------------------------------------------------- end reset()
    
# ------------------------------------------------------------------------- end class CircuitBreaker

# =========================================================================
# Circuit Breaker Exceptions
# =========================================================================

# ------------------------------------------------------------------------- class CircuitBreakerOpen
class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass
# ------------------------------------------------------------------------- end class CircuitBreakerOpen

# ------------------------------------------------------------------------- class CircuitBreakerTimeout
class CircuitBreakerTimeout(Exception):
    """Exception raised when call exceeds timeout threshold."""
    pass
# ------------------------------------------------------------------------- end class CircuitBreakerTimeout

# =========================================================================
# Circuit Breaker Registry
# =========================================================================

# ------------------------------------------------------------------------- class CircuitBreakerRegistry
class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger(__name__)
    # --------------------------------------------------------------------------------- end __init__()
    
    # --------------------------------------------------------------------------------- get_or_create()
    def get_or_create(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get an existing circuit breaker or create a new one.

        Args:
            name: Unique breaker name.
            config: Optional configuration to use when creating a new breaker.

        Returns:
            CircuitBreaker: Retrieved or newly created breaker instance.
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config)
            self.logger.info(f"Created circuit breaker: {name}")
        return self._breakers[name]
    # --------------------------------------------------------------------------------- end get_or_create()
    
    # --------------------------------------------------------------------------------- get_all_metrics()
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all registered circuit breakers.
        
        Returns:
            Dict[str, Dict[str, Any]]: Mapping of breaker name to metrics snapshot.
        """
        return {name: breaker.get_metrics() for name, breaker in self._breakers.items()}
    # --------------------------------------------------------------------------------- end get_all_metrics()
    
    # --------------------------------------------------------------------------------- reset_all()
    def reset_all(self) -> None:
        """Reset all circuit breakers to a closed state and clear histories."""
        for breaker in self._breakers.values():
            breaker.reset()
        self.logger.info("All circuit breakers reset")
    # --------------------------------------------------------------------------------- end reset_all()
    
    # --------------------------------------------------------------------------------- get_summary()
    def get_summary(self) -> Dict[str, Any]:
        """Get a compact summary of the registry's state.

        Returns:
            Dict[str, Any]: Total breakers, state distribution, overall health, and breaker names.
        """
        total_breakers = len(self._breakers)
        open_breakers = len([b for b in self._breakers.values() if b.is_open])
        half_open_breakers = len([b for b in self._breakers.values() if b.is_half_open])
        closed_breakers = len([b for b in self._breakers.values() if b.is_closed])
        
        return {
            "total_breakers": total_breakers,
            "state_distribution": {
                "open": open_breakers,
                "half_open": half_open_breakers,
                "closed": closed_breakers,
            },
            "health_status": "healthy"
            if open_breakers == 0
            else "degraded"
            if open_breakers < total_breakers
            else "critical",
            "breakers": list(self._breakers.keys()),
        }
    # --------------------------------------------------------------------------------- end get_summary()

# ------------------------------------------------------------------------- end class CircuitBreakerRegistry

# __________________________________________________________________________
# Standalone Function Definitions
#
# =========================================================================
# Global Registry and Decorators
# =========================================================================

_global_registry = CircuitBreakerRegistry()

# --------------------------------------------------------------------------------- get_circuit_breaker()
def get_circuit_breaker(
    name: str, config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry.

    Args:
        name: Unique breaker name.
        config: Optional breaker configuration for creation.

    Returns:
        CircuitBreaker: Registry-backed breaker instance.
    """
    return _global_registry.get_or_create(name, config)
# --------------------------------------------------------------------------------- end get_circuit_breaker()

# --------------------------------------------------------------------------------- circuit_breaker()
def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for protecting async functions with a circuit breaker.

    Args:
        name: Breaker name used in the registry.
        config: Optional configuration to apply when creating the breaker.

    Returns:
        Callable: A decorator that wraps an async function with breaker protection.
    """
    # ______________________
    # Embedded Function
    #
    # --------------------------------------------------------------------------------- decorator()
    def decorator(func):
        """Create a function wrapper bound to a named circuit breaker."""
        breaker = get_circuit_breaker(name, config)

        # ______________________
        # Embedded Function
        #
        # --------------------------------------------------------------------------------- wrapper()
        async def wrapper(*args, **kwargs):
            """Route the call through the breaker and return the result."""
            return await breaker.call(func, *args, **kwargs)
        # --------------------------------------------------------------------------------- end wrapper()

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.circuit_breaker = breaker
        return wrapper
    # --------------------------------------------------------------------------------- end decorator()

    return decorator
# --------------------------------------------------------------------------------- end circuit_breaker()

# =========================================================================
# Health Check Integration
# =========================================================================

# --------------------------------------------------------------------------------- circuit_breaker_health_check()
async def circuit_breaker_health_check() -> Dict[str, Any]:
    """Perform health check on all circuit breakers.

    Returns:
        Dict[str, Any]: Overall status, registry summary, detailed breaker metrics,
        and actionable recommendations.
    """
    summary = _global_registry.get_summary()
    all_metrics = _global_registry.get_all_metrics()
    
    # Calculate overall health
    if summary['state_distribution']['open'] == 0:
        if summary['state_distribution']['half_open'] == 0:
            status = "excellent"
        else:
            status = "recovering"
    else:
        open_ratio = summary['state_distribution']['open'] / summary['total_breakers']
        if open_ratio < 0.3:
            status = "degraded"
        else:
            status = "critical"
    
    return {
        'tool_name': 'circuit_breakers',
        'status': status,
        'summary': summary,
        'detailed_metrics': all_metrics,
        'recommendations': _generate_circuit_recommendations(summary, all_metrics)
    }
# --------------------------------------------------------------------------------- end circuit_breaker_health_check()

# --------------------------------------------------------------------------------- _generate_circuit_recommendations()
def _generate_circuit_recommendations(
    summary: Dict[str, Any], metrics: Dict[str, Dict[str, Any]]
) -> list[str]:
    """Generate recommendations based on circuit breaker status.

    Args:
        summary: Registry summary from `CircuitBreakerRegistry.get_summary`.
        metrics: Per-breaker metrics from `CircuitBreakerRegistry.get_all_metrics`.

    Returns:
        list[str]: Recommendations for improving resilience.
    """
    recommendations = []
    
    if summary['state_distribution']['open'] > 0:
        recommendations.append(f"{summary['state_distribution']['open']} circuit breakers are open - investigate underlying issues")
    
    if summary['state_distribution']['half_open'] > 0:
        recommendations.append(f"{summary['state_distribution']['half_open']} circuit breakers are recovering - monitor closely")
    
    # Check for frequently failing breakers
    frequent_failures = []
    for name, metric in metrics.items():
        if metric['recent_metrics']['failure_rate'] > 30:
            frequent_failures.append(name)
    
    if frequent_failures:
        recommendations.append(f"High failure rates detected in: {', '.join(frequent_failures)}")
    
    if summary['total_breakers'] == 0:
        recommendations.append("No circuit breakers configured - consider adding protection for critical operations")
    
    if not recommendations:
        recommendations.append("All circuit breakers are operating normally")
    
    return recommendations
# --------------------------------------------------------------------------------- end _generate_circuit_recommendations()

# __________________________________________________________________________
# End of File