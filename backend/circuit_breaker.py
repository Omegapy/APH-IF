# =========================================================================
# File: circuit_breaker.py
# Project: APH-IF Technology Framework
#          Advanced Parallel Hybrid - Intelligent Fusion System
# Author: Alexander Ricciardi
# Date: 2025-08-05
# File Path: backend/circuit_breaker.py
# =========================================================================

"""
Circuit Breaker Pattern Implementation for APH-IF Backend

Provides fault tolerance and resilience for external service calls including
database connections, LLM API calls, and other critical dependencies.
Implements automatic failure detection, circuit opening, and recovery testing.
"""

import time
import logging
import asyncio
from enum import Enum
from typing import Callable, Any, Optional, Dict, Union
from datetime import datetime, timedelta
from functools import wraps

# =========================================================================
# Circuit Breaker States
# =========================================================================
class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Failing, blocking calls
    HALF_OPEN = "half_open" # Testing recovery

# =========================================================================
# Circuit Breaker Exceptions
# =========================================================================
class CircuitBreakerError(Exception):
    """Base exception for circuit breaker errors"""
    pass

class CircuitOpenError(CircuitBreakerError):
    """Raised when circuit is open and blocking calls"""
    pass

class CircuitBreakerTimeoutError(CircuitBreakerError):
    """Raised when operation times out"""
    pass

# =========================================================================
# Circuit Breaker Implementation
# =========================================================================
class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance
    
    Monitors service calls and automatically opens circuit when failure
    threshold is reached, preventing cascading failures.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Union[Exception, tuple] = Exception,
        timeout: Optional[int] = None
    ):
        """
        Initialize circuit breaker
        
        Args:
            name: Circuit breaker identifier
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception types that trigger circuit opening
            timeout: Operation timeout in seconds
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.timeout = timeout
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.total_timeouts = 0
        
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker"""
        if asyncio.iscoroutinefunction(func):
            return self._async_wrapper(func)
        else:
            return self._sync_wrapper(func)
    
    def _sync_wrapper(self, func: Callable) -> Callable:
        """Synchronous function wrapper"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def _async_wrapper(self, func: Callable) -> Callable:
        """Asynchronous function wrapper"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call_async(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        self.total_calls += 1
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._attempt_reset()
            else:
                self._handle_open_circuit()
        
        # Execute function
        start_time = time.time()
        try:
            if self.timeout:
                # TODO: Implement timeout for sync functions
                result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._handle_success()
            return result
            
        except self.expected_exception as e:
            self._handle_failure(e)
            raise
        except Exception as e:
            # Unexpected exceptions don't count as circuit breaker failures
            self.logger.warning(f"Unexpected exception in {self.name}: {e}")
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        self.total_calls += 1
        
        # Check if circuit is open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._attempt_reset()
            else:
                self._handle_open_circuit()
        
        # Execute function
        start_time = time.time()
        try:
            if self.timeout:
                result = await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=self.timeout
                )
            else:
                result = await func(*args, **kwargs)
            
            self._handle_success()
            return result
            
        except asyncio.TimeoutError:
            self.total_timeouts += 1
            self._handle_failure(CircuitBreakerTimeoutError("Operation timed out"))
            raise CircuitBreakerTimeoutError(f"Operation timed out after {self.timeout}s")
        except self.expected_exception as e:
            self._handle_failure(e)
            raise
        except Exception as e:
            # Unexpected exceptions don't count as circuit breaker failures
            self.logger.warning(f"Unexpected exception in {self.name}: {e}")
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.next_attempt_time is None:
            return True
        return datetime.now() >= self.next_attempt_time
    
    def _attempt_reset(self):
        """Attempt to reset circuit to half-open state"""
        self.state = CircuitState.HALF_OPEN
        self.logger.info(f"Circuit breaker {self.name} attempting reset to HALF_OPEN")
    
    def _handle_open_circuit(self):
        """Handle calls when circuit is open"""
        self.logger.warning(f"Circuit breaker {self.name} is OPEN, blocking call")
        raise CircuitOpenError(f"Circuit breaker {self.name} is open")
    
    def _handle_success(self):
        """Handle successful function execution"""
        self.total_successes += 1
        self.last_success_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self._close_circuit()
        
        # Reset failure count on success
        self.failure_count = 0
    
    def _handle_failure(self, exception: Exception):
        """Handle failed function execution"""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        self.logger.warning(
            f"Circuit breaker {self.name} recorded failure "
            f"({self.failure_count}/{self.failure_threshold}): {exception}"
        )
        
        if self.failure_count >= self.failure_threshold:
            self._open_circuit()
    
    def _open_circuit(self):
        """Open the circuit breaker"""
        self.state = CircuitState.OPEN
        self.next_attempt_time = datetime.now() + timedelta(seconds=self.recovery_timeout)
        
        self.logger.error(
            f"Circuit breaker {self.name} opened due to {self.failure_count} failures. "
            f"Next attempt at {self.next_attempt_time}"
        )
    
    def _close_circuit(self):
        """Close the circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.next_attempt_time = None
        
        self.logger.info(f"Circuit breaker {self.name} closed - normal operation resumed")
    
    def reset(self):
        """Manually reset circuit breaker to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.next_attempt_time = None
        self.logger.info(f"Circuit breaker {self.name} manually reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_timeouts": self.total_timeouts,
            "success_rate": self.total_successes / max(self.total_calls, 1),
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "next_attempt_time": self.next_attempt_time.isoformat() if self.next_attempt_time else None
        }

# =========================================================================
# Circuit Breaker Manager
# =========================================================================
class CircuitBreakerManager:
    """Manages multiple circuit breakers"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger("circuit_breaker_manager")
    
    def create_circuit_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Union[Exception, tuple] = Exception,
        timeout: Optional[int] = None
    ) -> CircuitBreaker:
        """Create and register a new circuit breaker"""
        if name in self.circuit_breakers:
            self.logger.warning(f"Circuit breaker {name} already exists, returning existing instance")
            return self.circuit_breakers[name]
        
        circuit_breaker = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            timeout=timeout
        )
        
        self.circuit_breakers[name] = circuit_breaker
        self.logger.info(f"Created circuit breaker: {name}")
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers"""
        return {
            name: cb.get_stats() 
            for name, cb in self.circuit_breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers"""
        for cb in self.circuit_breakers.values():
            cb.reset()
        self.logger.info("All circuit breakers reset")
    
    def get_health_status(self) -> Dict[str, str]:
        """Get health status of all circuit breakers"""
        return {
            name: "healthy" if cb.state == CircuitState.CLOSED else "unhealthy"
            for name, cb in self.circuit_breakers.items()
        }

# =========================================================================
# Global Circuit Breaker Manager Instance
# =========================================================================
_circuit_breaker_manager = CircuitBreakerManager()

def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager instance"""
    return _circuit_breaker_manager
