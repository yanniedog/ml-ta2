#!/usr/bin/env python3
"""
Circuit Breaker Pattern Implementation for ML-TA System

This module provides a robust circuit breaker pattern to prevent cascading failures
when external services (like Binance API) become unavailable or unreliable.
"""

import time
import threading
from enum import Enum
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, rejecting requests  
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5           # Number of failures before opening
    recovery_timeout: float = 60.0       # Seconds before trying half-open
    success_threshold: int = 3           # Successes needed to close from half-open
    timeout: float = 30.0                # Request timeout
    expected_exception: tuple = (Exception,)  # Exceptions that count as failures


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracking for circuit breaker."""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    state_changes: int = 0


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, message: str, state: CircuitState, last_failure: Optional[str] = None):
        super().__init__(message)
        self.state = state
        self.last_failure = last_failure


class CircuitBreaker:
    """
    Circuit breaker implementation following the pattern described by Martin Fowler.
    
    The circuit breaker monitors the failure rate of calls to external services.
    When failures exceed a threshold, the circuit opens and fast-fails subsequent
    calls. After a timeout period, it enters half-open state to test recovery.
    """
    
    def __init__(self, 
                 name: str,
                 config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            config: Configuration parameters
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.metrics = CircuitBreakerMetrics()
        
        # State management
        self._state = CircuitState.CLOSED
        self._last_failure_exception: Optional[Exception] = None
        self._lock = threading.RLock()
        
        # Logging
        self.logger = logger.bind(circuit_breaker=name)
        
        self.logger.info("Circuit breaker initialized", 
                        failure_threshold=self.config.failure_threshold,
                        recovery_timeout=self.config.recovery_timeout)
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset from open to half-open."""
        return (time.time() - self.metrics.last_failure_time) >= self.config.recovery_timeout
    
    def _transition_to_open(self, exception: Exception) -> None:
        """Transition circuit to open state."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                self._state = CircuitState.OPEN
                self._last_failure_exception = exception
                self.metrics.state_changes += 1
                
                self.logger.warning("Circuit breaker opened",
                                  failure_count=self.metrics.failure_count,
                                  exception=str(exception))
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit to half-open state."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                self._state = CircuitState.HALF_OPEN
                self.metrics.success_count = 0  # Reset success count for half-open test
                self.metrics.state_changes += 1
                
                self.logger.info("Circuit breaker entering half-open state")
    
    def _transition_to_closed(self) -> None:
        """Transition circuit to closed state."""
        with self._lock:
            if self._state != CircuitState.CLOSED:
                self._state = CircuitState.CLOSED
                self.metrics.failure_count = 0  # Reset failure count
                self.metrics.state_changes += 1
                
                self.logger.info("Circuit breaker closed",
                               success_count=self.metrics.success_count)
    
    def _record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self.metrics.success_count += 1
            self.metrics.total_successes += 1
            self.metrics.last_success_time = time.time()
            
            # If in half-open state, check if we should close
            if self._state == CircuitState.HALF_OPEN:
                if self.metrics.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
    
    def _record_failure(self, exception: Exception) -> None:
        """Record a failed operation."""
        with self._lock:
            self.metrics.failure_count += 1
            self.metrics.total_failures += 1
            self.metrics.last_failure_time = time.time()
            
            # Check if we should open the circuit
            if self._state == CircuitState.CLOSED:
                if self.metrics.failure_count >= self.config.failure_threshold:
                    self._transition_to_open(exception)
            elif self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open state opens the circuit
                self._transition_to_open(exception)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call a function through the circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function
        
        Returns:
            Result of function call
        
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception raised by the function
        """
        with self._lock:
            self.metrics.total_requests += 1
            
            # Check circuit state
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    # Circuit is open, fail fast
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN",
                        state=self._state,
                        last_failure=str(self._last_failure_exception) if self._last_failure_exception else None
                    )
        
        # Attempt the call
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
            
        except self.config.expected_exception as e:
            self._record_failure(e)
            raise
        except Exception as e:
            # Unexpected exceptions don't count as circuit breaker failures
            self.logger.warning("Unexpected exception in circuit breaker",
                              exception=str(e),
                              exception_type=type(e).__name__)
            raise
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator interface for circuit breaker.
        
        Args:
            func: Function to wrap
        
        Returns:
            Wrapped function
        """
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        wrapper.__name__ = f"circuit_breaker_{func.__name__}"
        wrapper.__doc__ = f"Circuit breaker wrapped: {func.__doc__}"
        return wrapper
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get circuit breaker metrics.
        
        Returns:
            Dictionary with current metrics
        """
        with self._lock:
            return {
                'name': self.name,
                'state': self._state.value,
                'failure_count': self.metrics.failure_count,
                'success_count': self.metrics.success_count,
                'total_requests': self.metrics.total_requests,
                'total_failures': self.metrics.total_failures,
                'total_successes': self.metrics.total_successes,
                'state_changes': self.metrics.state_changes,
                'last_failure_time': self.metrics.last_failure_time,
                'last_success_time': self.metrics.last_success_time,
                'failure_rate': (self.metrics.total_failures / max(1, self.metrics.total_requests)) * 100,
                'uptime_seconds': time.time() - self.metrics.last_failure_time if self.metrics.last_failure_time > 0 else 0
            }
    
    def reset(self) -> None:
        """
        Manually reset circuit breaker to closed state.
        
        This should be used carefully, typically only for testing
        or manual recovery operations.
        """
        with self._lock:
            self._state = CircuitState.CLOSED
            self.metrics.failure_count = 0
            self.metrics.success_count = 0
            self.metrics.state_changes += 1
            
            self.logger.info("Circuit breaker manually reset")


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_or_create(self, 
                     name: str, 
                     config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """
        Get existing circuit breaker or create new one.
        
        Args:
            name: Circuit breaker name
            config: Configuration (only used for new breakers)
        
        Returns:
            Circuit breaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all registered circuit breakers.
        
        Returns:
            Dictionary mapping names to metrics
        """
        with self._lock:
            return {name: breaker.get_metrics() 
                   for name, breaker in self._breakers.items()}
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str, 
                       config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """
    Get or create a circuit breaker.
    
    Args:
        name: Circuit breaker name
        config: Configuration for new breakers
    
    Returns:
        Circuit breaker instance
    """
    return _registry.get_or_create(name, config)


def circuit_breaker(name: str, 
                   config: Optional[CircuitBreakerConfig] = None) -> Callable:
    """
    Decorator for applying circuit breaker pattern.
    
    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration
    
    Returns:
        Decorator function
    """
    breaker = get_circuit_breaker(name, config)
    
    def decorator(func: Callable) -> Callable:
        return breaker(func)
    
    return decorator


def get_all_circuit_breaker_metrics() -> Dict[str, Dict[str, Any]]:
    """
    Get metrics for all circuit breakers.
    
    Returns:
        Dictionary mapping names to metrics
    """
    return _registry.get_all_metrics()


# Example usage
if __name__ == "__main__":
    import random
    
    # Create circuit breaker with custom config
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=10.0,
        success_threshold=2
    )
    
    @circuit_breaker("example_service", config)
    def flaky_service(success_rate: float = 0.5) -> str:
        """Simulate a flaky external service."""
        if random.random() < success_rate:
            return "Success!"
        else:
            raise Exception("Service unavailable")
    
    # Test the circuit breaker
    for i in range(20):
        try:
            result = flaky_service(0.3)  # 30% success rate
            print(f"Call {i+1}: {result}")
        except Exception as e:
            print(f"Call {i+1}: Failed - {e}")
        
        time.sleep(0.5)
    
    # Print final metrics
    metrics = get_all_circuit_breaker_metrics()
    print("\nFinal metrics:")
    for name, stats in metrics.items():
        print(f"{name}: {stats}")
