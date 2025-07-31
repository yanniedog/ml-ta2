"""
Comprehensive Error Handling for ML-TA System

This module provides a custom exception hierarchy and error handling framework
with recovery strategies, retry mechanisms, and graceful degradation.
"""

import time
import random
import logging
import traceback
from typing import Dict, Any, Optional, Callable, Type, Union
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps
import structlog

logger = structlog.get_logger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA = "data"
    MODEL = "model"
    API = "api"
    CONFIG = "config"
    SECURITY = "security"
    SYSTEM = "system"
    VALIDATION = "validation"
    NETWORK = "network"


@dataclass
class ErrorContext:
    """Context information for errors."""
    operation: str
    component: str
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


class MLTAException(Exception):
    """Base exception class for ML-TA system."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "MLTA_ERROR",
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True
    ):
        """Initialize ML-TA exception."""
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.category = category
        self.context = context or ErrorContext("unknown", "unknown")
        self.cause = cause
        self.recoverable = recoverable
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp,
            "context": {
                "operation": self.context.operation,
                "component": self.context.component,
                "correlation_id": self.context.correlation_id,
                "user_id": self.context.user_id,
                "additional_data": self.context.additional_data
            },
            "cause": str(self.cause) if self.cause else None,
            "traceback": traceback.format_exc()
        }


class DataFetchError(MLTAException):
    """Exception for data fetching failures."""
    
    def __init__(self, message: str, source: str = "unknown", **kwargs):
        super().__init__(
            message,
            error_code="DATA_FETCH_ERROR",
            category=ErrorCategory.DATA,
            **kwargs
        )
        self.source = source


class DataQualityError(MLTAException):
    """Exception for data quality issues."""
    
    def __init__(self, message: str, quality_check: str = "unknown", **kwargs):
        super().__init__(
            message,
            error_code="DATA_QUALITY_ERROR",
            category=ErrorCategory.DATA,
            **kwargs
        )
        self.quality_check = quality_check


class FeatureEngineeringError(MLTAException):
    """Exception for feature engineering failures."""
    
    def __init__(self, message: str, feature_name: str = "unknown", **kwargs):
        super().__init__(
            message,
            error_code="FEATURE_ENGINEERING_ERROR",
            category=ErrorCategory.DATA,
            **kwargs
        )
        self.feature_name = feature_name


class ModelTrainingError(MLTAException):
    """Exception for model training failures."""
    
    def __init__(self, message: str, model_type: str = "unknown", **kwargs):
        super().__init__(
            message,
            error_code="MODEL_TRAINING_ERROR",
            category=ErrorCategory.MODEL,
            **kwargs
        )
        self.model_type = model_type


class PredictionError(MLTAException):
    """Exception for prediction failures."""
    
    def __init__(self, message: str, model_id: str = "unknown", **kwargs):
        super().__init__(
            message,
            error_code="PREDICTION_ERROR",
            category=ErrorCategory.MODEL,
            **kwargs
        )
        self.model_id = model_id


class ConfigurationError(MLTAException):
    """Exception for configuration issues."""
    
    def __init__(self, message: str, config_key: str = "unknown", **kwargs):
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            category=ErrorCategory.CONFIG,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs
        )
        self.config_key = config_key


class SecurityError(MLTAException):
    """Exception for security violations."""
    
    def __init__(self, message: str, violation_type: str = "unknown", **kwargs):
        super().__init__(
            message,
            error_code="SECURITY_ERROR",
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            **kwargs
        )
        self.violation_type = violation_type


class ValidationError(MLTAException):
    """Exception for data validation failures."""
    
    def __init__(self, message: str, validation_rule: str = "unknown", **kwargs):
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            category=ErrorCategory.VALIDATION,
            **kwargs
        )
        self.validation_rule = validation_rule


class APIError(MLTAException):
    """Exception for API-related failures."""
    
    def __init__(self, message: str, status_code: int = 500, endpoint: str = "unknown", **kwargs):
        super().__init__(
            message,
            error_code="API_ERROR",
            category=ErrorCategory.API,
            **kwargs
        )
        self.status_code = status_code
        self.endpoint = endpoint


class NetworkError(MLTAException):
    """Exception for network-related failures."""
    
    def __init__(self, message: str, url: str = "unknown", **kwargs):
        super().__init__(
            message,
            error_code="NETWORK_ERROR",
            category=ErrorCategory.NETWORK,
            **kwargs
        )
        self.url = url


class SystemResourceError(MLTAException):
    """Exception for system resource limitations."""
    
    def __init__(self, message: str, resource_type: str = "unknown", **kwargs):
        super().__init__(
            message,
            error_code="SYSTEM_RESOURCE_ERROR",
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.resource_type = resource_type


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_factor: float = 1.0


class RetryManager:
    """Manages retry logic with exponential backoff and jitter."""
    
    def __init__(self, config: RetryConfig = None):
        """Initialize retry manager."""
        self.config = config or RetryConfig()
        self.logger = logger.bind(component="retry_manager")
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        delay = min(delay * self.config.backoff_factor, self.config.max_delay)
        
        if self.config.jitter:
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0.1, 0.9)
            delay *= jitter
        
        return delay
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.config.max_attempts:
            return False
        
        # Don't retry certain types of errors
        if isinstance(exception, (ConfigurationError, SecurityError, ValidationError)):
            return False
        
        # Don't retry if explicitly marked as non-recoverable
        if isinstance(exception, MLTAException) and not exception.recoverable:
            return False
        
        return True
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                self.logger.debug(f"Attempting operation (attempt {attempt}/{self.config.max_attempts})")
                return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    self.logger.error(f"Operation failed permanently", exception=str(e), attempt=attempt)
                    raise
                
                if attempt < self.config.max_attempts:
                    delay = self.calculate_delay(attempt)
                    self.logger.warning(
                        f"Operation failed, retrying in {delay:.2f}s",
                        exception=str(e),
                        attempt=attempt,
                        delay=delay
                    )
                    time.sleep(delay)
        
        # If we get here, all attempts failed
        self.logger.error(f"All retry attempts exhausted", exception=str(last_exception))
        raise last_exception


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.logger = logger.bind(component="circuit_breaker")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
                self.logger.info("Circuit breaker entering half-open state")
            else:
                raise SystemResourceError(
                    "Circuit breaker is open",
                    context=ErrorContext("circuit_breaker", "protection")
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        if self.state == "half-open":
            self.state = "closed"
            self.logger.info("Circuit breaker reset to closed state")
        
        self.failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures",
                failure_threshold=self.failure_threshold
            )


class ErrorHandler:
    """Central error handler with consistent formatting and recovery strategies."""
    
    def __init__(self):
        """Initialize error handler."""
        self.logger = logger.bind(component="error_handler")
        self.retry_manager = RetryManager()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
    
    def handle_error(
        self,
        exception: Exception,
        context: Optional[ErrorContext] = None,
        notify: bool = True
    ) -> Dict[str, Any]:
        """Handle error with logging and optional notification."""
        
        # Convert to MLTAException if needed
        if not isinstance(exception, MLTAException):
            mlta_exception = MLTAException(
                str(exception),
                context=context,
                cause=exception
            )
        else:
            mlta_exception = exception
        
        # Log error
        error_data = mlta_exception.to_dict()
        
        if mlta_exception.severity == ErrorSeverity.CRITICAL:
            self.logger.critical("Critical error occurred", **error_data)
        elif mlta_exception.severity == ErrorSeverity.HIGH:
            self.logger.error("High severity error occurred", **error_data)
        elif mlta_exception.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity error occurred", **error_data)
        else:
            self.logger.info("Low severity error occurred", **error_data)
        
        # TODO: Add notification logic here (email, Slack, etc.)
        if notify and mlta_exception.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_notification(mlta_exception)
        
        return error_data
    
    def _send_notification(self, exception: MLTAException) -> None:
        """Send notification for high-severity errors."""
        # Placeholder for notification implementation
        self.logger.info(
            "Error notification triggered",
            error_code=exception.error_code,
            severity=exception.severity.value
        )
    
    def get_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create circuit breaker for named operation."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(**kwargs)
        return self.circuit_breakers[name]


class GracefulDegradation:
    """Implements graceful degradation strategies."""
    
    def __init__(self):
        """Initialize graceful degradation manager."""
        self.logger = logger.bind(component="graceful_degradation")
        self.fallback_strategies: Dict[str, Callable] = {}
    
    def register_fallback(self, operation: str, fallback_func: Callable) -> None:
        """Register fallback strategy for an operation."""
        self.fallback_strategies[operation] = fallback_func
        self.logger.info(f"Registered fallback strategy for {operation}")
    
    def execute_with_fallback(self, operation: str, primary_func: Callable, *args, **kwargs) -> Any:
        """Execute operation with fallback if primary fails."""
        try:
            return primary_func(*args, **kwargs)
        
        except Exception as e:
            self.logger.warning(
                f"Primary operation failed, attempting fallback",
                operation=operation,
                error=str(e)
            )
            
            if operation in self.fallback_strategies:
                try:
                    result = self.fallback_strategies[operation](*args, **kwargs)
                    self.logger.info(f"Fallback successful for {operation}")
                    return result
                
                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback also failed for {operation}",
                        fallback_error=str(fallback_error)
                    )
                    raise
            else:
                self.logger.error(f"No fallback strategy registered for {operation}")
                raise


class ErrorRecovery:
    """Implements automatic error recovery mechanisms."""
    
    def __init__(self):
        """Initialize error recovery manager."""
        self.logger = logger.bind(component="error_recovery")
        self.recovery_strategies: Dict[str, Callable] = {}
    
    def register_recovery_strategy(self, error_type: str, recovery_func: Callable) -> None:
        """Register recovery strategy for error type."""
        self.recovery_strategies[error_type] = recovery_func
        self.logger.info(f"Registered recovery strategy for {error_type}")
    
    def attempt_recovery(self, exception: MLTAException) -> bool:
        """Attempt to recover from error."""
        error_type = exception.__class__.__name__
        
        if error_type in self.recovery_strategies:
            try:
                self.logger.info(f"Attempting recovery for {error_type}")
                self.recovery_strategies[error_type](exception)
                self.logger.info(f"Recovery successful for {error_type}")
                return True
            
            except Exception as recovery_error:
                self.logger.error(
                    f"Recovery failed for {error_type}",
                    recovery_error=str(recovery_error)
                )
                return False
        
        self.logger.warning(f"No recovery strategy available for {error_type}")
        return False


# Decorator for automatic error handling
def handle_errors(
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_name: Optional[str] = None,
    context_operation: str = "unknown",
    context_component: str = "unknown"
):
    """Decorator for automatic error handling with retry and circuit breaker."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            context = ErrorContext(context_operation, context_component)
            
            # Setup retry if configured
            if retry_config:
                retry_manager = RetryManager(retry_config)
                func_to_call = lambda: func(*args, **kwargs)
                
                # Setup circuit breaker if configured
                if circuit_breaker_name:
                    circuit_breaker = error_handler.get_circuit_breaker(circuit_breaker_name)
                    func_to_call = lambda: circuit_breaker.call(func, *args, **kwargs)
                
                return retry_manager.retry(func_to_call)
            
            # Just circuit breaker
            elif circuit_breaker_name:
                circuit_breaker = error_handler.get_circuit_breaker(circuit_breaker_name)
                return circuit_breaker.call(func, *args, **kwargs)
            
            # No special handling, just execute
            else:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_handler.handle_error(e, context)
                    raise
        
        return wrapper
    return decorator


# Global instances
error_handler = ErrorHandler()
graceful_degradation = GracefulDegradation()
error_recovery = ErrorRecovery()
