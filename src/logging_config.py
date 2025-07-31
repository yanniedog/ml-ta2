"""
Production-Grade Logging Configuration for ML-TA System

This module provides structured logging with JSON formatting, correlation IDs,
contextual information, and comprehensive monitoring capabilities.
"""

import os
import sys
import json
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import structlog
import colorlog
from pythonjsonlogger import jsonlogger
import psutil
import threading
import time
from collections import defaultdict, deque

from .config import get_config


class CorrelationIDProcessor:
    """Adds correlation IDs to log entries for request tracing."""
    
    def __init__(self):
        self._local = threading.local()
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for current thread."""
        self._local.correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """Get correlation ID for current thread."""
        return getattr(self._local, 'correlation_id', None)
    
    def __call__(self, logger, method_name, event_dict):
        """Add correlation ID to log event."""
        correlation_id = self.get_correlation_id()
        if correlation_id:
            event_dict['correlation_id'] = correlation_id
        return event_dict


class PerformanceProcessor:
    """Adds performance metrics to log entries."""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def __call__(self, logger, method_name, event_dict):
        """Add performance metrics to log event."""
        try:
            # Add memory usage
            memory_info = self.process.memory_info()
            event_dict['memory_rss_mb'] = round(memory_info.rss / 1024 / 1024, 2)
            event_dict['memory_vms_mb'] = round(memory_info.vms / 1024 / 1024, 2)
            
            # Add CPU usage
            event_dict['cpu_percent'] = self.process.cpu_percent()
            
            # Add thread count
            event_dict['thread_count'] = self.process.num_threads()
            
        except Exception:
            # Don't fail logging if performance metrics can't be gathered
            pass
        
        return event_dict


class SecurityProcessor:
    """Processes security-related log events and sanitizes sensitive data."""
    
    SENSITIVE_KEYS = {
        'password', 'api_key', 'secret', 'token', 'authorization',
        'jwt', 'private_key', 'credential', 'auth'
    }
    
    def __call__(self, logger, method_name, event_dict):
        """Sanitize sensitive information from log events."""
        return self._sanitize_dict(event_dict)
    
    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively sanitize dictionary data."""
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in self.SENSITIVE_KEYS):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [self._sanitize_dict(item) if isinstance(item, dict) else item for item in value]
            else:
                sanitized[key] = value
        
        return sanitized


class StructuredLogger:
    """Main structured logger with JSON formatting and contextual information."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize structured logger."""
        self.name = name
        self.config = config or get_config().monitoring.dict()
        self.correlation_processor = CorrelationIDProcessor()
        self.performance_processor = PerformanceProcessor()
        self.security_processor = SecurityProcessor()
        
        self._setup_structlog()
    
    def _setup_structlog(self) -> None:
        """Configure structlog with processors and formatters."""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            self.correlation_processor,
            self.security_processor,
        ]
        
        # Add performance processor if profiling is enabled
        if self.config.get('enable_profiling', False):
            processors.append(self.performance_processor)
        
        # Configure structlog
        structlog.configure(
            processors=processors + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def get_logger(self) -> structlog.BoundLogger:
        """Get configured structlog logger."""
        return structlog.get_logger(self.name)


class LoggerFactory:
    """Factory for creating module-specific loggers with appropriate configurations."""
    
    def __init__(self):
        self.config = get_config()
        self.loggers: Dict[str, StructuredLogger] = {}
        self._setup_root_logger()
    
    def _setup_root_logger(self) -> None:
        """Setup root logger configuration."""
        log_level = getattr(logging, self.config.monitoring.log_level.upper())
        
        # Create logs directory
        log_dir = Path(self.config.paths.logs)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler with color formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        if self.config.app.debug:
            # Colored output for development
            console_formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            # JSON output for production
            console_formatter = jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s'
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # Add file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "ml_ta.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        file_handler.setLevel(log_level)
        
        file_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(pathname)s %(lineno)d %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Add error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "ml_ta_errors.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
    
    def get_logger(self, name: str) -> StructuredLogger:
        """Get or create a logger for the specified module."""
        if name not in self.loggers:
            self.loggers[name] = StructuredLogger(name)
        return self.loggers[name]


class LogAggregator:
    """Collects logs from multiple sources and formats them for analysis."""
    
    def __init__(self, buffer_size: int = 10000):
        """Initialize log aggregator."""
        self.buffer_size = buffer_size
        self.log_buffer = deque(maxlen=buffer_size)
        self.metrics = defaultdict(int)
        self.lock = threading.Lock()
    
    def add_log_entry(self, log_entry: Dict[str, Any]) -> None:
        """Add a log entry to the buffer."""
        with self.lock:
            self.log_buffer.append({
                **log_entry,
                'timestamp': datetime.utcnow().isoformat(),
                'buffer_size': len(self.log_buffer)
            })
            
            # Update metrics
            level = log_entry.get('level', 'unknown')
            self.metrics[f'log_count_{level.lower()}'] += 1
            self.metrics['total_log_count'] += 1
    
    def get_recent_logs(self, count: int = 100) -> list[Dict[str, Any]]:
        """Get recent log entries."""
        with self.lock:
            return list(self.log_buffer)[-count:]
    
    def get_logs_by_level(self, level: str, count: int = 100) -> list[Dict[str, Any]]:
        """Get logs filtered by level."""
        with self.lock:
            filtered_logs = [
                log for log in self.log_buffer 
                if log.get('level', '').lower() == level.lower()
            ]
            return filtered_logs[-count:]
    
    def get_metrics(self) -> Dict[str, int]:
        """Get aggregated log metrics."""
        with self.lock:
            return dict(self.metrics)
    
    def clear_buffer(self) -> None:
        """Clear the log buffer."""
        with self.lock:
            self.log_buffer.clear()
            self.metrics.clear()


class SecurityLogger:
    """Specialized logger for security events and audit trails."""
    
    def __init__(self):
        self.logger = LoggerFactory().get_logger("security").get_logger()
        self.config = get_config()
    
    def log_authentication_attempt(self, user_id: str, success: bool, ip_address: str, user_agent: str) -> None:
        """Log authentication attempt."""
        self.logger.info(
            "authentication_attempt",
            user_id=user_id,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            event_type="authentication"
        )
    
    def log_authorization_check(self, user_id: str, resource: str, action: str, granted: bool) -> None:
        """Log authorization check."""
        self.logger.info(
            "authorization_check",
            user_id=user_id,
            resource=resource,
            action=action,
            granted=granted,
            event_type="authorization"
        )
    
    def log_security_violation(self, violation_type: str, details: Dict[str, Any], severity: str = "high") -> None:
        """Log security violation."""
        self.logger.warning(
            "security_violation",
            violation_type=violation_type,
            details=details,
            severity=severity,
            event_type="security_violation"
        )
    
    def log_data_access(self, user_id: str, data_type: str, operation: str, record_count: int) -> None:
        """Log data access for audit trail."""
        self.logger.info(
            "data_access",
            user_id=user_id,
            data_type=data_type,
            operation=operation,
            record_count=record_count,
            event_type="data_access"
        )


class PerformanceLogger:
    """Specialized logger for performance metrics and system monitoring."""
    
    def __init__(self):
        self.logger = LoggerFactory().get_logger("performance").get_logger()
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation_id: str) -> None:
        """Start timing an operation."""
        self.start_times[operation_id] = time.time()
    
    def end_timer(self, operation_id: str, **kwargs) -> float:
        """End timing an operation and log the duration."""
        if operation_id not in self.start_times:
            self.logger.warning(f"Timer not found for operation: {operation_id}")
            return 0.0
        
        duration = time.time() - self.start_times.pop(operation_id)
        
        self.logger.info(
            "operation_completed",
            operation_id=operation_id,
            duration_seconds=round(duration, 4),
            **kwargs
        )
        
        return duration
    
    def log_memory_usage(self, operation: str, memory_before: float, memory_after: float) -> None:
        """Log memory usage for an operation."""
        memory_delta = memory_after - memory_before
        
        self.logger.info(
            "memory_usage",
            operation=operation,
            memory_before_mb=round(memory_before, 2),
            memory_after_mb=round(memory_after, 2),
            memory_delta_mb=round(memory_delta, 2)
        )
    
    def log_system_metrics(self) -> None:
        """Log current system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.logger.info(
                "system_metrics",
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=round(memory.available / 1024**3, 2),
                disk_percent=disk.percent,
                disk_free_gb=round(disk.free / 1024**3, 2)
            )
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")


class AlertLogger:
    """Specialized logger for alerts and notifications."""
    
    def __init__(self):
        self.logger = LoggerFactory().get_logger("alerts").get_logger()
        self.config = get_config()
        self.alert_history = deque(maxlen=1000)
    
    def log_alert(self, alert_type: str, message: str, severity: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log an alert event."""
        alert_data = {
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.alert_history.append(alert_data)
        
        # Log at appropriate level based on severity
        if severity.lower() == "critical":
            self.logger.critical("alert_triggered", **alert_data)
        elif severity.lower() == "high":
            self.logger.error("alert_triggered", **alert_data)
        elif severity.lower() == "medium":
            self.logger.warning("alert_triggered", **alert_data)
        else:
            self.logger.info("alert_triggered", **alert_data)
    
    def get_recent_alerts(self, count: int = 50) -> list[Dict[str, Any]]:
        """Get recent alerts."""
        return list(self.alert_history)[-count:]


class LogAnalyzer:
    """Analyzes logs for insights and anomaly detection."""
    
    def __init__(self, aggregator: LogAggregator):
        self.aggregator = aggregator
        self.logger = LoggerFactory().get_logger("log_analyzer").get_logger()
    
    def analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns in recent logs."""
        error_logs = self.aggregator.get_logs_by_level("error", 1000)
        
        error_patterns = defaultdict(int)
        error_sources = defaultdict(int)
        
        for log_entry in error_logs:
            # Count error types
            message = log_entry.get('message', '')
            error_patterns[message] += 1
            
            # Count error sources
            source = log_entry.get('name', 'unknown')
            error_sources[source] += 1
        
        analysis = {
            "total_errors": len(error_logs),
            "most_common_errors": dict(sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
            "error_sources": dict(sorted(error_sources.items(), key=lambda x: x[1], reverse=True)[:10]),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.info("error_pattern_analysis", **analysis)
        return analysis
    
    def detect_anomalies(self) -> list[Dict[str, Any]]:
        """Detect anomalies in log patterns."""
        recent_logs = self.aggregator.get_recent_logs(1000)
        anomalies = []
        
        # Check for unusual error rates
        error_count = len([log for log in recent_logs if log.get('level') == 'error'])
        error_rate = error_count / len(recent_logs) if recent_logs else 0
        
        if error_rate > 0.1:  # More than 10% errors
            anomalies.append({
                "type": "high_error_rate",
                "value": error_rate,
                "threshold": 0.1,
                "description": f"Error rate of {error_rate:.2%} exceeds threshold"
            })
        
        # Check for memory usage spikes
        memory_logs = [log for log in recent_logs if 'memory_rss_mb' in log]
        if memory_logs:
            max_memory = max(log['memory_rss_mb'] for log in memory_logs)
            memory_threshold = self.aggregator.config.get('max_memory_gb', 4) * 1024 * 0.9  # 90% of max
            
            if max_memory > memory_threshold:
                anomalies.append({
                    "type": "high_memory_usage",
                    "value": max_memory,
                    "threshold": memory_threshold,
                    "description": f"Memory usage of {max_memory:.2f}MB exceeds threshold"
                })
        
        if anomalies:
            self.logger.warning("log_anomalies_detected", anomalies=anomalies)
        
        return anomalies


# Global logger factory instance
logger_factory = LoggerFactory()


def get_logger(name: str) -> StructuredLogger:
    """Get a logger for the specified module."""
    return logger_factory.get_logger(name)


def setup_logging() -> None:
    """Setup logging configuration for the application."""
    logger_factory._setup_root_logger()


# Initialize logging on module import
setup_logging()
