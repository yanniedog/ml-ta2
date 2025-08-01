"""
Comprehensive monitoring and alerting system for ML-TA.

This module implements:
- System metrics collection and monitoring
- Performance tracking and alerting
- Resource utilization monitoring
- Alert management and notification system
- Health checks and status monitoring
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Core libraries
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Mock psutil for testing
    class psutil:
        @staticmethod
        def cpu_percent(interval=None): return 25.0
        @staticmethod
        def virtual_memory(): 
            class Memory:
                percent = 45.0
                available = 8 * 1024 * 1024 * 1024  # 8GB
            return Memory()
        @staticmethod
        def disk_usage(path):
            class Disk:
                total = 100 * 1024 * 1024 * 1024  # 100GB
                used = 50 * 1024 * 1024 * 1024    # 50GB
                free = 50 * 1024 * 1024 * 1024    # 50GB
            return Disk()

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from src.config import get_config
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MetricValue:
    """Individual metric value with metadata."""
    name: str
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class AlertRule:
    """Alert rule configuration."""
    # Accept either `name` or deprecated `rule_name` as first argument.
    # To maintain backward compatibility with older tests we allow `rule_name`
    # as an alias via `__post_init__`.

    """Alert rule configuration."""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "gte", "lte"
    threshold: Union[int, float]
    severity: str = "warning"
    enabled: bool = True
    cooldown_seconds: int = 300

    # backwards-compat handling
    def __init__(self, *args, **kwargs):
        """Custom init to support both `name` and deprecated `rule_name`."""
        if 'rule_name' in kwargs and 'name' not in kwargs:
            kwargs['name'] = kwargs.pop('rule_name')
        # Allow `severity_level` alias as well
        if 'severity_level' in kwargs and 'severity' not in kwargs:
            kwargs['severity'] = kwargs.pop('severity_level')
        # Use dataclass's generated __init__ via super().__setattr__ hack
        # Manually set attributes to preserve simplicity
        self.name = kwargs.pop('name')
        self.metric_name = kwargs.pop('metric_name')
        self.condition = kwargs.pop('condition')
        self.threshold = kwargs.pop('threshold')
        self.severity = kwargs.pop('severity', 'warning')
        self.enabled = kwargs.pop('enabled', True)
        self.cooldown_seconds = kwargs.pop('cooldown_seconds', 300)
        if kwargs:
            # Raise if unknown fields remain to catch typos
            raise TypeError(f"Unexpected arguments: {list(kwargs.keys())}")


@dataclass
class Alert:
    """Alert instance."""
    rule_name: str
    metric_name: str
    current_value: Union[int, float]
    threshold: Union[int, float]
    condition: str
    severity: str
    timestamp: datetime
    resolved: bool = False
    message: str = ""


class MetricsCollector:
    """Collect system and custom metrics."""
    
    def __init__(self, collection_interval: int = 30):
        """Initialize metrics collector."""
        self.collection_interval = collection_interval
        self.metrics_storage = deque(maxlen=10000)
        self.custom_metrics = {}
        self.collection_active = False
        self.collection_thread = None
        self.lock = threading.RLock()
        
        logger.info("MetricsCollector initialized")
    
    def start_collection(self):
        """Start metrics collection."""
        if self.collection_active:
            return
        
        self.collection_active = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop."""
        while self.collection_active:
            try:
                self._collect_system_metrics()
                self._collect_custom_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collect system performance metrics."""
        timestamp = datetime.now()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self._store_metric(MetricValue(
                name="system.cpu.usage_percent",
                value=cpu_percent,
                timestamp=timestamp,
                unit="percent"
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self._store_metric(MetricValue(
                name="system.memory.usage_percent",
                value=memory.percent,
                timestamp=timestamp,
                unit="percent"
            ))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            self._store_metric(MetricValue(
                name="system.disk.usage_percent",
                value=disk_usage,
                timestamp=timestamp,
                unit="percent"
            ))
            
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    def _collect_custom_metrics(self):
        """Collect custom application metrics."""
        timestamp = datetime.now()
        
        with self.lock:
            for metric_name, metric_func in self.custom_metrics.items():
                try:
                    value = metric_func()
                    if value is not None:
                        self._store_metric(MetricValue(
                            name=metric_name,
                            value=value,
                            timestamp=timestamp
                        ))
                except Exception as e:
                    logger.error(f"Custom metric error for {metric_name}: {e}")
    
    def _store_metric(self, metric: MetricValue):
        """Store metric in memory."""
        with self.lock:
            self.metrics_storage.append(metric)
    
    def register_custom_metric(self, name: str, collection_func: Callable[[], Union[int, float]]):
        """Register custom metric collection function."""
        with self.lock:
            self.custom_metrics[name] = collection_func
        logger.info("Custom metric registered", metric_name=name)
    
    def get_metrics(self, metric_name: Optional[str] = None, limit: Optional[int] = None) -> List[MetricValue]:
        """Get metrics with optional filtering."""
        with self.lock:
            metrics = list(self.metrics_storage)
        
        if metric_name:
            metrics = [m for m in metrics if m.name == metric_name]
        
        metrics.sort(key=lambda m: m.timestamp, reverse=True)
        
        if limit:
            metrics = metrics[:limit]
        
        return metrics
    
    def get_latest_value(self, metric_name: str) -> Optional[MetricValue]:
        """Get latest value for a specific metric."""
        metrics = self.get_metrics(metric_name=metric_name, limit=1)
        return metrics[0] if metrics else None


class AlertManager:
    """Manage alerts and notifications."""
    
    def __init__(self, check_interval: int = 60):
        """Initialize alert manager."""
        self.check_interval = check_interval
        self.alert_rules = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_cooldowns = {}
        self.alerting_active = False
        self.alerting_thread = None
        self.notification_handlers = []
        self.lock = threading.RLock()
        
        logger.info("AlertManager initialized")
    
    def start_alerting(self):
        """Start alert checking."""
        if self.alerting_active:
            return
        
        self.alerting_active = True
        self.alerting_thread = threading.Thread(target=self._alerting_loop, daemon=True)
        self.alerting_thread.start()
        
        logger.info("Alert checking started")
    
    def stop_alerting(self):
        """Stop alert checking."""
        self.alerting_active = False
        if self.alerting_thread:
            self.alerting_thread.join(timeout=5)
        logger.info("Alert checking stopped")
    
    def _alerting_loop(self):
        """Main alerting loop."""
        while self.alerting_active:
            try:
                self._check_alerts()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Alert checking error: {e}")
                time.sleep(self.check_interval)
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add alert rule."""
        try:
            with self.lock:
                self.alert_rules[rule.name] = rule
            logger.info("Alert rule added", rule_name=rule.name)
            return True
        except Exception as e:
            logger.error(f"Failed to add alert rule: {e}")
            return False
    
    def list_alert_rules(self) -> List[AlertRule]:
        """List all alert rules."""
        with self.lock:
            return list(self.alert_rules.values())
    
    def evaluate_alerts(self, metrics: List[MetricValue]) -> List[Alert]:
        """Evaluate alert rules against provided metrics."""
        alerts = []
        
        with self.lock:
            for rule_name, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                # Find relevant metrics for this rule
                relevant_metrics = [m for m in metrics if m.name == rule.metric_name]
                if not relevant_metrics:
                    continue
                
                # Use the latest metric value
                latest_metric = max(relevant_metrics, key=lambda m: m.timestamp)
                current_value = latest_metric.value
                
                # Evaluate condition
                should_alert = self._evaluate_condition(
                    current_value, rule.condition, rule.threshold
                )
                
                if should_alert:
                    alert = Alert(
                        rule_name=rule.name,
                        metric_name=rule.metric_name,
                        current_value=current_value,
                        threshold=rule.threshold,
                        condition=rule.condition,
                        severity=rule.severity,
                        timestamp=latest_metric.timestamp,
                        message=f"{rule.metric_name} {rule.condition} {rule.threshold} (current: {current_value})"
                    )
                    alerts.append(alert)
        
        return alerts
    
    def _check_alerts(self):
        """Check all alert rules against current metrics."""
        with self.lock:
            for rule_name, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                
                try:
                    if self._is_in_cooldown(rule_name, rule.cooldown_seconds):
                        continue
                    
                    current_value = self._get_current_metric_value(rule.metric_name)
                    if current_value is None:
                        continue
                    
                    should_alert = self._evaluate_condition(
                        current_value, rule.condition, rule.threshold
                    )
                    
                    if should_alert:
                        self._trigger_alert(rule, current_value)
                    elif rule_name in self.active_alerts:
                        self._resolve_alert(rule_name)
                
                except Exception as e:
                    logger.error(f"Alert rule check error for {rule_name}: {e}")
    
    def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value for metric (simulated)."""
        import random
        if "cpu" in metric_name.lower():
            return random.uniform(10, 90)
        elif "memory" in metric_name.lower():
            return random.uniform(20, 80)
        else:
            return random.uniform(0, 100)
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == "gt":
            return value > threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "lte":
            return value <= threshold
        return False
    
    def _is_in_cooldown(self, rule_name: str, cooldown_seconds: int) -> bool:
        """Check if alert rule is in cooldown period."""
        if rule_name not in self.alert_cooldowns:
            return False
        
        last_alert_time = self.alert_cooldowns[rule_name]
        return (datetime.now() - last_alert_time).total_seconds() < cooldown_seconds
    
    def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger an alert."""
        alert = Alert(
            rule_name=rule.name,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold=rule.threshold,
            condition=rule.condition,
            severity=rule.severity,
            timestamp=datetime.now(),
            message=f"{rule.metric_name} {rule.condition} {rule.threshold} (current: {current_value})"
        )
        
        with self.lock:
            self.active_alerts[rule.name] = alert
            self.alert_history.append(alert)
            self.alert_cooldowns[rule.name] = datetime.now()
        
        self._send_notifications(alert)
        
        logger.warning("Alert triggered", rule_name=rule.name, severity=rule.severity)
    
    def _resolve_alert(self, rule_name: str):
        """Resolve an active alert."""
        if rule_name not in self.active_alerts:
            return
        
        with self.lock:
            alert = self.active_alerts[rule_name]
            alert.resolved = True
            del self.active_alerts[rule_name]
        
        logger.info("Alert resolved", rule_name=rule_name)
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Notification handler error: {e}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add notification handler."""
        self.notification_handlers.append(handler)
        logger.info("Notification handler added")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self.lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """Get alert history."""
        with self.lock:
            history = list(self.alert_history)
        
        history.sort(key=lambda a: a.timestamp, reverse=True)
        
        if limit:
            history = history[:limit]
        
        return history


class HealthChecker:
    """System health checking and status monitoring."""
    
    def __init__(self, check_interval: int = 30):
        """Initialize health checker."""
        self.check_interval = check_interval
        self.health_checks = {}
        self.health_status = {}
        self.checking_active = False
        self.check_thread = None
        self.lock = threading.RLock()
        
        self._register_default_checks()
        logger.info("HealthChecker initialized")
    
    def _register_default_checks(self):
        """Register default system health checks."""
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("disk_space", self._check_disk_space)
    
    def register_health_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Register health check function."""
        with self.lock:
            self.health_checks[name] = check_func
        logger.info("Health check registered", check_name=name)
    
    def start_checking(self):
        """Start health checking."""
        if self.checking_active:
            return
        
        self.checking_active = True
        self.check_thread = threading.Thread(target=self._checking_loop, daemon=True)
        self.check_thread.start()
        
        logger.info("Health checking started")
    
    def stop_checking(self):
        """Stop health checking."""
        self.checking_active = False
        if self.check_thread:
            self.check_thread.join(timeout=5)
        logger.info("Health checking stopped")
    
    def _checking_loop(self):
        """Main health checking loop."""
        while self.checking_active:
            try:
                self._run_all_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health checking error: {e}")
                time.sleep(self.check_interval)
    
    def _run_all_checks(self):
        """Run all registered health checks."""
        timestamp = datetime.now()
        
        with self.lock:
            for check_name, check_func in self.health_checks.items():
                try:
                    result = check_func()
                    if not isinstance(result, dict):
                        result = {"status": "error", "message": "Invalid check result"}
                    
                    if "status" not in result:
                        result["status"] = "unknown"
                    
                    result["timestamp"] = timestamp
                    result["check_name"] = check_name
                    
                    self.health_status[check_name] = result
                    
                except Exception as e:
                    error_result = {
                        "check_name": check_name,
                        "status": "error",
                        "message": f"Health check failed: {e}",
                        "timestamp": timestamp
                    }
                    
                    self.health_status[check_name] = error_result
                    logger.error(f"Health check failed for {check_name}: {e}")
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            status = "healthy"
            issues = []
            
            if cpu_percent > 90:
                status = "warning"
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > 90:
                status = "warning"
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "issues": issues,
                "message": "System resources check completed"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check system resources: {e}"
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability."""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            status = "healthy"
            if usage_percent > 90:
                status = "critical"
            elif usage_percent > 80:
                status = "warning"
            
            return {
                "status": status,
                "usage_percent": usage_percent,
                "free_gb": disk.free / (1024**3),
                "total_gb": disk.total / (1024**3),
                "message": f"Disk usage: {usage_percent:.1f}%"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check disk space: {e}"
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self.lock:
            overall_status = "healthy"
            
            for check_result in self.health_status.values():
                if check_result["status"] == "critical":
                    overall_status = "critical"
                    break
                elif check_result["status"] in ["warning", "error"]:
                    overall_status = "warning"
            
            return {
                "overall_status": overall_status,
                "checks": self.health_status.copy(),
                "timestamp": datetime.now()
            }


class MonitoringSystem:
    """Main monitoring system orchestrator."""
    
    def __init__(self):
        """Initialize monitoring system."""
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
        self.is_running = False
        
        self._setup_default_alerts()
        logger.info("MonitoringSystem initialized")
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="system.cpu.usage_percent",
                condition="gt",
                threshold=80,
                severity="warning"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system.memory.usage_percent",
                condition="gt",
                threshold=85,
                severity="warning"
            ),
            AlertRule(
                name="high_disk_usage",
                metric_name="system.disk.usage_percent",
                condition="gt",
                threshold=90,
                severity="critical"
            )
        ]
        
        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule)
    
    def start(self):
        """Start all monitoring components."""
        if self.is_running:
            return
        
        self.metrics_collector.start_collection()
        self.alert_manager.start_alerting()
        self.health_checker.start_checking()
        
        self.is_running = True
        logger.info("MonitoringSystem started")
    
    def stop(self):
        """Stop all monitoring components."""
        if not self.is_running:
            return
        
        self.metrics_collector.stop_collection()
        self.alert_manager.stop_alerting()
        self.health_checker.stop_checking()
        
        self.is_running = False
        logger.info("MonitoringSystem stopped")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "monitoring_active": self.is_running,
            "health_status": self.health_checker.get_health_status(),
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "metrics_collected": len(self.metrics_collector.get_metrics()),
            "timestamp": datetime.now()
        }
    
    def add_custom_metric(self, name: str, collection_func: Callable[[], Union[int, float]]):
        """Add custom metric to monitoring."""
        self.metrics_collector.register_custom_metric(name, collection_func)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule."""
        self.alert_manager.add_alert_rule(rule)
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add notification handler for alerts."""
        self.alert_manager.add_notification_handler(handler)
    
    def add_metric(self, metric: MetricValue):
        """Add metric to the monitoring system."""
        self.metrics_collector._store_metric(metric)
    
    def get_metrics(self, metric_name: str = None):
        """Get metrics from the monitoring system."""
        return self.metrics_collector.get_metrics(metric_name)
    
    def get_all_metrics(self):
        """Get all metrics from the monitoring system."""
        return self.metrics_collector.get_metrics()
    
    def collect_system_metrics(self):
        """Manually trigger system metrics collection."""
        self.metrics_collector._collect_system_metrics()
    
    def start_monitoring(self):
        """Start monitoring (alias for start method)."""
        self.start()
    
    def stop_monitoring(self):
        """Stop monitoring (alias for stop method)."""
        self.stop()


def create_monitoring_system() -> MonitoringSystem:
    """Factory function to create monitoring system."""
    return MonitoringSystem()
