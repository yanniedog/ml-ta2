"""
Advanced monitoring integration for ML-TA System.

This module implements:
- Prometheus metrics integration
- Grafana dashboard configuration
- Advanced alerting channels (email, Slack, PagerDuty)
- Custom monitoring dashboards
- Performance and business metrics
"""

import os
import json
import time
import smtplib
import requests
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import warnings
warnings.filterwarnings('ignore')

# Prometheus client
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock Prometheus for testing
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    class CollectorRegistry:
        def __init__(self): pass
    def generate_latest(registry): return b""

from src.config import get_config
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AlertChannel:
    """Alert channel configuration."""
    name: str
    channel_type: str  # 'email', 'slack', 'pagerduty', 'webhook'
    config: Dict[str, Any]
    enabled: bool = True


@dataclass
class Dashboard:
    """Dashboard configuration."""
    name: str
    dashboard_type: str  # 'grafana', 'custom'
    panels: List[Dict[str, Any]]
    refresh_interval: str = "30s"
    time_range: str = "1h"


class PrometheusMetrics:
    """Prometheus metrics collector."""
    
    def __init__(self):
        """Initialize Prometheus metrics."""
        self.registry = CollectorRegistry()
        self._setup_metrics()
        
        logger.info("PrometheusMetrics initialized")
    
    def _setup_metrics(self):
        """Setup Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, using mock metrics")
        
        # System metrics
        self.cpu_usage = Gauge('mlta_cpu_usage_percent', 'CPU usage percentage', registry=self.registry)
        self.memory_usage = Gauge('mlta_memory_usage_bytes', 'Memory usage in bytes', registry=self.registry)
        self.disk_usage = Gauge('mlta_disk_usage_percent', 'Disk usage percentage', registry=self.registry)
        
        # Application metrics
        self.request_count = Counter('mlta_requests_total', 'Total requests', ['method', 'endpoint', 'status'], registry=self.registry)
        self.request_duration = Histogram('mlta_request_duration_seconds', 'Request duration', ['method', 'endpoint'], registry=self.registry)
        self.active_connections = Gauge('mlta_active_connections', 'Active connections', registry=self.registry)
        
        # ML metrics
        self.model_predictions = Counter('mlta_model_predictions_total', 'Total model predictions', ['model_name', 'status'], registry=self.registry)
        self.prediction_latency = Histogram('mlta_prediction_latency_seconds', 'Prediction latency', ['model_name'], registry=self.registry)
        self.model_accuracy = Gauge('mlta_model_accuracy', 'Model accuracy', ['model_name'], registry=self.registry)
        self.feature_drift = Gauge('mlta_feature_drift_score', 'Feature drift score', ['feature_name'], registry=self.registry)
        
        # Business metrics
        self.trading_signals = Counter('mlta_trading_signals_total', 'Trading signals generated', ['signal_type', 'confidence'], registry=self.registry)
        self.portfolio_value = Gauge('mlta_portfolio_value_usd', 'Portfolio value in USD', registry=self.registry)
        self.pnl_daily = Gauge('mlta_pnl_daily_usd', 'Daily P&L in USD', registry=self.registry)
        
        # Data metrics
        self.data_points_processed = Counter('mlta_data_points_processed_total', 'Data points processed', ['source'], registry=self.registry)
        self.data_quality_score = Gauge('mlta_data_quality_score', 'Data quality score', ['source'], registry=self.registry)
        self.api_errors = Counter('mlta_api_errors_total', 'API errors', ['api', 'error_type'], registry=self.registry)
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        self.request_count.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_prediction(self, model_name: str, latency: float, success: bool = True):
        """Record model prediction metrics."""
        status = 'success' if success else 'error'
        self.model_predictions.labels(model_name=model_name, status=status).inc()
        if success:
            self.prediction_latency.labels(model_name=model_name).observe(latency)
    
    def update_model_accuracy(self, model_name: str, accuracy: float):
        """Update model accuracy metric."""
        self.model_accuracy.labels(model_name=model_name).set(accuracy)
    
    def record_trading_signal(self, signal_type: str, confidence: str):
        """Record trading signal generation."""
        self.trading_signals.labels(signal_type=signal_type, confidence=confidence).inc()
    
    def update_portfolio_metrics(self, portfolio_value: float, daily_pnl: float):
        """Update portfolio metrics."""
        self.portfolio_value.set(portfolio_value)
        self.pnl_daily.set(daily_pnl)
    
    def record_data_processing(self, source: str, points: int, quality_score: float):
        """Record data processing metrics."""
        self.data_points_processed.labels(source=source).inc(points)
        self.data_quality_score.labels(source=source).set(quality_score)
    
    def record_api_error(self, api: str, error_type: str):
        """Record API error."""
        self.api_errors.labels(api=api, error_type=error_type).inc()
    
    def update_system_metrics(self, cpu_percent: float, memory_bytes: int, disk_percent: float):
        """Update system metrics."""
        self.cpu_usage.set(cpu_percent)
        self.memory_usage.set(memory_bytes)
        self.disk_usage.set(disk_percent)
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in exposition format."""
        return generate_latest(self.registry)


class AlertManager:
    """Advanced alert manager with multiple channels."""
    
    def __init__(self, channels: List[AlertChannel] = None):
        """Initialize alert manager."""
        self.channels = channels or []
        self.alert_history = []
        self.cooldown_periods = {}  # Track alert cooldowns
        
        logger.info(f"AlertManager initialized with {len(self.channels)} channels")
    
    def add_channel(self, channel: AlertChannel):
        """Add alert channel."""
        self.channels.append(channel)
        logger.info(f"Added alert channel: {channel.name}")
    
    def send_alert(self, title: str, message: str, severity: str = 'warning', 
                   tags: List[str] = None) -> Dict[str, Any]:
        """Send alert through all enabled channels."""
        alert_id = f"alert_{int(time.time())}_{hash(title) % 10000}"
        
        # Check cooldown
        cooldown_key = f"{title}_{severity}"
        if self._is_in_cooldown(cooldown_key):
            logger.info(f"Alert {alert_id} suppressed due to cooldown")
            return {'alert_id': alert_id, 'status': 'suppressed', 'reason': 'cooldown'}
        
        results = {}
        for channel in self.channels:
            if not channel.enabled:
                continue
            
            try:
                if channel.channel_type == 'email':
                    result = self._send_email_alert(channel, title, message, severity)
                elif channel.channel_type == 'slack':
                    result = self._send_slack_alert(channel, title, message, severity)
                elif channel.channel_type == 'pagerduty':
                    result = self._send_pagerduty_alert(channel, title, message, severity)
                elif channel.channel_type == 'webhook':
                    result = self._send_webhook_alert(channel, title, message, severity)
                else:
                    result = {'status': 'error', 'message': f'Unknown channel type: {channel.channel_type}'}
                
                results[channel.name] = result
                
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.name}: {e}")
                results[channel.name] = {'status': 'error', 'message': str(e)}
        
        # Record alert
        alert_record = {
            'alert_id': alert_id,
            'title': title,
            'message': message,
            'severity': severity,
            'tags': tags or [],
            'timestamp': datetime.now(),
            'channels': results
        }
        
        self.alert_history.append(alert_record)
        self._set_cooldown(cooldown_key)
        
        logger.info(f"Alert sent: {alert_id}")
        return {'alert_id': alert_id, 'status': 'sent', 'channels': results}
    
    def _is_in_cooldown(self, key: str, cooldown_minutes: int = 15) -> bool:
        """Check if alert is in cooldown period."""
        if key not in self.cooldown_periods:
            return False
        
        last_sent = self.cooldown_periods[key]
        cooldown_period = timedelta(minutes=cooldown_minutes)
        
        return datetime.now() - last_sent < cooldown_period
    
    def _set_cooldown(self, key: str):
        """Set cooldown for alert type."""
        self.cooldown_periods[key] = datetime.now()
    
    def _send_email_alert(self, channel: AlertChannel, title: str, 
                         message: str, severity: str) -> Dict[str, Any]:
        """Send email alert."""
        try:
            config = channel.config
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"[{severity.upper()}] ML-TA Alert: {title}"
            
            body = f"""
ML-TA System Alert

Severity: {severity.upper()}
Title: {title}
Time: {datetime.now().isoformat()}

Message:
{message}

--
ML-TA Monitoring System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            if config.get('use_tls', True):
                server.starttls()
            
            if 'username' in config and 'password' in config:
                server.login(config['username'], config['password'])
            
            server.send_message(msg)
            server.quit()
            
            return {'status': 'success', 'message': 'Email sent successfully'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _send_slack_alert(self, channel: AlertChannel, title: str, 
                         message: str, severity: str) -> Dict[str, Any]:
        """Send Slack alert."""
        try:
            config = channel.config
            webhook_url = config['webhook_url']
            
            # Color coding for severity
            color_map = {
                'info': '#36a64f',
                'warning': '#ff9900',
                'error': '#ff0000',
                'critical': '#8B0000'
            }
            
            payload = {
                'text': f"ML-TA Alert: {title}",
                'attachments': [
                    {
                        'color': color_map.get(severity, '#36a64f'),
                        'fields': [
                            {
                                'title': 'Severity',
                                'value': severity.upper(),
                                'short': True
                            },
                            {
                                'title': 'Time',
                                'value': datetime.now().isoformat(),
                                'short': True
                            },
                            {
                                'title': 'Message',
                                'value': message,
                                'short': False
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            return {'status': 'success', 'message': 'Slack alert sent successfully'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _send_pagerduty_alert(self, channel: AlertChannel, title: str, 
                             message: str, severity: str) -> Dict[str, Any]:
        """Send PagerDuty alert."""
        try:
            config = channel.config
            
            # Map severity to PagerDuty severity
            severity_map = {
                'info': 'info',
                'warning': 'warning',
                'error': 'error',
                'critical': 'critical'
            }
            
            payload = {
                'routing_key': config['integration_key'],
                'event_action': 'trigger',
                'payload': {
                    'summary': f"ML-TA Alert: {title}",
                    'source': 'ml-ta-system',
                    'severity': severity_map.get(severity, 'warning'),
                    'custom_details': {
                        'message': message,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            }
            
            response = requests.post(
                'https://events.pagerduty.com/v2/enqueue',
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            return {'status': 'success', 'message': 'PagerDuty alert sent successfully'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _send_webhook_alert(self, channel: AlertChannel, title: str, 
                           message: str, severity: str) -> Dict[str, Any]:
        """Send webhook alert."""
        try:
            config = channel.config
            
            payload = {
                'title': title,
                'message': message,
                'severity': severity,
                'timestamp': datetime.now().isoformat(),
                'source': 'ml-ta-system'
            }
            
            headers = config.get('headers', {})
            response = requests.post(
                config['webhook_url'],
                json=payload,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            
            return {'status': 'success', 'message': 'Webhook alert sent successfully'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alert history."""
        return sorted(self.alert_history, key=lambda x: x['timestamp'], reverse=True)[:limit]


class DashboardManager:
    """Dashboard configuration and management."""
    
    def __init__(self):
        """Initialize dashboard manager."""
        self.dashboards = {}
        self._create_default_dashboards()
        
        logger.info("DashboardManager initialized")
    
    def _create_default_dashboards(self):
        """Create default monitoring dashboards."""
        # System overview dashboard
        system_dashboard = Dashboard(
            name="ML-TA System Overview",
            dashboard_type="grafana",
            panels=[
                {
                    'title': 'CPU Usage',
                    'type': 'graph',
                    'targets': [{'expr': 'mlta_cpu_usage_percent'}],
                    'yAxes': [{'max': 100, 'min': 0, 'unit': 'percent'}]
                },
                {
                    'title': 'Memory Usage',
                    'type': 'graph',
                    'targets': [{'expr': 'mlta_memory_usage_bytes'}],
                    'yAxes': [{'unit': 'bytes'}]
                },
                {
                    'title': 'Request Rate',
                    'type': 'graph',
                    'targets': [{'expr': 'rate(mlta_requests_total[5m])'}],
                    'yAxes': [{'unit': 'reqps'}]
                },
                {
                    'title': 'Request Duration',
                    'type': 'graph',
                    'targets': [{'expr': 'histogram_quantile(0.95, mlta_request_duration_seconds_bucket)'}],
                    'yAxes': [{'unit': 's'}]
                }
            ]
        )
        
        # ML performance dashboard
        ml_dashboard = Dashboard(
            name="ML Performance",
            dashboard_type="grafana",
            panels=[
                {
                    'title': 'Model Predictions',
                    'type': 'graph',
                    'targets': [{'expr': 'rate(mlta_model_predictions_total[5m])'}],
                    'yAxes': [{'unit': 'predps'}]
                },
                {
                    'title': 'Prediction Latency',
                    'type': 'graph',
                    'targets': [{'expr': 'histogram_quantile(0.95, mlta_prediction_latency_seconds_bucket)'}],
                    'yAxes': [{'unit': 's'}]
                },
                {
                    'title': 'Model Accuracy',
                    'type': 'stat',
                    'targets': [{'expr': 'mlta_model_accuracy'}],
                    'thresholds': [{'color': 'red', 'value': 0.7}, {'color': 'yellow', 'value': 0.8}, {'color': 'green', 'value': 0.9}]
                },
                {
                    'title': 'Feature Drift',
                    'type': 'heatmap',
                    'targets': [{'expr': 'mlta_feature_drift_score'}],
                    'yAxes': [{'max': 1, 'min': 0}]
                }
            ]
        )
        
        # Business metrics dashboard
        business_dashboard = Dashboard(
            name="Business Metrics",
            dashboard_type="grafana",
            panels=[
                {
                    'title': 'Portfolio Value',
                    'type': 'stat',
                    'targets': [{'expr': 'mlta_portfolio_value_usd'}],
                    'yAxes': [{'unit': 'currencyUSD'}]
                },
                {
                    'title': 'Daily P&L',
                    'type': 'graph',
                    'targets': [{'expr': 'mlta_pnl_daily_usd'}],
                    'yAxes': [{'unit': 'currencyUSD'}]
                },
                {
                    'title': 'Trading Signals',
                    'type': 'graph',
                    'targets': [{'expr': 'rate(mlta_trading_signals_total[5m])'}],
                    'yAxes': [{'unit': 'sigps'}]
                },
                {
                    'title': 'Data Quality',
                    'type': 'stat',
                    'targets': [{'expr': 'avg(mlta_data_quality_score)'}],
                    'thresholds': [{'color': 'red', 'value': 0.7}, {'color': 'yellow', 'value': 0.8}, {'color': 'green', 'value': 0.9}]
                }
            ]
        )
        
        self.dashboards['system'] = system_dashboard
        self.dashboards['ml'] = ml_dashboard
        self.dashboards['business'] = business_dashboard
    
    def add_dashboard(self, dashboard: Dashboard):
        """Add custom dashboard."""
        self.dashboards[dashboard.name.lower().replace(' ', '_')] = dashboard
        logger.info(f"Added dashboard: {dashboard.name}")
    
    def get_dashboard_config(self, dashboard_name: str) -> Optional[Dict[str, Any]]:
        """Get dashboard configuration."""
        dashboard = self.dashboards.get(dashboard_name)
        if not dashboard:
            return None
        
        return {
            'dashboard': {
                'title': dashboard.name,
                'panels': dashboard.panels,
                'refresh': dashboard.refresh_interval,
                'time': {
                    'from': f'now-{dashboard.time_range}',
                    'to': 'now'
                },
                'tags': ['ml-ta', 'monitoring']
            }
        }
    
    def export_grafana_dashboard(self, dashboard_name: str) -> Optional[str]:
        """Export dashboard as Grafana JSON."""
        config = self.get_dashboard_config(dashboard_name)
        if not config:
            return None
        
        return json.dumps(config, indent=2)
    
    def list_dashboards(self) -> List[str]:
        """List available dashboards."""
        return list(self.dashboards.keys())


class MonitoringOrchestrator:
    """Orchestrates all monitoring components."""
    
    def __init__(self):
        """Initialize monitoring orchestrator."""
        self.metrics = PrometheusMetrics()
        self.alerts = AlertManager()
        self.dashboards = DashboardManager()
        self.monitoring_thread = None
        self.running = False
        
        logger.info("MonitoringOrchestrator initialized")
    
    def setup_alert_channels(self, channels_config: List[Dict[str, Any]]):
        """Setup alert channels from configuration."""
        for channel_config in channels_config:
            channel = AlertChannel(
                name=channel_config['name'],
                channel_type=channel_config['type'],
                config=channel_config['config'],
                enabled=channel_config.get('enabled', True)
            )
            self.alerts.add_channel(channel)
    
    def start_monitoring(self, interval: int = 30):
        """Start background monitoring."""
        if self.running:
            logger.warning("Monitoring already running")
            return
        
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"Started monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Stopped monitoring")
    
    def _monitoring_loop(self, interval: int):
        """Background monitoring loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            import psutil
            
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.metrics.update_system_metrics(
                cpu_percent=cpu_percent,
                memory_bytes=memory.used,
                disk_percent=disk.percent
            )
            
            # Check for alerts
            if cpu_percent > 80:
                self.alerts.send_alert(
                    title="High CPU Usage",
                    message=f"CPU usage is {cpu_percent:.1f}%",
                    severity="warning"
                )
            
            if memory.percent > 85:
                self.alerts.send_alert(
                    title="High Memory Usage",
                    message=f"Memory usage is {memory.percent:.1f}%",
                    severity="warning"
                )
            
            if disk.percent > 90:
                self.alerts.send_alert(
                    title="High Disk Usage",
                    message=f"Disk usage is {disk.percent:.1f}%",
                    severity="critical"
                )
                
        except ImportError:
            logger.warning("psutil not available for system metrics")
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine overall health
            health_score = 100
            issues = []
            
            if cpu_percent > 80:
                health_score -= 20
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > 85:
                health_score -= 25
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            if disk.percent > 90:
                health_score -= 30
                issues.append(f"High disk usage: {disk.percent:.1f}%")
            
            status = "healthy"
            if health_score < 50:
                status = "critical"
            elif health_score < 75:
                status = "degraded"
            elif health_score < 90:
                status = "warning"
            
            return {
                'status': status,
                'health_score': max(0, health_score),
                'issues': issues,
                'metrics': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health status check failed: {e}")
            return {
                'status': 'unknown',
                'health_score': 0,
                'issues': [f"Health check failed: {e}"],
                'timestamp': datetime.now().isoformat()
            }


def create_monitoring_system(alert_channels: List[Dict[str, Any]] = None) -> MonitoringOrchestrator:
    """Factory function to create monitoring system."""
    orchestrator = MonitoringOrchestrator()
    
    if alert_channels:
        orchestrator.setup_alert_channels(alert_channels)
    
    return orchestrator


if __name__ == '__main__':
    # Example usage
    
    # Setup alert channels
    channels = [
        {
            'name': 'email_alerts',
            'type': 'email',
            'config': {
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'from_email': 'alerts@mlta.com',
                'to_emails': ['admin@mlta.com'],
                'username': 'alerts@mlta.com',
                'password': 'app_password'
            }
        },
        {
            'name': 'slack_alerts',
            'type': 'slack',
            'config': {
                'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
            }
        }
    ]
    
    # Create monitoring system
    monitoring = create_monitoring_system(channels)
    
    # Record some metrics
    monitoring.metrics.record_request('GET', '/api/v1/predict', 200, 0.15)
    monitoring.metrics.record_prediction('lightgbm_model', 0.05, True)
    monitoring.metrics.update_model_accuracy('lightgbm_model', 0.92)
    
    # Send test alert
    monitoring.alerts.send_alert(
        title="Test Alert",
        message="This is a test alert from ML-TA monitoring system",
        severity="info"
    )
    
    # Get dashboard config
    dashboard_config = monitoring.dashboards.get_dashboard_config('system')
    print(f"System dashboard has {len(dashboard_config['dashboard']['panels'])} panels")
    
    # Check health status
    health = monitoring.get_health_status()
    print(f"System health: {health['status']} (score: {health['health_score']})")
    
    # Start monitoring (would run in background)
    # monitoring.start_monitoring(interval=30)
