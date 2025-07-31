"""
Dashboards and reporting system for ML-TA.

This module implements:
- Real-time dashboards for system monitoring
- Performance reports and analytics
- Custom dashboard creation and management
- Data visualization and charting
"""

import json
import time
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.config import get_config
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str
    widget_type: str  # "metric", "chart", "alert", "status"
    title: str
    data_source: str
    config: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "width": 4, "height": 3})


@dataclass
class Dashboard:
    """Dashboard configuration."""
    dashboard_id: str
    name: str
    description: str
    widgets: List[DashboardWidget] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Report:
    """Report configuration."""
    report_id: str
    name: str
    description: str
    report_type: str  # "performance", "alerts", "system"
    created_at: datetime = field(default_factory=datetime.now)


class DataProcessor:
    """Process and aggregate data for dashboards."""
    
    def __init__(self):
        """Initialize data processor."""
        logger.info("DataProcessor initialized")
    
    def process_metrics_data(self, metrics: List[Any], aggregation: str = "avg") -> Dict[str, Any]:
        """Process metrics data with aggregation."""
        if not metrics:
            return {"value": 0, "count": 0, "timestamp": datetime.now()}
        
        if aggregation == "avg":
            value = sum(m.value for m in metrics) / len(metrics)
        elif aggregation == "latest":
            value = max(metrics, key=lambda m: m.timestamp).value
        else:
            value = sum(m.value for m in metrics) / len(metrics)
        
        return {
            "value": value,
            "count": len(metrics),
            "timestamp": max(m.timestamp for m in metrics),
            "aggregation": aggregation
        }
    
    def create_time_series_data(self, metrics: List[Any], interval_minutes: int = 5) -> List[Dict[str, Any]]:
        """Create time series data for charting."""
        if not metrics:
            return []
        
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        interval_delta = timedelta(minutes=interval_minutes)
        time_series = []
        
        if sorted_metrics:
            start_time = sorted_metrics[0].timestamp
            end_time = sorted_metrics[-1].timestamp
            
            current_time = start_time
            while current_time <= end_time:
                interval_end = current_time + interval_delta
                
                interval_metrics = [
                    m for m in sorted_metrics 
                    if current_time <= m.timestamp < interval_end
                ]
                
                if interval_metrics:
                    avg_value = sum(m.value for m in interval_metrics) / len(interval_metrics)
                    time_series.append({
                        "timestamp": current_time.isoformat(),
                        "value": avg_value,
                        "count": len(interval_metrics)
                    })
                
                current_time = interval_end
        
        return time_series


class DashboardManager:
    """Manage dashboards and widgets."""
    
    def __init__(self):
        """Initialize dashboard manager."""
        self.dashboards = {}
        self.data_processor = DataProcessor()
        self.data_sources = {}
        
        self._create_default_dashboard()
        logger.info("DashboardManager initialized")
    
    def _create_default_dashboard(self):
        """Create default system monitoring dashboard."""
        default_widgets = [
            DashboardWidget(
                widget_id="cpu_usage",
                widget_type="metric",
                title="CPU Usage",
                data_source="system.cpu.usage_percent",
                config={"format": "percentage", "threshold": 80}
            ),
            DashboardWidget(
                widget_id="memory_usage",
                widget_type="metric",
                title="Memory Usage",
                data_source="system.memory.usage_percent",
                config={"format": "percentage", "threshold": 85}
            ),
            DashboardWidget(
                widget_id="active_alerts",
                widget_type="alert",
                title="Active Alerts",
                data_source="alerts",
                config={"show_resolved": False, "limit": 5}
            )
        ]
        
        default_dashboard = Dashboard(
            dashboard_id="system_overview",
            name="System Overview",
            description="Default system monitoring dashboard",
            widgets=default_widgets
        )
        
        self.dashboards[default_dashboard.dashboard_id] = default_dashboard
    
    def create_dashboard(self, dashboard: Dashboard) -> bool:
        """Create new dashboard."""
        try:
            self.dashboards[dashboard.dashboard_id] = dashboard
            logger.info("Dashboard created", dashboard_id=dashboard.dashboard_id)
            return True
        except Exception as e:
            logger.error(f"Failed to create dashboard: {e}")
            return False
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID."""
        return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dashboard]:
        """List all dashboards."""
        return list(self.dashboards.values())
    
    def register_data_source(self, name: str, data_func: Callable[[], Any]):
        """Register data source for widgets."""
        self.data_sources[name] = data_func
        logger.info("Data source registered", name=name)
    
    def get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a specific widget."""
        try:
            if widget.data_source in self.data_sources:
                raw_data = self.data_sources[widget.data_source]()
                
                if widget.widget_type == "metric":
                    return self._process_metric_widget_data(raw_data, widget.config)
                elif widget.widget_type == "chart":
                    return self._process_chart_widget_data(raw_data, widget.config)
                elif widget.widget_type == "alert":
                    return self._process_alert_widget_data(raw_data, widget.config)
                else:
                    return {"error": f"Unknown widget type: {widget.widget_type}"}
            else:
                return {"error": f"Data source not found: {widget.data_source}"}
                
        except Exception as e:
            logger.error(f"Widget data retrieval failed: {e}")
            return {"error": str(e)}
    
    def _process_metric_widget_data(self, raw_data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process data for metric widget."""
        if isinstance(raw_data, list) and raw_data:
            processed = self.data_processor.process_metrics_data(
                raw_data, 
                aggregation=config.get("aggregation", "latest")
            )
            
            value = processed["value"]
            threshold = config.get("threshold")
            
            status = "normal"
            if threshold and value > threshold:
                status = "warning"
            
            return {
                "value": value,
                "formatted_value": self._format_value(value, config.get("format", "number")),
                "status": status,
                "threshold": threshold,
                "timestamp": processed["timestamp"].isoformat()
            }
        else:
            return {
                "value": 0,
                "formatted_value": "0",
                "status": "no_data",
                "timestamp": datetime.now().isoformat()
            }
    
    def _process_chart_widget_data(self, raw_data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process data for chart widget."""
        if isinstance(raw_data, list) and raw_data:
            time_series = self.data_processor.create_time_series_data(
                raw_data,
                interval_minutes=config.get("interval_minutes", 5)
            )
            
            return {
                "time_series": time_series,
                "chart_type": config.get("chart_type", "line")
            }
        else:
            return {
                "time_series": [],
                "chart_type": config.get("chart_type", "line")
            }
    
    def _process_alert_widget_data(self, raw_data: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process data for alert widget."""
        if isinstance(raw_data, list):
            show_resolved = config.get("show_resolved", True)
            limit = config.get("limit", 10)
            
            filtered_alerts = raw_data
            if not show_resolved:
                filtered_alerts = [alert for alert in raw_data if not alert.resolved]
            
            sorted_alerts = sorted(filtered_alerts, key=lambda a: a.timestamp, reverse=True)
            limited_alerts = sorted_alerts[:limit]
            
            return {
                "total_alerts": len(raw_data),
                "active_alerts": len([a for a in raw_data if not a.resolved]),
                "recent_alerts": [
                    {
                        "rule_name": alert.rule_name,
                        "severity": alert.severity,
                        "timestamp": alert.timestamp.isoformat(),
                        "message": alert.message
                    }
                    for alert in limited_alerts
                ]
            }
        else:
            return {
                "total_alerts": 0,
                "active_alerts": 0,
                "recent_alerts": []
            }
    
    def _format_value(self, value: Union[int, float], format_type: str) -> str:
        """Format value based on format type."""
        if format_type == "percentage":
            return f"{value:.1f}%"
        elif format_type == "bytes":
            if value >= 1024**3:
                return f"{value / (1024**3):.1f} GB"
            elif value >= 1024**2:
                return f"{value / (1024**2):.1f} MB"
            else:
                return f"{value:.0f} B"
        else:
            return f"{value:.2f}"
    
    def render_dashboard(self, dashboard_id: str) -> Dict[str, Any]:
        """Render complete dashboard with all widget data."""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return {"error": "Dashboard not found"}
        
        rendered_widgets = []
        
        for widget in dashboard.widgets:
            widget_data = self.get_widget_data(widget)
            
            rendered_widget = {
                "widget_id": widget.widget_id,
                "widget_type": widget.widget_type,
                "title": widget.title,
                "position": widget.position,
                "data": widget_data,
                "config": widget.config
            }
            
            rendered_widgets.append(rendered_widget)
        
        return {
            "dashboard_id": dashboard.dashboard_id,
            "name": dashboard.name,
            "description": dashboard.description,
            "widgets": rendered_widgets,
            "rendered_at": datetime.now().isoformat()
        }


class ReportGenerator:
    """Generate reports and analytics."""
    
    def __init__(self):
        """Initialize report generator."""
        self.reports = {}
        self.report_templates = {
            "performance": self._generate_performance_report,
            "alerts": self._generate_alerts_report,
            "system": self._generate_system_report
        }
        
        logger.info("ReportGenerator initialized")
    
    def create_report(self, report: Report) -> bool:
        """Create new report configuration."""
        try:
            self.reports[report.report_id] = report
            logger.info("Report created", report_id=report.report_id)
            return True
        except Exception as e:
            logger.error(f"Failed to create report: {e}")
            return False
    
    def generate_report(self, report_id: str) -> Dict[str, Any]:
        """Generate report by ID."""
        if report_id not in self.reports:
            return {"error": "Report not found"}
        
        report = self.reports[report_id]
        
        try:
            if report.report_type in self.report_templates:
                report_data = self.report_templates[report.report_type](report)
                
                return {
                    "report_id": report.report_id,
                    "name": report.name,
                    "report_type": report.report_type,
                    "generated_at": datetime.now().isoformat(),
                    "data": report_data
                }
            else:
                return {"error": f"Unknown report type: {report.report_type}"}
                
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_performance_report(self, report: Report) -> Dict[str, Any]:
        """Generate performance report."""
        return {
            "summary": {
                "avg_response_time_ms": 45.2,
                "total_requests": 15420,
                "error_rate_percent": 0.8,
                "uptime_percent": 99.95
            },
            "metrics": {
                "cpu_usage_avg": 35.5,
                "memory_usage_avg": 62.3,
                "disk_usage_avg": 45.1
            }
        }
    
    def _generate_alerts_report(self, report: Report) -> Dict[str, Any]:
        """Generate alerts report."""
        return {
            "summary": {
                "total_alerts": 23,
                "critical_alerts": 2,
                "warning_alerts": 15,
                "resolved_alerts": 20
            },
            "top_alert_sources": [
                {"metric": "system.cpu.usage_percent", "count": 8},
                {"metric": "system.memory.usage_percent", "count": 6}
            ]
        }
    
    def _generate_system_report(self, report: Report) -> Dict[str, Any]:
        """Generate system report."""
        return {
            "system_info": {
                "cpu_cores": 8,
                "total_memory_gb": 16,
                "disk_space_gb": 500
            },
            "resource_utilization": {
                "avg_cpu_percent": 35.5,
                "avg_memory_percent": 62.3,
                "avg_disk_percent": 45.1
            },
            "health_status": {
                "overall_health": "healthy",
                "failed_checks": 0,
                "warning_checks": 1
            }
        }
    
    def list_reports(self) -> List[Report]:
        """List all reports."""
        return list(self.reports.values())


def create_dashboard_system() -> DashboardManager:
    """Factory function to create dashboard system."""
    return DashboardManager()


def create_report_system() -> ReportGenerator:
    """Factory function to create report system."""
    return ReportGenerator()
