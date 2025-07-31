"""
Comprehensive test suite for Phase 7: Monitoring and Operations.

Tests cover:
- Monitoring and alerting system
- Dashboards and reporting
- Security scanning and compliance
- Backup and disaster recovery
"""

import unittest
import os
import tempfile
import shutil
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import Phase 7 modules
from src.monitoring import (
    MonitoringSystem, AlertManager, HealthChecker, MetricValue, Alert, AlertRule
)
from src.dashboards import (
    DashboardManager, ReportGenerator, Dashboard, DashboardWidget, Report
)
from src.security import (
    SecurityScanner, ComplianceFramework, SecurityAuditor,
    SecurityLevel, ComplianceStandard, SecurityVulnerability
)
from src.backup_recovery import (
    BackupManager, DisasterRecoveryManager, BackupJob, BackupType, RecoveryPlan
)


class TestMonitoringSystem(unittest.TestCase):
    """Test monitoring and alerting system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitoring = MonitoringSystem()
        self.alert_manager = AlertManager()
        self.health_checker = HealthChecker()
    
    def test_monitoring_system_initialization(self):
        """Test monitoring system initialization."""
        self.assertIsInstance(self.monitoring, MonitoringSystem)
        self.assertIsInstance(self.monitoring.metrics, list)
        self.assertIsInstance(self.monitoring.collectors, dict)
    
    def test_metric_collection(self):
        """Test metric collection."""
        # Add a test metric
        metric = MetricValue(
            name="test.metric",
            value=42.5,
            timestamp=datetime.now(),
            tags={"source": "test"}
        )
        
        self.monitoring.add_metric(metric)
        
        # Verify metric was added
        metrics = self.monitoring.get_metrics("test.metric")
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].value, 42.5)
    
    def test_system_metrics_collection(self):
        """Test system metrics collection."""
        # Collect system metrics
        self.monitoring.collect_system_metrics()
        
        # Verify system metrics were collected
        cpu_metrics = self.monitoring.get_metrics("system.cpu.usage_percent")
        memory_metrics = self.monitoring.get_metrics("system.memory.usage_percent")
        
        self.assertGreater(len(cpu_metrics), 0)
        self.assertGreater(len(memory_metrics), 0)
    
    def test_alert_rule_creation(self):
        """Test alert rule creation."""
        rule = AlertRule(
            rule_name="high_cpu",
            metric_name="system.cpu.usage_percent",
            condition="greater_than",
            threshold=80.0,
            severity="warning"
        )
        
        success = self.alert_manager.add_alert_rule(rule)
        self.assertTrue(success)
        
        # Verify rule was added
        rules = self.alert_manager.list_alert_rules()
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].rule_name, "high_cpu")
    
    def test_alert_evaluation(self):
        """Test alert evaluation."""
        # Add alert rule
        rule = AlertRule(
            rule_name="test_alert",
            metric_name="test.metric",
            condition="greater_than",
            threshold=50.0,
            severity="warning"
        )
        self.alert_manager.add_alert_rule(rule)
        
        # Add metric that should trigger alert
        metric = MetricValue(
            name="test.metric",
            value=75.0,
            timestamp=datetime.now()
        )
        
        # Evaluate alerts
        alerts = self.alert_manager.evaluate_alerts([metric])
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].rule_name, "test_alert")
        self.assertEqual(alerts[0].severity, "warning")
    
    def test_health_checks(self):
        """Test health checking system."""
        # Run health checks
        results = self.health_checker.run_all_checks()
        
        self.assertIsInstance(results, dict)
        self.assertIn("overall_health", results)
        self.assertIn("checks", results)
        
        # Verify default checks exist
        checks = results["checks"]
        check_names = [check["name"] for check in checks]
        self.assertIn("System Resources", check_names)
        self.assertIn("Disk Space", check_names)
    
    def test_monitoring_orchestration(self):
        """Test monitoring system orchestration."""
        # Start monitoring (non-blocking)
        self.monitoring.start_monitoring()
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Stop monitoring
        self.monitoring.stop_monitoring()
        
        # Verify metrics were collected
        metrics = self.monitoring.get_all_metrics()
        self.assertGreater(len(metrics), 0)


class TestDashboards(unittest.TestCase):
    """Test dashboards and reporting system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dashboard_manager = DashboardManager()
        self.report_generator = ReportGenerator()
    
    def test_dashboard_manager_initialization(self):
        """Test dashboard manager initialization."""
        self.assertIsInstance(self.dashboard_manager, DashboardManager)
        
        # Verify default dashboard exists
        dashboards = self.dashboard_manager.list_dashboards()
        self.assertGreater(len(dashboards), 0)
        
        default_dashboard = self.dashboard_manager.get_dashboard("system_overview")
        self.assertIsNotNone(default_dashboard)
        self.assertEqual(default_dashboard.name, "System Overview")
    
    def test_dashboard_creation(self):
        """Test dashboard creation."""
        # Create test widget
        widget = DashboardWidget(
            widget_id="test_widget",
            widget_type="metric",
            title="Test Metric",
            data_source="test.metric"
        )
        
        # Create test dashboard
        dashboard = Dashboard(
            dashboard_id="test_dashboard",
            name="Test Dashboard",
            description="Test dashboard for unit tests",
            widgets=[widget]
        )
        
        success = self.dashboard_manager.create_dashboard(dashboard)
        self.assertTrue(success)
        
        # Verify dashboard was created
        retrieved = self.dashboard_manager.get_dashboard("test_dashboard")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Test Dashboard")
        self.assertEqual(len(retrieved.widgets), 1)
    
    def test_data_source_registration(self):
        """Test data source registration."""
        # Register test data source
        def test_data_source():
            return [
                MetricValue(
                    name="test.metric",
                    value=42.0,
                    timestamp=datetime.now()
                )
            ]
        
        self.dashboard_manager.register_data_source("test.metric", test_data_source)
        
        # Verify data source was registered
        self.assertIn("test.metric", self.dashboard_manager.data_sources)
    
    def test_widget_data_processing(self):
        """Test widget data processing."""
        # Register test data source
        def test_data_source():
            return [
                MetricValue(
                    name="test.metric",
                    value=75.0,
                    timestamp=datetime.now()
                )
            ]
        
        self.dashboard_manager.register_data_source("test.metric", test_data_source)
        
        # Create test widget
        widget = DashboardWidget(
            widget_id="test_widget",
            widget_type="metric",
            title="Test Metric",
            data_source="test.metric",
            config={"format": "percentage", "threshold": 80}
        )
        
        # Get widget data
        widget_data = self.dashboard_manager.get_widget_data(widget)
        
        self.assertIn("value", widget_data)
        self.assertIn("formatted_value", widget_data)
        self.assertIn("status", widget_data)
        self.assertEqual(widget_data["value"], 75.0)
        self.assertEqual(widget_data["formatted_value"], "75.0%")
        self.assertEqual(widget_data["status"], "normal")
    
    def test_dashboard_rendering(self):
        """Test dashboard rendering."""
        # Register test data source
        def test_data_source():
            return [
                MetricValue(
                    name="test.metric",
                    value=50.0,
                    timestamp=datetime.now()
                )
            ]
        
        self.dashboard_manager.register_data_source("test.metric", test_data_source)
        
        # Render default dashboard
        rendered = self.dashboard_manager.render_dashboard("system_overview")
        
        self.assertIn("dashboard_id", rendered)
        self.assertIn("name", rendered)
        self.assertIn("widgets", rendered)
        self.assertIn("rendered_at", rendered)
        
        # Verify widgets have data
        widgets = rendered["widgets"]
        self.assertGreater(len(widgets), 0)
        
        for widget in widgets:
            self.assertIn("data", widget)
    
    def test_report_creation(self):
        """Test report creation."""
        report = Report(
            report_id="test_report",
            name="Test Report",
            description="Test report for unit tests",
            report_type="performance"
        )
        
        success = self.report_generator.create_report(report)
        self.assertTrue(success)
        
        # Verify report was created
        reports = self.report_generator.list_reports()
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0].name, "Test Report")
    
    def test_report_generation(self):
        """Test report generation."""
        # Create test report
        report = Report(
            report_id="performance_report",
            name="Performance Report",
            description="System performance report",
            report_type="performance"
        )
        self.report_generator.create_report(report)
        
        # Generate report
        generated = self.report_generator.generate_report("performance_report")
        
        self.assertIn("report_id", generated)
        self.assertIn("name", generated)
        self.assertIn("generated_at", generated)
        self.assertIn("data", generated)
        
        # Verify report data structure
        data = generated["data"]
        self.assertIn("summary", data)
        self.assertIn("metrics", data)


class TestSecurity(unittest.TestCase):
    """Test security scanning and compliance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scanner = SecurityScanner()
        self.compliance = ComplianceFramework()
        self.auditor = SecurityAuditor()
        
        # Create temporary test files
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_code.py")
        
        # Write test code with vulnerabilities
        with open(self.test_file, 'w') as f:
            f.write("""
import hashlib
import subprocess

# Vulnerable code patterns
password = "hardcoded_password"
result = eval(user_input)
subprocess.call(command, shell=True)
hash_value = hashlib.md5(data).hexdigest()
""")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_security_scanner_initialization(self):
        """Test security scanner initialization."""
        self.assertIsInstance(self.scanner, SecurityScanner)
        self.assertIsInstance(self.scanner.vulnerability_patterns, dict)
        
        # Verify pattern categories exist
        expected_categories = ["injection", "crypto", "auth", "data"]
        for category in expected_categories:
            self.assertIn(category, self.scanner.vulnerability_patterns)
    
    def test_file_vulnerability_scanning(self):
        """Test file vulnerability scanning."""
        vulnerabilities = self.scanner.scan_file(self.test_file)
        
        self.assertGreater(len(vulnerabilities), 0)
        
        # Verify vulnerability types found
        vuln_titles = [v.title for v in vulnerabilities]
        self.assertIn("Hardcoded Password", vuln_titles)
        self.assertIn("Code Injection Risk", vuln_titles)
        self.assertIn("Weak Hash Algorithm", vuln_titles)
    
    def test_directory_vulnerability_scanning(self):
        """Test directory vulnerability scanning."""
        vulnerabilities = self.scanner.scan_directory(self.test_dir)
        
        self.assertGreater(len(vulnerabilities), 0)
        
        # Verify scan results are stored
        self.assertEqual(len(self.scanner.scan_results), len(vulnerabilities))
    
    def test_vulnerability_summary(self):
        """Test vulnerability summary generation."""
        # Scan test file first
        self.scanner.scan_file(self.test_file)
        
        summary = self.scanner.get_vulnerability_summary()
        
        self.assertIn("total", summary)
        self.assertIn("by_severity", summary)
        self.assertIn("by_category", summary)
        self.assertGreater(summary["total"], 0)
    
    def test_compliance_framework_initialization(self):
        """Test compliance framework initialization."""
        self.assertIsInstance(self.compliance, ComplianceFramework)
        self.assertIsInstance(self.compliance.compliance_checks, dict)
        
        # Verify compliance standards exist
        expected_standards = [ComplianceStandard.SOC2, ComplianceStandard.GDPR, ComplianceStandard.ISO27001]
        for standard in expected_standards:
            self.assertIn(standard, self.compliance.compliance_checks)
    
    def test_compliance_checks(self):
        """Test compliance check execution."""
        # Run SOC2 compliance checks
        results = self.compliance.run_compliance_checks(ComplianceStandard.SOC2)
        
        self.assertGreater(len(results), 0)
        
        # Verify check results structure
        for result in results:
            self.assertIsInstance(result.check_id, str)
            self.assertIsInstance(result.standard, ComplianceStandard)
            self.assertIn(result.status, ["pass", "fail", "warning", "not_applicable"])
    
    def test_compliance_summary(self):
        """Test compliance summary generation."""
        # Run compliance checks first
        self.compliance.run_compliance_checks(ComplianceStandard.SOC2)
        
        summary = self.compliance.get_compliance_summary()
        
        self.assertIn("total", summary)
        self.assertIn("by_status", summary)
        self.assertIn("by_standard", summary)
        self.assertIn("pass_rate", summary)
    
    def test_security_audit_logging(self):
        """Test security audit logging."""
        # Log authentication event
        event_id = self.auditor.log_authentication_event(
            user_id="test_user",
            result="success",
            ip_address="192.168.1.100",
            auth_method="jwt"
        )
        
        self.assertIsInstance(event_id, str)
        
        # Verify event was logged
        events = self.auditor.get_events(event_type="auth")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].user_id, "test_user")
        self.assertEqual(events[0].result, "success")
    
    def test_security_metrics(self):
        """Test security metrics generation."""
        # Log some test events
        self.auditor.log_authentication_event("user1", "success")
        self.auditor.log_authentication_event("user2", "failure")
        self.auditor.log_access_event("user1", "api/predictions", "GET", "success")
        
        metrics = self.auditor.get_security_metrics()
        
        self.assertIn("total_events", metrics)
        self.assertIn("by_type", metrics)
        self.assertIn("by_result", metrics)
        self.assertIn("failed_auth_count", metrics)
        self.assertIn("success_rate", metrics)
        
        self.assertEqual(metrics["total_events"], 3)
        self.assertEqual(metrics["failed_auth_count"], 1)


class TestBackupRecovery(unittest.TestCase):
    """Test backup and disaster recovery system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_backup_root = tempfile.mkdtemp()
        self.backup_manager = BackupManager(self.test_backup_root)
        self.recovery_manager = DisasterRecoveryManager()
        
        # Create test source directory
        self.test_source = tempfile.mkdtemp()
        test_file = os.path.join(self.test_source, "test_file.txt")
        with open(test_file, 'w') as f:
            f.write("Test file content")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_backup_root)
        shutil.rmtree(self.test_source)
    
    def test_backup_manager_initialization(self):
        """Test backup manager initialization."""
        self.assertIsInstance(self.backup_manager, BackupManager)
        self.assertTrue(os.path.exists(self.test_backup_root))
        
        # Verify default backup jobs exist
        jobs = list(self.backup_manager.jobs.values())
        self.assertGreater(len(jobs), 0)
        
        job_names = [job.name for job in jobs]
        self.assertIn("Configuration Backup", job_names)
        self.assertIn("Model Backup", job_names)
        self.assertIn("Source Code Backup", job_names)
    
    def test_backup_job_creation(self):
        """Test backup job creation."""
        job = BackupJob(
            job_id="test_backup",
            name="Test Backup",
            backup_type=BackupType.FULL,
            source_paths=[self.test_source],
            destination_path=os.path.join(self.test_backup_root, "test")
        )
        
        success = self.backup_manager.create_backup_job(job)
        self.assertTrue(success)
        
        # Verify job was created
        self.assertIn("test_backup", self.backup_manager.jobs)
    
    def test_backup_execution(self):
        """Test backup execution."""
        # Create test backup job
        job = BackupJob(
            job_id="test_execution",
            name="Test Execution",
            backup_type=BackupType.FULL,
            source_paths=[self.test_source],
            destination_path=os.path.join(self.test_backup_root, "test_exec")
        )
        self.backup_manager.create_backup_job(job)
        
        # Execute backup
        execution = self.backup_manager.execute_backup("test_execution")
        
        self.assertIsNotNone(execution)
        self.assertEqual(execution.job_id, "test_execution")
        self.assertEqual(execution.status.value, "completed")
        self.assertGreater(execution.files_backed_up, 0)
        self.assertTrue(os.path.exists(execution.backup_path))
    
    def test_backup_restoration(self):
        """Test backup restoration."""
        # Create and execute backup
        job = BackupJob(
            job_id="test_restore",
            name="Test Restore",
            backup_type=BackupType.FULL,
            source_paths=[self.test_source],
            destination_path=os.path.join(self.test_backup_root, "test_restore")
        )
        self.backup_manager.create_backup_job(job)
        execution = self.backup_manager.execute_backup("test_restore")
        
        # Create restore directory
        restore_dir = tempfile.mkdtemp()
        
        try:
            # Restore backup
            success = self.backup_manager.restore_backup(execution.execution_id, restore_dir)
            self.assertTrue(success)
            
            # Verify restored files
            restored_files = os.listdir(restore_dir)
            self.assertGreater(len(restored_files), 0)
            
        finally:
            shutil.rmtree(restore_dir)
    
    def test_backup_status(self):
        """Test backup status reporting."""
        status = self.backup_manager.get_backup_status()
        
        self.assertIn("total_jobs", status)
        self.assertIn("enabled_jobs", status)
        self.assertIn("recent_executions", status)
        self.assertIn("successful_executions", status)
        self.assertIn("success_rate", status)
    
    def test_disaster_recovery_manager_initialization(self):
        """Test disaster recovery manager initialization."""
        self.assertIsInstance(self.recovery_manager, DisasterRecoveryManager)
        
        # Verify default recovery plans exist
        plans = self.recovery_manager.list_recovery_plans()
        self.assertGreater(len(plans), 0)
        
        plan_names = [plan.name for plan in plans]
        self.assertIn("Database Recovery", plan_names)
        self.assertIn("Application Recovery", plan_names)
        self.assertIn("Infrastructure Recovery", plan_names)
    
    def test_recovery_plan_creation(self):
        """Test recovery plan creation."""
        plan = RecoveryPlan(
            plan_id="test_recovery",
            name="Test Recovery",
            description="Test recovery plan",
            priority=1,
            recovery_time_objective_hours=1.0,
            recovery_point_objective_hours=0.5,
            procedures=["Step 1", "Step 2", "Step 3"]
        )
        
        success = self.recovery_manager.create_recovery_plan(plan)
        self.assertTrue(success)
        
        # Verify plan was created
        retrieved = self.recovery_manager.get_recovery_plan("test_recovery")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Test Recovery")
    
    def test_recovery_test_execution(self):
        """Test disaster recovery test execution."""
        # Execute recovery test for default plan
        test_results = self.recovery_manager.execute_recovery_test("database_recovery")
        
        self.assertIn("plan_id", test_results)
        self.assertIn("test_date", test_results)
        self.assertIn("procedures_tested", test_results)
        self.assertIn("overall_status", test_results)
        self.assertEqual(test_results["plan_id"], "database_recovery")
    
    def test_recovery_status(self):
        """Test recovery status reporting."""
        status = self.recovery_manager.get_recovery_status()
        
        self.assertIn("total_plans", status)
        self.assertIn("tested_plans", status)
        self.assertIn("recently_tested", status)
        self.assertIn("test_coverage", status)


class TestPhase7Integration(unittest.TestCase):
    """Test Phase 7 component integration."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.monitoring = MonitoringSystem()
        self.dashboard_manager = DashboardManager()
        self.scanner = SecurityScanner()
        self.backup_manager = BackupManager()
    
    def test_monitoring_dashboard_integration(self):
        """Test integration between monitoring and dashboards."""
        # Collect some metrics
        self.monitoring.collect_system_metrics()
        
        # Register monitoring data source
        def get_cpu_metrics():
            return self.monitoring.get_metrics("system.cpu.usage_percent")
        
        self.dashboard_manager.register_data_source("system.cpu.usage_percent", get_cpu_metrics)
        
        # Render dashboard with monitoring data
        rendered = self.dashboard_manager.render_dashboard("system_overview")
        
        self.assertIn("widgets", rendered)
        
        # Find CPU widget and verify it has data
        cpu_widget = None
        for widget in rendered["widgets"]:
            if widget["widget_id"] == "cpu_usage":
                cpu_widget = widget
                break
        
        if cpu_widget:
            self.assertIn("data", cpu_widget)
            self.assertIn("value", cpu_widget["data"])
    
    def test_security_monitoring_integration(self):
        """Test integration between security and monitoring."""
        # Create test file with vulnerabilities
        test_dir = tempfile.mkdtemp()
        test_file = os.path.join(test_dir, "vuln_test.py")
        
        try:
            with open(test_file, 'w') as f:
                f.write('password = "secret123"')
            
            # Scan for vulnerabilities
            vulnerabilities = self.scanner.scan_directory(test_dir)
            
            # Create security metrics
            security_metric = MetricValue(
                name="security.vulnerabilities.count",
                value=len(vulnerabilities),
                timestamp=datetime.now(),
                tags={"severity": "high"}
            )
            
            self.monitoring.add_metric(security_metric)
            
            # Verify metric was added
            security_metrics = self.monitoring.get_metrics("security.vulnerabilities.count")
            self.assertEqual(len(security_metrics), 1)
            self.assertGreater(security_metrics[0].value, 0)
            
        finally:
            shutil.rmtree(test_dir)
    
    def test_backup_monitoring_integration(self):
        """Test integration between backup and monitoring."""
        # Create test backup job
        test_source = tempfile.mkdtemp()
        test_file = os.path.join(test_source, "backup_test.txt")
        
        try:
            with open(test_file, 'w') as f:
                f.write("Test backup content")
            
            job = BackupJob(
                job_id="integration_test",
                name="Integration Test Backup",
                backup_type=BackupType.FULL,
                source_paths=[test_source],
                destination_path=os.path.join(tempfile.gettempdir(), "integration_backup")
            )
            
            self.backup_manager.create_backup_job(job)
            execution = self.backup_manager.execute_backup("integration_test")
            
            # Create backup metrics
            backup_metric = MetricValue(
                name="backup.execution.duration_seconds",
                value=execution.duration_seconds or 0,
                timestamp=datetime.now(),
                tags={"job_id": job.job_id, "status": execution.status.value}
            )
            
            self.monitoring.add_metric(backup_metric)
            
            # Verify metric was added
            backup_metrics = self.monitoring.get_metrics("backup.execution.duration_seconds")
            self.assertEqual(len(backup_metrics), 1)
            
        finally:
            shutil.rmtree(test_source)


def run_phase7_tests():
    """Run all Phase 7 tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMonitoringSystem,
        TestDashboards,
        TestSecurity,
        TestBackupRecovery,
        TestPhase7Integration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_phase7_tests()
    print(f"\nPhase 7 tests {'PASSED' if success else 'FAILED'}")
