"""
Direct Phase 7 validation test for ML-TA monitoring and operations.

This test validates core Phase 7 functionality:
- Monitoring system basic operations
- Dashboard creation and rendering
- Security scanning capabilities
- Backup and recovery operations
"""

import os
import sys
import tempfile
import shutil
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_monitoring_system():
    """Test monitoring system core functionality."""
    print("Testing monitoring system...")
    
    try:
        from src.monitoring import MonitoringSystem, MetricValue, AlertManager, AlertRule
        
        # Test monitoring system initialization
        monitoring = MonitoringSystem()
        assert monitoring is not None, "Monitoring system failed to initialize"
        
        # Test metric collection
        metric = MetricValue(
            name="test.metric",
            value=42.5,
            timestamp=datetime.now()
        )
        monitoring.add_metric(metric)
        
        metrics = monitoring.get_metrics("test.metric")
        assert len(metrics) == 1, "Metric not added correctly"
        assert metrics[0].value == 42.5, "Metric value incorrect"
        
        # Test system metrics collection
        monitoring.collect_system_metrics()
        all_metrics = monitoring.get_all_metrics()
        assert len(all_metrics) > 1, "System metrics not collected"
        
        # Test alert manager
        alert_manager = AlertManager()
        rule = AlertRule(
            name="test_rule",
            metric_name="test.metric",
            condition="gt",
            threshold=40.0,
            severity="warning"
        )
        
        success = alert_manager.add_alert_rule(rule)
        assert success, "Alert rule not added"
        
        alerts = alert_manager.evaluate_alerts([metric])
        assert len(alerts) == 1, "Alert not triggered"
        assert alerts[0].rule_name == "test_rule", "Alert rule name incorrect"
        
        print("✓ Monitoring system tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Monitoring system test failed: {e}")
        return False


def test_dashboard_system():
    """Test dashboard system core functionality."""
    print("Testing dashboard system...")
    
    try:
        from src.dashboards import DashboardManager, Dashboard, DashboardWidget, ReportGenerator, Report
        from src.monitoring import MetricValue
        
        # Test dashboard manager initialization
        dashboard_manager = DashboardManager()
        assert dashboard_manager is not None, "Dashboard manager failed to initialize"
        
        # Test default dashboard exists
        default_dashboard = dashboard_manager.get_dashboard("system_overview")
        assert default_dashboard is not None, "Default dashboard not found"
        assert default_dashboard.name == "System Overview", "Default dashboard name incorrect"
        
        # Test custom dashboard creation
        widget = DashboardWidget(
            widget_id="test_widget",
            widget_type="metric",
            title="Test Widget",
            data_source="test.data"
        )
        
        dashboard = Dashboard(
            dashboard_id="test_dashboard",
            name="Test Dashboard",
            description="Test dashboard",
            widgets=[widget]
        )
        
        success = dashboard_manager.create_dashboard(dashboard)
        assert success, "Dashboard creation failed"
        
        # Test data source registration
        def test_data_source():
            return [MetricValue(name="test.data", value=100.0, timestamp=datetime.now())]
        
        dashboard_manager.register_data_source("test.data", test_data_source)
        
        # Test dashboard rendering
        rendered = dashboard_manager.render_dashboard("test_dashboard")
        assert "dashboard_id" in rendered, "Dashboard rendering failed"
        assert rendered["dashboard_id"] == "test_dashboard", "Dashboard ID incorrect"
        assert len(rendered["widgets"]) == 1, "Widget count incorrect"
        
        # Test report generator
        report_generator = ReportGenerator()
        report = Report(
            report_id="test_report",
            name="Test Report",
            description="Test report",
            report_type="performance"
        )
        
        success = report_generator.create_report(report)
        assert success, "Report creation failed"
        
        generated = report_generator.generate_report("test_report")
        assert "report_id" in generated, "Report generation failed"
        assert "data" in generated, "Report data missing"
        
        print("✓ Dashboard system tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Dashboard system test failed: {e}")
        return False


def test_security_system():
    """Test security system core functionality."""
    print("Testing security system...")
    
    try:
        from src.security import SecurityScanner, ComplianceFramework, SecurityAuditor, ComplianceStandard
        
        # Test security scanner
        scanner = SecurityScanner()
        assert scanner is not None, "Security scanner failed to initialize"
        assert len(scanner.vulnerability_patterns) > 0, "No vulnerability patterns loaded"
        
        # Create test file with vulnerabilities
        test_dir = tempfile.mkdtemp()
        test_file = os.path.join(test_dir, "test_vuln.py")
        
        try:
            with open(test_file, 'w') as f:
                f.write("""
password = "hardcoded_secret"
result = eval(user_input)
import hashlib
hash_val = hashlib.md5(data).hexdigest()
""")
            
            # Test vulnerability scanning
            vulnerabilities = scanner.scan_file(test_file)
            assert len(vulnerabilities) > 0, "No vulnerabilities detected"
            
            # Verify specific vulnerabilities found
            vuln_titles = [v.title for v in vulnerabilities]
            assert "Hardcoded Password" in vuln_titles, "Hardcoded password not detected"
            assert "Code Injection Risk" in vuln_titles, "Code injection not detected"
            
            # Test vulnerability summary
            summary = scanner.get_vulnerability_summary()
            assert summary["total"] > 0, "Vulnerability summary incorrect"
            
        finally:
            shutil.rmtree(test_dir)
        
        # Test compliance framework
        compliance = ComplianceFramework()
        assert compliance is not None, "Compliance framework failed to initialize"
        
        # Test compliance checks
        results = compliance.run_compliance_checks(ComplianceStandard.SOC2)
        assert len(results) > 0, "No compliance checks executed"
        
        # Verify check results structure
        for result in results:
            assert hasattr(result, 'check_id'), "Check ID missing"
            assert hasattr(result, 'status'), "Check status missing"
            assert result.status in ["pass", "fail", "warning", "not_applicable"], "Invalid check status"
        
        # Test security auditor
        auditor = SecurityAuditor()
        assert auditor is not None, "Security auditor failed to initialize"
        
        # Test audit logging
        event_id = auditor.log_authentication_event(
            user_id="test_user",
            result="success",
            auth_method="jwt"
        )
        assert event_id is not None, "Audit event not logged"
        
        # Test event retrieval
        events = auditor.get_events(event_type="auth")
        assert len(events) == 1, "Audit event not retrieved"
        assert events[0].user_id == "test_user", "Audit event data incorrect"
        
        print("✓ Security system tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Security system test failed: {e}")
        return False


def test_backup_recovery_system():
    """Test backup and recovery system core functionality."""
    print("Testing backup and recovery system...")
    
    try:
        from src.backup_recovery import BackupManager, DisasterRecoveryManager, BackupJob, BackupType, RecoveryPlan
        
        # Test backup manager
        test_backup_root = tempfile.mkdtemp()
        backup_manager = BackupManager(test_backup_root)
        
        try:
            assert backup_manager is not None, "Backup manager failed to initialize"
            assert os.path.exists(test_backup_root), "Backup directory not created"
            
            # Test default jobs exist
            jobs = list(backup_manager.jobs.values())
            assert len(jobs) > 0, "No default backup jobs created"
            
            job_names = [job.name for job in jobs]
            assert "Configuration Backup" in job_names, "Configuration backup job missing"
            
            # Test custom backup job
            test_source = tempfile.mkdtemp()
            test_file = os.path.join(test_source, "test_file.txt")
            
            try:
                with open(test_file, 'w') as f:
                    f.write("Test backup content")
                
                job = BackupJob(
                    job_id="test_backup",
                    name="Test Backup",
                    backup_type=BackupType.FULL,
                    source_paths=[test_source],
                    destination_path=os.path.join(test_backup_root, "test")
                )
                
                success = backup_manager.create_backup_job(job)
                assert success, "Backup job creation failed"
                
                # Test backup execution
                execution = backup_manager.execute_backup("test_backup")
                assert execution is not None, "Backup execution failed"
                assert execution.status.value == "completed", "Backup not completed"
                assert execution.files_backed_up > 0, "No files backed up"
                
                # Test backup restoration
                restore_dir = tempfile.mkdtemp()
                try:
                    success = backup_manager.restore_backup(execution.execution_id, restore_dir)
                    assert success, "Backup restoration failed"
                    
                    restored_files = os.listdir(restore_dir)
                    assert len(restored_files) > 0, "No files restored"
                    
                finally:
                    shutil.rmtree(restore_dir)
                
            finally:
                shutil.rmtree(test_source)
            
        finally:
            shutil.rmtree(test_backup_root)
        
        # Test disaster recovery manager
        recovery_manager = DisasterRecoveryManager()
        assert recovery_manager is not None, "Recovery manager failed to initialize"
        
        # Test default recovery plans
        plans = recovery_manager.list_recovery_plans()
        assert len(plans) > 0, "No default recovery plans created"
        
        plan_names = [plan.name for plan in plans]
        assert "Database Recovery" in plan_names, "Database recovery plan missing"
        
        # Test custom recovery plan
        plan = RecoveryPlan(
            plan_id="test_recovery",
            name="Test Recovery",
            description="Test recovery plan",
            priority=1,
            recovery_time_objective_hours=1.0,
            recovery_point_objective_hours=0.5,
            procedures=["Step 1", "Step 2"]
        )
        
        success = recovery_manager.create_recovery_plan(plan)
        assert success, "Recovery plan creation failed"
        
        # Test recovery test execution
        test_results = recovery_manager.execute_recovery_test("test_recovery")
        assert "plan_id" in test_results, "Recovery test failed"
        assert test_results["plan_id"] == "test_recovery", "Recovery test plan ID incorrect"
        
        print("✓ Backup and recovery system tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Backup and recovery system test failed: {e}")
        return False


def run_direct_phase7_validation():
    """Run direct Phase 7 validation tests."""
    print("=" * 60)
    print("PHASE 7 DIRECT VALIDATION TEST")
    print("=" * 60)
    
    tests = [
        ("Monitoring System", test_monitoring_system),
        ("Dashboard System", test_dashboard_system),
        ("Security System", test_security_system),
        ("Backup Recovery System", test_backup_recovery_system)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"FAILED: {test_name}")
    
    print("\n" + "=" * 60)
    print(f"PHASE 7 VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL PHASE 7 CORE FUNCTIONALITY VALIDATED")
        print("✓ Phase 7 quality gate: PASSED")
        return True
    else:
        print("✗ PHASE 7 VALIDATION FAILED")
        print("✗ Phase 7 quality gate: FAILED")
        return False


if __name__ == "__main__":
    success = run_direct_phase7_validation()
    sys.exit(0 if success else 1)
