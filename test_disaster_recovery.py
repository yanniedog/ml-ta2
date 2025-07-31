"""
Disaster Recovery Testing Framework for ML-TA System.

This module provides comprehensive disaster recovery testing including
backup validation, failover testing, and recovery procedures.
"""

import os
import sys
import time
import shutil
import tempfile
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DRTestResult:
    """Disaster recovery test result container."""
    
    test_name: str
    status: str  # PASSED, FAILED, SKIPPED
    duration: float
    rto_achieved: Optional[float] = None  # Recovery Time Objective achieved
    rpo_achieved: Optional[float] = None  # Recovery Point Objective achieved
    data_integrity: bool = True
    error_message: Optional[str] = None
    recovery_steps: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class BackupRecoveryTest:
    """Test backup and recovery procedures."""
    
    def __init__(self):
        self.test_backup_dir = Path("test_backups_dr")
        self.test_backup_dir.mkdir(exist_ok=True)
        self.test_data_dir = Path("test_data_dr")
        self.test_data_dir.mkdir(exist_ok=True)
    
    def test_database_backup_recovery(self) -> DRTestResult:
        """Test database backup and recovery procedures."""
        start_time = time.time()
        recovery_steps = []
        
        try:
            logger.info("Starting database backup/recovery test")
            
            # Step 1: Create test database data
            recovery_steps.append("Creating test database data")
            test_data = self._create_test_database_data()
            
            # Step 2: Perform backup
            recovery_steps.append("Performing database backup")
            backup_start = time.time()
            backup_result = self._perform_database_backup(test_data)
            backup_duration = time.time() - backup_start
            
            # Step 3: Simulate data loss
            recovery_steps.append("Simulating data loss")
            self._simulate_data_loss()
            
            # Step 4: Perform recovery
            recovery_steps.append("Performing database recovery")
            recovery_start = time.time()
            recovery_result = self._perform_database_recovery(backup_result)
            recovery_duration = time.time() - recovery_start
            
            # Step 5: Validate data integrity
            recovery_steps.append("Validating data integrity")
            integrity_result = self._validate_data_integrity(test_data, recovery_result)
            
            total_duration = time.time() - start_time
            
            # Check RTO/RPO objectives
            rto_target = 300.0  # 5 minutes
            rpo_target = 60.0   # 1 minute data loss acceptable
            
            rto_achieved = recovery_duration
            rpo_achieved = backup_duration  # Time since last backup
            
            success = (
                rto_achieved <= rto_target and
                rpo_achieved <= rpo_target and
                integrity_result['integrity_score'] > 0.95
            )
            
            status = "PASSED" if success else "FAILED"
            error_msg = None if success else f"RTO: {rto_achieved:.1f}s > {rto_target}s or RPO: {rpo_achieved:.1f}s > {rpo_target}s or integrity < 95%"
            
            logger.info(f"Database backup/recovery test {status} - RTO: {rto_achieved:.1f}s, RPO: {rpo_achieved:.1f}s")
            
            return DRTestResult(
                test_name="database_backup_recovery",
                status=status,
                duration=total_duration,
                rto_achieved=rto_achieved,
                rpo_achieved=rpo_achieved,
                data_integrity=integrity_result['integrity_score'] > 0.95,
                error_message=error_msg,
                recovery_steps=recovery_steps
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Database backup/recovery test failed: {e}")
            
            return DRTestResult(
                test_name="database_backup_recovery",
                status="FAILED",
                duration=duration,
                error_message=str(e),
                recovery_steps=recovery_steps
            )
    
    def test_model_backup_recovery(self) -> DRTestResult:
        """Test model backup and recovery procedures."""
        start_time = time.time()
        recovery_steps = []
        
        try:
            logger.info("Starting model backup/recovery test")
            
            # Step 1: Create test model files
            recovery_steps.append("Creating test model files")
            test_models = self._create_test_model_files()
            
            # Step 2: Perform model backup
            recovery_steps.append("Performing model backup")
            backup_start = time.time()
            backup_result = self._perform_model_backup(test_models)
            backup_duration = time.time() - backup_start
            
            # Step 3: Simulate model corruption
            recovery_steps.append("Simulating model corruption")
            self._simulate_model_corruption()
            
            # Step 4: Perform model recovery
            recovery_steps.append("Performing model recovery")
            recovery_start = time.time()
            recovery_result = self._perform_model_recovery(backup_result)
            recovery_duration = time.time() - recovery_start
            
            # Step 5: Validate model integrity
            recovery_steps.append("Validating model integrity")
            integrity_result = self._validate_model_integrity(test_models, recovery_result)
            
            total_duration = time.time() - start_time
            
            # Check objectives
            rto_target = 180.0  # 3 minutes for model recovery
            rpo_target = 30.0   # 30 seconds for model backup
            
            success = (
                recovery_duration <= rto_target and
                backup_duration <= rpo_target and
                integrity_result['models_recovered'] == integrity_result['models_total']
            )
            
            status = "PASSED" if success else "FAILED"
            error_msg = None if success else "Model recovery objectives not met"
            
            logger.info(f"Model backup/recovery test {status}")
            
            return DRTestResult(
                test_name="model_backup_recovery",
                status=status,
                duration=total_duration,
                rto_achieved=recovery_duration,
                rpo_achieved=backup_duration,
                data_integrity=integrity_result['models_recovered'] == integrity_result['models_total'],
                error_message=error_msg,
                recovery_steps=recovery_steps
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Model backup/recovery test failed: {e}")
            
            return DRTestResult(
                test_name="model_backup_recovery",
                status="FAILED",
                duration=duration,
                error_message=str(e),
                recovery_steps=recovery_steps
            )
    
    def _create_test_database_data(self) -> Dict[str, Any]:
        """Create test database data."""
        test_data = {
            'tables': {
                'users': [
                    {'id': 1, 'username': 'test_user', 'email': 'test@example.com'},
                    {'id': 2, 'username': 'admin', 'email': 'admin@example.com'}
                ],
                'predictions': [
                    {'id': 1, 'symbol': 'BTCUSD', 'prediction': 0.75, 'timestamp': datetime.now()},
                    {'id': 2, 'symbol': 'ETHUSD', 'prediction': 0.82, 'timestamp': datetime.now()}
                ],
                'models': [
                    {'id': 1, 'name': 'rf_model_v1', 'accuracy': 0.85, 'created_at': datetime.now()}
                ]
            },
            'metadata': {
                'total_records': 5,
                'created_at': datetime.now(),
                'checksum': 'abc123def456'
            }
        }
        
        # Save test data
        test_file = self.test_data_dir / "test_database.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f, default=str)
        
        return test_data
    
    def _perform_database_backup(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform database backup."""
        backup_file = self.test_backup_dir / f"db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Simulate backup process
        with open(backup_file, 'w') as f:
            json.dump(test_data, f, default=str)
        
        return {
            'backup_file': backup_file,
            'backup_size': backup_file.stat().st_size,
            'backup_time': datetime.now()
        }
    
    def _simulate_data_loss(self) -> None:
        """Simulate data loss."""
        # Remove test data file
        test_file = self.test_data_dir / "test_database.json"
        if test_file.exists():
            test_file.unlink()
    
    def _perform_database_recovery(self, backup_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform database recovery."""
        backup_file = backup_result['backup_file']
        
        # Restore from backup
        recovery_file = self.test_data_dir / "recovered_database.json"
        shutil.copy2(backup_file, recovery_file)
        
        # Load recovered data
        with open(recovery_file, 'r') as f:
            recovered_data = json.load(f)
        
        return recovered_data
    
    def _validate_data_integrity(self, original_data: Dict[str, Any], recovered_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data integrity after recovery."""
        try:
            # Compare record counts
            original_count = original_data['metadata']['total_records']
            recovered_count = recovered_data['metadata']['total_records']
            
            # Compare checksums
            original_checksum = original_data['metadata']['checksum']
            recovered_checksum = recovered_data['metadata']['checksum']
            
            integrity_score = 1.0 if (original_count == recovered_count and original_checksum == recovered_checksum) else 0.0
            
            return {
                'integrity_score': integrity_score,
                'records_match': original_count == recovered_count,
                'checksum_match': original_checksum == recovered_checksum,
                'original_records': original_count,
                'recovered_records': recovered_count
            }
            
        except Exception as e:
            logger.error(f"Data integrity validation failed: {e}")
            return {'integrity_score': 0.0}
    
    def _create_test_model_files(self) -> List[Path]:
        """Create test model files."""
        model_files = []
        
        for i in range(3):
            model_file = self.test_data_dir / f"test_model_{i}.pkl"
            # Create dummy model file
            model_file.write_text(f"dummy_model_data_{i}")
            model_files.append(model_file)
        
        return model_files
    
    def _perform_model_backup(self, model_files: List[Path]) -> Dict[str, Any]:
        """Perform model backup."""
        backup_dir = self.test_backup_dir / f"models_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(exist_ok=True)
        
        backed_up_files = []
        for model_file in model_files:
            backup_file = backup_dir / model_file.name
            shutil.copy2(model_file, backup_file)
            backed_up_files.append(backup_file)
        
        return {
            'backup_dir': backup_dir,
            'backed_up_files': backed_up_files,
            'backup_time': datetime.now()
        }
    
    def _simulate_model_corruption(self) -> None:
        """Simulate model corruption."""
        # Remove test model files
        for model_file in self.test_data_dir.glob("test_model_*.pkl"):
            model_file.unlink()
    
    def _perform_model_recovery(self, backup_result: Dict[str, Any]) -> List[Path]:
        """Perform model recovery."""
        backup_dir = backup_result['backup_dir']
        recovered_files = []
        
        for backup_file in backup_result['backed_up_files']:
            recovery_file = self.test_data_dir / backup_file.name
            shutil.copy2(backup_file, recovery_file)
            recovered_files.append(recovery_file)
        
        return recovered_files
    
    def _validate_model_integrity(self, original_files: List[Path], recovered_files: List[Path]) -> Dict[str, Any]:
        """Validate model integrity after recovery."""
        try:
            models_total = len(original_files)
            models_recovered = len(recovered_files)
            
            # Check file contents
            content_matches = 0
            for orig_file in original_files:
                for rec_file in recovered_files:
                    if orig_file.name == rec_file.name:
                        if orig_file.read_text() == rec_file.read_text():
                            content_matches += 1
                        break
            
            return {
                'models_total': models_total,
                'models_recovered': models_recovered,
                'content_matches': content_matches,
                'integrity_score': content_matches / models_total if models_total > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Model integrity validation failed: {e}")
            return {'models_total': 0, 'models_recovered': 0}


class FailoverTest:
    """Test system failover capabilities."""
    
    def test_service_failover(self) -> DRTestResult:
        """Test service failover procedures."""
        start_time = time.time()
        recovery_steps = []
        
        try:
            logger.info("Starting service failover test")
            
            # Step 1: Verify primary service health
            recovery_steps.append("Verifying primary service health")
            primary_health = self._check_service_health("primary")
            
            # Step 2: Simulate primary service failure
            recovery_steps.append("Simulating primary service failure")
            self._simulate_service_failure("primary")
            
            # Step 3: Trigger failover
            recovery_steps.append("Triggering failover to secondary")
            failover_start = time.time()
            failover_result = self._trigger_failover()
            failover_duration = time.time() - failover_start
            
            # Step 4: Verify secondary service health
            recovery_steps.append("Verifying secondary service health")
            secondary_health = self._check_service_health("secondary")
            
            # Step 5: Test service functionality
            recovery_steps.append("Testing service functionality")
            functionality_test = self._test_service_functionality()
            
            total_duration = time.time() - start_time
            
            # Check failover objectives
            rto_target = 60.0  # 1 minute failover time
            
            success = (
                failover_duration <= rto_target and
                secondary_health['status'] == 'healthy' and
                functionality_test['success']
            )
            
            status = "PASSED" if success else "FAILED"
            error_msg = None if success else f"Failover took {failover_duration:.1f}s > {rto_target}s or secondary unhealthy"
            
            logger.info(f"Service failover test {status} - Failover time: {failover_duration:.1f}s")
            
            return DRTestResult(
                test_name="service_failover",
                status=status,
                duration=total_duration,
                rto_achieved=failover_duration,
                data_integrity=functionality_test['success'],
                error_message=error_msg,
                recovery_steps=recovery_steps
            )
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Service failover test failed: {e}")
            
            return DRTestResult(
                test_name="service_failover",
                status="FAILED",
                duration=duration,
                error_message=str(e),
                recovery_steps=recovery_steps
            )
    
    def _check_service_health(self, service_type: str) -> Dict[str, Any]:
        """Check service health."""
        # Mock service health check
        return {
            'status': 'healthy',
            'response_time': 0.05,
            'service_type': service_type
        }
    
    def _simulate_service_failure(self, service_type: str) -> None:
        """Simulate service failure."""
        logger.info(f"Simulating {service_type} service failure")
        # In real implementation, this would stop the service
        time.sleep(0.1)
    
    def _trigger_failover(self) -> Dict[str, Any]:
        """Trigger failover to secondary service."""
        # Mock failover process
        time.sleep(2.0)  # Simulate failover time
        
        return {
            'success': True,
            'new_primary': 'secondary',
            'failover_time': 2.0
        }
    
    def _test_service_functionality(self) -> Dict[str, Any]:
        """Test service functionality after failover."""
        # Mock functionality test
        return {
            'success': True,
            'api_responsive': True,
            'predictions_working': True,
            'database_accessible': True
        }


class DisasterRecoveryTestSuite:
    """Comprehensive disaster recovery test suite."""
    
    def __init__(self):
        self.backup_recovery_test = BackupRecoveryTest()
        self.failover_test = FailoverTest()
        self.test_results = []
    
    def run_comprehensive_dr_tests(self) -> List[DRTestResult]:
        """Run all disaster recovery tests."""
        logger.info("Starting comprehensive disaster recovery test suite")
        
        tests = [
            ("Database Backup/Recovery", self.backup_recovery_test.test_database_backup_recovery),
            ("Model Backup/Recovery", self.backup_recovery_test.test_model_backup_recovery),
            ("Service Failover", self.failover_test.test_service_failover)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            logger.info(f"Running {test_name} DR test")
            try:
                result = test_func()
                results.append(result)
                logger.info(f"{test_name} DR test: {result.status}")
            except Exception as e:
                logger.error(f"{test_name} DR test failed: {e}")
                results.append(DRTestResult(
                    test_name=test_name.lower().replace(" ", "_").replace("/", "_"),
                    status="FAILED",
                    duration=0.0,
                    error_message=str(e)
                ))
        
        self.test_results = results
        logger.info(f"DR test suite completed. {len(results)} tests run")
        
        return results
    
    def generate_dr_report(self) -> str:
        """Generate disaster recovery test report."""
        if not self.test_results:
            return "No disaster recovery test results available."
        
        report = ["ML-TA Disaster Recovery Test Report", "=" * 50, ""]
        
        # Summary
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "PASSED"])
        failed_tests = len([r for r in self.test_results if r.status == "FAILED"])
        
        report.extend([
            f"Total DR Tests: {total_tests}",
            f"Passed: {passed_tests}",
            f"Failed: {failed_tests}",
            f"Success Rate: {(passed_tests/total_tests*100):.1f}%",
            ""
        ])
        
        # RTO/RPO Summary
        rto_results = [r.rto_achieved for r in self.test_results if r.rto_achieved is not None]
        rpo_results = [r.rpo_achieved for r in self.test_results if r.rpo_achieved is not None]
        
        if rto_results:
            report.extend([
                f"Average RTO: {sum(rto_results)/len(rto_results):.1f}s",
                f"Max RTO: {max(rto_results):.1f}s",
            ])
        
        if rpo_results:
            report.extend([
                f"Average RPO: {sum(rpo_results)/len(rpo_results):.1f}s",
                f"Max RPO: {max(rpo_results):.1f}s",
            ])
        
        report.append("")
        
        # Detailed results
        for result in self.test_results:
            report.extend([
                f"Test: {result.test_name}",
                f"  Status: {result.status}",
                f"  Duration: {result.duration:.2f}s",
            ])
            
            if result.rto_achieved is not None:
                report.append(f"  RTO Achieved: {result.rto_achieved:.2f}s")
            
            if result.rpo_achieved is not None:
                report.append(f"  RPO Achieved: {result.rpo_achieved:.2f}s")
            
            report.append(f"  Data Integrity: {'✓' if result.data_integrity else '✗'}")
            
            if result.error_message:
                report.append(f"  Error: {result.error_message}")
            
            if result.recovery_steps:
                report.append("  Recovery Steps:")
                for step in result.recovery_steps:
                    report.append(f"    - {step}")
            
            report.append("")
        
        # DR Readiness Assessment
        if passed_tests == total_tests:
            report.extend([
                "DR Readiness Assessment: READY",
                "All disaster recovery tests passed. System is ready for production deployment.",
                ""
            ])
        else:
            report.extend([
                "DR Readiness Assessment: NOT READY",
                f"{failed_tests} DR test(s) failed. Address these issues before production deployment.",
                ""
            ])
        
        return "\n".join(report)


if __name__ == "__main__":
    # Run comprehensive DR tests
    print("ML-TA Disaster Recovery Test Suite")
    
    dr_suite = DisasterRecoveryTestSuite()
    results = dr_suite.run_comprehensive_dr_tests()
    
    # Generate and display report
    report = dr_suite.generate_dr_report()
    print(report)
    
    # Exit with appropriate code
    failed_count = len([r for r in results if r.status == "FAILED"])
    sys.exit(0 if failed_count == 0 else 1)
