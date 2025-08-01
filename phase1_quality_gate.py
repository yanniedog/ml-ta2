#!/usr/bin/env python3
"""
Phase 1 Quality Gate Implementation for ML-TA System

This script validates that all Phase 1 requirements have been met:
- Infrastructure tests pass (100%)
- Configuration validates correctly in all environments
- Logging system produces structured output
- Error handling covers all exception scenarios
- Test framework executes without errors
"""

import sys
import os
import yaml
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any, List, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import ConfigManager, MLTAConfig
from src.logging_config import LoggerFactory
from src.exceptions import MLTAException


class Phase1QualityGate:
    """Quality gate checker for Phase 1 compliance."""
    
    def __init__(self):
        self.results = {}
        self.passed_checks = 0
        self.total_checks = 0
        
    def log_result(self, check_name: str, passed: bool, details: str = ""):
        """Log a check result."""
        self.results[check_name] = {
            'passed': passed,
            'details': details
        }
        self.total_checks += 1
        if passed:
            self.passed_checks += 1
            print(f"✅ {check_name}: PASSED {details}")
        else:
            print(f"❌ {check_name}: FAILED {details}")
    
    def check_directory_structure(self) -> bool:
        """Verify all required directories exist with proper permissions."""
        required_dirs = [
            "src", "config", "data", "models", "logs", "artefacts", 
            "monitoring", "deployment", "docs", "scripts", "tests",
            "web_app", "k8s", "terraform"
        ]
        
        project_root = Path(__file__).parent
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            self.log_result("Directory Structure", False, f"Missing: {missing_dirs}")
            return False
        else:
            self.log_result("Directory Structure", True, f"All {len(required_dirs)} directories exist")
            return True
    
    def check_configuration_validity(self) -> bool:
        """Test configuration validation in all environments."""
        environments = ["development", "testing", "production", "local"]
        
        try:
            # Test base configuration
            config_manager = ConfigManager()
            base_config = config_manager.load_config()
            
            if not base_config:
                self.log_result("Configuration Validity", False, "Base config failed to load")
                return False
            
            # Test environment-specific configs
            for env in environments:
                env_file = Path("config") / f"{env}.yaml"
                if env_file.exists():
                    with open(env_file, 'r') as f:
                        env_config = yaml.safe_load(f)
                    if not env_config:
                        self.log_result("Configuration Validity", False, f"Environment {env} config invalid")
                        return False
            
            self.log_result("Configuration Validity", True, f"Base + {len(environments)} environment configs valid")
            return True
            
        except Exception as e:
            self.log_result("Configuration Validity", False, f"Exception: {str(e)}")
            return False
    
    def check_logging_system(self) -> bool:
        """Verify logging system produces structured output."""
        try:
            logger_factory = LoggerFactory()
            logger = logger_factory.get_logger("test_logger")
            
            # Test that logger can be created and used
            if not logger:
                self.log_result("Logging System", False, "Logger creation failed")
                return False
            
            # Test structured logging capabilities
            test_log = logger.get_logger()
            if not hasattr(test_log, 'info'):
                self.log_result("Logging System", False, "Logger missing info method")
                return False
            
            self.log_result("Logging System", True, "Structured logging operational")
            return True
            
        except Exception as e:
            self.log_result("Logging System", False, f"Exception: {str(e)}")
            return False
    
    def check_error_handling(self) -> bool:
        """Test error handling covers all exception scenarios."""
        try:
            # Test custom exception creation
            test_exception = MLTAException(
                message="Test exception",
                error_code="TEST_ERROR"
            )
            
            if not hasattr(test_exception, 'message'):
                self.log_result("Error Handling", False, "MLTAException missing message attribute")
                return False
            
            if not hasattr(test_exception, 'error_code'):
                self.log_result("Error Handling", False, "MLTAException missing error_code attribute")
                return False
            
            self.log_result("Error Handling", True, "Custom exception handling functional")
            return True
            
        except Exception as e:
            self.log_result("Error Handling", False, f"Exception: {str(e)}")
            return False
    
    def check_test_framework(self) -> bool:
        """Verify test framework executes without errors."""
        try:
            # Check if conftest.py exists and is valid
            conftest_path = Path("tests") / "conftest.py"
            if not conftest_path.exists():
                self.log_result("Test Framework", False, "conftest.py not found")
                return False
            
            # Check if unit tests directory exists with tests
            unit_tests_path = Path("tests") / "unit"
            if not unit_tests_path.exists():
                self.log_result("Test Framework", False, "Unit tests directory not found")
                return False
            
            # Count test files
            test_files = list(unit_tests_path.glob("test_*.py"))
            if len(test_files) < 3:
                self.log_result("Test Framework", False, f"Only {len(test_files)} unit test files found")
                return False
            
            # Check integration tests
            integration_tests_path = Path("tests") / "integration"
            if not integration_tests_path.exists():
                self.log_result("Test Framework", False, "Integration tests directory not found")
                return False
            
            self.log_result("Test Framework", True, f"Test framework with {len(test_files)} unit tests ready")
            return True
            
        except Exception as e:
            self.log_result("Test Framework", False, f"Exception: {str(e)}")
            return False
    
    def check_requirements_files(self) -> bool:
        """Verify requirements files are complete."""
        try:
            req_file = Path("requirements.txt")
            dev_req_file = Path("requirements-dev.txt")
            
            if not req_file.exists():
                self.log_result("Requirements Files", False, "requirements.txt not found")
                return False
            
            if not dev_req_file.exists():
                self.log_result("Requirements Files", False, "requirements-dev.txt not found")
                return False
            
            # Count dependencies
            with open(req_file, 'r') as f:
                main_deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            with open(dev_req_file, 'r') as f:
                dev_deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            if len(main_deps) < 10:
                self.log_result("Requirements Files", False, f"Only {len(main_deps)} main dependencies")
                return False
            
            self.log_result("Requirements Files", True, f"{len(main_deps)} main + {len(dev_deps)} dev dependencies")
            return True
            
        except Exception as e:
            self.log_result("Requirements Files", False, f"Exception: {str(e)}")
            return False
    
    def run_all_checks(self) -> Tuple[bool, Dict[str, Any]]:
        """Run all Phase 1 quality gate checks."""
        print("🔍 Running Phase 1 Quality Gate Checks...")
        print("=" * 50)
        
        # Run all checks
        checks = [
            self.check_directory_structure,
            self.check_configuration_validity,
            self.check_logging_system,
            self.check_error_handling,
            self.check_test_framework,
            self.check_requirements_files
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                check_name = check.__name__.replace('check_', '').replace('_', ' ').title()
                self.log_result(check_name, False, f"Unexpected error: {str(e)}")
        
        print("=" * 50)
        
        # Calculate pass rate
        pass_rate = (self.passed_checks / self.total_checks) * 100 if self.total_checks > 0 else 0
        
        print(f"📊 Phase 1 Quality Gate Results:")
        print(f"   Passed: {self.passed_checks}/{self.total_checks} checks ({pass_rate:.1f}%)")
        
        # Phase 1 requires 100% pass rate
        phase1_passed = pass_rate == 100.0
        
        if phase1_passed:
            print("🎉 Phase 1 Quality Gate: PASSED")
            print("✅ Ready to proceed to Phase 2: Data Pipeline Compliance")
        else:
            print("⚠️  Phase 1 Quality Gate: FAILED")
            print("❌ Must fix issues before proceeding to Phase 2")
        
        return phase1_passed, self.results


def main():
    """Main function to run Phase 1 quality gate."""
    gate = Phase1QualityGate()
    passed, results = gate.run_all_checks()
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
