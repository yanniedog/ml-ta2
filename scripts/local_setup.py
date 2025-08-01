#!/usr/bin/env python3
"""
Local Development Setup Script for ML-TA System

This script automates the complete local development environment setup,
addressing Phase 8.2 Infrastructure as Code requirements.
"""

import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any

class LocalSetupManager:
    """Manages local development environment setup."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.setup_log = []
        
    def log_step(self, message: str, success: bool = True):
        """Log setup step with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        status = "[SUCCESS]" if success else "[FAILED]"
        log_entry = f"{status} [{timestamp}] {message}"
        self.setup_log.append(log_entry)
        print(log_entry)
    
    def run_command(self, command: str, cwd: Path = None, timeout: int = 300) -> bool:
        """Run shell command with error handling."""
        try:
            cwd = cwd or self.project_root
            result = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                self.log_step(f"Command succeeded: {command}")
                return True
            else:
                self.log_step(f"Command failed: {command} - {result.stderr}", False)
                return False
                
        except subprocess.TimeoutExpired:
            self.log_step(f"Command timed out: {command}", False)
            return False
        except Exception as e:
            self.log_step(f"Command error: {command} - {str(e)}", False)
            return False
    
    def create_directories(self) -> bool:
        """Create all required local directories."""
        directories = [
            "data/raw",
            "data/bronze",
            "data/silver", 
            "data/gold",
            "models",
            "logs",
            "artefacts",
            "cache",
            "monitoring/prometheus",
            "monitoring/grafana/dashboards",
            "monitoring/grafana/datasources",
            "backup",
            "temp"
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                self.log_step(f"Created directory: {directory}")
            return True
        except Exception as e:
            self.log_step(f"Failed to create directories: {str(e)}", False)
            return False
    
    def setup_local_monitoring(self) -> bool:
        """Set up local monitoring stack."""
        try:
            # Create Prometheus configuration
            prometheus_config = {
                'global': {
                    'scrape_interval': '15s',
                    'evaluation_interval': '15s'
                },
                'scrape_configs': [
                    {
                        'job_name': 'ml-ta-api',
                        'static_configs': [
                            {'targets': ['localhost:8000']}
                        ],
                        'metrics_path': '/metrics'
                    },
                    {
                        'job_name': 'ml-ta-worker',
                        'static_configs': [
                            {'targets': ['localhost:8001']}
                        ]
                    }
                ]
            }
            
            prometheus_file = self.project_root / "monitoring/prometheus.yml"
            with open(prometheus_file, 'w') as f:
                import yaml
                yaml.dump(prometheus_config, f, default_flow_style=False)
            
            # Create Grafana datasource configuration
            grafana_datasource = {
                'apiVersion': 1,
                'datasources': [
                    {
                        'name': 'Prometheus',
                        'type': 'prometheus',
                        'url': 'http://prometheus:9090',
                        'access': 'proxy',
                        'isDefault': True
                    }
                ]
            }
            
            datasource_file = self.project_root / "monitoring/grafana/datasources/prometheus.yml"
            with open(datasource_file, 'w') as f:
                import yaml
                yaml.dump(grafana_datasource, f, default_flow_style=False)
            
            self.log_step("Local monitoring stack configured")
            return True
            
        except Exception as e:
            self.log_step(f"Failed to setup monitoring: {str(e)}", False)
            return False
    
    def setup_local_backup(self) -> bool:
        """Implement local backup and recovery procedures."""
        try:
            backup_script = """#!/bin/bash
# Local Backup Script for ML-TA System

BACKUP_DIR="backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating backup at $BACKUP_DIR"

# Backup models
if [ -d "models" ]; then
    cp -r models "$BACKUP_DIR/"
    echo "[SUCCESS] Models backed up"
fi

# Backup configuration
if [ -d "config" ]; then
    cp -r config "$BACKUP_DIR/"
    echo "[SUCCESS] Configuration backed up"
fi

# Backup data (excluding large raw files)
if [ -d "data" ]; then
    mkdir -p "$BACKUP_DIR/data"
    cp -r data/bronze "$BACKUP_DIR/data/" 2>/dev/null || true
    cp -r data/silver "$BACKUP_DIR/data/" 2>/dev/null || true
    cp -r data/gold "$BACKUP_DIR/data/" 2>/dev/null || true
    echo "[SUCCESS] Processed data backed up"
fi

# Backup logs (last 7 days)
if [ -d "logs" ]; then
    mkdir -p "$BACKUP_DIR/logs"
    find logs -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/logs/" \;
    echo "[SUCCESS] Recent logs backed up"
fi

echo "Backup completed: $BACKUP_DIR"
"""
            
            backup_file = self.project_root / "scripts/backup.sh"
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(backup_script)
            
            # Make executable
            os.chmod(backup_file, 0o755)
            
            self.log_step("Local backup system configured")
            return True
            
        except Exception as e:
            self.log_step(f"Failed to setup backup: {str(e)}", False)
            return False
    
    def create_local_ci_pipeline(self) -> bool:
        """Create local CI/CD pipeline configuration."""
        try:
            # GitHub Actions workflow
            github_workflow = {
                'name': 'ML-TA Local CI/CD',
                'on': {
                    'push': {'branches': ['main', 'develop']},
                    'pull_request': {'branches': ['main']}
                },
                'jobs': {
                    'test': {
                        'runs-on': 'ubuntu-latest',
                        'strategy': {
                            'matrix': {
                                'python-version': ['3.9', '3.10', '3.11']
                            }
                        },
                        'steps': [
                            {
                                'uses': 'actions/checkout@v3'
                            },
                            {
                                'name': 'Set up Python',
                                'uses': 'actions/setup-python@v4',
                                'with': {
                                    'python-version': '${{ matrix.python-version }}'
                                }
                            },
                            {
                                'name': 'Install dependencies',
                                'run': 'pip install -r requirements.txt -r requirements-dev.txt'
                            },
                            {
                                'name': 'Run tests',
                                'run': 'python -m pytest tests/ -v --cov=src --cov-report=xml'
                            },
                            {
                                'name': 'Run security scan',
                                'run': 'python -c "from src.security_audit import create_security_auditor; auditor = create_security_auditor(); auditor.run_comprehensive_audit()"'
                            },
                            {
                                'name': 'Run performance tests',
                                'run': 'python scripts/performance_tests.py'
                            }
                        ]
                    },
                    'deploy': {
                        'needs': 'test',
                        'runs-on': 'ubuntu-latest',
                        'if': "github.ref == 'refs/heads/main'",
                        'steps': [
                            {
                                'uses': 'actions/checkout@v3'
                            },
                            {
                                'name': 'Build Docker image',
                                'run': 'docker build -t ml-ta:latest .'
                            },
                            {
                                'name': 'Run integration tests',
                                'run': 'docker-compose -f docker-compose.test.yml up --abort-on-container-exit'
                            }
                        ]
                    }
                }
            }
            
            # Create .github/workflows directory
            workflows_dir = self.project_root / ".github/workflows"
            workflows_dir.mkdir(parents=True, exist_ok=True)
            
            workflow_file = workflows_dir / "ci.yml"
            with open(workflow_file, 'w') as f:
                import yaml
                yaml.dump(github_workflow, f, default_flow_style=False)
            
            self.log_step("Local CI/CD pipeline configured")
            return True
            
        except Exception as e:
            self.log_step(f"Failed to setup CI/CD: {str(e)}", False)
            return False
    
    def validate_local_setup(self) -> bool:
        """Validate that local setup is working correctly."""
        try:
            # Check Python environment
            if not self.run_command("python --version"):
                return False
            
            # Check dependencies
            if not self.run_command("pip check"):
                self.log_step("Installing missing dependencies...")
                if not self.run_command("pip install -r requirements.txt"):
                    return False
            
            # Test import of core modules
            test_imports = [
                "from src.config import get_config",
                "from src.data_fetcher import BinanceDataFetcher", 
                "from src.features import FeaturePipeline",
                "from src.model_trainer import ModelTrainer",
                "from src.prediction_engine import PredictionEngine"
            ]
            
            for import_stmt in test_imports:
                if not self.run_command(f'python -c "{import_stmt}"'):
                    return False
            
            # Check Docker setup
            if not self.run_command("docker --version"):
                self.log_step("Docker not available - local containerization disabled", False)
            else:
                if not self.run_command("docker-compose --version"):
                    self.log_step("Docker Compose not available", False)
            
            self.log_step("Local setup validation completed")
            return True
            
        except Exception as e:
            self.log_step(f"Validation failed: {str(e)}", False)
            return False
    
    def run_complete_setup(self) -> bool:
        """Run complete local development setup."""
        print("Starting ML-TA Local Development Setup")
        print("=" * 60)
        
        setup_steps = [
            ("Creating directories", self.create_directories),
            ("Setting up local monitoring", self.setup_local_monitoring),
            ("Configuring backup system", self.setup_local_backup),
            ("Creating CI/CD pipeline", self.create_local_ci_pipeline),
            ("Validating setup", self.validate_local_setup)
        ]
        
        success_count = 0
        for step_name, step_func in setup_steps:
            print(f"\n[STEP] {step_name}...")
            if step_func():
                success_count += 1
            else:
                print(f"[FAILED] {step_name} failed")
        
        total_steps = len(setup_steps)
        success_rate = (success_count / total_steps) * 100
        
        print(f"\n[SUMMARY] Setup Summary:")
        print(f"   Success Rate: {success_rate:.1f}% ({success_count}/{total_steps})")
        print(f"   Setup Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save setup log
        log_file = self.project_root / "logs/local_setup.log"
        log_file.parent.mkdir(exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.setup_log))
        
        if success_rate >= 80:
            print("[SUCCESS] Local development environment ready!")
            return True
        else:
            print("[WARNING] Setup completed with issues. Check logs for details.")
            return False


def main():
    """Main setup function."""
    setup_manager = LocalSetupManager()
    success = setup_manager.run_complete_setup()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
