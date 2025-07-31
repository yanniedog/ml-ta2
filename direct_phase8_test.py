"""
Direct Phase 8 validation test for ML-TA deployment and integration.

This test validates core Phase 8 functionality:
- Docker containerization setup
- Kubernetes deployment manifests
- Terraform infrastructure as code
- CI/CD pipeline configuration
- Deployment scripts and runbooks
"""

import os
import sys
import yaml
import json
from pathlib import Path

def test_docker_configuration():
    """Test Docker containerization setup."""
    print("Testing Docker configuration...")
    
    try:
        # Check Dockerfile exists and has required components
        dockerfile_path = Path("Dockerfile")
        assert dockerfile_path.exists(), "Dockerfile not found"
        
        dockerfile_content = dockerfile_path.read_text()
        
        # Verify multi-stage build
        assert "FROM python:3.11-slim as base" in dockerfile_content, "Base stage not found"
        assert "FROM base as dependencies" in dockerfile_content, "Dependencies stage not found"
        assert "FROM dependencies as application" in dockerfile_content, "Application stage not found"
        
        # Verify security practices
        assert "RUN groupadd -r mlta && useradd -r -g mlta mlta" in dockerfile_content, "Non-root user not created"
        assert "USER mlta" in dockerfile_content, "Not switching to non-root user"
        
        # Verify health check
        assert "HEALTHCHECK" in dockerfile_content, "Health check not configured"
        
        # Verify exposed ports
        assert "EXPOSE 8000 9090" in dockerfile_content, "Required ports not exposed"
        
        print("✓ Docker configuration tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Docker configuration test failed: {e}")
        return False


def test_kubernetes_manifests():
    """Test Kubernetes deployment manifests."""
    print("Testing Kubernetes manifests...")
    
    try:
        k8s_dir = Path("k8s")
        assert k8s_dir.exists(), "k8s directory not found"
        
        # Check required manifest files
        required_files = ["deployment.yaml", "service.yaml", "storage.yaml"]
        for file_name in required_files:
            file_path = k8s_dir / file_name
            assert file_path.exists(), f"Required manifest {file_name} not found"
        
        # Test deployment.yaml
        deployment_path = k8s_dir / "deployment.yaml"
        deployment_content = deployment_path.read_text()
        
        # Parse YAML to validate structure
        deployment_docs = list(yaml.safe_load_all(deployment_content))
        assert len(deployment_docs) >= 3, "Expected at least 3 deployment documents"
        
        # Verify deployment components
        deployment_names = []
        for doc in deployment_docs:
            if doc and doc.get("kind") == "Deployment":
                deployment_names.append(doc["metadata"]["name"])
        
        expected_deployments = ["ml-ta-api", "ml-ta-worker", "ml-ta-scheduler"]
        for expected in expected_deployments:
            assert expected in deployment_names, f"Deployment {expected} not found"
        
        # Test service.yaml
        service_path = k8s_dir / "service.yaml"
        service_content = service_path.read_text()
        service_docs = list(yaml.safe_load_all(service_content))
        
        # Verify service and ingress
        has_service = any(doc.get("kind") == "Service" for doc in service_docs if doc)
        has_ingress = any(doc.get("kind") == "Ingress" for doc in service_docs if doc)
        has_network_policy = any(doc.get("kind") == "NetworkPolicy" for doc in service_docs if doc)
        
        assert has_service, "Service not found in service.yaml"
        assert has_ingress, "Ingress not found in service.yaml"
        assert has_network_policy, "NetworkPolicy not found in service.yaml"
        
        # Test storage.yaml
        storage_path = k8s_dir / "storage.yaml"
        storage_content = storage_path.read_text()
        storage_docs = list(yaml.safe_load_all(storage_content))
        
        # Verify storage components
        has_namespace = any(doc.get("kind") == "Namespace" for doc in storage_docs if doc)
        has_pvc = any(doc.get("kind") == "PersistentVolumeClaim" for doc in storage_docs if doc)
        has_configmap = any(doc.get("kind") == "ConfigMap" for doc in storage_docs if doc)
        has_secret = any(doc.get("kind") == "Secret" for doc in storage_docs if doc)
        
        assert has_namespace, "Namespace not found in storage.yaml"
        assert has_pvc, "PersistentVolumeClaim not found in storage.yaml"
        assert has_configmap, "ConfigMap not found in storage.yaml"
        assert has_secret, "Secret not found in storage.yaml"
        
        print("✓ Kubernetes manifests tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Kubernetes manifests test failed: {e}")
        return False


def test_terraform_infrastructure():
    """Test Terraform infrastructure as code."""
    print("Testing Terraform infrastructure...")
    
    try:
        terraform_dir = Path("terraform")
        assert terraform_dir.exists(), "terraform directory not found"
        
        # Check required Terraform files
        required_files = ["main.tf", "security.tf"]
        for file_name in required_files:
            file_path = terraform_dir / file_name
            assert file_path.exists(), f"Required Terraform file {file_name} not found"
        
        # Test main.tf
        main_tf_path = terraform_dir / "main.tf"
        main_tf_content = main_tf_path.read_text()
        
        # Verify required providers
        assert "hashicorp/aws" in main_tf_content, "AWS provider not configured"
        assert "hashicorp/kubernetes" in main_tf_content, "Kubernetes provider not configured"
        assert "hashicorp/helm" in main_tf_content, "Helm provider not configured"
        
        # Verify backend configuration
        assert "backend \"s3\"" in main_tf_content, "S3 backend not configured"
        
        # Verify infrastructure components
        assert "module \"vpc\"" in main_tf_content, "VPC module not found"
        assert "module \"eks\"" in main_tf_content, "EKS module not found"
        assert "module \"rds\"" in main_tf_content, "RDS module not found"
        assert "module \"redis\"" in main_tf_content, "Redis module not found"
        
        # Verify S3 buckets
        assert "aws_s3_bucket" in main_tf_content, "S3 buckets not configured"
        assert "ml_ta_data" in main_tf_content, "Data bucket not found"
        assert "ml_ta_models" in main_tf_content, "Models bucket not found"
        
        # Test security.tf
        security_tf_path = terraform_dir / "security.tf"
        security_tf_content = security_tf_path.read_text()
        
        # Verify security components
        assert "aws_kms_key" in security_tf_content, "KMS keys not configured"
        assert "aws_security_group" in security_tf_content, "Security groups not configured"
        assert "aws_iam_role" in security_tf_content, "IAM roles not configured"
        assert "aws_wafv2_web_acl" in security_tf_content, "WAF not configured"
        assert "aws_cloudwatch_metric_alarm" in security_tf_content, "CloudWatch alarms not configured"
        
        print("✓ Terraform infrastructure tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Terraform infrastructure test failed: {e}")
        return False


def test_cicd_pipeline():
    """Test CI/CD pipeline configuration."""
    print("Testing CI/CD pipeline...")
    
    try:
        github_dir = Path(".github/workflows")
        assert github_dir.exists(), ".github/workflows directory not found"
        
        # Check CI/CD workflow file
        cicd_path = github_dir / "ci-cd.yml"
        assert cicd_path.exists(), "ci-cd.yml workflow not found"
        
        cicd_content = cicd_path.read_text()
        
        # Parse YAML to validate structure
        cicd_config = yaml.safe_load(cicd_content)
        assert cicd_config is not None, "Invalid YAML in ci-cd.yml"
        
        # Verify workflow structure
        assert "name" in cicd_config, "Workflow name not specified"
        assert "on" in cicd_config or "'on'" in cicd_config, "Workflow triggers not specified"
        assert "jobs" in cicd_config, "Workflow jobs not specified"
        
        # Verify required jobs
        jobs = cicd_config["jobs"]
        expected_jobs = ["security-scan", "test", "build", "deploy-staging", "deploy-production"]
        
        for expected_job in expected_jobs:
            assert expected_job in jobs, f"Job {expected_job} not found"
        
        # Verify security scan job
        security_job = jobs["security-scan"]
        assert "bandit" in str(security_job), "Bandit security scan not configured"
        assert "safety" in str(security_job), "Safety vulnerability check not configured"
        assert "semgrep" in str(security_job), "Semgrep security scan not configured"
        
        # Verify test job
        test_job = jobs["test"]
        assert "services" in test_job, "Test services not configured"
        assert "postgres" in str(test_job), "PostgreSQL test service not configured"
        assert "redis" in str(test_job), "Redis test service not configured"
        
        # Verify build job
        build_job = jobs["build"]
        assert "docker/build-push-action" in str(build_job), "Docker build action not configured"
        assert "linux/amd64" in str(build_job) and "linux/arm64" in str(build_job), "Multi-platform build not configured"
        
        # Verify deployment jobs
        staging_job = jobs["deploy-staging"]
        prod_job = jobs["deploy-production"]
        
        assert "aws-actions/configure-aws-credentials" in str(staging_job), "AWS credentials not configured for staging"
        assert "aws-actions/configure-aws-credentials" in str(prod_job), "AWS credentials not configured for production"
        
        print("✓ CI/CD pipeline tests passed")
        return True
        
    except Exception as e:
        print(f"✗ CI/CD pipeline test failed: {e}")
        return False


def test_deployment_scripts():
    """Test deployment scripts and automation."""
    print("Testing deployment scripts...")
    
    try:
        scripts_dir = Path("scripts")
        assert scripts_dir.exists(), "scripts directory not found"
        
        # Check deployment script
        deploy_script = scripts_dir / "deploy.sh"
        assert deploy_script.exists(), "deploy.sh script not found"
        
        deploy_content = deploy_script.read_text()
        
        # Verify script components
        assert "#!/bin/bash" in deploy_content, "Bash shebang not found"
        assert "set -euo pipefail" in deploy_content, "Strict error handling not enabled"
        
        # Verify required functions
        required_functions = [
            "check_prerequisites",
            "create_namespace",
            "apply_manifests",
            "wait_for_deployments",
            "run_health_checks",
            "run_smoke_tests",
            "rollback_deployment"
        ]
        
        for func in required_functions:
            assert func in deploy_content, f"Function {func} not found in deploy script"
        
        # Verify error handling
        assert "trap cleanup EXIT" in deploy_content, "Error trap not configured"
        assert "cleanup()" in deploy_content, "Cleanup function not found"
        
        # Verify logging functions
        assert "log_info()" in deploy_content, "Info logging function not found"
        assert "log_error()" in deploy_content, "Error logging function not found"
        assert "log_success()" in deploy_content, "Success logging function not found"
        
        print("✓ Deployment scripts tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Deployment scripts test failed: {e}")
        return False


def test_documentation():
    """Test deployment documentation and runbooks."""
    print("Testing documentation...")
    
    try:
        docs_dir = Path("docs")
        assert docs_dir.exists(), "docs directory not found"
        
        # Check deployment runbook
        runbook_path = docs_dir / "deployment-runbook.md"
        assert runbook_path.exists(), "deployment-runbook.md not found"
        
        runbook_content = runbook_path.read_text()
        
        # Verify runbook sections
        required_sections = [
            "## Overview",
            "## Prerequisites",
            "## Infrastructure Setup",
            "## Application Deployment",
            "## Configuration Management",
            "## Monitoring and Observability",
            "## Scaling Operations",
            "## Backup and Recovery",
            "## Troubleshooting",
            "## Security Procedures",
            "## Performance Optimization",
            "## Maintenance Windows"
        ]
        
        for section in required_sections:
            assert section in runbook_content, f"Section {section} not found in runbook"
        
        # Verify practical examples
        assert "kubectl" in runbook_content, "kubectl commands not documented"
        assert "terraform" in runbook_content, "Terraform commands not documented"
        assert "helm" in runbook_content, "Helm commands not documented"
        
        # Verify troubleshooting procedures
        assert "Emergency Rollback" in runbook_content, "Emergency rollback procedure not documented"
        assert "Emergency Scale Down" in runbook_content, "Emergency scale down procedure not documented"
        
        print("✓ Documentation tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Documentation test failed: {e}")
        return False


def run_direct_phase8_validation():
    """Run direct Phase 8 validation tests."""
    print("=" * 60)
    print("PHASE 8 DIRECT VALIDATION TEST")
    print("=" * 60)
    
    tests = [
        ("Docker Configuration", test_docker_configuration),
        ("Kubernetes Manifests", test_kubernetes_manifests),
        ("Terraform Infrastructure", test_terraform_infrastructure),
        ("CI/CD Pipeline", test_cicd_pipeline),
        ("Deployment Scripts", test_deployment_scripts),
        ("Documentation", test_documentation)
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
    print(f"PHASE 8 VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ ALL PHASE 8 DEPLOYMENT AND INTEGRATION VALIDATED")
        print("✓ Phase 8 quality gate: PASSED")
        return True
    else:
        print("✗ PHASE 8 VALIDATION FAILED")
        print("✗ Phase 8 quality gate: FAILED")
        return False


if __name__ == "__main__":
    success = run_direct_phase8_validation()
    sys.exit(0 if success else 1)
