#!/bin/bash
# ML-TA Deployment Script
# Production-grade deployment automation for Kubernetes

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${ENVIRONMENT:-staging}"
NAMESPACE="ml-ta${ENVIRONMENT:+-$ENVIRONMENT}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
KUBECTL_TIMEOUT="${KUBECTL_TIMEOUT:-600s}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm is not installed or not in PATH"
        exit 1
    fi
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed or not in PATH"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    log_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        kubectl label namespace "$NAMESPACE" environment="$ENVIRONMENT"
        log_success "Created namespace: $NAMESPACE"
    fi
}

# Apply Kubernetes manifests
apply_manifests() {
    log_info "Applying Kubernetes manifests..."
    
    # Apply storage and RBAC first
    kubectl apply -f "$PROJECT_ROOT/k8s/storage.yaml" -n "$NAMESPACE"
    
    # Wait for PVCs to be bound
    log_info "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=Bound pvc --all -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    
    # Apply services
    kubectl apply -f "$PROJECT_ROOT/k8s/service.yaml" -n "$NAMESPACE"
    
    # Apply deployments with image tag substitution
    envsubst < "$PROJECT_ROOT/k8s/deployment.yaml" | kubectl apply -f - -n "$NAMESPACE"
    
    log_success "Applied all Kubernetes manifests"
}

# Wait for deployments to be ready
wait_for_deployments() {
    log_info "Waiting for deployments to be ready..."
    
    local deployments=("ml-ta-api" "ml-ta-worker" "ml-ta-scheduler")
    
    for deployment in "${deployments[@]}"; do
        log_info "Waiting for deployment: $deployment"
        kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    done
    
    log_success "All deployments are ready"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=ml-ta-api -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    
    # Get API pod for health check
    local api_pod
    api_pod=$(kubectl get pods -l app=ml-ta-api -n "$NAMESPACE" -o jsonpath='{.items[0].metadata.name}')
    
    if [[ -z "$api_pod" ]]; then
        log_error "No API pods found"
        exit 1
    fi
    
    # Check health endpoint
    log_info "Checking health endpoint..."
    if kubectl exec "$api_pod" -n "$NAMESPACE" -- curl -f http://localhost:8000/health; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        exit 1
    fi
    
    # Check readiness endpoint
    log_info "Checking readiness endpoint..."
    if kubectl exec "$api_pod" -n "$NAMESPACE" -- curl -f http://localhost:8000/ready; then
        log_success "Readiness check passed"
    else
        log_error "Readiness check failed"
        exit 1
    fi
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Port forward to API service for testing
    kubectl port-forward service/ml-ta-api-service 8080:80 -n "$NAMESPACE" &
    local port_forward_pid=$!
    
    # Wait for port forward to be ready
    sleep 5
    
    # Run smoke tests
    if python "$PROJECT_ROOT/tests/smoke/test_api_health.py" --base-url http://localhost:8080; then
        log_success "Smoke tests passed"
    else
        log_error "Smoke tests failed"
        kill $port_forward_pid
        exit 1
    fi
    
    # Clean up port forward
    kill $port_forward_pid
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    local deployments=("ml-ta-api" "ml-ta-worker" "ml-ta-scheduler")
    
    for deployment in "${deployments[@]}"; do
        kubectl rollout undo deployment/"$deployment" -n "$NAMESPACE"
        kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout="$KUBECTL_TIMEOUT"
    done
    
    log_success "Rollback completed"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code: $exit_code"
        
        if [[ "${ROLLBACK_ON_FAILURE:-true}" == "true" ]]; then
            rollback_deployment
        fi
    fi
    
    # Kill any background processes
    jobs -p | xargs -r kill
}

# Set up trap for cleanup
trap cleanup EXIT

# Main deployment function
deploy() {
    log_info "Starting ML-TA deployment to $ENVIRONMENT environment"
    log_info "Image tag: $IMAGE_TAG"
    log_info "Namespace: $NAMESPACE"
    
    check_prerequisites
    create_namespace
    apply_manifests
    wait_for_deployments
    run_health_checks
    run_smoke_tests
    
    log_success "Deployment completed successfully!"
    
    # Display useful information
    echo ""
    log_info "Deployment Information:"
    echo "  Environment: $ENVIRONMENT"
    echo "  Namespace: $NAMESPACE"
    echo "  Image Tag: $IMAGE_TAG"
    echo ""
    log_info "Useful Commands:"
    echo "  View pods: kubectl get pods -n $NAMESPACE"
    echo "  View logs: kubectl logs -f deployment/ml-ta-api -n $NAMESPACE"
    echo "  Port forward: kubectl port-forward service/ml-ta-api-service 8080:80 -n $NAMESPACE"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            NAMESPACE="ml-ta-$ENVIRONMENT"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE="false"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -e, --environment ENV    Target environment (default: staging)"
            echo "  -t, --tag TAG           Docker image tag (default: latest)"
            echo "  --no-rollback           Don't rollback on failure"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  ENVIRONMENT             Target environment"
            echo "  IMAGE_TAG               Docker image tag"
            echo "  KUBECTL_TIMEOUT         Kubectl timeout (default: 600s)"
            echo "  ROLLBACK_ON_FAILURE     Rollback on failure (default: true)"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Export variables for envsubst
export IMAGE_TAG
export ENVIRONMENT
export NAMESPACE

# Run deployment
deploy
