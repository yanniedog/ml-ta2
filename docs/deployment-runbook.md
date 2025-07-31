# ML-TA Deployment Runbook

## Overview

This runbook provides comprehensive instructions for deploying the ML-TA (Machine Learning Trading Analysis) system to production environments using Kubernetes, Terraform, and automated CI/CD pipelines.

## Prerequisites

### Required Tools
- **kubectl** (v1.27+): Kubernetes command-line tool
- **helm** (v3.12+): Kubernetes package manager
- **terraform** (v1.5+): Infrastructure as Code tool
- **aws-cli** (v2.13+): AWS command-line interface
- **docker** (v24.0+): Container runtime
- **git** (v2.40+): Version control system

### Access Requirements
- AWS account with appropriate IAM permissions
- Kubernetes cluster access (EKS recommended)
- Container registry access (GitHub Container Registry)
- Domain name and SSL certificates
- Monitoring and alerting endpoints

## Infrastructure Setup

### 1. Terraform Infrastructure Deployment

```bash
# Navigate to terraform directory
cd terraform/

# Initialize Terraform
terraform init

# Plan infrastructure changes
terraform plan -var-file="environments/production.tfvars"

# Apply infrastructure
terraform apply -var-file="environments/production.tfvars"

# Save important outputs
terraform output > ../infrastructure-outputs.txt
```

### 2. EKS Cluster Configuration

```bash
# Update kubeconfig
aws eks update-kubeconfig --region us-west-2 --name ml-ta-production

# Verify cluster access
kubectl cluster-info

# Install required add-ons
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo add cert-manager https://charts.jetstack.io
helm repo update

# Install NGINX Ingress Controller
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.type=LoadBalancer

# Install Cert-Manager
helm install cert-manager cert-manager/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set installCRDs=true
```

## Application Deployment

### 1. Environment Preparation

```bash
# Create production namespace
kubectl create namespace ml-ta
kubectl label namespace ml-ta environment=production

# Apply RBAC and storage configurations
kubectl apply -f k8s/storage.yaml

# Create secrets (replace with actual values)
kubectl create secret generic ml-ta-secrets \
  --from-literal=database-url="postgresql://user:pass@host:5432/mlta" \
  --from-literal=redis-url="redis://redis:pass@host:6379/0" \
  --from-literal=jwt-secret="your-jwt-secret" \
  --from-literal=encryption-key="your-encryption-key" \
  --namespace ml-ta
```

### 2. Application Deployment

```bash
# Deploy using deployment script
./scripts/deploy.sh --environment production --tag v1.0.0

# Or deploy manually
export IMAGE_TAG=v1.0.0
envsubst < k8s/deployment.yaml | kubectl apply -f - -n ml-ta
kubectl apply -f k8s/service.yaml -n ml-ta

# Wait for deployments
kubectl rollout status deployment/ml-ta-api -n ml-ta --timeout=600s
kubectl rollout status deployment/ml-ta-worker -n ml-ta --timeout=600s
kubectl rollout status deployment/ml-ta-scheduler -n ml-ta --timeout=600s
```

### 3. Post-Deployment Verification

```bash
# Check pod status
kubectl get pods -n ml-ta

# Check service endpoints
kubectl get services -n ml-ta

# Check ingress
kubectl get ingress -n ml-ta

# Run health checks
kubectl exec deployment/ml-ta-api -n ml-ta -- curl http://localhost:8000/health

# Check logs
kubectl logs -f deployment/ml-ta-api -n ml-ta
```

## Configuration Management

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `ENVIRONMENT` | Deployment environment | Yes | - |
| `DATABASE_URL` | PostgreSQL connection string | Yes | - |
| `REDIS_URL` | Redis connection string | Yes | - |
| `JWT_SECRET_KEY` | JWT signing key | Yes | - |
| `LOG_LEVEL` | Logging level | No | INFO |
| `API_WORKERS` | Number of API workers | No | 4 |
| `PREDICTION_TIMEOUT` | Prediction timeout (seconds) | No | 5.0 |

### ConfigMap Updates

```bash
# Update configuration
kubectl create configmap ml-ta-config \
  --from-file=config/production.yaml \
  --namespace ml-ta \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart deployments to pick up config changes
kubectl rollout restart deployment/ml-ta-api -n ml-ta
kubectl rollout restart deployment/ml-ta-worker -n ml-ta
```

## Monitoring and Observability

### 1. Metrics Collection

```bash
# Check Prometheus metrics
kubectl port-forward service/ml-ta-api-service 9090:9090 -n ml-ta
curl http://localhost:9090/metrics

# View Grafana dashboards
kubectl port-forward service/grafana 3000:3000 -n monitoring
# Access: http://localhost:3000
```

### 2. Log Aggregation

```bash
# View application logs
kubectl logs -f deployment/ml-ta-api -n ml-ta --tail=100

# View worker logs
kubectl logs -f deployment/ml-ta-worker -n ml-ta --tail=100

# Search logs with specific criteria
kubectl logs deployment/ml-ta-api -n ml-ta | grep "ERROR"
```

### 3. Health Monitoring

```bash
# Check application health
curl https://api.ml-ta.com/health

# Check readiness
curl https://api.ml-ta.com/ready

# Check metrics endpoint
curl https://api.ml-ta.com/metrics
```

## Scaling Operations

### 1. Horizontal Pod Autoscaling

```bash
# Create HPA for API pods
kubectl autoscale deployment ml-ta-api \
  --cpu-percent=70 \
  --min=3 \
  --max=20 \
  --namespace ml-ta

# Create HPA for worker pods
kubectl autoscale deployment ml-ta-worker \
  --cpu-percent=80 \
  --min=2 \
  --max=10 \
  --namespace ml-ta

# Check HPA status
kubectl get hpa -n ml-ta
```

### 2. Manual Scaling

```bash
# Scale API pods
kubectl scale deployment ml-ta-api --replicas=5 -n ml-ta

# Scale worker pods
kubectl scale deployment ml-ta-worker --replicas=3 -n ml-ta

# Verify scaling
kubectl get pods -n ml-ta
```

## Backup and Recovery

### 1. Database Backup

```bash
# Create database backup
kubectl exec deployment/ml-ta-api -n ml-ta -- \
  pg_dump $DATABASE_URL > backup-$(date +%Y%m%d-%H%M%S).sql

# Restore from backup
kubectl exec -i deployment/ml-ta-api -n ml-ta -- \
  psql $DATABASE_URL < backup-20231201-120000.sql
```

### 2. Model Backup

```bash
# Backup models to S3
kubectl exec deployment/ml-ta-api -n ml-ta -- \
  aws s3 sync /app/models s3://ml-ta-models-backup/$(date +%Y%m%d)/

# Restore models from S3
kubectl exec deployment/ml-ta-api -n ml-ta -- \
  aws s3 sync s3://ml-ta-models-backup/20231201/ /app/models/
```

## Troubleshooting

### Common Issues

#### 1. Pod Startup Issues

```bash
# Check pod events
kubectl describe pod <pod-name> -n ml-ta

# Check pod logs
kubectl logs <pod-name> -n ml-ta --previous

# Check resource constraints
kubectl top pods -n ml-ta
kubectl describe nodes
```

#### 2. Database Connection Issues

```bash
# Test database connectivity
kubectl exec deployment/ml-ta-api -n ml-ta -- \
  psql $DATABASE_URL -c "SELECT 1;"

# Check database pod status
kubectl get pods -l app=postgresql -n ml-ta

# Check database logs
kubectl logs deployment/postgresql -n ml-ta
```

#### 3. Performance Issues

```bash
# Check resource usage
kubectl top pods -n ml-ta
kubectl top nodes

# Check HPA status
kubectl get hpa -n ml-ta

# Check application metrics
curl https://api.ml-ta.com/metrics | grep -E "(cpu|memory|requests)"
```

### Emergency Procedures

#### 1. Emergency Rollback

```bash
# Quick rollback to previous version
kubectl rollout undo deployment/ml-ta-api -n ml-ta
kubectl rollout undo deployment/ml-ta-worker -n ml-ta
kubectl rollout undo deployment/ml-ta-scheduler -n ml-ta

# Check rollback status
kubectl rollout status deployment/ml-ta-api -n ml-ta
```

#### 2. Emergency Scale Down

```bash
# Scale down to minimum resources
kubectl scale deployment ml-ta-api --replicas=1 -n ml-ta
kubectl scale deployment ml-ta-worker --replicas=1 -n ml-ta

# Disable HPA temporarily
kubectl delete hpa ml-ta-api-hpa -n ml-ta
kubectl delete hpa ml-ta-worker-hpa -n ml-ta
```

#### 3. Circuit Breaker Activation

```bash
# Enable maintenance mode
kubectl patch configmap ml-ta-config -n ml-ta --patch '
{
  "data": {
    "maintenance_mode": "true"
  }
}'

# Restart API pods to pick up changes
kubectl rollout restart deployment/ml-ta-api -n ml-ta
```

## Security Procedures

### 1. Secret Rotation

```bash
# Rotate JWT secret
kubectl patch secret ml-ta-secrets -n ml-ta --patch '
{
  "data": {
    "jwt-secret": "'$(echo -n "new-jwt-secret" | base64)'"
  }
}'

# Restart deployments
kubectl rollout restart deployment/ml-ta-api -n ml-ta
```

### 2. Certificate Management

```bash
# Check certificate status
kubectl get certificates -n ml-ta

# Force certificate renewal
kubectl delete certificate ml-ta-tls -n ml-ta
kubectl apply -f k8s/service.yaml -n ml-ta
```

### 3. Security Scanning

```bash
# Scan container images
trivy image ghcr.io/ml-ta/ml-ta:latest

# Check for vulnerabilities
kubectl exec deployment/ml-ta-api -n ml-ta -- \
  python -m src.security scan_system
```

## Performance Optimization

### 1. Resource Tuning

```bash
# Update resource requests/limits
kubectl patch deployment ml-ta-api -n ml-ta --patch '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "ml-ta-api",
          "resources": {
            "requests": {"memory": "1Gi", "cpu": "500m"},
            "limits": {"memory": "4Gi", "cpu": "2000m"}
          }
        }]
      }
    }
  }
}'
```

### 2. Cache Optimization

```bash
# Clear Redis cache
kubectl exec deployment/ml-ta-api -n ml-ta -- \
  redis-cli -u $REDIS_URL FLUSHALL

# Warm up model cache
kubectl exec deployment/ml-ta-api -n ml-ta -- \
  python -m src.models warm_cache
```

## Maintenance Windows

### 1. Planned Maintenance

```bash
# 1. Enable maintenance mode
kubectl patch configmap ml-ta-config -n ml-ta --patch '
{
  "data": {
    "maintenance_mode": "true"
  }
}'

# 2. Wait for current requests to complete
sleep 30

# 3. Perform maintenance tasks
# (database migrations, model updates, etc.)

# 4. Disable maintenance mode
kubectl patch configmap ml-ta-config -n ml-ta --patch '
{
  "data": {
    "maintenance_mode": "false"
  }
}'

# 5. Restart services
kubectl rollout restart deployment/ml-ta-api -n ml-ta
```

### 2. Database Migrations

```bash
# Run database migrations
kubectl exec deployment/ml-ta-api -n ml-ta -- \
  python -m src.database migrate

# Verify migration status
kubectl exec deployment/ml-ta-api -n ml-ta -- \
  python -m src.database status
```

## Contact Information

### On-Call Procedures
- **Primary**: ML-TA DevOps Team
- **Secondary**: Platform Engineering Team
- **Escalation**: Engineering Management

### Communication Channels
- **Slack**: #ml-ta-alerts
- **Email**: ml-ta-oncall@company.com
- **Phone**: +1-XXX-XXX-XXXX

### Documentation Links
- **Architecture**: [Architecture Documentation](./architecture.md)
- **API Documentation**: [API Reference](./api.md)
- **Monitoring**: [Monitoring Guide](./monitoring.md)
- **Security**: [Security Guide](./security.md)
