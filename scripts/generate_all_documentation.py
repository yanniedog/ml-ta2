#!/usr/bin/env python3
"""
Complete Documentation Generation Script for ML-TA System

This script generates all required documentation for Phase 10.3 compliance:
- Technical documentation
- User guides and tutorials  
- Troubleshooting guides
- Deployment runbooks
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def create_user_guide():
    """Create comprehensive user guide."""
    user_guide = """# ML-TA User Guide

## Getting Started

### Quick Start
1. Open browser to http://localhost:8000
2. Complete setup wizard
3. Select cryptocurrency pair (e.g., BTCUSDT)
4. Click "Generate Prediction"
5. View results and confidence levels

### Main Features
- **Predictions**: Get AI-powered trading predictions
- **Data Explorer**: Analyze market trends and patterns
- **Monitoring**: Track system and model performance
- **Help**: Access tutorials and support

## User Interface Guide

### Dashboard
- Real-time system status
- Recent predictions summary
- Performance metrics
- Quick action buttons

### Prediction Page
- Symbol selection dropdown
- Timeframe options (1h, 4h, 1d)
- Generate prediction button
- Results display with confidence

### Data Explorer
- Historical price charts
- Technical indicators
- Feature importance plots
- Export functionality

## Troubleshooting
- **Slow predictions**: Check system resources
- **No data**: Verify internet connection
- **Errors**: Check Help section or logs
"""
    
    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    with open(docs_dir / "USER_GUIDE.md", 'w', encoding='utf-8') as f:
        f.write(user_guide)


def create_deployment_runbook():
    """Create deployment runbook."""
    runbook = """# ML-TA Deployment Runbook

## Prerequisites
- Python 3.9+
- 4GB RAM minimum
- 10GB disk space
- Internet connection

## Local Deployment

### Quick Setup
```bash
# Clone repository
git clone https://github.com/your-org/ml-ta2.git
cd ml-ta2

# Install dependencies
pip install -r requirements.txt

# Run setup script
python scripts/local_setup.py

# Start system
python src/web_frontend.py
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f ml-ta
```

### Kubernetes Deployment
```bash
# Apply manifests
kubectl apply -f deployment/kubernetes/

# Check deployment
kubectl get pods -n ml-ta

# Access service
kubectl port-forward svc/ml-ta-api 8000:8000
```

## Production Checklist
- [ ] Configure environment variables
- [ ] Set up monitoring and alerting
- [ ] Enable security features
- [ ] Configure backup procedures
- [ ] Verify compliance settings
- [ ] Run health checks
- [ ] Load test the system

## Monitoring and Maintenance
- Check system health daily
- Review prediction accuracy weekly
- Update models monthly
- Security audit quarterly
"""
    
    with open(project_root / "docs" / "DEPLOYMENT_RUNBOOK.md", 'w', encoding='utf-8') as f:
        f.write(runbook)


def create_troubleshooting_guide():
    """Create troubleshooting guide."""
    troubleshooting = """# ML-TA Troubleshooting Guide

## Common Issues

### System Won't Start
**Symptoms**: Application fails to launch
**Causes**: Missing dependencies, port conflicts, configuration errors
**Solutions**:
1. Check Python version: `python --version`
2. Install dependencies: `pip install -r requirements.txt`
3. Verify port availability: `netstat -an | grep 8000`
4. Check configuration: `python -m src.config validate`

### Predictions Not Working
**Symptoms**: Prediction requests fail or return errors
**Causes**: Data connection issues, model loading problems
**Solutions**:
1. Check internet connection
2. Verify Binance API access
3. Check model files: `ls models/`
4. Review logs: `tail -f logs/application.log`

### Slow Performance
**Symptoms**: High response times, timeouts
**Causes**: Resource constraints, large datasets
**Solutions**:
1. Check system resources: `top` or Task Manager
2. Reduce data processing size
3. Enable caching
4. Increase timeout values

### Data Quality Issues
**Symptoms**: Unrealistic predictions, poor accuracy
**Causes**: Data feed problems, feature engineering issues
**Solutions**:
1. Validate data sources
2. Check feature engineering pipeline
3. Review model performance metrics
4. Retrain models if necessary

## Diagnostic Commands
```bash
# Check system health
python -c "from src.monitoring import MonitoringSystem; m=MonitoringSystem(); print(m.get_system_status())"

# Validate configuration
python -m src.config validate

# Test prediction engine
python -c "from src.prediction_engine import PredictionEngine; p=PredictionEngine(); print('Engine OK')"

# Check data connection
python -c "from src.data_fetcher import BinanceDataFetcher; d=BinanceDataFetcher(); print('Data OK')"
```

## Log Analysis
- **Application logs**: `logs/application.log`
- **Error logs**: `logs/error.log`
- **Audit logs**: `logs/audit/`
- **Performance logs**: `logs/performance.log`

## Recovery Procedures
1. **Service restart**: `systemctl restart ml-ta`
2. **Database recovery**: Restore from backup
3. **Model recovery**: Retrain from historical data
4. **Configuration reset**: Restore default configuration

## Contact Support
- **Documentation**: Check docs/ directory
- **Issues**: GitHub Issues
- **Community**: Discussion forums
"""
    
    with open(project_root / "docs" / "TROUBLESHOOTING.md", 'w', encoding='utf-8') as f:
        f.write(troubleshooting)


def create_api_documentation():
    """Create API documentation."""
    api_doc = """# ML-TA API Documentation

## Base URL
http://localhost:8000/api/v1

## Authentication
API Key required for production. Development mode: no auth.

## Endpoints

### Health Check
```
GET /health
Response: {"status": "healthy", "components": {...}}
```

### Generate Prediction
```
POST /predict
Body: {"symbol": "BTCUSDT", "timeframe": "1h"}
Response: {"prediction": {...}, "confidence": 0.85}
```

### Get Historical Data
```
GET /data/{symbol}?interval=1h&limit=100
Response: {"data": [...], "count": 100}
```

### System Metrics
```
GET /metrics
Response: Prometheus metrics format
```

## Error Codes
- 400: Bad Request
- 404: Not Found  
- 500: Internal Error
- 503: Service Unavailable

## Rate Limits
- Default: 100 requests/minute
- Predictions: 10 requests/minute
"""
    
    with open(project_root / "docs" / "API_REFERENCE.md", 'w', encoding='utf-8') as f:
        f.write(api_doc)


def generate_comprehensive_documentation():
    """Generate all documentation components."""
    print("Generating ML-TA Documentation...")
    print("=" * 50)
    
    # Create documentation directory
    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)
    
    # Generate all documentation
    documentation_components = [
        ("User Guide", create_user_guide),
        ("Deployment Runbook", create_deployment_runbook),  
        ("Troubleshooting Guide", create_troubleshooting_guide),
        ("API Documentation", create_api_documentation)
    ]
    
    success_count = 0
    for doc_name, create_func in documentation_components:
        try:
            print(f"Creating {doc_name}...")
            create_func()
            print(f"✓ {doc_name} created successfully")
            success_count += 1
        except Exception as e:
            print(f"✗ Failed to create {doc_name}: {str(e)}")
    
    # Create documentation index
    create_documentation_index()
    
    # Generate documentation report
    total_docs = len(documentation_components) + 1  # +1 for index
    success_rate = ((success_count + 1) / total_docs) * 100
    
    print(f"\nDocumentation Generation Summary:")
    print(f"Success Rate: {success_rate:.1f}% ({success_count + 1}/{total_docs})")
    print(f"Documentation Location: {docs_dir}")
    
    return success_rate >= 90


def create_documentation_index():
    """Create main documentation index."""
    index = f"""# ML-TA System Documentation

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Documentation Index

### For Users
- [**User Guide**](USER_GUIDE.md) - Complete guide for end users
- [**API Reference**](API_REFERENCE.md) - API endpoints and usage
- [**Troubleshooting**](TROUBLESHOOTING.md) - Common issues and solutions

### For Administrators  
- [**Deployment Runbook**](DEPLOYMENT_RUNBOOK.md) - Installation and deployment
- [**Configuration Guide**](CONFIGURATION.md) - System configuration
- [**Security Documentation**](SECURITY.md) - Security implementation

### For Developers
- [**Architecture Overview**](ARCHITECTURE.md) - System architecture
- [**Database Schema**](DATABASE_SCHEMA.md) - Database structure
- [**Development Setup**](../README.md) - Development environment

## Quick Links

### Getting Started
1. [Installation Guide](DEPLOYMENT_RUNBOOK.md#local-deployment)
2. [First Time Setup](USER_GUIDE.md#getting-started)
3. [Basic Usage](USER_GUIDE.md#main-features)

### Administration
1. [System Monitoring](ARCHITECTURE.md#monitoring)
2. [Backup Procedures](DEPLOYMENT_RUNBOOK.md#monitoring-and-maintenance)
3. [Security Configuration](SECURITY.md)

### Support
1. [Troubleshooting Guide](TROUBLESHOOTING.md)
2. [FAQ](USER_GUIDE.md#troubleshooting)
3. [Contact Information](TROUBLESHOOTING.md#contact-support)

## System Requirements
- **Hardware**: 4GB RAM, 2+ CPU cores, 10GB storage
- **Software**: Python 3.9+, modern web browser
- **Network**: Internet connection for data feeds

## Compliance Documentation
- SOC2 Type II controls implemented
- GDPR data protection measures
- ISO 27001 security framework
- Comprehensive audit logging

## Documentation Maintenance
This documentation is automatically generated and should be kept up to date with system changes. Last updated: {datetime.now().strftime('%Y-%m-%d')}.
"""
    
    with open(project_root / "docs" / "README.md", 'w', encoding='utf-8') as f:
        f.write(index)


def main():
    """Main documentation generation function."""
    try:
        success = generate_comprehensive_documentation()
        
        if success:
            print("\n✅ All documentation generated successfully!")
            print("📖 Access documentation at: docs/README.md")
            return 0
        else:
            print("\n❌ Documentation generation completed with errors")
            return 1
            
    except Exception as e:
        print(f"\n💥 Documentation generation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
