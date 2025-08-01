# ML-TA Troubleshooting Guide

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
