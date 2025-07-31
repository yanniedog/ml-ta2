# ML-TA: Machine Learning Technical Analysis System

A production-grade cryptocurrency trading analysis platform that combines technical analysis with machine learning for predictive modeling.

## 🚀 Features

- **Advanced Technical Analysis**: 145+ technical indicators with mathematical validation
- **Machine Learning Models**: LightGBM, XGBoost, CatBoost with hyperparameter optimization
- **Real-time Predictions**: <100ms latency with 99.9% uptime
- **Data Quality Assurance**: Comprehensive validation and leakage prevention
- **Production-Grade Infrastructure**: Monitoring, alerting, security, and compliance
- **Scalable Architecture**: Three-tier data pipeline (Bronze → Silver → Gold)

## 🏗️ Architecture

```
ML-TA System
├── Data Pipeline (Bronze → Silver → Gold)
├── Feature Engineering (200+ features, zero leakage)
├── Model Training (Statistical significance validation)
├── Real-time Prediction Engine
├── Monitoring & Alerting
└── Security & Compliance
```

## 📁 Project Structure

```
ml-ta/
├── src/                    # Core Python modules (16+)
│   ├── config.py          # Configuration management
│   ├── logging_config.py  # Structured logging
│   ├── exceptions.py      # Error handling
│   ├── utils.py           # Core utilities
│   ├── data_fetcher.py    # Data acquisition
│   ├── data_loader.py     # Data loading
│   ├── data_quality.py    # Quality assurance
│   ├── indicators.py      # Technical indicators
│   ├── features.py        # Feature engineering
│   ├── model_trainer.py   # Model training
│   ├── predictor.py       # Real-time prediction
│   ├── backtest_engine.py # Backtesting
│   ├── api.py             # REST API
│   ├── monitoring.py      # System monitoring
│   └── security.py        # Security framework
├── config/                # YAML configurations
│   ├── settings.yaml      # Base configuration
│   ├── development.yaml   # Dev overrides
│   ├── testing.yaml       # Test overrides
│   └── production.yaml    # Prod overrides
├── data/                  # Data storage
│   ├── raw/              # Raw data
│   ├── bronze/           # Ingested data
│   ├── silver/           # Processed data
│   └── gold/             # ML-ready data
├── models/               # Trained models
├── tests/                # Comprehensive test suite
├── artefacts/           # Analysis outputs
├── monitoring/          # Dashboards & alerts
├── deployment/          # Docker & K8s
├── docs/                # Documentation
└── scripts/             # Utility scripts
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 4GB+ RAM
- 10GB+ disk space

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-ta
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

4. Set environment variables:
```bash
export ML_TA_ENV=development
export JWT_SECRET_KEY=your-secret-key
```

5. Run the demo:
```bash
python demo.py
```

## 🔧 Configuration

The system uses YAML-based configuration with environment-specific overrides:

- `config/settings.yaml` - Base configuration
- `config/development.yaml` - Development overrides
- `config/testing.yaml` - Testing overrides
- `config/production.yaml` - Production overrides

Environment is determined by the `ML_TA_ENV` environment variable.

## 📊 Data Pipeline

### Bronze Layer (Raw Data)
- Direct API ingestion from Binance
- Rate limiting and error handling
- Data validation and quality checks

### Silver Layer (Processed Data)
- Technical indicator calculations
- Data cleaning and normalization
- Feature engineering pipeline

### Gold Layer (ML-Ready Data)
- Feature selection and validation
- Train/test splits with temporal awareness
- Model-ready datasets

## 🤖 Machine Learning

### Supported Algorithms
- **LightGBM**: Gradient boosting with categorical support
- **XGBoost**: Extreme gradient boosting
- **CatBoost**: Categorical boosting
- **Random Forest**: Ensemble method
- **Neural Networks**: Deep learning models

### Model Validation
- Time series cross-validation
- Statistical significance testing
- Walk-forward analysis
- Out-of-time testing

### Hyperparameter Optimization
- Optuna-based optimization
- Multi-objective optimization
- Pruning strategies
- Parallel execution

## 📈 Technical Indicators

### Trend Indicators
- Simple/Exponential Moving Averages
- MACD, ADX, Parabolic SAR
- Ichimoku Cloud

### Momentum Indicators
- RSI, Stochastic, Williams %R
- Rate of Change, Momentum

### Volatility Indicators
- Bollinger Bands, ATR
- Keltner Channels

### Volume Indicators
- OBV, MFI, Chaikin Money Flow
- VWAP, Volume ROC

## 🔍 Model Interpretability

- **SHAP Analysis**: Feature importance and explanations
- **Permutation Importance**: Feature relevance ranking
- **Partial Dependence Plots**: Feature effect visualization
- **Model Diagnostics**: Residual analysis and calibration

## 🚨 Monitoring & Alerting

### Business Metrics
- Model accuracy and performance
- Prediction latency
- Data quality scores

### System Metrics
- CPU, memory, disk usage
- API response times
- Error rates

### Alerting
- Configurable thresholds
- Multiple notification channels
- Escalation policies

## 🔒 Security

- JWT-based authentication
- Role-based access control
- Input validation and sanitization
- Audit logging
- Data encryption

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Performance tests
pytest tests/performance/ -v

# All tests with coverage
pytest --cov=src --cov-report=html
```

## 📦 Deployment

### Docker
```bash
docker build -t ml-ta .
docker run -p 8000:8000 ml-ta
```

### Docker Compose
```bash
docker-compose up -d
```

### Kubernetes
```bash
kubectl apply -f deployment/kubernetes/
```

## 📚 API Documentation

Start the API server:
```bash
uvicorn src.api:app --reload
```

Access interactive documentation at: http://localhost:8000/docs

### Key Endpoints

- `POST /train` - Train new models
- `POST /predict` - Get predictions
- `GET /models` - List available models
- `GET /health` - Health check
- `GET /metrics` - System metrics

## 🔄 Backtesting

Run backtesting analysis:

```python
from src.backtest_engine import BacktestEngine

engine = BacktestEngine(config)
results = engine.run_backtest(
    strategy="ml_predictions",
    start_date="2023-01-01",
    end_date="2023-12-31"
)
```

## 📊 Performance Requirements

- **Latency**: <100ms prediction time
- **Throughput**: 1000+ concurrent requests
- **Accuracy**: >70% directional accuracy
- **Uptime**: 99.9% availability
- **Memory**: <4GB RAM usage
- **Coverage**: >95% test coverage

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Write comprehensive tests
- Update documentation
- Ensure security best practices

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## 🏆 Acknowledgments

- Technical analysis libraries: TA-Lib, pandas-ta
- Machine learning frameworks: scikit-learn, LightGBM, XGBoost, CatBoost
- Data processing: pandas, NumPy, PyArrow
- API framework: FastAPI
- Monitoring: Prometheus, Grafana

---

**⚠️ Disclaimer**: This system is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results.
