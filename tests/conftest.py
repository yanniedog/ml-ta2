"""
Test Configuration and Fixtures for ML-TA System

This module provides comprehensive pytest fixtures for database connections,
API clients, mock data generators, and test configurations.
"""

import os
import sys
import pytest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Generator
from unittest.mock import Mock, MagicMock
import sqlite3

# Add src to path for imports
# Add project root to path for consistent imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from src.config import ConfigManager, MLTAConfig
from src.logging_config import LoggerFactory
from src.exceptions import MLTAException


@pytest.fixture(scope="session")
def test_config() -> MLTAConfig:
    """Create test configuration."""
    # Set test environment
    os.environ["ML_TA_ENV"] = "testing"
    
    config_manager = ConfigManager(config_path="config")
    return config_manager.load_config()


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def test_database(temp_dir: Path) -> Generator[str, None, None]:
    """Create temporary SQLite database for tests."""
    db_path = temp_dir / "test.db"
    db_url = f"sqlite:///{db_path}"
    
    # Create test tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create sample tables
    cursor.execute("""
        CREATE TABLE market_data (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            prediction REAL NOT NULL,
            confidence REAL NOT NULL,
            model_id TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()
    
    yield db_url


@pytest.fixture(scope="function")
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate 1000 data points
    n_points = 1000
    base_price = 50000  # Starting price for BTCUSDT
    
    # Generate realistic price movements
    returns = np.random.normal(0, 0.02, n_points)  # 2% daily volatility
    prices = [base_price]
    
    for i in range(1, n_points):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1))  # Ensure positive prices
    
    # Create OHLCV data
    data = []
    for i in range(n_points):
        open_price = prices[i]
        close_price = prices[i] if i == n_points - 1 else prices[i + 1]
        
        # Generate high and low based on volatility
        volatility = abs(returns[i]) * open_price
        high = max(open_price, close_price) + np.random.uniform(0, volatility)
        low = min(open_price, close_price) - np.random.uniform(0, volatility)
        
        # Ensure OHLC relationships
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        volume = np.random.uniform(100, 1000)
        
        timestamp = datetime.now() - timedelta(minutes=n_points - i)
        
        data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2),
            'symbol': 'BTCUSDT'
        })
    
    return pd.DataFrame(data)


@pytest.fixture(scope="function")
def sample_features_data(sample_ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    """Generate sample features data for testing."""
    df = sample_ohlcv_data.copy()
    
    # Add simple technical indicators
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['rsi'] = 50 + np.random.normal(0, 10, len(df))  # Mock RSI
    df['volume_sma'] = df['volume'].rolling(window=10).mean()
    
    # Add price-based features
    df['price_change'] = df['close'].pct_change()
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Add target variable (next period return)
    df['target'] = df['close'].pct_change().shift(-1)
    df['target_binary'] = (df['target'] > 0).astype(int)
    
    return df.dropna()


@pytest.fixture(scope="function")
def mock_binance_api() -> Mock:
    """Create mock Binance API client."""
    mock_api = Mock()
    
    # Mock successful response
    mock_api.get_klines.return_value = [
        [
            1640995200000,  # timestamp
            "50000.00",     # open
            "51000.00",     # high
            "49500.00",     # low
            "50500.00",     # close
            "100.50",       # volume
            1640995259999,  # close time
            "5025000.00",   # quote asset volume
            1000,           # number of trades
            "50.25",        # taker buy base asset volume
            "2512500.00",   # taker buy quote asset volume
            "0"             # ignore
        ]
    ]
    
    return mock_api


@pytest.fixture(scope="function")
def mock_model() -> Mock:
    """Create mock ML model."""
    mock_model = Mock()
    
    # Mock prediction methods
    mock_model.predict.return_value = np.array([0.6, 0.7, 0.4, 0.8, 0.3])
    mock_model.predict_proba.return_value = np.array([
        [0.4, 0.6],
        [0.3, 0.7],
        [0.6, 0.4],
        [0.2, 0.8],
        [0.7, 0.3]
    ])
    
    # Mock feature importance
    mock_model.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
    
    return mock_model


@pytest.fixture(scope="function")
def mock_redis_client() -> Mock:
    """Create mock Redis client."""
    mock_redis = Mock()
    
    # Mock Redis operations
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.exists.return_value = False
    mock_redis.ping.return_value = True
    
    return mock_redis


@pytest.fixture(scope="function")
def test_data_factory():
    """Factory for generating test data."""
    
    class TestDataFactory:
        @staticmethod
        def create_market_data(
            symbol: str = "BTCUSDT",
            start_date: datetime = None,
            periods: int = 100,
            base_price: float = 50000
        ) -> pd.DataFrame:
            """Create market data for testing."""
            if start_date is None:
                start_date = datetime.now() - timedelta(days=periods)
            
            dates = pd.date_range(start=start_date, periods=periods, freq='1H')
            
            # Generate price series
            np.random.seed(42)
            returns = np.random.normal(0, 0.01, periods)
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            data = []
            for i, date in enumerate(dates):
                open_price = prices[i]
                close_price = prices[i] * (1 + np.random.normal(0, 0.005))
                high = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.002)))
                low = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.002)))
                volume = np.random.uniform(100, 1000)
                
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'volume': round(volume, 2)
                })
            
            return pd.DataFrame(data)
        
        @staticmethod
        def create_prediction_data(
            model_id: str = "test_model",
            n_predictions: int = 50
        ) -> pd.DataFrame:
            """Create prediction data for testing."""
            np.random.seed(42)
            
            data = []
            for i in range(n_predictions):
                data.append({
                    'timestamp': datetime.now() - timedelta(hours=i),
                    'symbol': 'BTCUSDT',
                    'prediction': np.random.uniform(0, 1),
                    'confidence': np.random.uniform(0.5, 0.95),
                    'model_id': model_id,
                    'actual': np.random.choice([0, 1]) if i < 30 else None  # Some with actuals
                })
            
            return pd.DataFrame(data)
        
        @staticmethod
        def create_feature_matrix(
            n_samples: int = 100,
            n_features: int = 20
        ) -> pd.DataFrame:
            """Create feature matrix for testing."""
            np.random.seed(42)
            
            # Generate feature names
            feature_names = [f"feature_{i}" for i in range(n_features)]
            
            # Generate random features
            data = np.random.randn(n_samples, n_features)
            
            df = pd.DataFrame(data, columns=feature_names)
            
            # Add timestamp
            df['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(hours=n_samples),
                periods=n_samples,
                freq='1H'
            )
            
            # Add target
            df['target'] = (df[feature_names].sum(axis=1) > 0).astype(int)
            
            return df
    
    return TestDataFactory


@pytest.fixture(scope="function")
def mock_services():
    """Create collection of mock services."""
    
    class MockServices:
        def __init__(self):
            self.data_fetcher = Mock()
            self.feature_engineer = Mock()
            self.model_trainer = Mock()
            self.predictor = Mock()
            self.monitor = Mock()
            
            # Setup default behaviors
            self._setup_data_fetcher()
            self._setup_feature_engineer()
            self._setup_model_trainer()
            self._setup_predictor()
            self._setup_monitor()
        
        def _setup_data_fetcher(self):
            """Setup data fetcher mock."""
            self.data_fetcher.fetch_data.return_value = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
                'open': np.random.uniform(49000, 51000, 100),
                'high': np.random.uniform(50000, 52000, 100),
                'low': np.random.uniform(48000, 50000, 100),
                'close': np.random.uniform(49500, 51500, 100),
                'volume': np.random.uniform(100, 1000, 100)
            })
        
        def _setup_feature_engineer(self):
            """Setup feature engineer mock."""
            self.feature_engineer.engineer_features.return_value = pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                'feature_3': np.random.randn(100),
                'target': np.random.choice([0, 1], 100)
            })
        
        def _setup_model_trainer(self):
            """Setup model trainer mock."""
            mock_model = Mock()
            mock_model.predict.return_value = np.random.uniform(0, 1, 50)
            mock_model.feature_importances_ = np.random.uniform(0, 1, 3)
            
            self.model_trainer.train_model.return_value = mock_model
            self.model_trainer.validate_model.return_value = {
                'accuracy': 0.75,
                'precision': 0.73,
                'recall': 0.78,
                'f1_score': 0.75
            }
        
        def _setup_predictor(self):
            """Setup predictor mock."""
            self.predictor.predict.return_value = {
                'prediction': 0.65,
                'confidence': 0.82,
                'timestamp': datetime.now()
            }
        
        def _setup_monitor(self):
            """Setup monitor mock."""
            self.monitor.get_metrics.return_value = {
                'model_accuracy': 0.75,
                'prediction_latency_ms': 45,
                'memory_usage_mb': 512,
                'cpu_usage_percent': 25
            }
    
    return MockServices()


@pytest.fixture(scope="session")
def logger():
    """Create test logger."""
    logger_factory = LoggerFactory()
    return logger_factory.get_logger("test").get_logger()


@pytest.fixture(autouse=True)
def setup_test_environment(test_config: MLTAConfig, temp_dir: Path):
    """Setup test environment before each test."""
    # Set deterministic seed
    np.random.seed(42)
    
    # Create test directories
    from src.config import get_model_dict
    for path_name, path_value in get_model_dict(test_config.paths).items():
        test_path = temp_dir / path_value.lstrip('./')
        test_path.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables for testing
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "WARNING"  # Reduce log noise in tests


@pytest.fixture(scope="function")
def api_client():
    """Create test API client."""
    from fastapi.testclient import TestClient
    
    # This will be implemented when we create the API module
    # For now, return a mock
    mock_client = Mock()
    mock_client.get.return_value.status_code = 200
    mock_client.post.return_value.status_code = 200
    mock_client.put.return_value.status_code = 200
    mock_client.delete.return_value.status_code = 200
    
    return mock_client


# Utility functions for tests
def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = True):
    """Assert two DataFrames are equal with better error messages."""
    try:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
    except AssertionError as e:
        pytest.fail(f"DataFrames are not equal: {e}")


def assert_model_performance(metrics: Dict[str, float], min_accuracy: float = 0.6):
    """Assert model performance meets minimum requirements."""
    assert 'accuracy' in metrics, "Accuracy metric missing"
    assert metrics['accuracy'] >= min_accuracy, f"Accuracy {metrics['accuracy']} below minimum {min_accuracy}"
    
    if 'precision' in metrics:
        assert metrics['precision'] >= 0.5, f"Precision {metrics['precision']} too low"
    
    if 'recall' in metrics:
        assert metrics['recall'] >= 0.5, f"Recall {metrics['recall']} too low"


def create_test_exception(
    message: str = "Test exception",
    error_code: str = "TEST_ERROR"
) -> MLTAException:
    """Create test exception for error handling tests."""
    return MLTAException(
        message=message,
        error_code=error_code
    )
