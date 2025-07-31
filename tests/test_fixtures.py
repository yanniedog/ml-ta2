"""
Comprehensive test fixtures and data factories for ML-TA system testing.
Provides realistic test data generation and mock services for isolated testing.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock
import tempfile
import shutil
from pathlib import Path
import json
import sqlite3
from contextlib import contextmanager

from src.config import ConfigManager
from src.data_fetcher import BinanceDataFetcher
from src.models import ModelManager
from src.utils import ensure_directory


class TestDataFactory:
    """Factory for generating realistic test data for various scenarios."""
    
    @staticmethod
    def generate_ohlcv_data(
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        periods: int = 1000,
        start_price: float = 50000.0,
        volatility: float = 0.02,
        trend: float = 0.0001
    ) -> pd.DataFrame:
        """Generate realistic OHLCV data with configurable parameters."""
        np.random.seed(42)  # For reproducible tests
        
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=periods),
            periods=periods,
            freq='H'
        )
        
        # Generate price series with trend and volatility
        returns = np.random.normal(trend, volatility, periods)
        prices = [start_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data
        data = []
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            # Generate realistic OHLC from close price
            volatility_factor = np.random.uniform(0.5, 1.5)
            high = price * (1 + volatility * volatility_factor * np.random.uniform(0, 1))
            low = price * (1 - volatility * volatility_factor * np.random.uniform(0, 1))
            open_price = prices[i-1] if i > 0 else price
            
            # Ensure OHLC consistency
            high = max(high, open_price, price)
            low = min(low, open_price, price)
            
            # Generate realistic volume
            base_volume = 1000 + np.random.exponential(500)
            volume = base_volume * (1 + abs(returns[i]) * 10)  # Higher volume on big moves
            
            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume,
                'symbol': symbol,
                'interval': interval
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def generate_features_data(n_samples: int = 1000, n_features: int = 200) -> pd.DataFrame:
        """Generate realistic feature matrix for testing."""
        np.random.seed(42)
        
        # Generate correlated features to simulate real feature engineering
        base_features = np.random.randn(n_samples, 20)
        
        features = []
        feature_names = []
        
        # Add base features
        for i in range(20):
            features.append(base_features[:, i])
            feature_names.append(f'base_feature_{i}')
        
        # Add derived features (moving averages, ratios, etc.)
        for i in range(20):
            # Moving averages
            ma_5 = pd.Series(base_features[:, i]).rolling(5, min_periods=1).mean().values
            ma_20 = pd.Series(base_features[:, i]).rolling(20, min_periods=1).mean().values
            
            features.extend([ma_5, ma_20])
            feature_names.extend([f'ma5_feature_{i}', f'ma20_feature_{i}'])
            
            # Ratios
            if i < 19:
                ratio = base_features[:, i] / (base_features[:, i+1] + 1e-8)
                features.append(ratio)
                feature_names.append(f'ratio_{i}_{i+1}')
        
        # Add noise features to reach target count
        while len(features) < n_features:
            noise = np.random.randn(n_samples) * 0.1
            features.append(noise)
            feature_names.append(f'noise_feature_{len(features)}')
        
        # Truncate if we have too many
        features = features[:n_features]
        feature_names = feature_names[:n_features]
        
        df = pd.DataFrame(np.column_stack(features), columns=feature_names)
        
        # Add timestamp column
        df['timestamp'] = pd.date_range(
            start=datetime.now() - timedelta(hours=n_samples),
            periods=n_samples,
            freq='H'
        )
        
        return df
    
    @staticmethod
    def generate_predictions_data(n_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate realistic prediction data for testing."""
        np.random.seed(42)
        
        predictions = []
        base_time = datetime.now()
        
        for i in range(n_samples):
            pred = {
                'timestamp': base_time + timedelta(minutes=i),
                'symbol': 'BTCUSDT',
                'prediction': np.random.choice([0, 1], p=[0.6, 0.4]),
                'confidence': np.random.uniform(0.5, 0.95),
                'features_used': np.random.randint(150, 200),
                'model_version': f'v1.{np.random.randint(0, 10)}',
                'latency_ms': np.random.uniform(10, 100)
            }
            predictions.append(pred)
        
        return predictions


class MockServices:
    """Mock services for isolated testing."""
    
    @staticmethod
    def mock_binance_api() -> Mock:
        """Create a mock Binance API with realistic responses."""
        mock_api = Mock()
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [
                1640995200000,  # timestamp
                "50000.00",     # open
                "51000.00",     # high
                "49500.00",     # low
                "50500.00",     # close
                "100.50",       # volume
                1640998800000,  # close_time
                "5050000.00",   # quote_volume
                1000,           # count
                "50.25",        # taker_buy_base_volume
                "2525000.00",   # taker_buy_quote_volume
                "0"             # ignore
            ]
        ]
        
        mock_api.get.return_value = mock_response
        return mock_api
    
    @staticmethod
    def mock_database() -> Mock:
        """Create a mock database connection."""
        mock_db = Mock()
        mock_cursor = Mock()
        
        # Mock query results
        mock_cursor.fetchall.return_value = [
            (1, 'BTCUSDT', '2024-01-01', 50000.0, 51000.0, 49500.0, 50500.0, 100.5)
        ]
        mock_cursor.fetchone.return_value = (1, 'BTCUSDT', '2024-01-01', 50000.0)
        
        mock_db.cursor.return_value = mock_cursor
        mock_db.execute = mock_cursor.execute
        mock_db.commit = Mock()
        mock_db.rollback = Mock()
        
        return mock_db
    
    @staticmethod
    def mock_redis() -> Mock:
        """Create a mock Redis connection."""
        mock_redis = Mock()
        
        # Mock Redis operations
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.exists.return_value = False
        mock_redis.expire.return_value = True
        
        return mock_redis
    
    @staticmethod
    def mock_model() -> Mock:
        """Create a mock ML model."""
        mock_model = Mock()
        
        # Mock prediction methods
        mock_model.predict.return_value = np.array([0, 1, 0, 1])
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]])
        mock_model.feature_importances_ = np.random.random(200)
        
        return mock_model


class TestEnvironment:
    """Test environment setup and teardown."""
    
    def __init__(self):
        self.temp_dir = None
        self.test_db_path = None
        self.original_config = None
    
    def setup(self) -> str:
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix='ml_ta_test_')
        
        # Create test directory structure
        test_dirs = [
            'data/raw', 'data/bronze', 'data/silver', 'data/gold',
            'models', 'logs', 'artefacts', 'cache'
        ]
        
        for dir_path in test_dirs:
            ensure_directory(Path(self.temp_dir) / dir_path)
        
        # Create test database
        self.test_db_path = Path(self.temp_dir) / 'test.db'
        self._create_test_database()
        
        return self.temp_dir
    
    def teardown(self):
        """Clean up test environment."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def _create_test_database(self):
        """Create test database with sample data."""
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                interval TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                confidence REAL NOT NULL,
                model_version TEXT NOT NULL
            )
        ''')
        
        # Insert sample data
        sample_data = TestDataFactory.generate_ohlcv_data(periods=100)
        for _, row in sample_data.iterrows():
            cursor.execute('''
                INSERT INTO ohlcv_data (symbol, timestamp, open, high, low, close, volume, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row['symbol'], row['timestamp'], row['open'], row['high'],
                row['low'], row['close'], row['volume'], row['interval']
            ))
        
        conn.commit()
        conn.close()
    
    @contextmanager
    def isolated_config(self, config_overrides: Dict[str, Any] = None):
        """Context manager for isolated configuration testing."""
        original_config = {}
        
        try:
            # Apply overrides if provided
            if config_overrides:
                # Store original values
                config_manager = ConfigManager()
                for key, value in config_overrides.items():
                    if hasattr(config_manager, key):
                        original_config[key] = getattr(config_manager, key)
                        setattr(config_manager, key, value)
            
            yield
            
        finally:
            # Restore original values
            if config_overrides:
                config_manager = ConfigManager()
                for key, value in original_config.items():
                    setattr(config_manager, key, value)


# Pytest fixtures
@pytest.fixture(scope="session")
def test_environment():
    """Session-scoped test environment."""
    env = TestEnvironment()
    temp_dir = env.setup()
    yield env
    env.teardown()


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    return TestDataFactory.generate_ohlcv_data()


@pytest.fixture
def sample_features_data():
    """Generate sample features data for testing."""
    return TestDataFactory.generate_features_data()


@pytest.fixture
def sample_predictions_data():
    """Generate sample predictions data for testing."""
    return TestDataFactory.generate_predictions_data()


@pytest.fixture
def mock_binance_api():
    """Mock Binance API for testing."""
    return MockServices.mock_binance_api()


@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    return MockServices.mock_database()


@pytest.fixture
def mock_redis():
    """Mock Redis connection for testing."""
    return MockServices.mock_redis()


@pytest.fixture
def mock_model():
    """Mock ML model for testing."""
    return MockServices.mock_model()


@pytest.fixture
def isolated_config():
    """Isolated configuration for testing."""
    env = TestEnvironment()
    with env.isolated_config() as config:
        yield config


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    import random
    random.seed(42)


# Phase 5: Prediction System Test Fixtures

class SimpleMockModel:
    """A simple mock model for prediction testing."""
    
    def __init__(self):
        self.random_state = np.random.RandomState(42)
    
    def predict(self, X):
        """Make deterministic predictions for testing."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        n_samples = len(X)
        return self.random_state.rand(n_samples)
    
    def predict_proba(self, X):
        """Generate probability predictions for testing."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        n_samples = len(X)
        probas = self.random_state.rand(n_samples, 2)
        # Normalize to sum to 1
        probas = probas / probas.sum(axis=1, keepdims=True)
        return probas


@pytest.fixture
def sample_data_fixture():
    """Generate sample input data for prediction tests."""
    # Create simple feature dataframe
    data = pd.DataFrame({
        'feature1': np.random.randn(10),
        'feature2': np.random.randn(10),
        'feature3': np.random.randn(10),
        'feature4': np.random.uniform(0, 1, 10),
        'feature5': np.random.choice([0, 1], 10)
    })
    return data


@pytest.fixture
def sample_model_fixture():
    """Create a sample model for prediction testing."""
    return SimpleMockModel()


@pytest.fixture
def sample_prediction_fixture():
    """Generate sample prediction output for testing."""
    from src.prediction_engine import PredictionResponse
    
    # Create mock prediction response
    return PredictionResponse(
        request_id="test-request-123",
        timestamp=datetime.now(),
        predictions=np.array([0.1, 0.2, 0.3]),
        probabilities=np.array([[0.6, 0.4], [0.7, 0.3], [0.8, 0.2]]),
        processing_time_ms=15.5,
        model_name="test_model",
        metadata={"source": "test_fixture"}
    )
