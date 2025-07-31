"""
Integration tests for end-to-end ML pipeline functionality.

Tests cover complete workflows from data fetching to prediction.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from src.data_fetcher import BinanceDataFetcher
from src.features import FeaturePipeline
from src.models import ModelTrainer
from src.prediction_engine import PredictionEngine
from src.config import ConfigManager
from src.exceptions import DataFetchError, FeatureEngineeringError, ModelTrainingError


class TestEndToEndPipeline:
    """Test complete ML pipeline integration."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager()
        
        # Create sample data for testing
        self.sample_data = self._create_sample_data()
    
    def teardown_method(self):
        """Cleanup test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create realistic sample OHLCV data."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic price data with trend and volatility
        base_price = 50000
        trend = np.linspace(0, 5000, n_samples)  # Upward trend
        noise = np.random.randn(n_samples).cumsum() * 100
        
        close_prices = base_price + trend + noise
        
        # Generate OHLC from close prices
        opens = np.roll(close_prices, 1)
        opens[0] = base_price
        
        # Add realistic spreads
        highs = close_prices + np.random.uniform(50, 500, n_samples)
        lows = close_prices - np.random.uniform(50, 500, n_samples)
        
        # Ensure OHLC relationships
        highs = np.maximum(highs, np.maximum(opens, close_prices))
        lows = np.minimum(lows, np.minimum(opens, close_prices))
        
        # Generate volume data
        volumes = np.random.lognormal(mean=6, sigma=0.5, size=n_samples)
        
        return pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'symbol': 'BTCUSDT',
            'open': opens,
            'high': highs,
            'low': lows,
            'close': close_prices,
            'volume': volumes
        })
    
    def test_data_to_features_pipeline(self):
        """Test data fetching to feature engineering pipeline."""
        # Initialize components
        feature_pipeline = FeaturePipeline()
        
        # Engineer features
        features_df = feature_pipeline.engineer_features(
            self.sample_data,
            fit_scalers=True,
            validate_temporal=True
        )
        
        # Verify feature engineering results
        assert len(features_df) == len(self.sample_data)
        assert len(features_df.columns) > len(self.sample_data.columns)
        
        # Check for expected feature types
        feature_names = feature_pipeline.get_feature_names(features_df)
        
        # Should have technical indicators
        assert any('sma' in name for name in feature_names)
        assert any('rsi' in name for name in feature_names)
        
        # Should have lagged features
        assert any('lag' in name for name in feature_names)
        
        # Should have rolling features
        assert any('rolling' in name for name in feature_names)
        
        # Should have interaction features
        assert any('div' in name or 'mult' in name for name in feature_names)
    
    def test_features_to_model_pipeline(self):
        """Test feature engineering to model training pipeline."""
        # Engineer features
        feature_pipeline = FeaturePipeline()
        features_df = feature_pipeline.engineer_features(self.sample_data, fit_scalers=True)
        
        # Create target variable (next period return)
        features_df['target'] = (features_df['close'].shift(-1) / features_df['close'] - 1) > 0
        features_df = features_df.dropna()
        
        # Prepare training data
        feature_names = feature_pipeline.get_feature_names(features_df)
        X = features_df[feature_names]
        y = features_df['target']
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        model_trainer = ModelTrainer()
        model, metrics = model_trainer.train_model(X_train, y_train, model_name='test_model')
        
        # Verify model training
        assert model is not None
        assert metrics.accuracy is not None
        assert metrics.training_time is not None
        
        # Test prediction
        predictions = model_trainer.predict('test_model', X_test)
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1] for pred in predictions)  # Binary classification
    
    def test_complete_pipeline_workflow(self):
        """Test complete pipeline from data to prediction."""
        # Step 1: Feature Engineering
        feature_pipeline = FeaturePipeline()
        features_df = feature_pipeline.engineer_features(self.sample_data, fit_scalers=True)
        
        # Step 2: Target Creation
        features_df['target'] = (features_df['close'].shift(-1) / features_df['close'] - 1) > 0
        features_df = features_df.dropna()
        
        # Step 3: Data Splitting
        feature_names = feature_pipeline.get_feature_names(features_df)
        X = features_df[feature_names]
        y = features_df['target']
        
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Step 4: Model Training
        model_trainer = ModelTrainer()
        model, metrics = model_trainer.train_model(X_train, y_train, model_name='pipeline_test')
        
        # Step 5: Prediction
        predictions = model_trainer.predict('pipeline_test', X_test)
        
        # Step 6: Validation
        accuracy = np.mean(predictions == y_test)
        
        # Verify complete pipeline
        assert len(predictions) == len(y_test)
        assert 0 <= accuracy <= 1
        assert metrics.accuracy is not None
        
        # Pipeline should produce reasonable results
        assert accuracy > 0.4  # Better than random for most cases
    
    def test_pipeline_with_different_symbols(self):
        """Test pipeline with different trading symbols."""
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        for symbol in symbols:
            # Create symbol-specific data
            symbol_data = self.sample_data.copy()
            symbol_data['symbol'] = symbol
            
            # Run pipeline
            feature_pipeline = FeaturePipeline()
            features_df = feature_pipeline.engineer_features(symbol_data, fit_scalers=True)
            
            # Verify results
            assert len(features_df) == len(symbol_data)
            assert all(features_df['symbol'] == symbol)
    
    def test_pipeline_with_different_timeframes(self):
        """Test pipeline with different timeframes."""
        timeframes = ['1H', '4H', '1D']
        
        for timeframe in timeframes:
            # Create timeframe-specific data
            if timeframe == '1H':
                data = self.sample_data
            elif timeframe == '4H':
                # Resample to 4H
                data = self.sample_data.set_index('timestamp').resample('4H').agg({
                    'symbol': 'first',
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).reset_index().dropna()
            else:  # 1D
                # Resample to 1D
                data = self.sample_data.set_index('timestamp').resample('1D').agg({
                    'symbol': 'first',
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).reset_index().dropna()
            
            if len(data) < 50:  # Skip if too little data after resampling
                continue
            
            # Run pipeline
            feature_pipeline = FeaturePipeline()
            features_df = feature_pipeline.engineer_features(data, fit_scalers=True)
            
            # Verify results
            assert len(features_df) == len(data)
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling and recovery."""
        # Test with insufficient data
        small_data = self.sample_data.head(10)
        
        feature_pipeline = FeaturePipeline()
        
        with pytest.raises(FeatureEngineeringError):
            feature_pipeline.engineer_features(small_data, fit_scalers=True)
        
        # Test with missing columns
        incomplete_data = self.sample_data.drop('close', axis=1)
        
        with pytest.raises(FeatureEngineeringError):
            feature_pipeline.engineer_features(incomplete_data, fit_scalers=True)
        
        # Test with invalid data types
        invalid_data = self.sample_data.copy()
        invalid_data['close'] = 'invalid'
        
        with pytest.raises(FeatureEngineeringError):
            feature_pipeline.engineer_features(invalid_data, fit_scalers=True)
    
    def test_pipeline_memory_efficiency(self):
        """Test pipeline memory usage with large datasets."""
        # Create larger dataset
        large_data = pd.concat([self.sample_data] * 5, ignore_index=True)
        large_data['timestamp'] = pd.date_range('2023-01-01', periods=len(large_data), freq='1H')
        
        # Monitor memory usage (simplified)
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run pipeline
        feature_pipeline = FeaturePipeline()
        features_df = feature_pipeline.engineer_features(large_data, fit_scalers=True)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        # Verify results and memory usage
        assert len(features_df) == len(large_data)
        assert memory_increase < 1000  # Should not use more than 1GB additional memory
    
    def test_pipeline_reproducibility(self):
        """Test pipeline reproducibility across runs."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Run pipeline twice
        feature_pipeline1 = FeaturePipeline()
        features_df1 = feature_pipeline1.engineer_features(self.sample_data.copy(), fit_scalers=True)
        
        np.random.seed(42)
        feature_pipeline2 = FeaturePipeline()
        features_df2 = feature_pipeline2.engineer_features(self.sample_data.copy(), fit_scalers=True)
        
        # Results should be identical
        pd.testing.assert_frame_equal(features_df1, features_df2)
    
    def test_pipeline_performance_benchmarks(self):
        """Test pipeline performance meets requirements."""
        import time
        
        # Test processing speed: should handle 10,000+ records in <30 seconds
        large_data = pd.concat([self.sample_data] * 10, ignore_index=True)
        large_data['timestamp'] = pd.date_range('2023-01-01', periods=len(large_data), freq='1H')
        
        start_time = time.time()
        
        feature_pipeline = FeaturePipeline()
        features_df = feature_pipeline.engineer_features(large_data, fit_scalers=True)
        
        processing_time = time.time() - start_time
        
        # Verify performance requirements
        assert len(large_data) >= 10000
        assert processing_time < 30  # Should complete in under 30 seconds
        assert len(features_df) == len(large_data)
        
        # Test feature count requirement: should generate 200+ features
        feature_names = feature_pipeline.get_feature_names(features_df)
        assert len(feature_names) >= 200


class TestDataQualityIntegration:
    """Test data quality assurance integration."""
    
    def setup_method(self):
        """Setup test environment."""
        self.sample_data = self._create_sample_data_with_issues()
    
    def _create_sample_data_with_issues(self) -> pd.DataFrame:
        """Create sample data with various quality issues."""
        np.random.seed(42)
        n_samples = 500
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'symbol': 'BTCUSDT',
            'open': np.random.uniform(45000, 55000, n_samples),
            'high': np.random.uniform(45000, 55000, n_samples),
            'low': np.random.uniform(45000, 55000, n_samples),
            'close': np.random.uniform(45000, 55000, n_samples),
            'volume': np.random.uniform(100, 1000, n_samples)
        })
        
        # Introduce quality issues
        # Missing values
        data.loc[10:15, 'close'] = np.nan
        
        # Outliers
        data.loc[50, 'close'] = 1000000  # Extreme outlier
        
        # Invalid OHLC relationships
        data.loc[100, 'high'] = data.loc[100, 'low'] - 100  # High < Low
        
        return data
    
    def test_data_quality_detection(self):
        """Test data quality issue detection."""
        from src.data_quality import DataQualityFramework
        
        quality_framework = DataQualityFramework()
        quality_report = quality_framework.assess_data_quality(self.sample_data)
        
        # Should detect quality issues
        assert quality_report.completeness_score < 1.0  # Missing values detected
        assert quality_report.validity_score < 1.0  # Invalid OHLC detected
        assert len(quality_report.issues) > 0
    
    def test_data_quality_remediation(self):
        """Test data quality issue remediation."""
        from src.data_quality import DataQualityFramework
        
        quality_framework = DataQualityFramework()
        cleaned_data = quality_framework.clean_data(self.sample_data)
        
        # Should have fewer quality issues
        assert not cleaned_data['close'].isna().any()  # Missing values handled
        assert all(cleaned_data['high'] >= cleaned_data['low'])  # OHLC relationships fixed


class TestModelIntegration:
    """Test model training and serving integration."""
    
    def setup_method(self):
        """Setup test environment."""
        # Create prepared training data
        np.random.seed(42)
        n_samples = 1000
        n_features = 50
        
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_train = np.random.randint(0, 2, n_samples)
        
        self.X_test = pd.DataFrame(
            np.random.randn(200, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_test = np.random.randint(0, 2, 200)
    
    def test_model_training_integration(self):
        """Test model training with different algorithms."""
        algorithms = ['random_forest', 'gradient_boosting']
        
        for algorithm in algorithms:
            model_trainer = ModelTrainer()
            model_trainer.config.model_type = algorithm
            
            model, metrics = model_trainer.train_model(
                self.X_train, self.y_train, model_name=f'{algorithm}_test'
            )
            
            # Verify training results
            assert model is not None
            assert metrics.accuracy is not None
            assert metrics.training_time is not None
            assert 0 <= metrics.accuracy <= 1
    
    def test_model_serving_integration(self):
        """Test model serving and prediction."""
        from src.model_serving import ModelServer
        
        # Train model
        model_trainer = ModelTrainer()
        model, metrics = model_trainer.train_model(
            self.X_train, self.y_train, model_name='serving_test'
        )
        
        # Initialize model server
        model_server = ModelServer()
        model_server.load_model('serving_test', model)
        
        # Test prediction
        predictions = model_server.predict('serving_test', self.X_test)
        
        # Verify predictions
        assert len(predictions) == len(self.X_test)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_validation_integration(self):
        """Test model validation and cross-validation."""
        from src.model_validation import ValidationFramework
        
        # Train model
        model_trainer = ModelTrainer()
        model, metrics = model_trainer.train_model(
            self.X_train, self.y_train, model_name='validation_test'
        )
        
        # Validate model
        validation_framework = ValidationFramework()
        validation_results = validation_framework.validate_model(
            model, self.X_test, self.y_test
        )
        
        # Verify validation results
        assert validation_results.accuracy is not None
        assert validation_results.precision is not None
        assert validation_results.recall is not None
        assert validation_results.f1_score is not None


if __name__ == '__main__':
    pytest.main([__file__])
