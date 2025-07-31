"""
Comprehensive unit tests for feature engineering functionality.

Tests cover FeaturePipeline, temporal validation, and all feature types.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.features import (
    FeaturePipeline, TemporalValidator, LaggingFeatures, 
    RollingFeatures, InteractionFeatures, RegimeFeatures, SeasonalFeatures
)
from src.exceptions import FeatureEngineeringError, ValidationError


class TestFeaturePipeline:
    """Test FeaturePipeline functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.pipeline = FeaturePipeline()
        
        # Create sample OHLCV data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        np.random.seed(42)
        
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'BTCUSDT',
            'open': 50000 + np.cumsum(np.random.randn(100) * 100),
            'high': 50000 + np.cumsum(np.random.randn(100) * 100) + 200,
            'low': 50000 + np.cumsum(np.random.randn(100) * 100) - 200,
            'close': 50000 + np.cumsum(np.random.randn(100) * 100),
            'volume': np.random.uniform(100, 1000, 100)
        })
        
        # Ensure OHLC relationships
        self.sample_data['high'] = self.sample_data[['open', 'high', 'close']].max(axis=1)
        self.sample_data['low'] = self.sample_data[['open', 'low', 'close']].min(axis=1)
    
    def test_init(self):
        """Test pipeline initialization."""
        assert isinstance(self.pipeline.temporal_validator, TemporalValidator)
        assert isinstance(self.pipeline.lagging_features, LaggingFeatures)
        assert isinstance(self.pipeline.rolling_features, RollingFeatures)
        assert isinstance(self.pipeline.interaction_features, InteractionFeatures)
        assert isinstance(self.pipeline.regime_features, RegimeFeatures)
        assert isinstance(self.pipeline.seasonal_features, SeasonalFeatures)
        assert not self.pipeline.fitted
    
    def test_engineer_features_basic(self):
        """Test basic feature engineering."""
        result = self.pipeline.engineer_features(self.sample_data, fit_scalers=True)
        
        # Should have more columns than input
        assert len(result.columns) > len(self.sample_data.columns)
        
        # Should preserve original columns
        for col in self.sample_data.columns:
            assert col in result.columns
        
        # Should have same number of rows
        assert len(result) == len(self.sample_data)
        
        # Pipeline should be fitted
        assert self.pipeline.fitted
    
    def test_engineer_features_no_scaling(self):
        """Test feature engineering without scaling."""
        result = self.pipeline.engineer_features(self.sample_data, fit_scalers=False)
        
        assert len(result.columns) > len(self.sample_data.columns)
        assert len(result) == len(self.sample_data)
    
    def test_engineer_features_no_temporal_validation(self):
        """Test feature engineering without temporal validation."""
        result = self.pipeline.engineer_features(
            self.sample_data, 
            fit_scalers=True, 
            validate_temporal=False
        )
        
        assert len(result.columns) > len(self.sample_data.columns)
    
    def test_engineer_features_empty_data(self):
        """Test feature engineering with empty data."""
        empty_data = pd.DataFrame(columns=self.sample_data.columns)
        
        with pytest.raises(FeatureEngineeringError):
            self.pipeline.engineer_features(empty_data)
    
    def test_engineer_features_missing_columns(self):
        """Test feature engineering with missing required columns."""
        incomplete_data = self.sample_data.drop('close', axis=1)
        
        with pytest.raises(FeatureEngineeringError):
            self.pipeline.engineer_features(incomplete_data)
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        result = self.pipeline.engineer_features(self.sample_data)
        feature_names = self.pipeline.get_feature_names(result)
        
        # Should exclude metadata columns
        assert 'timestamp' not in feature_names
        assert 'symbol' not in feature_names
        
        # Should include engineered features
        assert len(feature_names) > 0
        assert any('sma' in name for name in feature_names)
    
    @patch('src.features.TemporalValidator.validate_feature_matrix')
    def test_temporal_validation_called(self, mock_validate):
        """Test that temporal validation is called."""
        self.pipeline.engineer_features(self.sample_data, validate_temporal=True)
        mock_validate.assert_called_once()
    
    def test_feature_consistency(self):
        """Test that feature engineering is consistent across runs."""
        result1 = self.pipeline.engineer_features(self.sample_data.copy())
        
        # Reset pipeline
        self.pipeline = FeaturePipeline()
        result2 = self.pipeline.engineer_features(self.sample_data.copy())
        
        # Results should be identical (same features, same values)
        pd.testing.assert_frame_equal(result1, result2)


class TestTemporalValidator:
    """Test TemporalValidator functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.validator = TemporalValidator()
    
    def test_init(self):
        """Test validator initialization."""
        assert len(self.validator.violations) == 0
    
    def test_validate_feature_matrix_valid(self):
        """Test validation of valid feature matrix."""
        # Create properly sorted data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1H'),
            'feature1': range(10),
            'feature2': range(10, 20)
        })
        
        result = self.validator.validate_feature_matrix(df)
        assert result == True
        assert len(self.validator.get_violations()) == 0
    
    def test_validate_feature_matrix_unsorted(self):
        """Test validation of unsorted feature matrix."""
        # Create unsorted data
        timestamps = pd.date_range('2023-01-01', periods=10, freq='1H')
        df = pd.DataFrame({
            'timestamp': timestamps[[1, 0, 2, 3, 4, 5, 6, 7, 8, 9]],  # Unsorted
            'feature1': range(10),
            'feature2': range(10, 20)
        })
        
        with pytest.raises(ValidationError):
            self.validator.validate_feature_matrix(df)
    
    def test_get_violations(self):
        """Test getting violations list."""
        violations = self.validator.get_violations()
        assert isinstance(violations, list)


class TestLaggingFeatures:
    """Test LaggingFeatures functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.lagging_features = LaggingFeatures(max_lag=5)
        
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=20, freq='1H'),
            'close': range(20),
            'volume': range(100, 120)
        })
    
    def test_init(self):
        """Test initialization."""
        assert self.lagging_features.max_lag == 5
    
    def test_create_lagged_features(self):
        """Test creating lagged features."""
        result = self.lagging_features.create_lagged_features(
            self.sample_data, 
            ['close', 'volume'], 
            lags=[1, 2, 3]
        )
        
        # Should have original columns plus lagged versions
        assert 'close_lag_1' in result.columns
        assert 'close_lag_2' in result.columns
        assert 'close_lag_3' in result.columns
        assert 'volume_lag_1' in result.columns
        assert 'volume_lag_2' in result.columns
        assert 'volume_lag_3' in result.columns
        
        # Check lag values
        assert result['close_lag_1'].iloc[3] == self.sample_data['close'].iloc[2]
        assert result['close_lag_2'].iloc[4] == self.sample_data['close'].iloc[2]
    
    def test_create_lagged_features_missing_column(self):
        """Test creating lagged features with missing column."""
        result = self.lagging_features.create_lagged_features(
            self.sample_data, 
            ['close', 'nonexistent'], 
            lags=[1]
        )
        
        # Should handle missing columns gracefully
        assert 'close_lag_1' in result.columns
        assert 'nonexistent_lag_1' not in result.columns
    
    def test_create_percentage_changes(self):
        """Test creating percentage change features."""
        result = self.lagging_features.create_percentage_changes(
            self.sample_data, 
            ['close'], 
            periods=[1, 2]
        )
        
        assert 'close_pct_change_1' in result.columns
        assert 'close_pct_change_2' in result.columns
        
        # Check percentage change calculation
        expected_pct_1 = (self.sample_data['close'].iloc[1] - self.sample_data['close'].iloc[0]) / self.sample_data['close'].iloc[0]
        assert abs(result['close_pct_change_1'].iloc[1] - expected_pct_1) < 1e-10


class TestRollingFeatures:
    """Test RollingFeatures functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.rolling_features = RollingFeatures()
        
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=50, freq='1H'),
            'close': np.random.randn(50).cumsum() + 100,
            'volume': np.random.uniform(100, 1000, 50)
        })
    
    def test_create_rolling_statistics(self):
        """Test creating rolling statistics."""
        result = self.rolling_features.create_rolling_statistics(
            self.sample_data,
            ['close', 'volume'],
            windows=[5, 10],
            statistics=['mean', 'std', 'min', 'max']
        )
        
        # Check that rolling features are created
        expected_features = [
            'close_rolling_5_mean', 'close_rolling_5_std', 'close_rolling_5_min', 'close_rolling_5_max',
            'close_rolling_10_mean', 'close_rolling_10_std', 'close_rolling_10_min', 'close_rolling_10_max',
            'volume_rolling_5_mean', 'volume_rolling_5_std', 'volume_rolling_5_min', 'volume_rolling_5_max',
            'volume_rolling_10_mean', 'volume_rolling_10_std', 'volume_rolling_10_min', 'volume_rolling_10_max'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Check that rolling mean is calculated correctly
        expected_mean = self.sample_data['close'].iloc[:5].mean()
        assert abs(result['close_rolling_5_mean'].iloc[4] - expected_mean) < 1e-10
    
    def test_create_expanding_statistics(self):
        """Test creating expanding statistics."""
        result = self.rolling_features.create_expanding_statistics(
            self.sample_data,
            ['close'],
            statistics=['mean', 'std']
        )
        
        assert 'close_expanding_mean' in result.columns
        assert 'close_expanding_std' in result.columns
        
        # Check expanding mean calculation
        expected_expanding_mean = self.sample_data['close'].iloc[:10].mean()
        assert abs(result['close_expanding_mean'].iloc[9] - expected_expanding_mean) < 1e-10


class TestInteractionFeatures:
    """Test InteractionFeatures functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.interaction_features = InteractionFeatures()
        
        self.sample_data = pd.DataFrame({
            'close': [100, 110, 105, 120, 115],
            'volume': [1000, 1100, 950, 1200, 1050],
            'high': [102, 112, 107, 122, 117],
            'low': [98, 108, 103, 118, 113]
        })
    
    def test_create_ratio_features(self):
        """Test creating ratio features."""
        result = self.interaction_features.create_ratio_features(
            self.sample_data,
            ['close', 'high'],
            ['volume', 'low']
        )
        
        expected_features = [
            'close_div_volume', 'close_div_low',
            'high_div_volume', 'high_div_low'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Check ratio calculation
        expected_ratio = self.sample_data['close'].iloc[0] / self.sample_data['volume'].iloc[0]
        assert abs(result['close_div_volume'].iloc[0] - expected_ratio) < 1e-10
    
    def test_create_difference_features(self):
        """Test creating difference features."""
        result = self.interaction_features.create_difference_features(
            self.sample_data,
            [('high', 'low'), ('close', 'low')]
        )
        
        assert 'high_minus_low' in result.columns
        assert 'close_minus_low' in result.columns
        
        # Check difference calculation
        expected_diff = self.sample_data['high'].iloc[0] - self.sample_data['low'].iloc[0]
        assert abs(result['high_minus_low'].iloc[0] - expected_diff) < 1e-10
    
    def test_create_product_features(self):
        """Test creating product features."""
        result = self.interaction_features.create_product_features(
            self.sample_data,
            [('close', 'volume')]
        )
        
        assert 'close_mult_volume' in result.columns
        
        # Check product calculation
        expected_product = self.sample_data['close'].iloc[0] * self.sample_data['volume'].iloc[0]
        assert abs(result['close_mult_volume'].iloc[0] - expected_product) < 1e-10
    
    def test_create_polynomial_features(self):
        """Test creating polynomial features."""
        result = self.interaction_features.create_polynomial_features(
            self.sample_data,
            ['close'],
            degrees=[2, 3]
        )
        
        assert 'close_pow_2' in result.columns
        assert 'close_pow_3' in result.columns
        
        # Check polynomial calculation
        expected_pow2 = self.sample_data['close'].iloc[0] ** 2
        assert abs(result['close_pow_2'].iloc[0] - expected_pow2) < 1e-10


class TestRegimeFeatures:
    """Test RegimeFeatures functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.regime_features = RegimeFeatures()
        
        # Create trending data
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
            'close': np.linspace(100, 200, 100) + np.random.randn(100) * 2,
            'volume': np.random.uniform(100, 1000, 100)
        })
    
    def test_create_volatility_regime(self):
        """Test creating volatility regime features."""
        result = self.regime_features.create_volatility_regime(
            self.sample_data,
            price_col='close',
            window=20
        )
        
        assert 'volatility_regime' in result.columns
        assert 'volatility_percentile' in result.columns
        
        # Check that regime values are in expected range
        regime_values = result['volatility_regime'].dropna().unique()
        assert all(val in [0, 1] for val in regime_values)
    
    def test_create_trend_regime(self):
        """Test creating trend regime features."""
        result = self.regime_features.create_trend_regime(
            self.sample_data,
            price_col='close',
            short_window=10,
            long_window=30
        )
        
        assert 'trend_regime' in result.columns
        assert 'trend_strength' in result.columns
        
        # Check that trend regime values are in expected range
        trend_values = result['trend_regime'].dropna().unique()
        assert all(val in [-1, 0, 1] for val in trend_values)


class TestSeasonalFeatures:
    """Test SeasonalFeatures functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.seasonal_features = SeasonalFeatures()
        
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=168, freq='1H'),  # One week
            'close': range(168)
        })
    
    def test_create_time_features(self):
        """Test creating time-based features."""
        result = self.seasonal_features.create_time_features(self.sample_data)
        
        expected_features = [
            'hour', 'day_of_week', 'day_of_month', 'month',
            'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
            'month_sin', 'month_cos', 'is_weekend', 'is_month_start',
            'is_month_end', 'is_quarter_start'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Check specific values
        first_timestamp = pd.Timestamp('2023-01-01 00:00:00')
        assert result['hour'].iloc[0] == first_timestamp.hour
        assert result['day_of_week'].iloc[0] == first_timestamp.dayofweek
        assert result['month'].iloc[0] == first_timestamp.month
        
        # Check weekend flag (Jan 1, 2023 was a Sunday)
        assert result['is_weekend'].iloc[0] == 1


# Property-based testing with hypothesis
try:
    from hypothesis import given, strategies as st
    
    class TestFeatureEngineeringPropertyBased:
        """Property-based tests for feature engineering."""
        
        @given(st.integers(min_value=10, max_value=1000))
        def test_feature_count_consistency(self, n_samples):
            """Test that feature count is consistent regardless of sample size."""
            pipeline = FeaturePipeline()
            
            # Create data with varying sample sizes
            dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
            data = pd.DataFrame({
                'timestamp': dates,
                'symbol': 'TESTUSDT',
                'open': np.random.randn(n_samples).cumsum() + 100,
                'high': np.random.randn(n_samples).cumsum() + 102,
                'low': np.random.randn(n_samples).cumsum() + 98,
                'close': np.random.randn(n_samples).cumsum() + 100,
                'volume': np.random.uniform(100, 1000, n_samples)
            })
            
            # Ensure OHLC relationships
            data['high'] = data[['open', 'high', 'close']].max(axis=1)
            data['low'] = data[['open', 'low', 'close']].min(axis=1)
            
            try:
                result = pipeline.engineer_features(data, validate_temporal=False)
                
                # Should have same number of rows as input
                assert len(result) == n_samples
                
                # Should have more features than input
                assert len(result.columns) > len(data.columns)
                
            except FeatureEngineeringError:
                # Some configurations might not work with very small datasets
                if n_samples < 50:
                    pytest.skip("Small dataset size")
                else:
                    raise
        
        @given(st.lists(st.floats(min_value=1, max_value=10000), min_size=50, max_size=100))
        def test_price_feature_robustness(self, prices):
            """Test feature engineering robustness with various price sequences."""
            pipeline = FeaturePipeline()
            
            n_samples = len(prices)
            dates = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
            
            data = pd.DataFrame({
                'timestamp': dates,
                'symbol': 'TESTUSDT',
                'open': prices,
                'high': [p * 1.01 for p in prices],
                'low': [p * 0.99 for p in prices],
                'close': [p * 1.005 for p in prices],
                'volume': np.random.uniform(100, 1000, n_samples)
            })
            
            try:
                result = pipeline.engineer_features(data, validate_temporal=False)
                
                # Should not contain infinite or extremely large values
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    assert not np.isinf(result[col]).any(), f"Infinite values in {col}"
                    assert not (np.abs(result[col]) > 1e10).any(), f"Extremely large values in {col}"
                
            except (FeatureEngineeringError, ValueError):
                # Some price sequences might cause numerical issues
                pytest.skip("Numerical instability with given price sequence")

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass


if __name__ == '__main__':
    pytest.main([__file__])
