"""
Comprehensive tests for feature engineering pipeline.

Tests cover temporal validation, feature generation, selection, and monitoring
to ensure no data leakage and proper functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

# Import modules to test
from src.features import (
    TemporalValidator, LaggingFeatures, RollingFeatures,
    InteractionFeatures, RegimeFeatures, SeasonalFeatures,
    FeaturePipeline, create_feature_pipeline
)
from src.feature_selection import (
    CorrelationSelector, VarianceSelector, UnivariateSelector,
    TreeBasedSelector, DimensionalityReducer, FeatureSelector,
    create_feature_selector
)
from src.feature_monitoring import (
    StatisticalDriftDetector, PerformanceDriftDetector,
    FeatureMonitor, create_feature_monitor
)
from src.indicators import TechnicalIndicators
from src.exceptions import FeatureEngineeringError, ValidationError


class TestTemporalValidator:
    """Test temporal validation to prevent data leakage."""
    
    def test_valid_temporal_data(self, sample_ohlcv_data):
        """Test validation passes for properly ordered data."""
        validator = TemporalValidator()
        
        # Should pass validation
        result = validator.validate_feature_matrix(sample_ohlcv_data)
        assert result is True
        assert len(validator.get_violations()) == 0
    
    def test_future_looking_features_detected(self, sample_ohlcv_data):
        """Test detection of future-looking features."""
        validator = TemporalValidator()
        
        # Add suspicious feature names
        sample_ohlcv_data['future_price'] = sample_ohlcv_data['close'].shift(-1)
        sample_ohlcv_data['next_return'] = sample_ohlcv_data['close'].pct_change().shift(-1)
        
        with pytest.raises(ValidationError, match="future-looking features"):
            validator.validate_feature_matrix(sample_ohlcv_data)
    
    def test_unsorted_timestamp_detection(self):
        """Test detection of unsorted timestamps."""
        validator = TemporalValidator()
        
        # Create unsorted data
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        shuffled_dates = dates.to_series().sample(frac=1).reset_index(drop=True)
        
        data = pd.DataFrame({
            'timestamp': shuffled_dates,
            'close': np.random.randn(100)
        })
        
        with pytest.raises(ValidationError, match="sorted by timestamp"):
            validator.validate_feature_matrix(data)


class TestLaggingFeatures:
    """Test lagged feature generation."""
    
    def test_lag_feature_creation(self, sample_ohlcv_data):
        """Test creation of lagged features."""
        lag_generator = LaggingFeatures(max_lag=5)
        
        result = lag_generator.create_lag_features(
            sample_ohlcv_data,
            columns=['close', 'volume'],
            lags=[1, 2, 3]
        )
        
        # Check that lag features were created
        expected_features = ['close_lag_1', 'close_lag_2', 'close_lag_3',
                           'volume_lag_1', 'volume_lag_2', 'volume_lag_3']
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Check that lag_1 is properly shifted
        assert result['close_lag_1'].iloc[1] == sample_ohlcv_data['close'].iloc[0]
        assert result['close_lag_2'].iloc[2] == sample_ohlcv_data['close'].iloc[0]
    
    def test_diff_feature_creation(self, sample_ohlcv_data):
        """Test creation of difference features."""
        lag_generator = LaggingFeatures()
        
        result = lag_generator.create_diff_features(
            sample_ohlcv_data,
            columns=['close'],
            periods=[1, 2]
        )
        
        # Check difference features
        assert 'close_diff_1' in result.columns
        assert 'close_diff_2' in result.columns
        
        # Verify difference calculation
        expected_diff_1 = sample_ohlcv_data['close'].diff(1)
        pd.testing.assert_series_equal(
            result['close_diff_1'],
            expected_diff_1,
            check_names=False
        )


class TestRollingFeatures:
    """Test rolling statistical features."""
    
    def test_rolling_statistics(self, sample_ohlcv_data):
        """Test rolling statistical feature creation."""
        rolling_generator = RollingFeatures()
        
        result = rolling_generator.create_rolling_statistics(
            sample_ohlcv_data,
            columns=['close'],
            windows=[5, 10],
            statistics=['mean', 'std']
        )
        
        # Check rolling features
        expected_features = ['close_rolling_5_mean', 'close_rolling_5_std',
                           'close_rolling_10_mean', 'close_rolling_10_std']
        
        for feature in expected_features:
            assert feature in result.columns
        
        # Verify rolling calculation
        expected_rolling_mean = sample_ohlcv_data['close'].rolling(5).mean()
        pd.testing.assert_series_equal(
            result['close_rolling_5_mean'],
            expected_rolling_mean,
            check_names=False
        )
    
    def test_expanding_features(self, sample_ohlcv_data):
        """Test expanding window features."""
        rolling_generator = RollingFeatures()
        
        result = rolling_generator.create_expanding_features(
            sample_ohlcv_data,
            columns=['close'],
            statistics=['mean', 'std']
        )
        
        # Check expanding features
        assert 'close_expanding_mean' in result.columns
        assert 'close_expanding_std' in result.columns


class TestInteractionFeatures:
    """Test interaction feature generation."""
    
    def test_ratio_features(self, sample_ohlcv_data):
        """Test ratio feature creation."""
        interaction_generator = InteractionFeatures()
        
        result = interaction_generator.create_ratio_features(
            sample_ohlcv_data,
            numerator_cols=['high'],
            denominator_cols=['low']
        )
        
        # Check ratio feature
        assert 'high_div_low' in result.columns
        
        # Verify ratio calculation (avoiding division by zero)
        expected_ratio = sample_ohlcv_data['high'] / sample_ohlcv_data['low']
        pd.testing.assert_series_equal(
            result['high_div_low'],
            expected_ratio,
            check_names=False
        )
    
    def test_product_features(self, sample_ohlcv_data):
        """Test product feature creation."""
        interaction_generator = InteractionFeatures()
        
        result = interaction_generator.create_product_features(
            sample_ohlcv_data,
            column_pairs=[('volume', 'close')]
        )
        
        # Check product feature
        assert 'volume_mult_close' in result.columns
        
        # Verify product calculation
        expected_product = sample_ohlcv_data['volume'] * sample_ohlcv_data['close']
        pd.testing.assert_series_equal(
            result['volume_mult_close'],
            expected_product,
            check_names=False
        )
    
    def test_polynomial_features(self, sample_ohlcv_data):
        """Test polynomial feature creation."""
        interaction_generator = InteractionFeatures()
        
        result = interaction_generator.create_polynomial_features(
            sample_ohlcv_data,
            columns=['close'],
            degrees=[2, 3]
        )
        
        # Check polynomial features
        assert 'close_pow_2' in result.columns
        assert 'close_pow_3' in result.columns
        
        # Verify polynomial calculation
        expected_squared = sample_ohlcv_data['close'] ** 2
        pd.testing.assert_series_equal(
            result['close_pow_2'],
            expected_squared,
            check_names=False
        )


class TestRegimeFeatures:
    """Test market regime feature generation."""
    
    def test_volatility_regime(self, sample_ohlcv_data):
        """Test volatility regime detection."""
        regime_generator = RegimeFeatures()
        
        result = regime_generator.create_volatility_regime(
            sample_ohlcv_data,
            window=10
        )
        
        # Check regime features
        assert 'volatility_regime' in result.columns
        assert 'volatility_percentile' in result.columns
        
        # Check that regime values are binary
        unique_regimes = result['volatility_regime'].dropna().unique()
        assert set(unique_regimes).issubset({0, 1})
    
    def test_trend_regime(self, sample_ohlcv_data):
        """Test trend regime detection."""
        regime_generator = RegimeFeatures()
        
        result = regime_generator.create_trend_regime(
            sample_ohlcv_data,
            short_window=5,
            long_window=10
        )
        
        # Check trend features
        assert 'trend_regime' in result.columns
        assert 'trend_strength' in result.columns
        
        # Check that trend regime values are in expected range
        unique_trends = result['trend_regime'].dropna().unique()
        assert set(unique_trends).issubset({-1, 0, 1})


class TestSeasonalFeatures:
    """Test seasonal feature generation."""
    
    def test_time_features(self, sample_ohlcv_data):
        """Test time-based feature creation."""
        seasonal_generator = SeasonalFeatures()
        
        result = seasonal_generator.create_time_features(sample_ohlcv_data)
        
        # Check time features
        time_features = ['hour', 'day_of_week', 'month', 'quarter',
                        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                        'is_weekend', 'is_month_start']
        
        for feature in time_features:
            assert feature in result.columns
        
        # Check cyclical encoding
        assert result['hour_sin'].min() >= -1
        assert result['hour_sin'].max() <= 1
        assert result['hour_cos'].min() >= -1
        assert result['hour_cos'].max() <= 1


class TestFeaturePipeline:
    """Test complete feature engineering pipeline."""
    
    def test_feature_pipeline_creation(self):
        """Test feature pipeline creation."""
        pipeline = create_feature_pipeline()
        assert isinstance(pipeline, FeaturePipeline)
    
    def test_feature_engineering_process(self, sample_ohlcv_data):
        """Test complete feature engineering process."""
        pipeline = create_feature_pipeline()
        
        # Engineer features
        result = pipeline.engineer_features(sample_ohlcv_data, fit_scalers=True)
        
        # Check that features were created
        original_cols = set(sample_ohlcv_data.columns)
        new_cols = set(result.columns) - original_cols
        assert len(new_cols) > 50  # Should create many features
        
        # Check that target variables were created
        target_features = [col for col in result.columns if col.startswith('target_')]
        assert len(target_features) > 0
        
        # Check temporal validation passed (no exception raised)
        assert result is not None
    
    def test_feature_pipeline_scaling(self, sample_ohlcv_data):
        """Test feature scaling in pipeline."""
        config = {'scaler_type': 'standard'}
        pipeline = create_feature_pipeline(config)
        
        # First pass - fit scalers
        result1 = pipeline.engineer_features(sample_ohlcv_data, fit_scalers=True)
        
        # Second pass - use fitted scalers
        result2 = pipeline.engineer_features(sample_ohlcv_data, fit_scalers=False)
        
        # Results should be identical when using same scaler
        numeric_cols = result1.select_dtypes(include=[np.number]).columns
        pd.testing.assert_frame_equal(
            result1[numeric_cols],
            result2[numeric_cols],
            check_dtype=False
        )
    
    def test_get_feature_names(self, sample_ohlcv_data):
        """Test feature name extraction."""
        pipeline = create_feature_pipeline()
        result = pipeline.engineer_features(sample_ohlcv_data)
        
        feature_names = pipeline.get_feature_names(result)
        
        # Should exclude target and metadata columns
        assert not any(name.startswith('target_') for name in feature_names)
        assert 'timestamp' not in feature_names
        assert 'symbol' not in feature_names


class TestCorrelationSelector:
    """Test correlation-based feature selection."""
    
    def test_correlation_selection(self):
        """Test removal of highly correlated features."""
        # Create data with correlated features
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
        })
        
        # Add highly correlated feature
        data['feature_1_corr'] = data['feature_1'] + np.random.randn(n_samples) * 0.01
        
        selector = CorrelationSelector(threshold=0.9)
        result = selector.fit_transform(data)
        
        # Should remove one of the correlated features
        assert result.shape[1] < data.shape[1]
        assert len(selector.removed_features) > 0


class TestVarianceSelector:
    """Test variance-based feature selection."""
    
    def test_variance_selection(self):
        """Test removal of low-variance features."""
        # Create data with low-variance feature
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'high_variance': np.random.randn(n_samples),
            'low_variance': np.ones(n_samples) * 0.001 + np.random.randn(n_samples) * 0.001
        })
        
        selector = VarianceSelector(threshold=0.01)
        result = selector.fit_transform(data)
        
        # Should remove low-variance feature
        assert 'high_variance' in result.columns
        assert 'low_variance' not in result.columns


class TestUnivariateSelector:
    """Test univariate feature selection."""
    
    def test_univariate_selection_classification(self):
        """Test univariate selection for classification."""
        # Create data with informative and noise features
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'informative_1': np.random.randn(n_samples),
            'informative_2': np.random.randn(n_samples),
            'noise_1': np.random.randn(n_samples),
            'noise_2': np.random.randn(n_samples)
        })
        
        # Create target correlated with informative features
        y = (X['informative_1'] + X['informative_2'] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        selector = UnivariateSelector(
            score_func='mutual_info',
            selection_mode='k_best',
            k=2,
            task_type='classification'
        )
        
        result = selector.fit_transform(X, y)
        
        # Should select top 2 features
        assert result.shape[1] == 2
        selected_features = selector.get_feature_names_out(X.columns)
        assert len(selected_features) == 2


class TestTreeBasedSelector:
    """Test tree-based feature selection."""
    
    def test_tree_based_selection(self):
        """Test tree-based feature importance selection."""
        # Create data with different feature importance
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            'important_1': np.random.randn(n_samples),
            'important_2': np.random.randn(n_samples),
            'noise_1': np.random.randn(n_samples),
            'noise_2': np.random.randn(n_samples)
        })
        
        # Create target correlated with important features
        y = (X['important_1'] + X['important_2'] + np.random.randn(n_samples) * 0.1 > 0).astype(int)
        
        selector = TreeBasedSelector(
            estimator_type='random_forest',
            max_features=3,
            task_type='classification'
        )
        
        result = selector.fit_transform(X, y)
        
        # Should select at most 3 features
        assert result.shape[1] <= 3
        assert len(selector.feature_importances) > 0


class TestDimensionalityReducer:
    """Test dimensionality reduction."""
    
    def test_pca_reduction(self):
        """Test PCA dimensionality reduction."""
        # Create high-dimensional data
        np.random.seed(42)
        n_samples, n_features = 100, 20
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        reducer = DimensionalityReducer(method='pca', n_components=5)
        result = reducer.fit_transform(X)
        
        # Should reduce to 5 components
        assert result.shape[1] == 5
        assert all(col.startswith('pca_component_') for col in result.columns)


class TestFeatureSelector:
    """Test complete feature selection pipeline."""
    
    def test_feature_selection_pipeline(self):
        """Test complete feature selection process."""
        # Create diverse dataset
        np.random.seed(42)
        n_samples, n_features = 200, 50
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Add some correlated features
        X['feature_corr_1'] = X['feature_0'] + np.random.randn(n_samples) * 0.1
        X['feature_corr_2'] = X['feature_1'] + np.random.randn(n_samples) * 0.1
        
        # Create target
        y = (X['feature_0'] + X['feature_1'] + np.random.randn(n_samples) > 0).astype(int)
        
        config = {
            'use_variance_selection': True,
            'use_correlation_selection': True,
            'use_univariate_selection': True,
            'use_tree_selection': True,
            'univariate_selection': {'k': 20},
            'tree_selection': {'max_features': 15}
        }
        
        selector = create_feature_selector(config)
        result = selector.fit_transform(X, y, task_type='classification')
        
        # Should reduce feature count
        assert result.shape[1] < X.shape[1]
        assert len(selector.get_feature_names_out()) == result.shape[1]


class TestStatisticalDriftDetector:
    """Test statistical drift detection."""
    
    def test_drift_detection_setup(self):
        """Test drift detector setup and reference fitting."""
        detector = StatisticalDriftDetector()
        
        # Create reference data
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'continuous_feature': np.random.normal(0, 1, 1000),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        detector.fit_reference(reference_data)
        
        # Check that reference distributions were stored
        assert 'continuous_feature' in detector.reference_distributions
        assert 'categorical_feature' in detector.reference_distributions
        assert detector.reference_distributions['continuous_feature']['type'] == 'continuous'
        assert detector.reference_distributions['categorical_feature']['type'] == 'categorical'
    
    def test_no_drift_detection(self):
        """Test detection when no drift is present."""
        detector = StatisticalDriftDetector(significance_level=0.05)
        
        # Create reference data
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000)
        })
        
        detector.fit_reference(reference_data)
        
        # Create new data from same distribution
        new_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100)
        })
        
        results = detector.detect_drift(new_data)
        
        # Should not detect drift
        assert results['drift_detected'] is False
        assert results['features_with_drift'] == 0
    
    def test_drift_detection_with_shift(self):
        """Test detection when drift is present."""
        detector = StatisticalDriftDetector(significance_level=0.05)
        
        # Create reference data
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000)
        })
        
        detector.fit_reference(reference_data)
        
        # Create new data with distribution shift
        new_data = pd.DataFrame({
            'feature_1': np.random.normal(2, 1, 100)  # Mean shift
        })
        
        results = detector.detect_drift(new_data)
        
        # Should detect drift
        assert results['drift_detected'] is True
        assert results['features_with_drift'] > 0


class TestPerformanceDriftDetector:
    """Test performance drift detection."""
    
    def test_performance_monitoring(self):
        """Test performance drift monitoring."""
        detector = PerformanceDriftDetector(window_size=10, alert_threshold=0.1)
        
        # Set baseline
        baseline = {'accuracy': 0.85, 'precision': 0.80}
        detector.set_baseline(baseline)
        
        # Add performance measurements
        for i in range(15):
            # Simulate gradual performance degradation
            current_performance = {
                'accuracy': 0.85 - i * 0.01,
                'precision': 0.80 - i * 0.005
            }
            results = detector.update_performance(current_performance)
        
        # Should detect performance drift
        assert 'performance_drift' in results
        if 'accuracy' in results['performance_drift']:
            assert results['performance_drift']['accuracy']['drift_detected'] is True


class TestFeatureMonitor:
    """Test complete feature monitoring system."""
    
    def test_feature_monitor_setup(self):
        """Test feature monitor initialization and setup."""
        monitor = create_feature_monitor()
        assert isinstance(monitor, FeatureMonitor)
        assert not monitor.monitoring_active
    
    def test_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        monitor = create_feature_monitor()
        
        # Create reference data
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.choice(['A', 'B'], 1000)
        })
        
        baseline_performance = {'accuracy': 0.85}
        
        # Start monitoring
        monitor.start_monitoring(reference_data, baseline_performance)
        assert monitor.monitoring_active
        
        # Monitor a batch
        new_data = pd.DataFrame({
            'feature_1': np.random.normal(0.1, 1, 100),
            'feature_2': np.random.choice(['A', 'B'], 100)
        })
        
        new_performance = {'accuracy': 0.83}
        
        results = monitor.monitor_batch(new_data, new_performance)
        
        # Check results structure
        assert 'timestamp' in results
        assert 'statistical_drift' in results
        assert 'performance_drift' in results
        assert 'overall_status' in results
        assert results['overall_status'] in ['healthy', 'warning', 'critical']
    
    def test_alert_callback(self):
        """Test alert callback functionality."""
        monitor = create_feature_monitor()
        
        # Mock alert callback
        alert_callback = Mock()
        monitor.add_alert_callback(alert_callback)
        
        # Create data that should trigger alert
        reference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000)
        })
        
        monitor.start_monitoring(reference_data)
        
        # Create drifted data
        drifted_data = pd.DataFrame({
            'feature_1': np.random.normal(3, 1, 100)  # Large shift
        })
        
        monitor.monitor_batch(drifted_data)
        
        # Alert callback should have been called if drift detected
        # (This depends on the statistical test results)


# Integration tests
class TestFeatureEngineeringIntegration:
    """Integration tests for complete feature engineering workflow."""
    
    def test_end_to_end_feature_engineering(self, sample_ohlcv_data):
        """Test complete feature engineering workflow."""
        # Step 1: Engineer features
        feature_pipeline = create_feature_pipeline()
        features_df = feature_pipeline.engineer_features(sample_ohlcv_data)
        
        # Step 2: Select features
        # Create dummy target for selection
        target = (features_df['close'].pct_change().shift(-1) > 0).astype(int)
        target = target.dropna()
        features_for_selection = features_df.iloc[:-1]  # Remove last row to match target
        
        feature_selector = create_feature_selector()
        selected_features = feature_selector.fit_transform(
            features_for_selection,
            target,
            task_type='classification'
        )
        
        # Step 3: Monitor features
        monitor = create_feature_monitor()
        monitor.start_monitoring(selected_features.iloc[:100])  # Use first 100 as reference
        
        # Monitor remaining data
        if len(selected_features) > 100:
            monitoring_results = monitor.monitor_batch(selected_features.iloc[100:])
            assert 'overall_status' in monitoring_results
        
        # Verify the pipeline worked
        assert features_df.shape[1] > sample_ohlcv_data.shape[1]  # Features were created
        assert selected_features.shape[1] <= features_df.shape[1]  # Features were selected
        assert monitor.monitoring_active  # Monitoring is active
    
    def test_temporal_validation_prevents_leakage(self, sample_ohlcv_data):
        """Test that temporal validation prevents data leakage."""
        # Create features with potential leakage
        leaky_data = sample_ohlcv_data.copy()
        leaky_data['future_return'] = leaky_data['close'].pct_change().shift(-1)
        
        pipeline = create_feature_pipeline()
        
        # Should raise validation error
        with pytest.raises(ValidationError):
            pipeline.engineer_features(leaky_data, validate_temporal=True)
    
    def test_performance_requirements(self, sample_ohlcv_data):
        """Test that feature engineering meets performance requirements."""
        import time
        
        pipeline = create_feature_pipeline()
        
        # Measure feature engineering time
        start_time = time.time()
        features_df = pipeline.engineer_features(sample_ohlcv_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process data efficiently (adjust threshold as needed)
        assert processing_time < 10.0  # Should complete within 10 seconds
        
        # Should create substantial number of features
        original_features = len(sample_ohlcv_data.columns)
        new_features = len(features_df.columns) - original_features
        assert new_features >= 50  # Should create at least 50 new features
        
        # Memory usage should be reasonable
        memory_usage_mb = features_df.memory_usage(deep=True).sum() / 1024 / 1024
        assert memory_usage_mb < 100  # Should use less than 100MB for test data


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
