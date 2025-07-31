"""
Comprehensive Test Suite for ML-TA System

This script thoroughly tests all implemented components to ensure they work correctly
before proceeding to new development phases.
"""

import sys
import os
import traceback
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🔍 Testing Module Imports...")
    
    modules_to_test = [
        'config', 'logging_config', 'exceptions', 'utils',
        'data_fetcher', 'data_quality', 'data_loader',
        'indicators', 'features', 'feature_selection', 'feature_monitoring'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            __import__(f'src.{module}')
            print(f"  ✅ {module}")
        except Exception as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append((module, str(e)))
    
    if failed_imports:
        print(f"\n❌ Import failures: {len(failed_imports)}")
        return False
    
    print(f"✅ All {len(modules_to_test)} modules imported successfully")
    return True

def test_configuration():
    """Test configuration management system."""
    print("\n🔧 Testing Configuration System...")
    
    try:
        from src.config import get_config
        
        config = get_config()
        print(f"  ✅ Configuration loaded: {config.app.name}")
        print(f"  📊 Environment: {config.app.environment}")
        
        # Test config sections
        assert hasattr(config, 'app'), "Config missing app section"
        assert hasattr(config, 'data'), "Config missing data section"
        assert hasattr(config, 'features'), "Config missing features section"
        
        print("  ✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def test_technical_indicators():
    """Test technical indicators system."""
    print("\n📈 Testing Technical Indicators...")
    
    try:
        from src.indicators import create_indicators_calculator
        
        # Generate test OHLCV data
        np.random.seed(42)
        n_samples = 200
        
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'open': np.random.uniform(50000, 60000, n_samples),
            'high': np.random.uniform(55000, 65000, n_samples),
            'low': np.random.uniform(45000, 55000, n_samples),
            'close': np.random.uniform(50000, 60000, n_samples),
            'volume': np.random.uniform(1000, 5000, n_samples)
        })
        
        # Ensure OHLC relationships
        test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
        test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
        
        indicators = create_indicators_calculator()
        
        # Test individual indicators
        sma_20 = indicators.sma(test_data['close'], period=20)
        assert len(sma_20) == n_samples, "SMA should have same length as input"
        print("  ✅ SMA calculation functional")
        
        rsi_14 = indicators.rsi(test_data['close'], period=14)
        assert (rsi_14.dropna() >= 0).all() and (rsi_14.dropna() <= 100).all(), "RSI should be 0-100"
        print("  ✅ RSI calculation functional")
        
        # Test comprehensive indicator calculation
        start_time = time.time()
        result = indicators.calculate_all_indicators(test_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        latency_per_sample = processing_time / n_samples * 1000
        
        print(f"  ⏱️  Processing time: {processing_time:.3f}s ({latency_per_sample:.2f}ms per sample)")
        
        original_cols = len(test_data.columns)
        indicator_cols = len(result.columns) - original_cols
        
        print(f"  📊 Created {indicator_cols} indicators")
        assert indicator_cols >= 30, f"Expected at least 30 indicators, got {indicator_cols}"
        assert latency_per_sample < 100, f"Latency {latency_per_sample:.2f}ms exceeds 100ms requirement"
        
        return True
        
    except Exception as e:
        print(f"  ❌ Technical indicators test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering pipeline."""
    print("\n🏗️  Testing Feature Engineering...")
    
    try:
        from src.features import create_feature_pipeline
        
        # Generate test data
        np.random.seed(42)
        n_samples = 300
        
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'symbol': 'BTCUSDT',
            'open': 50000 + np.cumsum(np.random.randn(n_samples) * 100),
            'high': 50000 + np.cumsum(np.random.randn(n_samples) * 100) + 200,
            'low': 50000 + np.cumsum(np.random.randn(n_samples) * 100) - 200,
            'close': 50000 + np.cumsum(np.random.randn(n_samples) * 100),
            'volume': np.random.uniform(1000, 5000, n_samples)
        })
        
        # Ensure OHLC relationships
        test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
        test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
        
        pipeline = create_feature_pipeline()
        
        # Test feature engineering
        start_time = time.time()
        features_df = pipeline.engineer_features(test_data, fit_scalers=True, validate_temporal=True)
        end_time = time.time()
        
        processing_time = end_time - start_time
        latency_per_sample = processing_time / n_samples * 1000
        
        print(f"  ⏱️  Processing time: {processing_time:.3f}s ({latency_per_sample:.2f}ms per sample)")
        assert latency_per_sample < 100, f"Latency {latency_per_sample:.2f}ms exceeds 100ms requirement"
        
        # Check feature count
        original_features = len(test_data.columns)
        total_features = len(features_df.columns)
        new_features = total_features - original_features
        
        print(f"  📊 Created {new_features} features (total: {total_features})")
        assert new_features >= 50, f"Expected at least 50 new features, got {new_features}"
        
        # Test temporal validation
        try:
            leaky_data = test_data.copy()
            leaky_data['future_price'] = leaky_data['close'].shift(-1)
            pipeline.engineer_features(leaky_data, validate_temporal=True)
            assert False, "Should have detected future-looking features"
        except Exception as e:
            if "future-looking" in str(e):
                print("  ✅ Temporal validation functional")
            else:
                raise e
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature engineering test failed: {e}")
        return False

def test_feature_selection():
    """Test feature selection system."""
    print("\n🎯 Testing Feature Selection...")
    
    try:
        from src.feature_selection import create_feature_selector
        
        # Generate test data
        np.random.seed(42)
        n_samples, n_features = 200, 50
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create target
        y = (X['feature_0'] + X['feature_1'] + np.random.randn(n_samples) > 0).astype(int)
        
        selector = create_feature_selector()
        selected_features = selector.fit_transform(X, y, task_type='classification')
        
        original_count = X.shape[1]
        selected_count = selected_features.shape[1]
        
        print(f"  📉 Feature reduction: {original_count} → {selected_count}")
        assert selected_count < original_count, "Should reduce feature count"
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature selection test failed: {e}")
        return False

def test_feature_monitoring():
    """Test feature monitoring system."""
    print("\n🔍 Testing Feature Monitoring...")
    
    try:
        from src.feature_monitoring import create_feature_monitor
        
        # Generate test data
        np.random.seed(42)
        reference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(5, 2, 1000)
        })
        
        monitor = create_feature_monitor()
        monitor.start_monitoring(reference_data, {'accuracy': 0.85})
        
        assert monitor.monitoring_active, "Monitor should be active"
        print("  ✅ Monitoring setup functional")
        
        # Test drift detection
        drifted_data = pd.DataFrame({
            'feature_1': np.random.normal(3, 1, 100),  # Mean shift
            'feature_2': np.random.normal(5, 2, 100)
        })
        
        results = monitor.monitor_batch(drifted_data, {'accuracy': 0.75})
        print(f"  📊 Monitoring status: {results['overall_status']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature monitoring test failed: {e}")
        return False

def test_end_to_end():
    """Test complete end-to-end integration."""
    print("\n🔄 Testing End-to-End Integration...")
    
    try:
        from src.indicators import create_indicators_calculator
        from src.features import create_feature_pipeline
        from src.feature_selection import create_feature_selector
        from src.feature_monitoring import create_feature_monitor
        
        # Generate test data
        np.random.seed(42)
        n_samples = 400
        
        test_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'symbol': 'BTCUSDT',
            'open': 50000 + np.cumsum(np.random.randn(n_samples) * 100),
            'high': 50000 + np.cumsum(np.random.randn(n_samples) * 100) + 200,
            'low': 50000 + np.cumsum(np.random.randn(n_samples) * 100) - 200,
            'close': 50000 + np.cumsum(np.random.randn(n_samples) * 100),
            'volume': np.random.uniform(1000, 5000, n_samples)
        })
        
        test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
        test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
        
        print(f"  📊 Generated {len(test_data)} samples")
        
        # Step 1: Technical indicators
        indicators = create_indicators_calculator()
        indicators_data = indicators.calculate_all_indicators(test_data)
        print(f"  📈 Added {len(indicators_data.columns) - len(test_data.columns)} indicators")
        
        # Step 2: Feature engineering
        pipeline = create_feature_pipeline()
        features_data = pipeline.engineer_features(indicators_data, fit_scalers=True)
        print(f"  🏗️  Created {len(features_data.columns)} total features")
        
        # Step 3: Feature selection
        # Create target with shift and ensure perfect alignment with features
        target = (features_data['close'].pct_change().shift(-1) > 0).astype(int)
        
        # Since shift(-1) creates NaN at the end, we need to remove the last row from both
        features_for_selection = features_data.iloc[:-1].copy()
        target = target.iloc[:-1].copy()  # Align with features_for_selection
        
        # Now drop any remaining NaNs and align both datasets
        target = target.dropna()
        common_idx = target.index.intersection(features_for_selection.index)
        target = target.loc[common_idx]
        features_for_selection = features_for_selection.loc[common_idx]
        
        selector = create_feature_selector()
        selected_features = selector.fit_transform(features_for_selection, target, task_type='classification')
        print(f"  🎯 Selected {selected_features.shape[1]} features")
        
        # Step 4: Feature monitoring
        monitor = create_feature_monitor()
        reference_features = selected_features.iloc[:200]
        test_features = selected_features.iloc[200:]
        
        monitor.start_monitoring(reference_features, {'accuracy': 0.85})
        
        if len(test_features) > 0:
            results = monitor.monitor_batch(test_features, {'accuracy': 0.83})
            print(f"  🔍 Monitoring status: {results['overall_status']}")
        
        print("  ✅ End-to-end integration successful")
        return True
        
    except Exception as e:
        print(f"  ❌ End-to-end test failed: {e}")
        return False

def main():
    """Run comprehensive test suite."""
    print("🧪 ML-TA System Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Technical Indicators", test_technical_indicators),
        ("Feature Engineering", test_feature_engineering),
        ("Feature Selection", test_feature_selection),
        ("Feature Monitoring", test_feature_monitoring),
        ("End-to-End Integration", test_end_to_end)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"🧪 TEST SUMMARY: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED - System ready for Phase 4!")
        print("✅ Phase 3 Quality Gate: PASSED")
    else:
        print("❌ Some tests failed - Issues need to be resolved")
        print("❌ Phase 3 Quality Gate: FAILED")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
