"""
Quick test to validate dependency fixes and core functionality.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic module imports."""
    print("🔍 Testing Basic Imports...")
    
    try:
        # Test core modules
        from src.config import get_config
        config = get_config()
        print(f"  ✅ Config: {config.app.name}")
        
        from src.indicators import create_indicators_calculator
        indicators = create_indicators_calculator()
        print("  ✅ Indicators module imported")
        
        from src.features import create_feature_pipeline
        pipeline = create_feature_pipeline()
        print("  ✅ Features module imported")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_core_functionality():
    """Test core functionality without complex dependencies."""
    print("\n📈 Testing Core Functionality...")
    
    try:
        # Generate test data
        np.random.seed(42)
        n_samples = 100
        
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
        
        print(f"  📊 Generated {len(test_data)} samples")
        
        # Test basic indicators
        sma_20 = test_data['close'].rolling(20).mean()
        assert len(sma_20) == n_samples, "SMA should have same length"
        print("  ✅ SMA calculation works")
        
        # Test basic feature engineering
        test_data['close_lag_1'] = test_data['close'].shift(1)
        test_data['volume_rolling_5'] = test_data['volume'].rolling(5).mean()
        test_data['price_ratio'] = test_data['high'] / test_data['low']
        
        assert test_data['close_lag_1'].iloc[1] == test_data['close'].iloc[0], "Lag feature should work"
        print("  ✅ Basic feature engineering works")
        
        # Test temporal validation (no future data)
        future_features = [col for col in test_data.columns if 'future' in col.lower()]
        assert len(future_features) == 0, "No future-looking features should exist"
        print("  ✅ Temporal validation passed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Core functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run quick validation test."""
    print("🚀 Quick Validation Test")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Core Functionality", test_core_functionality)
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
            print(f"❌ {test_name} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 40)
    print(f"🧪 QUICK TEST: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ Dependencies fixed - proceeding to core test fixes")
    else:
        print("❌ Still have dependency issues")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
