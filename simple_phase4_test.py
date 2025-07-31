"""
Simplified Phase 4 Model Training validation test.
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_phase4_core():
    """Test Phase 4 core model training functionality."""
    print("\n🤖 Testing Phase 4: Model Training Core...")
    
    try:
        # Test basic model training without complex dependencies
        from src.models import create_model_trainer
        
        # Generate simple test data
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.uniform(0, 1, n_samples)
        })
        
        y = (X['feature_1'] + X['feature_2'] > 0).astype(int)
        
        print(f"  📊 Generated test data: {len(X)} samples, {len(X.columns)} features")
        
        # Test model trainer creation
        trainer = create_model_trainer(task_type='classification', model_type='random_forest')
        assert trainer is not None, "Model trainer should be created"
        print("  ✅ Model trainer created successfully")
        
        # Test model training
        model, metrics = trainer.train_model(X, y, "test_model")
        assert model is not None, "Model should be trained"
        assert metrics is not None, "Metrics should be calculated"
        print(f"  ✅ Model trained successfully")
        
        # Test predictions
        predictions = trainer.predict("test_model", X.head(10))
        assert len(predictions) == 10, "Should predict for all samples"
        print("  ✅ Model predictions working")
        
        # Test model metrics
        model_metrics = trainer.get_model_metrics("test_model")
        assert model_metrics is not None, "Should return model metrics"
        print("  ✅ Model metrics working")
        
        print("  🎉 Phase 4 core functionality validated!")
        return True
        
    except Exception as e:
        print(f"  ❌ Phase 4 core test failed: {e}")
        return False

def main():
    """Run simplified Phase 4 validation."""
    print("🤖 Phase 4: Model Training - Simplified Validation")
    print("=" * 60)
    
    success = test_phase4_core()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 PHASE 4 CORE VALIDATION: PASSED")
        print("✅ Model training system functional")
        print("✅ Basic model training working")
        print("✅ Model predictions working")
        print("✅ Model metrics calculation working")
        print("\n📋 PHASE 4 QUALITY GATE: PASSED")
        print("🚀 Ready to proceed to Phase 5: Prediction System")
    else:
        print("❌ PHASE 4 CORE VALIDATION: FAILED")
        print("❌ Phase 4 Quality Gate: FAILED")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
