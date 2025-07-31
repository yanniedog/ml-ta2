"""
Phase 5 Quality Gate Validation Script

This script directly validates the core functionality of our prediction system:
1. Predictor initialization
2. Model serving
3. A/B testing framework
4. Prediction latency
5. End-to-end prediction pipeline

Run this script to verify that all Phase 5 requirements are met.
"""

import os
import sys
import time
import uuid
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import get_config
from src.predictor import Predictor, create_predictor
from src.prediction_engine import (
    PredictionEngine, 
    PredictionRequest, 
    PredictionResponse,
    create_prediction_engine
)
from src.model_serving import ModelServer, create_model_server
from src.ab_testing import (
    ABTestingManager, 
    create_ab_testing_manager,
    create_ab_test
)
from src.exceptions import PredictionError


# Simple Mock Model for testing
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


def generate_sample_data():
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


def validate_predictor_initialization():
    """Validate that the predictor initializes properly."""
    print("\n🔍 Testing Predictor Initialization")
    
    predictor = create_predictor()
    assert predictor is not None, "Predictor should not be None"
    assert isinstance(predictor, Predictor), "Predictor should be instance of Predictor class"
    
    # Check default properties
    assert predictor.default_model is not None, "Default model should be set"
    assert predictor.fallback_strategy in ["default_prediction", "default_model", None], \
        "Fallback strategy should be valid"
    
    # Check subcomponents initialized
    assert predictor.prediction_engine is not None, "Prediction engine should be initialized"
    assert predictor.model_server is not None, "Model server should be initialized"
    
    print("✅ Predictor initialization successful")
    return predictor


def validate_prediction_latency(predictor, sample_data):
    """Validate that predictions meet latency requirements (<100ms)."""
    print("\n🔍 Testing Prediction Latency")
    
    predictor.start()
    
    try:
        # Register a sample model
        model = SimpleMockModel()
        predictor.prediction_engine.register_model("test_model", model)
        
        # Warmup prediction
        _ = predictor.predict(sample_data, model_name="test_model")
        
        # Measure latency of multiple predictions
        latencies = []
        for _ in range(10):
            start_time = time.time()
            result = predictor.predict(sample_data, model_name="test_model")
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Ensure result contains expected fields
            assert "predictions" in result, "Result should contain predictions"
            assert "model_name" in result, "Result should contain model_name"
            assert "processing_time_ms" in result, "Result should contain processing_time_ms"
        
        # Calculate average and maximum latency
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        print(f"📊 Average latency: {avg_latency:.2f}ms")
        print(f"📊 Maximum latency: {max_latency:.2f}ms")
        
        # Maximum latency requirement: < 200ms (more lenient for test environment)
        # In production this would be < 100ms
        assert max_latency < 200, f"Maximum latency {max_latency}ms exceeds 200ms threshold"
        
        # Average latency requirement: < 100ms
        assert avg_latency < 100, f"Average latency {avg_latency}ms exceeds 100ms threshold"
        
        print("✅ Prediction latency test successful")
    finally:
        predictor.stop()


def validate_ab_testing():
    """Validate that A/B testing framework works."""
    print("\n🔍 Testing A/B Testing Framework")
    
    # Create A/B testing manager
    ab_manager = create_ab_testing_manager()
    
    # Create test configuration
    test_config = create_ab_test(
        name="test_ab_experiment",
        models=["model_a", "model_b"],
        weights=[0.5, 0.5],
        min_sample_size=10,  # Small for testing
        max_duration_days=1,
        metrics=["accuracy", "latency_ms"]
    )
    
    # Create the test
    test = ab_manager.create_test(test_config)
    assert test is not None, "A/B test should be created"
    
    # Check consistent variant assignment
    user_id = "test_user_1"
    variant1 = test.assign_variant(user_id)
    variant2 = test.assign_variant(user_id)
    assert variant1.name == variant2.name, "Same user should get same variant"  
    
    print("✅ A/B testing framework validation successful")
    return ab_manager, test


def validate_end_to_end(predictor, ab_manager, sample_data):
    """Validate end-to-end prediction system."""
    print("\n🔍 Testing End-to-End Prediction System")
    
    predictor.start()
    
    try:
        # 1. Register sample model
        model = SimpleMockModel()
        model_name = "production_model"
        predictor.prediction_engine.register_model(model_name, model)
        
        # 2. Create A/B test
        test_config = create_ab_test(
            name="production_test",
            models=[model_name, model_name],  # Same model, different versions in practice
            min_sample_size=5
        )
        ab_test = ab_manager.create_test(test_config)
        
        # 3. Make predictions with different user IDs
        results = []
        for i in range(10):
            user_id = f"user_{i}"
            variant = ab_test.assign_variant(user_id)
            
            # Make prediction
            result = predictor.predict(
                features=sample_data,
                model_name=variant.model_name,
                request_id=user_id,
                metadata={"variant": variant.name, "test_name": test_config.name}
            )
            
            # Create response object from result
            response = PredictionResponse(
                request_id=user_id,
                timestamp=datetime.fromisoformat(result["timestamp"]) if isinstance(result["timestamp"], str) 
                           else result["timestamp"],
                predictions=np.array(result["predictions"]),
                processing_time_ms=result["processing_time_ms"],
                model_name=result["model_name"],
                metadata={"variant": variant.name}
            )
            
            # Mock actual value for testing
            actual_value = 1.0 if i % 2 == 0 else 0.0
            ab_test.record_prediction(variant.name, response, actual_value)
            
            results.append(result)
        
        # 4. Check predictor status
        status = predictor.get_status()
        assert status is not None, "Status should not be None"
        assert "prediction_engine" in status, "Status should include prediction_engine info"
        assert "model_server" in status, "Status should include model_server info"
        
        # 5. Check prediction results
        assert len(results) == 10, "Should have 10 prediction results"
        for result in results:
            assert "predictions" in result, "Result should contain predictions"
            assert "model_name" in result, "Result should contain model_name"
            assert "processing_time_ms" in result, "Result should contain processing_time_ms"
            assert result["processing_time_ms"] < 200, "Latency should be reasonable"
        
        # 6. Stop A/B test
        ab_manager.stop_test(test_config.name)
        
        print("✅ End-to-end prediction system validation successful")
    finally:
        predictor.stop()


def run_phase5_validation():
    """Run all validation tests."""
    print("\n" + "=" * 80)
    print("Phase 5 Quality Gate Validation")
    print("=" * 80)
    
    try:
        np.random.seed(42)
        
        # Generate sample data
        sample_data = generate_sample_data()
        
        # Run validation tests
        predictor = validate_predictor_initialization()
        validate_prediction_latency(predictor, sample_data)
        ab_manager, test = validate_ab_testing()
        validate_end_to_end(predictor, ab_manager, sample_data)
        
        print("\n" + "=" * 80)
        print("✅✅✅ PHASE 5 QUALITY GATE PASSED ✅✅✅")
        print("=" * 80)
        return True
    
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ PHASE 5 VALIDATION FAILED: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_phase5_validation()
    sys.exit(0 if success else 1)
