"""
Quality Gate Test for Phase 5: Prediction System.

This script tests:
1. Real-time prediction engine with <100ms latency
2. Model serving and monitoring
3. A/B testing framework
"""

import os
import sys
import time
import uuid
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import warnings
warnings.filterwarnings('ignore')

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

# Test fixtures
from tests.test_fixtures import (
    sample_data_fixture, 
    sample_model_fixture, 
    sample_prediction_fixture
)


@pytest.mark.phase5
def test_predictor_initialization():
    """Test that the predictor initializes properly."""
    predictor = create_predictor()
    assert predictor is not None
    assert isinstance(predictor, Predictor)
    
    # Check default properties
    assert predictor.default_model is not None
    assert predictor.fallback_strategy in ["default_prediction", "default_model", None]
    
    # Check subcomponents initialized
    assert predictor.prediction_engine is not None
    assert predictor.model_server is not None


@pytest.mark.phase5
def test_prediction_latency(sample_data_fixture):
    """Test that predictions meet latency requirements (<100ms)."""
    # Create predictor
    predictor = create_predictor()
    predictor.start()
    
    try:
        # Register a sample model
        model = sample_model_fixture()
        predictor.prediction_engine.register_model("test_model", model)
        
        # Get sample data
        features = sample_data_fixture()
        
        # Warmup prediction
        _ = predictor.predict(features, model_name="test_model")
        
        # Measure latency of multiple predictions
        latencies = []
        for _ in range(10):
            start_time = time.time()
            result = predictor.predict(features, model_name="test_model")
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Ensure result contains expected fields
            assert "predictions" in result
            assert "model_name" in result
            assert "processing_time_ms" in result
        
        # Calculate average and maximum latency
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        # Maximum latency requirement: < 200ms (more lenient for test environment)
        # In production this would be < 100ms
        assert max_latency < 200, f"Maximum latency {max_latency}ms exceeds 200ms threshold"
        
        # Average latency requirement: < 100ms
        assert avg_latency < 100, f"Average latency {avg_latency}ms exceeds 100ms threshold"
        
    finally:
        predictor.stop()


@pytest.mark.phase5
def test_prediction_monitoring(sample_data_fixture):
    """Test that prediction monitoring is working."""
    # Create prediction engine with monitoring enabled
    prediction_engine = create_prediction_engine(enable_monitoring=True)
    prediction_engine.start()
    
    try:
        # Register a sample model
        model = sample_model_fixture()
        prediction_engine.register_model("test_model", model)
        
        # Get sample data
        features = sample_data_fixture()
        
        # Make multiple predictions
        for i in range(20):
            request = PredictionRequest(
                request_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                features=features,
                model_name="test_model",
                metadata={"test_run": True}
            )
            response = prediction_engine.predict(request)
            assert response.predictions is not None
        
        # Get monitoring metrics
        status = prediction_engine.get_status()
        assert "monitoring_metrics" in status
        
        metrics = status["monitoring_metrics"]
        
        # Check that monitoring is collecting data
        assert "total_predictions" in metrics
        assert metrics["total_predictions"] >= 20
        
        assert "avg_latency_ms" in metrics
        assert metrics["avg_latency_ms"] > 0
        
        # Check drift detection (if enough predictions)
        if metrics["total_predictions"] >= prediction_engine.config.drift_detection_window:
            assert "drift_detected" in metrics
            
    finally:
        prediction_engine.stop()


@pytest.mark.phase5
def test_model_serving(sample_data_fixture):
    """Test that model serving works correctly."""
    # Create model server
    model_server = create_model_server(
        max_concurrent_requests=10,
        request_timeout_seconds=5.0,
        health_check_interval=10
    )
    model_server.start()
    
    try:
        # Get sample data
        features = sample_data_fixture()
        
        # Create a simple model
        model = sample_model_fixture()
        
        # Deploy the model
        model_name = "test_model"
        version = "1.0.0"
        model_path = "/tmp/test_model.pkl"  # Mock path
        
        # Mock metadata
        from src.model_serving import ModelMetadata
        metadata = ModelMetadata(
            name=model_name,
            version=version,
            created_at=datetime.now(),
            model_type="regression",
            framework="scikit-learn",
            input_schema={"features": ["feature1", "feature2"]},
            output_schema={"predictions": "float"},
            description="Test model for quality gate"
        )
        
        # Deploy the model
        model_server.deployed_models[f"{model_name}:{version}"] = {
            'model': model,
            'metadata': metadata,
            'status': 'active',
            'deployed_at': datetime.now(),
            'request_count': 0,
            'error_count': 0
        }
        
        # Make a prediction using the model server
        prediction = model_server.predict(model_name, features)
        
        # Verify prediction
        assert prediction is not None
        assert hasattr(prediction, "predictions")
        assert len(prediction.predictions) == len(features)
        
        # Check model status
        status = model_server.get_model_status(model_name)
        assert status is not None
        
        # Check metrics
        metrics = model_server.get_metrics()
        assert metrics is not None
        assert "total_requests" in metrics
        assert metrics["total_requests"] >= 1
        
    finally:
        model_server.stop()


@pytest.mark.phase5
def test_ab_testing():
    """Test that A/B testing framework works."""
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
    assert test is not None
    
    # Check consistent variant assignment
    user_id = "test_user_1"
    variant1 = test.assign_variant(user_id)
    variant2 = test.assign_variant(user_id)
    assert variant1.name == variant2.name  # Same user should get same variant
    
    # Record some mock predictions
    for i in range(20):
        user_id = f"user_{i}"
        variant = test.assign_variant(user_id)
        
        # Create mock response
        response = PredictionResponse(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            predictions=np.array([float(i % 2)]),  # Alternating predictions
            probabilities=np.array([[0.2, 0.8]]) if i % 2 else np.array([[0.7, 0.3]]),
            processing_time_ms=float(50 + i),  # Varying latency
            model_name=variant.model_name,
            metadata={"impact_score": 0.5 + (i / 100)}
        )
        
        # Record with different actual values to create a difference
        actual = float(i % 2) if variant.name == "variant_0" else float((i + 1) % 2)
        test.record_prediction(variant.name, response, actual)
    
    # Analyze results
    test.analyze_results()
    result = test.get_result()
    
    # Verify result structure
    assert result.metrics is not None
    assert len(result.metrics) > 0
    assert "accuracy" in result.metrics
    assert "latency_ms" in result.metrics
    
    # Check sample counts
    for variant_name, count in result.sample_counts.items():
        assert count > 0
        
    # Verify we can get a list of tests
    test_list = ab_manager.list_tests()
    assert test_config.name in test_list
    
    # Stop test
    result = ab_manager.stop_test(test_config.name)
    assert result is not None
    assert result.status == "completed"


@pytest.mark.phase5
def test_prediction_fallback():
    """Test that prediction fallbacks work correctly."""
    # Create predictor with fallback strategy
    config = {"prediction": {"fallback_strategy": "default_prediction"}}
    predictor = create_predictor(config)
    predictor.start()
    
    try:
        # Try to predict with non-existent model
        features = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [4.0, 5.0, 6.0]
        })
        
        result = predictor.predict(features, model_name="non_existent_model")
        
        # Verify fallback result
        assert result is not None
        assert "predictions" in result
        assert "error" in result
        assert "is_fallback" in result
        assert result["is_fallback"] is True
        
    finally:
        predictor.stop()


@pytest.mark.phase5
def test_prediction_batch():
    """Test batch prediction capabilities."""
    # Create predictor
    predictor = create_predictor()
    predictor.start()
    
    try:
        # Register a sample model
        model = sample_model_fixture()
        predictor.prediction_engine.register_model("test_model", model)
        
        # Create multiple feature sets
        features_list = [
            pd.DataFrame({
                "feature1": [float(i), float(i+1), float(i+2)],
                "feature2": [float(i*2), float(i*2+1), float(i*2+2)]
            })
            for i in range(5)
        ]
        
        # Make batch prediction
        results = predictor.predict_batch(
            features_list=features_list,
            model_name="test_model"
        )
        
        # Verify results
        assert results is not None
        assert len(results) == len(features_list)
        
        for result in results:
            assert "predictions" in result
            assert "model_name" in result
            assert "processing_time_ms" in result
        
    finally:
        predictor.stop()


@pytest.mark.phase5
def test_end_to_end_prediction_system(sample_data_fixture):
    """End-to-end test of the prediction system."""
    # Create predictor
    predictor = create_predictor()
    predictor.start()
    
    # Create A/B testing manager
    ab_manager = create_ab_testing_manager()
    
    try:
        # 1. Register sample model
        model = sample_model_fixture()
        model_name = "production_model"
        predictor.prediction_engine.register_model(model_name, model)
        
        # 2. Create A/B test
        test_config = create_ab_test(
            name="production_test",
            models=[model_name, model_name],  # Same model, different versions in practice
            min_sample_size=5
        )
        ab_test = ab_manager.create_test(test_config)
        
        # 3. Get sample data
        features = sample_data_fixture()
        
        # 4. Make predictions with different user IDs
        results = []
        for i in range(10):
            user_id = f"user_{i}"
            variant = ab_test.assign_variant(user_id)
            
            # Make prediction
            result = predictor.predict(
                features=features,
                model_name=variant.model_name,
                request_id=user_id,
                metadata={"variant": variant.name, "test_name": test_config.name}
            )
            
            # Record in A/B test
            response = PredictionResponse(
                request_id=user_id,
                timestamp=datetime.fromisoformat(result["timestamp"]),
                predictions=np.array(result["predictions"]),
                processing_time_ms=result["processing_time_ms"],
                model_name=result["model_name"],
                metadata={"variant": variant.name}
            )
            
            # Mock actual value for testing
            actual_value = 1.0 if i % 2 == 0 else 0.0
            ab_test.record_prediction(variant.name, response, actual_value)
            
            results.append(result)
        
        # 5. Check predictor status
        status = predictor.get_status()
        assert status is not None
        assert "prediction_engine" in status
        assert "model_server" in status
        
        # 6. Check A/B test results
        test_result = ab_test.get_result()
        assert test_result is not None
        assert "accuracy" in test_result.metrics
        
        # 7. Check prediction results
        assert len(results) == 10
        for result in results:
            assert "predictions" in result
            assert "model_name" in result
            assert "processing_time_ms" in result
            assert result["processing_time_ms"] < 200  # Reasonable latency for tests
        
        # 8. Stop A/B test
        ab_manager.stop_test(test_config.name)
        
    finally:
        predictor.stop()


if __name__ == "__main__":
    print("Running Phase 5 Quality Gate Tests")
    
    # Indicator for test start
    print("-" * 80)
    print("Testing Predictor Initialization")
    test_predictor_initialization()
    print("✓ Predictor initialization test passed")
    
    print("\nTesting Prediction Latency")
    data = sample_data_fixture()
    test_prediction_latency(data)
    print("✓ Prediction latency test passed")
    
    print("\nTesting Prediction Monitoring")
    test_prediction_monitoring(data)
    print("✓ Prediction monitoring test passed")
    
    print("\nTesting Model Serving")
    test_model_serving(data)
    print("✓ Model serving test passed")
    
    print("\nTesting A/B Testing Framework")
    test_ab_testing()
    print("✓ A/B testing framework test passed")
    
    print("\nTesting Prediction Fallback")
    test_prediction_fallback()
    print("✓ Prediction fallback test passed")
    
    print("\nTesting Batch Prediction")
    test_prediction_batch()
    print("✓ Batch prediction test passed")
    
    print("\nRunning End-to-End Prediction System Test")
    test_end_to_end_prediction_system(data)
    print("✓ End-to-end prediction system test passed")
    
    # Final result
    print("-" * 80)
    print("✓✓✓ ALL PHASE 5 QUALITY GATE TESTS PASSED ✓✓✓")
    print("-" * 80)
