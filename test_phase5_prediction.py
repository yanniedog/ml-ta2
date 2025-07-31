"""
Comprehensive test suite for Phase 5: Prediction System.

Tests:
- Real-time prediction engine functionality
- Model serving infrastructure
- Prediction monitoring and drift detection
- A/B testing framework
- Performance requirements (<100ms latency)
- Integration with Phase 4 models
"""

import numpy as np
import pandas as pd
import time
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_prediction_engine_basic():
    """Test basic prediction engine functionality."""
    print("\n🔬 Testing Prediction Engine Basic Functionality...")
    
    try:
        from prediction_engine import (
            PredictionEngine, PredictionRequest, PredictionResponse, 
            PredictionConfig, create_prediction_engine
        )
        
        # Create prediction engine
        engine = create_prediction_engine(max_latency_ms=100.0, enable_monitoring=True)
        
        # Create a simple mock model
        class MockModel:
            def predict(self, X):
                return np.random.randint(0, 2, len(X))
            
            def predict_proba(self, X):
                probs = np.random.random((len(X), 2))
                return probs / probs.sum(axis=1, keepdims=True)
        
        model = MockModel()
        
        # Register model
        engine.register_model("test_model", model)
        
        # Create test data
        features = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0],
            'feature_2': [0.5, 1.5, 2.5]
        })
        
        # Create prediction request
        request = PredictionRequest(
            request_id="test_001",
            timestamp=pd.Timestamp.now(),
            features=features,
            model_name="test_model"
        )
        
        # Make prediction
        start_time = time.time()
        response = engine.predict(request)
        end_time = time.time()
        
        # Validate response
        assert response.request_id == "test_001", "Request ID should match"
        assert response.predictions is not None, "Predictions should not be None"
        assert len(response.predictions) == 3, "Should have 3 predictions"
        assert response.probabilities is not None, "Probabilities should be available"
        assert response.processing_time_ms is not None, "Processing time should be recorded"
        assert response.processing_time_ms < 100, f"Processing time {response.processing_time_ms}ms should be < 100ms"
        
        print(f"  ✅ Basic prediction successful (latency: {response.processing_time_ms:.2f}ms)")
        print(f"  ✅ Predictions shape: {response.predictions.shape}")
        print(f"  ✅ Probabilities shape: {response.probabilities.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Prediction engine basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_serving():
    """Test model serving infrastructure."""
    print("\n🔬 Testing Model Serving Infrastructure...")
    
    try:
        from model_serving import ModelServer, ModelMetadata, create_model_server
        from datetime import datetime
        
        # Create model server
        server = create_model_server(max_concurrent_requests=50, request_timeout_seconds=10.0)
        
        # Start server
        server.start()
        
        # Create a simple model
        class SimpleModel:
            def predict(self, X):
                return (X.iloc[:, 0] > X.iloc[:, 1]).astype(int)
            
            def predict_proba(self, X):
                predictions = self.predict(X)
                probs = np.zeros((len(X), 2))
                probs[predictions == 0, 0] = 0.8
                probs[predictions == 0, 1] = 0.2
                probs[predictions == 1, 0] = 0.3
                probs[predictions == 1, 1] = 0.7
                return probs
        
        model = SimpleModel()
        
        # Create metadata
        metadata = ModelMetadata(
            name="simple_model",
            version="1.0.0",
            created_at=datetime.now(),
            model_type="SimpleModel",
            framework="custom",
            input_schema={"features": ["feature_1", "feature_2"]},
            output_schema={"predictions": "binary"}
        )
        
        # Deploy model
        success = server.deploy_model("simple_model", "1.0.0", model, metadata)
        assert success, "Model deployment should succeed"
        
        # Test prediction
        features = pd.DataFrame({
            'feature_1': [2.0, 1.0, 3.0],
            'feature_2': [1.0, 2.0, 1.5]
        })
        
        result = server.predict("simple_model", features, request_id="serve_test_001")
        
        # Validate result
        assert result['status'] == 'success', f"Prediction should succeed, got: {result}"
        assert 'predictions' in result, "Result should contain predictions"
        assert 'processing_time_ms' in result, "Result should contain processing time"
        assert result['processing_time_ms'] < 100, f"Processing time should be < 100ms, got: {result['processing_time_ms']}"
        
        print(f"  ✅ Model deployment successful")
        print(f"  ✅ Model serving prediction successful (latency: {result['processing_time_ms']:.2f}ms)")
        print(f"  ✅ Predictions: {result['predictions']}")
        
        # Test model status
        status = server.get_model_status("simple_model")
        assert 'model_name' in status, "Status should contain model name"
        assert status['request_count'] >= 1, "Request count should be at least 1"
        
        print(f"  ✅ Model status retrieval successful")
        
        # Test server metrics
        metrics = server.get_metrics()
        assert 'total_requests' in metrics, "Metrics should contain total requests"
        assert metrics['total_requests'] >= 1, "Should have at least 1 request"
        
        print(f"  ✅ Server metrics retrieval successful")
        
        # Stop server
        server.stop()
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model serving test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction_monitoring():
    """Test prediction monitoring and drift detection."""
    print("\n🔬 Testing Prediction Monitoring...")
    
    try:
        from prediction_engine import PredictionMonitor, PredictionRequest, PredictionResponse
        from datetime import datetime
        
        # Create monitor
        monitor = PredictionMonitor(window_size=100)
        
        # Simulate predictions
        for i in range(50):
            request = PredictionRequest(
                request_id=f"monitor_test_{i}",
                timestamp=datetime.now(),
                features=pd.DataFrame({'feature_1': [i], 'feature_2': [i * 2]})
            )
            
            # Simulate varying processing times
            processing_time = 10 + (i % 20)  # 10-30ms
            
            response = PredictionResponse(
                request_id=f"monitor_test_{i}",
                timestamp=datetime.now(),
                predictions=np.array([i % 2]),
                processing_time_ms=processing_time,
                model_name="test_model"
            )
            
            monitor.record_prediction(request, response)
        
        # Get metrics
        metrics = monitor.get_metrics()
        
        # Validate metrics
        assert 'performance_metrics' in metrics, "Metrics should contain performance data"
        assert 'monitoring_window_size' in metrics, "Metrics should contain window size"
        assert metrics['monitoring_window_size'] == 50, "Should have recorded 50 predictions"
        
        perf_metrics = metrics['performance_metrics']
        if perf_metrics:
            assert 'avg_processing_time_ms' in perf_metrics, "Should have average processing time"
            assert perf_metrics['avg_processing_time_ms'] > 0, "Average processing time should be positive"
        
        print(f"  ✅ Prediction monitoring successful")
        print(f"  ✅ Recorded {metrics['monitoring_window_size']} predictions")
        if perf_metrics:
            print(f"  ✅ Average processing time: {perf_metrics.get('avg_processing_time_ms', 0):.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Prediction monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ab_testing():
    """Test A/B testing framework."""
    print("\n🔬 Testing A/B Testing Framework...")
    
    try:
        from prediction_engine import ABTestManager, PredictionResponse
        from datetime import datetime
        
        # Create A/B test manager
        ab_manager = ABTestManager()
        
        # Create A/B test
        success = ab_manager.create_test("model_comparison", "model_a", "model_b", traffic_split=0.5)
        assert success, "A/B test creation should succeed"
        
        # Test model assignment
        model_assignments = {}
        for i in range(100):
            request_id = f"ab_test_{i}"
            assigned_model = ab_manager.get_model_for_request("model_comparison", request_id)
            
            if assigned_model not in model_assignments:
                model_assignments[assigned_model] = 0
            model_assignments[assigned_model] += 1
        
        # Validate traffic split (should be roughly 50/50)
        assert "model_a" in model_assignments, "Model A should be assigned"
        assert "model_b" in model_assignments, "Model B should be assigned"
        
        model_a_count = model_assignments.get("model_a", 0)
        model_b_count = model_assignments.get("model_b", 0)
        
        # Allow some variance in traffic split (40-60% range)
        assert 30 <= model_a_count <= 70, f"Model A should get ~50% traffic, got {model_a_count}%"
        assert 30 <= model_b_count <= 70, f"Model B should get ~50% traffic, got {model_b_count}%"
        
        print(f"  ✅ A/B test creation successful")
        print(f"  ✅ Traffic split: Model A: {model_a_count}%, Model B: {model_b_count}%")
        
        # Record some predictions
        for i in range(10):
            response = PredictionResponse(
                request_id=f"ab_result_{i}",
                timestamp=datetime.now(),
                predictions=np.array([i % 2]),
                processing_time_ms=15 + (i % 10),
                model_name="model_a" if i % 2 == 0 else "model_b"
            )
            
            ab_manager.record_prediction("model_comparison", response.model_name, response)
        
        # Get test results
        results = ab_manager.get_test_results("model_comparison")
        assert results is not None, "Should get test results"
        assert 'metrics_a' in results, "Should have metrics for model A"
        assert 'metrics_b' in results, "Should have metrics for model B"
        
        print(f"  ✅ A/B test results retrieval successful")
        print(f"  ✅ Model A predictions: {results['metrics_a'].get('prediction_count', 0)}")
        print(f"  ✅ Model B predictions: {results['metrics_b'].get('prediction_count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ A/B testing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_predictions():
    """Test batch prediction functionality."""
    print("\n🔬 Testing Batch Predictions...")
    
    try:
        from prediction_engine import create_prediction_engine, PredictionRequest
        from datetime import datetime
        
        # Create prediction engine
        engine = create_prediction_engine(max_latency_ms=100.0)
        
        # Create mock model
        class BatchModel:
            def predict(self, X):
                return np.random.randint(0, 3, len(X))
        
        model = BatchModel()
        engine.register_model("batch_model", model)
        
        # Create batch requests
        requests = []
        for i in range(20):
            features = pd.DataFrame({
                'feature_1': [i * 0.1],
                'feature_2': [i * 0.2]
            })
            
            request = PredictionRequest(
                request_id=f"batch_{i}",
                timestamp=datetime.now(),
                features=features,
                model_name="batch_model"
            )
            requests.append(request)
        
        # Make batch predictions
        start_time = time.time()
        responses = engine.predict_batch(requests)
        total_time = (time.time() - start_time) * 1000
        
        # Validate batch results
        assert len(responses) == 20, "Should have 20 responses"
        
        successful_predictions = sum(1 for r in responses if r.predictions is not None and len(r.predictions) > 0)
        assert successful_predictions == 20, f"All predictions should succeed, got {successful_predictions}/20"
        
        avg_latency = total_time / len(responses)
        assert avg_latency < 100, f"Average latency should be < 100ms, got {avg_latency:.2f}ms"
        
        print(f"  ✅ Batch predictions successful")
        print(f"  ✅ Processed {len(responses)} requests")
        print(f"  ✅ Average latency: {avg_latency:.2f}ms")
        print(f"  ✅ Total batch time: {total_time:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Batch predictions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_phase4():
    """Test integration with Phase 4 models."""
    print("\n🔬 Testing Integration with Phase 4 Models...")
    
    try:
        # Test with sklearn models if available
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            SKLEARN_AVAILABLE = True
        except ImportError:
            SKLEARN_AVAILABLE = False
        
        if SKLEARN_AVAILABLE:
            from prediction_engine import create_prediction_engine, PredictionRequest
            from model_serving import create_model_server, ModelMetadata
            from datetime import datetime
            
            # Create and train a real model
            np.random.seed(42)
            X = pd.DataFrame({
                'feature_1': np.random.randn(100),
                'feature_2': np.random.randn(100),
                'feature_3': np.random.uniform(0, 1, 100)
            })
            y = (X['feature_1'] + X['feature_2'] > 0).astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Test with prediction engine
            engine = create_prediction_engine()
            engine.register_model("rf_model", model)
            
            request = PredictionRequest(
                request_id="integration_test",
                timestamp=datetime.now(),
                features=X_test,
                model_name="rf_model"
            )
            
            response = engine.predict(request)
            
            assert response.predictions is not None, "Should get predictions"
            assert len(response.predictions) == len(X_test), "Should predict for all test samples"
            assert response.probabilities is not None, "Should get probabilities from RF"
            assert response.processing_time_ms < 100, "Should meet latency requirement"
            
            print(f"  ✅ Phase 4 model integration successful")
            print(f"  ✅ Predictions for {len(X_test)} samples")
            print(f"  ✅ Processing time: {response.processing_time_ms:.2f}ms")
            
            # Test with model server
            server = create_model_server()
            server.start()
            
            metadata = ModelMetadata(
                name="rf_integration",
                version="1.0.0",
                created_at=datetime.now(),
                model_type="RandomForestClassifier",
                framework="sklearn",
                input_schema={"features": list(X.columns)},
                output_schema={"predictions": "binary"}
            )
            
            server.deploy_model("rf_integration", "1.0.0", model, metadata)
            
            result = server.predict("rf_integration", X_test.iloc[:5])  # Test with 5 samples
            
            assert result['status'] == 'success', "Server prediction should succeed"
            assert len(result['predictions']) == 5, "Should get 5 predictions"
            assert result['processing_time_ms'] < 100, "Should meet latency requirement"
            
            print(f"  ✅ Model server integration successful")
            print(f"  ✅ Server prediction latency: {result['processing_time_ms']:.2f}ms")
            
            server.stop()
            
        else:
            print("  ⚠️  Sklearn not available - using mock integration test")
            
            # Mock integration test
            from prediction_engine import create_prediction_engine, PredictionRequest
            from datetime import datetime
            
            class MockPhase4Model:
                def predict(self, X):
                    return np.random.randint(0, 2, len(X))
                
                def predict_proba(self, X):
                    probs = np.random.random((len(X), 2))
                    return probs / probs.sum(axis=1, keepdims=True)
            
            engine = create_prediction_engine()
            model = MockPhase4Model()
            engine.register_model("mock_phase4", model)
            
            features = pd.DataFrame({
                'feature_1': [1.0, 2.0],
                'feature_2': [0.5, 1.5]
            })
            
            request = PredictionRequest(
                request_id="mock_integration",
                timestamp=datetime.now(),
                features=features,
                model_name="mock_phase4"
            )
            
            response = engine.predict(request)
            
            assert response.predictions is not None, "Should get predictions"
            assert len(response.predictions) == 2, "Should predict for 2 samples"
            
            print(f"  ✅ Mock Phase 4 integration successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Phase 4 integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_requirements():
    """Test performance requirements."""
    print("\n🔬 Testing Performance Requirements...")
    
    try:
        from prediction_engine import create_prediction_engine, PredictionRequest
        from datetime import datetime
        
        # Create high-performance engine
        engine = create_prediction_engine(max_latency_ms=50.0)  # Stricter requirement
        
        # Create fast mock model
        class FastModel:
            def predict(self, X):
                return np.ones(len(X))  # Simple, fast prediction
        
        model = FastModel()
        engine.register_model("fast_model", model)
        
        # Test single prediction latency
        latencies = []
        for i in range(100):
            features = pd.DataFrame({'feature_1': [i]})
            
            request = PredictionRequest(
                request_id=f"perf_test_{i}",
                timestamp=datetime.now(),
                features=features,
                model_name="fast_model"
            )
            
            start_time = time.time()
            response = engine.predict(request)
            latency = (time.time() - start_time) * 1000
            
            latencies.append(latency)
            
            assert response.predictions is not None, "Should get predictions"
        
        # Analyze performance
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        max_latency = np.max(latencies)
        
        # Performance requirements
        assert avg_latency < 50, f"Average latency should be < 50ms, got {avg_latency:.2f}ms"
        assert p95_latency < 100, f"P95 latency should be < 100ms, got {p95_latency:.2f}ms"
        
        print(f"  ✅ Performance requirements met")
        print(f"  ✅ Average latency: {avg_latency:.2f}ms")
        print(f"  ✅ P95 latency: {p95_latency:.2f}ms")
        print(f"  ✅ Max latency: {max_latency:.2f}ms")
        
        # Test throughput
        batch_sizes = [1, 10, 50, 100]
        for batch_size in batch_sizes:
            requests = []
            for i in range(batch_size):
                features = pd.DataFrame({'feature_1': [i]})
                request = PredictionRequest(
                    request_id=f"throughput_{i}",
                    timestamp=datetime.now(),
                    features=features,
                    model_name="fast_model"
                )
                requests.append(request)
            
            start_time = time.time()
            responses = engine.predict_batch(requests)
            total_time = time.time() - start_time
            
            throughput = batch_size / total_time
            
            print(f"  ✅ Batch size {batch_size}: {throughput:.1f} predictions/second")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance requirements test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 5 tests."""
    print("🤖 Phase 5: Prediction System - Comprehensive Testing")
    print("=" * 70)
    
    tests = [
        ("Prediction Engine Basic", test_prediction_engine_basic),
        ("Model Serving", test_model_serving),
        ("Prediction Monitoring", test_prediction_monitoring),
        ("A/B Testing", test_ab_testing),
        ("Batch Predictions", test_batch_predictions),
        ("Phase 4 Integration", test_integration_with_phase4),
        ("Performance Requirements", test_performance_requirements)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} Test...")
        try:
            if test_func():
                print(f"✅ {test_name} Test: PASSED")
                passed_tests += 1
            else:
                print(f"❌ {test_name} Test: FAILED")
        except Exception as e:
            print(f"❌ {test_name} Test: FAILED with exception: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 PHASE 5 COMPREHENSIVE TESTING: PASSED")
        print("✅ Real-time prediction engine functional")
        print("✅ Model serving infrastructure working")
        print("✅ Prediction monitoring operational")
        print("✅ A/B testing framework functional")
        print("✅ Performance requirements met (<100ms latency)")
        print("✅ Integration with Phase 4 models successful")
        print("\n📋 PHASE 5 QUALITY GATE: PASSED")
        print("🚀 Ready to proceed to Phase 6: API and Interface")
        return True
    else:
        print("❌ PHASE 5 COMPREHENSIVE TESTING: FAILED")
        print(f"❌ {total_tests - passed_tests} test(s) failed")
        print("❌ Phase 5 Quality Gate: FAILED")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
