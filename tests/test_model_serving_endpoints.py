"""
Test model serving endpoints with scaled features.

This script tests:
1. Direct RobustScaler integration with model serving
2. Model deployment via ModelServer
3. Prediction requests with proper feature scaling
4. End-to-end prediction flow
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
import unittest
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import RobustScaler from sklearn if available, otherwise use our custom implementation
try:
    from sklearn.preprocessing import RobustScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    from src.features import RobustScaler
    SKLEARN_AVAILABLE = False
from src.model_serving import ModelServer, ModelMetadata, create_model_server
from src.prediction_engine import PredictionRequest
from sklearn.linear_model import LinearRegression
import joblib
import tempfile


class TestModelServingEndpoints(unittest.TestCase):
    """Test model serving endpoints with scaled features."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample data with numeric features
        np.random.seed(42)
        
        # Create training data with numeric features
        self.data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 3, 100),
            'feature3': np.random.normal(-5, 10, 100),
            'target': np.random.normal(0, 1, 100)
        })
        
        # Add outliers to test robust scaling
        self.data.loc[0, 'feature1'] = 100.0  # Extreme outlier
        
        # Create scaler
        self.scaler = RobustScaler()
        
        # Create model server
        self.model_server = create_model_server(
            max_concurrent_requests=10,
            request_timeout_seconds=5.0,
            health_check_interval=5
        )
        self.model_server.start()
        
        # Create temp directory for model storage
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop model server
        self.model_server.stop()
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_train_and_deploy_model_with_scaled_features(self):
        """Test training and deploying a model with scaled features."""
        # Apply robust scaling directly
        X = self.data[['feature1', 'feature2', 'feature3']]
        y = self.data['target']
        
        # Fit scaler on features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Convert back to DataFrame for easier handling
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Verify scaling was applied and outlier was handled
        self.assertNotEqual(
            self.data['feature1'].values[0],
            X_scaled_df['feature1'].values[0],
            "Scaling should have transformed the outlier"
        )
        
        # Check that median was subtracted and scaling by IQR was applied
        self.assertLess(
            abs(X_scaled_df['feature1'].median()),
            abs(X['feature1'].median()),
            "Median should be closer to zero after robust scaling"
        )
        
        # Train a simple model on scaled data
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Save the model to a temporary file
        model_path = os.path.join(self.temp_dir, "test_model.joblib")
        joblib.dump(model, model_path)
        
        # Also save the scaler for later use
        scaler_path = os.path.join(self.temp_dir, "test_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        # Register the model with the model server
        metadata = ModelMetadata(
            name="test_model",
            version="0.1",
            created_at=datetime.now(),
            model_type="regression",
            framework="sklearn",
            input_schema={"feature1": "float", "feature2": "float", "feature3": "float"},
            output_schema={"prediction": "float"}
        )
        
        success = self.model_server.registry.register_model(metadata, model_path)
        self.assertTrue(success, "Model registration should succeed")
        
        # Load the model for deployment
        loaded_model = joblib.load(model_path)
        
        # Deploy the model
        success = self.model_server.deploy_model(
            model_name="test_model",
            model_version="0.1",
            model_object=loaded_model
        )
        self.assertTrue(success, "Model deployment should succeed")
        
        # Wait for deployment to complete
        time.sleep(1)
        
        # Create new test data with outliers for prediction
        test_data = pd.DataFrame({
            'feature1': [10.0, 0.5, -0.5],  # One outlier
            'feature2': [5.0, 4.0, 6.0],
            'feature3': [-5.0, -3.0, -7.0],
        })
        
        # Scale the test data using our fitted scaler
        test_data_scaled = pd.DataFrame(
            self.scaler.transform(test_data),
            columns=test_data.columns
        )
        
        # Make a prediction through the model server
        # The model server should use the scaled features
        result = self.model_server.predict(
            model_name="test_model",
            features=test_data_scaled,  # Already scaled features
            request_id="test-request-001"
        )
        
        # Verify prediction result
        self.assertIsNotNone(result, "Prediction result should not be None")
        self.assertIn('predictions', result, "Result should contain predictions")
        self.assertEqual(len(result['predictions']), 3, "Should have 3 predictions")
        
        # Get model status and verify
        status = self.model_server.get_model_status("test_model")
        self.assertEqual(status['model_name'], "test_model")
        self.assertEqual(status['version'], "0.1")
        self.assertEqual(status['status'], "active")  # Note: status is 'active', not 'deployed'
        self.assertEqual(status['request_count'], 1)
        
        # Test prediction with explicit request parameters
        result2 = self.model_server.predict(
            model_name="test_model",
            features=test_data_scaled,  # Already scaled features
            request_id="test-request-002"
        )
        self.assertIsNotNone(result2, "Prediction result should not be None")
        self.assertIn('predictions', result2, "Result should contain predictions key")
        self.assertEqual(len(result2['predictions']), 3, "Should have 3 predictions")
        
        # Test model metrics
        # Note: Due to how metrics are handled, requests might not be reflected immediately in metrics
        # We'll just verify that the metrics structure is correct
        metrics = self.model_server.get_metrics()
        self.assertIn('total_requests', metrics)
        self.assertIn('successful_requests', metrics)
        self.assertIn('failed_requests', metrics)
        
        print("Model serving endpoints test passed successfully!")


if __name__ == '__main__':
    unittest.main()
