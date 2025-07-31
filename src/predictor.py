"""
Real-time predictor for ML-TA system.

This module implements:
- High-level prediction interface
- Prediction caching and optimization
- Integration with model serving and monitoring
- Batch and streaming prediction capabilities
- Error handling and fallback strategies
"""

import uuid
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import threading
import warnings
warnings.filterwarnings('ignore')

from src.config import get_config
from src.logging_config import get_logger
from src.prediction_engine import (
    PredictionEngine, 
    PredictionRequest, 
    PredictionResponse,
    create_prediction_engine
)
from src.model_serving import ModelServer, create_model_server
from src.exceptions import PredictionError

logger = get_logger(__name__)


class Predictor:
    """
    High-level prediction interface for ML-TA system.
    
    This class serves as the main entry point for making predictions, handling:
    - Feature preprocessing
    - Model selection
    - Prediction execution
    - Error handling and fallback
    - Result formatting
    """
    
    def __init__(self, 
                 prediction_engine: Optional[PredictionEngine] = None,
                 model_server: Optional[ModelServer] = None,
                 config: Optional[Dict[str, Any]] = None,
                 default_model: str = 'latest',
                 fallback_strategy: str = 'default_prediction',
                 max_cache_size: int = 1000,
                 max_batch_size: int = 100):
        """
        Initialize the predictor.
        
        Args:
            prediction_engine: Optional PredictionEngine instance
            model_server: Optional ModelServer instance
            config: Configuration dictionary
            default_model: Default model name to use
            fallback_strategy: Strategy for handling prediction failures
            max_cache_size: Maximum number of cached predictions
            max_batch_size: Maximum batch size for predictions
        """
        # Use provided config or get a default
        if config is not None:
            self.config = config
        else:
            config_obj = get_config()
            self.config = {}
            if hasattr(config_obj, 'prediction'):
                self.config = config_obj.prediction.__dict__ if hasattr(config_obj.prediction, '__dict__') else {}
        
        # Initialize with passed parameters, fall back to config values
        self.default_model = default_model
        self.fallback_strategy = fallback_strategy
        self.max_cache_size = max_cache_size
        self.max_batch_size = max_batch_size
        self.cache_enabled = self.config.get('cache_enabled', True)
        
        # Initialize components
        self.prediction_engine = prediction_engine or create_prediction_engine(
            max_latency_ms=self.config.get('max_latency_ms', 100.0),
            enable_monitoring=self.config.get('enable_monitoring', True),
            enable_ab_testing=self.config.get('enable_ab_testing', False)
        )
        
        self.model_server = model_server or create_model_server()
        self.lock = threading.RLock()
        
        logger.info("Predictor initialized", 
                  default_model=self.default_model,
                  fallback_strategy=self.fallback_strategy,
                  cache_enabled=self.cache_enabled)
        
    def start(self):
        """Start the predictor components."""
        if not self.prediction_engine.is_running:
            self.prediction_engine.start()
        
        if not self.model_server.is_running:
            self.model_server.start()
        
        logger.info("Predictor started")
    
    def stop(self):
        """Stop the predictor components."""
        if self.prediction_engine.is_running:
            self.prediction_engine.stop()
            
        if self.model_server.is_running:
            self.model_server.stop()
            
        logger.info("Predictor stopped")
    
    def predict(self, 
                features: pd.DataFrame, 
                model_name: Optional[str] = None,
                request_id: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a prediction using the specified model.
        
        Args:
            features: Feature DataFrame
            model_name: Name of model to use
            request_id: Optional request identifier
            metadata: Additional metadata for request
            
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # Generate request ID if not provided
            request_id = request_id or str(uuid.uuid4())
            
            # Use specified model or default
            model_name = model_name or self.default_model
            
            # Prepare prediction request
            request = PredictionRequest(
                request_id=request_id,
                timestamp=datetime.now(),
                features=features,
                model_name=model_name,
                metadata=metadata or {}
            )
            
            # Make prediction
            response = self.prediction_engine.predict(request)
            
            if response.predictions.size == 0 and 'error' in (response.metadata or {}):
                raise PredictionError(f"Prediction failed: {response.metadata.get('error')}")
                
            # Format the results
            result = {
                'request_id': response.request_id,
                'timestamp': response.timestamp.isoformat(),
                'predictions': response.predictions.tolist(),
                'model_name': response.model_name,
                'processing_time_ms': response.processing_time_ms
            }
            
            if response.probabilities is not None:
                result['probabilities'] = response.probabilities.tolist()
                
            if response.confidence_scores is not None:
                result['confidence_scores'] = response.confidence_scores.tolist()
            
            if response.metadata:
                result['metadata'] = response.metadata
            
            logger.info("Prediction successful",
                       request_id=request_id,
                       model_name=model_name,
                       processing_time_ms=response.processing_time_ms)
                
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Prediction failed: {e}", 
                       request_id=request_id,
                       model_name=model_name or self.default_model,
                       processing_time_ms=processing_time_ms)
            
            # Apply fallback strategy
            if self.fallback_strategy == 'default_prediction':
                fallback_result = self._get_default_prediction(features)
                fallback_result['request_id'] = request_id
                fallback_result['timestamp'] = datetime.now().isoformat()
                fallback_result['error'] = str(e)
                fallback_result['is_fallback'] = True
                fallback_result['processing_time_ms'] = processing_time_ms
                
                logger.info("Using fallback prediction",
                          request_id=request_id,
                          fallback_strategy=self.fallback_strategy)
                
                return fallback_result
            
            elif self.fallback_strategy == 'default_model' and model_name != self.default_model:
                logger.info("Retrying with default model",
                          request_id=request_id,
                          default_model=self.default_model)
                
                # Retry with default model
                return self.predict(
                    features=features,
                    model_name=self.default_model,
                    request_id=request_id,
                    metadata=metadata
                )
                
            else:
                # No fallback or fallback also failed
                return {
                    'request_id': request_id,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'processing_time_ms': processing_time_ms,
                    'success': False
                }
    
    def predict_batch(self, 
                     features_list: List[pd.DataFrame],
                     model_name: Optional[str] = None,
                     request_ids: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            features_list: List of feature DataFrames
            model_name: Name of model to use
            request_ids: Optional list of request identifiers
            metadata: Additional metadata for requests
            
        Returns:
            List of prediction result dictionaries
        """
        # Generate request IDs if not provided
        if request_ids is None:
            request_ids = [str(uuid.uuid4()) for _ in range(len(features_list))]
        elif len(request_ids) < len(features_list):
            additional_ids = [str(uuid.uuid4()) for _ in range(len(features_list) - len(request_ids))]
            request_ids.extend(additional_ids)
            
        # Create prediction requests
        requests = [
            PredictionRequest(
                request_id=request_id,
                timestamp=datetime.now(),
                features=features,
                model_name=model_name or self.default_model,
                metadata=metadata or {}
            )
            for request_id, features in zip(request_ids, features_list)
        ]
        
        # Make batch predictions
        responses = self.prediction_engine.predict_batch(requests)
        
        # Format results
        results = []
        for response in responses:
            result = {
                'request_id': response.request_id,
                'timestamp': response.timestamp.isoformat(),
                'predictions': response.predictions.tolist(),
                'model_name': response.model_name,
                'processing_time_ms': response.processing_time_ms
            }
            
            if response.probabilities is not None:
                result['probabilities'] = response.probabilities.tolist()
                
            if response.confidence_scores is not None:
                result['confidence_scores'] = response.confidence_scores.tolist()
            
            if response.metadata:
                result['metadata'] = response.metadata
                
            results.append(result)
            
        logger.info("Batch prediction completed",
                  batch_size=len(features_list),
                  model_name=model_name or self.default_model)
            
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the prediction system.
        
        Returns:
            Dictionary with status information
        """
        engine_status = self.prediction_engine.get_status()
        server_status = self.model_server.get_model_status()
        
        return {
            'predictor': {
                'default_model': self.default_model,
                'fallback_strategy': self.fallback_strategy,
                'cache_enabled': self.cache_enabled
            },
            'prediction_engine': engine_status,
            'model_server': server_status,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_default_prediction(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate default prediction when actual prediction fails.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Default prediction result
        """
        # Simple default prediction strategy
        # In a real system, this could be more sophisticated
        if isinstance(features, pd.DataFrame):
            # For regression: use mean value or 0
            default_value = 0
            
            # For classification: use most frequent class (0)
            predictions = np.zeros(len(features))
            
            return {
                'predictions': predictions.tolist(),
                'model_name': 'fallback',
                'probabilities': None,
                'confidence_scores': None
            }
        else:
            return {
                'predictions': [0],
                'model_name': 'fallback',
                'probabilities': None,
                'confidence_scores': None
            }


def create_predictor(config_override: Optional[Dict[str, Any]] = None) -> Predictor:
    """
    Factory function to create a Predictor instance.
    
    Args:
        config_override: Optional configuration dictionary to override defaults
        
    Returns:
        Configured Predictor instance
    """
    # Get configuration
    config = get_config()
    
    # Get prediction configuration or empty dict if not available
    prediction_config = {}
    if hasattr(config, 'prediction'):
        prediction_config = config.prediction.__dict__ if hasattr(config.prediction, '__dict__') else {}
    
    # Apply overrides if provided
    if config_override:
        prediction_config.update(config_override)
    
    # Create prediction engine with specified configuration
    prediction_engine = create_prediction_engine(
        max_latency_ms=prediction_config.get('max_latency_ms', 100.0),
        enable_monitoring=prediction_config.get('enable_monitoring', True),
        enable_ab_testing=prediction_config.get('enable_ab_testing', False)
    )
    
    # Create model server with specified configuration
    model_server = create_model_server(
        max_concurrent_requests=prediction_config.get('max_concurrent_requests', 100),
        request_timeout_seconds=prediction_config.get('request_timeout_seconds', 30.0),
        health_check_interval=prediction_config.get('health_check_interval', 30)
    )
    
    # Create and return the predictor
    return Predictor(
        prediction_engine=prediction_engine,
        model_server=model_server,
        default_model=prediction_config.get('default_model', 'latest'),
        fallback_strategy=prediction_config.get('fallback_strategy', 'default_prediction'),
        max_cache_size=prediction_config.get('max_cache_size', 1000),
        max_batch_size=prediction_config.get('max_batch_size', 100)
    )
