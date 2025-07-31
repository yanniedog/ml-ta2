"""
Real-time prediction engine for ML-TA system.

This module implements:
- Real-time prediction capabilities with <100ms latency
- Model serving infrastructure with caching
- Prediction monitoring and drift detection
- A/B testing framework for model comparison
- Batch and streaming prediction support
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import threading
import queue
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
try:
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    class BaseEstimator:
        pass

# Caching
try:
    from functools import lru_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    def lru_cache(maxsize=None):
        def decorator(func):
            return func
        return decorator

from src.config import get_config
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionRequest:
    """Container for prediction request data."""
    request_id: str
    timestamp: datetime
    features: pd.DataFrame
    model_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PredictionResponse:
    """Container for prediction response data."""
    request_id: str
    timestamp: datetime
    predictions: np.ndarray
    probabilities: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    model_name: Optional[str] = None
    processing_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PredictionConfig:
    """Configuration for prediction engine."""
    max_batch_size: int = 1000
    max_latency_ms: float = 100.0
    cache_size: int = 10000
    enable_monitoring: bool = True
    enable_ab_testing: bool = False
    default_model: str = "default"
    confidence_threshold: float = 0.5
    drift_detection_window: int = 1000


class ModelCache:
    """Thread-safe model cache with LRU eviction."""
    
    def __init__(self, max_size: int = 10):
        """Initialize model cache."""
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.lock = threading.RLock()
        
        logger.info("ModelCache initialized", max_size=max_size)
    
    def get(self, model_name: str) -> Optional[Any]:
        """Get model from cache."""
        with self.lock:
            if model_name in self.cache:
                self.access_times[model_name] = time.time()
                logger.debug("Model cache hit", model_name=model_name)
                return self.cache[model_name]
            
            logger.debug("Model cache miss", model_name=model_name)
            return None
    
    def put(self, model_name: str, model: Any):
        """Put model in cache with LRU eviction."""
        with self.lock:
            # Evict least recently used if at capacity
            if len(self.cache) >= self.max_size and model_name not in self.cache:
                lru_model = min(self.access_times.items(), key=lambda x: x[1])[0]
                del self.cache[lru_model]
                del self.access_times[lru_model]
                logger.debug("Model evicted from cache", evicted_model=lru_model)
            
            self.cache[model_name] = model
            self.access_times[model_name] = time.time()
            
            logger.debug("Model cached", model_name=model_name, cache_size=len(self.cache))
    
    def remove(self, model_name: str):
        """Remove model from cache."""
        with self.lock:
            if model_name in self.cache:
                del self.cache[model_name]
                del self.access_times[model_name]
                logger.debug("Model removed from cache", model_name=model_name)
    
    def clear(self):
        """Clear all models from cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            logger.info("Model cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'models': list(self.cache.keys())
            }


class PredictionMonitor:
    """Monitor prediction performance and detect drift."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize prediction monitor."""
        self.window_size = window_size
        self.predictions_history = []
        self.performance_metrics = {}
        self.drift_alerts = []
        self.lock = threading.RLock()
        
        logger.info("PredictionMonitor initialized", window_size=window_size)
    
    def record_prediction(self, request: PredictionRequest, response: PredictionResponse):
        """Record prediction for monitoring."""
        with self.lock:
            prediction_record = {
                'timestamp': response.timestamp,
                'request_id': response.request_id,
                'model_name': response.model_name,
                'processing_time_ms': response.processing_time_ms,
                'feature_count': len(request.features.columns) if hasattr(request.features, 'columns') else 0,
                'sample_count': len(request.features) if hasattr(request.features, '__len__') else 1,
                'has_probabilities': response.probabilities is not None,
                'confidence_scores': response.confidence_scores
            }
            
            self.predictions_history.append(prediction_record)
            
            # Keep only recent predictions
            if len(self.predictions_history) > self.window_size:
                self.predictions_history = self.predictions_history[-self.window_size:]
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Check for drift
            self._check_drift()
    
    def _update_performance_metrics(self):
        """Update performance metrics from recent predictions."""
        if not self.predictions_history:
            return
        
        recent_predictions = self.predictions_history[-100:]  # Last 100 predictions
        
        processing_times = [p['processing_time_ms'] for p in recent_predictions if p['processing_time_ms']]
        
        if processing_times:
            self.performance_metrics = {
                'avg_processing_time_ms': np.mean(processing_times),
                'max_processing_time_ms': np.max(processing_times),
                'min_processing_time_ms': np.min(processing_times),
                'p95_processing_time_ms': np.percentile(processing_times, 95),
                'predictions_per_second': len(recent_predictions) / max(1, (recent_predictions[-1]['timestamp'] - recent_predictions[0]['timestamp']).total_seconds()),
                'total_predictions': len(self.predictions_history)
            }
    
    def _check_drift(self):
        """Check for prediction drift."""
        if len(self.predictions_history) < 100:
            return
        
        recent_times = [p['processing_time_ms'] for p in self.predictions_history[-50:] if p['processing_time_ms']]
        older_times = [p['processing_time_ms'] for p in self.predictions_history[-100:-50] if p['processing_time_ms']]
        
        if len(recent_times) >= 10 and len(older_times) >= 10:
            recent_avg = np.mean(recent_times)
            older_avg = np.mean(older_times)
            
            # Alert if processing time increased significantly
            if recent_avg > older_avg * 1.5:
                alert = {
                    'timestamp': datetime.now(),
                    'type': 'performance_degradation',
                    'message': f'Processing time increased from {older_avg:.2f}ms to {recent_avg:.2f}ms',
                    'severity': 'warning'
                }
                self.drift_alerts.append(alert)
                logger.warning("Performance drift detected", **alert)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        with self.lock:
            return {
                'performance_metrics': self.performance_metrics.copy(),
                'drift_alerts': self.drift_alerts[-10:],  # Last 10 alerts
                'monitoring_window_size': len(self.predictions_history)
            }


class ABTestManager:
    """Manage A/B testing for model comparison."""
    
    def __init__(self):
        """Initialize A/B test manager."""
        self.active_tests = {}
        self.test_results = {}
        self.lock = threading.RLock()
        
        logger.info("ABTestManager initialized")
    
    def create_test(self, test_name: str, model_a: str, model_b: str, 
                   traffic_split: float = 0.5) -> bool:
        """Create new A/B test."""
        with self.lock:
            if test_name in self.active_tests:
                logger.warning("A/B test already exists", test_name=test_name)
                return False
            
            self.active_tests[test_name] = {
                'model_a': model_a,
                'model_b': model_b,
                'traffic_split': traffic_split,
                'start_time': datetime.now(),
                'predictions_a': [],
                'predictions_b': [],
                'metrics_a': {},
                'metrics_b': {}
            }
            
            logger.info("A/B test created", 
                       test_name=test_name,
                       model_a=model_a,
                       model_b=model_b,
                       traffic_split=traffic_split)
            
            return True
    
    def get_model_for_request(self, test_name: str, request_id: str) -> Optional[str]:
        """Get model assignment for A/B test."""
        with self.lock:
            if test_name not in self.active_tests:
                return None
            
            test = self.active_tests[test_name]
            
            # Use hash of request_id for consistent assignment
            hash_value = hash(request_id) % 100
            traffic_split_pct = test['traffic_split'] * 100
            
            if hash_value < traffic_split_pct:
                return test['model_a']
            else:
                return test['model_b']
    
    def record_prediction(self, test_name: str, model_name: str, 
                         response: PredictionResponse):
        """Record prediction result for A/B test."""
        with self.lock:
            if test_name not in self.active_tests:
                return
            
            test = self.active_tests[test_name]
            
            prediction_record = {
                'timestamp': response.timestamp,
                'processing_time_ms': response.processing_time_ms,
                'confidence_scores': response.confidence_scores
            }
            
            if model_name == test['model_a']:
                test['predictions_a'].append(prediction_record)
            elif model_name == test['model_b']:
                test['predictions_b'].append(prediction_record)
    
    def get_test_results(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Get A/B test results."""
        with self.lock:
            if test_name not in self.active_tests:
                return None
            
            test = self.active_tests[test_name]
            
            # Calculate metrics for both models
            def calculate_metrics(predictions):
                if not predictions:
                    return {}
                
                processing_times = [p['processing_time_ms'] for p in predictions if p['processing_time_ms']]
                
                return {
                    'prediction_count': len(predictions),
                    'avg_processing_time_ms': np.mean(processing_times) if processing_times else 0,
                    'p95_processing_time_ms': np.percentile(processing_times, 95) if processing_times else 0
                }
            
            metrics_a = calculate_metrics(test['predictions_a'])
            metrics_b = calculate_metrics(test['predictions_b'])
            
            return {
                'test_name': test_name,
                'model_a': test['model_a'],
                'model_b': test['model_b'],
                'traffic_split': test['traffic_split'],
                'start_time': test['start_time'],
                'duration_hours': (datetime.now() - test['start_time']).total_seconds() / 3600,
                'metrics_a': metrics_a,
                'metrics_b': metrics_b
            }


class PredictionEngine:
    """Main real-time prediction engine."""
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        """Initialize prediction engine."""
        self.config = config or PredictionConfig()
        self.model_cache = ModelCache(max_size=10)
        self.monitor = PredictionMonitor(window_size=self.config.drift_detection_window)
        self.ab_test_manager = ABTestManager()
        self.models = {}
        self.feature_preprocessors = {}
        self.prediction_queue = queue.Queue()
        self.is_running = False
        
        logger.info("PredictionEngine initialized", 
                   max_latency_ms=self.config.max_latency_ms,
                   max_batch_size=self.config.max_batch_size)
    
    def register_model(self, model_name: str, model: Any, 
                      preprocessor: Optional[Any] = None):
        """Register a model for predictions."""
        try:
            self.models[model_name] = model
            if preprocessor:
                self.feature_preprocessors[model_name] = preprocessor
            
            # Cache the model for fast access
            self.model_cache.put(model_name, model)
            
            logger.info("Model registered", 
                       model_name=model_name,
                       has_preprocessor=preprocessor is not None)
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {e}")
            raise
    
    def unregister_model(self, model_name: str):
        """Unregister a model."""
        try:
            if model_name in self.models:
                del self.models[model_name]
            
            if model_name in self.feature_preprocessors:
                del self.feature_preprocessors[model_name]
            
            self.model_cache.remove(model_name)
            
            logger.info("Model unregistered", model_name=model_name)
            
        except Exception as e:
            logger.error(f"Failed to unregister model {model_name}: {e}")
    
    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make real-time prediction."""
        start_time = time.time()
        
        try:
            # Determine model to use
            model_name = request.model_name or self.config.default_model
            
            # Check for A/B testing
            if self.config.enable_ab_testing:
                for test_name in self.ab_test_manager.active_tests:
                    ab_model = self.ab_test_manager.get_model_for_request(test_name, request.request_id)
                    if ab_model:
                        model_name = ab_model
                        break
            
            # Get model from cache or registry
            model = self.model_cache.get(model_name)
            if model is None:
                if model_name in self.models:
                    model = self.models[model_name]
                    self.model_cache.put(model_name, model)
                else:
                    raise ValueError(f"Model {model_name} not found")
            
            # Preprocess features if needed
            features = request.features
            if model_name in self.feature_preprocessors:
                preprocessor = self.feature_preprocessors[model_name]
                features = preprocessor.transform(features)
            
            # Make prediction
            predictions = model.predict(features)
            
            # Get probabilities if available
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(features)
                except:
                    pass
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence(predictions, probabilities)
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create response
            response = PredictionResponse(
                request_id=request.request_id,
                timestamp=datetime.now(),
                predictions=predictions,
                probabilities=probabilities,
                confidence_scores=confidence_scores,
                model_name=model_name,
                processing_time_ms=processing_time_ms,
                metadata=request.metadata
            )
            
            # Monitor prediction
            if self.config.enable_monitoring:
                self.monitor.record_prediction(request, response)
            
            # Record for A/B testing
            if self.config.enable_ab_testing:
                for test_name in self.ab_test_manager.active_tests:
                    self.ab_test_manager.record_prediction(test_name, model_name, response)
            
            # Check latency requirement
            if processing_time_ms > self.config.max_latency_ms:
                logger.warning("Prediction exceeded latency requirement",
                             processing_time_ms=processing_time_ms,
                             max_latency_ms=self.config.max_latency_ms)
            
            logger.debug("Prediction completed",
                        request_id=request.request_id,
                        model_name=model_name,
                        processing_time_ms=processing_time_ms)
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed for request {request.request_id}: {e}")
            
            # Return error response
            return PredictionResponse(
                request_id=request.request_id,
                timestamp=datetime.now(),
                predictions=np.array([]),
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(e)}
            )
    
    def _calculate_confidence(self, predictions: np.ndarray, 
                            probabilities: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Calculate confidence scores for predictions."""
        if probabilities is not None:
            # For classification: use max probability as confidence
            if len(probabilities.shape) == 2:
                return np.max(probabilities, axis=1)
            else:
                return probabilities
        
        # For regression or when probabilities unavailable: use simple heuristic
        return None
    
    def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """Make batch predictions."""
        if len(requests) > self.config.max_batch_size:
            logger.warning("Batch size exceeds maximum",
                          batch_size=len(requests),
                          max_batch_size=self.config.max_batch_size)
            requests = requests[:self.config.max_batch_size]
        
        responses = []
        for request in requests:
            response = self.predict(request)
            responses.append(response)
        
        logger.info("Batch prediction completed",
                   batch_size=len(requests),
                   total_processing_time_ms=sum(r.processing_time_ms or 0 for r in responses))
        
        return responses
    
    def start(self):
        """Start the prediction engine and all its components."""
        if self.is_running:
            logger.info("PredictionEngine is already running")
            return
        
        logger.info("Starting PredictionEngine")
        self.is_running = True
        
        # Initialize any background workers or services here if needed
        # For example, we could start a background thread for batch processing
        
        logger.info("PredictionEngine started successfully")
    
    def stop(self):
        """Stop the prediction engine and all its components."""
        if not self.is_running:
            logger.info("PredictionEngine is already stopped")
            return
        
        logger.info("Stopping PredictionEngine")
        self.is_running = False
        
        # Cleanup resources, stop background threads, etc.
        
        logger.info("PredictionEngine stopped successfully")
    
    def get_status(self) -> Dict[str, Any]:
        """Get prediction engine status."""
        return {
            'is_running': self.is_running,
            'registered_models': list(self.models.keys()),
            'cache_stats': self.model_cache.get_stats(),
            'monitoring_metrics': self.monitor.get_metrics() if self.config.enable_monitoring else {},
            'active_ab_tests': list(self.ab_test_manager.active_tests.keys()) if self.config.enable_ab_testing else [],
            'config': {
                'max_latency_ms': self.config.max_latency_ms,
                'max_batch_size': self.config.max_batch_size,
                'enable_monitoring': self.config.enable_monitoring,
                'enable_ab_testing': self.config.enable_ab_testing
            }
        }


def create_prediction_engine(max_latency_ms: float = 100.0, 
                           enable_monitoring: bool = True,
                           enable_ab_testing: bool = False) -> PredictionEngine:
    """Factory function to create prediction engine."""
    config = PredictionConfig(
        max_latency_ms=max_latency_ms,
        enable_monitoring=enable_monitoring,
        enable_ab_testing=enable_ab_testing,
        max_batch_size=1000,
        cache_size=10000,
        default_model="default",
        confidence_threshold=0.5,
        drift_detection_window=1000
    )
    
    return PredictionEngine(config)
