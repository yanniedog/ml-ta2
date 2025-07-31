"""
Model serving infrastructure for ML-TA system.

This module implements:
- Model deployment and versioning
- Health checks and monitoring
- Load balancing and scaling
- Model lifecycle management
- Performance optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import threading
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Core libraries
try:
    import pickle
    PICKLE_AVAILABLE = True
except ImportError:
    PICKLE_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from src.config import get_config
from src.logging_config import get_logger
from src.prediction_engine import PredictionEngine, PredictionRequest, PredictionResponse
try:
    from src.ab_testing import ABTestingManager, create_ab_testing_manager
    AB_TESTING_AVAILABLE = True
except ImportError:
    AB_TESTING_AVAILABLE = False
    def create_ab_testing_manager():
        return None

logger = get_logger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for deployed models."""
    name: str
    version: str
    created_at: datetime
    model_type: str
    framework: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    checksum: str = ""


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    max_memory_mb: int = 1024
    max_cpu_percent: float = 80.0
    health_check_interval_seconds: int = 30
    max_concurrent_requests: int = 100
    request_timeout_seconds: float = 30.0
    auto_scaling_enabled: bool = False
    min_replicas: int = 1
    max_replicas: int = 5


class ModelRegistry:
    """Registry for managing model metadata and versions."""
    
    def __init__(self):
        """Initialize model registry."""
        self.models = {}
        self.versions = {}
        self.lock = threading.RLock()
        
        logger.info("ModelRegistry initialized")
    
    def register_model(self, metadata: ModelMetadata, model_path: str) -> bool:
        """Register a new model or version."""
        try:
            with self.lock:
                model_key = f"{metadata.name}:{metadata.version}"
                
                if model_key in self.models:
                    logger.warning("Model version already exists", 
                                 name=metadata.name, 
                                 version=metadata.version)
                    return False
                
                # Calculate checksum
                metadata.checksum = self._calculate_checksum(model_path)
                
                # Store metadata
                self.models[model_key] = {
                    'metadata': metadata,
                    'model_path': model_path,
                    'registered_at': datetime.now(),
                    'status': 'registered'
                }
                
                # Update version tracking
                if metadata.name not in self.versions:
                    self.versions[metadata.name] = []
                
                self.versions[metadata.name].append(metadata.version)
                self.versions[metadata.name].sort(reverse=True)  # Latest first
                
                logger.info("Model registered", 
                           name=metadata.name,
                           version=metadata.version,
                           checksum=metadata.checksum[:8])
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to register model {metadata.name}:{metadata.version}: {e}")
            return False
    
    def get_model_info(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get model information."""
        with self.lock:
            if version is None:
                # Get latest version
                if name in self.versions and self.versions[name]:
                    version = self.versions[name][0]
                else:
                    return None
            
            model_key = f"{name}:{version}"
            return self.models.get(model_key)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        with self.lock:
            models = []
            for model_key, model_info in self.models.items():
                models.append({
                    'name': model_info['metadata'].name,
                    'version': model_info['metadata'].version,
                    'created_at': model_info['metadata'].created_at,
                    'model_type': model_info['metadata'].model_type,
                    'status': model_info['status'],
                    'registered_at': model_info['registered_at']
                })
            
            return sorted(models, key=lambda x: x['registered_at'], reverse=True)
    
    def get_versions(self, name: str) -> List[str]:
        """Get all versions of a model."""
        with self.lock:
            return self.versions.get(name, []).copy()
    
    def delete_model(self, name: str, version: str) -> bool:
        """Delete a model version."""
        try:
            with self.lock:
                model_key = f"{name}:{version}"
                
                if model_key not in self.models:
                    logger.warning("Model version not found", name=name, version=version)
                    return False
                
                del self.models[model_key]
                
                if name in self.versions:
                    self.versions[name] = [v for v in self.versions[name] if v != version]
                    if not self.versions[name]:
                        del self.versions[name]
                
                logger.info("Model deleted", name=name, version=version)
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete model {name}:{version}: {e}")
            return False
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum."""
        try:
            import hashlib
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return "unknown"


class HealthChecker:
    """Health monitoring for deployed models."""
    
    def __init__(self, check_interval: int = 30):
        """Initialize health checker."""
        self.check_interval = check_interval
        self.health_status = {}
        self.health_history = {}
        self.is_running = False
        self.check_thread = None
        self.lock = threading.RLock()
        
        logger.info("HealthChecker initialized", check_interval=check_interval)
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.check_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_running = False
        if self.check_thread:
            self.check_thread.join(timeout=5)
        
        logger.info("Health monitoring stopped")
    
    def register_model(self, model_name: str, health_check_func: Callable[[], bool]):
        """Register a model for health monitoring."""
        with self.lock:
            self.health_status[model_name] = {
                'status': 'unknown',
                'last_check': None,
                'check_function': health_check_func,
                'consecutive_failures': 0
            }
            
            if model_name not in self.health_history:
                self.health_history[model_name] = []
        
        logger.info("Model registered for health monitoring", model_name=model_name)
    
    def unregister_model(self, model_name: str):
        """Unregister a model from health monitoring."""
        with self.lock:
            if model_name in self.health_status:
                del self.health_status[model_name]
        
        logger.info("Model unregistered from health monitoring", model_name=model_name)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                self._perform_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _perform_health_checks(self):
        """Perform health checks on all registered models."""
        with self.lock:
            for model_name, status_info in self.health_status.items():
                try:
                    check_func = status_info['check_function']
                    is_healthy = check_func()
                    
                    # Update status
                    previous_status = status_info['status']
                    status_info['status'] = 'healthy' if is_healthy else 'unhealthy'
                    status_info['last_check'] = datetime.now()
                    
                    if is_healthy:
                        status_info['consecutive_failures'] = 0
                    else:
                        status_info['consecutive_failures'] += 1
                    
                    # Record in history
                    self.health_history[model_name].append({
                        'timestamp': datetime.now(),
                        'status': status_info['status'],
                        'consecutive_failures': status_info['consecutive_failures']
                    })
                    
                    # Keep only recent history
                    if len(self.health_history[model_name]) > 1000:
                        self.health_history[model_name] = self.health_history[model_name][-1000:]
                    
                    # Log status changes
                    if previous_status != status_info['status']:
                        logger.info("Model health status changed",
                                   model_name=model_name,
                                   previous_status=previous_status,
                                   new_status=status_info['status'])
                    
                    # Alert on consecutive failures
                    if status_info['consecutive_failures'] >= 3:
                        logger.error("Model health check failing consistently",
                                   model_name=model_name,
                                   consecutive_failures=status_info['consecutive_failures'])
                
                except Exception as e:
                    logger.error(f"Health check failed for {model_name}: {e}")
                    status_info['status'] = 'error'
                    status_info['last_check'] = datetime.now()
    
    def get_health_status(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get health status for models."""
        with self.lock:
            if model_name:
                return self.health_status.get(model_name, {})
            else:
                return self.health_status.copy()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        with self.lock:
            total_models = len(self.health_status)
            healthy_models = sum(1 for status in self.health_status.values() 
                               if status['status'] == 'healthy')
            
            return {
                'total_models': total_models,
                'healthy_models': healthy_models,
                'unhealthy_models': total_models - healthy_models,
                'health_percentage': (healthy_models / total_models * 100) if total_models > 0 else 0,
                'last_check': max([status['last_check'] for status in self.health_status.values()], 
                                default=None)
            }


class ModelServer:
    """Main model serving infrastructure."""
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        """Initialize model server."""
        self.config = config or DeploymentConfig()
        self.registry = ModelRegistry()
        self.health_checker = HealthChecker(self.config.health_check_interval_seconds)
        self.prediction_engine = PredictionEngine()
        self.deployed_models = {}
        self.request_stats = {}
        self.lock = threading.RLock()
        
        logger.info("ModelServer initialized", 
                   max_concurrent_requests=self.config.max_concurrent_requests,
                   request_timeout=self.config.request_timeout_seconds)
    
    def start(self):
        """Start the model server."""
        try:
            self.health_checker.start_monitoring()
            logger.info("ModelServer started")
            
        except Exception as e:
            logger.error(f"Failed to start ModelServer: {e}")
            raise
    
    def stop(self):
        """Stop the model server."""
        try:
            self.health_checker.stop_monitoring()
            logger.info("ModelServer stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop ModelServer: {e}")
    
    def deploy_model(self, model_name: str, model_version: str, 
                    model_object: Any, metadata: Optional[ModelMetadata] = None) -> bool:
        """Deploy a model for serving."""
        try:
            with self.lock:
                # Create metadata if not provided
                if metadata is None:
                    metadata = ModelMetadata(
                        name=model_name,
                        version=model_version,
                        created_at=datetime.now(),
                        model_type=type(model_object).__name__,
                        framework="sklearn" if hasattr(model_object, 'predict') else "unknown",
                        input_schema={},
                        output_schema={}
                    )
                
                # Register model in prediction engine
                self.prediction_engine.register_model(model_name, model_object)
                
                # Store deployment info
                deployment_key = f"{model_name}:{model_version}"
                self.deployed_models[deployment_key] = {
                    'model': model_object,
                    'metadata': metadata,
                    'deployed_at': datetime.now(),
                    'status': 'active',
                    'request_count': 0,
                    'error_count': 0,
                    'total_processing_time_ms': 0
                }
                
                # Register for health monitoring
                def health_check():
                    try:
                        # Simple health check: try to make a dummy prediction
                        if hasattr(model_object, 'predict'):
                            # Create dummy data based on expected input
                            dummy_data = pd.DataFrame({'feature_1': [0.0]})
                            model_object.predict(dummy_data)
                        return True
                    except Exception:
                        return False
                
                self.health_checker.register_model(deployment_key, health_check)
                
                # Initialize request stats
                self.request_stats[deployment_key] = {
                    'total_requests': 0,
                    'successful_requests': 0,
                    'failed_requests': 0,
                    'avg_processing_time_ms': 0,
                    'last_request_time': None
                }
                
                logger.info("Model deployed successfully",
                           model_name=model_name,
                           model_version=model_version,
                           model_type=metadata.model_type)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to deploy model {model_name}:{model_version}: {e}")
            return False
    
    def undeploy_model(self, model_name: str, model_version: str) -> bool:
        """Undeploy a model from serving."""
        try:
            with self.lock:
                deployment_key = f"{model_name}:{model_version}"
                
                if deployment_key not in self.deployed_models:
                    logger.warning("Model not deployed", 
                                 model_name=model_name, 
                                 model_version=model_version)
                    return False
                
                # Unregister from prediction engine
                self.prediction_engine.unregister_model(model_name)
                
                # Unregister from health monitoring
                self.health_checker.unregister_model(deployment_key)
                
                # Remove deployment info
                del self.deployed_models[deployment_key]
                
                # Clean up request stats
                if deployment_key in self.request_stats:
                    del self.request_stats[deployment_key]
                
                logger.info("Model undeployed successfully",
                           model_name=model_name,
                           model_version=model_version)
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to undeploy model {model_name}:{model_version}: {e}")
            return False
    
    def predict(self, model_name: str, features: pd.DataFrame, 
               request_id: Optional[str] = None) -> Dict[str, Any]:
        """Make prediction using deployed model."""
        start_time = time.time()
        request_id = request_id or f"req_{int(time.time() * 1000)}"
        
        try:
            # Find deployed model
            deployed_model = None
            deployment_key = None
            
            with self.lock:
                for key, deployment in self.deployed_models.items():
                    if key.startswith(f"{model_name}:"):
                        deployed_model = deployment
                        deployment_key = key
                        break
            
            if not deployed_model:
                raise ValueError(f"Model {model_name} not deployed")
            
            # Create prediction request
            request = PredictionRequest(
                request_id=request_id,
                timestamp=datetime.now(),
                features=features,
                model_name=model_name
            )
            
            # Make prediction
            response = self.prediction_engine.predict(request)
            
            # Update stats
            processing_time_ms = (time.time() - start_time) * 1000
            
            with self.lock:
                if deployment_key in self.deployed_models:
                    deployment = self.deployed_models[deployment_key]
                    deployment['request_count'] += 1
                    deployment['total_processing_time_ms'] += processing_time_ms
                
                if deployment_key in self.request_stats:
                    stats = self.request_stats[deployment_key]
                    stats['total_requests'] += 1
                    stats['last_request_time'] = datetime.now()
                    
                    if response.metadata and 'error' not in response.metadata:
                        stats['successful_requests'] += 1
                    else:
                        stats['failed_requests'] += 1
                    
                    # Update average processing time
                    total_time = deployment['total_processing_time_ms']
                    total_requests = deployment['request_count']
                    stats['avg_processing_time_ms'] = total_time / total_requests
            
            # Format response
            result = {
                'request_id': response.request_id,
                'predictions': response.predictions.tolist() if hasattr(response.predictions, 'tolist') else response.predictions,
                'probabilities': response.probabilities.tolist() if response.probabilities is not None else None,
                'confidence_scores': response.confidence_scores.tolist() if response.confidence_scores is not None else None,
                'model_name': response.model_name,
                'processing_time_ms': processing_time_ms,
                'timestamp': response.timestamp.isoformat(),
                'status': 'success' if 'error' not in (response.metadata or {}) else 'error'
            }
            
            if response.metadata and 'error' in response.metadata:
                result['error'] = response.metadata['error']
            
            return result
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update error stats
            with self.lock:
                for key in self.deployed_models:
                    if key.startswith(f"{model_name}:"):
                        if key in self.request_stats:
                            self.request_stats[key]['total_requests'] += 1
                            self.request_stats[key]['failed_requests'] += 1
                            self.request_stats[key]['last_request_time'] = datetime.now()
                        break
            
            logger.error(f"Prediction failed for model {model_name}: {e}")
            
            return {
                'request_id': request_id,
                'status': 'error',
                'error': str(e),
                'processing_time_ms': processing_time_ms,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_status(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get status of deployed models."""
        with self.lock:
            if model_name:
                # Get specific model status
                for key, deployment in self.deployed_models.items():
                    if key.startswith(f"{model_name}:"):
                        stats = self.request_stats.get(key, {})
                        health = self.health_checker.get_health_status(key)
                        
                        return {
                            'model_name': deployment['metadata'].name,
                            'version': deployment['metadata'].version,
                            'status': deployment['status'],
                            'deployed_at': deployment['deployed_at'],
                            'request_count': deployment['request_count'],
                            'error_count': deployment['error_count'],
                            'health_status': health.get('status', 'unknown'),
                            'request_stats': stats
                        }
                
                return {}
            
            else:
                # Get all models status
                models_status = []
                for key, deployment in self.deployed_models.items():
                    stats = self.request_stats.get(key, {})
                    health = self.health_checker.get_health_status(key)
                    
                    models_status.append({
                        'model_name': deployment['metadata'].name,
                        'version': deployment['metadata'].version,
                        'status': deployment['status'],
                        'deployed_at': deployment['deployed_at'],
                        'request_count': deployment['request_count'],
                        'error_count': deployment['error_count'],
                        'health_status': health.get('status', 'unknown'),
                        'request_stats': stats
                    })
                
                return {
                    'models': models_status,
                    'total_models': len(models_status),
                    'health_summary': self.health_checker.get_health_summary(),
                    'server_status': 'running'
                }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        with self.lock:
            total_requests = sum(stats.get('total_requests', 0) 
                               for stats in self.request_stats.values())
            
            successful_requests = sum(stats.get('successful_requests', 0) 
                                    for stats in self.request_stats.values())
            
            failed_requests = sum(stats.get('failed_requests', 0) 
                                for stats in self.request_stats.values())
            
            avg_processing_times = [stats.get('avg_processing_time_ms', 0) 
                                  for stats in self.request_stats.values() if stats.get('avg_processing_time_ms', 0) > 0]
            
            return {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,
                'avg_processing_time_ms': np.mean(avg_processing_times) if avg_processing_times else 0,
                'deployed_models_count': len(self.deployed_models),
                'health_summary': self.health_checker.get_health_summary()
            }


def create_model_server(max_concurrent_requests: int = 100,
                       request_timeout_seconds: float = 30.0,
                       health_check_interval: int = 30) -> ModelServer:
    """Factory function to create model server."""
    config = DeploymentConfig(
        max_concurrent_requests=max_concurrent_requests,
        request_timeout_seconds=request_timeout_seconds,
        health_check_interval_seconds=health_check_interval,
        max_memory_mb=1024,
        max_cpu_percent=80.0,
        auto_scaling_enabled=False,
        min_replicas=1,
        max_replicas=5
    )
    
    return ModelServer(config)
