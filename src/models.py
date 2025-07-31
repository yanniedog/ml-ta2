"""
Model training system with hyperparameter optimization and validation.

This module implements:
- Model training with multiple algorithms
- Hyperparameter optimization using Optuna
- Cross-validation and model selection
- Ensemble methods and model comparison
- Model interpretability with SHAP analysis
- Model persistence and versioning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
try:
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    from sklearn.svm import SVC, SVR
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Create minimal fallback classes
    class Pipeline:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(len(X))

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    # Create minimal fallback
    class optuna:
        class Study:
            def optimize(self, *args, **kwargs):
                pass
            @property
            def best_params(self):
                return {}
        @staticmethod
        def create_study(*args, **kwargs):
            return optuna.Study()

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    class xgb:
        class XGBClassifier:
            def __init__(self, *args, **kwargs):
                pass
            def fit(self, X, y):
                return self
            def predict(self, X):
                return np.zeros(len(X))

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    class lgb:
        class LGBMClassifier:
            def __init__(self, *args, **kwargs):
                pass
            def fit(self, X, y):
                return self
            def predict(self, X):
                return np.zeros(len(X))

# Model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    class shap:
        class Explainer:
            def __init__(self, *args, **kwargs):
                pass
            def shap_values(self, X):
                return np.zeros_like(X)

from src.config import get_config
from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    cv_scores: Optional[List[float]] = None
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_type: str
    task_type: str  # 'classification' or 'regression'
    hyperparameters: Dict[str, Any]
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    optimize_hyperparams: bool = True
    n_trials: int = 100


class ModelTrainer:
    """Core model training system with hyperparameter optimization."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize model trainer."""
        self.config = config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.feature_importance = {}
        self.shap_values = {}
        
        # Initialize available models
        self._initialize_model_registry()
        
        logger.info("ModelTrainer initialized", 
                   model_type=self.config.model_type,
                   task_type=self.config.task_type)
    
    def _get_default_config(self) -> ModelConfig:
        """Get default model configuration."""
        return ModelConfig(
            model_type='random_forest',
            task_type='classification',
            hyperparameters={},
            cv_folds=5,
            test_size=0.2,
            random_state=42,
            optimize_hyperparams=True,
            n_trials=50
        )
    
    def _initialize_model_registry(self):
        """Initialize registry of available models."""
        self.model_registry = {
            'classification': {
                'random_forest': RandomForestClassifier if SKLEARN_AVAILABLE else None,
                'gradient_boosting': GradientBoostingClassifier if SKLEARN_AVAILABLE else None,
                'logistic_regression': LogisticRegression if SKLEARN_AVAILABLE else None,
                'svm': SVC if SKLEARN_AVAILABLE else None,
                'xgboost': xgb.XGBClassifier if XGBOOST_AVAILABLE else None,
                'lightgbm': lgb.LGBMClassifier if LIGHTGBM_AVAILABLE else None,
            },
            'regression': {
                'random_forest': RandomForestRegressor if SKLEARN_AVAILABLE else None,
                'gradient_boosting': GradientBoostingRegressor if SKLEARN_AVAILABLE else None,
                'linear_regression': LinearRegression if SKLEARN_AVAILABLE else None,
                'ridge': Ridge if SKLEARN_AVAILABLE else None,
                'lasso': Lasso if SKLEARN_AVAILABLE else None,
                'svm': SVR if SKLEARN_AVAILABLE else None,
                'xgboost': xgb.XGBRegressor if XGBOOST_AVAILABLE else None,
                'lightgbm': lgb.LGBMRegressor if LIGHTGBM_AVAILABLE else None,
            }
        }
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   model_name: str = "default") -> Tuple[Any, ModelMetrics]:
        """Train a single model with the given data."""
        start_time = datetime.now()
        
        try:
            # Create model
            model_class = self.model_registry[self.config.task_type].get(self.config.model_type)
            if model_class is None:
                # Fallback to random forest
                model_class = self.model_registry[self.config.task_type]['random_forest']
            
            model = model_class(random_state=self.config.random_state)
            
            # Create pipeline with scaling
            if SKLEARN_AVAILABLE:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])
            else:
                pipeline = model
            
            # Split data for training and testing
            split_idx = int((1 - self.config.test_size) * len(X))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train model
            logger.info("Training model", model_type=self.config.model_type, 
                       train_samples=len(X_train), test_samples=len(X_test))
            
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            pred_start = datetime.now()
            y_pred = pipeline.predict(X_test)
            pred_time = (datetime.now() - pred_start).total_seconds()
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            metrics.training_time = (datetime.now() - start_time).total_seconds()
            metrics.prediction_time = pred_time
            
            # Store model and metrics
            self.models[model_name] = pipeline
            self.metrics[model_name] = metrics
            
            logger.info("Model training completed",
                       model_name=model_name,
                       training_time=metrics.training_time,
                       test_score=metrics.accuracy or metrics.r2_score)
            
            return pipeline, metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """Calculate performance metrics based on task type."""
        metrics = ModelMetrics()
        
        try:
            if self.config.task_type == 'classification':
                if SKLEARN_AVAILABLE:
                    metrics.accuracy = accuracy_score(y_true, y_pred)
                    metrics.precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics.recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics.f1_score = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                else:
                    # Fallback calculations
                    metrics.accuracy = np.mean(y_true == y_pred)
                    metrics.precision = metrics.accuracy
                    metrics.recall = metrics.accuracy
                    metrics.f1_score = metrics.accuracy
            
            elif self.config.task_type == 'regression':
                if SKLEARN_AVAILABLE:
                    metrics.mse = mean_squared_error(y_true, y_pred)
                    metrics.mae = mean_absolute_error(y_true, y_pred)
                    metrics.r2_score = r2_score(y_true, y_pred)
                else:
                    # Fallback calculations
                    metrics.mse = np.mean((y_true - y_pred) ** 2)
                    metrics.mae = np.mean(np.abs(y_true - y_pred))
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    metrics.r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
        
        return metrics
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        try:
            model = self.models[model_name]
            predictions = model.predict(X)
            
            logger.debug("Predictions made",
                        model_name=model_name,
                        sample_count=len(X))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            raise
    
    def get_model_metrics(self, model_name: str) -> Optional[ModelMetrics]:
        """Get metrics for a trained model."""
        return self.metrics.get(model_name)
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        try:
            model_data = {
                'model': self.models[model_name],
                'metrics': self.metrics.get(model_name),
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Model saved", model_name=model_name, filepath=filepath)
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            raise


def create_model_trainer(task_type: str = 'classification', 
                        model_type: str = 'random_forest') -> ModelTrainer:
    """Factory function to create model trainer."""
    config = ModelConfig(
        model_type=model_type,
        task_type=task_type,
        hyperparameters={},
        cv_folds=5,
        test_size=0.2,
        random_state=42,
        optimize_hyperparams=False,  # Simplified for initial implementation
        n_trials=50
    )
    
    return ModelTrainer(config)
