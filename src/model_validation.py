"""
Model validation framework with comprehensive testing and evaluation.

This module implements:
- Cross-validation strategies for time series data
- Model performance evaluation and comparison
- Ensemble methods and model selection
- Model interpretability and feature analysis
- Validation reporting and visualization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML libraries
try:
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
from src.models import ModelTrainer, ModelMetrics

logger = get_logger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    cv_folds: int = 5
    test_size: float = 0.2
    validation_size: float = 0.2
    time_series_split: bool = True
    shuffle: bool = False
    random_state: int = 42
    scoring_metrics: List[str] = None
    
    def __post_init__(self):
        if self.scoring_metrics is None:
            self.scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']


class ModelValidator:
    """Comprehensive model validation and evaluation system."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """Initialize model validator."""
        self.config = config or ValidationConfig()
        self.validation_results = {}
        self.comparison_results = {}
        self.feature_analysis = {}
        
        logger.info("ModelValidator initialized", 
                   cv_folds=self.config.cv_folds,
                   test_size=self.config.test_size)
    
    def validate_model(self, trainer: ModelTrainer, X: pd.DataFrame, y: pd.Series,
                      model_name: str = "default") -> Dict[str, Any]:
        """Perform comprehensive validation of a single model."""
        start_time = datetime.now()
        
        try:
            logger.info("Starting model validation", 
                       model_name=model_name,
                       data_shape=X.shape)
            
            results = {
                'model_name': model_name,
                'model_type': trainer.config.model_type,
                'task_type': trainer.config.task_type,
                'validation_timestamp': start_time.isoformat(),
                'data_info': {
                    'n_samples': len(X),
                    'n_features': len(X.columns),
                    'feature_names': list(X.columns)
                }
            }
            
            # Time series cross-validation
            cv_results = self._perform_cross_validation(trainer, X, y)
            results['cross_validation'] = cv_results
            
            # Train-test split validation
            holdout_results = self._perform_holdout_validation(trainer, X, y)
            results['holdout_validation'] = holdout_results
            
            # Feature importance analysis
            feature_analysis = self._analyze_feature_importance(trainer, X, model_name)
            results['feature_analysis'] = feature_analysis
            
            # Model interpretability
            if SHAP_AVAILABLE:
                shap_analysis = self._perform_shap_analysis(trainer, X, model_name)
                results['shap_analysis'] = shap_analysis
            
            # Performance summary
            performance_summary = self._create_performance_summary(results)
            results['performance_summary'] = performance_summary
            
            # Store results
            self.validation_results[model_name] = results
            
            validation_time = (datetime.now() - start_time).total_seconds()
            logger.info("Model validation completed",
                       model_name=model_name,
                       validation_time=validation_time,
                       cv_score=cv_results.get('mean_score', 0))
            
            return results
            
        except Exception as e:
            logger.error(f"Model validation failed for {model_name}: {e}")
            raise
    
    def _perform_cross_validation(self, trainer: ModelTrainer, 
                                 X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform time series cross-validation."""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("Sklearn not available, skipping cross-validation")
                return {'error': 'sklearn_not_available'}
            
            # Create time series split
            if self.config.time_series_split:
                cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            else:
                from sklearn.model_selection import KFold
                cv = KFold(n_splits=self.config.cv_folds, 
                          shuffle=self.config.shuffle,
                          random_state=self.config.random_state)
            
            # Get model for cross-validation
            model_class = trainer.model_registry[trainer.config.task_type].get(trainer.config.model_type)
            if model_class is None:
                model_class = trainer.model_registry[trainer.config.task_type]['random_forest']
            
            model = model_class(random_state=trainer.config.random_state)
            
            # Perform cross-validation for different metrics
            cv_results = {}
            
            for metric in self.config.scoring_metrics:
                try:
                    if trainer.config.task_type == 'classification':
                        if metric in ['accuracy', 'precision', 'recall', 'f1']:
                            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                            cv_results[metric] = {
                                'scores': scores.tolist(),
                                'mean': scores.mean(),
                                'std': scores.std(),
                                'min': scores.min(),
                                'max': scores.max()
                            }
                    else:  # regression
                        if metric in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
                            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                            cv_results[metric] = {
                                'scores': scores.tolist(),
                                'mean': scores.mean(),
                                'std': scores.std(),
                                'min': scores.min(),
                                'max': scores.max()
                            }
                except Exception as e:
                    logger.warning(f"Cross-validation failed for metric {metric}: {e}")
            
            # Overall summary
            primary_metric = 'accuracy' if trainer.config.task_type == 'classification' else 'r2'
            if primary_metric in cv_results:
                cv_results['mean_score'] = cv_results[primary_metric]['mean']
                cv_results['std_score'] = cv_results[primary_metric]['std']
            
            return cv_results
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {'error': str(e)}
    
    def _perform_holdout_validation(self, trainer: ModelTrainer,
                                   X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform holdout validation with train-test split."""
        try:
            # Split data
            split_idx = int((1 - self.config.test_size) * len(X))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Train model
            model, metrics = trainer.train_model(X_train, y_train, "validation_model")
            
            # Make predictions on test set
            y_pred = trainer.predict("validation_model", X_test)
            
            # Calculate detailed metrics
            holdout_results = {
                'train_size': len(X_train),
                'test_size': len(X_test),
                'training_time': metrics.training_time,
                'prediction_time': metrics.prediction_time
            }
            
            if trainer.config.task_type == 'classification':
                if SKLEARN_AVAILABLE:
                    # Classification metrics
                    holdout_results.update({
                        'accuracy': float(np.mean(y_test == y_pred)),
                        'classification_report': classification_report(y_test, y_pred, output_dict=True),
                        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
                    })
                else:
                    holdout_results['accuracy'] = float(np.mean(y_test == y_pred))
            
            else:  # regression
                if SKLEARN_AVAILABLE:
                    holdout_results.update({
                        'mse': float(mean_squared_error(y_test, y_pred)),
                        'mae': float(mean_absolute_error(y_test, y_pred)),
                        'r2_score': float(r2_score(y_test, y_pred))
                    })
                else:
                    # Fallback calculations
                    mse = np.mean((y_test - y_pred) ** 2)
                    mae = np.mean(np.abs(y_test - y_pred))
                    ss_res = np.sum((y_test - y_pred) ** 2)
                    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    holdout_results.update({
                        'mse': float(mse),
                        'mae': float(mae),
                        'r2_score': float(r2)
                    })
            
            return holdout_results
            
        except Exception as e:
            logger.error(f"Holdout validation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_feature_importance(self, trainer: ModelTrainer, 
                                   X: pd.DataFrame, model_name: str) -> Dict[str, Any]:
        """Analyze feature importance for the model."""
        try:
            if model_name not in trainer.models:
                return {'error': 'model_not_found'}
            
            model = trainer.models[model_name]
            
            # Extract the actual model from pipeline
            if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                actual_model = model.named_steps['model']
            else:
                actual_model = model
            
            feature_analysis = {
                'feature_names': list(X.columns),
                'n_features': len(X.columns)
            }
            
            # Get feature importance if available
            if hasattr(actual_model, 'feature_importances_'):
                importance = actual_model.feature_importances_
                
                # Create feature importance dictionary
                feature_importance = dict(zip(X.columns, importance))
                
                # Sort by importance
                sorted_features = sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
                
                feature_analysis.update({
                    'feature_importance': feature_importance,
                    'top_features': sorted_features[:10],
                    'bottom_features': sorted_features[-5:],
                    'importance_stats': {
                        'mean': float(np.mean(importance)),
                        'std': float(np.std(importance)),
                        'min': float(np.min(importance)),
                        'max': float(np.max(importance))
                    }
                })
                
                logger.info("Feature importance analyzed",
                           model_name=model_name,
                           top_feature=sorted_features[0][0] if sorted_features else None)
            
            # Store for later use
            self.feature_analysis[model_name] = feature_analysis
            
            return feature_analysis
            
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            return {'error': str(e)}
    
    def _perform_shap_analysis(self, trainer: ModelTrainer, 
                              X: pd.DataFrame, model_name: str,
                              max_samples: int = 100) -> Dict[str, Any]:
        """Perform SHAP analysis for model interpretability."""
        if not SHAP_AVAILABLE:
            return {'error': 'shap_not_available'}
        
        try:
            if model_name not in trainer.models:
                return {'error': 'model_not_found'}
            
            model = trainer.models[model_name]
            
            # Sample data if too large
            X_sample = X.sample(n=min(max_samples, len(X)), 
                               random_state=self.config.random_state)
            
            # Create SHAP explainer
            explainer = shap.Explainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Calculate SHAP statistics
            if isinstance(shap_values, list):
                # Multi-class classification
                shap_mean = [np.mean(np.abs(sv), axis=0) for sv in shap_values]
                shap_analysis = {
                    'shap_values_shape': [sv.shape for sv in shap_values],
                    'mean_shap_values': [sv.tolist() for sv in shap_mean],
                    'feature_names': list(X.columns),
                    'sample_size': len(X_sample),
                    'analysis_type': 'multi_class'
                }
            else:
                # Binary classification or regression
                shap_mean = np.mean(np.abs(shap_values), axis=0)
                
                # Create feature importance from SHAP
                shap_importance = dict(zip(X.columns, shap_mean))
                sorted_shap = sorted(shap_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
                
                shap_analysis = {
                    'shap_values_shape': shap_values.shape,
                    'mean_shap_values': shap_mean.tolist(),
                    'shap_importance': shap_importance,
                    'top_shap_features': sorted_shap[:10],
                    'feature_names': list(X.columns),
                    'sample_size': len(X_sample),
                    'analysis_type': 'binary_or_regression'
                }
            
            logger.info("SHAP analysis completed",
                       model_name=model_name,
                       sample_size=len(X_sample))
            
            return shap_analysis
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return {'error': str(e)}
    
    def _create_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive performance summary."""
        summary = {
            'model_name': results['model_name'],
            'model_type': results['model_type'],
            'task_type': results['task_type'],
            'validation_timestamp': results['validation_timestamp']
        }
        
        # Cross-validation summary
        cv_results = results.get('cross_validation', {})
        if 'mean_score' in cv_results:
            summary['cv_score'] = cv_results['mean_score']
            summary['cv_std'] = cv_results['std_score']
        
        # Holdout validation summary
        holdout_results = results.get('holdout_validation', {})
        if results['task_type'] == 'classification':
            summary['test_accuracy'] = holdout_results.get('accuracy')
        else:
            summary['test_r2'] = holdout_results.get('r2_score')
            summary['test_mse'] = holdout_results.get('mse')
        
        summary['training_time'] = holdout_results.get('training_time')
        summary['prediction_time'] = holdout_results.get('prediction_time')
        
        # Feature analysis summary
        feature_analysis = results.get('feature_analysis', {})
        if 'top_features' in feature_analysis:
            summary['top_features'] = feature_analysis['top_features'][:5]
        
        return summary
    
    def compare_models(self, model_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple model validation results."""
        try:
            comparison_data = []
            
            for result in model_results:
                summary = result.get('performance_summary', {})
                
                row = {
                    'model_name': summary.get('model_name', 'unknown'),
                    'model_type': summary.get('model_type', 'unknown'),
                    'cv_score': summary.get('cv_score'),
                    'cv_std': summary.get('cv_std'),
                    'training_time': summary.get('training_time'),
                    'prediction_time': summary.get('prediction_time')
                }
                
                # Add task-specific metrics
                if summary.get('task_type') == 'classification':
                    row['test_accuracy'] = summary.get('test_accuracy')
                else:
                    row['test_r2'] = summary.get('test_r2')
                    row['test_mse'] = summary.get('test_mse')
                
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Store comparison results
            self.comparison_results = {
                'comparison_df': comparison_df,
                'timestamp': datetime.now().isoformat(),
                'n_models': len(model_results)
            }
            
            logger.info("Model comparison completed",
                       n_models=len(model_results))
            
            return comparison_df
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return pd.DataFrame()
    
    def get_best_model(self, model_results: List[Dict[str, Any]], 
                      metric: str = 'auto') -> Optional[Dict[str, Any]]:
        """Identify the best performing model based on specified metric."""
        if not model_results:
            return None
        
        try:
            # Determine metric if auto
            if metric == 'auto':
                task_type = model_results[0].get('task_type', 'classification')
                metric = 'cv_score' if task_type == 'classification' else 'cv_score'
            
            best_model = None
            best_score = float('-inf')
            
            for result in model_results:
                summary = result.get('performance_summary', {})
                score = summary.get(metric)
                
                if score is not None and score > best_score:
                    best_score = score
                    best_model = result
            
            if best_model:
                logger.info("Best model identified",
                           model_name=best_model['model_name'],
                           metric=metric,
                           score=best_score)
            
            return best_model
            
        except Exception as e:
            logger.error(f"Best model selection failed: {e}")
            return None
    
    def generate_validation_report(self, model_name: str) -> Dict[str, Any]:
        """Generate comprehensive validation report for a model."""
        if model_name not in self.validation_results:
            return {'error': 'model_not_validated'}
        
        result = self.validation_results[model_name]
        
        report = {
            'model_info': {
                'name': result['model_name'],
                'type': result['model_type'],
                'task': result['task_type'],
                'validation_date': result['validation_timestamp']
            },
            'data_info': result['data_info'],
            'performance': result['performance_summary'],
            'cross_validation': result.get('cross_validation', {}),
            'holdout_validation': result.get('holdout_validation', {}),
            'feature_analysis': result.get('feature_analysis', {}),
            'interpretability': result.get('shap_analysis', {})
        }
        
        logger.info("Validation report generated", model_name=model_name)
        
        return report


def create_model_validator(cv_folds: int = 5, test_size: float = 0.2) -> ModelValidator:
    """Factory function to create model validator."""
    config = ValidationConfig(
        cv_folds=cv_folds,
        test_size=test_size,
        validation_size=0.2,
        time_series_split=True,
        shuffle=False,
        random_state=42,
        scoring_metrics=['accuracy', 'precision', 'recall', 'f1']
    )
    
    return ModelValidator(config)
