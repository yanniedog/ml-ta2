"""
SHAP-based Model Interpretability for ML-TA System.

This module implements comprehensive model interpretability using SHAP values,
including feature importance analysis, prediction explanations, and monitoring.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# SHAP library
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    # Mock SHAP for testing
    class shap:
        class Explainer:
            def __init__(self, model, data=None):
                self.model = model
                self.data = data
            
            def shap_values(self, X):
                return np.random.randn(*X.shape)
        
        class TreeExplainer(Explainer):
            pass
        
        class LinearExplainer(Explainer):
            pass

# Plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from src.config import get_config
from src.logging_config import get_logger
from src.exceptions import ModelAnalysisError

logger = get_logger(__name__)


@dataclass
class SHAPAnalysisResult:
    """Results from SHAP analysis."""
    shap_values: np.ndarray
    expected_value: float
    feature_names: List[str]
    feature_importance: Dict[str, float]
    global_importance: Dict[str, float]
    analysis_timestamp: str
    model_name: str
    sample_size: int


class SHAPAnalyzer:
    """Main SHAP analysis class for model interpretability."""
    
    def __init__(self):
        """Initialize SHAP analyzer."""
        self.explainers = {}
        self.analysis_cache = {}
        
        logger.info("SHAPAnalyzer initialized")
    
    def create_explainer(self, model, X_background: pd.DataFrame, 
                        model_type: str = 'tree') -> Any:
        """Create appropriate SHAP explainer for the model."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, using mock explainer")
            return shap.Explainer(model, X_background)
        
        try:
            if model_type in ['lightgbm', 'xgboost', 'catboost', 'random_forest']:
                explainer = shap.TreeExplainer(model)
            elif model_type in ['linear', 'logistic']:
                explainer = shap.LinearExplainer(model, X_background)
            else:
                # General explainer for other models
                explainer = shap.Explainer(model, X_background)
            
            logger.info(f"Created {type(explainer).__name__} for {model_type}")
            return explainer
            
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            raise ModelAnalysisError(f"SHAP explainer creation failed: {e}")
    
    def analyze_model(self, model, X: pd.DataFrame, 
                     model_name: str, model_type: str = 'tree',
                     sample_size: Optional[int] = None) -> SHAPAnalysisResult:
        """Perform comprehensive SHAP analysis on a model."""
        logger.info(f"Starting SHAP analysis for {model_name}")
        
        # Sample data if needed for performance
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X
        
        try:
            # Create explainer
            explainer = self.create_explainer(model, X_sample, model_type)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary
            
            # Get expected value
            if hasattr(explainer, 'expected_value'):
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
            else:
                expected_value = 0.0
            
            # Calculate feature importance
            feature_importance = self._calculate_feature_importance(shap_values, X_sample.columns)
            global_importance = self._calculate_global_importance(shap_values, X_sample.columns)
            
            # Create result
            result = SHAPAnalysisResult(
                shap_values=shap_values,
                expected_value=float(expected_value),
                feature_names=list(X_sample.columns),
                feature_importance=feature_importance,
                global_importance=global_importance,
                analysis_timestamp=pd.Timestamp.now().isoformat(),
                model_name=model_name,
                sample_size=len(X_sample)
            )
            
            # Cache result
            self.analysis_cache[model_name] = result
            
            logger.info(f"SHAP analysis completed for {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            raise ModelAnalysisError(f"SHAP analysis failed: {e}")
    
    def _calculate_feature_importance(self, shap_values: np.ndarray, 
                                    feature_names: List[str]) -> Dict[str, float]:
        """Calculate mean absolute SHAP values for feature importance."""
        importance = np.mean(np.abs(shap_values), axis=0)
        return dict(zip(feature_names, importance.tolist()))
    
    def _calculate_global_importance(self, shap_values: np.ndarray,
                                   feature_names: List[str]) -> Dict[str, float]:
        """Calculate global feature importance (normalized)."""
        importance = np.mean(np.abs(shap_values), axis=0)
        total_importance = np.sum(importance)
        
        if total_importance > 0:
            normalized_importance = importance / total_importance
        else:
            normalized_importance = np.zeros_like(importance)
        
        return dict(zip(feature_names, normalized_importance.tolist()))
    
    def explain_prediction(self, model, X_instance: pd.DataFrame,
                          model_name: str, model_type: str = 'tree') -> Dict[str, Any]:
        """Explain a single prediction using SHAP values."""
        try:
            # Get or create explainer
            if model_name in self.explainers:
                explainer = self.explainers[model_name]
            else:
                explainer = self.create_explainer(model, X_instance, model_type)
                self.explainers[model_name] = explainer
            
            # Calculate SHAP values for instance
            shap_values = explainer.shap_values(X_instance)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            # Get prediction
            prediction = model.predict(X_instance)[0]
            prediction_proba = None
            
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(X_instance)[0]
            
            # Create explanation
            explanation = {
                'prediction': prediction,
                'prediction_probability': prediction_proba.tolist() if prediction_proba is not None else None,
                'shap_values': shap_values[0].tolist(),
                'feature_values': X_instance.iloc[0].to_dict(),
                'feature_contributions': dict(zip(X_instance.columns, shap_values[0])),
                'expected_value': getattr(explainer, 'expected_value', 0.0),
                'model_name': model_name
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Prediction explanation failed: {e}")
            raise ModelAnalysisError(f"Prediction explanation failed: {e}")
    
    def generate_summary_plot_data(self, result: SHAPAnalysisResult,
                                 max_features: int = 20) -> Dict[str, Any]:
        """Generate data for SHAP summary plot."""
        try:
            # Get top features by importance
            sorted_features = sorted(
                result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:max_features]
            
            top_feature_names = [f[0] for f in sorted_features]
            top_feature_indices = [result.feature_names.index(name) for name in top_feature_names]
            
            # Extract SHAP values for top features
            top_shap_values = result.shap_values[:, top_feature_indices]
            
            plot_data = {
                'shap_values': top_shap_values.tolist(),
                'feature_names': top_feature_names,
                'feature_importance': [f[1] for f in sorted_features],
                'expected_value': result.expected_value,
                'sample_size': result.sample_size
            }
            
            return plot_data
            
        except Exception as e:
            logger.error(f"Summary plot data generation failed: {e}")
            return {}
    
    def get_top_features(self, result: SHAPAnalysisResult, 
                        n_features: int = 10) -> List[Tuple[str, float]]:
        """Get top N features by SHAP importance."""
        sorted_features = sorted(
            result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_features[:n_features]
    
    def compare_feature_importance(self, results: Dict[str, SHAPAnalysisResult]) -> pd.DataFrame:
        """Compare feature importance across multiple models."""
        if not results:
            return pd.DataFrame()
        
        # Get all unique features
        all_features = set()
        for result in results.values():
            all_features.update(result.feature_names)
        
        all_features = sorted(list(all_features))
        
        # Create comparison DataFrame
        comparison_data = {}
        for model_name, result in results.items():
            model_importance = []
            for feature in all_features:
                importance = result.feature_importance.get(feature, 0.0)
                model_importance.append(importance)
            comparison_data[model_name] = model_importance
        
        comparison_df = pd.DataFrame(comparison_data, index=all_features)
        return comparison_df


class SHAPMonitor:
    """Monitor SHAP values and feature importance over time."""
    
    def __init__(self):
        """Initialize SHAP monitor."""
        self.importance_history = []
        self.drift_thresholds = {
            'feature_importance': 0.1,  # 10% change threshold
            'top_features': 0.2  # 20% change in top features
        }
        
        logger.info("SHAPMonitor initialized")
    
    def track_importance_changes(self, current_result: SHAPAnalysisResult,
                               previous_result: Optional[SHAPAnalysisResult] = None) -> Dict[str, Any]:
        """Track changes in feature importance over time."""
        if previous_result is None:
            # First analysis, just store
            self.importance_history.append(current_result)
            return {'status': 'baseline_established', 'changes': {}}
        
        try:
            # Calculate importance changes
            changes = {}
            drift_detected = False
            
            # Compare feature importance
            for feature in current_result.feature_names:
                current_imp = current_result.feature_importance.get(feature, 0.0)
                previous_imp = previous_result.feature_importance.get(feature, 0.0)
                
                if previous_imp > 0:
                    change_ratio = abs(current_imp - previous_imp) / previous_imp
                    if change_ratio > self.drift_thresholds['feature_importance']:
                        changes[feature] = {
                            'previous': previous_imp,
                            'current': current_imp,
                            'change_ratio': change_ratio
                        }
                        drift_detected = True
            
            # Compare top features
            current_top = [f[0] for f in sorted(
                current_result.feature_importance.items(),
                key=lambda x: x[1], reverse=True
            )[:10]]
            
            previous_top = [f[0] for f in sorted(
                previous_result.feature_importance.items(),
                key=lambda x: x[1], reverse=True
            )[:10]]
            
            top_features_overlap = len(set(current_top) & set(previous_top)) / len(current_top)
            
            if top_features_overlap < (1 - self.drift_thresholds['top_features']):
                drift_detected = True
                changes['top_features_change'] = {
                    'overlap_ratio': top_features_overlap,
                    'current_top': current_top,
                    'previous_top': previous_top
                }
            
            # Store current result
            self.importance_history.append(current_result)
            
            result = {
                'status': 'drift_detected' if drift_detected else 'stable',
                'changes': changes,
                'drift_detected': drift_detected,
                'analysis_timestamp': current_result.analysis_timestamp
            }
            
            if drift_detected:
                logger.warning("Feature importance drift detected", changes=changes)
            
            return result
            
        except Exception as e:
            logger.error(f"Importance change tracking failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_importance_trends(self, feature_name: str) -> Dict[str, Any]:
        """Get importance trend for a specific feature."""
        if not self.importance_history:
            return {}
        
        timestamps = []
        importance_values = []
        
        for result in self.importance_history:
            timestamps.append(result.analysis_timestamp)
            importance_values.append(result.feature_importance.get(feature_name, 0.0))
        
        return {
            'feature_name': feature_name,
            'timestamps': timestamps,
            'importance_values': importance_values,
            'trend': 'increasing' if importance_values[-1] > importance_values[0] else 'decreasing'
        }


class SHAPReports:
    """Generate comprehensive SHAP analysis reports."""
    
    def __init__(self):
        """Initialize SHAP reports generator."""
        self.analyzer = SHAPAnalyzer()
        
    def generate_model_interpretability_report(self, 
                                             results: Dict[str, SHAPAnalysisResult]) -> Dict[str, Any]:
        """Generate comprehensive interpretability report."""
        if not results:
            return {'error': 'No SHAP results available'}
        
        report = {
            'summary': {
                'models_analyzed': len(results),
                'total_features': len(next(iter(results.values())).feature_names),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            },
            'model_comparison': {},
            'feature_analysis': {},
            'recommendations': []
        }
        
        # Model comparison
        comparison_df = self.analyzer.compare_feature_importance(results)
        if not comparison_df.empty:
            report['model_comparison'] = {
                'feature_importance_correlation': comparison_df.corr().to_dict(),
                'top_features_by_model': {
                    model: comparison_df[model].nlargest(10).to_dict()
                    for model in comparison_df.columns
                }
            }
        
        # Feature analysis
        all_features = set()
        for result in results.values():
            all_features.update(result.feature_names)
        
        for feature in all_features:
            feature_importance_across_models = {}
            for model_name, result in results.items():
                feature_importance_across_models[model_name] = result.feature_importance.get(feature, 0.0)
            
            report['feature_analysis'][feature] = {
                'importance_across_models': feature_importance_across_models,
                'average_importance': np.mean(list(feature_importance_across_models.values())),
                'importance_std': np.std(list(feature_importance_across_models.values()))
            }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(results, report)
        
        return report
    
    def _generate_recommendations(self, results: Dict[str, SHAPAnalysisResult],
                                report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on SHAP analysis."""
        recommendations = []
        
        # Find consistently important features
        if 'feature_analysis' in report:
            important_features = [
                feature for feature, analysis in report['feature_analysis'].items()
                if analysis['average_importance'] > 0.01  # Threshold
            ]
            
            if important_features:
                recommendations.append(
                    f"Focus on top {len(important_features)} consistently important features: "
                    f"{', '.join(important_features[:5])}{'...' if len(important_features) > 5 else ''}"
                )
        
        # Check for feature redundancy
        if 'model_comparison' in report and 'feature_importance_correlation' in report['model_comparison']:
            correlations = report['model_comparison']['feature_importance_correlation']
            high_correlations = []
            
            models = list(correlations.keys())
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    if correlations[model1][model2] > 0.9:
                        high_correlations.append((model1, model2))
            
            if high_correlations:
                recommendations.append(
                    "High correlation in feature importance between models suggests "
                    "consistent feature selection across algorithms"
                )
        
        # Model-specific recommendations
        for model_name, result in results.items():
            top_features = sorted(result.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            
            recommendations.append(
                f"{model_name}: Top features are {', '.join([f[0] for f in top_features])}"
            )
        
        return recommendations


# Factory functions
def create_shap_analyzer() -> SHAPAnalyzer:
    """Factory function to create SHAP analyzer."""
    return SHAPAnalyzer()


def create_shap_monitor() -> SHAPMonitor:
    """Factory function to create SHAP monitor."""
    return SHAPMonitor()


if __name__ == '__main__':
    # Example usage
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate sample data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])
    y = np.random.randint(0, 2, 1000)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Analyze with SHAP
    analyzer = create_shap_analyzer()
    result = analyzer.analyze_model(model, X, 'test_model', 'random_forest')
    
    print(f"Top 5 features: {analyzer.get_top_features(result, 5)}")
    
    # Explain single prediction
    explanation = analyzer.explain_prediction(model, X.head(1), 'test_model', 'random_forest')
    print(f"Prediction: {explanation['prediction']}")
    print(f"Top contributions: {sorted(explanation['feature_contributions'].items(), key=lambda x: abs(x[1]), reverse=True)[:3]}")
