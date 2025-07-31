"""
Model Analysis Framework for ML-TA System.

This module implements comprehensive model analysis and interpretability:
1. Feature importance analysis across different model types
2. Partial dependence plots for understanding feature relationships
3. Model performance metrics and diagnostics
4. Model comparison utilities
5. Integration with SHAP analysis for unified interpretability

This complements the specific SHAP analysis provided in shap_analysis.py.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML libraries
try:
    import sklearn
    from sklearn.inspection import permutation_importance, partial_dependence
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, r2_score,
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Local imports
from src.config import get_config
from src.logging_config import get_logger
from src.exceptions import ModelAnalysisError

# Import SHAP analyzer if available
try:
    from src.shap_analysis import SHAPAnalyzer, create_shap_analyzer, SHAPAnalysisResult
    SHAP_MODULE_AVAILABLE = True
except ImportError:
    SHAP_MODULE_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class ModelAnalysisConfig:
    """Configuration for model analysis."""
    n_permutation_repeats: int = 10
    permutation_random_state: int = 42
    partial_dependence_grid_resolution: int = 20
    feature_importance_top_k: int = 20
    store_analysis_results: bool = True
    artifacts_path: str = "artefacts/model_analysis"


@dataclass
class ModelAnalysisResult:
    """Results from model analysis."""
    model_name: str
    model_type: str
    analysis_timestamp: str
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    analysis_config: Dict[str, Any]
    shap_analysis: Optional[Dict[str, Any]] = None


class ModelAnalyzer:
    """Main model analysis class for interpretability and diagnostics."""
    
    def __init__(self, config: Optional[ModelAnalysisConfig] = None):
        """Initialize model analyzer."""
        self.config = config or ModelAnalysisConfig()
        self.analysis_results = {}
        self.comparison_results = {}
        
        # Create artifacts directory if storing results
        if self.config.store_analysis_results:
            os.makedirs(Path(self.config.artifacts_path), exist_ok=True)
        
        # Initialize SHAP analyzer if available
        self.shap_analyzer = create_shap_analyzer() if SHAP_MODULE_AVAILABLE else None
        
        logger.info("ModelAnalyzer initialized", 
                   n_permutation_repeats=self.config.n_permutation_repeats,
                   shap_available=self.shap_analyzer is not None)
    
    def analyze_model(self, model, X: pd.DataFrame, y: Optional[pd.Series] = None,
                     model_name: str = "default", model_type: str = "unknown",
                     task_type: str = "regression") -> ModelAnalysisResult:
        """
        Perform comprehensive model analysis.
        
        Args:
            model: Trained model instance
            X: Feature matrix used for analysis
            y: Target values (optional, for performance metrics)
            model_name: Name identifier for the model
            model_type: Type of model (e.g. "lightgbm", "xgboost", "linear")
            task_type: "regression" or "classification"
            
        Returns:
            ModelAnalysisResult with comprehensive analysis data
        """
        start_time = datetime.now()
        logger.info(f"Starting model analysis for {model_name}", 
                   model_type=model_type,
                   task_type=task_type)
        
        # Initialize result container
        analysis_result = {
            "model_name": model_name,
            "model_type": model_type,
            "task_type": task_type,
            "analysis_timestamp": start_time.isoformat(),
            "data_info": {
                "n_samples": len(X),
                "n_features": len(X.columns),
                "feature_names": list(X.columns)
            },
            "performance_metrics": {},
            "feature_importance": {},
            "partial_dependence": {},
            "analysis_config": self.config.__dict__
        }
        
        # Calculate model predictions if y is provided
        if y is not None:
            try:
                y_pred = model.predict(X)
                analysis_result["performance_metrics"] = self._calculate_metrics(
                    y, y_pred, task_type
                )
            except Exception as e:
                logger.warning(f"Performance metrics calculation failed: {e}")
        
        # Calculate feature importance
        try:
            feature_importance = self._calculate_feature_importance(
                model, X, model_type
            )
            analysis_result["feature_importance"] = feature_importance
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
        
        # Calculate partial dependence for top features
        if SKLEARN_AVAILABLE:
            try:
                top_features = sorted(
                    analysis_result["feature_importance"].items(),
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )[:5]  # Only top 5 features
                
                for feature_name, importance in top_features:
                    feature_idx = list(X.columns).index(feature_name)
                    pdp_result = self._calculate_partial_dependence(
                        model, X, feature_idx
                    )
                    analysis_result["partial_dependence"][feature_name] = pdp_result
            except Exception as e:
                logger.warning(f"Partial dependence calculation failed: {e}")
        
        # Run SHAP analysis if available
        if self.shap_analyzer and model_type in ["lightgbm", "xgboost", "catboost", "random_forest"]:
            try:
                shap_result = self.shap_analyzer.analyze_model(
                    model, X, model_name, model_type
                )
                analysis_result["shap_analysis"] = {
                    "expected_value": float(shap_result.expected_value),
                    "feature_importance": shap_result.feature_importance,
                    "global_importance": shap_result.global_importance
                }
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
        
        # Store results
        result_obj = ModelAnalysisResult(
            model_name=model_name,
            model_type=model_type,
            analysis_timestamp=start_time.isoformat(),
            performance_metrics=analysis_result["performance_metrics"],
            feature_importance=analysis_result["feature_importance"],
            analysis_config=self.config.__dict__,
            shap_analysis=analysis_result.get("shap_analysis")
        )
        
        self.analysis_results[model_name] = result_obj
        
        # Persist results if configured
        if self.config.store_analysis_results:
            self._persist_analysis(analysis_result)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Model analysis completed in {duration:.2f}s")
        
        return result_obj
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                         task_type: str) -> Dict[str, float]:
        """Calculate performance metrics based on task type."""
        metrics = {}
        
        if task_type == "regression":
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["r2"] = float(r2_score(y_true, y_pred))
            
        elif task_type == "classification":
            # For binary classification
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            
            try:
                # These might fail for multiclass
                metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted"))
                metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted"))
                metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted"))
            
                # Only for binary classification
                if len(np.unique(y_true)) == 2:
                    # Some models don't have predict_proba
                    try:
                        y_prob = model.predict_proba(X)[:, 1]
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
                    except:
                        pass
            except Exception as e:
                logger.warning(f"Some classification metrics failed: {e}")
                
        return metrics
    
    def _calculate_feature_importance(self, model, X: pd.DataFrame, 
                                   model_type: str) -> Dict[str, float]:
        """Calculate feature importance based on model type."""
        feature_importance = {}
        
        # Try native feature importance if available
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for i, col in enumerate(X.columns):
                feature_importance[col] = float(importances[i])
        
        # Try coef_ for linear models
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            # Handle multi-class models
            if len(coefs.shape) > 1:
                coefs = np.abs(coefs).mean(axis=0)
            for i, col in enumerate(X.columns):
                feature_importance[col] = float(coefs[i])
        
        # Use permutation importance as fallback
        else:
            if SKLEARN_AVAILABLE:
                perm_importance = permutation_importance(
                    model, X, model.predict(X),  # Use predictions as target 
                    n_repeats=self.config.n_permutation_repeats,
                    random_state=self.config.permutation_random_state
                )
                
                for i, col in enumerate(X.columns):
                    feature_importance[col] = float(perm_importance.importances_mean[i])
        
        # Normalize values to [0, 1] range
        if feature_importance:
            max_importance = max(abs(v) for v in feature_importance.values())
            if max_importance > 0:
                feature_importance = {
                    k: float(v / max_importance) 
                    for k, v in feature_importance.items()
                }
        
        return feature_importance
    
    def _calculate_partial_dependence(self, model, X: pd.DataFrame, 
                                   feature_idx: int) -> Dict[str, List[float]]:
        """Calculate partial dependence for a feature."""
        if not SKLEARN_AVAILABLE:
            return {}
            
        try:
            # For sklearn ≥0.22
            pdp_result = partial_dependence(
                model, X, [feature_idx], 
                grid_resolution=self.config.partial_dependence_grid_resolution
            )
            
            # Handle different scikit-learn versions
            if isinstance(pdp_result, tuple):  # Older versions
                grid_values = pdp_result[1][0].tolist()
                pdp_values = pdp_result[0][0].tolist()
            else:  # Newer versions
                grid_values = pdp_result["values"][0].tolist()
                pdp_values = pdp_result["average"][0].tolist()
                
            return {
                "grid_values": grid_values,
                "pdp_values": pdp_values
            }
            
        except Exception as e:
            logger.warning(f"Partial dependence calculation failed: {e}")
            return {}
    
    def _persist_analysis(self, analysis_result: Dict[str, Any]) -> None:
        """Persist analysis results to disk."""
        model_name = analysis_result["model_name"]
        timestamp = analysis_result["analysis_timestamp"].replace(":", "-")
        
        # Create directory if needed
        output_dir = Path(self.config.artifacts_path) / model_name
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save JSON result
        output_path = output_dir / f"analysis_{timestamp}.json"
        
        # Convert numpy types to Python native types
        def convert_for_json(obj):
            if isinstance(obj, (np.int8, np.int16, np.int32,
                               np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            else:
                return obj
        
        with open(output_path, "w") as f:
            # Convert numpy values to Python native types for JSON serialization
            json_ready = {
                k: (
                    {kk: convert_for_json(vv) for kk, vv in v.items()}
                    if isinstance(v, dict) else convert_for_json(v)
                )
                for k, v in analysis_result.items()
            }
            json.dump(json_ready, f, indent=2)
        
        logger.info(f"Analysis results saved to {output_path}")
    
    def generate_model_diagnostics(self, model_name: str) -> Dict[str, Any]:
        """Generate comprehensive diagnostics for a model."""
        if model_name not in self.analysis_results:
            return {"error": "model_not_found"}
        
        result = self.analysis_results[model_name]
        
        # Basic diagnostics report
        diagnostics = {
            "model_info": {
                "name": result.model_name,
                "type": result.model_type,
                "analysis_time": result.analysis_timestamp
            },
            "performance": result.performance_metrics,
            "feature_importance": {
                k: v for k, v in sorted(
                    result.feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:self.config.feature_importance_top_k]
            }
        }
        
        # Add SHAP-based diagnostics if available
        if result.shap_analysis:
            diagnostics["shap_analysis"] = {
                "expected_value": result.shap_analysis["expected_value"],
                "global_importance": result.shap_analysis["global_importance"]
            }
        
        # Generate warnings and recommendations
        diagnostics["warnings"] = []
        diagnostics["recommendations"] = []
        
        # Performance warnings
        if "r2" in result.performance_metrics:
            r2 = result.performance_metrics["r2"]
            if r2 < 0.3:
                diagnostics["warnings"].append(
                    f"Low R² score ({r2:.2f}) indicates poor model fit"
                )
            elif r2 < 0.5:
                diagnostics["warnings"].append(
                    f"Moderate R² score ({r2:.2f}) suggests room for improvement"
                )
        
        # Feature importance warnings
        top_features = sorted(
            result.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        if len(top_features) > 0:
            top_importance = top_features[0][1]
            if top_importance > 0.5:
                diagnostics["warnings"].append(
                    f"Feature '{top_features[0][0]}' dominates with {top_importance:.2f} importance"
                )
                diagnostics["recommendations"].append(
                    f"Consider whether feature '{top_features[0][0]}' might be causing leakage"
                )
        
        logger.info(f"Generated diagnostics for {model_name}")
        return diagnostics
    
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """Compare multiple models based on their analysis results."""
        # Ensure all requested models have been analyzed
        missing_models = [name for name in model_names 
                         if name not in self.analysis_results]
        if missing_models:
            logger.warning(f"Some models not found: {missing_models}")
            model_names = [name for name in model_names 
                          if name in self.analysis_results]
        
        if not model_names:
            return {"error": "no_valid_models_to_compare"}
        
        # Get result objects
        results = [self.analysis_results[name] for name in model_names]
        
        # Build comparison structure
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "models_compared": model_names,
            "metrics_comparison": {},
            "feature_importance_comparison": {},
            "recommendations": []
        }
        
        # Compare performance metrics
        metrics_by_model = {}
        all_metrics = set()
        
        for result in results:
            metrics_by_model[result.model_name] = result.performance_metrics
            all_metrics.update(result.performance_metrics.keys())
        
        for metric in all_metrics:
            comparison["metrics_comparison"][metric] = {
                result.model_name: result.performance_metrics.get(metric, None)
                for result in results
            }
        
        # Compare feature importance
        all_features = set()
        for result in results:
            all_features.update(result.feature_importance.keys())
        
        for feature in all_features:
            comparison["feature_importance_comparison"][feature] = {
                result.model_name: result.feature_importance.get(feature, 0.0)
                for result in results
            }
        
        # Generate rankings and recommendations
        for metric, values in comparison["metrics_comparison"].items():
            # Skip metrics where any model has None value
            if None in values.values():
                continue
                
            # Determine if higher or lower is better for this metric
            higher_is_better = metric not in ["rmse", "mae"]
            
            # Sort by metric
            ranked_models = sorted(
                values.items(),
                key=lambda x: x[1],
                reverse=higher_is_better
            )
            
            comparison[f"{metric}_ranking"] = [
                {"model": model, "value": value}
                for model, value in ranked_models
            ]
            
            # Add recommendation for best model by this metric
            best_model, best_value = ranked_models[0]
            comparison["recommendations"].append(
                f"Best model for {metric}: {best_model} ({best_value:.4f})"
            )
        
        # Store comparison result
        comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.comparison_results[comparison_id] = comparison
        
        logger.info(f"Generated comparison between {len(model_names)} models")
        return comparison

    def generate_feature_impact_analysis(self, model_name: str) -> Dict[str, Any]:
        """Generate in-depth feature impact analysis for a specific model."""
        if model_name not in self.analysis_results:
            return {"error": "model_not_found"}
            
        result = self.analysis_results[model_name]
        
        # Feature importance analysis
        feature_impact = {
            "model_name": model_name,
            "model_type": result.model_type,
            "feature_importance": result.feature_importance,
            "feature_groups": {},
            "recommendations": []
        }
        
        # Group features by name patterns to identify related features
        grouped_features = {}
        for feature, importance in result.feature_importance.items():
            # Extract feature prefix (before first underscore or digit)
            import re
            prefix_match = re.match(r'^([a-zA-Z]+)[_0-9]', feature)
            if prefix_match:
                prefix = prefix_match.group(1)
                if prefix not in grouped_features:
                    grouped_features[prefix] = []
                grouped_features[prefix].append((feature, importance))
        
        # Calculate group importance
        for group, features in grouped_features.items():
            if len(features) > 1:  # Only include actual groups
                total_importance = sum(abs(imp) for _, imp in features)
                avg_importance = total_importance / len(features)
                feature_impact["feature_groups"][group] = {
                    "features": [f for f, _ in features],
                    "total_importance": float(total_importance),
                    "average_importance": float(avg_importance),
                    "count": len(features)
                }
        
        # Generate insights based on feature groups
        if grouped_features:
            # Find most impactful group
            most_impactful = max(
                feature_impact["feature_groups"].items(),
                key=lambda x: x[1]["total_importance"],
                default=(None, {})
            )
            
            if most_impactful[0]:
                group_name, group_info = most_impactful
                feature_impact["recommendations"].append(
                    f"The {group_name} group contains {group_info['count']} features "
                    f"with {group_info['total_importance']:.2f} total importance"
                )
        
        # Add SHAP integration if available
        if result.shap_analysis:
            feature_impact["shap_importance"] = result.shap_analysis["global_importance"]
            
            # Compare standard vs SHAP importance
            importance_diff = {}
            for feature in set(result.feature_importance.keys()) & set(result.shap_analysis["global_importance"].keys()):
                standard_imp = result.feature_importance[feature]
                shap_imp = result.shap_analysis["global_importance"][feature]
                importance_diff[feature] = abs(standard_imp - shap_imp)
            
            # Find features with high discrepancy
            high_diff_features = sorted(
                importance_diff.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5 with highest difference
            
            if high_diff_features:
                feature_impact["importance_discrepancies"] = {
                    feature: {
                        "standard_importance": result.feature_importance[feature],
                        "shap_importance": result.shap_analysis["global_importance"][feature],
                        "difference": diff
                    }
                    for feature, diff in high_diff_features
                }
                
                # Add recommendation
                feature_impact["recommendations"].append(
                    f"Features with different standard vs SHAP importance may have complex relationships: "
                    f"{', '.join(f for f, _ in high_diff_features)}"
                )
        
        return feature_impact


# Factory functions
def create_model_analyzer(
    n_permutation_repeats: int = 10,
    store_analysis_results: bool = True
) -> ModelAnalyzer:
    """Factory function to create model analyzer."""
    config = ModelAnalysisConfig(
        n_permutation_repeats=n_permutation_repeats,
        store_analysis_results=store_analysis_results
    )
    return ModelAnalyzer(config)


def analyze_model(
    model, 
    X: pd.DataFrame, 
    y: Optional[pd.Series] = None,
    model_name: str = "default",
    model_type: str = "unknown",
    task_type: str = "regression"
) -> Dict[str, Any]:
    """Convenience function for one-off model analysis."""
    analyzer = create_model_analyzer()
    result = analyzer.analyze_model(
        model=model,
        X=X,
        y=y,
        model_name=model_name,
        model_type=model_type,
        task_type=task_type
    )
    
    # Return dictionary representation for convenience
    return {
        "model_name": result.model_name,
        "model_type": result.model_type,
        "performance_metrics": result.performance_metrics,
        "feature_importance": dict(sorted(
            result.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
    }


if __name__ == "__main__":
    # Example usage
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    
    # Generate sample data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 10), columns=[f'feature_{i}' for i in range(10)])
    y = X['feature_0'] * 2 + X['feature_1'] - X['feature_2'] * 0.5 + np.random.randn(1000) * 0.1
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Analyze model
    result = analyze_model(model, X, y, "test_model", "random_forest", "regression")
    
    print(f"Model: {result['model_name']}")
    print(f"RMSE: {result['performance_metrics'].get('rmse', 'N/A')}")
    print(f"R²: {result['performance_metrics'].get('r2', 'N/A')}")
    print("\nTop 3 features by importance:")
    for i, (feature, importance) in enumerate(list(result['feature_importance'].items())[:3]):
        print(f"{i+1}. {feature}: {importance:.4f}")
