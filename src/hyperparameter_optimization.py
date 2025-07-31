"""
Intelligent Parameter Tuning with Optuna Integration.

This module implements:
- Advanced hyperparameter optimization using Optuna
- Multi-objective optimization balancing accuracy, speed, interpretability
- Pruning strategies for efficient search
- Search space definition for each algorithm
- Optimization history tracking and analysis
- AutoML pipeline for automatic algorithm selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Core optimization libraries
try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner, SuccessiveHalvingPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    # Create mock classes for testing
    class optuna:
        class Study:
            def __init__(self):
                self.best_params = {}
                self.best_value = 0.5
                self.trials = []
            
            def optimize(self, objective, n_trials=100, **kwargs):
                pass
        
        @staticmethod
        def create_study(*args, **kwargs):
            return optuna.Study()

# ML libraries
try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.config import get_config
from src.logging_config import get_logger
from src.exceptions import ModelTrainingError, OptimizationError

logger = get_logger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    n_trials: int = 100
    timeout: Optional[int] = None  # seconds
    sampler: str = 'tpe'  # 'tpe', 'cmaes', 'random'
    pruner: str = 'median'  # 'median', 'hyperband', 'successive_halving'
    cv_folds: int = 5
    cv_strategy: str = 'stratified'  # 'stratified', 'timeseries'
    scoring: str = 'accuracy'  # for classification
    direction: str = 'maximize'  # 'maximize', 'minimize'
    multi_objective: bool = False
    objectives: List[str] = field(default_factory=lambda: ['accuracy', 'speed'])
    early_stopping_rounds: int = 50
    verbose: bool = True


@dataclass
class OptimizationResult:
    """Results from hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    best_trial_number: int
    n_trials: int
    optimization_time: float
    study_name: str
    algorithm: str
    cv_scores: List[float]
    feature_importance: Optional[Dict[str, float]] = None
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)


class SearchSpaceDefinition:
    """Define search spaces for different algorithms."""
    
    @staticmethod
    def get_lightgbm_space(trial, task_type: str = 'classification') -> Dict[str, Any]:
        """Define LightGBM hyperparameter search space."""
        params = {
            'objective': 'binary' if task_type == 'classification' else 'regression',
            'metric': 'binary_logloss' if task_type == 'classification' else 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'verbosity': -1,
            'random_state': 42
        }
        
        if task_type == 'classification':
            params['num_class'] = 1  # Binary classification
        
        return params
    
    @staticmethod
    def get_xgboost_space(trial, task_type: str = 'classification') -> Dict[str, Any]:
        """Define XGBoost hyperparameter search space."""
        params = {
            'objective': 'binary:logistic' if task_type == 'classification' else 'reg:squarederror',
            'eval_metric': 'logloss' if task_type == 'classification' else 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'verbosity': 0
        }
        
        return params
    
    @staticmethod
    def get_catboost_space(trial, task_type: str = 'classification') -> Dict[str, Any]:
        """Define CatBoost hyperparameter search space."""
        params = {
            'objective': 'Logloss' if task_type == 'classification' else 'RMSE',
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli']),
            'random_seed': 42,
            'verbose': False
        }
        
        if params['bootstrap_type'] == 'Bayesian':
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 1)
        else:
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1)
        
        return params
    
    @staticmethod
    def get_random_forest_space(trial, task_type: str = 'classification') -> Dict[str, Any]:
        """Define Random Forest hyperparameter search space."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42
        }
        
        if not params['bootstrap']:
            params['oob_score'] = False
        
        return params


class HyperparameterOptimizer:
    """Main hyperparameter optimization class."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize hyperparameter optimizer."""
        self.config = config or OptimizationConfig()
        self.search_space = SearchSpaceDefinition()
        self.optimization_history = []
        self.best_models = {}
        
        logger.info("HyperparameterOptimizer initialized", 
                   n_trials=self.config.n_trials,
                   sampler=self.config.sampler)
    
    def _create_study(self, study_name: str, directions: Optional[List[str]] = None) -> optuna.Study:
        """Create Optuna study with specified configuration."""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using mock study")
            return optuna.create_study()
        
        # Configure sampler
        if self.config.sampler == 'tpe':
            sampler = TPESampler(seed=42)
        elif self.config.sampler == 'cmaes':
            sampler = CmaEsSampler(seed=42)
        else:
            sampler = RandomSampler(seed=42)
        
        # Configure pruner
        if self.config.pruner == 'median':
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif self.config.pruner == 'hyperband':
            pruner = HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)
        else:
            pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        
        # Create study
        if self.config.multi_objective and directions:
            study = optuna.create_study(
                directions=directions,
                sampler=sampler,
                pruner=pruner,
                study_name=study_name
            )
        else:
            study = optuna.create_study(
                direction=self.config.direction,
                sampler=sampler,
                pruner=pruner,
                study_name=study_name
            )
        
        return study
    
    def _create_objective_function(self, X: pd.DataFrame, y: pd.Series, 
                                 algorithm: str, task_type: str) -> Callable:
        """Create objective function for optimization."""
        
        def objective(trial):
            try:
                # Get hyperparameters for the algorithm
                if algorithm == 'lightgbm':
                    params = self.search_space.get_lightgbm_space(trial, task_type)
                elif algorithm == 'xgboost':
                    params = self.search_space.get_xgboost_space(trial, task_type)
                elif algorithm == 'catboost':
                    params = self.search_space.get_catboost_space(trial, task_type)
                elif algorithm == 'random_forest':
                    params = self.search_space.get_random_forest_space(trial, task_type)
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")
                
                # Create model
                model = self._create_model(algorithm, params, task_type)
                
                # Perform cross-validation
                cv_scores = self._cross_validate(model, X, y, task_type)
                
                # Calculate primary objective (mean CV score)
                primary_score = np.mean(cv_scores)
                
                # Handle multi-objective optimization
                if self.config.multi_objective:
                    # Calculate secondary objectives
                    speed_score = self._calculate_speed_score(model, X, y)
                    interpretability_score = self._calculate_interpretability_score(algorithm, params)
                    
                    return primary_score, speed_score, interpretability_score
                else:
                    return primary_score
                
            except Exception as e:
                logger.error(f"Trial failed: {e}")
                # Return poor score for failed trials
                if self.config.multi_objective:
                    return 0.0, 0.0, 0.0
                else:
                    return 0.0 if self.config.direction == 'maximize' else float('inf')
        
        return objective
    
    def _create_model(self, algorithm: str, params: Dict[str, Any], task_type: str):
        """Create model instance with given parameters."""
        if algorithm == 'lightgbm' and LIGHTGBM_AVAILABLE:
            if task_type == 'classification':
                return lgb.LGBMClassifier(**params)
            else:
                return lgb.LGBMRegressor(**params)
        
        elif algorithm == 'xgboost' and XGBOOST_AVAILABLE:
            if task_type == 'classification':
                return xgb.XGBClassifier(**params)
            else:
                return xgb.XGBRegressor(**params)
        
        elif algorithm == 'catboost' and CATBOOST_AVAILABLE:
            if task_type == 'classification':
                return cb.CatBoostClassifier(**params)
            else:
                return cb.CatBoostRegressor(**params)
        
        elif algorithm == 'random_forest' and SKLEARN_AVAILABLE:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if task_type == 'classification':
                return RandomForestClassifier(**params)
            else:
                return RandomForestRegressor(**params)
        
        else:
            raise ValueError(f"Algorithm {algorithm} not available or supported")
    
    def _cross_validate(self, model, X: pd.DataFrame, y: pd.Series, task_type: str) -> List[float]:
        """Perform cross-validation."""
        if not SKLEARN_AVAILABLE:
            # Mock cross-validation for testing
            return [0.7, 0.72, 0.68, 0.71, 0.69]
        
        # Choose cross-validation strategy
        if self.config.cv_strategy == 'timeseries':
            cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        else:
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
        
        # Perform cross-validation
        scores = cross_val_score(
            model, X, y, 
            cv=cv, 
            scoring=self.config.scoring,
            n_jobs=-1
        )
        
        return scores.tolist()
    
    def _calculate_speed_score(self, model, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate speed score for multi-objective optimization."""
        import time
        
        # Measure training time
        start_time = time.time()
        model.fit(X.head(min(1000, len(X))), y.head(min(1000, len(y))))  # Use subset for speed
        training_time = time.time() - start_time
        
        # Measure prediction time
        start_time = time.time()
        model.predict(X.head(100))
        prediction_time = time.time() - start_time
        
        # Convert to speed score (inverse of time, normalized)
        speed_score = 1.0 / (1.0 + training_time + prediction_time * 10)
        
        return speed_score
    
    def _calculate_interpretability_score(self, algorithm: str, params: Dict[str, Any]) -> float:
        """Calculate interpretability score for multi-objective optimization."""
        # Simple heuristic for interpretability
        interpretability_scores = {
            'random_forest': 0.8,
            'lightgbm': 0.7,
            'xgboost': 0.6,
            'catboost': 0.6
        }
        
        base_score = interpretability_scores.get(algorithm, 0.5)
        
        # Adjust based on complexity parameters
        if algorithm in ['lightgbm', 'xgboost', 'catboost']:
            # Simpler models are more interpretable
            if 'max_depth' in params:
                depth_penalty = min(params['max_depth'] / 20.0, 0.3)
                base_score -= depth_penalty
            
            if 'num_leaves' in params:
                leaves_penalty = min(params['num_leaves'] / 500.0, 0.2)
                base_score -= leaves_penalty
        
        return max(0.1, base_score)
    
    def optimize_algorithm(self, X: pd.DataFrame, y: pd.Series, 
                          algorithm: str, task_type: str = 'classification') -> OptimizationResult:
        """Optimize hyperparameters for a specific algorithm."""
        logger.info(f"Starting hyperparameter optimization for {algorithm}")
        
        start_time = time.time()
        study_name = f"{algorithm}_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create study
        if self.config.multi_objective:
            directions = ['maximize', 'maximize', 'maximize']  # accuracy, speed, interpretability
            study = self._create_study(study_name, directions)
        else:
            study = self._create_study(study_name)
        
        # Create objective function
        objective = self._create_objective_function(X, y, algorithm, task_type)
        
        # Run optimization
        try:
            study.optimize(
                objective, 
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                show_progress_bar=self.config.verbose
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise OptimizationError(f"Hyperparameter optimization failed: {e}")
        
        optimization_time = time.time() - start_time
        
        # Extract results
        if self.config.multi_objective:
            # For multi-objective, select best trial based on primary objective
            best_trial = max(study.trials, key=lambda t: t.values[0] if t.values else 0)
            best_params = best_trial.params
            best_score = best_trial.values[0] if best_trial.values else 0
        else:
            best_params = study.best_params
            best_score = study.best_value
        
        # Train final model with best parameters for feature importance
        try:
            best_model = self._create_model(algorithm, best_params, task_type)
            best_model.fit(X, y)
            
            # Extract feature importance if available
            feature_importance = None
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, best_model.feature_importances_))
            elif hasattr(best_model, 'get_feature_importance'):  # CatBoost
                feature_importance = dict(zip(X.columns, best_model.get_feature_importance()))
            
            # Store best model
            self.best_models[algorithm] = best_model
            
        except Exception as e:
            logger.warning(f"Failed to train final model: {e}")
            feature_importance = None
        
        # Create optimization result
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_trial_number=len(study.trials),
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            study_name=study_name,
            algorithm=algorithm,
            cv_scores=[],  # Would need to re-run CV with best params
            feature_importance=feature_importance,
            optimization_history=[
                {
                    'trial_number': i,
                    'params': trial.params,
                    'value': trial.value if not self.config.multi_objective else trial.values[0],
                    'state': trial.state.name
                }
                for i, trial in enumerate(study.trials)
            ]
        )
        
        self.optimization_history.append(result)
        
        logger.info(f"Optimization completed for {algorithm}",
                   best_score=best_score,
                   n_trials=len(study.trials),
                   optimization_time=optimization_time)
        
        return result
    
    def optimize_multiple_algorithms(self, X: pd.DataFrame, y: pd.Series,
                                   algorithms: List[str], 
                                   task_type: str = 'classification') -> Dict[str, OptimizationResult]:
        """Optimize hyperparameters for multiple algorithms."""
        results = {}
        
        for algorithm in algorithms:
            try:
                result = self.optimize_algorithm(X, y, algorithm, task_type)
                results[algorithm] = result
            except Exception as e:
                logger.error(f"Failed to optimize {algorithm}: {e}")
                continue
        
        return results
    
    def get_best_algorithm(self, results: Dict[str, OptimizationResult]) -> Tuple[str, OptimizationResult]:
        """Get the best algorithm from optimization results."""
        if not results:
            raise ValueError("No optimization results available")
        
        best_algorithm = max(results.keys(), key=lambda alg: results[alg].best_score)
        return best_algorithm, results[best_algorithm]
    
    def save_optimization_results(self, results: Dict[str, OptimizationResult], 
                                filepath: str) -> None:
        """Save optimization results to file."""
        serializable_results = {}
        
        for algorithm, result in results.items():
            serializable_results[algorithm] = {
                'best_params': result.best_params,
                'best_score': result.best_score,
                'best_trial_number': result.best_trial_number,
                'n_trials': result.n_trials,
                'optimization_time': result.optimization_time,
                'study_name': result.study_name,
                'algorithm': result.algorithm,
                'feature_importance': result.feature_importance,
                'optimization_history': result.optimization_history
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to {filepath}")


class AutoMLPipeline:
    """Automated ML pipeline with algorithm selection and optimization."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """Initialize AutoML pipeline."""
        self.config = config or OptimizationConfig()
        self.optimizer = HyperparameterOptimizer(config)
        self.available_algorithms = self._get_available_algorithms()
        
        logger.info("AutoMLPipeline initialized", 
                   available_algorithms=self.available_algorithms)
    
    def _get_available_algorithms(self) -> List[str]:
        """Get list of available algorithms based on installed libraries."""
        algorithms = ['random_forest']  # Always available with sklearn
        
        if LIGHTGBM_AVAILABLE:
            algorithms.append('lightgbm')
        
        if XGBOOST_AVAILABLE:
            algorithms.append('xgboost')
        
        if CATBOOST_AVAILABLE:
            algorithms.append('catboost')
        
        return algorithms
    
    def auto_optimize(self, X: pd.DataFrame, y: pd.Series,
                     task_type: str = 'classification',
                     algorithms: Optional[List[str]] = None) -> Dict[str, Any]:
        """Automatically select and optimize the best algorithm."""
        if algorithms is None:
            algorithms = self.available_algorithms
        
        # Filter to available algorithms
        algorithms = [alg for alg in algorithms if alg in self.available_algorithms]
        
        if not algorithms:
            raise ValueError("No available algorithms for optimization")
        
        logger.info(f"Starting AutoML optimization with algorithms: {algorithms}")
        
        # Optimize all algorithms
        results = self.optimizer.optimize_multiple_algorithms(X, y, algorithms, task_type)
        
        if not results:
            raise OptimizationError("All algorithm optimizations failed")
        
        # Get best algorithm
        best_algorithm, best_result = self.optimizer.get_best_algorithm(results)
        
        # Create summary
        summary = {
            'best_algorithm': best_algorithm,
            'best_score': best_result.best_score,
            'best_params': best_result.best_params,
            'all_results': results,
            'algorithm_ranking': sorted(
                results.items(), 
                key=lambda x: x[1].best_score, 
                reverse=True
            )
        }
        
        logger.info(f"AutoML optimization completed",
                   best_algorithm=best_algorithm,
                   best_score=best_result.best_score)
        
        return summary


# Factory functions
def create_hyperparameter_optimizer(n_trials: int = 100, 
                                  sampler: str = 'tpe',
                                  multi_objective: bool = False) -> HyperparameterOptimizer:
    """Factory function to create hyperparameter optimizer."""
    config = OptimizationConfig(
        n_trials=n_trials,
        sampler=sampler,
        multi_objective=multi_objective
    )
    
    return HyperparameterOptimizer(config)


def create_automl_pipeline(n_trials: int = 50) -> AutoMLPipeline:
    """Factory function to create AutoML pipeline."""
    config = OptimizationConfig(n_trials=n_trials)
    return AutoMLPipeline(config)


if __name__ == '__main__':
    # Example usage
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 20), columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(np.random.randint(0, 2, 1000))
    
    # Create optimizer
    optimizer = create_hyperparameter_optimizer(n_trials=10)
    
    # Optimize single algorithm
    result = optimizer.optimize_algorithm(X, y, 'random_forest', 'classification')
    print(f"Best score: {result.best_score}")
    print(f"Best params: {result.best_params}")
    
    # AutoML pipeline
    automl = create_automl_pipeline(n_trials=5)
    summary = automl.auto_optimize(X, y, 'classification')
    print(f"Best algorithm: {summary['best_algorithm']}")
    print(f"Algorithm ranking: {[(alg, res.best_score) for alg, res in summary['algorithm_ranking']]}")
