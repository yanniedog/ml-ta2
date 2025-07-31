"""
Feature Selection and Dimensionality Reduction for ML-TA System

This module provides advanced feature selection techniques to reduce dimensionality
while maintaining predictive power and preventing overfitting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings

# Handle optional dependencies gracefully
try:
    from sklearn.feature_selection import (
        SelectKBest, SelectPercentile, RFE, RFECV,
        mutual_info_classif, mutual_info_regression,
        f_classif, f_regression, chi2
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.decomposition import PCA, FastICA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Simple fallback classes
    class SelectKBest:
        def __init__(self, score_func=None, k=10):
            self.k = k
        
        def fit(self, X, y):
            return self
        
        def transform(self, X):
            return X.iloc[:, :self.k] if hasattr(X, 'iloc') else X[:, :self.k]
        
        def get_support(self):
            return [True] * self.k
    
    SelectPercentile = SelectKBest
    RFE = SelectKBest
    RFECV = SelectKBest
    RandomForestClassifier = None
    RandomForestRegressor = None
    PCA = None
    FastICA = None
    TSNE = None
    LabelEncoder = None
    mutual_info_classif = None
    mutual_info_regression = None
    f_classif = None
    f_regression = None
    chi2 = None

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    import logging
    structlog = logging

from .config import get_config
from .exceptions import FeatureEngineeringError, ValidationError
from .logging_config import get_logger

logger = get_logger("feature_selection").get_logger()
warnings.filterwarnings('ignore', category=FutureWarning)


class CorrelationSelector:
    """Removes highly correlated features to reduce multicollinearity."""
    
    def __init__(self, threshold: float = 0.95):
        """
        Initialize correlation selector.
        
        Args:
            threshold: Correlation threshold above which features are removed
        """
        self.threshold = threshold
        self.logger = logger.bind(component="correlation_selector")
        self.selected_features = []
        self.correlation_matrix = None
        self.removed_features = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CorrelationSelector':
        """
        Fit the correlation selector.
        
        Args:
            X: Feature matrix
            y: Target variable (not used)
        
        Returns:
            Self for method chaining
        """
        # Calculate correlation matrix
        self.correlation_matrix = X.corr().abs()
        
        # Find features to remove
        upper_triangle = np.triu(np.ones_like(self.correlation_matrix, dtype=bool), k=1)
        high_corr_pairs = np.where((self.correlation_matrix > self.threshold) & upper_triangle)
        
        # Select features to keep (prefer features with higher variance)
        features_to_remove = set()
        
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            feature1 = X.columns[i]
            feature2 = X.columns[j]
            
            # Remove feature with lower variance
            if X[feature1].var() < X[feature2].var():
                features_to_remove.add(feature1)
            else:
                features_to_remove.add(feature2)
        
        self.removed_features = list(features_to_remove)
        self.selected_features = [col for col in X.columns if col not in features_to_remove]
        
        self.logger.info(
            f"Correlation selection completed",
            total_features=len(X.columns),
            selected_features=len(self.selected_features),
            removed_features=len(self.removed_features)
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform feature matrix by removing correlated features.
        
        Args:
            X: Feature matrix
        
        Returns:
            Transformed feature matrix
        """
        if not self.selected_features:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self) -> List[str]:
        """Get selected feature names."""
        return self.selected_features.copy()


class VarianceSelector:
    """Removes low-variance features that provide little information."""
    
    def __init__(self, threshold: float = 0.01):
        """
        Initialize variance selector.
        
        Args:
            threshold: Variance threshold below which features are removed
        """
        self.threshold = threshold
        self.logger = logger.bind(component="variance_selector")
        self.selected_features = []
        self.feature_variances = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'VarianceSelector':
        """
        Fit the variance selector.
        
        Args:
            X: Feature matrix
            y: Target variable (not used)
        
        Returns:
            Self for method chaining
        """
        # Calculate variances
        self.feature_variances = X.var().to_dict()
        
        # Select features with variance above threshold
        self.selected_features = [
            col for col, var in self.feature_variances.items()
            if var > self.threshold
        ]
        
        removed_count = len(X.columns) - len(self.selected_features)
        self.logger.info(
            f"Variance selection completed",
            total_features=len(X.columns),
            selected_features=len(self.selected_features),
            removed_features=removed_count
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform feature matrix by removing low-variance features."""
        if not self.selected_features:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self) -> List[str]:
        """Get selected feature names."""
        return self.selected_features.copy()


class UnivariateSelector:
    """Selects features based on univariate statistical tests."""
    
    def __init__(
        self,
        score_func: str = 'mutual_info',
        selection_mode: str = 'k_best',
        k: int = 100,
        percentile: float = 50,
        task_type: str = 'classification'
    ):
        """
        Initialize univariate selector.
        
        Args:
            score_func: Scoring function ('mutual_info', 'f_test', 'chi2')
            selection_mode: Selection mode ('k_best', 'percentile')
            k: Number of features to select (for k_best mode)
            percentile: Percentile of features to select (for percentile mode)
            task_type: Type of task ('classification' or 'regression')
        """
        self.score_func = score_func
        self.selection_mode = selection_mode
        self.k = k
        self.percentile = percentile
        self.task_type = task_type
        self.logger = logger.bind(component="univariate_selector")
        
        self.selector = None
        self.feature_scores = {}
    
    def _get_score_function(self):
        """Get the appropriate scoring function."""
        if self.score_func == 'mutual_info':
            if self.task_type == 'classification':
                return mutual_info_classif
            else:
                return mutual_info_regression
        elif self.score_func == 'f_test':
            if self.task_type == 'classification':
                return f_classif
            else:
                return f_regression
        elif self.score_func == 'chi2':
            return chi2
        else:
            raise ValueError(f"Unknown score function: {self.score_func}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'UnivariateSelector':
        """
        Fit the univariate selector.
        
        Args:
            X: Feature matrix
            y: Target variable
        
        Returns:
            Self for method chaining
        """
        # Handle fallback if scikit-learn is not available
        if not SKLEARN_AVAILABLE:
            self.selected_features = list(X.columns[:self.k])
            return self
        
        # Filter out datetime columns to avoid DatetimeArray reduction errors
        datetime_cols = [col for col in X.columns if pd.api.types.is_datetime64_any_dtype(X[col])]
        if datetime_cols:
            self.logger.warning(f"Excluding datetime columns from univariate feature selection: {datetime_cols}")
            X = X.drop(columns=datetime_cols)
        
        # Choose score function
        if self.score_func == 'mutual_info':
            if self.task_type == 'classification':
                score_func = mutual_info_classif
            else:
                score_func = mutual_info_regression
        elif self.score_func == 'f_test':
            if self.task_type == 'classification':
                score_func = f_classif
            else:
                score_func = f_regression
        elif self.score_func == 'chi2':
            score_func = chi2
        else:
            raise ValueError(f"Unknown score function: {self.score_func}")
        
        # Create selector based on mode
        if self.selection_mode == 'k_best':
            self.selector = SelectKBest(score_func=score_func, k=min(self.k, X.shape[1]))
        elif self.selection_mode == 'percentile':
            self.selector = SelectPercentile(score_func=score_func, percentile=self.percentile)
        else:
            raise ValueError(f"Unknown selection mode: {self.selection_mode}")
        
        # Handle missing values and infinite values
        X_clean = X.fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        
        # For chi2, ensure non-negative values
        if self.score_func == 'chi2':
            X_clean = X_clean - X_clean.min() + 1e-6
        
        # Fit selector
        self.selector.fit(X_clean, y)
        
        # Store feature scores
        if hasattr(self.selector, 'scores_'):
            self.feature_scores = dict(zip(X.columns, self.selector.scores_))
        
        selected_count = len(self.get_feature_names_out(X.columns))
        self.logger.info(
            f"Univariate selection completed",
            score_func=self.score_func,
            total_features=len(X.columns),
            selected_features=selected_count
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform feature matrix using fitted selector."""
        if self.selector is None:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        X_clean = X.fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        
        if self.score_func == 'chi2':
            X_clean = X_clean - X_clean.min() + 1e-6
        
        X_selected = self.selector.transform(X_clean)
        selected_features = self.get_feature_names_out(X.columns)
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features: List[str]) -> List[str]:
        """Get selected feature names."""
        if self.selector is None:
            return []
        
        mask = self.selector.get_support()
        return [feature for feature, selected in zip(input_features, mask) if selected]


class TreeBasedSelector:
    """Selects features based on tree-based feature importance."""
    
    def __init__(
        self,
        estimator_type: str = 'random_forest',
        n_estimators: int = 100,
        max_features: int = 100,
        task_type: str = 'classification',
        importance_threshold: float = 0.01
    ):
        """
        Initialize tree-based selector.
        
        Args:
            estimator_type: Type of estimator ('random_forest', 'rfe')
            n_estimators: Number of trees
            max_features: Maximum number of features to select
            task_type: Type of task ('classification' or 'regression')
            importance_threshold: Minimum importance threshold
        """
        self.estimator_type = estimator_type
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.task_type = task_type
        self.importance_threshold = importance_threshold
        self.logger = logger.bind(component="tree_based_selector")
        
        self.estimator = None
        self.selector = None
        self.feature_importances = {}
        self.selected_features = []
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TreeBasedSelector':
        """
        Fit the tree-based selector.
        
        Args:
            X: Feature matrix
            y: Target variable
        
        Returns:
            Self for method chaining
        """
        # Create base estimator
        if self.task_type == 'classification':
            self.estimator = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.estimator = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1
            )
        
        # Filter out datetime columns to avoid DatetimeArray reduction errors
        datetime_cols = [col for col in X.columns if pd.api.types.is_datetime64_any_dtype(X[col])]
        if datetime_cols:
            self.logger.warning(f"Excluding datetime columns from tree-based feature selection: {datetime_cols}")
            X = X.drop(columns=datetime_cols)
            
        # Handle missing values
        X_clean = X.fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        
        if self.estimator_type == 'random_forest':
            # Fit estimator and use feature importances
            self.estimator.fit(X_clean, y)
            self.feature_importances = dict(zip(X.columns, self.estimator.feature_importances_))
            
            # Select features based on importance threshold
            important_features = [
                feature for feature, importance in self.feature_importances.items()
                if importance >= self.importance_threshold
            ]
            
            # Limit to max_features
            sorted_features = sorted(
                important_features,
                key=lambda x: self.feature_importances[x],
                reverse=True
            )
            self.selected_features = sorted_features[:self.max_features]
        
        elif self.estimator_type == 'rfe':
            # Use Recursive Feature Elimination
            self.selector = RFE(
                estimator=self.estimator,
                n_features_to_select=min(self.max_features, X.shape[1])
            )
            self.selector.fit(X_clean, y)
            self.selected_features = self.get_feature_names_out(X.columns)
        
        self.logger.info(
            f"Tree-based selection completed",
            estimator_type=self.estimator_type,
            total_features=len(X.columns),
            selected_features=len(self.selected_features)
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform feature matrix using selected features."""
        if not self.selected_features:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        return X[self.selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features: List[str] = None) -> List[str]:
        """Get selected feature names."""
        if self.estimator_type == 'rfe' and self.selector is not None:
            if input_features is None:
                return self.selected_features
            mask = self.selector.get_support()
            return [feature for feature, selected in zip(input_features, mask) if selected]
        
        return self.selected_features.copy()


class DimensionalityReducer:
    """Applies dimensionality reduction techniques like PCA."""
    
    def __init__(
        self,
        method: str = 'pca',
        n_components: Union[int, float] = 0.95,
        random_state: int = 42
    ):
        """
        Initialize dimensionality reducer.
        
        Args:
            method: Reduction method ('pca', 'ica', 'tsne')
            n_components: Number of components or variance ratio to retain
            random_state: Random state for reproducibility
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.logger = logger.bind(component="dimensionality_reducer")
        
        self.reducer = None
        self.feature_names = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DimensionalityReducer':
        """
        Fit the dimensionality reducer.
        
        Args:
            X: Feature matrix
            y: Target variable (not used for unsupervised methods)
        
        Returns:
            Self for method chaining
        """
        # Handle missing values
        X_clean = X.fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        
        if self.method == 'pca':
            self.reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state
            )
        elif self.method == 'ica':
            # For ICA, n_components must be an integer
            n_comp = self.n_components if isinstance(self.n_components, int) else min(50, X.shape[1])
            self.reducer = FastICA(
                n_components=n_comp,
                random_state=self.random_state
            )
        elif self.method == 'tsne':
            # t-SNE is typically used for visualization, not feature reduction
            n_comp = min(2, X.shape[1])
            self.reducer = TSNE(
                n_components=n_comp,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown reduction method: {self.method}")
        
        # Fit reducer
        self.reducer.fit(X_clean)
        
        # Generate feature names
        n_comp = self.reducer.n_components_ if hasattr(self.reducer, 'n_components_') else self.reducer.n_components
        self.feature_names = [f"{self.method}_component_{i}" for i in range(n_comp)]
        
        self.logger.info(
            f"Dimensionality reduction fitted",
            method=self.method,
            original_features=X.shape[1],
            reduced_features=len(self.feature_names)
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform feature matrix using fitted reducer."""
        if self.reducer is None:
            raise ValueError("Reducer not fitted. Call fit() first.")
        
        X_clean = X.fillna(0)
        X_clean = X_clean.replace([np.inf, -np.inf], 0)
        
        X_reduced = self.reducer.transform(X_clean)
        
        return pd.DataFrame(
            X_reduced,
            columns=self.feature_names,
            index=X.index
        )
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self) -> List[str]:
        """Get reduced feature names."""
        return self.feature_names.copy()


class FeatureSelector:
    """Main feature selection pipeline combining multiple selection methods."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature selector with configuration."""
        from .config import get_model_dict
        self.config = config or get_model_dict(get_config().features)
        self.logger = logger.bind(component="feature_selector")
        
        # Initialize selectors
        self.selectors = {}
        self.selected_features = []
        self.feature_scores = {}
        self.fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> 'FeatureSelector':
        """
        Fit all feature selectors.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: Type of task ('classification' or 'regression')
        
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Starting feature selection", features=X.shape[1], task_type=task_type)
        
        # Preemptively remove datetime columns from the entire feature selection process
        datetime_cols = [col for col in X.columns if pd.api.types.is_datetime64_any_dtype(X[col])]
        
        # Also identify and remove string/object columns that can't be converted to float
        string_cols = [col for col in X.columns if X[col].dtype == 'object' or 
                      (hasattr(X[col], 'dtype') and str(X[col].dtype).startswith('string'))]
        
        columns_to_remove = list(set(datetime_cols + string_cols))
        
        if columns_to_remove:
            self.logger.warning(f"Removing non-numeric columns from feature selection: {columns_to_remove}")
            current_X = X.drop(columns=columns_to_remove).copy()
        else:
            current_X = X.copy()
        
        # Step 1: Remove low-variance features
        if self.config.get('use_variance_selection', True):
            variance_threshold = self.config.get('variance_threshold', 0.01)
            self.selectors['variance'] = VarianceSelector(threshold=variance_threshold)
            current_X = self.selectors['variance'].fit_transform(current_X)
        
        # Step 2: Remove highly correlated features
        if self.config.get('use_correlation_selection', True):
            correlation_threshold = self.config.get('correlation_threshold', 0.95)
            self.selectors['correlation'] = CorrelationSelector(threshold=correlation_threshold)
            current_X = self.selectors['correlation'].fit_transform(current_X)
        
        # Step 3: Univariate feature selection
        if self.config.get('use_univariate_selection', True):
            univariate_config = self.config.get('univariate_selection', {})
            self.selectors['univariate'] = UnivariateSelector(
                score_func=univariate_config.get('score_func', 'mutual_info'),
                selection_mode=univariate_config.get('selection_mode', 'k_best'),
                k=univariate_config.get('k', 100),
                task_type=task_type
            )
            current_X = self.selectors['univariate'].fit_transform(current_X, y)
        
        # Step 4: Tree-based feature selection
        if self.config.get('use_tree_selection', True):
            tree_config = self.config.get('tree_selection', {})
            self.selectors['tree'] = TreeBasedSelector(
                estimator_type=tree_config.get('estimator_type', 'random_forest'),
                max_features=tree_config.get('max_features', 50),
                task_type=task_type
            )
            current_X = self.selectors['tree'].fit_transform(current_X, y)
        
        # Step 5: Dimensionality reduction (optional)
        if self.config.get('use_dimensionality_reduction', False):
            reduction_config = self.config.get('dimensionality_reduction', {})
            self.selectors['reduction'] = DimensionalityReducer(
                method=reduction_config.get('method', 'pca'),
                n_components=reduction_config.get('n_components', 0.95)
            )
            current_X = self.selectors['reduction'].fit_transform(current_X)
        
        self.selected_features = list(current_X.columns)
        self.fitted = True
        
        self.logger.info(
            f"Feature selection completed",
            original_features=X.shape[1],
            selected_features=len(self.selected_features),
            reduction_ratio=len(self.selected_features) / X.shape[1]
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform feature matrix using fitted selectors."""
        if not self.fitted:
            raise ValueError("Selector not fitted. Call fit() first.")
        
        current_X = X.copy()
        
        # Apply selectors in order
        for selector_name in ['variance', 'correlation', 'univariate', 'tree', 'reduction']:
            if selector_name in self.selectors:
                current_X = self.selectors[selector_name].transform(current_X)
        
        return current_X
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y, task_type).transform(X)
    
    def get_feature_names_out(self) -> List[str]:
        """Get final selected feature names."""
        return self.selected_features.copy()
    
    def get_feature_importance_scores(self) -> Dict[str, float]:
        """Get feature importance scores from tree-based selector."""
        if 'tree' in self.selectors and hasattr(self.selectors['tree'], 'feature_importances'):
            return self.selectors['tree'].feature_importances.copy()
        return {}


# Factory function
def create_feature_selector(config: Optional[Dict[str, Any]] = None) -> FeatureSelector:
    """Create feature selector with configuration."""
    return FeatureSelector(config)


# Example usage
if __name__ == "__main__":
    # Test feature selection
    selector = create_feature_selector()
    
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 200
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    
    # Add some correlated features
    X['feature_corr_1'] = X['feature_0'] + np.random.randn(n_samples) * 0.1
    X['feature_corr_2'] = X['feature_1'] + np.random.randn(n_samples) * 0.1
    
    # Create target variable
    y = (X['feature_0'] + X['feature_1'] + np.random.randn(n_samples) > 0).astype(int)
    
    try:
        X_selected = selector.fit_transform(X, y, task_type='classification')
        selected_features = selector.get_feature_names_out()
        
        print(f"Original features: {X.shape[1]}")
        print(f"Selected features: {len(selected_features)}")
        print(f"Reduction ratio: {len(selected_features) / X.shape[1]:.2f}")
        
    except Exception as e:
        print(f"Error: {e}")
