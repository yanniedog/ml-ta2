"""
Advanced Feature Engineering for ML-TA System

This module generates 200+ features from OHLCV data while preventing data leakage
through temporal validation and proper train/test separation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import warnings

# Handle optional dependencies gracefully
try:
    from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Simple fallback scalers
    class RobustScaler:
        def __init__(self):
            self.median_ = None
            self.scale_ = None
        
        def fit(self, X):
            self.median_ = np.median(X, axis=0)
            self.scale_ = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
            return self
        
        def transform(self, X):
            return (X - self.median_) / self.scale_
    
    StandardScaler = RobustScaler
    MinMaxScaler = RobustScaler
    SelectKBest = None
    mutual_info_classif = None
    chi2 = None
    f_classif = None

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    import logging
    structlog = logging

from .config import get_config
from .exceptions import FeatureEngineeringError, ValidationError, ErrorContext
from .utils import optimize_dataframe_memory
from .indicators import TechnicalIndicators
from .logging_config import get_logger

logger = get_logger("features").get_logger()
warnings.filterwarnings('ignore', category=FutureWarning)


class TemporalValidator:
    """Ensures all features use only historical data up to the current timestamp."""
    
    def __init__(self):
        """Initialize temporal validator."""
        self.logger = logger.bind(component="temporal_validator")
        self.violations = []
    
    def validate_feature_matrix(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> bool:
        """
        Validate that feature matrix doesn't contain future information.
        
        Args:
            df: Feature matrix to validate
            timestamp_col: Name of timestamp column
        
        Returns:
            True if validation passes
        
        Raises:
            ValidationError: If data leakage detected
        """
        self.violations = []
        
        if timestamp_col not in df.columns:
            raise ValidationError(f"Timestamp column '{timestamp_col}' not found")
        
        # Check for any features that might contain future information
        future_indicators = []
        
        # Check for forward-looking features (common mistakes)
        suspicious_patterns = [
            'future_', 'next_', 'lead_', 'forward_', 'ahead_'
        ]
        
        for col in df.columns:
            if any(pattern in col.lower() for pattern in suspicious_patterns):
                future_indicators.append(col)
        
        if future_indicators:
            self.violations.extend(future_indicators)
            raise ValidationError(f"Potential future-looking features detected: {future_indicators}")
        
        # Validate rolling calculations don't use future data
        self._validate_rolling_calculations(df, timestamp_col)
        
        self.logger.info("Temporal validation passed", features=len(df.columns))
        return True
    
    def _validate_rolling_calculations(self, df: pd.DataFrame, timestamp_col: str) -> None:
        """Validate rolling calculations for temporal correctness."""
        # This is a simplified check - in practice, you'd need more sophisticated validation
        # Check that data is properly sorted by timestamp
        if not df[timestamp_col].is_monotonic_increasing:
            self.violations.append("timestamp_not_sorted")
            raise ValidationError("Data must be sorted by timestamp for temporal validation")
    
    def get_violations(self) -> List[str]:
        """Get list of temporal violations found."""
        return self.violations.copy()


class LaggingFeatures:
    """Creates time-shifted versions of indicators and prices with configurable lags."""
    
    def __init__(self, max_lag: int = 10):
        """Initialize lagging features generator."""
        self.max_lag = max_lag
        self.logger = logger.bind(component="lagging_features")
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = None
    ) -> pd.DataFrame:
        """
        Create lagged features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to create lags for
            lags: List of lag periods (default: [1, 2, 3, 5, 10])
        
        Returns:
            DataFrame with lagged features added
        """
        if lags is None:
            lags = [1, 2, 3, 5, 10]
        
        result_df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                self.logger.warning(f"Column {col} not found, skipping")
                continue
            
            for lag in lags:
                if lag > self.max_lag:
                    continue
                
                lag_col_name = f"{col}_lag_{lag}"
                result_df[lag_col_name] = df[col].shift(lag)
        
        lag_count = len(columns) * len([l for l in lags if l <= self.max_lag])
        self.logger.info(f"Created {lag_count} lagged features")
        
        return result_df
    
    def create_diff_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        periods: List[int] = None
    ) -> pd.DataFrame:
        """
        Create difference features (current - previous).
        
        Args:
            df: Input DataFrame
            columns: Columns to create differences for
            periods: Difference periods (default: [1, 2, 5])
        
        Returns:
            DataFrame with difference features added
        """
        if periods is None:
            periods = [1, 2, 5]
        
        result_df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for period in periods:
                diff_col_name = f"{col}_diff_{period}"
                result_df[diff_col_name] = df[col].diff(periods=period)
        
        diff_count = len(columns) * len(periods)
        self.logger.info(f"Created {diff_count} difference features")
        
        return result_df


class RollingFeatures:
    """Computes rolling statistics over various windows."""
    
    def __init__(self):
        """Initialize rolling features generator."""
        self.logger = logger.bind(component="rolling_features")
    
    def create_rolling_statistics(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = None,
        statistics: List[str] = None
    ) -> pd.DataFrame:
        """
        Create rolling statistical features.
        
        Args:
            df: Input DataFrame
            columns: Columns to compute statistics for
            windows: Rolling window sizes (default: [5, 10, 20, 50])
            statistics: Statistics to compute (default: ['mean', 'std', 'min', 'max'])
        
        Returns:
            DataFrame with rolling features added
        """
        if windows is None:
            windows = [5, 10, 20, 50]
        
        if statistics is None:
            statistics = ['mean', 'std', 'min', 'max', 'skew', 'kurt']
        
        result_df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            # Skip datetime columns to avoid DatetimeArray reduction errors
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.logger.warning(f"Skipping rolling statistics for datetime column: {col}")
                continue
            
            for window in windows:
                rolling_obj = df[col].rolling(window=window)
                
                for stat in statistics:
                    try:
                        if stat == 'skew':
                            result_df[f"{col}_rolling_{window}_{stat}"] = rolling_obj.skew()
                        elif stat == 'kurt':
                            result_df[f"{col}_rolling_{window}_{stat}"] = rolling_obj.kurt()
                        else:
                            result_df[f"{col}_rolling_{window}_{stat}"] = getattr(rolling_obj, stat)()
                    except AttributeError:
                        self.logger.warning(f"Statistic {stat} not available for rolling window")
                        continue
        
        feature_count = len(columns) * len(windows) * len(statistics)
        self.logger.info(f"Created {feature_count} rolling statistical features")
        
        return result_df
    
    def create_expanding_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        statistics: List[str] = None
    ) -> pd.DataFrame:
        """
        Create expanding window features.
        
        Args:
            df: Input DataFrame
            columns: Columns to compute statistics for
            statistics: Statistics to compute
        
        Returns:
            DataFrame with expanding features added
        """
        if statistics is None:
            statistics = ['mean', 'std', 'min', 'max']
        
        result_df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            # Skip datetime columns to avoid DatetimeArray reduction errors
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.logger.warning(f"Skipping expanding features for datetime column: {col}")
                continue
            
            expanding_obj = df[col].expanding()
            
            for stat in statistics:
                try:
                    result_df[f"{col}_expanding_{stat}"] = getattr(expanding_obj, stat)()
                except AttributeError:
                    continue
        
        feature_count = len(columns) * len(statistics)
        self.logger.info(f"Created {feature_count} expanding window features")
        
        return result_df


class InteractionFeatures:
    """Generates ratios, differences, products, and polynomial combinations."""
    
    def __init__(self):
        """Initialize interaction features generator."""
        self.logger = logger.bind(component="interaction_features")
    
    def create_ratio_features(
        self,
        df: pd.DataFrame,
        numerator_cols: List[str],
        denominator_cols: List[str]
    ) -> pd.DataFrame:
        """
        Create ratio features between columns.
        
        Args:
            df: Input DataFrame
            numerator_cols: Columns to use as numerators
            denominator_cols: Columns to use as denominators
        
        Returns:
            DataFrame with ratio features added
        """
        result_df = df.copy()
        
        for num_col in numerator_cols:
            if num_col not in df.columns:
                continue
                
            # Skip datetime columns to avoid DatetimeArray reduction errors
            if pd.api.types.is_datetime64_any_dtype(df[num_col]):
                self.logger.warning(f"Skipping ratio features for datetime column: {num_col}")
                continue
            
            for den_col in denominator_cols:
                if den_col not in df.columns or num_col == den_col:
                    continue
                    
                # Skip datetime columns to avoid DatetimeArray reduction errors
                if pd.api.types.is_datetime64_any_dtype(df[den_col]):
                    self.logger.warning(f"Skipping ratio features for datetime column: {den_col}")
                    continue
                
                ratio_col_name = f"{num_col}_div_{den_col}"
                
                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = df[num_col] / df[den_col]
                    ratio = ratio.replace([np.inf, -np.inf], np.nan)
                    result_df[ratio_col_name] = ratio
        
        ratio_count = len(numerator_cols) * len(denominator_cols)
        self.logger.info(f"Created {ratio_count} ratio features")
        
        return result_df
    
    def create_product_features(
        self,
        df: pd.DataFrame,
        column_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Create product features between column pairs.
        
        Args:
            df: Input DataFrame
            column_pairs: List of column pairs to multiply
        
        Returns:
            DataFrame with product features added
        """
        result_df = df.copy()
        
        for col1, col2 in column_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Skip datetime columns to avoid DatetimeArray reduction errors
                if pd.api.types.is_datetime64_any_dtype(df[col1]) or pd.api.types.is_datetime64_any_dtype(df[col2]):
                    self.logger.warning(f"Skipping product features for datetime columns: {col1} or {col2}")
                    continue
                    
                product_col_name = f"{col1}_mult_{col2}"
                result_df[product_col_name] = df[col1] * df[col2]
        
        self.logger.info(f"Created {len(column_pairs)} product features")
        
        return result_df
    
    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        degrees: List[int] = None
    ) -> pd.DataFrame:
        """
        Create polynomial features.
        
        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features for
            degrees: Polynomial degrees (default: [2, 3])
        
        Returns:
            DataFrame with polynomial features added
        """
        if degrees is None:
            degrees = [2, 3]
        
        result_df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            # Skip datetime columns to avoid DatetimeArray reduction errors
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.logger.warning(f"Skipping polynomial features for datetime column: {col}")
                continue
            
            for degree in degrees:
                poly_col_name = f"{col}_pow_{degree}"
                result_df[poly_col_name] = df[col] ** degree
        
        poly_count = len(columns) * len(degrees)
        self.logger.info(f"Created {poly_count} polynomial features")
        
        return result_df


class RegimeFeatures:
    """Identifies market conditions (trending, ranging, volatile, calm)."""
    
    def __init__(self):
        """Initialize regime features generator."""
        self.logger = logger.bind(component="regime_features")
    
    def create_volatility_regime(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        window: int = 20,
        threshold_percentile: float = 70
    ) -> pd.DataFrame:
        """
        Create volatility regime features.
        
        Args:
            df: Input DataFrame
            price_col: Price column to analyze
            window: Rolling window for volatility calculation
            threshold_percentile: Percentile threshold for high volatility
        
        Returns:
            DataFrame with volatility regime features
        """
        result_df = df.copy()
        
        if price_col not in df.columns:
            return result_df
        
        # Calculate rolling volatility
        returns = df[price_col].pct_change()
        volatility = returns.rolling(window=window).std()
        
        # Define volatility regimes
        vol_threshold = volatility.quantile(threshold_percentile / 100)
        
        result_df['volatility_regime'] = np.where(
            volatility > vol_threshold, 1, 0  # 1 = high vol, 0 = low vol
        )
        
        result_df['volatility_percentile'] = volatility.rolling(window=window*2).rank(pct=True)
        
        self.logger.info("Created volatility regime features")
        
        return result_df
    
    def create_trend_regime(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        short_window: int = 10,
        long_window: int = 50
    ) -> pd.DataFrame:
        """
        Create trend regime features.
        
        Args:
            df: Input DataFrame
            price_col: Price column to analyze
            short_window: Short-term moving average window
            long_window: Long-term moving average window
        
        Returns:
            DataFrame with trend regime features
        """
        result_df = df.copy()
        
        if price_col not in df.columns:
            return result_df
        
        # Calculate moving averages
        sma_short = df[price_col].rolling(window=short_window).mean()
        sma_long = df[price_col].rolling(window=long_window).mean()
        
        # Define trend regimes
        result_df['trend_regime'] = np.where(
            sma_short > sma_long, 1,  # Uptrend
            np.where(sma_short < sma_long, -1, 0)  # Downtrend, Sideways
        )
        
        # Trend strength
        result_df['trend_strength'] = np.abs(sma_short - sma_long) / sma_long
        
        self.logger.info("Created trend regime features")
        
        return result_df


class SeasonalFeatures:
    """Captures time-based patterns (hour of day, day of week, month effects)."""
    
    def __init__(self):
        """Initialize seasonal features generator."""
        self.logger = logger.bind(component="seasonal_features")
    
    def create_time_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: Input DataFrame
            timestamp_col: Timestamp column name
        
        Returns:
            DataFrame with time features added
        """
        result_df = df.copy()
        
        if timestamp_col not in df.columns:
            return result_df
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            result_df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        dt = result_df[timestamp_col]
        
        # Extract time components - handle pandas datetime attributes safely
        result_df['hour'] = dt.dt.hour.astype('int64')
        result_df['day_of_week'] = dt.dt.dayofweek.astype('int64')
        result_df['day_of_month'] = dt.dt.day.astype('int64')
        
        # Safe handling for isocalendar week which can cause DatetimeArray reduction issues
        try:
            # Handle pandas >= 1.1.0 where isocalendar returns a DataFrame
            if hasattr(dt.dt, 'isocalendar') and callable(getattr(dt.dt, 'isocalendar')):
                iso_calendar = dt.dt.isocalendar()
                if hasattr(iso_calendar, 'week'):
                    result_df['week_of_year'] = iso_calendar.week.astype('int64')
                elif isinstance(iso_calendar, pd.DataFrame) and 'week' in iso_calendar.columns:
                    result_df['week_of_year'] = iso_calendar['week'].astype('int64')
                else:
                    # Fallback for older pandas versions
                    result_df['week_of_year'] = dt.dt.week.astype('int64') if hasattr(dt.dt, 'week') else dt.apply(lambda x: x.isocalendar()[1]).astype('int64')
            else:
                # Very old pandas versions
                result_df['week_of_year'] = dt.dt.week.astype('int64') if hasattr(dt.dt, 'week') else dt.apply(lambda x: x.isocalendar()[1]).astype('int64')
        except Exception as e:
            self.logger.warning(f"Could not extract week_of_year: {e}")
            # Safe fallback
            result_df['week_of_year'] = 0
        
        result_df['month'] = dt.dt.month.astype('int64')
        result_df['quarter'] = dt.dt.quarter.astype('int64')
        result_df['year'] = dt.dt.year.astype('int64')
        
        # Cyclical encoding for time features
        result_df['hour_sin'] = np.sin(2 * np.pi * result_df['hour'] / 24)
        result_df['hour_cos'] = np.cos(2 * np.pi * result_df['hour'] / 24)
        result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
        
        # Binary features - explicitly convert to int to avoid potential issues
        result_df['is_weekend'] = (result_df['day_of_week'] >= 5).astype('int64')
        result_df['is_month_start'] = (result_df['day_of_month'] <= 3).astype('int64')
        result_df['is_month_end'] = (result_df['day_of_month'] >= 28).astype('int64')
        result_df['is_quarter_start'] = result_df['month'].isin([1, 4, 7, 10]).astype('int64')
        
        self.logger.info("Created time-based seasonal features")
        
        return result_df


class FeaturePipeline:
    """Main feature engineering pipeline with proper train/test separation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature pipeline."""
        from .config import get_model_dict
        self.config = config or get_model_dict(get_config().features)
        self.logger = logger.bind(component="feature_pipeline")
        
        # Initialize components
        self.temporal_validator = TemporalValidator()
        self.indicators_calculator = TechnicalIndicators()
        self.lagging_features = LaggingFeatures()
        self.rolling_features = RollingFeatures()
        self.interaction_features = InteractionFeatures()
        self.regime_features = RegimeFeatures()
        self.seasonal_features = SeasonalFeatures()
        
        # Scalers (will be fitted on training data)
        self.scalers = {}
        self.fitted = False
    
    def engineer_features(
        self,
        df: pd.DataFrame,
        fit_scalers: bool = True,
        validate_temporal: bool = True
    ) -> pd.DataFrame:
        """
        Engineer comprehensive features from OHLCV data.
        
        Args:
            df: Input DataFrame with OHLCV data
            fit_scalers: Whether to fit scalers (True for training data)
            validate_temporal: Whether to validate temporal correctness
        
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering", records=len(df))
        
        try:
            # Start with input data
            result_df = df.copy()
            
            # Step 1: Calculate technical indicators
            result_df = self.indicators_calculator.calculate_all_indicators(result_df)
            
            # Step 2: Create lagged features
            price_volume_cols = ['open', 'high', 'low', 'close', 'volume']
            indicator_cols = [col for col in result_df.columns 
                            if any(indicator in col for indicator in ['SMA', 'EMA', 'RSI', 'MACD'])]
            
            lag_periods = self.config.get('lag_periods', [1, 2, 3, 5, 10])
            result_df = self.lagging_features.create_lag_features(
                result_df, price_volume_cols + indicator_cols[:10], lag_periods
            )
            
            # Step 3: Create rolling statistical features
            rolling_windows = self.config.get('rolling_windows', [5, 10, 20, 50])
            result_df = self.rolling_features.create_rolling_statistics(
                result_df, price_volume_cols, rolling_windows
            )
            
            # Step 4: Create interaction features
            # Price ratios
            result_df = self.interaction_features.create_ratio_features(
                result_df, ['high', 'close'], ['low', 'open']
            )
            
            # Volume interactions
            volume_price_pairs = [('volume', 'close'), ('volume', 'high'), ('volume', 'low')]
            result_df = self.interaction_features.create_product_features(
                result_df, volume_price_pairs
            )
            
            # Polynomial features for key indicators
            key_indicators = ['RSI_14', 'close']
            result_df = self.interaction_features.create_polynomial_features(
                result_df, key_indicators, [2]
            )
            
            # Step 5: Create regime features
            result_df = self.regime_features.create_volatility_regime(result_df)
            result_df = self.regime_features.create_trend_regime(result_df)
            
            # Step 6: Create seasonal features
            result_df = self.seasonal_features.create_time_features(result_df)
            
            # Step 7: Create target variables (for supervised learning)
            horizons = self.config.get('horizons', [1, 3, 5])
            for horizon in horizons:
                # Return targets
                result_df[f'target_return_{horizon}'] = result_df['close'].pct_change(periods=horizon).shift(-horizon)
                
                # Direction targets (binary classification)
                result_df[f'target_direction_{horizon}'] = (result_df[f'target_return_{horizon}'] > 0).astype(int)
                
                # Volatility targets
                result_df[f'target_volatility_{horizon}'] = result_df['close'].pct_change().rolling(window=horizon).std().shift(-horizon)
            
            # Step 8: Apply scaling if requested
            if fit_scalers or self.fitted:
                result_df = self._apply_scaling(result_df, fit_scalers)
            
            # Step 9: Temporal validation
            if validate_temporal:
                self.temporal_validator.validate_feature_matrix(result_df)
            
            # Step 10: Optimize memory usage
            result_df = optimize_dataframe_memory(result_df)
            
            feature_count = len([col for col in result_df.columns if col not in df.columns])
            self.logger.info(f"Feature engineering completed: {feature_count} features created")
            
            return result_df
        
        except Exception as e:
            raise FeatureEngineeringError(f"Feature engineering failed: {e}")
    
    def _apply_scaling(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Apply feature scaling."""
        result_df = df.copy()
        
        # Identify numeric columns to scale (exclude targets and metadata)
        exclude_patterns = ['target_', 'timestamp', 'symbol', 'regime']
        numeric_cols = [col for col in df.columns 
                       if df[col].dtype in ['float64', 'int64'] 
                       and not any(pattern in col for pattern in exclude_patterns)]
        
        if not numeric_cols:
            return result_df
        
        scaler_type = self.config.get('scaler_type', 'robust')
        
        if scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            self.logger.warning(f"Unknown scaler type: {scaler_type}, using RobustScaler")
            scaler = RobustScaler()
        
        if fit:
            # Fit scaler on non-null data
            valid_data = df[numeric_cols].dropna()
            if len(valid_data) > 0:
                scaler.fit(valid_data)
                self.scalers['main'] = scaler
                self.fitted = True
        
        if 'main' in self.scalers:
            # Transform data
            scaled_data = self.scalers['main'].transform(df[numeric_cols].fillna(0))
            result_df[numeric_cols] = scaled_data
        
        return result_df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature column names (excluding targets and metadata)."""
        exclude_patterns = ['target_', 'timestamp', 'symbol']
        return [col for col in df.columns 
                if not any(pattern in col for pattern in exclude_patterns)]


# Factory function
def create_feature_pipeline(config: Optional[Dict[str, Any]] = None) -> FeaturePipeline:
    """Create feature engineering pipeline."""
    return FeaturePipeline(config)


# Example usage
if __name__ == "__main__":
    # Test feature engineering
    pipeline = create_feature_pipeline()
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=200, freq='1H')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'BTCUSDT',
        'open': 50000 + np.cumsum(np.random.randn(200) * 100),
        'high': 50000 + np.cumsum(np.random.randn(200) * 100) + 200,
        'low': 50000 + np.cumsum(np.random.randn(200) * 100) - 200,
        'close': 50000 + np.cumsum(np.random.randn(200) * 100),
        'volume': np.random.uniform(100, 1000, 200)
    })
    
    # Ensure OHLC relationships
    sample_data['high'] = sample_data[['open', 'high', 'close']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'low', 'close']].min(axis=1)
    
    try:
        result = pipeline.engineer_features(sample_data)
        feature_names = pipeline.get_feature_names(result)
        print(f"Successfully engineered {len(feature_names)} features")
        print("Sample features:", feature_names[:20])
    except Exception as e:
        print(f"Error: {e}")
