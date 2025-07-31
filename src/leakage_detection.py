"""
Data Leakage Detection and Prevention System.
Comprehensive framework to detect and prevent data leakage in time series ML pipelines.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import warnings

from .logging_config import StructuredLogger
from .exceptions import DataLeakageError, ValidationError
from .utils import ensure_directory


@dataclass
class LeakageViolation:
    """Represents a data leakage violation."""
    feature_name: str
    violation_type: str
    description: str
    severity: str  # 'critical', 'warning', 'info'
    timestamp: datetime
    affected_samples: List[int]
    suggested_fix: str


class TemporalValidator:
    """Validates temporal consistency of features to prevent data leakage."""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
        self.violations: List[LeakageViolation] = []
    
    def validate_feature_matrix(
        self, 
        features: pd.DataFrame, 
        timestamp_col: str = 'timestamp',
        target_col: Optional[str] = None
    ) -> Tuple[bool, List[LeakageViolation]]:
        """
        Validate entire feature matrix for temporal consistency.
        
        Args:
            features: Feature matrix with timestamp column
            timestamp_col: Name of timestamp column
            target_col: Name of target column (if present)
            
        Returns:
            Tuple of (is_valid, violations_list)
        """
        self.violations = []
        
        if timestamp_col not in features.columns:
            violation = LeakageViolation(
                feature_name=timestamp_col,
                violation_type='missing_timestamp',
                description=f"Timestamp column '{timestamp_col}' not found",
                severity='critical',
                timestamp=datetime.now(),
                affected_samples=[],
                suggested_fix="Ensure timestamp column is present in feature matrix"
            )
            self.violations.append(violation)
            return False, self.violations
        
        # Sort by timestamp for validation
        features_sorted = features.sort_values(timestamp_col)
        
        # Check for future information leakage
        self._check_future_information(features_sorted, timestamp_col)
        
        # Check rolling calculations
        self._check_rolling_calculations(features_sorted, timestamp_col)
        
        # Check lagged features
        self._check_lagged_features(features_sorted, timestamp_col)
        
        # Check for target leakage
        if target_col and target_col in features.columns:
            self._check_target_leakage(features_sorted, timestamp_col, target_col)
        
        # Check feature naming conventions
        self._check_feature_naming(features.columns)
        
        is_valid = not any(v.severity == 'critical' for v in self.violations)
        
        self.logger.info(
            "Temporal validation completed",
            total_violations=len(self.violations),
            critical_violations=sum(1 for v in self.violations if v.severity == 'critical'),
            is_valid=is_valid
        )
        
        return is_valid, self.violations
    
    def _check_future_information(self, features: pd.DataFrame, timestamp_col: str):
        """Check for features that might contain future information."""
        timestamps = features[timestamp_col]
        
        # Check for non-monotonic timestamps (potential future info)
        if not timestamps.is_monotonic_increasing:
            violation = LeakageViolation(
                feature_name=timestamp_col,
                violation_type='non_monotonic_timestamps',
                description="Timestamps are not monotonically increasing",
                severity='critical',
                timestamp=datetime.now(),
                affected_samples=list(range(len(features))),
                suggested_fix="Ensure data is properly sorted by timestamp"
            )
            self.violations.append(violation)
        
        # Check for features with suspicious forward-looking patterns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != timestamp_col]
        
        for col in numeric_cols:
            if self._has_forward_looking_pattern(features[col], timestamps):
                violation = LeakageViolation(
                    feature_name=col,
                    violation_type='forward_looking_pattern',
                    description=f"Feature '{col}' shows forward-looking patterns",
                    severity='warning',
                    timestamp=datetime.now(),
                    affected_samples=[],
                    suggested_fix="Review feature calculation to ensure only past data is used"
                )
                self.violations.append(violation)
    
    def _check_rolling_calculations(self, features: pd.DataFrame, timestamp_col: str):
        """Check rolling calculations for proper temporal boundaries."""
        rolling_features = [col for col in features.columns 
                          if any(keyword in col.lower() for keyword in 
                               ['rolling', 'ma', 'sma', 'ema', 'mean', 'std', 'min', 'max'])]
        
        for col in rolling_features:
            # Check for NaN values at the beginning (expected for rolling calculations)
            first_valid_idx = features[col].first_valid_index()
            if first_valid_idx is None or first_valid_idx == 0:
                violation = LeakageViolation(
                    feature_name=col,
                    violation_type='suspicious_rolling_calculation',
                    description=f"Rolling feature '{col}' has no NaN values at start",
                    severity='warning',
                    timestamp=datetime.now(),
                    affected_samples=[0],
                    suggested_fix="Ensure rolling calculations have proper warm-up period"
                )
                self.violations.append(violation)
    
    def _check_lagged_features(self, features: pd.DataFrame, timestamp_col: str):
        """Check lagged features for proper temporal offset."""
        lagged_features = [col for col in features.columns 
                         if any(keyword in col.lower() for keyword in 
                              ['lag', 'shift', 'prev', 'past'])]
        
        for col in lagged_features:
            # Extract lag period from feature name if possible
            lag_period = self._extract_lag_period(col)
            if lag_period and lag_period > 0:
                # Check if feature values are properly shifted
                if not self._validate_lag_shift(features[col], lag_period):
                    violation = LeakageViolation(
                        feature_name=col,
                        violation_type='incorrect_lag_calculation',
                        description=f"Lagged feature '{col}' may not be properly shifted",
                        severity='warning',
                        timestamp=datetime.now(),
                        affected_samples=[],
                        suggested_fix=f"Ensure feature is shifted by {lag_period} periods"
                    )
                    self.violations.append(violation)
    
    def _check_target_leakage(self, features: pd.DataFrame, timestamp_col: str, target_col: str):
        """Check for target leakage in features."""
        target_values = features[target_col]
        
        # Check correlation between features and future targets
        for col in features.columns:
            if col in [timestamp_col, target_col]:
                continue
                
            if features[col].dtype in [np.number]:
                # Check correlation with future target values
                future_correlation = self._calculate_future_correlation(
                    features[col], target_values
                )
                
                if abs(future_correlation) > 0.8:  # High correlation threshold
                    violation = LeakageViolation(
                        feature_name=col,
                        violation_type='high_future_correlation',
                        description=f"Feature '{col}' highly correlated with future targets",
                        severity='critical',
                        timestamp=datetime.now(),
                        affected_samples=[],
                        suggested_fix="Remove or modify feature to eliminate future information"
                    )
                    self.violations.append(violation)
    
    def _check_feature_naming(self, columns: List[str]):
        """Check feature naming conventions for potential leakage indicators."""
        suspicious_keywords = [
            'future', 'next', 'forward', 'ahead', 'tomorrow', 
            'target', 'label', 'actual', 'real'
        ]
        
        for col in columns:
            col_lower = col.lower()
            for keyword in suspicious_keywords:
                if keyword in col_lower:
                    violation = LeakageViolation(
                        feature_name=col,
                        violation_type='suspicious_naming',
                        description=f"Feature name '{col}' contains suspicious keyword '{keyword}'",
                        severity='warning',
                        timestamp=datetime.now(),
                        affected_samples=[],
                        suggested_fix="Review feature calculation and rename if necessary"
                    )
                    self.violations.append(violation)
                    break
    
    def _has_forward_looking_pattern(self, series: pd.Series, timestamps: pd.Series) -> bool:
        """Check if series has forward-looking patterns."""
        # Simple heuristic: check if series has perfect predictive power
        # This is a simplified check - in practice, more sophisticated methods would be used
        if len(series) < 10:
            return False
        
        # Check for unrealistic patterns
        diff = series.diff()
        if diff.std() == 0 and diff.mean() != 0:  # Perfect trend
            return True
        
        # Check for values that are too predictable
        autocorr = series.autocorr(lag=1)
        if autocorr and abs(autocorr) > 0.99:
            return True
        
        return False
    
    def _extract_lag_period(self, feature_name: str) -> Optional[int]:
        """Extract lag period from feature name."""
        import re
        
        # Look for patterns like "lag_5", "shift_10", etc.
        patterns = [
            r'lag_(\d+)', r'shift_(\d+)', r'prev_(\d+)', 
            r'lag(\d+)', r'shift(\d+)', r'prev(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, feature_name.lower())
            if match:
                return int(match.group(1))
        
        return None
    
    def _validate_lag_shift(self, series: pd.Series, lag_period: int) -> bool:
        """Validate that series is properly lagged."""
        if len(series) <= lag_period:
            return True  # Too short to validate
        
        # Check if first lag_period values are NaN (expected for proper lagging)
        first_values = series.iloc[:lag_period]
        return first_values.isna().all()
    
    def _calculate_future_correlation(self, feature: pd.Series, target: pd.Series) -> float:
        """Calculate correlation between feature and future target values."""
        if len(feature) < 2 or len(target) < 2:
            return 0.0
        
        # Shift target to create "future" values
        future_target = target.shift(-1)
        
        # Calculate correlation, handling NaN values
        valid_mask = ~(feature.isna() | future_target.isna())
        if valid_mask.sum() < 2:
            return 0.0
        
        correlation = np.corrcoef(
            feature[valid_mask], 
            future_target[valid_mask]
        )[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0


class LeakageDetector:
    """Main class for comprehensive data leakage detection."""
    
    def __init__(self, output_dir: str = "./artefacts/leakage_reports"):
        self.logger = StructuredLogger(__name__)
        self.output_dir = Path(output_dir)
        ensure_directory(self.output_dir)
        
        self.temporal_validator = TemporalValidator()
        self.detected_violations: List[LeakageViolation] = []
    
    def scan_feature_matrix(
        self, 
        features: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        target_col: Optional[str] = None,
        save_report: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive scan of feature matrix for data leakage.
        
        Args:
            features: Feature matrix to scan
            timestamp_col: Name of timestamp column
            target_col: Name of target column (if present)
            save_report: Whether to save detailed report
            
        Returns:
            Dictionary with scan results
        """
        self.logger.info("Starting comprehensive leakage detection scan")
        
        # Temporal validation
        is_temporally_valid, temporal_violations = self.temporal_validator.validate_feature_matrix(
            features, timestamp_col, target_col
        )
        
        self.detected_violations.extend(temporal_violations)
        
        # Feature lineage analysis
        lineage_violations = self._analyze_feature_lineage(features.columns)
        self.detected_violations.extend(lineage_violations)
        
        # Statistical analysis
        statistical_violations = self._statistical_leakage_analysis(features, timestamp_col)
        self.detected_violations.extend(statistical_violations)
        
        # Generate report
        report = self._generate_leakage_report(features, timestamp_col)
        
        if save_report:
            self._save_report(report)
        
        self.logger.info(
            "Leakage detection scan completed",
            total_violations=len(self.detected_violations),
            critical_violations=sum(1 for v in self.detected_violations if v.severity == 'critical'),
            is_clean=len([v for v in self.detected_violations if v.severity == 'critical']) == 0
        )
        
        return report
    
    def _analyze_feature_lineage(self, feature_names: List[str]) -> List[LeakageViolation]:
        """Analyze feature lineage for potential leakage sources."""
        violations = []
        
        # Check for features that might be derived from targets
        target_derived_patterns = [
            'target_', 'label_', 'y_', 'actual_', 'real_', 'true_'
        ]
        
        for feature in feature_names:
            feature_lower = feature.lower()
            for pattern in target_derived_patterns:
                if pattern in feature_lower:
                    violation = LeakageViolation(
                        feature_name=feature,
                        violation_type='target_derived_feature',
                        description=f"Feature '{feature}' appears to be derived from target",
                        severity='critical',
                        timestamp=datetime.now(),
                        affected_samples=[],
                        suggested_fix="Remove feature or verify it doesn't contain target information"
                    )
                    violations.append(violation)
                    break
        
        return violations
    
    def _statistical_leakage_analysis(
        self, 
        features: pd.DataFrame, 
        timestamp_col: str
    ) -> List[LeakageViolation]:
        """Statistical analysis to detect potential leakage."""
        violations = []
        
        # Check for features with unrealistic predictive power
        numeric_features = features.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col != timestamp_col]
        
        for feature in numeric_features:
            if self._has_unrealistic_predictive_power(features[feature]):
                violation = LeakageViolation(
                    feature_name=feature,
                    violation_type='unrealistic_predictive_power',
                    description=f"Feature '{feature}' has unrealistic predictive characteristics",
                    severity='warning',
                    timestamp=datetime.now(),
                    affected_samples=[],
                    suggested_fix="Review feature calculation methodology"
                )
                violations.append(violation)
        
        return violations
    
    def _has_unrealistic_predictive_power(self, series: pd.Series) -> bool:
        """Check if feature has unrealistic predictive characteristics."""
        if len(series) < 10:
            return False
        
        # Check for constant values (except NaN)
        non_nan_values = series.dropna()
        if len(non_nan_values) > 0 and non_nan_values.nunique() == 1:
            return True
        
        # Check for perfect linear trends
        if len(non_nan_values) > 2:
            x = np.arange(len(non_nan_values))
            correlation = np.corrcoef(x, non_nan_values)[0, 1]
            if abs(correlation) > 0.999:
                return True
        
        return False
    
    def _generate_leakage_report(
        self, 
        features: pd.DataFrame, 
        timestamp_col: str
    ) -> Dict[str, Any]:
        """Generate comprehensive leakage detection report."""
        critical_violations = [v for v in self.detected_violations if v.severity == 'critical']
        warning_violations = [v for v in self.detected_violations if v.severity == 'warning']
        
        report = {
            'scan_timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'num_samples': len(features),
                'num_features': len(features.columns) - 1,  # Exclude timestamp
                'time_range': {
                    'start': features[timestamp_col].min().isoformat() if timestamp_col in features.columns else None,
                    'end': features[timestamp_col].max().isoformat() if timestamp_col in features.columns else None
                }
            },
            'summary': {
                'total_violations': len(self.detected_violations),
                'critical_violations': len(critical_violations),
                'warning_violations': len(warning_violations),
                'is_leakage_free': len(critical_violations) == 0
            },
            'violations': [
                {
                    'feature_name': v.feature_name,
                    'violation_type': v.violation_type,
                    'description': v.description,
                    'severity': v.severity,
                    'suggested_fix': v.suggested_fix,
                    'timestamp': v.timestamp.isoformat()
                }
                for v in self.detected_violations
            ],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on detected violations."""
        recommendations = []
        
        critical_count = sum(1 for v in self.detected_violations if v.severity == 'critical')
        warning_count = sum(1 for v in self.detected_violations if v.severity == 'warning')
        
        if critical_count > 0:
            recommendations.append(
                f"CRITICAL: {critical_count} critical data leakage violations detected. "
                "These must be fixed before model training."
            )
        
        if warning_count > 0:
            recommendations.append(
                f"WARNING: {warning_count} potential leakage issues detected. "
                "Review these features carefully."
            )
        
        if critical_count == 0 and warning_count == 0:
            recommendations.append(
                "No data leakage violations detected. Feature matrix appears clean."
            )
        
        # Specific recommendations based on violation types
        violation_types = set(v.violation_type for v in self.detected_violations)
        
        if 'target_derived_feature' in violation_types:
            recommendations.append(
                "Remove features that appear to be derived from target variables."
            )
        
        if 'forward_looking_pattern' in violation_types:
            recommendations.append(
                "Review rolling calculations and ensure they only use historical data."
            )
        
        if 'high_future_correlation' in violation_types:
            recommendations.append(
                "Investigate features with high correlation to future target values."
            )
        
        return recommendations
    
    def _save_report(self, report: Dict[str, Any]):
        """Save leakage detection report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"leakage_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Leakage detection report saved to {report_path}")
    
    def get_clean_features(
        self, 
        features: pd.DataFrame, 
        remove_critical: bool = True,
        remove_warnings: bool = False
    ) -> pd.DataFrame:
        """
        Return cleaned feature matrix with leakage violations removed.
        
        Args:
            features: Original feature matrix
            remove_critical: Remove features with critical violations
            remove_warnings: Remove features with warning violations
            
        Returns:
            Cleaned feature matrix
        """
        features_to_remove = set()
        
        for violation in self.detected_violations:
            if (violation.severity == 'critical' and remove_critical) or \
               (violation.severity == 'warning' and remove_warnings):
                features_to_remove.add(violation.feature_name)
        
        cleaned_features = features.drop(columns=list(features_to_remove), errors='ignore')
        
        self.logger.info(
            "Feature matrix cleaned",
            original_features=len(features.columns),
            removed_features=len(features_to_remove),
            remaining_features=len(cleaned_features.columns)
        )
        
        return cleaned_features


class AutomaticLeakageCorrection:
    """Automatic correction of common data leakage patterns."""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
    
    def correct_rolling_calculations(
        self, 
        features: pd.DataFrame, 
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """Correct rolling calculations to prevent leakage."""
        corrected_features = features.copy()
        
        rolling_features = [col for col in features.columns 
                          if any(keyword in col.lower() for keyword in 
                               ['rolling', 'ma', 'sma', 'ema'])]
        
        for col in rolling_features:
            # Extract window size from column name if possible
            window_size = self._extract_window_size(col)
            if window_size:
                # Recalculate with proper temporal boundaries
                base_col = self._get_base_column(col, features.columns)
                if base_col:
                    corrected_features[col] = (
                        features[base_col]
                        .rolling(window=window_size, min_periods=window_size)
                        .mean()
                    )
        
        return corrected_features
    
    def correct_lagged_features(
        self, 
        features: pd.DataFrame, 
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """Correct lagged features to ensure proper temporal offset."""
        corrected_features = features.copy()
        
        lagged_features = [col for col in features.columns 
                         if any(keyword in col.lower() for keyword in 
                              ['lag', 'shift', 'prev'])]
        
        for col in lagged_features:
            lag_period = self._extract_lag_period(col)
            if lag_period:
                base_col = self._get_base_column(col, features.columns)
                if base_col:
                    corrected_features[col] = features[base_col].shift(lag_period)
        
        return corrected_features
    
    def _extract_window_size(self, feature_name: str) -> Optional[int]:
        """Extract window size from feature name."""
        import re
        
        patterns = [
            r'ma_?(\d+)', r'sma_?(\d+)', r'ema_?(\d+)', 
            r'rolling_?(\d+)', r'window_?(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, feature_name.lower())
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_lag_period(self, feature_name: str) -> Optional[int]:
        """Extract lag period from feature name."""
        import re
        
        patterns = [
            r'lag_?(\d+)', r'shift_?(\d+)', r'prev_?(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, feature_name.lower())
            if match:
                return int(match.group(1))
        
        return None
    
    def _get_base_column(self, derived_col: str, all_columns: List[str]) -> Optional[str]:
        """Try to identify the base column for a derived feature."""
        # Simple heuristic: look for columns with similar names
        base_name = derived_col.lower()
        
        # Remove common suffixes/prefixes
        for pattern in ['_ma_', '_sma_', '_ema_', '_lag_', '_shift_', '_prev_']:
            if pattern in base_name:
                base_name = base_name.split(pattern)[0]
                break
        
        # Look for matching column
        for col in all_columns:
            if col.lower() == base_name or base_name in col.lower():
                return col
        
        return None
