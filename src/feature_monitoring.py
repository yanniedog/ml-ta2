"""
Feature Monitoring and Drift Detection for ML-TA System

This module monitors feature distributions and detects drift in real-time
to ensure model performance doesn't degrade due to changing data patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import warnings
import json

# Handle optional dependencies gracefully
try:
    from scipy import stats
    from scipy.stats import ks_2samp, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Simple fallback for statistical tests
    class stats:
        @staticmethod
        def chi2():
            return type('chi2', (), {'cdf': lambda x, df: 0.5})()
    
    def ks_2samp(x, y):
        # Simple fallback - just return moderate p-value
        return 0.1, 0.5
    
    def chi2_contingency(observed):
        # Simple fallback
        return 1.0, 0.5, 1, None

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
from .utils import ensure_directory

logger = get_logger("feature_monitoring").get_logger()
warnings.filterwarnings('ignore', category=FutureWarning)


class StatisticalDriftDetector:
    """Detects feature drift using statistical tests."""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize statistical drift detector.
        
        Args:
            significance_level: P-value threshold for detecting drift
        """
        self.significance_level = significance_level
        self.logger = logger.bind(component="statistical_drift_detector")
        self.reference_distributions = {}
        self.drift_history = []
    
    def fit_reference(self, X: pd.DataFrame, feature_types: Optional[Dict[str, str]] = None) -> None:
        """
        Fit reference distributions from training data.
        
        Args:
            X: Reference feature matrix
            feature_types: Dictionary mapping feature names to types ('continuous' or 'categorical')
        """
        self.reference_distributions = {}
        
        if feature_types is None:
            # Auto-detect feature types
            feature_types = self._detect_feature_types(X)
        
        for column in X.columns:
            if column not in feature_types:
                continue
            
            feature_type = feature_types[column]
            data = X[column].dropna()
            
            if len(data) == 0:
                continue
            
            if feature_type == 'continuous':
                # Store statistics for continuous features
                self.reference_distributions[column] = {
                    'type': 'continuous',
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'quantiles': data.quantile([0.25, 0.5, 0.75]).to_dict(),
                    'data_sample': data.sample(min(1000, len(data))).tolist()
                }
            else:
                # Store value counts for categorical features
                value_counts = data.value_counts(normalize=True)
                self.reference_distributions[column] = {
                    'type': 'categorical',
                    'value_counts': value_counts.to_dict(),
                    'unique_values': data.unique().tolist()
                }
        
        self.logger.info(f"Reference distributions fitted for {len(self.reference_distributions)} features")
    
    def detect_drift(self, X: pd.DataFrame, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Detect drift in new data compared to reference.
        
        Args:
            X: New feature matrix
            timestamp: Timestamp of the data
        
        Returns:
            Dictionary containing drift detection results
        """
        if not self.reference_distributions:
            raise ValueError("Reference distributions not fitted. Call fit_reference() first.")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        drift_results = {
            'timestamp': timestamp,
            'features_tested': 0,
            'features_with_drift': 0,
            'drift_detected': False,
            'feature_results': {}
        }
        
        for column in X.columns:
            if column not in self.reference_distributions:
                continue
            
            drift_results['features_tested'] += 1
            ref_dist = self.reference_distributions[column]
            new_data = X[column].dropna()
            
            if len(new_data) == 0:
                continue
            
            if ref_dist['type'] == 'continuous':
                # Use Kolmogorov-Smirnov test for continuous features
                ref_sample = ref_dist['data_sample']
                statistic, p_value = ks_2samp(ref_sample, new_data)
                
                drift_detected = p_value < self.significance_level
                
                drift_results['feature_results'][column] = {
                    'test': 'kolmogorov_smirnov',
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'drift_detected': drift_detected,
                    'mean_shift': float(new_data.mean() - ref_dist['mean']),
                    'std_shift': float(new_data.std() - ref_dist['std'])
                }
            
            else:
                # Use Chi-square test for categorical features
                new_value_counts = new_data.value_counts(normalize=True)
                ref_value_counts = ref_dist['value_counts']
                
                # Align categories
                all_categories = set(ref_value_counts.keys()) | set(new_value_counts.keys())
                ref_probs = [ref_value_counts.get(cat, 0) for cat in all_categories]
                new_probs = [new_value_counts.get(cat, 0) for cat in all_categories]
                
                # Expected frequencies
                expected = np.array(ref_probs) * len(new_data)
                observed = np.array(new_probs) * len(new_data)
                
                # Avoid zero frequencies
                expected = np.maximum(expected, 1e-6)
                observed = np.maximum(observed, 1e-6)
                
                try:
                    statistic = np.sum((observed - expected) ** 2 / expected)
                    df = len(all_categories) - 1
                    p_value = 1 - stats.chi2.cdf(statistic, df) if df > 0 else 1.0
                    
                    drift_detected = p_value < self.significance_level
                    
                    drift_results['feature_results'][column] = {
                        'test': 'chi_square',
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'drift_detected': drift_detected,
                        'new_categories': list(set(new_value_counts.keys()) - set(ref_value_counts.keys()))
                    }
                
                except Exception as e:
                    self.logger.warning(f"Chi-square test failed for {column}: {e}")
                    continue
            
            if drift_results['feature_results'][column]['drift_detected']:
                drift_results['features_with_drift'] += 1
        
        drift_results['drift_detected'] = drift_results['features_with_drift'] > 0
        drift_results['drift_ratio'] = (
            drift_results['features_with_drift'] / drift_results['features_tested']
            if drift_results['features_tested'] > 0 else 0
        )
        
        # Store in history
        self.drift_history.append(drift_results)
        
        self.logger.info(
            f"Drift detection completed",
            features_tested=drift_results['features_tested'],
            features_with_drift=drift_results['features_with_drift'],
            drift_ratio=drift_results['drift_ratio']
        )
        
        return drift_results
    
    def _detect_feature_types(self, X: pd.DataFrame) -> Dict[str, str]:
        """Auto-detect feature types."""
        feature_types = {}
        
        for column in X.columns:
            if X[column].dtype in ['object', 'category']:
                feature_types[column] = 'categorical'
            elif X[column].nunique() < 10:  # Assume categorical if few unique values
                feature_types[column] = 'categorical'
            else:
                feature_types[column] = 'continuous'
        
        return feature_types
    
    def get_drift_summary(self, window_hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent drift detections."""
        if not self.drift_history:
            return {'no_data': True}
        
        cutoff_time = datetime.now() - timedelta(hours=window_hours)
        recent_results = [
            result for result in self.drift_history
            if result['timestamp'] >= cutoff_time
        ]
        
        if not recent_results:
            return {'no_recent_data': True}
        
        # Aggregate statistics
        total_tests = sum(r['features_tested'] for r in recent_results)
        total_drift = sum(r['features_with_drift'] for r in recent_results)
        
        # Feature-level drift frequency
        feature_drift_counts = {}
        for result in recent_results:
            for feature, feature_result in result['feature_results'].items():
                if feature not in feature_drift_counts:
                    feature_drift_counts[feature] = {'total': 0, 'drift': 0}
                feature_drift_counts[feature]['total'] += 1
                if feature_result['drift_detected']:
                    feature_drift_counts[feature]['drift'] += 1
        
        # Calculate drift rates
        feature_drift_rates = {
            feature: counts['drift'] / counts['total']
            for feature, counts in feature_drift_counts.items()
        }
        
        return {
            'window_hours': window_hours,
            'total_tests': total_tests,
            'total_drift_detections': total_drift,
            'overall_drift_rate': total_drift / total_tests if total_tests > 0 else 0,
            'feature_drift_rates': feature_drift_rates,
            'most_drifted_features': sorted(
                feature_drift_rates.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }


class PerformanceDriftDetector:
    """Monitors model performance metrics for drift."""
    
    def __init__(self, window_size: int = 100, alert_threshold: float = 0.05):
        """
        Initialize performance drift detector.
        
        Args:
            window_size: Size of rolling window for performance calculation
            alert_threshold: Threshold for performance degradation alert
        """
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.logger = logger.bind(component="performance_drift_detector")
        
        self.performance_history = []
        self.baseline_performance = None
    
    def set_baseline(self, performance_metrics: Dict[str, float]) -> None:
        """
        Set baseline performance metrics.
        
        Args:
            performance_metrics: Dictionary of metric names to values
        """
        self.baseline_performance = performance_metrics.copy()
        self.logger.info(f"Baseline performance set: {performance_metrics}")
    
    def update_performance(
        self,
        performance_metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Update performance metrics and check for drift.
        
        Args:
            performance_metrics: Current performance metrics
            timestamp: Timestamp of the measurement
        
        Returns:
            Dictionary containing drift analysis results
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Add to history
        self.performance_history.append({
            'timestamp': timestamp,
            'metrics': performance_metrics.copy()
        })
        
        # Keep only recent history
        if len(self.performance_history) > self.window_size * 2:
            self.performance_history = self.performance_history[-self.window_size * 2:]
        
        # Calculate drift
        drift_results = self._calculate_performance_drift()
        
        return drift_results
    
    def _calculate_performance_drift(self) -> Dict[str, Any]:
        """Calculate performance drift metrics."""
        if len(self.performance_history) < self.window_size:
            return {'insufficient_data': True}
        
        # Get recent performance
        recent_metrics = self.performance_history[-self.window_size:]
        
        # Calculate rolling averages
        metric_names = recent_metrics[0]['metrics'].keys()
        rolling_averages = {}
        
        for metric in metric_names:
            values = [entry['metrics'][metric] for entry in recent_metrics]
            rolling_averages[metric] = np.mean(values)
        
        # Compare to baseline
        drift_results = {
            'timestamp': datetime.now(),
            'window_size': self.window_size,
            'rolling_averages': rolling_averages,
            'performance_drift': {}
        }
        
        if self.baseline_performance:
            for metric in metric_names:
                if metric in self.baseline_performance:
                    baseline_value = self.baseline_performance[metric]
                    current_value = rolling_averages[metric]
                    
                    # Calculate relative change
                    relative_change = (current_value - baseline_value) / baseline_value
                    
                    # Determine if drift is significant
                    drift_detected = abs(relative_change) > self.alert_threshold
                    
                    drift_results['performance_drift'][metric] = {
                        'baseline_value': baseline_value,
                        'current_value': current_value,
                        'relative_change': relative_change,
                        'drift_detected': drift_detected
                    }
        
        return drift_results


class FeatureMonitor:
    """Main feature monitoring system combining multiple drift detection methods."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature monitor."""
        self.config = config or get_config().features.dict()
        self.logger = logger.bind(component="feature_monitor")
        
        # Initialize detectors
        self.statistical_detector = StatisticalDriftDetector(
            significance_level=self.config.get('drift_significance_level', 0.05)
        )
        
        self.performance_detector = PerformanceDriftDetector(
            window_size=self.config.get('performance_window_size', 100),
            alert_threshold=self.config.get('performance_alert_threshold', 0.05)
        )
        
        # Monitoring state
        self.monitoring_active = False
        self.alert_callbacks = []
        
        # Storage for monitoring results
        self.monitoring_results_path = self.config.get('monitoring_results_path', 'artefacts/monitoring')
        ensure_directory(self.monitoring_results_path)
    
    def start_monitoring(
        self,
        reference_data: pd.DataFrame,
        baseline_performance: Optional[Dict[str, float]] = None,
        feature_types: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Start feature monitoring.
        
        Args:
            reference_data: Reference dataset for drift detection
            baseline_performance: Baseline performance metrics
            feature_types: Feature type mapping
        """
        self.logger.info("Starting feature monitoring")
        
        # Fit reference distributions
        self.statistical_detector.fit_reference(reference_data, feature_types)
        
        # Set baseline performance
        if baseline_performance:
            self.performance_detector.set_baseline(baseline_performance)
        
        self.monitoring_active = True
        self.logger.info("Feature monitoring started successfully")
    
    def monitor_batch(
        self,
        data: pd.DataFrame,
        performance_metrics: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Monitor a batch of data for drift.
        
        Args:
            data: New data batch
            performance_metrics: Performance metrics for this batch
            timestamp: Timestamp of the data
        
        Returns:
            Comprehensive monitoring results
        """
        if not self.monitoring_active:
            raise ValueError("Monitoring not started. Call start_monitoring() first.")
        
        if timestamp is None:
            timestamp = datetime.now()
        
        self.logger.info(f"Monitoring batch", records=len(data), timestamp=timestamp)
        
        # Statistical drift detection
        statistical_results = self.statistical_detector.detect_drift(data, timestamp)
        
        # Performance drift detection
        performance_results = {}
        if performance_metrics:
            performance_results = self.performance_detector.update_performance(
                performance_metrics, timestamp
            )
        
        # Combine results
        monitoring_results = {
            'timestamp': timestamp,
            'batch_size': len(data),
            'statistical_drift': statistical_results,
            'performance_drift': performance_results,
            'overall_status': self._determine_overall_status(statistical_results, performance_results)
        }
        
        # Save results
        self._save_monitoring_results(monitoring_results)
        
        # Check for alerts
        self._check_alerts(monitoring_results)
        
        return monitoring_results
    
    def _determine_overall_status(
        self,
        statistical_results: Dict[str, Any],
        performance_results: Dict[str, Any]
    ) -> str:
        """Determine overall monitoring status."""
        # Check statistical drift
        statistical_drift = statistical_results.get('drift_detected', False)
        
        # Check performance drift
        performance_drift = False
        if 'performance_drift' in performance_results:
            performance_drift = any(
                result.get('drift_detected', False)
                for result in performance_results['performance_drift'].values()
            )
        
        if statistical_drift and performance_drift:
            return 'critical'
        elif statistical_drift or performance_drift:
            return 'warning'
        else:
            return 'healthy'
    
    def _save_monitoring_results(self, results: Dict[str, Any]) -> None:
        """Save monitoring results to file."""
        try:
            timestamp_str = results['timestamp'].strftime('%Y%m%d_%H%M%S')
            filename = f"monitoring_results_{timestamp_str}.json"
            filepath = f"{self.monitoring_results_path}/{filename}"
            
            # Convert datetime objects to strings for JSON serialization
            serializable_results = self._make_json_serializable(results)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save monitoring results: {e}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def _check_alerts(self, results: Dict[str, Any]) -> None:
        """Check if alerts should be triggered."""
        status = results['overall_status']
        
        if status in ['warning', 'critical']:
            alert_data = {
                'timestamp': results['timestamp'],
                'status': status,
                'statistical_drift': results['statistical_drift']['drift_detected'],
                'features_with_drift': results['statistical_drift']['features_with_drift'],
                'drift_ratio': results['statistical_drift']['drift_ratio']
            }
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert_data)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")
            
            self.logger.warning(f"Drift alert triggered", status=status, alert_data=alert_data)
    
    def add_alert_callback(self, callback) -> None:
        """Add callback function for drift alerts."""
        self.alert_callbacks.append(callback)
    
    def get_monitoring_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get monitoring summary for the specified time window."""
        statistical_summary = self.statistical_detector.get_drift_summary(hours)
        
        return {
            'window_hours': hours,
            'statistical_drift_summary': statistical_summary,
            'monitoring_active': self.monitoring_active
        }
    
    def stop_monitoring(self) -> None:
        """Stop feature monitoring."""
        self.monitoring_active = False
        self.logger.info("Feature monitoring stopped")


# Factory function
def create_feature_monitor(config: Optional[Dict[str, Any]] = None) -> FeatureMonitor:
    """Create feature monitor with configuration."""
    return FeatureMonitor(config)


# Example usage
if __name__ == "__main__":
    # Test feature monitoring
    monitor = create_feature_monitor()
    
    # Generate reference data
    np.random.seed(42)
    n_samples = 1000
    
    reference_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_samples),
        'feature_2': np.random.normal(5, 2, n_samples),
        'feature_3': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2])
    })
    
    # Start monitoring
    baseline_performance = {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88}
    monitor.start_monitoring(reference_data, baseline_performance)
    
    # Generate new data with drift
    new_data = pd.DataFrame({
        'feature_1': np.random.normal(0.5, 1.2, 100),  # Mean shift and variance change
        'feature_2': np.random.normal(5, 2, 100),      # No change
        'feature_3': np.random.choice(['A', 'B', 'C'], 100, p=[0.3, 0.3, 0.4])  # Distribution change
    })
    
    # Monitor batch
    new_performance = {'accuracy': 0.80, 'precision': 0.78, 'recall': 0.85}
    results = monitor.monitor_batch(new_data, new_performance)
    
    print(f"Monitoring status: {results['overall_status']}")
    print(f"Statistical drift detected: {results['statistical_drift']['drift_detected']}")
    print(f"Features with drift: {results['statistical_drift']['features_with_drift']}")
    
    # Get summary
    summary = monitor.get_monitoring_summary()
    print(f"Monitoring summary: {summary}")
