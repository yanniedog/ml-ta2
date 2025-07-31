"""
Script to fix syntax errors in ab_testing.py by completely recreating the file
"""

def fix_ab_testing_file():
    # Create a new file with correct syntax
    content = '''"""
A/B testing framework for ML-TA system.

This module implements:
- Comprehensive A/B testing for model comparison
- Statistical significance testing
- Traffic splitting and user cohort assignment
- Result analysis and reporting
- Test lifecycle management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import threading
import uuid
import hashlib
import random
import json
import warnings
import scipy.stats
warnings.filterwarnings('ignore')

from src.config import get_config
from src.logging_config import get_logger
from src.prediction_engine import PredictionResponse
from src.exceptions import TestingError

logger = get_logger(__name__)


@dataclass
class TestVariant:
    """Configuration for a test variant."""
    name: str
    model_name: str
    weight: float = 0.5  # Percentage of traffic
    description: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    

@dataclass
class TestConfig:
    """Configuration for an A/B test."""
    name: str
    variants: List[TestVariant]
    min_sample_size: int = 1000
    max_duration_days: int = 14
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "latency_ms", "business_impact"])
    significance_level: float = 0.05
    description: str = ""
    is_active: bool = True


@dataclass
class TestResult:
    """Results of an A/B test."""
    test_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    sample_counts: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    significance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    winner: Optional[str] = None
    confidence: float = 0.0
    status: str = "running"  # running, completed, stopped


class ABTest:
    """
    A/B test implementation.
    
    Handles:
    - Test configuration
    - Traffic assignment
    - Result collection and analysis
    - Statistical significance calculation
    """
    
    def __init__(self, config: TestConfig):
        """
        Initialize A/B test.
        
        Args:
            config: Test configuration
        """
        self.config = config
        self.start_time = datetime.now()
        self.end_time = None
        
        # Initialize tracking data structures
        self.predictions = {variant.name: [] for variant in config.variants}
        self.actuals = {variant.name: [] for variant in config.variants}
        self.timestamps = {variant.name: [] for variant in config.variants}
        self.latencies = {variant.name: [] for variant in config.variants}
        self.metadata = {variant.name: [] for variant in config.variants}
        
        self.result = TestResult(
            test_name=config.name,
            start_time=self.start_time,
            sample_counts={variant.name: 0 for variant in config.variants}
        )
        
        self.lock = threading.RLock()
        self._normalize_weights()
        
        logger.info("A/B test created",
                  test_name=config.name,
                  variants=[v.name for v in config.variants],
                  metrics=config.metrics)

    def _normalize_weights(self):
        """Normalize variant weights to sum to 1.0."""
        total_weight = sum(v.weight for v in self.config.variants)
        for variant in self.config.variants:
            variant.weight /= total_weight
            
    def assign_variant(self, user_id: str) -> TestVariant:
        """
        Assign a user to a test variant.
        
        Args:
            user_id: User identifier
            
        Returns:
            Assigned test variant
        """
        with self.lock:
            # Use consistent hashing for stable assignment
            digest = hashlib.md5(f"{self.config.name}:{user_id}".encode()).hexdigest()
            value = int(digest[:8], 16) / 0xFFFFFFFF  # 0 to 1
            
            # Assign based on weights
            cumulative_weight = 0
            for variant in self.config.variants:
                cumulative_weight += variant.weight
                if value <= cumulative_weight:
                    return variant
                    
            # Fallback to last variant
            return self.config.variants[-1]
            
    def record_prediction(self, 
                         variant_name: str, 
                         response: PredictionResponse,
                         actual: Optional[Union[float, int, List]] = None):
        """
        Record a prediction for analysis.
        
        Args:
            variant_name: Test variant name
            response: Prediction response
            actual: Optional actual value/label for the prediction
        """
        with self.lock:
            if variant_name not in self.predictions:
                logger.warning(f"Unknown variant: {variant_name}")
                return
            
            self.predictions[variant_name].append(response.predictions)
            if actual is not None:
                self.actuals[variant_name].append(actual)
            self.timestamps[variant_name].append(response.timestamp)
            self.latencies[variant_name].append(response.latency_ms)
            self.metadata[variant_name].append(response.metadata)
            
            self.result.sample_counts[variant_name] += 1
            
    def is_ready_for_analysis(self) -> bool:
        """
        Check if test is ready for analysis.
        
        Returns:
            True if all variants have sufficient samples
        """
        for variant in self.config.variants:
            if self.result.sample_counts.get(variant.name, 0) < self.config.min_sample_size:
                return False
        return True
            
    def analyze_results(self):
        """Analyze test results and compute metrics."""
        with self.lock:
            # Skip if not enough data
            if not self.is_ready_for_analysis():
                logger.info("Not enough data for analysis",
                          test_name=self.config.name,
                          sample_counts=self.result.sample_counts)
                return
            
            # Initialize metrics and significance
            metrics = {metric_name: {} for metric_name in self.config.metrics}
            significance = {metric_name: {} for metric_name in self.config.metrics}
            
            # Calculate metrics
            for metric_name in metrics:
                for variant_name in self.predictions:
                    if metric_name == "latency_ms":
                        metrics[metric_name][variant_name] = np.mean(self.latencies[variant_name])
                    
                    elif metric_name == "accuracy" and len(self.actuals[variant_name]) > 0:
                        try:
                            # First attempt to flatten all predictions into a single array
                            predictions = np.concatenate(self.predictions[variant_name])
                            
                            # If actuals are lists/arrays, concatenate them too
                            if isinstance(self.actuals[variant_name][0], (list, np.ndarray)):
                                actuals = np.concatenate(self.actuals[variant_name])
                            else:
                                # Otherwise, convert to array directly
                                actuals = np.array(self.actuals[variant_name])
                                
                            # Ensure shapes match
                            if len(predictions) != len(actuals):
                                logger.warning(f"Shape mismatch in {variant_name}: predictions {predictions.shape}, actuals {actuals.shape}")
                                # Take the minimum length to avoid shape errors
                                min_len = min(len(predictions), len(actuals))
                                predictions = predictions[:min_len]
                                actuals = actuals[:min_len]
                                
                            metrics[metric_name][variant_name] = np.mean(predictions == actuals)
                        except Exception as e:
                            logger.error(f"Error computing accuracy for {variant_name}: {e}")
                            metrics[metric_name][variant_name] = 0.0
                            
                    elif metric_name == "business_impact" and len(self.metadata[variant_name]) > 0 and "impact_score" in self.metadata[variant_name][0]:
                        impact_scores = [m.get("impact_score", 0) for m in self.metadata[variant_name]]
                        metrics[metric_name][variant_name] = np.mean(impact_scores)
            
            # Calculate statistical significance
            for metric_name in metrics:
                # Use first variant as baseline
                baseline_variant = self.config.variants[0].name
                baseline_values = self._get_metric_values(metric_name, baseline_variant)
                
                for variant_name in metrics[metric_name]:
                    if variant_name == baseline_variant:
                        significance[metric_name][variant_name] = 1.0
                        continue
                        
                    variant_values = self._get_metric_values(metric_name, variant_name)
                    
                    # Skip if insufficient data
                    if len(baseline_values) < 30 or len(variant_values) < 30:
                        significance[metric_name][variant_name] = None
                        continue
                    
                    # Calculate p-value
                    _, p_value = scipy.stats.ttest_ind(baseline_values, variant_values)
                    significance[metric_name][variant_name] = p_value
                    
            # Update result
            self.result.metrics = metrics
            self.result.significance = significance
            
            # Determine winner based on primary metric (first in the list)
            primary_metric = self.config.metrics[0]
            if primary_metric in metrics:
                metric_values = metrics[primary_metric]
                winner = max(metric_values, key=metric_values.get)
                
                # Only declare winner if statistically significant
                if (primary_metric in significance and 
                    significance[primary_metric].get(winner) and 
                    significance[primary_metric][winner] < self.config.significance_level):
                    self.result.winner = winner
                    self.result.confidence = 1.0 - significance[primary_metric][winner]
            
            logger.info("A/B test analyzed",
                      test_name=self.config.name,
                      metrics=metrics,
                      winner=self.result.winner,
                      confidence=self.result.confidence)
            
    def _get_metric_values(self, metric_name: str, variant_name: str) -> List[float]:
        """
        Get values for a specific metric and variant.
        
        Args:
            metric_name: Metric name
            variant_name: Variant name
            
        Returns:
            List of values for the metric
        """
        if metric_name == "latency_ms":
            return self.latencies[variant_name]
            
        elif metric_name == "accuracy" and len(self.actuals[variant_name]) > 0:
            try:
                # First attempt to flatten all predictions into a single array
                predictions = np.concatenate(self.predictions[variant_name])
                
                # If actuals are lists/arrays, concatenate them too
                if isinstance(self.actuals[variant_name][0], (list, np.ndarray)):
                    actuals = np.concatenate(self.actuals[variant_name])
                else:
                    # Otherwise, convert to array directly
                    actuals = np.array(self.actuals[variant_name])
                    
                # Ensure shapes match
                if len(predictions) != len(actuals):
                    logger.warning(f"Shape mismatch in {variant_name}: predictions {predictions.shape}, actuals {actuals.shape}")
                    # Take the minimum length to avoid shape errors
                    min_len = min(len(predictions), len(actuals))
                    predictions = predictions[:min_len]
                    actuals = actuals[:min_len]
                    
                # Convert to binary correct/incorrect
                return (predictions == actuals).astype(float).tolist()
            except Exception as e:
                logger.error(f"Error computing accuracy values for {variant_name}: {e}")
                # Return empty list on error
                return []
                
        elif metric_name == "business_impact" and self.metadata[variant_name]:
            return [m.get("impact_score", 0) for m in self.metadata[variant_name]]
            
        return []
            
    def stop(self):
        """Stop the test and finalize results."""
        with self.lock:
            if not self.end_time:
                self.end_time = datetime.now()
                self.result.end_time = self.end_time
                self.result.status = "completed"
                
                # Final analysis
                self.analyze_results()
                
                logger.info("A/B test stopped",
                          test_name=self.config.name,
                          duration_days=(self.end_time - self.start_time).days,
                          winner=self.result.winner)
                
    def get_result(self) -> TestResult:
        """
        Get the current test result.
        
        Returns:
            Current test result
        """
        with self.lock:
            # Do an analysis if we haven't yet
            if not self.result.metrics and self.is_ready_for_analysis():
                self.analyze_results()
                
            return self.result


class ABTestingManager:
    """
    Manager for A/B tests.
    
    Handles:
    - Test creation and lifecycle management
    - User assignment to tests
    - Result collection and reporting
    """
    
    def __init__(self):
        """Initialize A/B testing manager."""
        self.active_tests = {}  # name -> test
        self.completed_tests = {}  # name -> test
        self.lock = threading.RLock()
        
    def create_test(self, config: TestConfig) -> ABTest:
        """
        Create a new A/B test.
        
        Args:
            config: Test configuration
            
        Returns:
            Created A/B test
        """
        with self.lock:
            test_name = config.name
            
            # Check if test already exists
            if test_name in self.active_tests or test_name in self.completed_tests:
                logger.warning("Test already exists",
                             test_name=test_name)
                return self.active_tests.get(test_name) or self.completed_tests.get(test_name)
                
            # Create and register test
            test = ABTest(config)
            self.active_tests[test_name] = test
            
            logger.info("A/B test created",
                      test_name=test_name,
                      variants=[v.name for v in config.variants])
                      
            return test
            
    def get_test(self, test_name: str) -> Optional[ABTest]:
        """
        Get an A/B test by name.
        
        Args:
            test_name: Test name
            
        Returns:
            A/B test if found, None otherwise
        """
        with self.lock:
            return self.active_tests.get(test_name) or self.completed_tests.get(test_name)
            
    def stop_test(self, test_name: str) -> Optional[TestResult]:
        """
        Stop an A/B test.
        
        Args:
            test_name: Test name
            
        Returns:
            Test result if test was found, None otherwise
        """
        with self.lock:
            test = self.active_tests.get(test_name)
            
            if not test:
                logger.warning("Test not found",
                             test_name=test_name)
                return None
                
            # Stop test
            test.stop()
            
            # Move to completed tests
            self.completed_tests[test_name] = test
            del self.active_tests[test_name]
            
            logger.info("A/B test stopped",
                      test_name=test_name,
                      winner=test.result.winner)
                      
            return test.result
            
    def get_assignment(self, test_name: str, user_id: str) -> Optional[TestVariant]:
        """
        Get variant assignment for a user.
        
        Args:
            test_name: Test name
            user_id: User identifier
            
        Returns:
            Assigned variant if test found, None otherwise
        """
        with self.lock:
            test = self.active_tests.get(test_name)
            
            if not test:
                logger.warning("Test not found or not active",
                             test_name=test_name)
                return None
                
            return test.assign_variant(user_id)
            
    def record_prediction(self,
                         test_name: str,
                         variant_name: str,
                         response: PredictionResponse,
                         actual: Optional[Union[float, int, List]] = None):
        """
        Record a prediction for a test.
        
        Args:
            test_name: Test name
            variant_name: Variant name
            response: Prediction response
            actual: Optional actual value
        """
        with self.lock:
            test = self.active_tests.get(test_name)
            
            if not test:
                logger.warning("Test not found or not active",
                             test_name=test_name)
                return
                
            test.record_prediction(variant_name, response, actual)
            
    def list_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        List all tests and their status.
        
        Returns:
            Dictionary mapping test names to their summary
        """
        with self.lock:
            result = {}
            
            # Active tests
            for name, test in self.active_tests.items():
                test_result = test.get_result()
                result[name] = {
                    'status': test_result.status,
                    'start_time': test_result.start_time.isoformat(),
                    'sample_counts': test_result.sample_counts,
                    'variants': [v.name for v in test.config.variants],
                    'metrics': test.config.metrics,
                    'duration_days': (datetime.now() - test_result.start_time).days
                }
                
            # Completed tests
            for name, test in self.completed_tests.items():
                test_result = test.get_result()
                result[name] = {
                    'status': test_result.status,
                    'start_time': test_result.start_time.isoformat(),
                    'end_time': test_result.end_time.isoformat() if test_result.end_time else None,
                    'sample_counts': test_result.sample_counts,
                    'variants': [v.name for v in test.config.variants],
                    'metrics': test.config.metrics,
                    'winner': test_result.winner,
                    'confidence': test_result.confidence,
                    'duration_days': (test_result.end_time - test_result.start_time).days if test_result.end_time else None
                }
                
            return result
                
    def get_test_result(self, test_name: str) -> Optional[TestResult]:
        """
        Get result of a test.
        
        Args:
            test_name: Test name
            
        Returns:
            Test result if found, None otherwise
        """
        with self.lock:
            test = self.active_tests.get(test_name) or self.completed_tests.get(test_name)
            
            if not test:
                logger.warning("Test not found",
                             test_name=test_name)
                return None
                
            return test.get_result()


def create_ab_testing_manager() -> ABTestingManager:
    """
    Factory function to create an A/B testing manager.
    
    Returns:
        A/B testing manager instance
    """
    return ABTestingManager()


def create_ab_test(name: str,
                  models: List[str],
                  weights: Optional[List[float]] = None,
                  min_sample_size: int = 1000,
                  max_duration_days: int = 14,
                  metrics: Optional[List[str]] = None) -> TestConfig:
    """
    Create an A/B test configuration.
    
    Args:
        name: Test name
        models: List of model names to compare
        weights: Optional traffic weights for models
        min_sample_size: Minimum sample size per variant
        max_duration_days: Maximum test duration in days
        metrics: Metrics to track
        
    Returns:
        Test configuration
    """
    if not weights:
        # Equal weights
        weights = [1.0 / len(models)] * len(models)
    elif len(weights) != len(models):
        raise ValueError("Number of weights must match number of models")
    
    metrics = metrics or ["accuracy", "latency_ms"]
    
    variants = [
        TestVariant(name=f"variant_{i}", model_name=model, weight=weight)
        for i, (model, weight) in enumerate(zip(models, weights))
    ]
    
    return TestConfig(
        name=name,
        variants=variants,
        min_sample_size=min_sample_size,
        max_duration_days=max_duration_days,
        metrics=metrics,
        significance_level=0.05
    )'''
    
    # Write the fixed content to the file
    with open('src/ab_testing.py', 'w') as f:
        f.write(content)
    
    print("File has been completely recreated with correct syntax.")

if __name__ == "__main__":
    fix_ab_testing_file()
