"""
Technical Indicator Validation and Mathematical Verification System.
Ensures mathematical correctness and handles edge cases for all technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import warnings
from abc import ABC, abstractmethod
import math
from scipy import stats
from hypothesis import given, strategies as st, settings
import pytest

from .logging_config import StructuredLogger
from .exceptions import ValidationError, IndicatorError
from .indicators import TechnicalIndicators


@dataclass
class ValidationResult:
    """Result of indicator validation."""
    indicator_name: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    performance_metrics: Dict[str, float]
    test_cases_passed: int
    test_cases_total: int


class IndicatorValidator:
    """Validates technical indicators for mathematical correctness."""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
        self.indicators = TechnicalIndicators()
        self.validation_results: Dict[str, ValidationResult] = {}
    
    def validate_all_indicators(self) -> Dict[str, ValidationResult]:
        """Validate all implemented technical indicators."""
        self.logger.info("Starting comprehensive indicator validation")
        
        # Get all indicator methods
        indicator_methods = [
            method for method in dir(self.indicators)
            if not method.startswith('_') and callable(getattr(self.indicators, method))
        ]
        
        for method_name in indicator_methods:
            try:
                result = self.validate_indicator(method_name)
                self.validation_results[method_name] = result
            except Exception as e:
                self.logger.error(f"Failed to validate indicator {method_name}: {str(e)}")
                self.validation_results[method_name] = ValidationResult(
                    indicator_name=method_name,
                    is_valid=False,
                    errors=[f"Validation failed: {str(e)}"],
                    warnings=[],
                    performance_metrics={},
                    test_cases_passed=0,
                    test_cases_total=0
                )
        
        self.logger.info(
            "Indicator validation completed",
            total_indicators=len(self.validation_results),
            valid_indicators=sum(1 for r in self.validation_results.values() if r.is_valid)
        )
        
        return self.validation_results
    
    def validate_indicator(self, indicator_name: str) -> ValidationResult:
        """Validate a specific technical indicator."""
        self.logger.info(f"Validating indicator: {indicator_name}")
        
        errors = []
        warnings = []
        performance_metrics = {}
        test_cases_passed = 0
        test_cases_total = 0
        
        try:
            # Get indicator method
            indicator_method = getattr(self.indicators, indicator_name)
            
            # Test with standard data
            test_cases_total += 1
            if self._test_standard_data(indicator_method, indicator_name):
                test_cases_passed += 1
            else:
                errors.append(f"Failed standard data test")
            
            # Test edge cases
            edge_case_results = self._test_edge_cases(indicator_method, indicator_name)
            test_cases_total += len(edge_case_results)
            test_cases_passed += sum(edge_case_results.values())
            
            for case, passed in edge_case_results.items():
                if not passed:
                    errors.append(f"Failed edge case: {case}")
            
            # Test mathematical properties
            math_test_results = self._test_mathematical_properties(indicator_method, indicator_name)
            test_cases_total += len(math_test_results)
            test_cases_passed += sum(math_test_results.values())
            
            for test, passed in math_test_results.items():
                if not passed:
                    warnings.append(f"Mathematical property test failed: {test}")
            
            # Performance benchmarking
            performance_metrics = self._benchmark_performance(indicator_method, indicator_name)
            
            # Property-based testing
            property_test_results = self._run_property_tests(indicator_method, indicator_name)
            test_cases_total += property_test_results['total']
            test_cases_passed += property_test_results['passed']
            
            if property_test_results['errors']:
                errors.extend(property_test_results['errors'])
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        is_valid = len(errors) == 0
        
        result = ValidationResult(
            indicator_name=indicator_name,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            performance_metrics=performance_metrics,
            test_cases_passed=test_cases_passed,
            test_cases_total=test_cases_total
        )
        
        self.logger.info(
            f"Indicator validation completed: {indicator_name}",
            is_valid=is_valid,
            test_cases_passed=test_cases_passed,
            test_cases_total=test_cases_total
        )
        
        return result
    
    def _test_standard_data(self, indicator_method: Callable, indicator_name: str) -> bool:
        """Test indicator with standard OHLCV data."""
        try:
            # Generate standard test data
            test_data = self._generate_standard_test_data()
            
            # Call indicator method
            if indicator_name in ['sma', 'ema', 'rsi', 'macd']:
                result = indicator_method(test_data, period=14)
            elif indicator_name in ['bollinger_bands', 'keltner_channels']:
                result = indicator_method(test_data, period=20)
            elif indicator_name in ['stochastic', 'williams_r']:
                result = indicator_method(test_data, k_period=14)
            else:
                # Try with default parameters
                result = indicator_method(test_data)
            
            # Basic validation
            if result is None:
                return False
            
            if isinstance(result, pd.Series):
                return not result.empty and not result.isna().all()
            elif isinstance(result, pd.DataFrame):
                return not result.empty and not result.isna().all().all()
            elif isinstance(result, dict):
                return all(v is not None for v in result.values())
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Standard data test failed for {indicator_name}: {str(e)}")
            return False
    
    def _test_edge_cases(self, indicator_method: Callable, indicator_name: str) -> Dict[str, bool]:
        """Test indicator with edge cases."""
        edge_cases = {
            'empty_data': self._test_empty_data,
            'single_value': self._test_single_value,
            'constant_values': self._test_constant_values,
            'nan_values': self._test_nan_values,
            'extreme_values': self._test_extreme_values,
            'insufficient_data': self._test_insufficient_data
        }
        
        results = {}
        for case_name, test_func in edge_cases.items():
            try:
                results[case_name] = test_func(indicator_method, indicator_name)
            except Exception as e:
                self.logger.warning(f"Edge case test {case_name} failed for {indicator_name}: {str(e)}")
                results[case_name] = False
        
        return results
    
    def _test_empty_data(self, indicator_method: Callable, indicator_name: str) -> bool:
        """Test with empty data."""
        empty_data = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
        try:
            result = indicator_method(empty_data)
            # Should handle gracefully, not crash
            return True
        except Exception:
            return False
    
    def _test_single_value(self, indicator_method: Callable, indicator_name: str) -> bool:
        """Test with single data point."""
        single_data = pd.DataFrame({
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [102.0],
            'volume': [1000.0]
        })
        
        try:
            result = indicator_method(single_data)
            return True
        except Exception:
            return False
    
    def _test_constant_values(self, indicator_method: Callable, indicator_name: str) -> bool:
        """Test with constant values."""
        constant_data = pd.DataFrame({
            'open': [100.0] * 50,
            'high': [100.0] * 50,
            'low': [100.0] * 50,
            'close': [100.0] * 50,
            'volume': [1000.0] * 50
        })
        
        try:
            result = indicator_method(constant_data)
            return True
        except Exception:
            return False
    
    def _test_nan_values(self, indicator_method: Callable, indicator_name: str) -> bool:
        """Test with NaN values."""
        test_data = self._generate_standard_test_data()
        # Introduce some NaN values
        test_data.loc[10:15, 'close'] = np.nan
        test_data.loc[25:30, 'volume'] = np.nan
        
        try:
            result = indicator_method(test_data)
            return True
        except Exception:
            return False
    
    def _test_extreme_values(self, indicator_method: Callable, indicator_name: str) -> bool:
        """Test with extreme values."""
        extreme_data = pd.DataFrame({
            'open': [1e-10, 1e10, -1e10, 0],
            'high': [1e-10, 1e10, -1e10, 0],
            'low': [1e-10, 1e10, -1e10, 0],
            'close': [1e-10, 1e10, -1e10, 0],
            'volume': [1e-10, 1e10, 0, 1]
        })
        
        try:
            result = indicator_method(extreme_data)
            return True
        except Exception:
            return False
    
    def _test_insufficient_data(self, indicator_method: Callable, indicator_name: str) -> bool:
        """Test with insufficient data for calculation."""
        insufficient_data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [95.0, 96.0],
            'close': [102.0, 103.0],
            'volume': [1000.0, 1100.0]
        })
        
        try:
            # Try with period larger than data
            if indicator_name in ['sma', 'ema', 'rsi']:
                result = indicator_method(insufficient_data, period=50)
            else:
                result = indicator_method(insufficient_data)
            return True
        except Exception:
            return False
    
    def _test_mathematical_properties(self, indicator_method: Callable, indicator_name: str) -> Dict[str, bool]:
        """Test mathematical properties of indicators."""
        tests = {}
        
        try:
            test_data = self._generate_standard_test_data()
            
            # Test monotonicity where applicable
            if indicator_name in ['sma', 'ema']:
                tests['monotonicity'] = self._test_monotonicity(indicator_method, test_data)
            
            # Test bounds where applicable
            if indicator_name in ['rsi', 'stochastic', 'williams_r']:
                tests['bounds'] = self._test_bounds(indicator_method, test_data, indicator_name)
            
            # Test continuity
            tests['continuity'] = self._test_continuity(indicator_method, test_data)
            
            # Test stability
            tests['stability'] = self._test_stability(indicator_method, test_data)
            
        except Exception as e:
            self.logger.warning(f"Mathematical property tests failed for {indicator_name}: {str(e)}")
        
        return tests
    
    def _test_monotonicity(self, indicator_method: Callable, data: pd.DataFrame) -> bool:
        """Test if indicator preserves monotonic trends."""
        try:
            # Create monotonically increasing data
            mono_data = data.copy()
            mono_data['close'] = np.linspace(100, 200, len(data))
            mono_data['high'] = mono_data['close'] * 1.02
            mono_data['low'] = mono_data['close'] * 0.98
            mono_data['open'] = mono_data['close'].shift(1).fillna(mono_data['close'].iloc[0])
            
            result = indicator_method(mono_data, period=10)
            
            if isinstance(result, pd.Series):
                # Check if trend is generally preserved (allowing for some noise)
                valid_result = result.dropna()
                if len(valid_result) < 2:
                    return True  # Not enough data to test
                
                # Check if at least 70% of consecutive differences are positive
                diffs = valid_result.diff().dropna()
                positive_ratio = (diffs > 0).mean()
                return positive_ratio > 0.7
            
            return True
            
        except Exception:
            return False
    
    def _test_bounds(self, indicator_method: Callable, data: pd.DataFrame, indicator_name: str) -> bool:
        """Test if indicator respects expected bounds."""
        try:
            if indicator_name == 'rsi':
                result = indicator_method(data, period=14)
                if isinstance(result, pd.Series):
                    valid_values = result.dropna()
                    return ((valid_values >= 0) & (valid_values <= 100)).all()
            
            elif indicator_name == 'stochastic':
                result = indicator_method(data, k_period=14)
                if isinstance(result, dict) and 'K' in result:
                    valid_values = result['K'].dropna()
                    return ((valid_values >= 0) & (valid_values <= 100)).all()
            
            elif indicator_name == 'williams_r':
                result = indicator_method(data, period=14)
                if isinstance(result, pd.Series):
                    valid_values = result.dropna()
                    return ((valid_values >= -100) & (valid_values <= 0)).all()
            
            return True
            
        except Exception:
            return False
    
    def _test_continuity(self, indicator_method: Callable, data: pd.DataFrame) -> bool:
        """Test if indicator produces continuous output."""
        try:
            result = indicator_method(data)
            
            if isinstance(result, pd.Series):
                valid_result = result.dropna()
                if len(valid_result) < 2:
                    return True
                
                # Check for unrealistic jumps (more than 10x change)
                ratios = (valid_result / valid_result.shift(1)).dropna()
                extreme_jumps = ((ratios > 10) | (ratios < 0.1)).sum()
                return extreme_jumps < len(ratios) * 0.05  # Less than 5% extreme jumps
            
            return True
            
        except Exception:
            return False
    
    def _test_stability(self, indicator_method: Callable, data: pd.DataFrame) -> bool:
        """Test if indicator is stable with small input changes."""
        try:
            # Original result
            result1 = indicator_method(data)
            
            # Add small noise
            noisy_data = data.copy()
            noise_factor = 0.001  # 0.1% noise
            for col in ['open', 'high', 'low', 'close']:
                noise = np.random.normal(0, noise_factor, len(data))
                noisy_data[col] = data[col] * (1 + noise)
            
            result2 = indicator_method(noisy_data)
            
            if isinstance(result1, pd.Series) and isinstance(result2, pd.Series):
                valid_mask = ~(result1.isna() | result2.isna())
                if valid_mask.sum() < 2:
                    return True
                
                # Check relative difference
                rel_diff = abs((result1[valid_mask] - result2[valid_mask]) / result1[valid_mask])
                return rel_diff.mean() < 0.1  # Less than 10% average difference
            
            return True
            
        except Exception:
            return False
    
    def _benchmark_performance(self, indicator_method: Callable, indicator_name: str) -> Dict[str, float]:
        """Benchmark indicator performance."""
        import time
        
        metrics = {}
        
        try:
            # Test with different data sizes
            data_sizes = [100, 1000, 10000]
            
            for size in data_sizes:
                test_data = self._generate_standard_test_data(size)
                
                # Measure execution time
                start_time = time.perf_counter()
                result = indicator_method(test_data)
                end_time = time.perf_counter()
                
                execution_time = end_time - start_time
                metrics[f'execution_time_{size}'] = execution_time
                metrics[f'throughput_{size}'] = size / execution_time if execution_time > 0 else float('inf')
            
            # Memory usage would require additional profiling tools
            
        except Exception as e:
            self.logger.warning(f"Performance benchmarking failed for {indicator_name}: {str(e)}")
        
        return metrics
    
    def _run_property_tests(self, indicator_method: Callable, indicator_name: str) -> Dict[str, Any]:
        """Run property-based tests using Hypothesis."""
        results = {'total': 0, 'passed': 0, 'errors': []}
        
        try:
            # Define property tests based on indicator type
            if indicator_name in ['sma', 'ema']:
                results.update(self._test_moving_average_properties(indicator_method))
            elif indicator_name == 'rsi':
                results.update(self._test_rsi_properties(indicator_method))
            elif indicator_name == 'macd':
                results.update(self._test_macd_properties(indicator_method))
            
        except Exception as e:
            results['errors'].append(f"Property testing failed: {str(e)}")
        
        return results
    
    def _test_moving_average_properties(self, indicator_method: Callable) -> Dict[str, Any]:
        """Property tests for moving averages."""
        results = {'total': 0, 'passed': 0, 'errors': []}
        
        @given(
            prices=st.lists(
                st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
                min_size=20, max_size=100
            ),
            period=st.integers(min_value=2, max_value=20)
        )
        @settings(max_examples=10, deadline=1000)  # Reduced for performance
        def test_ma_properties(prices, period):
            try:
                data = pd.DataFrame({
                    'open': prices,
                    'high': [p * 1.02 for p in prices],
                    'low': [p * 0.98 for p in prices],
                    'close': prices,
                    'volume': [1000] * len(prices)
                })
                
                result = indicator_method(data, period=period)
                
                if isinstance(result, pd.Series):
                    valid_result = result.dropna()
                    if len(valid_result) > 0:
                        # MA should be between min and max of input
                        assert valid_result.min() >= min(prices) * 0.98
                        assert valid_result.max() <= max(prices) * 1.02
                
                return True
                
            except Exception as e:
                raise AssertionError(f"MA property test failed: {str(e)}")
        
        try:
            test_ma_properties()
            results['total'] += 1
            results['passed'] += 1
        except Exception as e:
            results['total'] += 1
            results['errors'].append(str(e))
        
        return results
    
    def _test_rsi_properties(self, indicator_method: Callable) -> Dict[str, Any]:
        """Property tests for RSI."""
        results = {'total': 0, 'passed': 0, 'errors': []}
        
        @given(
            prices=st.lists(
                st.floats(min_value=10.0, max_value=1000.0, allow_nan=False),
                min_size=30, max_size=100
            ),
            period=st.integers(min_value=5, max_value=20)
        )
        @settings(max_examples=5, deadline=1000)
        def test_rsi_bounds(prices, period):
            try:
                data = pd.DataFrame({
                    'open': prices,
                    'high': [p * 1.02 for p in prices],
                    'low': [p * 0.98 for p in prices],
                    'close': prices,
                    'volume': [1000] * len(prices)
                })
                
                result = indicator_method(data, period=period)
                
                if isinstance(result, pd.Series):
                    valid_result = result.dropna()
                    if len(valid_result) > 0:
                        # RSI should be between 0 and 100
                        assert (valid_result >= 0).all()
                        assert (valid_result <= 100).all()
                
                return True
                
            except Exception as e:
                raise AssertionError(f"RSI property test failed: {str(e)}")
        
        try:
            test_rsi_bounds()
            results['total'] += 1
            results['passed'] += 1
        except Exception as e:
            results['total'] += 1
            results['errors'].append(str(e))
        
        return results
    
    def _test_macd_properties(self, indicator_method: Callable) -> Dict[str, Any]:
        """Property tests for MACD."""
        results = {'total': 0, 'passed': 0, 'errors': []}
        
        # MACD property tests would be implemented here
        # For now, just return empty results
        return results
    
    def _generate_standard_test_data(self, size: int = 100) -> pd.DataFrame:
        """Generate standard test data for validation."""
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic price series
        returns = np.random.normal(0.001, 0.02, size)
        prices = [100.0]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'open': [prices[i-1] if i > 0 else prices[0] for i in range(size)],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': [1000 + np.random.exponential(500) for _ in range(size)]
        })
        
        # Ensure OHLC consistency
        for i in range(size):
            data.loc[i, 'high'] = max(data.loc[i, ['open', 'high', 'low', 'close']])
            data.loc[i, 'low'] = min(data.loc[i, ['open', 'high', 'low', 'close']])
        
        return data
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report."""
        if not self.validation_results:
            return "No validation results available. Run validate_all_indicators() first."
        
        report_lines = [
            "=" * 80,
            "TECHNICAL INDICATOR VALIDATION REPORT",
            "=" * 80,
            f"Total Indicators Tested: {len(self.validation_results)}",
            f"Valid Indicators: {sum(1 for r in self.validation_results.values() if r.is_valid)}",
            f"Invalid Indicators: {sum(1 for r in self.validation_results.values() if not r.is_valid)}",
            "",
            "DETAILED RESULTS:",
            "-" * 40
        ]
        
        for indicator_name, result in self.validation_results.items():
            status = "PASS" if result.is_valid else "FAIL"
            report_lines.extend([
                f"Indicator: {indicator_name} [{status}]",
                f"  Test Cases: {result.test_cases_passed}/{result.test_cases_total}",
                f"  Errors: {len(result.errors)}",
                f"  Warnings: {len(result.warnings)}"
            ])
            
            if result.errors:
                report_lines.extend([f"    ERROR: {error}" for error in result.errors])
            
            if result.warnings:
                report_lines.extend([f"    WARNING: {warning}" for warning in result.warnings])
            
            if result.performance_metrics:
                report_lines.append("  Performance Metrics:")
                for metric, value in result.performance_metrics.items():
                    report_lines.append(f"    {metric}: {value:.6f}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


class PerformanceBenchmarks:
    """Performance benchmarking for technical indicators."""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
        self.indicators = TechnicalIndicators()
    
    def benchmark_all_indicators(self, data_sizes: List[int] = None) -> Dict[str, Dict[str, float]]:
        """Benchmark performance of all indicators."""
        if data_sizes is None:
            data_sizes = [100, 1000, 10000, 50000]
        
        results = {}
        
        indicator_methods = [
            method for method in dir(self.indicators)
            if not method.startswith('_') and callable(getattr(self.indicators, method))
        ]
        
        for method_name in indicator_methods:
            try:
                results[method_name] = self.benchmark_indicator(method_name, data_sizes)
            except Exception as e:
                self.logger.error(f"Benchmarking failed for {method_name}: {str(e)}")
                results[method_name] = {}
        
        return results
    
    def benchmark_indicator(self, indicator_name: str, data_sizes: List[int]) -> Dict[str, float]:
        """Benchmark a specific indicator."""
        import time
        import psutil
        import os
        
        results = {}
        indicator_method = getattr(self.indicators, indicator_name)
        
        for size in data_sizes:
            # Generate test data
            test_data = self._generate_benchmark_data(size)
            
            # Measure memory before
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Measure execution time
            start_time = time.perf_counter()
            
            try:
                result = indicator_method(test_data)
                success = True
            except Exception as e:
                self.logger.warning(f"Benchmark failed for {indicator_name} with size {size}: {str(e)}")
                success = False
            
            end_time = time.perf_counter()
            
            # Measure memory after
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            if success:
                execution_time = end_time - start_time
                memory_used = memory_after - memory_before
                
                results[f'time_{size}'] = execution_time
                results[f'memory_{size}'] = memory_used
                results[f'throughput_{size}'] = size / execution_time if execution_time > 0 else float('inf')
        
        return results
    
    def _generate_benchmark_data(self, size: int) -> pd.DataFrame:
        """Generate benchmark data of specified size."""
        np.random.seed(42)
        
        returns = np.random.normal(0.001, 0.02, size)
        prices = [100.0]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'open': [prices[i-1] if i > 0 else prices[0] for i in range(size)],
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000.0] * size
        })


# Test runner for continuous validation
class ContinuousValidator:
    """Continuous validation system for indicators."""
    
    def __init__(self, validation_interval: int = 3600):  # 1 hour default
        self.logger = StructuredLogger(__name__)
        self.validation_interval = validation_interval
        self.validator = IndicatorValidator()
        self.last_validation = None
    
    def should_validate(self) -> bool:
        """Check if validation should be run."""
        if self.last_validation is None:
            return True
        
        time_since_last = (datetime.now() - self.last_validation).total_seconds()
        return time_since_last >= self.validation_interval
    
    def run_validation(self) -> bool:
        """Run validation if needed."""
        if not self.should_validate():
            return True
        
        try:
            results = self.validator.validate_all_indicators()
            self.last_validation = datetime.now()
            
            # Check if all indicators are valid
            all_valid = all(result.is_valid for result in results.values())
            
            if not all_valid:
                self.logger.error("Indicator validation failed", results=results)
            
            return all_valid
            
        except Exception as e:
            self.logger.error(f"Validation run failed: {str(e)}")
            return False
