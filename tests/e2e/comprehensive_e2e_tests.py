#!/usr/bin/env python3
"""
Comprehensive End-to-End Testing Framework for ML-TA System

This module implements Phase 10.1 requirements for comprehensive E2E testing,
including realistic data volumes, complete user workflows, and stress testing.
"""

import os
import sys
import time
import json
import asyncio
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import ML-TA components
from config import get_config
from data_fetcher import BinanceDataFetcher
from features import FeaturePipeline
from model_trainer import ModelTrainer
from prediction_engine import PredictionEngine
from monitoring import MonitoringSystem
from web_frontend import create_app


@dataclass
class E2ETestResult:
    """Test result data structure."""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    success: bool
    details: Dict[str, Any]
    metrics: Dict[str, float]
    error_message: Optional[str] = None


class ComprehensiveE2ETestSuite:
    """Comprehensive end-to-end test suite for ML-TA system."""
    
    def __init__(self):
        self.config = get_config()
        self.results: List[E2ETestResult] = []
        self.logger = self._setup_logging()
        self.data_fetcher = None
        self.feature_pipeline = None
        self.model_trainer = None
        self.prediction_engine = None
        self.monitoring_system = None
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for E2E tests."""
        logger = logging.getLogger("e2e_tests")
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        log_file = log_dir / f"e2e_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> E2ETestResult:
        """Run a single test with timing and error handling."""
        start_time = datetime.now()
        self.logger.info(f"Starting test: {test_name}")
        
        try:
            result = test_func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            test_result = E2ETestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                success=True,
                details=result.get('details', {}),
                metrics=result.get('metrics', {})
            )
            
            self.logger.info(f"Test passed: {test_name} ({duration:.2f}s)")
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            test_result = E2ETestResult(
                test_name=test_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                success=False,
                details={},
                metrics={},
                error_message=str(e)
            )
            
            self.logger.error(f"Test failed: {test_name} - {str(e)}")
        
        self.results.append(test_result)
        return test_result
    
    def test_data_pipeline_realistic_volume(self) -> Dict[str, Any]:
        """Test data pipeline with realistic data volumes (10,000+ records)."""
        # Initialize data fetcher
        self.data_fetcher = BinanceDataFetcher()
        
        # Fetch realistic volume of data
        symbol = "BTCUSDT"
        interval = "1h"
        limit = 10000  # Realistic volume
        
        start_time = time.time()
        data = self.data_fetcher.fetch_historical_data(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        fetch_time = time.time() - start_time
        
        # Validate data quality
        assert data is not None, "Failed to fetch data"
        assert len(data) >= 9000, f"Insufficient data: {len(data)} records"
        assert not data.empty, "Data is empty"
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in data.columns, f"Missing column: {col}"
        
        # Check data integrity
        assert data['high'].min() >= data['low'].max() * 0.8, "Price integrity check failed"
        assert data['volume'].min() >= 0, "Volume integrity check failed"
        
        return {
            'details': {
                'records_fetched': len(data),
                'time_range': f"{data.index[0]} to {data.index[-1]}",
                'columns': list(data.columns),
                'data_integrity_passed': True
            },
            'metrics': {
                'fetch_time_seconds': fetch_time,
                'records_per_second': len(data) / fetch_time,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
            }
        }
    
    def test_feature_engineering_workflow(self) -> Dict[str, Any]:
        """Test complete feature engineering workflow with 200+ features."""
        # Use data from previous test or fetch new data
        if self.data_fetcher is None:
            self.data_fetcher = BinanceDataFetcher()
        
        # Fetch data for feature engineering
        data = self.data_fetcher.fetch_historical_data("BTCUSDT", "1h", 5000)
        
        # Initialize feature pipeline
        self.feature_pipeline = FeaturePipeline()
        
        start_time = time.time()
        features = self.feature_pipeline.engineer_features(
            data, 
            fit_scalers=True, 
            validate_temporal=True
        )
        feature_time = time.time() - start_time
        
        # Validate feature generation
        assert features is not None, "Feature engineering failed"
        assert len(features.columns) >= 200, f"Insufficient features: {len(features.columns)}"
        assert not features.isnull().all().any(), "Features contain all-null columns"
        
        # Check for data leakage
        temporal_valid = self.feature_pipeline.validate_temporal_consistency(features)
        assert temporal_valid, "Temporal validation failed - potential data leakage"
        
        return {
            'details': {
                'input_records': len(data),
                'output_records': len(features),
                'feature_count': len(features.columns),
                'temporal_validation_passed': temporal_valid,
                'feature_types': self._analyze_feature_types(features)
            },
            'metrics': {
                'feature_engineering_time': feature_time,
                'features_per_second': len(features.columns) / feature_time,
                'memory_usage_mb': features.memory_usage(deep=True).sum() / 1024 / 1024
            }
        }
    
    def test_model_training_workflow(self) -> Dict[str, Any]:
        """Test complete model training workflow with multiple algorithms."""
        # Ensure feature pipeline is available
        if self.feature_pipeline is None:
            self.test_feature_engineering_workflow()
        
        # Initialize model trainer
        self.model_trainer = ModelTrainer()
        
        # Prepare training data
        data = self.data_fetcher.fetch_historical_data("BTCUSDT", "1h", 3000)
        features = self.feature_pipeline.engineer_features(data, fit_scalers=True)
        
        start_time = time.time()
        
        # Train models
        training_results = self.model_trainer.train_ensemble(
            features=features,
            target_col='target_1h',
            test_size=0.2,
            optimize_hyperparameters=True
        )
        
        training_time = time.time() - start_time
        
        # Validate training results
        assert training_results is not None, "Model training failed"
        assert 'ensemble_score' in training_results, "Missing ensemble score"
        assert training_results['ensemble_score'] > 0.5, "Poor model performance"
        
        return {
            'details': {
                'training_samples': len(features),
                'feature_count': len(features.columns),
                'models_trained': len(training_results.get('model_scores', {})),
                'ensemble_score': training_results.get('ensemble_score', 0),
                'best_model': training_results.get('best_model', 'unknown')
            },
            'metrics': {
                'training_time_seconds': training_time,
                'accuracy_score': training_results.get('ensemble_score', 0),
                'cv_scores': training_results.get('cv_scores', [])
            }
        }
    
    def test_prediction_engine_workflow(self) -> Dict[str, Any]:
        """Test real-time prediction engine with latency requirements."""
        # Initialize prediction engine
        self.prediction_engine = PredictionEngine()
        
        # Prepare test data
        test_data = self.data_fetcher.fetch_historical_data("BTCUSDT", "1h", 100)
        
        # Test prediction latency
        latencies = []
        predictions = []
        
        for i in range(50):  # Test 50 predictions
            start_time = time.time()
            
            prediction = self.prediction_engine.predict(
                symbol="BTCUSDT",
                timeframe="1h",
                data=test_data.iloc[i:i+10]
            )
            
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            predictions.append(prediction)
        
        # Validate latency requirements
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
        
        assert avg_latency < 100, f"Average latency too high: {avg_latency:.2f}ms"
        assert p95_latency < 200, f"P95 latency too high: {p95_latency:.2f}ms"
        
        # Validate predictions
        valid_predictions = [p for p in predictions if p is not None]
        assert len(valid_predictions) >= 45, "Too many failed predictions"
        
        return {
            'details': {
                'total_predictions': len(predictions),
                'successful_predictions': len(valid_predictions),
                'success_rate': len(valid_predictions) / len(predictions),
                'latency_stats': {
                    'average_ms': avg_latency,
                    'p95_ms': p95_latency,
                    'p99_ms': p99_latency,
                    'min_ms': min(latencies),
                    'max_ms': max(latencies)
                }
            },
            'metrics': {
                'average_latency_ms': avg_latency,
                'p95_latency_ms': p95_latency,
                'success_rate': len(valid_predictions) / len(predictions)
            }
        }
    
    def test_monitoring_system_workflow(self) -> Dict[str, Any]:
        """Test monitoring system integration and alerting."""
        # Initialize monitoring system
        self.monitoring_system = MonitoringSystem()
        
        start_time = time.time()
        
        # Start monitoring
        self.monitoring_system.start()
        
        # Simulate system activity
        time.sleep(5)  # Let monitoring collect metrics
        
        # Check system status
        status = self.monitoring_system.get_system_status()
        
        # Test alert system
        test_alert = self.monitoring_system.create_alert(
            alert_type="test",
            message="E2E test alert",
            severity="info"
        )
        
        # Stop monitoring
        self.monitoring_system.stop()
        
        monitoring_time = time.time() - start_time
        
        # Validate monitoring functionality
        assert status is not None, "Failed to get system status"
        assert 'cpu_usage' in status, "Missing CPU usage metric"
        assert 'memory_usage' in status, "Missing memory usage metric"
        assert test_alert is not None, "Failed to create test alert"
        
        return {
            'details': {
                'monitoring_duration': monitoring_time,
                'system_status': status,
                'alert_created': test_alert is not None,
                'metrics_collected': len(status)
            },
            'metrics': {
                'cpu_usage': status.get('cpu_usage', 0),
                'memory_usage': status.get('memory_usage', 0),
                'monitoring_overhead': monitoring_time
            }
        }
    
    def test_stress_testing(self) -> Dict[str, Any]:
        """Perform stress testing with concurrent requests."""
        if self.prediction_engine is None:
            self.prediction_engine = PredictionEngine()
        
        # Prepare test data
        test_data = self.data_fetcher.fetch_historical_data("BTCUSDT", "1h", 100)
        
        # Stress test parameters
        concurrent_users = 100
        requests_per_user = 10
        total_requests = concurrent_users * requests_per_user
        
        start_time = time.time()
        successful_requests = 0
        failed_requests = 0
        response_times = []
        
        def make_prediction_request():
            """Make a single prediction request."""
            nonlocal successful_requests, failed_requests
            try:
                request_start = time.time()
                
                prediction = self.prediction_engine.predict(
                    symbol="BTCUSDT",
                    timeframe="1h",
                    data=test_data.tail(10)
                )
                
                request_time = (time.time() - request_start) * 1000
                response_times.append(request_time)
                
                if prediction is not None:
                    successful_requests += 1
                else:
                    failed_requests += 1
                    
            except Exception:
                failed_requests += 1
        
        # Execute stress test with thread pool
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            
            for user in range(concurrent_users):
                for request in range(requests_per_user):
                    future = executor.submit(make_prediction_request)
                    futures.append(future)
            
            # Wait for all requests to complete
            for future in as_completed(futures):
                future.result()
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        success_rate = successful_requests / total_requests
        throughput = total_requests / total_time
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Validate stress test requirements
        assert success_rate >= 0.95, f"Success rate too low: {success_rate:.2%}"
        assert throughput >= 50, f"Throughput too low: {throughput:.2f} req/s"
        
        return {
            'details': {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'concurrent_users': concurrent_users,
                'test_duration': total_time
            },
            'metrics': {
                'success_rate': success_rate,
                'throughput_req_per_sec': throughput,
                'avg_response_time_ms': avg_response_time,
                'total_time_seconds': total_time
            }
        }
    
    def _analyze_feature_types(self, features) -> Dict[str, int]:
        """Analyze feature types in the dataset."""
        numeric_cols = features.select_dtypes(include=['number']).columns
        categorical_cols = features.select_dtypes(include=['object', 'category']).columns
        datetime_cols = features.select_dtypes(include=['datetime']).columns
        
        return {
            'numeric': len(numeric_cols),
            'categorical': len(categorical_cols),
            'datetime': len(datetime_cols),
            'total': len(features.columns)
        }
    
    def run_comprehensive_suite(self) -> Dict[str, Any]:
        """Run the complete comprehensive E2E test suite."""
        self.logger.info("Starting Comprehensive E2E Test Suite")
        self.logger.info("=" * 80)
        
        # Define test cases
        test_cases = [
            ("Data Pipeline - Realistic Volume", self.test_data_pipeline_realistic_volume),
            ("Feature Engineering Workflow", self.test_feature_engineering_workflow),
            ("Model Training Workflow", self.test_model_training_workflow),
            ("Prediction Engine Workflow", self.test_prediction_engine_workflow),
            ("Monitoring System Workflow", self.test_monitoring_system_workflow),
            ("Stress Testing", self.test_stress_testing)
        ]
        
        # Execute all tests
        suite_start_time = datetime.now()
        
        for test_name, test_func in test_cases:
            self.run_test(test_name, test_func)
        
        suite_end_time = datetime.now()
        suite_duration = (suite_end_time - suite_start_time).total_seconds()
        
        # Calculate suite summary
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results if result.success)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests) * 100
        
        # Generate comprehensive report
        report = {
            'suite_summary': {
                'start_time': suite_start_time.isoformat(),
                'end_time': suite_end_time.isoformat(),
                'duration_seconds': suite_duration,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate_percent': success_rate
            },
            'test_results': [asdict(result) for result in self.results],
            'performance_metrics': self._calculate_performance_metrics(),
            'compliance_status': self._check_compliance_requirements()
        }
        
        # Save report
        self._save_test_report(report)
        
        # Log summary
        self.logger.info(f"E2E Test Suite Completed")
        self.logger.info(f"Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
        self.logger.info(f"Total Duration: {suite_duration:.2f} seconds")
        
        if success_rate >= 90:
            self.logger.info("✅ E2E Test Suite PASSED - System ready for production")
        else:
            self.logger.error("❌ E2E Test Suite FAILED - System needs improvement")
        
        return report
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate aggregated performance metrics."""
        if not self.results:
            return {}
        
        durations = [r.duration_seconds for r in self.results]
        
        return {
            'total_execution_time': sum(durations),
            'average_test_duration': sum(durations) / len(durations),
            'longest_test_duration': max(durations),
            'shortest_test_duration': min(durations),
            'performance_requirements_met': self._check_performance_requirements()
        }
    
    def _check_performance_requirements(self) -> Dict[str, bool]:
        """Check if performance requirements are met."""
        requirements = {
            'data_processing_under_30s': False,
            'feature_generation_200_plus': False,
            'prediction_latency_under_100ms': False,
            'stress_test_1000_concurrent': False
        }
        
        for result in self.results:
            if result.test_name == "Data Pipeline - Realistic Volume":
                if result.success and result.duration_seconds < 30:
                    requirements['data_processing_under_30s'] = True
            
            elif result.test_name == "Feature Engineering Workflow":
                if result.success and result.details.get('feature_count', 0) >= 200:
                    requirements['feature_generation_200_plus'] = True
            
            elif result.test_name == "Prediction Engine Workflow":
                if result.success and result.metrics.get('average_latency_ms', 999) < 100:
                    requirements['prediction_latency_under_100ms'] = True
            
            elif result.test_name == "Stress Testing":
                if result.success and result.details.get('concurrent_users', 0) >= 100:
                    requirements['stress_test_1000_concurrent'] = True
        
        return requirements
    
    def _check_compliance_requirements(self) -> Dict[str, bool]:
        """Check compliance requirements status."""
        return {
            'e2e_tests_passed': all(r.success for r in self.results),
            'realistic_data_volumes_tested': any(
                r.test_name == "Data Pipeline - Realistic Volume" and r.success 
                for r in self.results
            ),
            'complete_workflows_validated': len([
                r for r in self.results 
                if r.success and 'workflow' in r.test_name.lower()
            ]) >= 3,
            'stress_testing_completed': any(
                r.test_name == "Stress Testing" and r.success 
                for r in self.results
            )
        }
    
    def _save_test_report(self, report: Dict[str, Any]):
        """Save comprehensive test report."""
        # Create reports directory
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = reports_dir / f"e2e_test_report_{timestamp}.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Test report saved: {json_file}")


def main():
    """Main function to run comprehensive E2E tests."""
    print("Starting ML-TA Comprehensive E2E Test Suite")
    print("=" * 80)
    
    # Initialize test suite
    test_suite = ComprehensiveE2ETestSuite()
    
    # Run comprehensive tests
    report = test_suite.run_comprehensive_suite()
    
    # Return exit code based on success rate
    success_rate = report['suite_summary']['success_rate_percent']
    return 0 if success_rate >= 90 else 1


if __name__ == "__main__":
    exit(main())
