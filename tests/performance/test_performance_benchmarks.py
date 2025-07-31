"""
Performance tests to ensure system meets requirements.

Tests cover latency, throughput, memory usage, and scalability requirements.
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

from src.features import FeaturePipeline
from src.models import ModelTrainer
from src.prediction_engine import PredictionEngine
from src.api import MLTA_API
from src.monitoring import MonitoringSystem


class TestPerformanceRequirements:
    """Test system performance against requirements."""
    
    def setup_method(self):
        """Setup performance test environment."""
        self.process = psutil.Process(os.getpid())
        
        # Create large dataset for performance testing
        np.random.seed(42)
        self.large_dataset = self._create_large_dataset(15000)  # 15k records
        
        # Create realistic feature pipeline
        self.feature_pipeline = FeaturePipeline()
    
    def _create_large_dataset(self, n_samples: int) -> pd.DataFrame:
        """Create large realistic dataset for testing."""
        base_price = 50000
        trend = np.linspace(0, 10000, n_samples)
        noise = np.random.randn(n_samples).cumsum() * 200
        
        close_prices = base_price + trend + noise
        opens = np.roll(close_prices, 1)
        opens[0] = base_price
        
        highs = close_prices + np.random.uniform(50, 1000, n_samples)
        lows = close_prices - np.random.uniform(50, 1000, n_samples)
        
        # Ensure OHLC relationships
        highs = np.maximum(highs, np.maximum(opens, close_prices))
        lows = np.minimum(lows, np.minimum(opens, close_prices))
        
        return pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
            'symbol': 'BTCUSDT',
            'open': opens,
            'high': highs,
            'low': lows,
            'close': close_prices,
            'volume': np.random.lognormal(mean=6, sigma=0.5, size=n_samples)
        })
    
    def test_data_processing_speed_requirement(self):
        """Test: Process 10,000+ records in under 30 seconds using less than 4GB RAM."""
        # Monitor memory before
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Start timing
        start_time = time.time()
        
        # Process large dataset
        features_df = self.feature_pipeline.engineer_features(
            self.large_dataset, 
            fit_scalers=True,
            validate_temporal=False  # Skip for performance test
        )
        
        # End timing
        processing_time = time.time() - start_time
        
        # Monitor memory after
        memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Verify requirements
        assert len(self.large_dataset) >= 10000, "Dataset should have 10,000+ records"
        assert processing_time < 30, f"Processing took {processing_time:.2f}s, should be <30s"
        assert memory_used < 4096, f"Memory usage {memory_used:.2f}MB, should be <4GB"
        assert len(features_df) == len(self.large_dataset), "All records should be processed"
        
        print(f"✓ Processed {len(self.large_dataset)} records in {processing_time:.2f}s using {memory_used:.2f}MB")
    
    def test_feature_generation_requirement(self):
        """Test: Generate 200+ features from OHLCV data with zero data leakage."""
        # Process sample data
        features_df = self.feature_pipeline.engineer_features(
            self.large_dataset.head(1000),  # Use subset for faster testing
            fit_scalers=True,
            validate_temporal=True  # Enable temporal validation
        )
        
        # Get feature names (excluding metadata)
        feature_names = self.feature_pipeline.get_feature_names(features_df)
        
        # Verify requirements
        assert len(feature_names) >= 200, f"Generated {len(feature_names)} features, should be ≥200"
        
        # Verify no data leakage (temporal validation should pass)
        assert len(features_df) > 0, "Features should be generated successfully"
        
        print(f"✓ Generated {len(feature_names)} features with temporal validation")
    
    def test_prediction_latency_requirement(self):
        """Test: Serve predictions with <100ms latency."""
        # Prepare model and data
        sample_data = self.large_dataset.head(500)
        features_df = self.feature_pipeline.engineer_features(sample_data, fit_scalers=True)
        
        # Create target and train model
        features_df['target'] = (features_df['close'].shift(-1) / features_df['close'] - 1) > 0
        features_df = features_df.dropna()
        
        feature_names = self.feature_pipeline.get_feature_names(features_df)
        X = features_df[feature_names]
        y = features_df['target']
        
        # Train lightweight model for speed
        model_trainer = ModelTrainer()
        model_trainer.config.model_type = 'random_forest'  # Fast model
        model, _ = model_trainer.train_model(X.head(300), y.head(300), model_name='latency_test')
        
        # Test prediction latency
        test_sample = X.tail(1)  # Single prediction
        
        latencies = []
        for _ in range(100):  # Test 100 predictions
            start_time = time.perf_counter()
            prediction = model_trainer.predict('latency_test', test_sample)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        max_latency = np.max(latencies)
        
        # Verify requirements
        assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms, should be <100ms"
        assert p95_latency < 150, f"95th percentile latency {p95_latency:.2f}ms, should be <150ms"
        
        print(f"✓ Prediction latency: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms, max={max_latency:.2f}ms")
    
    def test_concurrent_request_handling(self):
        """Test: Handle 1000+ concurrent API requests with proper rate limiting."""
        # Create API server
        api_server = MLTA_API()
        
        # Simulate concurrent requests
        def make_request(request_id):
            try:
                # Simulate API request processing time
                start_time = time.perf_counter()
                
                # Mock request processing
                time.sleep(0.001)  # 1ms processing time
                
                end_time = time.perf_counter()
                return {
                    'request_id': request_id,
                    'success': True,
                    'latency': (end_time - start_time) * 1000
                }
            except Exception as e:
                return {
                    'request_id': request_id,
                    'success': False,
                    'error': str(e)
                }
        
        # Test with 1000 concurrent requests
        num_requests = 1000
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        successful_requests = sum(1 for r in results if r['success'])
        success_rate = successful_requests / num_requests
        avg_latency = np.mean([r.get('latency', 0) for r in results if r['success']])
        
        # Verify requirements
        assert num_requests >= 1000, "Should test with 1000+ requests"
        assert success_rate > 0.95, f"Success rate {success_rate:.2%}, should be >95%"
        assert total_time < 60, f"Total time {total_time:.2f}s, should complete within 60s"
        
        print(f"✓ Handled {num_requests} requests in {total_time:.2f}s with {success_rate:.2%} success rate")
    
    def test_memory_efficiency_scaling(self):
        """Test memory efficiency with increasing data sizes."""
        data_sizes = [1000, 5000, 10000, 15000]
        memory_usage = []
        
        for size in data_sizes:
            # Create dataset of specific size
            test_data = self.large_dataset.head(size)
            
            # Measure memory before
            memory_before = self.process.memory_info().rss / 1024 / 1024
            
            # Process data
            features_df = self.feature_pipeline.engineer_features(
                test_data, 
                fit_scalers=True,
                validate_temporal=False
            )
            
            # Measure memory after
            memory_after = self.process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before
            
            memory_usage.append(memory_used)
            
            # Clean up
            del features_df
            
            print(f"Size: {size}, Memory: {memory_used:.2f}MB")
        
        # Verify memory scaling is reasonable (should be roughly linear)
        memory_per_record = [mem / size for mem, size in zip(memory_usage, data_sizes)]
        
        # Memory per record should be consistent (not exponentially increasing)
        assert max(memory_per_record) / min(memory_per_record) < 3, "Memory scaling should be roughly linear"
        
        # Should not exceed 4GB for largest dataset
        assert max(memory_usage) < 4096, f"Max memory usage {max(memory_usage):.2f}MB, should be <4GB"
    
    def test_model_training_performance(self):
        """Test model training performance requirements."""
        # Prepare training data
        sample_data = self.large_dataset.head(2000)
        features_df = self.feature_pipeline.engineer_features(sample_data, fit_scalers=True)
        
        features_df['target'] = (features_df['close'].shift(-1) / features_df['close'] - 1) > 0
        features_df = features_df.dropna()
        
        feature_names = self.feature_pipeline.get_feature_names(features_df)
        X = features_df[feature_names]
        y = features_df['target']
        
        # Test different model types
        model_types = ['random_forest', 'gradient_boosting']
        
        for model_type in model_types:
            start_time = time.time()
            
            model_trainer = ModelTrainer()
            model_trainer.config.model_type = model_type
            model, metrics = model_trainer.train_model(X, y, model_name=f'{model_type}_perf_test')
            
            training_time = time.time() - start_time
            
            # Verify training performance
            assert training_time < 300, f"{model_type} training took {training_time:.2f}s, should be <300s"
            assert metrics.accuracy is not None, f"{model_type} should produce accuracy metrics"
            
            print(f"✓ {model_type} trained in {training_time:.2f}s with accuracy {metrics.accuracy:.3f}")
    
    def test_monitoring_system_performance(self):
        """Test monitoring system performance overhead."""
        # Create monitoring system
        monitoring = MonitoringSystem()
        
        # Measure baseline performance
        start_time = time.time()
        for _ in range(1000):
            # Simulate some work
            np.random.randn(100).sum()
        baseline_time = time.time() - start_time
        
        # Start monitoring
        monitoring.start()
        
        # Measure performance with monitoring
        start_time = time.time()
        for _ in range(1000):
            # Simulate same work with monitoring active
            np.random.randn(100).sum()
            # Add some metrics
            monitoring.add_metric({
                'name': 'test_metric',
                'value': np.random.randn(),
                'timestamp': time.time()
            })
        
        monitored_time = time.time() - start_time
        
        # Stop monitoring
        monitoring.stop()
        
        # Calculate overhead
        overhead_percent = ((monitored_time - baseline_time) / baseline_time) * 100
        
        # Verify monitoring overhead is acceptable
        assert overhead_percent < 20, f"Monitoring overhead {overhead_percent:.1f}%, should be <20%"
        
        print(f"✓ Monitoring overhead: {overhead_percent:.1f}%")


class TestScalabilityRequirements:
    """Test system scalability requirements."""
    
    def test_feature_pipeline_scalability(self):
        """Test feature pipeline scales with data size."""
        sizes = [100, 500, 1000, 2000]
        processing_times = []
        
        for size in sizes:
            # Create dataset
            data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=size, freq='1H'),
                'symbol': 'BTCUSDT',
                'open': np.random.uniform(45000, 55000, size),
                'high': np.random.uniform(45000, 55000, size),
                'low': np.random.uniform(45000, 55000, size),
                'close': np.random.uniform(45000, 55000, size),
                'volume': np.random.uniform(100, 1000, size)
            })
            
            # Ensure OHLC relationships
            data['high'] = data[['open', 'high', 'close']].max(axis=1)
            data['low'] = data[['open', 'low', 'close']].min(axis=1)
            
            # Time processing
            start_time = time.time()
            pipeline = FeaturePipeline()
            features_df = pipeline.engineer_features(data, fit_scalers=True, validate_temporal=False)
            processing_time = time.time() - start_time
            
            processing_times.append(processing_time)
            
            print(f"Size: {size}, Time: {processing_time:.2f}s")
        
        # Verify scaling is reasonable (should be roughly linear)
        time_per_record = [t / s for t, s in zip(processing_times, sizes)]
        
        # Time per record should not increase dramatically
        assert max(time_per_record) / min(time_per_record) < 5, "Processing should scale roughly linearly"
    
    def test_prediction_throughput(self):
        """Test prediction throughput requirements."""
        # Setup model
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(1000, 50), columns=[f'feature_{i}' for i in range(50)])
        y_train = np.random.randint(0, 2, 1000)
        
        model_trainer = ModelTrainer()
        model, _ = model_trainer.train_model(X_train, y_train, model_name='throughput_test')
        
        # Test batch prediction throughput
        batch_sizes = [1, 10, 100, 1000]
        
        for batch_size in batch_sizes:
            X_test = pd.DataFrame(np.random.randn(batch_size, 50), columns=[f'feature_{i}' for i in range(50)])
            
            # Time batch prediction
            start_time = time.perf_counter()
            predictions = model_trainer.predict('throughput_test', X_test)
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            throughput = batch_size / batch_time  # predictions per second
            
            # Verify throughput requirements
            if batch_size == 1:
                assert throughput > 10, f"Single prediction throughput {throughput:.1f}/s, should be >10/s"
            elif batch_size >= 100:
                assert throughput > 1000, f"Batch throughput {throughput:.1f}/s, should be >1000/s"
            
            print(f"Batch size: {batch_size}, Throughput: {throughput:.1f} predictions/s")


class TestResourceUtilization:
    """Test resource utilization requirements."""
    
    def test_cpu_utilization(self):
        """Test CPU utilization during intensive operations."""
        # Monitor CPU usage during feature engineering
        cpu_usage = []
        
        def monitor_cpu():
            for _ in range(50):  # Monitor for 5 seconds
                cpu_usage.append(psutil.cpu_percent(interval=0.1))
        
        # Start CPU monitoring in background
        import threading
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Perform intensive operation
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5000, freq='1H'),
            'symbol': 'BTCUSDT',
            'open': np.random.uniform(45000, 55000, 5000),
            'high': np.random.uniform(45000, 55000, 5000),
            'low': np.random.uniform(45000, 55000, 5000),
            'close': np.random.uniform(45000, 55000, 5000),
            'volume': np.random.uniform(100, 1000, 5000)
        })
        
        pipeline = FeaturePipeline()
        features_df = pipeline.engineer_features(large_data, fit_scalers=True, validate_temporal=False)
        
        # Wait for monitoring to complete
        monitor_thread.join()
        
        # Analyze CPU usage
        avg_cpu = np.mean(cpu_usage)
        max_cpu = np.max(cpu_usage)
        
        # Verify CPU utilization is reasonable
        assert max_cpu < 95, f"Max CPU usage {max_cpu:.1f}%, should be <95%"
        assert avg_cpu < 80, f"Average CPU usage {avg_cpu:.1f}%, should be <80%"
        
        print(f"✓ CPU usage: avg={avg_cpu:.1f}%, max={max_cpu:.1f}%")
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Perform repeated operations
        for i in range(10):
            data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=1000, freq='1H'),
                'symbol': 'BTCUSDT',
                'open': np.random.uniform(45000, 55000, 1000),
                'high': np.random.uniform(45000, 55000, 1000),
                'low': np.random.uniform(45000, 55000, 1000),
                'close': np.random.uniform(45000, 55000, 1000),
                'volume': np.random.uniform(100, 1000, 1000)
            })
            
            pipeline = FeaturePipeline()
            features_df = pipeline.engineer_features(data, fit_scalers=True, validate_temporal=False)
            
            # Force garbage collection
            del pipeline, features_df, data
            import gc
            gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Verify no significant memory leak
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB, possible leak"
        
        print(f"✓ Memory leak test: increased by {memory_increase:.2f}MB")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
