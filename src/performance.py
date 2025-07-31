"""
Performance testing and optimization framework for ML-TA system.

This module provides comprehensive performance testing, profiling, and optimization
capabilities for the ML-TA trading analysis platform.
"""

import time
import psutil
import threading
import statistics
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import asyncio
import concurrent.futures
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import numpy as np
    import pandas as pd
    from memory_profiler import profile
    import cProfile
    import pstats
    import io
except ImportError:
    # Fallback for missing optional dependencies
    np = None
    pd = None
    profile = lambda x: x
    cProfile = None
    pstats = None
    io = None

from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    
    # Timing metrics
    execution_time: float = 0.0
    avg_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # Throughput metrics
    requests_per_second: float = 0.0
    transactions_per_second: float = 0.0
    operations_per_second: float = 0.0
    
    # Resource metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_peak_mb: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_io_sent_mb: float = 0.0
    network_io_recv_mb: float = 0.0
    
    # Error metrics
    error_count: int = 0
    error_rate: float = 0.0
    timeout_count: int = 0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    test_name: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0


@dataclass
class LoadTestConfig:
    """Load test configuration."""
    
    concurrent_users: int = 10
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    ramp_down_seconds: int = 10
    request_timeout: float = 30.0
    think_time_seconds: float = 1.0
    
    # Test scenarios
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    
    # Resource limits
    max_cpu_percent: float = 90.0
    max_memory_mb: float = 4096.0
    
    # Assertions
    max_avg_response_time: float = 1.0
    min_requests_per_second: float = 100.0
    max_error_rate: float = 0.01


class PerformanceProfiler:
    """Performance profiler for detailed code analysis."""
    
    def __init__(self):
        self.profiles = {}
        self.active_profiles = {}
    
    def start_profile(self, name: str) -> None:
        """Start profiling a code section."""
        if cProfile:
            profiler = cProfile.Profile()
            profiler.enable()
            self.active_profiles[name] = {
                'profiler': profiler,
                'start_time': time.time()
            }
        
        logger.info(f"Started profiling: {name}")
    
    def stop_profile(self, name: str) -> Optional[Dict[str, Any]]:
        """Stop profiling and return results."""
        if name not in self.active_profiles:
            logger.warning(f"No active profile found: {name}")
            return None
        
        profile_data = self.active_profiles.pop(name)
        
        if cProfile and pstats and io:
            profiler = profile_data['profiler']
            profiler.disable()
            
            # Capture profile stats
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            profile_result = {
                'name': name,
                'duration': time.time() - profile_data['start_time'],
                'stats': stats_stream.getvalue(),
                'timestamp': datetime.now()
            }
            
            self.profiles[name] = profile_result
            logger.info(f"Completed profiling: {name}")
            return profile_result
        
        return None
    
    def get_profile_report(self, name: str) -> Optional[str]:
        """Get formatted profile report."""
        if name in self.profiles:
            profile = self.profiles[name]
            return f"""
Performance Profile: {profile['name']}
Duration: {profile['duration']:.3f} seconds
Timestamp: {profile['timestamp']}

Top Functions by Cumulative Time:
{profile['stats']}
"""
        return None


class ResourceMonitor:
    """System resource monitoring."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.metrics_history = []
        self.monitor_thread = None
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics_history = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Started resource monitoring")
    
    def stop_monitoring(self) -> List[Dict[str, Any]]:
        """Stop monitoring and return collected metrics."""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        logger.info(f"Stopped resource monitoring. Collected {len(self.metrics_history)} samples")
        return self.metrics_history.copy()
    
    def _monitor_loop(self) -> None:
        """Resource monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                process_cpu = process.cpu_percent()
                
                # Memory metrics
                memory = psutil.virtual_memory()
                process_memory = process.memory_info()
                
                # Disk I/O metrics
                disk_io = psutil.disk_io_counters()
                process_io = process.io_counters() if hasattr(process, 'io_counters') else None
                
                # Network I/O metrics
                network_io = psutil.net_io_counters()
                
                metrics = {
                    'timestamp': datetime.now(),
                    'cpu_percent_system': cpu_percent,
                    'cpu_percent_process': process_cpu,
                    'memory_percent': memory.percent,
                    'memory_available_mb': memory.available / (1024 * 1024),
                    'memory_used_mb': memory.used / (1024 * 1024),
                    'process_memory_rss_mb': process_memory.rss / (1024 * 1024),
                    'process_memory_vms_mb': process_memory.vms / (1024 * 1024),
                }
                
                if disk_io:
                    metrics.update({
                        'disk_read_mb': disk_io.read_bytes / (1024 * 1024),
                        'disk_write_mb': disk_io.write_bytes / (1024 * 1024),
                    })
                
                if process_io:
                    metrics.update({
                        'process_disk_read_mb': process_io.read_bytes / (1024 * 1024),
                        'process_disk_write_mb': process_io.write_bytes / (1024 * 1024),
                    })
                
                if network_io:
                    metrics.update({
                        'network_sent_mb': network_io.bytes_sent / (1024 * 1024),
                        'network_recv_mb': network_io.bytes_recv / (1024 * 1024),
                    })
                
                self.metrics_history.append(metrics)
                
            except Exception as e:
                logger.error(f"Error collecting resource metrics: {e}")
            
            time.sleep(self.interval)
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """Get summary statistics from collected metrics."""
        if not self.metrics_history:
            return {}
        
        summary = {}
        numeric_keys = [k for k in self.metrics_history[0].keys() if k != 'timestamp']
        
        for key in numeric_keys:
            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                summary[f"{key}_avg"] = statistics.mean(values)
                summary[f"{key}_max"] = max(values)
                summary[f"{key}_min"] = min(values)
                if len(values) > 1:
                    summary[f"{key}_std"] = statistics.stdev(values)
        
        return summary


class LoadTester:
    """Load testing framework."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results = []
        self.resource_monitor = ResourceMonitor()
        self.profiler = PerformanceProfiler()
    
    async def run_load_test(self, test_function: Callable, *args, **kwargs) -> PerformanceMetrics:
        """Run load test with specified function."""
        logger.info(f"Starting load test with {self.config.concurrent_users} users for {self.config.duration_seconds}s")
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        self.profiler.start_profile("load_test")
        
        start_time = time.time()
        response_times = []
        error_count = 0
        timeout_count = 0
        
        try:
            # Create semaphore for concurrent users
            semaphore = asyncio.Semaphore(self.config.concurrent_users)
            
            # Run test scenarios
            tasks = []
            end_time = start_time + self.config.duration_seconds
            
            while time.time() < end_time:
                task = asyncio.create_task(
                    self._run_test_scenario(semaphore, test_function, *args, **kwargs)
                )
                tasks.append(task)
                
                # Control request rate
                await asyncio.sleep(self.config.think_time_seconds / self.config.concurrent_users)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                elif isinstance(result, dict):
                    if result.get('timeout'):
                        timeout_count += 1
                    elif result.get('error'):
                        error_count += 1
                    else:
                        response_times.append(result.get('response_time', 0))
        
        except Exception as e:
            logger.error(f"Load test failed: {e}")
            error_count += 1
        
        finally:
            # Stop monitoring
            resource_metrics = self.resource_monitor.stop_monitoring()
            self.profiler.stop_profile("load_test")
        
        # Calculate metrics
        total_time = time.time() - start_time
        total_requests = len(response_times) + error_count + timeout_count
        
        metrics = PerformanceMetrics(
            test_name="load_test",
            execution_time=total_time,
            duration_seconds=total_time,
            error_count=error_count,
            timeout_count=timeout_count
        )
        
        if response_times:
            metrics.avg_response_time = statistics.mean(response_times)
            metrics.min_response_time = min(response_times)
            metrics.max_response_time = max(response_times)
            metrics.p95_response_time = np.percentile(response_times, 95) if np else 0
            metrics.p99_response_time = np.percentile(response_times, 99) if np else 0
            metrics.requests_per_second = len(response_times) / total_time
        
        if total_requests > 0:
            metrics.error_rate = error_count / total_requests
        
        # Add resource metrics
        if resource_metrics:
            resource_summary = self.resource_monitor.get_summary_metrics()
            metrics.cpu_usage_percent = resource_summary.get('cpu_percent_process_avg', 0)
            metrics.memory_usage_mb = resource_summary.get('process_memory_rss_mb_avg', 0)
            metrics.memory_peak_mb = resource_summary.get('process_memory_rss_mb_max', 0)
        
        logger.info(f"Load test completed: {metrics.requests_per_second:.2f} RPS, {metrics.error_rate:.3f} error rate")
        return metrics
    
    async def _run_test_scenario(self, semaphore: asyncio.Semaphore, test_function: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Run individual test scenario."""
        async with semaphore:
            start_time = time.time()
            
            try:
                # Run test function with timeout
                if asyncio.iscoroutinefunction(test_function):
                    await asyncio.wait_for(
                        test_function(*args, **kwargs),
                        timeout=self.config.request_timeout
                    )
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, 
                        lambda: test_function(*args, **kwargs)
                    )
                
                response_time = time.time() - start_time
                return {'response_time': response_time}
                
            except asyncio.TimeoutError:
                return {'timeout': True}
            except Exception as e:
                logger.debug(f"Test scenario error: {e}")
                return {'error': str(e)}


class BenchmarkSuite:
    """Comprehensive benchmark suite for ML-TA system."""
    
    def __init__(self):
        self.benchmarks = {}
        self.results = {}
        self.profiler = PerformanceProfiler()
    
    def register_benchmark(self, name: str, function: Callable, *args, **kwargs) -> None:
        """Register a benchmark function."""
        self.benchmarks[name] = {
            'function': function,
            'args': args,
            'kwargs': kwargs
        }
        logger.info(f"Registered benchmark: {name}")
    
    def run_benchmark(self, name: str, iterations: int = 100) -> PerformanceMetrics:
        """Run a specific benchmark."""
        if name not in self.benchmarks:
            raise ValueError(f"Benchmark not found: {name}")
        
        benchmark = self.benchmarks[name]
        function = benchmark['function']
        args = benchmark['args']
        kwargs = benchmark['kwargs']
        
        logger.info(f"Running benchmark: {name} ({iterations} iterations)")
        
        # Start profiling
        self.profiler.start_profile(name)
        
        execution_times = []
        error_count = 0
        
        start_time = time.time()
        
        for i in range(iterations):
            try:
                iter_start = time.time()
                function(*args, **kwargs)
                iter_time = time.time() - iter_start
                execution_times.append(iter_time)
                
            except Exception as e:
                logger.debug(f"Benchmark iteration {i} failed: {e}")
                error_count += 1
        
        total_time = time.time() - start_time
        
        # Stop profiling
        self.profiler.stop_profile(name)
        
        # Calculate metrics
        metrics = PerformanceMetrics(
            test_name=name,
            execution_time=total_time,
            duration_seconds=total_time,
            error_count=error_count
        )
        
        if execution_times:
            metrics.avg_response_time = statistics.mean(execution_times)
            metrics.min_response_time = min(execution_times)
            metrics.max_response_time = max(execution_times)
            metrics.operations_per_second = len(execution_times) / max(total_time, 0.001)
            
            if np:
                metrics.p95_response_time = np.percentile(execution_times, 95)
                metrics.p99_response_time = np.percentile(execution_times, 99)
        
        if iterations > 0:
            metrics.error_rate = error_count / iterations
        
        self.results[name] = metrics
        logger.info(f"Benchmark {name} completed: {metrics.operations_per_second:.2f} ops/sec")
        
        return metrics
    
    def run_all_benchmarks(self, iterations: int = 100) -> Dict[str, PerformanceMetrics]:
        """Run all registered benchmarks."""
        logger.info(f"Running {len(self.benchmarks)} benchmarks")
        
        for name in self.benchmarks:
            self.run_benchmark(name, iterations)
        
        return self.results.copy()
    
    def generate_report(self) -> str:
        """Generate benchmark report."""
        if not self.results:
            return "No benchmark results available."
        
        report = ["ML-TA Performance Benchmark Report", "=" * 50, ""]
        
        for name, metrics in self.results.items():
            report.extend([
                f"Benchmark: {name}",
                f"  Operations/sec: {metrics.operations_per_second:.2f}",
                f"  Avg Response Time: {metrics.avg_response_time*1000:.2f}ms",
                f"  Min Response Time: {metrics.min_response_time*1000:.2f}ms",
                f"  Max Response Time: {metrics.max_response_time*1000:.2f}ms",
                f"  P95 Response Time: {metrics.p95_response_time*1000:.2f}ms",
                f"  P99 Response Time: {metrics.p99_response_time*1000:.2f}ms",
                f"  Error Rate: {metrics.error_rate:.3f}",
                f"  Total Duration: {metrics.duration_seconds:.2f}s",
                ""
            ])
        
        return "\n".join(report)


# Factory functions
def create_performance_profiler() -> PerformanceProfiler:
    """Create performance profiler instance."""
    return PerformanceProfiler()


def create_resource_monitor(interval: float = 1.0) -> ResourceMonitor:
    """Create resource monitor instance."""
    return ResourceMonitor(interval)


def create_load_tester(config: LoadTestConfig) -> LoadTester:
    """Create load tester instance."""
    return LoadTester(config)


def create_benchmark_suite() -> BenchmarkSuite:
    """Create benchmark suite instance."""
    return BenchmarkSuite()


# Decorators for easy performance testing
def benchmark(name: str = None, iterations: int = 100):
    """Decorator for benchmarking functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            benchmark_name = name or func.__name__
            
            # Create temporary benchmark suite
            suite = BenchmarkSuite()
            suite.register_benchmark(benchmark_name, func, *args, **kwargs)
            result = suite.run_benchmark(benchmark_name, iterations)
            
            return result
        
        return wrapper
    return decorator


def profile_performance(name: str = None):
    """Decorator for profiling function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            profile_name = name or func.__name__
            
            profiler = PerformanceProfiler()
            profiler.start_profile(profile_name)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.stop_profile(profile_name)
                report = profiler.get_profile_report(profile_name)
                if report:
                    logger.info(f"Performance profile for {profile_name}:\n{report}")
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    print("ML-TA Performance Testing Framework")
    
    # Create benchmark suite
    suite = create_benchmark_suite()
    
    # Example benchmark functions
    def cpu_intensive_task():
        """CPU intensive benchmark."""
        total = 0
        for i in range(100000):
            total += i ** 2
        return total
    
    def memory_intensive_task():
        """Memory intensive benchmark."""
        data = list(range(100000))
        return sum(data)
    
    # Register benchmarks
    suite.register_benchmark("cpu_intensive", cpu_intensive_task)
    suite.register_benchmark("memory_intensive", memory_intensive_task)
    
    # Run benchmarks
    results = suite.run_all_benchmarks(iterations=50)
    
    # Generate report
    print(suite.generate_report())
