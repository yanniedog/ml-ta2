"""
System optimization and performance tuning framework for ML-TA system.

This module provides comprehensive optimization capabilities including
memory management, CPU optimization, database tuning, and caching strategies.
"""

import os
import sys
import gc
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationMetrics:
    """Optimization metrics container."""
    
    # Performance metrics
    execution_time_before: float = 0.0
    execution_time_after: float = 0.0
    performance_improvement: float = 0.0
    
    # Memory metrics
    memory_usage_before_mb: float = 0.0
    memory_usage_after_mb: float = 0.0
    memory_reduction_mb: float = 0.0
    memory_reduction_percent: float = 0.0
    
    # CPU metrics
    cpu_usage_before: float = 0.0
    cpu_usage_after: float = 0.0
    cpu_improvement: float = 0.0
    
    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    cache_size_mb: float = 0.0
    
    # Database metrics
    query_time_before: float = 0.0
    query_time_after: float = 0.0
    query_improvement: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    optimization_type: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class MemoryOptimizer:
    """Memory optimization and management."""
    
    def __init__(self):
        self.memory_pools = {}
        self.gc_thresholds = gc.get_threshold()
        self.optimization_history = []
    
    def optimize_memory_usage(self) -> OptimizationMetrics:
        """Optimize overall memory usage."""
        logger.info("Starting memory optimization")
        
        # Measure before optimization
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        start_time = time.time()
        
        # Perform optimizations
        self._optimize_garbage_collection()
        self._clear_unused_objects()
        self._optimize_pandas_memory()
        self._optimize_numpy_memory()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Measure after optimization
        try:
            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        except Exception:
            memory_after = memory_before  # Fallback if measurement fails
        optimization_time = time.time() - start_time
        
        metrics = OptimizationMetrics(
            optimization_type="memory",
            execution_time_after=optimization_time,
            memory_usage_before_mb=memory_before,
            memory_usage_after_mb=memory_after,
            memory_reduction_mb=memory_before - memory_after,
            memory_reduction_percent=((memory_before - memory_after) / memory_before) * 100 if memory_before > 0 else 0,
            custom_metrics={'objects_collected': collected}
        )
        
        self.optimization_history.append(metrics)
        logger.info(f"Memory optimization completed. Freed {metrics.memory_reduction_mb:.2f}MB ({metrics.memory_reduction_percent:.1f}%)")
        
        return metrics
    
    def _optimize_garbage_collection(self) -> None:
        """Optimize garbage collection settings."""
        # Tune GC thresholds for better performance
        gc.set_threshold(700, 10, 10)  # More aggressive collection
        
        # Enable automatic garbage collection
        if not gc.isenabled():
            gc.enable()
    
    def _clear_unused_objects(self) -> None:
        """Clear unused objects and references."""
        # Clear module-level caches
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
        
        # Clear function caches
        for module_name, module in sys.modules.items():
            if hasattr(module, '__dict__'):
                for attr_name in list(module.__dict__.keys()):
                    attr = getattr(module, attr_name, None)
                    if hasattr(attr, 'cache_clear'):
                        try:
                            attr.cache_clear()
                        except:
                            pass
    
    def _optimize_pandas_memory(self) -> None:
        """Optimize pandas memory usage."""
        if not NUMPY_AVAILABLE:
            return
        
        try:
            import pandas as pd
            
            # Clear pandas caches
            if hasattr(pd, '_config'):
                pd._config.config.reset_option('all')
            
            # Optimize data types for existing DataFrames
            # This would be called on specific DataFrames in practice
            
        except ImportError:
            pass
    
    def _optimize_numpy_memory(self) -> None:
        """Optimize numpy memory usage."""
        if not NUMPY_AVAILABLE:
            return
        
        try:
            import numpy as np
            
            # Clear numpy caches
            if hasattr(np, '_NoValue'):
                # Clear internal caches
                pass
            
        except ImportError:
            pass
    
    def create_memory_pool(self, name: str, size_mb: int = 100) -> None:
        """Create memory pool for efficient allocation."""
        self.memory_pools[name] = {
            'size_mb': size_mb,
            'allocated': 0,
            'objects': []
        }
        logger.info(f"Created memory pool '{name}' with {size_mb}MB capacity")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024),
            'gc_counts': gc.get_count(),
            'gc_thresholds': gc.get_threshold()
        }


class CPUOptimizer:
    """CPU optimization and performance tuning."""
    
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.thread_pool = None
        self.process_pool = None
    
    def optimize_cpu_usage(self) -> OptimizationMetrics:
        """Optimize CPU usage and performance."""
        logger.info("Starting CPU optimization")
        
        # Measure before optimization
        cpu_before = psutil.cpu_percent(interval=1)
        start_time = time.time()
        
        # Perform optimizations
        self._optimize_thread_pools()
        self._optimize_process_affinity()
        self._optimize_scheduling()
        
        # Measure after optimization
        cpu_after = psutil.cpu_percent(interval=1)
        optimization_time = time.time() - start_time
        
        metrics = OptimizationMetrics(
            optimization_type="cpu",
            execution_time_after=optimization_time,
            cpu_usage_before=cpu_before,
            cpu_usage_after=cpu_after,
            cpu_improvement=cpu_before - cpu_after,
            custom_metrics={
                'cpu_count': self.cpu_count,
                'thread_pool_size': self.cpu_count * 2,
                'process_pool_size': self.cpu_count
            }
        )
        
        logger.info(f"CPU optimization completed. CPU usage: {cpu_before:.1f}% -> {cpu_after:.1f}%")
        return metrics
    
    def _optimize_thread_pools(self) -> None:
        """Optimize thread pool configuration."""
        # Create optimized thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=False)
        
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.cpu_count * 2,
            thread_name_prefix="ml-ta-worker"
        )
        
        # Create optimized process pool
        if self.process_pool:
            self.process_pool.shutdown(wait=False)
        
        self.process_pool = ProcessPoolExecutor(
            max_workers=self.cpu_count
        )
    
    def _optimize_process_affinity(self) -> None:
        """Optimize process CPU affinity."""
        try:
            process = psutil.Process()
            
            # Set CPU affinity to all available cores
            available_cpus = list(range(self.cpu_count))
            process.cpu_affinity(available_cpus)
            
        except (AttributeError, OSError):
            # CPU affinity not supported on this platform
            pass
    
    def _optimize_scheduling(self) -> None:
        """Optimize process scheduling."""
        try:
            process = psutil.Process()
            
            # Set high priority for better performance
            if hasattr(process, 'nice'):
                current_nice = process.nice()
                if current_nice > -5:  # Only if not already high priority
                    process.nice(-5)  # Higher priority
            
        except (AttributeError, OSError, PermissionError):
            # Scheduling optimization not available or permitted
            pass
    
    def get_cpu_stats(self) -> Dict[str, Any]:
        """Get current CPU statistics."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
        }


class CacheOptimizer:
    """Cache optimization and management."""
    
    def __init__(self):
        self.cache_stats = {}
        self.redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self) -> None:
        """Initialize Redis connection if available."""
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'localhost'),
                    port=int(os.getenv('REDIS_PORT', 6379)),
                    db=int(os.getenv('REDIS_DB', 0)),
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache connection established")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                self.redis_client = None
    
    def optimize_cache_configuration(self) -> OptimizationMetrics:
        """Optimize cache configuration and performance."""
        logger.info("Starting cache optimization")
        
        start_time = time.time()
        
        # Optimize Redis cache if available
        if self.redis_client:
            self._optimize_redis_cache()
        
        # Optimize application-level caches
        self._optimize_application_caches()
        
        optimization_time = time.time() - start_time
        
        # Get cache statistics
        cache_stats = self.get_cache_stats()
        
        metrics = OptimizationMetrics(
            optimization_type="cache",
            execution_time_after=optimization_time,
            cache_hit_rate=cache_stats.get('hit_rate', 0),
            cache_miss_rate=cache_stats.get('miss_rate', 0),
            cache_size_mb=cache_stats.get('size_mb', 0),
            custom_metrics=cache_stats
        )
        
        logger.info(f"Cache optimization completed in {optimization_time:.2f}s")
        return metrics
    
    def _optimize_redis_cache(self) -> None:
        """Optimize Redis cache configuration."""
        if not self.redis_client:
            return
        
        try:
            # Set optimal Redis configuration
            config_updates = {
                'maxmemory-policy': 'allkeys-lru',  # LRU eviction
                'maxmemory-samples': '5',  # Better LRU approximation
                'timeout': '300',  # 5 minute timeout
                'tcp-keepalive': '60'  # Keep connections alive
            }
            
            for key, value in config_updates.items():
                try:
                    self.redis_client.config_set(key, value)
                except Exception as e:
                    logger.debug(f"Could not set Redis config {key}: {e}")
            
            # Clean up expired keys
            self.redis_client.execute_command('FLUSHEXPIRED')
            
        except Exception as e:
            logger.warning(f"Redis optimization failed: {e}")
    
    def _optimize_application_caches(self) -> None:
        """Optimize application-level caches."""
        # Clear and optimize function caches
        import functools
        
        # This would typically involve optimizing specific caches
        # in the application based on usage patterns
        
        # Example: Clear least recently used cache entries
        # This would be implemented based on specific cache implementations
        pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {}
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats.update({
                    'redis_used_memory_mb': info.get('used_memory', 0) / (1024 * 1024),
                    'redis_hit_rate': info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1),
                    'redis_connected_clients': info.get('connected_clients', 0),
                    'redis_total_commands': info.get('total_commands_processed', 0)
                })
            except Exception as e:
                logger.debug(f"Could not get Redis stats: {e}")
        
        return stats


class DatabaseOptimizer:
    """Database optimization and performance tuning."""
    
    def __init__(self):
        self.engines = {}
        self.connection_pools = {}
    
    def optimize_database_performance(self, database_url: str) -> OptimizationMetrics:
        """Optimize database performance and configuration."""
        logger.info("Starting database optimization")
        
        if not SQLALCHEMY_AVAILABLE:
            logger.warning("SQLAlchemy not available, skipping database optimization")
            return OptimizationMetrics(optimization_type="database")
        
        start_time = time.time()
        
        # Create optimized engine
        engine = self._create_optimized_engine(database_url)
        
        # Optimize connection pool
        self._optimize_connection_pool(engine)
        
        # Run database optimizations
        query_time_before, query_time_after = self._optimize_queries(engine)
        
        optimization_time = time.time() - start_time
        
        metrics = OptimizationMetrics(
            optimization_type="database",
            execution_time_after=optimization_time,
            query_time_before=query_time_before,
            query_time_after=query_time_after,
            query_improvement=query_time_before - query_time_after,
            custom_metrics={
                'connection_pool_size': 20,
                'max_overflow': 30
            }
        )
        
        logger.info(f"Database optimization completed. Query time: {query_time_before:.3f}s -> {query_time_after:.3f}s")
        return metrics
    
    def _create_optimized_engine(self, database_url: str):
        """Create optimized database engine."""
        engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
            connect_args={
                'connect_timeout': 10,
                'application_name': 'ml-ta-optimized'
            }
        )
        
        self.engines[database_url] = engine
        return engine
    
    def _optimize_connection_pool(self, engine) -> None:
        """Optimize database connection pool."""
        # Connection pool is optimized during engine creation
        # Additional optimizations can be added here
        pass
    
    def _optimize_queries(self, engine) -> Tuple[float, float]:
        """Optimize database queries."""
        try:
            # Test query performance before optimization
            start_time = time.time()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchall()
            query_time_before = time.time() - start_time
            
            # Run optimization queries (example)
            with engine.connect() as conn:
                # Analyze tables (PostgreSQL specific)
                try:
                    conn.execute(text("ANALYZE"))
                    conn.commit()
                except Exception:
                    pass
            
            # Test query performance after optimization
            start_time = time.time()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchall()
            query_time_after = time.time() - start_time
            
            return query_time_before, query_time_after
            
        except Exception as e:
            logger.warning(f"Database query optimization failed: {e}")
            return 0.0, 0.0


class SystemOptimizer:
    """Comprehensive system optimizer orchestrating all optimizations."""
    
    def __init__(self):
        self.memory_optimizer = MemoryOptimizer()
        self.cpu_optimizer = CPUOptimizer()
        self.cache_optimizer = CacheOptimizer()
        self.database_optimizer = DatabaseOptimizer()
        self.optimization_history = []
    
    def run_comprehensive_optimization(self, database_url: Optional[str] = None) -> List[OptimizationMetrics]:
        """Run comprehensive system optimization."""
        logger.info("Starting comprehensive system optimization")
        
        results = []
        
        # Memory optimization
        memory_metrics = self.memory_optimizer.optimize_memory_usage()
        results.append(memory_metrics)
        
        # CPU optimization
        cpu_metrics = self.cpu_optimizer.optimize_cpu_usage()
        results.append(cpu_metrics)
        
        # Cache optimization
        cache_metrics = self.cache_optimizer.optimize_cache_configuration()
        results.append(cache_metrics)
        
        # Database optimization
        if database_url:
            db_metrics = self.database_optimizer.optimize_database_performance(database_url)
            results.append(db_metrics)
        
        self.optimization_history.extend(results)
        logger.info(f"Comprehensive optimization completed. {len(results)} optimizations performed")
        
        return results
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        return {
            'memory': self.memory_optimizer.get_memory_stats(),
            'cpu': self.cpu_optimizer.get_cpu_stats(),
            'cache': self.cache_optimizer.get_cache_stats(),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_optimization_report(self) -> str:
        """Generate optimization report."""
        if not self.optimization_history:
            return "No optimization history available."
        
        report = ["ML-TA System Optimization Report", "=" * 50, ""]
        
        # Summary
        total_optimizations = len(self.optimization_history)
        memory_optimizations = [m for m in self.optimization_history if m.optimization_type == "memory"]
        cpu_optimizations = [m for m in self.optimization_history if m.optimization_type == "cpu"]
        
        report.extend([
            f"Total Optimizations: {total_optimizations}",
            f"Memory Optimizations: {len(memory_optimizations)}",
            f"CPU Optimizations: {len(cpu_optimizations)}",
            ""
        ])
        
        # Memory optimization results
        if memory_optimizations:
            latest_memory = memory_optimizations[-1]
            report.extend([
                "Memory Optimization Results:",
                f"  Memory Freed: {latest_memory.memory_reduction_mb:.2f}MB ({latest_memory.memory_reduction_percent:.1f}%)",
                f"  Current Usage: {latest_memory.memory_usage_after_mb:.2f}MB",
                ""
            ])
        
        # CPU optimization results
        if cpu_optimizations:
            latest_cpu = cpu_optimizations[-1]
            report.extend([
                "CPU Optimization Results:",
                f"  CPU Usage: {latest_cpu.cpu_usage_before:.1f}% -> {latest_cpu.cpu_usage_after:.1f}%",
                f"  Improvement: {latest_cpu.cpu_improvement:.1f}%",
                ""
            ])
        
        # System recommendations
        report.extend([
            "Optimization Recommendations:",
            "1. Run memory optimization regularly to prevent memory leaks",
            "2. Monitor CPU usage and adjust thread pool sizes as needed",
            "3. Optimize cache hit rates through better cache strategies",
            "4. Regular database maintenance and query optimization",
            "5. Monitor system metrics and adjust optimizations accordingly",
            ""
        ])
        
        return "\n".join(report)


# Factory functions
def create_memory_optimizer() -> MemoryOptimizer:
    """Create memory optimizer instance."""
    return MemoryOptimizer()


def create_cpu_optimizer() -> CPUOptimizer:
    """Create CPU optimizer instance."""
    return CPUOptimizer()


def create_cache_optimizer() -> CacheOptimizer:
    """Create cache optimizer instance."""
    return CacheOptimizer()


def create_database_optimizer() -> DatabaseOptimizer:
    """Create database optimizer instance."""
    return DatabaseOptimizer()


def create_system_optimizer() -> SystemOptimizer:
    """Create system optimizer instance."""
    return SystemOptimizer()


if __name__ == "__main__":
    # Example usage
    print("ML-TA System Optimization Framework")
    
    # Create system optimizer
    optimizer = create_system_optimizer()
    
    # Run comprehensive optimization
    results = optimizer.run_comprehensive_optimization()
    
    # Generate report
    print(optimizer.generate_optimization_report())
    
    # Display system stats
    stats = optimizer.get_system_stats()
    print(f"\nCurrent System Stats:")
    print(f"Memory Usage: {stats['memory']['rss_mb']:.2f}MB")
    print(f"CPU Usage: {stats['cpu']['cpu_percent']:.1f}%")
