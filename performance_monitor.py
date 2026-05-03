"""
Performance monitoring and profiling utilities for COTCAgent
"""

import time
import logging
import psutil
import threading
from functools import wraps
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from collections import defaultdict
import statistics
import gc

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    cpu_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    api_calls: int = 0
    cache_hits: int = 0
    errors: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.end_time is not None

    def complete(self, **kwargs):
        """Mark operation as completed"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

        # Update metadata
        self.metadata.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'cpu_percent': self.cpu_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'memory_delta_mb': self.memory_delta_mb,
            'api_calls': self.api_calls,
            'cache_hits': self.cache_hits,
            'errors': self.errors,
            'metadata': self.metadata
        }


class PerformanceMonitor:
    """Performance monitoring system"""

    def __init__(self, enabled: bool = True, log_interval: int = 60):
        self.enabled = enabled
        self.log_interval = log_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = False

        if enabled:
            self.start_monitoring()

    def start_monitoring(self):
        """Start background performance monitoring"""
        if not self.enabled:
            return

        self._stop_monitoring = False
        self._monitor_thread = threading.Thread(
            target=self._background_monitor,
            daemon=True,
            name="PerformanceMonitor"
        )
        self._monitor_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop background performance monitoring"""
        if not self.enabled:
            return

        self._stop_monitoring = True
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")

    def _background_monitor(self):
        """Background monitoring thread"""
        last_log_time = time.time()

        while not self._stop_monitoring:
            try:
                current_time = time.time()

                # Periodic cleanup and logging
                if current_time - last_log_time >= self.log_interval:
                    self._periodic_cleanup()
                    self._log_performance_summary()
                    last_log_time = current_time

                time.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error in performance monitoring thread: {e}")
                time.sleep(5)  # Wait longer on error

    def _periodic_cleanup(self):
        """Periodic cleanup of old metrics"""
        cutoff_time = time.time() - (self.log_interval * 2)  # Keep last 2 intervals

        with self.lock:
            # Remove old completed metrics
            self.metrics_history = [
                m for m in self.metrics_history
                if not m.completed or m.end_time > cutoff_time
            ]

    def _log_performance_summary(self):
        """Log performance summary"""
        with self.lock:
            if not self.operation_stats:
                return

            summary = []
            for operation, durations in self.operation_stats.items():
                if durations:
                    avg_duration = statistics.mean(durations)
                    max_duration = max(durations)
                    min_duration = min(durations)
                    count = len(durations)

                    summary.append(
                        f"{operation}: count={count}, "
                        f"avg={avg_duration:.2f}s, "
                        f"min={min_duration:.2f}s, "
                        f"max={max_duration:.2f}s"
                    )

            if summary:
                logger.info(f"Performance summary: {'; '.join(summary)}")

    def start_operation(self, operation_name: str, **metadata) -> PerformanceMetrics:
        """Start monitoring an operation"""
        if not self.enabled:
            return PerformanceMetrics(operation_name, time.time())

        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        initial_cpu = psutil.cpu_percent()

        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            memory_usage_mb=initial_memory,
            cpu_percent=initial_cpu,
            metadata=metadata
        )

        with self.lock:
            self.metrics_history.append(metrics)

        return metrics

    def complete_operation(self, metrics: PerformanceMetrics, **kwargs):
        """Complete monitoring an operation"""
        if not self.enabled:
            return

        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        end_cpu = psutil.cpu_percent()

        metrics.memory_delta_mb = end_memory - (metrics.memory_usage_mb or 0)
        metrics.complete(**kwargs)

        with self.lock:
            self.operation_stats[metrics.operation_name].append(metrics.duration)

            # Keep only last 100 measurements per operation
            if len(self.operation_stats[metrics.operation_name]) > 100:
                self.operation_stats[metrics.operation_name] = \
                    self.operation_stats[metrics.operation_name][-100:]

        logger.debug(
            f"Operation '{metrics.operation_name}' completed in {metrics.duration:.2f}s "
            f"(CPU: {end_cpu:.1f}%, Memory: {end_memory:.1f}MB, "
            f"Delta: {metrics.memory_delta_mb:.1f}MB)"
        )

    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for an operation"""
        with self.lock:
            durations = self.operation_stats.get(operation_name, [])
            if not durations:
                return {}

            return {
                'count': len(durations),
                'average_duration': statistics.mean(durations),
                'median_duration': statistics.median(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'standard_deviation': statistics.stdev(durations) if len(durations) > 1 else 0
            }

    def get_recent_metrics(self, limit: int = 10) -> List[PerformanceMetrics]:
        """Get recent performance metrics"""
        with self.lock:
            return self.metrics_history[-limit:] if self.metrics_history else []

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': self.get_memory_usage(),
            'disk_usage': psutil.disk_usage('/').percent,
            'active_operations': len([m for m in self.metrics_history if not m.completed])
        }


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def performance_monitor(operation_name: Optional[str] = None):
    """
    Decorator for performance monitoring

    Args:
        operation_name: Optional custom operation name. If not provided,
                       uses function.__name__
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            metrics = _performance_monitor.start_operation(op_name)

            try:
                result = await func(*args, **kwargs)
                _performance_monitor.complete_operation(metrics, success=True)
                return result
            except Exception as e:
                _performance_monitor.complete_operation(metrics, success=False, error=str(e))
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            metrics = _performance_monitor.start_operation(op_name)

            try:
                result = func(*args, **kwargs)
                _performance_monitor.complete_operation(metrics, success=True)
                return result
            except Exception as e:
                _performance_monitor.complete_operation(metrics, success=False, error=str(e))
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    return _performance_monitor


def enable_performance_monitoring():
    """Enable performance monitoring"""
    _performance_monitor.enabled = True
    _performance_monitor.start_monitoring()


def disable_performance_monitoring():
    """Disable performance monitoring"""
    _performance_monitor.enabled = False
    _performance_monitor.stop_monitoring()


def get_performance_stats() -> Dict[str, Any]:
    """Get comprehensive performance statistics"""
    monitor = get_performance_monitor()

    return {
        'enabled': monitor.enabled,
        'operation_stats': {
            op: monitor.get_operation_stats(op)
            for op in monitor.operation_stats.keys()
        },
        'recent_metrics': [m.to_dict() for m in monitor.get_recent_metrics(20)],
        'system_stats': monitor.get_system_stats(),
        'total_operations': len(monitor.metrics_history)
    }


def cleanup_memory():
    """Force garbage collection and memory cleanup"""
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

    # Force garbage collection
    gc.collect()

    # Clear any caches if available
    try:
        import asyncio
        # Clear asyncio tasks
        loop = asyncio.get_event_loop()
        if loop:
            # Cancel any pending tasks (be careful with this)
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            if len(pending) > 10:  # Only cancel if many pending
                logger.warning(f"Cleaning up {len(pending)} pending asyncio tasks")
    except:
        pass

    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_saved = initial_memory - final_memory

    logger.info(f"Memory cleanup completed. Saved {memory_saved:.1f}MB")

    return memory_saved
