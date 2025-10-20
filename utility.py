import datetime
import gc
import hashlib
import platform
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from collections import deque, defaultdict
from dataclasses import field, dataclass
from datetime import datetime, date, timedelta
from enum import Enum
from fcntl import fcntl
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Type, Callable

import ollama
import psutil
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from config import SUPPORTED_EXTENSIONS, MODEL_PATHS


class LogLevel(Enum):
    """Log levels"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add custom fields
        if hasattr(record, 'custom_fields'):
            log_data.update(record.custom_fields)

        return json.dumps(log_data)


def setup_logging(
        log_dir: Path,
        log_level: LogLevel,
        enable_console: bool = True,
        enable_file: bool = True,
        enable_json: bool = True,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
) -> logging.Logger:
    """
    Setup comprehensive logging system with UTF-8 support.

    Args:
        log_dir: Directory for log files
        log_level: Minimum log level
        enable_console: Enable console output
        enable_file: Enable file logging
        enable_json: Enable JSON structured logging
        max_bytes: Max size per log file
        backup_count: Number of backup files to keep

    Returns:
        Configured root logger
    """

    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(log_level.value)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with UTF-8 encoding
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Set UTF-8 encoding for Windows compatibility
        if sys.platform == 'win32':
            # Reconfigure stdout to use UTF-8
            import io
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer,
                encoding='utf-8',
                errors='replace',  # Replace problematic chars instead of crashing
                line_buffering=True
            )

        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File handler for regular logs with UTF-8
    file_format = None
    if enable_file:
        file_handler = RotatingFileHandler(
            log_dir / 'processing.log',
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'  # Explicit UTF-8 encoding
        )
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    # JSON structured logging with UTF-8
    if enable_json:
        json_handler = RotatingFileHandler(
            log_dir / 'processing.json',
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'  # Explicit UTF-8 encoding
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(StructuredFormatter())
        logger.addHandler(json_handler)

    # Error-only file with UTF-8
    error_handler = RotatingFileHandler(
        log_dir / 'errors.log',
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'  # Explicit UTF-8 encoding
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format if file_format else logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(error_handler)

    logger.info(f"Logging initialized - Level: {log_level}, Dir: {log_dir}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger for specific module"""
    return logging.getLogger(name)


def create_logger_with_context(name: str, **context) -> logging.LoggerAdapter:
    """
    Create logger with context fields.

    Args:
        name: Logger name
        **context: Context fields to include in all log messages

    Returns:
        LoggerAdapter with context
    """
    logger = get_logger(name)

    # Create adapter that properly adds context to extra dict
    class ContextAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            # Merge context into extra
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            if 'custom_fields' not in kwargs['extra']:
                kwargs['extra']['custom_fields'] = {}
            kwargs['extra']['custom_fields'].update(self.extra)
            return msg, kwargs

    return ContextAdapter(logger, context)


def log_exception(logger: logging.Logger, exception: Exception, context: Dict[str, Any] = None):
    """
    Log exception with full context.

    Args:
        logger: Logger instance
        exception: Exception to log
        context: Additional context
    """
    logger.error(
        f"Exception: {type(exception).__name__}: {exception}",
        exc_info=True,
        extra={'custom_fields': context or {}}
    )
    record_metric('exception', 1, exception_type=type(exception).__name__)


@dataclass
class MetricPoint:
    """Single metric measurement"""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Collect and track performance metrics.

    Features:
    - Time-series data
    - Aggregations (min, max, avg, percentiles)
    - Metric persistence
    """

    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector.

        Args:
            retention_hours: How long to keep metrics in memory
        """
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.retention_hours = retention_hours
        self.logger = get_logger(__name__)
        self._lock = threading.Lock()

    def record(self, metric_name: str, value: float, **metadata):
        """
        Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Numeric value
            **metadata: Additional context
        """
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            metadata=metadata
        )
        self.metrics[metric_name].append(point)
        self._cleanup_old_metrics()
        with self._lock:
            self.metrics[metric_name].append(point)
            self._cleanup_old_metrics()

    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)

        for metric_name, points in self.metrics.items():
            # Remove old points from the left
            while points and points[0].timestamp < cutoff:
                points.popleft()

    def get_stats(self, metric_name: str, window_minutes: Optional[int] = None) -> Dict[str, float]:
        """
        Get statistics for a metric.

        Args:
            metric_name: Name of metric
            window_minutes: Optional time window (default: all retained data)

        Returns:
            Dictionary with min, max, avg, count, p50, p95, p99
        """
        points = list(self.metrics.get(metric_name, []))

        if not points:
            return {
                'count': 0,
                'min': 0,
                'max': 0,
                'avg': 0,
                'p50': 0,
                'p95': 0,
                'p99': 0
            }

        # Filter by time window
        if window_minutes:
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            points = [p for p in points if p.timestamp >= cutoff]

        if not points:
            return {'count': 0, 'min': 0, 'max': 0, 'avg': 0, 'p50': 0, 'p95': 0, 'p99': 0}

        values = sorted([p.value for p in points])
        count = len(values)

        def percentile(p: float) -> float:
            """Calculate percentile"""
            k = (count - 1) * (p / 100)
            f = int(k)
            c = f + 1 if f < count - 1 else f
            return values[f] + (k - f) * (values[c] - values[f])

        return {
            'count': count,
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / count,
            'p50': percentile(50),
            'p95': percentile(95),
            'p99': percentile(99)
        }

    def export_metrics(self, output_path: Path):
        """Export all metrics to JSON"""
        data = {
            metric_name: [
                {
                    'timestamp': p.timestamp.isoformat(),
                    'value': p.value,
                    'metadata': p.metadata
                }
                for p in points
            ]
            for metric_name, points in self.metrics.items()
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Exported metrics to {output_path}")


_global_metrics = MetricsCollector()


def record_metric(metric_name: str, value: float, **metadata):
    """Convenience function to record metric"""
    _global_metrics.record(metric_name, value, **metadata)


def get_metric_stats(metric_name: str, window_minutes: Optional[int] = None) -> Dict[
    str, float]:
    """Convenience function to get metric stats"""
    return _global_metrics.get_stats(metric_name, window_minutes)


def export_metrics(output_path: Path):
    """Export metrics from global collector"""
    _global_metrics.export_metrics(output_path)


def track_performance(metric_name: Optional[str] = None):
    """
    Decorator to track function execution time.

    Args:
        metric_name: Custom metric name (default: function name)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}"
            start = time.time()
            logger = get_logger(func.__module__)

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                record_metric(f"{name}.duration", duration, status='success')
                logger.debug(f"{name} completed in {duration:.2f}s")
                return result

            except Exception as e:
                duration = time.time() - start
                record_metric(f"{name}.duration", duration, status='error')
                logger.error(f"{name} failed after {duration:.2f}s: {e}")
                raise

        return wrapper

    return decorator


def track_errors(error_type: str = "general"):
    """
    Decorator to track error rates.

    Args:
        error_type: Category of error
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                record_metric(f"error.{error_type}", 0, status='success')
                return result
            except Exception as e:
                record_metric(f"error.{error_type}", 1, exception=type(e).__name__)
                raise

        return wrapper

    return decorator


@dataclass
class HealthStatus:
    """Health check result"""
    component: str
    healthy: bool
    message: str
    response_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Comprehensive system health checks.

    Checks:
    - Ollama service availability
    - OCR model availability
    - File system access
    - Memory usage
    - Disk space
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self.last_check: Dict[str, HealthStatus] = {}

    def check_ollama(self, model_name: str, timeout: int = 10) -> HealthStatus:
        """
        Check Ollama service and model availability.

        Args:
            model_name: Model to verify
            timeout: Timeout in seconds

        Returns:
            HealthStatus object
        """
        start = time.time()

        try:
            # Check service
            models = ollama.list()
            model_list = models.models if hasattr(models, 'models') else []

            # Check model
            model_names = []
            for m in model_list:
                if hasattr(m, 'model'):
                    model_names.append(m.model)
                elif hasattr(m, 'name'):
                    model_names.append(m.name)

            model_available = any(model_name in name for name in model_names)

            # Test generation
            if model_available:
                try:
                    test_result = ollama.generate(
                        model=model_name,
                        prompt="Test",
                        options={'num_predict': 5}
                    )
                    response_text = test_result.get("response", "").strip()

                    duration_ms = (time.time() - start) * 1000

                    return HealthStatus(
                        component="ollama",
                        healthy=True,
                        message=f"Ollama service operational with model {model_name}",
                        response_time_ms=duration_ms,
                        details={
                            'model': model_name,
                            'available_models': model_names[:5],  # First 5
                            'test_response_length': len(response_text)
                        }
                    )
                except Exception as e:
                    duration_ms = (time.time() - start) * 1000
                    return HealthStatus(
                        component="ollama",
                        healthy=False,
                        message=f"Model test failed: {e}",
                        response_time_ms=duration_ms,
                        details={'error': str(e)}
                    )
            else:
                duration_ms = (time.time() - start) * 1000
                return HealthStatus(
                    component="ollama",
                    healthy=False,
                    message=f"Model {model_name} not found",
                    response_time_ms=duration_ms,
                    details={'available_models': model_names}
                )

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return HealthStatus(
                component="ollama",
                healthy=False,
                message=f"Ollama service unreachable: {e}",
                response_time_ms=duration_ms,
                details={'error': str(e)}
            )

    def check_ocr_models(self) -> HealthStatus:
        """Check if OCR models are available (lightweight - just checks paths)"""
        start = time.time()

        try:

            # Just check if model directories exist, don't load them
            det_dir = Path(MODEL_PATHS['detection folder'])
            rec_dir = Path(MODEL_PATHS['recognition folder'])

            models_exist = det_dir.exists() and rec_dir.exists()

            if models_exist:
                # Check for actual model files
                det_files = list(det_dir.glob('*.pdiparams')) or list(det_dir.glob('*.pdmodel'))
                rec_files = list(rec_dir.glob('*.pdiparams')) or list(rec_dir.glob('*.pdmodel'))

                models_valid = bool(det_files and rec_files)

                duration_ms = (time.time() - start) * 1000

                if models_valid:
                    return HealthStatus(
                        component="ocr",
                        healthy=True,
                        message="OCR models found and ready",
                        response_time_ms=duration_ms,
                        details={
                            'backend': 'PaddleOCR',
                            'detection_model': MODEL_PATHS['detection model'],
                            'recognition_model': MODEL_PATHS['recognition model'],
                            'check_type': 'path_validation'
                        }
                    )
                else:
                    return HealthStatus(
                        component="ocr",
                        healthy=False,
                        message="OCR model files not found in directories",
                        response_time_ms=duration_ms,
                        details={'detection_dir': str(det_dir), 'recognition_dir': str(rec_dir)}
                    )
            else:
                duration_ms = (time.time() - start) * 1000
                return HealthStatus(
                    component="ocr",
                    healthy=False,
                    message="OCR model directories not found (will download on first use)",
                    response_time_ms=duration_ms,
                    details={
                        'detection_dir': str(det_dir),
                        'recognition_dir': str(rec_dir),
                        'note': 'PaddleOCR will auto-download models on first initialization'
                    }
                )

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return HealthStatus(
                component="ocr",
                healthy=False,
                message=f"OCR check failed: {e}",
                response_time_ms=duration_ms,
                details={'error': str(e)}
            )

    def check_filesystem(self, path: Path) -> HealthStatus:
        """
        Check filesystem access and space.

        Args:
            path: Directory to check

        Returns:
            HealthStatus object
        """
        start = time.time()

        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            # Check read/write access
            test_file = path / '.health_check'
            test_file.write_text('test')
            test_file.unlink()

            # Check disk space
            disk = psutil.disk_usage(str(path))
            free_gb = disk.free / (1024 ** 3)
            percent_used = disk.percent

            duration_ms = (time.time() - start) * 1000

            # Warn if less than 5GB free or >90% used
            healthy = free_gb > 5.0 and percent_used < 90

            return HealthStatus(
                component="filesystem",
                healthy=healthy,
                message=f"Filesystem accessible, {free_gb:.1f}GB free ({100 - percent_used:.1f}% available)",
                response_time_ms=duration_ms,
                details={
                    'path': str(path),
                    'free_gb': round(free_gb, 2),
                    'used_percent': round(percent_used, 2),
                    'total_gb': round(disk.total / (1024 ** 3), 2)
                }
            )

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return HealthStatus(
                component="filesystem",
                healthy=False,
                message=f"Filesystem check failed: {e}",
                response_time_ms=duration_ms,
                details={'error': str(e), 'path': str(path)}
            )

    def check_memory(self, warn_threshold_percent: float = 80) -> HealthStatus:
        """
        Check system memory usage.

        Args:
            warn_threshold_percent: Warn if memory usage exceeds this

        Returns:
            HealthStatus object
        """
        start = time.time()

        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)
            used_percent = memory.percent

            duration_ms = (time.time() - start) * 1000

            healthy = used_percent < warn_threshold_percent

            return HealthStatus(
                component="memory",
                healthy=healthy,
                message=f"Memory usage: {used_percent:.1f}%, {available_gb:.1f}GB available",
                response_time_ms=duration_ms,
                details={
                    'available_gb': round(available_gb, 2),
                    'used_percent': round(used_percent, 2),
                    'total_gb': round(memory.total / (1024 ** 3), 2)
                }
            )

        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            return HealthStatus(
                component="memory",
                healthy=False,
                message=f"Memory check failed: {e}",
                response_time_ms=duration_ms,
                details={'error': str(e)}
            )

    def run_all_checks(
            self,
            model_name: str,
            data_path: Path
    ) -> Dict[str, HealthStatus]:
        """
        Run all health checks.

        Args:
            model_name: Ollama model to check
            data_path: Data directory to check

        Returns:
            Dictionary of component -> HealthStatus
        """
        self.logger.info("Running health checks...")

        checks = {
            'ollama': self.check_ollama(model_name),
            'filesystem': self.check_filesystem(data_path),
            'memory': self.check_memory()
        }

        # Cache results
        self.last_check = checks

        # Log results
        for component, status in checks.items():
            if status.healthy:
                self.logger.info(f"[OK] {component}: {status.message}")
            else:
                self.logger.error(f"[FAIL] {component}: {status.message}")

        return checks

    def is_system_healthy(self) -> bool:
        """Check if all components are healthy"""
        return all(status.healthy for status in self.last_check.values())

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of last health check"""
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_healthy': self.is_system_healthy(),
            'components': {
                name: {
                    'healthy': status.healthy,
                    'message': status.message,
                    'response_time_ms': status.response_time_ms
                }
                for name, status in self.last_check.items()
            }
        }


class RateLimiter:
    """
    Token bucket rate limiter.

    Prevents resource exhaustion by limiting operations per time window.
    """

    def __init__(
            self,
            max_operations: int,
            window_seconds: int = 60,
            burst_allowance: float = 1.5
    ):
        """
        Initialize rate limiter.

        Args:
            max_operations: Maximum operations per window
            window_seconds: Time window in seconds
            burst_allowance: Multiplier for burst capacity
        """
        self.max_operations = max_operations
        self.window_seconds = window_seconds
        self.burst_capacity = int(max_operations * burst_allowance)
        self.tokens = self.burst_capacity
        self.last_update = time.time()
        self.logger = get_logger(__name__)

    def _refill_tokens(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_update

        # Calculate tokens to add
        tokens_to_add = (elapsed / self.window_seconds) * self.max_operations
        self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
        self.last_update = now

    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False if rate limit exceeded
        """
        self._refill_tokens()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        else:
            self.logger.warning(f"Rate limit exceeded: {self.tokens:.1f} tokens available, {tokens} requested")
            record_metric('rate_limit.exceeded', 1)
            return False

    def wait_time(self, tokens: int = 1) -> float:
        """
        Calculate wait time until tokens available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds to wait
        """
        self._refill_tokens()

        if self.tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self.tokens
        return (tokens_needed / self.max_operations) * self.window_seconds


class ResourceLimits:
    """
    Enforce resource limits for file processing.

    Prevents:
    - Processing files too large
    - Processing too many pages
    - Excessive memory usage
    """

    def __init__(
            self,
            max_file_size_mb: int = 50,
            max_pdf_pages: int = 100,
            max_total_size_mb: int = 500
    ):
        """
        Initialize resource limits.

        Args:
            max_file_size_mb: Maximum single file size
            max_pdf_pages: Maximum PDF pages to process
            max_total_size_mb: Maximum total size per batch
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_pdf_pages = max_pdf_pages
        self.max_total_size_bytes = max_total_size_mb * 1024 * 1024
        self.current_batch_size = 0
        self.logger = get_logger(__name__)

    def validate_file_size(self, file_path: Path) -> tuple[bool, str]:
        """
        Validate file size.

        Args:
            file_path: Path to file

        Returns:
            Tuple of (valid, error_message)
        """
        try:
            size = file_path.stat().st_size

            if size > self.max_file_size_bytes:
                size_mb = size / (1024 * 1024)
                max_mb = self.max_file_size_bytes / (1024 * 1024)
                return False, f"File too large: {size_mb:.1f}MB (max: {max_mb}MB)"

            return True, ""

        except Exception as e:
            return False, f"Error checking file size: {e}"

    def validate_pdf_pages(self, page_count: int) -> tuple[bool, str]:
        """
        Validate PDF page count.

        Args:
            page_count: Number of pages

        Returns:
            Tuple of (valid, error_message)
        """
        if page_count > self.max_pdf_pages:
            return False, f"Too many pages: {page_count} (max: {self.max_pdf_pages})"
        return True, ""

    def can_add_to_batch(self, file_size: int) -> bool:
        """Check if file can be added to current batch"""
        return (self.current_batch_size + file_size) <= self.max_total_size_bytes

    def add_to_batch(self, file_size: int):
        """Add file to batch tracking"""
        self.current_batch_size += file_size

    def reset_batch(self):
        """Reset batch size counter"""
        self.current_batch_size = 0


class InputValidator:
    """
    Validate all user inputs and file contents.

    Security checks:
    - Path traversal prevention
    - File type validation
    - Size limits
    - Malicious content detection
    """

    ALLOWED_EXTENSIONS = set(
        SUPPORTED_EXTENSIONS['office'] + SUPPORTED_EXTENSIONS['documents'] +
        SUPPORTED_EXTENSIONS['text'] + SUPPORTED_EXTENSIONS['images']
    )

    # ENHANCED: Multiple patterns to catch various traversal techniques
    DANGEROUS_PATTERNS = [
        # Direct traversal
        (r'\.\.[/\\]', 'direct_traversal'),

        # URL-encoded traversal (%2F = /, %5C = \, %2E = .)
        (r'%2[eE]%2[fF]', 'url_encoded_traversal'),
        (r'%2[eE]%5[cC]', 'url_encoded_traversal_backslash'),

        # Double-encoded (%252F = %2F which decodes to /)
        (r'%25[0-9a-fA-F]{2}', 'double_encoded'),

        # Obfuscated: ....// means .. when dots are stripped
        (r'\.{4,}[/\\]', 'obfuscated_traversal'),

        # Unicode/alternate encoding
        (r'\\u002e\\u002e', 'unicode_traversal'),
        (r'\\x2e\\x2e', 'hex_traversal'),

        # Null byte injection (though mostly historical)
        (r'\.\.[\\/]\x00', 'null_byte_traversal'),

        # Windows device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9)
        (r'(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])', 'windows_device_name'),
    ]

    def __init__(self):
        self.logger = get_logger(__name__)
        # Pre-compile all patterns for performance
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), name)
            for pattern, name in self.DANGEROUS_PATTERNS
        ]

    def validate_path(self, path: Path, base_dir: Optional[Path] = None) -> tuple[bool, str]:
        """
        Validate file path for security issues with comprehensive checks.

        Args:
            path: Path to validate
            base_dir: Optional base directory for containment check

        Returns:
            (is_valid, error_message)
        """
        try:
            path = Path(path).resolve()
            path_str = str(path)

            # Check 1: Containment in base directory (prevents escaping)
            if base_dir:
                try:
                    base_dir = Path(base_dir).resolve()
                    path.relative_to(base_dir)
                except ValueError:
                    return False, f"Path outside base directory: {path} not under {base_dir}"

            # Check 2: Multiple traversal techniques
            for compiled_pattern, threat_type in self.compiled_patterns:
                if compiled_pattern.search(path_str):
                    self.logger.warning(
                        f"Security threat detected: {threat_type}",
                        extra={'custom_fields': {
                            'threat_type': threat_type,
                            'path': path_str[:100]
                        }}
                    )
                    return False, f"Invalid path pattern ({threat_type}): {path}"

            # Check 3: Excessive dots or slashes (could indicate obfuscation)
            if re.search(r'\.{5,}', path_str) or re.search(r'[/\\]{3,}', path_str):
                return False, f"Suspicious path pattern detected: {path}"

            # Check 4: Control characters or unusual whitespace
            if any(ord(c) < 32 for c in path_str):
                return False, f"Path contains control characters: {path}"

            return True, ""

        except Exception as e:
            self.logger.debug(f"Path validation exception: {e}")
            return False, f"Path validation error: {e}"

    def validate_file_type(self, file_path: Path) -> tuple[bool, str]:
        """Validate file extension"""
        try:
            ext = file_path.suffix.lower()

            if not ext:
                return False, f"File has no extension: {file_path.name}"

            if ext not in self.ALLOWED_EXTENSIONS:
                return False, f"File type not allowed: {ext}"

            return True, ""
        except Exception as e:
            self.logger.debug(f"File type validation exception: {e}")
            return False, f"File type validation error: {e}"

    def validate_file_exists(self, file_path: Path) -> tuple[bool, str]:
        """Check if file exists and is readable"""
        try:
            if not file_path.exists():
                return False, f"File not found: {file_path}"

            if not file_path.is_file():
                return False, f"Not a file: {file_path}"

            if not os.access(file_path, os.R_OK):
                return False, f"File not readable: {file_path}"

            return True, ""
        except Exception as e:
            self.logger.debug(f"File existence check exception: {e}")
            return False, f"File check error: {e}"

    def validate_file(
            self,
            file_path: Path,
            base_dir: Optional[Path] = None,
            resource_limits: Optional[ResourceLimits] = None
    ) -> tuple[bool, List[str]]:
        """Complete file validation"""
        errors = []

        try:
            file_path = Path(file_path).resolve()
        except Exception as e:
            return False, [f"Cannot resolve path: {e}"]

        # Path validation (critical)
        valid, error = self.validate_path(file_path, base_dir)
        if not valid:
            errors.append(error)
            return False, errors

        # Existence check (critical)
        valid, error = self.validate_file_exists(file_path)
        if not valid:
            errors.append(error)
            return False, errors

        # File type check (warning level)
        valid, error = self.validate_file_type(file_path)
        if not valid:
            errors.append(error)

        # Size check
        if resource_limits:
            valid, error = resource_limits.validate_file_size(file_path)
            if not valid:
                errors.append(error)

        critical_errors = [e for e in errors if any(
            keyword in e.lower()
            for keyword in ['not found', 'traversal', 'readable', 'outside base']
        )]

        return len(critical_errors) == 0, errors


class MemoryMonitor:
    """Monitor and manage memory usage"""

    def __init__(self, warning_threshold_percent: float = 80, critical_threshold_percent: float = 90):
        """
        Initialize memory monitor.

        Args:
            warning_threshold_percent: Log warning at this usage
            critical_threshold_percent: Trigger GC and quality reduction
        """
        self.warning_threshold = warning_threshold_percent
        self.critical_threshold = critical_threshold_percent
        self.logger = get_logger(__name__)

    def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent

    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical"""
        usage = self.get_memory_usage()
        return usage >= self.critical_threshold

    def is_memory_warning(self) -> bool:
        """Check if memory usage is at warning level"""
        usage = self.get_memory_usage()
        return usage >= self.warning_threshold

    def force_cleanup(self):
        """Force garbage collection and log memory stats"""
        before = self.get_memory_usage()
        gc.collect()
        after = self.get_memory_usage()
        freed = before - after

        if freed > 0:
            self.logger.info(f"Memory cleanup: {before:.1f}% -> {after:.1f}% (freed {freed:.1f}%)")

        record_metric('memory.cleanup', freed)

    def check_and_cleanup(self):
        """Check memory and cleanup if needed"""
        if self.is_memory_critical():
            self.logger.warning("Critical memory usage detected, forcing cleanup")
            self.force_cleanup()
        elif self.is_memory_warning():
            self.logger.warning(f"Memory usage at {self.get_memory_usage():.1f}%")


class AdaptiveQualityController:
    """
    Automatically adjust processing quality based on memory pressure.

    Reduces DPI and processing complexity when memory is constrained.
    """

    def __init__(
            self,
            initial_dpi: int = 300,
            min_dpi: int = 150,
            max_dpi: int = 600
    ):
        """
        Initialize quality controller.

        Args:
            initial_dpi: Starting DPI
            min_dpi: Minimum DPI (lowest quality)
            max_dpi: Maximum DPI (highest quality)
        """
        self.current_dpi = initial_dpi
        self.min_dpi = min_dpi
        self.max_dpi = max_dpi
        self.logger = get_logger(__name__)

    def reduce_quality(self) -> int:
        """
        Reduce quality to save memory.

        Returns:
            New DPI value
        """
        old_dpi = self.current_dpi
        self.current_dpi = max(self.min_dpi, int(self.current_dpi * 0.75))

        if old_dpi != self.current_dpi:
            self.logger.warning(f"Reducing quality: {old_dpi} -> {self.current_dpi} DPI")
            record_metric('pdf.quality_reduction', 1, old_dpi=old_dpi, new_dpi=self.current_dpi)

        return self.current_dpi

    def increase_quality(self) -> int:
        """
        Increase quality when memory allows.

        Returns:
            New DPI value
        """
        old_dpi = self.current_dpi
        self.current_dpi = min(self.max_dpi, int(self.current_dpi * 1.25))

        if old_dpi != self.current_dpi:
            self.logger.info(f"Increasing quality: {old_dpi} -> {self.current_dpi} DPI")

        return self.current_dpi


class ProcessingStatus(str, Enum):
    """Status of document processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ExtractionMethod(str, Enum):
    """Method used for text extraction"""

    # OCR-based methods
    OCR = "ocr"
    OCR_FALLBACK = "ocr_fallback"
    IMAGE_OCR = "image_ocr"
    HYBRID_OCR = "hybrid_ocr"  # PDF with both text and images

    # Direct extraction methods
    DIRECT = "direct"
    PDF_TEXT_LAYER = "pdf_text_layer"

    # Office document extraction (no images)
    DOCX_EXTRACTION = "docx_extraction"
    PPTX_EXTRACTION = "pptx_extraction"
    XLSX_EXTRACTION = "xlsx_extraction"
    CSV_EXTRACTION = "csv_extraction"

    # Office document extraction WITH embedded image OCR
    DOCX_WITH_IMAGE_OCR = "docx_extraction_with_image_ocr"
    PPTX_WITH_IMAGE_OCR = "pptx_extraction_with_image_ocr"

    @classmethod
    def from_file_extension(cls, ext: str, has_images: bool = False) -> 'ExtractionMethod':
        """
        Get appropriate extraction method based on file extension.

        Args:
            ext: File extension (e.g., '.docx', '.pdf')
            has_images: Whether the document contains embedded images

        Returns:
            Appropriate ExtractionMethod enum value
        """
        ext = ext.lower()

        if ext == '.docx':
            return cls.DOCX_WITH_IMAGE_OCR if has_images else cls.DOCX_EXTRACTION
        elif ext == '.pptx':
            return cls.PPTX_WITH_IMAGE_OCR if has_images else cls.PPTX_EXTRACTION
        elif ext == '.xlsx':
            return cls.XLSX_EXTRACTION
        elif ext == '.csv':
            return cls.CSV_EXTRACTION
        elif ext == '.pdf':
            return cls.PDF_TEXT_LAYER  # Default, can be overridden to OCR/HYBRID
        elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            return cls.IMAGE_OCR
        elif ext in ['.txt', '.log']:
            return cls.DIRECT
        else:
            return cls.OCR_FALLBACK

    @classmethod
    def is_ocr_method(cls, method: str) -> bool:
        """Check if method involves OCR processing"""
        return method in [
            cls.OCR.value,
            cls.OCR_FALLBACK.value,
            cls.IMAGE_OCR.value,
            cls.HYBRID_OCR.value,
            cls.DOCX_WITH_IMAGE_OCR.value,
            cls.PPTX_WITH_IMAGE_OCR.value
        ]

    @classmethod
    def requires_paddleocr(cls, method: str) -> bool:
        """Check if method requires PaddleOCR initialization"""
        return cls.is_ocr_method(method)


@dataclass
class PatternMatch:
    """Structured container for pattern match results"""
    value: str
    start: int
    end: int
    confidence: float = 1.0
    context: str = ""
    pattern_name: str = ""


class ReinsurancePatterns:
    """
    Compiled regex patterns for extracting reinsurance document fields.

    Each pattern is:
    1. Compiled for performance
    2. Documented with examples
    3. Tagged with confidence levels
    4. Designed to handle common variations
    """

    def __init__(self):
        """Initialize and compile all regex patterns"""

        # Currency components - used in multiple patterns
        self.currency_symbol = r'[\$\€\£\¥\₹\₩\₪\฿\¢]'
        self.currency_code = r'[A-Z]{3}'  # ISO 4217 codes: USD, EUR, GBP, etc.

        # Money pattern: handles both prefix and suffix currency
        # Examples: "$1,000,000", "1000000 USD", "€ 1.000.000,00"
        self.money_pattern = self._compile_money_pattern()

        # Entity patterns
        self.insured_pattern = self._compile_insured_pattern()
        self.cedant_pattern = self._compile_cedant_pattern()
        self.broker_pattern = self._compile_broker_pattern()

        # Financial patterns
        self.tsi_pattern = self._compile_tsi_pattern()
        self.premium_pattern = self._compile_premium_pattern()
        self.retention_pattern = self._compile_retention_pattern()
        self.share_pattern = self._compile_share_pattern()

        # Date and period patterns
        self.period_pattern = self._compile_period_pattern()
        self.date_pattern = self._compile_date_pattern()

        # Location patterns
        self.country_pattern = self._compile_country_pattern()
        self.geographical_limit_pattern = self._compile_geographical_limit_pattern()

        # Risk patterns
        self.perils_pattern = self._compile_perils_pattern()
        self.excess_pattern = self._compile_excess_pattern()

        self.policy_ref_pattern = self._compile_policy_ref_pattern()
        self.reinsurance_type_pattern = self._compile_reinsurance_type_pattern()

    def _compile_policy_ref_pattern(self) -> re.Pattern:
        labels = [r'Policy\s+Number', r'Certificate\s+No', r'Slip\s+Number',
                  r'Reference\s+No', r'FAC[\/\-]']
        pattern = r'(?:' + '|'.join(labels) + r')[:\s\-]*([A-Z0-9\/\-]{5,30})'
        return re.compile(pattern, re.IGNORECASE)

    def _compile_reinsurance_type_pattern(self) -> re.Pattern:
        """
        Compile pattern to extract the reinsurance type.

        Matches examples like:
          - "Reinsurance Type: Quota Share (Proportional)"
          - "Type of Reinsurance: Excess of Loss - Cat XL"
          - "Type: Facultative / Obligatory"
          - "Type of Cover: Proportional (Surplus)"
          - "Reinsurance Type: XL (Excess Loss)"

        Captures:
          - 'type'  : the canonical matched type token (e.g. "Quota Share", "Excess of Loss", "Facultative", "Treaty", "XL", "QS")
          - 'detail': optional trailing text with subtype/notes (up to a line) e.g. "(Proportional)", "Cat XL", "Layer 1"
        """

        types = [
            r'Proportional(?:\s*\(Quota\s*Share\))?',
            r'Quota[-\s]?Share\b',
            r'\bQS\b',
            r'Surplus\b',
            r'Non[-\s]?Proportional\b',
            r'Excess\s+of\s+Loss\b',
            r'Excess\s+Loss\b',
            r'\bXoL\b',
            r'\bXL\b',
            r'\bXS\b',
            r'Stop[-\s]?Loss\b',
            r'Cat(?:astrophe)?\s+Excess\s+of\s+Loss\b',
            r'\bCat\s*XL\b',
            r'Facultative\b',
            r'\bFac(?:\.)?\b',
            r'Facultative\s*\/\s*Obligatory\b',
            r'Treaty\b',
            r'Proportional\s+Treaty\b',
            r'Non[-\s]?Proportional\s+Treaty\b',
            r'Per\s+Risk\b',
            r'Per\s+Occurrence\b',
            r'Occurrence\b',
            r'Aggregate\s+Excess\b',
            r'Aggregate\s+Limit\b',
            r'Layered\b',
            r'Follow\s+the\s+Fortunes\b',
            r'Line\s+Slip\b'
        ]

        # Leading labels that introduce reinsurance type
        leading_labels = [
            r'Reinsurance\s+Type',
            r'Type\s+of\s+Reinsurance',
            r'Type\s+of\s+Cover',
            r'Type\s+of\s+Insurance',
            r'Type\s+of\s+Contract',
            r'Type'
        ]

        # Build the pattern with FIXED parentheses grouping
        # Structure: (label)(separator)(type_group)(optional_detail)
        pattern = (
                r'(?:' + r'|'.join(leading_labels) + r')'  # Match one of the leading labels
                                                     r'[:\-\s]*'  # Optional separators (colon, dash, space)
                                                     r'(?P<type>(?:' + r'|'.join(types) + r'))'  # Capture the main type
                                                                                          r'(?:'  # Optional detail group (non-capturing)
                                                                                          r'[\s\-\:]*'  # Optional separator before details
                                                                                          r'(?P<detail>[\(\[\/\-\w\s\,\.]{0,180}?)'  # Capture detail (up to 180 chars)
                                                                                          r')?'
            # Detail group is optional
        )

        return re.compile(pattern, re.IGNORECASE)

    def _compile_money_pattern(self) -> re.Pattern:
        """
        Compile pattern for monetary values with currency.

        Handles:
        - Currency before: $1,000,000 or USD 1000000
        - Currency after: 1,000,000 USD or 1000000$
        - Various separators: 1,000,000.00 or 1.000.000,00 or 1 000 000.00
        - Optional decimals

        Examples:
            "$1,000,000.00" -> groups: cur_before=$, num=1,000,000.00
            "1000000 USD" -> groups: num=1000000, cur_after=USD
            "€ 1.000.000,50" -> groups: cur_before=€, num=1.000.000,50
        """
        pattern = (
                r'(?P<cur_before>' + self.currency_code + r'|' + self.currency_symbol + r')?\s*'
                                                                                        r'(?P<num>[-+]?\d[\d,\s.]*\.?\d*)'
                                                                                        r'\s*(?P<cur_after>' + self.currency_code + r'|' + self.currency_symbol + r')?'
        )
        return re.compile(pattern, re.IGNORECASE)

    def _compile_insured_pattern(self) -> re.Pattern:
        """
        Pattern for insured party name extraction.

        Captures:
        - Name of the Original Insured: XYZ Corporation
        - Named Insured: ABC Ltd.
        - Insured: Company Name & Co.
        - Assured: Business Name (Pvt) Ltd

        Returns up to 200 characters after the label, typically containing:
        - Company names with &, -, ., ', ", /, ()
        - Abbreviations: Ltd, Inc, Corp, GmbH, SA, Pvt, Co
        """
        labels = [
            r'Name\s+of\s+the\s+Original\s+Insured',
            r'Named\s+Insured',
            r'Insured',
            r'Assured',
            r'Policyholder'
        ]
        pattern = (
                r'(?:' + '|'.join(labels) + r')'
                                            r'[:\s\-]*'
                                            r'([A-Z0-9&\-\.,\(\)\'\"\/\s]{3,200})'
        )
        return re.compile(pattern, re.IGNORECASE)

    def _compile_cedant_pattern(self) -> re.Pattern:
        """
        Pattern for cedant (ceding company) extraction.

        Captures:
        - Cedant: Insurance Company Ltd
        - Ceding Company: XYZ Reinsurance
        - Ceding Insurer: ABC Insurance Co.
        - Insurer: Company Name
        """
        labels = [
            r'Cedant\s+Name',
            r'Cedant',
            r'Ceding\s+Company',
            r'Ceding\s+Insurer',
            r'Ceding\s+Party',
            r'Reinsured',  # Sometimes used interchangeably
        ]
        pattern = (
                r'(?:' + '|'.join(labels) + r')'
                                            r'[:\s\-]*'
                                            r'([A-Z0-9&\-\.,\(\)\'\"\/\s]{3,200})'
        )
        return re.compile(pattern, re.IGNORECASE)

    def _compile_broker_pattern(self) -> re.Pattern:
        """
        Pattern for broker/intermediary extraction.

        Captures:
        - Broker: Marsh LLC
        - Intermediary: Aon Corporation
        - Reinsurance Broker: Willis Towers Watson
        """
        labels = [
            r'Reinsurance\s+Broker',
            r'Broker',
            r'Intermediary',
            r'Placing\s+Broker'
        ]
        pattern = (
                r'(?:' + '|'.join(labels) + r')'
                                            r'[:\s\-]*'
                                            r'([A-Z0-9&\-\.,\(\)\'\"\/\s]{3,200})'
        )
        return re.compile(pattern, re.IGNORECASE)

    def _compile_tsi_pattern(self) -> re.Pattern:
        """
        Pattern for Total Sum Insured (TSI) extraction.

        Captures variations like:
        - Total Sum Insured: USD 10,000,000
        - TSI: 10000000 USD
        - Total (QAR): 5,000,000
        - Sum Insured: $1M
        - Aggregate Limit: 10,000,000.00
        - Limit of Indemnity: 5000000

        Returns context up to 160 characters for amount parsing.
        """
        labels = [
            r'Total\s+Sum\s+Insured\s*\(TSI\)',
            r'Total\s+Sum\s+Insured',
            r'Total\s*\([A-Z]{3}\)',  # e.g., Total (QAR)
            r'SUM\s+INSURED',
            r'TSI',
            r'Total\s+Limit',
            r'Aggregate\s+Limit',
            r'Limit\s+of\s+Indemnity',
            r'Sum\s+Insured'
        ]
        pattern = (
                r'(?:' + '|'.join(labels) + r')'
                                            r'[^\n\r]{0,200}'  # Capture context
                                            r'([^\n\r]{1,160})'  # Capture value area
        )
        return re.compile(pattern, re.IGNORECASE)

    def _compile_premium_pattern(self) -> re.Pattern:
        """
        Pattern for premium amount extraction.

        Captures:
        - Premium: USD 50,000
        - Gross Premium: 50000
        - Net Premium: $50,000.00
        - Reinsurance Premium: 50K
        """
        labels = [
            r'Gross\s+Premium',
            r'Net\s+Premium',
            r'Reinsurance\s+Premium',
            r'Premium\s+Amount',
            r'Premium'
        ]
        pattern = (
                r'(?:' + '|'.join(labels) + r')'
                                            r'[:\s\-]*'
                                            r'([^\n\r]{1,100})'
        )
        return re.compile(pattern, re.IGNORECASE)

    def _compile_retention_pattern(self) -> re.Pattern:
        """
        Pattern for cedant retention percentage.

        Examples matched:
          - "Cedant's retention: 25%"
          - "Retention of Cedant 25 %"
          - "Retention: 25"
        """
        labels = [
            r"Cedant(?:'s)?\s+retention\s+in\s*%?",
            r"Cedant(?:'s)?\s+retention",
            r'Cedant\s+retention\s+in\s*%?',
            r'Cedant\s+retention',
            r'Retention\s+of\s+Cedant',
            r'Retention'
        ]

        # Allow small separators like ":" or "-" then capture 1-3 digits with optional decimal and optional %
        pattern = (
                r'(?:' + '|'.join(labels) + r')'
                                            r'[:\s\-]{0,6}'
                                            r'([0-9]{1,3}(?:\.[0-9]{1,2})?\s*%?)'
        )
        return re.compile(pattern, re.IGNORECASE)

    def _compile_share_pattern(self) -> re.Pattern:
        """
        Pattern for share offered percentage.

        Captures:
        - Share Offered: 75%
        - Offered Share: 75 %
        - Share: 75
        """
        labels = [
            r'Share\s+Offered',
            r'Offered\s+Share',
            r'Participation\s+Share',
            r'Share'
        ]
        pattern = (
                r'(?:' + '|'.join(labels) + r')'
                                            r'[:\s\-]*'
                                            r'([0-9]{1,3}(?:\.[0-9]{1,2})?\s*%?)'
        )
        return re.compile(pattern, re.IGNORECASE)

    def _compile_period_pattern(self) -> re.Pattern:
        """
        Pattern for insurance period extraction.

        Captures various date range formats:
        - From 01/01/2024 to 31/12/2024
        - From January 1, 2024 to December 31, 2024
        - 01-Jan-2024 to 31-Dec-2024
        - Period: 2024-01-01 to 2024-12-31
        """
        labels = [
            r'Period\s+of\s+Insurance',
            r'Insurance\s+Period',
            r'Period\s+of\s+Cover',
            r'Coverage\s+Period',
            r'Period'
        ]
        # Date range patterns
        date_ranges = [
            r'From\s+[^\n\r]{3,60}?\s+to\s+[^\n\r]{3,60}',
            r'\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\s*to\s*\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}',
            r'[A-Za-z]{3,9}\s+\d{1,2},?\s*\d{2,4}\s*to\s*[A-Za-z]{3,9}\s+\d{1,2},?\s*\d{2,4}',
            r'\d{4}-\d{2}-\d{2}\s*to\s*\d{4}-\d{2}-\d{2}'
        ]
        pattern = (
                r'(?:' + '|'.join(labels) + r')'
                                            r'[:\s\-]*'
                                            r'(' + '|'.join(date_ranges) + r')'
        )
        return re.compile(pattern, re.IGNORECASE)

    def _compile_date_pattern(self) -> re.Pattern:
        """
        Generic date pattern for single dates.

        Formats supported:
        - 01/01/2024, 01-01-2024, 01.01.2024
        - January 1, 2024
        - 1 Jan 2024
        - 2024-01-01 (ISO format)
        """
        patterns = [
            r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
            r'[A-Za-z]{3,9}\s+\d{1,2},?\s*\d{2,4}',
            r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}'
        ]
        return re.compile('|'.join(patterns), re.IGNORECASE)

    def _compile_country_pattern(self) -> re.Pattern:
        """
        Pattern for country/risk location extraction.

        Captures:
        - Risk Location
        - Country
        - Territorial Limit
        - State
        """
        labels = [
            r'Risk\s+Location',
            r'Location\s+of\s+Risk',
            r'Country',
            r'Territorial\s+Limit',
            r'Territory',
            r'State\s+of',
            r'Jurisdiction'
        ]
        pattern = (
                r'(?:' + '|'.join(labels) + r')'
                                            r'[:\s\-]*'
                                            r'([A-Za-z\s\-\(\)\/]{2,80})'
        )
        return re.compile(pattern, re.IGNORECASE)

    def _compile_geographical_limit_pattern(self) -> re.Pattern:
        """
        Pattern for geographical scope/limits.

        Captures:
        - Geographical Limit: Worldwide
        - Geographical Scope: Asia Pacific
        - Territory: Global excluding specified
        """
        labels = [
            r'Geographical\s+Limit',
            r'Geographical\s+Scope',
            r'Geographic\s+Coverage',
            r'Territorial\s+Scope'
        ]
        pattern = (
                r'(?:' + '|'.join(labels) + r')'
                                            r'[:\s\-]*'
                                            r'([A-Za-z\s\-\(\)\/,]{2,150})'
        )
        return re.compile(pattern, re.IGNORECASE)

    def _compile_perils_pattern(self) -> re.Pattern:
        """
        Pattern for covered perils/risks.

        Captures:
        - Perils Covered: Fire, Flood, Earthquake
        - Covered Risks: All risks
        - Coverage: Comprehensive
        """
        labels = [
            r'Perils\s+Covered',
            r'Covered\s+Perils',
            r'Risks\s+Covered',
            r'Covered\s+Risks',
            r'Type\s+of\s+Coverage',
            r'Coverage'
        ]
        pattern = (
                r'(?:' + '|'.join(labels) + r')'
                                            r'[:\s\-]*'
                                            r'([A-Za-z\s\-\(\)\/,]{2,200})'
        )
        return re.compile(pattern, re.IGNORECASE)

    def _compile_excess_pattern(self) -> re.Pattern:
        """
        Pattern for excess/deductible amounts.

        Captures:
        - Excess: eg USD 100,000
        - Deductible: eg $50,000
        - Excess/Deductible: 100000
        """
        labels = [
            r'Excess\s*\/\s*Deductible',
            r'Excess',
            r'Deductible',
            r'Retention'
        ]
        pattern = (
                r'(?:' + '|'.join(labels) + r')'
                                            r'[:\s\-]*'
                                            r'([^\n\r]{1,100})'
        )
        return re.compile(pattern, re.IGNORECASE)


class MoneyParser:
    """
    Utility class for parsing monetary values from text.

    Handles various international formats:
    - US: 1,000,000.00
    - EU: 1.000.000,00
    - International: 1 000 000.00
    """

    @staticmethod
    def parse_amount(text: str) -> Optional[float]:
        """
        Parse monetary amount from text string.

        Args:
            text: String containing monetary value

        Returns:
            Parsed float value or None if parsing fails

        Examples:
            "1,000,000.00" -> 1000000.0
            "1.000.000,00" -> 1000000.0
            "1 000 000" -> 1000000.0
        """
        if not text:
            return None

        # Remove currency symbols and codes
        cleaned = re.sub(r'[A-Z]{3}', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'[\$\€\£\¥\₹\₩\₪\฿\¢]', '', cleaned)
        cleaned = cleaned.strip()

        # Detect format: comma as decimal (EU) vs period as decimal (US)
        comma_count = cleaned.count(',')
        period_count = cleaned.count('.')

        try:
            # EU format: 1.000.000,00
            if comma_count == 1 and period_count > 0:
                if cleaned.rfind('.') < cleaned.rfind(','):
                    normalized = cleaned.replace('.', '').replace(',', '.')
                    return float(normalized)

            # US format: 1,000,000.00 or plain 1000000
            normalized = cleaned.replace(',', '').replace(' ', '')
            return float(normalized)

        except ValueError:
            return None

    @staticmethod
    def extract_currency(text: str) -> Optional[str]:
        """
        Extract currency code or symbol from text.

        Prioritizes ISO codes over symbols.

        Args:
            text: String containing currency information

        Returns:
            Currency code (e.g., "USD") or symbol (e.g., "$")
        """

        if not text:
            return None

        # Try ISO codes (any 3 uppercase letters)
        code_match = re.search(r'\b([A-Z]{3})\b', text)
        if code_match:
            return code_match.group(1).upper()

        # Try any currency symbol
        symbol_match = re.search(r'[\$€£\¥₹₩\₪\฿\¢]', text)
        if symbol_match:
            return symbol_match.group(0)

        return None


def extract_with_patterns(
        text: str,
        patterns: ReinsurancePatterns,
        window_size: int = 80
) -> Dict[str, Any]:
    """
    Extract all candidate values using compiled patterns.

    Args:
        text: Source text to extract from
        patterns: Instance of ReinsurancePatterns
        window_size: Context window for currency detection

    Returns:
        Dictionary of extracted candidates with metadata
    """

    candidates = {}
    parser = MoneyParser()

    # Extract insured -
    match = patterns.insured_pattern.search(text)
    if match:
        candidates['insured'] = PatternMatch(
            value=match.group(1).strip(),
            start=match.start(),
            end=match.end(),
            pattern_name='insured'
        )

    # Extract TSI
    match = patterns.tsi_pattern.search(text)
    if match:
        raw = match.group(1).strip()
        amount = parser.parse_amount(raw)
        currency = parser.extract_currency(raw)

        # Return PatternMatch object instead of raw string
        candidates['total_sum_insured'] = PatternMatch(
            value=raw,
            start=match.start(),
            end=match.end(),
            pattern_name='total_sum_insured',
            context=f"amount={amount}, currency={currency}"
        )
        if amount is not None:
            candidates['total_sum_insured_float'] = amount
        if currency:
            candidates['currency'] = currency

    # Extract period
    match = patterns.period_pattern.search(text)
    if match:
        candidates['period_of_insurance'] = PatternMatch(
            value=match.group(1).strip(),
            start=match.start(),
            end=match.end(),
            pattern_name='period'
        )

    # Extract country
    match = patterns.country_pattern.search(text)
    if match:
        candidates['country'] = PatternMatch(
            value=match.group(1).strip(),
            start=match.start(),
            end=match.end(),
            pattern_name='country'
        )

    # Extract retention
    match = patterns.retention_pattern.search(text)
    if match:
        candidates['retention_of_cedant'] = PatternMatch(
            value=match.group(1).strip(),
            start=match.start(),
            end=match.end(),
            pattern_name='retention'
        )

    # Extract share offered
    match = patterns.share_pattern.search(text)
    if match:
        candidates['share_offered'] = PatternMatch(
            value=match.group(1).strip(),
            start=match.start(),
            end=match.end(),
            pattern_name='share'
        )

    # If no TSI found by label, find largest monetary value
    if 'total_sum_insured_and_breakdown' not in candidates:
        all_money = []
        for money_match in patterns.money_pattern.finditer(text):
            amount = parser.parse_amount(money_match.group('num'))
            if amount and amount > 1000:  # Filter small amounts
                all_money.append((amount, money_match))

        if all_money:
            largest = max(all_money, key=lambda x: x[0])
            amount, money_match = largest
            candidates['total_sum_insured_and_breakdown'] = money_match.group(0).strip()
            candidates['total_sum_insured_float'] = amount

            # Try to find currency in context
            start = max(0, money_match.start() - window_size)
            end = min(len(text), money_match.end() + window_size)
            context = text[start:end]
            currency = parser.extract_currency(context)
            if currency:
                candidates['currency'] = currency

    return candidates


class OllamaConfig(BaseModel):
    """Configuration for Ollama LLM"""
    model_name: str = Field(..., min_length=1, description="Ollama model name")
    max_tokens: int = Field(4000, ge=100, le=32000, description="Maximum tokens to generate")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="Sampling temperature")
    timeout: int = Field(300, ge=30, le=3600, description="Timeout in seconds")

    class Config:
        validate_assignment = True


class OCRConfig(BaseModel):
    """Configuration for OCR processing"""
    device: str = Field("cpu", pattern=r"^(cpu|gpu|gpu:[0-9,]+)$")  # Changed regex to pattern
    cpu_threads: int = Field(10, ge=1, le=64)
    enable_mkldnn: bool = Field(True)
    det_limit_side_len: int = Field(960, ge=320, le=4096)
    text_det_thresh: float = Field(0.3, ge=0.0, le=1.0)
    text_det_box_thresh: float = Field(0.6, ge=0.0, le=1.0)
    text_recognition_batch_size: int = Field(6, ge=1, le=64)
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)

    class Config:
        validate_assignment = True


class ProcessingConfig(BaseModel):
    """Configuration for document processing"""
    save_visuals: bool = Field(True)
    pdf_dpi: int = Field(300, ge=72, le=600)
    max_text_length: int = Field(11000, ge=1000, le=100000)
    max_attachment_length: int = Field(8000, ge=1000, le=50000)
    skip_processed: bool = Field(True)
    max_file_size_mb: int = Field(50, ge=1, le=500, description="Maximum file size in MB")
    max_pdf_pages: int = Field(100, ge=1, le=1000, description="Maximum PDF pages to process")

    class Config:
        validate_assignment = True


class AppConfig(BaseModel):
    """Complete application configuration"""
    ollama: OllamaConfig
    ocr: OCRConfig
    processing: ProcessingConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create config from nested dictionary"""
        return cls(
            ollama=OllamaConfig(**config_dict.get('OLLAMA_CONFIG', {})),
            ocr=OCRConfig(**config_dict.get('OCR_CONFIG', {})),
            processing=ProcessingConfig(**config_dict.get('PROCESSING_CONFIG', {}))
        )

    class Config:
        validate_assignment = True


class TimingMetrics(BaseModel):
    """Timing information for processing steps"""
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    def complete(self):
        """Mark timing as complete and calculate duration"""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()


class OCRMetrics(BaseModel):
    """Metrics for OCR processing"""
    total_pages: int = Field(0, ge=0)
    successful_pages: int = Field(0, ge=0)
    failed_pages: int = Field(0, ge=0)
    total_text_blocks: int = Field(0, ge=0)
    average_confidence: float = Field(0.0, ge=0.0, le=1.0)
    processing_time_seconds: float = Field(0.0, ge=0.0)

    @field_validator('successful_pages', 'failed_pages')
    def validate_page_counts(cls, v, values):
        """Ensure page counts don't exceed total"""
        if 'total_pages' in values and v > values['total_pages']:
            raise ValueError(f"Page count {v} exceeds total pages {values['total_pages']}")
        return v


class ProcessingMetrics(BaseModel):
    """Overall processing metrics"""
    total_files: int = Field(0, ge=0)
    successful_files: int = Field(0, ge=0)
    failed_files: int = Field(0, ge=0)
    total_size_bytes: int = Field(0, ge=0)
    ocr_metrics: Optional[OCRMetrics] = None
    timing: TimingMetrics = Field(default_factory=TimingMetrics)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_files == 0:
            return 0.0
        return (self.successful_files / self.total_files) * 100


class DateRange(BaseModel):

    model_config = ConfigDict(validate_assignment=True)

    start_date: Optional[date] = None
    end_date: Optional[date] = None
    raw_text: str = Field(..., min_length=1)

    @model_validator(mode='after')
    def validate_date_order(self) -> 'DateRange':
        """Ensure start date is before end date"""
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError(f"Start date {self.start_date} is after end date {self.end_date}")
        return self

    @field_validator('raw_text')
    @classmethod
    def clean_raw_text(cls, v: str) -> str:
        """Clean and normalize raw text"""
        return v.strip()


class MonetaryAmount(BaseModel):
    """Monetary amount with currency """

    model_config = ConfigDict(validate_assignment=True)

    amount: Optional[float] = Field(None, ge=0)
    currency: Optional[str] = Field(None, description="Currency code or symbol")
    raw_text: str = Field(..., min_length=1)

    @field_validator('amount')
    @classmethod
    def validate_reasonable_amount(cls, v: Optional[float]) -> Optional[float]:
        """Validate amount is within reasonable bounds"""
        if v is not None and v > 1e15:
            raise ValueError(f"Amount {v} exceeds reasonable bounds")
        return v

    @field_validator('raw_text')
    @classmethod
    def clean_raw_text(cls, v: str) -> str:
        """Clean and normalize raw text"""
        return v.strip()


class PercentageValue(BaseModel):
    """Percentage value with validation """

    model_config = ConfigDict(validate_assignment=True)

    value: Optional[float] = Field(None, ge=0, le=100)
    raw_text: str = Field(..., min_length=1)

    @field_validator('value')
    @classmethod
    def round_percentage(cls, v: Optional[float]) -> Optional[float]:
        """Round percentage to 2 decimal places"""
        if v is not None:
            return round(v, 2)
        return v


class ReinsuranceExtraction(BaseModel):
    """
    Complete reinsurance document extraction result.

    This is the main validation model for LLM extraction output.
    All fields default to "TBD" if not found in the source document.
    """

    # Entity Information
    insured: str = Field("TBD", description="Name of the insured party")
    cedant: str = Field("TBD", description="Name of the ceding company")
    broker: str = Field("TBD", description="Name of the broker/intermediary")

    # Risk Description
    occupation_of_insured: str = Field("TBD", description="Occupation or industry of insured")
    main_activities: str = Field("TBD", description="Main business activities")
    perils_covered: str = Field("TBD", description="Covered perils or risks")

    # Location
    geographical_limit: str = Field("TBD", description="Geographical coverage limits")
    situation_of_risk: str = Field("TBD", description="Specific risk location/situation")
    country: str = Field("TBD", description="Country of risk")

    # Financial Terms
    total_sum_insured: str = Field("TBD", description="Total insured sum (raw string)")
    total_sum_insured_float: Optional[float] = Field(None, ge=0, description="Total insured sum (numeric)")
    currency: Optional[str] = Field(None, description="Currency code or symbol")
    premium: str = Field("TBD", description="Premium amount")
    premium_rates: str = Field("TBD", description="Premium rates")

    # Coverage Terms
    period_of_insurance: str = Field("TBD", description="Insurance coverage period")
    excess_deductible: str = Field("TBD", description="Excess or deductible amount")
    retention_of_cedant: str = Field("TBD", description="Cedant's retention percentage")
    share_offered: str = Field("TBD", description="Reinsurance share offered")

    # Risk Assessment
    possible_maximum_loss: str = Field("TBD", description="Possible maximum loss (PML)")
    cat_exposure: str = Field("TBD", description="Catastrophe exposure")
    claims_experience: str = Field("TBD", description="Historical claims experience")

    # Additional Terms
    reinsurance_deductions: str = Field("TBD", description="Reinsurance deductions")
    inward_acceptances: str = Field("TBD", description="Inward acceptances")
    risk_surveyor_report: str = Field("TBD", description="Risk surveyor's report details")

    # ESG and Climate
    climate_change_risk_factors: str = Field("TBD", description="Climate change considerations")
    esg_risk_assessment: str = Field("TBD", description="ESG risk assessment")

    # Metadata
    extraction_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence score")
    extraction_timestamp: datetime = Field(default_factory=datetime.now)

    policy_reference: str = Field("TBD", description="Policy/certificate/slip number")
    email_subject: str = Field("TBD", description="Subject line of email")

    reinsurance_type: str = Field("TBD", description="Facultative | Treaty | etc.")
    coverage_basis: str = Field("TBD", description="Proportional | Non-Proportional | etc.")
    layer_structure: str = Field("TBD", description="For XL treaties")

    valuation_basis: str = Field("TBD", description="Replacement Cost | Market Value | etc.")
    reinstatements: str = Field("TBD", description="Reinstatement terms")
    commission_rate: str = Field("TBD", description="Brokerage commission")
    profit_commission: str = Field("TBD", description="Profit sharing terms")
    aggregate_limit: str = Field("TBD", description="Annual aggregate limit")

    loss_ratio: str = Field("TBD", description="Historical loss ratio")

    co_reinsurers: str = Field("TBD", description="Other reinsurers and shares")
    lead_reinsurer: str = Field("TBD", description="Lead reinsurer")
    warranties_conditions: str = Field("TBD", description="Special conditions")
    territorial_exclusions: str = Field("TBD", description="Excluded territories")

    cyber_risk_exposure: str = Field("TBD", description="Cyber risk details")
    pandemic_exclusions: str = Field("TBD", description="Pandemic terms")

    @field_validator('currency')
    @classmethod
    def normalize_currency(cls, v):
        """Normalize currency to uppercase ISO code if possible"""
        if not v or v == "TBD":
            return v
        if len(v) == 3 and v.isalpha():
            return v.upper()
        return v

    @field_validator('*', mode='before')
    @classmethod
    def empty_string_to_tbd(cls, v):
        """Convert empty strings to TBD"""
        if isinstance(v, str) and not v.strip():
            return "TBD"
        return v

    @model_validator(mode='after')
    def validate_financial_consistency(self) -> 'ReinsuranceExtraction':
        """
        Validate financial fields for consistency.
        """
        logger = logging.getLogger(__name__)

        try:
            # 1. Check TSI string vs numeric consistency
            tsi_str = self.total_sum_insured
            tsi_float = self.total_sum_insured_float

            # If we have numeric TSI but string is TBD, update string
            if tsi_float and tsi_float > 0 and tsi_str == "TBD":
                self.total_sum_insured = str(tsi_float)

            # 2. Check TSI has currency
            if tsi_float and tsi_float > 0:
                if not self.currency or self.currency == "TBD":
                    logger.warning(
                        f"TSI has numeric value ({tsi_float}) but currency is missing or TBD."
                    )

                    # Try to extract currency from TSI string
                    if tsi_str and tsi_str != "TBD":
                        extracted_currency = self._try_extract_currency(tsi_str)
                        if extracted_currency:
                            self.currency = extracted_currency
                            logger.info(f"Extracted currency from TSI string: {extracted_currency}")

            # 3. Validate percentage fields
            self._validate_percentages(logger)

        except Exception as e:
            logger.warning(f"Financial consistency validation warning: {e}")
            # Don't fail validation - just log the issue

        return self

    def _try_extract_currency(self, text: str) -> Optional[str]:
        """Try to extract currency from a text string."""
        import re

        if not text:
            return None

        # Try ISO codes (3 uppercase letters)
        code_match = re.search(r'\b([A-Z]{3})\b', text)
        if code_match:
            return code_match.group(1)

        # Try currency symbols
        symbol_match = re.search(r'[$€£¥₹₩₪฿¢]', text)
        if symbol_match:
            return symbol_match.group(0)

        return None

    def _validate_percentages(self, logger):
        """Validate that retention + share_offered don't exceed 100%"""
        try:
            retention_str = self.retention_of_cedant
            share_str = self.share_offered

            if retention_str == "TBD" or share_str == "TBD":
                return

            retention_val = self._parse_percentage(retention_str)
            share_val = self._parse_percentage(share_str)

            if retention_val is not None and share_val is not None:
                total = retention_val + share_val

                if total > 100:
                    logger.warning(
                        f"Retention ({retention_val}%) + Share ({share_val}%) = {total}% exceeds 100%"
                    )
                elif total < 100:
                    logger.info(
                        f"Retention ({retention_val}%) + Share ({share_val}%) = {total}% (undersubscribed)"
                    )
        except Exception as e:
            logger.debug(f"Could not validate percentages: {e}")

    def _parse_percentage(self, text: str) -> Optional[float]:
        """Parse percentage from string like '25%' or '25' or '0.25'"""
        import re

        if not text or text == "TBD":
            return None

        try:
            cleaned = text.replace('%', '').strip()
            value = float(cleaned)

            if 0 < value < 1:
                value = value * 100

            return value
        except:
            return None

    def to_dict_clean(self) -> Dict[str, Any]:
        """Export as dictionary, excluding None and metadata"""
        return self.model_dump(
            exclude_none=True,
            exclude={'extraction_timestamp'},
            mode='json'
        )

    def get_completeness_score(self) -> float:
        """
        Calculate completeness score (0-100) based on filled fields.
        """
        all_fields = self.model_dump(
            exclude={'extraction_confidence', 'extraction_timestamp'},
            mode='json'
        )

        total_fields = len(all_fields)
        if total_fields == 0:
            return 0.0

        # Count non-empty fields
        filled_fields = sum(
            1 for v in all_fields.values()
            if v not in [None, "TBD", "", 0, 0.0]
        )

        return (filled_fields / total_fields) * 100

    def get_missing_critical_fields(self) -> List[str]:
        """Return list of critical fields that are missing."""
        critical_fields = {
            'insured': self.insured,
            'cedant': self.cedant,
            'total_sum_insured': self.total_sum_insured,
            'period_of_insurance': self.period_of_insurance,
            'currency': self.currency,
            'policy_reference': self.policy_reference,
            'reinsurance_type': self.reinsurance_type
        }

        missing = [
            field for field, value in critical_fields.items()
            if value in [None, "TBD", ""]
        ]

        return missing


class AttachmentMetadata(BaseModel):
    """Metadata for processed attachment"""
    filename: str = Field(..., min_length=1)
    file_path: str = Field(..., min_length=1)
    file_size_bytes: int = Field(..., ge=0)
    file_type: str = Field(..., min_length=1)
    extraction_method: ExtractionMethod
    processing_time_seconds: float = Field(..., ge=0)
    text_length: int = Field(0, ge=0)
    page_number: Optional[int] = Field(None, ge=1)
    ocr_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator('extraction_method', mode='before')
    @classmethod
    def validate_method(cls, v):
        """Accept both enum and string values"""
        if isinstance(v, str):
            # Try to convert string to enum
            try:
                return ExtractionMethod(v)
            except ValueError:
                raise ValueError(f"Invalid extraction method: {v}")
        return v


class EmailMetadata(BaseModel):
    """Email metadata with validation"""
    subject: str = Field("", description="Email subject")
    sender: str = Field("", description="Sender email or name")
    sender_email: Optional[str] = Field(None, description="Sender email address")
    date: Optional[str] = Field(None, description="Email date")
    message_id: Optional[str] = Field(None, description="Email message ID")

    @field_validator('sender_email')
    def validate_email_format(cls, v):
        """Basic email format validation"""
        if v and '@' not in v:
            raise ValueError(f"Invalid email format: {v}")
        return v

    @field_validator('subject')
    def clean_subject(cls, v):
        """Clean email subject"""
        return v.strip()


class ProcessingResult(BaseModel):
    """Complete processing result with validation"""
    status: ProcessingStatus
    email_metadata: EmailMetadata
    attachments: List[AttachmentMetadata] = Field(default_factory=list)
    extraction: Optional[ReinsuranceExtraction] = None
    metrics: ProcessingMetrics = Field(default_factory=ProcessingMetrics)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @property
    def is_successful(self) -> bool:
        """Check if processing was successful"""
        return self.status == ProcessingStatus.COMPLETED and self.extraction is not None

    @property
    def completeness_score(self) -> float:
        """Get extraction completeness score"""
        if self.extraction:
            return self.extraction.get_completeness_score()
        return 0.0

    def add_error(self, error: str):
        """Add error message"""
        self.errors.append(error)
        if self.status == ProcessingStatus.PROCESSING:
            self.status = ProcessingStatus.FAILED

    def add_warning(self, warning: str):
        """Add warning message"""
        self.warnings.append(warning)

    def to_summary(self) -> Dict[str, Any]:
        """Generate processing summary"""
        return {
            'status': self.status.value,
            'success': self.is_successful,
            'completeness_score': self.completeness_score,
            'total_attachments': len(self.attachments),
            'total_errors': len(self.errors),
            'total_warnings': len(self.warnings),
            'processing_time': self.metrics.timing.duration_seconds,
            'missing_critical_fields': (
                self.extraction.get_missing_critical_fields()
                if self.extraction else []
            )
        }


class OllamaException(Exception):
    """
    Base exception for Ollama-related errors.

    Attributes:
        message: Error description
        model_name: Name of the model involved (if applicable)
        details: Additional context about the error
    """

    def __init__(
            self,
            message: str,
            model_name: Optional[str] = None,
            details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.model_name = model_name
        self.details = details or {}

        # Build full error message
        full_message = message
        if model_name:
            full_message = f"[Model: {model_name}] {message}"
        if details:
            full_message += f" | Details: {details}"

        super().__init__(full_message)

        # Log the error
        logger = get_logger(__name__)
        logger.error(
            f"OllamaException: {message}",
            extra={
                'custom_fields': {
                    'model_name': model_name,
                    'error_type': self.__class__.__name__,
                    **details
                }
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'model_name': self.model_name,
            'details': self.details,
            'timestamp': datetime.now().isoformat()
        }


class OllamaModelNotLoadedException(OllamaException):
    """
    Raised when attempting to use a model that is not loaded in Ollama.

    This typically means:
    - Model needs to be pulled: ollama pull <model_name>
    - Model name is misspelled
    - Ollama service has unloaded the model due to inactivity
    """

    def __init__(
            self,
            model_name: str,
            available_models: Optional[List[str]] = None
    ):
        self.available_models = available_models or []

        message = f"Model '{model_name}' is not loaded or available"

        if available_models:
            message += f". Available models: {', '.join(available_models[:5])}"
            if len(available_models) > 5:
                message += f" (and {len(available_models) - 5} more)"

        details = {
            'available_models': available_models,
            'suggestion': f"Run: ollama pull {model_name}"
        }

        super().__init__(message, model_name=model_name, details=details)

    @classmethod
    def from_ollama_list(cls, model_name: str, models_response: Any) -> 'OllamaModelNotLoadedException':
        """
        Create exception from ollama.list() response.

        Args:
            model_name: The model that wasn't found
            models_response: Response from ollama.list()
        """
        try:
            model_list = models_response.models if hasattr(models_response, 'models') else []
            available = [
                m.model if hasattr(m, 'model') else m.name
                for m in model_list
            ]
        except Exception:
            available = []

        return cls(model_name, available_models=available)


class OllamaTimeoutException(OllamaException):
    """
    Raised when Ollama request exceeds timeout duration.

    This can happen when:
    - Model is loading for the first time (can take 30+ seconds)
    - Generation is taking too long (complex prompt, low hardware)
    - System is under heavy load
    - Network issues (if Ollama is remote)
    """

    def __init__(
            self,
            model_name: str,
            timeout_seconds: float,
            operation: str = "generate",
            partial_response: Optional[str] = None
    ):
        self.timeout_seconds = timeout_seconds
        self.operation = operation
        self.partial_response = partial_response

        message = (
            f"Operation '{operation}' timed out after {timeout_seconds:.1f}s"
        )

        if partial_response:
            message += f" (got {len(partial_response)} chars before timeout)"

        details = {
            'timeout_seconds': timeout_seconds,
            'operation': operation,
            'partial_response_length': len(partial_response) if partial_response else 0,
            'suggestion': 'Increase timeout or use a faster model'
        }

        super().__init__(message, model_name=model_name, details=details)

    def has_partial_response(self) -> bool:
        """Check if any response was received before timeout"""
        return bool(self.partial_response)


class OllamaRateLimitException(OllamaException):
    """
    Raised when Ollama rate limit is exceeded.

    This is application-level rate limiting, not from Ollama itself.
    Occurs when RateLimiter prevents excessive requests.
    """

    def __init__(
            self,
            model_name: str,
            wait_time_seconds: float,
            current_tokens: float,
            required_tokens: int = 1
    ):
        self.wait_time_seconds = wait_time_seconds
        self.current_tokens = current_tokens
        self.required_tokens = required_tokens

        message = (
            f"Rate limit exceeded. "
            f"Need {required_tokens} token(s), have {current_tokens:.2f}. "
            f"Wait {wait_time_seconds:.1f}s"
        )

        details = {
            'wait_time_seconds': wait_time_seconds,
            'current_tokens': current_tokens,
            'required_tokens': required_tokens,
            'suggestion': 'Wait or reduce request frequency'
        }

        super().__init__(message, model_name=model_name, details=details)

    def get_wait_time(self) -> float:
        """Get recommended wait time in seconds"""
        return self.wait_time_seconds


class OllamaConnectionException(OllamaException):
    """
    Raised when cannot connect to Ollama service.

    Common causes:
    - Ollama service not running
    - Wrong host/port configuration
    - Firewall blocking connection
    - Service crashed or restarting
    """

    def __init__(
            self,
            message: str = "Cannot connect to Ollama service",
            host: str = "localhost",
            port: int = 11434,
            original_error: Optional[Exception] = None
    ):
        self.host = host
        self.port = port
        self.original_error = original_error

        full_message = f"{message} at {host}:{port}"

        if original_error:
            full_message += f" - {type(original_error).__name__}: {original_error}"

        details = {
            'host': host,
            'port': port,
            'original_error': str(original_error) if original_error else None,
            'suggestion': 'Check if Ollama is running: systemctl status ollama or ollama serve'
        }

        super().__init__(full_message, details=details)


class OllamaResponseException(OllamaException):
    """
    Raised when Ollama returns malformed or unexpected response.

    Common causes:
    - JSON parsing failure
    - Missing required fields in response
    - Unexpected response structure
    - Empty response when content expected
    """

    def __init__(
            self,
            message: str,
            model_name: str,
            response_preview: Optional[str] = None,
            expected_format: Optional[str] = None
    ):
        self.response_preview = response_preview
        self.expected_format = expected_format

        full_message = f"Invalid response: {message}"

        if response_preview:
            preview = response_preview[:200]
            if len(response_preview) > 200:
                preview += "..."
            full_message += f" | Preview: {preview}"

        details = {
            'response_preview': response_preview[:500] if response_preview else None,
            'expected_format': expected_format,
            'suggestion': 'Check model output format or adjust parsing logic'
        }

        super().__init__(full_message, model_name=model_name, details=details)


class OllamaMemoryException(OllamaException):
    """
    Raised when Ollama runs out of memory during generation.

    Common causes:
    - Model too large for available RAM/VRAM
    - Context window too large
    - Multiple models loaded simultaneously
    - System memory pressure
    """

    def __init__(
            self,
            model_name: str,
            memory_available_mb: Optional[float] = None,
            memory_required_mb: Optional[float] = None,
            context_length: Optional[int] = None
    ):
        self.memory_available_mb = memory_available_mb
        self.memory_required_mb = memory_required_mb
        self.context_length = context_length

        message = "Insufficient memory for model"

        if memory_available_mb and memory_required_mb:
            message += (
                f" (need ~{memory_required_mb:.0f}MB, "
                f"have {memory_available_mb:.0f}MB)"
            )

        details = {
            'memory_available_mb': memory_available_mb,
            'memory_required_mb': memory_required_mb,
            'context_length': context_length,
            'suggestion': 'Use smaller model, reduce context, or close other applications'
        }

        super().__init__(message, model_name=model_name, details=details)

    @classmethod
    def from_system_info(cls, model_name: str) -> 'OllamaMemoryException':
        """Create exception with current system memory info"""
        try:
            mem = psutil.virtual_memory()
            available_mb = mem.available / (1024 * 1024)
        except Exception:
            available_mb = None

        return cls(
            model_name=model_name,
            memory_available_mb=available_mb
        )


def retry_on_failure(
        max_attempts: int = 3,
        delay_seconds: float = 5.0,  # Increased from 2.0
        backoff_multiplier: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        logger: Optional[logging.Logger] = None,
        on_retry: Optional[Callable] = None
):
    """
    Retry decorator with exponential backoff and optional cleanup callback.

    Args:
        max_attempts: Maximum retry attempts
        delay_seconds: Initial delay between retries
        backoff_multiplier: Multiply delay by this each retry
        exceptions: Tuple of exceptions to catch
        logger: Optional logger for retry messages
        on_retry: Optional callback to run before each retry (e.g., cleanup)

    Example:
        @retry_on_failure(
            max_attempts=3,
            exceptions=(ConnectionError, TimeoutError),
            on_retry=lambda: gc.collect()
        )
        def work_function():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or get_logger(func.__module__)
            current_delay = delay_seconds
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        _logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts",
                            extra={'last_error': str(e)}
                        )
                        raise

                    _logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )

                    record_metric(
                        f'retry.{func.__name__}',
                        1,
                        attempt=attempt,
                        error_type=type(e).__name__
                    )

                    # Run cleanup callback if provided
                    if on_retry:
                        try:
                            on_retry()
                        except Exception as cleanup_error:
                            _logger.warning(f"Retry cleanup failed: {cleanup_error}")

                    time.sleep(current_delay)
                    current_delay *= backoff_multiplier

            # Should never reach here, but for type safety
            raise last_exception

        return wrapper

    return decorator


class ExtractionCache:
    """
    Manages cached extraction results to avoid reprocessing.

    Cache structure per folder:
    <folder>/
      ├── text_detected_and_recognized/  # OCR outputs
      ├── <model_name>/                  # Model-specific LLM outputs
      │   ├── llm_context.json
      │   ├── master_prompt.json
      │   ├── llm_response_<timestamp>.json
      │   └── extracted_reinsurance_data_<timestamp>.json
      └── extraction_manifest.json       # Tracks all extractions
    """

    def __init__(self, folder_path: Path):
        self.folder_path = Path(folder_path)
        self.manifest_path = self.folder_path / "extraction_manifest.json"
        self.manifest = self._load_manifest()
        self._lock = threading.Lock()

    def _load_manifest(self) -> Dict[str, Any]:
        """Load extraction manifest or create new one"""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'ocr_cache': {},
            'llm_cache': {},
            'created': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }

    @retry_on_failure(
        max_attempts=3,
        delay_seconds=1.0,
        exceptions=(IOError, OSError, PermissionError),
        logger=get_logger(__name__)
    )
    def _save_manifest(self):
        """Thread-safe manifest save with file locking"""
        with self._lock:
            self.manifest['last_updated'] = datetime.now().isoformat()

            # Atomic write: write to temp, then rename
            temp_path = self.manifest_path.with_suffix('.json.tmp')

            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    # File lock (Unix/Linux)
                    if platform.system() != 'Windows' and hasattr(fcntl, 'flock'):
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    # Windows: rely on atomic rename + threading lock (no action needed)

                    json.dump(self.manifest, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk

                # Atomic rename (POSIX-compliant)
                temp_path.replace(self.manifest_path)

            except Exception as e:
                if temp_path.exists():
                    temp_path.unlink()
                raise IOError(f"Failed to save manifest: {e}")

    def _hash_file(self, file_path: Path) -> str:
        """Generate SHA256 hash of file for cache validation"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def has_ocr_cache(self, file_path: Path) -> bool:
        """Check if valid OCR cache exists for file"""
        file_name = file_path.name

        if file_name not in self.manifest['ocr_cache']:
            return False

        cache_info = self.manifest['ocr_cache'][file_name]

        # Validate file hasn't changed
        current_hash = self._hash_file(file_path)
        if cache_info.get('file_hash') != current_hash:
            return False

        # Check if output exists
        ocr_dir = self.folder_path / "text_detected_and_recognized"
        if not ocr_dir.exists():
            return False

        # Check for expected outputs based on file type
        base_name = file_path.stem
        ext = file_path.suffix.lower()

        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']:
            expected = ocr_dir / cache_info.get('latest_timestamp', '') / f"{base_name}_image"
        elif ext == '.pdf':
            expected = ocr_dir / cache_info.get('latest_timestamp', '')
        else:
            expected = ocr_dir / f"{base_name}_{cache_info.get('method', 'extracted')}.json"

        return expected.exists()

    def register_ocr_cache(
            self,
            file_path: Path,
            method: str,
            timestamp: str,
            output_dir: Path
    ) -> None:

        """
        Register OCR cache entry with STRICT path validation
        """
        logger = get_logger(__name__)

        # CRITICAL: Validate all paths are resolved and safe
        try:
            file_path = file_path.resolve(strict=True)
            output_dir = output_dir.resolve(strict=True)
            folder_path = self.folder_path.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            logger.error(f"Path resolution failed: {e}")
            raise ValueError(f"Invalid path during cache registration: {e}")

        # MUST be under folder_path or explicitly allowed
        try:
            relative_output = output_dir.relative_to(folder_path)
            output_dir_str = str(relative_output)
        except ValueError:
            # Absolute path outside folder - REJECT unless explicitly allowed
            logger.error(
                f"Security: Attempted cache registration outside folder boundary",
                extra={'output': str(output_dir), 'folder': str(folder_path)}
            )
            raise ValueError(
                f"Cache output directory must be within email folder: "
                f"{output_dir} not under {folder_path}"
            )

        file_name = file_path.name
        file_hash = self._hash_file(file_path)

        self.manifest['ocr_cache'][file_name] = {
            'file_hash': file_hash,
            'method': method,
            'latest_timestamp': timestamp,
            'output_dir': output_dir_str,
            'cached_at': datetime.now().isoformat()
        }

        logger.debug(f"Registered OCR cache for {file_name}")
        self._save_manifest()

    def has_llm_cache(self, model_name: str) -> Tuple[bool, Optional[Path]]:
        """
        Check if valid LLM extraction exists for model.

        Returns:
            (has_cache, latest_response_path)
        """
        if model_name not in self.manifest['llm_cache']:
            return False, None

        cache_info = self.manifest['llm_cache'][model_name]
        latest_timestamp = cache_info.get('latest_timestamp')

        if not latest_timestamp:
            return False, None

        model_dir = self.folder_path / model_name
        response_file = model_dir / f"llm_response_{latest_timestamp}.json"

        if not response_file.exists():
            return False, None

        return True, response_file

    def register_llm_cache(
            self,
            model_name: str,
            timestamp: str,
            context_hash: str,
            prompt_hash: str
    ):
        """Register LLM cache entry"""
        if model_name not in self.manifest['llm_cache']:
            self.manifest['llm_cache'][model_name] = {
                'extractions': []
            }

        self.manifest['llm_cache'][model_name].update({
            'latest_timestamp': timestamp,
            'context_hash': context_hash,
            'prompt_hash': prompt_hash,
            'cached_at': datetime.now().isoformat()
        })

        # Keep history
        self.manifest['llm_cache'][model_name]['extractions'].append({
            'timestamp': timestamp,
            'context_hash': context_hash,
            'prompt_hash': prompt_hash
        })

        self._save_manifest()

    def _extract_text_from_cached_data(
            self,
            data: Dict[str, Any],
            ext: str
    ) -> List[str]:
        """
        Extract text from cached office document data.
        Mimics the text extraction logic in EmailProcessor.
        """
        text_parts = []

        if ext == '.docx':
            # Extract paragraphs
            for para in data.get('paragraphs', []):
                if isinstance(para, dict):
                    text = para.get('text', '')
                    style = para.get('style', 'Normal')
                    if 'Heading' in style:
                        text = f"\n## {text} ##\n"
                    text_parts.append(text)

            # Extract tables
            for table in data.get('tables', []):
                table_text = f"\n[TABLE {table.get('table_number', '')} - {table.get('rows', 0)}x{table.get('columns', 0)}]\n"
                for row in table.get('data', []):
                    table_text += " | ".join(str(cell) for cell in row) + "\n"
                text_parts.append(table_text)

        elif ext == '.pptx':
            # Extract slides
            for slide in data.get('slides', []):
                slide_text = f"\n[SLIDE {slide.get('slide_number', '')}: {slide.get('title', '')}]\n"
                for text_box in slide.get('text_boxes', []):
                    slide_text += text_box.get('text', '') + "\n"
                for table in slide.get('tables', []):
                    for row in table.get('data', []):
                        slide_text += " | ".join(str(cell) for cell in row) + "\n"
                text_parts.append(slide_text)

        elif ext == '.xlsx':
            # Extract sheets
            for sheet in data.get('sheets', []):
                sheet_text = f"\n[SHEET: {sheet.get('sheet_name', '')}]\n"
                for row in sheet.get('data', [])[:100]:  # Limit rows
                    sheet_text += " | ".join(str(cell) if cell is not None else "" for cell in row) + "\n"
                text_parts.append(sheet_text)

        elif ext == '.csv':
            text_parts.append(f"[CSV FILE - {data.get('metadata', {}).get('rows', 0)} rows]\n")
            for row in data.get('data', [])[:100]:
                text_parts.append(" | ".join(str(cell) if cell is not None else "" for cell in row))

        return text_parts

    def _extract_text_from_ocr_result(self, ocr_data: Dict[str, Any]) -> str:
        """
        Extract text from OCR result JSON.
        Handles PaddleOCR result format.
        """
        text_parts = []

        rec_texts = ocr_data.get('rec_texts', []) or []
        rec_scores = ocr_data.get('rec_scores', []) or []

        if rec_scores and len(rec_scores) == len(rec_texts):
            for text, score in zip(rec_texts, rec_scores):
                try:
                    if float(score) >= 0.7 and str(text).strip():
                        text_parts.append(str(text).strip())
                except Exception:
                    continue
        else:
            # No scores available, take all text
            text_parts = [str(t).strip() for t in rec_texts if str(t).strip()]

        return ' '.join(text_parts)

    def get_cached_ocr_data(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load cached OCR data with corrected path resolution.

        Handles the actual file structure:
        text_detected_and_recognized/
          └── {timestamp}/
              ├── {basename}_docx_extracted.json
              ├── {basename}_pptx_extracted.json
              ├── {basename}_xlsx_extracted.json
              └── {basename}_csv_extracted.json
        """
        logger = get_logger(__name__)

        if not self.has_ocr_cache(file_path):
            return None

        file_name = file_path.name
        cache_info = self.manifest['ocr_cache'][file_name]

        base_name = file_path.stem
        ext = file_path.suffix.lower()
        output_dir_str = cache_info['output_dir']
        timestamp = cache_info['latest_timestamp']  # NOW WE USE IT!
        method = cache_info['method']

        # Resolve output directory
        if Path(output_dir_str).is_absolute():
            base_output_dir = Path(output_dir_str)
        else:
            email_folder = file_path.parent
            base_output_dir = email_folder / output_dir_str

        # CRITICAL: Files are saved in timestamp subdirectory
        timestamped_dir = base_output_dir / timestamp

        try:
            # Load based on file type
            if ext in SUPPORTED_EXTENSIONS['office']:
                # Office files have method-specific suffixes with "_extracted"
                # e.g., "document_docx_extracted.json"
                json_filename = f"{base_name}_{method}_extracted.json"
                json_file = timestamped_dir / json_filename

                if not json_file.exists():
                    # Fallback: try without "_extracted" suffix (old format)
                    json_file_fallback = timestamped_dir / f"{base_name}_{method}.json"
                    if json_file_fallback.exists():
                        json_file = json_file_fallback
                        logger.debug(f"Using fallback cache format: {json_file.name}")
                    else:
                        logger.warning(f"Cached JSON not found: {json_file}")
                        logger.debug(f"Expected: {json_file}")
                        logger.debug(f"Also tried: {json_file_fallback}")
                        return None

                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.debug(f"Loaded cached OCR data: {json_file.name}")

                    text_parts = self._extract_text_from_cached_data(data, ext)

                    return {
                        'text': '\n'.join(text_parts),
                        'method': method,
                        'structured_content': data,
                        'cached': True
                    }

            # For images and PDFs (OCR results)
            elif ext in SUPPORTED_EXTENSIONS['images']:
                # Image OCR saved in: {timestamp}/{basename}_image/ocr_result.json
                img_dir = timestamped_dir / f"{base_name}_image"
                ocr_json = img_dir / "ocr_result.json"

                if not ocr_json.exists():
                    logger.warning(f"Cached image OCR not found: {ocr_json}")
                    return None

                with open(ocr_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.debug(f"Loaded cached image OCR: {ocr_json.name}")

                    # Extract text from OCR result
                    text = self._extract_text_from_ocr_result(data)

                    return {
                        'text': text,
                        'method': 'ocr',
                        'ocr_data': data,
                        'cached': True
                    }

            elif ext in SUPPORTED_EXTENSIONS['documents']:  # PDFs
                # PDF OCR saved in: {timestamp}/{basename}_page{N}_OCR/ocr_result.json
                # Need to aggregate all pages
                all_text = []
                page_num = 1

                while True:
                    page_dir = timestamped_dir / f"{base_name}_page{page_num}_OCR"
                    ocr_json = page_dir / "ocr_result.json"

                    if not ocr_json.exists():
                        break

                    with open(ocr_json, 'r', encoding='utf-8') as f:
                        page_data = json.load(f)
                        page_text = self._extract_text_from_ocr_result(page_data)
                        if page_text:
                            all_text.append(page_text)

                    page_num += 1

                if all_text:
                    logger.debug(f"Loaded cached PDF OCR: {page_num - 1} pages")
                    return {
                        'text': '\n\n'.join(all_text),
                        'method': 'ocr',
                        'pages_processed': page_num - 1,
                        'cached': True
                    }
                else:
                    logger.warning(f"No cached PDF pages found for {base_name}")
                    return None

            return None

        except Exception as e:
            logger.warning(f"Error loading cached OCR data: {e}", exc_info=True)
            return None

    def invalidate_ocr_cache(self, file_path: Path):
        """Invalidate OCR cache for a file"""
        file_name = file_path.name
        if file_name in self.manifest['ocr_cache']:
            del self.manifest['ocr_cache'][file_name]
            self._save_manifest()

    def invalidate_llm_cache(self, model_name: str):
        """Invalidate LLM cache for a model"""
        if model_name in self.manifest['llm_cache']:
            del self.manifest['llm_cache'][model_name]
            self._save_manifest()


class PromptCache:
    """
    Semantic prompt caching system.

    Caches LLM responses based on prompt similarity.
    Useful when processing similar documents.
    Better if we incorporate semantic caching if necessary; sentence transformers
    """

    def __init__(self, cache_dir: Path, similarity_threshold: float = 0.95):
        """
        Initialize prompt cache.

        Args:
            cache_dir: Directory to store cache
            similarity_threshold: Minimum similarity to use cached response (0-1)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index = self.cache_dir / "cache_index.json"
        self.similarity_threshold = similarity_threshold
        self.index = self._load_index()
        self.logger = get_logger(__name__)

    def _load_index(self) -> Dict[str, Any]:
        """Load cache index"""
        if self.cache_index.exists():
            with open(self.cache_index, 'r') as f:
                return json.load(f)
        return {'entries': [], 'total_hits': 0, 'total_misses': 0}

    def _save_index(self):
        """Save cache index"""
        with open(self.cache_index, 'w') as f:
            json.dump(self.index, f, indent=2)

    def _hash_prompt(self, prompt: str) -> str:
        """Generate hash of prompt"""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple Jaccard similarity between prompts.

        For production, consider using sentence transformers for semantic similarity.
        """
        # Tokenize (simple word-based)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def get_cached_response(
            self,
            prompt: str,
            model: str
    ) -> Optional[Dict[str, Any]]:
        """
        Try to get cached response for similar prompt.

        Returns:
            Cached response dict or None
        """
        prompt_hash = self._hash_prompt(prompt)

        # Exact match first
        for entry in self.index['entries']:
            if entry['prompt_hash'] == prompt_hash and entry['model'] == model:
                self.logger.info(f"Cache HIT (exact): {prompt_hash[:8]}...")
                self.index['total_hits'] += 1
                self._save_index()

                cache_file = self.cache_dir / f"{prompt_hash}.json"
                with open(cache_file, 'r') as f:
                    return json.load(f)

        # Similarity-based match
        for entry in self.index['entries']:
            if entry['model'] != model:
                continue

            # Load and compare
            cache_file = self.cache_dir / f"{entry['prompt_hash']}.json"
            if not cache_file.exists():
                continue

            with open(cache_file, 'r') as f:
                cached_data = json.load(f)

            similarity = self._compute_similarity(prompt, cached_data['prompt'])

            if similarity >= self.similarity_threshold:
                self.logger.info(
                    f"Cache HIT (similar: {similarity:.2%}): {entry['prompt_hash'][:8]}..."
                )
                self.index['total_hits'] += 1
                self._save_index()
                return cached_data

        self.logger.debug("Cache MISS")
        self.index['total_misses'] += 1
        self._save_index()
        return None

    def cache_response(
            self,
            prompt: str,
            response: Dict[str, Any],
            model: str
    ):
        """Cache a response"""
        prompt_hash = self._hash_prompt(prompt)

        cache_data = {
            'prompt': prompt,
            'response': response,
            'model': model,
            'cached_at': datetime.now().isoformat()
        }

        cache_file = self.cache_dir / f"{prompt_hash}.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

        # Update index
        entry = {
            'prompt_hash': prompt_hash,
            'model': model,
            'timestamp': datetime.now().isoformat(),
            'prompt_length': len(prompt)
        }

        # Remove old entry if exists
        self.index['entries'] = [
            e for e in self.index['entries']
            if e['prompt_hash'] != prompt_hash
        ]
        self.index['entries'].append(entry)

        # Keep only recent 1000 entries
        if len(self.index['entries']) > 1000:
            self.index['entries'] = sorted(
                self.index['entries'],
                key=lambda x: x['timestamp'],
                reverse=True
            )[:1000]

        self._save_index()
        self.logger.debug(f"Cached response: {prompt_hash[:8]}...")

    def clear_cache(self):
        """Clear all cached responses"""
        for entry in self.index['entries']:
            cache_file = self.cache_dir / f"{entry['prompt_hash']}.json"
            if cache_file.exists():
                cache_file.unlink()

        self.index = {'entries': [], 'total_hits': 0, 'total_misses': 0}
        self._save_index()
        self.logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.index['total_hits'] + self.index['total_misses']
        hit_rate = (
            self.index['total_hits'] / total_requests
            if total_requests > 0 else 0
        )

        return {
            'total_entries': len(self.index['entries']),
            'total_hits': self.index['total_hits'],
            'total_misses': self.index['total_misses'],
            'hit_rate': f"{hit_rate:.1%}",
            'cache_size_mb': sum(
                (self.cache_dir / f"{e['prompt_hash']}.json").stat().st_size
                for e in self.index['entries']
                if (self.cache_dir / f"{e['prompt_hash']}.json").exists()
            ) / (1024 * 1024)
        }
