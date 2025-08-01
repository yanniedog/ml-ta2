"""
Core Utilities and Helpers for ML-TA System

This module provides essential utility functions for file operations, data processing,
memory optimization, validation, and system monitoring.
"""

import os
import sys
import hashlib
import random
import pickle
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

# Handle optional dependencies gracefully
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

try:
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    # Fallback encryption (basic base64 - not secure, for development only)
    import base64
    class Fernet:
        def __init__(self, key):
            self.key = key
        
        @staticmethod
        def generate_key():
            return base64.b64encode(b'development_key_not_secure').decode()
        
        def encrypt(self, data):
            return base64.b64encode(data).decode()
        
        def decrypt(self, data):
            return base64.b64decode(data.encode())

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    import logging
    structlog = logging

from src.exceptions import MLTAException, ValidationError, SystemResourceError

logger = structlog.get_logger(__name__)


def ensure_directory(path: Union[str, Path], permissions: int = 0o755) -> Path:
    """
    Ensure directory exists with proper permissions.
    
    Args:
        path: Directory path to create
        permissions: Directory permissions (default: 0o755)
    
    Returns:
        Path object of created directory
    
    Raises:
        SystemResourceError: If directory cannot be created
    """
    try:
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        # Set permissions on Unix-like systems
        if os.name != 'nt':  # Not Windows
            os.chmod(path_obj, permissions)
        
        logger.debug(f"Directory ensured: {path_obj}")
        return path_obj
    
    except Exception as e:
        raise SystemResourceError(
            f"Failed to create directory {path}: {e}",
            resource_type="filesystem"
        )


def save_parquet(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    compression: str = "snappy",
    validate_schema: bool = True
) -> None:
    """
    Save DataFrame to Parquet format with compression and validation.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        compression: Compression algorithm ('snappy', 'gzip', 'brotli')
        validate_schema: Whether to validate schema before saving
    
    Raises:
        ValidationError: If DataFrame validation fails
        SystemResourceError: If file cannot be saved
    """
    try:
        # Validate DataFrame
        if validate_schema:
            if df.empty:
                raise ValidationError("Cannot save empty DataFrame")
            
            # Check for invalid column names
            invalid_chars = set('()[]{}.,;:')
            for col in df.columns:
                if any(char in str(col) for char in invalid_chars):
                    logger.warning(f"Column name contains special characters: {col}")
        
        # Ensure directory exists
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        
        # Optimize data types before saving
        df_optimized = optimize_dataframe_memory(df)
        
        # Convert to PyArrow table for better control
        table = pa.Table.from_pandas(df_optimized)
        
        # Save with compression
        pq.write_table(
            table,
            file_path,
            compression=compression,
            use_dictionary=True,
            row_group_size=10000
        )
        
        file_size = file_path.stat().st_size / 1024 / 1024  # MB
        logger.info(
            f"Saved DataFrame to Parquet",
            file_path=str(file_path),
            rows=len(df),
            columns=len(df.columns),
            size_mb=round(file_size, 2),
            compression=compression
        )
    
    except Exception as e:
        raise SystemResourceError(
            f"Failed to save Parquet file {file_path}: {e}",
            resource_type="filesystem"
        )


def load_parquet(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
    filters: Optional[List[Tuple]] = None,
    validate_data: bool = True
) -> pd.DataFrame:
    """
    Load DataFrame from Parquet format with optional column selection and filtering.
    
    Args:
        file_path: Input file path
        columns: List of columns to load (None for all)
        filters: PyArrow filters for data selection
        validate_data: Whether to validate loaded data
    
    Returns:
        Loaded DataFrame
    
    Raises:
        ValidationError: If file validation fails
        SystemResourceError: If file cannot be loaded
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValidationError(f"Parquet file does not exist: {file_path}")
        
        # Load with PyArrow for better performance
        table = pq.read_table(
            file_path,
            columns=columns,
            filters=filters
        )
        
        df = table.to_pandas()
        
        if validate_data:
            if df.empty:
                logger.warning(f"Loaded empty DataFrame from {file_path}")
            
            # Check for data quality issues
            null_counts = df.isnull().sum()
            if null_counts.sum() > 0:
                logger.info(f"Loaded DataFrame contains {null_counts.sum()} null values")
        
        file_size = file_path.stat().st_size / 1024 / 1024  # MB
        logger.info(
            f"Loaded DataFrame from Parquet",
            file_path=str(file_path),
            rows=len(df),
            columns=len(df.columns),
            size_mb=round(file_size, 2)
        )
        
        return df
    
    except Exception as e:
        raise SystemResourceError(
            f"Failed to load Parquet file {file_path}: {e}",
            resource_type="filesystem"
        )


def set_deterministic_seed(seed: int = 42) -> None:
    """
    Set deterministic seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variables for other libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to set seeds for ML libraries
    try:
        import lightgbm as lgb
        lgb.LGBMModel.set_params(random_state=seed)
    except ImportError:
        pass
    
    try:
        import xgboost as xgb
        # XGBoost uses random_state parameter
        pass
    except ImportError:
        pass
    
    try:
        import catboost as cb
        # CatBoost uses random_seed parameter
        pass
    except ImportError:
        pass
    
    logger.info(f"Set deterministic seed: {seed}")


class MemoryOptimizer:
    """Optimizes DataFrame memory usage through dtype optimization and chunked processing."""
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by converting to optimal dtypes.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Memory-optimized DataFrame
        """
        start_memory = df.memory_usage(deep=True).sum() / 1024**2
        
        df_optimized = df.copy()
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            if col_type != object:
                c_min = df_optimized[col].min()
                c_max = df_optimized[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df_optimized[col] = df_optimized[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df_optimized[col] = df_optimized[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df_optimized[col] = df_optimized[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df_optimized[col] = df_optimized[col].astype(np.int64)
                
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df_optimized[col] = df_optimized[col].astype(np.float32)
                    else:
                        df_optimized[col] = df_optimized[col].astype(np.float64)
        
        end_memory = df_optimized.memory_usage(deep=True).sum() / 1024**2
        memory_reduction = 100 * (start_memory - end_memory) / start_memory
        
        logger.info(
            f"Memory optimization completed",
            start_memory_mb=round(start_memory, 2),
            end_memory_mb=round(end_memory, 2),
            reduction_percent=round(memory_reduction, 2)
        )
        
        return df_optimized
    
    @staticmethod
    def process_in_chunks(
        df: pd.DataFrame,
        chunk_size: int,
        process_func: callable,
        **kwargs
    ) -> pd.DataFrame:
        """
        Process large DataFrame in chunks to manage memory usage.
        
        Args:
            df: Input DataFrame
            chunk_size: Number of rows per chunk
            process_func: Function to apply to each chunk
            **kwargs: Additional arguments for process_func
        
        Returns:
            Processed DataFrame
        """
        chunks = []
        total_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size else 0)
        
        logger.info(f"Processing DataFrame in {total_chunks} chunks of {chunk_size} rows")
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size].copy()
            processed_chunk = process_func(chunk, **kwargs)
            chunks.append(processed_chunk)
            
            if (i // chunk_size + 1) % 10 == 0:
                logger.debug(f"Processed chunk {i // chunk_size + 1}/{total_chunks}")
        
        result = pd.concat(chunks, ignore_index=True)
        logger.info(f"Chunk processing completed: {len(result)} rows")
        
        return result


class TimeUtils:
    """Utilities for time and date operations."""
    
    @staticmethod
    def parse_date(date_str: str, format_str: str = "%Y-%m-%d") -> datetime:
        """Parse date string to datetime object."""
        try:
            return datetime.strptime(date_str, format_str)
        except ValueError as e:
            raise ValidationError(f"Invalid date format {date_str}: {e}")
    
    @staticmethod
    def to_utc_timestamp(dt: datetime) -> int:
        """Convert datetime to UTC timestamp."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    
    @staticmethod
    def from_utc_timestamp(timestamp: int) -> datetime:
        """Convert UTC timestamp to datetime."""
        return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
    
    @staticmethod
    def get_trading_hours_mask(df: pd.DataFrame, datetime_col: str = 'timestamp') -> pd.Series:
        """Get mask for trading hours (assuming 24/7 crypto trading)."""
        # For crypto, trading is 24/7, but we might want to filter weekends for traditional analysis
        dt_series = pd.to_datetime(df[datetime_col])
        return pd.Series(True, index=df.index)  # Always trading for crypto


class FileUtils:
    """Utilities for secure file operations."""
    
    @staticmethod
    def atomic_write(file_path: Union[str, Path], data: Any, mode: str = 'wb') -> None:
        """
        Atomic file write operation to prevent corruption.
        
        Args:
            file_path: Target file path
            data: Data to write
            mode: File open mode
        """
        file_path = Path(file_path)
        temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
        
        try:
            ensure_directory(file_path.parent)
            
            with open(temp_path, mode) as f:
                if isinstance(data, (str, bytes)):
                    f.write(data)
                else:
                    # Assume it's a serializable object
                    if mode == 'wb':
                        joblib.dump(data, f)
                    else:
                        f.write(str(data))
            
            # Atomic move
            shutil.move(str(temp_path), str(file_path))
            logger.debug(f"Atomic write completed: {file_path}")
        
        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                temp_path.unlink()
            raise SystemResourceError(f"Atomic write failed for {file_path}: {e}")
    
    @staticmethod
    def safe_delete(file_path: Union[str, Path], max_retries: int = 3) -> bool:
        """
        Safely delete file with retries.
        
        Args:
            file_path: File path to delete
            max_retries: Maximum retry attempts
        
        Returns:
            True if deletion successful, False otherwise
        """
        file_path = Path(file_path)
        
        for attempt in range(max_retries):
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"File deleted: {file_path}")
                return True
            
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to delete file {file_path} after {max_retries} attempts: {e}")
                    return False
                else:
                    logger.warning(f"Delete attempt {attempt + 1} failed for {file_path}: {e}")
                    time.sleep(0.1 * (attempt + 1))
        
        return False
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path], algorithm: str = 'sha256') -> str:
        """
        Calculate file hash for integrity checking.
        
        Args:
            file_path: File path
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256', 'sha512')
        
        Returns:
            Hex digest of file hash
        """
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()


class ValidationUtils:
    """Utilities for data validation and constraint checking."""
    
    @staticmethod
    def validate_dataframe_schema(
        df: pd.DataFrame,
        required_columns: List[str],
        column_types: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Validate DataFrame schema against requirements.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            column_types: Expected column types
        
        Returns:
            True if validation passes
        
        Raises:
            ValidationError: If validation fails
        """
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
        
        # Check column types if specified
        if column_types:
            for col, expected_type in column_types.items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if not actual_type.startswith(expected_type):
                        raise ValidationError(
                            f"Column {col} has type {actual_type}, expected {expected_type}"
                        )
        
        return True
    
    @staticmethod
    def validate_numeric_range(
        series: pd.Series,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_nan: bool = True
    ) -> bool:
        """
        Validate numeric series is within specified range.
        
        Args:
            series: Numeric series to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_nan: Whether NaN values are allowed
        
        Returns:
            True if validation passes
        
        Raises:
            ValidationError: If validation fails
        """
        if not allow_nan and series.isna().any():
            raise ValidationError(f"Series contains NaN values: {series.isna().sum()}")
        
        if min_value is not None:
            below_min = series < min_value
            if below_min.any():
                raise ValidationError(f"Series contains values below minimum {min_value}")
        
        if max_value is not None:
            above_max = series > max_value
            if above_max.any():
                raise ValidationError(f"Series contains values above maximum {max_value}")
        
        return True
    
    @staticmethod
    def validate_no_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> bool:
        """
        Validate DataFrame has no duplicate rows.
        
        Args:
            df: DataFrame to validate
            subset: Columns to check for duplicates
        
        Returns:
            True if validation passes
        
        Raises:
            ValidationError: If duplicates found
        """
        duplicates = df.duplicated(subset=subset)
        if duplicates.any():
            raise ValidationError(f"DataFrame contains {duplicates.sum()} duplicate rows")
        
        return True


class CryptoUtils:
    """Utilities for cryptographic operations."""
    
    @staticmethod
    def generate_key() -> bytes:
        """Generate encryption key."""
        return Fernet.generate_key()
    
    @staticmethod
    def encrypt_data(data: str, key: bytes) -> str:
        """Encrypt string data."""
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data.encode())
        return encrypted_data.decode()
    
    @staticmethod
    def decrypt_data(encrypted_data: str, key: bytes) -> str:
        """Decrypt string data."""
        fernet = Fernet(key)
        decrypted_data = fernet.decrypt(encrypted_data.encode())
        return decrypted_data.decode()
    
    @staticmethod
    def hash_string(data: str, algorithm: str = 'sha256') -> str:
        """Hash string data."""
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data.encode())
        return hash_obj.hexdigest()
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate random token."""
        return hashlib.sha256(os.urandom(length)).hexdigest()


class SystemUtils:
    """Utilities for system monitoring and resource management."""
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            'system_total_gb': round(memory.total / 1024**3, 2),
            'system_available_gb': round(memory.available / 1024**3, 2),
            'system_used_percent': memory.percent,
            'process_rss_mb': round(process_memory.rss / 1024**2, 2),
            'process_vms_mb': round(process_memory.vms / 1024**2, 2)
        }
    
    @staticmethod
    def get_cpu_usage() -> Dict[str, float]:
        """Get current CPU usage statistics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
    
    @staticmethod
    def get_disk_usage(path: str = '/') -> Dict[str, float]:
        """Get disk usage statistics."""
        if os.name == 'nt':  # Windows
            path = 'C:\\'
        
        disk = psutil.disk_usage(path)
        
        return {
            'total_gb': round(disk.total / 1024**3, 2),
            'used_gb': round(disk.used / 1024**3, 2),
            'free_gb': round(disk.free / 1024**3, 2),
            'used_percent': round((disk.used / disk.total) * 100, 2)
        }
    
    @staticmethod
    def check_resource_limits(config: Dict[str, Any]) -> Dict[str, bool]:
        """Check if system resources are within configured limits."""
        memory_stats = SystemUtils.get_memory_usage()
        cpu_stats = SystemUtils.get_cpu_usage()
        disk_stats = SystemUtils.get_disk_usage()
        
        checks = {
            'memory_ok': memory_stats['system_used_percent'] < config.get('max_memory_percent', 85),
            'cpu_ok': cpu_stats['cpu_percent'] < config.get('max_cpu_percent', 80),
            'disk_ok': disk_stats['used_percent'] < config.get('max_disk_percent', 90)
        }
        
        return checks


# Convenience functions using the optimized implementation
def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage."""
    return MemoryOptimizer.optimize_dataframe_memory(df)


def process_in_chunks(df: pd.DataFrame, chunk_size: int, process_func: callable, **kwargs) -> pd.DataFrame:
    """Process DataFrame in chunks."""
    return MemoryOptimizer.process_in_chunks(df, chunk_size, process_func, **kwargs)
