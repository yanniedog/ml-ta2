"""
ML-TA: Machine Learning Technical Analysis System

A production-grade cryptocurrency trading analysis platform that combines
technical analysis with machine learning for predictive modeling.
"""

__version__ = "2.0.0"
__author__ = "ML-TA Development Team"
__email__ = "dev@ml-ta.com"
__description__ = "Machine Learning Technical Analysis System"

# Core imports for easy access
from .config import get_config, reload_config
from .logging_config import get_logger, setup_logging
from .exceptions import MLTAException, ErrorSeverity, ErrorCategory
from .utils import (
    ensure_directory,
    save_parquet,
    load_parquet,
    set_deterministic_seed,
    optimize_dataframe_memory
)

# Initialize logging on import
setup_logging()

__all__ = [
    # Configuration
    'get_config',
    'reload_config',
    
    # Logging
    'get_logger',
    'setup_logging',
    
    # Exceptions
    'MLTAException',
    'ErrorSeverity',
    'ErrorCategory',
    
    # Utilities
    'ensure_directory',
    'save_parquet',
    'load_parquet',
    'set_deterministic_seed',
    'optimize_dataframe_memory',
    
    # Metadata
    '__version__',
    '__author__',
    '__email__',
    '__description__'
]
