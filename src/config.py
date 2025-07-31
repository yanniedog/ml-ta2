"""
Enhanced Configuration Management for ML-TA System

This module provides comprehensive configuration management with validation,
environment-specific overrides, and secure secret handling.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Handle optional dependencies gracefully
try:
    from pydantic import BaseModel, Field
    # Handle Pydantic v1 vs v2 compatibility
    try:
        from pydantic import validator
        PYDANTIC_V1 = True
    except ImportError:
        from pydantic import field_validator as validator
        PYDANTIC_V1 = False
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for when pydantic is not available
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
        
        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        def model_dump(self):
            return self.dict()
    
    def Field(**kwargs):
        return None
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    PYDANTIC_AVAILABLE = False
    PYDANTIC_V1 = True


def get_model_dict(model):
    """Get dictionary representation of Pydantic model, compatible with v1 and v2."""
    if hasattr(model, 'model_dump'):
        return model.model_dump()
    elif hasattr(model, 'dict'):
        return model.dict()
    else:
        # Fallback for non-Pydantic objects
        return model.__dict__ if hasattr(model, '__dict__') else {}


try:
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
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
    
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import structlog
    logger = structlog.get_logger(__name__)
    STRUCTLOG_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    STRUCTLOG_AVAILABLE = False
    
    # Create structlog-compatible interface
    class StructlogCompat:
        @staticmethod
        def get_logger(name):
            return logging.getLogger(name)
    
    structlog = StructlogCompat()


class AppConfig(BaseModel):
    """Application configuration model."""
    name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Environment name")
    seed: int = Field(42, description="Random seed for reproducibility")
    debug: bool = Field(False, description="Debug mode flag")


class DataConfig(BaseModel):
    """Data configuration model."""
    symbols: list[str] = Field(..., description="Trading symbols to process")
    intervals: list[str] = Field(..., description="Time intervals for data")
    horizons: list[int] = Field(..., description="Prediction horizons")
    start_date: str = Field(..., description="Data start date")
    end_date: str = Field(..., description="Data end date")
    max_records_per_fetch: int = Field(1000, description="Max records per API call")
    data_retention_days: int = Field(365, description="Data retention period")


class BinanceConfig(BaseModel):
    """Binance API configuration model."""
    base_url: str = Field(..., description="Binance API base URL")
    klines_endpoint: str = Field(..., description="Klines endpoint path")
    limit: int = Field(1000, description="Request limit")
    request_delay_ms: int = Field(200, description="Delay between requests")
    max_retries: int = Field(5, description="Maximum retry attempts")
    backoff_factor: int = Field(2, description="Backoff multiplier")
    timeout_seconds: int = Field(30, description="Request timeout")
    rate_limit_requests_per_minute: int = Field(1200, description="Rate limit")


class PathsConfig(BaseModel):
    """Paths configuration model."""
    data: str = Field("./data", description="Data directory path")
    models: str = Field("./models", description="Models directory path")
    logs: str = Field("./logs", description="Logs directory path")
    artefacts: str = Field("./artefacts", description="Artefacts directory path")
    cache: str = Field("./cache", description="Cache directory path")


class DatabaseConfig(BaseModel):
    """Database configuration model."""
    url: str = Field(..., description="Database connection URL")
    echo: bool = Field(False, description="Echo SQL statements")
    pool_size: int = Field(10, description="Connection pool size")
    max_overflow: int = Field(20, description="Max overflow connections")


class RedisConfig(BaseModel):
    """Redis configuration model."""
    host: str = Field("localhost", description="Redis host")
    port: int = Field(6379, description="Redis port")
    db: int = Field(0, description="Redis database number")
    password: Optional[str] = Field(None, description="Redis password")
    socket_timeout: int = Field(5, description="Socket timeout")


class SecurityConfig(BaseModel):
    """Security configuration model."""
    api_key_header: str = Field("X-API-Key", description="API key header name")
    rate_limit_per_minute: int = Field(60, description="Rate limit per minute")
    max_request_size_mb: int = Field(10, description="Max request size in MB")
    allowed_origins: list[str] = Field(..., description="Allowed CORS origins")
    jwt_secret_key: str = Field(..., description="JWT secret key")
    jwt_algorithm: str = Field("HS256", description="JWT algorithm")
    jwt_expiration_hours: int = Field(24, description="JWT expiration time")


class MonitoringConfig(BaseModel):
    """Monitoring configuration model."""
    metrics_port: int = Field(8000, description="Metrics server port")
    health_check_interval: int = Field(30, description="Health check interval")
    log_level: str = Field("INFO", description="Logging level")
    enable_profiling: bool = Field(False, description="Enable profiling")
    alert_thresholds: Dict[str, Union[int, float]] = Field(..., description="Alert thresholds")


class PerformanceConfig(BaseModel):
    """Performance configuration model."""
    max_memory_gb: int = Field(4, description="Maximum memory usage in GB")
    max_cpu_percent: int = Field(80, description="Maximum CPU usage percentage")
    batch_size: int = Field(1000, description="Processing batch size")
    parallel_jobs: int = Field(-1, description="Number of parallel jobs")
    cache_size_mb: int = Field(512, description="Cache size in MB")
    prediction_timeout_seconds: int = Field(10, description="Prediction timeout")


class MLTAConfig(BaseModel):
    """Main configuration model for ML-TA system."""
    app: AppConfig
    data: DataConfig
    binance: BinanceConfig
    paths: PathsConfig
    database: DatabaseConfig
    redis: RedisConfig
    security: SecurityConfig
    monitoring: MonitoringConfig
    performance: PerformanceConfig
    indicators: Dict[str, Any] = Field(..., description="Technical indicators config")
    model: Dict[str, Any] = Field(..., description="Model configuration")
    features: Dict[str, Any] = Field(..., description="Feature engineering config")
    backtest: Dict[str, Any] = Field(..., description="Backtesting configuration")


class ConfigValidator:
    """Validates configuration parameters and ensures consistency."""
    
    @staticmethod
    def validate_config(config: MLTAConfig) -> bool:
        """Validate the complete configuration."""
        try:
            # Validate date formats
            from datetime import datetime
            datetime.strptime(config.data.start_date, "%Y-%m-%d")
            datetime.strptime(config.data.end_date, "%Y-%m-%d")
            
            # Validate paths exist or can be created
            for path_name, path_value in get_model_dict(config.paths).items():
                path_obj = Path(path_value)
                path_obj.mkdir(parents=True, exist_ok=True)
            
            # Validate performance constraints
            if config.performance.max_memory_gb < 1:
                raise ValueError("max_memory_gb must be at least 1")
            
            if config.performance.max_cpu_percent < 10 or config.performance.max_cpu_percent > 100:
                raise ValueError("max_cpu_percent must be between 10 and 100")
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False


class SecretManager:
    """Manages secure handling of sensitive configuration data."""
    
    def __init__(self, key: Optional[bytes] = None):
        """Initialize secret manager with encryption key."""
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)
    
    def encrypt_secret(self, secret: str) -> str:
        """Encrypt a secret string."""
        return self.cipher.encrypt(secret.encode()).decode()
    
    def decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt an encrypted secret."""
        return self.cipher.decrypt(encrypted_secret.encode()).decode()
    
    def get_secret_from_env(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from environment variable."""
        value = os.getenv(key, default)
        if value and value.startswith("encrypted:"):
            return self.decrypt_secret(value[10:])
        return value


class EnvironmentConfig:
    """Manages environment-specific configuration selection."""
    
    @staticmethod
    def get_environment() -> str:
        """Determine current environment from environment variables."""
        return os.getenv("ML_TA_ENV", "development").lower()
    
    @staticmethod
    def get_config_files(base_path: str = "config") -> list[str]:
        """Get list of configuration files to load."""
        env = EnvironmentConfig.get_environment()
        base_config = os.path.join(base_path, "settings.yaml")
        env_config = os.path.join(base_path, f"{env}.yaml")
        
        files = [base_config]
        if os.path.exists(env_config):
            files.append(env_config)
        
        return files


class ConfigurationAudit:
    """Tracks configuration changes and provides audit trail."""
    
    def __init__(self):
        self.changes = []
    
    def log_change(self, key: str, old_value: Any, new_value: Any, source: str):
        """Log a configuration change."""
        change = {
            "timestamp": logger.info,
            "key": key,
            "old_value": old_value,
            "new_value": new_value,
            "source": source
        }
        self.changes.append(change)
        logger.info(f"Config change: {key} changed from {old_value} to {new_value} (source: {source})")


class ConfigManager:
    """Main configuration manager that loads and validates configuration."""
    
    def __init__(self, config_path: str = "config"):
        """Initialize configuration manager."""
        self.config_path = config_path
        self.secret_manager = SecretManager()
        self.audit = ConfigurationAudit()
        self._config: Optional[MLTAConfig] = None
    
    def load_yaml_file(self, file_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = yaml.safe_load(file)
                logger.info(f"Loaded configuration from {file_path}")
                return content or {}
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {file_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            raise
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries with deep merge."""
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(base_config, override_config)
    
    def substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration values."""
        def substitute_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return self.secret_manager.get_secret_from_env(env_var, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            return value
        
        return substitute_value(config)
    
    def load_config(self) -> MLTAConfig:
        """Load and validate complete configuration."""
        if self._config is not None:
            return self._config
        
        # Get configuration files
        config_files = EnvironmentConfig.get_config_files(self.config_path)
        
        # Load and merge configurations
        merged_config = {}
        for config_file in config_files:
            file_config = self.load_yaml_file(config_file)
            merged_config = self.merge_configs(merged_config, file_config)
        
        # Substitute environment variables
        merged_config = self.substitute_env_vars(merged_config)
        
        # Create and validate configuration model
        try:
            self._config = MLTAConfig(**merged_config)
            
            # Validate configuration
            if not ConfigValidator.validate_config(self._config):
                raise ValueError("Configuration validation failed")
            
            logger.info("Configuration loaded and validated successfully")
            return self._config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get_config(self) -> MLTAConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> MLTAConfig:
        """Reload configuration from files."""
        self._config = None
        return self.load_config()
    
    def update_config(self, updates: Dict[str, Any], source: str = "runtime") -> None:
        """Update configuration at runtime."""
        if self._config is None:
            self.load_config()
        
        # Apply updates and track changes
        config_dict = get_model_dict(self._config)
        for key, value in updates.items():
            old_value = config_dict.get(key)
            config_dict[key] = value
            self.audit.log_change(key, old_value, value, source)
        
        # Recreate configuration object
        self._config = MLTAConfig(**config_dict)
        
        # Validate updated configuration
        if not ConfigValidator.validate_config(self._config):
            raise ValueError("Updated configuration validation failed")


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> MLTAConfig:
    """Get the global configuration instance."""
    return config_manager.get_config()


def reload_config() -> MLTAConfig:
    """Reload the global configuration."""
    return config_manager.reload_config()
