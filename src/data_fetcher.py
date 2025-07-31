"""
Robust Data Acquisition for ML-TA System

This module provides comprehensive data fetching capabilities with error handling,
rate limiting, data validation, and caching for cryptocurrency market data.
"""

import time
import asyncio
import aiohttp
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from pathlib import Path
import hashlib
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import structlog

from .config import get_config
from .exceptions import (
    DataFetchError, NetworkError, ValidationError, 
    handle_errors, RetryConfig, ErrorContext
)
from .utils import ensure_directory, save_parquet, TimeUtils
from .logging_config import get_logger

logger = get_logger("data_fetcher").get_logger()


class RequestManager:
    """Manages HTTP requests with rate limiting and concurrent request handling."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize request manager with configuration."""
        self.config = config
        self.logger = logger.bind(component="request_manager")
        
        # Rate limiting
        self.request_delay = config.get('request_delay_ms', 200) / 1000  # Convert to seconds
        self.rate_limit = config.get('rate_limit_requests_per_minute', 1200)
        self.request_times = []
        
        # Request configuration
        self.timeout = config.get('timeout_seconds', 30)
        self.max_retries = config.get('max_retries', 5)
        self.backoff_factor = config.get('backoff_factor', 2)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ML-TA/2.0.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Check if we're at the rate limit
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (current_time - self.request_times[0])
            if sleep_time > 0:
                self.logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        # Add current request time
        self.request_times.append(current_time)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, NetworkError))
    )
    def make_request(
        self,
        url: str,
        method: str = 'GET',
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request with rate limiting and error handling.
        
        Args:
            url: Request URL
            method: HTTP method
            params: Query parameters
            data: Request body data
            headers: Additional headers
        
        Returns:
            Response data as dictionary
        
        Raises:
            NetworkError: If request fails
            DataFetchError: If response is invalid
        """
        self._check_rate_limit()
        
        try:
            # Prepare request
            request_headers = self.session.headers.copy()
            if headers:
                request_headers.update(headers)
            
            # Add delay between requests
            if self.request_delay > 0:
                time.sleep(self.request_delay)
            
            self.logger.debug(f"Making {method} request to {url}", params=params)
            
            # Make request
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=request_headers,
                timeout=self.timeout
            )
            
            # Check response status
            response.raise_for_status()
            
            # Parse JSON response
            try:
                data = response.json()
                self.logger.debug(f"Request successful: {url}")
                return data
            
            except json.JSONDecodeError as e:
                raise DataFetchError(
                    f"Invalid JSON response from {url}: {e}",
                    source="response_parsing"
                )
        
        except requests.RequestException as e:
            self.logger.error(f"Request failed: {url}", error=str(e))
            raise NetworkError(f"Request failed for {url}: {e}", url=url)
    
    def close(self) -> None:
        """Close the session."""
        self.session.close()


class DataValidator:
    """Validates fetched data for completeness, consistency, and quality."""
    
    def __init__(self):
        """Initialize data validator."""
        self.logger = logger.bind(component="data_validator")
    
    def validate_kline_data(self, data: List[List], symbol: str, interval: str) -> bool:
        """
        Validate kline/candlestick data from API response.
        
        Args:
            data: Raw kline data from API
            symbol: Trading symbol
            interval: Time interval
        
        Returns:
            True if validation passes
        
        Raises:
            ValidationError: If validation fails
        """
        if not data:
            raise ValidationError(f"Empty kline data for {symbol} {interval}")
        
        if not isinstance(data, list):
            raise ValidationError(f"Kline data must be a list, got {type(data)}")
        
        # Check each kline record
        for i, kline in enumerate(data):
            if not isinstance(kline, list) or len(kline) < 11:
                raise ValidationError(
                    f"Invalid kline format at index {i}: expected list with 11+ elements"
                )
            
            try:
                # Validate timestamp
                timestamp = int(kline[0])
                if timestamp <= 0:
                    raise ValidationError(f"Invalid timestamp at index {i}: {timestamp}")
                
                # Validate OHLCV values
                open_price = float(kline[1])
                high_price = float(kline[2])
                low_price = float(kline[3])
                close_price = float(kline[4])
                volume = float(kline[5])
                
                # Check price relationships
                if not (low_price <= open_price <= high_price):
                    raise ValidationError(f"Invalid OHLC relationship at index {i}: L={low_price}, O={open_price}, H={high_price}")
                
                if not (low_price <= close_price <= high_price):
                    raise ValidationError(f"Invalid OHLC relationship at index {i}: L={low_price}, C={close_price}, H={high_price}")
                
                # Check for negative values
                if any(val < 0 for val in [open_price, high_price, low_price, close_price, volume]):
                    raise ValidationError(f"Negative values found at index {i}")
                
                # Check for zero prices (volume can be zero)
                if any(val <= 0 for val in [open_price, high_price, low_price, close_price]):
                    raise ValidationError(f"Zero or negative prices at index {i}")
            
            except (ValueError, TypeError) as e:
                raise ValidationError(f"Invalid numeric data at index {i}: {e}")
        
        self.logger.debug(f"Validation passed for {len(data)} klines", symbol=symbol, interval=interval)
        return True
    
    def validate_dataframe(self, df: pd.DataFrame, symbol: str, interval: str) -> bool:
        """
        Validate processed DataFrame.
        
        Args:
            df: Processed DataFrame
            symbol: Trading symbol
            interval: Time interval
        
        Returns:
            True if validation passes
        
        Raises:
            ValidationError: If validation fails
        """
        if df.empty:
            raise ValidationError(f"Empty DataFrame for {symbol} {interval}")
        
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
        
        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            raise ValidationError(f"Null values found: {null_counts.to_dict()}")
        
        # Check data types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValidationError(f"Column {col} must be numeric, got {df[col].dtype}")
        
        # Check OHLC relationships
        invalid_ohlc = (
            (df['low'] > df['open']) |
            (df['low'] > df['high']) |
            (df['low'] > df['close']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['open'] <= 0) |
            (df['high'] <= 0) |
            (df['low'] <= 0) |
            (df['close'] <= 0)
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            raise ValidationError(f"Found {invalid_count} records with invalid OHLC relationships")
        
        # Check for duplicate timestamps
        duplicate_timestamps = df['timestamp'].duplicated().sum()
        if duplicate_timestamps > 0:
            raise ValidationError(f"Found {duplicate_timestamps} duplicate timestamps")
        
        # Check timestamp ordering
        if not df['timestamp'].is_monotonic_increasing:
            raise ValidationError("Timestamps are not in ascending order")
        
        self.logger.debug(f"DataFrame validation passed", 
                         symbol=symbol, 
                         interval=interval,
                         records=len(df))
        return True


class CacheManager:
    """Manages data caching with TTL and invalidation strategies."""
    
    def __init__(self, cache_dir: str = "cache", ttl_hours: int = 1):
        """Initialize cache manager."""
        self.cache_dir = Path(cache_dir)
        self.ttl_hours = ttl_hours
        ensure_directory(self.cache_dir)
        self.logger = logger.bind(component="cache_manager")
    
    def _get_cache_key(self, symbol: str, interval: str, start_time: int, end_time: int) -> str:
        """Generate cache key for data request."""
        key_data = f"{symbol}_{interval}_{start_time}_{end_time}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.parquet"
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file is still valid based on TTL."""
        if not cache_path.exists():
            return False
        
        file_age = time.time() - cache_path.stat().st_mtime
        return file_age < (self.ttl_hours * 3600)
    
    def get_cached_data(
        self,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data if available and valid.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            start_time: Start timestamp
            end_time: End timestamp
        
        Returns:
            Cached DataFrame or None if not available
        """
        cache_key = self._get_cache_key(symbol, interval, start_time, end_time)
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                self.logger.debug(f"Cache hit for {symbol} {interval}", cache_key=cache_key)
                return df
            except Exception as e:
                self.logger.warning(f"Failed to read cache file: {e}", cache_path=str(cache_path))
                # Remove corrupted cache file
                cache_path.unlink(missing_ok=True)
        
        return None
    
    def cache_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        start_time: int,
        end_time: int
    ) -> None:
        """
        Cache DataFrame data.
        
        Args:
            df: DataFrame to cache
            symbol: Trading symbol
            interval: Time interval
            start_time: Start timestamp
            end_time: End timestamp
        """
        cache_key = self._get_cache_key(symbol, interval, start_time, end_time)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            df.to_parquet(cache_path, compression='snappy')
            self.logger.debug(f"Data cached for {symbol} {interval}", 
                            cache_key=cache_key,
                            records=len(df))
        except Exception as e:
            self.logger.error(f"Failed to cache data: {e}", cache_path=str(cache_path))
    
    def clear_cache(self, older_than_hours: Optional[int] = None) -> int:
        """
        Clear cache files.
        
        Args:
            older_than_hours: Only clear files older than this many hours
        
        Returns:
            Number of files cleared
        """
        cleared_count = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.parquet"):
            should_clear = True
            
            if older_than_hours is not None:
                file_age = current_time - cache_file.stat().st_mtime
                should_clear = file_age > (older_than_hours * 3600)
            
            if should_clear:
                try:
                    cache_file.unlink()
                    cleared_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to clear cache file: {e}", file=str(cache_file))
        
        self.logger.info(f"Cleared {cleared_count} cache files")
        return cleared_count


class BinanceDataFetcher:
    """Fetches cryptocurrency market data from Binance API with comprehensive error handling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Binance data fetcher."""
        self.config = config or get_config().binance.dict()
        self.logger = logger.bind(component="binance_fetcher")
        
        # Initialize components
        self.request_manager = RequestManager(self.config)
        self.validator = DataValidator()
        self.cache_manager = CacheManager()
        
        # API configuration
        self.base_url = self.config['base_url']
        self.klines_endpoint = self.config['klines_endpoint']
        self.limit = self.config['limit']
    
    def _parse_kline_data(self, raw_data: List[List], symbol: str) -> pd.DataFrame:
        """
        Parse raw kline data into DataFrame.
        
        Args:
            raw_data: Raw kline data from API
            symbol: Trading symbol
        
        Returns:
            Parsed DataFrame with OHLCV data
        """
        if not raw_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(raw_data, columns=columns)
        
        # Convert data types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                          'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Select main columns
        main_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        df = df[main_columns].copy()
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    @handle_errors(
        retry_config=RetryConfig(max_attempts=3, base_delay=1.0),
        context_operation="fetch_klines",
        context_component="binance_fetcher"
    )
    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch kline/candlestick data from Binance API.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1h', '1d')
            start_time: Start time for data
            end_time: End time for data
            limit: Maximum number of records
        
        Returns:
            DataFrame with OHLCV data
        
        Raises:
            DataFetchError: If data fetching fails
            ValidationError: If data validation fails
        """
        # Prepare parameters
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit or self.limit
        }
        
        if start_time:
            params['startTime'] = TimeUtils.to_utc_timestamp(start_time)
        
        if end_time:
            params['endTime'] = TimeUtils.to_utc_timestamp(end_time)
        
        # Check cache first
        if start_time and end_time:
            cached_data = self.cache_manager.get_cached_data(
                symbol, interval,
                params.get('startTime', 0),
                params.get('endTime', 0)
            )
            if cached_data is not None:
                return cached_data
        
        try:
            # Make API request
            url = f"{self.base_url}{self.klines_endpoint}"
            self.logger.info(f"Fetching klines for {symbol} {interval}", params=params)
            
            response_data = self.request_manager.make_request(url, params=params)
            
            # Validate raw data
            self.validator.validate_kline_data(response_data, symbol, interval)
            
            # Parse data
            df = self._parse_kline_data(response_data, symbol)
            
            # Validate DataFrame
            if not df.empty:
                self.validator.validate_dataframe(df, symbol, interval)
            
            # Cache data
            if start_time and end_time and not df.empty:
                self.cache_manager.cache_data(
                    df, symbol, interval,
                    params.get('startTime', 0),
                    params.get('endTime', 0)
                )
            
            self.logger.info(f"Successfully fetched {len(df)} klines for {symbol} {interval}")
            return df
        
        except Exception as e:
            error_context = ErrorContext(
                operation="fetch_klines",
                component="binance_fetcher",
                additional_data={'symbol': symbol, 'interval': interval}
            )
            
            if isinstance(e, (ValidationError, NetworkError)):
                raise
            else:
                raise DataFetchError(
                    f"Failed to fetch klines for {symbol} {interval}: {e}",
                    source="binance_api",
                    context=error_context
                )
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            interval: Time interval
            start_time: Start time for data
            end_time: End time for data
        
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self.fetch_klines(symbol, interval, start_time, end_time)
                results[symbol] = df
                self.logger.info(f"Fetched {len(df)} records for {symbol}")
            
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}: {e}")
                results[symbol] = pd.DataFrame()  # Empty DataFrame for failed fetches
        
        return results
    
    def get_latest_price(self, symbol: str) -> Dict[str, float]:
        """
        Get latest price for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Dictionary with latest price information
        """
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {'symbol': symbol.upper()}
            
            response_data = self.request_manager.make_request(url, params=params)
            
            return {
                'symbol': response_data['symbol'],
                'price': float(response_data['price']),
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            raise DataFetchError(f"Failed to get latest price for {symbol}: {e}")
    
    def close(self) -> None:
        """Close connections and cleanup resources."""
        self.request_manager.close()
        self.logger.info("Data fetcher closed")


# Factory function for easy instantiation
def create_data_fetcher(source: str = "binance", config: Optional[Dict[str, Any]] = None) -> BinanceDataFetcher:
    """
    Create data fetcher instance.
    
    Args:
        source: Data source ('binance' currently supported)
        config: Optional configuration override
    
    Returns:
        Data fetcher instance
    
    Raises:
        ValueError: If unsupported source specified
    """
    if source.lower() == "binance":
        return BinanceDataFetcher(config)
    else:
        raise ValueError(f"Unsupported data source: {source}")


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_data_fetcher():
        """Test data fetcher functionality."""
        fetcher = create_data_fetcher("binance")
        
        try:
            # Test single symbol fetch
            df = fetcher.fetch_klines("BTCUSDT", "1h", limit=100)
            print(f"Fetched {len(df)} records for BTCUSDT")
            print(df.head())
            
            # Test latest price
            price_info = fetcher.get_latest_price("BTCUSDT")
            print(f"Latest price: {price_info}")
            
        except Exception as e:
            print(f"Test failed: {e}")
        
        finally:
            fetcher.close()
    
    # Run test
    asyncio.run(test_data_fetcher())
