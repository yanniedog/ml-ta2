"""
Comprehensive unit tests for data fetching functionality.

Tests cover BinanceDataFetcher, rate limiting, caching, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from datetime import datetime, timedelta
import time
import json

from src.data_fetcher import BinanceDataFetcher, RequestManager, DataValidator, CacheManager
from src.exceptions import DataFetchError, ValidationError


class TestBinanceDataFetcher:
    """Test BinanceDataFetcher functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = {
            'base_url': 'https://api.binance.com',
            'klines_endpoint': '/api/v3/klines',
            'limit': 1000,
            'request_delay_ms': 200,
            'max_retries': 3,
            'timeout_seconds': 30
        }
        self.fetcher = BinanceDataFetcher(self.config)
    
    def test_init(self):
        """Test fetcher initialization."""
        assert self.fetcher.base_url == 'https://api.binance.com'
        assert self.fetcher.klines_endpoint == '/api/v3/klines'
        assert self.fetcher.limit == 1000
        assert isinstance(self.fetcher.request_manager, RequestManager)
        assert isinstance(self.fetcher.validator, DataValidator)
        assert isinstance(self.fetcher.cache_manager, CacheManager)
    
    def test_parse_kline_data_valid(self):
        """Test parsing valid kline data."""
        raw_data = [
            [1609459200000, "29000.00", "29500.00", "28500.00", "29200.00", "100.5", 
             1609462799999, "2920000.0", 1000, "50.25", "1460000.0", "0"],
            [1609462800000, "29200.00", "29800.00", "29000.00", "29600.00", "150.3", 
             1609466399999, "4440000.0", 1500, "75.15", "2220000.0", "0"]
        ]
        
        df = self.fetcher._parse_kline_data(raw_data, 'BTCUSDT')
        
        assert len(df) == 2
        assert list(df.columns) == ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        assert df['symbol'].iloc[0] == 'BTCUSDT'
        assert df['open'].iloc[0] == 29000.00
        assert df['high'].iloc[0] == 29500.00
        assert df['low'].iloc[0] == 28500.00
        assert df['close'].iloc[0] == 29200.00
        assert df['volume'].iloc[0] == 100.5
        assert isinstance(df['timestamp'].iloc[0], pd.Timestamp)
    
    def test_parse_kline_data_empty(self):
        """Test parsing empty kline data."""
        df = self.fetcher._parse_kline_data([], 'BTCUSDT')
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)
    
    def test_parse_kline_data_invalid_numbers(self):
        """Test parsing kline data with invalid numbers."""
        raw_data = [
            [1609459200000, "invalid", "29500.00", "28500.00", "29200.00", "100.5", 
             1609462799999, "2920000.0", 1000, "50.25", "1460000.0", "0"]
        ]
        
        df = self.fetcher._parse_kline_data(raw_data, 'BTCUSDT')
        
        # Should handle invalid numbers gracefully (convert to NaN)
        assert len(df) == 1
        assert pd.isna(df['open'].iloc[0])
    
    @patch('src.data_fetcher.RequestManager.make_request')
    @patch('src.data_fetcher.DataValidator.validate_kline_data')
    @patch('src.data_fetcher.DataValidator.validate_dataframe')
    @patch('src.data_fetcher.CacheManager.get_cached_data')
    @patch('src.data_fetcher.CacheManager.cache_data')
    def test_fetch_klines_success(self, mock_cache_data, mock_get_cached, 
                                 mock_validate_df, mock_validate_kline, mock_request):
        """Test successful kline fetching."""
        # Setup mocks
        mock_get_cached.return_value = None  # No cached data
        mock_request.return_value = [
            [1609459200000, "29000.00", "29500.00", "28500.00", "29200.00", "100.5", 
             1609462799999, "2920000.0", 1000, "50.25", "1460000.0", "0"]
        ]
        mock_validate_kline.return_value = None
        mock_validate_df.return_value = None
        
        # Test fetch
        df = self.fetcher.fetch_klines('BTCUSDT', '1h')
        
        # Assertions
        assert len(df) == 1
        assert df['symbol'].iloc[0] == 'BTCUSDT'
        mock_request.assert_called_once()
        mock_validate_kline.assert_called_once()
        mock_validate_df.assert_called_once()
    
    @patch('src.data_fetcher.CacheManager.get_cached_data')
    def test_fetch_klines_cached(self, mock_get_cached):
        """Test fetching klines from cache."""
        # Setup cached data
        cached_df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2021-01-01')],
            'symbol': ['BTCUSDT'],
            'open': [29000.0],
            'high': [29500.0],
            'low': [28500.0],
            'close': [29200.0],
            'volume': [100.5]
        })
        mock_get_cached.return_value = cached_df
        
        # Test fetch
        start_time = datetime(2021, 1, 1)
        end_time = datetime(2021, 1, 2)
        df = self.fetcher.fetch_klines('BTCUSDT', '1h', start_time, end_time)
        
        # Should return cached data
        assert len(df) == 1
        assert df['symbol'].iloc[0] == 'BTCUSDT'
        mock_get_cached.assert_called_once()
    
    @patch('src.data_fetcher.RequestManager.make_request')
    def test_fetch_klines_request_error(self, mock_request):
        """Test kline fetching with request error."""
        mock_request.side_effect = DataFetchError("API request failed")
        
        with pytest.raises(DataFetchError):
            self.fetcher.fetch_klines('BTCUSDT', '1h')
    
    @patch('src.data_fetcher.RequestManager.make_request')
    @patch('src.data_fetcher.DataValidator.validate_kline_data')
    def test_fetch_klines_validation_error(self, mock_validate, mock_request):
        """Test kline fetching with validation error."""
        mock_request.return_value = [["invalid", "data"]]
        mock_validate.side_effect = ValidationError("Invalid data format")
        
        with pytest.raises(ValidationError):
            self.fetcher.fetch_klines('BTCUSDT', '1h')
    
    def test_fetch_klines_parameters(self):
        """Test kline fetching parameter handling."""
        with patch('src.data_fetcher.RequestManager.make_request') as mock_request:
            mock_request.return_value = []
            
            start_time = datetime(2021, 1, 1)
            end_time = datetime(2021, 1, 2)
            
            self.fetcher.fetch_klines('BTCUSDT', '1h', start_time, end_time, limit=500)
            
            # Verify request parameters
            call_args = mock_request.call_args
            params = call_args[1]['params']
            
            assert params['symbol'] == 'BTCUSDT'
            assert params['interval'] == '1h'
            assert params['limit'] == 500
            assert 'startTime' in params
            assert 'endTime' in params


class TestRequestManager:
    """Test RequestManager functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.config = {
            'request_delay_ms': 200,
            'max_retries': 3,
            'timeout_seconds': 30,
            'backoff_factor': 2
        }
        self.request_manager = RequestManager(self.config)
    
    @patch('requests.get')
    def test_make_request_success(self, mock_get):
        """Test successful API request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.request_manager.make_request('https://api.test.com', {'param': 'value'})
        
        assert result == {"data": "test"}
        mock_get.assert_called_once()
    
    @patch('requests.get')
    def test_make_request_retry_on_failure(self, mock_get):
        """Test request retry on failure."""
        # First two calls fail, third succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status.side_effect = Exception("Server error")
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"data": "test"}
        mock_response_success.raise_for_status.return_value = None
        
        mock_get.side_effect = [mock_response_fail, mock_response_fail, mock_response_success]
        
        result = self.request_manager.make_request('https://api.test.com')
        
        assert result == {"data": "test"}
        assert mock_get.call_count == 3
    
    @patch('requests.get')
    def test_make_request_max_retries_exceeded(self, mock_get):
        """Test request failure after max retries."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = Exception("Server error")
        mock_get.return_value = mock_response
        
        with pytest.raises(DataFetchError):
            self.request_manager.make_request('https://api.test.com')
        
        assert mock_get.call_count == self.config['max_retries']
    
    @patch('time.sleep')
    @patch('requests.get')
    def test_rate_limiting(self, mock_get, mock_sleep):
        """Test rate limiting between requests."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Make two requests
        self.request_manager.make_request('https://api.test.com')
        self.request_manager.make_request('https://api.test.com')
        
        # Should have slept between requests
        mock_sleep.assert_called()


class TestDataValidator:
    """Test DataValidator functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.validator = DataValidator()
    
    def test_validate_kline_data_valid(self):
        """Test validation of valid kline data."""
        valid_data = [
            [1609459200000, "29000.00", "29500.00", "28500.00", "29200.00", "100.5", 
             1609462799999, "2920000.0", 1000, "50.25", "1460000.0", "0"]
        ]
        
        # Should not raise exception
        self.validator.validate_kline_data(valid_data, 'BTCUSDT', '1h')
    
    def test_validate_kline_data_empty(self):
        """Test validation of empty kline data."""
        # Should not raise exception for empty data
        self.validator.validate_kline_data([], 'BTCUSDT', '1h')
    
    def test_validate_kline_data_invalid_format(self):
        """Test validation of invalid kline data format."""
        invalid_data = [
            ["invalid", "format"]  # Too few fields
        ]
        
        with pytest.raises(ValidationError):
            self.validator.validate_kline_data(invalid_data, 'BTCUSDT', '1h')
    
    def test_validate_dataframe_valid(self):
        """Test validation of valid DataFrame."""
        df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2021-01-01')],
            'symbol': ['BTCUSDT'],
            'open': [29000.0],
            'high': [29500.0],
            'low': [28500.0],
            'close': [29200.0],
            'volume': [100.5]
        })
        
        # Should not raise exception
        self.validator.validate_dataframe(df, 'BTCUSDT', '1h')
    
    def test_validate_dataframe_missing_columns(self):
        """Test validation of DataFrame with missing columns."""
        df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2021-01-01')],
            'symbol': ['BTCUSDT']
            # Missing required columns
        })
        
        with pytest.raises(ValidationError):
            self.validator.validate_dataframe(df, 'BTCUSDT', '1h')
    
    def test_validate_dataframe_invalid_ohlc(self):
        """Test validation of DataFrame with invalid OHLC relationships."""
        df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2021-01-01')],
            'symbol': ['BTCUSDT'],
            'open': [29000.0],
            'high': [28000.0],  # High < Open (invalid)
            'low': [28500.0],
            'close': [29200.0],
            'volume': [100.5]
        })
        
        with pytest.raises(ValidationError):
            self.validator.validate_dataframe(df, 'BTCUSDT', '1h')


class TestCacheManager:
    """Test CacheManager functionality."""
    
    def setup_method(self):
        """Setup test environment."""
        self.cache_manager = CacheManager()
    
    def test_cache_and_retrieve_data(self):
        """Test caching and retrieving data."""
        df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2021-01-01')],
            'symbol': ['BTCUSDT'],
            'open': [29000.0],
            'high': [29500.0],
            'low': [28500.0],
            'close': [29200.0],
            'volume': [100.5]
        })
        
        start_time = int(datetime(2021, 1, 1).timestamp() * 1000)
        end_time = int(datetime(2021, 1, 2).timestamp() * 1000)
        
        # Cache data
        self.cache_manager.cache_data(df, 'BTCUSDT', '1h', start_time, end_time)
        
        # Retrieve data
        cached_df = self.cache_manager.get_cached_data('BTCUSDT', '1h', start_time, end_time)
        
        assert cached_df is not None
        assert len(cached_df) == 1
        assert cached_df['symbol'].iloc[0] == 'BTCUSDT'
    
    def test_cache_miss(self):
        """Test cache miss scenario."""
        start_time = int(datetime(2021, 1, 1).timestamp() * 1000)
        end_time = int(datetime(2021, 1, 2).timestamp() * 1000)
        
        cached_df = self.cache_manager.get_cached_data('BTCUSDT', '1h', start_time, end_time)
        
        assert cached_df is None
    
    def test_cache_expiry(self):
        """Test cache expiry functionality."""
        df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2021-01-01')],
            'symbol': ['BTCUSDT'],
            'open': [29000.0],
            'high': [29500.0],
            'low': [28500.0],
            'close': [29200.0],
            'volume': [100.5]
        })
        
        start_time = int(datetime(2021, 1, 1).timestamp() * 1000)
        end_time = int(datetime(2021, 1, 2).timestamp() * 1000)
        
        # Cache with short TTL
        self.cache_manager.ttl_seconds = 0.1  # 100ms
        self.cache_manager.cache_data(df, 'BTCUSDT', '1h', start_time, end_time)
        
        # Wait for expiry
        time.sleep(0.2)
        
        # Should be expired
        cached_df = self.cache_manager.get_cached_data('BTCUSDT', '1h', start_time, end_time)
        assert cached_df is None


# Property-based testing with hypothesis
try:
    from hypothesis import given, strategies as st
    
    class TestDataFetcherPropertyBased:
        """Property-based tests for data fetcher."""
        
        @given(st.text(min_size=1, max_size=20).filter(lambda x: x.isalnum()))
        def test_symbol_handling(self, symbol):
            """Test that symbol handling is consistent."""
            fetcher = BinanceDataFetcher()
            
            # Should handle various symbol formats
            raw_data = [
                [1609459200000, "29000.00", "29500.00", "28500.00", "29200.00", "100.5", 
                 1609462799999, "2920000.0", 1000, "50.25", "1460000.0", "0"]
            ]
            
            df = fetcher._parse_kline_data(raw_data, symbol)
            
            assert len(df) == 1
            assert df['symbol'].iloc[0] == symbol
        
        @given(st.lists(st.floats(min_value=0, max_value=100000), min_size=1, max_size=100))
        def test_price_data_handling(self, prices):
            """Test handling of various price data."""
            fetcher = BinanceDataFetcher()
            
            raw_data = []
            for i, price in enumerate(prices):
                raw_data.append([
                    1609459200000 + i * 60000,  # Increment timestamp
                    str(price), str(price * 1.01), str(price * 0.99), str(price * 1.005),
                    "100.5", 1609462799999 + i * 60000, "2920000.0", 1000, "50.25", "1460000.0", "0"
                ])
            
            df = fetcher._parse_kline_data(raw_data, 'TESTUSDT')
            
            assert len(df) == len(prices)
            assert all(df['open'] >= 0)  # Prices should be non-negative

except ImportError:
    # Hypothesis not available, skip property-based tests
    pass


if __name__ == '__main__':
    pytest.main([__file__])
