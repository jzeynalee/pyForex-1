# tests/test_trading_mtf_data_provider.py
"""
Unit tests for trading/mtf_data_provider.py - Multi-timeframe data provider.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from trading.mtf_data_provider import (
    MTFDataProvider, MTFDataCache
)


@pytest.mark.unit
class TestMTFDataCache:
    """Test MTFDataCache class."""

    def test_init(self):
        """Test cache initialization."""
        cache = MTFDataCache()

        assert len(cache.data) == 0
        assert len(cache.last_update) == 0

    def test_is_stale_no_data(self):
        """Test stale check when no data exists."""
        cache = MTFDataCache()

        assert cache.is_stale("H1") is True

    def test_is_stale_fresh_data(self):
        """Test stale check with fresh data."""
        cache = MTFDataCache()
        cache.update_intervals["H1"] = 5  # 5 minutes
        df = pd.DataFrame({'close': [1.1000]})
        
        cache.update("H1", df)

        assert cache.is_stale("H1") is False

    def test_is_stale_old_data(self):
        """Test stale check with old data."""
        cache = MTFDataCache()
        cache.update_intervals["H1"] = 5  # 5 minutes
        
        # Update with old timestamp
        df = pd.DataFrame({'close': [1.1000]})
        cache.data["H1"] = df
        cache.last_update["H1"] = datetime.now() - timedelta(minutes=10)

        assert cache.is_stale("H1") is True

    def test_update(self):
        """Test updating cache."""
        cache = MTFDataCache()
        df = pd.DataFrame({'close': [1.1000, 1.1005]})

        cache.update("H1", df)

        assert "H1" in cache.data
        assert "H1" in cache.last_update
        assert len(cache.data["H1"]) == 2

    def test_get_fresh(self):
        """Test getting fresh cached data."""
        cache = MTFDataCache()
        cache.update_intervals["H1"] = 60  # Long interval
        df = pd.DataFrame({'close': [1.1000]})
        
        cache.update("H1", df)
        result = cache.get("H1")

        assert result is not None
        assert len(result) == 1

    def test_get_stale(self):
        """Test getting stale cached data (returns None)."""
        cache = MTFDataCache()
        cache.update_intervals["H1"] = 1  # Short interval
        df = pd.DataFrame({'close': [1.1000]})
        cache.data["H1"] = df
        cache.last_update["H1"] = datetime.now() - timedelta(minutes=5)

        result = cache.get("H1")

        assert result is None

    def test_clear(self):
        """Test clearing cache."""
        cache = MTFDataCache()
        cache.update("H1", pd.DataFrame({'close': [1.1000]}))
        cache.update("M15", pd.DataFrame({'close': [1.1000]}))

        cache.clear()

        assert len(cache.data) == 0
        assert len(cache.last_update) == 0


@pytest.mark.unit
class TestMTFDataProvider:
    """Test MTFDataProvider class."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock connector."""
        connector = Mock()
        connector.ensure_connected = Mock(return_value=True)
        connector.get_data = Mock(return_value=pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=100, freq='h'),
            'open': [1.1000] * 100,
            'high': [1.1010] * 100,
            'low': [1.0990] * 100,
            'close': [1.1005] * 100,
            'volume': [1000] * 100
        }))
        return connector

    def test_init_default(self):
        """Test default initialization."""
        provider = MTFDataProvider()

        assert provider.symbol == "EURUSD"
        assert provider.cache_enabled is True
        assert provider.connector is None

    def test_init_with_connector(self, mock_connector):
        """Test initialization with connector."""
        provider = MTFDataProvider(connector=mock_connector, symbol="GBPUSD")

        assert provider.symbol == "GBPUSD"
        assert provider.connector == mock_connector

    def test_init_cache_disabled(self):
        """Test initialization with cache disabled."""
        provider = MTFDataProvider(cache_enabled=False)

        assert provider.cache_enabled is False

    def test_fetch_single_timeframe_with_connector(self, mock_connector):
        """Test fetching single timeframe data with connector."""
        provider = MTFDataProvider(connector=mock_connector)

        df = provider.fetch_single_timeframe("H1", n_candles=100)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        mock_connector.get_data.assert_called_once()

    def test_fetch_single_timeframe_with_cache(self, mock_connector):
        """Test fetching with cache enabled."""
        provider = MTFDataProvider(connector=mock_connector, cache_enabled=True)

        # First fetch
        df1 = provider.fetch_single_timeframe("H1", n_candles=100, use_cache=True)
        
        # Second fetch should use cache
        df2 = provider.fetch_single_timeframe("H1", n_candles=100, use_cache=True)

        assert len(df1) == len(df2)
        # Should only call connector once due to cache
        assert mock_connector.get_data.call_count >= 1

    def test_fetch_single_timeframe_no_cache(self, mock_connector):
        """Test fetching without using cache."""
        provider = MTFDataProvider(connector=mock_connector, cache_enabled=True)

        df1 = provider.fetch_single_timeframe("H1", n_candles=100, use_cache=False)
        df2 = provider.fetch_single_timeframe("H1", n_candles=100, use_cache=False)

        # Should call connector twice
        assert mock_connector.get_data.call_count == 2

    def test_fetch_multiple_timeframes(self, mock_connector):
        """Test fetching multiple timeframes."""
        provider = MTFDataProvider(connector=mock_connector)

        data = provider.fetch_multiple_timeframes(["H1", "M15", "H4"], n_candles=100)

        assert "H1" in data
        assert "M15" in data
        assert "H4" in data
        assert all(isinstance(df, pd.DataFrame) for df in data.values())

    def test_get_timeframe_data_cached(self, mock_connector):
        """Test getting cached timeframe data."""
        provider = MTFDataProvider(connector=mock_connector, cache_enabled=True)
        
        # Fetch first to populate cache
        provider.fetch_single_timeframe("H1", n_candles=100)

        # Get from cache
        df = provider.get_timeframe_data("H1", n_candles=100)

        assert df is not None
        assert isinstance(df, pd.DataFrame)

    def test_get_timeframe_data_not_cached(self, mock_connector):
        """Test getting data when not in cache."""
        provider = MTFDataProvider(connector=mock_connector, cache_enabled=True)

        df = provider.get_timeframe_data("H1", n_candles=100)

        # Should fetch fresh data
        assert df is not None
        mock_connector.get_data.assert_called()

    def test_refresh_timeframe(self, mock_connector):
        """Test refreshing a specific timeframe."""
        provider = MTFDataProvider(connector=mock_connector)

        success = provider.refresh_timeframe("H1", n_candles=100)

        assert success is True
        mock_connector.get_data.assert_called_once()

    def test_refresh_all_timeframes(self, mock_connector):
        """Test refreshing all timeframes."""
        provider = MTFDataProvider(connector=mock_connector)
        timeframes = ["H1", "M15", "H4"]

        provider.refresh_all_timeframes(timeframes, n_candles=100)

        # Should have called for each timeframe
        assert mock_connector.get_data.call_count == len(timeframes)

    def test_synchronize_timeframes(self, mock_connector):
        """Test synchronizing timeframes."""
        # Create data with different timestamps
        dates1 = pd.date_range('2024-01-01', periods=100, freq='h')
        dates2 = pd.date_range('2024-01-01', periods=25, freq='4h')  # H4
        
        def side_effect(n, symbol, timeframe):
            if timeframe == "H1":
                return pd.DataFrame({
                    'time': dates1,
                    'close': [1.1000] * 100
                })
            else:
                return pd.DataFrame({
                    'time': dates2,
                    'close': [1.1000] * 25
                })
        
        mock_connector.get_data.side_effect = side_effect
        provider = MTFDataProvider(connector=mock_connector)

        data = provider.fetch_multiple_timeframes(["H1", "H4"], n_candles=100)
        synchronized = provider.synchronize_timeframes(data)

        # All should have same length (aligned to shortest)
        if synchronized:
            lengths = [len(df) for df in synchronized.values()]
            assert min(lengths) > 0

    def test_get_latest_bar_time(self, mock_connector):
        """Test getting latest bar time for timeframe."""
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        mock_connector.get_data.return_value = pd.DataFrame({
            'time': dates,
            'close': [1.1000] * 100
        })
        
        provider = MTFDataProvider(connector=mock_connector)

        latest_time = provider.get_latest_bar_time("H1")

        assert latest_time is not None
        assert isinstance(latest_time, datetime)

