# tests/test_trading_mt5_connector.py
"""
Unit tests for trading/mt5_connector.py - MetaTrader 5 connection and order execution.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from datetime import datetime
from trading.mt5_connector import (
    MT5Connector, MockMT5Connector, OrderResult, AccountInfo, Timeframe
)


@pytest.mark.unit
class TestOrderResult:
    """Test OrderResult NamedTuple."""

    def test_order_result_creation(self):
        """Test creating OrderResult."""
        result = OrderResult(
            success=True,
            ticket=12345,
            price=1.1000,
            volume=0.1,
            error=None
        )

        assert result.success is True
        assert result.ticket == 12345
        assert result.price == 1.1000
        assert result.volume == 0.1
        assert result.error is None

    def test_order_result_failure(self):
        """Test OrderResult for failed order."""
        result = OrderResult(
            success=False,
            ticket=None,
            price=None,
            volume=0.1,
            error="Connection failed"
        )

        assert result.success is False
        assert result.ticket is None
        assert result.error == "Connection failed"


@pytest.mark.unit
class TestAccountInfo:
    """Test AccountInfo dataclass."""

    def test_account_info_creation(self):
        """Test creating AccountInfo."""
        info = AccountInfo(
            balance=10000.0,
            equity=10050.0,
            margin=500.0,
            free_margin=9550.0,
            profit=50.0
        )

        assert info.balance == 10000.0
        assert info.equity == 10050.0
        assert info.margin == 500.0
        assert info.free_margin == 9550.0
        assert info.profit == 50.0


@pytest.mark.unit
class TestTimeframe:
    """Test Timeframe enum."""

    def test_timeframe_values(self):
        """Test timeframe enum values."""
        assert Timeframe.M1.value == 1
        assert Timeframe.M5.value == 5
        assert Timeframe.H1.value == 60
        assert Timeframe.H4.value == 240
        assert Timeframe.D1.value == 1440


@pytest.mark.unit
class TestMockMT5Connector:
    """Test MockMT5Connector class."""

    def test_init_default(self):
        """Test default initialization."""
        connector = MockMT5Connector()

        assert connector.symbol == "EURUSD"
        assert connector.connected is True
        assert len(connector._positions) == 0

    def test_init_custom_symbol(self):
        """Test initialization with custom symbol."""
        connector = MockMT5Connector(symbol="GBPUSD")

        assert connector.symbol == "GBPUSD"

    def test_connect(self):
        """Test connection."""
        connector = MockMT5Connector()
        connector.connected = False

        result = connector.connect()

        assert result is True
        assert connector.connected is True

    def test_disconnect(self):
        """Test disconnection."""
        connector = MockMT5Connector()

        connector.disconnect()

        assert connector.connected is False

    def test_ensure_connected(self):
        """Test ensure_connected."""
        connector = MockMT5Connector()

        result = connector.ensure_connected()

        assert result is True

    def test_get_account_info(self):
        """Test getting account info."""
        connector = MockMT5Connector()

        info = connector.get_account_info()

        assert isinstance(info, AccountInfo)
        assert info.balance == 10000.0
        assert info.equity == 10000.0

    def test_get_data(self):
        """Test getting mock market data."""
        connector = MockMT5Connector()

        df = connector.get_data(n=100)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert 'time' in df.columns
        assert 'open' in df.columns
        assert 'close' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns

    def test_execute_order(self):
        """Test executing an order."""
        connector = MockMT5Connector()

        result = connector.execute_order(
            signal="BUY",
            volume=0.1,
            sl=1.0950,
            tp=1.1100
        )

        assert result.success is True
        assert result.ticket is not None
        assert result.volume == 0.1

    def test_entry_alias(self):
        """Test entry alias for execute_order."""
        connector = MockMT5Connector()

        result = connector.entry("SELL", 0.1, 1.1050, 1.0900)

        assert result.success is True

    def test_get_open_positions(self):
        """Test getting open positions."""
        connector = MockMT5Connector()

        positions = connector.get_open_positions()

        assert isinstance(positions, list)

    def test_get_current_price(self):
        """Test getting current price."""
        connector = MockMT5Connector()

        price_info = connector.get_current_price()

        assert 'bid' in price_info
        assert 'ask' in price_info
        assert 'spread' in price_info
        assert price_info['ask'] > price_info['bid']

    def test_get_symbol_info(self):
        """Test getting symbol info."""
        connector = MockMT5Connector()

        symbol_info = connector.get_symbol_info()

        assert 'name' in symbol_info
        assert 'point' in symbol_info
        assert 'volume_min' in symbol_info
        assert symbol_info['name'] == connector.symbol


@pytest.mark.unit
class TestMT5Connector:
    """Test MT5Connector class (with mocking)."""

    @pytest.fixture
    def mock_mt5(self):
        """Mock MetaTrader5 module."""
        with patch('trading.mt5_connector.MT5_AVAILABLE', True):
            mock_mt5 = MagicMock()
            mock_mt5.TIMEFRAME_M1 = 1
            mock_mt5.TIMEFRAME_M5 = 5
            mock_mt5.TIMEFRAME_M15 = 15
            mock_mt5.TIMEFRAME_M30 = 30
            mock_mt5.TIMEFRAME_H1 = 60
            mock_mt5.TIMEFRAME_H4 = 240
            mock_mt5.TIMEFRAME_D1 = 1440
            mock_mt5.TRADE_ACTION_DEAL = 1
            mock_mt5.ORDER_TYPE_BUY = 0
            mock_mt5.ORDER_TYPE_SELL = 1
            mock_mt5.ORDER_TIME_GTC = 0
            mock_mt5.ORDER_FILLING_IOC = 1
            mock_mt5.TRADE_RETCODE_DONE = 10009
            
            with patch('trading.mt5_connector.mt5', mock_mt5):
                yield mock_mt5

    def test_init_raises_when_mt5_unavailable(self):
        """Test initialization raises when MT5 unavailable."""
        with patch('trading.mt5_connector.MT5_AVAILABLE', False):
            with pytest.raises(RuntimeError, match="MetaTrader5 library not available"):
                MT5Connector()

    def test_init_default(self, mock_mt5):
        """Test default initialization."""
        connector = MT5Connector()

        assert connector.symbol == "EURUSD"
        assert connector.timeframe == "H1"
        assert connector.connected is False

    def test_init_custom(self, mock_mt5):
        """Test initialization with custom parameters."""
        connector = MT5Connector(
            account=12345,
            password="pass",
            server="server",
            symbol="GBPUSD",
            timeframe="M15"
        )

        assert connector.account == 12345
        assert connector.password == "pass"
        assert connector.symbol == "GBPUSD"
        assert connector.timeframe == "M15"

    def test_get_timeframe_mapping(self, mock_mt5):
        """Test timeframe string to MT5 constant mapping."""
        connector = MT5Connector()

        # Test valid timeframes
        assert connector._get_mt5_timeframe("H1") == mock_mt5.TIMEFRAME_H1
        assert connector._get_mt5_timeframe("M15") == mock_mt5.TIMEFRAME_M15
        assert connector._get_mt5_timeframe("H4") == mock_mt5.TIMEFRAME_H4

    def test_get_timeframe_unknown_defaults(self, mock_mt5):
        """Test unknown timeframe defaults to H1."""
        connector = MT5Connector()

        result = connector._get_mt5_timeframe("UNKNOWN")

        assert result == mock_mt5.TIMEFRAME_H1

    def test_connect_success(self, mock_mt5):
        """Test successful connection."""
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True
        
        connector = MT5Connector(account=12345, password="pass", server="server")

        result = connector.connect()

        assert result is True
        assert connector.connected is True
        mock_mt5.initialize.assert_called_once()
        mock_mt5.login.assert_called_once()

    def test_connect_initialize_fails(self, mock_mt5):
        """Test connection fails when initialize fails."""
        mock_mt5.initialize.return_value = False
        mock_mt5.last_error.return_value = (1, "Init failed")
        
        connector = MT5Connector()

        result = connector.connect()

        assert result is False
        assert connector.connected is False

    def test_connect_login_fails(self, mock_mt5):
        """Test connection fails when login fails."""
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = False
        mock_mt5.last_error.return_value = (2, "Login failed")
        
        connector = MT5Connector(account=12345, password="pass", server="server")

        result = connector.connect()

        assert result is False
        mock_mt5.shutdown.assert_called_once()

    def test_disconnect(self, mock_mt5):
        """Test disconnection."""
        mock_mt5.initialize.return_value = True
        connector = MT5Connector()
        connector.connected = True

        connector.disconnect()

        assert connector.connected is False
        mock_mt5.shutdown.assert_called_once()

    def test_ensure_connected_already_connected(self, mock_mt5):
        """Test ensure_connected when already connected."""
        mock_mt5.terminal_info.return_value = Mock()
        connector = MT5Connector()
        connector.connected = True

        result = connector.ensure_connected()

        assert result is True

    def test_ensure_connected_reconnects(self, mock_mt5):
        """Test ensure_connected reconnects when needed."""
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = None
        connector = MT5Connector()
        connector.connect = Mock(return_value=True)

        result = connector.ensure_connected()

        assert result is True
        connector.connect.assert_called_once()

    def test_get_account_info_success(self, mock_mt5):
        """Test getting account info successfully."""
        mock_account = Mock()
        mock_account.balance = 10000.0
        mock_account.equity = 10050.0
        mock_account.margin = 500.0
        mock_account.margin_free = 9550.0
        mock_account.profit = 50.0
        
        mock_mt5.initialize.return_value = True
        mock_mt5.account_info.return_value = mock_account
        
        connector = MT5Connector()
        connector.ensure_connected = Mock(return_value=True)

        info = connector.get_account_info()

        assert info is not None
        assert info.balance == 10000.0
        assert info.equity == 10050.0

    def test_get_account_info_fails(self, mock_mt5):
        """Test getting account info when connection fails."""
        connector = MT5Connector()
        connector.ensure_connected = Mock(return_value=False)

        info = connector.get_account_info()

        assert info is None

    def test_get_data_success(self, mock_mt5):
        """Test getting market data successfully."""
        mock_rates = [
            {'time': 1609459200, 'open': 1.1000, 'high': 1.1010, 
             'low': 1.0990, 'close': 1.1005, 'tick_volume': 1000}
        ] * 10
        
        mock_mt5.initialize.return_value = True
        mock_mt5.copy_rates_from_pos.return_value = mock_rates
        
        connector = MT5Connector()
        connector.ensure_connected = Mock(return_value=True)

        df = connector.get_data(n=10)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert 'time' in df.columns

    def test_get_data_fails(self, mock_mt5):
        """Test getting data when request fails."""
        mock_mt5.initialize.return_value = True
        mock_mt5.copy_rates_from_pos.return_value = None
        mock_mt5.last_error.return_value = (1, "Data error")
        
        connector = MT5Connector()
        connector.ensure_connected = Mock(return_value=True)

        df = connector.get_data(n=10)

        assert df.empty

    def test_get_current_price_success(self, mock_mt5):
        """Test getting current price successfully."""
        mock_tick = Mock()
        mock_tick.bid = 1.0999
        mock_tick.ask = 1.1001
        mock_tick.time = 1609459200
        
        mock_mt5.initialize.return_value = True
        mock_mt5.symbol_info_tick.return_value = mock_tick
        
        connector = MT5Connector()
        connector.ensure_connected = Mock(return_value=True)

        price_info = connector.get_current_price()

        assert price_info is not None
        assert price_info['bid'] == 1.0999
        assert price_info['ask'] == 1.1001
        assert price_info['spread'] == pytest.approx(0.0002, rel=1e-6)

    def test_execute_order_buy_success(self, mock_mt5):
        """Test executing BUY order successfully."""
        mock_tick = Mock()
        mock_tick.bid = 1.0999
        mock_tick.ask = 1.1001
        
        mock_result = Mock()
        mock_result.retcode = mock_mt5.TRADE_RETCODE_DONE
        mock_result.order = 12345
        mock_result.price = 1.1001
        mock_result.volume = 0.1
        
        mock_mt5.initialize.return_value = True
        mock_mt5.symbol_info_tick.return_value = mock_tick
        mock_mt5.order_send.return_value = mock_result
        
        connector = MT5Connector()
        connector.ensure_connected = Mock(return_value=True)

        result = connector.execute_order("BUY", 0.1, 1.0950, 1.1100)

        assert result.success is True
        assert result.ticket == 12345
        assert result.price == 1.1001

    def test_execute_order_invalid_signal(self, mock_mt5):
        """Test executing order with invalid signal."""
        connector = MT5Connector()
        connector.ensure_connected = Mock(return_value=True)

        result = connector.execute_order("INVALID", 0.1, 1.0950, 1.1100)

        assert result.success is False
        assert "Invalid signal" in result.error

    def test_execute_order_not_connected(self, mock_mt5):
        """Test executing order when not connected."""
        connector = MT5Connector()
        connector.ensure_connected = Mock(return_value=False)

        result = connector.execute_order("BUY", 0.1, 1.0950, 1.1100)

        assert result.success is False
        assert "Not connected" in result.error

    def test_get_open_positions(self, mock_mt5):
        """Test getting open positions."""
        mock_position = Mock()
        mock_position.ticket = 12345
        mock_position.symbol = "EURUSD"
        mock_position.type = mock_mt5.ORDER_TYPE_BUY
        mock_position.volume = 0.1
        mock_position.price_open = 1.1000
        mock_position.price_current = 1.1005
        mock_position.sl = 1.0950
        mock_position.tp = 1.1100
        mock_position.profit = 50.0
        mock_position.magic = 123456
        mock_position.time = 1609459200
        
        mock_mt5.initialize.return_value = True
        mock_mt5.positions_get.return_value = [mock_position]
        
        connector = MT5Connector()
        connector.ensure_connected = Mock(return_value=True)

        positions = connector.get_open_positions()

        assert len(positions) == 1
        assert positions[0]['ticket'] == 12345
        assert positions[0]['type'] == 'BUY'

    def test_close_position_success(self, mock_mt5):
        """Test closing position successfully."""
        mock_position = Mock()
        mock_position.type = mock_mt5.ORDER_TYPE_BUY
        mock_position.symbol = "EURUSD"
        mock_position.volume = 0.1
        
        mock_tick = Mock()
        mock_tick.bid = 1.1005
        mock_tick.ask = 1.1007
        
        mock_result = Mock()
        mock_result.retcode = mock_mt5.TRADE_RETCODE_DONE
        mock_result.order = 54321
        mock_result.price = 1.1005
        mock_result.volume = 0.1
        
        mock_mt5.initialize.return_value = True
        mock_mt5.positions_get.return_value = [mock_position]
        mock_mt5.symbol_info_tick.return_value = mock_tick
        mock_mt5.order_send.return_value = mock_result
        
        connector = MT5Connector()
        connector.ensure_connected = Mock(return_value=True)

        result = connector.close_position(12345)

        assert result.success is True
        assert result.ticket == 54321

    def test_close_position_not_found(self, mock_mt5):
        """Test closing non-existent position."""
        mock_mt5.initialize.return_value = True
        mock_mt5.positions_get.return_value = None
        
        connector = MT5Connector()
        connector.ensure_connected = Mock(return_value=True)

        result = connector.close_position(99999)

        assert result.success is False
        assert "not found" in result.error

    def test_get_symbol_info(self, mock_mt5):
        """Test getting symbol information."""
        mock_symbol = Mock()
        mock_symbol.name = "EURUSD"
        mock_symbol.point = 0.00001
        mock_symbol.digits = 5
        mock_symbol.spread = 10
        mock_symbol.volume_min = 0.01
        mock_symbol.volume_max = 100.0
        mock_symbol.volume_step = 0.01
        mock_symbol.trade_tick_value = 1.0
        mock_symbol.trade_tick_size = 0.00001
        mock_symbol.trade_contract_size = 100000
        
        mock_mt5.initialize.return_value = True
        mock_mt5.symbol_info.return_value = mock_symbol
        
        connector = MT5Connector()
        connector.ensure_connected = Mock(return_value=True)

        symbol_info = connector.get_symbol_info()

        assert symbol_info is not None
        assert symbol_info['name'] == "EURUSD"
        assert symbol_info['point'] == 0.00001

