# tests/test_trading_bot.py
"""
Unit tests for trading/bot.py - Main trading bot orchestration.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import pandas as pd
from trading.bot import BotConfig, TradingBot, BacktestBot


@pytest.mark.unit
class TestBotConfig:
    """Test BotConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BotConfig()

        assert config.symbol == "EURUSD"
        assert config.timeframe == "H1"
        assert config.tick_interval == 10.0
        assert config.data_window == 100
        assert config.use_mock is False
        assert config.log_level == "INFO"

    def test_custom_values(self):
        """Test custom configuration."""
        config = BotConfig(
            symbol="GBPUSD",
            timeframe="M15",
            tick_interval=5.0,
            data_window=200,
            use_mock=True,
            log_level="DEBUG"
        )

        assert config.symbol == "GBPUSD"
        assert config.timeframe == "M15"
        assert config.tick_interval == 5.0
        assert config.data_window == 200
        assert config.use_mock is True
        assert config.log_level == "DEBUG"


@pytest.mark.unit
class TestTradingBot:
    """Test TradingBot class."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock connector."""
        connector = Mock()
        connector.connect.return_value = True
        connector.disconnect.return_value = True
        connector.get_account_info.return_value = Mock(balance=10000.0)
        connector.get_data.return_value = pd.DataFrame({
            'time': [datetime(2024, 1, 1, 12, 0)],
            'open': [1.1000],
            'high': [1.1010],
            'low': [1.0990],
            'close': [1.1005],
            'volume': [1000]
        })
        connector.get_open_positions.return_value = []
        return connector

    @pytest.fixture
    def mock_strategy(self):
        """Create a mock strategy."""
        strategy = Mock()
        strategy.name = "MockStrategy"
        strategy.on_bar.return_value = None
        strategy.get_stats.return_value = {"trades": 0}
        return strategy

    @pytest.fixture
    def mock_risk_manager(self):
        """Create a mock risk manager."""
        risk_mgr = Mock()
        risk_mgr.get_status.return_value = {"balance": 10000.0}
        return risk_mgr

    def test_init_with_mock_connector(self):
        """Test bot initialization with mock connector."""
        config = BotConfig(use_mock=True)

        with patch('trading.bot.MockMT5Connector') as MockConnector:
            with patch('trading.bot.RiskManager') as MockRiskMgr:
                with patch('trading.bot.NeuralHybridStrategy') as MockStrategy:
                    bot = TradingBot(config)

                    assert bot.config.symbol == "EURUSD"
                    assert bot.running is False
                    assert bot.iteration_count == 0
                    assert bot.last_bar_time is None
                    MockConnector.assert_called_once()

    def test_init_with_real_connector(self):
        """Test bot initialization with real MT5 connector."""
        config = BotConfig(use_mock=False)

        with patch('trading.bot.MT5Connector') as MockConnector:
            with patch('trading.bot.RiskManager') as MockRiskMgr:
                with patch('trading.bot.NeuralHybridStrategy') as MockStrategy:
                    with patch('trading.bot.settings') as mock_settings:
                        mock_settings.SYMBOL = "EURUSD"
                        mock_settings.TIMEFRAME = "H1"
                        mock_settings.TICK_INTERVAL = 10.0
                        mock_settings.MT5_ACCOUNT = 12345
                        mock_settings.MT5_PASSWORD = "password"
                        mock_settings.MT5_SERVER = "server"
                        mock_settings.MT5_PATH = "path"
                        mock_settings.MAGIC_NUMBER = 123
                        mock_settings.MAX_DAILY_LOSS_PCT = 3.0
                        mock_settings.RISK_PER_TRADE_PCT = 1.0
                        mock_settings.MAX_DRAWDOWN_PCT = 10.0

                        bot = TradingBot(config)

                        MockConnector.assert_called_once()

    def test_init_custom_strategy(self):
        """Test bot initialization with custom strategy class."""
        config = BotConfig(use_mock=True)
        custom_strategy_class = Mock()

        with patch('trading.bot.MockMT5Connector'):
            with patch('trading.bot.RiskManager'):
                bot = TradingBot(config, strategy_class=custom_strategy_class)

                custom_strategy_class.assert_called_once()

    def test_get_status(self, mock_connector, mock_strategy, mock_risk_manager):
        """Test getting bot status."""
        config = BotConfig(use_mock=True)

        with patch('trading.bot.MockMT5Connector', return_value=mock_connector):
            with patch('trading.bot.RiskManager', return_value=mock_risk_manager):
                with patch('trading.bot.NeuralHybridStrategy', return_value=mock_strategy):
                    bot = TradingBot(config)
                    bot.running = True
                    bot.iteration_count = 5
                    bot.last_bar_time = datetime(2024, 1, 1, 12, 0)

                    status = bot.get_status()

                    assert status['running'] is True
                    assert status['iteration_count'] == 5
                    assert status['last_bar_time'] == "2024-01-01 12:00:00"
                    assert 'strategy' in status
                    assert 'risk' in status
                    assert 'positions' in status

    def test_stop(self, mock_connector, mock_strategy, mock_risk_manager):
        """Test stopping the bot."""
        config = BotConfig(use_mock=True)

        with patch('trading.bot.MockMT5Connector', return_value=mock_connector):
            with patch('trading.bot.RiskManager', return_value=mock_risk_manager):
                with patch('trading.bot.NeuralHybridStrategy', return_value=mock_strategy):
                    bot = TradingBot(config)
                    bot.running = True

                    bot.stop()

                    assert bot.running is False

    def test_run_iteration_with_data(self, mock_connector, mock_strategy, mock_risk_manager):
        """Test single iteration with valid data."""
        config = BotConfig(use_mock=True)

        df = pd.DataFrame({
            'time': [datetime(2024, 1, 1, 12, 0)],
            'open': [1.1000],
            'high': [1.1010],
            'low': [1.0990],
            'close': [1.1005],
            'volume': [1000]
        })
        mock_connector.get_data.return_value = df

        with patch('trading.bot.MockMT5Connector', return_value=mock_connector):
            with patch('trading.bot.RiskManager', return_value=mock_risk_manager):
                with patch('trading.bot.NeuralHybridStrategy', return_value=mock_strategy):
                    bot = TradingBot(config)
                    bot._run_iteration()

                    assert bot.iteration_count == 1
                    mock_strategy.on_bar.assert_called_once()
                    assert bot.last_bar_time == df['time'].iloc[-1]

    def test_run_iteration_empty_data(self, mock_connector, mock_strategy, mock_risk_manager):
        """Test iteration with empty data."""
        config = BotConfig(use_mock=True)

        mock_connector.get_data.return_value = pd.DataFrame()

        with patch('trading.bot.MockMT5Connector', return_value=mock_connector):
            with patch('trading.bot.RiskManager', return_value=mock_risk_manager):
                with patch('trading.bot.NeuralHybridStrategy', return_value=mock_strategy):
                    bot = TradingBot(config)
                    bot._run_iteration()

                    assert bot.iteration_count == 1
                    mock_strategy.on_bar.assert_not_called()

    def test_run_iteration_same_bar(self, mock_connector, mock_strategy, mock_risk_manager):
        """Test iteration skips same bar."""
        config = BotConfig(use_mock=True)

        bar_time = datetime(2024, 1, 1, 12, 0)
        df = pd.DataFrame({
            'time': [bar_time],
            'open': [1.1000],
            'high': [1.1010],
            'low': [1.0990],
            'close': [1.1005],
            'volume': [1000]
        })
        mock_connector.get_data.return_value = df

        with patch('trading.bot.MockMT5Connector', return_value=mock_connector):
            with patch('trading.bot.RiskManager', return_value=mock_risk_manager):
                with patch('trading.bot.NeuralHybridStrategy', return_value=mock_strategy):
                    bot = TradingBot(config)

                    # First iteration
                    bot._run_iteration()
                    assert bot.last_bar_time == bar_time
                    assert mock_strategy.on_bar.call_count == 1

                    # Second iteration with same bar
                    bot._run_iteration()
                    assert mock_strategy.on_bar.call_count == 1  # Should not increase

    def test_run_iteration_new_bar(self, mock_connector, mock_strategy, mock_risk_manager):
        """Test iteration processes new bar."""
        config = BotConfig(use_mock=True)

        with patch('trading.bot.MockMT5Connector', return_value=mock_connector):
            with patch('trading.bot.RiskManager', return_value=mock_risk_manager):
                with patch('trading.bot.NeuralHybridStrategy', return_value=mock_strategy):
                    bot = TradingBot(config)

                    # First bar
                    df1 = pd.DataFrame({
                        'time': [datetime(2024, 1, 1, 12, 0)],
                        'close': [1.1005]
                    })
                    mock_connector.get_data.return_value = df1
                    bot._run_iteration()

                    # New bar
                    df2 = pd.DataFrame({
                        'time': [datetime(2024, 1, 1, 13, 0)],
                        'close': [1.1010]
                    })
                    mock_connector.get_data.return_value = df2
                    bot._run_iteration()

                    assert mock_strategy.on_bar.call_count == 2

    def test_run_connection_failure(self, mock_connector, mock_strategy, mock_risk_manager):
        """Test run exits on connection failure."""
        config = BotConfig(use_mock=True)
        mock_connector.connect.return_value = False

        with patch('trading.bot.MockMT5Connector', return_value=mock_connector):
            with patch('trading.bot.RiskManager', return_value=mock_risk_manager):
                with patch('trading.bot.NeuralHybridStrategy', return_value=mock_strategy):
                    bot = TradingBot(config)
                    bot.run()

                    # Should exit immediately
                    assert bot.iteration_count == 0


@pytest.mark.unit
class TestBacktestBot:
    """Test BacktestBot class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range('2024-01-01', periods=200, freq='h')
        return pd.DataFrame({
            'time': dates,
            'open': [1.1000 + i*0.0001 for i in range(200)],
            'high': [1.1010 + i*0.0001 for i in range(200)],
            'low': [1.0990 + i*0.0001 for i in range(200)],
            'close': [1.1005 + i*0.0001 for i in range(200)],
            'volume': [1000] * 200
        })

    @pytest.fixture
    def mock_strategy_class(self):
        """Create a mock strategy class."""
        strategy_class = Mock()
        strategy_instance = Mock()
        strategy_instance.name = "MockStrategy"
        strategy_instance.on_bar.return_value = None
        strategy_class.return_value = strategy_instance
        return strategy_class

    def test_init(self, sample_data, mock_strategy_class):
        """Test BacktestBot initialization."""
        bot = BacktestBot(
            data=sample_data,
            strategy_class=mock_strategy_class,
            initial_balance=10000.0
        )

        assert bot.current_idx == 0
        assert bot.window_size == 100
        assert len(bot.data) == 200
        mock_strategy_class.assert_called_once()

    def test_init_custom_balance(self, sample_data, mock_strategy_class):
        """Test initialization with custom balance."""
        bot = BacktestBot(
            data=sample_data,
            strategy_class=mock_strategy_class,
            initial_balance=50000.0
        )

        assert bot.executor.balance == 50000.0

    def test_get_data(self, sample_data, mock_strategy_class):
        """Test getting data window."""
        bot = BacktestBot(
            data=sample_data,
            strategy_class=mock_strategy_class
        )

        bot.current_idx = 150
        window = bot.get_data(n=50)

        assert len(window) == 50
        assert window.index[-1] == 150

    def test_get_data_at_start(self, sample_data, mock_strategy_class):
        """Test getting data at start of backtest."""
        bot = BacktestBot(
            data=sample_data,
            strategy_class=mock_strategy_class
        )

        bot.current_idx = 10
        window = bot.get_data(n=50)

        # Should return from start
        assert len(window) == 11
        assert window.index[0] == 0

    def test_run_completes(self, sample_data, mock_strategy_class):
        """Test backtest runs to completion."""
        bot = BacktestBot(
            data=sample_data,
            strategy_class=mock_strategy_class
        )

        result = bot.run()

        assert 'trades' in result
        assert 'final_balance' in result
        assert 'signals' in result
        assert len(result['signals']) == 100  # 200 - window_size

    def test_run_updates_executor_price(self, sample_data, mock_strategy_class):
        """Test run updates executor with prices."""
        mock_strategy_instance = mock_strategy_class.return_value

        bot = BacktestBot(
            data=sample_data,
            strategy_class=mock_strategy_class
        )

        bot.run()

        # Strategy should be called for each bar after window
        expected_calls = len(sample_data) - bot.window_size
        assert mock_strategy_instance.on_bar.call_count == expected_calls

    def test_run_collects_signals(self, sample_data, mock_strategy_class):
        """Test run collects signals."""
        mock_strategy_instance = mock_strategy_class.return_value
        mock_strategy_instance.on_bar.return_value = "BUY"

        bot = BacktestBot(
            data=sample_data,
            strategy_class=mock_strategy_class
        )

        result = bot.run()

        # Check signals are collected
        assert all(s['signal'] == "BUY" for s in result['signals'])

    def test_run_final_balance(self, sample_data, mock_strategy_class):
        """Test final balance is returned."""
        bot = BacktestBot(
            data=sample_data,
            strategy_class=mock_strategy_class,
            initial_balance=10000.0
        )

        result = bot.run()

        assert result['final_balance'] <= 10000.0  # May have commissions

    def test_window_size_limits_start(self, sample_data, mock_strategy_class):
        """Test window size prevents early trading."""
        bot = BacktestBot(
            data=sample_data,
            strategy_class=mock_strategy_class
        )

        # Backtest starts at window_size
        result = bot.run()

        # First signal should be at index window_size
        first_signal_time = result['signals'][0]['time']
        expected_time = sample_data['time'].iloc[bot.window_size]
        assert first_signal_time == expected_time


@pytest.mark.unit
class TestBotIntegration:
    """Integration tests for bot components."""

    def test_bot_with_backtest_executor(self):
        """Test bot can work with BacktestExecutor."""
        from trading.backtest import BacktestExecutor

        config = BotConfig(use_mock=True)

        with patch('trading.bot.MockMT5Connector', return_value=BacktestExecutor()):
            with patch('trading.bot.RiskManager') as MockRiskMgr:
                with patch('trading.bot.NeuralHybridStrategy') as MockStrategy:
                    bot = TradingBot(config)
                    assert bot.connector is not None

    def test_backtest_bot_data_provider(self):
        """Test BacktestBot acts as data provider."""
        dates = pd.date_range('2024-01-01', periods=200, freq='h')
        data = pd.DataFrame({
            'time': dates,
            'close': [1.1000] * 200
        })

        mock_strategy_class = Mock()

        bot = BacktestBot(data, mock_strategy_class)

        # Bot should be passed as data_provider
        call_kwargs = mock_strategy_class.call_args[1]
        assert call_kwargs['data_provider'] == bot
