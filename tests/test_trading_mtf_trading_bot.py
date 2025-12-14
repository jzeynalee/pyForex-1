# tests/test_trading_mtf_trading_bot.py
"""
Unit tests for trading/mtf_trading_bot.py - Multi-Timeframe Trading Bot.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from trading.mtf_trading_bot import MTFTradingBot, MTFBotConfig


@pytest.mark.unit
class TestMTFBotConfig:
    """Test MTFBotConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MTFBotConfig()

        assert config.symbol == "EURUSD"
        assert config.profile == "SWING"
        assert config.tick_interval == 10.0
        assert config.use_mock is False
        assert config.enable_trading is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = MTFBotConfig(
            symbol="GBPUSD",
            profile="SCALP",
            tick_interval=5.0,
            use_mock=True
        )

        assert config.symbol == "GBPUSD"
        assert config.profile == "SCALP"
        assert config.tick_interval == 5.0
        assert config.use_mock is True


@pytest.mark.unit
class TestMTFTradingBot:
    """Test MTFTradingBot class."""

    @pytest.fixture
    def mock_connector(self):
        """Create a mock connector."""
        connector = Mock()
        connector.connect.return_value = True
        connector.get_account_info.return_value = Mock(balance=10000.0)
        connector.get_data.return_value = None
        return connector

    def test_init_with_mock_connector(self, mock_connector):
        """Test initialization with mock connector."""
        config = MTFBotConfig(use_mock=True)

        with patch('trading.mtf_trading_bot.MockMT5Connector', return_value=mock_connector), \
             patch('trading.mtf_trading_bot.MTFDataProvider') as MockProvider, \
             patch('trading.mtf_trading_bot.MTFTrendDetector') as MockDetector, \
             patch('trading.mtf_trading_bot.get_profile') as MockProfile, \
             patch('trading.mtf_trading_bot.RiskManager') as MockRisk:
            
            MockProvider.return_value = Mock()
            MockDetector.return_value = Mock()
            MockProfile.return_value = Mock()
            MockRisk.return_value = Mock()

            bot = MTFTradingBot(config, connector=mock_connector)

            assert bot.config == config
            assert bot.connector == mock_connector
            assert bot.running is False

    def test_init_with_real_connector(self, mock_connector):
        """Test initialization with real connector."""
        config = MTFBotConfig(use_mock=False)

        with patch('trading.mtf_trading_bot.MT5Connector') as MockMT5, \
             patch('trading.mtf_trading_bot.settings') as mock_settings, \
             patch('trading.mtf_trading_bot.MTFDataProvider'), \
             patch('trading.mtf_trading_bot.MTFTrendDetector'), \
             patch('trading.mtf_trading_bot.get_profile'), \
             patch('trading.mtf_trading_bot.RiskManager'):
            
            mock_settings.MT5_ACCOUNT = 12345
            mock_settings.MT5_PASSWORD = "pass"
            mock_settings.MT5_SERVER = "server"
            mock_settings.MT5_PATH = "path"
            mock_settings.MAGIC_NUMBER = 123
            
            MockMT5.return_value = mock_connector

            bot = MTFTradingBot(config)

            assert bot.config == config

    def test_get_status(self, mock_connector):
        """Test getting bot status."""
        config = MTFBotConfig(use_mock=True)

        with patch('trading.mtf_trading_bot.MockMT5Connector', return_value=mock_connector), \
             patch('trading.mtf_trading_bot.MTFDataProvider'), \
             patch('trading.mtf_trading_bot.MTFTrendDetector'), \
             patch('trading.mtf_trading_bot.get_profile'), \
             patch('trading.mtf_trading_bot.RiskManager'):
            
            bot = MTFTradingBot(config, connector=mock_connector)
            bot.running = True
            bot.iteration_count = 5

            status = bot.get_status()

            assert status['running'] is True
            assert status['iteration_count'] == 5

    def test_stop(self, mock_connector):
        """Test stopping the bot."""
        config = MTFBotConfig(use_mock=True)

        with patch('trading.mtf_trading_bot.MockMT5Connector', return_value=mock_connector), \
             patch('trading.mtf_trading_bot.MTFDataProvider'), \
             patch('trading.mtf_trading_bot.MTFTrendDetector'), \
             patch('trading.mtf_trading_bot.get_profile'), \
             patch('trading.mtf_trading_bot.RiskManager'):
            
            bot = MTFTradingBot(config, connector=mock_connector)
            bot.running = True

            bot.stop()

            assert bot.running is False

