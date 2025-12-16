# tests/test_trading_prop_firm_bot.py
"""
Unit tests for trading/prop_firm_bot.py - Prop Firm Trading Bot.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from trading.prop_firm_bot import (
    PropFirmTradingBot, PropFirmBotConfig, PropFirmBotState
)


@pytest.mark.unit
class TestPropFirmBotState:
    """Test PropFirmBotState enum."""

    def test_state_values(self):
        """Test prop firm bot state enum values."""
        assert PropFirmBotState.STOPPED.value == "stopped"
        assert PropFirmBotState.CHALLENGE_PASSED.value == "challenge_passed"
        assert PropFirmBotState.CHALLENGE_FAILED.value == "challenge_failed"


@pytest.mark.unit
class TestPropFirmBotConfig:
    """Test PropFirmBotConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PropFirmBotConfig()

        assert config.firm == 'FTMO'
        assert config.account_size == 100000
        assert config.phase == 'evaluation'
        assert config.symbol == 'EURUSD'
        assert config.conservative_mode is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = PropFirmBotConfig(
            firm='MYFUNDED',
            account_size=50000,
            phase='funded',
            symbol='GBPUSD',
            conservative_mode=False
        )

        assert config.firm == 'MYFUNDED'
        assert config.account_size == 50000
        assert config.phase == 'funded'
        assert config.symbol == 'GBPUSD'
        assert config.conservative_mode is False


@pytest.mark.unit
class TestPropFirmTradingBot:
    """Test PropFirmTradingBot class."""

    @pytest.fixture
    def mock_data_provider(self):
        """Create a mock data provider."""
        return Mock()

    @pytest.fixture
    def mock_executor(self):
        """Create a mock executor."""
        executor = Mock()
        executor.get_account_balance.return_value = 100000.0
        return executor

    def test_init(self, mock_data_provider, mock_executor):
        """Test initialization."""
        config = PropFirmBotConfig()

        with patch('trading.prop_firm_bot.get_prop_firm_config') as MockConfig, \
             patch('trading.prop_firm_bot.PropFirmMonitor') as MockMonitor:
            
            MockConfig.return_value = Mock()
            MockMonitor.return_value = Mock()

            bot = PropFirmTradingBot(
                config=config,
                data_provider=mock_data_provider,
                executor=mock_executor
            )

            assert bot.config == config
            assert bot.data_provider == mock_data_provider
            assert bot.executor == mock_executor
            assert bot.state == PropFirmBotState.STOPPED

    def test_get_state(self, mock_data_provider, mock_executor):
        """Test getting bot state."""
        config = PropFirmBotConfig()

        with patch('trading.prop_firm_bot.get_prop_firm_config'), \
             patch('trading.prop_firm_bot.PropFirmMonitor'):
            
            bot = PropFirmTradingBot(config, mock_data_provider, mock_executor)
            bot.state = PropFirmBotState.RUNNING

            state = bot.get_state()

            assert state == PropFirmBotState.RUNNING

