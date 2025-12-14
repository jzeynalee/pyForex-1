# tests/test_trading_live_trading_bot.py
"""
Unit tests for trading/live_trading_bot.py - Live Trading Bot with Risk Management.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from trading.live_trading_bot import (
    LiveTradingBot, BotConfig, BotState, OpenPosition
)


@pytest.mark.unit
class TestBotState:
    """Test BotState enum."""

    def test_bot_state_values(self):
        """Test bot state enum values."""
        assert BotState.STOPPED.value == "stopped"
        assert BotState.RUNNING.value == "running"
        assert BotState.PROTECTION_HALT.value == "protection_halt"


@pytest.mark.unit
class TestOpenPosition:
    """Test OpenPosition dataclass."""

    def test_open_position_creation(self):
        """Test creating OpenPosition."""
        position = OpenPosition(
            ticket="12345",
            symbol="EURUSD",
            direction=1,
            entry_price=1.1000,
            entry_time=datetime.now(),
            volume=0.1,
            stop_loss=1.0950,
            take_profit=1.1100
        )

        assert position.ticket == "12345"
        assert position.symbol == "EURUSD"
        assert position.direction == 1
        assert position.current_pnl == 0.0


@pytest.mark.unit
class TestBotConfig:
    """Test BotConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BotConfig()

        assert config.symbol == 'EURUSD'
        assert config.profile == 'INTRADAY'
        assert config.check_interval_seconds == 60
        assert config.base_risk_percent == 1.0
        assert config.enable_capital_protection is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = BotConfig(
            symbol='GBPUSD',
            profile='SWING',
            check_interval_seconds=120,
            dry_run=True
        )

        assert config.symbol == 'GBPUSD'
        assert config.profile == 'SWING'
        assert config.check_interval_seconds == 120
        assert config.dry_run is True


@pytest.mark.unit
class TestLiveTradingBot:
    """Test LiveTradingBot class."""

    @pytest.fixture
    def mock_data_provider(self):
        """Create a mock data provider."""
        provider = Mock()
        return provider

    @pytest.fixture
    def mock_executor(self):
        """Create a mock executor."""
        executor = Mock()
        executor.get_account_balance.return_value = 10000.0
        return executor

    def test_init_default(self, mock_data_provider, mock_executor):
        """Test default initialization."""
        config = BotConfig()

        bot = LiveTradingBot(
            config=config,
            data_provider=mock_data_provider,
            executor=mock_executor
        )

        assert bot.config == config
        assert bot.data_provider == mock_data_provider
        assert bot.executor == mock_executor
        assert bot.state == BotState.STOPPED

    def test_get_state(self, mock_data_provider, mock_executor):
        """Test getting bot state."""
        config = BotConfig()
        bot = LiveTradingBot(config, mock_data_provider, mock_executor)
        bot.state = BotState.RUNNING

        state = bot.get_state()

        assert state == BotState.RUNNING

