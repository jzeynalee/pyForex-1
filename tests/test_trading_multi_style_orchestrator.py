# tests/test_trading_multi_style_orchestrator.py
"""
Unit tests for trading/multi_style_orchestrator.py - Multi-Style Trading Orchestrator.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from trading.multi_style_orchestrator import (
    MultiStyleOrchestrator, SignalEvent, ExecutionResult
)
from trading.style_config import TradingStyle, OrchestratorConfig


@pytest.mark.unit
class TestSignalEvent:
    """Test SignalEvent dataclass."""

    def test_signal_event_creation(self):
        """Test creating SignalEvent."""
        event = SignalEvent(
            style=TradingStyle.INTRADAY,
            signal="BUY",
            confidence=0.75,
            timestamp=datetime.now(),
            trade_params={'volume': 0.1}
        )

        assert event.style == TradingStyle.INTRADAY
        assert event.signal == "BUY"
        assert event.confidence == 0.75


@pytest.mark.unit
class TestExecutionResult:
    """Test ExecutionResult dataclass."""

    def test_execution_result_creation(self):
        """Test creating ExecutionResult."""
        result = ExecutionResult(
            style=TradingStyle.SCALP,
            success=True,
            ticket=12345,
            error=None,
            trade_params={'volume': 0.1}
        )

        assert result.style == TradingStyle.SCALP
        assert result.success is True
        assert result.ticket == 12345


@pytest.mark.unit
class TestMultiStyleOrchestrator:
    """Test MultiStyleOrchestrator class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock orchestrator config."""
        config = OrchestratorConfig()
        return config

    @pytest.fixture
    def mock_connector(self):
        """Create a mock connector."""
        connector = Mock()
        connector.connect.return_value = True
        connector.get_account_info.return_value = Mock(balance=10000.0)
        return connector

    def test_init(self, mock_config, mock_connector):
        """Test initialization."""
        with patch('trading.multi_style_orchestrator.MultiTimeframeDataManager') as MockData, \
             patch('trading.multi_style_orchestrator.PositionCoordinator') as MockCoord, \
             patch('trading.multi_style_orchestrator.RiskManager') as MockRisk, \
             patch('trading.multi_style_orchestrator.create_style_strategy') as MockStrategy:
            
            MockData.return_value = Mock()
            MockCoord.return_value = Mock()
            MockRisk.return_value = Mock()
            MockStrategy.return_value = Mock()

            orchestrator = MultiStyleOrchestrator(
                config=mock_config,
                connector=mock_connector
            )

            assert orchestrator.config == mock_config
            assert orchestrator.connector == mock_connector

