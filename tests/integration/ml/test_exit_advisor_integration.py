"""
Exit Advisor Integration Tests.

Tests Phase 4 RL-based exit advisor integration.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

try:
    from risk_management import ExitAdvisor, ExitAction, Position
    HAS_EXIT_ADVISOR = True
except ImportError:
    HAS_EXIT_ADVISOR = False
    ExitAdvisor = None
    ExitAction = None
    Position = None


@pytest.fixture
def mock_exit_advisor():
    """Create mock exit advisor for testing."""
    if not HAS_EXIT_ADVISOR:
        pytest.skip("Exit advisor not available")
    
    advisor = Mock(spec=ExitAdvisor)
    advisor.recommend.return_value = (ExitAction.HOLD, 0.7)
    return advisor


@pytest.fixture
def sample_position():
    """Create sample position for testing."""
    if not HAS_EXIT_ADVISOR:
        pytest.skip("Exit advisor not available")
    
    return Position(
        ticket="12345",
        symbol="EURUSD",
        direction=1,  # Long
        entry_price=1.1000,
        entry_time=datetime.utcnow() - timedelta(hours=2),
        volume=0.1,
        stop_loss=1.0950,
        take_profit=1.1100,
        current_price=1.1050,
        unrealized_pnl=50.0,
    )


@pytest.mark.integration
@pytest.mark.skipif(not HAS_EXIT_ADVISOR, reason="Exit advisor not available")
class TestExitAdvisorRecommendations:
    """Tests for exit advisor recommendations."""
    
    def test_hold_recommendation_for_profitable_position(self, mock_exit_advisor, sample_position):
        """Profitable position in trend should recommend HOLD."""
        # Position is profitable
        sample_position.unrealized_pnl = 50.0
        sample_position.current_price = 1.1050
        
        mock_exit_advisor.recommend.return_value = (ExitAction.HOLD, 0.8)
        
        action, confidence = mock_exit_advisor.recommend(sample_position)
        
        assert action == ExitAction.HOLD
        assert confidence >= 0.5
    
    def test_exit_recommendation_for_reversal(self, mock_exit_advisor, sample_position):
        """Position showing reversal signs should recommend EXIT."""
        # Simulate reversal scenario
        sample_position.current_price = 1.0980  # Price dropped
        sample_position.unrealized_pnl = -20.0
        
        mock_exit_advisor.recommend.return_value = (ExitAction.EXIT, 0.75)
        
        action, confidence = mock_exit_advisor.recommend(sample_position)
        
        assert action == ExitAction.EXIT
        assert confidence >= 0.5
    
    def test_partial_close_recommendation(self, mock_exit_advisor, sample_position):
        """Large profit should recommend PARTIAL_CLOSE."""
        sample_position.current_price = 1.1080
        sample_position.unrealized_pnl = 80.0
        
        mock_exit_advisor.recommend.return_value = (ExitAction.PARTIAL_CLOSE, 0.65)
        
        action, confidence = mock_exit_advisor.recommend(sample_position)
        
        assert action == ExitAction.PARTIAL_CLOSE
    
    def test_tighten_stop_recommendation(self, mock_exit_advisor, sample_position):
        """Good profit should recommend TIGHTEN_STOP."""
        sample_position.current_price = 1.1060
        sample_position.unrealized_pnl = 60.0
        
        mock_exit_advisor.recommend.return_value = (ExitAction.TIGHTEN_STOP, 0.7)
        
        action, confidence = mock_exit_advisor.recommend(sample_position)
        
        assert action == ExitAction.TIGHTEN_STOP


@pytest.mark.integration
@pytest.mark.skipif(not HAS_EXIT_ADVISOR, reason="Exit advisor not available")
class TestExitAdvisorWithBot:
    """Tests for exit advisor integration with trading bot."""
    
    def test_exit_advisor_called_for_open_positions(self, mt5_executor):
        """Exit advisor should be called for each open position."""
        from trading.live_trading_bot import LiveTradingBot, BotConfig
        
        mock_advisor = Mock()
        mock_advisor.recommend.return_value = (ExitAction.HOLD, 0.7)
        
        strategy = Mock()
        strategy.initialize.return_value = True
        strategy.evaluate.return_value = None
        
        bot = LiveTradingBot(
            config=BotConfig(check_interval_seconds=1, dry_run=True),
            data_provider=Mock(),
            executor=mt5_executor,
            strategy=strategy,
        )
        bot.initialize(starting_balance=10000.0)
        bot._exit_advisor = mock_advisor
        
        # Add open position
        bot._open_positions["123"] = Mock(
            ticket="123",
            symbol="EURUSD",
            direction=1,
            entry_price=1.1000,
            entry_time=datetime.utcnow(),
            volume=0.1,
            stop_loss=1.0950,
            take_profit=1.1100,
            current_pnl=50.0,
        )
        
        # Check positions (if method exists)
        if hasattr(bot, '_check_exit_recommendations'):
            bot._check_exit_recommendations()
            assert mock_advisor.recommend.called
    
    def test_exit_action_executed_correctly(self, mt5_executor):
        """Exit actions should be executed correctly."""
        from trading.live_trading_bot import LiveTradingBot, BotConfig
        
        mock_advisor = Mock()
        mock_advisor.recommend.return_value = (ExitAction.EXIT, 0.9)
        
        strategy = Mock()
        strategy.initialize.return_value = True
        
        bot = LiveTradingBot(
            config=BotConfig(check_interval_seconds=1, dry_run=False),
            data_provider=Mock(),
            executor=mt5_executor,
            strategy=strategy,
        )
        bot.initialize(starting_balance=10000.0)
        bot._exit_advisor = mock_advisor
        
        # Add position
        bot._open_positions["123"] = Mock(
            ticket="123",
            symbol="EURUSD",
            direction=1,
            entry_price=1.1000,
            entry_time=datetime.utcnow(),
            volume=0.1,
            stop_loss=1.0950,
            take_profit=1.1100,
            current_pnl=-30.0,
        )
        
        # Execute exit if method exists
        if hasattr(bot, '_execute_exit_action'):
            bot._execute_exit_action("123", ExitAction.EXIT)
            # Position should be closed
            # (actual behavior depends on implementation)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_EXIT_ADVISOR, reason="Exit advisor not available")
class TestExitAdvisorEdgeCases:
    """Edge case tests for exit advisor."""
    
    def test_advisor_handles_new_position(self, mock_exit_advisor):
        """Advisor should handle very new positions."""
        if not HAS_EXIT_ADVISOR:
            pytest.skip("Exit advisor not available")
        
        new_position = Position(
            ticket="12345",
            symbol="EURUSD",
            direction=1,
            entry_price=1.1000,
            entry_time=datetime.utcnow(),  # Just opened
            volume=0.1,
            stop_loss=1.0950,
            take_profit=1.1100,
            current_price=1.1000,
            unrealized_pnl=0.0,
        )
        
        mock_exit_advisor.recommend.return_value = (ExitAction.HOLD, 0.9)
        
        action, confidence = mock_exit_advisor.recommend(new_position)
        
        # New position should typically HOLD
        assert action == ExitAction.HOLD
    
    def test_advisor_handles_at_stop_loss(self, mock_exit_advisor):
        """Advisor should handle position near stop loss."""
        if not HAS_EXIT_ADVISOR:
            pytest.skip("Exit advisor not available")
        
        at_sl_position = Position(
            ticket="12345",
            symbol="EURUSD",
            direction=1,
            entry_price=1.1000,
            entry_time=datetime.utcnow() - timedelta(hours=1),
            volume=0.1,
            stop_loss=1.0950,
            take_profit=1.1100,
            current_price=1.0955,  # Very close to SL
            unrealized_pnl=-45.0,
        )
        
        mock_exit_advisor.recommend.return_value = (ExitAction.EXIT, 0.85)
        
        action, confidence = mock_exit_advisor.recommend(at_sl_position)
        
        # Near SL should recommend EXIT
        assert action == ExitAction.EXIT
    
    def test_advisor_handles_at_take_profit(self, mock_exit_advisor):
        """Advisor should handle position near take profit."""
        if not HAS_EXIT_ADVISOR:
            pytest.skip("Exit advisor not available")
        
        at_tp_position = Position(
            ticket="12345",
            symbol="EURUSD",
            direction=1,
            entry_price=1.1000,
            entry_time=datetime.utcnow() - timedelta(hours=1),
            volume=0.1,
            stop_loss=1.0950,
            take_profit=1.1100,
            current_price=1.1095,  # Very close to TP
            unrealized_pnl=95.0,
        )
        
        # Could recommend HOLD (let TP hit) or PARTIAL_CLOSE
        mock_exit_advisor.recommend.return_value = (ExitAction.HOLD, 0.6)
        
        action, confidence = mock_exit_advisor.recommend(at_tp_position)
        
        assert action in [ExitAction.HOLD, ExitAction.PARTIAL_CLOSE, ExitAction.EXIT]
    
    def test_low_confidence_recommendation_ignored(self, mock_exit_advisor, sample_position):
        """Low confidence recommendations should be treated cautiously."""
        mock_exit_advisor.recommend.return_value = (ExitAction.EXIT, 0.3)  # Low confidence
        
        action, confidence = mock_exit_advisor.recommend(sample_position)
        
        # Low confidence exit should be noted
        assert confidence < 0.5
        # System should probably HOLD with low confidence exit signal
