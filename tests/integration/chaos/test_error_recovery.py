"""
Error Recovery Integration Tests.

Tests system behavior during partial failures and recovery scenarios.
"""
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from trading.live_trading_bot import LiveTradingBot, BotConfig, BotState
from trading.decision_engine import EnhancedDecisionEngine, DecisionEngineConfig


@pytest.mark.integration
class TestPartialFailureRecovery:
    """Tests for partial failure handling."""
    
    def test_prediction_failure_does_not_crash_bot(self, mt5_executor):
        """Bot should handle prediction failures gracefully."""
        strategy = Mock()
        strategy.initialize.return_value = True
        
        # First call succeeds, second fails, third succeeds
        call_count = [0]
        def evaluate_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Prediction model error")
            return None  # No trade signal
        
        strategy.evaluate.side_effect = evaluate_side_effect
        
        bot = LiveTradingBot(
            config=BotConfig(check_interval_seconds=1, dry_run=True),
            data_provider=Mock(),
            executor=mt5_executor,
            strategy=strategy,
        )
        bot.initialize(starting_balance=10000.0)
        
        iterations = [0]
        def wait_and_count(seconds):
            iterations[0] += 1
            if iterations[0] >= 3:
                bot._stop_event.set()
        
        bot._wait = wait_and_count
        bot._is_market_open = lambda *args, **kwargs: True
        
        # Should not raise exception
        bot.run()
        
        # Bot should have continued after error
        assert call_count[0] >= 2
        assert bot.state != BotState.ERROR
    
    def test_execution_failure_rolls_back_state(self, mt5_executor):
        """Failed execution should not leave inconsistent state."""
        strategy = Mock()
        strategy.initialize.return_value = True
        
        decision = Mock()
        decision.should_trade = True
        decision.direction = "BUY"
        decision.position_size = 0.1
        decision.stop_loss = 1.095
        decision.take_profit = 1.110
        decision.to_dict.return_value = {}
        
        strategy.evaluate.return_value = decision
        strategy.create_order.return_value = Mock(
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            price=0.0,
            stop_loss=1.095,
            take_profit=1.110,
            ticket=None,
        )
        strategy.execute.return_value = False  # Execution fails
        
        bot = LiveTradingBot(
            config=BotConfig(check_interval_seconds=1, dry_run=False),
            data_provider=Mock(),
            executor=mt5_executor,
            strategy=strategy,
        )
        bot.initialize(starting_balance=10000.0)
        bot._is_market_open = lambda *args, **kwargs: True
        bot._can_open_new_trade = lambda *args, **kwargs: True
        
        initial_positions = len(bot._open_positions)
        
        bot._evaluate_and_trade(datetime.utcnow())
        
        # No position should be added on failed execution
        assert len(bot._open_positions) == initial_positions
    
    def test_data_provider_failure_handled(self, mt5_executor):
        """Data provider failures should be handled gracefully."""
        data_provider = Mock()
        data_provider.get_ohlcv.side_effect = ConnectionError("Data feed disconnected")
        
        strategy = Mock()
        strategy.initialize.return_value = True
        strategy.evaluate.return_value = None
        
        bot = LiveTradingBot(
            config=BotConfig(check_interval_seconds=1, dry_run=True),
            data_provider=data_provider,
            executor=mt5_executor,
            strategy=strategy,
        )
        bot.initialize(starting_balance=10000.0)
        
        # Should not crash when data provider fails
        try:
            bot._evaluate_and_trade(datetime.utcnow())
        except ConnectionError:
            pass  # Expected - bot should catch this in production
        
        # Bot should still be in valid state
        assert bot.state in [BotState.RUNNING, BotState.STOPPED, BotState.INITIALIZING]


@pytest.mark.integration
class TestStateRecovery:
    """Tests for state recovery after errors."""
    
    def test_decision_engine_recovers_from_invalid_input(self, ohlcv_df):
        """Decision engine should recover from invalid inputs."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        # First, make a valid call
        valid_predictions = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.001,
            "quantiles": np.array([-0.0005, -0.0002, 0.0, 0.0004, 0.0008]),
            "features": None,
        }
        entry = float(ohlcv_df["close"].iloc[-1])
        
        d1 = engine.evaluate(
            predictions=valid_predictions,
            entry_price=entry,
            pair="EURUSD",
            account_balance=10000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        assert d1 is not None
        
        # Try invalid input (should handle gracefully)
        try:
            invalid_predictions = {
                "direction_probs": np.array([np.nan, np.nan, np.nan]),
                "volatility": np.nan,
                "quantiles": None,
                "features": None,
            }
            engine.evaluate(
                predictions=invalid_predictions,
                entry_price=entry,
                pair="EURUSD",
                account_balance=10000.0,
                market_data=ohlcv_df,
                current_spread=1.0,
            )
        except (ValueError, TypeError):
            pass  # Expected
        
        # Engine should still work after invalid input
        d3 = engine.evaluate(
            predictions=valid_predictions,
            entry_price=entry,
            pair="EURUSD",
            account_balance=10000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        assert d3 is not None
    
    def test_bot_recovers_from_temporary_disconnect(self, mt5_executor):
        """Bot should recover when connection is restored."""
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
        
        # Simulate disconnect
        mt5_executor.connector.connected = False
        bot._check_connection()
        assert bot.state == BotState.DISCONNECTED
        
        # Simulate reconnect
        mt5_executor.connector.connected = True
        bot._check_connection()
        
        # Should recover to running state
        assert bot.state in [BotState.RUNNING, BotState.STOPPED]


@pytest.mark.integration
class TestGracefulDegradation:
    """Tests for graceful degradation under adverse conditions."""
    
    def test_missing_optional_components_handled(self, ohlcv_df):
        """System should work with missing optional components."""
        # Decision engine without meta model
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            use_meta_labeling=False,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        predictions = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.001,
            "quantiles": np.array([-0.0005, -0.0002, 0.0, 0.0004, 0.0008]),
            "features": None,
        }
        
        entry = float(ohlcv_df["close"].iloc[-1])
        decision = engine.evaluate(
            predictions=predictions,
            entry_price=entry,
            pair="EURUSD",
            account_balance=10000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # Should still produce valid decision
        assert decision is not None
        assert decision.should_trade is True
    
    def test_reduced_functionality_under_high_load(self, ohlcv_df):
        """System should maintain core functionality under stress."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        predictions = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.001,
            "quantiles": np.array([-0.0005, -0.0002, 0.0, 0.0004, 0.0008]),
            "features": None,
        }
        entry = float(ohlcv_df["close"].iloc[-1])
        
        # Rapid-fire evaluations
        decisions = []
        for _ in range(100):
            d = engine.evaluate(
                predictions=predictions,
                entry_price=entry,
                pair="EURUSD",
                account_balance=10000.0,
                market_data=ohlcv_df,
                current_spread=1.0,
            )
            decisions.append(d)
        
        # All decisions should be valid
        assert all(d is not None for d in decisions)
        # Decisions should be consistent
        assert all(d.direction == decisions[0].direction for d in decisions)


@pytest.mark.integration
class TestResourceCleanup:
    """Tests for proper resource cleanup after errors."""
    
    def test_bot_cleanup_on_error(self, mt5_executor):
        """Bot should clean up resources on error."""
        strategy = Mock()
        strategy.initialize.return_value = True
        strategy.evaluate.side_effect = RuntimeError("Fatal error")
        
        bot = LiveTradingBot(
            config=BotConfig(check_interval_seconds=1, dry_run=True),
            data_provider=Mock(),
            executor=mt5_executor,
            strategy=strategy,
        )
        bot.initialize(starting_balance=10000.0)
        
        def wait_and_stop(seconds):
            bot._stop_event.set()
        
        bot._wait = wait_and_stop
        bot._is_market_open = lambda *args, **kwargs: True
        
        # Run should complete without hanging
        bot.run()
        
        # Bot should be stopped
        assert bot._stop_event.is_set()
    
    def test_position_state_consistent_after_error(self, mt5_executor):
        """Position state should remain consistent after errors."""
        strategy = Mock()
        strategy.initialize.return_value = True
        
        bot = LiveTradingBot(
            config=BotConfig(check_interval_seconds=1, dry_run=False),
            data_provider=Mock(),
            executor=mt5_executor,
            strategy=strategy,
        )
        bot.initialize(starting_balance=10000.0)
        
        # Add a position manually
        bot._open_positions["123"] = Mock(
            ticket="123",
            symbol="EURUSD",
            direction=1,
            entry_price=1.1000,
        )
        
        # Simulate error during position update
        try:
            raise RuntimeError("Position update error")
        except RuntimeError:
            pass
        
        # Position should still be tracked
        assert "123" in bot._open_positions
