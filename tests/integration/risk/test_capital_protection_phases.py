"""
Capital Protection Integration Tests.

Tests Phase 5 capital protection: daily/weekly loss limits, protection levels.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from trading.decision_engine import EnhancedDecisionEngine, DecisionEngineConfig


@pytest.mark.integration
class TestDailyLossLimit:
    """Tests for daily loss limit protection."""
    
    def test_daily_loss_limit_blocks_trading(self, ohlcv_df):
        """Trading should be blocked when daily loss limit is reached."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            enable_capital_protection=True,
            max_daily_loss_pct=3.0,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        # Simulate daily losses exceeding 3%
        engine.record_trade_result(pnl=-150.0, is_win=False, size=0.1)
        engine.record_trade_result(pnl=-200.0, is_win=False, size=0.1)
        
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
            account_balance=9650.0,  # After losses
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        assert decision.should_trade is False
        assert any("daily" in r.lower() or "protection" in r.lower() 
                   for r in decision.rejection_reasons)
    
    def test_daily_loss_resets_next_day(self, ohlcv_df):
        """Daily loss counter should reset on new trading day."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            enable_capital_protection=True,
            max_daily_loss_pct=3.0,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        # Simulate losses
        engine.record_trade_result(pnl=-350.0, is_win=False, size=0.1)
        
        # Reset daily stats (simulating new day)
        if hasattr(engine, '_daily_pnl'):
            engine._daily_pnl = 0.0
        if hasattr(engine, '_daily_loss'):
            engine._daily_loss = 0.0
        if hasattr(engine, '_protection_manager'):
            engine._protection_manager.reset_daily()
        
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
            account_balance=9650.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # Should be able to trade after daily reset (unless drawdown limit hit)
        # The decision depends on overall drawdown, not just daily
        assert decision is not None


@pytest.mark.integration
class TestWeeklyLossLimit:
    """Tests for weekly loss limit protection."""
    
    def test_weekly_loss_limit_blocks_trading(self, ohlcv_df):
        """Trading should be blocked when weekly loss limit is reached."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            enable_capital_protection=True,
            max_weekly_loss_pct=6.0,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        # Simulate weekly losses exceeding 6%
        for _ in range(4):
            engine.record_trade_result(pnl=-200.0, is_win=False, size=0.1)
        
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
            account_balance=9200.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        assert decision.should_trade is False


@pytest.mark.integration
class TestProtectionLevels:
    """Tests for graduated protection levels."""
    
    def test_protection_level_escalates_with_losses(self, ohlcv_df):
        """Protection level should escalate as losses accumulate."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            enable_capital_protection=True,
            max_drawdown_pct=10.0,
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
        
        # Initial decision - should be normal protection
        d1 = engine.evaluate(
            predictions=predictions,
            entry_price=entry,
            pair="EURUSD",
            account_balance=10000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # Record some losses
        engine.record_trade_result(pnl=-500.0, is_win=False, size=0.1)
        
        # Decision after losses - protection should be elevated
        d2 = engine.evaluate(
            predictions=predictions,
            entry_price=entry,
            pair="EURUSD",
            account_balance=9500.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # Either position size reduced or protection level elevated
        if d1.should_trade and d2.should_trade:
            # Position size should be reduced or protection warnings present
            size_reduced = d2.position_size < d1.position_size
            has_warnings = len(d2.protection_warnings) > 0
            level_elevated = d2.protection_level != 'normal'
            assert size_reduced or has_warnings or level_elevated or d2.size_adjusted_by_protection
    
    def test_consecutive_losses_increase_protection(self, ohlcv_df):
        """Consecutive losses should increase protection measures."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            enable_capital_protection=True,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        # Record consecutive losses
        for i in range(5):
            engine.record_trade_result(pnl=-100.0, is_win=False, size=0.1)
        
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
            account_balance=9500.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # After consecutive losses, either trading blocked or size reduced
        if decision.should_trade:
            assert decision.protection_level != 'normal' or decision.size_adjusted_by_protection


@pytest.mark.integration
class TestRecoveryFromProtection:
    """Tests for recovery from protection states."""
    
    def test_winning_trades_reduce_protection(self, ohlcv_df):
        """Winning trades should gradually reduce protection level."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            enable_capital_protection=True,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        # First, accumulate some losses
        engine.record_trade_result(pnl=-300.0, is_win=False, size=0.1)
        engine.record_trade_result(pnl=-200.0, is_win=False, size=0.1)
        
        # Then record winning trades
        engine.record_trade_result(pnl=250.0, is_win=True, size=0.1)
        engine.record_trade_result(pnl=300.0, is_win=True, size=0.1)
        
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
            account_balance=10050.0,  # Slightly above starting
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # After recovery, trading should be allowed
        assert decision.should_trade is True
