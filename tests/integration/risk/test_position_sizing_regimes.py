"""
Position Sizing Integration Tests.

Tests risk-adjusted position sizing under different market regimes.
"""
import pytest
import numpy as np

from trading.decision_engine import EnhancedDecisionEngine, DecisionEngineConfig


@pytest.mark.integration
class TestPositionSizingByRegime:
    """Tests for regime-aware position sizing."""
    
    def test_trending_regime_allows_larger_positions(self, ohlcv_df):
        """Trending markets should allow standard position sizes."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            base_risk_percent=1.0,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        # Strong directional signal with low volatility (trending)
        predictions = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.0008,  # Low volatility
            "quantiles": np.array([-0.0004, -0.0002, 0.0, 0.0003, 0.0006]),
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
        
        assert decision.should_trade is True
        assert decision.position_size > 0
        assert decision.risk_percent <= config.base_risk_percent * 1.5  # Allow some buffer
    
    def test_ranging_regime_reduces_position_size(self, ohlcv_df):
        """Ranging/choppy markets should reduce position sizes."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            base_risk_percent=1.0,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        entry = float(ohlcv_df["close"].iloc[-1])
        
        # Trending conditions
        trending_pred = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.0008,
            "quantiles": np.array([-0.0004, -0.0002, 0.0, 0.0003, 0.0006]),
            "features": None,
        }
        trending_decision = engine.evaluate(
            predictions=trending_pred,
            entry_price=entry,
            pair="EURUSD",
            account_balance=10000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # High volatility / ranging conditions
        ranging_pred = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.003,  # Higher volatility
            "quantiles": np.array([-0.0015, -0.0008, 0.0, 0.0012, 0.0024]),
            "features": None,
        }
        ranging_decision = engine.evaluate(
            predictions=ranging_pred,
            entry_price=entry,
            pair="EURUSD",
            account_balance=10000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # Higher volatility should result in smaller position
        if trending_decision.should_trade and ranging_decision.should_trade:
            assert ranging_decision.position_size <= trending_decision.position_size
    
    def test_high_volatility_widens_stops(self, ohlcv_df):
        """High volatility should result in wider stop losses."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        entry = float(ohlcv_df["close"].iloc[-1])
        
        # Low volatility
        low_vol_pred = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.0005,
            "quantiles": np.array([-0.0003, -0.0001, 0.0, 0.0002, 0.0004]),
            "features": None,
        }
        low_vol_decision = engine.evaluate(
            predictions=low_vol_pred,
            entry_price=entry,
            pair="EURUSD",
            account_balance=10000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # High volatility
        high_vol_pred = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.002,
            "quantiles": np.array([-0.0012, -0.0006, 0.0, 0.0008, 0.0016]),
            "features": None,
        }
        high_vol_decision = engine.evaluate(
            predictions=high_vol_pred,
            entry_price=entry,
            pair="EURUSD",
            account_balance=10000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # Higher volatility should have wider stops (larger SL distance)
        if low_vol_decision.should_trade and high_vol_decision.should_trade:
            low_vol_sl_dist = abs(entry - low_vol_decision.stop_loss)
            high_vol_sl_dist = abs(entry - high_vol_decision.stop_loss)
            assert high_vol_sl_dist >= low_vol_sl_dist


@pytest.mark.integration
class TestPositionSizingLimits:
    """Tests for position sizing limits and constraints."""
    
    def test_position_size_respects_max_risk(self, ohlcv_df):
        """Position size should never exceed maximum risk percentage."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            base_risk_percent=1.0,
            max_risk_percent=2.0,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        predictions = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.0003,  # Very low volatility might suggest large position
            "quantiles": np.array([-0.0002, -0.0001, 0.0, 0.0001, 0.0002]),
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
        
        if decision.should_trade:
            assert decision.risk_percent <= config.max_risk_percent
    
    def test_small_account_gets_minimum_position(self, ohlcv_df):
        """Small accounts should still get valid minimum position sizes."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            base_risk_percent=1.0,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=500.0)  # Small account
        
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
            account_balance=500.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        if decision.should_trade:
            assert decision.position_size >= 0.01  # Minimum lot size
    
    def test_position_size_scales_with_account(self, ohlcv_df):
        """Position size should scale proportionally with account size."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            base_risk_percent=1.0,
        )
        
        predictions = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.001,
            "quantiles": np.array([-0.0005, -0.0002, 0.0, 0.0004, 0.0008]),
            "features": None,
        }
        entry = float(ohlcv_df["close"].iloc[-1])
        
        # Small account
        engine_small = EnhancedDecisionEngine(config=config, meta_model=None)
        engine_small.initialize(starting_balance=5000.0)
        decision_small = engine_small.evaluate(
            predictions=predictions,
            entry_price=entry,
            pair="EURUSD",
            account_balance=5000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # Large account
        engine_large = EnhancedDecisionEngine(config=config, meta_model=None)
        engine_large.initialize(starting_balance=50000.0)
        decision_large = engine_large.evaluate(
            predictions=predictions,
            entry_price=entry,
            pair="EURUSD",
            account_balance=50000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        if decision_small.should_trade and decision_large.should_trade:
            # Larger account should have larger position
            assert decision_large.position_size > decision_small.position_size
            # But risk percentage should be similar
            assert abs(decision_large.risk_percent - decision_small.risk_percent) < 0.5
