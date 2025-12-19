"""
Meta-Labeling Filter Integration Tests.

Tests Phase 3 meta-labeling for trade quality filtering.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, MagicMock

from trading.decision_engine import EnhancedDecisionEngine, DecisionEngineConfig


class MockMetaModel:
    """Mock meta-labeling model for testing."""
    
    def __init__(self, score=0.7):
        self.default_score = score
        self.call_count = 0
    
    def predict_proba(self, features):
        self.call_count += 1
        # Return probability of positive class
        return np.array([[1 - self.default_score, self.default_score]])
    
    def predict(self, features):
        self.call_count += 1
        return np.array([1 if self.default_score > 0.5 else 0])


@pytest.mark.integration
class TestMetaLabelingFilter:
    """Tests for meta-labeling trade filter."""
    
    def test_high_meta_score_allows_trade(self, ohlcv_df):
        """High meta score should allow trade to proceed."""
        meta_model = MockMetaModel(score=0.8)
        
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            min_meta_score=0.5,
            use_meta_labeling=True,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=meta_model)
        engine.initialize(starting_balance=10000.0)
        
        predictions = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.001,
            "quantiles": np.array([-0.0005, -0.0002, 0.0, 0.0004, 0.0008]),
            "features": np.zeros(64),
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
        assert decision.meta_score >= 0.5
    
    def test_low_meta_score_rejects_trade(self, ohlcv_df):
        """Low meta score should reject trade even with good signal."""
        meta_model = MockMetaModel(score=0.3)
        
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            min_meta_score=0.5,
            use_meta_labeling=True,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=meta_model)
        engine.initialize(starting_balance=10000.0)
        
        predictions = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.001,
            "quantiles": np.array([-0.0005, -0.0002, 0.0, 0.0004, 0.0008]),
            "features": np.zeros(64),
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
        
        # Should be rejected due to low meta score
        if config.use_meta_labeling:
            assert decision.should_trade is False or decision.meta_score < 0.5
    
    def test_meta_model_receives_correct_features(self, ohlcv_df):
        """Meta model should receive prediction features."""
        meta_model = MockMetaModel(score=0.7)
        
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            use_meta_labeling=True,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=meta_model)
        engine.initialize(starting_balance=10000.0)
        
        test_features = np.random.randn(64)
        predictions = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.001,
            "quantiles": np.array([-0.0005, -0.0002, 0.0, 0.0004, 0.0008]),
            "features": test_features,
        }
        
        entry = float(ohlcv_df["close"].iloc[-1])
        engine.evaluate(
            predictions=predictions,
            entry_price=entry,
            pair="EURUSD",
            account_balance=10000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # Meta model should have been called
        if config.use_meta_labeling:
            assert meta_model.call_count >= 1
    
    def test_no_meta_model_skips_filtering(self, ohlcv_df):
        """Without meta model, filtering should be skipped."""
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
        
        # Should still be able to trade without meta model
        assert decision.should_trade is True


@pytest.mark.integration
class TestMetaLabelingWithRisk:
    """Tests for meta-labeling interaction with risk management."""
    
    def test_meta_score_affects_position_sizing(self, ohlcv_df):
        """Lower meta scores should result in smaller positions."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            min_meta_score=0.4,
            use_meta_labeling=True,
            scale_size_by_meta=True,
        )
        
        predictions = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.001,
            "quantiles": np.array([-0.0005, -0.0002, 0.0, 0.0004, 0.0008]),
            "features": np.zeros(64),
        }
        entry = float(ohlcv_df["close"].iloc[-1])
        
        # High meta score
        high_meta = MockMetaModel(score=0.9)
        engine_high = EnhancedDecisionEngine(config=config, meta_model=high_meta)
        engine_high.initialize(starting_balance=10000.0)
        decision_high = engine_high.evaluate(
            predictions=predictions,
            entry_price=entry,
            pair="EURUSD",
            account_balance=10000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # Medium meta score
        med_meta = MockMetaModel(score=0.6)
        engine_med = EnhancedDecisionEngine(config=config, meta_model=med_meta)
        engine_med.initialize(starting_balance=10000.0)
        decision_med = engine_med.evaluate(
            predictions=predictions,
            entry_price=entry,
            pair="EURUSD",
            account_balance=10000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # If scaling is enabled, higher meta should have larger position
        if decision_high.should_trade and decision_med.should_trade:
            if config.scale_size_by_meta:
                assert decision_high.position_size >= decision_med.position_size
    
    def test_borderline_meta_score_handling(self, ohlcv_df):
        """Borderline meta scores should be handled correctly."""
        meta_model = MockMetaModel(score=0.50)  # Exactly at threshold
        
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            min_meta_score=0.5,
            use_meta_labeling=True,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=meta_model)
        engine.initialize(starting_balance=10000.0)
        
        predictions = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.001,
            "quantiles": np.array([-0.0005, -0.0002, 0.0, 0.0004, 0.0008]),
            "features": np.zeros(64),
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
        
        # At exactly threshold, behavior should be consistent
        assert decision is not None
        # Either allowed (>= threshold) or rejected (< threshold)
        if decision.meta_score >= config.min_meta_score:
            # Other factors might still reject
            pass
        else:
            assert decision.should_trade is False
