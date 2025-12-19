"""
End-to-End Pipeline Integration Tests.

Tests the complete flow: Data → Features → Prediction → Decision → Execution
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from trading.decision_engine import EnhancedDecisionEngine, DecisionEngineConfig
from strategies.neural_hybrid import NeuralHybridStrategy, StrategyConfig, Order, OrderType


class MockPredictor:
    """Mock predictor that returns controlled predictions."""
    
    def __init__(self, direction_probs=None, volatility=0.001, quantiles=None):
        self.direction_probs = direction_probs or np.array([0.05, 0.05, 0.90])
        self.volatility = volatility
        self.quantiles = quantiles or np.array([-0.0005, -0.0002, 0.0, 0.0004, 0.0008])
        self.call_count = 0
        self.last_features = None
    
    def predict(self, features, *args, **kwargs):
        self.call_count += 1
        self.last_features = features
        
        result = Mock()
        result.probabilities = self.direction_probs
        result.predicted_class = int(np.argmax(self.direction_probs))
        result.confidence = float(np.max(self.direction_probs))
        result.signal_name = ['BEAR', 'SIDEWAYS', 'BULL'][result.predicted_class]
        result.volatility = self.volatility
        result.quantiles = self.quantiles
        result.gate_weights = np.array([0.4, 0.35, 0.25])
        result.features = np.zeros(64)
        return result


@pytest.mark.integration
class TestEndToEndPipeline:
    """Tests for complete trading pipeline."""
    
    def test_bullish_signal_produces_buy_decision(self, ohlcv_df):
        """High bull probability should produce BUY decision with valid SL/TP."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            min_risk_reward=1.5,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        predictions = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.001,
            "quantiles": np.array([-0.0006, -0.0002, 0.0, 0.0004, 0.0009]),
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
        assert decision.direction == "BUY"
        assert decision.stop_loss < entry
        assert decision.take_profit > entry
        assert decision.position_size > 0
        assert decision.risk_reward_ratio >= 1.5
    
    def test_bearish_signal_produces_sell_decision(self, ohlcv_df):
        """High bear probability should produce SELL decision with valid SL/TP."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            min_risk_reward=1.5,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        predictions = {
            "direction_probs": np.array([0.90, 0.05, 0.05]),
            "volatility": 0.001,
            "quantiles": np.array([-0.0009, -0.0004, 0.0, 0.0002, 0.0006]),
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
        assert decision.direction == "SELL"
        assert decision.stop_loss > entry
        assert decision.take_profit < entry
        assert decision.position_size > 0
    
    def test_sideways_signal_rejects_trade(self, ohlcv_df):
        """High sideways probability should reject trade."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        predictions = {
            "direction_probs": np.array([0.20, 0.60, 0.20]),
            "volatility": 0.001,
            "quantiles": np.array([-0.0003, -0.0001, 0.0, 0.0001, 0.0003]),
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
        
        assert decision.should_trade is False
    
    def test_low_confidence_rejects_trade(self, ohlcv_df):
        """Low confidence should reject trade even with directional bias."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        predictions = {
            "direction_probs": np.array([0.30, 0.30, 0.40]),  # Max is 0.40 < 0.55
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
        
        assert decision.should_trade is False
        assert decision.direction_confidence < 0.55
    
    def test_strategy_creates_valid_order_from_decision(self, ohlcv_df):
        """Strategy should create valid order from decision."""
        data_provider = Mock()
        data_provider.get_ohlcv.return_value = ohlcv_df
        data_provider.get_spread.return_value = 1.0
        
        cfg = StrategyConfig(
            sequence_length=60,
            use_vision=False,
            use_yolo=False,
            min_direction_confidence=0.55,
        )
        strategy = NeuralHybridStrategy(config=cfg, data_provider=data_provider, executor=None)
        strategy._initialized = True
        strategy.predictor = MockPredictor(direction_probs=np.array([0.05, 0.05, 0.90]))
        
        decision = strategy.evaluate()
        
        if decision and decision.should_trade:
            order = strategy.create_order(decision)
            
            assert isinstance(order, Order)
            assert order.symbol == cfg.symbol
            assert order.direction in ("BUY", "SELL")
            assert order.volume > 0
            assert order.stop_loss != 0
            assert order.take_profit != 0
    
    def test_full_pipeline_data_integrity(self, ohlcv_df):
        """Verify data flows correctly through entire pipeline without corruption."""
        data_provider = Mock()
        data_provider.get_ohlcv.return_value = ohlcv_df.copy()
        data_provider.get_spread.return_value = 1.0
        
        predictor = MockPredictor()
        
        cfg = StrategyConfig(
            sequence_length=60,
            use_vision=False,
            use_yolo=False,
        )
        strategy = NeuralHybridStrategy(config=cfg, data_provider=data_provider, executor=None)
        strategy._initialized = True
        strategy.predictor = predictor
        
        # Run evaluation
        strategy.evaluate()
        
        # Verify predictor received valid features
        assert predictor.call_count == 1
        assert predictor.last_features is not None
        assert not np.isnan(predictor.last_features).any()
        assert predictor.last_features.shape[0] == cfg.sequence_length


@pytest.mark.integration
class TestPipelineEdgeCases:
    """Edge case tests for pipeline robustness."""
    
    def test_extreme_volatility_adjusts_position_size(self, ohlcv_df):
        """High volatility should reduce position size."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
        )
        engine = EnhancedDecisionEngine(config=config, meta_model=None)
        engine.initialize(starting_balance=10000.0)
        
        entry = float(ohlcv_df["close"].iloc[-1])
        
        # Normal volatility
        normal_pred = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.001,
            "quantiles": np.array([-0.0005, -0.0002, 0.0, 0.0004, 0.0008]),
            "features": None,
        }
        normal_decision = engine.evaluate(
            predictions=normal_pred,
            entry_price=entry,
            pair="EURUSD",
            account_balance=10000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # High volatility
        high_vol_pred = {
            "direction_probs": np.array([0.05, 0.05, 0.90]),
            "volatility": 0.005,  # 5x higher
            "quantiles": np.array([-0.0025, -0.0010, 0.0, 0.0020, 0.0040]),
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
        
        # High volatility should result in smaller position or rejection
        if normal_decision.should_trade and high_vol_decision.should_trade:
            assert high_vol_decision.position_size <= normal_decision.position_size
    
    def test_wide_spread_affects_decision(self, ohlcv_df):
        """Very wide spread should affect trade viability."""
        config = DecisionEngineConfig(
            profile="INTRADAY",
            min_direction_confidence=0.55,
            min_risk_reward=1.5,
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
        
        # Normal spread
        normal_decision = engine.evaluate(
            predictions=predictions,
            entry_price=entry,
            pair="EURUSD",
            account_balance=10000.0,
            market_data=ohlcv_df,
            current_spread=1.0,
        )
        
        # Very wide spread (50 pips)
        wide_decision = engine.evaluate(
            predictions=predictions,
            entry_price=entry,
            pair="EURUSD",
            account_balance=10000.0,
            market_data=ohlcv_df,
            current_spread=50.0,
        )
        
        # Wide spread should either reject or have worse risk/reward
        if normal_decision.should_trade and wide_decision.should_trade:
            assert wide_decision.risk_reward_ratio <= normal_decision.risk_reward_ratio
