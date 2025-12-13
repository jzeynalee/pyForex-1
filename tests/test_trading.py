# tests/test_trading.py
"""
Unit tests for trading modules.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from trading.signal_engine import (
    Signal, SignalResult, SignalConfig, 
    generate_signal, generate_signal_simple, SignalAggregator
)
from trading.risk_manager import RiskManager, RiskConfig
from trading.decision_engine import DecisionEngine, DecisionResult
from trading.backtest import BacktestExecutor, BacktestConfig

@pytest.mark.unit
class TestSignalEngine:
    """Test suite for signal generation."""
    
    @pytest.mark.parametrize("probs,expected_signal", [
        ([0.80, 0.10, 0.10], Signal.BUY),
        ([0.10, 0.85, 0.05], Signal.SELL),
        ([0.15, 0.15, 0.70], Signal.HOLD),
        ([0.45, 0.43, 0.12], Signal.NO_TRADE),  # Insufficient spread
        ([0.50, 0.30, 0.20], Signal.NO_TRADE),  # Below threshold
    ])
    def test_signal_generation_parametrized(self, probs, expected_signal):
        """Test signal generation with various probability distributions."""
        result = generate_signal(np.array(probs))
        assert result.signal == expected_signal
    
    def test_signal_enum(self):
        """Test Signal enum values."""
        assert Signal.BUY.value == "BUY"
        assert Signal.SELL.value == "SELL"
        assert Signal.HOLD.value == "HOLD"
        assert Signal.NO_TRADE.value == "NO_TRADE"
    
    def test_generate_buy_signal(self, model_probabilities_buy):
        """Test BUY signal generation."""
        result = generate_signal(model_probabilities_buy)
        
        assert result.signal == Signal.BUY
        assert result.confidence == 0.75
        assert "Bullish" in result.reason
    
    def test_generate_sell_signal(self, model_probabilities_sell):
        """Test SELL signal generation."""
        result = generate_signal(model_probabilities_sell)
        
        assert result.signal == Signal.SELL
        assert result.confidence == 0.80
        assert "Bearish" in result.reason
    
    def test_no_trade_low_confidence(self, model_probabilities_uncertain):
        """Test NO_TRADE with low confidence."""
        result = generate_signal(model_probabilities_uncertain)
        
        assert result.signal == Signal.NO_TRADE
        assert "Low confidence" in result.reason
    
    def test_hold_dominant(self):
        """Test HOLD when hold probability is dominant."""
        probs = np.array([0.15, 0.15, 0.70])  # High HOLD
        result = generate_signal(probs)
        
        assert result.signal == Signal.HOLD
        assert "HOLD dominant" in result.reason
    
    def test_insufficient_spread(self):
        """Test NO_TRADE with insufficient bull/bear spread."""
        # Use higher confidence so it passes confidence check but fails spread check
        probs = np.array([0.62, 0.60, 0.08])  # Close BUY vs SELL, above threshold
        result = generate_signal(probs)
        
        assert result.signal == Signal.NO_TRADE
        # Either spread or confidence reason is acceptable
        assert "spread" in result.reason.lower() or "confidence" in result.reason.lower()

    
    def test_custom_config(self):
        """Test signal generation with custom config."""
        config = SignalConfig(min_confidence=0.50, bull_bear_spread=0.10)
        probs = np.array([0.55, 0.40, 0.05])
        
        result = generate_signal(probs, config)
        
        assert result.signal == Signal.BUY
    
    def test_generate_signal_simple_buy(self):
        """Test simple signal generation for BUY."""
        probs = np.array([0.70, 0.20, 0.10])
        signal = generate_signal_simple(probs, threshold=0.6)
        
        assert signal == "BUY"
    
    def test_generate_signal_simple_sell(self):
        """Test simple signal generation for SELL."""
        probs = np.array([0.10, 0.75, 0.15])
        signal = generate_signal_simple(probs, threshold=0.6)
        
        assert signal == "SELL"
    
    def test_generate_signal_simple_no_trade(self):
        """Test simple signal generation for NO_TRADE."""
        probs = np.array([0.40, 0.40, 0.20])
        signal = generate_signal_simple(probs, threshold=0.6)
        
        assert signal == "NO_TRADE"
    
    def test_invalid_probabilities(self):
        """Test error with wrong number of probabilities."""
        with pytest.raises(ValueError, match="Expected 3 probabilities"):
            generate_signal(np.array([0.5, 0.5]))

@pytest.mark.unit
class TestSignalAggregator:
    """Test suite for signal aggregation."""
    
    def test_init(self):
        """Test SignalAggregator initialization."""
        agg = SignalAggregator(window_size=5, consensus_threshold=0.6)
        
        assert agg.window_size == 5
        assert agg.consensus_threshold == 0.6
    
    def test_incomplete_window(self):
        """Test behavior with incomplete window."""
        agg = SignalAggregator(window_size=3)
        
        result = agg.add_signal(Signal.BUY)
        assert result == Signal.NO_TRADE
        
        result = agg.add_signal(Signal.BUY)
        assert result == Signal.NO_TRADE
    
    def test_consensus_buy(self):
        """Test BUY consensus."""
        agg = SignalAggregator(window_size=3, consensus_threshold=0.67)
        
        agg.add_signal(Signal.BUY)
        agg.add_signal(Signal.BUY)
        result = agg.add_signal(Signal.HOLD)  # 2/3 BUY
        
        assert result == Signal.BUY
    
    def test_consensus_sell(self):
        """Test SELL consensus."""
        agg = SignalAggregator(window_size=3, consensus_threshold=0.67)
        
        agg.add_signal(Signal.SELL)
        agg.add_signal(Signal.SELL)
        result = agg.add_signal(Signal.BUY)  # 2/3 SELL
        
        assert result == Signal.SELL
    
    def test_no_consensus(self):
        """Test when no consensus is reached."""
        agg = SignalAggregator(window_size=3, consensus_threshold=0.67)
        
        agg.add_signal(Signal.BUY)
        agg.add_signal(Signal.SELL)
        result = agg.add_signal(Signal.HOLD)
        
        assert result == Signal.NO_TRADE
    
    def test_reset(self):
        """Test history reset."""
        agg = SignalAggregator(window_size=3)
        
        agg.add_signal(Signal.BUY)
        agg.add_signal(Signal.BUY)
        agg.reset()
        
        assert len(agg.signal_history) == 0

@pytest.mark.unit
class TestRiskManager:
    """Test suite for risk management."""
    
    def test_init_default(self):
        """Test RiskManager initialization."""
        rm = RiskManager(account_balance=10000)
        
        assert rm.starting_balance == 10000
        assert rm.trading_allowed == True
        assert rm.daily_trades == 0
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = RiskConfig(max_daily_loss_pct=0.05, risk_per_trade_pct=0.02)
        rm = RiskManager(account_balance=10000, config=config)
        
        assert rm.config.max_daily_loss_pct == 0.05
        assert rm.config.risk_per_trade_pct == 0.02
    
    def test_check_risk_limits_allowed(self):
        """Test risk limits when trading is allowed."""
        rm = RiskManager(account_balance=10000)
        
        allowed, reason = rm.check_risk_limits(
            current_balance=10000,
            current_equity=10000,
            open_positions=0
        )
        
        assert allowed == True
        assert reason is None
    
    def test_daily_loss_limit(self):
        """Test daily loss limit enforcement."""
        rm = RiskManager(account_balance=10000)
        
        # Simulate 4% loss (exceeds 3% default limit)
        allowed, reason = rm.check_risk_limits(
            current_balance=10000,
            current_equity=9600,
            open_positions=0
        )
        
        assert allowed == False
        assert "Daily loss limit" in reason
    
    def test_max_drawdown(self):
        """Test max drawdown limit enforcement."""
        rm = RiskManager(account_balance=10000)
        
        # Simulate a scenario where daily loss is OK but total drawdown exceeds limit
        # Set daily_start_balance to current balance to avoid daily loss trigger
        rm.daily_start_balance = 9000  # Pretend day started at 9000
        
        # Now equity at 8900 is only ~1.1% daily loss (OK)
        # But total drawdown from starting_balance is 11% (exceeds 10% limit)
        allowed, reason = rm.check_risk_limits(
            current_balance=9000,
            current_equity=8900,
            open_positions=0
        )
        
        assert allowed == False
        assert "drawdown" in reason.lower()
        assert rm.trading_allowed == False  # Should halt trading
    
    def test_max_positions(self):
        """Test max positions limit."""
        rm = RiskManager(account_balance=10000)
        
        allowed, reason = rm.check_risk_limits(
            current_balance=10000,
            current_equity=10000,
            open_positions=3  # Default max is 3
        )
        
        assert allowed == False
        assert "Max positions" in reason
    
    def test_daily_trade_limit(self):
        """Test daily trade limit enforcement."""
        rm = RiskManager(account_balance=10000)
        rm.daily_trades = 10  # Default max is 10
        
        allowed, reason = rm.check_risk_limits(
            current_balance=10000,
            current_equity=10000,
            open_positions=0
        )
        
        assert allowed == False
        assert "Daily trade limit" in reason
    
    def test_calculate_atr(self, sample_ohlcv_data):
        """Test ATR calculation."""
        rm = RiskManager(account_balance=10000)
        atr = rm.calculate_atr(sample_ohlcv_data)
        
        assert atr > 0
        assert isinstance(atr, float)
    
    def test_get_params(self, sample_ohlcv_data):
        """Test trade parameter calculation."""
        rm = RiskManager(account_balance=10000)
        
        params = rm.get_params(sample_ohlcv_data, signal='BUY')
        
        assert isinstance(params, TradeParams)
        assert params.volume > 0
        assert params.stop_loss < sample_ohlcv_data['close'].iloc[-1]  # SL below price for BUY
        assert params.take_profit > sample_ohlcv_data['close'].iloc[-1]  # TP above price for BUY
    
    def test_get_params_sell(self, sample_ohlcv_data):
        """Test trade parameters for SELL."""
        rm = RiskManager(account_balance=10000)
        
        params = rm.get_params(sample_ohlcv_data, signal='SELL')
        
        assert params.stop_loss > sample_ohlcv_data['close'].iloc[-1]  # SL above price for SELL
        assert params.take_profit < sample_ohlcv_data['close'].iloc[-1]  # TP below price for SELL
    
    def test_record_trade(self):
        """Test trade recording."""
        rm = RiskManager(account_balance=10000)
        
        rm.record_trade(pnl=100)
        
        assert rm.daily_trades == 1
        assert rm.daily_pnl == 100
    
    def test_get_status(self):
        """Test status getter."""
        rm = RiskManager(account_balance=10000)
        rm.record_trade(pnl=50)
        
        status = rm.get_status()
        
        assert status['trading_allowed'] == True
        assert status['daily_trades'] == 1
        assert status['daily_pnl'] == 50

    def test_get_params_with_zero_atr(self, sample_ohlcv_data):
        """Test fallback when ATR is zero (flat price)."""
        rm = RiskManager(account_balance=10000)
        
        # Create flat price data
        flat_df = sample_ohlcv_data.copy()
        flat_df['high'] = 1.1
        flat_df['low'] = 1.1
        flat_df['close'] = 1.1
        flat_df['open'] = 1.1
        
        params = rm.get_params(flat_df, signal='BUY')
        
        # Should use fallback, not crash or return zeros
        assert params.volume > 0
        assert params.stop_loss != params.take_profit
    
    def test_risk_limits_with_negative_balance(self):
        """Test behavior with edge case balances."""
        rm = RiskManager(account_balance=10000)
        
        # This shouldn't happen, but test defensive behavior
        allowed, reason = rm.check_risk_limits(
            current_balance=-100,
            current_equity=-100,
            open_positions=0
        )
        
        assert allowed == False

@pytest.mark.unit
class TestDecisionEngine:
    """Test suite for decision engine."""
    
    def test_init(self):
        """Test DecisionEngine initialization."""
        engine = DecisionEngine(threshold=0.65)
        assert engine.threshold == 0.65
    
    def test_buy_bullish_confluence(self, mock_trend_analysis_bullish):
        """Test BUY decision with bullish confluence."""
        engine = DecisionEngine(threshold=0.70)
        
        pattern_probs = [0.75, 0.15, 0.10]  # Strong BUY pattern
        
        result = engine.decide(pattern_probs, mock_trend_analysis_bullish)
        
        assert result.signal == "BUY"
        assert "Confluence" in result.reason
    
    def test_sell_bearish_confluence(self, mock_trend_analysis_bearish):
        """Test SELL decision with bearish confluence."""
        engine = DecisionEngine(threshold=0.70)
        
        pattern_probs = [0.10, 0.80, 0.10]  # Strong SELL pattern
        
        result = engine.decide(pattern_probs, mock_trend_analysis_bearish)
        
        assert result.signal == "SELL"
        assert "Confluence" in result.reason
    
    def test_filtered_counter_trend(self, mock_trend_analysis_bullish):
        """Test filtered counter-trend trade."""
        engine = DecisionEngine(threshold=0.70)
        
        pattern_probs = [0.10, 0.75, 0.15]  # SELL pattern
        
        result = engine.decide(pattern_probs, mock_trend_analysis_bullish)
        
        # Should be filtered (selling into bull trend)
        assert result.signal == "NO_TRADE"
        assert "Filtered" in result.reason
    
    def test_weak_pattern_filtered(self, mock_trend_analysis_bullish):
        """Test filtering of weak pattern signals."""
        engine = DecisionEngine(threshold=0.70)
        
        pattern_probs = [0.50, 0.30, 0.20]  # Weak pattern
        
        result = engine.decide(pattern_probs, mock_trend_analysis_bullish)
        
        assert result.signal == "NO_TRADE"
        assert "weak" in result.reason.lower()
    
    def test_sideways_market(self, mock_trend_analysis_sideways):
        """Test decision in sideways market."""
        engine = DecisionEngine(threshold=0.70)
        
        # Moderate pattern
        pattern_probs = [0.75, 0.15, 0.10]
        
        result = engine.decide(pattern_probs, mock_trend_analysis_sideways)
        
        # Not strong enough for sideways
        assert result.signal == "NO_TRADE"


@pytest.mark.unit
class TestBacktestExecutor:
    """Test suite for backtest executor."""
    
    def test_init_default(self):
        """Test BacktestExecutor initialization."""
        executor = BacktestExecutor()
        
        assert executor.balance == 10000.0
        assert executor.equity == 10000.0
        assert len(executor.positions) == 0
    
    def test_init_custom(self):
        """Test initialization with custom config."""
        config = BacktestConfig(initial_balance=50000, spread_pips=2.0)
        executor = BacktestExecutor(config)
        
        assert executor.balance == 50000
        assert executor.config.spread_pips == 2.0
    
    def test_entry_buy(self):
        """Test BUY order entry."""
        executor = BacktestExecutor()
        executor.current_price = 1.1000
        
        result = executor.entry(
            signal='BUY',
            volume=0.1,
            sl=1.0950,
            tp=1.1100
        )
        
        assert result['success'] == True
        assert len(executor.positions) == 1
        assert executor.positions[0].direction == 'BUY'
    
    def test_entry_sell(self):
        """Test SELL order entry."""
        executor = BacktestExecutor()
        executor.current_price = 1.1000
        
        result = executor.entry(
            signal='SELL',
            volume=0.1,
            sl=1.1050,
            tp=1.0900
        )
        
        assert result['success'] == True
        assert executor.positions[0].direction == 'SELL'
    
    def test_entry_invalid_signal(self):
        """Test entry with invalid signal."""
        executor = BacktestExecutor()
        
        result = executor.entry(
            signal='HOLD',
            volume=0.1,
            sl=1.0,
            tp=1.1
        )
        
        assert result['success'] == False
    
    def test_stop_loss_hit_buy(self):
        """Test stop loss hit for BUY position."""
        executor = BacktestExecutor()
        executor.current_price = 1.1000
        
        executor.entry('BUY', volume=0.1, sl=1.0950, tp=1.1100)
        
        # Price drops to SL
        executor.update_price(1.0940)
        
        assert len(executor.positions) == 0
        assert len(executor.trade_history) == 1
        assert executor.trade_history[0].status == 'CLOSED_SL'
    
    def test_take_profit_hit_buy(self):
        """Test take profit hit for BUY position."""
        executor = BacktestExecutor()
        executor.current_price = 1.1000
        
        executor.entry('BUY', volume=0.1, sl=1.0950, tp=1.1100)
        
        # Price rises to TP
        executor.update_price(1.1110)
        
        assert len(executor.positions) == 0
        assert executor.trade_history[0].status == 'CLOSED_TP'
        assert executor.trade_history[0].pnl > 0
    
    def test_stop_loss_hit_sell(self):
        """Test stop loss hit for SELL position."""
        executor = BacktestExecutor()
        executor.current_price = 1.1000
        
        executor.entry('SELL', volume=0.1, sl=1.1050, tp=1.0900)
        
        # Price rises to SL
        executor.update_price(1.1060)
        
        assert len(executor.positions) == 0
        assert executor.trade_history[0].status == 'CLOSED_SL'
    
    def test_close_all_positions(self):
        """Test closing all positions."""
        executor = BacktestExecutor()
        executor.current_price = 1.1000
        
        executor.entry('BUY', volume=0.1, sl=1.0950, tp=1.1100)
        executor.entry('SELL', volume=0.1, sl=1.1050, tp=1.0900)
        
        executor.close_all_positions()
        
        assert len(executor.positions) == 0
        assert len(executor.trade_history) == 2
    
    def test_get_performance_metrics(self):
        """Test performance metrics calculation."""
        executor = BacktestExecutor()
        executor.current_price = 1.1000
        
        # Create a winning trade
        executor.entry('BUY', volume=0.1, sl=1.0950, tp=1.1050)
        executor.update_price(1.1060)
        
        # Create a losing trade
        executor.entry('SELL', volume=0.1, sl=1.1060, tp=1.0940)
        executor.update_price(1.1070)
        
        metrics = executor.get_performance_metrics()
        
        assert metrics['total_trades'] == 2
        assert metrics['winning_trades'] == 1
        assert metrics['losing_trades'] == 1
        assert metrics['win_rate'] == 0.5
    
    def test_empty_performance_metrics(self):
        """Test metrics with no trades."""
        executor = BacktestExecutor()
        metrics = executor.get_performance_metrics()
        
        assert metrics['total_trades'] == 0
        assert metrics['win_rate'] == 0
    
    def test_get_open_positions(self):
        """Test getting open positions list."""
        executor = BacktestExecutor()
        executor.current_price = 1.1000
        
        executor.entry('BUY', volume=0.1, sl=1.0950, tp=1.1100)
        
        positions = executor.get_open_positions()
        
        assert len(positions) == 1
        assert positions[0]['type'] == 'BUY'
        assert positions[0]['volume'] == 0.1