# tests/test_trading_signal_engine.py
"""
Unit tests for trading/signal_engine.py - Signal generation from model predictions.
"""

import pytest
import numpy as np
from trading.signal_engine import (
    Signal, SignalResult, SignalConfig, generate_signal,
    generate_signal_simple, SignalAggregator
)


@pytest.mark.unit
class TestSignal:
    """Test Signal enum."""

    def test_signal_values(self):
        """Test signal enum values."""
        assert Signal.BUY.value == "BUY"
        assert Signal.SELL.value == "SELL"
        assert Signal.HOLD.value == "HOLD"
        assert Signal.NO_TRADE.value == "NO_TRADE"


@pytest.mark.unit
class TestSignalConfig:
    """Test SignalConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SignalConfig()

        assert config.min_confidence == 0.60
        assert config.max_confidence_hold == 0.45
        assert config.bull_bear_spread == 0.15

    def test_custom_values(self):
        """Test custom configuration."""
        config = SignalConfig(
            min_confidence=0.70,
            max_confidence_hold=0.40,
            bull_bear_spread=0.20
        )

        assert config.min_confidence == 0.70
        assert config.max_confidence_hold == 0.40
        assert config.bull_bear_spread == 0.20


@pytest.mark.unit
class TestGenerateSignal:
    """Test generate_signal function."""

    def test_hold_dominant(self):
        """Test when HOLD probability is dominant."""
        probs = np.array([0.20, 0.10, 0.70])  # BUY, SELL, HOLD

        result = generate_signal(probs)

        assert result.signal == Signal.HOLD
        assert result.confidence == 0.70
        assert "HOLD dominant" in result.reason

    def test_low_confidence_no_trade(self):
        """Test low confidence results in NO_TRADE."""
        config = SignalConfig(min_confidence=0.60)
        probs = np.array([0.55, 0.25, 0.20])  # BUY below threshold

        result = generate_signal(probs, config)

        assert result.signal == Signal.NO_TRADE
        assert "Low confidence" in result.reason

    def test_insufficient_spread_no_trade(self):
        """Test insufficient spread results in NO_TRADE."""
        config = SignalConfig(bull_bear_spread=0.15)
        probs = np.array([0.50, 0.45, 0.05])  # Spread = 0.05 < 0.15

        result = generate_signal(probs, config)

        assert result.signal == Signal.NO_TRADE
        assert "Insufficient spread" in result.reason

    def test_buy_signal(self):
        """Test BUY signal generation."""
        probs = np.array([0.75, 0.15, 0.10])  # Strong BUY

        result = generate_signal(probs)

        assert result.signal == Signal.BUY
        assert result.confidence == 0.75
        assert "Bullish signal" in result.reason

    def test_sell_signal(self):
        """Test SELL signal generation."""
        probs = np.array([0.15, 0.75, 0.10])  # Strong SELL

        result = generate_signal(probs)

        assert result.signal == Signal.SELL
        assert result.confidence == 0.75
        assert "Bearish signal" in result.reason

    def test_invalid_probability_length(self):
        """Test error on invalid probability array length."""
        probs = np.array([0.5, 0.5])  # Only 2 values

        with pytest.raises(ValueError, match="Expected 3 probabilities"):
            generate_signal(probs)

    def test_multidimensional_array(self):
        """Test handling of multidimensional arrays."""
        probs = np.array([[0.75, 0.15, 0.10]])  # 2D array

        result = generate_signal(probs)

        assert result.signal == Signal.BUY
        assert result.confidence == 0.75


@pytest.mark.unit
class TestGenerateSignalSimple:
    """Test generate_signal_simple function."""

    def test_buy_signal(self):
        """Test BUY signal generation."""
        probs = np.array([0.65, 0.25, 0.10])
        threshold = 0.60

        result = generate_signal_simple(probs, threshold)

        assert result == "BUY"

    def test_sell_signal(self):
        """Test SELL signal generation."""
        probs = np.array([0.25, 0.65, 0.10])
        threshold = 0.60

        result = generate_signal_simple(probs, threshold)

        assert result == "SELL"

    def test_no_trade_below_threshold(self):
        """Test NO_TRADE when below threshold."""
        probs = np.array([0.55, 0.30, 0.15])
        threshold = 0.60

        result = generate_signal_simple(probs, threshold)

        assert result == "NO_TRADE"

    def test_no_trade_equal_probabilities(self):
        """Test NO_TRADE when probabilities are equal."""
        probs = np.array([0.50, 0.50, 0.00])
        threshold = 0.60

        result = generate_signal_simple(probs, threshold)

        assert result == "NO_TRADE"

    def test_two_probabilities(self):
        """Test with only 2 probabilities (no HOLD)."""
        probs = np.array([0.70, 0.30])
        threshold = 0.60

        result = generate_signal_simple(probs, threshold)

        assert result == "BUY"

    def test_invalid_length(self):
        """Test error on invalid array length."""
        probs = np.array([0.5])  # Only 1 value

        with pytest.raises(ValueError):
            generate_signal_simple(probs)


@pytest.mark.unit
class TestSignalAggregator:
    """Test SignalAggregator class."""

    def test_init_default(self):
        """Test default initialization."""
        aggregator = SignalAggregator()

        assert aggregator.window_size == 3
        assert aggregator.consensus_threshold == 0.67
        assert len(aggregator.signal_history) == 0

    def test_init_custom(self):
        """Test custom initialization."""
        aggregator = SignalAggregator(window_size=5, consensus_threshold=0.80)

        assert aggregator.window_size == 5
        assert aggregator.consensus_threshold == 0.80

    def test_add_signal_insufficient_window(self):
        """Test NO_TRADE when window not full."""
        aggregator = SignalAggregator(window_size=3)

        result = aggregator.add_signal(Signal.BUY)

        assert result == Signal.NO_TRADE
        assert len(aggregator.signal_history) == 1

    def test_add_signal_buy_consensus(self):
        """Test BUY consensus after multiple signals."""
        aggregator = SignalAggregator(window_size=3, consensus_threshold=0.67)

        # Add 2 BUY signals (67% = 2/3)
        aggregator.add_signal(Signal.BUY)
        aggregator.add_signal(Signal.BUY)
        result = aggregator.add_signal(Signal.BUY)

        assert result == Signal.BUY
        assert len(aggregator.signal_history) == 3

    def test_add_signal_sell_consensus(self):
        """Test SELL consensus after multiple signals."""
        aggregator = SignalAggregator(window_size=3, consensus_threshold=0.67)

        # Add 2 SELL signals
        aggregator.add_signal(Signal.SELL)
        aggregator.add_signal(Signal.SELL)
        result = aggregator.add_signal(Signal.HOLD)

        assert result == Signal.SELL  # 2/3 = 67% consensus
        assert len(aggregator.signal_history) == 3

    def test_add_signal_no_consensus(self):
        """Test NO_TRADE when no consensus."""
        aggregator = SignalAggregator(window_size=3, consensus_threshold=0.67)

        # Mixed signals
        aggregator.add_signal(Signal.BUY)
        aggregator.add_signal(Signal.SELL)
        result = aggregator.add_signal(Signal.HOLD)

        assert result == Signal.NO_TRADE

    def test_window_sliding(self):
        """Test that window slides correctly."""
        aggregator = SignalAggregator(window_size=3)

        # Fill window
        aggregator.add_signal(Signal.BUY)
        aggregator.add_signal(Signal.BUY)
        aggregator.add_signal(Signal.BUY)

        # Add another signal (should remove oldest)
        result = aggregator.add_signal(Signal.SELL)

        assert len(aggregator.signal_history) == 3
        assert aggregator.signal_history[0] == Signal.BUY  # Oldest
        assert aggregator.signal_history[-1] == Signal.SELL  # Newest

    def test_reset(self):
        """Test reset clears history."""
        aggregator = SignalAggregator(window_size=3)

        aggregator.add_signal(Signal.BUY)
        aggregator.add_signal(Signal.BUY)
        aggregator.reset()

        assert len(aggregator.signal_history) == 0

    def test_consensus_exact_threshold(self):
        """Test consensus at exact threshold."""
        aggregator = SignalAggregator(window_size=5, consensus_threshold=0.60)

        # Add 3 BUY signals (60% = 3/5)
        aggregator.add_signal(Signal.BUY)
        aggregator.add_signal(Signal.BUY)
        aggregator.add_signal(Signal.BUY)
        aggregator.add_signal(Signal.HOLD)
        result = aggregator.add_signal(Signal.HOLD)

        assert result == Signal.BUY

