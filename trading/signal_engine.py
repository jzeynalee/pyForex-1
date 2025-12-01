# trading/signal_engine.py
"""
Signal generation from model predictions.
"""
import numpy as np
from typing import Literal, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"


class SignalResult(NamedTuple):
    """Structured signal output."""
    signal: Signal
    confidence: float
    reason: str


@dataclass
class SignalConfig:
    """Configuration for signal generation."""
    min_confidence: float = 0.60        # Minimum confidence to trade
    max_confidence_hold: float = 0.45   # Max hold prob before forcing decision
    bull_bear_spread: float = 0.15      # Min difference between bull/bear
    

def generate_signal(
    probabilities: np.ndarray,
    config: Optional[SignalConfig] = None,
) -> SignalResult:
    """
    Generate trading signal from model probabilities.
    
    Args:
        probabilities: Array of [P(BUY), P(SELL), P(HOLD)]
        config: Signal generation parameters
    
    Returns:
        SignalResult with signal, confidence, and reason
    """
    config = config or SignalConfig()
    
    # Handle different input shapes
    probs = np.array(probabilities).flatten()
    if len(probs) != 3:
        raise ValueError(f"Expected 3 probabilities, got {len(probs)}")
    
    p_buy, p_sell, p_hold = probs
    
    # Rule 1: If HOLD is dominant, don't trade
    if p_hold > config.max_confidence_hold and p_hold > max(p_buy, p_sell):
        return SignalResult(
            signal=Signal.HOLD,
            confidence=p_hold,
            reason=f"HOLD dominant ({p_hold:.2%})"
        )
    
    # Rule 2: Check if any directional signal is confident enough
    max_directional = max(p_buy, p_sell)
    
    if max_directional < config.min_confidence:
        return SignalResult(
            signal=Signal.NO_TRADE,
            confidence=max_directional,
            reason=f"Low confidence ({max_directional:.2%} < {config.min_confidence:.2%})"
        )
    
    # Rule 3: Check bull/bear spread (avoid conflicting signals)
    spread = abs(p_buy - p_sell)
    if spread < config.bull_bear_spread:
        return SignalResult(
            signal=Signal.NO_TRADE,
            confidence=max_directional,
            reason=f"Insufficient spread ({spread:.2%} < {config.bull_bear_spread:.2%})"
        )
    
    # Rule 4: Generate signal
    if p_buy > p_sell:
        return SignalResult(
            signal=Signal.BUY,
            confidence=p_buy,
            reason=f"Bullish signal ({p_buy:.2%} confidence)"
        )
    else:
        return SignalResult(
            signal=Signal.SELL,
            confidence=p_sell,
            reason=f"Bearish signal ({p_sell:.2%} confidence)"
        )


def generate_signal_simple(
    probabilities: np.ndarray,
    threshold: float = 0.6,
) -> str:
    """
    Simple signal generation (backwards compatible).
    
    Args:
        probabilities: Array of [P(BUY), P(SELL), P(HOLD)]
        threshold: Minimum probability threshold
    
    Returns:
        'BUY', 'SELL', or 'NO_TRADE'
    """
    probs = np.array(probabilities).flatten()
    
    if len(probs) == 3:
        p_buy, p_sell, p_hold = probs
    elif len(probs) == 2:
        p_buy, p_sell = probs
        p_hold = 0
    else:
        raise ValueError(f"Expected 2 or 3 probabilities, got {len(probs)}")
    
    if p_buy > threshold and p_buy > p_sell:
        return "BUY"
    if p_sell > threshold and p_sell > p_buy:
        return "SELL"
    return "NO_TRADE"


class SignalAggregator:
    """
    Aggregates signals over time for more robust decisions.
    Reduces whipsawing by requiring consistent signals.
    """
    
    def __init__(
        self,
        window_size: int = 3,
        consensus_threshold: float = 0.67,
    ):
        self.window_size = window_size
        self.consensus_threshold = consensus_threshold
        self.signal_history: list[Signal] = []
    
    def add_signal(self, signal: Signal) -> Signal:
        """
        Add signal to history and return consensus signal.
        
        Args:
            signal: Latest signal
        
        Returns:
            Consensus signal (BUY/SELL only if consistent)
        """
        self.signal_history.append(signal)
        
        # Keep only recent signals
        if len(self.signal_history) > self.window_size:
            self.signal_history.pop(0)
        
        # Need full window
        if len(self.signal_history) < self.window_size:
            return Signal.NO_TRADE
        
        # Count signals
        buy_count = sum(1 for s in self.signal_history if s == Signal.BUY)
        sell_count = sum(1 for s in self.signal_history if s == Signal.SELL)
        
        # Check consensus
        required = int(self.window_size * self.consensus_threshold)
        
        if buy_count >= required:
            return Signal.BUY
        if sell_count >= required:
            return Signal.SELL
        
        return Signal.NO_TRADE
    
    def reset(self):
        """Clear signal history."""
        self.signal_history.clear()
