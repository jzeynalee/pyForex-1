# trading/decision_engine.py
"""
Decision Engine: The 'Prefrontal Cortex' that gates impulses.
Combines Deep Learning (Fast, Pattern) with Trend Engine (Slow, Context).
"""
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DecisionResult:
    signal: str          # 'BUY', 'SELL', 'NO_TRADE'
    confidence: float    # 0.0 to 1.0
    reason: str
    meta: Dict           # Debug info

class DecisionEngine:
    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def decide(
        self, 
        pattern_probs: list, 
        trend_analysis: Dict
    ) -> DecisionResult:
        """
        Gating Logic:
        1. Pattern Recognition (Deep Learning) proposes a trade.
        2. Trend Engine (Structural/MTF) validates or rejects it.
        """
        p_buy, p_sell, p_hold = pattern_probs
        
        # 1. Determine the "Impulse" (Pattern Signal)
        if max(p_buy, p_sell) < self.threshold:
            pattern_signal = "NO_TRADE"
            pattern_conf = max(p_buy, p_sell)
        elif p_buy > p_sell:
            pattern_signal = "BUY"
            pattern_conf = p_buy
        else:
            pattern_signal = "SELL"
            pattern_conf = p_sell

        # 2. Get Context (Trend)
        trend_direction = trend_analysis.get('direction', 'SIDEWAYS')
        trend_conf = trend_analysis.get('confidence', 0.5)

        # 3. Apply Gating Rules
        final_signal = "NO_TRADE"
        final_conf = 0.0
        reason = ""

        if pattern_signal == "NO_TRADE":
            reason = f"Pattern weak ({pattern_conf:.2f})"
            
        elif trend_direction == "SIDEWAYS":
            # Rule: In ranging markets, demand higher pattern confidence
            if pattern_conf >= 0.80:
                final_signal = pattern_signal
                final_conf = pattern_conf * 0.85
                reason = "Range scalp (Strong Pattern)"
            else:
                reason = "Filtered: Sideways market"

        elif pattern_signal == "BUY":
            if trend_direction == "BULLISH":
                # Confluence
                final_signal = "BUY"
                final_conf = (pattern_conf + trend_conf) / 2
                reason = "Confluence: Trend + Pattern"
            elif trend_direction == "BEARISH":
                # Counter-trend
                if pattern_conf > 0.90 and trend_conf < 0.60:
                    final_signal = "BUY"
                    final_conf = pattern_conf * 0.6
                    reason = "Counter-trend Reversal (High Risk)"
                else:
                    reason = "Filtered: Trading against Bear Trend"

        elif pattern_signal == "SELL":
            if trend_direction == "BEARISH":
                # Confluence
                final_signal = "SELL"
                final_conf = (pattern_conf + trend_conf) / 2
                reason = "Confluence: Trend + Pattern"
            elif trend_direction == "BULLISH":
                # Counter-trend
                if pattern_conf > 0.90 and trend_conf < 0.60:
                    final_signal = "SELL"
                    final_conf = pattern_conf * 0.6
                    reason = "Counter-trend Reversal (High Risk)"
                else:
                    reason = "Filtered: Trading against Bull Trend"

        return DecisionResult(
            signal=final_signal,
            confidence=final_conf,
            reason=reason,
            meta={
                'pattern_signal': pattern_signal,
                'pattern_conf': pattern_conf,
                'trend_direction': trend_direction,
                'trend_conf': trend_conf
            }
        )