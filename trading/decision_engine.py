# trading/decision_engine.py
"""
Decision Engine: The 'Prefrontal Cortex' that gates impulses.

UPDATED: Now integrates with MTF (Multi-Timeframe) analysis system.

Combines:
- Deep Learning (TCN/Fusion) for fast pattern recognition
- MTF Trend Analysis for context and confluence
- Regime Classification for market state awareness

The decision engine acts as a gatekeeper, ensuring trades only execute
when multiple systems agree.
"""

import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength classification."""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NONE = "NONE"


@dataclass
class DecisionResult:
    """Structured decision output."""
    signal: str          # 'BUY', 'SELL', 'NO_TRADE'
    confidence: float    # 0.0 to 1.0
    reason: str          # Human-readable explanation
    strength: SignalStrength
    meta: Dict           # Debug info
    
    def is_tradeable(self) -> bool:
        """Check if signal is actionable."""
        return self.signal in ('BUY', 'SELL')


@dataclass
class DecisionConfig:
    """Configuration for DecisionEngine."""
    # Thresholds
    pattern_threshold: float = 0.70       # Min pattern confidence to consider
    trend_threshold: float = 0.55         # Min trend confidence for alignment
    confluence_threshold: float = 0.60    # Min MTF confluence score
    
    # Counter-trend settings
    allow_counter_trend: bool = True
    counter_trend_pattern_threshold: float = 0.90
    counter_trend_trend_threshold: float = 0.60
    counter_trend_penalty: float = 0.6    # Reduce confidence for counter-trend
    
    # Regime settings
    ranging_market_boost: float = 0.80    # Required pattern confidence in ranging
    volatile_market_penalty: float = 0.85 # Confidence multiplier in volatile
    
    # MTF settings
    require_higher_tf_alignment: bool = True
    higher_tf_penalty: float = 0.75       # Penalty if higher TF disagrees


class DecisionEngine:
    """
    Decision Engine integrating pattern recognition with MTF trend analysis.
    
    Pipeline:
    1. Evaluate pattern signal (from TCN/Fusion model)
    2. Check MTF trend alignment
    3. Apply regime filters
    4. Compute final signal with confidence
    
    Usage:
        engine = DecisionEngine()
        result = engine.decide(
            pattern_probs=[0.75, 0.15, 0.10],
            mtf_result=mtf_trend_detector.detect(dfs_dict)
        )
        
        if result.is_tradeable():
            execute_trade(result.signal, result.confidence)
    """
    
    def __init__(self, config: Optional[DecisionConfig] = None):
        self.config = config or DecisionConfig()
    
    def decide(
        self,
        pattern_probs: List[float],
        trend_analysis: Optional[Dict] = None,
        mtf_result: Optional["MTFTrendResult"] = None,
    ) -> DecisionResult:
        """
        Main decision method.
        
        Args:
            pattern_probs: [P(BUY), P(SELL), P(HOLD)] from ML model
            trend_analysis: Legacy dict format (for backward compatibility)
            mtf_result: MTFTrendResult from MTFTrendDetector (preferred)
        
        Returns:
            DecisionResult with signal, confidence, and reasoning
        """
        # Handle both legacy dict and new MTFTrendResult
        if mtf_result is not None:
            trend_info = self._extract_trend_info(mtf_result)
        elif trend_analysis is not None:
            trend_info = trend_analysis
        else:
            # No trend info - use neutral defaults
            trend_info = {
                'direction': 'SIDEWAYS',
                'confidence': 0.5,
                'regime': 'UNKNOWN',
                'mtf_score': 0.5,
                'alignment': 0.5,
                'higher_tf_aligned': True,
            }
        
        # Step 1: Evaluate pattern signal
        pattern_signal, pattern_conf = self._evaluate_pattern(pattern_probs)
        
        # Step 2: Get trend context
        trend_direction = trend_info.get('direction', 'SIDEWAYS')
        trend_conf = trend_info.get('confidence', 0.5)
        regime = trend_info.get('regime', 'UNKNOWN')
        mtf_score = trend_info.get('mtf_score', 0.5)
        alignment = trend_info.get('alignment', 0.5)
        higher_tf_aligned = trend_info.get('higher_tf_aligned', True)
        
        # Step 3: Apply decision logic
        return self._apply_decision_logic(
            pattern_signal=pattern_signal,
            pattern_conf=pattern_conf,
            trend_direction=trend_direction,
            trend_conf=trend_conf,
            regime=regime,
            mtf_score=mtf_score,
            alignment=alignment,
            higher_tf_aligned=higher_tf_aligned,
        )
    
    def _extract_trend_info(self, mtf_result: "MTFTrendResult") -> Dict:
        """Extract trend info from MTFTrendResult."""
        return {
            'direction': mtf_result.direction,
            'confidence': mtf_result.confidence,
            'regime': mtf_result.regime,
            'mtf_score': mtf_result.mtf_score,
            'alignment': mtf_result.mtf_alignment,
            'higher_tf_aligned': mtf_result.higher_tf_aligned,
        }
    
    def _evaluate_pattern(
        self,
        probs: List[float],
    ) -> Tuple[str, float]:
        """
        Evaluate pattern signal from probabilities.
        
        Returns:
            (signal, confidence)
        """
        p_buy, p_sell, p_hold = probs[0], probs[1], probs[2]
        
        # Check if any directional signal meets threshold
        max_directional = max(p_buy, p_sell)
        
        if max_directional < self.config.pattern_threshold:
            return "NO_TRADE", max_directional
        
        if p_buy > p_sell:
            return "BUY", p_buy
        else:
            return "SELL", p_sell
    
    def _apply_decision_logic(
        self,
        pattern_signal: str,
        pattern_conf: float,
        trend_direction: str,
        trend_conf: float,
        regime: str,
        mtf_score: float,
        alignment: float,
        higher_tf_aligned: bool,
    ) -> DecisionResult:
        """
        Apply comprehensive decision logic.
        
        Rules:
        1. NO_TRADE if pattern weak
        2. SIDEWAYS regime requires higher pattern confidence
        3. Trend alignment boosts confidence
        4. Counter-trend requires very high pattern confidence
        5. Higher TF disagreement penalizes signal
        """
        meta = {
            'pattern_signal': pattern_signal,
            'pattern_conf': pattern_conf,
            'trend_direction': trend_direction,
            'trend_conf': trend_conf,
            'regime': regime,
            'mtf_score': mtf_score,
            'alignment': alignment,
            'higher_tf_aligned': higher_tf_aligned,
        }
        
        # Rule 1: Pattern too weak
        if pattern_signal == "NO_TRADE":
            return DecisionResult(
                signal="NO_TRADE",
                confidence=pattern_conf,
                reason=f"Pattern weak ({pattern_conf:.2f} < {self.config.pattern_threshold:.2f})",
                strength=SignalStrength.NONE,
                meta=meta,
            )
        
        # Rule 2: SIDEWAYS/RANGING regime - require higher confidence
        if trend_direction == "SIDEWAYS" or regime in ("RANGING", "TRANSITIONAL"):
            if pattern_conf >= self.config.ranging_market_boost:
                final_conf = pattern_conf * 0.85  # Slight penalty
                return DecisionResult(
                    signal=pattern_signal,
                    confidence=final_conf,
                    reason=f"Range scalp (strong pattern {pattern_conf:.2f})",
                    strength=SignalStrength.MODERATE,
                    meta=meta,
                )
            else:
                return DecisionResult(
                    signal="NO_TRADE",
                    confidence=pattern_conf,
                    reason=f"Sideways market, pattern not strong enough ({pattern_conf:.2f} < {self.config.ranging_market_boost:.2f})",
                    strength=SignalStrength.WEAK,
                    meta=meta,
                )
        
        # Rule 3: Check trend alignment
        is_aligned = self._check_alignment(pattern_signal, trend_direction)
        
        if is_aligned:
            # Confluence: Pattern + Trend agree
            base_conf = (pattern_conf + trend_conf) / 2
            
            # Boost for good MTF alignment
            if alignment > 0.7:
                base_conf *= 1.1
            
            # Boost for strong regime
            if regime == "TRENDING":
                base_conf *= 1.05
            
            # Penalty for volatile regime
            if regime == "VOLATILE":
                base_conf *= self.config.volatile_market_penalty
            
            final_conf = min(1.0, base_conf)
            
            strength = SignalStrength.STRONG if final_conf > 0.75 else SignalStrength.MODERATE
            
            return DecisionResult(
                signal=pattern_signal,
                confidence=final_conf,
                reason=f"Confluence: Trend + Pattern aligned (p={pattern_conf:.2f}, t={trend_conf:.2f})",
                strength=strength,
                meta=meta,
            )
        
        # Rule 4: Counter-trend trade
        if self.config.allow_counter_trend:
            if pattern_conf >= self.config.counter_trend_pattern_threshold:
                if trend_conf < self.config.counter_trend_trend_threshold:
                    # Weak trend + strong pattern = possible reversal
                    final_conf = pattern_conf * self.config.counter_trend_penalty
                    
                    # Extra penalty if higher TF disagrees
                    if self.config.require_higher_tf_alignment and not higher_tf_aligned:
                        final_conf *= self.config.higher_tf_penalty
                    
                    return DecisionResult(
                        signal=pattern_signal,
                        confidence=final_conf,
                        reason=f"Counter-trend (weak trend {trend_conf:.2f}, strong pattern {pattern_conf:.2f})",
                        strength=SignalStrength.WEAK,
                        meta=meta,
                    )
        
        # Rule 5: Pattern vs Trend disagreement
        reason_dir = "bearish" if trend_direction == "BEARISH" else "bullish"
        return DecisionResult(
            signal="NO_TRADE",
            confidence=pattern_conf * 0.5,
            reason=f"Filtered: {pattern_signal} against {reason_dir} trend ({trend_conf:.2f})",
            strength=SignalStrength.NONE,
            meta=meta,
        )
    
    def _check_alignment(self, signal: str, trend: str) -> bool:
        """Check if signal aligns with trend direction."""
        if signal == "BUY" and trend == "BULLISH":
            return True
        if signal == "SELL" and trend == "BEARISH":
            return True
        return False
    
    def decide_with_mtf(
        self,
        pattern_probs: List[float],
        dfs_dict: Dict,
        trend_detector: "MTFTrendDetector",
    ) -> DecisionResult:
        """
        Convenience method that runs MTF analysis and decision in one call.
        
        Args:
            pattern_probs: [P(BUY), P(SELL), P(HOLD)]
            dfs_dict: Multi-timeframe data dict
            trend_detector: MTFTrendDetector instance
        
        Returns:
            DecisionResult
        """
        # Run MTF analysis
        mtf_result = trend_detector.detect(dfs_dict)
        
        # Make decision
        return self.decide(pattern_probs, mtf_result=mtf_result)


class MTFDecisionEngine(DecisionEngine):
    """
    Extended decision engine with built-in MTF trend detector.
    
    All-in-one solution for decision making with MTF context.
    """
    
    def __init__(
        self,
        profile: str = "SWING",
        config: Optional[DecisionConfig] = None,
        ml_model: Optional = None,
    ):
        super().__init__(config)
        
        from trend_detection.mtf_trend_detector import MTFTrendDetector
        self.trend_detector = MTFTrendDetector.from_profile(profile, ml_model=ml_model)
        self.profile = profile
    
    def analyze_and_decide(
        self,
        pattern_probs: List[float],
        dfs_dict: Dict,
    ) -> Tuple[DecisionResult, "MTFTrendResult"]:
        """
        Run full analysis and decision pipeline.
        
        Args:
            pattern_probs: [P(BUY), P(SELL), P(HOLD)]
            dfs_dict: Multi-timeframe data dict
        
        Returns:
            (DecisionResult, MTFTrendResult)
        """
        # Run MTF analysis
        mtf_result = self.trend_detector.detect(dfs_dict)
        
        # Make decision
        decision = self.decide(pattern_probs, mtf_result=mtf_result)
        
        return decision, mtf_result
    
    def get_recommendation(
        self,
        pattern_probs: List[float],
        dfs_dict: Dict,
    ) -> Dict:
        """
        Get comprehensive trading recommendation.
        
        Returns dict with all relevant information for trade execution.
        """
        decision, mtf_result = self.analyze_and_decide(pattern_probs, dfs_dict)
        
        return {
            # Decision
            'signal': decision.signal,
            'confidence': decision.confidence,
            'strength': decision.strength.value,
            'reason': decision.reason,
            'tradeable': decision.is_tradeable(),
            
            # Trend context
            'trend': mtf_result.trend_name,
            'trend_direction': mtf_result.direction,
            'trend_strength': mtf_result.trend_strength,
            
            # MTF info
            'mtf_score': mtf_result.mtf_score,
            'mtf_alignment': mtf_result.mtf_alignment,
            'higher_tf_aligned': mtf_result.higher_tf_aligned,
            
            # Regime
            'regime': mtf_result.regime,
            
            # Per-TF breakdown
            'timeframe_scores': mtf_result.timeframe_scores,
            'timeframe_directions': mtf_result.timeframe_directions,
            
            # Meta
            'pattern_probs': pattern_probs,
            'profile': self.profile,
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_decision_engine(
    engine_type: str = "standard",
    profile: str = "SWING",
    **kwargs
) -> DecisionEngine:
    """
    Factory function for creating decision engines.
    
    Args:
        engine_type: 'standard' or 'mtf'
        profile: MTF profile for 'mtf' type
        **kwargs: Additional arguments
    
    Returns:
        DecisionEngine instance
    """
    if engine_type == "mtf":
        return MTFDecisionEngine(profile=profile, **kwargs)
    else:
        return DecisionEngine(**kwargs)