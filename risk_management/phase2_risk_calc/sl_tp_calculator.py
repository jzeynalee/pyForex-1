"""
Phase 2: Stop-Loss / Take-Profit Calculator

This module calculates optimal SL/TP levels using:
- Quantile predictions for price distribution
- Volatility forecasts
- Market regime adjustments
- Direction confidence

Key principle: SL/TP should be data-driven, not arbitrary pip values.
"""

import numpy as np
from typing import Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification for adaptive risk."""
    TRENDING_STRONG = "trending_strong"
    TRENDING_WEAK = "trending_weak"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"


class TradeDirection(Enum):
    """Trade direction."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class SLTPConfig:
    """Configuration for SL/TP calculation."""
    # Quantile-based SL/TP selection
    sl_quantile_buy: float = 0.05    # Q5 for buy SL (worst case down)
    sl_quantile_sell: float = 0.95   # Q95 for sell SL (worst case up)
    tp_quantile_buy: float = 0.75    # Q75 for buy TP (good case up)
    tp_quantile_sell: float = 0.25   # Q25 for sell TP (good case down)
    
    # Volatility multipliers
    min_sl_atr_multiple: float = 1.0   # Minimum SL distance in ATR
    max_sl_atr_multiple: float = 3.0   # Maximum SL distance in ATR
    min_tp_atr_multiple: float = 1.5   # Minimum TP distance in ATR
    max_tp_atr_multiple: float = 5.0   # Maximum TP distance in ATR
    
    # Risk-reward constraints
    min_risk_reward: float = 1.5       # Minimum acceptable R:R ratio
    target_risk_reward: float = 2.0    # Target R:R ratio
    
    # Regime adjustments
    regime_sl_multiplier: Dict[MarketRegime, float] = field(default_factory=lambda: {
        MarketRegime.TRENDING_STRONG: 1.2,   # Wider SL in strong trends
        MarketRegime.TRENDING_WEAK: 1.0,
        MarketRegime.RANGING: 0.8,            # Tighter SL in ranges
        MarketRegime.VOLATILE: 1.5,           # Much wider in volatile
        MarketRegime.LOW_VOLATILITY: 0.7      # Tighter in calm markets
    })
    
    regime_tp_multiplier: Dict[MarketRegime, float] = field(default_factory=lambda: {
        MarketRegime.TRENDING_STRONG: 1.5,   # Larger targets in trends
        MarketRegime.TRENDING_WEAK: 1.2,
        MarketRegime.RANGING: 0.8,            # Smaller targets in ranges
        MarketRegime.VOLATILE: 1.3,
        MarketRegime.LOW_VOLATILITY: 0.9
    })
    
    # Confidence-based adjustments
    low_confidence_threshold: float = 0.5     # Below this, widen SL
    high_confidence_threshold: float = 0.75   # Above this, can tighten SL


class SLTPResult(NamedTuple):
    """Result of SL/TP calculation."""
    stop_loss: float
    take_profit: float
    sl_distance: float       # Distance from entry to SL
    tp_distance: float       # Distance from entry to TP
    risk_reward_ratio: float
    sl_method: str           # Which method determined SL
    tp_method: str           # Which method determined TP
    regime_adjusted: bool    # Whether regime adjustment was applied
    confidence_adjusted: bool # Whether confidence adjustment was applied


class SLTPCalculator:
    """
    Calculates optimal Stop-Loss and Take-Profit levels.
    
    Uses a multi-method approach:
    1. Primary: Quantile-based (distribution of price movements)
    2. Secondary: Volatility-based (ATR multiples)
    3. Constraints: Risk-reward ratio enforcement
    4. Adjustments: Regime and confidence-based modifications
    """
    
    def __init__(self, config: Optional[SLTPConfig] = None):
        self.config = config or SLTPConfig()
    
    def calculate(
        self,
        entry_price: float,
        direction: TradeDirection,
        quantiles: np.ndarray,          # [Q5, Q25, Q50, Q75, Q95]
        volatility: float,              # Predicted sigma
        regime: Optional[MarketRegime] = None,
        direction_confidence: Optional[float] = None,
        atr: Optional[float] = None     # Actual ATR if available
    ) -> SLTPResult:
        """
        Calculate optimal SL/TP levels.
        
        Args:
            entry_price: Trade entry price
            direction: BUY or SELL
            quantiles: Predicted price movement quantiles [Q5, Q25, Q50, Q75, Q95]
            volatility: Predicted volatility (sigma)
            regime: Optional market regime for adjustment
            direction_confidence: Optional confidence score [0, 1]
            atr: Optional actual ATR (uses volatility if not provided)
        
        Returns:
            SLTPResult with calculated levels and metadata
        """
        # Use ATR or volatility for distance calculations
        vol_measure = atr if atr is not None else volatility
        
        # Step 1: Calculate quantile-based SL/TP
        sl_quant, tp_quant = self._calculate_quantile_based(
            entry_price, direction, quantiles
        )
        
        # Step 2: Calculate volatility-based SL/TP (fallback/bounds)
        sl_vol, tp_vol = self._calculate_volatility_based(
            entry_price, direction, vol_measure
        )
        
        # Step 3: Select primary values with bounds
        sl_price, sl_method = self._select_sl(
            entry_price, direction, sl_quant, sl_vol, vol_measure
        )
        tp_price, tp_method = self._select_tp(
            entry_price, direction, tp_quant, tp_vol, vol_measure
        )
        
        # Step 4: Apply regime adjustments
        regime_adjusted = False
        if regime is not None:
            sl_price, tp_price = self._apply_regime_adjustment(
                entry_price, direction, sl_price, tp_price, regime
            )
            regime_adjusted = True
        
        # Step 5: Apply confidence adjustments
        confidence_adjusted = False
        if direction_confidence is not None:
            sl_price, tp_price = self._apply_confidence_adjustment(
                entry_price, direction, sl_price, tp_price,
                direction_confidence, vol_measure
            )
            confidence_adjusted = True
        
        # Step 6: Enforce minimum risk-reward ratio
        sl_price, tp_price = self._enforce_risk_reward(
            entry_price, direction, sl_price, tp_price
        )
        
        # Calculate final distances and ratio
        sl_distance = abs(entry_price - sl_price)
        tp_distance = abs(tp_price - entry_price)
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
        
        return SLTPResult(
            stop_loss=sl_price,
            take_profit=tp_price,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            risk_reward_ratio=rr_ratio,
            sl_method=sl_method,
            tp_method=tp_method,
            regime_adjusted=regime_adjusted,
            confidence_adjusted=confidence_adjusted
        )
    
    def _calculate_quantile_based(
        self,
        entry_price: float,
        direction: TradeDirection,
        quantiles: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate SL/TP based on predicted price distribution.
        
        Quantiles represent expected price movements:
        - Q5: 5th percentile (worst case move)
        - Q95: 95th percentile (best case move)
        """
        q5, q25, q50, q75, q95 = quantiles
        
        if direction == TradeDirection.BUY:
            # For BUY: SL below entry (use Q5), TP above (use Q75)
            sl_price = entry_price + q5   # q5 is typically negative
            tp_price = entry_price + q75  # q75 is typically positive
        else:
            # For SELL: SL above entry (use Q95), TP below (use Q25)
            sl_price = entry_price + q95  # q95 is typically positive
            tp_price = entry_price + q25  # q25 is typically negative
        
        return sl_price, tp_price
    
    def _calculate_volatility_based(
        self,
        entry_price: float,
        direction: TradeDirection,
        volatility: float
    ) -> Tuple[float, float]:
        """
        Calculate SL/TP based on volatility multiples.
        
        This serves as a fallback and provides bounds.
        """
        sl_distance = volatility * 1.5  # 1.5x volatility for SL
        tp_distance = volatility * 2.5  # 2.5x volatility for TP
        
        if direction == TradeDirection.BUY:
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        
        return sl_price, tp_price
    
    def _select_sl(
        self,
        entry_price: float,
        direction: TradeDirection,
        sl_quant: float,
        sl_vol: float,
        vol_measure: float
    ) -> Tuple[float, str]:
        """
        Select SL price with bounds enforcement.
        
        Returns (sl_price, method_used)
        """
        # Calculate distances
        dist_quant = abs(entry_price - sl_quant)
        dist_vol = abs(entry_price - sl_vol)
        
        # Bounds
        min_dist = vol_measure * self.config.min_sl_atr_multiple
        max_dist = vol_measure * self.config.max_sl_atr_multiple
        
        # Primary: use quantile-based if within bounds
        if min_dist <= dist_quant <= max_dist:
            return sl_quant, "quantile"
        
        # Clamp to bounds
        if dist_quant < min_dist:
            clamped_dist = min_dist
            method = "min_bound"
        else:
            clamped_dist = max_dist
            method = "max_bound"
        
        if direction == TradeDirection.BUY:
            sl_price = entry_price - clamped_dist
        else:
            sl_price = entry_price + clamped_dist
        
        return sl_price, method
    
    def _select_tp(
        self,
        entry_price: float,
        direction: TradeDirection,
        tp_quant: float,
        tp_vol: float,
        vol_measure: float
    ) -> Tuple[float, str]:
        """
        Select TP price with bounds enforcement.
        
        Returns (tp_price, method_used)
        """
        dist_quant = abs(tp_quant - entry_price)
        dist_vol = abs(tp_vol - entry_price)
        
        min_dist = vol_measure * self.config.min_tp_atr_multiple
        max_dist = vol_measure * self.config.max_tp_atr_multiple
        
        if min_dist <= dist_quant <= max_dist:
            return tp_quant, "quantile"
        
        if dist_quant < min_dist:
            clamped_dist = min_dist
            method = "min_bound"
        else:
            clamped_dist = max_dist
            method = "max_bound"
        
        if direction == TradeDirection.BUY:
            tp_price = entry_price + clamped_dist
        else:
            tp_price = entry_price - clamped_dist
        
        return tp_price, method
    
    def _apply_regime_adjustment(
        self,
        entry_price: float,
        direction: TradeDirection,
        sl_price: float,
        tp_price: float,
        regime: MarketRegime
    ) -> Tuple[float, float]:
        """Apply regime-based adjustments to SL/TP."""
        sl_mult = self.config.regime_sl_multiplier.get(regime, 1.0)
        tp_mult = self.config.regime_tp_multiplier.get(regime, 1.0)
        
        sl_dist = abs(entry_price - sl_price) * sl_mult
        tp_dist = abs(tp_price - entry_price) * tp_mult
        
        if direction == TradeDirection.BUY:
            return entry_price - sl_dist, entry_price + tp_dist
        else:
            return entry_price + sl_dist, entry_price - tp_dist
    
    def _apply_confidence_adjustment(
        self,
        entry_price: float,
        direction: TradeDirection,
        sl_price: float,
        tp_price: float,
        confidence: float,
        vol_measure: float
    ) -> Tuple[float, float]:
        """
        Adjust SL/TP based on prediction confidence.
        
        Low confidence -> wider SL (more room for error)
        High confidence -> can tighten SL slightly
        """
        sl_dist = abs(entry_price - sl_price)
        tp_dist = abs(tp_price - entry_price)
        
        if confidence < self.config.low_confidence_threshold:
            # Low confidence: widen SL by 20%
            sl_dist *= 1.2
            # Reduce TP slightly (less aggressive)
            tp_dist *= 0.9
        elif confidence > self.config.high_confidence_threshold:
            # High confidence: can tighten SL by 10%
            sl_dist *= 0.9
            # Can be more aggressive with TP
            tp_dist *= 1.1
        
        # Enforce minimum distance
        min_sl = vol_measure * self.config.min_sl_atr_multiple
        sl_dist = max(sl_dist, min_sl)
        
        if direction == TradeDirection.BUY:
            return entry_price - sl_dist, entry_price + tp_dist
        else:
            return entry_price + sl_dist, entry_price - tp_dist
    
    def _enforce_risk_reward(
        self,
        entry_price: float,
        direction: TradeDirection,
        sl_price: float,
        tp_price: float
    ) -> Tuple[float, float]:
        """
        Ensure minimum risk-reward ratio is met.
        
        If R:R is too low, extend TP to meet minimum.
        """
        sl_dist = abs(entry_price - sl_price)
        tp_dist = abs(tp_price - entry_price)
        
        current_rr = tp_dist / sl_dist if sl_dist > 0 else 0
        
        if current_rr < self.config.min_risk_reward:
            # Extend TP to meet minimum R:R
            required_tp_dist = sl_dist * self.config.min_risk_reward
            
            if direction == TradeDirection.BUY:
                tp_price = entry_price + required_tp_dist
            else:
                tp_price = entry_price - required_tp_dist
            
            logger.debug(
                f"Extended TP to meet min R:R. "
                f"Original R:R: {current_rr:.2f}, New R:R: {self.config.min_risk_reward}"
            )
        
        return sl_price, tp_price
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        direction: TradeDirection,
        original_sl: float,
        volatility: float,
        profit_lock_ratio: float = 0.5
    ) -> float:
        """
        Calculate a trailing stop level.
        
        Trails the stop to lock in profits as price moves favorably.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            direction: Trade direction
            original_sl: Original stop loss level
            volatility: Current volatility measure
            profit_lock_ratio: Fraction of profit to lock (0.5 = 50%)
        
        Returns:
            New stop loss level (may be same as original if no improvement)
        """
        if direction == TradeDirection.BUY:
            profit = current_price - entry_price
            if profit <= 0:
                return original_sl
            
            # New SL = entry + (profit * lock_ratio) - volatility buffer
            new_sl = entry_price + (profit * profit_lock_ratio) - (volatility * 0.5)
            return max(new_sl, original_sl)  # Only trail up, never down
        else:
            profit = entry_price - current_price
            if profit <= 0:
                return original_sl
            
            new_sl = entry_price - (profit * profit_lock_ratio) + (volatility * 0.5)
            return min(new_sl, original_sl)  # Only trail down, never up


class PartialExitCalculator:
    """
    Calculates partial exit levels for scaling out of positions.
    
    Strategy: Take partial profits at intermediate levels to lock in gains
    while letting remainder run for larger targets.
    """
    
    def __init__(
        self,
        exit_levels: Tuple[float, ...] = (0.33, 0.67),  # Exit at 33% and 67% of TP
        exit_portions: Tuple[float, ...] = (0.5, 0.3)   # Exit 50%, then 30%
    ):
        """
        Args:
            exit_levels: Fractions of TP distance to trigger exits
            exit_portions: Fraction of remaining position to close at each level
        """
        if len(exit_levels) != len(exit_portions):
            raise ValueError("exit_levels and exit_portions must have same length")
        
        self.exit_levels = exit_levels
        self.exit_portions = exit_portions
    
    def calculate_levels(
        self,
        entry_price: float,
        direction: TradeDirection,
        sl_price: float,
        tp_price: float
    ) -> list:
        """
        Calculate partial exit levels.
        
        Returns:
            List of (price, portion_to_close) tuples
        """
        tp_distance = abs(tp_price - entry_price)
        
        exits = []
        for level_frac, portion in zip(self.exit_levels, self.exit_portions):
            distance = tp_distance * level_frac
            
            if direction == TradeDirection.BUY:
                price = entry_price + distance
            else:
                price = entry_price - distance
            
            exits.append({
                'price': price,
                'portion': portion,
                'level_name': f"TP_{int(level_frac * 100)}%"
            })
        
        return exits


def calculate_sl_tp_from_predictions(
    entry_price: float,
    direction: str,  # 'BUY' or 'SELL'
    predictions: Dict,  # From MultiHeadTCN.predict_risk_params()
    regime: Optional[str] = None,
    atr: Optional[float] = None,
    config: Optional[SLTPConfig] = None
) -> SLTPResult:
    """
    Convenience function to calculate SL/TP directly from model predictions.
    
    Args:
        entry_price: Trade entry price
        direction: 'BUY' or 'SELL'
        predictions: Dictionary with 'quantiles', 'volatility', 'direction_probs'
        regime: Optional regime string
        atr: Optional ATR value
        config: Optional SLTPConfig
    
    Returns:
        SLTPResult with calculated levels
    """
    calculator = SLTPCalculator(config)
    
    # Convert direction
    trade_dir = TradeDirection[direction.upper()]
    
    # Convert regime if provided
    market_regime = None
    if regime:
        try:
            market_regime = MarketRegime(regime.lower())
        except ValueError:
            logger.warning(f"Unknown regime: {regime}, ignoring")
    
    # Get confidence from direction probabilities
    confidence = None
    if 'direction_probs' in predictions:
        probs = predictions['direction_probs']
        if hasattr(probs, 'numpy'):
            probs = probs.numpy()
        confidence = float(np.max(probs))
    
    # Get quantiles
    quantiles = predictions['quantiles']
    if hasattr(quantiles, 'numpy'):
        quantiles = quantiles.numpy()
    if quantiles.ndim > 1:
        quantiles = quantiles[0]  # Take first batch element
    
    # Get volatility
    volatility = predictions['volatility']
    if hasattr(volatility, 'item'):
        volatility = volatility.item()
    
    return calculator.calculate(
        entry_price=entry_price,
        direction=trade_dir,
        quantiles=quantiles,
        volatility=volatility,
        regime=market_regime,
        direction_confidence=confidence,
        atr=atr
    )
