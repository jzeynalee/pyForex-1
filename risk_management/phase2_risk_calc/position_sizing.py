"""
Phase 2: Position Sizing Calculator

This module calculates optimal position sizes using:
- Account risk percentage
- Volatility-based adjustments
- Direction confidence weighting
- Kelly criterion (optional)
- Portfolio correlation constraints

Key principle: Risk a fixed percentage of capital, adjusted by market conditions.
"""

import numpy as np
from typing import Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AccountCurrency(Enum):
    """Account base currency."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""
    # Base risk parameters
    base_risk_percent: float = 1.0      # Risk 1% of account per trade
    max_risk_percent: float = 2.0       # Maximum risk per trade
    min_risk_percent: float = 0.25      # Minimum risk per trade
    
    # Confidence-based adjustments
    confidence_scaling: bool = True
    min_confidence_for_full_size: float = 0.7
    low_confidence_risk_reduction: float = 0.5  # 50% reduction at low confidence
    
    # Volatility-based adjustments
    volatility_scaling: bool = True
    normal_volatility_percentile: Tuple[float, float] = (25, 75)  # Normal range
    high_vol_risk_reduction: float = 0.6  # 40% reduction in high vol
    low_vol_risk_increase: float = 1.2    # 20% increase in low vol
    
    # Kelly criterion
    use_kelly: bool = False
    kelly_fraction: float = 0.25         # Fractional Kelly (more conservative)
    
    # Streak adjustments
    use_streak_adjustment: bool = True
    losing_streak_threshold: int = 3
    losing_streak_reduction: float = 0.7  # 30% reduction after losing streak
    
    # Max exposure limits
    max_single_pair_exposure: float = 5.0    # Max 5% of account in one pair
    max_correlated_exposure: float = 10.0    # Max 10% in correlated pairs
    max_total_exposure: float = 20.0         # Max 20% total risk exposure
    
    # Rounding
    lot_size_precision: int = 2              # Round to 0.01 lots


class PositionSizeResult(NamedTuple):
    """Result of position size calculation."""
    position_size: float        # In lots (standard lots)
    units: int                  # Number of units
    risk_amount: float          # Dollar amount at risk
    risk_percent: float         # Actual risk percentage
    adjustment_factors: Dict    # What adjustments were applied
    warnings: list             # Any warnings about the position


class PositionSizingCalculator:
    """
    Calculates optimal position sizes for forex trades.
    
    Core formula:
    Position Size = (Account Risk Amount) / (SL Distance in Price * Pip Value)
    
    With adjustments for:
    - Model confidence
    - Market volatility
    - Win/loss streaks
    - Portfolio exposure
    """
    
    def __init__(self, config: Optional[PositionSizingConfig] = None):
        self.config = config or PositionSizingConfig()
        
        # Historical volatility for percentile calculations
        self._volatility_history = []
        self._max_history_length = 100
    
    def calculate(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        pair: str,
        direction_confidence: Optional[float] = None,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
        recent_streak: Optional[int] = None,  # Negative for losing streak
        current_exposure: Optional[Dict[str, float]] = None,
        account_currency: AccountCurrency = AccountCurrency.USD
    ) -> PositionSizeResult:
        """
        Calculate optimal position size.
        
        Args:
            account_balance: Current account balance
            entry_price: Trade entry price
            stop_loss: Stop loss price
            pair: Currency pair (e.g., 'EURUSD')
            direction_confidence: Model confidence [0, 1]
            volatility: Current volatility measure
            win_rate: Historical win rate [0, 1]
            avg_win_loss_ratio: Average win / average loss
            recent_streak: Recent win/loss streak (negative = losses)
            current_exposure: Dict of pair -> exposure amount
            account_currency: Account base currency
        
        Returns:
            PositionSizeResult with calculated size and metadata
        """
        warnings = []
        adjustment_factors = {}
        
        # Step 1: Calculate base risk amount
        risk_percent = self.config.base_risk_percent
        
        # Step 2: Apply confidence adjustment
        if self.config.confidence_scaling and direction_confidence is not None:
            conf_factor = self._calculate_confidence_factor(direction_confidence)
            risk_percent *= conf_factor
            adjustment_factors['confidence'] = conf_factor
        
        # Step 3: Apply volatility adjustment
        if self.config.volatility_scaling and volatility is not None:
            vol_factor = self._calculate_volatility_factor(volatility)
            risk_percent *= vol_factor
            adjustment_factors['volatility'] = vol_factor
        
        # Step 4: Apply Kelly criterion (if enabled and data available)
        if self.config.use_kelly and win_rate is not None and avg_win_loss_ratio is not None:
            kelly_factor = self._calculate_kelly_factor(win_rate, avg_win_loss_ratio)
            risk_percent = min(risk_percent, kelly_factor * 100)  # Kelly caps the risk
            adjustment_factors['kelly'] = kelly_factor
        
        # Step 5: Apply streak adjustment
        if self.config.use_streak_adjustment and recent_streak is not None:
            streak_factor = self._calculate_streak_factor(recent_streak)
            risk_percent *= streak_factor
            adjustment_factors['streak'] = streak_factor
        
        # Step 6: Clamp to min/max risk
        original_risk = risk_percent
        risk_percent = np.clip(
            risk_percent,
            self.config.min_risk_percent,
            self.config.max_risk_percent
        )
        if risk_percent != original_risk:
            adjustment_factors['clamped'] = True
        
        # Calculate risk amount in account currency
        risk_amount = account_balance * (risk_percent / 100)
        
        # Step 7: Calculate SL distance in pips
        sl_distance_price = abs(entry_price - stop_loss)
        sl_distance_pips = self._price_to_pips(sl_distance_price, pair)
        
        # Step 8: Calculate pip value
        pip_value = self._get_pip_value(pair, entry_price, account_currency)
        
        # Step 9: Calculate position size
        if sl_distance_pips > 0 and pip_value > 0:
            # Position size in standard lots
            position_size = risk_amount / (sl_distance_pips * pip_value)
        else:
            position_size = 0
            warnings.append("Invalid SL distance or pip value")
        
        # Step 10: Check exposure limits
        if current_exposure is not None:
            position_size, exposure_warnings = self._apply_exposure_limits(
                position_size, pair, risk_amount, current_exposure, account_balance
            )
            warnings.extend(exposure_warnings)
        
        # Step 11: Round to lot precision
        position_size = round(position_size, self.config.lot_size_precision)
        
        # Calculate units (1 standard lot = 100,000 units)
        units = int(position_size * 100000)
        
        # Recalculate actual risk
        actual_risk_amount = position_size * sl_distance_pips * pip_value
        actual_risk_percent = (actual_risk_amount / account_balance) * 100
        
        return PositionSizeResult(
            position_size=position_size,
            units=units,
            risk_amount=actual_risk_amount,
            risk_percent=actual_risk_percent,
            adjustment_factors=adjustment_factors,
            warnings=warnings
        )
    
    def _calculate_confidence_factor(self, confidence: float) -> float:
        """
        Calculate risk adjustment based on model confidence.
        
        Low confidence -> reduced risk
        High confidence -> full risk
        """
        if confidence >= self.config.min_confidence_for_full_size:
            return 1.0
        
        # Linear interpolation from low_confidence_reduction to 1.0
        min_factor = self.config.low_confidence_risk_reduction
        factor = min_factor + (1 - min_factor) * (
            confidence / self.config.min_confidence_for_full_size
        )
        return max(min_factor, factor)
    
    def _calculate_volatility_factor(self, volatility: float) -> float:
        """
        Calculate risk adjustment based on volatility.
        
        Uses percentile ranking against historical volatility.
        """
        # Update history
        self._volatility_history.append(volatility)
        if len(self._volatility_history) > self._max_history_length:
            self._volatility_history.pop(0)
        
        # Need enough history for percentile calculation
        if len(self._volatility_history) < 20:
            return 1.0
        
        # Calculate percentile
        percentile = np.percentile(
            self._volatility_history,
            [self.config.normal_volatility_percentile[0],
             self.config.normal_volatility_percentile[1]]
        )
        
        if volatility < percentile[0]:
            # Low volatility - can increase risk slightly
            return self.config.low_vol_risk_increase
        elif volatility > percentile[1]:
            # High volatility - reduce risk
            return self.config.high_vol_risk_reduction
        else:
            # Normal volatility
            return 1.0
    
    def _calculate_kelly_factor(
        self,
        win_rate: float,
        avg_win_loss_ratio: float
    ) -> float:
        """
        Calculate Kelly criterion for optimal bet sizing.
        
        Kelly % = W - (1-W)/R
        where W = win rate, R = win/loss ratio
        
        Returns fractional Kelly for more conservative sizing.
        """
        if avg_win_loss_ratio <= 0:
            return 0.01  # Minimum
        
        kelly = win_rate - ((1 - win_rate) / avg_win_loss_ratio)
        
        # Apply fractional Kelly (more conservative)
        kelly *= self.config.kelly_fraction
        
        # Clamp to reasonable range
        return max(0.01, min(kelly, self.config.max_risk_percent / 100))
    
    def _calculate_streak_factor(self, streak: int) -> float:
        """
        Adjust risk based on recent win/loss streak.
        
        Reduces risk after losing streaks to preserve capital.
        """
        if streak >= 0:
            # Winning streak - no adjustment (don't get overconfident)
            return 1.0
        
        # Losing streak
        if abs(streak) >= self.config.losing_streak_threshold:
            return self.config.losing_streak_reduction
        
        return 1.0
    
    def _price_to_pips(self, price_diff: float, pair: str) -> float:
        """Convert price difference to pips."""
        # JPY pairs have 2 decimal places, others have 4
        if 'JPY' in pair.upper():
            return price_diff * 100
        else:
            return price_diff * 10000
    
    def _get_pip_value(
        self,
        pair: str,
        current_price: float,
        account_currency: AccountCurrency
    ) -> float:
        """
        Calculate pip value in account currency.
        
        For standard lot (100,000 units):
        - If quote currency = account currency: pip value = 10
        - If quote currency ≠ account currency: convert
        """
        pair = pair.upper()
        base = pair[:3]
        quote = pair[3:]
        
        # Standard pip value for 1 standard lot
        if quote == account_currency.value:
            # Direct quote (e.g., EURUSD for USD account)
            return 10.0
        elif base == account_currency.value:
            # Indirect quote (e.g., USDJPY for USD account)
            if 'JPY' in pair:
                return 1000 / current_price
            else:
                return 10 / current_price
        else:
            # Cross rate - need conversion (simplified)
            # In production, would look up actual exchange rate
            return 10.0  # Approximate
    
    def _apply_exposure_limits(
        self,
        position_size: float,
        pair: str,
        risk_amount: float,
        current_exposure: Dict[str, float],
        account_balance: float
    ) -> Tuple[float, list]:
        """
        Apply exposure limits to position size.
        
        Returns (adjusted_size, warnings)
        """
        warnings = []
        
        # Current exposure for this pair
        pair_exposure = current_exposure.get(pair, 0)
        new_pair_exposure = pair_exposure + risk_amount
        max_pair = account_balance * (self.config.max_single_pair_exposure / 100)
        
        if new_pair_exposure > max_pair:
            # Reduce position to stay within limit
            available = max(0, max_pair - pair_exposure)
            if available <= 0:
                warnings.append(f"Max exposure reached for {pair}")
                return 0, warnings
            
            reduction_ratio = available / risk_amount
            position_size *= reduction_ratio
            warnings.append(f"Position reduced to stay within {pair} exposure limit")
        
        # Total exposure check
        total_exposure = sum(current_exposure.values()) + risk_amount
        max_total = account_balance * (self.config.max_total_exposure / 100)
        
        if total_exposure > max_total:
            available = max(0, max_total - sum(current_exposure.values()))
            if available <= 0:
                warnings.append("Max total exposure reached")
                return 0, warnings
            
            reduction_ratio = available / risk_amount
            position_size *= min(position_size, position_size * reduction_ratio)
            warnings.append("Position reduced to stay within total exposure limit")
        
        return position_size, warnings
    
    def calculate_correlated_exposure(
        self,
        positions: Dict[str, float],
        correlations: Dict[Tuple[str, str], float]
    ) -> float:
        """
        Calculate effective exposure considering correlations.
        
        Highly correlated positions effectively multiply risk.
        """
        pairs = list(positions.keys())
        n = len(pairs)
        
        if n <= 1:
            return sum(positions.values())
        
        # Build correlation matrix
        corr_matrix = np.eye(n)
        for i, pair1 in enumerate(pairs):
            for j, pair2 in enumerate(pairs):
                if i != j:
                    key = (pair1, pair2) if pair1 < pair2 else (pair2, pair1)
                    corr_matrix[i, j] = correlations.get(key, 0)
        
        # Calculate portfolio variance
        exposures = np.array([positions[p] for p in pairs])
        portfolio_variance = exposures @ corr_matrix @ exposures
        
        # Effective exposure is sqrt of variance
        effective_exposure = np.sqrt(portfolio_variance)
        
        return effective_exposure


class ScaledPositionCalculator:
    """
    Calculates position sizes for scaled entries.
    
    Supports:
    - Multiple entry points (scaling in)
    - Pyramid entries in trending markets
    - Cost averaging strategies
    """
    
    def __init__(self, total_risk_percent: float = 2.0):
        """
        Args:
            total_risk_percent: Total risk across all scaled entries
        """
        self.total_risk_percent = total_risk_percent
    
    def calculate_scaled_entries(
        self,
        account_balance: float,
        entry_prices: list,
        stop_loss: float,
        pair: str,
        scaling_weights: Optional[list] = None
    ) -> list:
        """
        Calculate position sizes for multiple entry points.
        
        Args:
            account_balance: Account balance
            entry_prices: List of entry prices
            stop_loss: Common stop loss for all entries
            pair: Currency pair
            scaling_weights: Optional weights for each entry (must sum to 1)
        
        Returns:
            List of position sizes for each entry
        """
        n_entries = len(entry_prices)
        
        if scaling_weights is None:
            # Equal weighting by default
            scaling_weights = [1 / n_entries] * n_entries
        
        assert abs(sum(scaling_weights) - 1.0) < 0.01, "Weights must sum to 1"
        
        total_risk_amount = account_balance * (self.total_risk_percent / 100)
        
        positions = []
        base_calculator = PositionSizingCalculator()
        
        for entry_price, weight in zip(entry_prices, scaling_weights):
            allocated_risk = total_risk_amount * weight
            
            # Calculate position for this entry
            sl_distance_pips = base_calculator._price_to_pips(
                abs(entry_price - stop_loss), pair
            )
            pip_value = base_calculator._get_pip_value(
                pair, entry_price, AccountCurrency.USD
            )
            
            if sl_distance_pips > 0 and pip_value > 0:
                size = allocated_risk / (sl_distance_pips * pip_value)
            else:
                size = 0
            
            positions.append({
                'entry_price': entry_price,
                'position_size': round(size, 2),
                'allocated_risk': allocated_risk,
                'weight': weight
            })
        
        return positions


def calculate_position_from_predictions(
    account_balance: float,
    entry_price: float,
    sl_tp_result,  # SLTPResult from sl_tp_calculator
    pair: str,
    predictions: Dict,  # From MultiHeadTCN
    config: Optional[PositionSizingConfig] = None
) -> PositionSizeResult:
    """
    Convenience function to calculate position size from model predictions.
    
    Args:
        account_balance: Current account balance
        entry_price: Trade entry price
        sl_tp_result: SLTPResult from SLTPCalculator
        pair: Currency pair
        predictions: Model predictions dict
        config: Optional PositionSizingConfig
    
    Returns:
        PositionSizeResult
    """
    calculator = PositionSizingCalculator(config)
    
    # Extract confidence from direction probabilities
    confidence = None
    if 'direction_probs' in predictions:
        probs = predictions['direction_probs']
        if hasattr(probs, 'numpy'):
            probs = probs.numpy()
        confidence = float(np.max(probs))
    
    # Extract volatility
    volatility = predictions.get('volatility')
    if volatility is not None and hasattr(volatility, 'item'):
        volatility = volatility.item()
    
    return calculator.calculate(
        account_balance=account_balance,
        entry_price=entry_price,
        stop_loss=sl_tp_result.stop_loss,
        pair=pair,
        direction_confidence=confidence,
        volatility=volatility
    )
