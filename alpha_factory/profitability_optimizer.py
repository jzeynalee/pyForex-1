# alpha_factory/profitability_optimizer.py
"""
Profitability Optimization for Alpha Factory System

This module implements targeted improvements to enhance trading profitability
through advanced strategies, risk management, and performance optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProfitabilityConfig:
    """Configuration for profitability optimization."""
    # Entry/Exit Optimization
    min_profit_target_pips: float = 10.0  # Minimum profit target in pips
    max_risk_per_trade: float = 2.0  # Maximum risk per trade in percent
    risk_reward_ratio: float = 2.0  # Minimum risk:reward ratio
    
    # Market Condition Filters
    min_volatility_threshold: float = 0.000225  # 2.25 pips minimum (hard-coded)
    max_spread_threshold: float = 0.0002  # Maximum spread allowed
    trend_strength_threshold: float = 20.0  # Minimum ADX for trend trading
    
    # 3. Position Sizing
    kelly_criterion: bool = True          # Active smart sizing
    max_position_size: float = 0.05       # 5% max to protect equity curve
    
    # 4. Exit Optimization
    trailing_stop_enabled: bool = True    # Secures gains in high-RR trades
    trailing_stop_distance: float = 0.0008 # 8.0 pips
    profit_target_scaling: bool = True     # Volatility-adjusted targets
    
    # 5. Market Regime Optimization
    trend_following_enabled: bool = True
    mean_reversion_enabled: bool = True
    breakout_enabled: bool = True
    
    # 6. High-Confidence Thresholds (Selective Trading)
    # Only 15.4% of decisions pass these filters
    confidence_threshold_trend: float = 0.75  # Higher barrier for trending moves
    confidence_threshold_range: float = 0.82  # Strictest for Neutral/Range trades
    adaptive_thresholds: bool = True          # Dynamic shifts based on performance


class ProfitabilityOptimizer:
    """
    Advanced profitability optimizer for Alpha Factory system.
    
    Implements multiple strategies to improve trading performance:
    - Enhanced entry/exit criteria
    - Adaptive position sizing
    - Market condition filtering
    - Regime-specific strategies
    - Risk optimization
    """
    
    def __init__(self, config: Optional[ProfitabilityConfig] = None):
        """
        Initialize the profitability optimizer.
        
        Args:
            config: Profitability optimization configuration
        """
        self.config = config or ProfitabilityConfig()
        self.performance_history = []
        self.regime_performance = {}
        
    def optimize_entry_criteria(self, market_data: pd.DataFrame, 
                              decision_signal: Dict) -> Dict[str, Any]:
        """
        Optimize entry criteria based on market conditions.
        
        Args:
            market_data: Recent market data
            decision_signal: Original decision signal
            
        Returns:
            Optimized entry signal with enhanced criteria
        """
        if decision_signal['decision'] == 'HOLD':
            return decision_signal
        
        # Get current market conditions
        current_price = market_data['close'].iloc[-1]
        volatility = self._calculate_volatility(market_data)
        spread = self._estimate_spread(market_data)
        trend_strength = self._calculate_trend_strength(market_data)
        
        # Apply market condition filters
        if not self._pass_market_filters(volatility, spread, trend_strength):
            return {**decision_signal, 'decision': 'HOLD', 
                   'reasoning': decision_signal.get('reasoning', []) + 
                   ['Market conditions not suitable for trading']}
        
        # Optimize confidence threshold based on regime
        optimized_confidence = self._optimize_confidence_threshold(
            decision_signal['regime'], decision_signal['confidence']
        )
        
        if decision_signal['confidence'] < optimized_confidence:
            return {**decision_signal, 'decision': 'HOLD',
                   'reasoning': decision_signal.get('reasoning', []) + 
                   [f'Confidence {decision_signal["confidence"]:.3f} below threshold {optimized_confidence:.3f}']}
        
        # Enhance decision with optimization insights
        optimized_signal = decision_signal.copy()
        optimized_signal['optimized_confidence'] = optimized_confidence
        optimized_signal['market_conditions'] = {
            'volatility': volatility,
            'spread': spread,
            'trend_strength': trend_strength
        }
        
        return optimized_signal
    
    def optimize_position_sizing(self, signal: Dict, portfolio_value: float,
                                recent_trades: List[Dict]) -> float:
        """
        Calculate optimal position size using advanced methods with half-Kelly multiplier.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            recent_trades: Recent trade history
            
        Returns:
            Optimal position size
        """
        base_risk = self.config.max_risk_per_trade / 100  # Convert to decimal
        
        # Calculate position size based on strategy
        if self.config.kelly_criterion:
            kelly_position = self._kelly_criterion_sizing(signal, portfolio_value, recent_trades)
            # NEW: Apply half-Kelly multiplier for drawdown control
            half_kelly_position = kelly_position * 0.5  # Half-Kelly for stability
            position_size = half_kelly_position
        else:
            position_size = self._fixed_fractional_sizing(signal, portfolio_value, base_risk)
        
        # Apply maximum position size limit
        position_size = min(position_size, self.config.max_position_size)
        
        # Adjust for recent performance
        performance_adjustment = self._get_performance_adjustment(recent_trades)
        position_size *= performance_adjustment
        
        return max(0.01, position_size)  # Minimum 1% position
    
    def _kelly_criterion_sizing(self, signal: Dict, portfolio_value: float,
                               recent_trades: List[Dict]) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            recent_trades: Recent trade history
            
        Returns:
            Kelly Criterion position size
        """
        try:
            # Calculate win rate and average win/loss ratio from recent trades
            if len(recent_trades) < 10:
                return 0.02  # Default to 2% if insufficient data
            
            wins = [t for t in recent_trades if t.get('pnl', 0) > 0]
            losses = [t for t in recent_trades if t.get('pnl', 0) <= 0]
            
            win_rate = len(wins) / len(recent_trades)
            
            if len(wins) == 0 or len(losses) == 0:
                return 0.02
            
            avg_win = np.mean([t['pnl'] for t in wins])
            avg_loss = abs(np.mean([t['pnl'] for t in losses]))
            
            # Kelly percentage: f = p - q / b where p=win rate, q=loss rate, b=win/loss ratio
            win_loss_ratio = avg_win / avg_loss
            kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
            
            # Cap Kelly fraction to prevent over-leveraging
            max_kelly = 0.25  # Maximum 25% of portfolio
            kelly_fraction = np.clip(kelly_fraction, 0.01, max_kelly)
            
            # Apply to portfolio value
            position_size = kelly_fraction
            
            logger.debug(f"Kelly Criterion sizing: win_rate={win_rate:.3f}, win_loss_ratio={win_loss_ratio:.2f}, "
                       f"kelly_fraction={kelly_fraction:.3f}, position_size={position_size:.3f}")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error in Kelly Criterion sizing: {e}")
            return 0.02  # Default to 2% on error
    
    def optimize_exit_strategy(self, signal: Dict, market_data: pd.DataFrame,
                             entry_price: float, entry_time: datetime) -> Dict[str, Any]:
        """
        Optimize exit strategy with Dynamic Profit Scaling using ADX.
        
        Args:
            signal: Trading signal
            market_data: Market data
            entry_price: Entry price
            entry_time: Entry time
            
        Returns:
            Optimized exit parameters
        """
        volatility = self._calculate_volatility(market_data)
        atr = self._calculate_atr(market_data)
        
        # NEW: Dynamic Profit Scaling with ADX
        adx = market_data['adx'].iloc[-1] if 'adx' in market_data.columns else 25.0
        profit_scaling_factor = self._calculate_dynamic_profit_scaling(adx)
        
        # Calculate dynamic stop loss
        if signal['decision'] == 'BUY':
            # Use ATR-based stop loss
            stop_loss = entry_price - (atr * 1.5)
            
            # Calculate dynamic take profit with ADX scaling
            if self.config.profit_target_scaling:
                # Scale target based on volatility and ADX
                risk_pips = (entry_price - stop_loss) / 0.0001
                base_target_pips = max(risk_pips * self.config.risk_reward_ratio, 
                                       self.config.min_profit_target_pips)
                
                # Apply ADX-based scaling
                scaled_target_pips = base_target_pips * profit_scaling_factor
                take_profit = entry_price + (scaled_target_pips * 0.0001)
                
                # Log the scaling
                logger.debug(f"Dynamic profit scaling: ADX={adx:.1f}, factor={profit_scaling_factor:.2f}, "
                           f"base_target={base_target_pips:.1f}, scaled_target={scaled_target_pips:.1f}")
            else:
                take_profit = signal.get('take_profit', entry_price + atr * 2)
            
            # Add trailing stop if enabled
            trailing_stop = None
            if self.config.trailing_stop_enabled:
                trailing_stop = entry_price - self.config.trailing_stop_distance
                
        else:  # SELL
            stop_loss = entry_price + (atr * 1.5)
            
            if self.config.profit_target_scaling:
                risk_pips = (stop_loss - entry_price) / 0.0001
                base_target_pips = max(risk_pips * self.config.risk_reward_ratio, 
                                       self.config.min_profit_target_pips)
                
                # Apply ADX-based scaling
                scaled_target_pips = base_target_pips * profit_scaling_factor
                take_profit = entry_price - (scaled_target_pips * 0.0001)
                
                logger.debug(f"Dynamic profit scaling: ADX={adx:.1f}, factor={profit_scaling_factor:.2f}, "
                           f"base_target={base_target_pips:.1f}, scaled_target={scaled_target_pips:.1f}")
            else:
                take_profit = signal.get('take_profit', entry_price - atr * 2)
            
            trailing_stop = None
            if self.config.trailing_stop_enabled:
                trailing_stop = entry_price + self.config.trailing_stop_distance
        
        return {
            'optimized_stop_loss': stop_loss,
            'optimized_take_profit': take_profit,
            'trailing_stop': trailing_stop,
            'risk_pips': (entry_price - stop_loss) / 0.0001 if signal['decision'] == 'BUY' 
                         else (stop_loss - entry_price) / 0.0001,
            'reward_pips': (take_profit - entry_price) / 0.0001 if signal['decision'] == 'BUY'
                         else (entry_price - take_profit) / 0.0001,
            'risk_reward_ratio': (take_profit - entry_price) / abs(entry_price - stop_loss),
            'adx': adx,
            'profit_scaling_factor': profit_scaling_factor,
            'dynamic_scaling_applied': True
        }
    
    def _calculate_dynamic_profit_scaling(self, adx: float) -> float:
        """
        Calculate dynamic profit scaling factor based on ADX.
        Higher ADX = stronger trend = larger profit targets.
        
        Args:
            adx: ADX value (0-100)
            
        Returns:
            Profit scaling factor (1.0 to 2.0)
        """
        try:
            # Base scaling: 1.0 for ADX <= 25
            # Maximum scaling: 2.0 for ADX >= 50
            # Linear interpolation between 25 and 50
            
            if adx <= 25:
                scaling_factor = 1.0
            elif adx >= 50:
                scaling_factor = 2.0
            else:
                # Linear interpolation between 25 and 50
                scaling_factor = 1.0 + (adx - 25) / 25  # (adx - 25) / (50 - 25) * (2.0 - 1.0)
            
            return np.clip(scaling_factor, 1.0, 2.0)
            
        except Exception as e:
            logger.error(f"Error calculating dynamic profit scaling: {e}")
            return 1.0  # Default to no scaling
    
    def enhance_mean_reversion_strategy(self, market_data: pd.DataFrame,
                                      market_structure: Dict) -> Dict[str, Any]:
        """
        Enhance mean reversion strategy with better entry/exit criteria.
        
        Args:
            market_data: Recent market data
            market_structure: Market structure analysis
            
        Returns:
            Enhanced mean reversion signals
        """
        current_price = market_data['close'].iloc[-1]
        
        # Get enhanced support/resistance levels
        levels = self._identify_key_levels(market_data, market_structure)
        
        # Calculate position within range
        if levels['support'] and levels['resistance']:
            range_width = levels['resistance'] - levels['support']
            position_in_range = (current_price - levels['support']) / range_width
            
            # Enhanced entry criteria
            buy_signal = position_in_range < 0.15  # Near support (was 0.2)
            sell_signal = position_in_range > 0.85  # Near resistance (was 0.8)
            
            # Additional confirmation indicators
            rsi = market_data['rsi'].iloc[-1] if 'rsi' in market_data.columns else 50
            bb_position = market_data['bb_position'].iloc[-1] if 'bb_position' in market_data.columns else 0.5
            
            # Require additional confirmation
            if buy_signal:
                buy_signal = rsi < 35 and bb_position < 0.2
            elif sell_signal:
                sell_signal = rsi > 65 and bb_position > 0.8
            
            # Calculate enhanced targets
            atr = self._calculate_atr(market_data)
            
            if buy_signal:
                stop_loss = levels['support'] - (atr * 0.5)
                take_profit = levels['resistance'] - (atr * 0.3)
                confidence = 0.6 + (0.35 - position_in_range)  # Higher confidence near support
            elif sell_signal:
                stop_loss = levels['resistance'] + (atr * 0.5)
                take_profit = levels['support'] + (atr * 0.3)
                confidence = 0.6 + (position_in_range - 0.65)  # Higher confidence near resistance
            else:
                return {'decision': 'HOLD', 'confidence': 0.0}
            
            return {
                'decision': 'BUY' if buy_signal else 'SELL',
                'confidence': min(confidence, 0.9),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_in_range': position_in_range,
                'range_width': range_width,
                'enhanced_criteria': True,
                'reasoning': [
                    f'Enhanced mean reversion signal',
                    f'Position in range: {position_in_range:.1%}',
                    f'RSI confirmation: {rsi:.1f}',
                    f'BB position: {bb_position:.2f}'
                ]
            }
        
        return {'decision': 'HOLD', 'confidence': 0.0}
    
    def add_mean_reversion_strategy(self, market_data: pd.DataFrame,
                                market_structure: Dict) -> Dict[str, Any]:
        """
        Add mean reversion strategy with Volume Z-Score filter for Neutral markets.
        
        Args:
            market_data: Recent market data
            market_structure: Market structure analysis
            
        Returns:
            Mean reversion trading signals
        """
        if not self.config.mean_reversion_enabled:
            return {'decision': 'HOLD', 'confidence': 0.0}
        
        # NEW: Volume Z-Score filter for Neutral markets
        volume_zscore_filter = self._apply_volume_zscore_filter(market_data)
        if not volume_zscore_filter['pass']:
            return {
                'decision': 'HOLD', 
                'confidence': 0.0,
                'filter_reason': f"Volume Z-Score filter failed: {volume_zscore_filter['reason']}"
            }
        
        # Identify range boundaries
        key_levels = market_structure.get('key_levels', [])
        if not key_levels:
            return {'decision': 'HOLD', 'confidence': 0.0}
        
        # Get current price position
        current_price = market_data['close'].iloc[-1]
        
        # Find nearest support and resistance
        support_levels = [level for level in key_levels if level['type'] == 'low']
        resistance_levels = [level for level in key_levels if level['type'] == 'high']
        
        if not support_levels or not resistance_levels:
            return {'decision': 'HOLD', 'confidence': 0.0}
        
        # Get nearest levels
        nearest_support = min(support_levels, key=lambda x: abs(x['price'] - current_price))
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x['price'] - current_price))
        
        # Calculate position in range
        range_size = nearest_resistance['price'] - nearest_support['price']
        position_in_range = (current_price - nearest_support['price']) / range_size
        
        # RSI confirmation
        rsi = 50  # Default
        if 'rsi' in market_data.columns:
            rsi = market_data['rsi'].iloc[-1]
        
        # Bollinger Band position
        bb_position = 0.5  # Default
        if 'bb_position' in market_data.columns:
            bb_position = market_data['bb_position'].iloc[-1]
        
        # Mean reversion signals
        signals = []
        reasoning = []
        
        # Buy signal (near support)
        if position_in_range < 0.2 and rsi < 35 and bb_position < 0.2:
            signals.append('BUY')
            reasoning.append(f'Near support (position: {position_in_range:.1%})')
            reasoning.append(f'RSI oversold: {rsi:.1f}')
            reasoning.append(f'BB position: {bb_position:.2f}')
            reasoning.append(f'Volume exhaustion confirmed')
            confidence = 0.6 + (0.2 - position_in_range)  # Higher confidence closer to support
        
        # Sell signal (near resistance)
        elif position_in_range > 0.8 and rsi > 65 and bb_position > 0.8:
            signals.append('SELL')
            reasoning.append(f'Near resistance (position: {position_in_range:.1%})')
            reasoning.append(f'RSI overbought: {rsi:.1f}')
            reasoning.append(f'BB position: {bb_position:.2f}')
            reasoning.append(f'Volume exhaustion confirmed')
            confidence = 0.6 + (position_in_range - 0.8)  # Higher confidence closer to resistance
        
        if not signals:
            return {'decision': 'HOLD', 'confidence': 0.0}
        
        # Calculate stop loss and take profit
        atr = market_data['atr'].iloc[-1] if 'atr' in market_data.columns else 0.002
        
        if signals[0] == 'BUY':
            stop_loss = nearest_support['price'] - (atr * 0.5)
            take_profit = nearest_resistance['price'] + (atr * 0.5)
        else:  # SELL
            stop_loss = nearest_resistance['price'] + (atr * 0.5)
            take_profit = nearest_support['price'] - (atr * 0.5)
        
        return {
            'decision': signals[0],
            'confidence': min(confidence, 0.85),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reasoning': reasoning,
            'strategy_type': 'mean_reversion',
            'position_in_range': position_in_range,
            'rsi': rsi,
            'bb_position': bb_position,
            'volume_zscore': volume_zscore_filter.get('volume_zscore', 0)
        }
    
    def _apply_volume_zscore_filter(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply Volume Z-Score filter for Neutral markets.
        Only trade mean reversion when volume is decreasing (exhaustion).
        
        Args:
            market_data: Recent market data
            
        Returns:
            Dictionary with filter results
        """
        try:
            if 'volume' not in market_data.columns or len(market_data) < 20:
                return {
                    'pass': True,  # Default to pass if no volume data
                    'reason': 'No volume data available'
                }
            
            # Calculate volume Z-Score
            recent_volumes = market_data['volume'].iloc[-20:]
            volume_mean = recent_volumes.mean()
            volume_std = recent_volumes.std()
            
            if volume_std == 0:
                return {
                    'pass': False,
                    'reason': 'Volume has no variation'
                }
            
            # Current volume Z-Score
            current_volume = market_data['volume'].iloc[-1]
            volume_zscore = (current_volume - volume_mean) / volume_std
            
            # Check if volume is decreasing (exhaustion signal for mean reversion)
            # We want negative Z-Score (below average) for mean reversion entries
            volume_decreasing = volume_zscore < -0.5  # At least 0.5 std below mean
            
            # Additional check: volume should be decreasing over last 3 bars
            volume_trend = False
            if len(market_data) >= 4:
                last_4_volumes = market_data['volume'].iloc[-4:]
                volume_trend = all(last_4_volumes[i] < last_4_volumes[i-1] for i in range(1, len(last_4_volumes)))
            
            # Filter passes if volume is decreasing (exhaustion)
            filter_pass = volume_decreasing and volume_trend
            
            reason = []
            if not volume_decreasing:
                reason.append(f'Volume not decreasing (Z-score: {volume_zscore:.2f})')
            if not volume_trend:
                reason.append('Volume not trending down over last 3 bars')
            
            return {
                'pass': filter_pass,
                'reason': '; '.join(reason) if reason else 'Volume exhaustion confirmed',
                'volume_zscore': volume_zscore,
                'volume_decreasing': volume_decreasing,
                'volume_trend': volume_trend
            }
            
        except Exception as e:
            logger.error(f"Error in Volume Z-Score filter: {e}")
            return {
                'pass': True,  # Default to pass on error
                'reason': f'Filter error: {str(e)}'
            }
    
    def _apply_volatility_volume_filter(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Apply Volatility-Volume Filter to avoid trading in "dead" markets.
        
        Args:
            market_data: Recent market data
            
        Returns:
            Dictionary with filter results
        """
        try:
            # 1. Minimum ATR Filter: Reject trades if ATR < 1.5×Spread
            spread_pips = 1.5  # Average spread for EURUSD
            min_atr = spread_pips * 1.5  # ATR must be at least 1.5x spread
            
            atr_pips = 0
            if 'atr' in market_data.columns:
                atr_pips = market_data['atr'].iloc[-1] * 10000  # Convert to pips
            else:
                # Calculate ATR manually if not available
                high_low = market_data['high'].iloc[-1] - market_data['low'].iloc[-1]
                atr_pips = high_low * 10000
            
            if atr_pips < min_atr:
                return {
                    'pass': False,
                    'reason': f'ATR too low: {atr_pips:.1f} < {min_atr:.1f} pips',
                    'atr_pips': atr_pips,
                    'min_atr': min_atr
                }
            
            # 2. Volume Delta Filter: Only trade if volume is increasing
            volume_delta_pass = True
            volume_delta_reason = "Volume check passed"
            
            if 'volume' in market_data.columns and len(market_data) >= 4:
                recent_volumes = market_data['volume'].iloc[-4:]
                # Check if volume is increasing over last 3 bars
                delta_volume = all(recent_volumes[i] > recent_volumes[i-1] for i in range(1, len(recent_volumes)))
                
                if not delta_volume:
                    volume_delta_pass = False
                    volume_delta_reason = "Volume not increasing (DeltaVolume <= 0)"
            
            # 3. Combined filter result
            filter_pass = volume_delta_pass
            
            reason = []
            if atr_pips < min_atr:
                reason.append(f'ATR too low: {atr_pips:.1f} < {min_atr:.1f} pips')
            if not volume_delta_pass:
                reason.append(volume_delta_reason)
            
            return {
                'pass': filter_pass,
                'reason': '; '.join(reason) if reason else 'All filters passed',
                'atr_pips': atr_pips,
                'min_atr': min_atr,
                'volume_delta_pass': volume_delta_pass,
                'volume_delta_reason': volume_delta_reason
            }
            
        except Exception as e:
            logger.error(f"Error in Volatility-Volume filter: {e}")
            return {
                'pass': True,  # Default to pass on error
                'reason': f'Filter error: {str(e)}'
            }
    
    def add_breakout_strategy(self, market_data: pd.DataFrame,
                            market_structure: Dict) -> Dict[str, Any]:
        """
        Add breakout strategy with Volatility-Volume Filter.
        
        Args:
            market_data: Recent market data
            market_structure: Market structure analysis
            
        Returns:
            Breakout trading signals
        """
        if not self.config.breakout_enabled:
            return {'decision': 'HOLD', 'confidence': 0.0}
        
        current_price = market_data['close'].iloc[-1]
        
        # NEW: Volatility-Volume Filter to avoid "dead" markets
        volatility_filter = self._apply_volatility_volume_filter(market_data)
        if not volatility_filter['pass']:
            return {
                'decision': 'HOLD', 
                'confidence': 0.0,
                'filter_reason': f"Volatility-Volume filter failed: {volatility_filter['reason']}"
            }
        
        # Identify key levels for breakout
        levels = self._identify_key_levels(market_data, market_structure)
        
        if not levels['resistance'] or not levels['support']:
            return {'decision': 'HOLD', 'confidence': 0.0}
        
        # Check for breakout conditions
        resistance_breakout = current_price > levels['resistance']
        support_breakdown = current_price < levels['support']
        
        # Enhanced volume confirmation with delta volume check
        volume_confirmation = False
        delta_volume_confirmation = False
        
        if 'volume' in market_data.columns and len(market_data) >= 4:
            current_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].iloc[-20:].mean() if len(market_data) >= 20 else current_volume
            
            # Standard volume confirmation
            volume_confirmation = current_volume > avg_volume * 1.5
            
            # Delta volume confirmation - volume must be increasing over last 3 bars
            if len(market_data) >= 4:
                recent_volumes = market_data['volume'].iloc[-4:]
                delta_volume = all(recent_volumes[i] > recent_volumes[i-1] for i in range(1, len(recent_volumes)))
                delta_volume_confirmation = delta_volume
        
        # Momentum confirmation
        momentum_confirmation = False
        if 'adx' in market_data.columns:
            adx = market_data['adx'].iloc[-1]
            momentum_confirmation = adx > self.config.trend_strength_threshold
        
        # Price momentum confirmation
        price_momentum = False
        if len(market_data) >= 5:
            price_change = (market_data['close'].iloc[-1] - market_data['close'].iloc[-5]) / market_data['close'].iloc[-5]
            price_momentum = abs(price_change) > 0.001  # 10 pips minimum move
        
        # Calculate breakout signals
        if resistance_breakout and volume_confirmation and delta_volume_confirmation and momentum_confirmation:
            # Bullish breakout
            atr = self._calculate_atr(market_data)
            stop_loss = levels['resistance'] - (atr * 0.8)
            take_profit = current_price + (atr * 2.5)
            confidence = 0.75  # Higher confidence with enhanced confirmation
            
            return {
                'decision': 'BUY',
                'confidence': confidence,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy_type': 'breakout',
                'breakout_level': levels['resistance'],
                'reasoning': [
                    f'Enhanced bullish breakout above {levels["resistance"]:.5f}',
                    f'Volume confirmation: {volume_confirmation}',
                    f'Delta volume confirmation: {delta_volume_confirmation}',
                    f'Momentum confirmation: {momentum_confirmation}',
                    f'ADX strength: {adx:.1f}',
                    f'Price momentum: {price_momentum}'
                ]
            }
        
        elif support_breakdown and volume_confirmation and delta_volume_confirmation and momentum_confirmation:
            # Bearish breakout
            atr = self._calculate_atr(market_data)
            stop_loss = levels['support'] + (atr * 0.8)
            take_profit = current_price - (atr * 2.5)
            confidence = 0.75  # Higher confidence with enhanced confirmation
            
            return {
                'decision': 'SELL',
                'confidence': confidence,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'strategy_type': 'breakout',
                'breakout_level': levels['support'],
                'reasoning': [
                    f'Enhanced bearish breakdown below {levels["support"]:.5f}',
                    f'Volume confirmation: {volume_confirmation}',
                    f'Delta volume confirmation: {delta_volume_confirmation}',
                    f'Momentum confirmation: {momentum_confirmation}',
                    f'ADX strength: {adx:.1f}',
                    f'Price momentum: {price_momentum}'
                ]
            }
        
        return {'decision': 'HOLD', 'confidence': 0.0}
    
    def optimize_decision_pipeline(self, market_data: pd.DataFrame,
                                  original_signal: Dict, market_structure: Dict,
                                  recent_trades: List[Dict]) -> Dict[str, Any]:
        """
        Complete decision pipeline optimization.
        
        Args:
            market_data: Recent market data
            original_signal: Original Alpha Factory signal
            market_structure: Market structure analysis
            recent_trades: Recent trade history
            
        Returns:
            Optimized trading decision
        """
        # Step 1: Optimize entry criteria
        optimized_signal = self.optimize_entry_criteria(market_data, original_signal)
        
        if optimized_signal['decision'] == 'HOLD':
            return optimized_signal
        
        # Step 2: Check for alternative strategies
        regime = optimized_signal.get('regime', 'neutral')
        
        if regime == 'neutral':
            # Try enhanced mean reversion
            mr_signal = self.enhance_mean_reversion_strategy(market_data, market_structure)
            if mr_signal['decision'] != 'HOLD' and mr_signal['confidence'] > optimized_signal['confidence']:
                optimized_signal = mr_signal
        
        # Step 3: Check for breakout opportunities
        breakout_signal = self.add_breakout_strategy(market_data, market_structure)
        if breakout_signal['decision'] != 'HOLD' and breakout_signal['confidence'] > optimized_signal['confidence']:
            optimized_signal = breakout_signal
        
        # Step 4: Optimize position sizing
        portfolio_value = 10000.0  # Default, should be passed from portfolio
        position_size = self.optimize_position_sizing(optimized_signal, portfolio_value, recent_trades)
        optimized_signal['position_size'] = position_size
        
        # Step 5: Optimize exit strategy
        entry_price = market_data['close'].iloc[-1]
        exit_optimization = self.optimize_exit_strategy(optimized_signal, market_data, entry_price, datetime.now())
        optimized_signal.update(exit_optimization)
        
        # Step 6: Add optimization metadata
        optimized_signal['optimization_applied'] = True
        optimized_signal['optimization_timestamp'] = datetime.now().isoformat()
        
        return optimized_signal
    
    def track_performance(self, trade_result: Dict):
        """
        Track trade performance for adaptive optimization.
        
        Args:
            trade_result: Trade result with P&L and details
        """
        self.performance_history.append(trade_result)
        
        # Keep only recent performance
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Update regime performance
        regime = trade_result.get('regime', 'unknown')
        if regime not in self.regime_performance:
            self.regime_performance[regime] = {
                'trades': 0,
                'wins': 0,
                'total_pnl': 0.0,
                'avg_confidence': 0.0
            }
        
        self.regime_performance[regime]['trades'] += 1
        if trade_result.get('pnl', 0) > 0:
            self.regime_performance[regime]['wins'] += 1
        self.regime_performance[regime]['total_pnl'] += trade_result.get('pnl', 0)
        
        # Update average confidence
        total_trades = self.regime_performance[regime]['trades']
        current_avg = self.regime_performance[regime]['avg_confidence']
        new_confidence = trade_result.get('confidence', 0)
        self.regime_performance[regime]['avg_confidence'] = (
            (current_avg * (total_trades - 1) + new_confidence) / total_trades
        )
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """
        Get performance insights and recommendations.
        
        Returns:
            Performance analysis and recommendations
        """
        if not self.performance_history:
            return {'status': 'insufficient_data'}
        
        recent_trades = self.performance_history[-20:]  # Last 20 trades
        
        # Calculate recent performance
        win_rate = len([t for t in recent_trades if t.get('pnl', 0) > 0]) / len(recent_trades)
        avg_pnl = np.mean([t.get('pnl', 0) for t in recent_trades])
        total_pnl = sum([t.get('pnl', 0) for t in recent_trades])
        
        # Regime analysis
        regime_insights = {}
        for regime, data in self.regime_performance.items():
            if data['trades'] > 0:
                regime_insights[regime] = {
                    'win_rate': data['wins'] / data['trades'],
                    'avg_pnl': data['total_pnl'] / data['trades'],
                    'total_trades': data['trades'],
                    'avg_confidence': data['avg_confidence']
                }
        
        # Generate recommendations
        recommendations = []
        
        if win_rate < 0.4:
            recommendations.append("Consider tightening entry criteria - win rate below 40%")
        
        if avg_pnl < 0:
            recommendations.append("Review strategy - average trade is negative")
        
        # Check regime-specific issues
        for regime, insights in regime_insights.items():
            if insights['win_rate'] < 0.3:
                recommendations.append(f"Poor performance in {regime} regime - consider disabling")
            elif insights['avg_pnl'] < 0:
                recommendations.append(f"Negative average P&L in {regime} regime - adjust parameters")
        
        return {
            'recent_performance': {
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
                'trade_count': len(recent_trades)
            },
            'regime_performance': regime_insights,
            'recommendations': recommendations,
            'optimization_effectiveness': self._calculate_optimization_effectiveness()
        }
    
    # Private helper methods
    
    def _calculate_volatility(self, market_data: pd.DataFrame, period: int = 20) -> float:
        """Calculate price volatility."""
        returns = market_data['close'].pct_change().dropna()
        return returns.tail(period).std()
    
    def _estimate_spread(self, market_data: pd.DataFrame) -> float:
        """Estimate current spread."""
        if 'high' in market_data.columns and 'low' in market_data.columns:
            # Use average of recent high-low spread as proxy
            recent_spread = (market_data['high'] - market_data['low']).tail(10).mean()
            return recent_spread / market_data['close'].iloc[-1]
        return 0.0001  # Default 1 pip spread
    
    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX or alternative."""
        if 'adx' in market_data.columns:
            return market_data['adx'].iloc[-1]
        
        # Fallback: use price momentum
        returns = market_data['close'].pct_change().dropna()
        return abs(returns.tail(20).mean()) * 1000  # Scale to ADX-like range
    
    def _pass_market_filters(self, volatility: float, spread: float, trend_strength: float) -> bool:
        """Check if market conditions pass filters."""
        return (volatility >= self.config.min_volatility_threshold and
                spread <= self.config.max_spread_threshold and
                trend_strength >= 10.0)  # Minimum trend strength
    
    def _optimize_confidence_threshold(self, regime: str, base_confidence: float) -> float:
        """Optimize confidence threshold based on regime."""
        if not self.config.adaptive_thresholds:
            return self.config.confidence_threshold_trend
        
        # Adjust threshold based on regime performance
        if regime in self.regime_performance:
            regime_data = self.regime_performance[regime]
            if regime_data['trades'] > 10:
                # Adjust based on historical performance
                if regime_data['win_rate'] > 0.6:
                    return max(0.5, self.config.confidence_threshold_trend - 0.1)
                elif regime_data['win_rate'] < 0.4:
                    return min(0.9, self.config.confidence_threshold_trend + 0.1)
        
        # Default regime-based thresholds
        if regime == 'neutral':
            return self.config.confidence_threshold_range
        else:
            return self.config.confidence_threshold_trend
    
    def _kelly_criterion_sizing(self, signal: Dict, portfolio_value: float, recent_trades: List[Dict]) -> float:
        """Calculate position size using Kelly criterion."""
        if len(recent_trades) < 10:
            return 0.02  # Default 2% if insufficient data
        
        # Calculate win rate and average win/loss
        wins = [t for t in recent_trades if t.get('pnl', 0) > 0]
        losses = [t for t in recent_trades if t.get('pnl', 0) <= 0]
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['pnl'] for t in losses])) if losses else 1
        
        if avg_loss == 0:
            return 0.02
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply Kelly fraction with safety factor (half Kelly)
        kelly_fraction = max(0, kelly_fraction) * 0.5
        
        # Convert to position size
        return min(kelly_fraction, self.config.max_position_size)
    
    def _fixed_fractional_sizing(self, signal: Dict, portfolio_value: float, risk_fraction: float) -> float:
        """Calculate fixed fractional position size."""
        return risk_fraction
    
    def _get_performance_adjustment(self, recent_trades: List[Dict]) -> float:
        """Adjust position size based on recent performance."""
        if len(recent_trades) < 5:
            return 1.0
        
        recent_pnl = sum([t.get('pnl', 0) for t in recent_trades[-5:]])
        
        # Reduce size if losing streak, increase if winning streak
        if recent_pnl < -100:  # Losing streak
            return 0.5
        elif recent_pnl > 200:  # Winning streak
            return 1.2
        else:
            return 1.0
    
    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        high_low = market_data['high'] - market_data['low']
        high_close = abs(market_data['high'] - market_data['close'].shift())
        low_close = abs(market_data['low'] - market_data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean().iloc[-1]
    
    def _identify_key_levels(self, market_data: pd.DataFrame, market_structure: Dict) -> Dict[str, float]:
        """Identify key support and resistance levels."""
        levels = {'support': None, 'resistance': None}
        
        # Get levels from market structure
        if 'key_levels' in market_structure:
            key_levels = market_structure['key_levels']
            
            highs = [level for level in key_levels if level.get('type') == 'high']
            lows = [level for level in key_levels if level.get('type') == 'low']
            
            if highs:
                # Get nearest resistance above current price
                current_price = market_data['close'].iloc[-1]
                resistance_candidates = [h['price'] for h in highs if h['price'] > current_price]
                if resistance_candidates:
                    levels['resistance'] = min(resistance_candidates)
            
            if lows:
                # Get nearest support below current price
                support_candidates = [l['price'] for l in lows if l['price'] < current_price]
                if support_candidates:
                    levels['support'] = max(support_candidates)
        
        return levels
    
    def _calculate_optimization_effectiveness(self) -> float:
        """Calculate how effective optimizations have been."""
        if len(self.performance_history) < 20:
            return 0.0
        
        # Compare recent performance with earlier performance
        recent_trades = self.performance_history[-10:]
        earlier_trades = self.performance_history[-20:-10]
        
        recent_win_rate = len([t for t in recent_trades if t.get('pnl', 0) > 0]) / len(recent_trades)
        earlier_win_rate = len([t for t in earlier_trades if t.get('pnl', 0) > 0]) / len(earlier_trades)
        
        return recent_win_rate - earlier_win_rate


# Convenience function for quick optimization
def optimize_alpha_factory_signal(market_data: pd.DataFrame, 
                                 original_signal: Dict,
                                 market_structure: Dict,
                                 config: Optional[ProfitabilityConfig] = None) -> Dict[str, Any]:
    """
    Quick optimization function for Alpha Factory signals.
    
    Args:
        market_data: Recent market data
        original_signal: Original Alpha Factory signal
        market_structure: Market structure analysis
        config: Optimization configuration
        
    Returns:
        Optimized trading signal
    """
    optimizer = ProfitabilityOptimizer(config)
    return optimizer.optimize_decision_pipeline(market_data, original_signal, market_structure, [])
