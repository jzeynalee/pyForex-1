"""
Exit Logic Optimizer for Alpha Factory

This module implements professional exit optimization techniques:
1. Dynamic exits based on market structure
2. Partial take-profit at structure levels
3. Volatility contraction trailing stops
4. Probability collapse exit triggers
5. Regime-specific exit strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ExitType(Enum):
    """Exit strategy types."""
    FIXED = "fixed"
    STRUCTURE = "structure"
    TRAILING = "trailing"
    PARTIAL = "partial"
    PROBABILITY = "probability"
    VOLATILITY = "volatility"

class ExitReason(Enum):
    """Exit reason classifications."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TRAILING_STOP = "trailing_stop"
    STRUCTURE_LEVEL = "structure_level"
    PROBABILITY_COLLAPSE = "probability_collapse"
    VOLATILITY_EXPANSION = "volatility_expansion"
    PARTIAL_EXIT = "partial_exit"

@dataclass
class ExitConfig:
    """Configuration for exit optimization."""
    # Structure-based exits
    structure_exits_enabled: bool = True
    structure_lookback: int = 20
    min_structure_strength: float = 0.5
    
    # Partial take-profit
    partial_exits_enabled: bool = True
    partial_exit_levels: List[float] = None  # [0.5, 1.0] for 50% and 100%
    partial_exit_ratios: List[float] = None  # [0.3, 0.7] for 30% and 70%
    
    # Trailing stops
    trailing_stops_enabled: bool = True
    trailing_distance: float = 0.001  # 10 pips
    trailing_activation: float = 0.002  # 20 pips in profit
    volatility_trailing: bool = True
    
    # Probability-based exits
    probability_exits_enabled: bool = True
    probability_threshold: float = 0.3  # Exit if confidence drops below 30%
    probability_decay_rate: float = 0.1
    
    # Volatility exits
    volatility_exits_enabled: bool = True
    volatility_expansion_threshold: float = 2.0  # 2x ATR expansion
    volatility_contraction_threshold: float = 0.5  # 0.5x ATR contraction
    
    # Regime-specific exits
    regime_exits_enabled: bool = True
    regime_exit_multipliers: Dict[str, float] = None

class ExitOptimizer:
    """Professional exit optimization for Alpha Factory."""
    
    def __init__(self, config: ExitConfig = None):
        self.config = config or ExitConfig()
        
        # Initialize default partial exit levels
        if self.config.partial_exit_levels is None:
            self.config.partial_exit_levels = [0.5, 1.0]  # 50% and full
        
        if self.config.partial_exit_ratios is None:
            self.config.partial_exit_ratios = [0.3, 0.7]  # 30% at 50% TP, 70% at full
        
        # Initialize regime exit multipliers
        if self.config.regime_exit_multipliers is None:
            self.config.regime_exit_multipliers = {
                'bullish': 1.2,   # 20% wider exits for trends
                'bearish': 1.2,   # 20% wider exits for trends
                'neutral': 0.8,   # 20% tighter exits for ranges
                'volatile': 1.5    # 50% wider exits for volatility
            }
        
        # Exit tracking
        self.exit_history = []
        self.performance_by_exit_type = {}
    
    def identify_structure_levels(self, price_data: pd.DataFrame, current_price: float,
                                direction: str) -> Dict[str, float]:
        """
        Identify key structure levels for exit points.
        
        Args:
            price_data: DataFrame with OHLC data
            current_price: Current price
            direction: 'long' or 'short'
            
        Returns:
            Dictionary with structure levels
        """
        if not self.config.structure_exits_enabled:
            return {}
        
        try:
            lookback = min(self.config.structure_lookback, len(price_data))
            recent_data = price_data.tail(lookback)
            
            structure_levels = {}
            
            if direction == 'long':
                # Find resistance levels above current price
                resistance_levels = []
                for i in range(len(recent_data)):
                    high = recent_data.iloc[i]['high']
                    if high > current_price:
                        # Check if this is a swing high
                        if i > 0 and i < len(recent_data) - 1:
                            left_high = recent_data.iloc[i-1]['high']
                            right_high = recent_data.iloc[i+1]['high']
                            if high > left_high and high > right_high:
                                resistance_levels.append(high)
                
                if resistance_levels:
                    structure_levels['nearest_resistance'] = min(resistance_levels)
                    structure_levels['next_resistance'] = sorted(resistance_levels)[1] if len(resistance_levels) > 1 else None
            
            else:  # short
                # Find support levels below current price
                support_levels = []
                for i in range(len(recent_data)):
                    low = recent_data.iloc[i]['low']
                    if low < current_price:
                        # Check if this is a swing low
                        if i > 0 and i < len(recent_data) - 1:
                            left_low = recent_data.iloc[i-1]['low']
                            right_low = recent_data.iloc[i+1]['low']
                            if low < left_low and low < right_low:
                                support_levels.append(low)
                
                if support_levels:
                    structure_levels['nearest_support'] = max(support_levels)
                    structure_levels['next_support'] = sorted(support_levels, reverse=True)[1] if len(support_levels) > 1 else None
            
            return structure_levels
            
        except Exception as e:
            logger.error(f"Error identifying structure levels: {e}")
            return {}
    
    def calculate_trailing_stop(self, entry_price: float, current_price: float,
                              direction: str, highest_profit: float, 
                              current_atr: float = None) -> float:
        """
        Calculate trailing stop price.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            direction: 'long' or 'short'
            highest_profit: Highest profit achieved so far
            current_atr: Current ATR for volatility adjustment
            
        Returns:
            Trailing stop price
        """
        if not self.config.trailing_stops_enabled:
            return None
        
        try:
            # Check if trailing stop is activated
            profit = abs(current_price - entry_price)
            if profit < self.config.trailing_activation:
                return None
            
            # Calculate base trailing distance
            trailing_distance = self.config.trailing_distance
            
            # Adjust for volatility if enabled
            if self.config.volatility_trailing and current_atr is not None:
                volatility_multiplier = min(2.0, current_atr / 0.0001)  # Normalize to 10 pips ATR
                trailing_distance *= volatility_multiplier
            
            # Calculate trailing stop
            if direction == 'long':
                trailing_stop = current_price - trailing_distance
                # Ensure trailing stop only moves up
                if hasattr(self, '_last_trailing_stop'):
                    trailing_stop = max(trailing_stop, self._last_trailing_stop)
                self._last_trailing_stop = trailing_stop
            else:  # short
                trailing_stop = current_price + trailing_distance
                # Ensure trailing stop only moves down
                if hasattr(self, '_last_trailing_stop'):
                    trailing_stop = min(trailing_stop, self._last_trailing_stop)
                self._last_trailing_stop = trailing_stop
            
            return trailing_stop
            
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {e}")
            return None
    
    def check_probability_collapse(self, current_probability: float, 
                                 initial_probability: float, 
                                 trade_age_bars: int) -> bool:
        """
        Check if probability has collapsed enough to exit.
        
        Args:
            current_probability: Current trade probability
            initial_probability: Initial trade probability
            trade_age_bars: Age of trade in bars
            
        Returns:
            True if probability has collapsed
        """
        if not self.config.probability_exits_enabled:
            return False
        
        try:
            # Check absolute threshold
            if current_probability < self.config.probability_threshold:
                return True
            
            # Check relative decay
            probability_decay = (initial_probability - current_probability) / initial_probability
            decay_threshold = self.config.probability_decay_rate * (trade_age_bars / 100)  # Scale by age
            
            if probability_decay > decay_threshold:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking probability collapse: {e}")
            return False
    
    def check_volatility_exit(self, current_atr: float, initial_atr: float,
                            direction: str) -> Tuple[bool, str]:
        """
        Check if volatility conditions trigger an exit.
        
        Args:
            current_atr: Current ATR
            initial_atr: ATR at trade entry
            direction: 'long' or 'short'
            
        Returns:
            Tuple of (should_exit, exit_reason)
        """
        if not self.config.volatility_exits_enabled:
            return False, ""
        
        try:
            volatility_ratio = current_atr / initial_atr
            
            # Volatility expansion (exit in volatile conditions)
            if volatility_ratio > self.config.volatility_expansion_threshold:
                return True, ExitReason.VOLATILITY_EXPANSION.value
            
            # Volatility contraction (potential trend exhaustion)
            if volatility_ratio < self.config.volatility_contraction_threshold:
                return True, ExitReason.VOLATILITY_EXPANSION.value  # Use same reason for now
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Error checking volatility exit: {e}")
            return False, ""
    
    def calculate_partial_exits(self, entry_price: float, current_price: float,
                               direction: str, structure_levels: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Calculate partial exit levels.
        
        Args:
            entry_price: Entry price
            current_price: Current price
            direction: 'long' or 'short'
            structure_levels: Structure levels from identify_structure_levels
            
        Returns:
            List of partial exit configurations
        """
        if not self.config.partial_exits_enabled:
            return []
        
        try:
            partial_exits = []
            
            for i, (level_ratio, exit_ratio) in enumerate(zip(self.config.partial_exit_levels, self.config.partial_exit_ratios)):
                if direction == 'long':
                    # Calculate take profit price
                    profit_target = entry_price * (1 + level_ratio)
                    
                    # Check if structure level provides better target
                    if 'nearest_resistance' in structure_levels:
                        structure_target = structure_levels['nearest_resistance']
                        if structure_target < profit_target:
                            profit_target = structure_target
                else:  # short
                    profit_target = entry_price * (1 - level_ratio)
                    
                    # Check if structure level provides better target
                    if 'nearest_support' in structure_levels:
                        structure_target = structure_levels['nearest_support']
                        if structure_target > profit_target:
                            profit_target = structure_target
                
                partial_exits.append({
                    'level': i + 1,
                    'price': profit_target,
                    'ratio': exit_ratio,
                    'type': ExitType.PARTIAL.value
                })
            
            return partial_exits
            
        except Exception as e:
            logger.error(f"Error calculating partial exits: {e}")
            return []
    
    def optimize_exit_strategy(self, trade_data: Dict[str, Any], 
                             market_data: pd.DataFrame,
                             regime: str = 'neutral') -> Dict[str, Any]:
        """
        Optimize complete exit strategy for a trade.
        
        Args:
            trade_data: Dictionary with trade information
            market_data: DataFrame with market data
            regime: Current market regime
            
        Returns:
            Dictionary with optimized exit strategy
        """
        logger.info("Optimizing exit strategy")
        
        # Extract trade parameters
        entry_price = trade_data['entry_price']
        current_price = trade_data['current_price']
        direction = trade_data['direction']
        initial_probability = trade_data.get('probability', 0.7)
        current_probability = trade_data.get('current_probability', initial_probability)
        trade_age = trade_data.get('trade_age_bars', 0)
        initial_atr = trade_data.get('initial_atr', 0.0001)
        
        # Get current market data
        current_bar = market_data.iloc[-1]
        current_atr = current_bar.get('atr', initial_atr)
        
        # Initialize exit strategy
        exit_strategy = {
            'primary_exit': None,
            'partial_exits': [],
            'trailing_stop': None,
            'exit_conditions': [],
            'regime_adjustments': {}
        }
        
        # 1. Structure-based exits
        structure_levels = self.identify_structure_levels(market_data, current_price, direction)
        if structure_levels:
            exit_strategy['structure_levels'] = structure_levels
            
            # Set primary exit based on structure
            if direction == 'long' and 'nearest_resistance' in structure_levels:
                exit_strategy['primary_exit'] = {
                    'price': structure_levels['nearest_resistance'],
                    'type': ExitType.STRUCTURE.value,
                    'reason': ExitReason.STRUCTURE_LEVEL.value
                }
            elif direction == 'short' and 'nearest_support' in structure_levels:
                exit_strategy['primary_exit'] = {
                    'price': structure_levels['nearest_support'],
                    'type': ExitType.STRUCTURE.value,
                    'reason': ExitReason.STRUCTURE_LEVEL.value
                }
        
        # 2. Partial exits
        partial_exits = self.calculate_partial_exits(entry_price, current_price, direction, structure_levels)
        exit_strategy['partial_exits'] = partial_exits
        
        # 3. Trailing stop
        highest_profit = trade_data.get('highest_profit', abs(current_price - entry_price))
        trailing_stop = self.calculate_trailing_stop(entry_price, current_price, direction, highest_profit, current_atr)
        if trailing_stop:
            exit_strategy['trailing_stop'] = {
                'price': trailing_stop,
                'type': ExitType.TRAILING.value,
                'reason': ExitReason.TRAILING_STOP.value
            }
        
        # 4. Probability collapse check
        if self.check_probability_collapse(current_probability, initial_probability, trade_age):
            exit_strategy['exit_conditions'].append({
                'condition': 'probability_collapse',
                'threshold': self.config.probability_threshold,
                'current_value': current_probability,
                'action': 'exit_immediately'
            })
        
        # 5. Volatility exit check
        should_exit_vol, vol_reason = self.check_volatility_exit(current_atr, initial_atr, direction)
        if should_exit_vol:
            exit_strategy['exit_conditions'].append({
                'condition': 'volatility_exit',
                'reason': vol_reason,
                'current_atr': current_atr,
                'initial_atr': initial_atr,
                'action': 'exit_immediately'
            })
        
        # 6. Regime-specific adjustments
        if self.config.regime_exits_enabled:
            regime_multiplier = self.config.regime_exit_multipliers.get(regime, 1.0)
            exit_strategy['regime_adjustments'] = {
                'regime': regime,
                'multiplier': regime_multiplier,
                'adjusted_trailing_distance': self.config.trailing_distance * regime_multiplier
            }
        
        # Store exit strategy
        self.exit_history.append({
            'timestamp': datetime.now(),
            'trade_data': trade_data,
            'exit_strategy': exit_strategy,
            'regime': regime
        })
        
        logger.info("Exit strategy optimization completed")
        
        return exit_strategy
    
    def evaluate_exit_performance(self, exit_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate performance of different exit strategies.
        
        Args:
            exit_results: List of exit results
            
        Returns:
            Performance analysis by exit type
        """
        if not exit_results:
            return {'status': 'no_exit_data'}
        
        performance_by_type = {}
        
        for result in exit_results:
            exit_type = result.get('exit_type', 'unknown')
            pnl = result.get('pnl', 0)
            
            if exit_type not in performance_by_type:
                performance_by_type[exit_type] = {
                    'trades': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0,
                    'win_rate': 0,
                    'wins': 0,
                    'losses': 0
                }
            
            perf = performance_by_type[exit_type]
            perf['trades'] += 1
            perf['total_pnl'] += pnl
            perf['wins'] += 1 if pnl > 0 else 0
            perf['losses'] += 1 if pnl <= 0 else 0
        
        # Calculate derived metrics
        for exit_type, perf in performance_by_type.items():
            if perf['trades'] > 0:
                perf['avg_pnl'] = perf['total_pnl'] / perf['trades']
                perf['win_rate'] = perf['wins'] / perf['trades']
        
        return performance_by_type
    
    def get_exit_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of exit optimization performance."""
        if not self.exit_history:
            return {'status': 'no_exit_history'}
        
        summary = {
            'total_exits_optimized': len(self.exit_history),
            'exit_types_used': set(),
            'average_partial_exits': 0,
            'trailing_stop_usage': 0,
            'structure_exit_usage': 0,
            'config': {
                'structure_exits': self.config.structure_exits_enabled,
                'partial_exits': self.config.partial_exits_enabled,
                'trailing_stops': self.config.trailing_stops_enabled,
                'probability_exits': self.config.probability_exits_enabled,
                'volatility_exits': self.config.volatility_exits_enabled
            }
        }
        
        # Analyze exit strategy usage
        partial_exits_count = 0
        trailing_stop_count = 0
        structure_exit_count = 0
        
        for exit_record in self.exit_history:
            strategy = exit_record['exit_strategy']
            
            # Count partial exits
            partial_exits_count += len(strategy.get('partial_exits', []))
            summary['exit_types_used'].update([exit.get('type') for exit in strategy.get('partial_exits', [])])
            
            # Count trailing stops
            if strategy.get('trailing_stop'):
                trailing_stop_count += 1
                summary['exit_types_used'].add(strategy['trailing_stop']['type'])
            
            # Count structure exits
            if strategy.get('primary_exit') and strategy['primary_exit']['type'] == ExitType.STRUCTURE.value:
                structure_exit_count += 1
                summary['exit_types_used'].add(ExitType.STRUCTURE.value)
        
        summary['average_partial_exits'] = partial_exits_count / len(self.exit_history)
        summary['trailing_stop_usage'] = trailing_stop_count / len(self.exit_history)
        summary['structure_exit_usage'] = structure_exit_count / len(self.exit_history)
        summary['exit_types_used'] = list(summary['exit_types_used'])
        
        return summary
