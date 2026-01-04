"""
Regime-Conditional Execution for Alpha Factory

This module implements professional regime-specific execution strategies:
1. Different position sizing by regime
2. Regime-specific trade filtering
3. Dynamic risk management per regime
4. Regime-based entry/exit rules
5. Performance tracking by regime
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"

@dataclass
class RegimeExecutionConfig:
    """Configuration for regime-conditional execution."""
    # Position sizing multipliers
    regime_multipliers: Dict[str, float] = None
    
    # Trade filtering rules
    regime_filters: Dict[str, bool] = None
    
    # Risk management per regime
    regime_stop_multipliers: Dict[str, float] = None
    regime_target_multipliers: Dict[str, float] = None
    
    # Minimum confidence per regime
    regime_confidence_thresholds: Dict[str, float] = None
    
    # Performance tracking
    track_regime_performance: bool = True
    performance_window: int = 100  # Trades for performance tracking

class RegimeExecutor:
    """Professional regime-conditional execution for Alpha Factory."""
    
    def __init__(self, config: RegimeExecutionConfig = None):
        self.config = config or RegimeExecutionConfig()
        
        # Initialize default regime multipliers if not provided
        if self.config.regime_multipliers is None:
            self.config.regime_multipliers = {
                'bullish': 1.0,      # Full size in strong trends
                'bearish': 1.0,      # Full size in strong trends
                'neutral': 0.0,      # No trades in neutral/range
                'volatile': 0.5      # Half size in volatile
            }
        
        # Initialize default regime filters if not provided
        if self.config.regime_filters is None:
            self.config.regime_filters = {
                'bullish': True,      # Allow trades in bullish
                'bearish': True,      # Allow trades in bearish
                'neutral': False,     # Skip trades in neutral
                'volatile': True      # Allow trades in volatile (reduced size)
            }
        
        # Initialize default risk multipliers if not provided
        if self.config.regime_stop_multipliers is None:
            self.config.regime_stop_multipliers = {
                'bullish': 1.0,      # Standard stops for trends
                'bearish': 1.0,      # Standard stops for trends
                'neutral': 1.5,      # Wider stops for ranges
                'volatile': 2.0      # Widest stops for volatility
            }
        
        if self.config.regime_target_multipliers is None:
            self.config.regime_target_multipliers = {
                'bullish': 4.0,      # 4:1 RR for trends
                'bearish': 4.0,      # 4:1 RR for trends
                'neutral': 2.5,      # 2.5:1 RR for ranges
                'volatile': 3.0      # 3:1 RR for volatile
            }
        
        # Initialize default confidence thresholds if not provided
        if self.config.regime_confidence_thresholds is None:
            self.config.regime_confidence_thresholds = {
                'bullish': 0.75,     # Standard confidence for trends
                'bearish': 0.75,     # Standard confidence for trends
                'neutral': 0.82,     # Highest confidence for ranges
                'volatile': 0.70     # Lower confidence for volatile
            }
        
        # Performance tracking
        self.regime_performance = {regime.value: [] for regime in MarketRegime}
        self.trade_count = 0
    
    def apply_regime_filtering(self, signals: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
        """
        Apply regime-specific trade filtering.
        
        Args:
            signals: DataFrame with trading signals
            regimes: Series with regime classifications
            
        Returns:
            Filtered signals DataFrame
        """
        filtered_signals = signals.copy()
        
        # Apply regime filters
        for idx, signal in filtered_signals.iterrows():
            if idx < len(regimes):
                regime = regimes.iloc[idx]
                
                # Check if regime allows trading
                if not self.config.regime_filters.get(regime, True):
                    filtered_signals.at[idx, 'skip_trade'] = True
                    filtered_signals.at[idx, 'skip_reason'] = f"Regime filter: {regime}"
                
                # Check confidence threshold for regime
                if 'confidence' in signal:
                    min_confidence = self.config.regime_confidence_thresholds.get(regime, 0.75)
                    if signal['confidence'] < min_confidence:
                        filtered_signals.at[idx, 'skip_trade'] = True
                        filtered_signals.at[idx, 'skip_reason'] = f"Confidence threshold: {signal['confidence']:.3f} < {min_confidence:.3f}"
        
        # Log filtering results
        skipped_trades = filtered_signals.get('skip_trade', pd.Series([False] * len(filtered_signals))).sum()
        total_trades = len(filtered_signals)
        
        logger.info(f"Regime filtering: {skipped_trades}/{total_trades} trades skipped")
        
        return filtered_signals
    
    def apply_regime_position_sizing(self, signals: pd.DataFrame, regimes: pd.Series, 
                                   base_position_size: float = 1.0) -> pd.DataFrame:
        """
        Apply regime-specific position sizing.
        
        Args:
            signals: DataFrame with trading signals
            regimes: Series with regime classifications
            base_position_size: Base position size
            
        Returns:
            Signals with adjusted position sizes
        """
        sized_signals = signals.copy()
        
        for idx, signal in sized_signals.iterrows():
            if idx < len(regimes):
                regime = regimes.iloc[idx]
                
                # Get regime multiplier
                multiplier = self.config.regime_multipliers.get(regime, 1.0)
                
                # Calculate adjusted position size
                if 'position_size' in signal:
                    adjusted_size = signal['position_size'] * multiplier
                else:
                    adjusted_size = base_position_size * multiplier
                
                sized_signals.at[idx, 'adjusted_position_size'] = adjusted_size
                sized_signals.at[idx, 'regime_multiplier'] = multiplier
        
        return sized_signals
    
    def apply_regime_risk_management(self, signals: pd.DataFrame, regimes: pd.Series,
                                   atr_values: pd.Series = None) -> pd.DataFrame:
        """
        Apply regime-specific risk management (stop loss and take profit).
        
        Args:
            signals: DataFrame with trading signals
            regimes: Series with regime classifications
            atr_values: Series with ATR values for dynamic sizing
            
        Returns:
            Signals with regime-adjusted risk parameters
        """
        risk_adjusted_signals = signals.copy()
        
        for idx, signal in risk_adjusted_signals.iterrows():
            if idx < len(regimes):
                regime = regimes.iloc[idx]
                
                # Get regime risk multipliers
                stop_multiplier = self.config.regime_stop_multipliers.get(regime, 1.0)
                target_multiplier = self.config.regime_target_multipliers.get(regime, 2.0)
                
                # Get ATR for dynamic sizing
                atr = atr_values.iloc[idx] if atr_values is not None and idx < len(atr_values) else 0.001
                
                # Calculate regime-specific stop distance
                stop_distance = atr * stop_multiplier
                
                # Calculate regime-specific target distance
                target_distance = stop_distance * target_multiplier
                
                risk_adjusted_signals.at[idx, 'regime_stop_distance'] = stop_distance
                risk_adjusted_signals.at[idx, 'regime_target_distance'] = target_distance
                risk_adjusted_signals.at[idx, 'regime_rr_ratio'] = target_multiplier
        
        return risk_adjusted_signals
    
    def execute_regime_conditional_strategy(self, signals: pd.DataFrame, regimes: pd.Series,
                                          atr_values: pd.Series = None, 
                                          base_position_size: float = 1.0) -> pd.DataFrame:
        """
        Execute complete regime-conditional strategy.
        
        Args:
            signals: DataFrame with trading signals
            regimes: Series with regime classifications
            atr_values: Series with ATR values
            base_position_size: Base position size
            
        Returns:
            DataFrame with regime-adjusted signals
        """
        logger.info("Executing regime-conditional strategy")
        
        # Step 1: Apply regime filtering
        filtered_signals = self.apply_regime_filtering(signals, regimes)
        
        # Step 2: Apply regime position sizing
        sized_signals = self.apply_regime_position_sizing(
            filtered_signals, regimes, base_position_size
        )
        
        # Step 3: Apply regime risk management
        final_signals = self.apply_regime_risk_management(
            sized_signals, regimes, atr_values
        )
        
        # Log execution summary
        total_signals = len(signals)
        skip_trade_mask = final_signals.get('skip_trade', pd.Series([False] * len(final_signals)))
        filtered_signals_count = len(final_signals[skip_trade_mask == False])
        
        logger.info(f"Regime execution: {filtered_signals_count}/{total_signals} signals passed")
        
        # Log regime distribution
        regime_distribution = {}
        for idx, signal in final_signals.iterrows():
            if idx < len(regimes) and not signal.get('skip_trade', False):
                regime = regimes.iloc[idx]
                regime_distribution[regime] = regime_distribution.get(regime, 0) + 1
        
        logger.info(f"Regime distribution: {regime_distribution}")
        
        return final_signals
    
    def track_regime_performance(self, trade_results: pd.Series, regimes: pd.Series):
        """
        Track performance by regime for analysis and optimization.
        
        Args:
            trade_results: Series with trade P&L results
            regimes: Series with regime classifications
        """
        if not self.config.track_regime_performance:
            return
        
        for idx, result in trade_results.items():
            if idx < len(regimes):
                regime = regimes.iloc[idx]
                self.regime_performance[regime].append(result)
        
        # Keep only recent performance
        for regime in self.regime_performance:
            if len(self.regime_performance[regime]) > self.config.performance_window:
                self.regime_performance[regime] = self.regime_performance[regime][-self.config.performance_window:]
        
        self.trade_count += len(trade_results)
    
    def get_regime_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary by regime."""
        summary = {}
        
        for regime, results in self.regime_performance.items():
            if len(results) > 0:
                results_array = np.array(results)
                summary[regime] = {
                    'trades': len(results),
                    'total_pnl': np.sum(results_array),
                    'avg_pnl': np.mean(results_array),
                    'win_rate': np.mean(results_array > 0),
                    'std_pnl': np.std(results_array),
                    'sharpe': np.mean(results_array) / np.std(results_array) * np.sqrt(252) if np.std(results_array) > 0 else 0
                }
            else:
                summary[regime] = {
                    'trades': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0,
                    'win_rate': 0,
                    'std_pnl': 0,
                    'sharpe': 0
                }
        
        return summary
    
    def optimize_regime_parameters(self, performance_history: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Optimize regime parameters based on performance history.
        
        Args:
            performance_history: Historical performance by regime
            
        Returns:
            Optimization recommendations
        """
        recommendations = {}
        
        for regime, history in performance_history.items():
            if len(history) < 20:  # Need sufficient data
                recommendations[regime] = {'status': 'insufficient_data'}
                continue
            
            # Calculate recent performance
            recent_performance = history[-50:]  # Last 50 trades
            recent_win_rate = np.mean(np.array(recent_performance) > 0)
            recent_avg_pnl = np.mean(recent_performance)
            
            # Generate recommendations
            if recent_win_rate < 0.45:  # Poor win rate
                recommendations[regime] = {
                    'status': 'poor_performance',
                    'action': 'reduce_size',
                    'current_multiplier': self.config.regime_multipliers.get(regime, 1.0),
                    'suggested_multiplier': max(0.1, self.config.regime_multipliers.get(regime, 1.0) * 0.5),
                    'reason': f'Low win rate: {recent_win_rate:.1%}'
                }
            elif recent_win_rate > 0.65 and recent_avg_pnl > 30:  # Excellent performance
                recommendations[regime] = {
                    'status': 'excellent_performance',
                    'action': 'increase_size',
                    'current_multiplier': self.config.regime_multipliers.get(regime, 1.0),
                    'suggested_multiplier': min(2.0, self.config.regime_multipliers.get(regime, 1.0) * 1.2),
                    'reason': f'High win rate: {recent_win_rate:.1%}, Avg P&L: ${recent_avg_pnl:.1f}'
                }
            else:
                recommendations[regime] = {
                    'status': 'acceptable',
                    'action': 'maintain',
                    'current_multiplier': self.config.regime_multipliers.get(regime, 1.0),
                    'reason': f'Acceptable performance: {recent_win_rate:.1%} WR, ${recent_avg_pnl:.1f} avg'
                }
        
        return recommendations
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of regime execution configuration and performance."""
        return {
            'config': {
                'regime_multipliers': self.config.regime_multipliers,
                'regime_filters': self.config.regime_filters,
                'regime_stop_multipliers': self.config.regime_stop_multipliers,
                'regime_target_multipliers': self.config.regime_target_multipliers,
                'regime_confidence_thresholds': self.config.regime_confidence_thresholds
            },
            'performance': self.get_regime_performance_summary(),
            'trade_count': self.trade_count,
            'tracking_enabled': self.config.track_regime_performance
        }
