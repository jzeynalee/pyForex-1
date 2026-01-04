"""
Kelly-lite Capital Allocation for Alpha Factory

This module implements professional position sizing techniques:
1. Edge-based position sizing using Kelly criterion
2. Variance-adjusted position sizing
3. Regime-specific position multipliers
4. Risk-adjusted capital allocation
5. Portfolio-level risk management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import math

logger = logging.getLogger(__name__)

@dataclass
class KellyConfig:
    """Configuration for Kelly-lite position sizing."""
    # Kelly parameters
    kelly_fraction: float = 0.25  # Use 25% of full Kelly (conservative)
    max_position_size: float = 0.05  # 5% max position size
    min_position_size: float = 0.01  # 1% min position size
    
    # Edge calculation
    edge_window: int = 100  # Trades for edge calculation
    min_edge_threshold: float = 0.02  # 2% minimum edge
    
    # Variance adjustment
    variance_window: int = 50  # Trades for variance calculation
    variance_multiplier: float = 1.5  # Variance penalty multiplier
    
    # Risk management
    max_portfolio_risk: float = 0.10  # 10% max portfolio risk
    max_correlation_risk: float = 0.02  # 2% max correlated risk
    
    # Regime adjustments
    regime_multipliers: Dict[str, float] = None
    
    # Safety features
    drawdown_adjustment: bool = True
    volatility_adjustment: bool = True
    correlation_adjustment: bool = True

class KellyAllocator:
    """Professional Kelly-lite position sizing for Alpha Factory."""
    
    def __init__(self, config: KellyConfig = None):
        self.config = config or KellyConfig()
        
        # Initialize regime multipliers if not provided
        if self.config.regime_multipliers is None:
            self.config.regime_multipliers = {
                'bullish': 1.2,   # 20% larger in trends
                'bearish': 1.2,   # 20% larger in trends
                'neutral': 0.8,   # 20% smaller in ranges
                'volatile': 0.6   # 40% smaller in volatile
            }
        
        # Performance tracking
        self.trade_history = []
        self.edge_history = []
        self.variance_history = []
        self.position_sizes = []
        
        # Risk metrics
        self.current_portfolio_risk = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
    
    def calculate_edge(self, trade_results: List[float]) -> float:
        """
        Calculate trading edge from historical results.
        
        Args:
            trade_results: List of trade P&L results
            
        Returns:
            Edge (expected return as decimal)
        """
        if len(trade_results) < 10:
            return 0.0
        
        try:
            results_array = np.array(trade_results)
            
            # Calculate win rate and average win/loss
            wins = results_array[results_array > 0]
            losses = results_array[results_array < 0]
            
            win_rate = len(wins) / len(results_array)
            avg_win = np.mean(wins) if len(wins) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            
            # Calculate edge
            if avg_loss != 0:
                edge = win_rate * avg_win + (1 - win_rate) * avg_loss
                edge = edge / abs(avg_loss)  # Normalize by average loss
            else:
                edge = win_rate * avg_win
            
            return max(0, edge)  # Edge cannot be negative
            
        except Exception as e:
            logger.error(f"Error calculating edge: {e}")
            return 0.0
    
    def calculate_variance(self, trade_results: List[float]) -> float:
        """
        Calculate variance of trade results.
        
        Args:
            trade_results: List of trade P&L results
            
        Returns:
            Variance (as decimal)
        """
        if len(trade_results) < 10:
            return 1.0  # Default variance
        
        try:
            results_array = np.array(trade_results)
            variance = np.var(results_array)
            
            # Normalize variance (prevent extreme values)
            normalized_variance = min(10.0, max(0.1, variance))
            
            return normalized_variance
            
        except Exception as e:
            logger.error(f"Error calculating variance: {e}")
            return 1.0
    
    def calculate_kelly_fraction(self, edge: float, variance: float) -> float:
        """
        Calculate Kelly fraction with safety adjustments.
        
        Args:
            edge: Trading edge
            variance: Variance of returns
            
        Returns:
            Kelly fraction (position size as decimal)
        """
        try:
            # Basic Kelly formula: f = (bp - q) / b
            # Where b = average win/average loss ratio, p = win rate, q = loss rate
            # Simplified: f = edge / variance
            
            if variance <= 0:
                return 0.0
            
            # Calculate raw Kelly
            raw_kelly = edge / variance
            
            # Apply Kelly fraction (conservative approach)
            conservative_kelly = raw_kelly * self.config.kelly_fraction
            
            # Apply maximum position size limit
            limited_kelly = min(conservative_kelly, self.config.max_position_size)
            
            # Apply minimum position size
            final_kelly = max(limited_kelly, self.config.min_position_size) if limited_kelly > 0 else 0.0
            
            return final_kelly
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}")
            return self.config.min_position_size
    
    def adjust_for_regime(self, base_size: float, regime: str) -> float:
        """
        Adjust position size based on market regime.
        
        Args:
            base_size: Base position size
            regime: Market regime
            
        Returns:
            Regime-adjusted position size
        """
        regime_multiplier = self.config.regime_multipliers.get(regime, 1.0)
        return base_size * regime_multiplier
    
    def adjust_for_drawdown(self, base_size: float) -> float:
        """
        Adjust position size based on current drawdown.
        
        Args:
            base_size: Base position size
            
        Returns:
            Drawdown-adjusted position size
        """
        if not self.config.drawdown_adjustment:
            return base_size
        
        try:
            # Reduce position size based on drawdown
            if self.current_drawdown > 0.05:  # 5% drawdown
                drawdown_multiplier = max(0.5, 1 - self.current_drawdown * 2)
                return base_size * drawdown_multiplier
            
            return base_size
            
        except Exception as e:
            logger.error(f"Error adjusting for drawdown: {e}")
            return base_size
    
    def adjust_for_volatility(self, base_size: float, current_volatility: float, 
                            historical_volatility: float) -> float:
        """
        Adjust position size based on volatility.
        
        Args:
            base_size: Base position size
            current_volatility: Current market volatility
            historical_volatility: Historical average volatility
            
        Returns:
            Volatility-adjusted position size
        """
        if not self.config.volatility_adjustment:
            return base_size
        
        try:
            # Calculate volatility ratio
            volatility_ratio = current_volatility / historical_volatility
            
            # Adjust position size inversely to volatility
            if volatility_ratio > 1.5:  # High volatility
                volatility_multiplier = 0.7
            elif volatility_ratio < 0.5:  # Low volatility
                volatility_multiplier = 1.2
            else:
                volatility_multiplier = 1.0
            
            return base_size * volatility_multiplier
            
        except Exception as e:
            logger.error(f"Error adjusting for volatility: {e}")
            return base_size
    
    def calculate_optimal_position_size(self, trade_data: Dict[str, Any], 
                                     portfolio_data: Dict[str, Any] = None) -> Dict[str, float]:
        """
        Calculate optimal position size using Kelly-lite methodology.
        
        Args:
            trade_data: Dictionary with trade information
            portfolio_data: Dictionary with portfolio information
            
        Returns:
            Dictionary with position sizing details
        """
        logger.info("Calculating optimal position size")
        
        # Extract trade parameters
        confidence = trade_data.get('confidence', 0.7)
        expected_return = trade_data.get('expected_return', 0.02)
        regime = trade_data.get('regime', 'neutral')
        
        # Get recent trade results
        recent_results = self.trade_history[-self.config.edge_window:]
        
        # Calculate edge
        edge = self.calculate_edge(recent_results)
        self.edge_history.append(edge)
        
        # Calculate variance
        variance = self.calculate_variance(recent_results)
        self.variance_history.append(variance)
        
        # Calculate base Kelly fraction
        base_kelly = self.calculate_kelly_fraction(edge, variance)
        
        # Adjust for confidence
        confidence_adjustment = min(1.5, confidence / 0.7)  # Normalize to 0.7 base confidence
        confidence_adjusted_kelly = base_kelly * confidence_adjustment
        
        # Adjust for regime
        regime_adjusted_kelly = self.adjust_for_regime(confidence_adjusted_kelly, regime)
        
        # Adjust for drawdown
        drawdown_adjusted_kelly = self.adjust_for_drawdown(regime_adjusted_kelly)
        
        # Adjust for volatility if provided
        if portfolio_data and 'current_volatility' in portfolio_data:
            current_vol = portfolio_data['current_volatility']
            historical_vol = portfolio_data.get('historical_volatility', current_vol)
            final_kelly = self.adjust_for_volatility(drawdown_adjusted_kelly, current_vol, historical_vol)
        else:
            final_kelly = drawdown_adjusted_kelly
        
        # Apply final limits
        final_position_size = min(final_kelly, self.config.max_position_size)
        final_position_size = max(final_position_size, 0.0)
        
        # Store position size
        self.position_sizes.append(final_position_size)
        
        # Calculate position metrics
        position_metrics = {
            'base_kelly': base_kelly,
            'confidence_adjusted': confidence_adjusted_kelly,
            'regime_adjusted': regime_adjusted_kelly,
            'drawdown_adjusted': drawdown_adjusted_kelly,
            'final_position_size': final_position_size,
            'edge': edge,
            'variance': variance,
            'regime': regime,
            'confidence': confidence
        }
        
        logger.info(f"Position size calculated: {final_position_size:.3f}")
        
        return position_metrics
    
    def update_trade_result(self, pnl: float, confidence: float, regime: str):
        """
        Update trade results for edge and variance calculation.
        
        Args:
            pnl: Trade P&L result
            confidence: Trade confidence
            regime: Trade regime
        """
        self.trade_history.append(pnl)
        
        # Update drawdown
        if len(self.trade_history) > 1:
            cumulative_pnl = np.cumsum(self.trade_history)
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdown = (peak - cumulative_pnl) / peak
            self.current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        logger.info(f"Trade result updated: P&L={pnl:.2f}, Drawdown={self.current_drawdown:.3f}")
    
    def get_position_sizing_summary(self) -> Dict[str, Any]:
        """Get summary of position sizing performance."""
        if not self.position_sizes:
            return {'status': 'no_position_history'}
        
        summary = {
            'total_positions': len(self.position_sizes),
            'avg_position_size': np.mean(self.position_sizes),
            'max_position_size': np.max(self.position_sizes),
            'min_position_size': np.min(self.position_sizes),
            'current_edge': self.edge_history[-1] if self.edge_history else 0,
            'current_variance': self.variance_history[-1] if self.variance_history else 0,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'config': {
                'kelly_fraction': self.config.kelly_fraction,
                'max_position_size': self.config.max_position_size,
                'min_position_size': self.config.min_position_size
            }
        }
        
        # Calculate position size statistics
        if len(self.position_sizes) > 10:
            summary['position_size_std'] = np.std(self.position_sizes)
            summary['position_size_range'] = np.max(self.position_sizes) - np.min(self.position_sizes)
        
        return summary
    
    def analyze_regime_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze position sizing performance by regime."""
        if not self.trade_history or not self.position_sizes:
            return {'status': 'insufficient_data'}
        
        regime_performance = {}
        
        # This would need regime tracking per trade
        # For now, return placeholder
        for regime in ['bullish', 'bearish', 'neutral', 'volatile']:
            regime_performance[regime] = {
                'avg_position_size': 0.025,  # Placeholder
                'avg_pnl': 0.0,
                'win_rate': 0.0,
                'total_trades': 0
            }
        
        return regime_performance
    
    def optimize_parameters(self, performance_history: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Optimize Kelly parameters based on performance history.
        
        Args:
            performance_history: List of performance metrics
            
        Returns:
            Optimization recommendations
        """
        if len(performance_history) < 20:
            return {'status': 'insufficient_data'}
        
        recommendations = {}
        
        # Analyze recent performance
        recent_performance = performance_history[-50:]
        recent_sharpe = np.mean([p.get('sharpe', 0) for p in recent_performance])
        recent_drawdown = np.mean([p.get('max_drawdown', 0) for p in recent_performance])
        
        # Generate recommendations
        if recent_drawdown > 0.15:  # High drawdown
            recommendations['kelly_fraction'] = max(0.1, self.config.kelly_fraction * 0.8)
            recommendations['reason'] = f"High drawdown: {recent_drawdown:.1%}"
        elif recent_sharpe > 2.0 and recent_drawdown < 0.05:  # Excellent performance
            recommendations['kelly_fraction'] = min(0.5, self.config.kelly_fraction * 1.1)
            recommendations['reason'] = f"Excellent performance: Sharpe {recent_sharpe:.1f}"
        else:
            recommendations['kelly_fraction'] = self.config.kelly_fraction
            recommendations['reason'] = "Performance acceptable, maintain current settings"
        
        return recommendations
