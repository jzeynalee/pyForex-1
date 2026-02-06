"""
Portfolio-Level Alpha for Alpha Factory

Phase 5: Portfolio-Level Alpha (Professional Scaling)

Goal: Move from single-strategy thinking to portfolio alpha
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import itertools

logger = logging.getLogger(__name__)

@dataclass
class AlphaVariant:
    """Represents an alpha variant."""
    def __init__(self, name: str, horizon: str, sensitivity: str, 
                 base_performance: Dict[str, float]):
        self.name = name
        self.horizon = horizon  # 'short', 'medium', 'long'
        self.sensitivity = sensitivity  # 'low', 'medium', 'high'
        self.base_performance = base_performance
        self.current_performance = base_performance.copy()
        self.correlation_matrix = {}
        self.weight = 0.0
        self.active = True

@dataclass
class PortfolioConfig:
    """Configuration for portfolio alpha management."""
    # Alpha variants
    max_variants_per_strategy: int = 3
    min_variant_correlation: float = 0.7  # Drop variants with correlation > 0.7
    
    # Capital allocation
    allocation_method: str = 'ev_weighted'  # 'ev_weighted', 'correlation_adjusted', 'drawdown_aware'
    min_allocation_per_variant: float = 0.05  # 5% minimum
    max_allocation_per_variant: float = 0.4   # 40% maximum
    
    # Risk management
    max_portfolio_correlation: float = 0.8
    rebalance_threshold: float = 0.1  # 10% deviation triggers rebalance
    performance_window: int = 100  # Trades for performance evaluation
    
    # Drawdown management
    max_variant_drawdown: float = 0.15  # 15% max per variant
    portfolio_drawdown_threshold: float = 0.1  # 10% portfolio drawdown triggers action

class PortfolioAlphaManager:
    """Professional portfolio-level alpha management."""
    
    def __init__(self, config: PortfolioConfig = None):
        self.config = config or PortfolioConfig()
        
        # Alpha variants
        self.alpha_variants = {}
        self.variant_performance_history = defaultdict(list)
        
        # Portfolio allocation
        self.current_allocation = {}
        self.allocation_history = []
        
        # Correlation tracking
        self.correlation_matrix = {}
        self.correlation_history = []
        
        # Performance tracking
        self.portfolio_performance = {
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0
        }
        
        logger.info("Portfolio alpha manager initialized")
    
    def create_alpha_variants(self, base_strategy_name: str) -> List[AlphaVariant]:
        """
        Create alpha variants from base strategy.
        
        Args:
            base_strategy_name: Name of base strategy
            
        Returns:
            List of alpha variants
        """
        variants = []
        
        # Define variant configurations
        variant_configs = [
            # Horizon variants
            {'name': f'{base_strategy_name}_short_term', 'horizon': 'short', 'sensitivity': 'medium'},
            {'name': f'{base_strategy_name}_medium_term', 'horizon': 'medium', 'sensitivity': 'medium'},
            {'name': f'{base_strategy_name}_long_term', 'horizon': 'long', 'sensitivity': 'medium'},
            
            # Sensitivity variants
            {'name': f'{base_strategy_name}_low_sensitivity', 'horizon': 'medium', 'sensitivity': 'low'},
            {'name': f'{base_strategy_name}_high_sensitivity', 'horizon': 'medium', 'sensitivity': 'high'},
            
            # Hybrid variants
            {'name': f'{base_strategy_name}_short_low', 'horizon': 'short', 'sensitivity': 'low'},
            {'name': f'{base_strategy_name}_long_high', 'horizon': 'long', 'sensitivity': 'high'},
        ]
        
        # Create variants with adjusted base performance
        base_performance = {
            'win_rate': 0.65,
            'avg_win': 45.0,
            'avg_loss': -18.0,
            'expectancy': 25.0,
            'sharpe': 1.5,
            'max_drawdown': 0.12
        }
        
        for config in variant_configs[:self.config.max_variants_per_strategy]:
            # Adjust performance based on variant characteristics
            adjusted_performance = base_performance.copy()
            
            # Horizon adjustments
            if config['horizon'] == 'short':
                adjusted_performance['win_rate'] *= 0.95  # Slightly lower WR
                adjusted_performance['avg_win'] *= 0.8     # Smaller wins
                adjusted_performance['avg_loss'] *= 0.7    # Smaller losses
                adjusted_performance['sharpe'] *= 1.1       # Better risk-adjusted
            elif config['horizon'] == 'long':
                adjusted_performance['win_rate'] *= 1.05  # Higher WR
                adjusted_performance['avg_win'] *= 1.3     # Larger wins
                adjusted_performance['avg_loss'] *= 1.2    # Larger losses
                adjusted_performance['sharpe'] *= 0.9       # Lower risk-adjusted
            
            # Sensitivity adjustments
            if config['sensitivity'] == 'low':
                adjusted_performance['win_rate'] *= 1.02  # More stable
                adjusted_performance['sharpe'] *= 1.15     # Better risk-adjusted
            elif config['sensitivity'] == 'high':
                adjusted_performance['win_rate'] *= 0.98  # Less stable
                adjusted_performance['sharpe'] *= 0.85     # Lower risk-adjusted
            
            variant = AlphaVariant(
                config['name'],
                config['horizon'],
                config['sensitivity'],
                adjusted_performance
            )
            
            variants.append(variant)
            self.alpha_variants[variant.name] = variant
        
        logger.info(f"Created {len(variants)} alpha variants for {base_strategy_name}")
        
        return variants
    
    def calculate_variant_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlations between alpha variants.
        
        Returns:
            Correlation matrix dictionary
        """
        variants = list(self.alpha_variants.values())
        n_variants = len(variants)
        
        if n_variants < 2:
            return {}
        
        # Initialize correlation matrix
        correlation_matrix = {}
        
        for i, variant1 in enumerate(variants):
            correlation_matrix[variant1.name] = {}
            
            for j, variant2 in enumerate(variants):
                if i == j:
                    correlation_matrix[variant1.name][variant2.name] = 1.0
                else:
                    # Calculate correlation based on variant characteristics
                    correlation = self._estimate_variant_correlation(variant1, variant2)
                    correlation_matrix[variant1.name][variant2.name] = correlation
        
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
    
    def _estimate_variant_correlation(self, variant1: AlphaVariant, variant2: AlphaVariant) -> float:
        """
        Estimate correlation between two variants based on characteristics.
        
        Args:
            variant1: First variant
            variant2: Second variant
            
        Returns:
            Estimated correlation (0-1)
        """
        correlation = 0.5  # Base correlation
        
        # Horizon similarity
        if variant1.horizon == variant2.horizon:
            correlation += 0.3
        elif abs(self._horizon_value(variant1.horizon) - self._horizon_value(variant2.horizon)) == 2:
            correlation -= 0.2
        
        # Sensitivity similarity
        if variant1.sensitivity == variant2.sensitivity:
            correlation += 0.2
        elif abs(self._sensitivity_value(variant1.sensitivity) - self._sensitivity_value(variant2.sensitivity)) == 2:
            correlation -= 0.1
        
        # Ensure correlation is in valid range
        return max(0.0, min(1.0, correlation))
    
    def _horizon_value(self, horizon: str) -> int:
        """Convert horizon to numeric value."""
        mapping = {'short': 0, 'medium': 1, 'long': 2}
        return mapping.get(horizon, 1)
    
    def _sensitivity_value(self, sensitivity: str) -> int:
        """Convert sensitivity to numeric value."""
        mapping = {'low': 0, 'medium': 1, 'high': 2}
        return mapping.get(sensitivity, 1)
    
    def filter_high_correlation_variants(self) -> List[str]:
        """
        Filter out variants with high correlation.
        
        Returns:
            List of variant names to remove
        """
        variants_to_remove = []
        
        if not self.correlation_matrix:
            return variants_to_remove
        
        variant_names = list(self.alpha_variants.keys())
        
        for i, variant1 in enumerate(variant_names):
            if variant1 in variants_to_remove:
                continue
                
            for j, variant2 in enumerate(variant_names[i+1:], i+1):
                if variant2 in variants_to_remove:
                    continue
                
                correlation = self.correlation_matrix[variant1][variant2]
                
                if correlation > self.config.min_variant_correlation:
                    # Remove the variant with lower performance
                    perf1 = self.alpha_variants[variant1].current_performance['sharpe']
                    perf2 = self.alpha_variants[variant2].current_performance['sharpe']
                    
                    if perf1 < perf2:
                        variants_to_remove.append(variant1)
                    else:
                        variants_to_remove.append(variant2)
        
        return list(set(variants_to_remove))
    
    def calculate_ev_weighted_allocation(self) -> Dict[str, float]:
        """
        Calculate EV-weighted portfolio allocation.
        
        Returns:
            Dictionary of variant allocations
        """
        allocations = {}
        
        # Calculate total EV
        total_ev = sum(variant.current_performance['expectancy'] 
                       for variant in self.alpha_variants.values() 
                       if variant.active)
        
        if total_ev == 0:
            return allocations
        
        # Allocate based on EV contribution
        for variant_name, variant in self.alpha_variants.items():
            if not variant.active:
                allocations[variant_name] = 0.0
                continue
            
            ev_weight = variant.current_performance['expectancy'] / total_ev
            allocations[variant_name] = ev_weight
        
        return allocations
    
    def calculate_correlation_adjusted_allocation(self) -> Dict[str, float]:
        """
        Calculate correlation-adjusted portfolio allocation.
        
        Returns:
            Dictionary of variant allocations
        """
        ev_allocations = self.calculate_ev_weighted_allocation()
        
        if not ev_allocations or not self.correlation_matrix:
            return ev_allocations
        
        # Adjust for correlations
        adjusted_allocations = {}
        
        for variant_name, ev_allocation in ev_allocations.items():
            if ev_allocation == 0:
                adjusted_allocations[variant_name] = 0.0
                continue
            
            # Calculate correlation penalty
            correlation_penalty = 0.0
            total_correlation = 0.0
            
            for other_variant, allocation in ev_allocations.items():
                if other_variant != variant_name and allocation > 0:
                    correlation = self.correlation_matrix.get(variant_name, {}).get(other_variant, 0.0)
                    correlation_penalty += correlation * allocation
                    total_correlation += allocation
            
            if total_correlation > 0:
                penalty_factor = correlation_penalty / total_correlation
                adjusted_allocation = ev_allocation * (1 - penalty_factor * 0.5)  # 50% penalty weight
            else:
                adjusted_allocation = ev_allocation
            
            adjusted_allocations[variant_name] = max(0, adjusted_allocation)
        
        # Normalize to sum to 1
        total_adjusted = sum(adjusted_allocations.values())
        if total_adjusted > 0:
            adjusted_allocations = {k: v/total_adjusted for k, v in adjusted_allocations.items()}
        
        return adjusted_allocations
    
    def calculate_drawdown_aware_allocation(self) -> Dict[str, float]:
        """
        Calculate drawdown-aware portfolio allocation.
        
        Returns:
            Dictionary of variant allocations
        """
        base_allocations = self.calculate_correlation_adjusted_allocation()
        
        if not base_allocations:
            return base_allocations
        
        # Adjust for drawdowns
        drawdown_adjusted = {}
        
        for variant_name, base_allocation in base_allocations.items():
            if base_allocation == 0:
                drawdown_adjusted[variant_name] = 0.0
                continue
            
            variant = self.alpha_variants[variant_name]
            current_dd = variant.current_performance.get('current_drawdown', 0.0)
            max_dd = variant.current_performance.get('max_drawdown', 0.12)
            
            # Reduce allocation based on drawdown
            if max_dd > 0:
                drawdown_ratio = current_dd / max_dd
                if drawdown_ratio > 0.5:  # More than 50% of max drawdown
                    adjustment_factor = 0.5  # Reduce by 50%
                elif drawdown_ratio > 0.3:  # More than 30% of max drawdown
                    adjustment_factor = 0.8  # Reduce by 20%
                else:
                    adjustment_factor = 1.0  # No adjustment
                
                drawdown_adjusted[variant_name] = base_allocation * adjustment_factor
            else:
                drawdown_adjusted[variant_name] = base_allocation
        
        # Normalize to sum to 1
        total_adjusted = sum(drawdown_adjusted.values())
        if total_adjusted > 0:
            drawdown_adjusted = {k: v/total_adjusted for k, v in drawdown_adjusted.items()}
        
        return drawdown_adjusted
    
    def optimize_portfolio_allocation(self) -> Dict[str, float]:
        """
        Optimize portfolio allocation using selected method.
        
        Returns:
            Dictionary of optimized allocations
        """
        # Calculate correlations
        self.calculate_variant_correlations()
        
        # Filter high correlation variants
        variants_to_remove = self.filter_high_correlation_variants()
        for variant_name in variants_to_remove:
            if variant_name in self.alpha_variants:
                self.alpha_variants[variant_name].active = False
                logger.info(f"Deactivated high-correlation variant: {variant_name}")
        
        # Calculate allocation based on method
        if self.config.allocation_method == 'ev_weighted':
            allocation = self.calculate_ev_weighted_allocation()
        elif self.config.allocation_method == 'correlation_adjusted':
            allocation = self.calculate_correlation_adjusted_allocation()
        elif self.config.allocation_method == 'drawdown_aware':
            allocation = self.calculate_drawdown_aware_allocation()
        else:
            allocation = self.calculate_ev_weighted_allocation()
        
        # Apply allocation constraints
        constrained_allocation = {}
        
        for variant_name, weight in allocation.items():
            if weight < self.config.min_allocation_per_variant:
                constrained_allocation[variant_name] = 0.0
            elif weight > self.config.max_allocation_per_variant:
                constrained_allocation[variant_name] = self.config.max_allocation_per_variant
            else:
                constrained_allocation[variant_name] = weight
        
        # Renormalize
        total_weight = sum(constrained_allocation.values())
        if total_weight > 0:
            constrained_allocation = {k: v/total_weight for k, v in constrained_allocation.items()}
        
        self.current_allocation = constrained_allocation
        self.allocation_history.append({
            'timestamp': datetime.now(),
            'allocation': constrained_allocation.copy(),
            'method': self.config.allocation_method,
            'active_variants': len([v for v in self.alpha_variants.values() if v.active])
        })
        
        logger.info(f"Portfolio allocation optimized: {len(constrained_allocation)} active variants")
        
        return constrained_allocation
    
    def update_variant_performance(self, variant_name: str, performance_data: Dict[str, float]):
        """
        Update performance data for a variant.
        
        Args:
            variant_name: Name of the variant
            performance_data: Performance metrics
        """
        if variant_name not in self.alpha_variants:
            logger.warning(f"Variant {variant_name} not found")
            return
        
        variant = self.alpha_variants[variant_name]
        variant.current_performance.update(performance_data)
        
        # Store in history
        self.variant_performance_history[variant_name].append({
            'timestamp': datetime.now(),
            'performance': performance_data.copy()
        })
        
        # Keep history manageable
        if len(self.variant_performance_history[variant_name]) > 200:
            self.variant_performance_history[variant_name] = self.variant_performance_history[variant_name][-100:]
        
        logger.debug(f"Updated performance for {variant_name}")
    
    def calculate_portfolio_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio-level performance metrics.
        
        Returns:
            Dictionary with portfolio metrics
        """
        if not self.current_allocation:
            return {
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'current_drawdown': 0.0,
                'active_variants': 0
            }
        
        # Weighted average of variant metrics
        total_weight = sum(self.current_allocation.values())
        
        if total_weight == 0:
            return {
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'current_drawdown': 0.0,
                'active_variants': 0
            }
        
        weighted_metrics = {
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0
        }
        
        for variant_name, weight in self.current_allocation.items():
            if weight == 0 or variant_name not in self.alpha_variants:
                continue
            
            variant = self.alpha_variants[variant_name]
            perf = variant.current_performance
            
            weighted_metrics['total_pnl'] += perf.get('total_pnl', 0) * weight
            weighted_metrics['win_rate'] += perf.get('win_rate', 0) * weight
            weighted_metrics['sharpe'] += perf.get('sharpe', 0) * weight
            weighted_metrics['max_drawdown'] += perf.get('max_drawdown', 0) * weight
            weighted_metrics['current_drawdown'] += perf.get('current_drawdown', 0) * weight
        
        weighted_metrics['active_variants'] = len([v for v in self.alpha_variants.values() if v.active])
        
        return weighted_metrics
    
    def get_portfolio_report(self) -> Dict[str, Any]:
        """Get comprehensive portfolio report."""
        portfolio_metrics = self.calculate_portfolio_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'allocation_method': self.config.allocation_method,
            'current_allocation': self.current_allocation,
            'portfolio_metrics': portfolio_metrics,
            'active_variants': len([v for v in self.alpha_variants.values() if v.active]),
            'total_variants': len(self.alpha_variants),
            'correlation_matrix': self.correlation_matrix,
            'allocation_history': self.allocation_history[-5:],  # Last 5 allocations
            'config': {
                'max_variants_per_strategy': self.config.max_variants_per_strategy,
                'min_correlation_threshold': self.config.min_variant_correlation,
                'allocation_method': self.config.allocation_method
            }
        }
