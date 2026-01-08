"""
Advanced Alpha Survivability Validation
Institutional-Grade Notional Scaling Analysis

Tests alpha performance at increasing notional sizes to prove
survivability under institutional trading volumes.
"""

import sys
sys.path.append('e:/pyProject/pyForex-1')

from research_backtest import ResearchBacktest
from alpha_factory.risk_profile_manager import RiskProfile
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class NotionalScalingBacktest:
    """Advanced alpha survivability validation with notional scaling."""
    
    def __init__(self):
        self.base_notional = 100000  # $100K base notional
        self.scaling_factors = [1, 5, 10, 25, 50, 100]  # Up to $10M
        self.results = {}
        
    def calculate_capacity_impact(self, notional: float, trade_size: float) -> Dict[str, float]:
        """Calculate capacity impact based on notional size."""
        # Capacity constraints increase with notional
        base_capacity = 0.15  # 15% of notional per session
        max_position = 0.02   # 2% of notional per trade
        
        # Adjust for notional size (larger notional = tighter constraints)
        notional_multiplier = min(1.0, self.base_notional / notional)
        
        adjusted_max_position = max_position * notional_multiplier
        adjusted_session_capacity = base_capacity * notional_multiplier
        
        return {
            'max_position_size': adjusted_max_position,
            'session_capacity': adjusted_session_capacity,
            'capacity_impact': 1.0 - notional_multiplier
        }
    
    def apply_notional_adjustments(self, signals: List[Dict[str, Any]], notional: float) -> List[Dict[str, Any]]:
        """Apply notional-based adjustments to signals."""
        capacity_impact = self.calculate_capacity_impact(notional, 0.02)
        
        adjusted_signals = []
        for signal in signals:
            adjusted_signal = signal.copy()
            
            # Reduce position size based on capacity
            adjusted_signal['position_size'] *= capacity_impact['max_position_size'] / 0.02
            
            # Apply slippage impact (increases with notional)
            slippage_impact = 1.0 - (0.001 * (notional / self.base_notional))  # 0.1% per $100K
            adjusted_signal['expected_value'] *= slippage_impact
            
            adjusted_signals.append(adjusted_signal)
        
        return adjusted_signals
    
    def run_notional_scaling_test(self) -> Dict[str, Any]:
        """Run complete notional scaling analysis."""
        print("=" * 80)
        print("ADVANCED ALPHA SURVIVABILITY VALIDATION")
        print("Institutional-Grade Notional Scaling Analysis")
        print("=" * 80)
        
        # Generate base data
        print("\n📊 Generating base test data...")
        base_backtest = ResearchBacktest(RiskProfile.RESEARCH)
        data = base_backtest.generate_compatible_data(10000)
        
        # Test each notional size
        for factor in self.scaling_factors:
            notional = self.base_notional * factor
            print(f"\n🚀 Testing Notional: ${notional:,} ({factor}x base)")
            
            # Initialize backtest with notional adjustments
            backtest = NotionalAdjustedBacktest(RiskProfile.RESEARCH, notional)
            
            # Run backtest
            metrics = backtest.run_backtest(data)
            
            # Store results
            self.results[factor] = {
                'notional': notional,
                'metrics': metrics,
                'capacity_impact': self.calculate_capacity_impact(notional, 0.02)
            }
            
            # Display results
            print(f"   Trades: {metrics['summary']['total_trades']:,}")
            print(f"   Win Rate: {metrics['summary']['win_rate']:.2%}")
            print(f"   Avg P&L: ${metrics['summary']['avg_trade']:.2f}")
            print(f"   Mean EV: ${metrics['ev_distribution']['mean']:.2f}")
            print(f"   Capacity Impact: {self.results[factor]['capacity_impact']['capacity_impact']:.1%}")
        
        # Generate survivability analysis
        return self.generate_survivability_report()
    
    def generate_survivability_report(self) -> Dict[str, Any]:
        """Generate comprehensive survivability analysis."""
        print(f"\n📈 GENERATING SURVIVABILITY ANALYSIS...")
        
        # Calculate survivability metrics
        survivability = {
            'trade_degradation': {},
            'ev_degradation': {},
            'capacity_impact': {},
            'notional_threshold': None,
            'survivability_score': 0
        }
        
        base_trades = self.results[1]['metrics']['summary']['total_trades']
        base_ev = self.results[1]['metrics']['ev_distribution']['mean']
        
        for factor in self.scaling_factors:
            if factor == 1:
                continue
                
            current = self.results[factor]
            
            # Trade degradation
            trade_degradation = (base_trades - current['metrics']['summary']['total_trades']) / base_trades
            survivability['trade_degradation'][factor] = trade_degradation
            
            # EV degradation
            ev_degradation = (base_ev - current['metrics']['ev_distribution']['mean']) / base_ev
            survivability['ev_degradation'][factor] = ev_degradation
            
            # Capacity impact
            survivability['capacity_impact'][factor] = current['capacity_impact']['capacity_impact']
        
        # Find survivability threshold (where degradation > 50%)
        for factor in self.scaling_factors:
            if (survivability['trade_degradation'].get(factor, 0) > 0.5 or 
                survivability['ev_degradation'].get(factor, 0) > 0.5):
                survivability['notional_threshold'] = self.results[factor]['notional']
                break
        
        # Calculate survivability score
        total_score = 0
        max_score = 0
        
        for factor in self.scaling_factors:
            if factor == 1:
                continue
                
            trade_score = max(0, 1 - survivability['trade_degradation'][factor] * 2)
            ev_score = max(0, 1 - survivability['ev_degradation'][factor] * 2)
            capacity_score = max(0, 1 - survivability['capacity_impact'][factor])
            
            factor_score = (trade_score + ev_score + capacity_score) / 3
            total_score += factor_score
            max_score += 1
        
        survivability['survivability_score'] = (total_score / max_score) * 100 if max_score > 0 else 0
        
        # Display results
        print(f"\n🎯 SURVIVABILITY ANALYSIS RESULTS:")
        print(f"   Survivability Score: {survivability['survivability_score']:.1f}/100")
        
        if survivability['notional_threshold']:
            print(f"   Notional Threshold: ${survivability['notional_threshold']:,}")
            print(f"   ✅ Alpha survives up to threshold")
        else:
            print(f"   Notional Threshold: None")
            print(f"   ✅ Alpha survives all tested notional sizes")
        
        print(f"\n📊 DEGRADATION ANALYSIS:")
        for factor in self.scaling_factors:
            if factor == 1:
                continue
                
            trade_deg = survivability['trade_degradation'][factor] * 100
            ev_deg = survivability['ev_degradation'][factor] * 100
            cap_imp = survivability['capacity_impact'][factor] * 100
            
            print(f"   {factor}x (${self.results[factor]['notional']:,}):")
            print(f"     Trade Degradation: {trade_deg:.1f}%")
            print(f"     EV Degradation: {ev_deg:.1f}%")
            print(f"     Capacity Impact: {cap_imp:.1f}%")
        
        return survivability

class NotionalAdjustedBacktest(ResearchBacktest):
    """Backtest with notional-based adjustments."""
    
    def __init__(self, risk_profile, notional):
        super().__init__(risk_profile)
        self.notional = notional
        self.scaling_analyzer = NotionalScalingBacktest()
    
    def apply_risk_layer(self, signals: List[Dict[str, Any]], row: pd.Series) -> List[Dict[str, Any]]:
        """Apply risk layer with notional adjustments."""
        # Apply base risk layer
        base_signals = super().apply_risk_layer(signals, row)
        
        # Apply notional adjustments
        adjusted_signals = self.scaling_analyzer.apply_notional_adjustments(base_signals, self.notional)
        
        return adjusted_signals

def run_advanced_survivability_test():
    """Run complete advanced survivability validation."""
    analyzer = NotionalScalingBacktest()
    return analyzer.run_notional_scaling_test()

if __name__ == "__main__":
    run_advanced_survivability_test()
