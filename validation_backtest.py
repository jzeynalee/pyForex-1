"""
Validation Profile Backtest for Alpha Factory

PHASE 6: VALIDATION PROFILE TEST
"""

import sys
sys.path.append('e:/pyProject/pyForex-1')

from research_backtest import ResearchBacktest
from alpha_factory.risk_profile_manager import RiskProfile
import logging

logger = logging.getLogger(__name__)

def run_validation_backtest():
    """Run VALIDATION profile test."""
    print("=" * 80)
    print("VALIDATION PROFILE BACKTEST - ALPHA FACTORY")
    print("=" * 80)
    
    # Generate compatible data
    print("\n📊 Generating post-2010 compatible data...")
    data = ResearchBacktest(RiskProfile.RESEARCH).generate_compatible_data(10000)
    
    # Test VALIDATION profile
    print(f"\n🚀 Running VALIDATION profile test (PHASE 6)...")
    
    # Initialize backtest for VALIDATION profile
    backtest = ResearchBacktest(RiskProfile.VALIDATION)
    
    # Run backtest
    metrics = backtest.run_backtest(data)
    
    # Display results
    print(f"\n📈 VALIDATION Profile Results:")
    print(f"   Total Trades: {metrics['summary']['total_trades']:,}")
    print(f"   Win Rate: {metrics['summary']['win_rate']:.2%}")
    print(f"   Total P&L: ${metrics['summary']['total_pnl']:,.2f}")
    print(f"   Average P&L: ${metrics['summary']['avg_trade']:.2f}")
    print(f"   Mean EV: ${metrics['ev_distribution']['mean']:.2f}")
    print(f"   Median EV: ${metrics['ev_distribution']['median']:.2f}")
    
    # VALIDATION acceptance criteria
    trades_ok = metrics['summary']['total_trades'] >= 300  # PHASE 4 requirement
    ev_ok = metrics['ev_distribution']['mean'] > 0
    trade_reduction_ok = metrics['summary']['total_trades'] >= 200  # PHASE 4 requirement
    
    print(f"\n🎯 VALIDATION Acceptance Criteria:")
    print(f"   Trades >= 300: {'PASS' if trades_ok else 'FAIL'}")
    print(f"   Mean EV > 0: {'PASS' if ev_ok else 'FAIL'}")
    print(f"   Trades >= 200: {'PASS' if trade_reduction_ok else 'FAIL'}")
    
    # Calculate trade reduction from RESEARCH (need RESEARCH results)
    research_trades = 3873  # From previous run
    current_trades = metrics['summary']['total_trades']
    reduction_percentage = (research_trades - current_trades) / research_trades * 100
    
    print(f"   Trade Reduction: {reduction_percentage:.1f}% (from RESEARCH)")
    print(f"   Reduction <= 60%: {'PASS' if reduction_percentage <= 60 else 'FAIL'}")
    
    if trades_ok and ev_ok and trade_reduction_ok and reduction_percentage <= 60:
        print(f"\n✅ VALIDATION profile PASSED acceptance criteria!")
        
        # Check EV distribution improvement
        research_mean_ev = 26.52  # From RESEARCH profile
        validation_mean_ev = metrics['ev_distribution']['mean']
        ev_improvement = (validation_mean_ev - research_mean_ev) / research_mean_ev * 100
        
        print(f"   EV Improvement: {ev_improvement:+.1f}% (from RESEARCH)")
        
        if ev_improvement > 0:
            print(f"   ✅ EV distribution improved")
        else:
            print(f"   ⚠️ EV distribution declined")
        
    else:
        print(f"\n❌ VALIDATION profile FAILED acceptance criteria!")
        return None
    
    # Generate report
    report_file = backtest.generate_report()
    print(f"   Report: {report_file}")
    
    return metrics

if __name__ == "__main__":
    run_validation_backtest()
