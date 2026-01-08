"""
Production Profile Backtest for Alpha Factory

PHASE 7: PRODUCTION READINESS CHECK
"""

import sys
sys.path.append('e:/pyProject/pyForex-1')

from research_backtest import ResearchBacktest
from alpha_factory.risk_profile_manager import RiskProfile
import logging

logger = logging.getLogger(__name__)

def run_production_backtest():
    """Run PRODUCTION profile test."""
    print("=" * 80)
    print("PRODUCTION PROFILE BACKTEST - ALPHA FACTORY")
    print("=" * 80)
    
    # Generate compatible data
    print("\n📊 Generating post-2010 compatible data...")
    data = ResearchBacktest(RiskProfile.RESEARCH).generate_compatible_data(10000)
    
    # Test PRODUCTION profile
    print(f"\n🚀 Running PRODUCTION profile test (PHASE 7)...")
    
    # Initialize backtest for PRODUCTION profile
    backtest = ResearchBacktest(RiskProfile.PRODUCTION)
    
    # Run backtest
    metrics = backtest.run_backtest(data)
    
    # Display results
    print(f"\n📈 PRODUCTION Profile Results:")
    print(f"   Total Trades: {metrics['summary']['total_trades']:,}")
    print(f"   Win Rate: {metrics['summary']['win_rate']:.2%}")
    print(f"   Total P&L: ${metrics['summary']['total_pnl']:,.2f}")
    print(f"   Average P&L: ${metrics['summary']['avg_trade']:.2f}")
    print(f"   Mean EV: ${metrics['ev_distribution']['mean']:.2f}")
    print(f"   Median EV: ${metrics['ev_distribution']['median']:.2f}")
    
    # PRODUCTION acceptance criteria
    trades_ok = metrics['summary']['total_trades'] >= 10  # Low but acceptable for production
    ev_ok = metrics['ev_distribution']['mean'] > 0
    
    print(f"\n🎯 PRODUCTION Acceptance Criteria:")
    print(f"   Trades >= 10: {'PASS' if trades_ok else 'FAIL'}")
    print(f"   Mean EV > 0: {'PASS' if ev_ok else 'FAIL'}")
    
    # Calculate trade reduction from RESEARCH
    research_trades = 3873  # From RESEARCH profile
    current_trades = metrics['summary']['total_trades']
    reduction_percentage = (research_trades - current_trades) / research_trades * 100
    
    print(f"   Trade Reduction: {reduction_percentage:.1f}% (from RESEARCH)")
    print(f"   High Selectivity: {'PASS' if reduction_percentage >= 80 else 'FAIL'}")
    
    if trades_ok and ev_ok and reduction_percentage >= 80:
        print(f"\n✅ PRODUCTION profile PASSED acceptance criteria!")
        print(f"   System demonstrates high selectivity with capital protection")
        
    else:
        print(f"\n❌ PRODUCTION profile FAILED acceptance criteria!")
        return None
    
    # Generate report
    report_file = backtest.generate_report()
    print(f"   Report: {report_file}")
    
    return metrics

if __name__ == "__main__":
    run_production_backtest()
