"""
Comprehensive Backtesting Example
==================================

Demonstrates the complete backtesting system with all features:
- Event-driven execution
- Data validation
- Realistic execution simulation
- ML model tracking
- Risk management integration
- Comprehensive reporting
- Acceptance gate validation

This example shows how to run a production-grade backtest that gives
confidence for live trading.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Backtesting components
from backtesting import (
    BacktestEngine,
    BacktestConfig,
    BacktestMode,
    DataValidator,
    RealisticExecutionSimulator,
    ExecutionConfig,
    SlippageModel,
    LatencyModel,
    MetricsCalculator,
    BacktestReporter,
    AcceptanceGate,
    ReportConfig
)

# Trading components
from trading.decision_engine import EnhancedDecisionEngine, DecisionEngineConfig
from risk_management import RiskManager, RiskManagerConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleDataProvider:
    """Simple data provider for demonstration."""
    
    def __init__(self, data: pd.DataFrame):
        self.historical_data = data
    
    def load_historical_data(self, start_date=None, end_date=None):
        """Load historical data."""
        df = self.historical_data.copy()
        
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        return df


class SimpleStrategy:
    """Simple moving average crossover strategy for demonstration."""
    
    def __init__(self, data_provider, executor):
        self.data_provider = data_provider
        self.executor = executor
        self.fast_period = 10
        self.slow_period = 30
    
    def on_bar(self, data: pd.DataFrame) -> str:
        """Process bar and return signal."""
        if len(data) < self.slow_period:
            return 'NO_TRADE'
        
        # Calculate moving averages
        fast_ma = data['close'].rolling(self.fast_period).mean().iloc[-1]
        slow_ma = data['close'].rolling(self.slow_period).mean().iloc[-1]
        
        # Previous values
        fast_ma_prev = data['close'].rolling(self.fast_period).mean().iloc[-2]
        slow_ma_prev = data['close'].rolling(self.slow_period).mean().iloc[-2]
        
        # Check for crossover
        if fast_ma > slow_ma and fast_ma_prev <= slow_ma_prev:
            return 'BUY'
        elif fast_ma < slow_ma and fast_ma_prev >= slow_ma_prev:
            return 'SELL'
        
        return 'NO_TRADE'


def generate_sample_data(
    start_date: datetime,
    end_date: datetime,
    timeframe: str = 'H1'
) -> pd.DataFrame:
    """Generate sample OHLCV data for demonstration."""
    logger.info(f"Generating sample data from {start_date} to {end_date}")
    
    # Generate date range
    if timeframe == 'H1':
        freq = 'H'
    elif timeframe == 'M15':
        freq = '15T'
    else:
        freq = 'H'
    
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Generate realistic price data (random walk with trend)
    np.random.seed(42)
    n = len(dates)
    
    # Base price
    base_price = 1.1000
    
    # Add trend and noise
    trend = np.linspace(0, 0.02, n)  # 2% upward trend
    noise = np.random.randn(n) * 0.001
    returns = trend + noise
    
    # Generate prices
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC
    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(n)) * 0.0005),
        'low': prices * (1 - np.abs(np.random.randn(n)) * 0.0005),
        'close': prices * (1 + np.random.randn(n) * 0.0002),
        'volume': np.random.randint(100, 1000, n),
    }, index=dates)
    
    # Ensure OHLC relationships
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    logger.info(f"Generated {len(df)} bars")
    return df


def run_comprehensive_backtest():
    """Run comprehensive backtest with all features."""
    
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE BACKTESTING EXAMPLE")
    logger.info("=" * 80)
    
    # =========================================================================
    # 1. GENERATE/LOAD DATA
    # =========================================================================
    logger.info("\n1. Loading historical data...")
    
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Generate sample data (in production, load from database/files)
    data = generate_sample_data(start_date, end_date, timeframe='H1')
    
    # =========================================================================
    # 2. VALIDATE DATA
    # =========================================================================
    logger.info("\n2. Validating data integrity...")
    
    data_validator = DataValidator(
        check_monotonicity=True,
        check_gaps=True,
        check_lookahead=True,
        check_prices=True,
        check_completeness=True
    )
    
    validation_result = data_validator.validate(data)
    logger.info(validation_result.summary())
    
    if not validation_result.is_valid:
        logger.error("Data validation failed!")
        for error in validation_result.errors:
            logger.error(f"  {error.severity.value}: {error.message}")
        
        if validation_result.critical_errors:
            logger.error("Critical errors detected. Aborting backtest.")
            return
    
    # =========================================================================
    # 3. CONFIGURE BACKTESTING ENGINE
    # =========================================================================
    logger.info("\n3. Configuring backtesting engine...")
    
    backtest_config = BacktestConfig(
        mode=BacktestMode.HISTORICAL_REPLAY,
        start_date=start_date,
        end_date=end_date,
        warmup_bars=100,
        initial_balance=10000.0,
        commission_per_lot=7.0,
        base_spread_pips=1.0,
        slippage_enabled=True,
        latency_enabled=True,
        max_positions=1,
        max_daily_trades=5,
        freeze_model_weights=True,
        validate_data=True,
        validate_features=False,  # Disable for simple example
        check_lookahead=True,
        log_events=True,
        log_decisions=True,
        log_trades=True,
        save_artifacts=True,
        artifacts_dir="backtest_artifacts"
    )
    
    # =========================================================================
    # 4. CONFIGURE EXECUTION SIMULATOR
    # =========================================================================
    logger.info("\n4. Configuring realistic execution simulator...")
    
    execution_config = ExecutionConfig(
        initial_balance=10000.0,
        commission_per_lot=7.0,
        base_spread_pips=1.0,
        slippage_model=SlippageModel.REALISTIC,
        slippage_mean_pips=0.3,
        slippage_std_pips=0.5,
        slippage_max_pips=5.0,
        latency_model=LatencyModel.REALISTIC,
        latency_mean_ms=50,
        latency_std_ms=30,
        requote_probability=0.02,
        partial_fill_probability=0.01,
        rejection_probability=0.005,
        enable_market_impact=True,
        enable_spread_widening=True
    )
    
    execution_simulator = RealisticExecutionSimulator(execution_config)
    
    # =========================================================================
    # 5. SETUP STRATEGY
    # =========================================================================
    logger.info("\n5. Setting up trading strategy...")
    
    data_provider = SimpleDataProvider(data)
    strategy = SimpleStrategy(data_provider, execution_simulator)
    
    # =========================================================================
    # 6. CREATE AND RUN BACKTEST ENGINE
    # =========================================================================
    logger.info("\n6. Creating backtest engine...")
    
    engine = BacktestEngine(backtest_config)
    engine.set_data_provider(data_provider)
    engine.set_execution_simulator(execution_simulator)
    engine.set_strategy(strategy)
    engine.set_validators(data_validator=data_validator)
    
    logger.info("\n7. Running backtest...")
    
    # Progress callback
    def progress_callback(current, total):
        if current % 500 == 0:
            pct = (current / total) * 100
            logger.info(f"Progress: {current}/{total} ({pct:.1f}%)")
    
    results = engine.run(progress_callback=progress_callback)
    
    # =========================================================================
    # 7. CALCULATE METRICS
    # =========================================================================
    logger.info("\n8. Calculating performance metrics...")
    
    calculator = MetricsCalculator()
    metrics = calculator.calculate(
        trades=results['trades'],
        initial_balance=backtest_config.initial_balance,
        start_date=start_date,
        end_date=end_date
    )
    
    # =========================================================================
    # 8. GENERATE REPORT
    # =========================================================================
    logger.info("\n9. Generating comprehensive report...")
    
    report_config = ReportConfig(
        output_dir="backtest_reports",
        generate_plots=True,
        generate_html=True,
        generate_json=True,
        generate_csv=True
    )
    
    reporter = BacktestReporter(report_config)
    
    # Define acceptance gate
    acceptance_gate = AcceptanceGate(
        min_sharpe_ratio=1.5,
        max_drawdown_pct=20.0,
        min_profit_factor=1.5,
        min_win_rate=0.45,
        max_walk_forward_decay_pct=25.0,
        max_risk_violations=0,
        max_execution_failures=0,
        min_trades=30
    )
    
    report_metadata = reporter.generate_report(
        results=results,
        metrics=metrics.to_dict(),
        acceptance_gate=acceptance_gate
    )
    
    # =========================================================================
    # 9. DISPLAY SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST SUMMARY")
    logger.info("=" * 80)
    
    trade_metrics = metrics.trade_metrics
    risk_metrics = metrics.risk_metrics
    return_metrics = metrics.return_metrics
    exec_metrics = metrics.execution_metrics
    
    logger.info(f"\n📊 PERFORMANCE")
    logger.info(f"  Total Return:     ${return_metrics.total_return:,.2f} ({return_metrics.total_return_pct:.2f}%)")
    logger.info(f"  CAGR:             {return_metrics.cagr:.2f}%")
    logger.info(f"  Sharpe Ratio:     {risk_metrics.sharpe_ratio:.2f}")
    logger.info(f"  Sortino Ratio:    {risk_metrics.sortino_ratio:.2f}")
    
    logger.info(f"\n💼 TRADES")
    logger.info(f"  Total Trades:     {trade_metrics.total_trades}")
    logger.info(f"  Win Rate:         {trade_metrics.win_rate:.2%}")
    logger.info(f"  Profit Factor:    {trade_metrics.profit_factor:.2f}")
    logger.info(f"  Expectancy:       ${trade_metrics.expectancy:.2f}")
    logger.info(f"  Avg Win:          ${trade_metrics.avg_win:.2f}")
    logger.info(f"  Avg Loss:         ${trade_metrics.avg_loss:.2f}")
    
    logger.info(f"\n⚠️  RISK")
    logger.info(f"  Max Drawdown:     ${risk_metrics.max_drawdown:.2f} ({risk_metrics.max_drawdown_pct:.2f}%)")
    logger.info(f"  VaR (95%):        ${risk_metrics.var_95:.2f}")
    logger.info(f"  Max Consec Wins:  {risk_metrics.max_consecutive_wins}")
    logger.info(f"  Max Consec Loss:  {risk_metrics.max_consecutive_losses}")
    
    logger.info(f"\n⚙️  EXECUTION")
    logger.info(f"  Total Commission: ${exec_metrics.total_commission:.2f}")
    logger.info(f"  Total Slippage:   {exec_metrics.total_slippage_pips:.1f} pips")
    logger.info(f"  Avg Slippage:     {exec_metrics.avg_slippage_pips:.2f} pips/trade")
    
    # Acceptance gate result
    if report_metadata.get('gate_result'):
        gate_result = report_metadata['gate_result']
        logger.info(f"\n{'✅' if gate_result['passed'] else '❌'} ACCEPTANCE GATE: {'PASSED' if gate_result['passed'] else 'FAILED'}")
        if not gate_result['passed']:
            logger.info("  Failures:")
            for failure in gate_result['failures']:
                logger.info(f"    ❌ {failure}")
    
    logger.info(f"\n📁 REPORTS")
    logger.info(f"  Output Directory: {report_metadata['output_dir']}")
    logger.info(f"  Report Name:      {report_metadata['report_name']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST COMPLETE")
    logger.info("=" * 80)
    
    return results, metrics, report_metadata


def main():
    """Main entry point."""
    try:
        results, metrics, report_metadata = run_comprehensive_backtest()
        
        logger.info("\n✅ Backtest completed successfully!")
        logger.info(f"📊 View HTML report: {report_metadata['output_dir']}/{report_metadata['report_name']}.html")
        
        return 0
    
    except Exception as e:
        logger.error(f"\n❌ Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
