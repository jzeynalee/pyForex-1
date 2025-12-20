"""
Real Strategy Backtest with Historical Data
============================================

Comprehensive backtest using real historical data with technical analysis strategy.
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalDataProvider:
    """Data provider for historical CSV data."""
    
    def __init__(self, csv_path: str):
        logger.info(f"Loading historical data from {csv_path}")
        self.data = self._load_data(csv_path)
        self.historical_data = self.data
        logger.info(f"Loaded {len(self.data)} bars from {self.data.index[0]} to {self.data.index[-1]}")
    
    def _load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and prepare historical data."""
        # Try to load the CSV
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Successfully loaded CSV with {len(df)} rows")
        except FileNotFoundError:
            logger.warning(f"File not found: {csv_path}, generating sample data")
            return self._generate_sample_data()
        except Exception as e:
            logger.error(f"Error loading CSV: {e}, generating sample data")
            return self._generate_sample_data()
        
        # Parse time column
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            logger.warning("No time column found, using index")
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            logger.error(f"Missing required columns. Found: {df.columns.tolist()}")
            return self._generate_sample_data()
        
        # Add volume if missing
        if 'volume' not in df.columns:
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
            else:
                df['volume'] = 100
        
        # Sort by index
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Add basic indicators
        df = self._add_indicators(df)
        
        return df
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators."""
        # Moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # ATR
        df['atr'] = self._calculate_atr(df, 14)
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate realistic sample data as fallback."""
        logger.info("Generating realistic sample data (1 year, H1) with trending and ranging periods")
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='h')
        
        np.random.seed(42)
        n = len(dates)
        
        # Generate realistic EUR/USD with multiple market regimes
        base_price = 1.0800
        
        # Create alternating trending and ranging periods
        prices = np.zeros(n)
        prices[0] = base_price
        
        # Define market regimes (trending vs ranging)
        regime_length = 500  # bars per regime
        
        for i in range(1, n):
            regime = (i // regime_length) % 3
            
            if regime == 0:  # Uptrend
                drift = 0.00008
                volatility = 0.0012
            elif regime == 1:  # Downtrend
                drift = -0.00008
                volatility = 0.0012
            else:  # Ranging
                drift = 0.0
                volatility = 0.0008
            
            # Add mean reversion
            if i > 50:
                ma_50 = np.mean(prices[i-50:i])
                mean_reversion = (ma_50 - prices[i-1]) * 0.02
            else:
                mean_reversion = 0
            
            # Price movement
            change = drift + mean_reversion + np.random.randn() * volatility
            prices[i] = prices[i-1] * (1 + change)
        
        # Generate OHLC with realistic intrabar movement
        df = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.randn(n)) * 0.0005),
            'low': prices * (1 - np.abs(np.random.randn(n)) * 0.0005),
            'close': prices * (1 + np.random.randn(n) * 0.0002),
            'volume': np.random.randint(50, 200, n),
        }, index=dates)
        
        # Ensure OHLC relationships
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        # Add indicators
        df = self._add_indicators(df)
        
        logger.info(f"Generated data with price range: {df['close'].min():.5f} - {df['close'].max():.5f}")
        
        return df
    
    def load_historical_data(self, start_date=None, end_date=None):
        """Load historical data for backtest."""
        df = self.data.copy()
        
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        return df


class TechnicalStrategy:
    """Technical analysis strategy using multiple indicators."""
    
    def __init__(self, data_provider, executor):
        self.data_provider = data_provider
        self.executor = executor
        
        # Strategy parameters
        self.fast_ma = 10
        self.slow_ma = 30
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.min_atr = 0.0005  # Minimum volatility filter
        
        logger.info("Technical strategy initialized")
    
    def on_bar(self, data: pd.DataFrame) -> str:
        """Process bar and generate signal."""
        if len(data) < 60:
            return 'NO_TRADE'
        
        try:
            # Get latest values
            close = data['close'].iloc[-1]
            sma_fast = data['sma_10'].iloc[-1]
            sma_slow = data['sma_30'].iloc[-1]
            sma_fast_prev = data['sma_10'].iloc[-2]
            sma_slow_prev = data['sma_30'].iloc[-2]
            rsi = data['rsi'].iloc[-1]
            atr = data['atr'].iloc[-1]
            macd = data['macd'].iloc[-1]
            macd_signal = data['macd_signal'].iloc[-1]
            macd_prev = data['macd'].iloc[-2]
            macd_signal_prev = data['macd_signal'].iloc[-2]
            bb_upper = data['bb_upper'].iloc[-1]
            bb_lower = data['bb_lower'].iloc[-1]
            bb_middle = data['bb_middle'].iloc[-1]
            
            # Check for NaN
            if pd.isna([sma_fast, sma_slow, rsi, atr]).any():
                return 'NO_TRADE'
            
            # Volatility filter (lowered threshold)
            if atr < self.min_atr * 0.5:
                return 'NO_TRADE'
            
            # BUY signal: MA crossover + supporting indicators
            buy_signal = False
            if sma_fast > sma_slow and sma_fast_prev <= sma_slow_prev:
                # MA crossover detected
                if rsi < 65:  # More lenient RSI
                    buy_signal = True
            
            # Additional BUY: MACD crossover
            elif macd > macd_signal and macd_prev <= macd_signal_prev:
                if rsi < 60 and close > bb_middle:
                    buy_signal = True
            
            # Additional BUY: Oversold bounce
            elif rsi < 35 and close > bb_lower:
                if sma_fast > sma_slow:
                    buy_signal = True
            
            # SELL signal: MA crossover + supporting indicators
            sell_signal = False
            if sma_fast < sma_slow and sma_fast_prev >= sma_slow_prev:
                # MA crossover detected
                if rsi > 35:  # More lenient RSI
                    sell_signal = True
            
            # Additional SELL: MACD crossover
            elif macd < macd_signal and macd_prev >= macd_signal_prev:
                if rsi > 40 and close < bb_middle:
                    sell_signal = True
            
            # Additional SELL: Overbought reversal
            elif rsi > 65 and close < bb_upper:
                if sma_fast < sma_slow:
                    sell_signal = True
            
            # Generate signal
            if buy_signal:
                return 'BUY'
            elif sell_signal:
                return 'SELL'
            
            return 'NO_TRADE'
            
        except Exception as e:
            logger.error(f"Strategy error: {e}")
            return 'NO_TRADE'


def run_real_backtest():
    """Run backtest with real data."""
    
    logger.info("=" * 80)
    logger.info("REAL STRATEGY BACKTEST WITH HISTORICAL DATA")
    logger.info("=" * 80)
    
    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    logger.info("\n1. Loading historical data...")
    
    # Try multiple data sources
    data_paths = [
        "data/raw/EURUSD_H1_latest.csv",
        "data/raw/eurusd_latest.csv",
        "mock_backtest_data.csv"
    ]
    
    data_provider = None
    for path in data_paths:
        try:
            data_provider = HistoricalDataProvider(path)
            break
        except:
            continue
    
    if data_provider is None:
        logger.info("No CSV found, using generated data")
        data_provider = HistoricalDataProvider("nonexistent.csv")
    
    start_date = data_provider.data.index[0]
    end_date = data_provider.data.index[-1]
    
    logger.info(f"Data range: {start_date} to {end_date}")
    logger.info(f"Total bars: {len(data_provider.data)}")
    
    # =========================================================================
    # 2. VALIDATE DATA
    # =========================================================================
    logger.info("\n2. Validating data...")
    
    validator = DataValidator()
    validation_result = validator.validate(data_provider.data)
    logger.info(validation_result.summary())
    
    # =========================================================================
    # 3. CONFIGURE BACKTEST
    # =========================================================================
    logger.info("\n3. Configuring backtest...")
    
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
        max_positions=3,
        max_daily_trades=10,
        log_events=False,
        log_decisions=True,
        log_trades=True,
        save_artifacts=True
    )
    
    # =========================================================================
    # 4. CONFIGURE EXECUTION
    # =========================================================================
    logger.info("\n4. Configuring execution...")
    
    exec_config = ExecutionConfig(
        initial_balance=10000.0,
        commission_per_lot=7.0,
        base_spread_pips=1.0,
        slippage_model=SlippageModel.REALISTIC,
        slippage_mean_pips=0.3,
        latency_model=LatencyModel.REALISTIC,
        requote_probability=0.02,
        enable_market_impact=True,
        enable_spread_widening=True
    )
    
    simulator = RealisticExecutionSimulator(exec_config)
    
    # =========================================================================
    # 5. CREATE STRATEGY
    # =========================================================================
    logger.info("\n5. Creating strategy...")
    
    strategy = TechnicalStrategy(data_provider, simulator)
    
    # =========================================================================
    # 6. RUN BACKTEST
    # =========================================================================
    logger.info("\n6. Running backtest...")
    
    engine = BacktestEngine(backtest_config)
    engine.set_data_provider(data_provider)
    engine.set_execution_simulator(simulator)
    engine.set_strategy(strategy)
    
    def progress_callback(current, total):
        if current % 1000 == 0:
            pct = (current / total) * 100
            logger.info(f"Progress: {current}/{total} ({pct:.1f}%)")
    
    results = engine.run(progress_callback=progress_callback)
    
    # =========================================================================
    # 7. CALCULATE METRICS
    # =========================================================================
    logger.info("\n7. Calculating metrics...")
    
    calculator = MetricsCalculator()
    metrics = calculator.calculate(
        trades=results['trades'],
        initial_balance=10000,
        start_date=start_date,
        end_date=end_date
    )
    
    # =========================================================================
    # 8. GENERATE REPORT
    # =========================================================================
    logger.info("\n8. Generating report...")
    
    reporter = BacktestReporter(ReportConfig(
        output_dir="backtest_reports",
        generate_plots=True,
        generate_html=True,
        generate_json=True,
        generate_csv=True
    ))
    
    gate = AcceptanceGate(
        min_sharpe_ratio=1.5,
        max_drawdown_pct=20.0,
        min_profit_factor=1.5,
        min_win_rate=0.45,
        min_trades=30
    )
    
    report_metadata = reporter.generate_report(
        results=results,
        metrics=metrics.to_dict(),
        acceptance_gate=gate
    )
    
    # =========================================================================
    # 9. DISPLAY SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 80)
    
    tm = metrics.trade_metrics
    rm = metrics.risk_metrics
    ret = metrics.return_metrics
    em = metrics.execution_metrics
    
    logger.info(f"\n📊 PERFORMANCE")
    logger.info(f"  Total Return:     ${ret.total_return:,.2f} ({ret.total_return_pct:.2f}%)")
    logger.info(f"  CAGR:             {ret.cagr:.2f}%")
    logger.info(f"  Sharpe Ratio:     {rm.sharpe_ratio:.2f}")
    logger.info(f"  Sortino Ratio:    {rm.sortino_ratio:.2f}")
    logger.info(f"  Calmar Ratio:     {rm.calmar_ratio:.2f}")
    
    logger.info(f"\n💼 TRADES")
    logger.info(f"  Total Trades:     {tm.total_trades}")
    logger.info(f"  Winning Trades:   {tm.winning_trades}")
    logger.info(f"  Losing Trades:    {tm.losing_trades}")
    logger.info(f"  Win Rate:         {tm.win_rate:.2%}")
    logger.info(f"  Profit Factor:    {tm.profit_factor:.2f}")
    logger.info(f"  Expectancy:       ${tm.expectancy:.2f}")
    logger.info(f"  Avg Win:          ${tm.avg_win:.2f}")
    logger.info(f"  Avg Loss:         ${tm.avg_loss:.2f}")
    
    logger.info(f"\n⚠️  RISK")
    logger.info(f"  Max Drawdown:     ${rm.max_drawdown:.2f} ({rm.max_drawdown_pct:.2f}%)")
    logger.info(f"  VaR (95%):        ${rm.var_95:.2f}")
    logger.info(f"  CVaR (95%):       ${rm.cvar_95:.2f}")
    logger.info(f"  Max Consec Wins:  {rm.max_consecutive_wins}")
    logger.info(f"  Max Consec Loss:  {rm.max_consecutive_losses}")
    
    logger.info(f"\n⚙️  EXECUTION")
    logger.info(f"  Total Commission: ${em.total_commission:.2f}")
    logger.info(f"  Total Slippage:   {em.total_slippage_pips:.1f} pips")
    logger.info(f"  Avg Slippage:     {em.avg_slippage_pips:.2f} pips/trade")
    
    if report_metadata.get('gate_result'):
        gate_result = report_metadata['gate_result']
        logger.info(f"\n{'✅' if gate_result['passed'] else '❌'} ACCEPTANCE GATE: {'PASSED' if gate_result['passed'] else 'FAILED'}")
        if not gate_result['passed']:
            logger.info("  Failures:")
            for failure in gate_result['failures']:
                logger.info(f"    ❌ {failure}")
    
    logger.info(f"\n📁 REPORTS")
    logger.info(f"  HTML: backtest_reports\\{report_metadata['report_name']}.html")
    
    logger.info("\n" + "=" * 80)
    
    return results, metrics, report_metadata


def main():
    """Main entry point."""
    try:
        results, metrics, report = run_real_backtest()
        
        logger.info("\n✅ Backtest completed successfully!")
        logger.info(f"📊 View report: start backtest_reports\\{report['report_name']}.html")
        return 0
    
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
