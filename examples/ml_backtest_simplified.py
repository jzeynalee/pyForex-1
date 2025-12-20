"""
Simplified ML Strategy Backtest with TCN
========================================

Uses TCN predictions with simplified feature engineering to avoid complexity.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import torch

# Backtesting components
from backtesting import (
    BacktestEngine, BacktestConfig, BacktestMode, DataValidator,
    RealisticExecutionSimulator, ExecutionConfig, SlippageModel, LatencyModel,
    MetricsCalculator, BacktestReporter, AcceptanceGate, ReportConfig
)

# ML components
from inference.predictor import RiskAwareTCNPredictor, PredictorConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s')
logger = logging.getLogger(__name__)


class HistoricalDataProvider:
    """Data provider for historical data."""
    
    def __init__(self):
        logger.info("Generating realistic H1 data (1 year)")
        self.data = self._generate_data()
        self.historical_data = self.data
        logger.info(f"Loaded {len(self.data)} bars from {self.data.index[0]} to {self.data.index[-1]}")
    
    def _generate_data(self) -> pd.DataFrame:
        """Generate realistic EUR/USD data."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='h')
        
        np.random.seed(42)
        n = len(dates)
        base_price = 1.0800
        prices = np.zeros(n)
        prices[0] = base_price
        
        regime_length = 500
        for i in range(1, n):
            regime = (i // regime_length) % 3
            drift = 0.00008 if regime == 0 else (-0.00008 if regime == 1 else 0.0)
            volatility = 0.0012 if regime != 2 else 0.0008
            
            if i > 50:
                ma_50 = np.mean(prices[i-50:i])
                mean_reversion = (ma_50 - prices[i-1]) * 0.02
            else:
                mean_reversion = 0
            
            change = drift + mean_reversion + np.random.randn() * volatility
            prices[i] = prices[i-1] * (1 + change)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices * (1 + np.abs(np.random.randn(n)) * 0.0005),
            'low': prices * (1 - np.abs(np.random.randn(n)) * 0.0005),
            'close': prices * (1 + np.random.randn(n) * 0.0002),
            'volume': np.random.randint(50, 200, n),
        }, index=dates)
        
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    def load_historical_data(self, start_date=None, end_date=None):
        df = self.data.copy()
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        return df


class SimplifiedMLStrategy:
    """ML strategy with simplified feature engineering."""
    
    def __init__(self, data_provider, executor, profile='INTRADAY'):
        self.data_provider = data_provider
        self.executor = executor
        self.profile = profile
        self.sequence_length = 60
        self.min_confidence = 0.55
        
        # Try to initialize TCN
        self.use_ml = self._init_predictor()
        logger.info(f"ML Strategy initialized (ML enabled: {self.use_ml})")
    
    def _init_predictor(self) -> bool:
        """Initialize TCN predictor."""
        try:
            config = PredictorConfig(
                profile=self.profile,
                sequence_length=self.sequence_length,
                use_risk_heads=True,
                confidence_threshold=self.min_confidence
            )
            
            self.predictor = RiskAwareTCNPredictor(config=config)
            
            # Try to load weights
            weights_map = {
                'INTRADAY': 'models/weights/intraday_h1_best.pt',
                'SCALP': 'models/weights/scalp_m5_best.pt',
                'SWING': 'models/weights/swing_h4_best.pt'
            }
            weights_path = weights_map.get(self.profile, 'models/weights/tcn_best.pt')
            
            try:
                self.predictor.load_weights(weights_path)
                logger.info(f"✅ Loaded TCN weights from {weights_path}")
                return True
            except Exception as e:
                logger.warning(f"⚠️  Could not load weights: {e}. Using untrained model.")
                return True  # Still use ML, just untrained
                
        except Exception as e:
            logger.error(f"❌ Could not initialize predictor: {e}")
            return False
    
    def on_bar(self, data: pd.DataFrame) -> str:
        """Generate signal."""
        if len(data) < self.sequence_length:
            return 'NO_TRADE'
        
        try:
            if self.use_ml:
                return self._ml_signal(data)
            else:
                return self._technical_signal(data)
        except Exception as e:
            logger.error(f"Strategy error: {e}")
            return 'NO_TRADE'
    
    def _ml_signal(self, data: pd.DataFrame) -> str:
        """ML-based signal."""
        try:
            features = self._prepare_simple_features(data)
            if features is None or np.isnan(features).any():
                return 'NO_TRADE'
            
            prediction = self.predictor.predict(features)
            
            if prediction.confidence < self.min_confidence:
                return 'NO_TRADE'
            
            if prediction.volatility < 0.0001:
                return 'NO_TRADE'
            
            if prediction.signal_name == 'BULL':
                return 'BUY'
            elif prediction.signal_name == 'BEAR':
                return 'SELL'
            
            return 'NO_TRADE'
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return 'NO_TRADE'
    
    def _prepare_simple_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare simplified features for ML model."""
        try:
            window = data.tail(self.sequence_length).copy()
            
            # Calculate basic technical indicators
            close = window['close'].values
            high = window['high'].values
            low = window['low'].values
            volume = window['volume'].values
            
            # Returns
            returns = np.diff(close, prepend=close[0]) / close
            
            # Moving averages
            sma_10 = pd.Series(close).rolling(10, min_periods=1).mean().values
            sma_20 = pd.Series(close).rolling(20, min_periods=1).mean().values
            sma_50 = pd.Series(close).rolling(50, min_periods=1).mean().values
            
            # RSI
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            rsi = (100 - (100 / (1 + rs))).values
            
            # ATR
            tr1 = high - low
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[tr1[0]], np.maximum(tr1[1:], np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
            
            # Bollinger Bands
            bb_middle = sma_20
            bb_std = pd.Series(close).rolling(20, min_periods=1).std().values
            bb_upper = bb_middle + 2 * bb_std
            bb_lower = bb_middle - 2 * bb_std
            
            # MACD
            ema_12 = pd.Series(close).ewm(span=12, min_periods=1).mean().values
            ema_26 = pd.Series(close).ewm(span=26, min_periods=1).mean().values
            macd = ema_12 - ema_26
            macd_signal = pd.Series(macd).ewm(span=9, min_periods=1).mean().values
            
            # Normalize prices
            close_norm = (close - close.mean()) / (close.std() + 1e-10)
            high_norm = (high - high.mean()) / (high.std() + 1e-10)
            low_norm = (low - low.mean()) / (low.std() + 1e-10)
            volume_norm = (volume - volume.mean()) / (volume.std() + 1e-10)
            
            # Stack features (seq_len, n_features)
            features = np.column_stack([
                close_norm, high_norm, low_norm, volume_norm,
                returns, sma_10, sma_20, sma_50, rsi, atr,
                bb_upper, bb_middle, bb_lower, macd, macd_signal
            ])
            
            # Fill any remaining NaNs
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            return None
    
    def _technical_signal(self, data: pd.DataFrame) -> str:
        """Fallback technical signal."""
        try:
            close = data['close'].values
            sma_fast = pd.Series(close).rolling(10).mean().iloc[-1]
            sma_slow = pd.Series(close).rolling(30).mean().iloc[-1]
            sma_fast_prev = pd.Series(close).rolling(10).mean().iloc[-2]
            sma_slow_prev = pd.Series(close).rolling(30).mean().iloc[-2]
            
            delta = pd.Series(close).diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi = (100 - (100 / (1 + gain / (loss + 1e-10)))).iloc[-1]
            
            if pd.isna([sma_fast, sma_slow, rsi]).any():
                return 'NO_TRADE'
            
            if sma_fast > sma_slow and sma_fast_prev <= sma_slow_prev and rsi < 70:
                return 'BUY'
            if sma_fast < sma_slow and sma_fast_prev >= sma_slow_prev and rsi > 30:
                return 'SELL'
            
            return 'NO_TRADE'
        except:
            return 'NO_TRADE'


def run_ml_backtest():
    """Run ML backtest."""
    logger.info("=" * 80)
    logger.info("SIMPLIFIED ML STRATEGY BACKTEST (TCN)")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\n1. Loading data...")
    data_provider = HistoricalDataProvider()
    start_date = data_provider.data.index[0]
    end_date = data_provider.data.index[-1]
    
    # Validate
    logger.info("\n2. Validating data...")
    validator = DataValidator()
    validation_result = validator.validate(data_provider.data)
    logger.info(validation_result.summary())
    
    # Configure backtest
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
    
    # Configure execution
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
    
    # Create strategy
    logger.info("\n5. Creating ML strategy...")
    strategy = SimplifiedMLStrategy(data_provider, simulator, profile='INTRADAY')
    
    # Run backtest
    logger.info("\n6. Running backtest...")
    logger.info(f"Using ML: {strategy.use_ml}")
    
    engine = BacktestEngine(backtest_config)
    engine.set_data_provider(data_provider)
    engine.set_execution_simulator(simulator)
    engine.set_strategy(strategy)
    
    def progress_callback(current, total):
        if current % 1000 == 0:
            logger.info(f"Progress: {current}/{total} ({(current/total)*100:.1f}%)")
    
    results = engine.run(progress_callback=progress_callback)
    
    # Calculate metrics
    logger.info("\n7. Calculating metrics...")
    calculator = MetricsCalculator()
    metrics = calculator.calculate(trades=results['trades'], initial_balance=10000, start_date=start_date, end_date=end_date)
    
    # Generate report
    logger.info("\n8. Generating report...")
    reporter = BacktestReporter(ReportConfig(output_dir="backtest_reports", generate_plots=True, generate_html=True, generate_json=True, generate_csv=True))
    gate = AcceptanceGate(min_sharpe_ratio=1.5, max_drawdown_pct=20.0, min_profit_factor=1.5, min_win_rate=0.45, min_trades=30)
    report_metadata = reporter.generate_report(results=results, metrics=metrics.to_dict(), acceptance_gate=gate)
    
    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST RESULTS - ML STRATEGY (TCN)")
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
    
    logger.info(f"\n💼 TRADES")
    logger.info(f"  Total Trades:     {tm.total_trades}")
    logger.info(f"  Win Rate:         {tm.win_rate:.2%}")
    logger.info(f"  Profit Factor:    {tm.profit_factor:.2f}")
    logger.info(f"  Expectancy:       ${tm.expectancy:.2f}")
    
    logger.info(f"\n⚠️  RISK")
    logger.info(f"  Max Drawdown:     ${rm.max_drawdown:.2f} ({rm.max_drawdown_pct:.2f}%)")
    logger.info(f"  Max Consec Loss:  {rm.max_consecutive_losses}")
    
    logger.info(f"\n⚙️  EXECUTION")
    logger.info(f"  Total Commission: ${em.total_commission:.2f}")
    logger.info(f"  Avg Slippage:     {em.avg_slippage_pips:.2f} pips/trade")
    
    if report_metadata.get('gate_result'):
        gate_result = report_metadata['gate_result']
        logger.info(f"\n{'✅' if gate_result['passed'] else '❌'} ACCEPTANCE GATE: {'PASSED' if gate_result['passed'] else 'FAILED'}")
    
    logger.info(f"\n📁 HTML: backtest_reports\\{report_metadata['report_name']}.html")
    logger.info("\n" + "=" * 80)
    
    return results, metrics, report_metadata


if __name__ == "__main__":
    try:
        results, metrics, report = run_ml_backtest()
        logger.info("\n✅ ML Backtest completed!")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
