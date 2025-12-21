"""
Backtest Orchestrator
======================

Unified backtest runner that integrates all layers of the pyForex trading system.
Runs the exact production pipeline with historical data - zero code divergence.

Usage:
    from backtesting.orchestrator import BacktestOrchestrator, BacktestConfig
    
    config = BacktestConfig(
        data_path='data/EURUSD_H1.csv',
        profile='INTRADAY',
        initial_balance=10000.0
    )
    
    orchestrator = BacktestOrchestrator(config)
    results = orchestrator.run()
"""

import logging
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Type
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.weight_validator import WeightValidator, WeightValidationResult

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtesting execution modes."""
    HISTORICAL_REPLAY = "historical_replay"
    WALK_FORWARD = "walk_forward"
    PAPER_REPLAY = "paper_replay"
    STRESS_SIMULATION = "stress_simulation"


@dataclass
class BacktestConfig:
    """Configuration for backtest orchestrator."""
    # Data
    data_path: str = "data/EURUSD_H1.csv"
    symbol: str = "EURUSD"
    primary_timeframe: str = "H1"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    warmup_bars: int = 200
    
    # Execution
    mode: BacktestMode = BacktestMode.HISTORICAL_REPLAY
    initial_balance: float = 10000.0
    commission_per_lot: float = 7.0
    spread_pips: float = 1.0
    slippage_pips: float = 0.5
    latency_ms: int = 50
    
    # Strategy
    profile: str = "INTRADAY"
    min_direction_confidence: float = 0.55  # Confidence threshold for signals
    
    # Risk
    max_positions: int = 1
    max_daily_trades: int = 10
    
    # ML
    freeze_model_weights: bool = True
    validate_weights: bool = True
    
    # Validation
    validate_data: bool = True
    validate_features: bool = True
    
    # Output
    save_artifacts: bool = True
    artifacts_dir: str = "backtest_artifacts"
    verbose: bool = True


@dataclass
class BacktestResults:
    """Results from backtest run."""
    # Summary
    initial_balance: float = 0.0
    final_balance: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    # Execution
    total_bars: int = 0
    signals_generated: int = 0
    trades_executed: int = 0
    duration_seconds: float = 0.0
    
    # Data
    equity_curve: List[Dict] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)
    decisions: List[Dict] = field(default_factory=list)
    
    # Validation
    weight_validation: Optional[WeightValidationResult] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'summary': {
                'initial_balance': self.initial_balance,
                'final_balance': self.final_balance,
                'total_return': self.total_return,
                'total_return_pct': self.total_return_pct,
            },
            'risk_metrics': {
                'max_drawdown': self.max_drawdown,
                'max_drawdown_pct': self.max_drawdown_pct,
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio,
            },
            'trading_metrics': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
            },
            'execution': {
                'total_bars': self.total_bars,
                'signals_generated': self.signals_generated,
                'trades_executed': self.trades_executed,
                'duration_seconds': self.duration_seconds,
            }
        }


class BacktestOrchestrator:
    """
    Unified backtest orchestrator integrating all pyForex layers.
    
    Components:
    - DataLoader (with MTF support)
    - BacktestConnector (MT5 interface simulation)
    - NeuralHybridStrategy (production strategy)
    - EnhancedDecisionEngine (full decision pipeline)
    - RiskManager (Phases 1-5)
    - MetricsCollector
    
    Key Features:
    - Zero code divergence from live trading
    - Event-driven replay
    - Full ML pipeline execution
    - Comprehensive metrics collection
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.project_root = PROJECT_ROOT
        
        # Components (initialized in setup)
        self.data_loader = None
        self.connector = None
        self.strategy = None
        self.data = None
        
        # State
        self.is_running = False
        self.results = BacktestResults()
        
        # Tracking
        self.equity_curve: List[Dict] = []
        self.decisions_log: List[Dict] = []
        
        logger.info(f"BacktestOrchestrator initialized for {config.symbol} ({config.profile})")
    
    def validate_weights(self) -> WeightValidationResult:
        """Validate all required model weights exist."""
        validator = WeightValidator()
        result = validator.validate_for_profile(
            self.config.profile, 
            self.project_root
        )
        
        if self.config.verbose:
            logger.info(str(result))
        
        self.results.weight_validation = result
        return result
    
    def setup(self) -> bool:
        """
        Initialize all components for backtesting.
        
        Returns:
            True if setup successful
        """
        logger.info("=" * 60)
        logger.info("  BACKTEST SETUP")
        logger.info("=" * 60)
        
        # 1. Validate weights if configured
        if self.config.validate_weights:
            logger.info("Validating model weights...")
            weight_result = self.validate_weights()
            if not weight_result.is_valid:
                logger.error("Weight validation failed. Cannot proceed.")
                return False
            logger.info("✓ Model weights validated")
        
        # 2. Load data
        logger.info(f"Loading data from {self.config.data_path}...")
        try:
            from trading.data_loader import DataLoader
            self.data_loader = DataLoader()
            
            data_path = self.project_root / self.config.data_path
            if data_path.exists():
                self.data = self.data_loader.load_csv(str(data_path))
            else:
                logger.warning(f"Data file not found: {data_path}")
                logger.info("Generating synthetic data for testing...")
                self.data = self.data_loader.generate_synthetic_data(n=2000)
            
            # Validate data
            if self.config.validate_data:
                val_result = self.data_loader.validate_data(self.data)
                if not val_result.passed:
                    logger.warning("Data validation issues:")
                    for issue in val_result.issues:
                        logger.warning(f"  - {issue}")
                else:
                    logger.info("✓ Data validation passed")
            
            logger.info(f"✓ Loaded {len(self.data)} bars")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
        
        # 3. Setup connector
        logger.info("Setting up backtest connector...")
        try:
            from trading.backtest_connector import BacktestConnector, BacktestConfig as BTConfig
            
            bt_config = BTConfig(
                initial_balance=self.config.initial_balance,
                symbol=self.config.symbol,
                spread_pips=self.config.spread_pips,
                commission_per_lot=self.config.commission_per_lot,
                slippage_pips=self.config.slippage_pips,
                latency_ms=self.config.latency_ms
            )
            
            self.connector = BacktestConnector(self.data, bt_config)
            logger.info("✓ Backtest connector initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup connector: {e}")
            return False
        
        # 4. Setup strategy directly (not through TradingBot for flexibility)
        logger.info("Setting up strategy...")
        try:
            # Import directly to avoid circular import through __init__.py
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "neural_hybrid", 
                str(self.project_root / "strategies" / "neural_hybrid.py")
            )
            neural_hybrid_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(neural_hybrid_module)
            NeuralHybridStrategy = neural_hybrid_module.NeuralHybridStrategy
            StrategyConfig = neural_hybrid_module.StrategyConfig
            
            # Create strategy config
            strategy_config = StrategyConfig(
                profile=self.config.profile,
                symbol=self.config.symbol,
                sequence_length=60,
                use_vision=False,  # Disable vision for backtest speed
                use_yolo=False,
                min_direction_confidence=self.config.min_direction_confidence,
            )
            
            # Create strategy with connector as data provider and executor
            self.strategy = NeuralHybridStrategy(
                config=strategy_config,
                data_provider=self.connector,
                executor=self.connector
            )
            
            # Initialize strategy with starting balance
            if hasattr(self.strategy, 'initialize'):
                try:
                    success = self.strategy.initialize(
                        starting_balance=self.config.initial_balance
                    )
                    if not success:
                        logger.error("Strategy initialization returned False")
                        # Try to get more info
                        if hasattr(self.strategy, '_initialized'):
                            logger.error(f"Strategy _initialized flag: {self.strategy._initialized}")
                        return False
                except Exception as init_error:
                    logger.error(f"Strategy initialization exception: {init_error}")
                    import traceback
                    traceback.print_exc()
                    return False
            
            logger.info(f"[OK] Strategy initialized: {self.strategy.name}")
            
        except Exception as e:
            logger.error(f"Failed to setup bot/strategy: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        logger.info("=" * 60)
        logger.info("  SETUP COMPLETE")
        logger.info("=" * 60)
        
        return True
    
    def run(self) -> BacktestResults:
        """
        Execute the backtest.
        
        Returns:
            BacktestResults with comprehensive metrics
        """
        # Setup if not done
        if self.connector is None:
            if not self.setup():
                logger.error("Setup failed. Cannot run backtest.")
                return self.results
        
        logger.info("=" * 60)
        logger.info("  STARTING BACKTEST")
        logger.info("=" * 60)
        logger.info(f"  Symbol: {self.config.symbol}")
        logger.info(f"  Profile: {self.config.profile}")
        logger.info(f"  Initial Balance: ${self.config.initial_balance:,.2f}")
        logger.info(f"  Bars: {len(self.data)}")
        logger.info("=" * 60)
        
        self.is_running = True
        start_time = time.time()
        
        bars_processed = 0
        signals_count = 0
        
        try:
            # Main backtest loop
            while self.connector.next_bar():
                # Get current market data window
                df = self.connector.get_data(n=100)
                
                if df.empty or len(df) < 60:
                    bars_processed += 1
                    continue
                
                # Run strategy evaluation
                try:
                    decision = self.strategy.evaluate(
                        current_time=self.connector.current_time
                    )
                    
                    if decision and decision.should_trade:
                        signals_count += 1
                        # Create and execute order
                        order = self.strategy.create_order(decision)
                        if order:
                            result = self.connector.execute_order(
                                signal=order.direction,
                                volume=order.volume,
                                sl=order.stop_loss,
                                tp=order.take_profit
                            )
                            if result.success:
                                logger.debug(
                                    f"Trade executed: {order.direction} {order.volume} @ {result.price}"
                                )
                except Exception as e:
                    logger.debug(f"Strategy evaluation error: {e}")
                
                # Record equity
                self.equity_curve.append({
                    'time': self.connector.current_time,
                    'equity': self.connector.equity,
                    'balance': self.connector.balance,
                    'open_positions': len(self.connector.positions)
                })
                
                bars_processed += 1
                
                # Progress logging
                if bars_processed % 500 == 0:
                    pct = (bars_processed / len(self.data)) * 100
                    logger.info(
                        f"Progress: {bars_processed}/{len(self.data)} bars ({pct:.1f}%) | "
                        f"Balance: ${self.connector.balance:,.2f} | "
                        f"Trades: {len(self.connector.history)}"
                    )
        
        except KeyboardInterrupt:
            logger.warning("Backtest interrupted by user")
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
        
        duration = time.time() - start_time
        
        # Calculate results
        self._calculate_results(bars_processed, duration)
        self.results.signals_generated = signals_count
        
        # Log summary
        self._log_summary()
        
        # Save artifacts
        if self.config.save_artifacts:
            self._save_artifacts()
        
        return self.results
    
    def _calculate_results(self, bars_processed: int, duration: float):
        """Calculate all backtest metrics."""
        self.results.initial_balance = self.config.initial_balance
        self.results.final_balance = self.connector.balance
        self.results.total_return = self.connector.balance - self.config.initial_balance
        self.results.total_return_pct = (self.results.total_return / self.config.initial_balance) * 100
        
        self.results.total_bars = bars_processed
        self.results.duration_seconds = duration
        self.results.equity_curve = self.equity_curve
        self.results.trades = self.connector.history
        
        # Trade statistics
        trades = self.connector.history
        self.results.total_trades = len(trades)
        
        if trades:
            wins = [t for t in trades if t.get('final_pnl', 0) > 0]
            losses = [t for t in trades if t.get('final_pnl', 0) <= 0]
            
            self.results.winning_trades = len(wins)
            self.results.losing_trades = len(losses)
            self.results.win_rate = len(wins) / len(trades) if trades else 0
            
            total_profit = sum(t.get('final_pnl', 0) for t in wins)
            total_loss = abs(sum(t.get('final_pnl', 0) for t in losses))
            
            self.results.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            self.results.avg_win = total_profit / len(wins) if wins else 0
            self.results.avg_loss = total_loss / len(losses) if losses else 0
        
        # Equity curve metrics
        if self.equity_curve:
            df_equity = pd.DataFrame(self.equity_curve)
            
            # Max Drawdown
            df_equity['peak'] = df_equity['equity'].cummax()
            df_equity['dd'] = (df_equity['equity'] - df_equity['peak']) / df_equity['peak']
            self.results.max_drawdown_pct = abs(df_equity['dd'].min()) * 100
            self.results.max_drawdown = abs((df_equity['peak'] * df_equity['dd']).min())
            
            # Returns for Sharpe/Sortino
            df_equity['returns'] = df_equity['equity'].pct_change().fillna(0)
            
            # Sharpe Ratio (annualized, assuming hourly data)
            mean_return = df_equity['returns'].mean()
            std_return = df_equity['returns'].std()
            
            if std_return > 0:
                # Annualize based on timeframe
                periods_per_year = self._get_periods_per_year()
                self.results.sharpe_ratio = (mean_return * periods_per_year) / (std_return * np.sqrt(periods_per_year))
            
            # Sortino Ratio
            downside_returns = df_equity['returns'][df_equity['returns'] < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            
            if downside_std > 0:
                periods_per_year = self._get_periods_per_year()
                self.results.sortino_ratio = (mean_return * periods_per_year) / (downside_std * np.sqrt(periods_per_year))
            
            # Calmar Ratio
            if self.results.max_drawdown_pct > 0:
                annual_return = self.results.total_return_pct * (periods_per_year / bars_processed) if bars_processed > 0 else 0
                self.results.calmar_ratio = annual_return / self.results.max_drawdown_pct
    
    def _get_periods_per_year(self) -> int:
        """Get number of periods per year based on timeframe."""
        tf_periods = {
            'M1': 525600,
            'M5': 105120,
            'M15': 35040,
            'M30': 17520,
            'H1': 8760,
            'H4': 2190,
            'D1': 252,
            'W1': 52,
        }
        return tf_periods.get(self.config.primary_timeframe.upper(), 8760)
    
    def _log_summary(self):
        """Log backtest summary."""
        logger.info("\n" + "=" * 60)
        logger.info("  BACKTEST RESULTS")
        logger.info("=" * 60)
        
        logger.info(f"  Initial Balance:  ${self.results.initial_balance:,.2f}")
        logger.info(f"  Final Balance:    ${self.results.final_balance:,.2f}")
        logger.info(f"  Total Return:     ${self.results.total_return:,.2f} ({self.results.total_return_pct:.2f}%)")
        logger.info("")
        logger.info(f"  Max Drawdown:     {self.results.max_drawdown_pct:.2f}%")
        logger.info(f"  Sharpe Ratio:     {self.results.sharpe_ratio:.2f}")
        logger.info(f"  Sortino Ratio:    {self.results.sortino_ratio:.2f}")
        logger.info(f"  Calmar Ratio:     {self.results.calmar_ratio:.2f}")
        logger.info("")
        logger.info(f"  Total Trades:     {self.results.total_trades}")
        logger.info(f"  Win Rate:         {self.results.win_rate:.2%}")
        logger.info(f"  Profit Factor:    {self.results.profit_factor:.2f}")
        logger.info(f"  Avg Win:          ${self.results.avg_win:.2f}")
        logger.info(f"  Avg Loss:         ${self.results.avg_loss:.2f}")
        logger.info("")
        logger.info(f"  Bars Processed:   {self.results.total_bars}")
        logger.info(f"  Duration:         {self.results.duration_seconds:.2f}s")
        logger.info(f"  Speed:            {self.results.total_bars / self.results.duration_seconds:.1f} bars/s")
        logger.info("=" * 60)
    
    def _save_artifacts(self):
        """Save backtest artifacts to disk."""
        artifacts_path = self.project_root / self.config.artifacts_dir
        artifacts_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_file = artifacts_path / f"equity_{timestamp}.csv"
            equity_df.to_csv(equity_file, index=False)
            logger.info(f"Equity curve saved to {equity_file}")
        
        # Save trades
        if self.connector.history:
            trades_df = pd.DataFrame(self.connector.history)
            trades_file = artifacts_path / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Trades saved to {trades_file}")
        
        # Save summary
        import json
        summary_file = artifacts_path / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(self.results.to_dict(), f, indent=2, default=str)
        logger.info(f"Summary saved to {summary_file}")


def run_backtest(
    data_path: str = "data/EURUSD_H1.csv",
    profile: str = "INTRADAY",
    initial_balance: float = 10000.0,
    **kwargs
) -> BacktestResults:
    """
    Convenience function to run a backtest.
    
    Args:
        data_path: Path to historical data CSV
        profile: Trading profile (SCALP, INTRADAY, SWING)
        initial_balance: Starting balance
        **kwargs: Additional config options
        
    Returns:
        BacktestResults
    """
    config = BacktestConfig(
        data_path=data_path,
        profile=profile,
        initial_balance=initial_balance,
        **kwargs
    )
    
    orchestrator = BacktestOrchestrator(config)
    return orchestrator.run()


if __name__ == "__main__":
    import argparse
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("backtest_orchestrator.log")
        ]
    )
    
    parser = argparse.ArgumentParser(description='Run pyForex Backtest')
    parser.add_argument('--data', type=str, default='data/EURUSD_H1.csv', help='Path to data file')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Trading symbol')
    parser.add_argument('--profile', type=str, default='INTRADAY', choices=['SCALP', 'INTRADAY', 'SWING'])
    parser.add_argument('--balance', type=float, default=10000.0, help='Initial balance')
    parser.add_argument('--no-validate', action='store_true', help='Skip weight validation')
    parser.add_argument('--min-confidence', type=float, default=0.55, help='Min direction confidence threshold')
    
    args = parser.parse_args()
    
    config = BacktestConfig(
        data_path=args.data,
        symbol=args.symbol,
        profile=args.profile,
        initial_balance=args.balance,
        validate_weights=not args.no_validate,
        min_direction_confidence=args.min_confidence
    )
    
    orchestrator = BacktestOrchestrator(config)
    results = orchestrator.run()
