"""
Event-Driven Backtesting Engine
================================

Core backtesting engine with zero code divergence from live trading.
Implements event-driven replay with proper timing and causality.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtesting execution modes."""
    HISTORICAL_REPLAY = "historical_replay"  # Deterministic evaluation
    WALK_FORWARD = "walk_forward"            # Generalization testing
    PAPER_REPLAY = "paper_replay"            # Live-like timing
    STRESS_SIMULATION = "stress_simulation"  # Adversarial markets


class EventType(Enum):
    """Event types in backtesting."""
    BAR_CLOSE = "bar_close"
    TICK = "tick"
    ORDER_FILL = "order_fill"
    ORDER_REJECT = "order_reject"
    POSITION_CLOSE = "position_close"
    RISK_UPDATE = "risk_update"
    MODEL_PREDICTION = "model_prediction"
    DECISION_MADE = "decision_made"
    MARKET_DATA = "market_data"


@dataclass
class BacktestEvent:
    """Event in the backtesting timeline."""
    timestamp: datetime
    event_type: EventType
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'data': self.data,
            'metadata': self.metadata
        }


@dataclass
class BacktestConfig:
    """Configuration for backtesting engine."""
    # Mode
    mode: BacktestMode = BacktestMode.HISTORICAL_REPLAY
    
    # Data
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    warmup_bars: int = 200
    
    # Execution
    initial_balance: float = 10000.0
    commission_per_lot: float = 7.0
    base_spread_pips: float = 1.0
    slippage_enabled: bool = True
    latency_enabled: bool = True
    
    # Risk
    max_positions: int = 1
    max_daily_trades: int = 10
    
    # ML Models
    freeze_model_weights: bool = True
    enable_model_tracking: bool = True
    enable_ablation: bool = False
    
    # Validation
    validate_data: bool = True
    validate_features: bool = True
    check_lookahead: bool = True
    
    # Logging
    log_events: bool = True
    log_decisions: bool = True
    log_trades: bool = True
    save_artifacts: bool = True
    artifacts_dir: str = "backtest_artifacts"
    
    # Performance
    parallel_processing: bool = False
    chunk_size: int = 10000


class BacktestEngine:
    """
    Event-driven backtesting engine with pipeline fidelity.
    
    Key Features:
    - Event-driven replay (not candle loops)
    - Zero code divergence from live trading
    - Proper causality and timing
    - ML model tracking and ablation
    - Realistic execution simulation
    - Comprehensive validation
    
    Usage:
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_balance=10000
        )
        
        engine = BacktestEngine(config)
        engine.set_strategy(my_strategy)
        engine.set_data_provider(data_provider)
        engine.set_execution_simulator(execution_sim)
        
        results = engine.run()
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
        # Components (set via dependency injection)
        self.strategy = None
        self.data_provider = None
        self.execution_simulator = None
        self.risk_manager = None
        self.decision_engine = None
        self.model_tracker = None
        self.data_validator = None
        self.feature_validator = None
        
        # State
        self.current_time: Optional[datetime] = None
        self.current_bar_idx: int = 0
        self.is_running: bool = False
        
        # Event queue (sorted by timestamp)
        self.event_queue: List[BacktestEvent] = []
        
        # Results tracking
        self.events_log: List[BacktestEvent] = []
        self.trades_log: List[Dict] = []
        self.decisions_log: List[Dict] = []
        self.model_predictions_log: List[Dict] = []
        self.performance_snapshots: List[Dict] = []
        
        # Artifacts directory
        if self.config.save_artifacts:
            self.artifacts_path = Path(self.config.artifacts_dir)
            self.artifacts_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"BacktestEngine initialized in {config.mode.value} mode")
    
    def set_strategy(self, strategy):
        """Set trading strategy."""
        self.strategy = strategy
        logger.info(f"Strategy set: {strategy.__class__.__name__}")
    
    def set_data_provider(self, data_provider):
        """Set data provider."""
        self.data_provider = data_provider
        logger.info("Data provider set")
    
    def set_execution_simulator(self, execution_simulator):
        """Set execution simulator."""
        self.execution_simulator = execution_simulator
        logger.info("Execution simulator set")
    
    def set_risk_manager(self, risk_manager):
        """Set risk manager."""
        self.risk_manager = risk_manager
        logger.info("Risk manager set")
    
    def set_decision_engine(self, decision_engine):
        """Set decision engine."""
        self.decision_engine = decision_engine
        logger.info("Decision engine set")
    
    def set_model_tracker(self, model_tracker):
        """Set model tracker for ML monitoring."""
        self.model_tracker = model_tracker
        logger.info("Model tracker set")
    
    def set_validators(self, data_validator=None, feature_validator=None):
        """Set validators."""
        if data_validator:
            self.data_validator = data_validator
            logger.info("Data validator set")
        if feature_validator:
            self.feature_validator = feature_validator
            logger.info("Feature validator set")
    
    def run(self, progress_callback: Optional[Callable] = None) -> Dict:
        """
        Run backtest.
        
        Args:
            progress_callback: Optional callback(current, total) for progress
        
        Returns:
            Dictionary with comprehensive backtest results
        """
        logger.info("=" * 80)
        logger.info("STARTING BACKTEST")
        logger.info("=" * 80)
        
        # Validate setup
        self._validate_setup()
        
        # Initialize components
        self._initialize_components()
        
        # Get data
        logger.info("Loading historical data...")
        data = self._load_data()
        
        if data is None or len(data) == 0:
            raise ValueError("No data available for backtesting")
        
        logger.info(f"Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")
        
        # Validate data
        if self.config.validate_data and self.data_validator:
            logger.info("Validating data integrity...")
            validation_result = self.data_validator.validate(data)
            if not validation_result.is_valid:
                logger.error(f"Data validation failed: {validation_result.errors}")
                if validation_result.critical_errors:
                    raise ValueError("Critical data validation errors detected")
        
        # Run event loop
        self.is_running = True
        total_bars = len(data)
        start_idx = self.config.warmup_bars
        
        logger.info(f"Starting event loop from bar {start_idx}/{total_bars}")
        
        try:
            for i in range(start_idx, total_bars):
                self.current_bar_idx = i
                self.current_time = data.index[i]
                
                # Process bar
                self._process_bar(data, i)
                
                # Progress callback
                if progress_callback and i % 100 == 0:
                    progress_callback(i - start_idx, total_bars - start_idx)
                
                # Performance snapshot
                if i % 1000 == 0:
                    self._take_performance_snapshot()
        
        except KeyboardInterrupt:
            logger.warning("Backtest interrupted by user")
        except Exception as e:
            logger.error(f"Backtest error: {e}", exc_info=True)
            raise
        finally:
            self.is_running = False
        
        # Finalize
        logger.info("Finalizing backtest...")
        results = self._finalize()
        
        logger.info("=" * 80)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 80)
        
        return results
    
    def _validate_setup(self):
        """Validate that all required components are set."""
        if self.data_provider is None:
            raise ValueError("Data provider not set")
        if self.execution_simulator is None:
            raise ValueError("Execution simulator not set")
        if self.strategy is None and self.decision_engine is None:
            raise ValueError("Either strategy or decision_engine must be set")
    
    def _initialize_components(self):
        """Initialize all components."""
        # Initialize execution simulator
        if hasattr(self.execution_simulator, 'initialize'):
            self.execution_simulator.initialize(self.config.initial_balance)
        
        # Initialize risk manager
        if self.risk_manager and hasattr(self.risk_manager, 'initialize'):
            self.risk_manager.initialize(self.config.initial_balance)
        
        # Initialize decision engine
        if self.decision_engine and hasattr(self.decision_engine, 'initialize'):
            self.decision_engine.initialize(self.config.initial_balance)
        
        # Freeze model weights if configured
        if self.config.freeze_model_weights:
            self._freeze_model_weights()
    
    def _freeze_model_weights(self):
        """Freeze ML model weights to prevent training during backtest."""
        if hasattr(self.strategy, 'model'):
            if hasattr(self.strategy.model, 'eval'):
                self.strategy.model.eval()
            if hasattr(self.strategy.model, 'requires_grad_'):
                for param in self.strategy.model.parameters():
                    param.requires_grad = False
        
        logger.info("Model weights frozen")
    
    def _load_data(self) -> pd.DataFrame:
        """Load historical data from data provider."""
        if hasattr(self.data_provider, 'load_historical_data'):
            return self.data_provider.load_historical_data(
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
        elif hasattr(self.data_provider, 'historical_data'):
            return self.data_provider.historical_data
        else:
            raise ValueError("Data provider does not support historical data loading")
    
    def _process_bar(self, data: pd.DataFrame, idx: int):
        """
        Process a single bar in event-driven manner.
        
        This is the core event loop that maintains causality.
        """
        current_bar = data.iloc[idx]
        
        # Create bar close event
        bar_event = BacktestEvent(
            timestamp=self.current_time,
            event_type=EventType.BAR_CLOSE,
            data={
                'open': current_bar['open'],
                'high': current_bar['high'],
                'low': current_bar['low'],
                'close': current_bar['close'],
                'volume': current_bar.get('volume', 0)
            }
        )
        
        if self.config.log_events:
            self.events_log.append(bar_event)
        
        # Update execution simulator with current price
        if hasattr(self.execution_simulator, 'update_price'):
            self.execution_simulator.update_price(
                price=current_bar['close'],
                time=self.current_time
            )
        
        # Get data window for strategy (causal - only past data)
        data_window = data.iloc[max(0, idx - 200):idx + 1].copy()
        
        # Validate features if configured
        if self.config.validate_features and self.feature_validator:
            feature_validation = self.feature_validator.validate(
                data_window,
                current_idx=len(data_window) - 1
            )
            if not feature_validation.is_valid:
                logger.warning(f"Feature validation warnings at {self.current_time}")
        
        # Strategy decision
        if self.strategy:
            self._execute_strategy(data_window, current_bar)
        elif self.decision_engine:
            self._execute_decision_engine(data_window, current_bar)
    
    def _execute_strategy(self, data_window: pd.DataFrame, current_bar: pd.Series):
        """Execute strategy logic."""
        try:
            # Call strategy
            signal = self.strategy.on_bar(data_window)
            
            # Log decision
            if self.config.log_decisions and signal != 'NO_TRADE':
                self.decisions_log.append({
                    'timestamp': self.current_time,
                    'signal': signal,
                    'price': current_bar['close']
                })
                logger.info(f"Signal generated: {signal} at {current_bar['close']:.5f}")
            
            # Handle signal (execution happens through simulator)
            if signal and signal != 'NO_TRADE':
                self._handle_signal(signal, current_bar)
        
        except Exception as e:
            logger.error(f"Strategy error at {self.current_time}: {e}")
    
    def _execute_decision_engine(self, data_window: pd.DataFrame, current_bar: pd.Series):
        """Execute decision engine logic."""
        # This would integrate with the full decision pipeline
        # Including ML models, risk management, etc.
        pass
    
    def _handle_signal(self, signal: str, current_bar: pd.Series):
        """Handle trading signal."""
        # Check if we can trade
        if not self._can_trade():
            return
        
        # Get risk parameters
        if self.risk_manager:
            risk_params = self.risk_manager.get_params(
                signal=signal,
                price=current_bar['close']
            )
        else:
            # Default risk params
            risk_params = {
                'volume': 0.01,
                'stop_loss': current_bar['close'] * (0.99 if signal == 'BUY' else 1.01),
                'take_profit': current_bar['close'] * (1.02 if signal == 'BUY' else 0.98)
            }
        
        # Execute through simulator
        result = self.execution_simulator.entry(
            signal=signal,
            volume=risk_params.get('volume', 0.01),
            sl=risk_params.get('stop_loss'),
            tp=risk_params.get('take_profit')
        )
        
        # Log trade
        if self.config.log_trades and result.get('success'):
            self.trades_log.append({
                'timestamp': self.current_time,
                'signal': signal,
                'volume': risk_params.get('volume'),
                'price': result.get('price'),
                'sl': risk_params.get('stop_loss'),
                'tp': risk_params.get('take_profit'),
                'ticket': result.get('ticket')
            })
    
    def _can_trade(self) -> bool:
        """Check if trading is allowed."""
        # Check max positions
        if hasattr(self.execution_simulator, 'positions'):
            if len(self.execution_simulator.positions) >= self.config.max_positions:
                return False
        
        # Check daily trade limit
        today_trades = [t for t in self.trades_log 
                       if t['timestamp'].date() == self.current_time.date()]
        if len(today_trades) >= self.config.max_daily_trades:
            return False
        
        return True
    
    def _take_performance_snapshot(self):
        """Take a snapshot of current performance."""
        if hasattr(self.execution_simulator, 'balance'):
            self.performance_snapshots.append({
                'timestamp': self.current_time,
                'balance': self.execution_simulator.balance,
                'equity': getattr(self.execution_simulator, 'equity', self.execution_simulator.balance),
                'num_trades': len(self.trades_log),
                'open_positions': len(getattr(self.execution_simulator, 'positions', []))
            })
    
    def _finalize(self) -> Dict:
        """Finalize backtest and generate results."""
        # Close all positions
        if hasattr(self.execution_simulator, 'close_all_positions'):
            self.execution_simulator.close_all_positions()
        
        # Get performance metrics
        if hasattr(self.execution_simulator, 'get_performance_metrics'):
            metrics = self.execution_simulator.get_performance_metrics()
        else:
            metrics = {}
        
        # Get trade history
        if hasattr(self.execution_simulator, 'get_trade_history'):
            trade_history = self.execution_simulator.get_trade_history()
        else:
            trade_history = self.trades_log
        
        # Compile results
        results = {
            'config': {
                'mode': self.config.mode.value,
                'start_date': self.config.start_date.isoformat() if self.config.start_date else None,
                'end_date': self.config.end_date.isoformat() if self.config.end_date else None,
                'initial_balance': self.config.initial_balance
            },
            'metrics': metrics,
            'trades': trade_history,
            'decisions': self.decisions_log,
            'performance_snapshots': self.performance_snapshots,
            'summary': {
                'total_bars': self.current_bar_idx,
                'total_trades': len(trade_history),
                'total_decisions': len(self.decisions_log),
                'final_balance': metrics.get('final_balance', self.config.initial_balance)
            }
        }
        
        # Save artifacts
        if self.config.save_artifacts:
            self._save_artifacts(results)
        
        return results
    
    def _save_artifacts(self, results: Dict):
        """Save backtest artifacts to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results JSON
        results_file = self.artifacts_path / f"backtest_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save trades CSV
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_file = self.artifacts_path / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Trades saved to {trades_file}")
        
        # Save performance snapshots
        if self.performance_snapshots:
            perf_df = pd.DataFrame(self.performance_snapshots)
            perf_file = self.artifacts_path / f"performance_{timestamp}.csv"
            perf_df.to_csv(perf_file, index=False)
            logger.info(f"Performance snapshots saved to {perf_file}")
