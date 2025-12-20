"""
Comprehensive Backtesting System for pyForex
============================================

Pipeline-faithful, ML-aware, execution-realistic backtesting framework.

Core Principles:
1. Zero code divergence between live and backtest paths
2. Event-driven replay, not candle loops
3. ML weights frozen unless explicitly testing retraining
4. Execution realism > optimistic fills
5. Risk system tested as first-class ML component

Modules:
- engine: Core event-driven backtesting engine
- data_validator: Data ingestion validation and lookahead detection
- feature_validator: Feature engineering validation
- model_tracker: ML model execution tracking and ablation
- decision_validator: Decision-making layer validation
- risk_backtester: ML-based risk management backtesting
- execution_simulator: Realistic MT5 execution simulation
- metrics: Portfolio-level metrics and performance analysis
- walk_forward: Walk-forward and regime validation
- reporter: Comprehensive reporting and acceptance gates
"""

__version__ = '1.0.0'

from .engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestMode,
    BacktestEvent,
    EventType
)

from .data_validator import (
    DataValidator,
    DataValidationResult,
    ValidationError
)

from .execution_simulator import (
    RealisticExecutionSimulator,
    ExecutionConfig,
    SlippageModel,
    LatencyModel
)

from .metrics import (
    PerformanceMetrics,
    MetricsCalculator,
    RiskMetrics,
    TradeMetrics
)

from .reporter import (
    BacktestReporter,
    AcceptanceGate,
    ReportConfig
)

__all__ = [
    'BacktestEngine',
    'BacktestConfig',
    'BacktestMode',
    'BacktestEvent',
    'EventType',
    'DataValidator',
    'DataValidationResult',
    'ValidationError',
    'RealisticExecutionSimulator',
    'ExecutionConfig',
    'SlippageModel',
    'LatencyModel',
    'PerformanceMetrics',
    'MetricsCalculator',
    'RiskMetrics',
    'TradeMetrics',
    'BacktestReporter',
    'AcceptanceGate',
    'ReportConfig',
]
