"""
pyForex Risk Management System (v2)
===================================

A comprehensive ML-driven risk management system for forex trading.

5-Phase Architecture:
    Phase 1: Predictive Foundation (Multi-head TCN)
    Phase 2: Risk Calculations (SL/TP, Position Sizing, Hard Rules)
    Phase 3: Trade Filtering (Triple Barrier, Meta-labeling)
    Phase 4: Adaptive Exit (RL-based PPO agent)
    Phase 5: Capital Protection (Rule-based safety limits)

Quick Start:
    from risk_management import RiskManager
    
    manager = RiskManager.create_for_profile('INTRADAY', input_features=64)
    manager.train_predictive_model(features, prices)
    decision = manager.evaluate_trade_opportunity(...)
"""

__version__ = '2.0.0'

from .risk_manager import RiskManager, RiskManagerConfig, TradeDecision

# Phase 1: Predictive Foundation
from .phase1_predictive import (
    TCNConfig, TradingProfile, MultiHeadTCN, RiskPrediction,
    create_tcn_for_profile, TrainingConfig, RiskDataset, MultiHeadTCNTrainer
)

# Phase 2: Risk Calculations
from .phase2_risk_calc import (
    SLTPConfig, SLTPResult, SLTPCalculator, MarketRegime, TradeDirection,
    PositionSizingConfig, PositionSizeResult, PositionSizingCalculator,
    HardRulesConfig, HardRulesEngine, TradingSession, TradeGatekeeper
)

# Phase 3: Trade Filtering
from .phase3_filtering import (
    TripleBarrierConfig, TripleBarrierLabeler, BarrierOutcome,
    MetaLabelingConfig, MetaLabelingModel, TradeFilter
)

# Phase 4: RL Exit Optimization
from .phase4_rl_exit import (
    ExitTradingEnv, ExitEnvConfig, ExitAction, Position,
    PPOAgent, PPOConfig, ExitOptimizer,
    ExitAdvisor, train_exit_optimizer
)

# Phase 5: Capital Protection
from .phase5_capital_protection import (
    ProtectionLevel, ProtectionAction,
    ProtectionConfig, ProtectionState, TradingMetrics,
    CapitalProtector, ProtectionManager,
    TradingGuard, ProtectedTradingSession,
    integrate_with_risk_manager
)

# Utilities
from .utils import (
    calculate_atr, calculate_volatility, calculate_adx,
    RegimeDetector, normalize_features, generate_performance_report
)

__all__ = [
    # Version
    '__version__',
    
    # Core Manager
    'RiskManager', 'RiskManagerConfig', 'TradeDecision',
    
    # Phase 1: Predictive Foundation
    'TCNConfig', 'TradingProfile', 'MultiHeadTCN', 'RiskPrediction',
    'create_tcn_for_profile', 'TrainingConfig', 'RiskDataset', 'MultiHeadTCNTrainer',
    
    # Phase 2: Risk Calculations
    'SLTPConfig', 'SLTPResult', 'SLTPCalculator', 'MarketRegime', 'TradeDirection',
    'PositionSizingConfig', 'PositionSizeResult', 'PositionSizingCalculator',
    'HardRulesConfig', 'HardRulesEngine', 'TradingSession', 'TradeGatekeeper',
    
    # Phase 3: Trade Filtering
    'TripleBarrierConfig', 'TripleBarrierLabeler', 'BarrierOutcome',
    'MetaLabelingConfig', 'MetaLabelingModel', 'TradeFilter',
    
    # Phase 4: RL Exit Optimization
    'ExitTradingEnv', 'ExitEnvConfig', 'ExitAction', 'Position',
    'PPOAgent', 'PPOConfig', 'ExitOptimizer',
    'ExitAdvisor', 'train_exit_optimizer',
    
    # Phase 5: Capital Protection
    'ProtectionLevel', 'ProtectionAction',
    'ProtectionConfig', 'ProtectionState', 'TradingMetrics',
    'CapitalProtector', 'ProtectionManager',
    'TradingGuard', 'ProtectedTradingSession',
    'integrate_with_risk_manager',
    
    # Utilities
    'calculate_atr', 'calculate_volatility', 'calculate_adx',
    'RegimeDetector', 'normalize_features', 'generate_performance_report'
]
