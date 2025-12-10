"""
pyForex Risk Management System
==============================

A comprehensive ML-driven risk management system for forex trading.

Quick Start:
    from risk_management import RiskManager
    
    manager = RiskManager.create_for_profile('INTRADAY', input_features=64)
    manager.train_predictive_model(features, prices)
    decision = manager.evaluate_trade_opportunity(...)
"""

__version__ = '1.0.0'

from .risk_manager import RiskManager, RiskManagerConfig, TradeDecision

from .phase1_predictive import (
    TCNConfig, TradingProfile, MultiHeadTCN, RiskPrediction,
    create_tcn_for_profile, TrainingConfig, RiskDataset, MultiHeadTCNTrainer
)

from .phase2_risk_calc import (
    SLTPConfig, SLTPResult, SLTPCalculator, MarketRegime, TradeDirection,
    PositionSizingConfig, PositionSizeResult, PositionSizingCalculator,
    HardRulesConfig, HardRulesEngine, TradingSession, TradeGatekeeper
)

from .phase3_filtering import (
    TripleBarrierConfig, TripleBarrierLabeler, BarrierOutcome,
    MetaLabelingConfig, MetaLabelingModel, TradeFilter
)

from .utils import (
    calculate_atr, calculate_volatility, calculate_adx,
    RegimeDetector, normalize_features, generate_performance_report
)

__all__ = [
    '__version__', 'RiskManager', 'RiskManagerConfig', 'TradeDecision',
    'TCNConfig', 'TradingProfile', 'MultiHeadTCN', 'create_tcn_for_profile',
    'SLTPConfig', 'SLTPCalculator', 'PositionSizingCalculator',
    'HardRulesEngine', 'TradeGatekeeper', 'TripleBarrierLabeler',
    'MetaLabelingModel', 'TradeFilter', 'RegimeDetector'
]
