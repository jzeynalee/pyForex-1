"""
Phase 2: Risk Calculations

Components for calculating trade parameters:
- SL/TP Calculator: Data-driven stop-loss and take-profit levels
- Position Sizing: Risk-adjusted position sizes
- Hard Rules: Deterministic guardrails for leverage, exposure, sessions
"""

from .sl_tp_calculator import (
    SLTPConfig,
    SLTPResult,
    SLTPCalculator,
    MarketRegime,
    TradeDirection,
    PartialExitCalculator,
    calculate_sl_tp_from_predictions
)

from .position_sizing import (
    PositionSizingConfig,
    PositionSizeResult,
    PositionSizingCalculator,
    ScaledPositionCalculator,
    AccountCurrency,
    calculate_position_from_predictions
)

from .hard_rules import (
    HardRulesConfig,
    HardRulesEngine,
    TradingSession,
    RuleViolation,
    TradeGatekeeper
)

__all__ = [
    # SL/TP
    'SLTPConfig',
    'SLTPResult',
    'SLTPCalculator',
    'MarketRegime',
    'TradeDirection',
    'PartialExitCalculator',
    'calculate_sl_tp_from_predictions',
    # Position Sizing
    'PositionSizingConfig',
    'PositionSizeResult',
    'PositionSizingCalculator',
    'ScaledPositionCalculator',
    'AccountCurrency',
    'calculate_position_from_predictions',
    # Hard Rules
    'HardRulesConfig',
    'HardRulesEngine',
    'TradingSession',
    'RuleViolation',
    'TradeGatekeeper'
]
