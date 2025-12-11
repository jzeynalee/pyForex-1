# trading/__init__.py
"""
Trading module with risk management integration.

Includes:
- EnhancedDecisionEngine: ML-powered trade decisions
- LiveTradingBot: Production trading bot
- PropFirmTradingBot: Prop firm challenge specialized bot
"""

from .decision_engine import (
    EnhancedDecisionEngine,
    DecisionEngineConfig,
    TradeDecision,
    Signal,
    MTFDecisionEngine,  # Backward compatibility alias
    convert_legacy_predictions
)

from .live_trading_bot import (
    LiveTradingBot,
    BotConfig,
    BotState,
    create_bot
)

from .prop_firm_bot import (
    PropFirmTradingBot,
    PropFirmBotConfig,
    PropFirmBotState,
    create_prop_firm_bot
)

__all__ = [
    # Decision Engine
    'EnhancedDecisionEngine',
    'DecisionEngineConfig',
    'TradeDecision',
    'Signal',
    'MTFDecisionEngine',
    'convert_legacy_predictions',
    # Standard Bot
    'LiveTradingBot',
    'BotConfig',
    'BotState',
    'create_bot',
    # Prop Firm Bot
    'PropFirmTradingBot',
    'PropFirmBotConfig',
    'PropFirmBotState',
    'create_prop_firm_bot'
]
