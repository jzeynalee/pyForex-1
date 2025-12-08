# trading/style_config.py
"""
Trading Style Configurations for Multi-Style Orchestrator.

Defines parameters for:
- Scalping (M5/M15)
- Intraday (M30/H1)
- Swing (H4/D1)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


class TradingStyle(Enum):
    """Trading style enumeration."""
    SCALP = "scalp"
    INTRADAY = "intraday"
    SWING = "swing"


@dataclass
class StyleConfig:
    """Configuration for a single trading style."""
    
    # Identity
    name: str
    style: TradingStyle
    enabled: bool = True
    
    # Timeframes
    primary_timeframe: str = "H1"
    secondary_timeframes: List[str] = field(default_factory=list)
    
    # Signal Thresholds
    min_confidence: float = 0.65
    min_trend_alignment: float = 0.55
    
    # Risk Parameters
    risk_per_trade_pct: float = 0.01  # 1% per trade
    max_positions: int = 2
    max_daily_trades: int = 5
    risk_allocation_pct: float = 0.33  # Share of total risk budget
    
    # Position Sizing
    sl_atr_multiplier: float = 1.5
    tp_atr_multiplier: float = 3.0
    min_risk_reward: float = 1.5
    
    # Timing
    check_interval_seconds: float = 60.0
    min_bars_required: int = 100
    
    # Trade Management
    use_trailing_stop: bool = False
    trailing_stop_atr: float = 1.0
    break_even_at_rr: float = 1.0  # Move SL to BE at 1:1 RR
    partial_close_at_rr: float = 0.0  # 0 = disabled
    partial_close_pct: float = 0.5
    
    # Session Filters (hours in UTC)
    allowed_sessions: List[Tuple[int, int]] = field(default_factory=list)
    avoid_news_minutes: int = 30
    
    # Style-specific
    require_trend_confirmation: bool = True
    allow_counter_trend: bool = False
    counter_trend_confidence: float = 0.85


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

SCALP_CONFIG = StyleConfig(
    name="Scalper",
    style=TradingStyle.SCALP,
    enabled=True,
    
    # Fast timeframes
    primary_timeframe="M15",
    secondary_timeframes=["M5", "H1"],
    
    # Higher thresholds for quick trades
    min_confidence=0.75,
    min_trend_alignment=0.60,
    
    # Tighter risk
    risk_per_trade_pct=0.005,  # 0.5% per scalp
    max_positions=3,
    max_daily_trades=10,
    risk_allocation_pct=0.30,
    
    # Tight SL/TP
    sl_atr_multiplier=1.0,
    tp_atr_multiplier=1.5,
    min_risk_reward=1.2,
    
    # Frequent checks
    check_interval_seconds=30.0,
    min_bars_required=60,
    
    # Quick management
    use_trailing_stop=True,
    trailing_stop_atr=0.5,
    break_even_at_rr=0.8,
    partial_close_at_rr=1.0,
    partial_close_pct=0.5,
    
    # Session: London + NY overlap (best liquidity)
    allowed_sessions=[(7, 11), (13, 17)],  # UTC
    avoid_news_minutes=15,
    
    # No counter-trend scalps
    require_trend_confirmation=True,
    allow_counter_trend=False,
)


INTRADAY_CONFIG = StyleConfig(
    name="Intraday",
    style=TradingStyle.INTRADAY,
    enabled=True,
    
    # Medium timeframes
    primary_timeframe="H1",
    secondary_timeframes=["M15", "H4"],
    
    # Standard thresholds
    min_confidence=0.65,
    min_trend_alignment=0.55,
    
    # Standard risk
    risk_per_trade_pct=0.01,  # 1% per trade
    max_positions=2,
    max_daily_trades=5,
    risk_allocation_pct=0.40,
    
    # Standard SL/TP
    sl_atr_multiplier=1.5,
    tp_atr_multiplier=3.0,
    min_risk_reward=1.5,
    
    # Hourly checks
    check_interval_seconds=60.0,
    min_bars_required=100,
    
    # Moderate management
    use_trailing_stop=True,
    trailing_stop_atr=1.0,
    break_even_at_rr=1.0,
    partial_close_at_rr=1.5,
    partial_close_pct=0.5,
    
    # Active sessions
    allowed_sessions=[(6, 20)],  # UTC - Main sessions
    avoid_news_minutes=30,
    
    # Prefer trend, allow some counter
    require_trend_confirmation=True,
    allow_counter_trend=True,
    counter_trend_confidence=0.80,
)


SWING_CONFIG = StyleConfig(
    name="Swing",
    style=TradingStyle.SWING,
    enabled=True,
    
    # Slow timeframes
    primary_timeframe="H4",
    secondary_timeframes=["H1", "D1"],
    
    # Lower thresholds (bigger picture)
    min_confidence=0.60,
    min_trend_alignment=0.50,
    
    # Larger risk per trade, fewer trades
    risk_per_trade_pct=0.015,  # 1.5% per swing
    max_positions=2,
    max_daily_trades=2,
    risk_allocation_pct=0.30,
    
    # Wider SL/TP
    sl_atr_multiplier=2.0,
    tp_atr_multiplier=4.0,
    min_risk_reward=2.0,
    
    # Less frequent checks
    check_interval_seconds=300.0,  # 5 minutes
    min_bars_required=100,
    
    # Relaxed management
    use_trailing_stop=True,
    trailing_stop_atr=1.5,
    break_even_at_rr=1.5,
    partial_close_at_rr=2.0,
    partial_close_pct=0.5,
    
    # All sessions (swing doesn't care)
    allowed_sessions=[],  # Empty = all
    avoid_news_minutes=60,
    
    # Strong trend focus
    require_trend_confirmation=True,
    allow_counter_trend=True,
    counter_trend_confidence=0.85,
)


# Default configuration set
DEFAULT_STYLE_CONFIGS = {
    TradingStyle.SCALP: SCALP_CONFIG,
    TradingStyle.INTRADAY: INTRADAY_CONFIG,
    TradingStyle.SWING: SWING_CONFIG,
}


@dataclass
class OrchestratorConfig:
    """Master configuration for Multi-Style Orchestrator."""
    
    # Symbol
    symbol: str = "EURUSD"
    
    # Global Risk Limits
    max_total_positions: int = 5
    max_daily_loss_pct: float = 0.03  # 3% daily loss limit
    max_drawdown_pct: float = 0.10    # 10% max drawdown
    max_correlation_exposure: float = 0.6  # Max correlated position size
    
    # Style Configurations
    styles: Dict[TradingStyle, StyleConfig] = field(default_factory=lambda: DEFAULT_STYLE_CONFIGS.copy())
    
    # Coordination
    prevent_opposing_positions: bool = True  # No simultaneous BUY scalp + SELL swing
    aggregate_trend_weight: bool = True      # Use combined trend from all TFs
    
    # Execution
    use_mock_connector: bool = False
    magic_number_base: int = 100000  # Each style gets unique magic
    
    # Monitoring
    log_level: str = "INFO"
    heartbeat_seconds: float = 60.0
    
    def get_style_config(self, style: TradingStyle) -> StyleConfig:
        """Get configuration for a specific style."""
        return self.styles.get(style, INTRADAY_CONFIG)
    
    def get_enabled_styles(self) -> List[TradingStyle]:
        """Get list of enabled trading styles."""
        return [style for style, config in self.styles.items() if config.enabled]
    
    def get_magic_number(self, style: TradingStyle) -> int:
        """Get unique magic number for a style."""
        offsets = {
            TradingStyle.SCALP: 1,
            TradingStyle.INTRADAY: 2,
            TradingStyle.SWING: 3,
        }
        return self.magic_number_base + offsets.get(style, 0)


# =============================================================================
# TIMEFRAME UTILITIES
# =============================================================================

TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
}


def get_timeframe_minutes(tf: str) -> int:
    """Convert timeframe string to minutes."""
    return TIMEFRAME_MINUTES.get(tf.upper(), 60)


def get_all_required_timeframes(config: OrchestratorConfig) -> List[str]:
    """Get all unique timeframes needed by enabled styles."""
    timeframes = set()
    
    for style in config.get_enabled_styles():
        style_config = config.get_style_config(style)
        timeframes.add(style_config.primary_timeframe)
        timeframes.update(style_config.secondary_timeframes)
    
    # Sort by granularity (smallest first)
    return sorted(timeframes, key=lambda tf: get_timeframe_minutes(tf))


def style_for_timeframe(tf: str) -> TradingStyle:
    """Suggest appropriate style for a timeframe."""
    minutes = get_timeframe_minutes(tf)
    
    if minutes <= 15:
        return TradingStyle.SCALP
    elif minutes <= 60:
        return TradingStyle.INTRADAY
    else:
        return TradingStyle.SWING
