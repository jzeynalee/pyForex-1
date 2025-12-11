# config/prop_firm_config.py
"""
Prop Firm Challenge Configuration

Pre-configured settings for major prop trading firms:
- FTMO
- MyForexFunds
- The5ers
- Funded Next
- True Forex Funds
- TopStep Forex

Each configuration enforces:
- Daily loss limits
- Maximum drawdown
- Profit targets
- Trading restrictions
- Risk per trade limits

Usage:
    from config.prop_firm_config import get_prop_firm_config, PropFirmMode
    
    config = get_prop_firm_config('FTMO', account_size=100000)
    bot = create_bot_for_prop_firm(config)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import time, datetime
import logging

logger = logging.getLogger(__name__)


class PropFirm(Enum):
    """Supported prop trading firms."""
    FTMO = "ftmo"
    MYFOREXFUNDS = "myforexfunds"
    THE5ERS = "the5ers"
    FUNDED_NEXT = "funded_next"
    TRUE_FOREX_FUNDS = "true_forex_funds"
    TOPSTEP = "topstep"
    CUSTOM = "custom"


class ChallengePhase(Enum):
    """Challenge phases."""
    EVALUATION = "evaluation"      # Initial challenge
    VERIFICATION = "verification"  # Second phase (some firms)
    FUNDED = "funded"              # Live funded account


@dataclass
class PropFirmRules:
    """Trading rules for a prop firm."""
    # Loss Limits (as percentage of account)
    max_daily_loss_pct: float          # e.g., 5.0 = 5%
    max_total_drawdown_pct: float      # e.g., 10.0 = 10%
    
    # Profit Target (for challenge phase)
    profit_target_pct: float           # e.g., 10.0 = 10%
    
    # Time Limits
    min_trading_days: int = 0          # Minimum days must trade
    max_trading_days: int = 0          # 0 = unlimited
    
    # Trading Restrictions
    weekend_holding_allowed: bool = True
    news_trading_allowed: bool = True
    news_blackout_minutes: int = 0     # Minutes before/after news
    
    # Position Limits
    max_lots_per_trade: float = 0      # 0 = unlimited
    max_open_positions: int = 0        # 0 = unlimited
    
    # Instrument Restrictions
    allowed_pairs: List[str] = field(default_factory=list)  # Empty = all allowed
    forbidden_pairs: List[str] = field(default_factory=list)
    
    # Trading Hours (UTC)
    trading_start: Optional[time] = None
    trading_end: Optional[time] = None
    
    # Scaling Rules
    profit_split_pct: float = 80.0     # Your share of profits
    scaling_available: bool = True


@dataclass
class PropFirmConfig:
    """Complete configuration for prop firm trading."""
    firm: PropFirm
    firm_name: str
    phase: ChallengePhase
    account_size: float
    rules: PropFirmRules
    
    # Safety Margins (trade below limits)
    daily_loss_buffer_pct: float = 1.0    # Stay 1% below daily limit
    drawdown_buffer_pct: float = 2.0      # Stay 2% below max DD
    
    # Conservative Mode Settings
    max_risk_per_trade_pct: float = 0.5   # Risk per trade
    max_trades_per_day: int = 5
    min_risk_reward: float = 1.5
    
    # Trailing Drawdown (some firms use this)
    trailing_drawdown: bool = False
    trailing_dd_lock_profit_pct: float = 0  # Lock DD after X% profit
    
    @property
    def effective_daily_loss_pct(self) -> float:
        """Daily loss limit with safety buffer."""
        return self.rules.max_daily_loss_pct - self.daily_loss_buffer_pct
    
    @property
    def effective_max_drawdown_pct(self) -> float:
        """Max drawdown with safety buffer."""
        return self.rules.max_total_drawdown_pct - self.drawdown_buffer_pct
    
    @property
    def daily_loss_amount(self) -> float:
        """Maximum daily loss in currency."""
        return self.account_size * (self.effective_daily_loss_pct / 100)
    
    @property
    def max_drawdown_amount(self) -> float:
        """Maximum drawdown in currency."""
        return self.account_size * (self.effective_max_drawdown_pct / 100)
    
    @property
    def profit_target_amount(self) -> float:
        """Profit target in currency."""
        return self.account_size * (self.rules.profit_target_pct / 100)
    
    def to_protection_config(self) -> Dict:
        """Convert to Phase 5 ProtectionConfig parameters."""
        return {
            'max_daily_loss_pct': self.effective_daily_loss_pct,
            'max_weekly_loss_pct': self.effective_max_drawdown_pct * 0.7,  # Conservative
            'max_monthly_loss_pct': self.effective_max_drawdown_pct,
            'max_drawdown_pct': self.effective_max_drawdown_pct,
            'max_consecutive_losses': 4,  # Conservative
            'losing_streak_cooldown_minutes': 60,
        }
    
    def to_bot_config(self) -> Dict:
        """Convert to BotConfig parameters."""
        return {
            'base_risk_percent': self.max_risk_per_trade_pct,
            'max_daily_loss_percent': self.effective_daily_loss_pct,
            'max_weekly_loss_percent': self.effective_max_drawdown_pct * 0.7,
            'max_drawdown_percent': self.effective_max_drawdown_pct,
            'max_open_trades': min(self.rules.max_open_positions or 3, 3),
            'max_consecutive_losses': 4,
            'cooldown_minutes': 60,
        }
    
    def to_decision_engine_config(self) -> Dict:
        """Convert to DecisionEngineConfig parameters."""
        return {
            'base_risk_percent': self.max_risk_per_trade_pct,
            'min_risk_reward': self.min_risk_reward,
            'enable_capital_protection': True,
            'max_daily_loss_pct': self.effective_daily_loss_pct,
            'max_weekly_loss_pct': self.effective_max_drawdown_pct * 0.7,
            'max_drawdown_pct': self.effective_max_drawdown_pct,
        }


# =============================================================================
# PROP FIRM PRESETS
# =============================================================================

FTMO_RULES = {
    ChallengePhase.EVALUATION: PropFirmRules(
        max_daily_loss_pct=5.0,
        max_total_drawdown_pct=10.0,
        profit_target_pct=10.0,
        min_trading_days=4,
        max_trading_days=30,
        weekend_holding_allowed=True,
        news_trading_allowed=True,  # Allowed but risky
        profit_split_pct=0,  # No profit during challenge
    ),
    ChallengePhase.VERIFICATION: PropFirmRules(
        max_daily_loss_pct=5.0,
        max_total_drawdown_pct=10.0,
        profit_target_pct=5.0,
        min_trading_days=4,
        max_trading_days=60,
        weekend_holding_allowed=True,
        news_trading_allowed=True,
        profit_split_pct=0,
    ),
    ChallengePhase.FUNDED: PropFirmRules(
        max_daily_loss_pct=5.0,
        max_total_drawdown_pct=10.0,
        profit_target_pct=0,  # No target, just trade
        weekend_holding_allowed=True,
        news_trading_allowed=True,
        profit_split_pct=80.0,  # Up to 90% with scaling
        scaling_available=True,
    ),
}

MYFOREXFUNDS_RULES = {
    ChallengePhase.EVALUATION: PropFirmRules(
        max_daily_loss_pct=5.0,
        max_total_drawdown_pct=12.0,
        profit_target_pct=8.0,
        min_trading_days=5,
        max_trading_days=30,
        weekend_holding_allowed=True,
        news_trading_allowed=False,  # 2 min before/after
        news_blackout_minutes=2,
        profit_split_pct=0,
    ),
    ChallengePhase.FUNDED: PropFirmRules(
        max_daily_loss_pct=5.0,
        max_total_drawdown_pct=12.0,
        profit_target_pct=0,
        weekend_holding_allowed=True,
        news_trading_allowed=False,
        news_blackout_minutes=2,
        profit_split_pct=75.0,  # Up to 85%
        scaling_available=True,
    ),
}

THE5ERS_RULES = {
    ChallengePhase.EVALUATION: PropFirmRules(
        max_daily_loss_pct=3.0,  # Stricter!
        max_total_drawdown_pct=6.0,  # Stricter!
        profit_target_pct=6.0,
        min_trading_days=3,
        max_trading_days=0,  # Unlimited
        weekend_holding_allowed=False,  # Must close Friday
        news_trading_allowed=True,
        profit_split_pct=0,
    ),
    ChallengePhase.FUNDED: PropFirmRules(
        max_daily_loss_pct=3.0,
        max_total_drawdown_pct=6.0,
        profit_target_pct=0,
        weekend_holding_allowed=False,
        news_trading_allowed=True,
        profit_split_pct=50.0,  # Starts at 50%, scales to 100%
        scaling_available=True,
    ),
}

FUNDED_NEXT_RULES = {
    ChallengePhase.EVALUATION: PropFirmRules(
        max_daily_loss_pct=5.0,
        max_total_drawdown_pct=10.0,
        profit_target_pct=10.0,
        min_trading_days=0,  # No minimum
        max_trading_days=0,  # Unlimited
        weekend_holding_allowed=True,
        news_trading_allowed=True,
        profit_split_pct=0,
    ),
    ChallengePhase.FUNDED: PropFirmRules(
        max_daily_loss_pct=5.0,
        max_total_drawdown_pct=10.0,
        profit_target_pct=0,
        weekend_holding_allowed=True,
        news_trading_allowed=True,
        profit_split_pct=80.0,  # Up to 90%
        scaling_available=True,
    ),
}

TRUE_FOREX_FUNDS_RULES = {
    ChallengePhase.EVALUATION: PropFirmRules(
        max_daily_loss_pct=5.0,
        max_total_drawdown_pct=10.0,
        profit_target_pct=8.0,
        min_trading_days=5,
        max_trading_days=30,
        weekend_holding_allowed=True,
        news_trading_allowed=True,
        profit_split_pct=0,
    ),
    ChallengePhase.FUNDED: PropFirmRules(
        max_daily_loss_pct=5.0,
        max_total_drawdown_pct=10.0,
        profit_target_pct=0,
        weekend_holding_allowed=True,
        news_trading_allowed=True,
        profit_split_pct=80.0,
        scaling_available=True,
    ),
}

PROP_FIRM_PRESETS = {
    PropFirm.FTMO: FTMO_RULES,
    PropFirm.MYFOREXFUNDS: MYFOREXFUNDS_RULES,
    PropFirm.THE5ERS: THE5ERS_RULES,
    PropFirm.FUNDED_NEXT: FUNDED_NEXT_RULES,
    PropFirm.TRUE_FOREX_FUNDS: TRUE_FOREX_FUNDS_RULES,
}


def get_prop_firm_config(
    firm: str,
    account_size: float,
    phase: str = 'evaluation',
    conservative: bool = True
) -> PropFirmConfig:
    """
    Get configuration for a specific prop firm.
    
    Args:
        firm: Firm name ('FTMO', 'MyForexFunds', etc.)
        account_size: Account size in USD
        phase: 'evaluation', 'verification', or 'funded'
        conservative: Use extra safety margins
    
    Returns:
        PropFirmConfig ready to use
    
    Example:
        config = get_prop_firm_config('FTMO', 100000, 'evaluation')
    """
    # Parse firm
    firm_upper = firm.upper().replace(' ', '_').replace('-', '_')
    try:
        prop_firm = PropFirm[firm_upper]
    except KeyError:
        # Try partial match
        for pf in PropFirm:
            if firm_upper in pf.name:
                prop_firm = pf
                break
        else:
            raise ValueError(f"Unknown prop firm: {firm}. Available: {[f.name for f in PropFirm]}")
    
    # Parse phase
    phase_lower = phase.lower()
    if phase_lower in ['eval', 'evaluation', 'challenge']:
        challenge_phase = ChallengePhase.EVALUATION
    elif phase_lower in ['verify', 'verification', 'phase2']:
        challenge_phase = ChallengePhase.VERIFICATION
    elif phase_lower in ['funded', 'live']:
        challenge_phase = ChallengePhase.FUNDED
    else:
        challenge_phase = ChallengePhase.EVALUATION
    
    # Get rules
    if prop_firm not in PROP_FIRM_PRESETS:
        raise ValueError(f"No preset for {prop_firm.name}")
    
    firm_rules = PROP_FIRM_PRESETS[prop_firm]
    
    if challenge_phase not in firm_rules:
        # Fall back to evaluation if phase not found
        challenge_phase = ChallengePhase.EVALUATION
    
    rules = firm_rules[challenge_phase]
    
    # Create config
    config = PropFirmConfig(
        firm=prop_firm,
        firm_name=prop_firm.name.replace('_', ' ').title(),
        phase=challenge_phase,
        account_size=account_size,
        rules=rules,
        # Conservative settings
        daily_loss_buffer_pct=1.5 if conservative else 0.5,
        drawdown_buffer_pct=2.5 if conservative else 1.0,
        max_risk_per_trade_pct=0.5 if conservative else 1.0,
        max_trades_per_day=3 if conservative else 5,
        min_risk_reward=2.0 if conservative else 1.5,
    )
    
    logger.info(
        f"Prop firm config created: {config.firm_name} {challenge_phase.value} "
        f"${account_size:,.0f} | Daily Limit: {config.effective_daily_loss_pct}% "
        f"| Max DD: {config.effective_max_drawdown_pct}%"
    )
    
    return config


def create_custom_prop_config(
    account_size: float,
    max_daily_loss_pct: float = 5.0,
    max_drawdown_pct: float = 10.0,
    profit_target_pct: float = 10.0,
    **kwargs
) -> PropFirmConfig:
    """
    Create custom prop firm configuration.
    
    Useful for firms not in presets or custom rules.
    """
    rules = PropFirmRules(
        max_daily_loss_pct=max_daily_loss_pct,
        max_total_drawdown_pct=max_drawdown_pct,
        profit_target_pct=profit_target_pct,
        **kwargs
    )
    
    return PropFirmConfig(
        firm=PropFirm.CUSTOM,
        firm_name="Custom",
        phase=ChallengePhase.EVALUATION,
        account_size=account_size,
        rules=rules,
    )


# =============================================================================
# PROP FIRM MONITOR
# =============================================================================

@dataclass
class PropFirmStatus:
    """Current status relative to prop firm rules."""
    config: PropFirmConfig
    
    # Current metrics
    current_balance: float = 0.0
    starting_balance: float = 0.0
    daily_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Calculated
    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown from peak."""
        peak = max(self.starting_balance, self.current_balance)
        if peak <= 0:
            return 0.0
        return ((peak - self.current_balance) / peak) * 100
    
    @property
    def daily_loss_pct(self) -> float:
        """Today's loss as percentage."""
        if self.starting_balance <= 0:
            return 0.0
        return (abs(min(0, self.daily_pnl)) / self.starting_balance) * 100
    
    @property
    def profit_progress_pct(self) -> float:
        """Progress toward profit target."""
        if self.config.rules.profit_target_pct <= 0:
            return 100.0  # No target
        target = self.config.profit_target_amount
        return min(100.0, (self.total_pnl / target) * 100) if target > 0 else 0.0
    
    @property
    def daily_loss_remaining(self) -> float:
        """How much more can lose today."""
        limit = self.config.daily_loss_amount
        used = abs(min(0, self.daily_pnl))
        return max(0, limit - used)
    
    @property
    def drawdown_remaining(self) -> float:
        """How much more drawdown allowed."""
        limit = self.config.max_drawdown_amount
        current = (self.starting_balance - self.current_balance) if self.current_balance < self.starting_balance else 0
        return max(0, limit - current)
    
    @property
    def is_daily_limit_near(self) -> bool:
        """True if within 20% of daily limit."""
        return self.daily_loss_pct >= (self.config.effective_daily_loss_pct * 0.8)
    
    @property
    def is_drawdown_limit_near(self) -> bool:
        """True if within 20% of max drawdown."""
        return self.current_drawdown_pct >= (self.config.effective_max_drawdown_pct * 0.8)
    
    @property
    def should_stop_trading(self) -> bool:
        """True if should stop to protect account."""
        return (
            self.daily_loss_pct >= self.config.effective_daily_loss_pct or
            self.current_drawdown_pct >= self.config.effective_max_drawdown_pct
        )
    
    @property
    def challenge_passed(self) -> bool:
        """True if profit target reached."""
        return self.total_pnl >= self.config.profit_target_amount
    
    def get_status_message(self) -> str:
        """Get formatted status message."""
        return (
            f"📊 {self.config.firm_name} {self.config.phase.value.upper()}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Balance: ${self.current_balance:,.2f}\n"
            f"📈 Total P&L: ${self.total_pnl:+,.2f}\n"
            f"📅 Daily P&L: ${self.daily_pnl:+,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Target Progress: {self.profit_progress_pct:.1f}%\n"
            f"⚠️ Daily Loss: {self.daily_loss_pct:.2f}% / {self.config.effective_daily_loss_pct:.1f}%\n"
            f"📉 Drawdown: {self.current_drawdown_pct:.2f}% / {self.config.effective_max_drawdown_pct:.1f}%\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 Daily Loss Remaining: ${self.daily_loss_remaining:,.2f}\n"
            f"🛡️ DD Buffer: ${self.drawdown_remaining:,.2f}"
        )


class PropFirmMonitor:
    """
    Monitors trading against prop firm rules.
    
    Integrates with trading bot to:
    - Track progress toward targets
    - Warn when approaching limits
    - Block trading when limits hit
    - Generate status reports
    
    Usage:
        monitor = PropFirmMonitor(prop_firm_config)
        monitor.initialize(starting_balance=100000)
        
        # After each trade
        monitor.update(daily_pnl=-500, total_pnl=2500, balance=102500)
        
        # Check before trading
        if monitor.can_trade():
            execute_trade()
        else:
            print(monitor.block_reason)
    """
    
    def __init__(self, config: PropFirmConfig):
        self.config = config
        self.status = PropFirmStatus(config=config)
        self._peak_balance = 0.0
        self._trade_count_today = 0
        self._last_update = None
    
    def initialize(self, starting_balance: float):
        """Initialize monitor with starting balance."""
        self.status.starting_balance = starting_balance
        self.status.current_balance = starting_balance
        self._peak_balance = starting_balance
        logger.info(f"PropFirmMonitor initialized: ${starting_balance:,.2f}")
    
    def update(
        self,
        daily_pnl: float,
        total_pnl: float,
        balance: float
    ):
        """Update status with current metrics."""
        self.status.daily_pnl = daily_pnl
        self.status.total_pnl = total_pnl
        self.status.current_balance = balance
        
        # Update peak
        if balance > self._peak_balance:
            self._peak_balance = balance
        
        self._last_update = datetime.utcnow()
    
    def record_trade(self):
        """Record that a trade was made."""
        self._trade_count_today += 1
    
    def reset_daily(self):
        """Reset daily counters."""
        self.status.daily_pnl = 0.0
        self._trade_count_today = 0
    
    def can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed.
        
        Returns:
            (can_trade: bool, reason: str)
        """
        # Check daily loss limit
        if self.status.daily_loss_pct >= self.config.effective_daily_loss_pct:
            return False, f"Daily loss limit reached ({self.status.daily_loss_pct:.2f}%)"
        
        # Check drawdown limit
        if self.status.current_drawdown_pct >= self.config.effective_max_drawdown_pct:
            return False, f"Max drawdown reached ({self.status.current_drawdown_pct:.2f}%)"
        
        # Check trade count
        if self._trade_count_today >= self.config.max_trades_per_day:
            return False, f"Max daily trades reached ({self._trade_count_today})"
        
        # Check if approaching limits (warning)
        if self.status.is_daily_limit_near:
            return True, f"WARNING: Approaching daily limit ({self.status.daily_loss_pct:.2f}%)"
        
        if self.status.is_drawdown_limit_near:
            return True, f"WARNING: Approaching max drawdown ({self.status.current_drawdown_pct:.2f}%)"
        
        return True, "OK"
    
    def get_max_risk_for_trade(self) -> float:
        """Calculate maximum risk allowed for next trade."""
        # Based on remaining daily loss allowance
        daily_remaining = self.status.daily_loss_remaining
        
        # Based on remaining drawdown
        dd_remaining = self.status.drawdown_remaining
        
        # Use the more restrictive
        max_loss_allowed = min(daily_remaining, dd_remaining)
        
        # Cap at configured max risk per trade
        max_risk = self.status.starting_balance * (self.config.max_risk_per_trade_pct / 100)
        
        return min(max_loss_allowed, max_risk)
    
    def get_position_size_limit(self, sl_pips: float, pip_value: float = 10.0) -> float:
        """
        Calculate maximum position size based on remaining risk budget.
        
        Args:
            sl_pips: Stop loss in pips
            pip_value: Value per pip per lot (default $10 for majors)
        
        Returns:
            Maximum lot size
        """
        max_risk = self.get_max_risk_for_trade()
        
        if sl_pips <= 0 or pip_value <= 0:
            return 0.01  # Minimum
        
        max_lots = max_risk / (sl_pips * pip_value)
        
        # Apply firm's lot limit if any
        if self.config.rules.max_lots_per_trade > 0:
            max_lots = min(max_lots, self.config.rules.max_lots_per_trade)
        
        return max(0.01, round(max_lots, 2))
    
    def get_status(self) -> PropFirmStatus:
        """Get current status."""
        return self.status
    
    def get_alerts(self) -> List[str]:
        """Get list of current alerts/warnings."""
        alerts = []
        
        if self.status.should_stop_trading:
            alerts.append("🚨 STOP TRADING - Limit reached!")
        
        if self.status.is_daily_limit_near:
            alerts.append(f"⚠️ Daily loss at {self.status.daily_loss_pct:.1f}%")
        
        if self.status.is_drawdown_limit_near:
            alerts.append(f"⚠️ Drawdown at {self.status.current_drawdown_pct:.1f}%")
        
        if self.status.challenge_passed:
            alerts.append("🎉 PROFIT TARGET REACHED!")
        
        return alerts
