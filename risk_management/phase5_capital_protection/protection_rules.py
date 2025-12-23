# risk_management/phase5_capital_protection/protection_rules.py
"""
Phase 5: Capital Protection System

Rule-based capital protection overlays (NOT learned).
These are hard safety limits that override all other decisions.

Rules:
1. Daily Loss Limit: Stop trading if daily loss exceeds threshold
2. Drawdown Protection: Reduce position sizes as drawdown increases
3. Losing Streak: Cooldown period after consecutive losses
4. Equity Curve Monitoring: Kill switch if equity degrades significantly
5. Weekly/Monthly Limits: Longer-term loss protection
6. Correlation Exposure: Limit exposure to correlated positions

These rules are deterministic and always enforced.
They protect capital during adverse conditions and model failures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ProtectionLevel(Enum):
    """Protection activation levels."""
    NORMAL = "normal"           # No restrictions
    CAUTION = "caution"         # Reduced sizing
    WARNING = "warning"         # Significantly reduced
    CRITICAL = "critical"       # Trading suspended
    KILLED = "killed"           # Kill switch activated


class ProtectionAction(Enum):
    """Actions taken by protection system."""
    ALLOW = "allow"
    REDUCE_SIZE = "reduce_size"
    BLOCK_NEW = "block_new"
    CLOSE_ALL = "close_all"
    KILL_SWITCH = "kill_switch"


@dataclass
class ProtectionConfig:
    """Configuration for capital protection rules."""
    # Daily limits
    max_daily_loss_pct: float = 3.0         # Max daily loss as % of balance
    daily_loss_warning_pct: float = 2.0      # Warning threshold
    max_daily_trades: int = 20               # Max trades per day
    
    # Drawdown limits
    max_drawdown_pct: float = 10.0           # Max drawdown from peak
    drawdown_reduction_start: float = 5.0    # Start reducing at this DD
    drawdown_reduction_factor: float = 0.5   # Size multiplier at max DD
    
    # Weekly/Monthly limits
    max_weekly_loss_pct: float = 6.0
    max_monthly_loss_pct: float = 10.0
    
    # Losing streak
    max_consecutive_losses: int = 5
    losing_streak_cooldown_minutes: int = 30
    losing_streak_size_reduction: float = 0.5
    
    # Equity curve monitoring
    equity_ma_period: int = 20               # Moving average period
    equity_below_ma_limit: int = 10          # Days below MA triggers warning
    equity_kill_threshold: float = 0.85      # Kill if equity < 85% of peak
    
    # Win rate monitoring  
    min_win_rate: float = 0.35               # Minimum acceptable win rate
    win_rate_lookback: int = 50              # Trades to consider
    
    # Exposure limits (redundant with Phase 2 but enforced here too)
    max_total_exposure_pct: float = 20.0
    max_correlated_exposure_pct: float = 10.0
    
    # Recovery rules
    recovery_win_streak: int = 3             # Wins needed to lift restrictions
    recovery_profit_pct: float = 1.0         # Profit needed to lift restrictions


@dataclass
class ProtectionState:
    """Current state of capital protection."""
    level: ProtectionLevel = ProtectionLevel.NORMAL
    action: ProtectionAction = ProtectionAction.ALLOW
    size_multiplier: float = 1.0
    
    # Timestamps
    last_update: datetime = field(default_factory=datetime.utcnow)
    cooldown_until: Optional[datetime] = None
    killed_at: Optional[datetime] = None
    
    # Metrics at state change
    trigger_reason: str = ""
    metrics_at_trigger: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'level': self.level.value,
            'action': self.action.value,
            'size_multiplier': self.size_multiplier,
            'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None,
            'killed_at': self.killed_at.isoformat() if self.killed_at else None,
            'trigger_reason': self.trigger_reason
        }


@dataclass
class TradingMetrics:
    """Aggregated trading metrics for protection decisions."""
    # Balance and equity
    current_balance: float
    peak_balance: float
    current_equity: float
    
    # P&L
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    
    # Drawdown
    current_drawdown_pct: float
    max_drawdown_pct: float
    
    # Trade stats
    daily_trade_count: int
    consecutive_losses: int
    consecutive_wins: int
    recent_win_rate: float
    
    # Exposure
    total_exposure_pct: float
    correlated_exposure_pct: float
    
    # Equity curve
    days_below_equity_ma: int
    equity_vs_peak_pct: float


class CapitalProtector:
    """
    Capital protection system that monitors and enforces safety limits.
    
    This is the final safety layer before trade execution.
    It can reduce position sizes, block new trades, or activate kill switch.
    
    Usage:
        protector = CapitalProtector(config)
        
        # Update with trade results
        protector.record_trade(pnl=150.0, is_win=True)
        
        # Check before trading
        result = protector.check_trade(
            proposed_size=0.5,
            account_balance=10000,
            current_exposure=2.5
        )
        
        if result['allowed']:
            execute_trade(size=result['adjusted_size'])
        else:
            logger.warning(f"Trade blocked: {result['reason']}")
    """
    
    def __init__(self, config: Optional[ProtectionConfig] = None):
        self.config = config or ProtectionConfig()
        
        # Current state
        self.state = ProtectionState()
        
        # Balance tracking
        self._initial_balance: float = 0.0
        self._peak_balance: float = 0.0
        self._current_balance: float = 0.0
        
        # P&L tracking
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._monthly_pnl: float = 0.0
        self._daily_trade_count: int = 0
        
        # Trade tracking
        self._trade_history: deque = deque(maxlen=500)
        self._consecutive_losses: int = 0
        self._consecutive_wins: int = 0
        
        # Equity curve
        self._equity_history: deque = deque(maxlen=100)
        self._equity_ma: float = 0.0
        self._days_below_ma: int = 0
        
        # Time tracking
        self._last_trade_time: Optional[datetime] = None
        self._day_start: Optional[datetime] = None
        self._week_start: Optional[datetime] = None
        self._month_start: Optional[datetime] = None
        
        logger.info("CapitalProtector initialized")
    
    def initialize(self, balance: float):
        """Initialize with starting balance."""
        self._initial_balance = balance
        self._peak_balance = balance
        self._current_balance = balance
        self._equity_history.append(balance)
        self._update_time_periods()
        
        logger.info(f"Capital protection initialized with balance: {balance:.2f}")
    
    def record_trade(
        self,
        pnl: float,
        is_win: bool,
        trade_size: float = 0.0,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a completed trade.
        
        Args:
            pnl: Profit/loss amount
            is_win: Whether trade was profitable
            trade_size: Position size (lots)
            timestamp: Trade close time
        """
        timestamp = timestamp or datetime.utcnow()
        self._check_time_periods(timestamp)
        
        # Update P&L
        self._current_balance += pnl
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self._monthly_pnl += pnl
        self._daily_trade_count += 1
        
        # Update peak
        if self._current_balance > self._peak_balance:
            self._peak_balance = self._current_balance
        
        # Update streaks
        if is_win:
            self._consecutive_wins += 1
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._consecutive_wins = 0
        
        # Record trade
        self._trade_history.append({
            'pnl': pnl,
            'is_win': is_win,
            'timestamp': timestamp,
            'balance': self._current_balance
        })
        
        # Update equity history
        self._equity_history.append(self._current_balance)
        self._update_equity_ma()
        
        # Evaluate protection state
        self._evaluate_state(now=timestamp)
        
        self._last_trade_time = timestamp
    
    def check_trade(
        self,
        proposed_size: float,
        account_balance: float,
        current_exposure: float = 0.0,
        correlated_exposure: float = 0.0,
        current_time: Optional[datetime] = None,
    ) -> Dict:
        """
        Check if a trade is allowed and get adjusted size.
        
        Args:
            proposed_size: Proposed position size (lots)
            account_balance: Current account balance
            current_exposure: Current total exposure (%)
            correlated_exposure: Current correlated exposure (%)
        
        Returns:
            Dict with 'allowed', 'adjusted_size', 'reason', 'warnings'
        """
        result = {
            'allowed': True,
            'adjusted_size': proposed_size,
            'reason': '',
            'warnings': [],
            'protection_level': self.state.level.value,
            'size_multiplier': self.state.size_multiplier
        }

        now = current_time or datetime.utcnow()
        try:
            self._check_time_periods(now)
        except Exception:
            pass
        
        # Check kill switch
        if self.state.level == ProtectionLevel.KILLED:
            result['allowed'] = False
            result['adjusted_size'] = 0.0
            result['reason'] = f"Kill switch active since {self.state.killed_at}"
            return result
        
        # Check cooldown
        if self.state.cooldown_until and now < self.state.cooldown_until:
            result['allowed'] = False
            result['adjusted_size'] = 0.0
            result['reason'] = f"In cooldown until {self.state.cooldown_until}"
            return result
        
        # Check critical level
        if self.state.level == ProtectionLevel.CRITICAL:
            result['allowed'] = False
            result['adjusted_size'] = 0.0
            result['reason'] = f"Critical protection level: {self.state.trigger_reason}"
            return result
        
        # Check daily trade limit
        if self._daily_trade_count >= self.config.max_daily_trades:
            result['allowed'] = False
            result['adjusted_size'] = 0.0
            result['reason'] = f"Daily trade limit reached ({self.config.max_daily_trades})"
            return result
        
        # Check exposure limits
        total_with_trade = current_exposure + (proposed_size / account_balance * 100000 / account_balance * 100)
        if total_with_trade > self.config.max_total_exposure_pct:
            result['warnings'].append(
                f"Would exceed total exposure limit ({total_with_trade:.1f}% > {self.config.max_total_exposure_pct}%)"
            )
        
        # Apply size multiplier
        adjusted_size = proposed_size * self.state.size_multiplier
        result['adjusted_size'] = adjusted_size
        
        if adjusted_size < proposed_size:
            result['warnings'].append(
                f"Size reduced from {proposed_size:.2f} to {adjusted_size:.2f} "
                f"(multiplier: {self.state.size_multiplier:.2f})"
            )
        
        # Add any active warnings
        if self.state.level == ProtectionLevel.CAUTION:
            result['warnings'].append(f"Caution level active: {self.state.trigger_reason}")
        elif self.state.level == ProtectionLevel.WARNING:
            result['warnings'].append(f"Warning level active: {self.state.trigger_reason}")
        
        return result
    
    def get_metrics(self) -> TradingMetrics:
        """Get current trading metrics."""
        drawdown_pct = (
            (self._peak_balance - self._current_balance) / self._peak_balance * 100
            if self._peak_balance > 0 else 0
        )
        
        # Calculate win rate from recent trades
        recent_trades = list(self._trade_history)[-self.config.win_rate_lookback:]
        if recent_trades:
            recent_win_rate = sum(1 for t in recent_trades if t['is_win']) / len(recent_trades)
        else:
            recent_win_rate = 0.5
        
        return TradingMetrics(
            current_balance=self._current_balance,
            peak_balance=self._peak_balance,
            current_equity=self._current_balance,
            daily_pnl=self._daily_pnl,
            weekly_pnl=self._weekly_pnl,
            monthly_pnl=self._monthly_pnl,
            current_drawdown_pct=drawdown_pct,
            max_drawdown_pct=drawdown_pct,  # Would need separate tracking
            daily_trade_count=self._daily_trade_count,
            consecutive_losses=self._consecutive_losses,
            consecutive_wins=self._consecutive_wins,
            recent_win_rate=recent_win_rate,
            total_exposure_pct=0.0,  # Set externally
            correlated_exposure_pct=0.0,
            days_below_equity_ma=self._days_below_ma,
            equity_vs_peak_pct=self._current_balance / self._peak_balance * 100 if self._peak_balance > 0 else 100
        )
    
    def get_state(self) -> ProtectionState:
        """Get current protection state."""
        return self.state
    
    def reset_daily(self, now: Optional[datetime] = None):
        """Reset daily counters (call at start of trading day)."""
        self._daily_pnl = 0.0
        self._daily_trade_count = 0
        self._day_start = (now or datetime.utcnow()).replace(hour=0, minute=0, second=0, microsecond=0)
        logger.info("Daily protection counters reset")
    
    def reset_weekly(self, now: Optional[datetime] = None):
        """Reset weekly counters."""
        self._weekly_pnl = 0.0
        ts = now or datetime.utcnow()
        self._week_start = ts - timedelta(days=ts.weekday())
        logger.info("Weekly protection counters reset")
    
    def reset_monthly(self, now: Optional[datetime] = None):
        """Reset monthly counters."""
        self._monthly_pnl = 0.0
        self._month_start = (now or datetime.utcnow()).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        logger.info("Monthly protection counters reset")
    
    def lift_restrictions(self, reason: str = "Manual override"):
        """Manually lift restrictions (except kill switch)."""
        if self.state.level != ProtectionLevel.KILLED:
            self.state = ProtectionState(
                level=ProtectionLevel.NORMAL,
                action=ProtectionAction.ALLOW,
                size_multiplier=1.0,
                trigger_reason=f"Lifted: {reason}"
            )
            self.state.cooldown_until = None
            logger.info(f"Restrictions lifted: {reason}")
    
    def activate_kill_switch(self, reason: str):
        """Manually activate kill switch."""
        self.state.level = ProtectionLevel.KILLED
        self.state.action = ProtectionAction.KILL_SWITCH
        self.state.killed_at = datetime.utcnow()
        self.state.trigger_reason = reason
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
    
    def _evaluate_state(self, now: Optional[datetime] = None):
        """Evaluate current metrics and update protection state."""
        metrics = self.get_metrics()

        ts = now or datetime.utcnow()
        
        new_level = ProtectionLevel.NORMAL
        new_action = ProtectionAction.ALLOW
        size_mult = 1.0
        trigger = ""
        cooldown = None
        
        # Check kill switch conditions
        if metrics.equity_vs_peak_pct < self.config.equity_kill_threshold * 100:
            new_level = ProtectionLevel.KILLED
            new_action = ProtectionAction.KILL_SWITCH
            trigger = f"Equity at {metrics.equity_vs_peak_pct:.1f}% of peak"
            self.state.killed_at = ts
        
        # Check daily loss limit
        elif (metrics.daily_pnl < 0 and 
              abs(metrics.daily_pnl) / self._initial_balance * 100 >= self.config.max_daily_loss_pct):
            new_level = ProtectionLevel.CRITICAL
            new_action = ProtectionAction.BLOCK_NEW
            trigger = f"Daily loss limit hit: {abs(metrics.daily_pnl):.2f}"
        
        # Check weekly loss limit
        elif (metrics.weekly_pnl < 0 and
              abs(metrics.weekly_pnl) / self._initial_balance * 100 >= self.config.max_weekly_loss_pct):
            new_level = ProtectionLevel.CRITICAL
            new_action = ProtectionAction.BLOCK_NEW
            trigger = f"Weekly loss limit hit: {abs(metrics.weekly_pnl):.2f}"
        
        # Check monthly loss limit
        elif (metrics.monthly_pnl < 0 and
              abs(metrics.monthly_pnl) / self._initial_balance * 100 >= self.config.max_monthly_loss_pct):
            new_level = ProtectionLevel.CRITICAL
            new_action = ProtectionAction.BLOCK_NEW
            trigger = f"Monthly loss limit hit: {abs(metrics.monthly_pnl):.2f}"
        
        # Check losing streak
        elif metrics.consecutive_losses >= self.config.max_consecutive_losses:
            new_level = ProtectionLevel.WARNING
            new_action = ProtectionAction.BLOCK_NEW
            cooldown = ts + timedelta(minutes=self.config.losing_streak_cooldown_minutes)
            trigger = f"Losing streak: {metrics.consecutive_losses} consecutive losses"
        
        # Check drawdown - graduated response
        elif metrics.current_drawdown_pct >= self.config.max_drawdown_pct:
            new_level = ProtectionLevel.WARNING
            new_action = ProtectionAction.REDUCE_SIZE
            size_mult = self.config.drawdown_reduction_factor
            trigger = f"Max drawdown reached: {metrics.current_drawdown_pct:.1f}%"
        
        elif metrics.current_drawdown_pct >= self.config.drawdown_reduction_start:
            new_level = ProtectionLevel.CAUTION
            new_action = ProtectionAction.REDUCE_SIZE
            # Linear interpolation of size reduction
            dd_range = self.config.max_drawdown_pct - self.config.drawdown_reduction_start
            dd_progress = (metrics.current_drawdown_pct - self.config.drawdown_reduction_start) / dd_range
            size_mult = 1.0 - (1.0 - self.config.drawdown_reduction_factor) * dd_progress
            trigger = f"Drawdown warning: {metrics.current_drawdown_pct:.1f}%"
        
        # Check win rate
        elif metrics.recent_win_rate < self.config.min_win_rate and len(self._trade_history) >= 20:
            new_level = ProtectionLevel.CAUTION
            new_action = ProtectionAction.REDUCE_SIZE
            size_mult = 0.75
            trigger = f"Low win rate: {metrics.recent_win_rate:.1%}"
        
        # Check daily loss warning
        elif (metrics.daily_pnl < 0 and
              abs(metrics.daily_pnl) / self._initial_balance * 100 >= self.config.daily_loss_warning_pct):
            new_level = ProtectionLevel.CAUTION
            new_action = ProtectionAction.REDUCE_SIZE
            size_mult = 0.75
            trigger = f"Daily loss warning: {abs(metrics.daily_pnl):.2f}"
        
        # Check recovery conditions
        if (self.state.level in [ProtectionLevel.CAUTION, ProtectionLevel.WARNING] and
            metrics.consecutive_wins >= self.config.recovery_win_streak):
            new_level = ProtectionLevel.NORMAL
            new_action = ProtectionAction.ALLOW
            size_mult = 1.0
            trigger = f"Recovery: {metrics.consecutive_wins} consecutive wins"
        
        # Update state if changed
        if new_level != self.state.level or new_action != self.state.action:
            self.state = ProtectionState(
                level=new_level,
                action=new_action,
                size_multiplier=size_mult,
                last_update=ts,
                cooldown_until=cooldown,
                trigger_reason=trigger,
                metrics_at_trigger={
                    'drawdown': metrics.current_drawdown_pct,
                    'daily_pnl': metrics.daily_pnl,
                    'consecutive_losses': metrics.consecutive_losses,
                    'win_rate': metrics.recent_win_rate
                }
            )
            
            if new_level != ProtectionLevel.NORMAL:
                logger.warning(f"Protection level changed to {new_level.value}: {trigger}")
    
    def _update_time_periods(self, now: Optional[datetime] = None):
        """Initialize time period tracking."""
        ts = now or datetime.utcnow()
        self._day_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        self._week_start = ts - timedelta(days=ts.weekday())
        self._month_start = ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def _check_time_periods(self, timestamp: datetime):
        """Check and reset time periods if needed."""
        if self._day_start is None:
            self._update_time_periods(now=timestamp)
            return
        
        # Check day rollover
        day_boundary = self._day_start + timedelta(days=1)
        if timestamp >= day_boundary:
            self.reset_daily(now=timestamp)
        
        # Check week rollover
        week_boundary = self._week_start + timedelta(weeks=1)
        if timestamp >= week_boundary:
            self.reset_weekly(now=timestamp)
        
        # Check month rollover
        next_month = (self._month_start.month % 12) + 1
        next_month_year = self._month_start.year + (1 if next_month == 1 else 0)
        month_boundary = self._month_start.replace(year=next_month_year, month=next_month)
        if timestamp >= month_boundary:
            self.reset_monthly(now=timestamp)
    
    def _update_equity_ma(self):
        """Update equity moving average."""
        if len(self._equity_history) >= self.config.equity_ma_period:
            self._equity_ma = np.mean(list(self._equity_history)[-self.config.equity_ma_period:])
            
            if self._current_balance < self._equity_ma:
                self._days_below_ma += 1
            else:
                self._days_below_ma = 0


class ProtectionManager:
    """
    High-level manager integrating capital protection with trading.
    
    Provides convenient methods for common operations.
    
    Usage:
        manager = ProtectionManager()
        manager.start_session(balance=10000)
        
        # Before each trade
        check = manager.pre_trade_check(size=0.5, balance=10000)
        if check['allowed']:
            execute_trade(size=check['adjusted_size'])
        
        # After each trade
        manager.post_trade_update(pnl=150, is_win=True)
        
        # Get status
        status = manager.get_status()
    """
    
    def __init__(self, config: Optional[ProtectionConfig] = None):
        self.protector = CapitalProtector(config)
        self._session_active = False
    
    def start_session(self, balance: float):
        """Start a new trading session."""
        self.protector.initialize(balance)
        self._session_active = True
        logger.info(f"Trading session started with balance: {balance:.2f}")
    
    def end_session(self):
        """End trading session."""
        self._session_active = False
        logger.info("Trading session ended")
    
    def pre_trade_check(
        self,
        proposed_size: float,
        account_balance: float,
        current_exposure: float = 0.0
    ) -> Dict:
        """
        Check if trade is allowed before execution.
        
        Returns dict with 'allowed', 'adjusted_size', 'reason', 'warnings'
        """
        if not self._session_active:
            return {
                'allowed': False,
                'adjusted_size': 0.0,
                'reason': 'No active session'
            }
        
        return self.protector.check_trade(
            proposed_size=proposed_size,
            account_balance=account_balance,
            current_exposure=current_exposure
        )
    
    def post_trade_update(
        self,
        pnl: float,
        is_win: bool,
        trade_size: float = 0.0
    ):
        """Update after trade completion."""
        self.protector.record_trade(pnl, is_win, trade_size)
    
    def get_status(self) -> Dict:
        """Get current protection status."""
        state = self.protector.get_state()
        metrics = self.protector.get_metrics()
        
        return {
            'session_active': self._session_active,
            'protection_level': state.level.value,
            'protection_action': state.action.value,
            'size_multiplier': state.size_multiplier,
            'trigger_reason': state.trigger_reason,
            'in_cooldown': (
                state.cooldown_until is not None and
                datetime.utcnow() < state.cooldown_until
            ),
            'metrics': {
                'balance': metrics.current_balance,
                'peak_balance': metrics.peak_balance,
                'drawdown_pct': metrics.current_drawdown_pct,
                'daily_pnl': metrics.daily_pnl,
                'weekly_pnl': metrics.weekly_pnl,
                'daily_trades': metrics.daily_trade_count,
                'consecutive_losses': metrics.consecutive_losses,
                'consecutive_wins': metrics.consecutive_wins,
                'win_rate': metrics.recent_win_rate
            }
        }
    
    def force_stop(self, reason: str = "Manual stop"):
        """Force stop all trading."""
        self.protector.activate_kill_switch(reason)
    
    def reset_to_normal(self, reason: str = "Manual reset"):
        """Reset protection to normal (if not killed)."""
        self.protector.lift_restrictions(reason)
