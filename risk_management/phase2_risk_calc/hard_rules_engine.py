#!/usr/bin/env python3
# risk_management\phase2_risk_calc\hard_rules_engine.py
# Context: This code is part of a larger trading system that aims to ensure safe and compliant trading practices.

# TradeGatekeeper: Hard Rules Engine for Forex Trading
# This module implements a set of deterministic trading rules
# that cannot be overridden by machine learning models.

"""
Phase 2: Hard Rules Engine

Deterministic rules that ALWAYS apply regardless of model predictions:
- Maximum leverage limits
- Exposure limits per pair and portfolio
- Session-based filters (spread, liquidity, time)
- Correlation-based exposure controls
- News event blackouts

These are the "guardrails" that prevent catastrophic losses.
"""

import numpy as np
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """Major trading sessions."""
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    WEEKEND = "weekend"
    ROLLOVER = "rollover"


class RuleViolation(NamedTuple):
    """Represents a rule violation."""
    rule_name: str
    severity: str           # 'warning', 'block', 'critical'
    message: str
    current_value: float
    limit_value: float


@dataclass
class HardRulesConfig:
    """Configuration for hard trading rules."""
    
    # Leverage limits by regime
    max_leverage_default: float = 10.0
    max_leverage_by_regime: Dict[str, float] = field(default_factory=lambda: {
        'trending_strong': 15.0,
        'trending_weak': 10.0,
        'ranging': 8.0,
        'volatile': 5.0,
        'low_volatility': 12.0
    })
    
    # Exposure limits (as % of account)
    max_single_pair_exposure: float = 5.0
    max_correlated_group_exposure: float = 10.0
    max_total_exposure: float = 20.0
    max_single_direction_exposure: float = 15.0  # Max in one direction (long/short)
    
    # Spread limits (in pips)
    max_spread_by_pair: Dict[str, float] = field(default_factory=lambda: {
        'EURUSD': 2.0,
        'GBPUSD': 3.0,
        'USDJPY': 2.0,
        'USDCHF': 2.5,
        'AUDUSD': 2.5,
        'USDCAD': 2.5,
        'NZDUSD': 3.0,
        'EURJPY': 3.0,
        'GBPJPY': 4.0,
        'EURGBP': 2.5,
        'DEFAULT': 5.0
    })
    
    # Liquidity thresholds
    min_liquidity_depth: float = 0.0  # Minimum depth in lots
    
    # Session rules
    allowed_sessions: List[TradingSession] = field(default_factory=lambda: [
        TradingSession.TOKYO,
        TradingSession.LONDON,
        TradingSession.NEW_YORK,
        TradingSession.OVERLAP_LONDON_NY
    ])
    
    # Time windows (UTC)
    tokyo_open: time = time(0, 0)    # 00:00 UTC
    tokyo_close: time = time(9, 0)   # 09:00 UTC
    london_open: time = time(8, 0)   # 08:00 UTC
    london_close: time = time(17, 0) # 17:00 UTC
    ny_open: time = time(13, 0)      # 13:00 UTC
    ny_close: time = time(22, 0)     # 22:00 UTC
    
    # Rollover avoidance (typically 5PM EST = 22:00 UTC)
    rollover_start: time = time(21, 45)
    rollover_end: time = time(22, 15)
    avoid_rollover: bool = True
    
    # Weekend closing
    friday_close_hour: int = 21      # Close positions by 21:00 UTC Friday
    
    # News blackout (minutes before/after high-impact news)
    news_blackout_before: int = 30
    news_blackout_after: int = 15
    
    # Correlation groups (pairs that move together)
    correlation_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        'usd_pairs': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'],
        'jpy_pairs': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY'],
        'commodity_currencies': ['AUDUSD', 'NZDUSD', 'USDCAD']
    })


class HardRulesEngine:
    """
    Enforces hard trading rules that cannot be overridden.
    
    All rule checks return:
    - Whether the trade is allowed
    - List of any violations
    - Adjusted parameters if applicable
    """
    
    def __init__(self, config: Optional[HardRulesConfig] = None):
        self.config = config or HardRulesConfig()
        
        # Track current state
        self._open_positions: Dict[str, Dict] = {}  # pair -> position info
        self._scheduled_news: List[Dict] = []
    
    def check_all_rules(
        self,
        pair: str,
        direction: str,
        position_size: float,
        entry_price: float,
        current_spread: float,
        account_balance: float,
        current_time: Optional[datetime] = None,
        regime: Optional[str] = None,
        leverage_requested: Optional[float] = None
    ) -> Tuple[bool, List[RuleViolation], Dict]:
        """
        Check all hard rules before entering a trade.
        
        Args:
            pair: Currency pair
            direction: 'BUY' or 'SELL'
            position_size: Requested position size in lots
            entry_price: Entry price
            current_spread: Current spread in pips
            account_balance: Account balance
            current_time: Current datetime (UTC)
            regime: Market regime
            leverage_requested: Requested leverage
        
        Returns:
            (is_allowed, violations, adjusted_params)
        """
        current_time = current_time or datetime.utcnow()
        violations = []
        adjusted_params = {}
        
        # Check each rule
        violations.extend(self._check_spread_rule(pair, current_spread))
        violations.extend(self._check_session_rules(current_time))
        violations.extend(self._check_rollover_rule(current_time))
        violations.extend(self._check_weekend_rule(current_time))
        violations.extend(self._check_news_blackout(current_time))
        
        # Exposure checks
        exposure_violations, adjusted_size = self._check_exposure_rules(
            pair, direction, position_size, entry_price, account_balance
        )
        violations.extend(exposure_violations)
        if adjusted_size != position_size:
            adjusted_params['position_size'] = adjusted_size
        
        # Leverage check
        if leverage_requested is not None:
            lev_violations, adjusted_leverage = self._check_leverage_rule(
                leverage_requested, regime
            )
            violations.extend(lev_violations)
            if adjusted_leverage != leverage_requested:
                adjusted_params['leverage'] = adjusted_leverage
        
        # Determine if trade is blocked
        blocking_violations = [v for v in violations if v.severity in ('block', 'critical')]
        is_allowed = len(blocking_violations) == 0
        
        if not is_allowed:
            logger.warning(f"Trade blocked: {[v.message for v in blocking_violations]}")
        
        return is_allowed, violations, adjusted_params
    
    def _check_spread_rule(
        self,
        pair: str,
        current_spread: float
    ) -> List[RuleViolation]:
        """Check if spread is within acceptable limits."""
        violations = []
        
        max_spread = self.config.max_spread_by_pair.get(
            pair.upper(),
            self.config.max_spread_by_pair['DEFAULT']
        )
        
        if current_spread > max_spread:
            violations.append(RuleViolation(
                rule_name='max_spread',
                severity='block',
                message=f"Spread {current_spread:.1f} pips exceeds max {max_spread:.1f} for {pair}",
                current_value=current_spread,
                limit_value=max_spread
            ))
        elif current_spread > max_spread * 0.8:
            violations.append(RuleViolation(
                rule_name='high_spread_warning',
                severity='warning',
                message=f"Spread {current_spread:.1f} pips approaching max {max_spread:.1f}",
                current_value=current_spread,
                limit_value=max_spread
            ))
        
        return violations
    
    def _check_session_rules(self, current_time: datetime) -> List[RuleViolation]:
        """Check if current session is allowed for trading."""
        violations = []
        
        current_session = self._get_current_session(current_time)
        
        if current_session not in self.config.allowed_sessions:
            violations.append(RuleViolation(
                rule_name='session_filter',
                severity='block',
                message=f"Trading not allowed during {current_session.value} session",
                current_value=0,
                limit_value=0
            ))
        
        return violations
    
    def _check_rollover_rule(self, current_time: datetime) -> List[RuleViolation]:
        """Check if we're in rollover period."""
        violations = []
        
        if not self.config.avoid_rollover:
            return violations
        
        current = current_time.time()
        
        if self.config.rollover_start <= current <= self.config.rollover_end:
            violations.append(RuleViolation(
                rule_name='rollover_avoidance',
                severity='block',
                message="Trading blocked during rollover period",
                current_value=0,
                limit_value=0
            ))
        
        return violations
    
    def _check_weekend_rule(self, current_time: datetime) -> List[RuleViolation]:
        """Check if market is closed for weekend."""
        violations = []
        
        weekday = current_time.weekday()
        hour = current_time.hour
        
        # Friday after close time
        if weekday == 4 and hour >= self.config.friday_close_hour:
            violations.append(RuleViolation(
                rule_name='weekend_close',
                severity='block',
                message="Market closing for weekend - no new positions",
                current_value=0,
                limit_value=0
            ))
        
        # Saturday or Sunday
        if weekday in (5, 6):
            violations.append(RuleViolation(
                rule_name='weekend_close',
                severity='block',
                message="Market closed for weekend",
                current_value=0,
                limit_value=0
            ))
        
        return violations
    
    def _check_news_blackout(self, current_time: datetime) -> List[RuleViolation]:
        """Check if we're in a news blackout period."""
        violations = []
        
        for news_event in self._scheduled_news:
            event_time = news_event.get('time')
            if event_time is None:
                continue
            
            time_diff = (event_time - current_time).total_seconds() / 60
            
            # Before news
            if 0 < time_diff <= self.config.news_blackout_before:
                violations.append(RuleViolation(
                    rule_name='news_blackout',
                    severity='block',
                    message=f"News event in {int(time_diff)} minutes: {news_event.get('title', 'Unknown')}",
                    current_value=time_diff,
                    limit_value=self.config.news_blackout_before
                ))
            
            # After news
            if -self.config.news_blackout_after <= time_diff <= 0:
                violations.append(RuleViolation(
                    rule_name='news_blackout',
                    severity='warning',
                    message=f"News event {int(-time_diff)} minutes ago: {news_event.get('title', 'Unknown')}",
                    current_value=-time_diff,
                    limit_value=self.config.news_blackout_after
                ))
        
        return violations
    
    def _check_exposure_rules(
        self,
        pair: str,
        direction: str,
        position_size: float,
        entry_price: float,
        account_balance: float
    ) -> Tuple[List[RuleViolation], float]:
        """
        Check exposure limits.
        
        Returns (violations, adjusted_position_size)
        """
        violations = []
        adjusted_size = position_size
        
        # Calculate exposure of new position
        position_value = position_size * 100000 * entry_price  # Approximate
        new_exposure_pct = (position_value / account_balance) * 100
        
        # Single pair exposure
        current_pair_exposure = self._get_pair_exposure(pair, account_balance)
        total_pair_exposure = current_pair_exposure + new_exposure_pct
        
        if total_pair_exposure > self.config.max_single_pair_exposure:
            available = max(0, self.config.max_single_pair_exposure - current_pair_exposure)
            
            if available <= 0:
                violations.append(RuleViolation(
                    rule_name='max_pair_exposure',
                    severity='block',
                    message=f"Max exposure reached for {pair}",
                    current_value=current_pair_exposure,
                    limit_value=self.config.max_single_pair_exposure
                ))
                adjusted_size = 0
            else:
                violations.append(RuleViolation(
                    rule_name='max_pair_exposure',
                    severity='warning',
                    message=f"Position reduced to stay within {pair} limit",
                    current_value=total_pair_exposure,
                    limit_value=self.config.max_single_pair_exposure
                ))
                adjusted_size *= (available / new_exposure_pct)
        
        # Total exposure
        total_exposure = self._get_total_exposure(account_balance) + new_exposure_pct
        
        if total_exposure > self.config.max_total_exposure:
            violations.append(RuleViolation(
                rule_name='max_total_exposure',
                severity='block',
                message=f"Total exposure {total_exposure:.1f}% exceeds max {self.config.max_total_exposure}%",
                current_value=total_exposure,
                limit_value=self.config.max_total_exposure
            ))
            adjusted_size = 0
        
        # Directional exposure
        direction_exposure = self._get_directional_exposure(direction, account_balance)
        direction_exposure += new_exposure_pct
        
        if direction_exposure > self.config.max_single_direction_exposure:
            violations.append(RuleViolation(
                rule_name='max_direction_exposure',
                severity='warning',
                message=f"{direction} exposure {direction_exposure:.1f}% high",
                current_value=direction_exposure,
                limit_value=self.config.max_single_direction_exposure
            ))
        
        # Correlated group exposure
        group_violations = self._check_correlation_group_exposure(
            pair, new_exposure_pct, account_balance
        )
        violations.extend(group_violations)
        
        return violations, adjusted_size
    
    def _check_leverage_rule(
        self,
        leverage_requested: float,
        regime: Optional[str]
    ) -> Tuple[List[RuleViolation], float]:
        """Check leverage limits."""
        violations = []
        
        # Get max leverage for regime
        if regime and regime in self.config.max_leverage_by_regime:
            max_leverage = self.config.max_leverage_by_regime[regime]
        else:
            max_leverage = self.config.max_leverage_default
        
        if leverage_requested > max_leverage:
            violations.append(RuleViolation(
                rule_name='max_leverage',
                severity='warning',
                message=f"Leverage {leverage_requested}x exceeds max {max_leverage}x for {regime or 'default'}",
                current_value=leverage_requested,
                limit_value=max_leverage
            ))
            return violations, max_leverage
        
        return violations, leverage_requested
    
    def _check_correlation_group_exposure(
        self,
        pair: str,
        new_exposure: float,
        account_balance: float
    ) -> List[RuleViolation]:
        """Check exposure in correlated currency groups."""
        violations = []
        
        for group_name, group_pairs in self.config.correlation_groups.items():
            if pair.upper() not in [p.upper() for p in group_pairs]:
                continue
            
            # Calculate current group exposure
            group_exposure = sum(
                self._get_pair_exposure(p, account_balance)
                for p in group_pairs
            )
            total_group_exposure = group_exposure + new_exposure
            
            if total_group_exposure > self.config.max_correlated_group_exposure:
                violations.append(RuleViolation(
                    rule_name='correlated_group_exposure',
                    severity='warning',
                    message=f"Correlated group '{group_name}' exposure {total_group_exposure:.1f}% high",
                    current_value=total_group_exposure,
                    limit_value=self.config.max_correlated_group_exposure
                ))
        
        return violations
    
    def _get_current_session(self, current_time: datetime) -> TradingSession:
        """Determine current trading session."""
        current = current_time.time()
        weekday = current_time.weekday()
        
        if weekday in (5, 6):
            return TradingSession.WEEKEND
        
        # Check overlap first
        if (self.config.london_open <= current <= self.config.ny_close and
            self.config.ny_open <= current <= self.config.london_close):
            return TradingSession.OVERLAP_LONDON_NY
        
        # Check rollover
        if self.config.rollover_start <= current <= self.config.rollover_end:
            return TradingSession.ROLLOVER
        
        # Check individual sessions
        if self.config.tokyo_open <= current <= self.config.tokyo_close:
            return TradingSession.TOKYO
        
        if self.config.london_open <= current <= self.config.london_close:
            return TradingSession.LONDON
        
        if self.config.ny_open <= current <= self.config.ny_close:
            return TradingSession.NEW_YORK
        
        # Default to Tokyo (early morning)
        return TradingSession.TOKYO
    
    def _get_pair_exposure(self, pair: str, account_balance: float) -> float:
        """Get current exposure for a pair as percentage of account."""
        position = self._open_positions.get(pair.upper(), {})
        position_value = position.get('value', 0)
        return (position_value / account_balance) * 100 if account_balance > 0 else 0
    
    def _get_total_exposure(self, account_balance: float) -> float:
        """Get total exposure as percentage of account."""
        total_value = sum(p.get('value', 0) for p in self._open_positions.values())
        return (total_value / account_balance) * 100 if account_balance > 0 else 0
    
    def _get_directional_exposure(self, direction: str, account_balance: float) -> float:
        """Get exposure in one direction (BUY/SELL)."""
        direction = direction.upper()
        direction_value = sum(
            p.get('value', 0)
            for p in self._open_positions.values()
            if p.get('direction', '').upper() == direction
        )
        return (direction_value / account_balance) * 100 if account_balance > 0 else 0
    
    def update_positions(self, positions: Dict[str, Dict]):
        """
        Update tracked open positions.
        
        Args:
            positions: Dict of pair -> {'value': float, 'direction': str, ...}
        """
        self._open_positions = positions
    
    def add_news_event(
        self,
        event_time: datetime,
        title: str,
        impact: str = 'high',
        currencies: Optional[List[str]] = None
    ):
        """Add a scheduled news event for blackout consideration."""
        self._scheduled_news.append({
            'time': event_time,
            'title': title,
            'impact': impact,
            'currencies': currencies or []
        })
    
    def clear_old_news(self, before: datetime):
        """Remove old news events from tracking."""
        self._scheduled_news = [
            n for n in self._scheduled_news
            if n.get('time', datetime.min) > before
        ]


class TradeGatekeeper:
    """
    High-level interface that combines all Phase 2 components.
    
    Use this as the single entry point for trade validation.
    """
    
    def __init__(
        self,
        rules_config: Optional[HardRulesConfig] = None
    ):
        self.rules_engine = HardRulesEngine(rules_config)
    
    def validate_trade(
        self,
        pair: str,
        direction: str,
        position_size: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        account_balance: float,
        current_spread: float,
        current_time: Optional[datetime] = None,
        regime: Optional[str] = None
    ) -> Dict:
        """
        Validate a complete trade setup.
        
        Returns comprehensive validation result.
        """
        is_allowed, violations, adjustments = self.rules_engine.check_all_rules(
            pair=pair,
            direction=direction,
            position_size=position_size,
            entry_price=entry_price,
            current_spread=current_spread,
            account_balance=account_balance,
            current_time=current_time,
            regime=regime
        )
        
        # Categorize violations
        warnings = [v for v in violations if v.severity == 'warning']
        blocks = [v for v in violations if v.severity in ('block', 'critical')]
        
        return {
            'allowed': is_allowed,
            'violations': [
                {
                    'rule': v.rule_name,
                    'severity': v.severity,
                    'message': v.message,
                    'current': v.current_value,
                    'limit': v.limit_value
                }
                for v in violations
            ],
            'warnings_count': len(warnings),
            'blocks_count': len(blocks),
            'adjustments': adjustments,
            'original_position_size': position_size,
            'adjusted_position_size': adjustments.get('position_size', position_size)
        }