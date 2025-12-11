# risk_management/phase5_capital_protection/integration.py
"""
Phase 5: Capital Protection Integration

Integrates capital protection with the full risk management pipeline.
Provides hooks for different stages of trade lifecycle.

This module ensures capital protection rules are enforced
at every decision point in the trading system.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
import logging

from .protection_rules import (
    CapitalProtector, ProtectionConfig, ProtectionState,
    ProtectionLevel, ProtectionAction, ProtectionManager
)

logger = logging.getLogger(__name__)


@dataclass
class ProtectedTradeResult:
    """Result of a protected trade operation."""
    allowed: bool
    original_size: float
    adjusted_size: float
    protection_level: str
    warnings: List[str] = field(default_factory=list)
    block_reason: Optional[str] = None
    
    # Metadata
    checked_at: datetime = field(default_factory=datetime.utcnow)
    metrics_snapshot: Dict = field(default_factory=dict)


class TradingGuard:
    """
    Decorator-based protection for trading functions.
    
    Wraps trading operations to ensure capital protection
    rules are checked before execution.
    
    Usage:
        guard = TradingGuard(protection_config)
        
        @guard.protect_entry
        def open_trade(size: float, **kwargs) -> dict:
            return execute_trade(size=kwargs.get('adjusted_size', size))
        
        @guard.protect_exit
        def close_trade(ticket: int, pnl: float) -> dict:
            return close_position(ticket)
    """
    
    def __init__(
        self,
        config: Optional[ProtectionConfig] = None,
        on_block: Optional[Callable[[str], None]] = None,
        on_warning: Optional[Callable[[List[str]], None]] = None
    ):
        self.protector = CapitalProtector(config)
        self.on_block = on_block
        self.on_warning = on_warning
        
        # Track pending trades
        self._pending_trades: Dict[str, Dict] = {}
    
    def initialize(self, balance: float):
        """Initialize guard with starting balance."""
        self.protector.initialize(balance)
    
    def protect_entry(self, func: Callable) -> Callable:
        """
        Decorator for trade entry functions.
        
        The decorated function receives an 'adjusted_size' kwarg
        with the protection-adjusted position size.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract parameters
            size = kwargs.get('size', args[0] if args else 0.0)
            balance = kwargs.get('balance', 10000.0)
            exposure = kwargs.get('exposure', 0.0)
            
            # Check protection
            check = self.protector.check_trade(
                proposed_size=size,
                account_balance=balance,
                current_exposure=exposure
            )
            
            if not check['allowed']:
                if self.on_block:
                    self.on_block(check['reason'])
                
                logger.warning(f"Trade blocked: {check['reason']}")
                return {
                    'success': False,
                    'blocked': True,
                    'reason': check['reason']
                }
            
            if check['warnings'] and self.on_warning:
                self.on_warning(check['warnings'])
            
            # Add adjusted size to kwargs
            kwargs['adjusted_size'] = check['adjusted_size']
            kwargs['protection_level'] = check['protection_level']
            
            # Execute trade
            result = func(*args, **kwargs)
            
            # Track pending trade
            if isinstance(result, dict) and result.get('success'):
                ticket = result.get('ticket', str(datetime.utcnow().timestamp()))
                self._pending_trades[str(ticket)] = {
                    'size': check['adjusted_size'],
                    'entry_time': datetime.utcnow()
                }
            
            return result
        
        return wrapper
    
    def protect_exit(self, func: Callable) -> Callable:
        """
        Decorator for trade exit functions.
        
        Records trade results for protection tracking.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Execute exit
            result = func(*args, **kwargs)
            
            # Record if successful
            if isinstance(result, dict) and result.get('success'):
                pnl = result.get('pnl', 0.0)
                ticket = kwargs.get('ticket', args[0] if args else None)
                
                # Get trade info
                trade_info = self._pending_trades.pop(str(ticket), {})
                
                # Update protection
                self.protector.record_trade(
                    pnl=pnl,
                    is_win=pnl > 0,
                    trade_size=trade_info.get('size', 0.0)
                )
            
            return result
        
        return wrapper
    
    def get_state(self) -> ProtectionState:
        """Get current protection state."""
        return self.protector.get_state()


class ProtectedTradingSession:
    """
    Context manager for protected trading sessions.
    
    Automatically initializes protection at session start
    and handles cleanup at session end.
    
    Usage:
        with ProtectedTradingSession(balance=10000) as session:
            # Trading operations
            if session.can_trade():
                result = session.execute_trade(size=0.5)
                session.record_result(result)
    """
    
    def __init__(
        self,
        balance: float,
        config: Optional[ProtectionConfig] = None
    ):
        self.initial_balance = balance
        self.manager = ProtectionManager(config)
        self._trades_executed = 0
        self._session_pnl = 0.0
    
    def __enter__(self) -> 'ProtectedTradingSession':
        self.manager.start_session(self.initial_balance)
        logger.info(f"Protected trading session started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        status = self.manager.get_status()
        logger.info(
            f"Session ended: {self._trades_executed} trades, "
            f"PnL: {self._session_pnl:.2f}, "
            f"Final level: {status['protection_level']}"
        )
        self.manager.end_session()
        return False
    
    def can_trade(self, proposed_size: float = 0.1) -> bool:
        """Check if trading is currently allowed."""
        check = self.manager.pre_trade_check(
            proposed_size=proposed_size,
            account_balance=self.initial_balance + self._session_pnl
        )
        return check['allowed']
    
    def get_adjusted_size(
        self,
        proposed_size: float,
        current_exposure: float = 0.0
    ) -> ProtectedTradeResult:
        """Get protection-adjusted position size."""
        check = self.manager.pre_trade_check(
            proposed_size=proposed_size,
            account_balance=self.initial_balance + self._session_pnl,
            current_exposure=current_exposure
        )
        
        return ProtectedTradeResult(
            allowed=check['allowed'],
            original_size=proposed_size,
            adjusted_size=check.get('adjusted_size', 0.0),
            protection_level=check.get('protection_level', 'unknown'),
            warnings=check.get('warnings', []),
            block_reason=check.get('reason') if not check['allowed'] else None
        )
    
    def record_result(self, pnl: float, is_win: bool, size: float = 0.0):
        """Record trade result."""
        self.manager.post_trade_update(pnl, is_win, size)
        self._trades_executed += 1
        self._session_pnl += pnl
    
    def get_status(self) -> Dict:
        """Get current session status."""
        status = self.manager.get_status()
        status['session_trades'] = self._trades_executed
        status['session_pnl'] = self._session_pnl
        return status


def integrate_with_risk_manager(risk_manager, protection_config: Optional[ProtectionConfig] = None):
    """
    Integrate capital protection with existing RiskManager.
    
    This function patches the RiskManager to include
    capital protection checks.
    
    Args:
        risk_manager: Existing RiskManager instance
        protection_config: Protection configuration
    
    Returns:
        Enhanced RiskManager with capital protection
    """
    protector = CapitalProtector(protection_config)
    
    # Store original method
    original_evaluate = risk_manager.evaluate_trade_opportunity
    
    def protected_evaluate(*args, **kwargs):
        """Wrapper that adds capital protection checks."""
        # Get original decision
        decision = original_evaluate(*args, **kwargs)
        
        if not decision.should_trade:
            return decision
        
        # Check capital protection
        check = protector.check_trade(
            proposed_size=decision.position_size,
            account_balance=kwargs.get('account_balance', 10000),
            current_exposure=0.0  # Would need to track this
        )
        
        if not check['allowed']:
            decision.should_trade = False
            decision.rejection_reasons.append(f"Capital protection: {check['reason']}")
            decision.position_size = 0.0
            return decision
        
        # Apply size adjustment
        if check['adjusted_size'] != decision.position_size:
            decision.position_size = check['adjusted_size']
            decision.rejection_reasons.append(
                f"Size reduced by capital protection (level: {check['protection_level']})"
            )
        
        return decision
    
    # Patch the method
    risk_manager.evaluate_trade_opportunity = protected_evaluate
    risk_manager._capital_protector = protector
    
    # Add helper methods
    def record_trade_result(pnl: float, is_win: bool, size: float = 0.0):
        protector.record_trade(pnl, is_win, size)
    
    def get_protection_status() -> Dict:
        return {
            'state': protector.get_state().to_dict(),
            'metrics': protector.get_metrics().__dict__
        }
    
    risk_manager.record_trade_result = record_trade_result
    risk_manager.get_protection_status = get_protection_status
    
    logger.info("Capital protection integrated with RiskManager")
    
    return risk_manager


class CapitalProtectionCallback:
    """
    Callback class for integration with training/backtesting loops.
    
    Implements common callback interface for compatibility
    with various training frameworks.
    """
    
    def __init__(self, config: Optional[ProtectionConfig] = None):
        self.protector = CapitalProtector(config)
        self._episode_pnl = 0.0
        self._episode_trades = 0
    
    def on_episode_start(self, balance: float):
        """Called at start of each episode."""
        self.protector.initialize(balance)
        self._episode_pnl = 0.0
        self._episode_trades = 0
    
    def on_trade_start(self, proposed_size: float, balance: float) -> Dict:
        """Called before trade execution."""
        return self.protector.check_trade(
            proposed_size=proposed_size,
            account_balance=balance
        )
    
    def on_trade_end(self, pnl: float, is_win: bool, size: float):
        """Called after trade completion."""
        self.protector.record_trade(pnl, is_win, size)
        self._episode_pnl += pnl
        self._episode_trades += 1
    
    def on_episode_end(self) -> Dict:
        """Called at end of episode."""
        return {
            'episode_pnl': self._episode_pnl,
            'episode_trades': self._episode_trades,
            'final_state': self.protector.get_state().to_dict()
        }
    
    def should_stop(self) -> bool:
        """Check if trading should stop."""
        state = self.protector.get_state()
        return state.level in [ProtectionLevel.CRITICAL, ProtectionLevel.KILLED]
