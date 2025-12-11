# risk_management/phase5_capital_protection/__init__.py
"""
Phase 5: Capital Protection System

Rule-based safety overlays that protect trading capital.
These are NOT learned - they are deterministic hard limits.

Protection Rules:
1. Daily Loss Limit: Stop trading after X% daily loss
2. Weekly/Monthly Limits: Longer-term loss protection
3. Drawdown Protection: Reduce sizes as drawdown increases
4. Losing Streak: Cooldown after N consecutive losses
5. Equity Curve Monitoring: Kill switch if equity degrades
6. Win Rate Monitoring: Warning if win rate drops

Protection Levels:
- NORMAL: No restrictions
- CAUTION: Reduced position sizes
- WARNING: Significantly reduced sizes
- CRITICAL: Trading suspended (new trades blocked)
- KILLED: Kill switch active (all trading stopped)

These rules override all other decisions in the system.
They are the final safety layer before execution.

Usage:
    from risk_management.phase5_capital_protection import (
        CapitalProtector, ProtectionConfig, ProtectionManager
    )
    
    # Basic usage
    protector = CapitalProtector(ProtectionConfig(
        max_daily_loss_pct=3.0,
        max_drawdown_pct=10.0
    ))
    protector.initialize(balance=10000)
    
    # Before each trade
    check = protector.check_trade(
        proposed_size=0.5,
        account_balance=10000
    )
    
    if check['allowed']:
        execute_trade(size=check['adjusted_size'])
    
    # After each trade
    protector.record_trade(pnl=150, is_win=True)
    
    # Using ProtectionManager (simpler interface)
    manager = ProtectionManager()
    manager.start_session(balance=10000)
    
    check = manager.pre_trade_check(size=0.5, balance=10000)
    manager.post_trade_update(pnl=150, is_win=True)
    
    status = manager.get_status()
    
    # Using context manager
    with ProtectedTradingSession(balance=10000) as session:
        if session.can_trade():
            result = session.get_adjusted_size(0.5)
            # ... execute trade ...
            session.record_result(pnl=150, is_win=True)
"""

from .protection_rules import (
    # Enums
    ProtectionLevel,
    ProtectionAction,
    # Config
    ProtectionConfig,
    # State
    ProtectionState,
    TradingMetrics,
    # Main classes
    CapitalProtector,
    ProtectionManager
)

from .integration import (
    # Result types
    ProtectedTradeResult,
    # Decorators and guards
    TradingGuard,
    # Session management
    ProtectedTradingSession,
    # Integration helpers
    integrate_with_risk_manager,
    CapitalProtectionCallback
)

__all__ = [
    # Enums
    'ProtectionLevel',
    'ProtectionAction',
    # Config
    'ProtectionConfig',
    # State types
    'ProtectionState',
    'TradingMetrics',
    'ProtectedTradeResult',
    # Main classes
    'CapitalProtector',
    'ProtectionManager',
    # Integration
    'TradingGuard',
    'ProtectedTradingSession',
    'integrate_with_risk_manager',
    'CapitalProtectionCallback'
]
