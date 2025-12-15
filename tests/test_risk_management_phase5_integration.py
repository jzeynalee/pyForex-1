"""Unit tests for `risk_management/phase5_capital_protection/integration.py`.

Test Summary:
    Validates decorator-based protection wrappers and RiskManager patching helper.

Test Breakdown:
    - TradingGuard
        - protect_entry blocks trade execution when protection denies
    - ProtectedTradingSession
        - can_trade reflects protection rules
    - integrate_with_risk_manager
        - patches evaluate_trade_opportunity and can block trades
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from risk_management.phase5_capital_protection.integration import (
    ProtectedTradingSession,
    TradingGuard,
    integrate_with_risk_manager,
)
from risk_management.phase5_capital_protection.protection_rules import ProtectionConfig


class TestTradingGuard(unittest.TestCase):
    def test_protect_entry_blocks(self):
        cfg = ProtectionConfig(max_daily_trades=0)
        guard = TradingGuard(cfg)
        guard.initialize(balance=10_000.0)

        @guard.protect_entry
        def open_trade(size: float, **kwargs):
            return {"success": True, "ticket": 123, "size": kwargs.get("adjusted_size", size)}

        res = open_trade(size=0.5, balance=10_000.0)
        self.assertFalse(res["success"])
        self.assertTrue(res.get("blocked"))


class TestProtectedTradingSession(unittest.TestCase):
    def test_can_trade_reflects_protection(self):
        cfg = ProtectionConfig(max_daily_trades=0)
        with ProtectedTradingSession(balance=10_000.0, config=cfg) as session:
            self.assertFalse(session.can_trade(proposed_size=0.1))


class TestIntegrateWithRiskManager(unittest.TestCase):
    def test_integrate_blocks_trade_when_protection_blocks(self):
        class _Decision:
            def __init__(self):
                self.should_trade = True
                self.rejection_reasons = []
                self.position_size = 0.5

        class _RM:
            def evaluate_trade_opportunity(self, *args, **kwargs):
                return _Decision()

        rm = _RM()
        cfg = ProtectionConfig(max_daily_trades=0)
        rm2 = integrate_with_risk_manager(rm, protection_config=cfg)
        decision = rm2.evaluate_trade_opportunity(account_balance=10_000.0)
        self.assertFalse(decision.should_trade)
        self.assertTrue(any("Capital protection" in r for r in decision.rejection_reasons))


if __name__ == "__main__":
    unittest.main()
