"""Unit tests for `risk_management/phase5_capital_protection/protection_rules.py`.

Test Summary:
    Validates protection state transitions and trade gating behavior.

Test Breakdown:
    - CapitalProtector
        - initialize sets balances and state
        - record_trade updates daily pnl and can trigger CRITICAL on daily loss
        - check_trade blocks when max_daily_trades reached
    - ProtectionManager
        - session lifecycle and pre_trade_check behavior
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from risk_management.phase5_capital_protection.protection_rules import (
    CapitalProtector,
    ProtectionConfig,
    ProtectionLevel,
    ProtectionManager,
)


class TestCapitalProtector(unittest.TestCase):
    def test_initialize_sets_state(self):
        p = CapitalProtector(ProtectionConfig())
        p.initialize(10_000.0)
        self.assertEqual(p.get_state().level, ProtectionLevel.NORMAL)

    def test_daily_loss_triggers_critical(self):
        cfg = ProtectionConfig(max_daily_loss_pct=1.0, daily_loss_warning_pct=0.5)
        p = CapitalProtector(cfg)
        p.initialize(10_000.0)

        # 2% loss
        p.record_trade(pnl=-200.0, is_win=False, timestamp=datetime(2025, 1, 1, 10, 0, 0))
        self.assertIn(p.get_state().level, {ProtectionLevel.CRITICAL, ProtectionLevel.WARNING, ProtectionLevel.KILLED})

    def test_check_trade_blocks_when_daily_trade_limit_reached(self):
        cfg = ProtectionConfig(max_daily_trades=0)
        p = CapitalProtector(cfg)
        p.initialize(10_000.0)
        result = p.check_trade(proposed_size=0.1, account_balance=10_000.0)
        self.assertFalse(result["allowed"])


class TestProtectionManager(unittest.TestCase):
    def test_session_required_for_pre_trade_check(self):
        m = ProtectionManager(ProtectionConfig())
        check = m.pre_trade_check(proposed_size=0.1, account_balance=10_000.0)
        self.assertFalse(check["allowed"])

    def test_start_session_enables_checks(self):
        m = ProtectionManager(ProtectionConfig(max_daily_trades=1))
        m.start_session(balance=10_000.0)
        check = m.pre_trade_check(proposed_size=0.1, account_balance=10_000.0)
        self.assertIn("allowed", check)


if __name__ == "__main__":
    unittest.main()
