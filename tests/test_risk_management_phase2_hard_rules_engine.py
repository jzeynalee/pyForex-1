"""Unit tests for `risk_management/phase2_risk_calc/hard_rules_engine.py`.

Test Summary:
    Provides smoke + behavior tests for the alternate hard rules implementation
    in `hard_rules_engine.py`.

Test Breakdown:
    - HardRulesEngine
        - check_all_rules returns expected tuple types
        - spread rule blocks when above configured limits
    - TradeGatekeeper
        - validate_trade returns expected response schema
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from risk_management.phase2_risk_calc.hard_rules_engine import (
    HardRulesConfig,
    HardRulesEngine,
    TradeGatekeeper,
)


class TestHardRulesEngineAlt(unittest.TestCase):
    def test_spread_blocks(self):
        cfg = HardRulesConfig(max_spread_by_pair={"EURUSD": 2.0, "DEFAULT": 5.0})
        engine = HardRulesEngine(cfg)
        v = engine._check_spread_rule("EURUSD", 2.5)
        self.assertTrue(any(x.rule_name == "max_spread" and x.severity == "block" for x in v))

    def test_check_all_rules_returns_expected_types(self):
        engine = HardRulesEngine()
        allowed, violations, adjusted = engine.check_all_rules(
            pair="EURUSD",
            direction="BUY",
            position_size=0.01,
            entry_price=1.0,
            current_spread=1.0,
            account_balance=10_000.0,
            current_time=datetime(2025, 1, 1, 10, 0, 0),
        )

        self.assertIsInstance(allowed, bool)
        self.assertIsInstance(violations, list)
        self.assertIsInstance(adjusted, dict)


class TestTradeGatekeeperAlt(unittest.TestCase):
    def test_validate_trade_schema(self):
        gatekeeper = TradeGatekeeper()
        result = gatekeeper.validate_trade(
            pair="EURUSD",
            direction="BUY",
            position_size=0.01,
            entry_price=1.0,
            stop_loss=0.99,
            take_profit=1.02,
            account_balance=10_000.0,
            current_spread=1.0,
            current_time=datetime(2025, 1, 1, 10, 0, 0),
        )

        self.assertIn("allowed", result)
        self.assertIn("violations", result)
        self.assertIn("warnings_count", result)
        self.assertIn("blocks_count", result)
        self.assertIn("adjustments", result)


if __name__ == "__main__":
    unittest.main()
