"""Unit tests for `risk_management/phase2_risk_calc/hard_rules.py`.

Test Summary:
    Validates the deterministic rule checks and trade gatekeeping logic used in Phase 2.

Test Breakdown:
    - HardRulesEngine
        - Spread rule (block + warning)
        - Session/rollover/weekend rules
        - News blackout rule
        - Exposure rule adjustments (position_size reduced to 0 when limits exceeded)
    - TradeGatekeeper
        - validate_trade aggregates violations and exposes counts/adjustments
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from risk_management.phase2_risk_calc.hard_rules import (
    HardRulesConfig,
    HardRulesEngine,
    TradeGatekeeper,
    TradingSession,
)


class TestHardRulesEngine(unittest.TestCase):
    def setUp(self):
        self.config = HardRulesConfig(
            max_spread_by_pair={"EURUSD": 2.0, "DEFAULT": 5.0},
            allowed_sessions=[
                TradingSession.TOKYO,
                TradingSession.LONDON,
                TradingSession.NEW_YORK,
                TradingSession.OVERLAP_LONDON_NY,
            ],
            avoid_rollover=True,
            max_single_pair_exposure=5.0,
            max_total_exposure=20.0,
        )
        self.engine = HardRulesEngine(self.config)

    def test_spread_rule_blocks_when_exceeds_limit(self):
        violations = self.engine._check_spread_rule("EURUSD", current_spread=3.0)
        self.assertTrue(any(v.rule_name == "max_spread" and v.severity == "block" for v in violations))

    def test_spread_rule_warns_when_near_limit(self):
        violations = self.engine._check_spread_rule("EURUSD", current_spread=1.7)
        self.assertTrue(any(v.rule_name == "high_spread_warning" and v.severity == "warning" for v in violations))

    def test_weekend_rule_blocks(self):
        # Saturday
        t = datetime(2025, 1, 4, 12, 0, 0)
        violations = self.engine._check_weekend_rule(t)
        self.assertTrue(any(v.rule_name == "weekend_close" and v.severity == "block" for v in violations))

    def test_rollover_rule_blocks_in_window(self):
        # Default rollover window is 21:45-22:15
        t = datetime(2025, 1, 1, 22, 0, 0)
        violations = self.engine._check_rollover_rule(t)
        self.assertTrue(any(v.rule_name == "rollover_avoidance" and v.severity == "block" for v in violations))

    def test_news_blackout_blocks_before_event(self):
        now = datetime.utcnow()
        self.engine.add_news_event(event_time=now + timedelta(minutes=10), title="CPI")
        violations = self.engine._check_news_blackout(now)
        self.assertTrue(any(v.rule_name == "news_blackout" and v.severity == "block" for v in violations))

    def test_exposure_rule_blocks_when_pair_limit_reached(self):
        # 5% exposure already used.
        account_balance = 10_000.0
        self.engine.update_positions({"EURUSD": {"value": 500.0, "direction": "BUY"}})

        allowed, violations, adjustments = self.engine.check_all_rules(
            pair="EURUSD",
            direction="BUY",
            position_size=0.01,  # 0.01 lots at price 1.0 => ~1000 value => 10% exposure
            entry_price=1.0,
            current_spread=1.0,
            account_balance=account_balance,
            current_time=datetime(2025, 1, 1, 10, 0, 0),
        )

        self.assertFalse(allowed)
        self.assertTrue(any(v.rule_name == "max_pair_exposure" and v.severity == "block" for v in violations))
        self.assertEqual(adjustments.get("position_size"), 0)


class TestTradeGatekeeper(unittest.TestCase):
    def test_validate_trade_returns_expected_keys(self):
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
            regime=None,
        )

        self.assertIn("allowed", result)
        self.assertIn("violations", result)
        self.assertIn("warnings_count", result)
        self.assertIn("blocks_count", result)
        self.assertIn("adjustments", result)


if __name__ == "__main__":
    unittest.main()
