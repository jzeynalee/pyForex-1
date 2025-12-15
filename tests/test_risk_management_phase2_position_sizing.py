"""Unit tests for `risk_management/phase2_risk_calc/position_sizing.py`.

Test Summary:
    Validates position sizing math, adjustment factors, and basic utility helpers.

Test Breakdown:
    - PositionSizingCalculator
        - base calculation returns non-negative position size
        - confidence scaling reduces risk for low confidence
        - price-to-pips conversion for JPY vs non-JPY pairs
    - ScaledPositionCalculator
        - scaled entry weights sum validation
        - produces one output per entry price
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from risk_management.phase2_risk_calc.position_sizing import (
    PositionSizingCalculator,
    PositionSizingConfig,
    ScaledPositionCalculator,
)


class TestPositionSizingCalculator(unittest.TestCase):
    def test_calculate_base_non_negative(self):
        calc = PositionSizingCalculator(PositionSizingConfig(base_risk_percent=1.0))
        res = calc.calculate(
            account_balance=10_000.0,
            entry_price=1.2000,
            stop_loss=1.1900,
            pair="EURUSD",
        )
        self.assertGreaterEqual(res.position_size, 0.0)
        self.assertGreaterEqual(res.units, 0)
        self.assertIsInstance(res.adjustment_factors, dict)
        self.assertIsInstance(res.warnings, list)

    def test_confidence_scaling_reduces_risk(self):
        cfg = PositionSizingConfig(
            base_risk_percent=1.0,
            confidence_scaling=True,
            min_confidence_for_full_size=0.7,
            low_confidence_risk_reduction=0.5,
        )
        calc = PositionSizingCalculator(cfg)

        high = calc.calculate(
            account_balance=10_000.0,
            entry_price=1.2000,
            stop_loss=1.1900,
            pair="EURUSD",
            direction_confidence=0.9,
        )
        low = calc.calculate(
            account_balance=10_000.0,
            entry_price=1.2000,
            stop_loss=1.1900,
            pair="EURUSD",
            direction_confidence=0.1,
        )

        self.assertLessEqual(low.position_size, high.position_size)
        self.assertIn("confidence", low.adjustment_factors)

    def test_price_to_pips_jpy_vs_non_jpy(self):
        calc = PositionSizingCalculator()
        self.assertAlmostEqual(calc._price_to_pips(0.01, "USDJPY"), 1.0)
        self.assertAlmostEqual(calc._price_to_pips(0.0001, "EURUSD"), 1.0)


class TestScaledPositionCalculator(unittest.TestCase):
    def test_calculate_scaled_entries_outputs_match_inputs(self):
        spc = ScaledPositionCalculator(total_risk_percent=2.0)
        entries = spc.calculate_scaled_entries(
            account_balance=10_000.0,
            entry_prices=[1.2000, 1.1990, 1.1980],
            stop_loss=1.1900,
            pair="EURUSD",
            scaling_weights=[0.5, 0.3, 0.2],
        )

        self.assertEqual(len(entries), 3)
        self.assertAlmostEqual(sum(e["weight"] for e in entries), 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
