"""Unit tests for `risk_management/risk_manager.py`.

Test Summary:
    Provides a smoke-level validation of RiskManager factory construction and
    key object wiring.

Test Breakdown:
    - RiskManager.create_for_profile
        - returns RiskManager instance when dependencies are importable

Notes:
    This module can be heavy (torch + ML components). Tests are skipped if
    the module cannot be imported in the current environment.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestRiskManager(unittest.TestCase):
    def test_create_for_profile_smoke(self):
        try:
            rm_mod = import_module("risk_management.risk_manager")
            torch = import_module("torch")
            if getattr(torch, "__version__", "").startswith("0.0-fake"):
                raise unittest.SkipTest("Skipping RiskManager smoke test (fake torch stub)")
            if not hasattr(getattr(torch, "nn", None), "ModuleList"):
                raise unittest.SkipTest("Skipping RiskManager smoke test (torch.nn.ModuleList unavailable)")
        except Exception as e:
            raise unittest.SkipTest(f"Skipping RiskManager import (missing deps): {e}")

        RiskManager = rm_mod.RiskManager
        mgr = RiskManager.create_for_profile("INTRADAY", input_features=8, sequence_length=10)
        self.assertEqual(mgr.config.profile, "INTRADAY")
        self.assertEqual(mgr.config.input_features, 8)


if __name__ == "__main__":
    unittest.main()
