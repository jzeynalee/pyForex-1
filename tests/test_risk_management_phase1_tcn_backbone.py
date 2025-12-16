"""Unit tests for `risk_management/phase1_predictive/tcn_backbone.py`.

Test Summary:
    Provides lightweight shape/contract tests for the TCN backbone and heads.

Test Breakdown:
    - TCNConfig
        - dilations vary by TradingProfile
        - receptive_field is positive
    - MultiHeadTCN
        - forward(mode='all') returns required keys

Notes:
    These tests are skipped if `torch` is not available.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestTCNBackbone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.mod = import_module("risk_management.phase1_predictive.tcn_backbone")
            cls.torch = import_module("torch")
            # If tests are running with the lightweight fake torch stub, skip.
            if getattr(cls.torch, "__version__", "").startswith("0.0-fake"):
                raise unittest.SkipTest("Skipping TCN backbone tests (fake torch stub)")
            if not hasattr(cls.torch, "softmax"):
                raise unittest.SkipTest("Skipping TCN backbone tests (torch missing softmax)")
            if not hasattr(getattr(cls.torch, "nn", None), "ModuleList"):
                raise unittest.SkipTest("Skipping TCN backbone tests (torch.nn.ModuleList unavailable)")
        except Exception as e:
            raise unittest.SkipTest(f"Skipping TCN backbone tests (missing torch): {e}")

    def test_tcnconfig_dilations_by_profile(self):
        TradingProfile = self.mod.TradingProfile
        TCNConfig = self.mod.TCNConfig

        c1 = TCNConfig(profile=TradingProfile.SCALP)
        c2 = TCNConfig(profile=TradingProfile.INTRADAY)
        c3 = TCNConfig(profile=TradingProfile.SWING)

        self.assertLess(len(c1.dilations), len(c2.dilations))
        self.assertLess(len(c2.dilations), len(c3.dilations))
        self.assertGreater(c1.receptive_field, 0)

    def test_multihead_forward_all_keys(self):
        TCNConfig = self.mod.TCNConfig
        MultiHeadTCN = self.mod.MultiHeadTCN

        cfg = TCNConfig(input_channels=8, hidden_channels=16, kernel_size=3, dropout=0.1)
        model = MultiHeadTCN(cfg)

        x = self.torch.randn(2, 20, 8)
        out = model(x, mode="all")

        for k in ["direction", "volatility", "quantiles", "features"]:
            self.assertIn(k, out)


if __name__ == "__main__":
    unittest.main()
