"""Unit tests for `risk_management/phase1_predictive/training.py`.

Test Summary:
    Validates core loss functions, dataset sequencing, and metric computation.

Test Breakdown:
    - DirectionLoss / VolatilityLoss / QuantileLoss
        - returns finite scalar
    - RiskDataset
        - length and item schema
    - compute_metrics
        - returns expected metric keys

Notes:
    These tests are skipped if `torch` is not available.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestPhase1Training(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.mod = import_module("risk_management.phase1_predictive.training")
            cls.torch = import_module("torch")
            # If tests are running with the lightweight fake torch stub, skip.
            if getattr(cls.torch, "__version__", "").startswith("0.0-fake"):
                raise unittest.SkipTest("Skipping Phase1 training tests (fake torch stub)")
            required = ["softmax", "randn", "tensor", "float32", "long", "isfinite"]
            missing = [name for name in required if not hasattr(cls.torch, name)]
            if missing:
                raise unittest.SkipTest(f"Skipping Phase1 training tests (torch missing: {missing})")
        except Exception as e:
            raise unittest.SkipTest(f"Skipping Phase1 training tests (missing torch): {e}")

    def test_losses_return_scalar(self):
        DirectionLoss = self.mod.DirectionLoss
        VolatilityLoss = self.mod.VolatilityLoss
        QuantileLoss = self.mod.QuantileLoss

        # Direction
        dl = DirectionLoss()
        pred_probs = self.torch.softmax(self.torch.randn(4, 3), dim=-1)
        target_dir = self.torch.tensor([0, 1, 2, 1], dtype=self.torch.long)
        loss_dir = dl(pred_probs, target_dir)
        self.assertTrue(self.torch.isfinite(loss_dir).item())

        # Volatility
        vl = VolatilityLoss()
        pred_vol = self.torch.tensor([0.1, 0.2, 0.15, 0.05])
        true_vol = self.torch.tensor([0.1, 0.25, 0.10, 0.05])
        loss_vol = vl(pred_vol, true_vol)
        self.assertTrue(self.torch.isfinite(loss_vol).item())

        # Quantiles
        ql = QuantileLoss()
        pred_q = self.torch.randn(4, 5)
        true_move = self.torch.randn(4)
        loss_q = ql(pred_q, true_move)
        self.assertTrue(self.torch.isfinite(loss_q).item())

    def test_risk_dataset_len_and_item(self):
        RiskDataset = self.mod.RiskDataset

        features = np.random.randn(100, 8).astype(np.float32)
        direction = np.random.randint(0, 3, size=100)
        vol = np.random.rand(100).astype(np.float32)
        move = np.random.randn(100).astype(np.float32)

        ds = RiskDataset(features, direction, vol, move, sequence_length=10)
        self.assertEqual(len(ds), 100 - 10)

        seq, targets, vision = ds[0]
        self.assertEqual(tuple(seq.shape), (10, 8))
        self.assertIn("direction", targets)
        self.assertIn("volatility", targets)
        self.assertIn("price_move", targets)
        self.assertIsNone(vision)

    def test_compute_metrics_contains_keys(self):
        compute_metrics = self.mod.compute_metrics

        preds = {
            "direction": self.torch.softmax(self.torch.randn(6, 3), dim=-1),
            "volatility": self.torch.rand(6),
            "quantiles": self.torch.randn(6, 5),
        }
        targets = {
            "direction": self.torch.tensor([0, 1, 2, 1, 0, 2], dtype=self.torch.long),
            "volatility": self.torch.rand(6),
            "price_move": self.torch.randn(6),
        }

        metrics = compute_metrics(preds, targets)
        self.assertIn("direction_accuracy", metrics)
        self.assertIn("volatility_mae", metrics)
        self.assertIn("prediction_interval_width", metrics)


if __name__ == "__main__":
    unittest.main()
