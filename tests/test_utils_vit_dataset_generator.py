"""Unit tests for `utils/vit_dataset_generator.py`.

Test Summary:
    Covers ViT dataset generation components: labelers and the generator's
    validation/splitting/saving logic in a lightweight manner.

Test Breakdown:
    - Labelers
        - FuturePriceLabeler: boundary cases (missing/short future_df) and threshold logic
        - TrendStructureLabeler: insufficient swings returns SIDEWAYS
        - EMABasedLabeler: insufficient history returns SIDEWAYS
    - Generator
        - input validation (missing OHLC columns)
        - generate_from_dataframe creates directories and returns stats for small sample count

Notes:
    Image rendering is patched to return a tiny deterministic array.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestViTDatasetGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("utils.vit_dataset_generator")

    def _make_df(self, n=120):
        import pandas as pd

        close = [1.0 + i * 0.0001 for i in range(n)]
        return pd.DataFrame(
            {
                "open": close,
                "high": [c + 0.0002 for c in close],
                "low": [c - 0.0002 for c in close],
                "close": close,
                "tick_volume": [100] * n,
            }
        )

    def test_future_price_labeler_boundaries(self):
        import pandas as pd

        labeler = self.mod.FuturePriceLabeler(forward_bars=3, threshold_pct=0.5)
        cur = self._make_df(10)
        self.assertEqual(labeler.get_label(cur, None), self.mod.TrendLabel.SIDEWAYS)
        self.assertEqual(labeler.get_label(cur, pd.DataFrame()), self.mod.TrendLabel.SIDEWAYS)

    def test_future_price_labeler_thresholds(self):
        import pandas as pd

        labeler = self.mod.FuturePriceLabeler(forward_bars=1, threshold_pct=0.5)
        cur = self._make_df(10)
        future = cur.copy()
        future.loc[0, "close"] = cur["close"].iloc[-1] * 1.01
        self.assertEqual(labeler.get_label(cur, future), self.mod.TrendLabel.BULLISH)

        future2 = cur.copy()
        future2.loc[0, "close"] = cur["close"].iloc[-1] * 0.98
        self.assertEqual(labeler.get_label(cur, future2), self.mod.TrendLabel.BEARISH)

    def test_trend_structure_labeler_insufficient_swings(self):
        labeler = self.mod.TrendStructureLabeler(min_swings=10)
        df = self._make_df(30)
        self.assertEqual(labeler.get_label(df), self.mod.TrendLabel.SIDEWAYS)

    def test_ema_based_labeler_insufficient_history(self):
        labeler = self.mod.EMABasedLabeler(fast_period=10, slow_period=30)
        df = self._make_df(10)
        self.assertEqual(labeler.get_label(df), self.mod.TrendLabel.SIDEWAYS)

    def test_generator_missing_columns_raises(self):
        import pandas as pd

        with TemporaryDirectory() as td:
            gen = self.mod.ViTDatasetGenerator(output_dir=td, window_size=10, stride=5)
            with self.assertRaises(ValueError):
                gen.generate_from_dataframe(pd.DataFrame({"close": [1, 2, 3]}))

    def test_generate_from_dataframe_stubbed_renderer(self):
        import numpy as np

        df = self._make_df(120)

        class DummyRenderer:
            def render(self, window, include_volume=True):
                return np.zeros((16, 16, 3), dtype=np.uint8)

        with TemporaryDirectory() as td:
            gen = self.mod.ViTDatasetGenerator(output_dir=td, window_size=10, stride=10, val_split=0.5)
            gen.renderer = DummyRenderer()

            stats = gen.generate_from_dataframe(df, symbol="X", max_samples=6)
            self.assertTrue(isinstance(stats, dict))
            self.assertIn("train", stats)
            self.assertIn("val", stats)


if __name__ == "__main__":
    unittest.main()
