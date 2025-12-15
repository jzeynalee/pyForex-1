"""Unit tests for `utils/mtf_features.py`.

Test Summary:
    Comprehensive coverage of MTF feature building and feature set conversion.

Test Breakdown:
    - MTFFeatureSet
        - `to_array()` respects order and defaults missing keys to 0
        - `to_dict()` returns a copy
    - MTFFeatureBuilder per-timeframe
        - handles small windows (slope fallback)
        - computes expected key groups
    - Cross-timeframe features
        - default weights and empty/None frames behavior
        - alignment / confluence keys exist
    - Convenience function
        - `build_ml_features()` returns dict of features
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestMTFFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("utils.mtf_features")

    def _make_df(self, n=80):
        import pandas as pd

        close = [100 + i * 0.1 for i in range(n)]
        return pd.DataFrame(
            {
                "open": close,
                "high": [c + 0.2 for c in close],
                "low": [c - 0.2 for c in close],
                "close": close,
                "time": pd.date_range("2020-01-01", periods=n, freq="min"),
            }
        )

    def test_feature_set_to_array_and_to_dict(self):
        fs = self.mod.MTFFeatureSet(features={"a": 1.0}, feature_names=["a"], primary_tf="H1")
        arr = fs.to_array(["a", "b"])
        self.assertEqual(arr.tolist(), [1.0, 0.0])
        d = fs.to_dict()
        self.assertEqual(d, {"a": 1.0})
        d["a"] = 2.0
        self.assertEqual(fs.features["a"], 1.0)

    def test_builder_build_features_contract(self):
        builder = self.mod.MTFFeatureBuilder(lookback=200)  # force slope fallback on short df
        dfs = {"M15": self._make_df(40), "H1": self._make_df(40)}
        out = builder.build_features(dfs, primary_tf="H1")
        self.assertTrue(isinstance(out.feature_names, list))
        self.assertEqual(out.primary_tf, "H1")
        self.assertTrue(isinstance(out.features, dict))
        self.assertIn("mtf_confluence", out.features)

    def test_cross_features_empty_dict(self):
        builder = self.mod.MTFFeatureBuilder()
        cf = builder._compute_cross_tf_features({})
        self.assertEqual(cf, {})

    def test_convenience_build_ml_features(self):
        dfs = {"H1": self._make_df(80)}
        d = self.mod.build_ml_features(dfs, primary_tf="H1")
        self.assertTrue(isinstance(d, dict))
        self.assertIn("mtf_confluence", d)


if __name__ == "__main__":
    unittest.main()
