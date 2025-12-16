"""Unit tests for `utils/features_engineering.py`.

Test Summary:
    Focuses on deterministic, lightweight correctness checks for the optimized
    feature engineering utilities without running full 220+ feature generation
    on large datasets.

Test Breakdown:
    - Optional dependency flags
        - module imports even if optional deps are missing
        - HAS_POLARS / HAS_BOTTLENECK are booleans
    - Fast rolling helpers
        - `fast_rolling_mean/std/min/max/sum` return correct shapes and expected NaN prefix
    - I/O helpers
        - `load_parquet_fast` delegates to pandas when polars unavailable (patched)
        - `save_parquet_fast` delegates to pandas when polars unavailable (patched)
    - FeatureEngineerOptimized entrypoint
        - `generate_features` raises on missing required columns
        - `calculate_indicators` returns df unchanged when insufficient rows

Notes:
    `FeatureEngineerOptimized.__init__` triggers numba warmup; tests patch
    `_warmup_numba` to keep runtime deterministic and fast.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestFeaturesEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("utils.features_engineering")

    def test_optional_flags_are_bool(self):
        self.assertTrue(isinstance(getattr(self.mod, "HAS_POLARS"), bool))
        self.assertTrue(isinstance(getattr(self.mod, "HAS_BOTTLENECK"), bool))

    def test_fast_rolling_mean_contract(self):
        import numpy as np

        arr = np.arange(10, dtype=float)
        out = self.mod.fast_rolling_mean(arr, window=3)
        self.assertEqual(out.shape, arr.shape)
        self.assertTrue(np.isnan(out[:2]).all())
        self.assertAlmostEqual(out[2], np.mean(arr[0:3]))

    def test_fast_rolling_std_contract(self):
        import numpy as np

        arr = np.arange(10, dtype=float)
        out = self.mod.fast_rolling_std(arr, window=3)
        self.assertEqual(out.shape, arr.shape)

    def test_fast_rolling_min_max_sum_contract(self):
        import numpy as np

        arr = np.arange(10, dtype=float)
        mn = self.mod.fast_rolling_min(arr, window=3)
        mx = self.mod.fast_rolling_max(arr, window=3)
        sm = self.mod.fast_rolling_sum(arr, window=3)
        self.assertEqual(mn.shape, arr.shape)
        self.assertEqual(mx.shape, arr.shape)
        self.assertEqual(sm.shape, arr.shape)

    def test_load_save_parquet_fast_delegates_to_pandas(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})
        with patch.object(self.mod, "HAS_POLARS", False):
            with patch.object(self.mod.pd, "read_parquet", return_value=df) as rp:
                out = self.mod.load_parquet_fast("x.parquet")
                self.assertIs(out, df)
                self.assertTrue(rp.called)

            with patch.object(pd.DataFrame, "to_parquet", return_value=None) as tp:
                self.mod.save_parquet_fast(df, "x.parquet")
                self.assertTrue(tp.called)

    def test_generate_features_missing_required_columns_raises(self):
        import pandas as pd

        with patch.object(self.mod.FeatureEngineerOptimized, "_warmup_numba", return_value=None):
            fe = self.mod.FeatureEngineerOptimized(db_connector=None)
        with self.assertRaises(ValueError):
            fe.generate_features(pd.DataFrame({"close": [1, 2, 3]}), batch_processing=False)

    def test_calculate_indicators_insufficient_rows_returns_df(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "open": [1.0] * 10,
                "high": [1.1] * 10,
                "low": [0.9] * 10,
                "close": [1.0] * 10,
                "volume": [100] * 10,
            }
        )
        with patch.object(self.mod.FeatureEngineerOptimized, "_warmup_numba", return_value=None):
            fe = self.mod.FeatureEngineerOptimized(db_connector=None)
        out = fe.calculate_indicators(df)
        self.assertEqual(list(out.columns), list(df.columns))
        self.assertEqual(len(out), len(df))


if __name__ == "__main__":
    unittest.main()
