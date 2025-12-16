"""Unit tests for `utils/feature_adapter.py`.

Test Summary:
    Covers feature engineering and the EnhancedDataLoaderV2/V3 pipelines,
    including checkpoint-based feature selection and key failure modes.

Test Breakdown:
    - FeatureEngineer
        - `add_all_features()` normalizes columns and produces non-empty feature set
        - fills NaNs deterministically
    - EnhancedDataLoaderV2
        - `load_csv()` sets feature_columns excluding OHLCV/meta
        - `split_and_scale()` returns correctly shaped arrays
        - `create_sequences()` returns aligned X/y with correct label domain
    - EnhancedDataLoaderV3
        - `from_checkpoint()` uses `utils.checkpoint_loader.load_features` when available
        - `load_csv()` inserts missing checkpoint features with zeros
    - Convenience functions
        - `get_available_features()` returns list

Notes:
    Uses lightweight CSV fixtures in a temp directory.
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


class TestFeatureAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("utils.feature_adapter")

    def _write_csv(self, path: Path, n=120):
        import pandas as pd

        close = [100 + i * 0.01 for i in range(n)]
        df = pd.DataFrame(
            {
                "Open": close,
                "High": [c + 0.1 for c in close],
                "Low": [c - 0.1 for c in close],
                "Close": close,
                "Volume": [1000] * n,
            }
        )
        df.to_csv(path, index=False)
        return df

    def test_feature_engineer_add_all_features_contract(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "OPEN": [1, 1.1, 1.2, 1.3] * 20,
                "HIGH": [1.1, 1.2, 1.3, 1.4] * 20,
                "LOW": [0.9, 1.0, 1.1, 1.2] * 20,
                "CLOSE": [1.0, 1.1, 1.2, 1.3] * 20,
                "VOLUME": [10] * 80,
            }
        )
        out = self.mod.FeatureEngineer.add_all_features(df)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertIn("macd", out.columns)
        self.assertFalse(out.isna().any().any())

    def test_enhanced_loader_v2_load_csv_and_features(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "d.csv"
            self._write_csv(p)

            loader = self.mod.EnhancedDataLoaderV2(sequence_length=10)
            df = loader.load_csv(str(p))
            self.assertTrue(len(loader.feature_columns) > 0)
            self.assertNotIn("open", loader.feature_columns)
            self.assertNotIn("close", loader.feature_columns)
            self.assertIn("close", df.columns)

    def test_enhanced_loader_v2_split_and_scale_shapes(self):
        import numpy as np

        with TemporaryDirectory() as td:
            p = Path(td) / "d.csv"
            self._write_csv(p)

            loader = self.mod.EnhancedDataLoaderV2(sequence_length=10)
            df = loader.load_csv(str(p))
            train_scaled, val_scaled, test_scaled = loader.split_and_scale(df, split_ratio=0.7, validation_ratio=0.2)
            self.assertTrue(isinstance(train_scaled, np.ndarray))
            self.assertGreater(train_scaled.shape[0], 0)
            self.assertGreater(val_scaled.shape[0], 0)
            self.assertGreater(test_scaled.shape[0], 0)
            self.assertEqual(train_scaled.shape[1], val_scaled.shape[1])

    def test_enhanced_loader_v2_create_sequences_labels_domain(self):
        import numpy as np

        with TemporaryDirectory() as td:
            p = Path(td) / "d.csv"
            self._write_csv(p)

            loader = self.mod.EnhancedDataLoaderV2(sequence_length=10, trend_threshold=0.0001)
            df = loader.load_csv(str(p))
            train_scaled, _val_scaled, _test_scaled = loader.split_and_scale(df, split_ratio=0.8, validation_ratio=0.1)
            X, y = loader.create_sequences(train_scaled, loader.train_close, seq_len=10, horizon=1)
            self.assertTrue(isinstance(X, np.ndarray))
            self.assertTrue(isinstance(y, np.ndarray))
            self.assertEqual(X.shape[0], y.shape[0])
            self.assertTrue(set(np.unique(y)).issubset({0, 1, 2}))

    def test_loader_v3_checkpoint_features_missing_filled_with_zero(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "d.csv"
            self._write_csv(p)

            with patch("utils.checkpoint_loader.load_features", return_value=["does_not_exist", "macd"]):
                loader = self.mod.EnhancedDataLoaderV3.from_checkpoint("dummy.pt", sequence_length=10)
                df = loader.load_csv(str(p))
                self.assertIn("does_not_exist", df.columns)
                self.assertEqual(float(df["does_not_exist"].iloc[-1]), 0.0)
                self.assertIn("does_not_exist", loader.get_feature_columns())

    def test_get_available_features_returns_list(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "d.csv"
            self._write_csv(p)
            feats = self.mod.get_available_features(str(p))
            self.assertTrue(isinstance(feats, list))
            self.assertTrue(len(feats) > 0)


if __name__ == "__main__":
    unittest.main()
