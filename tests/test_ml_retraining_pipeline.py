"""Unit tests for `ml/retraining_pipeline.py`.

Test Summary:
    Comprehensive coverage of deterministic and dependency-light components of
    the retraining pipeline. These tests focus on data handling and decision
    logic, avoiding full model training.

Test Breakdown:
    - Feature engineering helpers
        - `_basic_feature_engineering()` produces expected feature columns
        - `_calc_atr()` and `_calc_rsi()` return Series with correct index/length
    - Target generation
        - `_generate_target()` produces labels in {-1, 0, 1}
    - Data validation
        - `_validate_data()` fails on insufficient samples
        - `_validate_data()` fails on excessive NaN ratio
    - Splitting
        - `_split_data()` returns numeric-only feature names and non-empty splits
    - Model comparison logic
        - `_compare_models()` when no baseline
        - `_compare_models()` when baseline missing
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestRetrainingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline_mod = import_module("ml.retraining_pipeline")
        cls.cfg_mod = import_module("ml.retraining_config")
        cls.mm_mod = import_module("ml.model_manager")

    def _make_pipeline(self, models_dir: str):
        cfg = self.cfg_mod.RetrainingConfig.for_intraday()
        cfg.data.min_training_samples = 1
        cfg.data.max_missing_ratio = 1.0
        cfg.validation.min_improvement_pct = 5.0
        mgr = self.mm_mod.ModelManager(self.mm_mod.ManagerConfig(models_dir=models_dir))
        return cfg, mgr, self.pipeline_mod.RetrainingPipeline(config=cfg, model_manager=mgr)

    def _make_base_ohlc(self, n: int = 60):
        import numpy as np
        import pandas as pd

        idx = pd.date_range(end=datetime.now(), periods=n, freq="min")
        base = np.linspace(1.0, 1.03, len(idx))
        df = pd.DataFrame(
            {
                "open": base,
                "high": base + 0.001,
                "low": base - 0.001,
                "close": base + 0.0002,
                "tick_volume": np.arange(len(idx)),
                "spread": np.ones(len(idx)),
                "real_volume": np.zeros(len(idx)),
            },
            index=idx,
        )
        return df

    def test_basic_feature_engineering_creates_expected_columns(self):
        import numpy as np
        import pandas as pd

        with TemporaryDirectory() as td:
            _cfg, _mgr, pipe = self._make_pipeline(td)
            df = self._make_base_ohlc(80)
            data = {"EURUSD": {"M1": df}}

            feats = pipe._basic_feature_engineering(data)
            self.assertIsInstance(feats, pd.DataFrame)
            for col in ["returns", "log_returns", "volatility", "atr", "rsi", "momentum", "symbol"]:
                self.assertIn(col, feats.columns)
            self.assertTrue(np.isfinite(feats["atr"].fillna(0)).all())

    def test_calc_atr_and_rsi_shapes(self):
        import pandas as pd

        with TemporaryDirectory() as td:
            _cfg, _mgr, pipe = self._make_pipeline(td)
            df = self._make_base_ohlc(60)

            atr = pipe._calc_atr(df, 14)
            self.assertIsInstance(atr, pd.Series)
            self.assertEqual(len(atr), len(df))
            self.assertTrue(atr.index.equals(df.index))

            rsi = pipe._calc_rsi(df["close"], 14)
            self.assertIsInstance(rsi, pd.Series)
            self.assertEqual(len(rsi), len(df))
            self.assertTrue(rsi.index.equals(df.index))

    def test_generate_target_domain(self):
        import pandas as pd

        with TemporaryDirectory() as td:
            _cfg, _mgr, pipe = self._make_pipeline(td)
            df = self._make_base_ohlc(60)
            data = {"EURUSD": {"M1": df}}

            idx = df.index
            features_df = pd.DataFrame({"f1": range(len(idx)), "symbol": ["EURUSD"] * len(idx)}, index=idx)
            target = pipe._generate_target(features_df, data)
            self.assertTrue(set(target.unique()).issubset({-1, 0, 1}))

    def test_validate_data_fails_on_insufficient_samples(self):
        import pandas as pd

        with TemporaryDirectory() as td:
            cfg, _mgr, pipe = self._make_pipeline(td)
            cfg.data.min_training_samples = 100

            idx = pd.date_range(end=datetime.now(), periods=10, freq="min")
            features_df = pd.DataFrame({"f1": range(10)}, index=idx)
            target = pd.Series([0] * 10, index=idx)
            ok = pipe._validate_data(features_df, target)
            self.assertFalse(ok)

    def test_validate_data_fails_on_nan_ratio(self):
        import numpy as np
        import pandas as pd

        with TemporaryDirectory() as td:
            cfg, _mgr, pipe = self._make_pipeline(td)
            cfg.data.min_training_samples = 1
            cfg.data.max_missing_ratio = 0.01

            idx = pd.date_range(end=datetime.now(), periods=50, freq="min")
            features_df = pd.DataFrame({"f1": np.nan, "f2": np.nan}, index=idx)
            target = pd.Series([0] * 50, index=idx)
            ok = pipe._validate_data(features_df, target)
            self.assertFalse(ok)

    def test_split_data_numeric_only_feature_names(self):
        import pandas as pd

        with TemporaryDirectory() as td:
            _cfg, _mgr, pipe = self._make_pipeline(td)
            idx = pd.date_range(end=datetime.now(), periods=120, freq="min")
            features_df = pd.DataFrame(
                {
                    "f1": list(range(len(idx))),
                    "f2": list(reversed(range(len(idx)))),
                    "symbol": ["EURUSD"] * len(idx),
                },
                index=idx,
            )
            target = pd.Series([0] * len(idx), index=idx)

            split = pipe._split_data(features_df, target)
            self.assertGreater(split.training_samples, 0)
            self.assertGreater(split.validation_samples, 0)
            self.assertNotIn("symbol", split.feature_names)

    def test_compare_models_no_baseline(self):
        with TemporaryDirectory() as td:
            cfg, mgr, pipe = self._make_pipeline(td)
            new_metrics = {"val_accuracy": 0.5}
            passed, comp = pipe._compare_models(new_metrics, profile_name=cfg.profile.value, force=False)
            self.assertTrue(passed)
            self.assertEqual(comp.get("reason"), "no_baseline")

    def test_compare_models_baseline_not_found(self):
        with TemporaryDirectory() as td:
            cfg, mgr, pipe = self._make_pipeline(td)
            mgr.active_models[cfg.profile.value] = "missing_id"
            new_metrics = {"val_accuracy": 0.5}
            passed, comp = pipe._compare_models(new_metrics, profile_name=cfg.profile.value, force=False)
            self.assertTrue(passed)
            self.assertEqual(comp.get("reason"), "baseline_not_found")


if __name__ == "__main__":
    unittest.main()
