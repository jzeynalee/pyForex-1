"""Unit tests for `risk_management/phase3_filtering/meta_labeling.py`.

Test Summary:
    Validates feature extraction, meta-label generation, lightweight model training,
    and high-level trade filtering behavior.

Test Breakdown:
    - MetaLabelingModel
        - create_meta_labels marks only (direction != 0 and outcome == WIN) as 1
        - train returns metrics dict (sklearn fallback)
    - MetaFeatureExtractor
        - extract_features returns correct shape and feature name list
    - TradeFilter
        - filter_signals combines confidence + meta-score thresholds
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from risk_management.phase3_filtering.meta_labeling import (
    MetaFeatureExtractor,
    MetaLabelingConfig,
    MetaLabelingModel,
    TradeFilter,
)


class TestMetaLabelingModel(unittest.TestCase):
    def test_create_meta_labels(self):
        model = MetaLabelingModel(MetaLabelingConfig(use_lightgbm=False))
        primary_dirs = np.array([1, 1, 0, -1])
        barrier = np.array([1, -1, 1, 1])
        y = model.create_meta_labels(primary_dirs, barrier)
        np.testing.assert_array_equal(y, np.array([1, 0, 0, 1]))

    def test_train_sklearn_fallback_returns_metrics(self):
        cfg = MetaLabelingConfig(
            use_lightgbm=False,
            n_estimators=10,
            max_depth=2,
            n_cv_splits=2,
            default_threshold=0.5,
        )
        model = MetaLabelingModel(cfg)

        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 6))
        y = rng.integers(0, 2, size=(80,))

        metrics = model.train(X, y, validation_split=0.25)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            self.assertIn(key, metrics)


class TestMetaFeatureExtractor(unittest.TestCase):
    def test_extract_features_shape_and_names(self):
        cfg = MetaLabelingConfig(use_lightgbm=False)
        extractor = MetaFeatureExtractor(cfg)

        n = 50
        primary_predictions = {
            "direction_probs": np.tile(np.array([[0.2, 0.3, 0.5]]), (n, 1)),
            "volatility": np.linspace(0.1, 0.2, n),
            "quantiles": np.tile(np.array([[ -0.001, -0.0005, 0.0, 0.0005, 0.001 ]]), (n, 1)),
        }
        market_data = pd.DataFrame(
            {
                "spread": np.full(n, 1.2),
                "atr": np.full(n, 0.001),
                "volume": np.arange(n) + 1,
            }
        )
        ts = pd.date_range("2025-01-01", periods=n, freq="h")

        X = extractor.extract_features(primary_predictions, market_data, timestamps=ts)
        self.assertEqual(X.shape[0], n)
        self.assertGreater(X.shape[1], 3)
        self.assertGreater(len(extractor.get_feature_names()), 0)


class TestTradeFilter(unittest.TestCase):
    def test_filter_signals_combines_thresholds(self):
        cfg = MetaLabelingConfig(use_lightgbm=False)
        meta_model = MetaLabelingModel(cfg)

        # Fake a trained model by stubbing predict_proba.
        class _DummyModel:
            def predict_proba(self, X):
                # Return [[p0, p1]]
                p1 = np.linspace(0.0, 1.0, X.shape[0])
                return np.stack([1 - p1, p1], axis=1)

        meta_model.model = _DummyModel()

        n = 10
        primary_predictions = {
            "direction_probs": np.tile(np.array([[0.2, 0.3, 0.5]]), (n, 1))
        }
        market_data = pd.DataFrame({"spread": np.full(n, 1.0)})

        trade_filter = TradeFilter(meta_model=meta_model, min_confidence=0.4, min_meta_score=0.5)
        should_trade, scores = trade_filter.filter_signals(primary_predictions, market_data)

        self.assertEqual(len(should_trade), n)
        self.assertEqual(len(scores), n)
        self.assertTrue(should_trade[-1])
        self.assertFalse(should_trade[0])


if __name__ == "__main__":
    unittest.main()
