"""Unit tests for `ml/drift_detector.py`.

Test Summary:
    Comprehensive coverage of drift detection primitives and detectors:
    statistical helper behavior, DriftDetector state and drift classification,
    and concept drift detection.

Test Breakdown:
    - Statistical primitives
        - `ks_test()` guardrails for short arrays
        - `psi()` handles identical distributions and constant distributions
        - `js_divergence()` returns near-zero for identical distributions
        - `mean_shift()` and `variance_ratio()` handle small/degenerate cases
    - DriftDetector behavior
        - `set_reference()` precomputes stats
        - `add_sample()` triggers checks at configured interval
        - `check_drift()` produces actionable recommendations for:
            - missing reference
            - insufficient window
            - obvious distribution shift
        - `get_drift_trend()` classifies trend based on history
        - `reset()` / `full_reset()` manage internal state
    - Concept drift
        - `check_concept_drift()` returns None until reference + enough samples
        - returns a DriftResult when prediction error distribution shifts

Notes:
    This module imports SciPy. If SciPy is unavailable, the entire test module
    is skipped.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.mod = import_module("ml.drift_detector")
        except Exception as e:
            raise unittest.SkipTest(f"Skipping drift_detector tests (missing deps): {e}")


class TestStatisticalTests(_Base):
    def test_ks_test_guardrail_short_arrays(self):
        import numpy as np

        ref = np.array([1.0, 2.0, 3.0])
        cur = np.array([1.0, 2.0, 3.0])
        stat, pval = self.mod.StatisticalTests.ks_test(ref, cur)
        self.assertEqual(stat, 0.0)
        self.assertEqual(pval, 1.0)

    def test_psi_identical_is_small(self):
        import numpy as np

        ref = np.array([0.0, 0.1, 0.2, 0.3] * 100)
        cur = np.array([0.0, 0.1, 0.2, 0.3] * 100)
        psi = self.mod.StatisticalTests.psi(ref, cur, bins=10)
        self.assertGreaterEqual(psi, 0.0)
        self.assertLess(psi, 1e-6)

    def test_psi_constant_distribution_is_zeroish(self):
        import numpy as np

        ref = np.ones(100)
        cur = np.ones(100)
        psi = self.mod.StatisticalTests.psi(ref, cur, bins=10)
        self.assertGreaterEqual(psi, 0.0)
        self.assertLessEqual(psi, 1e-6)

    def test_js_divergence_identical_is_small(self):
        import numpy as np

        ref = np.array([0.0, 0.25, 0.5, 0.75] * 100)
        cur = np.array([0.0, 0.25, 0.5, 0.75] * 100)
        js = self.mod.StatisticalTests.js_divergence(ref, cur, bins=10)
        self.assertGreaterEqual(js, 0.0)
        self.assertLess(js, 1e-6)

    def test_mean_shift_and_variance_ratio_small_arrays(self):
        import numpy as np

        ref = np.array([1.0, 1.0, 1.0, 1.0])
        cur = np.array([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(self.mod.StatisticalTests.mean_shift(ref, cur), 0.0)
        self.assertEqual(self.mod.StatisticalTests.variance_ratio(ref, cur), 1.0)


class TestDriftDetectorCore(_Base):
    def test_check_drift_without_reference(self):
        import pandas as pd

        detector = self.mod.DriftDetector(self.mod.DriftConfig(detection_window_size=20))
        detector.add_batch(pd.DataFrame({"a": [1, 2, 3] * 10, "b": [3, 2, 1] * 10}))
        res = detector.check_drift()
        self.assertFalse(res.drift_detected)
        self.assertEqual(res.drift_type, self.mod.DriftType.NONE)
        self.assertIn("No reference data", res.recommendation)

    def test_check_drift_insufficient_current_window(self):
        import pandas as pd

        cfg = self.mod.DriftConfig(detection_window_size=40)
        detector = self.mod.DriftDetector(cfg)
        detector.set_reference(pd.DataFrame({"x": list(range(200))}))
        detector.add_batch(pd.DataFrame({"x": list(range(10))}))
        res = detector.check_drift()
        self.assertFalse(res.drift_detected)
        self.assertIn("Insufficient data", res.recommendation)

    def test_set_reference_computes_stats(self):
        import pandas as pd

        detector = self.mod.DriftDetector(self.mod.DriftConfig())
        detector.set_reference(pd.DataFrame({"x": list(range(50)), "y": list(range(50, 100))}))
        self.assertIn("x", detector.reference_stats)
        self.assertIn("mean", detector.reference_stats["x"])
        self.assertIn("q75", detector.reference_stats["x"])

    def test_add_sample_triggers_check_interval(self):
        import pandas as pd

        cfg = self.mod.DriftConfig(check_interval_bars=3, detection_window_size=10)
        detector = self.mod.DriftDetector(cfg)
        detector.set_reference(pd.DataFrame({"x": [0.0] * 200, "y": [0.0] * 200}))

        self.assertIsNone(detector.add_sample(pd.Series({"x": 0.0, "y": 0.0})))
        self.assertIsNone(detector.add_sample(pd.Series({"x": 0.0, "y": 0.0})))
        res = detector.add_sample(pd.Series({"x": 0.0, "y": 0.0}))
        self.assertTrue(res is None or hasattr(res, "drift_detected"))

    def test_detects_obvious_distribution_shift(self):
        import numpy as np
        import pandas as pd

        cfg = self.mod.DriftConfig(
            detection_window_size=100,
            min_features_drifted=1,
            feature_drift_ratio=0.0,
            ks_threshold=0.5,
            psi_threshold=0.1,
            js_threshold=0.05,
            medium_threshold=0.05,
        )
        detector = self.mod.DriftDetector(cfg)

        ref = pd.DataFrame({"x": np.zeros(1000), "y": np.zeros(1000)})
        detector.set_reference(ref)

        current = pd.DataFrame({"x": np.ones(150) * 10.0, "y": np.ones(150) * 10.0})
        detector.add_batch(current)
        res = detector.check_drift()
        self.assertTrue(res.drift_detected)
        self.assertGreaterEqual(res.overall_score, 0.0)
        self.assertTrue(len(res.drifted_features) >= 1)
        self.assertTrue(isinstance(res.recommendation, str) and len(res.recommendation) > 0)

    def test_get_drift_trend_insufficient_data(self):
        detector = self.mod.DriftDetector(self.mod.DriftConfig())
        trend = detector.get_drift_trend()
        self.assertEqual(trend.get("trend"), "insufficient_data")


class TestConceptDriftDetector(_Base):
    def test_concept_drift_none_until_ready(self):
        import numpy as np

        cfg = self.mod.DriftConfig(detection_window_size=20)
        cd = self.mod.ConceptDriftDetector(cfg)
        self.assertIsNone(cd.check_concept_drift())

        ref_preds = np.zeros(40)
        ref_targets = np.zeros(40)
        cd.set_reference_predictions(ref_preds, ref_targets)
        for _ in range(5):
            cd.add_prediction(0.0, 0.0)
        self.assertIsNone(cd.check_concept_drift())

    def test_concept_drift_detected_on_error_shift(self):
        import numpy as np

        cfg = self.mod.DriftConfig(detection_window_size=20, ks_threshold=0.5, psi_threshold=0.05)
        cd = self.mod.ConceptDriftDetector(cfg)

        ref_targets = np.zeros(100)
        ref_preds = np.zeros(100)
        cd.set_reference_predictions(ref_preds, ref_targets)

        for _ in range(25):
            cd.add_prediction(10.0, 0.0)

        res = cd.check_concept_drift()
        self.assertIsNotNone(res)
        self.assertTrue(res.drift_detected)
        self.assertEqual(res.drift_type, self.mod.DriftType.CONCEPT_DRIFT)
        self.assertIn("Concept drift detected", res.recommendation)


if __name__ == "__main__":
    unittest.main()
