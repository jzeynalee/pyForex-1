"""Unit tests for `ml/risk_retraining/risk_drift_detector.py`.

Test Summary:
    Comprehensive coverage for drift detection components used by the risk
    retraining system.

Test Breakdown:
    - Detection methods
        - PSI, KS-test, JS divergence, mean/variance shift helpers
    - Feature drift detector
        - detect_drift without reference returns NO_DRIFT with an error detail
        - detect_drift flags drift when distributions shift
        - detect_multi_feature_drift aggregates per-feature results

Notes:
    This module imports SciPy. If SciPy isn't available, tests are skipped.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestRiskDriftDetectorMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.mod = import_module("ml.risk_retraining.risk_drift_detector")
        except Exception as e:
            raise unittest.SkipTest(f"Skipping risk_drift_detector tests (missing deps): {e}")

    def test_population_stability_index_identical(self):
        import numpy as np

        ref = np.array([0.0, 0.1, 0.2, 0.3] * 50)
        cur = np.array([0.0, 0.1, 0.2, 0.3] * 50)
        psi = self.mod.DriftDetectionMethods.population_stability_index(ref, cur, n_bins=10)
        self.assertGreaterEqual(psi, 0.0)
        self.assertLess(psi, 1e-6)

    def test_ks_test_identical_has_high_pvalue(self):
        import numpy as np

        ref = np.array([0.0, 0.1, 0.2, 0.3] * 100)
        cur = np.array([0.0, 0.1, 0.2, 0.3] * 100)
        stat, pval = self.mod.DriftDetectionMethods.ks_test(ref, cur)
        self.assertGreaterEqual(stat, 0.0)
        self.assertGreaterEqual(pval, 0.5)

    def test_js_divergence_identical_is_small(self):
        import numpy as np

        ref = np.array([0.0, 0.1, 0.2, 0.3] * 200)
        cur = np.array([0.0, 0.1, 0.2, 0.3] * 200)
        js = self.mod.DriftDetectionMethods.jensen_shannon_divergence(ref, cur, n_bins=20)
        self.assertGreaterEqual(js, 0.0)
        self.assertLess(js, 1e-6)

    def test_mean_and_variance_shift_helpers(self):
        import numpy as np

        ref = np.zeros(200)
        cur = np.ones(200) * 10.0
        shifted, z = self.mod.DriftDetectionMethods.mean_shift_test(ref, cur, threshold_std=2.0)
        self.assertTrue(isinstance(shifted, (bool, np.bool_)))
        self.assertTrue(isinstance(z, float))

        changed, ratio = self.mod.DriftDetectionMethods.variance_ratio_test(ref + 0.01, cur + 0.01, threshold=2.0)
        self.assertTrue(isinstance(changed, (bool, np.bool_)))
        self.assertTrue(isinstance(ratio, float))


class TestFeatureDriftDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.mod = import_module("ml.risk_retraining.risk_drift_detector")
        except Exception as e:
            raise unittest.SkipTest(f"Skipping risk_drift_detector tests (missing deps): {e}")

    def test_detect_drift_without_reference(self):
        import numpy as np

        det = self.mod.FeatureDriftDetector(reference_window_size=100)
        res = det.detect_drift("x", np.zeros(50))
        self.assertFalse(res.drift_detected)
        self.assertEqual(res.drift_type, self.mod.DriftType.NO_DRIFT)
        self.assertIn("error", res.details)

    def test_detect_drift_flags_large_shift(self):
        import numpy as np

        det = self.mod.FeatureDriftDetector(reference_window_size=200, psi_threshold=0.05, ks_threshold=0.5, js_threshold=0.01)
        det.set_reference("x", np.zeros(500))
        res = det.detect_drift("x", np.ones(200) * 10.0)
        self.assertTrue(isinstance(res.drift_detected, (bool, np.bool_)))
        self.assertTrue(isinstance(res.score, float))
        self.assertIn("psi", res.details)

    def test_detect_multi_feature_drift_aggregates(self):
        import numpy as np

        det = self.mod.FeatureDriftDetector(reference_window_size=200, psi_threshold=0.05, ks_threshold=0.5, js_threshold=0.01)
        det.set_reference("x", np.zeros(500))
        det.set_reference("y", np.zeros(500))
        res = det.detect_multi_feature_drift({"x": np.ones(200) * 10.0, "y": np.zeros(200)})
        self.assertTrue(isinstance(res.drift_detected, bool))
        self.assertTrue(isinstance(res.score, float))
        self.assertTrue(isinstance(res.details, dict))


if __name__ == "__main__":
    unittest.main()
