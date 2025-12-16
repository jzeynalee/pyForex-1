"""Unit tests for `ml/risk_retraining/risk_performance_monitor.py`.

Test Summary:
    Comprehensive coverage for risk-model performance metrics and monitoring.
    Includes deterministic metric calculator tests and monitor health behavior.

Test Breakdown:
    - Metric calculators (no sklearn-required paths)
        - direction accuracy
        - volatility MAE / MAPE / correlation
        - quantile pinball loss / coverage / crossing rate
        - GBM meta filter rate and improvement calculations (pure numpy)
        - RL metrics (sharpe ratio, entropy, exit ratios)
    - Monitor health
        - `get_model_health()` returns UNKNOWN when insufficient data
        - recording TCN predictions enables TCN metric calculation

Notes:
    Some calculator methods use sklearn; those paths are not exercised here.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestRiskPerformanceMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("ml.risk_retraining.risk_performance_monitor")

    def test_direction_accuracy(self):
        import numpy as np

        preds = np.array([[0.1, 0.9], [0.8, 0.2], [0.6, 0.4]])
        targets = np.array([1, 0, 0])
        acc = self.mod.TCNRiskMetricCalculator.direction_accuracy(preds, targets)
        self.assertAlmostEqual(acc, 1.0)

    def test_volatility_metrics(self):
        import numpy as np

        pred = np.array([0.10, 0.20, 0.30])
        real = np.array([0.10, 0.25, 0.35])
        mae = self.mod.TCNRiskMetricCalculator.volatility_mae(pred, real)
        self.assertAlmostEqual(mae, np.mean(np.abs(pred - real)))

        mape = self.mod.TCNRiskMetricCalculator.volatility_mape(pred, real)
        self.assertTrue(isinstance(mape, float))

        corr = self.mod.TCNRiskMetricCalculator.volatility_correlation(pred, real)
        self.assertTrue(isinstance(corr, float))

    def test_quantile_metrics(self):
        import numpy as np

        q_levels = [0.1, 0.5, 0.9]
        preds = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ]
        )
        actual = np.array([1.0, 1.0, 1.0])
        loss = self.mod.TCNRiskMetricCalculator.quantile_pinball_loss(preds, actual, q_levels)
        self.assertTrue(isinstance(loss, float))
        cov50 = self.mod.TCNRiskMetricCalculator.quantile_coverage(preds, actual, 1)
        self.assertTrue(0.0 <= cov50 <= 1.0)
        cross = self.mod.TCNRiskMetricCalculator.quantile_crossing_rate(preds)
        self.assertEqual(cross, 0.0)

    def test_gbm_meta_pure_helpers(self):
        import numpy as np

        preds = np.array([0.2, 0.7, 0.6, 0.1])
        results = np.array([1.0, -1.0, 2.0, -1.0])

        fr = self.mod.GBMMetaMetricCalculator.filter_rate(preds, threshold=0.5)
        self.assertAlmostEqual(fr, 0.5)

        fwr = self.mod.GBMMetaMetricCalculator.filtered_win_rate(preds, results, threshold=0.5)
        self.assertTrue(0.0 <= fwr <= 1.0)

        imp = self.mod.GBMMetaMetricCalculator.filter_improvement(preds, results, threshold=0.5)
        self.assertTrue(isinstance(imp, float))

        pf = self.mod.GBMMetaMetricCalculator.profit_factor(preds, results, threshold=0.5)
        self.assertTrue(isinstance(pf, float))

    def test_rl_exit_helpers(self):
        import numpy as np

        returns = np.array([0.01, -0.02, 0.03, 0.01])
        sharpe = self.mod.RLExitMetricCalculator.sharpe_ratio(returns)
        self.assertTrue(isinstance(sharpe, float))

        probs = np.array([[0.5, 0.5], [0.9, 0.1]])
        ent = self.mod.RLExitMetricCalculator.policy_entropy(probs)
        self.assertTrue(isinstance(ent, float))

        exit_pnls = np.array([1.0, -1.0, 2.0])
        pr = self.mod.RLExitMetricCalculator.profitable_exit_ratio(exit_pnls)
        self.assertAlmostEqual(pr, 2 / 3)

    def test_health_unknown_with_no_data(self):
        cfg = self.mod.RiskRetrainingConfig()
        monitor = self.mod.RiskPerformanceMonitor(cfg)

        health = monitor.get_model_health(self.mod.RiskModelType.TCN_RISK)
        self.assertEqual(health.status, self.mod.MetricStatus.UNKNOWN)
        self.assertFalse(health.needs_retraining)

    def test_record_tcn_prediction_enables_metric_calc(self):
        import numpy as np

        cfg = self.mod.RiskRetrainingConfig()
        monitor = self.mod.RiskPerformanceMonitor(cfg, window_size=50)

        for _ in range(60):
            monitor.record_tcn_prediction(
                direction_pred=np.array([0.2, 0.8]),
                direction_target=1,
                volatility_pred=0.01,
                volatility_realized=0.01,
                quantile_pred=np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
                quantile_actual=0.5,
            )

        metrics = monitor.calculate_tcn_metrics()
        self.assertTrue(isinstance(metrics, dict))
        health = monitor.get_model_health(self.mod.RiskModelType.TCN_RISK)
        self.assertTrue(hasattr(health, "status"))


if __name__ == "__main__":
    unittest.main()
