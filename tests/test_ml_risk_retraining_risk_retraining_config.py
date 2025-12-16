"""Unit tests for `ml/risk_retraining/risk_retraining_config.py`.

Test Summary:
    Comprehensive coverage of the Risk Retraining configuration module:
    enums, dataclass defaults, profile presets, and routing helpers.

Test Breakdown:
    - Enums
        - expected model/trigger types exist
    - Profile presets
        - `get_config_for_profile()` supports SCALP/SWING/INTRADAY
        - unknown profile raises ValueError
    - Routing helpers
        - `get_metrics_for_model()` returns dicts for each model type
        - `get_drift_config_for_model()` returns correct drift config object
        - `get_schedule_config_for_model()` returns correct schedule config object
    - Config presets
        - `RiskRetrainingConfig.for_scalp/for_swing/for_intraday()` set profile_name and schedules
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestRiskRetrainingConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("ml.risk_retraining.risk_retraining_config")

    def test_enum_members(self):
        self.assertTrue(hasattr(self.mod.RiskModelType, "TCN_RISK"))
        self.assertTrue(hasattr(self.mod.RiskModelType, "GBM_META"))
        self.assertTrue(hasattr(self.mod.RiskModelType, "RL_EXIT"))
        self.assertTrue(hasattr(self.mod.RetrainingTriggerType, "SCHEDULED"))
        self.assertTrue(hasattr(self.mod.RetrainingTriggerType, "PERFORMANCE"))
        self.assertTrue(hasattr(self.mod.RetrainingTriggerType, "DRIFT"))

    def test_get_config_for_profile(self):
        cfg = self.mod.get_config_for_profile("SCALP")
        self.assertEqual(cfg.profile_name, "SCALP")

        cfg2 = self.mod.get_config_for_profile("SWING")
        self.assertEqual(cfg2.profile_name, "SWING")

        cfg3 = self.mod.get_config_for_profile("INTRADAY")
        self.assertEqual(cfg3.profile_name, "INTRADAY")

        with self.assertRaises(ValueError):
            self.mod.get_config_for_profile("NOPE")

    def test_get_metrics_for_model(self):
        cfg = self.mod.RiskRetrainingConfig()
        m1 = cfg.get_metrics_for_model(self.mod.RiskModelType.TCN_RISK)
        m2 = cfg.get_metrics_for_model(self.mod.RiskModelType.GBM_META)
        m3 = cfg.get_metrics_for_model(self.mod.RiskModelType.RL_EXIT)
        self.assertIsInstance(m1, dict)
        self.assertIsInstance(m2, dict)
        self.assertIsInstance(m3, dict)

        with self.assertRaises(ValueError):
            cfg.get_metrics_for_model("NOPE")

    def test_get_drift_and_schedule_config_for_model(self):
        cfg = self.mod.RiskRetrainingConfig()

        self.assertIs(cfg.get_drift_config_for_model(self.mod.RiskModelType.TCN_RISK), cfg.tcn_drift)
        self.assertIs(cfg.get_drift_config_for_model(self.mod.RiskModelType.GBM_META), cfg.gbm_drift)
        self.assertIs(cfg.get_drift_config_for_model(self.mod.RiskModelType.RL_EXIT), cfg.rl_drift)

        self.assertIs(cfg.get_schedule_config_for_model(self.mod.RiskModelType.TCN_RISK), cfg.tcn_schedule)
        self.assertIs(cfg.get_schedule_config_for_model(self.mod.RiskModelType.GBM_META), cfg.gbm_schedule)
        self.assertIs(cfg.get_schedule_config_for_model(self.mod.RiskModelType.RL_EXIT), cfg.rl_schedule)

    def test_profile_preset_constructors(self):
        scalp = self.mod.RiskRetrainingConfig.for_scalp()
        self.assertEqual(scalp.profile_name, "SCALP")
        self.assertTrue(scalp.tcn_schedule.enabled)

        swing = self.mod.RiskRetrainingConfig.for_swing()
        self.assertEqual(swing.profile_name, "SWING")
        self.assertTrue(swing.tcn_schedule.enabled)

        intra = self.mod.RiskRetrainingConfig.for_intraday()
        self.assertEqual(intra.profile_name, "INTRADAY")
        self.assertTrue(intra.tcn_schedule.enabled)


if __name__ == "__main__":
    unittest.main()
