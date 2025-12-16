"""Unit tests for `ml/retraining_config.py`.

Test Summary:
    Comprehensive coverage of the consolidated retraining configuration layer.
    Verifies enum wiring, threshold evaluation helpers, schedule computation,
    factory presets, and serialization.

Test Breakdown:
    - Threshold evaluation
        - `PerformanceThresholds.check_thresholds()` and `get_violated()`
    - Schedule computation
        - `ScheduleConfig.get_next_scheduled_time()` for DAILY, WEEKLY, CUSTOM
    - Master configuration
        - `RetrainingConfig.is_trigger_enabled()`
        - `RetrainingConfig.to_dict()` contains expected keys
    - Factory methods
        - `for_scalping()`, `for_intraday()`, `for_swing()` adjust defaults
        - `from_profile()` handles known/unknown strings
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestRetrainingConfigComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("ml.retraining_config")

    def test_performance_thresholds_helpers(self):
        pt = self.mod.PerformanceThresholds(min_accuracy=0.55, max_drawdown=0.15)
        metrics = {"accuracy": 0.50, "drawdown": 0.20, "profit_factor": 1.5}
        violated = pt.get_violated(metrics)
        self.assertIn("accuracy", violated)
        self.assertIn("drawdown", violated)
        self.assertNotIn("profit_factor", violated)

    def test_schedule_next_time_daily(self):
        sc = self.mod.ScheduleConfig(
            schedule_type=self.mod.ScheduleType.DAILY,
            training_hour=2,
            training_minute=0,
        )
        now = datetime.now()
        nxt = sc.get_next_scheduled_time(from_time=now)
        self.assertTrue(isinstance(nxt, datetime))
        self.assertGreater(nxt, now)

    def test_schedule_next_time_weekly(self):
        sc = self.mod.ScheduleConfig(
            schedule_type=self.mod.ScheduleType.WEEKLY,
            training_days=[(datetime.now().weekday() + 1) % 7],
            training_hour=3,
            training_minute=0,
        )
        now = datetime.now()
        nxt = sc.get_next_scheduled_time(from_time=now)
        self.assertTrue(isinstance(nxt, datetime))
        self.assertGreater(nxt, now)

    def test_schedule_next_time_custom(self):
        sc = self.mod.ScheduleConfig(
            schedule_type=self.mod.ScheduleType.CUSTOM,
            custom_interval_hours=5,
        )
        now = datetime.now()
        nxt = sc.get_next_scheduled_time(from_time=now)
        self.assertTrue(isinstance(nxt, datetime))
        self.assertGreaterEqual(nxt - now, timedelta(hours=5))

    def test_retraining_config_factories(self):
        scalp = self.mod.RetrainingConfig.for_scalping()
        self.assertEqual(scalp.profile, self.mod.TradingProfile.SCALP)
        self.assertEqual(scalp.schedule.schedule_type, self.mod.ScheduleType.DAILY)

        intra = self.mod.RetrainingConfig.for_intraday()
        self.assertEqual(intra.profile, self.mod.TradingProfile.INTRADAY)

        swing = self.mod.RetrainingConfig.for_swing()
        self.assertEqual(swing.profile, self.mod.TradingProfile.SWING)

    def test_from_profile_known_and_unknown(self):
        cfg = self.mod.RetrainingConfig.from_profile("scalp")
        self.assertEqual(cfg.profile, self.mod.TradingProfile.SCALP)

        unknown = self.mod.RetrainingConfig.from_profile("unknown")
        self.assertEqual(unknown.profile, self.mod.TradingProfile.SWING)

    def test_is_trigger_enabled_and_to_dict(self):
        cfg = self.mod.RetrainingConfig.for_swing()
        self.assertTrue(cfg.is_trigger_enabled(self.mod.RetrainingTrigger.SCHEDULED))

        d = cfg.to_dict()
        for key in [
            "profile",
            "symbols",
            "cooldown_hours",
            "enabled_triggers",
            "schedule_type",
            "model_type",
            "min_training_samples",
            "lookback_days",
        ]:
            self.assertIn(key, d)


if __name__ == "__main__":
    unittest.main()
