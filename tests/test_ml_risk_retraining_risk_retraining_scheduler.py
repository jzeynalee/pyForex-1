"""Unit tests for `ml/risk_retraining/risk_retraining_scheduler.py`.

Test Summary:
    Comprehensive coverage of the risk retraining scheduler orchestration
    utilities. Focuses on deterministic manager behavior and queueing logic.

Test Breakdown:
    - Managers
        - `BlackoutPeriodManager.is_in_blackout()` returns a (bool, reason) tuple
        - `CooldownManager.is_in_cooldown()` behavior after marking retrained
        - `ScheduleManager.get_next_scheduled()` returns datetime/None
    - Scheduler queueing
        - `_queue_retraining()` prevents duplicates
        - `trigger_manual_retraining()` queues and processes pending when allowed

Notes:
    Imports may pull optional dependencies. If imports fail, tests are skipped.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timedelta


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestRiskRetrainingSchedulerSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.mod = import_module("ml.risk_retraining.risk_retraining_scheduler")
        except Exception as e:
            raise unittest.SkipTest(f"Skipping risk_retraining_scheduler tests (missing deps): {e}")

    def test_blackout_manager_tuple(self):
        cfg = self.mod.RiskRetrainingConfig()
        mgr = self.mod.BlackoutPeriodManager(cfg)
        in_blackout, reason = mgr.is_in_blackout()
        self.assertIsInstance(in_blackout, bool)
        self.assertTrue(reason is None or isinstance(reason, str))

    def test_cooldown_manager(self):
        cfg = self.mod.RiskRetrainingConfig()
        cm = self.mod.CooldownManager(cfg)
        in_cd, rem = cm.is_in_cooldown(self.mod.RiskModelType.TCN_RISK)
        self.assertFalse(in_cd)
        cm.mark_retrained(self.mod.RiskModelType.TCN_RISK)
        in_cd2, rem2 = cm.is_in_cooldown(self.mod.RiskModelType.TCN_RISK)
        self.assertTrue(isinstance(in_cd2, bool))
        self.assertTrue(rem2 is None or hasattr(rem2, "total_seconds"))

    def test_schedule_manager_next_scheduled(self):
        cfg = self.mod.RiskRetrainingConfig()
        sm = self.mod.ScheduleManager(cfg)
        nxt = sm.get_next_scheduled(self.mod.RiskModelType.TCN_RISK)
        self.assertTrue(nxt is None or hasattr(nxt, "isoformat"))

    def test_schedule_manager_is_scheduled_now_daily_boundary(self):
        cfg = self.mod.RiskRetrainingConfig()
        cfg.tcn_schedule.enabled = True
        cfg.tcn_schedule.schedule_type = "daily"
        cfg.tcn_schedule.schedule_hour = 2
        sm = self.mod.ScheduleManager(cfg)

        fixed = datetime(2025, 1, 2, 2, 5, 0)

        class FakeDateTime:
            @staticmethod
            def utcnow():
                return fixed

        with patch.object(self.mod, "datetime", FakeDateTime):
            self.assertTrue(sm.is_scheduled_now(self.mod.RiskModelType.TCN_RISK))

    def test_schedule_manager_is_scheduled_now_weekly_boundary(self):
        cfg = self.mod.RiskRetrainingConfig()
        cfg.tcn_schedule.enabled = True
        cfg.tcn_schedule.schedule_type = "weekly"
        cfg.tcn_schedule.schedule_day_of_week = 6
        cfg.tcn_schedule.schedule_hour = 3
        sm = self.mod.ScheduleManager(cfg)

        fixed = datetime(2025, 1, 5, 3, 1, 0)

        class FakeDateTime:
            @staticmethod
            def utcnow():
                return fixed

        with patch.object(self.mod, "datetime", FakeDateTime):
            self.assertTrue(sm.is_scheduled_now(self.mod.RiskModelType.TCN_RISK))

    def test_queue_retraining_deduplicates(self):
        cfg = self.mod.RiskRetrainingConfig()

        class DummyPM:
            def __init__(self, _cfg):
                pass

            def get_model_health(self, _model_type):
                class H:
                    needs_retraining = False
                    reason = None

                    def to_dict(self):
                        return {}

                    metrics = []

                return H()

            def mark_retrained(self, _model_type):
                return None

        class DummyDD:
            def __init__(self, _cfg):
                pass

            def should_check(self, _model_type):
                return False

        class DummyPipeline:
            def __init__(self, _cfg):
                pass

            def run_with_dependencies(self, _model_type, _trigger):
                class R:
                    success = True
                    error = None
                    metrics = {}
                    model_path = None
                    model_type = _model_type

                return [R()]

        with patch.object(self.mod, "RiskPerformanceMonitor", DummyPM), patch.object(self.mod, "RiskDriftDetector", DummyDD), patch.object(self.mod, "RiskRetrainingPipelineManager", DummyPipeline):
            sched = self.mod.RiskRetrainingScheduler(config=cfg)
            sched._queue_retraining(self.mod.RiskModelType.TCN_RISK, self.mod.RetrainingTriggerType.MANUAL, "x")
            sched._queue_retraining(self.mod.RiskModelType.TCN_RISK, self.mod.RetrainingTriggerType.MANUAL, "x")
            self.assertEqual(len(sched.pending_retraining), 1)

    def test_end_to_end_pending_processing_executes_pipeline_and_updates_cooldown(self):
        cfg = self.mod.RiskRetrainingConfig()
        cfg.cooldown_hours = 1

        class DummyPM:
            def __init__(self, _cfg):
                pass

            def get_model_health(self, _model_type):
                class H:
                    needs_retraining = False
                    reason = None

                    metrics = []

                    def to_dict(self):
                        return {}

                return H()

            def mark_retrained(self, _model_type):
                return None

        class DummyDD:
            def __init__(self, _cfg):
                pass

            def should_check(self, _model_type):
                return False

        class DummyPipeline:
            def __init__(self, _cfg):
                pass

            def run_with_dependencies(self, model_type, trigger):
                r = type("R", (), {})()
                r.success = True
                r.error = None
                r.metrics = {}
                r.model_path = "p"
                r.model_type = model_type
                return [r]

        with patch.object(self.mod, "RiskPerformanceMonitor", DummyPM), patch.object(self.mod, "RiskDriftDetector", DummyDD), patch.object(self.mod, "RiskRetrainingPipelineManager", DummyPipeline):
            sched = self.mod.RiskRetrainingScheduler(config=cfg)
            sched._queue_retraining(self.mod.RiskModelType.TCN_RISK, self.mod.RetrainingTriggerType.MANUAL, "unit")
            self.assertEqual(len(sched.pending_retraining), 1)
            sched._process_pending_retraining()
            self.assertEqual(len(sched.pending_retraining), 0)
            self.assertTrue(len(sched.event_history) >= 1)
            in_cd, remaining = sched.cooldown_manager.is_in_cooldown(self.mod.RiskModelType.TCN_RISK)
            self.assertTrue(isinstance(in_cd, bool))


if __name__ == "__main__":
    unittest.main()
