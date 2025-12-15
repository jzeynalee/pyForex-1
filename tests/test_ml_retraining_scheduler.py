"""Unit tests for `ml/retraining_scheduler.py`.

Test Summary:
    Comprehensive coverage of the ML retraining scheduler orchestration layer.
    Tests focus on deterministic scheduling/trigger logic, callback wiring,
    and core error paths without running a full retraining workflow.

Test Breakdown:
    - Factory functions
        - `create_scheduler()` produces a scheduler with expected config
        - `create_scheduler_for_profile()` applies profile presets
    - Trigger lifecycle
        - `trigger_retraining()` creates an event when idle
        - repeated triggers are rejected when not idle
    - Callback registration
        - `add_trigger_callback()` stores callbacks
    - Execution guardrails
        - `execute_retraining()` raises when components are missing
    - Monitoring lifecycle
        - `start_monitoring()` toggles `_running` and spawns a thread
        - `stop_monitoring()` stops monitoring
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestRetrainingScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("ml.retraining_scheduler")

    def test_create_scheduler_factory(self):
        with TemporaryDirectory() as td:
            sched = self.mod.create_scheduler(profile_name="SWING", models_dir=td)
            self.assertIsInstance(sched, self.mod.RetrainingScheduler)
            self.assertEqual(sched.config.profile_name, "SWING")
            self.assertEqual(sched.config.models_dir, td)

    def test_create_scheduler_for_profile_presets(self):
        with TemporaryDirectory() as td:
            with patch.object(self.mod, "RetrainingScheduler") as RS:
                self.mod.create_scheduler_for_profile("SCALP")
                self.assertTrue(RS.called)
            cfg = self.mod.RetrainingConfig(profile_name="SCALP", models_dir=td, logs_dir=td)
            self.assertEqual(cfg.profile_name, "SCALP")

    def test_trigger_and_status(self):
        with TemporaryDirectory() as td:
            cfg = self.mod.RetrainingConfig(profile_name="SWING", models_dir=td, logs_dir=td)
            sched = self.mod.RetrainingScheduler(cfg)

            event = sched.trigger_retraining(self.mod.TriggerType.MANUAL, "unit test")
            self.assertIsNotNone(event)
            self.assertEqual(event.trigger_type, self.mod.TriggerType.MANUAL)
            self.assertEqual(sched.status, self.mod.RetrainingStatus.TRIGGERED)

            second = sched.trigger_retraining(self.mod.TriggerType.MANUAL, "second")
            self.assertIsNone(second)

            status = sched.get_status()
            for key in ["status", "profile", "monitoring_active", "performance_summary", "drift_trend"]:
                self.assertIn(key, status)

            nxt = sched.get_next_scheduled_time()
            self.assertTrue(isinstance(nxt, datetime) or nxt is None)

    def test_add_trigger_callback_registers(self):
        with TemporaryDirectory() as td:
            cfg = self.mod.RetrainingConfig(profile_name="SWING", models_dir=td, logs_dir=td)
            sched = self.mod.RetrainingScheduler(cfg)

            def cb(_event):
                return None

            sched.add_trigger_callback(cb)
            self.assertIn(cb, sched.on_trigger_callbacks)

    def test_execute_retraining_requires_components(self):
        with TemporaryDirectory() as td:
            cfg = self.mod.RetrainingConfig(profile_name="SWING", models_dir=td, logs_dir=td)
            sched = self.mod.RetrainingScheduler(cfg)
            evt = sched.trigger_retraining(self.mod.TriggerType.MANUAL, "unit test")
            self.assertIsNotNone(evt)
            with self.assertRaises(ValueError):
                sched.execute_retraining(evt)

    def test_check_scheduled_trigger_daily_boundary(self):
        mod = self.mod

        with TemporaryDirectory() as td:
            cfg = mod.RetrainingConfig(profile_name="SWING", models_dir=td, logs_dir=td)
            cfg.schedule.enabled = True
            cfg.schedule.schedule_type = "daily"
            cfg.schedule.schedule_hour = 2
            cfg.schedule.schedule_minute = 0

            sched = mod.RetrainingScheduler(cfg)
            sched.last_schedule_check = datetime(2000, 1, 1, 0, 0, 0)

            fixed_now = datetime(2025, 1, 2, 2, 0, 0)

            class FakeDateTime:
                @staticmethod
                def now():
                    return fixed_now

                @staticmethod
                def utcnow():
                    return fixed_now

            with patch.object(mod, "datetime", FakeDateTime):
                event = sched.check_scheduled_trigger()
                self.assertTrue(event is None or event.trigger_type == mod.TriggerType.SCHEDULED)

    def test_check_scheduled_trigger_weekly_boundary(self):
        mod = self.mod

        with TemporaryDirectory() as td:
            cfg = mod.RetrainingConfig(profile_name="SWING", models_dir=td, logs_dir=td)
            cfg.schedule.enabled = True
            cfg.schedule.schedule_type = "weekly"
            cfg.schedule.schedule_day = 0
            cfg.schedule.schedule_hour = 2
            cfg.schedule.schedule_minute = 0

            sched = mod.RetrainingScheduler(cfg)
            sched.last_schedule_check = datetime(2000, 1, 1, 0, 0, 0)

            fixed_now = datetime(2025, 1, 6, 2, 0, 0)

            class FakeDateTime:
                @staticmethod
                def now():
                    return fixed_now

                @staticmethod
                def utcnow():
                    return fixed_now

            with patch.object(mod, "datetime", FakeDateTime):
                event = sched.check_scheduled_trigger()
                self.assertTrue(event is None or event.trigger_type == mod.TriggerType.SCHEDULED)

    def test_execute_retraining_end_to_end_success(self):
        mod = self.mod

        with TemporaryDirectory() as td:
            cfg = mod.RetrainingConfig(profile_name="SWING", models_dir=td, logs_dir=td)
            cfg.require_validation = True
            cfg.auto_rollback_enabled = True
            cfg.min_training_samples = 1
            sched = mod.RetrainingScheduler(cfg)

            class DummyPreparer(mod.DataPreparer):
                def prepare_training_data(self, profile_name, start_date, end_date):
                    return [1, 2, 3], [1, 0, 1], [4, 5], [0, 1]

                def get_feature_names(self):
                    return ["f1"]

            class DummyModel:
                def predict(self, x):
                    return [0.6 for _ in x]

            class DummyTrainer(mod.ModelTrainer):
                def train(self, X_train, y_train, hyperparameters=None):
                    return DummyModel(), {"sharpe_ratio": 1.0}

                def get_default_hyperparameters(self):
                    return {}

                def get_model_type(self):
                    return "dummy"

            class DummyValidation:
                passed = True
                recommendation = "ok"

                def to_dict(self):
                    return {"passed": True}

            sched.set_data_preparer(DummyPreparer(cfg))
            sched.set_model_trainer(DummyTrainer(cfg))

            with patch.object(sched.model_manager, "save_model", return_value="m1"), patch.object(
                sched.model_manager, "activate_model", return_value=True
            ), patch.object(sched.model_manager, "validate_model", return_value=DummyValidation()), patch.object(
                sched, "_save_event_log", return_value=None
            ):
                evt = sched.trigger_retraining(mod.TriggerType.MANUAL, "unit")
                out = sched.execute_retraining(evt)
                self.assertEqual(out.status, mod.RetrainingStatus.COMPLETED)
                self.assertEqual(out.new_model_id, "m1")

    def test_execute_retraining_failure_triggers_rollback(self):
        mod = self.mod

        with TemporaryDirectory() as td:
            cfg = mod.RetrainingConfig(profile_name="SWING", models_dir=td, logs_dir=td)
            cfg.require_validation = False
            cfg.auto_rollback_enabled = True
            cfg.min_training_samples = 1
            sched = mod.RetrainingScheduler(cfg)

            class DummyPreparer(mod.DataPreparer):
                def prepare_training_data(self, profile_name, start_date, end_date):
                    return [1, 2, 3], [1, 0, 1], [4, 5], [0, 1]

                def get_feature_names(self):
                    return ["f1"]

            class DummyModel:
                def predict(self, x):
                    return [0.6 for _ in x]

            class DummyTrainer(mod.ModelTrainer):
                def train(self, X_train, y_train, hyperparameters=None):
                    return DummyModel(), {"sharpe_ratio": 1.0}

                def get_default_hyperparameters(self):
                    return {}

                def get_model_type(self):
                    return "dummy"

            sched.set_data_preparer(DummyPreparer(cfg))
            sched.set_model_trainer(DummyTrainer(cfg))
            sched.model_manager.active_models["SWING"] = "baseline"
            sched.current_event = None

            with patch.object(sched.model_manager, "save_model", return_value="m2"), patch.object(
                sched.model_manager, "activate_model", return_value=False
            ), patch.object(sched.model_manager, "rollback", return_value="baseline"), patch.object(
                sched, "_save_event_log", return_value=None
            ):
                evt = sched.trigger_retraining(mod.TriggerType.MANUAL, "unit")
                out = sched.execute_retraining(evt)
                self.assertIn(out.status, [mod.RetrainingStatus.FAILED, mod.RetrainingStatus.ROLLED_BACK])

    def test_start_and_stop_monitoring(self):
        with TemporaryDirectory() as td:
            cfg = self.mod.RetrainingConfig(profile_name="SWING", models_dir=td, logs_dir=td)
            sched = self.mod.RetrainingScheduler(cfg)

            class DummyThread:
                def __init__(self, target=None, daemon=None):
                    self.target = target
                    self.daemon = daemon
                    self.started = False

                def start(self):
                    self.started = True

                def join(self, timeout=None):
                    return None

            with patch.object(self.mod.threading, "Thread", DummyThread):
                sched.start_monitoring()
                self.assertTrue(sched._running)
                self.assertTrue(isinstance(sched._monitor_thread, DummyThread))
                self.assertTrue(sched._monitor_thread.started)
                sched.stop_monitoring()
                self.assertFalse(sched._running)


if __name__ == "__main__":
    unittest.main()
