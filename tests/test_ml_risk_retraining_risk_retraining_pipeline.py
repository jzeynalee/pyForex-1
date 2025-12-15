"""Unit tests for `ml/risk_retraining/risk_retraining_pipeline.py`.

Test Summary:
    Comprehensive coverage of deterministic retraining pipeline components.
    These tests focus on data providers and dependency orchestration logic
    without running heavy model training.

Test Breakdown:
    - Data structures
        - `PipelineResult.to_dict()` output keys
    - Data provider
        - `RiskDataProvider.load_tcn_data()` raises when no data files exist
    - Dependency orchestration
        - `RiskRetrainingPipelineManager.run_with_dependencies()` triggers dependent
          models when the primary model succeeds

Notes:
    The module imports torch. Tests are skipped only if the module cannot be
    imported in the current environment.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestRiskRetrainingPipelineStructures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.mod = import_module("ml.risk_retraining.risk_retraining_pipeline")
        except unittest.SkipTest:
            raise
        except Exception as e:
            raise unittest.SkipTest(f"Skipping risk_retraining_pipeline tests (missing deps): {e}")

    def test_pipeline_result_to_dict(self):
        r = self.mod.PipelineResult(
            model_type=self.mod.RiskModelType.TCN_RISK,
            trigger=self.mod.RetrainingTriggerType.MANUAL,
            success=True,
            stage_reached=self.mod.PipelineStage.COMPLETED,
            duration_seconds=1.0,
            metrics={"a": 1.0},
            champion_metrics=None,
            challenger_metrics=None,
            model_path=None,
            error=None,
            timestamp=datetime.now(),
        )
        d = r.to_dict()
        for key in ["model_type", "trigger", "success", "stage_reached", "duration_seconds", "metrics", "timestamp"]:
            self.assertIn(key, d)

    def test_data_provider_raises_when_no_files(self):
        with TemporaryDirectory() as td:
            dp = self.mod.RiskDataProvider(data_dir=td, validation_ratio=0.2)
            with self.assertRaises(ValueError):
                dp.load_tcn_data(["EURUSD"], lookback_days=1)

    def test_data_provider_read_parquet_failure_propagates(self):
        with TemporaryDirectory() as td:
            data_dir = Path(td)
            (data_dir / "eurusd_features.parquet").write_bytes(b"not parquet")
            dp = self.mod.RiskDataProvider(data_dir=td, validation_ratio=0.2)

            with patch.object(self.mod.pd, "read_parquet", side_effect=OSError("read failed")):
                with self.assertRaises(OSError):
                    dp.load_tcn_data(["EURUSD"], lookback_days=1)

    def test_pipeline_manager_runs_dependencies(self):
        cfg = self.mod.RiskRetrainingConfig()
        cfg.dependencies.tcn_triggers_gbm = True
        cfg.dependencies.tcn_triggers_rl = False

        mgr = self.mod.RiskRetrainingPipelineManager(cfg)

        def mk_result(model_type):
            return self.mod.PipelineResult(
                model_type=model_type,
                trigger=self.mod.RetrainingTriggerType.MANUAL,
                success=True,
                stage_reached=self.mod.PipelineStage.COMPLETED,
                duration_seconds=1.0,
                metrics={},
                champion_metrics=None,
                challenger_metrics=None,
                model_path="/tmp/model",
                error=None,
                timestamp=datetime.now(),
            )

        mgr.run_pipeline = MagicMock(side_effect=lambda model_type, trigger, **kwargs: mk_result(model_type))
        results = mgr.run_with_dependencies(self.mod.RiskModelType.TCN_RISK, self.mod.RetrainingTriggerType.MANUAL)
        self.assertEqual(len(results), 2)
        called_types = [c.args[0] for c in mgr.run_pipeline.call_args_list]
        self.assertEqual(called_types[0], self.mod.RiskModelType.TCN_RISK)
        self.assertEqual(called_types[1], self.mod.RiskModelType.GBM_META)


if __name__ == "__main__":
    unittest.main()
