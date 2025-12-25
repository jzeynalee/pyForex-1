"""Unit tests for `ml/model_manager.py`.

Test Summary:
    Comprehensive coverage of model lifecycle management:
    registry persistence, model save/load, validation gating, activation,
    rollback, model comparisons, and export/import.

Test Breakdown:
    - Persistence
        - creating a new manager over an existing models_dir loads prior registry
    - Save/load
        - `save_model()` creates model files + metadata
        - `load_model()` returns model and metadata
    - Activation
        - activation is blocked when validation is required and model is not validated
        - activation succeeds when forced
    - Validation
        - `validate_model()` compares candidate vs baseline and updates metadata
    - Queries
        - `list_models()`, `get_model_info()`, `compare_models()`
    - Import/export
        - `export_model()` produces a zip
        - `import_model()` registers imported model
    - Rollback
        - `rollback()` activates the prior model version for a profile
"""

from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestModelManagerComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("ml.model_manager")

    def _save_two_models(self, mgr, profile: str = "SWING"):
        now = datetime.now()

        model_id_1 = mgr.save_model(
            model={"pred": 1.0},
            profile_name=profile,
            version="v1",
            model_type="dummy",
            hyperparameters={"a": 1},
            feature_names=["f1"],
            feature_schema_version=None,
            training_data=[1, 2, 3],
            training_start=now - timedelta(hours=2),
            training_end=now - timedelta(hours=1),
            validation_metrics={"sharpe_ratio": 1.0, "win_rate": 0.5, "profit_factor": 1.2},
        )

        model_id_2 = mgr.save_model(
            model={"pred": 2.0},
            profile_name=profile,
            version="v2",
            model_type="dummy",
            hyperparameters={"a": 2},
            feature_names=["f1"],
            feature_schema_version=None,
            training_data=[1, 2, 3, 4],
            training_start=now - timedelta(hours=1),
            training_end=now,
            validation_metrics={"sharpe_ratio": 1.2, "win_rate": 0.55, "profit_factor": 1.3},
        )

        mgr.registry[model_id_1].created_at = now - timedelta(minutes=10)
        mgr.registry[model_id_2].created_at = now
        mgr._save_registry()

        return model_id_1, model_id_2

    def test_save_and_load_model_roundtrip(self):
        mod = self.mod

        with TemporaryDirectory() as td:
            mgr = mod.ModelManager(mod.ManagerConfig(models_dir=td, max_versions=10))
            mid1, _mid2 = self._save_two_models(mgr)

            model, meta = mgr.load_model(mid1)
            self.assertIsInstance(model, dict)
            self.assertEqual(meta.model_id, mid1)
            self.assertEqual(meta.profile_name, "SWING")

    def test_registry_persists_across_instances(self):
        mod = self.mod

        with TemporaryDirectory() as td:
            mgr1 = mod.ModelManager(mod.ManagerConfig(models_dir=td, max_versions=10))
            mid1, _mid2 = self._save_two_models(mgr1)

            mgr2 = mod.ModelManager(mod.ManagerConfig(models_dir=td, max_versions=10))
            self.assertIn(mid1, mgr2.registry)

    def test_corrupted_registry_json_recovers_to_empty(self):
        mod = self.mod

        with TemporaryDirectory() as td:
            reg = Path(td) / "registry.json"
            reg.write_text("{not valid json", encoding="utf-8")

            mgr = mod.ModelManager(mod.ManagerConfig(models_dir=td, max_versions=10))
            self.assertEqual(mgr.registry, {})
            self.assertEqual(mgr.active_models, {})

    def test_save_registry_permission_error_does_not_raise(self):
        mod = self.mod

        with TemporaryDirectory() as td:
            mgr = mod.ModelManager(mod.ManagerConfig(models_dir=td, max_versions=10))
            mgr.registry["x"] = mod.ModelMetadata(
                model_id="x",
                version="v1",
                created_at=datetime.now(),
                training_start=datetime.now(),
                training_end=datetime.now(),
                model_type="dummy",
                hyperparameters={},
                feature_names=[],
                feature_schema_version=None,
                training_samples=0,
                validation_metrics={},
                profile_name="SWING",
                data_hash="h",
            )

            with patch("builtins.open", side_effect=PermissionError("denied")):
                mgr._save_registry()

    def test_save_model_raises_on_filesystem_error(self):
        mod = self.mod

        with TemporaryDirectory() as td:
            mgr = mod.ModelManager(mod.ManagerConfig(models_dir=td, max_versions=10))
            now = datetime.now()

            def open_side_effect(*args, **kwargs):
                raise PermissionError("denied")

            with patch("builtins.open", side_effect=open_side_effect):
                with self.assertRaises(PermissionError):
                    mgr.save_model(
                        model={"pred": 1.0},
                        profile_name="SWING",
                        version="v1",
                        model_type="dummy",
                        hyperparameters={},
                        feature_names=[],
                        feature_schema_version=None,
                        training_data=[1],
                        training_start=now,
                        training_end=now,
                        validation_metrics={},
                    )

    def test_activation_gated_by_validation(self):
        mod = self.mod

        with TemporaryDirectory() as td:
            cfg = mod.ManagerConfig(models_dir=td, require_validation=True)
            mgr = mod.ModelManager(cfg)
            _mid1, mid2 = self._save_two_models(mgr)

            ok = mgr.activate_model(mid2, force=False)
            self.assertFalse(ok)

            ok2 = mgr.activate_model(mid2, force=True)
            self.assertTrue(ok2)
            self.assertEqual(mgr.active_models.get("SWING"), mid2)

    def test_validate_model_updates_metadata_and_returns_result(self):
        mod = self.mod

        with TemporaryDirectory() as td:
            cfg = mod.ManagerConfig(models_dir=td, require_validation=True, min_improvement_pct=1.0)
            mgr = mod.ModelManager(cfg)
            mid1, mid2 = self._save_two_models(mgr)

            mgr.active_models["SWING"] = mid1
            mgr.registry[mid1].is_active = True
            mgr._save_registry()

            def predict_fn(model, _data):
                return model["pred"]

            metric_calculators = {
                "sharpe_ratio": lambda pred, _y: float(pred),
                "win_rate": lambda pred, _y: float(pred) / 10.0,
                "profit_factor": lambda pred, _y: float(pred),
            }

            res = mgr.validate_model(
                candidate_id=mid2,
                validation_data=[1, 2, 3],
                validation_targets=[0, 1, 0],
                predict_fn=predict_fn,
                metric_calculators=metric_calculators,
            )

            self.assertEqual(res.candidate_id, mid2)
            self.assertEqual(res.baseline_id, mid1)
            self.assertIn("sharpe_ratio", res.metrics_comparison)
            self.assertTrue(isinstance(res.passed, bool))
            self.assertIsInstance(res.improvement_pct, float)
            self.assertTrue(mgr.registry[mid2].validation_timestamp is not None)

    def test_list_models_get_info_compare(self):
        mod = self.mod

        with TemporaryDirectory() as td:
            mgr = mod.ModelManager(mod.ManagerConfig(models_dir=td, max_versions=10))
            mid1, mid2 = self._save_two_models(mgr)

            models = mgr.list_models(profile_name="SWING")
            self.assertTrue(len(models) >= 2)

            info = mgr.get_model_info(mid1)
            self.assertIsInstance(info, dict)
            self.assertEqual(info["model_id"], mid1)
            self.assertIn("model_exists", info)

            comp = mgr.compare_models([mid1, mid2])
            self.assertIn(mid1, comp)
            self.assertIn(mid2, comp)
            self.assertIn("validation_metrics", comp[mid1])

    def test_export_and_import(self):
        mod = self.mod

        with TemporaryDirectory() as td:
            mgr = mod.ModelManager(mod.ManagerConfig(models_dir=td, max_versions=10))
            mid1, _mid2 = self._save_two_models(mgr)

            export_base = str(Path(td) / "exported_model")
            ok = mgr.export_model(mid1, export_base + ".zip")
            self.assertTrue(ok)
            self.assertTrue(Path(export_base + ".zip").exists())

            mgr2 = mod.ModelManager(mod.ManagerConfig(models_dir=td, max_versions=10))
            imported_id = mgr2.import_model(export_base + ".zip")
            self.assertIsInstance(imported_id, str)
            self.assertIn(imported_id, mgr2.registry)

    def test_rollback_selects_previous(self):
        mod = self.mod

        with TemporaryDirectory() as td:
            mgr = mod.ModelManager(mod.ManagerConfig(models_dir=td, max_versions=10, require_validation=False))
            mid1, mid2 = self._save_two_models(mgr)

            mgr.activate_model(mid2, force=True)
            rolled = mgr.rollback("SWING", steps=1)
            self.assertEqual(rolled, mid1)
            self.assertEqual(mgr.active_models.get("SWING"), mid1)


if __name__ == "__main__":
    unittest.main()
