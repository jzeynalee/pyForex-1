"""Unit tests for `ml/__init__.py`.

Test Summary:
    Comprehensive coverage of the `ml` package import surface:
    metadata fields, `__all__` contract, and high-level convenience exports.

Test Breakdown:
    - Import and metadata
        - package imports successfully
        - `__version__` / `__author__` are present
    - Export contract
        - `__all__` is a list of strings
        - `__all__` contains critical public names
    - Convenience API
        - exported types/classes exist (config, detectors, managers)
        - factory functions exist and are callable

Notes:
    Importing `ml` may require optional dependencies (e.g., SciPy). If those
    are not installed, these tests are skipped.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestMLInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import ml  # noqa: F401
        except Exception as e:
            raise unittest.SkipTest(f"Skipping ml init tests (missing deps): {e}")

    def test_import_ml(self):
        import ml  # noqa: F401

    def test_metadata_fields(self):
        import ml

        self.assertTrue(hasattr(ml, "__version__"))
        self.assertTrue(isinstance(getattr(ml, "__version__"), str))
        self.assertTrue(hasattr(ml, "__author__"))
        self.assertTrue(isinstance(getattr(ml, "__author__"), str))

    def test_public_exports_present(self):
        import ml

        self.assertTrue(hasattr(ml, "RetrainingConfig"))
        self.assertTrue(hasattr(ml, "DriftDetector"))
        self.assertTrue(hasattr(ml, "PerformanceMonitor"))
        self.assertTrue(hasattr(ml, "ModelManager"))
        self.assertTrue(hasattr(ml, "RetrainingScheduler"))

        self.assertTrue(isinstance(getattr(ml, "__all__", []), list))
        self.assertTrue(all(isinstance(x, str) for x in ml.__all__))

        required = [
            "RetrainingConfig",
            "DriftDetector",
            "ConceptDriftDetector",
            "PerformanceMonitor",
            "ModelManager",
            "RetrainingPipeline",
            "RetrainingScheduler",
        ]
        for name in required:
            self.assertIn(name, ml.__all__)

        self.assertTrue(callable(getattr(ml, "create_scheduler", None)))
        self.assertTrue(callable(getattr(ml, "create_scheduler_for_profile", None)))

    def test_factory_functions_exist(self):
        import ml

        self.assertTrue(callable(getattr(ml, "create_scheduler", None)))
        self.assertTrue(callable(getattr(ml, "create_scheduler_for_profile", None)))

    def test_factory_functions_return_scheduler(self):
        import ml

        with TemporaryDirectory() as td:
            sched = ml.create_scheduler(profile_name="SWING", models_dir=td)
            self.assertIsInstance(sched, ml.RetrainingScheduler)

        sched2 = ml.create_scheduler_for_profile("SWING")
        self.assertIsInstance(sched2, ml.RetrainingScheduler)


if __name__ == "__main__":
    unittest.main()
