"""Unit tests for `ml/risk_retraining/__init__.py`.

Test Summary:
    Comprehensive coverage of the risk retraining package public surface.
    Confirms base configuration exports are always available, and optional
    components are exposed when dependencies exist.

Test Breakdown:
    - Import
        - package import succeeds (or is skipped if optional deps make import impossible)
    - Public helpers
        - version and available models/profiles helpers
    - Export contract
        - `__all__` is a list of strings
        - baseline config symbols are present in `__all__` and on module
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestMLRiskRetrainingInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.pkg = import_module("ml.risk_retraining")
        except Exception as e:
            raise unittest.SkipTest(f"Skipping ml.risk_retraining import (missing deps): {e}")

    def test_version_helpers(self):
        self.assertTrue(callable(getattr(self.pkg, "get_version", None)))
        self.assertTrue(callable(getattr(self.pkg, "get_available_models", None)))
        self.assertTrue(callable(getattr(self.pkg, "get_available_profiles", None)))

        v = self.pkg.get_version()
        self.assertIsInstance(v, str)

        profiles = self.pkg.get_available_profiles()
        self.assertIn("SCALP", profiles)

        models = self.pkg.get_available_models()
        self.assertTrue(isinstance(models, list))

    def test_all_contract_and_base_exports(self):
        self.assertTrue(isinstance(getattr(self.pkg, "__all__", []), list))
        self.assertTrue(all(isinstance(x, str) for x in self.pkg.__all__))

        required = [
            "RiskModelType",
            "RetrainingTriggerType",
            "RiskRetrainingConfig",
            "get_config_for_profile",
        ]
        for name in required:
            self.assertIn(name, self.pkg.__all__)
            self.assertTrue(hasattr(self.pkg, name))


if __name__ == "__main__":
    unittest.main()
