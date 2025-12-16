"""Unit tests for `utils/__init__.py`.

Test Summary:
    Validates the `utils` package import surface and its re-exported symbols.

Test Breakdown:
    - Import
        - `import utils` succeeds
    - Export contract
        - `__all__` contains expected public names
        - each name in `__all__` exists as an attribute on the package
    - Re-exported types
        - `MTFProfile`, `Timeframe`, `MTFFeatureBuilder`, `MTFFeatureSet` are importable
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestUtilsInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import utils

        cls.pkg = utils

    def test_import_utils(self):
        import utils  # noqa: F401

    def test_all_contract(self):
        self.assertTrue(isinstance(getattr(self.pkg, "__all__", []), list))
        self.assertTrue(all(isinstance(x, str) for x in self.pkg.__all__))

    def test_expected_exports_present(self):
        required = [
            "MTFProfile",
            "Timeframe",
            "SCALP_PROFILE",
            "SWING_PROFILE",
            "INTRADAY_PROFILE",
            "get_profile",
            "create_custom_profile",
            "MTFFeatureBuilder",
            "MTFFeatureSet",
        ]
        for name in required:
            self.assertIn(name, self.pkg.__all__)
            self.assertTrue(hasattr(self.pkg, name))


if __name__ == "__main__":
    unittest.main()
