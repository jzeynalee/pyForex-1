"""Unit tests for `models/__init__.py`.

Test Summary:
    Ensures the `models` package is importable and behaves as a package module.

Test Breakdown:
    - Package import
        - importing `models` succeeds
    - Basic package attributes
        - `models.__file__` exists
        - `models.__package__` is set
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestModelsInit(unittest.TestCase):
    def test_import_models_package(self):
        import models  # noqa: F401

    def test_basic_package_attributes(self):
        import models

        self.assertTrue(hasattr(models, "__file__"))
        self.assertTrue(bool(models.__file__))
        self.assertEqual(models.__package__, "models")


if __name__ == "__main__":
    unittest.main()
