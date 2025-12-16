"""Unit tests for `utils/yolo_dataset_generator.py`.

Test Summary:
    Covers YOLO dataset generator behavior using lightweight stubs to avoid
    heavy image generation.

Test Breakdown:
    - Input validation
        - missing required OHLC columns raises ValueError
    - Windowing
        - `_create_windows` returns correct number of windows
    - YAML config
        - `_create_yaml_config` writes expected keys
    - Generation pipeline (stubbed)
        - `generate_from_dataframe` returns stats dict and creates directory structure

Notes:
    Image rendering and pattern detection are patched with deterministic stubs.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestYOLODatasetGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("utils.yolo_dataset_generator")

    def _make_df(self, n=60):
        import pandas as pd

        close = [1.0 + i * 0.0001 for i in range(n)]
        return pd.DataFrame(
            {
                "open": close,
                "high": [c + 0.0002 for c in close],
                "low": [c - 0.0002 for c in close],
                "close": close,
                "tick_volume": [100] * n,
            }
        )

    def test_missing_columns_raises(self):
        import pandas as pd

        with TemporaryDirectory() as td:
            gen = self.mod.YOLODatasetGenerator(output_dir=td)
            with self.assertRaises(ValueError):
                gen.generate_from_dataframe(pd.DataFrame({"close": [1, 2, 3]}))

    def test_create_windows_count(self):
        df = self._make_df(100)
        with TemporaryDirectory() as td:
            gen = self.mod.YOLODatasetGenerator(output_dir=td, window_size=20, stride=10)
            windows = gen._create_windows(df, max_samples=None)
            self.assertEqual(len(windows), 9)

    def test_create_yaml_config_writes(self):
        import yaml

        with TemporaryDirectory() as td:
            gen = self.mod.YOLODatasetGenerator(output_dir=td)
            gen._setup_directories()
            gen._create_yaml_config()
            y = Path(td) / "data.yaml"
            self.assertTrue(y.exists())
            content = yaml.safe_load(y.read_text(encoding="utf-8"))
            self.assertIn("train", content)
            self.assertIn("val", content)
            self.assertIn("names", content)
            self.assertIn("nc", content)

    def test_generate_from_dataframe_stubbed(self):
        import numpy as np

        df = self._make_df(80)

        class DummyDetector:
            def detect_all_patterns(self, window):
                return []

            def to_yolo_annotations(self, patterns):
                return []

        class DummyRenderer:
            def render_with_annotations(self, window, annotations):
                img = np.zeros((16, 16, 3), dtype=np.uint8)
                return img, []

        with TemporaryDirectory() as td:
            gen = self.mod.YOLODatasetGenerator(output_dir=td, window_size=20, stride=20, val_split=0.5)
            gen.detector = DummyDetector()
            gen.renderer = DummyRenderer()

            stats = gen.generate_from_dataframe(df, symbol="X", max_samples=3)
            self.assertTrue(isinstance(stats, dict))
            self.assertIn("train", stats)
            self.assertIn("val", stats)


if __name__ == "__main__":
    unittest.main()
