"""Unit tests for `utils/generate_dataset.py`.

Test Summary:
    Tests the dataset generation orchestration script in a safe, stubbed way.

Test Breakdown:
    - `save_stats`
        - writes JSON to disk
    - Dataset generation functions
        - `generate_yolo_dataset` forwards parameters to YOLODatasetGenerator
        - `generate_vit_dataset` forwards parameters to ViTDatasetGenerator
    - CLI behavior
        - missing data file triggers SystemExit

Notes:
    Heavy image generation is replaced with stubs.
"""

from __future__ import annotations

import json
import sys
import unittest
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestGenerateDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("utils.generate_dataset")

    def test_save_stats_writes_json(self):
        with TemporaryDirectory() as td:
            out = Path(td) / "s.json"
            self.mod.save_stats({"a": 1}, str(out))
            self.assertTrue(out.exists())
            self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["a"], 1)

    def test_generate_yolo_dataset_forwards(self):
        class DummyGen:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def generate_synthetic(self, n_samples, symbol):
                return {"mode": "synthetic", "n": n_samples, "symbol": symbol}

            def generate_from_csv(self, data_path, max_samples):
                return {"mode": "csv", "path": data_path, "n": max_samples}

        with patch.object(self.mod, "YOLODatasetGenerator", DummyGen):
            stats = self.mod.generate_yolo_dataset(
                data_path="x.csv",
                output_dir="out",
                samples=3,
                synthetic=True,
                image_size=128,
                window_size=10,
            )
            self.assertEqual(stats["mode"], "synthetic")

    def test_generate_vit_dataset_forwards(self):
        class DummyGen:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def generate_synthetic(self, n_samples, symbol, class_balance):
                return {"mode": "synthetic", "n": n_samples, "symbol": symbol}

            def generate_from_csv(self, data_path, max_samples):
                return {"mode": "csv", "path": data_path, "n": max_samples}

        class DummyLabeler:
            def __init__(self, forward_bars, threshold_pct):
                self.forward_bars = forward_bars
                self.threshold_pct = threshold_pct

        with patch.object(self.mod, "ViTDatasetGenerator", DummyGen), patch.object(self.mod, "FuturePriceLabeler", DummyLabeler):
            stats = self.mod.generate_vit_dataset(
                data_path="x.csv",
                output_dir="out",
                samples=3,
                synthetic=True,
                image_size=128,
                window_size=10,
            )
            self.assertEqual(stats["mode"], "synthetic")

    def test_main_missing_file_exits(self):
        argv = ["generate_dataset.py", "--data", "does_not_exist.csv", "--output", "out"]
        with patch.object(sys, "argv", argv), patch.object(self.mod.Path, "exists", return_value=False), self.assertRaises(SystemExit):
            self.mod.main()


if __name__ == "__main__":
    unittest.main()
