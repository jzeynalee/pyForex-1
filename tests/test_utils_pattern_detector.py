"""Unit tests for `utils/pattern_detector.py`.

Test Summary:
    Covers candlestick pattern detection primitives, including single-, two-,
    and three-candle pattern rules, and annotation conversion.

Test Breakdown:
    - Enums and dataclasses
        - PatternClass members and PATTERN_NAMES consistency
        - PatternDetection fields
    - Detector basics
        - `detect_all_patterns()` returns list
        - `to_yolo_annotations()` maps detections to dicts
    - Deterministic pattern cases
        - Doji detection
        - Bullish engulfing detection
        - Morning star detection (basic)
    - Edge cases
        - empty dataframe returns empty list
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestPatternDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("utils.pattern_detector")

    def test_enum_and_names_consistency(self):
        self.assertEqual(len(self.mod.PATTERN_NAMES), len(list(self.mod.PatternClass)))
        self.assertEqual(self.mod.PATTERN_NAMES[self.mod.PatternClass.DOJI], "doji")

    def test_empty_dataframe_returns_empty(self):
        import pandas as pd

        det = self.mod.CandlestickPatternDetector()
        out = det.detect_all_patterns(pd.DataFrame(columns=["open", "high", "low", "close"]))
        self.assertEqual(out, [])

    def test_detect_dojo_single_candle(self):
        import pandas as pd

        df = pd.DataFrame(
            [
                {"open": 1.0, "high": 1.2, "low": 0.8, "close": 1.001},
            ]
        )
        det = self.mod.CandlestickPatternDetector(doji_threshold=0.05)
        patterns = det.detect_all_patterns(df)
        self.assertTrue(any(p.pattern_name == "doji" for p in patterns))

    def test_detect_bullish_engulfing(self):
        import pandas as pd

        # Candle 0 bearish, candle 1 bullish engulfing candle 0
        df = pd.DataFrame(
            [
                {"open": 1.10, "high": 1.12, "low": 1.05, "close": 1.06},
                {"open": 1.05, "high": 1.13, "low": 1.04, "close": 1.12},
            ]
        )
        det = self.mod.CandlestickPatternDetector(engulf_threshold=1.0)
        patterns = det.detect_all_patterns(df)
        names = [p.pattern_name for p in patterns]
        self.assertIn("bullish_engulfing", names)

        anns = det.to_yolo_annotations(patterns)
        self.assertTrue(all("class_id" in a and "start_idx" in a and "end_idx" in a for a in anns))

    def test_detect_morning_star_basic(self):
        import pandas as pd

        # Rough morning star proxy
        df = pd.DataFrame(
            [
                {"open": 1.20, "high": 1.21, "low": 1.10, "close": 1.11},
                {"open": 1.11, "high": 1.12, "low": 1.05, "close": 1.10},
                {"open": 1.09, "high": 1.20, "low": 1.08, "close": 1.18},
            ]
        )
        det = self.mod.CandlestickPatternDetector()
        patterns = det.detect_all_patterns(df)
        self.assertTrue(any(p.pattern_name == "morning_star" for p in patterns))


if __name__ == "__main__":
    unittest.main()
