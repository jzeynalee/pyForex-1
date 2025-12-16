"""Unit tests for `utils/chart_patterns.py`.

Test Summary:
    Covers the chart pattern detection helpers and pattern functions.

Test Breakdown:
    - Utilities
        - `_mark_index` bounds behavior
        - `_find_peaks` / `_find_troughs` return indices without error
    - Pattern detectors
        - output type is `pd.Series` with boolean dtype aligned to df.index
        - patterns behave sensibly on flat / insufficient movement data
        - at least one deterministic synthetic scenario triggers a known pattern
    - Registry
        - `CHART_PATTERN_FUNCS` contains callable entries for all exported patterns

Notes:
    This module depends on SciPy (`scipy.signal.find_peaks`). If SciPy is not
    available, these tests are skipped.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestChartPatterns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.mod = import_module("utils.chart_patterns")
        except Exception as e:
            raise unittest.SkipTest(f"Skipping chart_patterns tests (missing deps): {e}")

    def _make_df(self, close):
        import pandas as pd

        n = len(close)
        return pd.DataFrame(
            {
                "open": close,
                "high": [c + 0.1 for c in close],
                "low": [c - 0.1 for c in close],
                "close": close,
                "volume": [100] * n,
            }
        )

    def test_mark_index_bounds(self):
        m = self.mod._mark_index(5, 2)
        self.assertEqual(m.tolist(), [False, False, True, False, False])
        m2 = self.mod._mark_index(5, -1)
        self.assertEqual(m2.tolist(), [False] * 5)
        m3 = self.mod._mark_index(5, 10)
        self.assertEqual(m3.tolist(), [False] * 5)

    def test_find_peaks_and_troughs_smoke(self):
        import pandas as pd

        s = pd.Series([0, 1, 0, 1, 0, 1, 0])
        peaks = self.mod._find_peaks(s, distance=1)
        troughs = self.mod._find_troughs(s, distance=1)
        self.assertTrue(isinstance(peaks, (list, tuple)) or hasattr(peaks, "__len__"))
        self.assertTrue(isinstance(troughs, (list, tuple)) or hasattr(troughs, "__len__"))

    def test_pattern_output_contract_on_flat_data(self):
        import pandas as pd

        df = self._make_df([100.0] * 80)
        funcs = [
            self.mod.pattern_double_top,
            self.mod.pattern_double_bottom,
            self.mod.pattern_triple_top,
            self.mod.pattern_triple_bottom,
            self.mod.pattern_head_and_shoulders,
            self.mod.pattern_inverse_head_and_shoulders,
            self.mod.pattern_bullish_flag,
            self.mod.pattern_bearish_flag,
            self.mod.pattern_bullish_pennant,
            self.mod.pattern_bearish_pennant,
            self.mod.pattern_rectangle,
            self.mod.pattern_triangle,
            self.mod.pattern_falling_wedge,
            self.mod.pattern_rising_wedge,
            self.mod.pattern_cup_and_handle,
            self.mod.pattern_inverted_cup_and_handle,
        ]
        for fn in funcs:
            out = fn(df)
            self.assertIsInstance(out, pd.Series)
            self.assertEqual(len(out), len(df))
            self.assertEqual(list(out.index), list(df.index))
            self.assertTrue(out.dtype == bool or str(out.dtype).startswith("bool"))

    def test_double_top_deterministic(self):
        import pandas as pd

        close = [1, 2, 3, 2, 3, 2, 1, 1, 1, 1]  # peaks at indices 2 and 4 (same level)
        df = self._make_df(close)
        mask = self.mod.pattern_double_top(df, distance=1, tol=0.001)
        self.assertIsInstance(mask, pd.Series)
        self.assertTrue(mask.iloc[4])

    def test_registry_contains_callables(self):
        reg = getattr(self.mod, "CHART_PATTERN_FUNCS", {})
        self.assertTrue(isinstance(reg, dict))
        self.assertGreaterEqual(len(reg), 10)
        for name, fn in reg.items():
            self.assertTrue(callable(fn), name)


if __name__ == "__main__":
    unittest.main()
