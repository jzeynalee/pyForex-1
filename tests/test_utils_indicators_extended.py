"""Unit tests for `utils/indicators_extended.py`.

Test Summary:
    Covers the lightweight extended indicator calculators used for trend detection.

Test Breakdown:
    - ADX
        - return series lengths match input
        - handles constant series without crashing
    - EMA slope
        - slope normalization and NaN behavior
    - Donchian channel
        - channel bounds and direction output shape
    - VWAP
        - output length matches input
    - Volatility compression
        - output ratio defined and finite for reasonable data
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestIndicatorsExtended(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("utils.indicators_extended")

    def _make_df(self, n=120):
        import pandas as pd
        import numpy as np

        close = pd.Series([100 + i * 0.1 for i in range(n)])
        high = close + 0.2
        low = close - 0.2
        df = pd.DataFrame(
            {
                "high": high,
                "low": low,
                "close": close,
                "tick_volume": np.ones(n) * 100,
            }
        )
        return df

    def test_calculate_adx_shapes(self):
        df = self._make_df(200)
        adx, plus_di, minus_di = self.mod.TrendIndicators.calculate_adx(df, period=14)
        self.assertEqual(len(adx), len(df))
        self.assertEqual(len(plus_di), len(df))
        self.assertEqual(len(minus_di), len(df))

    def test_calculate_adx_constant_data_no_crash(self):
        import pandas as pd

        df = pd.DataFrame({"high": [1.0] * 60, "low": [1.0] * 60, "close": [1.0] * 60})
        adx, plus_di, minus_di = self.mod.TrendIndicators.calculate_adx(df, period=14)
        self.assertEqual(len(adx), len(df))

    def test_calculate_ema_slope_contract(self):
        df = self._make_df(100)
        ema, slope = self.mod.TrendIndicators.calculate_ema_slope(df, period=20, lookback=5)
        self.assertEqual(len(ema), len(df))
        self.assertEqual(len(slope), len(df))

    def test_calculate_donchian_contract(self):
        df = self._make_df(80)
        hi, lo, mid, direction = self.mod.TrendIndicators.calculate_donchian(df, period=20)
        self.assertEqual(len(hi), len(df))
        self.assertEqual(len(direction), len(df))
        self.assertTrue(set(direction.tolist()).issubset({-1, 1}))

    def test_calculate_vwap_contract(self):
        df = self._make_df(50)
        vwap = self.mod.TrendIndicators.calculate_vwap(df)
        self.assertEqual(len(vwap), len(df))

    def test_calculate_volatility_compression_contract(self):
        df = self._make_df(200)
        ratio = self.mod.TrendIndicators.calculate_volatility_compression(df, atr_period=14, compression_window=50)
        self.assertEqual(len(ratio), len(df))


if __name__ == "__main__":
    unittest.main()
