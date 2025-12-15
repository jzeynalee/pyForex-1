import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def ohlcv_df():
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    base = 1.1000
    close = base + np.linspace(0, 0.005, n)
    df = pd.DataFrame(
        {
            "time": idx,
            "open": close - 0.0001,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "volume": np.full(n, 1000, dtype=float),
        }
    )
    df["atr"] = 0.001
    return df
