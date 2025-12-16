import numpy as np
import pandas as pd
from utils.indicators_extended import TrendIndicators


def make_linear_df(n=60, start=100, end=120):
    close = np.linspace(start, end, n)
    high = close + 1.0
    low = close - 1.0
    tick_volume = np.ones(n) * 100
    df = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close,
        'tick_volume': tick_volume,
    })
    return df


def test_calculate_adx_basic_properties():
    df = make_linear_df()
    adx, plus_di, minus_di = TrendIndicators.calculate_adx(df, period=14)

    # outputs should be pandas Series of the same length
    assert len(adx) == len(df)
    assert len(plus_di) == len(df)
    assert len(minus_di) == len(df)

    # For monotonically increasing series, plus_di should dominate minus_di
    # check last non-NaN values
    last_plus = plus_di.dropna().iloc[-1]
    last_minus = minus_di.dropna().iloc[-1]
    assert last_plus >= 0
    assert last_plus >= last_minus


def test_calculate_ema_slope_increasing():
    df = make_linear_df(n=50, start=50, end=100)
    ema, slope_norm = TrendIndicators.calculate_ema_slope(df, period=10, lookback=3)

    # EMA should be increasing for increasing prices
    diffs = ema.diff().dropna()
    assert (diffs >= -1e-8).all()

    # slope normalized should be positive at the end
    assert slope_norm.dropna().iloc[-1] > 0


def test_calculate_donchian_and_direction():
    df = make_linear_df(n=30)
    high_ch, low_ch, mid_ch, direction = TrendIndicators.calculate_donchian(df, period=5)

    assert len(high_ch) == len(df)
    assert len(low_ch) == len(df)
    assert len(mid_ch) == len(df)
    assert len(direction) == len(df)

    # mid channel should be between high and low
    idx = -1
    assert low_ch.iloc[idx] <= mid_ch.iloc[idx] <= high_ch.iloc[idx]

    # direction values must be either 1 or -1
    unique_dirs = set(direction.tolist())
    assert unique_dirs.issubset({1, -1})


def test_calculate_vwap_constant_price():
    # If price is constant, VWAP should equal that price
    n = 20
    price = 123.45
    df = pd.DataFrame({
        'high': np.full(n, price),
        'low': np.full(n, price),
        'close': np.full(n, price),
        'tick_volume': np.arange(1, n + 1),
    })

    vwap = TrendIndicators.calculate_vwap(df)
    # last value should equal price (within tolerance)
    assert abs(vwap.iloc[-1] - price) < 1e-8


def test_calculate_volatility_compression_constant_atr():
    # Create data where true range is constant -> compression ratio ~1
    n = 200
    close = np.linspace(100, 100.5, n)
    high = close + 0.5
    low = close - 0.5
    df = pd.DataFrame({'high': high, 'low': low, 'close': close})

    comp = TrendIndicators.calculate_volatility_compression(df, atr_period=14, compression_window=50)
    # compression should be finite and near 1 for later values
    tail = comp.dropna().iloc[-10:]
    assert len(tail) > 0
    mean_tail = float(tail.mean())
    assert 0.9 < mean_tail < 1.1
