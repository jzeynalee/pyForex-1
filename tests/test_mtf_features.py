import numpy as np
import pandas as pd

from utils.mtf_features import build_ml_features, MTFFeatureBuilder, MTFFeatureSet


def make_tf_df(n=220, start=100.0, end=120.0, freq='T'):
    close = np.linspace(start, end, n)
    open_ = close - 0.1
    high = close + 0.5
    low = close - 0.5
    time = pd.date_range('2025-01-01', periods=n, freq=freq)
    df = pd.DataFrame({
        'time': time,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
    })
    return df


def test_build_ml_features_basic_keys_and_types():
    # single bullish timeframe
    df = make_tf_df(n=240, start=100, end=150)
    features = build_ml_features({'M1': df}, primary_tf='M1')

    # check presence of some expected keys
    assert 'M1_price_vs_ema20' in features
    assert 'M1_adx' in features
    assert 'mtf_weighted_direction' in features
    assert 'mtf_direction_alignment' in features

    # numeric types
    assert isinstance(features['M1_adx'], (int, float))
    assert isinstance(features['mtf_weighted_direction'], float)


def test_mtffeatureset_to_array_and_to_dict():
    # build using builder so feature ordering is realistic
    df = make_tf_df(n=240)
    builder = MTFFeatureBuilder()
    fs = builder.build_features({'M1': df}, primary_tf='M1')

    arr_default = fs.to_array()
    assert isinstance(arr_default, np.ndarray)
    assert arr_default.size == len(fs.feature_names)

    # custom order (reverse)
    order = list(fs.feature_names)[::-1]
    arr_custom = fs.to_array(feature_order=order)
    # values should differ if order changed
    assert arr_custom.shape == arr_default.shape
    assert (arr_custom != arr_default).any()

    d = fs.to_dict()
    assert isinstance(d, dict)
    assert d.get(next(iter(fs.feature_names))) == fs.features[next(iter(fs.feature_names))]


def test_direction_alignment_counts_multiple_timeframes():
    # create 3 timeframes: two bullish, one bearish
    bullish1 = make_tf_df(n=240, start=100, end=150)
    bullish2 = make_tf_df(n=240, start=200, end=260)
    bearish = make_tf_df(n=240, start=300, end=250)

    builder = MTFFeatureBuilder()
    features = builder.build_features({'M1': bullish1, 'H1': bullish2, 'D1': bearish}, primary_tf='M1')

    # counts
    assert features.features['mtf_bullish_count'] == 2
    assert features.features['mtf_bearish_count'] == 1
    # alignment should be max(bullish,bearish)/3 == 2/3
    assert abs(features.features['mtf_direction_alignment'] - (2/3)) < 1e-6


def test_candle_feature_ratios_edge_cases():
    # create last candle with zero range -> expect defaults (body_ratio 0.5)
    df = make_tf_df(n=50)
    # set last candle to have equal high and low
    df.loc[df.index[-1], 'high'] = 100.0
    df.loc[df.index[-1], 'low'] = 100.0
    df.loc[df.index[-1], 'open'] = 100.0
    df.loc[df.index[-1], 'close'] = 100.0

    builder = MTFFeatureBuilder()
    features = builder.build_features({'M1': df}, primary_tf='M1')

    assert features.features['M1_body_ratio'] == 0.5
    assert features.features['M1_upper_shadow'] == 0.0
    assert features.features['M1_lower_shadow'] == 0.0
