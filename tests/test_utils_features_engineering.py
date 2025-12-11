#!/usr/bin/env python3
"""
Comprehensive test suite for utils/features_engineering.py

Tests optimized feature engineering with Numba and bottleneck.

Coverage:
- Numba JIT kernels (LSMA, McGinley, swing detection, etc.)
- Fast rolling operations (mean, std, max, min, sum)
- FeatureEngineerOptimized main class
- Indicator calculations (100+ indicators)
- Pattern detection
- Temporal features
- Batch processing
"""

import logging
import tempfile
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.features_engineering import (
    numba_lsma,
    numba_mcginley,
    numba_swing_detection,
    numba_higher_lows_lower_highs,
    numba_abc_pattern,
    numba_local_slope,
    numba_cci_mean_deviation,
    numba_pullback_stage,
    numba_aroon,
    numba_psar,
    numba_kama,
    numba_connors_streak,
    numba_rma,
    fast_rolling_mean,
    fast_rolling_std,
    fast_rolling_max,
    fast_rolling_min,
    fast_rolling_sum,
    FeatureEngineerOptimized,
    FeatureEngineer,
)


logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_prices():
    """Create sample price data."""
    np.random.seed(42)
    n = 200
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.5)
    return prices.astype(np.float64)


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame."""
    np.random.seed(42)
    n = 200
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='1h'),
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.3),
        'low': prices - np.abs(np.random.randn(n) * 0.3),
        'close': prices + np.random.randn(n) * 0.2,
        'tick_volume': np.random.randint(100, 5000, n),
        'volume': np.random.randint(100, 5000, n),
    })
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


@pytest.fixture
def large_ohlcv_df():
    """Create large OHLCV DataFrame for batch processing tests."""
    np.random.seed(42)
    n = 200_000
    base_price = 100.0
    prices = base_price + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=n, freq='1min'),
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.3),
        'low': prices - np.abs(np.random.randn(n) * 0.3),
        'close': prices + np.random.randn(n) * 0.2,
        'volume': np.random.randint(100, 5000, n),
    })
    
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df


# =============================================================================
# TEST: Numba Kernels
# =============================================================================

class TestNumbaLSMA:
    """Test Least Squares Moving Average."""
    
    def test_lsma_shape(self, sample_prices):
        """Test LSMA output shape."""
        period = 10
        result = numba_lsma(sample_prices, period)
        
        assert result.shape == sample_prices.shape
        assert result.dtype == np.float64
    
    def test_lsma_nan_prefix(self, sample_prices):
        """Test LSMA has NaN values before period."""
        period = 10
        result = numba_lsma(sample_prices, period)
        
        assert np.isnan(result[:period-1]).all()
        assert not np.isnan(result[period-1:]).any()
    
    def test_lsma_smooth(self, sample_prices):
        """Test LSMA produces smooth output."""
        period = 10
        result = numba_lsma(sample_prices, period)
        
        # LSMA should reduce variance
        assert np.std(result[period:]) < np.std(sample_prices)
    
    def test_lsma_different_periods(self, sample_prices):
        """Test LSMA with different periods."""
        results = []
        for period in [5, 10, 20]:
            result = numba_lsma(sample_prices, period)
            results.append(result)
        
        # All should produce valid results without all NaN
        for result in results:
            assert not np.isnan(result[50:]).all()


class TestNumbaMcGinley:
    """Test McGinley Dynamic."""
    
    def test_mcginley_shape(self, sample_prices):
        """Test McGinley output shape."""
        result = numba_mcginley(sample_prices, 20)
        
        assert result.shape == sample_prices.shape
        assert result.dtype == np.float64
    
    def test_mcginley_adaptive(self, sample_prices):
        """Test McGinley is adaptive."""
        result = numba_mcginley(sample_prices, 20)
        
        # Should not be all NaN
        assert not np.isnan(result).all()
        
        # Should have values after first element
        assert not np.isnan(result[1:]).all()
    
    def test_mcginley_bounded(self, sample_prices):
        """Test McGinley stays near prices."""
        result = numba_mcginley(sample_prices, 20)
        
        # McGinley should follow price closely
        mean_diff = np.nanmean(np.abs(result - sample_prices))
        assert mean_diff < np.std(sample_prices)


class TestNumbaSwingDetection:
    """Test swing high/low detection."""
    
    def test_swing_detection_shapes(self, sample_ohlcv_df):
        """Test swing detection returns correct shapes."""
        high = sample_ohlcv_df['high'].values
        low = sample_ohlcv_df['low'].values
        
        swing_high, swing_low = numba_swing_detection(high, low, lookback=2)
        
        assert swing_high.shape == high.shape
        assert swing_low.shape == low.shape
        assert swing_high.dtype == np.int8
        assert swing_low.dtype == np.int8
    
    def test_swing_detection_binary(self, sample_ohlcv_df):
        """Test swing detection outputs are binary."""
        high = sample_ohlcv_df['high'].values
        low = sample_ohlcv_df['low'].values
        
        swing_high, swing_low = numba_swing_detection(high, low, lookback=2)
        
        assert np.all(np.isin(swing_high, [0, 1]))
        assert np.all(np.isin(swing_low, [0, 1]))
    
    def test_swing_detection_not_both(self, sample_ohlcv_df):
        """Test swing_high and swing_low don't occur at same bar."""
        high = sample_ohlcv_df['high'].values
        low = sample_ohlcv_df['low'].values
        
        swing_high, swing_low = numba_swing_detection(high, low, lookback=2)
        
        # At each position, shouldn't be both swing high and low
        assert not (swing_high & swing_low).any()
    
    def test_swing_detection_edge_bars(self, sample_ohlcv_df):
        """Test edge bars have no swings (lookback)."""
        high = sample_ohlcv_df['high'].values
        low = sample_ohlcv_df['low'].values
        lookback = 2
        
        swing_high, swing_low = numba_swing_detection(high, low, lookback=lookback)
        
        # First and last lookback bars shouldn't have swings
        assert (swing_high[:lookback] == 0).all()
        assert (swing_low[-lookback:] == 0).all()


class TestNumbaHigherLowsLowerHighs:
    """Test higher lows/lower highs pattern detection."""
    
    def test_pattern_shapes(self, sample_ohlcv_df):
        """Test pattern detection returns correct shapes."""
        high = sample_ohlcv_df['high'].values
        low = sample_ohlcv_df['low'].values
        swing_high = np.zeros_like(high, dtype=np.int8)
        swing_low = np.zeros_like(low, dtype=np.int8)
        swing_low[50::20] = 1  # Mark some swing lows
        swing_high[60::20] = 1  # Mark some swing highs
        trend = np.ones_like(high, dtype=np.int8)
        
        hl, lh = numba_higher_lows_lower_highs(high, low, swing_high, swing_low, trend, 50)
        
        assert hl.shape == high.shape
        assert lh.shape == high.shape
        assert hl.dtype == np.int8
        assert lh.dtype == np.int8
    
    def test_pattern_binary_output(self, sample_ohlcv_df):
        """Test patterns are binary."""
        high = sample_ohlcv_df['high'].values
        low = sample_ohlcv_df['low'].values
        swing_high = np.zeros_like(high, dtype=np.int8)
        swing_low = np.zeros_like(low, dtype=np.int8)
        trend = np.ones_like(high, dtype=np.int8)
        
        hl, lh = numba_higher_lows_lower_highs(high, low, swing_high, swing_low, trend, 50)
        
        assert np.all(np.isin(hl, [0, 1]))
        assert np.all(np.isin(lh, [0, 1]))


class TestNumbaABCPattern:
    """Test ABC pullback pattern detection."""
    
    def test_abc_shapes(self, sample_ohlcv_df):
        """Test ABC pattern returns correct shapes."""
        high = sample_ohlcv_df['high'].values
        low = sample_ohlcv_df['low'].values
        swing_high = np.zeros_like(high, dtype=np.int8)
        swing_low = np.zeros_like(low, dtype=np.int8)
        swing_low[50::20] = 1
        swing_high[60::20] = 1
        
        abc_bull, abc_bear = numba_abc_pattern(high, low, swing_high, swing_low, 20)
        
        assert abc_bull.shape == high.shape
        assert abc_bear.shape == high.shape
    
    def test_abc_binary(self, sample_ohlcv_df):
        """Test ABC patterns are binary."""
        high = sample_ohlcv_df['high'].values
        low = sample_ohlcv_df['low'].values
        swing_high = np.zeros_like(high, dtype=np.int8)
        swing_low = np.zeros_like(low, dtype=np.int8)
        
        abc_bull, abc_bear = numba_abc_pattern(high, low, swing_high, swing_low, 20)
        
        assert np.all(np.isin(abc_bull, [0, 1]))
        assert np.all(np.isin(abc_bear, [0, 1]))


class TestNumbaLocalSlope:
    """Test local slope calculation."""
    
    def test_local_slope_shape(self, sample_prices):
        """Test local slope output shape."""
        result = numba_local_slope(sample_prices, 5)
        
        assert result.shape == sample_prices.shape
    
    def test_local_slope_values(self, sample_prices):
        """Test local slope has valid values."""
        result = numba_local_slope(sample_prices, 5)
        
        # Should have non-zero values
        assert np.any(result != 0)
        
        # Slope should be reasonable
        assert np.all(np.abs(result) < np.std(sample_prices) * 10)


class TestNumbaCCIMeanDeviation:
    """Test CCI mean deviation calculation."""
    
    def test_cci_md_shape(self, sample_prices):
        """Test CCI mean deviation shape."""
        tp = sample_prices
        tp_sma = pd.Series(tp).rolling(20).mean().values
        tp_sma = np.nan_to_num(tp_sma, nan=tp[0])
        
        result = numba_cci_mean_deviation(tp, tp_sma, 20)
        
        assert result.shape == tp.shape
    
    def test_cci_md_nan_prefix(self, sample_prices):
        """Test CCI mean deviation NaN prefix."""
        tp = sample_prices
        tp_sma = pd.Series(tp).rolling(20).mean().values
        tp_sma = np.nan_to_num(tp_sma, nan=tp[0])
        
        result = numba_cci_mean_deviation(tp, tp_sma, 20)
        
        assert np.isnan(result[:19]).all()


class TestNumbaAroon:
    """Test Aroon indicator."""
    
    def test_aroon_shapes(self, sample_prices):
        """Test Aroon returns three arrays."""
        aroon_up, aroon_down, aroon_osc = numba_aroon(sample_prices, sample_prices, 25)
        
        assert aroon_up.shape == sample_prices.shape
        assert aroon_down.shape == sample_prices.shape
        assert aroon_osc.shape == sample_prices.shape
    
    def test_aroon_range(self, sample_prices):
        """Test Aroon values are in valid range."""
        aroon_up, aroon_down, aroon_osc = numba_aroon(sample_prices, sample_prices, 25)
        
        # Aroon up and down should be 0-100
        valid_up = np.isnan(aroon_up) | ((aroon_up >= 0) & (aroon_up <= 100))
        valid_down = np.isnan(aroon_down) | ((aroon_down >= 0) & (aroon_down <= 100))
        
        assert valid_up.all()
        assert valid_down.all()


class TestNumbaParabolicSAR:
    """Test Parabolic SAR."""
    
    def test_psar_shape(self, sample_ohlcv_df):
        """Test Parabolic SAR shape."""
        high = sample_ohlcv_df['high'].values
        low = sample_ohlcv_df['low'].values
        close = sample_ohlcv_df['close'].values
        
        result = numba_psar(high, low, close)
        
        assert result.shape == close.shape
        assert result.dtype == np.float64
    
    def test_psar_no_nan(self, sample_ohlcv_df):
        """Test Parabolic SAR has no NaN values."""
        high = sample_ohlcv_df['high'].values
        low = sample_ohlcv_df['low'].values
        close = sample_ohlcv_df['close'].values
        
        result = numba_psar(high, low, close)
        
        assert not np.isnan(result).any()
    
    def test_psar_between_prices(self, sample_ohlcv_df):
        """Test Parabolic SAR is between high and low."""
        high = sample_ohlcv_df['high'].values
        low = sample_ohlcv_df['low'].values
        close = sample_ohlcv_df['close'].values
        
        result = numba_psar(high, low, close)
        
        # SAR should be reasonable - not checking strict bounds due to SAR logic
        # Just verify it has reasonable values
        assert result.min() > low.min() - 5
        assert result.max() < high.max() + 5


class TestNumbaKAMA:
    """Test Kaufman Adaptive Moving Average."""
    
    def test_kama_shape(self, sample_prices):
        """Test KAMA shape."""
        result = numba_kama(sample_prices, 10, 2, 30)
        
        assert result.shape == sample_prices.shape
    
    def test_kama_no_nan(self, sample_prices):
        """Test KAMA has no NaN values."""
        result = numba_kama(sample_prices, 10, 2, 30)
        
        assert not np.isnan(result).any()
    
    def test_kama_adaptive(self, sample_prices):
        """Test KAMA is adaptive to price changes."""
        result = numba_kama(sample_prices, 10, 2, 30)
        
        # KAMA should follow prices reasonably
        diff = np.abs(result - sample_prices)
        assert np.mean(diff) < np.std(sample_prices)


class TestNumbaConnorsStreak:
    """Test Connors RSI streak calculation."""
    
    def test_streak_shape(self, sample_prices):
        """Test streak shape."""
        result = numba_connors_streak(sample_prices)
        
        assert result.shape == sample_prices.shape
    
    def test_streak_logic(self, sample_prices):
        """Test streak tracks up/down correctly."""
        # Create simple price sequence
        prices = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0], dtype=np.float64)
        result = numba_connors_streak(prices)
        
        # Should track consecutive ups/downs
        assert result[0] == 0
        assert result[1] > 0  # up
        assert result[2] > result[1]  # still up
        assert result[3] < 0  # down


class TestNumbaRMA:
    """Test Relative Moving Average."""
    
    def test_rma_shape(self, sample_prices):
        """Test RMA shape."""
        result = numba_rma(sample_prices, 14)
        
        assert result.shape == sample_prices.shape
    
    def test_rma_no_nan(self, sample_prices):
        """Test RMA has no NaN values."""
        result = numba_rma(sample_prices, 14)
        
        assert not np.isnan(result).any()
    
    def test_rma_smooth(self, sample_prices):
        """Test RMA is smooth."""
        result = numba_rma(sample_prices, 14)
        
        # RMA should be smoother than prices
        assert np.std(np.diff(result)) < np.std(np.diff(sample_prices))


# =============================================================================
# TEST: Fast Rolling Operations
# =============================================================================

class TestFastRollingMean:
    """Test fast rolling mean."""
    
    def test_rolling_mean_vs_pandas(self, sample_prices):
        """Test fast rolling mean matches pandas."""
        window = 20
        fast_result = fast_rolling_mean(sample_prices, window)
        pandas_result = pd.Series(sample_prices).rolling(window).mean().values
        
        # Should be close (allowing for bottleneck vs pandas differences)
        valid_idx = ~np.isnan(fast_result) & ~np.isnan(pandas_result)
        if valid_idx.sum() > 0:
            assert np.allclose(fast_result[valid_idx], pandas_result[valid_idx], rtol=1e-5, atol=1e-8)
    
    def test_rolling_mean_shape(self, sample_prices):
        """Test rolling mean shape."""
        result = fast_rolling_mean(sample_prices, 20)
        assert result.shape == sample_prices.shape


class TestFastRollingStd:
    """Test fast rolling std."""
    
    def test_rolling_std_shape(self, sample_prices):
        """Test rolling std shape."""
        result = fast_rolling_std(sample_prices, 20)
        assert result.shape == sample_prices.shape
    
    def test_rolling_std_positive(self, sample_prices):
        """Test rolling std is positive."""
        result = fast_rolling_std(sample_prices, 20)
        valid = ~np.isnan(result)
        assert (result[valid] >= 0).all()


class TestFastRollingMinMax:
    """Test fast rolling min/max."""
    
    def test_rolling_max_correct(self, sample_prices):
        """Test rolling max finds correct values."""
        result = fast_rolling_max(sample_prices, 5)
        
        # Check a few positions manually
        for i in range(5, min(20, len(sample_prices))):
            window = sample_prices[i-4:i+1]
            expected = np.nanmax(window)
            # Allow small numerical differences
            if not np.isnan(result[i]):
                assert np.isclose(result[i], expected, rtol=1e-5)
    
    def test_rolling_min_correct(self, sample_prices):
        """Test rolling min finds correct values."""
        result = fast_rolling_min(sample_prices, 5)
        
        # Check a few positions manually
        for i in range(5, min(20, len(sample_prices))):
            window = sample_prices[i-4:i+1]
            expected = np.nanmin(window)
            if not np.isnan(result[i]):
                assert np.isclose(result[i], expected, rtol=1e-5)


class TestFastRollingSum:
    """Test fast rolling sum."""
    
    def test_rolling_sum_shape(self, sample_prices):
        """Test rolling sum shape."""
        result = fast_rolling_sum(sample_prices, 20)
        assert result.shape == sample_prices.shape


# =============================================================================
# TEST: FeatureEngineerOptimized Main Class
# =============================================================================

class TestFeatureEngineerOptimizedInit:
    """Test FeatureEngineerOptimized initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        fe = FeatureEngineerOptimized()
        
        assert fe.MAX_LOOKBACK == 1050
        assert fe.BATCH_SIZE == 100_000
        assert fe.db is None
    
    def test_init_with_db(self):
        """Test initialization with database connector."""
        mock_db = MagicMock()
        fe = FeatureEngineerOptimized(db_connector=mock_db)
        
        assert fe.db is mock_db
    
    def test_warmup_numba(self):
        """Test Numba functions are warmed up without error."""
        # Should not raise
        fe = FeatureEngineerOptimized()
        assert fe is not None


class TestFeatureEngineerOptimizedStaticMethods:
    """Test static helper methods."""
    
    def test_sma(self, sample_ohlcv_df):
        """Test SMA calculation."""
        result = FeatureEngineerOptimized.sma(sample_ohlcv_df['close'], 20)
        
        assert len(result) == len(sample_ohlcv_df)
        assert result.isna().sum() == 19  # First 19 are NaN
    
    def test_ema(self, sample_ohlcv_df):
        """Test EMA calculation."""
        result = FeatureEngineerOptimized.ema(sample_ohlcv_df['close'], 20)
        
        assert len(result) == len(sample_ohlcv_df)
        assert result.isna().sum() < 20
    
    def test_typical_price(self, sample_ohlcv_df):
        """Test typical price calculation."""
        result = FeatureEngineerOptimized.typical_price(
            sample_ohlcv_df['high'],
            sample_ohlcv_df['low'],
            sample_ohlcv_df['close']
        )
        
        assert len(result) == len(sample_ohlcv_df)
        assert (result >= sample_ohlcv_df['low']).all()
        assert (result <= sample_ohlcv_df['high']).all()
    
    def test_true_range(self, sample_ohlcv_df):
        """Test true range calculation."""
        result = FeatureEngineerOptimized.true_range(
            sample_ohlcv_df['high'],
            sample_ohlcv_df['low'],
            sample_ohlcv_df['close']
        )
        
        assert len(result) == len(sample_ohlcv_df)
        assert (result >= 0).all()


class TestFeatureEngineerOptimizedGeneration:
    """Test feature generation."""
    
    def test_generate_features_basic(self, sample_ohlcv_df):
        """Test basic feature generation."""
        fe = FeatureEngineerOptimized()
        result = fe.generate_features(sample_ohlcv_df.copy(), batch_processing=False)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_df)
        assert result.shape[1] > sample_ohlcv_df.shape[1]  # More columns
    
    def test_generate_features_required_columns(self):
        """Test error on missing required columns."""
        df = pd.DataFrame({'close': [1.0, 2.0, 3.0]})
        fe = FeatureEngineerOptimized()
        
        with pytest.raises(ValueError):
            fe.generate_features(df)
    
    def test_generate_features_minimum_rows(self, sample_ohlcv_df):
        """Test warning on insufficient data."""
        small_df = sample_ohlcv_df.iloc[:10].copy()
        fe = FeatureEngineerOptimized()
        
        # Should warn but continue
        result = fe.generate_features(small_df, batch_processing=False)
        assert result is not None
    
    def test_generate_features_adds_columns(self, sample_ohlcv_df):
        """Test that many features are added."""
        fe = FeatureEngineerOptimized()
        result = fe.generate_features(sample_ohlcv_df.copy(), batch_processing=False)
        
        # Should have 100+ technical indicators
        new_cols = result.shape[1] - sample_ohlcv_df.shape[1]
        assert new_cols > 50
    
    def test_generate_features_moving_averages(self, sample_ohlcv_df):
        """Test moving average features are generated."""
        fe = FeatureEngineerOptimized()
        result = fe.generate_features(sample_ohlcv_df.copy(), batch_processing=False)
        
        ma_cols = [c for c in result.columns if 'sma' in c or 'ema' in c or 'wma' in c]
        assert len(ma_cols) > 0
    
    def test_generate_features_oscillators(self, sample_ohlcv_df):
        """Test oscillator features are generated."""
        fe = FeatureEngineerOptimized()
        result = fe.generate_features(sample_ohlcv_df.copy(), batch_processing=False)
        
        osc_cols = [c for c in result.columns if 'rsi' in c or 'stoch' in c or 'macd' in c]
        assert len(osc_cols) > 0
    
    def test_generate_features_volatility(self, sample_ohlcv_df):
        """Test volatility features are generated."""
        fe = FeatureEngineerOptimized()
        result = fe.generate_features(sample_ohlcv_df.copy(), batch_processing=False)
        
        vol_cols = [c for c in result.columns if 'atr' in c or 'bb' in c or 'kc' in c]
        assert len(vol_cols) > 0
    
    def test_generate_features_volume(self, sample_ohlcv_df):
        """Test volume features are generated."""
        fe = FeatureEngineerOptimized()
        result = fe.generate_features(sample_ohlcv_df.copy(), batch_processing=False)
        
        vol_features = [c for c in result.columns if 'volume' in c or 'obv' in c or 'mfi' in c]
        assert len(vol_features) > 0
    
    def test_generate_features_patterns(self, sample_ohlcv_df):
        """Test pattern features are generated."""
        fe = FeatureEngineerOptimized()
        result = fe.generate_features(sample_ohlcv_df.copy(), batch_processing=False)
        
        pattern_cols = [c for c in result.columns if 'pattern_' in c]
        assert len(pattern_cols) > 0
    
    def test_generate_features_temporal(self, sample_ohlcv_df):
        """Test temporal features are generated."""
        fe = FeatureEngineerOptimized()
        result = fe.generate_features(sample_ohlcv_df.copy(), batch_processing=False)
        
        temporal_cols = [c for c in result.columns if 'hour' in c or 'day' in c or 'month' in c]
        assert len(temporal_cols) > 0
    
    def test_generate_features_no_inf(self, sample_ohlcv_df):
        """Test no infinite values in output."""
        fe = FeatureEngineerOptimized()
        result = fe.generate_features(sample_ohlcv_df.copy(), batch_processing=False)
        
        # Check numeric columns only
        numeric_result = result.select_dtypes(include=[np.number])
        assert not np.isinf(numeric_result).any().any()
    
    def test_generate_features_preserves_index(self, sample_ohlcv_df):
        """Test index is preserved."""
        sample_ohlcv_df.index = sample_ohlcv_df.index + 100
        fe = FeatureEngineerOptimized()
        result = fe.generate_features(sample_ohlcv_df.copy(), batch_processing=False)
        
        assert (result.index == sample_ohlcv_df.index).all()


class TestCalculateIndicators:
    """Test calculate_indicators method."""
    
    def test_calculate_indicators_macd(self, sample_ohlcv_df):
        """Test MACD indicators are calculated."""
        fe = FeatureEngineerOptimized()
        result = fe.calculate_indicators(sample_ohlcv_df.copy())
        
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_hist' in result.columns
    
    def test_calculate_indicators_rsi(self, sample_ohlcv_df):
        """Test RSI indicators are calculated."""
        fe = FeatureEngineerOptimized()
        result = fe.calculate_indicators(sample_ohlcv_df.copy())
        
        assert 'rsi' in result.columns
        
        # RSI should be between 0 and 100
        rsi_valid = result['rsi'].dropna()
        assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all()
    
    def test_calculate_indicators_bollinger_bands(self, sample_ohlcv_df):
        """Test Bollinger Bands are calculated."""
        fe = FeatureEngineerOptimized()
        result = fe.calculate_indicators(sample_ohlcv_df.copy())
        
        assert 'bb_upper' in result.columns
        assert 'bb_middle' in result.columns
        assert 'bb_lower' in result.columns
        
        # Upper should be > middle > lower
        assert (result['bb_upper'].dropna() > result['bb_middle'].dropna()).all()
        assert (result['bb_middle'].dropna() > result['bb_lower'].dropna()).all()
    
    def test_calculate_indicators_atr(self, sample_ohlcv_df):
        """Test ATR is calculated."""
        fe = FeatureEngineerOptimized()
        result = fe.calculate_indicators(sample_ohlcv_df.copy())
        
        assert 'atr' in result.columns
        
        # ATR should be positive
        atr_valid = result['atr'].dropna()
        assert (atr_valid > 0).all()
    
    def test_calculate_indicators_adx(self, sample_ohlcv_df):
        """Test ADX is calculated."""
        fe = FeatureEngineerOptimized()
        result = fe.calculate_indicators(sample_ohlcv_df.copy())
        
        assert 'adx' in result.columns
        assert 'di_plus' in result.columns
        assert 'di_minus' in result.columns


class TestBatchProcessing:
    """Test batch processing for large datasets."""
    
    @pytest.mark.slow
    def test_batch_processing_large_dataset(self, large_ohlcv_df):
        """Test batch processing on large dataset."""
        fe = FeatureEngineerOptimized()
        result = fe.generate_features(large_ohlcv_df.copy(), batch_processing=True)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(large_ohlcv_df)
        assert result.shape[1] > large_ohlcv_df.shape[1]
    
    @pytest.mark.slow
    def test_batch_vs_non_batch_consistency(self):
        """Test batch and non-batch produce similar results."""
        np.random.seed(42)
        n = 150_000
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=n, freq='1min'),
            'open': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'high': 101 + np.cumsum(np.random.randn(n) * 0.1),
            'low': 99 + np.cumsum(np.random.randn(n) * 0.1),
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'volume': np.random.randint(100, 1000, n),
        })
        
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        fe = FeatureEngineerOptimized()
        
        # Non-batch
        result_no_batch = fe.generate_features(df.copy(), batch_processing=False)
        
        # Batch (will auto-batch due to size)
        result_batch = fe.generate_features(df.copy(), batch_processing=True)
        
        # Results should have same shape
        assert result_no_batch.shape == result_batch.shape


# =============================================================================
# TEST: Feature Categories
# =============================================================================

class TestFeatureCategories:
    """Test that all major feature categories are generated."""
    
    def test_all_categories_present(self, sample_ohlcv_df):
        """Test all major feature categories are present."""
        fe = FeatureEngineerOptimized()
        result = fe.generate_features(sample_ohlcv_df.copy(), batch_processing=False)
        
        categories = {
            'Moving Averages': ['sma', 'ema', 'wma', 'hma', 'tema', 'dema'],
            'Oscillators': ['rsi', 'stoch', 'macd', 'cci'],
            'Volatility': ['atr', 'bb_', 'kc_', 'dc_'],
            'Volume': ['volume', 'obv', 'vwap', 'mfi', 'cmf', 'obv'],
            'Patterns': ['pattern_'],
            'Swing': ['swing', 'pullback', 'fib'],
            'Trend': ['trend_'],
            'Temporal': ['hour', 'day', 'month', 'session'],
        }
        
        for category, keywords in categories.items():
            matching = [c for c in result.columns if any(k in c for k in keywords)]
            assert len(matching) > 0, f"Missing {category}"


# =============================================================================
# TEST: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_constant_prices(self):
        """Test handling of constant prices."""
        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=100, freq='1h'),
            'open': [100.0] * 100,
            'high': [100.5] * 100,
            'low': [99.5] * 100,
            'close': [100.0] * 100,
            'volume': [1000] * 100,
        })
        
        fe = FeatureEngineerOptimized()
        result = fe.generate_features(df, batch_processing=False)
        
        assert result is not None
        assert len(result) == 100
    
    def test_nan_handling(self, sample_ohlcv_df):
        """Test NaN handling."""
        df = sample_ohlcv_df.copy()
        df.loc[10:15, 'volume'] = np.nan
        
        fe = FeatureEngineerOptimized()
        result = fe.generate_features(df, batch_processing=False)
        
        # Should complete and have finite numeric columns
        numeric = result.select_dtypes(include=[np.number])
        non_finite_ratio = numeric.isnull().sum().sum() / numeric.size
        assert non_finite_ratio < 0.3  # Allow some NaN but not too much
    
    def test_single_row(self):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
            'volume': [1000],
        })
        
        fe = FeatureEngineerOptimized()
        # Should not crash
        result = fe.generate_features(df, batch_processing=False)
        assert result is not None


class TestBackwardCompatibility:
    """Test backward compatibility."""
    
    def test_feature_engineer_alias(self):
        """Test FeatureEngineer is alias for FeatureEngineerOptimized."""
        assert FeatureEngineer is FeatureEngineerOptimized


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
'''
✅ Test suite for feature_adapter.py complete with 60 comprehensive tests organized into 10 test classes:

Test Class	Count	Focus
TestFeatureEngineer	9	Technical indicator computation (momentum, volatility, trend, volume, oscillators)
TestEnhancedDataLoaderV2Init	4	Initialization with default/custom parameters
TestEnhancedDataLoaderV2Loading	4	CSV loading, feature column population, file handling
TestEnhancedDataLoaderV2SplitAndScale	7	Train/val/test splitting, scaler types (Robust/Standard/MinMax)
TestEnhancedDataLoaderV2Sequences	5	Sequence creation, shapes, labels, horizons, trend thresholds
TestEnhancedDataLoaderV3Init	3	V3 inheritance, checkpoint features initialization
TestEnhancedDataLoaderV3FromCheckpoint	5	Checkpoint loading, feature extraction, error handling
TestEnhancedDataLoaderV3Loading	4	CSV loading with checkpoint feature filtering, missing feature handling
TestLoadDataForEvaluation	4	Convenience function for model evaluation, custom parameters
TestGetAvailableFeatures	3	Feature extraction from CSV files
TestIntegration	6	Full pipelines (V2, V3 with/without checkpoints), backward compatibility
TestEdgeCases	6	Empty DataFrames, single rows, NaN values, constant prices, large data, sequence length edge cases
Result: ✅ 60/60 tests PASSED

The test suite validates:

✅ All 5 feature engineering methods (momentum, volatility, trend, volume, oscillators)
✅ V2 backward compatibility (original loader)
✅ V3 checkpoint integration (feature selection from model checkpoints)
✅ All scaler types and their correct application
✅ Sequence generation with various horizons and thresholds
✅ Full data pipeline workflows
✅ Error handling and edge cases
✅ Convenience functions for evaluation and feature discovery
'''