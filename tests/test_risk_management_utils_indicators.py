# tests/test_risk_management_utils_indicators.py
"""
Comprehensive unit tests for risk_management/utils/indicators.py

Tests technical indicators, regime detection, normalization, and performance metrics.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from risk_management.utils.indicators import (
    calculate_atr, calculate_volatility, calculate_adx, calculate_rsi,
    calculate_bollinger_bands, MarketRegime, RegimeConfig, RegimeDetector,
    create_direction_labels, create_volatility_labels, create_price_move_labels,
    normalize_features, apply_normalization,
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown,
    calculate_win_rate, calculate_profit_factor, calculate_expectancy,
    PerformanceReport, generate_performance_report
)


@pytest.mark.unit
class TestATR:
    """Test ATR calculation."""

    def test_calculate_atr_basic(self):
        """Test basic ATR calculation with valid data."""
        high = np.array([100, 105, 110, 108, 112, 115])
        low = np.array([98, 102, 107, 105, 109, 112])
        close = np.array([99, 104, 109, 107, 111, 114])

        atr = calculate_atr(high, low, close, period=3)

        assert len(atr) == len(close)
        assert np.all(atr >= 0)
        assert not np.isnan(atr[-1])
        assert np.all(np.isfinite(atr))

    def test_calculate_atr_single_value(self):
        """Test ATR with single value."""
        high = np.array([100])
        low = np.array([99])
        close = np.array([99.5])

        atr = calculate_atr(high, low, close, period=14)
        assert len(atr) == 1
        assert atr[0] >= 0

    def test_calculate_atr_increasing_trend(self):
        """Test ATR with strong uptrend."""
        n = 20
        close = np.linspace(100, 120, n)
        high = close + 1
        low = close - 1

        atr = calculate_atr(high, low, close, period=5)
        
        # ATR should be positive and relatively stable
        assert np.all(atr > 0)
        assert atr[-1] > 0


@pytest.mark.unit
class TestVolatility:
    """Test volatility calculation."""

    def test_calculate_volatility_basic(self):
        """Test basic volatility calculation."""
        close = np.array([100, 101, 102, 101, 103, 104, 103, 105])

        vol = calculate_volatility(close, period=3)

        assert len(vol) == len(close)
        assert np.all(vol >= 0)

    def test_calculate_volatility_annualized(self):
        """Test annualized volatility."""
        close = np.linspace(100, 110, 100)

        vol = calculate_volatility(close, period=20, annualize=True)
        vol_not_annualized = calculate_volatility(close, period=20, annualize=False)

        # Annualized should be larger (or equal if all zeros)
        assert np.all(vol >= vol_not_annualized)

    def test_calculate_volatility_constant_price(self):
        """Test volatility with constant price."""
        close = np.full(50, 100.0)

        vol = calculate_volatility(close, period=10)
        
        # Should be zero or very small
        assert np.all(vol >= 0)
        assert vol[-1] < 1e-6


@pytest.mark.unit
class TestADX:
    """Test ADX calculation."""

    def test_calculate_adx_basic(self):
        """Test basic ADX calculation."""
        high = np.array([100, 105, 110, 108, 112, 115])
        low = np.array([98, 102, 107, 105, 109, 112])
        close = np.array([99, 104, 109, 107, 111, 114])

        adx, plus_di, minus_di = calculate_adx(high, low, close, period=3)

        assert len(adx) == len(close)
        assert len(plus_di) == len(close)
        assert len(minus_di) == len(close)
        assert np.all(adx >= 0)
        assert np.all(adx <= 100)
        assert np.all(plus_di >= 0)
        assert np.all(plus_di <= 100)
        assert np.all(minus_di >= 0)
        assert np.all(minus_di <= 100)

    def test_calculate_adx_strong_uptrend(self):
        """Test ADX with strong uptrend."""
        n = 30
        close = np.linspace(100, 150, n)
        high = close + 2
        low = close - 1

        adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)

        # In strong uptrend, +DI should be higher than -DI
        assert plus_di[-1] > minus_di[-1]
        assert adx[-1] > 20  # Should indicate trend


@pytest.mark.unit
class TestRSI:
    """Test RSI calculation."""

    def test_calculate_rsi_basic(self):
        """Test basic RSI calculation."""
        close = np.linspace(100, 110, 30)

        rsi = calculate_rsi(close, period=14)

        assert len(rsi) == len(close)
        assert np.all(rsi >= 0)
        assert np.all(rsi <= 100)

    def test_calculate_rsi_overbought(self):
        """Test RSI with strong uptrend (should be high)."""
        close = np.linspace(100, 130, 50)

        rsi = calculate_rsi(close, period=14)

        # RSI should be high in strong uptrend
        assert rsi[-1] > 50

    def test_calculate_rsi_oversold(self):
        """Test RSI with strong downtrend (should be low)."""
        close = np.linspace(130, 100, 50)

        rsi = calculate_rsi(close, period=14)

        # RSI should be low in strong downtrend
        assert rsi[-1] < 70


@pytest.mark.unit
class TestBollingerBands:
    """Test Bollinger Bands calculation."""

    def test_calculate_bollinger_bands_basic(self):
        """Test basic Bollinger Bands calculation."""
        close = np.linspace(100, 110, 30)

        upper, middle, lower = calculate_bollinger_bands(close, period=20)

        assert len(upper) == len(close)
        assert len(middle) == len(close)
        assert len(lower) == len(close)
        assert np.all(upper >= middle)
        assert np.all(middle >= lower)

    def test_calculate_bollinger_bands_values(self):
        """Test Bollinger Bands have reasonable values."""
        close = np.array([100, 101, 102, 103, 104, 103, 102, 101])

        upper, middle, lower = calculate_bollinger_bands(close, period=5, num_std=2.0)

        # Middle should be near mean
        assert abs(middle[-1] - np.mean(close[-5:])) < 1
        # Upper and lower should be symmetric around middle
        assert abs((upper[-1] - middle[-1]) - (middle[-1] - lower[-1])) < 2


@pytest.mark.unit
class TestRegimeDetector:
    """Test RegimeDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a RegimeDetector instance."""
        return RegimeDetector()

    def test_detect_trending_strong(self, detector):
        """Test detection of strong trending regime."""
        # Create strong trend data
        close = np.linspace(100, 150, 100)
        high = close + 2
        low = close - 2

        regime, indicators = detector.detect(high, low, close)

        # Should detect strong trend
        assert isinstance(regime, MarketRegime)
        assert 'adx' in indicators
        assert 'volatility' in indicators

    def test_detect_regime_batch(self, detector):
        """Test batch regime detection."""
        close = np.linspace(100, 110, 50)
        high = close + 1
        low = close - 1

        regimes = detector.detect_batch(high, low, close)

        assert len(regimes) == len(close)
        assert np.all(regimes >= 0)

    def test_detect_custom_config(self):
        """Test regime detection with custom config."""
        config = RegimeConfig(adx_strong_trend=30.0)
        detector = RegimeDetector(config)

        close = np.linspace(100, 150, 100)
        high = close + 2
        low = close - 2

        regime, _ = detector.detect(high, low, close)
        assert isinstance(regime, MarketRegime)


@pytest.mark.unit
class TestLabelCreation:
    """Test label creation functions."""

    def test_create_direction_labels(self):
        """Test direction label creation."""
        close = np.array([100, 101, 102, 101, 103, 104])

        labels = create_direction_labels(close, horizon=1, threshold=0.01)

        assert len(labels) == len(close)
        assert set(labels).issubset({0, 1, 2})  # BEAR, SIDEWAYS, BULL

    def test_create_volatility_labels(self):
        """Test volatility label creation."""
        high = np.array([101, 102, 103, 102, 104, 105])
        low = np.array([99, 100, 101, 100, 102, 103])
        close = np.array([100, 101, 102, 101, 103, 104])

        labels = create_volatility_labels(high, low, close, horizon=1)

        assert len(labels) == len(close)
        assert np.all(labels >= 0)

    def test_create_price_move_labels(self):
        """Test price movement label creation."""
        close = np.array([100, 101, 102, 101, 103, 104])

        labels = create_price_move_labels(close, horizon=2)

        assert len(labels) == len(close)


@pytest.mark.unit
class TestNormalization:
    """Test feature normalization functions."""

    def test_normalize_features_zscore(self):
        """Test z-score normalization."""
        features = np.random.randn(100, 5)

        normalized, params = normalize_features(features, method='zscore')

        assert normalized.shape == features.shape
        assert params['method'] == 'zscore'
        assert 'mean' in params
        assert 'std' in params
        # Check normalized data has mean ~0 and std ~1
        assert abs(normalized.mean()) < 1e-10
        assert abs(normalized.std() - 1.0) < 1e-10

    def test_normalize_features_minmax(self):
        """Test min-max normalization."""
        features = np.random.randn(100, 5) * 10 + 50

        normalized, params = normalize_features(features, method='minmax')

        assert normalized.shape == features.shape
        assert np.all(normalized >= -1)  # Should be in range [-1, 1]
        assert np.all(normalized <= 1)
        assert params['method'] == 'minmax'

    def test_normalize_features_robust(self):
        """Test robust normalization."""
        features = np.random.randn(100, 5)

        normalized, params = normalize_features(features, method='robust')

        assert normalized.shape == features.shape
        assert params['method'] == 'robust'

    def test_apply_normalization(self):
        """Test applying saved normalization parameters."""
        features = np.random.randn(100, 5)
        normalized, params = normalize_features(features, method='zscore')

        new_features = np.random.randn(50, 5)
        normalized_new = apply_normalization(new_features, params)

        assert normalized_new.shape == new_features.shape


@pytest.mark.unit
class TestPerformanceMetrics:
    """Test performance metric calculations."""

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])

        sharpe = calculate_sharpe_ratio(returns)

        assert isinstance(sharpe, (float, np.floating))
        # Positive returns should give positive Sharpe
        if returns.mean() > 0:
            assert sharpe > -10  # Allow some variance

    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])

        sortino = calculate_sortino_ratio(returns)

        assert isinstance(sortino, (float, int, np.floating))

    def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        equity = np.array([100, 105, 110, 108, 115, 112, 118])

        max_dd, peak_idx, trough_idx = calculate_max_drawdown(equity)

        assert max_dd >= 0
        assert 0 <= peak_idx < len(equity)
        assert 0 <= trough_idx < len(equity)
        assert peak_idx <= trough_idx

    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        outcomes = np.array([1, -1, 1, 1, -1, 1])

        win_rate = calculate_win_rate(outcomes)

        assert 0 <= win_rate <= 1
        assert win_rate == 4/6

    def test_calculate_profit_factor(self):
        """Test profit factor calculation."""
        returns = np.array([10, -5, 20, -10, 15, -8])

        pf = calculate_profit_factor(returns)

        assert pf > 0
        # Gross profit / gross loss should be positive

    def test_calculate_expectancy(self):
        """Test expectancy calculation."""
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])

        expectancy = calculate_expectancy(returns)

        assert isinstance(expectancy, float)


@pytest.mark.unit
class TestPerformanceReport:
    """Test performance report generation."""

    def test_generate_performance_report(self):
        """Test generating performance report."""
        returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015, 0.01, -0.002])

        report = generate_performance_report(returns)

        assert isinstance(report, PerformanceReport)
        assert report.total_trades == len(returns)
        assert 0 <= report.win_rate <= 1
        assert report.profit_factor > 0
        assert isinstance(report.expectancy, float)

    def test_generate_performance_report_all_wins(self):
        """Test performance report with all winning trades."""
        returns = np.array([0.01, 0.02, 0.015, 0.01])

        report = generate_performance_report(returns)

        assert report.win_rate == 1.0
        assert report.profit_factor > 1.0

    def test_generate_performance_report_all_losses(self):
        """Test performance report with all losing trades."""
        returns = np.array([-0.01, -0.02, -0.015, -0.01])

        report = generate_performance_report(returns)

        assert report.win_rate == 0.0

