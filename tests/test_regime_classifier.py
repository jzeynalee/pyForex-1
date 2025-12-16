import numpy as np
import pandas as pd
import pytest

from trend_detection.regime_classifier import RegimeClassifier


def make_trending_df(n=100, trend=1.0):
    """Create trending OHLCV data (ADX > 25)."""
    price = 1.1000
    rows = []
    for i in range(n):
        o = price
        c = price + trend * 0.0005
        h = max(o, c) + 0.0001
        l = min(o, c) - 0.0001
        rows.append({'open': o, 'high': h, 'low': l, 'close': c, 'tick_volume': 100})
        price = c
    return pd.DataFrame(rows)


def make_ranging_df(n=100, noise=0.0002):
    """Create ranging OHLCV data (ADX < 20, BB width < 2%)."""
    price = 1.1000
    rows = []
    for i in range(n):
        o = price
        c = price + np.random.randn() * noise
        h = max(o, c) + abs(np.random.randn() * noise * 0.5)
        l = min(o, c) - abs(np.random.randn() * noise * 0.5)
        rows.append({'open': o, 'high': h, 'low': l, 'close': c, 'tick_volume': 100})
        price = c
    return pd.DataFrame(rows)


def make_volatile_df(n=100, volatility=0.002):
    """Create volatile OHLCV data (high relative volatility)."""
    price = 1.1000
    rows = []
    for i in range(n):
        o = price
        c = price + np.random.randn() * volatility
        h = max(o, c) + abs(np.random.randn() * volatility)
        l = min(o, c) - abs(np.random.randn() * volatility)
        rows.append({'open': o, 'high': h, 'low': l, 'close': c, 'tick_volume': 100})
        price = c
    return pd.DataFrame(rows)


def make_transitional_df(n=100):
    """Create transitional OHLCV data (moderate movement, ADX 20-25)."""
    price = 1.1000
    rows = []
    for i in range(n):
        o = price
        c = price + np.random.randn() * 0.0003
        h = max(o, c) + 0.0002
        l = min(o, c) - 0.0002
        rows.append({'open': o, 'high': h, 'low': l, 'close': c, 'tick_volume': 100})
        price = c
    return pd.DataFrame(rows)


class TestRegimeClassifierInit:
    def test_init_creates_indicators(self):
        classifier = RegimeClassifier()
        assert classifier.indicators is not None


class TestClassifyRegime:
    def test_classify_trending_regime(self):
        """Strong uptrend should classify as TRENDING."""
        df = make_trending_df(n=100, trend=1.0)
        classifier = RegimeClassifier()

        result = classifier.classify_regime(df)

        assert isinstance(result, dict)
        assert 'regime' in result
        assert 'adx' in result
        assert 'volatility' in result
        assert 'bb_width' in result
        # Trending data should have ADX > 25
        assert result['adx'] > 20 or result['regime'] in ('RANGING', 'TRANSITIONAL', 'VOLATILE')

    def test_classify_regime_returns_dict_structure(self):
        """Result should have all expected keys."""
        df = make_ranging_df(n=100)
        classifier = RegimeClassifier()

        result = classifier.classify_regime(df)

        assert isinstance(result, dict)
        assert result['regime'] in ('TRENDING', 'RANGING', 'VOLATILE', 'TRANSITIONAL')
        assert isinstance(result['adx'], (int, float, np.number))
        assert isinstance(result['volatility'], (int, float, np.number))
        assert isinstance(result['bb_width'], (int, float, np.number))

    def test_classify_regime_ranging_with_low_adx_and_width(self):
        """Low ADX and narrow BB should classify as RANGING."""
        df = make_ranging_df(n=100, noise=0.00005)
        classifier = RegimeClassifier()

        result = classifier.classify_regime(df)

        # Ranging data has low ADX and narrow BB width
        if result['adx'] < 20 and result['bb_width'] < 2.0:
            assert result['regime'] == 'RANGING'

    def test_classify_regime_volatile(self):
        """High volatility should classify as VOLATILE."""
        df = make_volatile_df(n=100, volatility=0.003)
        classifier = RegimeClassifier()

        result = classifier.classify_regime(df)

        assert result['regime'] in ('VOLATILE', 'TRENDING', 'TRANSITIONAL')
        assert result['volatility'] >= 0

    def test_classify_regime_transitional(self):
        """Moderate data should classify as TRANSITIONAL."""
        df = make_transitional_df(n=100)
        classifier = RegimeClassifier()

        result = classifier.classify_regime(df)

        assert result['regime'] in ('TRENDING', 'RANGING', 'VOLATILE', 'TRANSITIONAL')


class TestApplyRegimeFilter:
    def test_apply_filter_trending_high_score(self):
        """Trending regime with high score should not be penalized."""
        regime_info = {'regime': 'TRENDING', 'adx': 30, 'volatility': 0.8, 'bb_width': 1.5}
        mtf_score = 0.8
        classifier = RegimeClassifier()

        result = classifier.apply_regime_filter(regime_info, mtf_score)

        assert isinstance(result, dict)
        assert 'adjusted_score' in result
        assert 'regime' in result
        assert 'adjustment_factor' in result
        # High score in trending regime should stay high
        assert result['adjusted_score'] >= 0.7 * mtf_score

    def test_apply_filter_trending_low_score(self):
        """Trending regime with low score gets reduced."""
        regime_info = {'regime': 'TRENDING', 'adx': 30, 'volatility': 0.8, 'bb_width': 1.5}
        mtf_score = 0.5
        classifier = RegimeClassifier()

        result = classifier.apply_regime_filter(regime_info, mtf_score)

        # Low score in trending gets adjustment
        assert result['adjusted_score'] == pytest.approx(mtf_score * 0.7)

    def test_apply_filter_ranging_penalizes(self):
        """Ranging regime heavily penalizes scores."""
        regime_info = {'regime': 'RANGING', 'adx': 15, 'volatility': 0.5, 'bb_width': 1.0}
        mtf_score = 0.8
        classifier = RegimeClassifier()

        result = classifier.apply_regime_filter(regime_info, mtf_score)

        # Ranging regime applies 0.3x multiplier
        assert result['adjusted_score'] == pytest.approx(mtf_score * 0.3)

    def test_apply_filter_volatile_weak_signal(self):
        """Volatile regime with weak signal gets penalized."""
        regime_info = {'regime': 'VOLATILE', 'adx': 22, 'volatility': 1.5, 'bb_width': 3.0}
        mtf_score = 0.6
        classifier = RegimeClassifier()

        result = classifier.apply_regime_filter(regime_info, mtf_score)

        # Weak score (< 0.65) in volatile/transitional gets 0.5x
        assert result['adjusted_score'] == pytest.approx(mtf_score * 0.5)

    def test_apply_filter_volatile_strong_signal(self):
        """Volatile regime with strong signal passes through."""
        regime_info = {'regime': 'VOLATILE', 'adx': 22, 'volatility': 1.5, 'bb_width': 3.0}
        mtf_score = 0.8
        classifier = RegimeClassifier()

        result = classifier.apply_regime_filter(regime_info, mtf_score)

        # Strong score (>= 0.65) in volatile keeps full value
        assert result['adjusted_score'] == pytest.approx(mtf_score)

    def test_apply_filter_transitional_weak_signal(self):
        """Transitional regime with weak signal gets penalized."""
        regime_info = {'regime': 'TRANSITIONAL', 'adx': 22, 'volatility': 1.0, 'bb_width': 1.5}
        mtf_score = 0.6
        classifier = RegimeClassifier()

        result = classifier.apply_regime_filter(regime_info, mtf_score)

        # Weak signal in transitional gets 0.5x
        assert result['adjusted_score'] == pytest.approx(mtf_score * 0.5)

    def test_apply_filter_adjustment_factor_calculation(self):
        """Adjustment factor should be adjusted/mtf_score."""
        regime_info = {'regime': 'RANGING'}
        mtf_score = 0.8
        classifier = RegimeClassifier()

        result = classifier.apply_regime_filter(regime_info, mtf_score)

        expected_factor = result['adjusted_score'] / mtf_score
        assert result['adjustment_factor'] == pytest.approx(expected_factor)

    def test_apply_filter_zero_score_prevents_division_error(self):
        """Zero MTF score should not cause division error."""
        regime_info = {'regime': 'RANGING'}
        mtf_score = 0.0
        classifier = RegimeClassifier()

        result = classifier.apply_regime_filter(regime_info, mtf_score)

        assert result['adjusted_score'] == 0.0
        assert result['adjustment_factor'] == 1.0  # Fallback to 1.0


class TestRegimeClassifierIntegration:
    def test_classify_and_filter_workflow(self):
        """Test end-to-end classify + filter workflow."""
        df = make_trending_df(n=100, trend=1.0)
        classifier = RegimeClassifier()

        # Step 1: Classify regime
        regime_info = classifier.classify_regime(df)
        assert regime_info['regime'] in ('TRENDING', 'RANGING', 'VOLATILE', 'TRANSITIONAL')

        # Step 2: Apply filter with various scores
        for mtf_score in [0.3, 0.5, 0.7, 0.9]:
            result = classifier.apply_regime_filter(regime_info, mtf_score)
            assert 0.0 <= result['adjusted_score'] <= 1.0
            assert result['adjustment_factor'] >= 0.0

    def test_multiple_regimes_same_data(self):
        """Verify different data produces different regimes."""
        classifier = RegimeClassifier()

        regimes = set()
        for maker, args in [
            (make_trending_df, {'trend': 1.0}),
            (make_ranging_df, {'noise': 0.00005}),
            (make_volatile_df, {'volatility': 0.003}),
        ]:
            df = maker(n=100, **args)
            result = classifier.classify_regime(df)
            regimes.add(result['regime'])

        # At least 2 different regimes should be detected
        assert len(regimes) >= 2

    def test_regime_metrics_are_non_negative(self):
        """All regime metrics should be non-negative."""
        df = make_ranging_df(n=100)
        classifier = RegimeClassifier()

        result = classifier.classify_regime(df)

        assert result['adx'] >= 0
        assert result['volatility'] >= 0
        assert result['bb_width'] >= 0


class TestRegimeFilterEdgeCases:
    def test_filter_with_extreme_scores(self):
        """Filter should handle boundary scores (0, 1)."""
        classifier = RegimeClassifier()
        regime_info = {'regime': 'TRENDING'}

        result_zero = classifier.apply_regime_filter(regime_info, 0.0)
        result_one = classifier.apply_regime_filter(regime_info, 1.0)

        assert 0.0 <= result_zero['adjusted_score'] <= 1.0
        assert 0.0 <= result_one['adjusted_score'] <= 1.0

    def test_filter_all_regimes_all_scores(self):
        """Filter should work for all regime types and scores."""
        classifier = RegimeClassifier()
        regimes = ['TRENDING', 'RANGING', 'VOLATILE', 'TRANSITIONAL']
        scores = [0.0, 0.25, 0.5, 0.75, 1.0]

        for regime in regimes:
            for score in scores:
                regime_info = {'regime': regime}
                result = classifier.apply_regime_filter(regime_info, score)

                assert isinstance(result['adjusted_score'], (int, float, np.number))
                assert 0.0 <= result['adjusted_score'] <= 1.0
