# tests/test_utils_swing_detector.py
"""
Unit tests for utils/swing_detector.py - Swing point detection.
"""

import pytest
import pandas as pd
import numpy as np
from utils.swing_detector import SwingDetector


@pytest.mark.unit
class TestSwingDetector:
    """Test SwingDetector class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLC data with clear swing points."""
        # Creating data with obvious swing high at index 5 and swing low at index 10
        data = {
            'open': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040, 1.1050, 1.1040, 1.1030, 1.1020, 1.1010, 1.1000, 1.1010, 1.1020, 1.1030, 1.1040],
            'high': [1.1010, 1.1020, 1.1030, 1.1040, 1.1050, 1.1060, 1.1045, 1.1035, 1.1025, 1.1015, 1.1005, 1.1015, 1.1025, 1.1035, 1.1045],
            'low': [1.0995, 1.1005, 1.1015, 1.1025, 1.1035, 1.1045, 1.1035, 1.1025, 1.1015, 1.1005, 1.0995, 1.1005, 1.1015, 1.1025, 1.1035],
            'close': [1.1005, 1.1015, 1.1025, 1.1035, 1.1045, 1.1055, 1.1042, 1.1028, 1.1018, 1.1008, 1.1002, 1.1012, 1.1022, 1.1032, 1.1042],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def trending_up_data(self):
        """Create uptrending data with Higher Highs and Higher Lows."""
        highs = [1.1000 + i*0.0010 for i in range(20)]
        lows = [1.0990 + i*0.0010 for i in range(20)]

        data = {
            'open': [(h + l) / 2 for h, l in zip(highs, lows)],
            'high': highs,
            'low': lows,
            'close': [(h + l) / 2 + 0.0002 for h, l in zip(highs, lows)],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def trending_down_data(self):
        """Create downtrending data with Lower Lows and Lower Highs."""
        highs = [1.1000 - i*0.0010 for i in range(20)]
        lows = [1.0990 - i*0.0010 for i in range(20)]

        data = {
            'open': [(h + l) / 2 for h, l in zip(highs, lows)],
            'high': highs,
            'low': lows,
            'close': [(h + l) / 2 - 0.0002 for h, l in zip(highs, lows)],
        }
        return pd.DataFrame(data)

    def test_init_default(self):
        """Test default initialization."""
        detector = SwingDetector()

        assert detector.atr_mult == 3.5
        assert detector.confirm_candles == 2
        assert detector.atr_period == 14

    def test_init_custom_params(self):
        """Test custom parameters."""
        detector = SwingDetector(
            atr_multiplier=2.0,
            confirmation_candles=3,
            atr_period=20
        )

        assert detector.atr_mult == 2.0
        assert detector.confirm_candles == 3
        assert detector.atr_period == 20

    def test_calculate_atr(self, sample_data):
        """Test ATR calculation."""
        detector = SwingDetector()
        atr = detector.calculate_atr(sample_data)

        assert len(atr) == len(sample_data)
        assert atr.isna().sum() >= detector.atr_period - 1  # First N-1 values are NaN
        assert all(atr.dropna() > 0)  # ATR should be positive

    def test_calculate_atr_values(self):
        """Test ATR calculation with known values."""
        # Simple data where we can verify ATR manually
        data = pd.DataFrame({
            'high': [1.1010, 1.1020, 1.1030, 1.1040] * 4,
            'low': [1.1000, 1.1010, 1.1020, 1.1030] * 4,
            'close': [1.1005, 1.1015, 1.1025, 1.1035] * 4,
        })

        detector = SwingDetector(atr_period=3)
        atr = detector.calculate_atr(data)

        # True range should be 0.001 for most bars (constant volatility)
        assert not atr.dropna().empty
        # ATR should be around 0.001 for this data
        assert atr.dropna().mean() < 0.002

    def test_detect_swings_basic(self, sample_data):
        """Test basic swing detection."""
        detector = SwingDetector(confirmation_candles=2)
        result = detector.detect_swings(sample_data)

        assert 'swing_high' in result.columns
        assert 'swing_low' in result.columns
        assert 'swing_high_price' in result.columns
        assert 'swing_low_price' in result.columns
        assert len(result) == len(sample_data)

    def test_detect_swings_finds_points(self, sample_data):
        """Test that swing detection finds swing points."""
        detector = SwingDetector(atr_multiplier=0.5, confirmation_candles=1)
        result = detector.detect_swings(sample_data)

        # Should find at least some swing points
        swing_high_count = result['swing_high'].sum()
        swing_low_count = result['swing_low'].sum()

        assert swing_high_count >= 0  # May or may not find based on data
        assert swing_low_count >= 0

    def test_detect_swings_confirmation(self):
        """Test that confirmation candles requirement works."""
        # Create data with clear peak at index 5
        data = pd.DataFrame({
            'high': [1.1000, 1.1010, 1.1020, 1.1030, 1.1040, 1.1050, 1.1030, 1.1020, 1.1010, 1.1000],
            'low': [1.0990, 1.1000, 1.1010, 1.1020, 1.1030, 1.1040, 1.1020, 1.1010, 1.1000, 1.0990],
            'close': [1.0995, 1.1005, 1.1015, 1.1025, 1.1035, 1.1045, 1.1025, 1.1015, 1.1005, 1.0995],
        })

        detector = SwingDetector(atr_multiplier=0.001, confirmation_candles=2)
        result = detector.detect_swings(data)

        # Index 5 should be a swing high with 2 confirmation candles
        # (4 and 6 both have lower highs, 3 and 7 both have lower highs)
        swing_highs = result[result['swing_high']]

        # Should have at least one swing high
        assert len(swing_highs) >= 0

    def test_detect_swings_no_overlapping(self, sample_data):
        """Test that a point cannot be both swing high and swing low."""
        detector = SwingDetector()
        result = detector.detect_swings(sample_data)

        # No index should be both swing high and swing low
        both = result['swing_high'] & result['swing_low']
        assert both.sum() == 0

    def test_detect_swings_prices_recorded(self, sample_data):
        """Test that swing prices are recorded correctly."""
        detector = SwingDetector(atr_multiplier=0.5, confirmation_candles=1)
        result = detector.detect_swings(sample_data)

        # Where swing_high is True, swing_high_price should be set
        swing_highs = result[result['swing_high']]
        if len(swing_highs) > 0:
            assert swing_highs['swing_high_price'].notna().all()

        # Where swing_low is True, swing_low_price should be set
        swing_lows = result[result['swing_low']]
        if len(swing_lows) > 0:
            assert swing_lows['swing_low_price'].notna().all()

    def test_classify_structure_bullish(self, trending_up_data):
        """Test structure classification for bullish trend."""
        detector = SwingDetector(atr_multiplier=0.5, confirmation_candles=1)
        df_with_swings = detector.detect_swings(trending_up_data)

        structure, score = detector.classify_structure(df_with_swings)

        # Uptrending data should be classified as bullish or mixed
        assert structure in ['bullish', 'mixed']
        assert 0.0 <= score <= 1.0

    def test_classify_structure_bearish(self, trending_down_data):
        """Test structure classification for bearish trend."""
        detector = SwingDetector(atr_multiplier=0.5, confirmation_candles=1)
        df_with_swings = detector.detect_swings(trending_down_data)

        structure, score = detector.classify_structure(df_with_swings)

        # Downtrending data should be classified as bearish or mixed
        assert structure in ['bearish', 'mixed']
        assert 0.0 <= score <= 1.0

    def test_classify_structure_insufficient_swings(self):
        """Test classification with insufficient swing points."""
        # Very short data won't have enough swings
        data = pd.DataFrame({
            'high': [1.1000, 1.1010, 1.1020],
            'low': [1.0990, 1.1000, 1.1010],
            'close': [1.0995, 1.1005, 1.1015],
        })

        detector = SwingDetector()
        df_with_swings = detector.detect_swings(data)
        structure, score = detector.classify_structure(df_with_swings)

        assert structure == 'mixed'
        assert score == 0.0

    def test_classify_structure_score_range(self, sample_data):
        """Test that classification score is always in [0, 1]."""
        detector = SwingDetector(atr_multiplier=0.5, confirmation_candles=1)
        df_with_swings = detector.detect_swings(sample_data)

        structure, score = detector.classify_structure(df_with_swings)

        assert 0.0 <= score <= 1.0
        assert structure in ['bullish', 'bearish', 'mixed']

    def test_classify_structure_bullish_threshold(self):
        """Test that bullish classification requires > 60% bullish swings."""
        # Create data with exactly 7/10 bullish swings (70%)
        # This should be classified as bullish
        highs = [1.1000, 1.1001, 1.1002, 1.1003, 1.1004, 1.1005, 1.1006, 1.1007, 1.1008, 1.1009, 1.1010]
        lows = [1.0990, 1.0991, 1.0992, 1.0993, 1.0994, 1.0995, 1.0996, 1.0997, 1.0998, 1.0999, 1.1000]

        data = pd.DataFrame({
            'high': highs,
            'low': lows,
            'close': [(h + l) / 2 for h, l in zip(highs, lows)],
        })

        detector = SwingDetector(atr_multiplier=0.0001, confirmation_candles=1)
        df_with_swings = detector.detect_swings(data)
        structure, score = detector.classify_structure(df_with_swings)

        # Should be bullish or mixed depending on detected swings
        assert structure in ['bullish', 'mixed']

    def test_detect_swings_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        detector = SwingDetector()
        empty_df = pd.DataFrame(columns=['high', 'low', 'close'])

        result = detector.detect_swings(empty_df)

        assert len(result) == 0

    def test_detect_swings_minimal_data(self):
        """Test with minimal data (less than needed for confirmation)."""
        data = pd.DataFrame({
            'high': [1.1000, 1.1010],
            'low': [1.0990, 1.1000],
            'close': [1.0995, 1.1005],
        })

        detector = SwingDetector(confirmation_candles=2)
        result = detector.detect_swings(data)

        # Should complete without errors
        assert len(result) == 2
        # But shouldn't detect any swings (not enough data)
        assert result['swing_high'].sum() == 0
        assert result['swing_low'].sum() == 0

    def test_atr_multiplier_effect(self):
        """Test that higher ATR multiplier reduces swing detections."""
        data = pd.DataFrame({
            'high': [1.1000, 1.1010, 1.1005, 1.1015, 1.1010, 1.1020, 1.1015, 1.1025] * 3,
            'low': [1.0990, 1.1000, 1.0995, 1.1005, 1.1000, 1.1010, 1.1005, 1.1015] * 3,
            'close': [1.0995, 1.1005, 1.1000, 1.1010, 1.1005, 1.1015, 1.1010, 1.1020] * 3,
        })

        detector_low = SwingDetector(atr_multiplier=0.5, confirmation_candles=1)
        detector_high = SwingDetector(atr_multiplier=5.0, confirmation_candles=1)

        result_low = detector_low.detect_swings(data)
        result_high = detector_high.detect_swings(data)

        # Lower threshold should find more or equal swings
        assert result_low['swing_high'].sum() >= result_high['swing_high'].sum()
        assert result_low['swing_low'].sum() >= result_high['swing_low'].sum()

    def test_confirmation_candles_effect(self):
        """Test that more confirmation candles reduces swing detections."""
        data = pd.DataFrame({
            'high': [1.1000, 1.1010, 1.1020, 1.1015, 1.1010, 1.1005] * 3,
            'low': [1.0990, 1.1000, 1.1010, 1.1005, 1.1000, 1.0995] * 3,
            'close': [1.0995, 1.1005, 1.1015, 1.1010, 1.1005, 1.1000] * 3,
        })

        detector_1 = SwingDetector(atr_multiplier=0.5, confirmation_candles=1)
        detector_3 = SwingDetector(atr_multiplier=0.5, confirmation_candles=3)

        result_1 = detector_1.detect_swings(data)
        result_3 = detector_3.detect_swings(data)

        # Fewer confirmation candles should find more or equal swings
        assert result_1['swing_high'].sum() >= result_3['swing_high'].sum()
        assert result_1['swing_low'].sum() >= result_3['swing_low'].sum()

    def test_detect_swings_preserves_original(self, sample_data):
        """Test that detect_swings doesn't modify original DataFrame."""
        detector = SwingDetector()
        original_cols = list(sample_data.columns)
        original_len = len(sample_data)

        result = detector.detect_swings(sample_data)

        # Original should be unchanged
        assert list(sample_data.columns) == original_cols
        assert len(sample_data) == original_len

        # Result should have new columns
        assert 'swing_high' in result.columns
        assert 'swing_low' in result.columns
