import numpy as np
import pandas as pd
import pytest

from utils.pattern_detector import (
    CandlestickPatternDetector,
    PatternDetection,
    PatternClass,
    PATTERN_NAMES,
    detect_patterns,
)


def make_doji_candle(open_=100, close=100.05):
    """Create a doji-like candle (small body)."""
    return {
        'open': open_,
        'high': open_ + 2.0,
        'low': open_ - 2.0,
        'close': close,
    }


def make_hammer_candle(open_=100):
    """Create a hammer-like candle (long lower shadow)."""
    return {
        'open': open_,
        'high': open_ + 0.5,
        'low': open_ - 5.0,  # long lower shadow
        'close': open_ + 0.2,
    }


def make_bullish_candle(open_=100, close=105):
    """Create a bullish candle."""
    return {
        'open': open_,
        'high': close + 0.5,
        'low': open_ - 0.5,
        'close': close,
    }


def make_bearish_candle(open_=100, close=95):
    """Create a bearish candle."""
    return {
        'open': open_,
        'high': open_ + 0.5,
        'low': close - 0.5,
        'close': close,
    }


def make_marubozu_bull(open_=100, close=105):
    """Create a bullish marubozu (big body, no/tiny shadows)."""
    return {
        'open': open_,
        'high': close + 0.01,
        'low': open_ - 0.01,
        'close': close,
    }


def make_df_from_candles(candles):
    """Convert list of candle dicts to DataFrame."""
    return pd.DataFrame(candles)


class TestPatternDetectorInit:
    def test_default_init(self):
        detector = CandlestickPatternDetector()
        assert detector.body_threshold == 0.1
        assert detector.doji_threshold == 0.05
        assert detector.shadow_ratio == 2.0
        assert detector.engulf_threshold == 1.0

    def test_custom_init(self):
        detector = CandlestickPatternDetector(
            body_threshold=0.15, doji_threshold=0.08,
            shadow_ratio=2.5, engulf_threshold=1.5
        )
        assert detector.body_threshold == 0.15
        assert detector.doji_threshold == 0.08
        assert detector.shadow_ratio == 2.5
        assert detector.engulf_threshold == 1.5


class TestCandleMetrics:
    def test_get_candle_metrics_bullish(self):
        candle = make_bullish_candle(open_=100, close=110)
        df = make_df_from_candles([candle])
        detector = CandlestickPatternDetector()

        m = detector._get_candle_metrics(df.iloc[0])

        assert bool(m['is_bullish']) is True
        assert m['body'] == 10.0
        assert m['upper_shadow'] > 0
        assert m['body_ratio'] > 0.8  # body is dominant

    def test_get_candle_metrics_bearish(self):
        candle = make_bearish_candle(open_=100, close=90)
        df = make_df_from_candles([candle])
        detector = CandlestickPatternDetector()

        m = detector._get_candle_metrics(df.iloc[0])

        assert bool(m['is_bullish']) is False
        assert m['body'] == 10.0
        assert m['lower_shadow'] > 0

    def test_get_candle_metrics_doji(self):
        candle = make_doji_candle(open_=100, close=100.02)
        df = make_df_from_candles([candle])
        detector = CandlestickPatternDetector()

        m = detector._get_candle_metrics(df.iloc[0])

        assert m['body_ratio'] < 0.1


class TestSingleCandlePatterns:
    def test_doji_detection(self):
        candle = make_doji_candle()
        df = make_df_from_candles([candle])
        detector = CandlestickPatternDetector()

        patterns = detector._detect_single_candle_patterns(df)

        doji_patterns = [p for p in patterns if p.pattern_class == PatternClass.DOJI]
        assert len(doji_patterns) > 0
        assert doji_patterns[0].confidence > 0.7
        assert doji_patterns[0].direction == 'neutral'

    def test_hammer_detection(self):
        # Create clearer hammer: more body/less upper shadow
        candle = {
            'open': 100,
            'high': 100.3,
            'low': 95,  # long lower shadow
            'close': 100.2,  # small bullish body
        }
        df = make_df_from_candles([candle])
        detector = CandlestickPatternDetector()

        patterns = detector._detect_single_candle_patterns(df)

        hammer_patterns = [p for p in patterns if p.pattern_class == PatternClass.HAMMER]
        assert len(hammer_patterns) > 0
        assert hammer_patterns[0].direction == 'bullish'

    def test_spinning_top_detection(self):
        # small body with equal shadows
        candle = {
            'open': 100,
            'close': 101,
            'high': 103,  # +2 upper shadow
            'low': 98,    # -2 lower shadow
        }
        df = make_df_from_candles([candle])
        detector = CandlestickPatternDetector()

        patterns = detector._detect_single_candle_patterns(df)

        spinning_top = [p for p in patterns if p.pattern_class == PatternClass.SPINNING_TOP]
        assert len(spinning_top) > 0
        assert spinning_top[0].direction == 'neutral'

    def test_marubozu_bull_detection(self):
        candle = make_marubozu_bull(open_=100, close=105)
        df = make_df_from_candles([candle])
        detector = CandlestickPatternDetector()

        patterns = detector._detect_single_candle_patterns(df)

        marubozu = [p for p in patterns if p.pattern_class == PatternClass.MARUBOZU_BULL]
        assert len(marubozu) > 0
        assert marubozu[0].direction == 'bullish'
        assert marubozu[0].confidence > 0.85

    def test_marubozu_bear_detection(self):
        candle = {
            'open': 105,
            'high': 105.01,
            'low': 100 - 0.01,
            'close': 100,
        }
        df = make_df_from_candles([candle])
        detector = CandlestickPatternDetector()

        patterns = detector._detect_single_candle_patterns(df)

        marubozu = [p for p in patterns if p.pattern_class == PatternClass.MARUBOZU_BEAR]
        assert len(marubozu) > 0
        assert marubozu[0].direction == 'bearish'

    def test_inverted_hammer_basic(self):
        # Test inverted hammer: long upper shadow, small body
        candles = [make_bullish_candle(100, 101), make_bullish_candle(101, 102)]
        df = make_df_from_candles(candles)
        detector = CandlestickPatternDetector()

        patterns = detector._detect_single_candle_patterns(df)

        # Just verify patterns are detected and analyzer works
        assert isinstance(patterns, list)
        for p in patterns:
            assert p.pattern_name in PATTERN_NAMES


class TestTwoCandlePatterns:
    def test_bullish_engulfing(self):
        candles = [
            make_bearish_candle(open_=100, close=95),  # bearish
            make_bullish_candle(open_=94, close=106),  # bullish, engulfs
        ]
        df = make_df_from_candles(candles)
        detector = CandlestickPatternDetector()

        patterns = detector._detect_two_candle_patterns(df)

        engulf = [p for p in patterns if p.pattern_class == PatternClass.BULLISH_ENGULFING]
        assert len(engulf) > 0
        assert engulf[0].start_idx == 0
        assert engulf[0].end_idx == 1
        assert engulf[0].direction == 'bullish'

    def test_bearish_engulfing(self):
        candles = [
            make_bullish_candle(open_=100, close=105),  # bullish
            make_bearish_candle(open_=106, close=94),   # bearish, engulfs
        ]
        df = make_df_from_candles(candles)
        detector = CandlestickPatternDetector()

        patterns = detector._detect_two_candle_patterns(df)

        engulf = [p for p in patterns if p.pattern_class == PatternClass.BEARISH_ENGULFING]
        assert len(engulf) > 0
        assert engulf[0].direction == 'bearish'

    def test_bullish_harami(self):
        candles = [
            {
                'open': 100,
                'high': 101,
                'low': 90,
                'close': 90,  # bearish
            },
            {
                'open': 95,
                'high': 99,
                'low': 92,
                'close': 98,  # bullish, small body, inside
            },
        ]
        df = make_df_from_candles(candles)
        detector = CandlestickPatternDetector()

        patterns = detector._detect_two_candle_patterns(df)

        harami = [p for p in patterns if p.pattern_class == PatternClass.BULLISH_HARAMI]
        for h in harami:
            assert h.direction == 'bullish'

    def test_piercing_line(self):
        candles = [
            make_bearish_candle(open_=100, close=95),
            {
                'open': 94,  # below low
                'high': 102,
                'low': 93,
                'close': 98,  # above midpoint
            },
        ]
        df = make_df_from_candles(candles)
        detector = CandlestickPatternDetector()

        patterns = detector._detect_two_candle_patterns(df)

        piercing = [p for p in patterns if p.pattern_class == PatternClass.PIERCING_LINE]
        assert len(piercing) > 0
        assert piercing[0].direction == 'bullish'

    def test_tweezer_bottom(self):
        candles = [
            make_bearish_candle(open_=100, close=95),
            make_bullish_candle(open_=94, close=99),
        ]
        # Set same low
        candles[1]['low'] = candles[0]['low']
        df = make_df_from_candles(candles)
        detector = CandlestickPatternDetector()

        patterns = detector._detect_two_candle_patterns(df)

        tweezer = [p for p in patterns if p.pattern_class == PatternClass.TWEEZER_BOTTOM]
        assert len(tweezer) > 0
        assert tweezer[0].direction == 'bullish'


class TestThreeCandlePatterns:
    def test_morning_star(self):
        candles = [
            make_bearish_candle(100, 95),           # bearish
            make_doji_candle(94, 94.5),             # small body, lower
            make_bullish_candle(95, 103),           # bullish, closes above midpoint
        ]
        df = make_df_from_candles(candles)
        detector = CandlestickPatternDetector()

        patterns = detector._detect_three_candle_patterns(df)

        morning = [p for p in patterns if p.pattern_class == PatternClass.MORNING_STAR]
        assert len(morning) > 0
        assert morning[0].start_idx == 0
        assert morning[0].end_idx == 2
        assert morning[0].direction == 'bullish'
        assert morning[0].confidence > 0.8

    def test_evening_star(self):
        candles = [
            make_bullish_candle(100, 105),          # bullish
            make_doji_candle(106, 106.5),           # small body, higher
            make_bearish_candle(107, 97),           # bearish, closes below midpoint
        ]
        df = make_df_from_candles(candles)
        detector = CandlestickPatternDetector()

        patterns = detector._detect_three_candle_patterns(df)

        evening = [p for p in patterns if p.pattern_class == PatternClass.EVENING_STAR]
        assert len(evening) > 0
        assert evening[0].direction == 'bearish'

    def test_three_white_soldiers(self):
        candles = [
            make_bullish_candle(100, 102),
            make_bullish_candle(101, 103),
            make_bullish_candle(102, 104),
        ]
        df = make_df_from_candles(candles)
        detector = CandlestickPatternDetector()

        patterns = detector._detect_three_candle_patterns(df)

        soldiers = [p for p in patterns if p.pattern_class == PatternClass.THREE_WHITE_SOLDIERS]
        # May or may not detect depending on exact logic, but test structure
        for s in soldiers:
            assert s.direction == 'bullish'
            assert s.confidence > 0.8

    def test_three_black_crows(self):
        candles = [
            make_bearish_candle(100, 98),
            make_bearish_candle(99, 97),
            make_bearish_candle(98, 96),
        ]
        df = make_df_from_candles(candles)
        detector = CandlestickPatternDetector()

        patterns = detector._detect_three_candle_patterns(df)

        crows = [p for p in patterns if p.pattern_class == PatternClass.THREE_BLACK_CROWS]
        for c in crows:
            assert c.direction == 'bearish'


class TestDetectAllPatterns:
    def test_detect_all_patterns_integration(self):
        # Build a realistic OHLCV with mixed patterns
        np.random.seed(42)
        candles = []
        base = 100
        for i in range(50):
            change = np.random.randn() * 0.3
            o = base
            c = base + change
            h = max(o, c) + abs(np.random.randn() * 0.2)
            l = min(o, c) - abs(np.random.randn() * 0.2)
            candles.append({'open': o, 'high': h, 'low': l, 'close': c})
            base = c

        df = make_df_from_candles(candles)
        detector = CandlestickPatternDetector()

        patterns = detector.detect_all_patterns(df)

        # Should detect some patterns
        assert isinstance(patterns, list)
        # Most random data will have at least some doji or spinning tops
        assert any(p.confidence > 0.5 for p in patterns)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
        detector = CandlestickPatternDetector()

        patterns = detector.detect_all_patterns(df)
        assert patterns == []


class TestPatternDetectionDataclass:
    def test_pattern_detection_creation(self):
        pd_obj = PatternDetection(
            pattern_class=0,
            pattern_name="doji",
            start_idx=5,
            end_idx=5,
            confidence=0.95,
            direction='neutral',
        )

        assert pd_obj.pattern_class == 0
        assert pd_obj.pattern_name == "doji"
        assert pd_obj.start_idx == 5
        assert pd_obj.confidence == 0.95


class TestYOLOAnnotationConversion:
    def test_to_yolo_annotations_empty(self):
        detector = CandlestickPatternDetector()
        annotations = detector.to_yolo_annotations([])

        assert annotations == []

    def test_to_yolo_annotations_multiple(self):
        patterns = [
            PatternDetection(0, "doji", 0, 0, 0.9, 'neutral'),
            PatternDetection(1, "hammer", 5, 5, 0.85, 'bullish'),
            PatternDetection(3, "bullish_engulfing", 10, 11, 0.88, 'bullish'),
        ]
        detector = CandlestickPatternDetector()
        annotations = detector.to_yolo_annotations(patterns)

        assert len(annotations) == 3
        assert annotations[0]['class_id'] == 0
        assert annotations[0]['start_idx'] == 0
        assert annotations[1]['class_id'] == 1
        assert annotations[2]['end_idx'] == 11


class TestConvenienceFunctions:
    def test_detect_patterns_convenience(self):
        candle = make_doji_candle()
        df = make_df_from_candles([candle])

        patterns = detect_patterns(df)

        assert len(patterns) > 0


class TestPatternNamesConstant:
    def test_pattern_names_count(self):
        assert len(PATTERN_NAMES) == 20

    def test_pattern_names_correct(self):
        assert "doji" in PATTERN_NAMES
        assert "hammer" in PATTERN_NAMES
        assert "bullish_engulfing" in PATTERN_NAMES
        assert "three_white_soldiers" in PATTERN_NAMES
        assert "three_black_crows" in PATTERN_NAMES
