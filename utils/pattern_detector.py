# pattern_detector.py
"""
Rule-based candlestick pattern detector for dataset labeling.
Detects 20 standard candlestick patterns and returns bounding box info.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class PatternClass(IntEnum):
    """Standard candlestick pattern classes matching YOLO detector."""
    DOJI = 0
    HAMMER = 1
    INVERTED_HAMMER = 2
    BULLISH_ENGULFING = 3
    BEARISH_ENGULFING = 4
    MORNING_STAR = 5
    EVENING_STAR = 6
    THREE_WHITE_SOLDIERS = 7
    THREE_BLACK_CROWS = 8
    BULLISH_HARAMI = 9
    BEARISH_HARAMI = 10
    SHOOTING_STAR = 11
    HANGING_MAN = 12
    PIERCING_LINE = 13
    DARK_CLOUD_COVER = 14
    TWEEZER_TOP = 15
    TWEEZER_BOTTOM = 16
    SPINNING_TOP = 17
    MARUBOZU_BULL = 18
    MARUBOZU_BEAR = 19
    INSIDE_BAR = 20
    OUTSIDE_BAR = 21
    PIN_BAR = 22
    TWO_BAR_REVERSAL = 23
    THREE_BAR_PLAY = 24


PATTERN_NAMES = [
    "doji", "hammer", "inverted_hammer", "bullish_engulfing", "bearish_engulfing",
    "morning_star", "evening_star", "three_white_soldiers", "three_black_crows",
    "bullish_harami", "bearish_harami", "shooting_star", "hanging_man",
    "piercing_line", "dark_cloud_cover", "tweezer_top", "tweezer_bottom",
    "spinning_top", "marubozu_bull", "marubozu_bear",
    "inside_bar", "outside_bar", "pin_bar", "two_bar_reversal", "three_bar_play"
]


@dataclass
class PatternDetection:
    """Detected pattern with location info."""
    pattern_class: int
    pattern_name: str
    start_idx: int
    end_idx: int
    confidence: float
    direction: str  # 'bullish', 'bearish', 'neutral'


class CandlestickPatternDetector:
    """
    Detects candlestick patterns in OHLCV data.
    Returns pattern locations for YOLO annotation generation.
    """
    
    def __init__(
        self,
        body_threshold: float = 0.1,
        doji_threshold: float = 0.05,
        shadow_ratio: float = 2.0,
        engulf_threshold: float = 1.0,
    ):
        self.body_threshold = body_threshold
        self.doji_threshold = doji_threshold
        self.shadow_ratio = shadow_ratio
        self.engulf_threshold = engulf_threshold
    
    def detect_all_patterns(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect all patterns in the DataFrame."""
        patterns = []
        patterns.extend(self._detect_single_candle_patterns(df))
        patterns.extend(self._detect_two_candle_patterns(df))
        patterns.extend(self._detect_three_candle_patterns(df))
        return patterns

    def _get_ohlc_arrays(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        o = df['open'].to_numpy(dtype=np.float64, copy=False)
        h = df['high'].to_numpy(dtype=np.float64, copy=False)
        l = df['low'].to_numpy(dtype=np.float64, copy=False)
        c = df['close'].to_numpy(dtype=np.float64, copy=False)
        return o, h, l, c
    
    def _get_candle_metrics(self, row) -> Dict:
        """Calculate candle metrics."""
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        
        body = abs(c - o)
        total_range = h - l
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        is_bullish = c > o
        
        return {
            'body': body,
            'range': total_range if total_range > 0 else 1e-10,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'is_bullish': is_bullish,
            'body_ratio': body / total_range if total_range > 0 else 0,
        }
    
    def _detect_single_candle_patterns(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect single candle patterns."""
        patterns = []

        if df is None or len(df) == 0:
            return patterns

        o, h, l, c = self._get_ohlc_arrays(df)
        n = len(o)

        body = np.abs(c - o)
        rng = np.maximum(h - l, 1e-10)
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l
        is_bullish = c > o
        body_ratio = body / rng

        for i in range(n):
            b = float(body[i])
            r = float(rng[i])
            us = float(upper_shadow[i])
            ls = float(lower_shadow[i])
            br = float(body_ratio[i])
            bull = bool(is_bullish[i])
            
            # Doji
            if br < self.doji_threshold:
                patterns.append(PatternDetection(
                    pattern_class=PatternClass.DOJI,
                    pattern_name="doji",
                    start_idx=i, end_idx=i,
                    confidence=1.0 - br / self.doji_threshold,
                    direction='neutral',
                ))
            
            # Spinning Top
            elif self.doji_threshold <= br < 0.3:
                if us > b * 0.5 and ls > b * 0.5:
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.SPINNING_TOP,
                        pattern_name="spinning_top",
                        start_idx=i, end_idx=i,
                        confidence=0.7, direction='neutral',
                    ))
            
            # Hammer
            if ls > b * self.shadow_ratio and us < b * 0.5:
                if br < 0.4:
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.HAMMER,
                        pattern_name="hammer",
                        start_idx=i, end_idx=i,
                        confidence=min(1.0, ls / (b * self.shadow_ratio + 1e-10)),
                        direction='bullish',
                    ))
            
            # Inverted Hammer
            if us > b * self.shadow_ratio and ls < b * 0.5:
                if br < 0.4:
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.INVERTED_HAMMER,
                        pattern_name="inverted_hammer",
                        start_idx=i, end_idx=i,
                        confidence=min(1.0, us / (b * self.shadow_ratio + 1e-10)),
                        direction='bullish',
                    ))
            
            # Shooting Star (in uptrend)
            if i >= 3:
                if c[i - 3] < c[i - 2] < c[i - 1]:
                    if us > b * self.shadow_ratio and ls < b * 0.5:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.SHOOTING_STAR,
                            pattern_name="shooting_star",
                            start_idx=i, end_idx=i,
                            confidence=0.8, direction='bearish',
                        ))
            
            # Hanging Man (in uptrend)
            if i >= 3:
                if c[i - 3] < c[i - 2] < c[i - 1]:
                    if ls > b * self.shadow_ratio and us < b * 0.5:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.HANGING_MAN,
                            pattern_name="hanging_man",
                            start_idx=i, end_idx=i,
                            confidence=0.8, direction='bearish',
                        ))
            
            # Marubozu Bull
            if bull and br > 0.9:
                if us < b * 0.05 and ls < b * 0.05:
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.MARUBOZU_BULL,
                        pattern_name="marubozu_bull",
                        start_idx=i, end_idx=i,
                        confidence=br, direction='bullish',
                    ))
            
            # Marubozu Bear
            if (not bull) and br > 0.9:
                if us < b * 0.05 and ls < b * 0.05:
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.MARUBOZU_BEAR,
                        pattern_name="marubozu_bear",
                        start_idx=i, end_idx=i,
                        confidence=br, direction='bearish',
                    ))

            # Pin Bar (generic) - wick >= 3x body, body <= 25% of range
            if b > 0:
                if max(us, ls) >= 3.0 * b and (b / r) <= 0.25:
                    direction = 'bullish' if ls > us else 'bearish'
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.PIN_BAR,
                        pattern_name="pin_bar",
                        start_idx=i, end_idx=i,
                        confidence=min(1.0, max(us, ls) / (3.0 * b + 1e-10)),
                        direction=direction,
                    ))
        
        return patterns
    
    def _detect_two_candle_patterns(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect two-candle patterns."""
        patterns = []

        if df is None or len(df) < 2:
            return patterns

        o, h, l, c = self._get_ohlc_arrays(df)
        n = len(o)

        body = np.abs(c - o)
        rng = np.maximum(h - l, 1e-10)
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l
        is_bullish = c > o
        body_ratio = body / rng

        # A small rolling median range proxy for reversal qualification
        median_window = 20
        
        for i in range(1, n):
            prev_o = float(o[i - 1])
            prev_h = float(h[i - 1])
            prev_l = float(l[i - 1])
            prev_c = float(c[i - 1])
            curr_o = float(o[i])
            curr_h = float(h[i])
            curr_l = float(l[i])
            curr_c = float(c[i])

            prev_bull = bool(is_bullish[i - 1])
            curr_bull = bool(is_bullish[i])
            prev_body = float(body[i - 1])
            curr_body = float(body[i])
            
            # Bullish Engulfing
            if (not prev_bull) and curr_bull:
                if curr_o < prev_c and curr_c > prev_o:
                    if curr_body > prev_body * self.engulf_threshold:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.BULLISH_ENGULFING,
                            pattern_name="bullish_engulfing",
                            start_idx=i-1, end_idx=i,
                            confidence=min(1.0, curr_body / (prev_body + 1e-10)),
                            direction='bullish',
                        ))
            
            # Bearish Engulfing
            if prev_bull and (not curr_bull):
                if curr_o > prev_c and curr_c < prev_o:
                    if curr_body > prev_body * self.engulf_threshold:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.BEARISH_ENGULFING,
                            pattern_name="bearish_engulfing",
                            start_idx=i-1, end_idx=i,
                            confidence=min(1.0, curr_body / (prev_body + 1e-10)),
                            direction='bearish',
                        ))
            
            # Bullish Harami
            if (not prev_bull) and curr_bull:
                if curr_o > prev_c and curr_c < prev_o:
                    if curr_body < prev_body * 0.5:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.BULLISH_HARAMI,
                            pattern_name="bullish_harami",
                            start_idx=i-1, end_idx=i,
                            confidence=0.7, direction='bullish',
                        ))
            
            # Bearish Harami
            if prev_bull and (not curr_bull):
                if curr_o < prev_c and curr_c > prev_o:
                    if curr_body < prev_body * 0.5:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.BEARISH_HARAMI,
                            pattern_name="bearish_harami",
                            start_idx=i-1, end_idx=i,
                            confidence=0.7, direction='bearish',
                        ))
            
            # Piercing Line
            if (not prev_bull) and curr_bull:
                if curr_o < prev_l:
                    midpoint = (prev_o + prev_c) / 2
                    if curr_c > midpoint and curr_c < prev_o:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.PIERCING_LINE,
                            pattern_name="piercing_line",
                            start_idx=i-1, end_idx=i,
                            confidence=0.8, direction='bullish',
                        ))
            
            # Dark Cloud Cover
            if prev_bull and (not curr_bull):
                if curr_o > prev_h:
                    midpoint = (prev_o + prev_c) / 2
                    if curr_c < midpoint and curr_c > prev_o:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.DARK_CLOUD_COVER,
                            pattern_name="dark_cloud_cover",
                            start_idx=i-1, end_idx=i,
                            confidence=0.8, direction='bearish',
                        ))
            
            # Tweezer Top
            if abs(prev_h - curr_h) < float(rng[i - 1]) * 0.05:
                if prev_bull and (not curr_bull):
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.TWEEZER_TOP,
                        pattern_name="tweezer_top",
                        start_idx=i-1, end_idx=i,
                        confidence=0.7, direction='bearish',
                    ))
            
            # Tweezer Bottom
            if abs(prev_l - curr_l) < float(rng[i - 1]) * 0.05:
                if (not prev_bull) and curr_bull:
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.TWEEZER_BOTTOM,
                        pattern_name="tweezer_bottom",
                        start_idx=i-1, end_idx=i,
                        confidence=0.7, direction='bullish',
                    ))

            # Inside Bar
            if curr_h < prev_h and curr_l > prev_l:
                patterns.append(PatternDetection(
                    pattern_class=PatternClass.INSIDE_BAR,
                    pattern_name="inside_bar",
                    start_idx=i-1, end_idx=i,
                    confidence=0.8,
                    direction='neutral',
                ))

            # Outside Bar (generic engulfing)
            if curr_h > prev_h and curr_l < prev_l:
                patterns.append(PatternDetection(
                    pattern_class=PatternClass.OUTSIDE_BAR,
                    pattern_name="outside_bar",
                    start_idx=i-1, end_idx=i,
                    confidence=0.8,
                    direction='neutral',
                ))

            # Two-Bar Reversal
            # Sign flip between candle bodies + unusually large current range
            if (curr_c - curr_o) * (prev_c - prev_o) < 0:
                start = max(0, i - median_window)
                med = float(np.median(rng[start:i + 1])) if i > start else float(rng[i])
                if float(rng[i]) >= 1.5 * (med + 1e-10):
                    direction = 'bullish' if curr_bull else 'bearish'
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.TWO_BAR_REVERSAL,
                        pattern_name="two_bar_reversal",
                        start_idx=i-1, end_idx=i,
                        confidence=min(1.0, float(rng[i]) / (1.5 * (med + 1e-10))),
                        direction=direction,
                    ))
        
        return patterns
    
    def _detect_three_candle_patterns(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect three-candle patterns."""
        patterns = []

        if df is None or len(df) < 3:
            return patterns

        o, h, l, c = self._get_ohlc_arrays(df)
        n = len(o)

        body = np.abs(c - o)
        rng = np.maximum(h - l, 1e-10)
        is_bullish = c > o

        median_window = 20
        
        for i in range(2, n):
            c1_o = float(o[i - 2]); c1_c = float(c[i - 2]); c1_h = float(h[i - 2]); c1_l = float(l[i - 2])
            c2_o = float(o[i - 1]); c2_c = float(c[i - 1]); c2_h = float(h[i - 1]); c2_l = float(l[i - 1])
            c3_o = float(o[i]); c3_c = float(c[i]);

            m1_is_bull = bool(is_bullish[i - 2])
            m2_body_ratio = float(body[i - 1] / (rng[i - 1] + 1e-10))
            m3_is_bull = bool(is_bullish[i])
            
            # Morning Star
            if (not m1_is_bull) and m3_is_bull:
                if m2_body_ratio < 0.3:
                    if c2_l < c1_c and c3_c > (c1_o + c1_c) / 2:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.MORNING_STAR,
                            pattern_name="morning_star",
                            start_idx=i-2, end_idx=i,
                            confidence=0.85, direction='bullish',
                        ))
            
            # Evening Star
            if m1_is_bull and (not m3_is_bull):
                if m2_body_ratio < 0.3:
                    if c2_h > c1_c and c3_c < (c1_o + c1_c) / 2:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.EVENING_STAR,
                            pattern_name="evening_star",
                            start_idx=i-2, end_idx=i,
                            confidence=0.85, direction='bearish',
                        ))
            
            # Three White Soldiers
            if bool(is_bullish[i - 2]) and bool(is_bullish[i - 1]) and bool(is_bullish[i]):
                if c2_c > c1_c and c3_c > c2_c:
                    if c2_o > c1_o and c2_o < c1_c:
                        if c3_o > c2_o and c3_o < c2_c:
                            if (body[i - 2] / rng[i - 2]) > 0.5 and (body[i - 1] / rng[i - 1]) > 0.5 and (body[i] / rng[i]) > 0.5:
                                patterns.append(PatternDetection(
                                    pattern_class=PatternClass.THREE_WHITE_SOLDIERS,
                                    pattern_name="three_white_soldiers",
                                    start_idx=i-2, end_idx=i,
                                    confidence=0.9, direction='bullish',
                                ))
            
            # Three Black Crows
            if (not bool(is_bullish[i - 2])) and (not bool(is_bullish[i - 1])) and (not bool(is_bullish[i])):
                if c2_c < c1_c and c3_c < c2_c:
                    if c2_o < c1_o and c2_o > c1_c:
                        if c3_o < c2_o and c3_o > c2_c:
                            if (body[i - 2] / rng[i - 2]) > 0.5 and (body[i - 1] / rng[i - 1]) > 0.5 and (body[i] / rng[i]) > 0.5:
                                patterns.append(PatternDetection(
                                    pattern_class=PatternClass.THREE_BLACK_CROWS,
                                    pattern_name="three_black_crows",
                                    start_idx=i-2, end_idx=i,
                                    confidence=0.9, direction='bearish',
                                ))

            # Three-bar play (simplified)
            # Impulse bar + inside bar + reversal bar
            start = max(0, i - median_window)
            med = float(np.median(rng[start:i + 1])) if i > start else float(rng[i])
            impulse = float(rng[i - 2]) >= 1.5 * (med + 1e-10)
            inside = (float(h[i - 1]) < float(h[i - 2])) and (float(l[i - 1]) > float(l[i - 2]))
            if impulse and inside:
                # Reversal bar: direction opposite to impulse bar
                impulse_dir = (c1_c - c1_o)
                rev_dir = (c3_c - c3_o)
                if impulse_dir * rev_dir < 0:
                    direction = 'bullish' if rev_dir > 0 else 'bearish'
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.THREE_BAR_PLAY,
                        pattern_name="three_bar_play",
                        start_idx=i - 2, end_idx=i,
                        confidence=0.75,
                        direction=direction,
                    ))
        
        return patterns
    
    def to_yolo_annotations(self, patterns: List[PatternDetection]) -> List[Dict]:
        """Convert pattern detections to YOLO annotation format."""
        return [
            {
                'class_id': p.pattern_class,
                'start_idx': p.start_idx,
                'end_idx': p.end_idx,
            }
            for p in patterns
        ]


def detect_patterns(df: pd.DataFrame) -> List[PatternDetection]:
    """Convenience function to detect all patterns."""
    detector = CandlestickPatternDetector()
    return detector.detect_all_patterns(df)


if __name__ == "__main__":
    np.random.seed(42)
    n = 100
    base = 100
    data = []
    
    for i in range(n):
        change = np.random.randn() * 0.5
        o = base
        c = base + change
        h = max(o, c) + abs(np.random.randn() * 0.3)
        l = min(o, c) - abs(np.random.randn() * 0.3)
        data.append({'open': o, 'high': h, 'low': l, 'close': c})
        base = c
    
    df = pd.DataFrame(data)
    
    detector = CandlestickPatternDetector()
    patterns = detector.detect_all_patterns(df)
    
    print(f"Detected {len(patterns)} patterns:")
    for p in patterns[:10]:
        print(f"  {p.pattern_name}: idx {p.start_idx}-{p.end_idx}, conf={p.confidence:.2f}")
