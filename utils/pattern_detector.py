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


PATTERN_NAMES = [
    "doji", "hammer", "inverted_hammer", "bullish_engulfing", "bearish_engulfing",
    "morning_star", "evening_star", "three_white_soldiers", "three_black_crows",
    "bullish_harami", "bearish_harami", "shooting_star", "hanging_man",
    "piercing_line", "dark_cloud_cover", "tweezer_top", "tweezer_bottom",
    "spinning_top", "marubozu_bull", "marubozu_bear"
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
        
        for i in range(len(df)):
            row = df.iloc[i]
            m = self._get_candle_metrics(row)
            
            # Doji
            if m['body_ratio'] < self.doji_threshold:
                patterns.append(PatternDetection(
                    pattern_class=PatternClass.DOJI,
                    pattern_name="doji",
                    start_idx=i, end_idx=i,
                    confidence=1.0 - m['body_ratio'] / self.doji_threshold,
                    direction='neutral',
                ))
            
            # Spinning Top
            elif self.doji_threshold <= m['body_ratio'] < 0.3:
                if m['upper_shadow'] > m['body'] * 0.5 and m['lower_shadow'] > m['body'] * 0.5:
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.SPINNING_TOP,
                        pattern_name="spinning_top",
                        start_idx=i, end_idx=i,
                        confidence=0.7, direction='neutral',
                    ))
            
            # Hammer
            if m['lower_shadow'] > m['body'] * self.shadow_ratio and m['upper_shadow'] < m['body'] * 0.5:
                if m['body_ratio'] < 0.4:
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.HAMMER,
                        pattern_name="hammer",
                        start_idx=i, end_idx=i,
                        confidence=min(1.0, m['lower_shadow'] / (m['body'] * self.shadow_ratio)),
                        direction='bullish',
                    ))
            
            # Inverted Hammer
            if m['upper_shadow'] > m['body'] * self.shadow_ratio and m['lower_shadow'] < m['body'] * 0.5:
                if m['body_ratio'] < 0.4:
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.INVERTED_HAMMER,
                        pattern_name="inverted_hammer",
                        start_idx=i, end_idx=i,
                        confidence=min(1.0, m['upper_shadow'] / (m['body'] * self.shadow_ratio)),
                        direction='bullish',
                    ))
            
            # Shooting Star (in uptrend)
            if i >= 3:
                prev_closes = df['close'].iloc[i-3:i].values
                if all(prev_closes[j] < prev_closes[j+1] for j in range(len(prev_closes)-1)):
                    if m['upper_shadow'] > m['body'] * self.shadow_ratio and m['lower_shadow'] < m['body'] * 0.5:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.SHOOTING_STAR,
                            pattern_name="shooting_star",
                            start_idx=i, end_idx=i,
                            confidence=0.8, direction='bearish',
                        ))
            
            # Hanging Man (in uptrend)
            if i >= 3:
                prev_closes = df['close'].iloc[i-3:i].values
                if all(prev_closes[j] < prev_closes[j+1] for j in range(len(prev_closes)-1)):
                    if m['lower_shadow'] > m['body'] * self.shadow_ratio and m['upper_shadow'] < m['body'] * 0.5:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.HANGING_MAN,
                            pattern_name="hanging_man",
                            start_idx=i, end_idx=i,
                            confidence=0.8, direction='bearish',
                        ))
            
            # Marubozu Bull
            if m['is_bullish'] and m['body_ratio'] > 0.9:
                if m['upper_shadow'] < m['body'] * 0.05 and m['lower_shadow'] < m['body'] * 0.05:
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.MARUBOZU_BULL,
                        pattern_name="marubozu_bull",
                        start_idx=i, end_idx=i,
                        confidence=m['body_ratio'], direction='bullish',
                    ))
            
            # Marubozu Bear
            if not m['is_bullish'] and m['body_ratio'] > 0.9:
                if m['upper_shadow'] < m['body'] * 0.05 and m['lower_shadow'] < m['body'] * 0.05:
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.MARUBOZU_BEAR,
                        pattern_name="marubozu_bear",
                        start_idx=i, end_idx=i,
                        confidence=m['body_ratio'], direction='bearish',
                    ))
        
        return patterns
    
    def _detect_two_candle_patterns(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect two-candle patterns."""
        patterns = []
        
        for i in range(1, len(df)):
            prev = df.iloc[i-1]
            curr = df.iloc[i]
            
            m_prev = self._get_candle_metrics(prev)
            m_curr = self._get_candle_metrics(curr)
            
            # Bullish Engulfing
            if not m_prev['is_bullish'] and m_curr['is_bullish']:
                if curr['open'] < prev['close'] and curr['close'] > prev['open']:
                    if m_curr['body'] > m_prev['body'] * self.engulf_threshold:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.BULLISH_ENGULFING,
                            pattern_name="bullish_engulfing",
                            start_idx=i-1, end_idx=i,
                            confidence=min(1.0, m_curr['body'] / (m_prev['body'] + 1e-10)),
                            direction='bullish',
                        ))
            
            # Bearish Engulfing
            if m_prev['is_bullish'] and not m_curr['is_bullish']:
                if curr['open'] > prev['close'] and curr['close'] < prev['open']:
                    if m_curr['body'] > m_prev['body'] * self.engulf_threshold:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.BEARISH_ENGULFING,
                            pattern_name="bearish_engulfing",
                            start_idx=i-1, end_idx=i,
                            confidence=min(1.0, m_curr['body'] / (m_prev['body'] + 1e-10)),
                            direction='bearish',
                        ))
            
            # Bullish Harami
            if not m_prev['is_bullish'] and m_curr['is_bullish']:
                if curr['open'] > prev['close'] and curr['close'] < prev['open']:
                    if m_curr['body'] < m_prev['body'] * 0.5:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.BULLISH_HARAMI,
                            pattern_name="bullish_harami",
                            start_idx=i-1, end_idx=i,
                            confidence=0.7, direction='bullish',
                        ))
            
            # Bearish Harami
            if m_prev['is_bullish'] and not m_curr['is_bullish']:
                if curr['open'] < prev['close'] and curr['close'] > prev['open']:
                    if m_curr['body'] < m_prev['body'] * 0.5:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.BEARISH_HARAMI,
                            pattern_name="bearish_harami",
                            start_idx=i-1, end_idx=i,
                            confidence=0.7, direction='bearish',
                        ))
            
            # Piercing Line
            if not m_prev['is_bullish'] and m_curr['is_bullish']:
                if curr['open'] < prev['low']:
                    midpoint = (prev['open'] + prev['close']) / 2
                    if curr['close'] > midpoint and curr['close'] < prev['open']:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.PIERCING_LINE,
                            pattern_name="piercing_line",
                            start_idx=i-1, end_idx=i,
                            confidence=0.8, direction='bullish',
                        ))
            
            # Dark Cloud Cover
            if m_prev['is_bullish'] and not m_curr['is_bullish']:
                if curr['open'] > prev['high']:
                    midpoint = (prev['open'] + prev['close']) / 2
                    if curr['close'] < midpoint and curr['close'] > prev['open']:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.DARK_CLOUD_COVER,
                            pattern_name="dark_cloud_cover",
                            start_idx=i-1, end_idx=i,
                            confidence=0.8, direction='bearish',
                        ))
            
            # Tweezer Top
            if abs(prev['high'] - curr['high']) < m_prev['range'] * 0.05:
                if m_prev['is_bullish'] and not m_curr['is_bullish']:
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.TWEEZER_TOP,
                        pattern_name="tweezer_top",
                        start_idx=i-1, end_idx=i,
                        confidence=0.7, direction='bearish',
                    ))
            
            # Tweezer Bottom
            if abs(prev['low'] - curr['low']) < m_prev['range'] * 0.05:
                if not m_prev['is_bullish'] and m_curr['is_bullish']:
                    patterns.append(PatternDetection(
                        pattern_class=PatternClass.TWEEZER_BOTTOM,
                        pattern_name="tweezer_bottom",
                        start_idx=i-1, end_idx=i,
                        confidence=0.7, direction='bullish',
                    ))
        
        return patterns
    
    def _detect_three_candle_patterns(self, df: pd.DataFrame) -> List[PatternDetection]:
        """Detect three-candle patterns."""
        patterns = []
        
        for i in range(2, len(df)):
            c1 = df.iloc[i-2]
            c2 = df.iloc[i-1]
            c3 = df.iloc[i]
            
            m1 = self._get_candle_metrics(c1)
            m2 = self._get_candle_metrics(c2)
            m3 = self._get_candle_metrics(c3)
            
            # Morning Star
            if not m1['is_bullish'] and m3['is_bullish']:
                if m2['body_ratio'] < 0.3:
                    if c2['low'] < c1['close'] and c3['close'] > (c1['open'] + c1['close']) / 2:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.MORNING_STAR,
                            pattern_name="morning_star",
                            start_idx=i-2, end_idx=i,
                            confidence=0.85, direction='bullish',
                        ))
            
            # Evening Star
            if m1['is_bullish'] and not m3['is_bullish']:
                if m2['body_ratio'] < 0.3:
                    if c2['high'] > c1['close'] and c3['close'] < (c1['open'] + c1['close']) / 2:
                        patterns.append(PatternDetection(
                            pattern_class=PatternClass.EVENING_STAR,
                            pattern_name="evening_star",
                            start_idx=i-2, end_idx=i,
                            confidence=0.85, direction='bearish',
                        ))
            
            # Three White Soldiers
            if m1['is_bullish'] and m2['is_bullish'] and m3['is_bullish']:
                if c2['close'] > c1['close'] and c3['close'] > c2['close']:
                    if c2['open'] > c1['open'] and c2['open'] < c1['close']:
                        if c3['open'] > c2['open'] and c3['open'] < c2['close']:
                            if m1['body_ratio'] > 0.5 and m2['body_ratio'] > 0.5 and m3['body_ratio'] > 0.5:
                                patterns.append(PatternDetection(
                                    pattern_class=PatternClass.THREE_WHITE_SOLDIERS,
                                    pattern_name="three_white_soldiers",
                                    start_idx=i-2, end_idx=i,
                                    confidence=0.9, direction='bullish',
                                ))
            
            # Three Black Crows
            if not m1['is_bullish'] and not m2['is_bullish'] and not m3['is_bullish']:
                if c2['close'] < c1['close'] and c3['close'] < c2['close']:
                    if c2['open'] < c1['open'] and c2['open'] > c1['close']:
                        if c3['open'] < c2['open'] and c3['open'] > c2['close']:
                            if m1['body_ratio'] > 0.5 and m2['body_ratio'] > 0.5 and m3['body_ratio'] > 0.5:
                                patterns.append(PatternDetection(
                                    pattern_class=PatternClass.THREE_BLACK_CROWS,
                                    pattern_name="three_black_crows",
                                    start_idx=i-2, end_idx=i,
                                    confidence=0.9, direction='bearish',
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
