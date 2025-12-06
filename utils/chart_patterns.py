"""
chart_patterns.py

Detectors for common chart patterns (flags, pennants, wedges, triangles,
rectangles, double/triple tops/bottoms, head & shoulders, cup & handle).
Each detector returns a boolean Series aligned with df.index.
Designed for OHLCV DataFrames with columns: ['open','high','low','close','volume'].
"""

import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# -----------------------------
# Utilities
# -----------------------------
def _find_peaks(series: pd.Series, distance=5, prominence=None):
    """Indices of local maxima."""
    peaks, _ = find_peaks(series.values, distance=distance, prominence=prominence)
    return peaks

def _find_troughs(series: pd.Series, distance=5, prominence=None):
    """Indices of local minima."""
    troughs, _ = find_peaks((-series).values, distance=distance, prominence=prominence)
    return troughs

def _mark_index(index_len: int, mark_pos: int):
    """Helper to produce a boolean mask with True at mark_pos."""
    mask = np.zeros(index_len, dtype=bool)
    if 0 <= mark_pos < index_len:
        mask[mark_pos] = True
    return mask

# -----------------------------
# Double / Triple Tops & Bottoms
# -----------------------------
def pattern_double_top(df: pd.DataFrame, distance=5, tol=0.02) -> pd.Series:
    """
    Two peaks at similar levels with a dip between.
    Marks the second peak.
    """
    peaks = _find_peaks(df["close"], distance=distance)
    mask = np.zeros(len(df), dtype=bool)
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i + 1]
        price1, price2 = df["close"].iloc[p1], df["close"].iloc[p2]
        if abs(price1 - price2) / max(price1, 1e-12) <= tol and (p2 - p1) >= distance:
            mask[p2] = True
    return pd.Series(mask, index=df.index)

def pattern_double_bottom(df: pd.DataFrame, distance=5, tol=0.02) -> pd.Series:
    """Two troughs at similar levels with a peak between. Marks the second trough."""
    troughs = _find_troughs(df["close"], distance=distance)
    mask = np.zeros(len(df), dtype=bool)
    for i in range(len(troughs) - 1):
        t1, t2 = troughs[i], troughs[i + 1]
        price1, price2 = df["close"].iloc[t1], df["close"].iloc[t2]
        if abs(price1 - price2) / max(price1, 1e-12) <= tol and (t2 - t1) >= distance:
            mask[t2] = True
    return pd.Series(mask, index=df.index)

def pattern_triple_top(df: pd.DataFrame, distance=5, tol=0.02) -> pd.Series:
    """Three peaks at similar levels. Marks the third peak."""
    peaks = _find_peaks(df["close"], distance=distance)
    mask = np.zeros(len(df), dtype=bool)
    for i in range(len(peaks) - 2):
        p1, p2, p3 = peaks[i], peaks[i + 1], peaks[i + 2]
        prices = df["close"].iloc[[p1, p2, p3]]
        if (prices.max() - prices.min()) / max(prices.mean(), 1e-12) <= tol:
            mask[p3] = True
    return pd.Series(mask, index=df.index)

def pattern_triple_bottom(df: pd.DataFrame, distance=5, tol=0.02) -> pd.Series:
    """Three troughs at similar levels. Marks the third trough."""
    troughs = _find_troughs(df["close"], distance=distance)
    mask = np.zeros(len(df), dtype=bool)
    for i in range(len(troughs) - 2):
        t1, t2, t3 = troughs[i], troughs[i + 1], troughs[i + 2]
        prices = df["close"].iloc[[t1, t2, t3]]
        if (prices.max() - prices.min()) / max(prices.mean(), 1e-12) <= tol:
            mask[t3] = True
    return pd.Series(mask, index=df.index)

# -----------------------------
# Head & Shoulders
# -----------------------------
def pattern_head_and_shoulders(df: pd.DataFrame, distance=5, tol=0.03) -> pd.Series:
    """
    Three peaks with the middle (head) higher than shoulders (left/right) and shoulders similar.
    Marks the right shoulder.
    """
    peaks = _find_peaks(df["close"], distance=distance)
    mask = np.zeros(len(df), dtype=bool)
    for i in range(len(peaks) - 2):
        l, h, r = peaks[i], peaks[i + 1], peaks[i + 2]
        left, head, right = df["close"].iloc[[l, h, r]]
        if head > left and head > right and abs(left - right) / max(head, 1e-12) <= tol:
            mask[r] = True
    return pd.Series(mask, index=df.index)

def pattern_inverse_head_and_shoulders(df: pd.DataFrame, distance=5, tol=0.03) -> pd.Series:
    """
    Three troughs with the middle (head) lower than shoulders (left/right) and shoulders similar.
    Marks the right shoulder trough.
    """
    troughs = _find_troughs(df["close"], distance=distance)
    mask = np.zeros(len(df), dtype=bool)
    for i in range(len(troughs) - 2):
        l, h, r = troughs[i], troughs[i + 1], troughs[i + 2]
        left, head, right = df["close"].iloc[[l, h, r]]
        if head < left and head < right and abs(left - right) / max(abs(head), 1e-12) <= tol:
            mask[r] = True
    return pd.Series(mask, index=df.index)

# -----------------------------
# Flags & Pennants (approximate)
# -----------------------------
def _consolidation_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std()

def pattern_bullish_flag(df: pd.DataFrame, lookback=20, pole_return=0.08, cons_std_ratio=0.02) -> pd.Series:
    """
    Sharp rise (flagpole) then narrow rectangular consolidation.
    Marks end of consolidation window.
    """
    returns = df["close"].pct_change(lookback)
    cons_std = _consolidation_std(df["close"], lookback)
    cond = (returns > pole_return) & (cons_std < df["close"].rolling(lookback).mean() * cons_std_ratio)
    return cond.fillna(False)

def pattern_bearish_flag(df: pd.DataFrame, lookback=20, pole_return=-0.08, cons_std_ratio=0.02) -> pd.Series:
    returns = df["close"].pct_change(lookback)
    cons_std = _consolidation_std(df["close"], lookback)
    cond = (returns < pole_return) & (cons_std < df["close"].rolling(lookback).mean() * cons_std_ratio)
    return cond.fillna(False)

def pattern_bullish_pennant(df: pd.DataFrame, lookback=20, pole_return=0.08) -> pd.Series:
    """
    Sharp rise followed by small triangular consolidation (range compressing).
    """
    returns = df["close"].pct_change(lookback)
    range_width = (df["high"].rolling(lookback).max() - df["low"].rolling(lookback).min())
    cond = (returns > pole_return) & (range_width.diff().rolling(3).mean() < 0)
    return cond.fillna(False)

def pattern_bearish_pennant(df: pd.DataFrame, lookback=20, pole_return=-0.08) -> pd.Series:
    returns = df["close"].pct_change(lookback)
    range_width = (df["high"].rolling(lookback).max() - df["low"].rolling(lookback).min())
    cond = (returns < pole_return) & (range_width.diff().rolling(3).mean() < 0)
    return cond.fillna(False)

# -----------------------------
# Rectangles, Triangles, Wedges
# -----------------------------
def pattern_rectangle(df: pd.DataFrame, lookback=20, tol=0.02) -> pd.Series:
    """
    Horizontal consolidation between support/resistance.
    """
    high = df["high"].rolling(lookback).max()
    low = df["low"].rolling(lookback).min()
    cond = ((high - low) / df["close"].rolling(lookback).mean()) < tol
    return cond.fillna(False)

def pattern_triangle(df: pd.DataFrame, lookback=30) -> pd.Series:
    """
    Converging highs and lows (symmetrical triangle).
    Simple proxy: decreasing high range and increasing low range toward midline; or both ranges decreasing.
    """
    hi_max = df["high"].rolling(lookback).max()
    hi_min = df["high"].rolling(lookback).min()
    lo_max = df["low"].rolling(lookback).max()
    lo_min = df["low"].rolling(lookback).min()
    hi_range = (hi_max - hi_min).diff().rolling(3).mean()
    lo_range = (lo_max - lo_min).diff().rolling(3).mean()
    cond = (hi_range < 0) & (lo_range > 0) | ((hi_range < 0) & (lo_range < 0))
    return cond.fillna(False)

def pattern_falling_wedge(df: pd.DataFrame, lookback=30) -> pd.Series:
    """
    Both highs and lows making lower values, ranges converging.
    """
    hi_max = df["high"].rolling(lookback).max().diff()
    lo_min = df["low"].rolling(lookback).min().diff()
    range_narrowing = (df["high"].rolling(lookback).max() - df["low"].rolling(lookback).min()).diff().rolling(3).mean() < 0
    cond = (hi_max < 0) & (lo_min < 0) & range_narrowing
    return cond.fillna(False)

def pattern_rising_wedge(df: pd.DataFrame, lookback=30) -> pd.Series:
    """
    Both highs and lows making higher values, ranges converging.
    """
    hi_max = df["high"].rolling(lookback).max().diff()
    lo_min = df["low"].rolling(lookback).min().diff()
    range_narrowing = (df["high"].rolling(lookback).max() - df["low"].rolling(lookback).min()).diff().rolling(3).mean() < 0
    cond = (hi_max > 0) & (lo_min > 0) & range_narrowing
    return cond.fillna(False)

# -----------------------------
# Cup & Handle
# -----------------------------
def pattern_cup_and_handle(df: pd.DataFrame, lookback=50, min_depth=0.12, max_depth=0.5, handle_window=10) -> pd.Series:
    """
    U-shaped base followed by small pullback (handle).
    Proxy: sizeable drawdown then recovery near prior high, then small dip.
    Marks end of handle window.
    """
    roll_max = df["close"].rolling(lookback).max()
    roll_min = df["close"].rolling(lookback).min()
    depth = (roll_max - roll_min) / roll_max.replace(0, np.nan)
    recovered = df["close"] > (roll_max * (1 - 0.05))  # near prior highs
    handle = df["close"].diff(handle_window) < 0  # small dip
    cond = (depth.between(min_depth, max_depth)) & recovered & handle
    return cond.fillna(False)

def pattern_inverted_cup_and_handle(df: pd.DataFrame, lookback=50, min_depth=0.12, max_depth=0.5, handle_window=10) -> pd.Series:
    """
    Inverted U-shaped top followed by small bounce (handle).
    Proxy: sizeable run-up then drop near prior low, then small bounce.
    """
    roll_max = df["close"].rolling(lookback).max()
    roll_min = df["close"].rolling(lookback).min()
    depth = (roll_max - roll_min) / roll_min.replace(0, np.nan)
    dropped = df["close"] < (roll_min * (1 + 0.05))  # near prior lows
    handle = df["close"].diff(handle_window) > 0  # small bounce
    cond = (depth.between(min_depth, max_depth)) & dropped & handle
    return cond.fillna(False)

# -----------------------------
# Registry
# -----------------------------
CHART_PATTERN_FUNCS = {
    # Tops/Bottoms
    "pattern_double_top": pattern_double_top,
    "pattern_double_bottom": pattern_double_bottom,
    "pattern_triple_top": pattern_triple_top,
    "pattern_triple_bottom": pattern_triple_bottom,

    # Head & Shoulders
    "pattern_head_and_shoulders": pattern_head_and_shoulders,
    "pattern_inverse_head_and_shoulders": pattern_inverse_head_and_shoulders,

    # Flags & Pennants
    "pattern_bullish_flag": pattern_bullish_flag,
    "pattern_bearish_flag": pattern_bearish_flag,
    "pattern_bullish_pennant": pattern_bullish_pennant,
    "pattern_bearish_pennant": pattern_bearish_pennant,

    # Consolidations / Trendlines
    "pattern_rectangle": pattern_rectangle,
    "pattern_triangle": pattern_triangle,
    "pattern_falling_wedge": pattern_falling_wedge,
    "pattern_rising_wedge": pattern_rising_wedge,

    # Cup & Handle
    "pattern_cup_and_handle": pattern_cup_and_handle,
    "pattern_inverted_cup_and_handle": pattern_inverted_cup_and_handle,
}
