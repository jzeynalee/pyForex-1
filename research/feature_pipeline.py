"""
Shared feature engineering pipeline for all variants.

Wraps the existing FeatureEngineerOptimized and adds:
- Economic category grouping for Alpha2
- Standardized feature matrix output
- No future leakage guarantees
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Economic categories — must be orthogonal
# ---------------------------------------------------------------------------

CATEGORY_DEFINITIONS: Dict[str, List[str]] = {
    "trend": [
        "ema_8", "ema_21", "ema_50", "ema_200",
        "sma_20", "sma_50", "sma_200",
        "trend_short", "trend_medium", "trend_long",
        "adx", "dmi_plus", "dmi_minus",
        "lsma", "mcginley_dynamic",
        "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b",
        "aroon_up", "aroon_down", "aroon_oscillator",
        "supertrend", "parabolic_sar",
        "linear_regression_slope", "linear_regression_intercept",
    ],
    "momentum": [
        "momentum", "momentum_10", "momentum_20",
        "roc", "roc_10", "roc_20",
        "trix", "trix_signal",
        "awesome_oscillator",
        "williams_r",
        "ultimate_oscillator",
        "kst", "kst_signal",
        "ppo", "ppo_signal",
    ],
    "oscillator": [
        "rsi", "rsi_14",
        "stoch_k", "stoch_d", "stoch_rsi",
        "cci", "cci_20",
        "macd", "macd_signal", "macd_histogram",
        "mfi",
        "dpo",
        "cmo",
    ],
    "volatility": [
        "atr", "atr_14", "atr_ratio",
        "bb_upper", "bb_lower", "bb_middle", "bb_width", "bb_percent",
        "keltner_upper", "keltner_lower", "keltner_width",
        "donchian_upper", "donchian_lower", "donchian_width",
        "natr",
        "historical_volatility", "volatility_ratio",
        "true_range",
    ],
    "volume": [
        "obv", "obv_ema",
        "vwap",
        "cmf", "cmf_20",
        "ad_line", "ad_oscillator",
        "volume_sma", "volume_ratio",
        "volume_oscillator",
        "eom", "eom_14",
        "force_index",
    ],
    "structure": [
        "swing_high", "swing_low",
        "higher_lows", "lower_highs",
        "support_level", "resistance_level",
        "pivot_point", "pivot_r1", "pivot_s1",
        "fib_382", "fib_500", "fib_618",
        "fractal_up", "fractal_down",
        "heikin_ashi_trend",
    ],
}

ALL_CATEGORIES = list(CATEGORY_DEFINITIONS.keys())


def _safe_match(feature_cols: List[str], patterns: List[str]) -> List[str]:
    """Match feature columns to pattern list (case-insensitive substring)."""
    matched = []
    fc_lower = {c: c.lower().strip() for c in feature_cols}
    for pattern in patterns:
        p = pattern.lower().strip()
        for col, col_lower in fc_lower.items():
            if p == col_lower or p in col_lower:
                if col not in matched:
                    matched.append(col)
    return matched


class FeaturePipeline:
    """Shared, deterministic feature pipeline for all variants.

    Wraps FeatureEngineerOptimized from alpha_factory and provides:
    - Full feature DataFrame
    - Category-grouped feature subsets for Alpha2
    - Standardized numpy matrices for MH-TCN input
    """

    def __init__(self, max_lookback: int = 500):
        self.max_lookback = max_lookback
        self._feature_engineer = None
        self._category_columns: Dict[str, List[str]] = {}

    def _get_engineer(self):
        if self._feature_engineer is None:
            from alpha_factory.features_engineering import FeatureEngineerOptimized
            self._feature_engineer = FeatureEngineerOptimized()
        return self._feature_engineer

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all features from OHLCV data.

        Returns DataFrame aligned with input index. NaN rows at the start
        (from lookback) are preserved — callers must handle them.
        """
        engineer = self._get_engineer()
        features = engineer.generate_features(df, batch_processing=True)
        # Cache category mappings on first call
        if not self._category_columns:
            self._build_category_map(features.columns.tolist())
        return features

    def _build_category_map(self, columns: List[str]):
        """Map actual feature columns to economic categories."""
        assigned = set()
        for cat, patterns in CATEGORY_DEFINITIONS.items():
            matched = _safe_match(columns, patterns)
            self._category_columns[cat] = matched
            assigned.update(matched)
        # Unassigned features go into a catch-all (not used for Alpha2 fusion)
        remaining = [c for c in columns if c not in assigned]
        if remaining:
            self._category_columns["_other"] = remaining

    def get_category_columns(self) -> Dict[str, List[str]]:
        """Return mapping of category -> list of column names."""
        return dict(self._category_columns)

    def get_category_features(
        self, features: pd.DataFrame, category: str
    ) -> pd.DataFrame:
        """Extract features for a single category."""
        cols = self._category_columns.get(category, [])
        valid = [c for c in cols if c in features.columns]
        if not valid:
            return pd.DataFrame(index=features.index)
        return features[valid]

    def to_numpy_matrix(
        self, features: pd.DataFrame, fillna: float = 0.0
    ) -> np.ndarray:
        """Convert features to a clean float32 numpy array for MH-TCN."""
        arr = features.values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=fillna, posinf=fillna, neginf=fillna)
        return arr

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ATR from OHLCV DataFrame."""
        h = df["high"] if "high" in df.columns else df["High"]
        l = df["low"] if "low" in df.columns else df["Low"]
        c = df["close"] if "close" in df.columns else df["Close"]
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
