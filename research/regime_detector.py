"""
Shared regime detector for all variants.

Uses volatility-based regime classification with optional HMM.
All variants see the same regime label — this is NOT a differentiator.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .interfaces import RegimeLabel

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Volatility-based regime detector.

    Classification logic:
        - TRENDING:   ADX > adx_trend_thresh AND directional bias
        - RANGING:    ADX < adx_range_thresh AND low vol
        - VOLATILE:   ATR_ratio > vol_spike_thresh
        - TRANSITION: otherwise (ambiguous)
    """

    def __init__(
        self,
        adx_trend_thresh: float = 25.0,
        adx_range_thresh: float = 20.0,
        vol_spike_thresh: float = 1.5,
        lookback: int = 20,
    ):
        self.adx_trend_thresh = adx_trend_thresh
        self.adx_range_thresh = adx_range_thresh
        self.vol_spike_thresh = vol_spike_thresh
        self.lookback = lookback

    def detect(
        self,
        df: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> RegimeLabel:
        """Classify the current regime from the latest bar.

        Args:
            df: OHLCV data (at least `lookback` bars).
            features: Optional pre-computed features with 'adx', 'atr', etc.

        Returns:
            RegimeLabel for the current bar.
        """
        try:
            adx = self._get_adx(df, features)
            atr_ratio = self._get_atr_ratio(df, features)

            if atr_ratio > self.vol_spike_thresh:
                return RegimeLabel.VOLATILE
            if adx > self.adx_trend_thresh:
                return RegimeLabel.TRENDING
            if adx < self.adx_range_thresh:
                return RegimeLabel.RANGING
            return RegimeLabel.TRANSITION
        except Exception:
            return RegimeLabel.TRANSITION

    def detect_series(
        self,
        df: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Classify regime for every bar in the DataFrame."""
        regimes = []
        n = len(df)
        for i in range(n):
            end = i + 1
            start = max(0, end - max(self.lookback * 3, 60))
            sub_df = df.iloc[start:end]
            sub_feat = features.iloc[start:end] if features is not None else None
            regimes.append(self.detect(sub_df, sub_feat))
        return pd.Series(regimes, index=df.index)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _get_adx(self, df: pd.DataFrame, features: Optional[pd.DataFrame]) -> float:
        if features is not None:
            for col in ("adx", "ADX", "adx_14"):
                if col in features.columns:
                    val = features[col].iloc[-1]
                    if not np.isnan(val):
                        return float(val)
        return self._compute_adx(df)

    def _get_atr_ratio(self, df: pd.DataFrame, features: Optional[pd.DataFrame]) -> float:
        if features is not None:
            for col in ("atr_ratio", "ATR_ratio", "volatility_ratio"):
                if col in features.columns:
                    val = features[col].iloc[-1]
                    if not np.isnan(val):
                        return float(val)
        return self._compute_atr_ratio(df)

    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Minimal ADX computation."""
        h = df["high"] if "high" in df.columns else df["High"]
        l = df["low"] if "low" in df.columns else df["Low"]
        c = df["close"] if "close" in df.columns else df["Close"]
        if len(h) < period + 1:
            return 20.0  # neutral default

        plus_dm = h.diff().clip(lower=0)
        minus_dm = (-l.diff()).clip(lower=0)
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
        adx = dx.rolling(period).mean()
        val = adx.iloc[-1]
        return float(val) if not np.isnan(val) else 20.0

    def _compute_atr_ratio(self, df: pd.DataFrame) -> float:
        """ATR_short / ATR_long as a volatility spike measure."""
        h = df["high"] if "high" in df.columns else df["High"]
        l = df["low"] if "low" in df.columns else df["Low"]
        c = df["close"] if "close" in df.columns else df["Close"]
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        short = tr.rolling(5).mean().iloc[-1]
        long_ = tr.rolling(20).mean().iloc[-1]
        if np.isnan(short) or np.isnan(long_) or long_ < 1e-10:
            return 1.0
        return float(short / long_)
