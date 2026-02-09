"""
Alpha V1 — Rule-based + statistical alpha head.

Wraps the existing AlphaFactory pipeline:
    Swings → Features → Causality → Decision

Produces direction + base probability with NO ML inside.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..interfaces import AlphaHead, AlphaSignal, Direction, RegimeLabel

logger = logging.getLogger(__name__)


class AlphaHeadV1(AlphaHead):
    """Rule-based alpha using market structure, momentum, and causality.

    Internally uses a simplified version of the existing AlphaFactory
    decision logic without requiring the full orchestrator instantiation.
    This keeps the research framework self-contained.
    """

    def __init__(
        self,
        trend_weight: float = 0.30,
        momentum_weight: float = 0.25,
        sr_weight: float = 0.20,
        causal_weight: float = 0.25,
        min_confidence: float = 0.40,
    ):
        self.trend_weight = trend_weight
        self.momentum_weight = momentum_weight
        self.sr_weight = sr_weight
        self.causal_weight = causal_weight
        self.min_confidence = min_confidence
        self._eval_count = 0

    def name(self) -> str:
        return "AlphaV1_RuleBased"

    def reset(self) -> None:
        self._eval_count = 0

    def evaluate(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        regime: RegimeLabel,
    ) -> AlphaSignal:
        self._eval_count += 1

        if len(features) < 20:
            return self._hold(regime)

        try:
            trend_score = self._trend_score(features)
            momentum_score = self._momentum_score(features)
            sr_score = self._support_resistance_score(df, features)
            causal_score = self._causal_score(features)

            composite = (
                self.trend_weight * trend_score
                + self.momentum_weight * momentum_score
                + self.sr_weight * sr_score
                + self.causal_weight * causal_score
            )

            # Direction from sign of composite
            if composite > 0:
                direction = Direction.LONG
            elif composite < 0:
                direction = Direction.SHORT
            else:
                return self._hold(regime)

            # Probability = |composite| mapped through sigmoid
            raw_prob = 1.0 / (1.0 + np.exp(-3.0 * abs(composite)))
            confidence = abs(composite)

            if raw_prob < self.min_confidence:
                return self._hold(regime, raw_prob)

            reasoning = [
                f"trend={trend_score:.3f}",
                f"mom={momentum_score:.3f}",
                f"sr={sr_score:.3f}",
                f"causal={causal_score:.3f}",
                f"composite={composite:.3f}",
            ]

            return AlphaSignal(
                direction=direction,
                probability=float(np.clip(raw_prob, 0.0, 1.0)),
                confidence=float(confidence),
                regime=regime,
                reasoning=reasoning,
                feature_vector=features.iloc[-1].values.astype(np.float32),
                timestamp=df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None,
            )

        except Exception as e:
            logger.debug(f"AlphaV1 eval error: {e}")
            return self._hold(regime)

    # ------------------------------------------------------------------
    # Component scores — each returns [-1, +1]
    # ------------------------------------------------------------------

    def _trend_score(self, feat: pd.DataFrame) -> float:
        """Score based on moving-average alignment and ADX."""
        score = 0.0
        n = 0

        # EMA alignment
        for short, long in [("ema_8", "ema_21"), ("ema_21", "ema_50"), ("sma_20", "sma_50")]:
            if short in feat.columns and long in feat.columns:
                s = feat[short].iloc[-1]
                l = feat[long].iloc[-1]
                if not (np.isnan(s) or np.isnan(l)):
                    diff = (s - l) / (abs(l) + 1e-10)
                    score += float(np.clip(diff * 100, -1, 1))
                    n += 1

        # ADX contribution (strong trend = higher magnitude)
        for col in ("adx", "adx_14", "ADX"):
            if col in feat.columns:
                adx_val = feat[col].iloc[-1]
                if not np.isnan(adx_val):
                    adx_mult = float(np.clip(adx_val / 50.0, 0.2, 1.0))
                    if n > 0:
                        score *= adx_mult
                    break

        # Trend direction features
        for col in ("trend_short", "trend_medium"):
            if col in feat.columns:
                v = feat[col].iloc[-1]
                if not np.isnan(v):
                    score += float(np.clip(v, -1, 1))
                    n += 1

        return float(np.clip(score / max(n, 1), -1, 1))

    def _momentum_score(self, feat: pd.DataFrame) -> float:
        """Score from RSI, MACD, momentum indicators."""
        score = 0.0
        n = 0

        # RSI — centered around 50
        for col in ("rsi", "rsi_14"):
            if col in feat.columns:
                rsi = feat[col].iloc[-1]
                if not np.isnan(rsi):
                    score += float(np.clip((rsi - 50) / 30, -1, 1))
                    n += 1
                    break

        # MACD histogram sign/magnitude
        for col in ("macd_histogram", "macd_hist"):
            if col in feat.columns:
                mh = feat[col].iloc[-1]
                if not np.isnan(mh):
                    score += float(np.clip(mh * 200, -1, 1))
                    n += 1
                    break

        # Raw momentum
        for col in ("momentum", "momentum_10"):
            if col in feat.columns:
                m = feat[col].iloc[-1]
                if not np.isnan(m):
                    score += float(np.clip(m * 50, -1, 1))
                    n += 1
                    break

        # Stochastic
        for col in ("stoch_k",):
            if col in feat.columns:
                sk = feat[col].iloc[-1]
                if not np.isnan(sk):
                    score += float(np.clip((sk - 50) / 30, -1, 1))
                    n += 1
                    break

        return float(np.clip(score / max(n, 1), -1, 1))

    def _support_resistance_score(self, df: pd.DataFrame, feat: pd.DataFrame) -> float:
        """Score based on price position relative to S/R and Bollinger."""
        close_col = "close" if "close" in df.columns else "Close"
        close = float(df[close_col].iloc[-1])
        score = 0.0
        n = 0

        # Bollinger band position
        for bb_u, bb_l in [("bb_upper", "bb_lower")]:
            if bb_u in feat.columns and bb_l in feat.columns:
                upper = feat[bb_u].iloc[-1]
                lower = feat[bb_l].iloc[-1]
                if not (np.isnan(upper) or np.isnan(lower)) and (upper - lower) > 1e-10:
                    pct = (close - lower) / (upper - lower)
                    score += float(np.clip(-(pct - 0.5) * 2, -1, 1))  # mean-reversion bias
                    n += 1

        # Pivot points
        for col in ("pivot_point",):
            if col in feat.columns:
                pp = feat[col].iloc[-1]
                if not np.isnan(pp) and pp > 0:
                    diff = (close - pp) / pp
                    score += float(np.clip(diff * 50, -1, 1))
                    n += 1

        return float(np.clip(score / max(n, 1), -1, 1))

    def _causal_score(self, feat: pd.DataFrame) -> float:
        """Simplified causal score using lagged feature correlations.

        In a full system this would use Granger/transfer-entropy; here we
        approximate with lagged momentum–price correlation.
        """
        score = 0.0
        n = 0
        for col in ("momentum", "rsi", "macd_histogram"):
            if col in feat.columns and len(feat) >= 5:
                vals = feat[col].iloc[-5:]
                if vals.notna().sum() >= 4:
                    # Trend of the indicator over last 5 bars
                    diffs = vals.diff().dropna()
                    trend = diffs.mean()
                    score += float(np.clip(trend * 100, -1, 1))
                    n += 1
        return float(np.clip(score / max(n, 1), -1, 1))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _hold(self, regime: RegimeLabel, prob: float = 0.5) -> AlphaSignal:
        return AlphaSignal(
            direction=Direction.HOLD,
            probability=prob,
            confidence=0.0,
            regime=regime,
            reasoning=["insufficient_signal"],
        )
