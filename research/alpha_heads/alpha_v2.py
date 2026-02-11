"""
Alpha V2 — Category-based probabilistic alpha head.

Architecture:
    200+ indicators → grouped into 6 economic categories →
    intra-category weighted evidence → sigmoid → category probability →
    logit-space fusion across categories → calibrated final probability

Key design:
    - Intra-category: z_cat = Σ w_i · f(x_i), P_cat = σ(z_cat)
    - Cross-category: Final_logit = Σ α_k · logit(P_k)
    - α_k are regime-dependent weights
    - Post-aggregation calibration (Platt/isotonic via calibrator)
    - Indicator-count bias mitigation: normalize weights per category
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..interfaces import AlphaHead, AlphaSignal, Direction, RegimeLabel
from ..feature_pipeline import ALL_CATEGORIES, CATEGORY_DEFINITIONS

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -10, 10))
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return np.log(p / (1.0 - p))


# Regime-dependent category weights — rows sum to 1
_REGIME_WEIGHTS: Dict[RegimeLabel, Dict[str, float]] = {
    RegimeLabel.TRENDING: {
        "trend": 0.30, "momentum": 0.25, "oscillator": 0.10,
        "volatility": 0.10, "volume": 0.10, "structure": 0.15,
    },
    RegimeLabel.RANGING: {
        "trend": 0.10, "momentum": 0.15, "oscillator": 0.30,
        "volatility": 0.15, "volume": 0.10, "structure": 0.20,
    },
    RegimeLabel.VOLATILE: {
        "trend": 0.15, "momentum": 0.15, "oscillator": 0.15,
        "volatility": 0.30, "volume": 0.10, "structure": 0.15,
    },
    RegimeLabel.TRANSITION: {
        "trend": 0.17, "momentum": 0.17, "oscillator": 0.17,
        "volatility": 0.17, "volume": 0.16, "structure": 0.16,
    },
}


class CategoryScorer:
    """Scores a single economic category from its feature subset.

    Intra-category compression:
        1. For each feature, compute a directional z-score relative to its
           rolling mean/std (no future data).
        2. Weight by learned/heuristic reliability weights.
        3. Aggregate: z_cat = Σ w_i * z_i  (normalized by Σ|w_i|).
        4. P_cat = sigmoid(z_cat).

    Indicator-count bias mitigation:
        - Weights are normalized so that sum(|w|) = 1.0 per category.
        - Categories with more features don't get louder signals.
    """

    def __init__(self, category: str, lookback: int = 100):
        self.category = category
        self.lookback = lookback
        # Feature weights — initialized uniform, can be learned
        self._weights: Dict[str, float] = {}
        self._initialized = False

    def score(
        self,
        features: pd.DataFrame,
        category_cols: List[str],
    ) -> Tuple[float, float]:
        """Compute category z-score and probability.

        Returns:
            (z_cat, P_cat) where z_cat is signed and P_cat = sigmoid(z_cat).
        """
        valid_cols = [c for c in category_cols if c in features.columns]
        if not valid_cols or len(features) < 20:
            return 0.0, 0.5

        # Initialize uniform weights on first call
        if not self._initialized or set(valid_cols) != set(self._weights.keys()):
            n = len(valid_cols)
            self._weights = {c: 1.0 / n for c in valid_cols}
            self._initialized = True

        z_scores = []
        weights = []

        for col in valid_cols:
            vals = features[col].iloc[-self.lookback:]
            if vals.notna().sum() < 10:
                continue

            current = vals.iloc[-1]
            if np.isnan(current):
                continue

            # Rolling z-score (no future leak — uses only past data)
            mean = vals.iloc[:-1].mean()
            std = vals.iloc[:-1].std()
            if std < 1e-10:
                z = 0.0
            else:
                z = (current - mean) / std

            z = float(np.clip(z, -3.0, 3.0))
            w = self._weights.get(col, 1.0 / len(valid_cols))
            z_scores.append(z)
            weights.append(abs(w))

        if not z_scores:
            return 0.0, 0.5

        # Normalize weights to sum to 1 (mitigate count bias)
        w_arr = np.array(weights)
        w_sum = w_arr.sum()
        if w_sum > 1e-10:
            w_arr = w_arr / w_sum

        z_cat = float(np.dot(z_scores, w_arr))
        p_cat = _sigmoid(z_cat)
        return z_cat, p_cat


class AlphaHeadV2(AlphaHead):
    """Category-based probabilistic alpha with logit-space fusion.

    Pipeline:
        features → CategoryScorer (×6) → P_cat per category →
        logit-space fusion with regime-dependent weights →
        P_final = σ(Σ α_k · logit(P_k)) → direction from sign logic
    """

    def __init__(
        self,
        lookback: int = 100,
        min_probability: float = 0.40,
        directional_edge_min: float = 0.06,
    ):
        self.lookback = lookback
        self.min_probability = min_probability
        self.directional_edge_min = directional_edge_min
        self._scorers: Dict[str, CategoryScorer] = {}
        self._category_columns: Dict[str, List[str]] = {}
        self._eval_count = 0
        # Vectorized pre-computed scores (populated by precompute_all_scores)
        self._precomputed_cat_probs: Optional[Dict[str, np.ndarray]] = None
        self._precomputed_p_bull: Optional[np.ndarray] = None
        self._precomputed_p_bear: Optional[np.ndarray] = None
        self._precomputed_regimes: Optional[pd.Series] = None

        # Create scorers for each category
        for cat in ALL_CATEGORIES:
            self._scorers[cat] = CategoryScorer(cat, lookback=lookback)

    def name(self) -> str:
        return "AlphaV2_CategoryProb"

    def reset(self) -> None:
        self._eval_count = 0
        self._precomputed_cat_probs = None
        self._precomputed_p_bull = None
        self._precomputed_p_bear = None
        self._precomputed_regimes = None
        for scorer in self._scorers.values():
            scorer._initialized = False

    def precompute_all_scores(
        self,
        features: pd.DataFrame,
        regimes: pd.Series,
    ) -> None:
        """Vectorized pre-computation of all category scores for every bar.

        After calling this, evaluate() returns cached results in O(1) per bar
        instead of O(lookback * n_features).
        """
        n = len(features)
        lb = self.lookback
        logger.info(f"AlphaV2: pre-computing scores for {n:,} bars (lookback={lb})...")

        # Per-category z-scores via vectorized rolling
        cat_probs_all: Dict[str, np.ndarray] = {cat: np.full(n, 0.5) for cat in ALL_CATEGORIES}

        for cat in ALL_CATEGORIES:
            cols = self._category_columns.get(cat, [])
            valid_cols = [c for c in cols if c in features.columns]
            if not valid_cols:
                continue

            # Stack all feature columns and compute rolling z-scores
            sub = features[valid_cols].astype(np.float64)
            rolling_mean = sub.rolling(window=lb, min_periods=10).mean().shift(1)
            rolling_std = sub.rolling(window=lb, min_periods=10).std().shift(1)
            rolling_std = rolling_std.replace(0, np.nan)

            z_scores = ((sub - rolling_mean) / rolling_std).clip(-3.0, 3.0)
            z_scores = z_scores.fillna(0.0)

            # Uniform weights, normalized
            n_feats = len(valid_cols)
            z_cat = z_scores.mean(axis=1).values  # mean = sum * (1/n) / (n * 1/n) = mean z
            cat_probs_all[cat] = 1.0 / (1.0 + np.exp(-np.clip(z_cat, -10, 10)))

        # Per-regime logit-space fusion for every bar
        p_bull_all = np.full(n, 0.5)
        p_bear_all = np.full(n, 0.5)

        for regime_label in [RegimeLabel.TRENDING, RegimeLabel.RANGING,
                             RegimeLabel.VOLATILE, RegimeLabel.TRANSITION]:
            mask = (regimes == regime_label).values
            if not mask.any():
                continue
            weights = _REGIME_WEIGHTS.get(regime_label, _REGIME_WEIGHTS[RegimeLabel.TRANSITION])
            logit_sum = np.zeros(n)
            for cat in ALL_CATEGORIES:
                alpha_k = weights.get(cat, 1.0 / len(ALL_CATEGORIES))
                p_cat = np.clip(cat_probs_all[cat], 1e-6, 1.0 - 1e-6)
                logit_sum += alpha_k * np.log(p_cat / (1.0 - p_cat))
            p_bull = 1.0 / (1.0 + np.exp(-np.clip(logit_sum, -10, 10)))
            p_bull_all[mask] = p_bull[mask]
            p_bear_all[mask] = (1.0 - p_bull)[mask]

        self._precomputed_cat_probs = cat_probs_all
        self._precomputed_p_bull = p_bull_all
        self._precomputed_p_bear = p_bear_all
        self._precomputed_regimes = regimes
        logger.info(f"AlphaV2: pre-computation complete")

    def set_category_columns(self, mapping: Dict[str, List[str]]):
        """Inject actual feature→category mapping from FeaturePipeline."""
        self._category_columns = mapping

    def evaluate(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        regime: RegimeLabel,
        bar_idx: int = -1,
    ) -> AlphaSignal:
        self._eval_count += 1

        # ── Fast path: use pre-computed scores ──────────────────────────
        if self._precomputed_p_bull is not None and bar_idx >= 0:
            try:
                return self._evaluate_precomputed(bar_idx, regime, features)
            except Exception:
                pass  # fall through to slow path

        # ── Slow path: per-bar computation ──────────────────────────────
        if len(features) < 30:
            return self._hold(regime)

        try:
            # Stage 1: Compute per-category probabilities
            cat_probs: Dict[str, float] = {}
            cat_z_scores: Dict[str, float] = {}
            regime_weights = _REGIME_WEIGHTS.get(regime, _REGIME_WEIGHTS[RegimeLabel.TRANSITION])

            for cat in ALL_CATEGORIES:
                cols = self._category_columns.get(cat, [])
                if not cols:
                    cat_probs[cat] = 0.5
                    cat_z_scores[cat] = 0.0
                    continue
                z_cat, p_cat = self._scorers[cat].score(features, cols)
                cat_probs[cat] = p_cat
                cat_z_scores[cat] = z_cat

            # Stage 2: Logit-space fusion
            final_logit = 0.0
            for cat in ALL_CATEGORIES:
                alpha_k = regime_weights.get(cat, 1.0 / len(ALL_CATEGORIES))
                final_logit += alpha_k * _logit(cat_probs[cat])

            p_bull = _sigmoid(final_logit)
            p_bear = 1.0 - p_bull

            # Stage 3: Direction from dominant probability
            directional_edge = abs(p_bull - p_bear)
            if directional_edge < self.directional_edge_min:
                return self._hold(regime, max(p_bull, p_bear), cat_probs)

            if p_bull > p_bear:
                direction = Direction.LONG
                probability = p_bull
            else:
                direction = Direction.SHORT
                probability = p_bear

            if probability < self.min_probability:
                return self._hold(regime, probability, cat_probs)

            reasoning = [f"{cat}={cat_probs[cat]:.3f}" for cat in ALL_CATEGORIES]
            reasoning.append(f"logit={final_logit:.3f}")
            reasoning.append(f"edge={directional_edge:.3f}")

            return AlphaSignal(
                direction=direction,
                probability=float(np.clip(probability, 0.0, 1.0)),
                confidence=float(directional_edge),
                regime=regime,
                reasoning=reasoning,
                category_probs=cat_probs,
                feature_vector=features.iloc[-1].values.astype(np.float32),
                timestamp=df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None,
            )

        except Exception as e:
            logger.debug(f"AlphaV2 eval error: {e}")
            return self._hold(regime)

    def _evaluate_precomputed(
        self,
        bar_idx: int,
        regime: RegimeLabel,
        features: pd.DataFrame,
    ) -> AlphaSignal:
        """O(1) evaluation using pre-computed scores."""
        p_bull = float(self._precomputed_p_bull[bar_idx])
        p_bear = float(self._precomputed_p_bear[bar_idx])

        cat_probs = {cat: float(self._precomputed_cat_probs[cat][bar_idx])
                     for cat in ALL_CATEGORIES}

        directional_edge = abs(p_bull - p_bear)
        if directional_edge < self.directional_edge_min:
            return self._hold(regime, max(p_bull, p_bear), cat_probs)

        if p_bull > p_bear:
            direction = Direction.LONG
            probability = p_bull
        else:
            direction = Direction.SHORT
            probability = p_bear

        if probability < self.min_probability:
            return self._hold(regime, probability, cat_probs)

        return AlphaSignal(
            direction=direction,
            probability=float(np.clip(probability, 0.0, 1.0)),
            confidence=float(directional_edge),
            regime=regime,
            reasoning=[f"{cat}={cat_probs[cat]:.3f}" for cat in ALL_CATEGORIES],
            category_probs=cat_probs,
            timestamp=None,
        )

    def _hold(
        self,
        regime: RegimeLabel,
        prob: float = 0.5,
        cat_probs: Optional[Dict[str, float]] = None,
    ) -> AlphaSignal:
        return AlphaSignal(
            direction=Direction.HOLD,
            probability=prob,
            confidence=0.0,
            regime=regime,
            reasoning=["insufficient_signal"],
            category_probs=cat_probs or {},
        )
