# alpha_factory/probabilistic_alpha_factory.py
"""
Probabilistic Alpha Factory v2
==============================

A fully probabilistic trading decision system that:
1. Computes soft regime priors from swing analysis
2. Generates calibrated feature-level probabilities
3. Applies continuous causal confidence weighting
4. Uses MH-TCN for temporal refinement of probability sequences
5. Aggregates evidence via Noisy-OR / Bayesian methods
6. Applies dynamic thresholding with alpha decay

Architecture:
    Swings → Regime Prior → Feature Probs → Causal Adjustment
                                    ↓
                            MH-TCN Temporal Refinement
                                    ↓
                            Evidence Aggregation (Noisy-OR)
                                    ↓
                            Dynamic Threshold → Decision

Author: pyForex Team
"""

import logging
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ProbabilisticConfig:
    """Configuration for the Probabilistic Alpha Factory."""
    
    # Regime Prior
    regime_scale_factor: float = 2.0  # Scaling for vol-adjusted slope
    entropy_weight: float = 0.3  # Weight for volatility entropy in regime uncertainty
    
    # Feature Probability Calibration
    calibration_method: str = "logistic"  # "logistic", "kde", "quantile"
    calibration_lookback: int = 500  # Bars for calibration statistics

    # Feature Inclusion (no silent selection)
    key_features_only: bool = False
    key_features: Optional[List[str]] = None
    
    # Historical Reliability
    hit_rate_floor: float = 0.4  # Minimum hit rate to consider feature
    specificity_floor: float = 0.3  # Minimum regime specificity
    decay_halflife_days: float = 30.0  # Half-life for feature decay
    
    # Causal Confidence
    causal_neutral_floor: float = 0.4  # Neutral causal confidence
    causal_ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "granger": 0.3,
        "transfer_entropy": 0.4,
        "correlation": 0.3
    })
    
    # MH-TCN Integration
    mhtcn_weight: float = 0.4  # Weight of MH-TCN in final aggregation
    mhtcn_sequence_length: int = 20  # Probability sequence length for MH-TCN
    
    # Evidence Aggregation
    aggregation_method: str = "noisy_or"  # "noisy_or", "bayesian", "weighted_avg"
    evidence_threshold: float = 0.1  # Minimum evidence to include in aggregation
    evidence_threshold_enabled: bool = False
    
    # Dynamic Thresholding
    base_threshold: float = 0.35
    vol_factor: float = 1.5  # Threshold increase per unit volatility
    drawdown_penalty: float = 0.05  # Threshold increase per 10% drawdown
    
    # Alpha Decay
    alpha_decay_enabled: bool = True
    alpha_decay_rate: float = 0.02  # Per-bar decay rate
    alpha_min_confidence: float = 0.3  # Floor for decayed confidence
    
    # Stability
    stability_lookback: int = 20  # Bars for stability calculation
    stability_weight: float = 0.2  # Weight in final probability
    
    # Paths
    metadata_path: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata_path is None:
            self.metadata_path = str(Path(__file__).parent / "metadata" / "feature_stats.json")

        if self.key_features is None:
            self.key_features = [
                "rsi", "macd", "macd_histogram", "adx",
                "atr_ratio", "momentum", "trend_strength"
            ]


class RegimeType(Enum):
    """Soft regime types."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"


@dataclass
class RegimeProbabilities:
    """Container for regime probabilities."""
    p_bull: float
    p_bear: float
    p_neutral: float
    p_volatile: float = 0.0
    
    def __post_init__(self):
        self._normalize()
    
    def _normalize(self):
        """Ensure probabilities sum to 1."""
        total = self.p_bull + self.p_bear + self.p_neutral + self.p_volatile
        if total > 0:
            self.p_bull /= total
            self.p_bear /= total
            self.p_neutral /= total
            self.p_volatile /= total
        else:
            self.p_bull = 0.25
            self.p_bear = 0.25
            self.p_neutral = 0.25
            self.p_volatile = 0.25
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "bull": self.p_bull,
            "bear": self.p_bear,
            "neutral": self.p_neutral,
            "volatile": self.p_volatile
        }
    
    @property
    def dominant_regime(self) -> str:
        """Get the most probable regime."""
        probs = self.to_dict()
        return max(probs, key=probs.get)
    
    @property
    def entropy(self) -> float:
        """Calculate entropy of regime distribution (uncertainty measure)."""
        probs = [self.p_bull, self.p_bear, self.p_neutral, self.p_volatile]
        probs = [p for p in probs if p > 0]
        if not probs:
            return 0.0
        return float(-np.sum([p * np.log(p + 1e-10) for p in probs]))


@dataclass
class FeatureProbability:
    """Probability output for a single feature."""
    feature_name: str
    p_bull: float
    p_bear: float
    p_neutral: float
    raw_value: float
    hit_rate: float = 1.0
    specificity: float = 1.0
    causal_confidence: float = 0.5
    decay_factor: float = 1.0
    
    @property
    def reliability_weight(self) -> float:
        """Combined reliability weight for this feature.
        
        Uses a blended score instead of pure multiplication which would
        collapse to ~0 when any factor (especially specificity) is near-zero.
        """
        # Weighted blend of reliability factors
        w = (0.4 * self.hit_rate
             + 0.2 * self.specificity
             + 0.3 * self.causal_confidence
             + 0.1 * self.decay_factor)
        return float(np.clip(w, 0.05, 1.0))
    
    @property
    def weighted_p_bull(self) -> float:
        """Probability weighted by reliability."""
        return self.p_bull * self.reliability_weight
    
    @property
    def weighted_p_bear(self) -> float:
        """Probability weighted by reliability."""
        return self.p_bear * self.reliability_weight


@dataclass
class DecisionOutput:
    """Final decision output from the probabilistic system."""
    direction: str  # "LONG", "SHORT", "HOLD"
    confidence: float
    regime_probs: RegimeProbabilities
    final_p_bull: float
    final_p_bear: float
    threshold_used: float
    size_multiplier: float
    reasoning: List[str]
    feature_contributions: Dict[str, float]
    mhtcn_contribution: Optional[Dict[str, float]] = None
    mhtcn_prediction: Optional[Any] = None  # Full prediction object for efficiency
    stability_score: float = 1.0
    alpha_decay_applied: float = 1.0


# =============================================================================
# Metadata Store
# =============================================================================

class FeatureMetadataStore:
    """
    Persistent store for feature statistics and reliability metrics.
    
    Stores per-feature:
    - Historical quantiles per regime
    - Hit rates from walk-forward backtests
    - Regime specificity (KL divergence)
    - Last update timestamp for decay calculation
    """
    
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else Path(__file__).parent / "metadata" / "feature_stats.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Dict] = {}
        self._load()
    
    def _load(self):
        """Load metadata from disk."""
        if self.path.exists():
            try:
                with open(self.path, 'r') as f:
                    self._data = json.load(f)
                logger.info(f"Loaded feature metadata: {len(self._data)} features")
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
                self._data = {}
        else:
            self._data = {}
    
    def save(self):
        """Save metadata to disk."""
        try:
            with open(self.path, 'w') as f:
                json.dump(self._data, f, indent=2, default=str)
            logger.info(f"Saved feature metadata: {len(self._data)} features")
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def get_feature_stats(self, feature_name: str) -> Optional[Dict]:
        """Get statistics for a feature."""
        return self._data.get(feature_name)
    
    def update_feature_stats(
        self,
        feature_name: str,
        quantiles_bull: Dict[str, float],
        quantiles_bear: Dict[str, float],
        quantiles_neutral: Dict[str, float],
        hit_rate: float,
        specificity: float,
        sharpe_contribution: float = 0.0
    ):
        """Update statistics for a feature."""
        self._data[feature_name] = {
            "quantiles_bull": quantiles_bull,
            "quantiles_bear": quantiles_bear,
            "quantiles_neutral": quantiles_neutral,
            "hit_rate": float(hit_rate),
            "specificity": float(specificity),
            "sharpe_contribution": float(sharpe_contribution),
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def get_decay_factor(self, feature_name: str, halflife_days: float = 30.0) -> float:
        """Calculate decay factor based on time since last update."""
        stats = self._data.get(feature_name)
        if not stats or "last_updated" not in stats:
            return 1.0
        
        try:
            last_updated = datetime.fromisoformat(stats["last_updated"])
            days_since = (datetime.utcnow() - last_updated).days
            decay = np.exp(-np.log(2) * days_since / halflife_days)
            return float(np.clip(decay, 0.1, 1.0))
        except Exception:
            return 1.0
    
    def get_all_features(self) -> List[str]:
        """Get list of all tracked features."""
        return list(self._data.keys())


# =============================================================================
# Stage 1: Regime Prior from Swings
# =============================================================================

class RegimePriorCalculator:
    """
    Calculates soft regime probabilities from swing point analysis.
    
    Uses:
    - Volatility-adjusted slope of recent swings
    - Volatility regime entropy for uncertainty
    - HMM-inspired state transitions
    """
    
    def __init__(self, config: ProbabilisticConfig):
        self.config = config
    
    def calculate(
        self,
        df: pd.DataFrame,
        swing_points: Optional[List] = None
    ) -> RegimeProbabilities:
        """
        Calculate regime prior probabilities.
        
        Args:
            df: OHLCV DataFrame with at least 'close', 'high', 'low'
            swing_points: Optional list of swing point objects
        
        Returns:
            RegimeProbabilities with soft regime labels
        """
        if df is None or len(df) < 20:
            return RegimeProbabilities(0.25, 0.25, 0.25, 0.25)
        
        # Calculate volatility metrics
        returns = df['close'].pct_change().dropna()
        current_vol = float(returns.tail(20).std()) if len(returns) >= 20 else 0.01
        rolling_vol = float(returns.rolling(50).std().iloc[-1]) if len(returns) >= 50 else current_vol
        
        # Volatility regime entropy (uncertainty from vol clustering)
        vol_ratio = current_vol / (rolling_vol + 1e-10)
        vol_entropy = float(np.clip(abs(vol_ratio - 1.0), 0, 1))
        
        # Calculate trend strength from price action
        if swing_points and len(swing_points) >= 2:
            # Use swing points for slope calculation
            slope, duration = self._swing_slope(swing_points, df)
        else:
            # Fallback to simple price change
            lookback = min(20, len(df) - 1)
            price_change = df['close'].iloc[-1] - df['close'].iloc[-lookback]
            slope = price_change / (df['close'].iloc[-lookback] * lookback + 1e-10)
            duration = lookback
        
        # Volatility-adjusted slope
        vol_adj_slope = slope / (current_vol + 1e-10)
        
        # Convert to probabilities using tanh (bounded [-1, 1])
        strength = float(np.tanh(vol_adj_slope * self.config.regime_scale_factor))
        
        # Calculate raw regime probabilities
        # Positive strength → bullish, negative → bearish
        p_bull_raw = float(np.clip((strength + 1) / 2, 0, 1))
        p_bear_raw = float(np.clip((1 - strength) / 2, 0, 1))
        
        # Neutral probability increases when strength is low
        p_neutral_raw = float(np.clip(1 - abs(strength), 0, 1))
        
        # Volatile probability from volatility spike
        vol_z = (current_vol - rolling_vol) / (rolling_vol + 1e-10)
        p_volatile = float(self._sigmoid((vol_z - 1.5) * 2.0))
        
        # Apply entropy weighting (reduce confidence when uncertain)
        entropy_factor = 1.0 - self.config.entropy_weight * vol_entropy
        
        p_bull = p_bull_raw * entropy_factor * (1 - p_volatile)
        p_bear = p_bear_raw * entropy_factor * (1 - p_volatile)
        p_neutral = p_neutral_raw * (1 - p_volatile) + vol_entropy * 0.2
        
        return RegimeProbabilities(
            p_bull=p_bull,
            p_bear=p_bear,
            p_neutral=p_neutral,
            p_volatile=p_volatile
        )
    
    def _swing_slope(self, swing_points: List, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate slope from swing points."""
        try:
            # Sort by index
            swings = sorted(swing_points, key=lambda s: getattr(s, 'index', 0))
            if len(swings) < 2:
                return 0.0, 1.0
            
            a = swings[-2]
            b = swings[-1]
            
            price_change = float(getattr(b, 'price', 0) - getattr(a, 'price', 0))
            duration = float(max(1, getattr(b, 'index', 0) - getattr(a, 'index', 0)))
            
            base_price = df['close'].iloc[-1] if len(df) > 0 else 1.0
            slope = price_change / (base_price * duration + 1e-10)
            
            return slope, duration
        except Exception:
            return 0.0, 1.0
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid function."""
        try:
            return float(1.0 / (1.0 + np.exp(-x)))
        except Exception:
            return 0.5


# =============================================================================
# Stage 2: Feature-Level Probability Calibration
# =============================================================================

class FeatureProbabilityCalibrator:
    """
    Generates calibrated probabilities for each feature.
    
    Methods:
    - Logistic: Sigmoid calibration based on regime-specific quantiles
    - KDE: Kernel density estimation for smooth probabilities
    - Quantile: Direct quantile-based probability mapping
    """
    
    def __init__(self, config: ProbabilisticConfig, metadata_store: FeatureMetadataStore):
        self.config = config
        self.metadata = metadata_store
        
        # Default feature configurations (can be overridden by metadata)
        self._default_configs = self._get_default_feature_configs()
    
    def _get_default_feature_configs(self) -> Dict[str, Dict]:
        """Default calibration parameters for common features."""
        return {
            "rsi": {
                "bull_zone": (50, 70),  # RSI above 50 supports bullish trend
                "bear_zone": (30, 50),  # RSI below 50 supports bearish trend
                "scale": 20.0,
                "type": "oscillator"
            },
            "macd": {
                "bull_zone": (0, float('inf')),
                "bear_zone": (float('-inf'), 0),
                "scale": 0.001,
                "type": "momentum"
            },
            "macd_histogram": {
                "bull_zone": (0, float('inf')),
                "bear_zone": (float('-inf'), 0),
                "scale": 0.0005,
                "type": "momentum"
            },
            "adx": {
                "trend_threshold": 25,
                "scale": 10.0,
                "type": "trend_strength"
            },
            "bb_position": {
                "bull_zone": (0.2, 0.8),
                "bear_zone": (0.2, 0.8),
                "scale": 0.3,
                "type": "mean_reversion"
            },
            "atr_ratio": {
                "high_vol_threshold": 1.5,
                "scale": 0.5,
                "type": "volatility"
            }
        }
    
    def calibrate_feature(
        self,
        feature_name: str,
        value: float,
        regime_context: Optional[str] = None
    ) -> FeatureProbability:
        """
        Generate calibrated probabilities for a feature value.
        
        Args:
            feature_name: Name of the feature
            value: Current feature value
            regime_context: Optional regime context for conditional calibration
        
        Returns:
            FeatureProbability with calibrated probabilities
        """
        # Get feature statistics from metadata
        stats = self.metadata.get_feature_stats(feature_name)
        default_config = self._default_configs.get(feature_name.lower(), {})
        
        # Calculate probabilities based on method
        if self.config.calibration_method == "logistic":
            p_bull, p_bear, p_neutral = self._logistic_calibration(
                value, feature_name, stats, default_config, regime_context
            )
        elif self.config.calibration_method == "kde":
            p_bull, p_bear, p_neutral = self._kde_calibration(
                value, feature_name, stats, regime_context
            )
        else:  # quantile
            p_bull, p_bear, p_neutral = self._quantile_calibration(
                value, feature_name, stats, regime_context
            )
        
        # Get reliability metrics
        hit_rate = stats.get("hit_rate", 1.0) if stats else 1.0
        specificity = stats.get("specificity", 1.0) if stats else 1.0
        decay_factor = self.metadata.get_decay_factor(
            feature_name, self.config.decay_halflife_days
        )
        
        # Apply floors
        hit_rate = max(hit_rate, self.config.hit_rate_floor)
        specificity = max(specificity, self.config.specificity_floor)
        
        return FeatureProbability(
            feature_name=feature_name,
            p_bull=p_bull,
            p_bear=p_bear,
            p_neutral=p_neutral,
            raw_value=value,
            hit_rate=hit_rate,
            specificity=specificity,
            decay_factor=decay_factor
        )
    
    def _logistic_calibration(
        self,
        value: float,
        feature_name: str,
        stats: Optional[Dict],
        default_config: Dict,
        regime_context: Optional[str]
    ) -> Tuple[float, float, float]:
        """Logistic/sigmoid calibration."""
        feature_type = default_config.get("type", "generic")
        scale = default_config.get("scale", 1.0)
        
        if feature_type == "oscillator":
            # RSI-like: trend-following interpretation
            # High values (>70) = strong bullish momentum
            # Low values (<30) = strong bearish momentum
            # 40-60 = neutral/indeterminate zone
            bull_zone = default_config.get("bull_zone", (50, 70))
            bear_zone = default_config.get("bear_zone", (30, 50))
            
            midpoint = 50.0
            if value > bull_zone[1]:  # Strong bullish (e.g. RSI > 70)
                p_bull = self._sigmoid((value - bull_zone[1]) / scale)
                p_bear = 0.1
            elif value < bear_zone[0]:  # Strong bearish (e.g. RSI < 30)
                p_bear = self._sigmoid((bear_zone[0] - value) / scale)
                p_bull = 0.1
            else:
                # Interpolate: above midpoint favors bull, below favors bear
                normalized = (value - midpoint) / (scale + 1e-10)
                p_bull = self._sigmoid(normalized)
                p_bear = self._sigmoid(-normalized)
            
            p_neutral = float(np.clip(1.0 - abs(p_bull - p_bear), 0.1, 0.5))
            
        elif feature_type == "momentum":
            # MACD-like: positive = bullish, negative = bearish
            normalized = value / (scale + 1e-10)
            p_bull = self._sigmoid(normalized)
            p_bear = self._sigmoid(-normalized)
            p_neutral = 1.0 - abs(p_bull - p_bear)
            
        elif feature_type == "trend_strength":
            # ADX-like: high = trending, low = neutral
            threshold = default_config.get("trend_threshold", 25)
            trend_strength = self._sigmoid((value - threshold) / scale)
            p_neutral = 1.0 - trend_strength
            # Direction comes from other features
            p_bull = trend_strength * 0.5
            p_bear = trend_strength * 0.5
            
        else:
            # Generic: use quantiles if available
            if stats and "quantiles_bull" in stats:
                return self._quantile_calibration(value, feature_name, stats, regime_context)
            else:
                # Default: neutral
                p_bull = 0.33
                p_bear = 0.33
                p_neutral = 0.34
        
        # Normalize
        total = p_bull + p_bear + p_neutral
        if total > 0:
            p_bull /= total
            p_bear /= total
            p_neutral /= total
        
        return float(np.clip(p_bull, 0, 1)), float(np.clip(p_bear, 0, 1)), float(np.clip(p_neutral, 0, 1))
    
    def _kde_calibration(
        self,
        value: float,
        feature_name: str,
        stats: Optional[Dict],
        regime_context: Optional[str]
    ) -> Tuple[float, float, float]:
        """Kernel density estimation calibration."""
        # Simplified KDE using stored quantiles
        if not stats:
            return 0.33, 0.33, 0.34
        
        # Use quantiles to estimate density
        q_bull = stats.get("quantiles_bull", {})
        q_bear = stats.get("quantiles_bear", {})
        q_neutral = stats.get("quantiles_neutral", {})
        
        # Calculate likelihood under each regime
        p_bull = self._quantile_likelihood(value, q_bull)
        p_bear = self._quantile_likelihood(value, q_bear)
        p_neutral = self._quantile_likelihood(value, q_neutral)
        
        # Normalize
        total = p_bull + p_bear + p_neutral
        if total > 0:
            return p_bull / total, p_bear / total, p_neutral / total
        return 0.33, 0.33, 0.34
    
    def _quantile_calibration(
        self,
        value: float,
        feature_name: str,
        stats: Optional[Dict],
        regime_context: Optional[str]
    ) -> Tuple[float, float, float]:
        """Direct quantile-based calibration."""
        if not stats:
            return 0.33, 0.33, 0.34
        
        q_bull = stats.get("quantiles_bull", {})
        q_bear = stats.get("quantiles_bear", {})
        q_neutral = stats.get("quantiles_neutral", {})
        
        # Position in each regime's distribution
        pos_bull = self._quantile_position(value, q_bull)
        pos_bear = self._quantile_position(value, q_bear)
        pos_neutral = self._quantile_position(value, q_neutral)
        
        # Convert position to probability (closer to median = higher prob)
        p_bull = 1.0 - abs(pos_bull - 0.5) * 2
        p_bear = 1.0 - abs(pos_bear - 0.5) * 2
        p_neutral = 1.0 - abs(pos_neutral - 0.5) * 2
        
        # Normalize
        total = p_bull + p_bear + p_neutral
        if total > 0:
            return p_bull / total, p_bear / total, p_neutral / total
        return 0.33, 0.33, 0.34
    
    def _quantile_likelihood(self, value: float, quantiles: Dict) -> float:
        """Estimate likelihood from quantiles using Gaussian kernel."""
        if not quantiles:
            return 0.33
        
        q50 = quantiles.get("q50", quantiles.get("0.5", 0))
        q25 = quantiles.get("q25", quantiles.get("0.25", q50 - 1))
        q75 = quantiles.get("q75", quantiles.get("0.75", q50 + 1))
        
        # Estimate std from IQR
        iqr = q75 - q25
        std = iqr / 1.35 if iqr > 0 else 1.0
        
        # Gaussian likelihood
        z = (value - q50) / (std + 1e-10)
        likelihood = float(np.exp(-0.5 * z ** 2))
        
        return likelihood
    
    def _quantile_position(self, value: float, quantiles: Dict) -> float:
        """Get position in quantile distribution (0 to 1)."""
        if not quantiles:
            return 0.5
        
        # Get quantile values
        qs = []
        for key in ["q10", "q25", "q50", "q75", "q90", "0.1", "0.25", "0.5", "0.75", "0.9"]:
            if key in quantiles:
                qs.append((float(key.replace("q", "").replace("0.", "")) / 100 if "q" in key else float(key), 
                          quantiles[key]))
        
        if not qs:
            return 0.5
        
        qs.sort(key=lambda x: x[1])
        
        # Find position
        for i, (q_level, q_val) in enumerate(qs):
            if value <= q_val:
                if i == 0:
                    return q_level
                prev_level, prev_val = qs[i - 1]
                # Interpolate
                frac = (value - prev_val) / (q_val - prev_val + 1e-10)
                return prev_level + frac * (q_level - prev_level)
        
        return qs[-1][0]  # Above all quantiles
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        try:
            return float(1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))
        except Exception:
            return 0.5


# =============================================================================
# Stage 3: Causal Confidence Calculator
# =============================================================================

class CausalConfidenceCalculator:
    """
    Calculates continuous causal confidence for features.
    
    Uses ensemble of methods:
    - Granger causality (linear)
    - Transfer entropy (nonlinear)
    - Correlation analysis (baseline)
    """
    
    def __init__(self, config: ProbabilisticConfig):
        self.config = config
        self._cached_results: Dict[str, float] = {}
    
    def calculate(
        self,
        feature_name: str,
        causality_results: Optional[Dict] = None
    ) -> float:
        """
        Calculate causal confidence for a feature.
        
        Args:
            feature_name: Name of the feature
            causality_results: Pre-computed causality analysis results
        
        Returns:
            Causal confidence in [0, 1]
        """
        if causality_results is None:
            return self.config.causal_neutral_floor
        
        # Check cache
        cache_key = f"{feature_name}_{hash(str(causality_results))}"
        if cache_key in self._cached_results:
            return self._cached_results[cache_key]
        
        # Extract scores from different methods
        scores = []
        weights = self.config.causal_ensemble_weights
        
        # Granger causality
        granger_score = self._extract_granger_score(feature_name, causality_results)
        if granger_score is not None:
            scores.append((granger_score, weights.get("granger", 0.3)))
        
        # Transfer entropy
        te_score = self._extract_transfer_entropy(feature_name, causality_results)
        if te_score is not None:
            scores.append((te_score, weights.get("transfer_entropy", 0.4)))
        
        # Correlation
        corr_score = self._extract_correlation(feature_name, causality_results)
        if corr_score is not None:
            scores.append((corr_score, weights.get("correlation", 0.3)))
        
        # Weighted average
        if scores:
            total_weight = sum(w for _, w in scores)
            confidence = sum(s * w for s, w in scores) / (total_weight + 1e-10)
        else:
            confidence = self.config.causal_neutral_floor
        
        # Apply floor to avoid zeroing useful features
        confidence = max(confidence, self.config.causal_neutral_floor)
        confidence = float(np.clip(confidence, 0, 1))
        
        self._cached_results[cache_key] = confidence
        return confidence
    
    def _extract_granger_score(self, feature_name: str, results: Dict) -> Optional[float]:
        """Extract Granger causality score."""
        try:
            ranking = results.get("causal_ranking", {})
            if feature_name in ranking:
                data = ranking[feature_name]
                if isinstance(data, dict):
                    # Combine p-value and F-statistic
                    p_val = data.get("p_value", 0.5)
                    f_stat = data.get("f_statistic", 0)
                    
                    # Convert to score: low p-value + high F = high score
                    score = (1 - p_val) * min(f_stat / 10, 1.0)
                    return float(np.clip(score, 0, 1))
        except Exception:
            pass
        return None
    
    def _extract_transfer_entropy(self, feature_name: str, results: Dict) -> Optional[float]:
        """Extract transfer entropy score."""
        try:
            te_results = results.get("transfer_entropy", {})
            if feature_name in te_results:
                te_value = te_results[feature_name]
                if isinstance(te_value, (int, float)):
                    # Normalize TE (typical range 0-0.5)
                    return float(np.clip(te_value * 2, 0, 1))
        except Exception:
            pass
        return None
    
    def _extract_correlation(self, feature_name: str, results: Dict) -> Optional[float]:
        """Extract correlation-based score."""
        try:
            corr_results = results.get("correlations", results.get("feature_correlations", {}))
            if feature_name in corr_results:
                corr = abs(corr_results[feature_name])
                return float(np.clip(corr, 0, 1))
        except Exception:
            pass
        return None
    
    def clear_cache(self):
        """Clear cached results."""
        self._cached_results.clear()


# =============================================================================
# Stage 4: Evidence Aggregation
# =============================================================================

class EvidenceAggregator:
    """
    Aggregates feature probabilities into final regime probabilities.
    
    Methods:
    - Noisy-OR: P(any evidence) = 1 - ∏(1 - P_i)
    - Bayesian: Prior × Likelihood ratios
    - Weighted Average: Simple weighted combination
    """
    
    def __init__(self, config: ProbabilisticConfig):
        self.config = config
    
    def aggregate(
        self,
        regime_prior: RegimeProbabilities,
        feature_probs: List[FeatureProbability],
        stability_score: float = 1.0,
        mhtcn_probs: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """
        Aggregate all evidence into final probabilities.
        
        Args:
            regime_prior: Prior regime probabilities from swings
            feature_probs: List of calibrated feature probabilities
            stability_score: Market stability score
            mhtcn_probs: Optional MH-TCN probability outputs
        
        Returns:
            (p_bull_final, p_bear_final, p_neutral_final, contributions_dict)
        """
        method = str(getattr(self.config, 'aggregation_method', 'weighted_avg')).lower()
        
        if method == "noisy_or":
            p_bull, p_bear, p_neutral = self._noisy_or_aggregation(feature_probs)
        elif method == "bayesian":
            p_bull, p_bear, p_neutral = self._bayesian_aggregation(regime_prior, feature_probs)
        else:
            p_bull, p_bear, p_neutral = self._weighted_avg_aggregation(feature_probs)
        
        # Incorporate MH-TCN if available
        contributions = {}
        if mhtcn_probs is not None:
            mhtcn_weight = float(getattr(self.config, 'mhtcn_weight', 0.4))
            
            # Detect degenerate MH-TCN output: if one class dominates (>0.90),
            # the model has no useful directional info (softmax saturation).
            # Scale down MH-TCN weight proportionally to avoid killing classical signals.
            max_mhtcn_prob = max(mhtcn_probs.get("bull", 0), mhtcn_probs.get("bear", 0), mhtcn_probs.get("neutral", 0))
            if max_mhtcn_prob > 0.90:
                # Linearly reduce weight: at 0.90 keep full weight, at 1.0 use 0
                degenerate_factor = max(0.0, (1.0 - max_mhtcn_prob) / 0.10)
                mhtcn_weight *= degenerate_factor
                logger.debug(f"MH-TCN degenerate: max_prob={max_mhtcn_prob:.3f}, effective_weight={mhtcn_weight:.3f}")
            
            classical_weight = 1.0 - mhtcn_weight
            
            p_bull = classical_weight * p_bull + mhtcn_weight * mhtcn_probs.get("bull", 0.33)
            p_bear = classical_weight * p_bear + mhtcn_weight * mhtcn_probs.get("bear", 0.33)
            p_neutral = classical_weight * p_neutral + mhtcn_weight * mhtcn_probs.get("neutral", 0.34)
            
            contributions["mhtcn"] = mhtcn_weight
        
        # Apply regime prior as soft directional bias (additive blend, NOT multiplicative)
        # Multiplicative regime_prior * evidence compresses everything to ~0.06 then
        # normalisation produces ~0.33/0.33/0.33 regardless of evidence – defeating
        # the entire pipeline.  Instead, blend a small fraction of the prior to nudge
        # the aggregated evidence toward the dominant regime.
        prior_blend = 0.10  # 10 % prior influence
        p_bull  = (1.0 - prior_blend) * p_bull  + prior_blend * regime_prior.p_bull
        p_bear  = (1.0 - prior_blend) * p_bear  + prior_blend * regime_prior.p_bear
        p_neutral = (1.0 - prior_blend) * p_neutral + prior_blend * regime_prior.p_neutral
        
        # Stability score: used only as a confidence quality metric.
        # We do NOT penalise directional probabilities in volatile markets –
        # for scalping, volatility is opportunity.  Stability is still tracked
        # and reported so the gate logic can use it, but it no longer distorts
        # the probability distribution here.
        
        # Normalize
        total = p_bull + p_bear + p_neutral
        if total > 0:
            p_bull /= total
            p_bear /= total
            p_neutral /= total
        else:
            p_bull, p_bear, p_neutral = 0.33, 0.33, 0.34
        
        # Track feature contributions
        for fp in feature_probs:
            contributions[fp.feature_name] = fp.weighted_p_bull - fp.weighted_p_bear
        
        return (
            float(np.clip(p_bull, 0, 1)),
            float(np.clip(p_bear, 0, 1)),
            float(np.clip(p_neutral, 0, 1)),
            contributions
        )
    
    def _noisy_or_aggregation(
        self,
        feature_probs: List[FeatureProbability]
    ) -> Tuple[float, float, float]:
        """Noisy-OR aggregation with anti-saturation measures.
        
        Fixes: When N correlated features all produce ~0.33, classic Noisy-OR
        gives 1-(1-0.33)^N ≈ 0.97 for BOTH bull AND bear, making the result
        random. We fix this by:
        1. Using only the top-K most directional features (avoids correlated flooding)
        2. Dampening each probability contribution (sqrt) to reduce saturation
        """
        max_features = 4  # Limit to top-K to avoid independence violation
        
        # Sort features by directional strength (max of bull, bear prob)
        ranked = sorted(feature_probs, key=lambda fp: abs(fp.p_bull - fp.p_bear), reverse=True)
        top_features = ranked[:max_features]
        
        if bool(getattr(self.config, 'evidence_threshold_enabled', False)):
            threshold = self.config.evidence_threshold
            bull_probs = [fp.weighted_p_bull for fp in top_features if fp.weighted_p_bull > threshold]
            bear_probs = [fp.weighted_p_bear for fp in top_features if fp.weighted_p_bear > threshold]
        else:
            bull_probs = [fp.weighted_p_bull for fp in top_features]
            bear_probs = [fp.weighted_p_bear for fp in top_features]
        
        # Dampened Noisy-OR: use sqrt(p) to reduce saturation speed
        if bull_probs:
            dampened = [np.sqrt(p) * 0.5 for p in bull_probs]  # dampen
            p_bull = 1.0 - np.prod([1.0 - d for d in dampened])
        else:
            p_bull = 0.2
        
        if bear_probs:
            dampened = [np.sqrt(p) * 0.5 for p in bear_probs]
            p_bear = 1.0 - np.prod([1.0 - d for d in dampened])
        else:
            p_bear = 0.2
        
        p_neutral = float(np.clip(1.0 - max(p_bull, p_bear), 0.1, 0.6))
        
        return float(p_bull), float(p_bear), p_neutral
    
    def _bayesian_aggregation(
        self,
        prior: RegimeProbabilities,
        feature_probs: List[FeatureProbability]
    ) -> Tuple[float, float, float]:
        """Bayesian aggregation: prior × likelihood ratios."""
        # Start with prior
        log_odds_bull = np.log(prior.p_bull / (1 - prior.p_bull + 1e-10) + 1e-10)
        log_odds_bear = np.log(prior.p_bear / (1 - prior.p_bear + 1e-10) + 1e-10)
        
        # Update with each feature's likelihood ratio
        for fp in feature_probs:
            if (not bool(getattr(self.config, 'evidence_threshold_enabled', False))) or fp.weighted_p_bull > self.config.evidence_threshold:
                lr_bull = fp.weighted_p_bull / (1 - fp.weighted_p_bull + 1e-10)
                log_odds_bull += np.log(lr_bull + 1e-10) * 0.5  # Dampen updates
            
            if (not bool(getattr(self.config, 'evidence_threshold_enabled', False))) or fp.weighted_p_bear > self.config.evidence_threshold:
                lr_bear = fp.weighted_p_bear / (1 - fp.weighted_p_bear + 1e-10)
                log_odds_bear += np.log(lr_bear + 1e-10) * 0.5
        
        # Convert back to probabilities
        p_bull = 1.0 / (1.0 + np.exp(-np.clip(log_odds_bull, -10, 10)))
        p_bear = 1.0 / (1.0 + np.exp(-np.clip(log_odds_bear, -10, 10)))
        p_neutral = 1.0 - max(p_bull, p_bear)
        
        return float(p_bull), float(p_bear), float(np.clip(p_neutral, 0.1, 0.8))
    
    def _weighted_avg_aggregation(
        self,
        feature_probs: List[FeatureProbability]
    ) -> Tuple[float, float, float]:
        """Simple weighted average aggregation."""
        if not feature_probs:
            return 0.33, 0.33, 0.34
        
        total_weight = 0.0
        sum_bull = 0.0
        sum_bear = 0.0
        sum_neutral = 0.0
        
        for fp in feature_probs:
            weight = fp.reliability_weight
            sum_bull += fp.p_bull * weight
            sum_bear += fp.p_bear * weight
            sum_neutral += fp.p_neutral * weight
            total_weight += weight
        
        if total_weight > 0:
            p_bull = sum_bull / total_weight
            p_bear = sum_bear / total_weight
            p_neutral = sum_neutral / total_weight
        else:
            p_bull, p_bear, p_neutral = 0.33, 0.33, 0.34
        
        return float(p_bull), float(p_bear), float(p_neutral)


# =============================================================================
# Stage 5: Dynamic Thresholding
# =============================================================================

class DynamicThresholdCalculator:
    """
    Calculates dynamic decision thresholds based on market conditions.
    
    Threshold = base + vol_factor × volatility + drawdown_penalty
    """
    
    def __init__(self, config: ProbabilisticConfig):
        self.config = config
        self._recent_drawdown = 0.0
        self._equity_peak = 1.0
    
    def calculate_threshold(
        self,
        current_volatility: float,
        current_drawdown: float = 0.0
    ) -> float:
        """
        Calculate dynamic threshold for decision.
        
        Args:
            current_volatility: Current market volatility
            current_drawdown: Current drawdown from equity peak
        
        Returns:
            Threshold value for decision
        """
        base = self.config.base_threshold
        
        # Volatility adjustment (higher vol = higher threshold)
        vol_adjustment = self.config.vol_factor * current_volatility
        
        # Drawdown penalty (higher DD = higher threshold, more conservative)
        dd_adjustment = self.config.drawdown_penalty * (current_drawdown / 0.1)  # Per 10% DD
        
        threshold = base + vol_adjustment + dd_adjustment
        
        return float(np.clip(threshold, 0.34, 0.8))
    
    def update_drawdown(self, current_equity: float):
        """Update drawdown tracking."""
        if current_equity > self._equity_peak:
            self._equity_peak = current_equity
        
        self._recent_drawdown = (self._equity_peak - current_equity) / self._equity_peak


# =============================================================================
# Stage 6: Alpha Decay
# =============================================================================

class AlphaDecayManager:
    """
    Manages alpha decay for signal confidence.
    
    Signals lose confidence over time if not refreshed.
    """
    
    def __init__(self, config: ProbabilisticConfig):
        self.config = config
        self._signal_timestamps: Dict[str, datetime] = {}
    
    def apply_decay(
        self,
        signal_id: str,
        confidence: float,
        bars_since_signal: int = 0
    ) -> float:
        """
        Apply alpha decay to confidence.
        
        Args:
            signal_id: Unique identifier for the signal
            confidence: Original confidence
            bars_since_signal: Number of bars since signal was generated
        
        Returns:
            Decayed confidence
        """
        if not self.config.alpha_decay_enabled:
            return confidence
        
        decay_factor = np.exp(-self.config.alpha_decay_rate * bars_since_signal)
        decayed = confidence * decay_factor
        
        # Apply floor
        decayed = max(decayed, self.config.alpha_min_confidence)
        
        return float(decayed)
    
    def refresh_signal(self, signal_id: str):
        """Refresh a signal's timestamp."""
        self._signal_timestamps[signal_id] = datetime.utcnow()
    
    def get_bars_since_signal(self, signal_id: str, bar_duration_minutes: int = 60) -> int:
        """Get number of bars since signal was generated."""
        if signal_id not in self._signal_timestamps:
            return 0
        
        elapsed = datetime.utcnow() - self._signal_timestamps[signal_id]
        bars = int(elapsed.total_seconds() / (bar_duration_minutes * 60))
        return max(0, bars)


# =============================================================================
# Main Probabilistic Alpha Factory
# =============================================================================

class ProbabilisticAlphaFactory:
    """
    Main class for the Probabilistic Alpha Factory.
    
    Orchestrates all stages:
    1. Regime prior from swings
    2. Feature probability calibration
    3. Causal confidence adjustment
    4. MH-TCN temporal refinement
    5. Evidence aggregation
    6. Dynamic thresholding
    7. Alpha decay management
    """
    
    def __init__(
        self,
        config: Optional[ProbabilisticConfig] = None,
        mhtcn_provider: Optional[Any] = None
    ):
        self.config = config or ProbabilisticConfig()
        
        # Initialize components
        self.metadata_store = FeatureMetadataStore(self.config.metadata_path)
        self.regime_calculator = RegimePriorCalculator(self.config)
        self.feature_calibrator = FeatureProbabilityCalibrator(self.config, self.metadata_store)
        self.causal_calculator = CausalConfidenceCalculator(self.config)
        self.evidence_aggregator = EvidenceAggregator(self.config)
        self.threshold_calculator = DynamicThresholdCalculator(self.config)
        self.alpha_decay = AlphaDecayManager(self.config)
        
        # MH-TCN provider (optional)
        self.mhtcn_provider = mhtcn_provider
        
        # Probability sequence buffer for MH-TCN
        self._prob_sequence: List[Dict[str, float]] = []
        
        logger.info("ProbabilisticAlphaFactory initialized")
    
    def evaluate(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        timeframe: str = "H1",
        swing_points: Optional[List] = None,
        causality_results: Optional[Dict] = None,
        current_equity: float = 1.0,
        signal_id: str = "default"
    ) -> DecisionOutput:
        """
        Main evaluation method.
        
        Args:
            df: OHLCV DataFrame
            features: Extracted features DataFrame
            swing_points: List of swing point objects
            causality_results: Pre-computed causality analysis
            current_equity: Current equity for drawdown calculation
            signal_id: Unique signal identifier for decay tracking
        
        Returns:
            DecisionOutput with trading decision
        """
        reasoning = []
        
        # Stage 1: Regime Prior
        regime_prior = self.regime_calculator.calculate(df, swing_points)
        reasoning.append(f"Regime prior: {regime_prior.dominant_regime} (entropy={regime_prior.entropy:.2f})")
        
        # Stage 2: Feature Probabilities
        feature_probs = self._calibrate_features(features, regime_prior.dominant_regime, causality_results)
        reasoning.append(f"Calibrated {len(feature_probs)} features")
        
        # Stage 3: Calculate stability
        stability_score = self._calculate_stability(df)
        reasoning.append(f"Stability score: {stability_score:.2f}")
        
        # Stage 4: MH-TCN temporal refinement (if available)
        mhtcn_probs = None
        mhtcn_prediction = None
        if self.mhtcn_provider is not None:
            mhtcn_prediction = self.mhtcn_provider.predict(df, str(timeframe or "H1"))
            if mhtcn_prediction:
                direction_probs = mhtcn_prediction.direction_probs
                mhtcn_probs = {
                    "bear": float(direction_probs[0]),
                    "neutral": float(direction_probs[1]),
                    "bull": float(direction_probs[2])
                }
                reasoning.append(f"MH-TCN: bull={mhtcn_probs.get('bull', 0):.2f}")
        
        # Stage 5: Evidence aggregation
        p_bull, p_bear, p_neutral, contributions = self.evidence_aggregator.aggregate(
            regime_prior, feature_probs, stability_score, mhtcn_probs
        )
        reasoning.append(f"Aggregated: P(bull)={p_bull:.2f}, P(bear)={p_bear:.2f}")
        
        # Stage 6: Dynamic threshold
        current_vol = self._get_current_volatility(df)
        self.threshold_calculator.update_drawdown(current_equity)
        threshold = self.threshold_calculator.calculate_threshold(
            current_vol, self.threshold_calculator._recent_drawdown
        )
        reasoning.append(f"Dynamic threshold: {threshold:.2f}")
        
        # Stage 7: Alpha decay
        bars_since = self.alpha_decay.get_bars_since_signal(signal_id)
        confidence = max(p_bull, p_bear)
        decayed_confidence = self.alpha_decay.apply_decay(signal_id, confidence, bars_since)
        
        # Decision (sampled diagnostic logging)
        if not hasattr(self, '_eval_count'):
            self._eval_count = 0
        self._eval_count += 1
        if self._eval_count <= 10 or self._eval_count % 500 == 0:
            logger.info(
                f"[{signal_id}] p_bull={p_bull:.4f} p_bear={p_bear:.4f} p_neut={p_neutral:.4f} "
                f"thresh={threshold:.4f} conf={decayed_confidence:.4f} vol={current_vol:.6f}"
            )
        # Require minimum directional edge: |p_bull - p_bear| must exceed
        # min_edge to confirm the signal is not ambiguous noise.
        min_edge = 0.06
        directional_edge = abs(p_bull - p_bear)
        
        if p_bull > threshold and p_bull > p_bear and directional_edge >= min_edge:
            direction = "LONG"
            # Dampened size scaling: sqrt prevents oversizing on noisy confidence
            raw_excess = (p_bull - threshold) / (1 - threshold + 1e-10)
            size_mult = min(np.sqrt(raw_excess) * 0.6, 0.7)
            self.alpha_decay.refresh_signal(signal_id)
        elif p_bear > threshold and p_bear > p_bull and directional_edge >= min_edge:
            direction = "SHORT"
            raw_excess = (p_bear - threshold) / (1 - threshold + 1e-10)
            size_mult = min(np.sqrt(raw_excess) * 0.6, 0.7)
            self.alpha_decay.refresh_signal(signal_id)
        else:
            direction = "HOLD"
            size_mult = 0.0
        
        # Update probability sequence for MH-TCN
        self._update_prob_sequence({
            "bull": p_bull,
            "bear": p_bear,
            "neutral": p_neutral,
            "stability": stability_score
        })
        
        return DecisionOutput(
            direction=direction,
            confidence=decayed_confidence,
            regime_probs=regime_prior,
            final_p_bull=p_bull,
            final_p_bear=p_bear,
            threshold_used=threshold,
            size_multiplier=float(np.clip(size_mult, 0, 1)),
            reasoning=reasoning,
            feature_contributions=contributions,
            mhtcn_contribution=mhtcn_probs,
            mhtcn_prediction=mhtcn_prediction,
            stability_score=stability_score,
            alpha_decay_applied=decayed_confidence / (confidence + 1e-10)
        )
    
    def _calibrate_features(
        self,
        features: pd.DataFrame,
        regime_context: str,
        causality_results: Optional[Dict]
    ) -> List[FeatureProbability]:
        """Calibrate all features."""
        if features is None or features.empty:
            return []
        
        feature_probs: List[FeatureProbability] = []
        row = features.iloc[-1]

        if bool(getattr(self.config, 'key_features_only', False)):
            feature_names = [f for f in (self.config.key_features or []) if f in features.columns]
        else:
            exclude_patterns = (
                'timestamp', 'datetime', 'date', 'time',
                'year', 'month', 'day', 'hour', 'minute', 'second',
                'id', 'idx', 'index',
                'session', 'bar', 'candle',
            )

            numeric_cols = [c for c in features.select_dtypes(include=[np.number]).columns]
            feature_names = []
            for c in numeric_cols:
                cc = str(c).strip().lower()
                if any(p in cc for p in exclude_patterns):
                    continue
                feature_names.append(str(c))

        for feat_name in feature_names:
            try:
                value = row[feat_name]
            except Exception:
                continue

            if pd.notna(value) and np.isfinite(value):
                fp = self.feature_calibrator.calibrate_feature(
                    feat_name, float(value), regime_context
                )

                fp.causal_confidence = self.causal_calculator.calculate(
                    feat_name, causality_results
                )

                feature_probs.append(fp)

        if bool(getattr(self.config, 'key_features_only', False)):
            logger.debug(f"ProbabilisticAlphaFactory: key_features_only enabled ({len(feature_probs)} features)")
        else:
            logger.debug(f"ProbabilisticAlphaFactory: calibrating all numeric features ({len(feature_probs)} features)")

        return feature_probs
    
    def _calculate_stability(self, df: pd.DataFrame) -> float:
        """Calculate market stability score."""
        lookback = int(getattr(self.config, 'stability_lookback', 20))
        if df is None or len(df) < lookback:
            return 0.5
        
        recent = df.tail(lookback)
        returns = recent['close'].pct_change().dropna()
        
        if len(returns) < 5:
            return 0.5
        
        # Stability = inverse of volatility (normalized)
        vol = float(returns.std())
        
        # Adaptive volatility normalization based on typical ranges
        # For M5/M15, 0.01 is a reasonable high-vol threshold.
        high_vol_threshold = 0.01
        
        normalized_vol = np.clip(vol / high_vol_threshold, 0, 2)
        stability = 1.0 - (normalized_vol / 2.0)
        
        # Apply stability weight from config if present
        stability_weight = float(getattr(self.config, 'stability_weight', 0.2))
        
        # Ensure it doesn't drop too low unless volatility is extreme
        return float(np.clip(stability, 0.2, 1.0))
    
    def _get_current_volatility(self, df: pd.DataFrame) -> float:
        """Get current volatility."""
        if df is None or len(df) < 20:
            return 0.01
        
        returns = df['close'].pct_change().tail(20).dropna()
        return float(returns.std()) if len(returns) > 0 else 0.01
    
    def _get_mhtcn_probs(self, df: pd.DataFrame, timeframe: str = "H1") -> Optional[Dict[str, float]]:
        """Get MH-TCN probability predictions."""
        if self.mhtcn_provider is None:
            return None
        
        try:
            # Get prediction from MH-TCN
            prediction = self.mhtcn_provider.predict(df, str(timeframe or "H1"))
            if prediction is None:
                return None
            
            # Convert direction_probs to regime probs
            direction_probs = prediction.direction_probs
            return {
                "bear": float(direction_probs[0]),
                "neutral": float(direction_probs[1]),
                "bull": float(direction_probs[2])
            }
        except Exception as e:
            logger.warning(f"MH-TCN prediction failed: {e}")
            return None
    
    def _update_prob_sequence(self, probs: Dict[str, float]):
        """Update probability sequence buffer for MH-TCN."""
        self._prob_sequence.append(probs)
        
        # Keep only last N entries
        max_len = self.config.mhtcn_sequence_length
        if len(self._prob_sequence) > max_len:
            self._prob_sequence = self._prob_sequence[-max_len:]
    
    def get_probability_sequence(self) -> np.ndarray:
        """Get probability sequence for MH-TCN input."""
        if not self._prob_sequence:
            return np.zeros((self.config.mhtcn_sequence_length, 4))
        
        # Convert to array
        seq = []
        for p in self._prob_sequence:
            seq.append([
                p.get("bull", 0.33),
                p.get("bear", 0.33),
                p.get("neutral", 0.34),
                p.get("stability", 0.5)
            ])
        
        arr = np.array(seq, dtype=np.float32)
        
        # Pad if needed
        if len(arr) < self.config.mhtcn_sequence_length:
            pad_len = self.config.mhtcn_sequence_length - len(arr)
            pad = np.tile(arr[0:1], (pad_len, 1))
            arr = np.vstack([pad, arr])
        
        return arr
    
    def save_metadata(self):
        """Save feature metadata to disk."""
        self.metadata_store.save()
    
    def update_feature_stats_from_backtest(
        self,
        backtest_results: pd.DataFrame,
        feature_columns: List[str]
    ):
        """
        Update feature statistics from walk-forward backtest results.
        
        Args:
            backtest_results: DataFrame with columns including 'regime', 'pnl', and feature columns
            feature_columns: List of feature column names to update
        """
        if backtest_results is None or backtest_results.empty:
            return
        
        for feat_name in feature_columns:
            if feat_name not in backtest_results.columns:
                continue
            
            try:
                # Calculate quantiles per regime
                quantiles_bull = {}
                quantiles_bear = {}
                quantiles_neutral = {}
                
                for regime, q_dict in [
                    ("bull", quantiles_bull),
                    ("bear", quantiles_bear),
                    ("neutral", quantiles_neutral)
                ]:
                    regime_data = backtest_results[
                        backtest_results.get("regime", "neutral") == regime
                    ][feat_name].dropna()
                    
                    if len(regime_data) > 10:
                        q_dict["q10"] = float(regime_data.quantile(0.1))
                        q_dict["q25"] = float(regime_data.quantile(0.25))
                        q_dict["q50"] = float(regime_data.quantile(0.5))
                        q_dict["q75"] = float(regime_data.quantile(0.75))
                        q_dict["q90"] = float(regime_data.quantile(0.9))
                
                # Calculate hit rate (accuracy when feature signals direction)
                # Simplified: correlation with PnL
                if "pnl" in backtest_results.columns:
                    corr = backtest_results[feat_name].corr(backtest_results["pnl"])
                    hit_rate = (abs(corr) + 1) / 2  # Map [-1, 1] to [0, 1]
                else:
                    hit_rate = 0.5
                
                # Calculate specificity (how different is distribution across regimes)
                # Simplified: variance of means
                means = []
                for regime in ["bull", "bear", "neutral"]:
                    regime_data = backtest_results[
                        backtest_results.get("regime", "neutral") == regime
                    ][feat_name].dropna()
                    if len(regime_data) > 0:
                        means.append(regime_data.mean())
                
                if len(means) >= 2:
                    specificity = np.std(means) / (np.mean(np.abs(means)) + 1e-10)
                    specificity = float(np.clip(specificity, 0, 1))
                else:
                    specificity = 0.5
                
                # Update metadata
                self.metadata_store.update_feature_stats(
                    feat_name,
                    quantiles_bull,
                    quantiles_bear,
                    quantiles_neutral,
                    float(hit_rate),
                    specificity
                )
                
            except Exception as e:
                logger.warning(f"Could not update stats for {feat_name}: {e}")
        
        # Save to disk
        self.metadata_store.save()
        logger.info(f"Updated feature statistics for {len(feature_columns)} features")


# =============================================================================
# Factory Function
# =============================================================================

def create_probabilistic_alpha_factory(
    config: Optional[ProbabilisticConfig] = None,
    mhtcn_weights_dir: Optional[str] = None,
    profile: str = "INTRADAY"
) -> ProbabilisticAlphaFactory:
    """
    Factory function to create a ProbabilisticAlphaFactory with MH-TCN integration.
    
    Args:
        config: Optional configuration
        mhtcn_weights_dir: Directory containing MH-TCN weights
        profile: Trading profile (SCALP, INTRADAY, SWING)
    
    Returns:
        Configured ProbabilisticAlphaFactory instance
    """
    config = config or ProbabilisticConfig()
    
    # Try to create MH-TCN provider
    mhtcn_provider = None
    try:
        from .mhtcn_integration import MHTCNFeatureProvider
        from .trading_profiles import get_profile

        profile_key = str(profile or 'INTRADAY').upper().strip()
        if profile_key == 'SCALP':
            profile_key = 'SCALPING'

        trading_profile = get_profile(profile_key)
        mhtcn_provider = MHTCNFeatureProvider(
            profile=trading_profile,
            weights_dir=mhtcn_weights_dir
        )
        logger.info(f"MH-TCN provider initialized for {profile_key}")
    except Exception as e:
        logger.warning(f"Could not initialize MH-TCN provider: {e}")
    
    return ProbabilisticAlphaFactory(config=config, mhtcn_provider=mhtcn_provider)
