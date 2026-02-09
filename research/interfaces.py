"""
Core interfaces and data contracts for the research framework.

All alpha heads, MH-TCN filters, and the experiment harness communicate
through these standardized types.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class VariantID(Enum):
    """The six experimental variants."""
    V1_ALPHA = "V1_Alpha"
    V2_ALPHA_MHTCN = "V2_Alpha+MHTCN"
    V3_ALPHA2 = "V3_Alpha2"
    V4_ALPHA2_MHTCN = "V4_Alpha2+MHTCN"
    V5_ALPHA_PROB_MHTCN = "V5_Alpha+ProbMHTCN"
    V6_ALPHA2_PROB_MHTCN = "V6_Alpha2+ProbMHTCN"


class RegimeLabel(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    TRANSITION = "transition"


# =============================================================================
# Data contracts
# =============================================================================

@dataclass
class AlphaSignal:
    """Standardized output from any alpha head.

    Direction authority lives here — MH-TCN may NOT override direction.
    """
    direction: Direction
    probability: float          # P(direction is correct), calibrated [0,1]
    confidence: float           # raw confidence before calibration
    regime: RegimeLabel
    reasoning: List[str] = field(default_factory=list)
    # Per-category probabilities (Alpha2 populates; Alpha1 leaves empty)
    category_probs: Dict[str, float] = field(default_factory=dict)
    # Raw feature vector at this bar (shared across pipeline)
    feature_vector: Optional[np.ndarray] = None
    timestamp: Optional[pd.Timestamp] = None

    @property
    def is_trade(self) -> bool:
        return self.direction != Direction.HOLD


@dataclass
class MHTCNOutput:
    """Standardized output from any MH-TCN filter.

    g_factor is the multiplicative modulator: P_final = P_alpha * g_factor
    g_factor ∈ (0, 1].  A value of 1.0 means "no change".
    """
    g_factor: float             # multiplicative confidence factor
    signal_survival_prob: float # P(signal still valid after N bars)
    confidence_decay: float     # suggested decay rate
    regime_validity: float      # P(current regime assessment is correct)
    raw_output: Optional[np.ndarray] = None

    def __post_init__(self):
        self.g_factor = float(np.clip(self.g_factor, 0.0, 1.0))


@dataclass
class TradeRecord:
    """Single trade for backtest evaluation."""
    variant_id: str
    bar_index: int
    entry_time: Optional[pd.Timestamp]
    exit_time: Optional[pd.Timestamp]
    direction: Direction
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    volume: float
    pnl: float
    # Probabilities for attribution
    p_alpha: float              # raw alpha probability
    p_final: float              # after MH-TCN modulation (= p_alpha if no MHTCN)
    g_factor: float = 1.0       # MH-TCN multiplicative factor
    regime: str = ""
    # Forward label for Brier scoring
    forward_label: Optional[int] = None  # 1 = trade was correct, 0 = not


@dataclass
class VariantResult:
    """Aggregated results for one variant."""
    variant_id: str
    trades: List[TradeRecord]
    equity_curve: np.ndarray
    # Pre-computed metrics (filled by MetricsCollector)
    metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# Abstract base classes
# =============================================================================

class AlphaHead(ABC):
    """Abstract interface for all alpha generation heads.

    Produces direction + probability from market data and features.
    No ML allowed inside Alpha heads (rule-based / statistical only for V1;
    category-based probabilistic for V2).
    """

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        ...

    @abstractmethod
    def evaluate(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        regime: RegimeLabel,
    ) -> AlphaSignal:
        """Generate alpha signal for the current bar.

        Args:
            df: OHLCV data up to current bar (no future leak).
            features: Pre-computed technical features aligned with df.
            regime: Current regime from the shared regime detector.

        Returns:
            AlphaSignal with direction, probability, and metadata.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a fresh backtest run."""
        ...


class MHTCNFilter(ABC):
    """Abstract interface for MH-TCN confidence filters.

    Receives an AlphaSignal and returns a multiplicative g-factor.
    MUST NOT override direction.
    """

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def filter(
        self,
        signal: AlphaSignal,
        df: pd.DataFrame,
        features: pd.DataFrame,
    ) -> MHTCNOutput:
        """Apply temporal filtering to an alpha signal.

        Args:
            signal: The alpha signal to filter.
            df: OHLCV data up to current bar.
            features: Pre-computed features aligned with df.

        Returns:
            MHTCNOutput with g_factor for multiplicative modulation.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        ...


class NullMHTCNFilter(MHTCNFilter):
    """Pass-through filter (g_factor = 1.0) for variants without MH-TCN."""

    def name(self) -> str:
        return "NullFilter"

    def filter(self, signal, df, features) -> MHTCNOutput:
        return MHTCNOutput(
            g_factor=1.0,
            signal_survival_prob=1.0,
            confidence_decay=0.0,
            regime_validity=1.0,
        )

    def reset(self):
        pass


# =============================================================================
# Variant configuration
# =============================================================================

@dataclass
class VariantConfig:
    """Defines one experimental variant."""
    variant_id: VariantID
    alpha_head: AlphaHead
    mhtcn_filter: MHTCNFilter
    description: str = ""

    # Execution / risk params (MUST be identical across variants)
    initial_balance: float = 10_000.0
    risk_per_trade: float = 0.01        # 1% risk
    commission_per_lot: float = 7.0
    spread_pips: float = 1.0
    pip_value: float = 10.0
    min_rr: float = 1.5                 # minimum risk-reward
    atr_sl_mult: float = 2.0
    max_open_trades: int = 1
    cooldown_bars: int = 6              # bars between trades

    # Probability gate
    min_probability: float = 0.55       # P_final must exceed this to trade
