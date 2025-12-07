# utils/mtf_config.py
"""
Multi-Timeframe Configuration System

Defines MTF profiles for different trading strategies:
- SCALP: M5, M15, H1 (short-term, faster signals)
- SWING: M15, H1, H4 (medium-term, more reliable signals)

Each profile defines:
- Timeframes to use
- Weights for confluence scoring
- Primary timeframe for signal generation
- Feature engineering parameters
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Literal
from enum import Enum


class Timeframe(str, Enum):
    """Supported timeframes."""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    
    @property
    def minutes(self) -> int:
        """Return timeframe duration in minutes."""
        mapping = {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30,
            "H1": 60, "H4": 240, "D1": 1440
        }
        return mapping[self.value]
    
    @classmethod
    def from_string(cls, s: str) -> "Timeframe":
        """Parse timeframe from string."""
        return cls(s.upper())


@dataclass
class MTFProfile:
    """
    Multi-Timeframe Profile Configuration.
    
    Defines a set of timeframes to analyze together with their
    relative weights for confluence scoring.
    """
    name: str
    description: str
    
    # Timeframes: (higher, primary, lower)
    higher_tf: Timeframe
    primary_tf: Timeframe  
    lower_tf: Timeframe
    
    # Weights for confluence scoring (must sum to 1.0)
    weights: Dict[str, float] = field(default_factory=dict)
    
    # Number of candles to fetch per timeframe
    candle_counts: Dict[str, int] = field(default_factory=dict)
    
    # Minimum bars required for analysis
    min_bars: int = 100
    
    # Signal generation settings
    min_confluence_score: float = 0.60
    require_higher_tf_alignment: bool = True
    
    def __post_init__(self):
        """Set defaults if not provided."""
        if not self.weights:
            self.weights = {
                self.higher_tf.value: 0.35,
                self.primary_tf.value: 0.45,
                self.lower_tf.value: 0.20,
            }
        
        if not self.candle_counts:
            self.candle_counts = {
                self.higher_tf.value: 200,
                self.primary_tf.value: 200,
                self.lower_tf.value: 200,
            }
    
    @property
    def timeframes(self) -> List[Timeframe]:
        """Return all timeframes in order (higher to lower)."""
        return [self.higher_tf, self.primary_tf, self.lower_tf]
    
    @property
    def timeframe_strings(self) -> List[str]:
        """Return timeframe strings."""
        return [tf.value for tf in self.timeframes]
    
    def get_weight(self, tf: str) -> float:
        """Get weight for a timeframe."""
        return self.weights.get(tf.upper(), 0.0)
    
    def get_candle_count(self, tf: str) -> int:
        """Get number of candles to fetch for a timeframe."""
        return self.candle_counts.get(tf.upper(), 200)


# Pre-defined profiles
SCALP_PROFILE = MTFProfile(
    name="SCALP",
    description="Short-term scalping: M5 primary with M15/H1 context",
    higher_tf=Timeframe.H1,
    primary_tf=Timeframe.M15,
    lower_tf=Timeframe.M5,
    weights={
        "H1": 0.30,   # Higher TF trend direction
        "M15": 0.45,  # Primary signal generation
        "M5": 0.25,   # Entry timing
    },
    candle_counts={
        "H1": 100,
        "M15": 200,
        "M5": 300,
    },
    min_confluence_score=0.55,
    require_higher_tf_alignment=True,
)


SWING_PROFILE = MTFProfile(
    name="SWING",
    description="Medium-term swing trading: H1 primary with M15/H4 context",
    higher_tf=Timeframe.H4,
    primary_tf=Timeframe.H1,
    lower_tf=Timeframe.M15,
    weights={
        "H4": 0.35,   # Higher TF trend direction
        "H1": 0.45,   # Primary signal generation  
        "M15": 0.20,  # Entry refinement
    },
    candle_counts={
        "H4": 200,
        "H1": 200,
        "M15": 200,
    },
    min_confluence_score=0.60,
    require_higher_tf_alignment=True,
)


INTRADAY_PROFILE = MTFProfile(
    name="INTRADAY",
    description="Intraday trading: M15 primary with M5/H1 context",
    higher_tf=Timeframe.H1,
    primary_tf=Timeframe.M15,
    lower_tf=Timeframe.M5,
    weights={
        "H1": 0.35,
        "M15": 0.40,
        "M5": 0.25,
    },
    min_confluence_score=0.58,
)


# Profile registry
MTF_PROFILES: Dict[str, MTFProfile] = {
    "SCALP": SCALP_PROFILE,
    "SWING": SWING_PROFILE,
    "INTRADAY": INTRADAY_PROFILE,
}


def get_profile(name: str) -> MTFProfile:
    """Get MTF profile by name."""
    name_upper = name.upper()
    if name_upper not in MTF_PROFILES:
        available = list(MTF_PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return MTF_PROFILES[name_upper]


def create_custom_profile(
    name: str,
    timeframes: Tuple[str, str, str],  # (higher, primary, lower)
    weights: Tuple[float, float, float] = (0.35, 0.45, 0.20),
    **kwargs
) -> MTFProfile:
    """
    Create a custom MTF profile.
    
    Args:
        name: Profile name
        timeframes: Tuple of (higher_tf, primary_tf, lower_tf)
        weights: Tuple of weights for each timeframe
        **kwargs: Additional MTFProfile parameters
    
    Returns:
        Configured MTFProfile
    """
    higher, primary, lower = timeframes
    w_higher, w_primary, w_lower = weights
    
    return MTFProfile(
        name=name,
        description=kwargs.get('description', f"Custom profile: {timeframes}"),
        higher_tf=Timeframe.from_string(higher),
        primary_tf=Timeframe.from_string(primary),
        lower_tf=Timeframe.from_string(lower),
        weights={
            higher.upper(): w_higher,
            primary.upper(): w_primary,
            lower.upper(): w_lower,
        },
        **{k: v for k, v in kwargs.items() if k != 'description'}
    )


@dataclass
class MTFAnalysisConfig:
    """
    Configuration for MTF analysis operations.
    """
    # Active profile
    profile: MTFProfile
    
    # Analysis settings
    use_structural_analysis: bool = True
    use_regime_filter: bool = True
    use_ml_confirmation: bool = True
    
    # Feature engineering
    compute_ema_slopes: bool = True
    compute_adx: bool = True
    compute_rsi: bool = True
    compute_volatility: bool = True
    
    # Signal thresholds
    signal_threshold: float = 0.60
    counter_trend_threshold: float = 0.85
    
    # Risk adjustment based on confluence
    adjust_risk_by_confluence: bool = True
    high_confluence_risk_mult: float = 1.3  # Risk multiplier for strong confluence
    low_confluence_risk_mult: float = 0.7   # Risk multiplier for weak confluence


# Default configurations
DEFAULT_SCALP_CONFIG = MTFAnalysisConfig(
    profile=SCALP_PROFILE,
    signal_threshold=0.55,
    counter_trend_threshold=0.80,
)

DEFAULT_SWING_CONFIG = MTFAnalysisConfig(
    profile=SWING_PROFILE,
    signal_threshold=0.60,
    counter_trend_threshold=0.85,
)
