# utils/mtf_config.py
"""
Multi-Timeframe Configuration Profiles

Defines MTF profiles for different trading styles:
- SCALP: M5, M15, H1 (fast, short-term)
- INTRADAY: M15, H1, H4 (balanced)
- SWING: H1, H4, D1 (slow, long-term)

Each profile specifies:
- Timeframes and their weights
- Minimum confluence requirements
- Signal generation parameters
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class Timeframe(Enum):
    """Supported timeframes."""
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    H1 = "H1"
    H4 = "H4"
    D1 = "D1"
    W1 = "W1"


# Timeframe to minutes mapping
TF_MINUTES = {
    Timeframe.M1: 1,
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
    Timeframe.D1: 1440,
    Timeframe.W1: 10080,
}


@dataclass
class MTFProfile:
    """Configuration profile for multi-timeframe analysis."""
    
    # Identity
    name: str
    description: str = ""
    
    # Timeframes (ordered from lowest to highest)
    timeframes: Tuple[Timeframe, ...] = (Timeframe.M15, Timeframe.H1, Timeframe.H4)
    
    # Which TF is the primary trading TF
    primary_tf: Timeframe = Timeframe.H1
    
    # Weights for each timeframe (should sum to 1.0)
    weights: Dict[str, float] = field(default_factory=dict)
    
    # Analysis parameters
    min_bars: int = 100
    min_confluence_score: float = 0.60
    require_higher_tf_alignment: bool = True
    
    # Signal generation
    signal_threshold: float = 0.65
    counter_trend_threshold: float = 0.85
    
    # Feature generation
    ema_periods: Tuple[int, ...] = (20, 50, 200)
    adx_period: int = 14
    atr_period: int = 14
    
    # Candle counts per timeframe
    candle_counts: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize weights and candle counts if not provided."""
        if not self.weights:
            # Default equal weights
            n = len(self.timeframes)
            self.weights = {tf.value: 1.0 / n for tf in self.timeframes}
        
        if not self.candle_counts:
            # Default candle counts
            self.candle_counts = {tf.value: 200 for tf in self.timeframes}
    
    @property
    def timeframe_strings(self) -> List[str]:
        """Get timeframes as string list."""
        return [tf.value for tf in self.timeframes]
    
    @property
    def higher_tf(self) -> Timeframe:
        """Get highest timeframe."""
        return self.timeframes[-1]
    
    @property
    def lower_tf(self) -> Timeframe:
        """Get lowest timeframe."""
        return self.timeframes[0]
    
    def get_weight(self, tf: str) -> float:
        """Get weight for a timeframe."""
        return self.weights.get(tf.upper(), 0.0)


# =============================================================================
# PRESET PROFILES
# =============================================================================

SCALP_PROFILE = MTFProfile(
    name="SCALP",
    description="Fast scalping profile using M5/M15/H1",
    timeframes=(Timeframe.M5, Timeframe.M15, Timeframe.H1),
    primary_tf=Timeframe.M15,
    weights={
        "M5": 0.25,
        "M15": 0.45,
        "H1": 0.30,
    },
    min_bars=60,
    min_confluence_score=0.70,  # Higher threshold for scalps
    require_higher_tf_alignment=True,
    signal_threshold=0.70,
    counter_trend_threshold=0.90,
    candle_counts={
        "M5": 200,
        "M15": 150,
        "H1": 100,
    },
)


INTRADAY_PROFILE = MTFProfile(
    name="INTRADAY",
    description="Balanced intraday profile using M15/H1/H4",
    timeframes=(Timeframe.M15, Timeframe.H1, Timeframe.H4),
    primary_tf=Timeframe.H1,
    weights={
        "M15": 0.20,
        "H1": 0.45,
        "H4": 0.35,
    },
    min_bars=100,
    min_confluence_score=0.65,
    require_higher_tf_alignment=True,
    signal_threshold=0.65,
    counter_trend_threshold=0.85,
    candle_counts={
        "M15": 200,
        "H1": 150,
        "H4": 100,
    },
)


SWING_PROFILE = MTFProfile(
    name="SWING",
    description="Swing trading profile using H1/H4/D1",
    timeframes=(Timeframe.H1, Timeframe.H4, Timeframe.D1),
    primary_tf=Timeframe.H4,
    weights={
        "H1": 0.20,
        "H4": 0.45,
        "D1": 0.35,
    },
    min_bars=100,
    min_confluence_score=0.60,
    require_higher_tf_alignment=True,
    signal_threshold=0.60,
    counter_trend_threshold=0.80,
    candle_counts={
        "H1": 200,
        "H4": 150,
        "D1": 100,
    },
)


# Profile registry
PROFILES = {
    "SCALP": SCALP_PROFILE,
    "INTRADAY": INTRADAY_PROFILE,
    "SWING": SWING_PROFILE,
}


def get_profile(name: str) -> MTFProfile:
    """
    Get MTF profile by name.
    
    Args:
        name: Profile name ('SCALP', 'INTRADAY', 'SWING')
    
    Returns:
        MTFProfile instance
    
    Raises:
        ValueError: If profile not found
    """
    name = name.upper()
    if name not in PROFILES:
        raise ValueError(f"Unknown profile: {name}. Available: {list(PROFILES.keys())}")
    return PROFILES[name]


def create_custom_profile(
    name: str,
    timeframes: List[str],
    primary_tf: str,
    weights: Optional[Dict[str, float]] = None,
    **kwargs
) -> MTFProfile:
    """
    Create a custom MTF profile.
    
    Args:
        name: Profile name
        timeframes: List of timeframe strings (e.g., ['M15', 'H1', 'H4'])
        primary_tf: Primary trading timeframe
        weights: Optional weight dict
        **kwargs: Additional MTFProfile parameters
    
    Returns:
        Custom MTFProfile instance
    """
    # Convert strings to Timeframe enum
    tf_enums = tuple(Timeframe(tf.upper()) for tf in timeframes)
    primary_enum = Timeframe(primary_tf.upper())
    
    return MTFProfile(
        name=name,
        timeframes=tf_enums,
        primary_tf=primary_enum,
        weights=weights or {},
        **kwargs
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_tf_minutes(tf: str) -> int:
    """Convert timeframe string to minutes."""
    try:
        return TF_MINUTES[Timeframe(tf.upper())]
    except (ValueError, KeyError):
        # Fallback parsing
        tf = tf.upper()
        if tf.startswith('M'):
            return int(tf[1:])
        elif tf.startswith('H'):
            return int(tf[1:]) * 60
        elif tf.startswith('D'):
            return int(tf[1:]) * 1440
        elif tf.startswith('W'):
            return int(tf[1:]) * 10080
        return 60  # Default to H1


def sort_timeframes(timeframes: List[str]) -> List[str]:
    """Sort timeframes from lowest to highest."""
    return sorted(timeframes, key=lambda tf: get_tf_minutes(tf))


def get_higher_timeframe(tf: str, available: List[str]) -> Optional[str]:
    """Get the next higher timeframe from available list."""
    tf_minutes = get_tf_minutes(tf)
    
    higher = [t for t in available if get_tf_minutes(t) > tf_minutes]
    if not higher:
        return None
    
    return min(higher, key=lambda t: get_tf_minutes(t))


def get_lower_timeframe(tf: str, available: List[str]) -> Optional[str]:
    """Get the next lower timeframe from available list."""
    tf_minutes = get_tf_minutes(tf)
    
    lower = [t for t in available if get_tf_minutes(t) < tf_minutes]
    if not lower:
        return None
    
    return max(lower, key=lambda t: get_tf_minutes(t))