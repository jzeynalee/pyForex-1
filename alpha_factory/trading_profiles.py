# alpha_factory/trading_profiles.py
"""
Trading Profile Definitions for 3TF System.
Defines the valid timeframe hierarchies for different trading styles.
"""

from dataclasses import dataclass
from enum import Enum

class TimeFrame(Enum):
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class ProfileType(Enum):
    SCALPING = "SCALPING"
    INTRADAY = "INTRADAY"
    SWING = "SWING"

@dataclass(frozen=True)
class TradingProfile:
    """Immutable configuration for a trading profile."""
    type: ProfileType
    ltf: TimeFrame  # Execution / Timing
    mtf: TimeFrame  # Structure / Setup
    htf: TimeFrame  # Participation / Bias

    def __str__(self):
        return f"{self.type.value} [LTF:{self.ltf.value} | MTF:{self.mtf.value} | HTF:{self.htf.value}]"

# Registry of authorized profiles
PROFILES = {
    ProfileType.SCALPING: TradingProfile(
        ProfileType.SCALPING, 
        ltf=TimeFrame.M5, 
        mtf=TimeFrame.M15, 
        htf=TimeFrame.H1
    ),
    ProfileType.INTRADAY: TradingProfile(
        ProfileType.INTRADAY, 
        ltf=TimeFrame.M15, 
        mtf=TimeFrame.H1, 
        htf=TimeFrame.H4
    ),
    ProfileType.SWING: TradingProfile(
        ProfileType.SWING, 
        ltf=TimeFrame.H1, 
        mtf=TimeFrame.H4, 
        htf=TimeFrame.D1
    ),
}

def get_profile(profile_type: str) -> TradingProfile:
    """Factory method to retrieve a profile by string name."""
    try:
        return PROFILES[ProfileType[profile_type.upper()]]
    except KeyError:
        raise ValueError(f"Invalid profile type: {profile_type}. Available: {[p.name for p in ProfileType]}")# alpha_factory/trading_profiles.py
"""
Trading Profile Definitions for 3TF System.
Defines the valid timeframe hierarchies for different trading styles.
"""

from dataclasses import dataclass
from enum import Enum

class TimeFrame(Enum):
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

class ProfileType(Enum):
    SCALPING = "SCALPING"
    INTRADAY = "INTRADAY"
    SWING = "SWING"

@dataclass(frozen=True)
class TradingProfile:
    """Immutable configuration for a trading profile."""
    type: ProfileType
    ltf: TimeFrame  # Execution / Timing
    mtf: TimeFrame  # Structure / Setup
    htf: TimeFrame  # Participation / Bias

    def __str__(self):
        return f"{self.type.value} [LTF:{self.ltf.value} | MTF:{self.mtf.value} | HTF:{self.htf.value}]"

# Registry of authorized profiles
PROFILES = {
    ProfileType.SCALPING: TradingProfile(
        ProfileType.SCALPING, 
        ltf=TimeFrame.M5, 
        mtf=TimeFrame.M15, 
        htf=TimeFrame.H1
    ),
    ProfileType.INTRADAY: TradingProfile(
        ProfileType.INTRADAY, 
        ltf=TimeFrame.M15, 
        mtf=TimeFrame.H1, 
        htf=TimeFrame.H4
    ),
    ProfileType.SWING: TradingProfile(
        ProfileType.SWING, 
        ltf=TimeFrame.H1, 
        mtf=TimeFrame.H4, 
        htf=TimeFrame.D1
    ),
}

def get_profile(profile_type: str) -> TradingProfile:
    """Factory method to retrieve a profile by string name."""
    try:
        return PROFILES[ProfileType[profile_type.upper()]]
    except KeyError:
        raise ValueError(f"Invalid profile type: {profile_type}. Available: {[p.name for p in ProfileType]}")