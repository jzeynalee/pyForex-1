# utils/__init__.py
"""
Utilities package for pyForex.
"""

from .mtf_config import (
    MTFProfile,
    Timeframe,
    SCALP_PROFILE,
    SWING_PROFILE,
    INTRADAY_PROFILE,
    get_profile,
    create_custom_profile,
    # MTFAnalysisConfig,  # REMOVE THIS LINE - class doesn't exist
)

from .mtf_features import (
    MTFFeatureBuilder,
    MTFFeatureSet,
    build_ml_features,
)

__all__ = [
    'MTFProfile',
    'Timeframe',
    'SCALP_PROFILE',
    'SWING_PROFILE', 
    'INTRADAY_PROFILE',
    'get_profile',
    'create_custom_profile',
    # 'MTFAnalysisConfig',  # REMOVE THIS LINE
    'MTFFeatureBuilder',
    'MTFFeatureSet',
    'build_ml_features',
]