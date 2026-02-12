# utils/config.py
"""
Centralized configuration with validation.
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Literal

class Settings(BaseSettings):
    # MT5 Credentials
    MT5_ACCOUNT: int = 0
    MT5_PASSWORD: str = ""
    MT5_SERVER: str = ""
    MT5_PATH: str = ""
    
    # Trading Config
    SYMBOL: str = "EURUSD"
    TIMEFRAME: Literal["M1", "M5", "M15", "M30", "H1", "H4", "D1"] = "H1"
    MAGIC_NUMBER: int = 123456
    
    # Risk Config
    MAX_DAILY_LOSS_PCT: float = 0.03
    RISK_PER_TRADE_PCT: float = 0.01
    MAX_DRAWDOWN_PCT: float = 0.10
    
    # Model Config
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    CONFIDENCE_THRESHOLD: float = 0.60
    ASSETS_DIR: Path = Path("E:/pyProject/pyForex-assets")
    WEIGHTS_DIR: Path = Path("E:/pyProject/pyForex-assets/models/weights")
    TREND_MODEL_PATH: Path = Path("E:/pyProject/pyForex-assets/models/trend_classifier.joblib")
    
    # Data Config
    SEQUENCE_LENGTH: int = 60
    IMAGE_SIZE: int = 224
    
    # Polling
    TICK_INTERVAL: float = 10.0  # seconds between checks

    # Guardrails
    ENFORCE_AUTHORITATIVE_PIPELINE: bool = True

    # Profile Selection
    PROFILE: Literal["SCALP", "INTRADAY", "SWING"] = "INTRADAY"

    # Timeframes
    SCALP_HTF: str = "H1"
    SCALP_MTF: str = "M15"
    SCALP_LTF: str = "M5"
    INTRADAY_HTF: str = "H4"
    INTRADAY_MTF: str = "H1"
    INTRADAY_LTF: str = "M15"
    SWING_HTF: str = "D1"
    SWING_MTF: str = "H4"
    SWING_LTF: str = "H1"

    # Thresholds
    SCALP_MIN_HTF_CONF: float = 0.38
    SCALP_MIN_MTF_CONF: float = 0.40
    SCALP_MIN_LTF_CONF: float = 0.40
    SCALP_MIN_STABILITY: float = 0.30
    SCALP_MIN_DIR_SCORE: float = 0.10
    SCALP_RELAXED: bool = True

    INTRADAY_MIN_HTF_CONF: float = 0.40
    INTRADAY_MIN_MTF_CONF: float = 0.42
    INTRADAY_MIN_LTF_CONF: float = 0.44
    INTRADAY_MIN_STABILITY: float = 0.40
    INTRADAY_MIN_DIR_SCORE: float = 0.05
    INTRADAY_RELAXED: bool = False

    SWING_MIN_HTF_CONF: float = 0.42
    SWING_MIN_MTF_CONF: float = 0.44
    SWING_MIN_LTF_CONF: float = 0.46
    SWING_MIN_STABILITY: float = 0.45
    SWING_MIN_DIR_SCORE: float = 0.08
    SWING_RELAXED: bool = False

    # Risk Management
    SCALP_BASE_RISK: float = 0.25
    SCALP_MIN_RR: float = 1.5
    SCALP_MAX_OPEN: int = 1
    SCALP_MAX_DAILY: int = 10
    SCALP_MAX_LOSS: float = 1.5
    SCALP_MIN_SL_PIPS: float = 6.0
    SCALP_MAX_LOT: float = 0.3
    SCALP_COOLDOWN: float = 30.0
    SCALP_ATR_SL_MULT: float = 2.8
    SCALP_ATR_PERIOD: int = 20

    INTRADAY_BASE_RISK: float = 0.5
    INTRADAY_MIN_RR: float = 2.0
    INTRADAY_MAX_OPEN: int = 2
    INTRADAY_MAX_DAILY: int = 5
    INTRADAY_MAX_LOSS: float = 2.0
    INTRADAY_MIN_SL_PIPS: float = 10.0
    INTRADAY_MAX_LOT: float = 1.0
    INTRADAY_COOLDOWN: float = 60.0
    INTRADAY_ATR_SL_MULT: float = 2.0
    INTRADAY_ATR_PERIOD: int = 14

    SWING_BASE_RISK: float = 1.0
    SWING_MIN_RR: float = 3.0
    SWING_MAX_OPEN: int = 3
    SWING_MAX_DAILY: int = 2
    SWING_MAX_LOSS: float = 3.0
    SWING_MIN_SL_PIPS: float = 30.0
    SWING_MAX_LOT: float = 2.0
    SWING_COOLDOWN: float = 240.0
    SWING_ATR_SL_MULT: float = 2.5
    SWING_ATR_PERIOD: int = 14

    # Decision Engine
    SCALP_AGG_METHOD: str = "weighted_avg"
    SCALP_MHTCN_WEIGHT: float = 0.55
    SCALP_STABILITY_WEIGHT: float = 0.05
    SCALP_REGIME_SCALE: float = 1.0
    SCALP_ENTROPY_WEIGHT: float = 0.15
    SCALP_CALIB_METHOD: str = "logistic"
    SCALP_DECAY_RATE: float = 0.06
    SCALP_KEY_FEATS_ONLY: bool = True

    INTRADAY_AGG_METHOD: str = "weighted_avg"
    INTRADAY_MHTCN_WEIGHT: float = 0.4
    INTRADAY_STABILITY_WEIGHT: float = 0.2
    INTRADAY_REGIME_SCALE: float = 2.5
    INTRADAY_ENTROPY_WEIGHT: float = 0.3
    INTRADAY_CALIB_METHOD: str = "logistic"
    INTRADAY_DECAY_RATE: float = 0.02
    INTRADAY_KEY_FEATS_ONLY: bool = False

    # V6 Strategy (AlphaV2 + ProbabilisticTCN)
    V6_WEIGHTS_DIR: Path = Path("E:/pyProject/pyForex-assets/models/v6_profiles")
    V6_MIN_G_FACTOR: float = 0.52  # Phase 1: g_factor IS trade probability
    V6_DEVICE: str = "cpu"

    SWING_AGG_METHOD: str = "bayesian"
    SWING_MHTCN_WEIGHT: float = 0.3
    SWING_STABILITY_WEIGHT: float = 0.15
    SWING_REGIME_SCALE: float = 3.0
    SWING_ENTROPY_WEIGHT: float = 0.2
    SWING_CALIB_METHOD: str = "logistic"
    SWING_DECAY_RATE: float = 0.01
    SWING_KEY_FEATS_ONLY: bool = False

    @field_validator('CONFIDENCE_THRESHOLD')
    @classmethod
    def validate_threshold(cls, v):
        if not 0.5 <= v <= 0.95:
            raise ValueError("Threshold must be between 0.5 and 0.95")
        return v

    class Config:
        env_file = "config.env"
        extra = "ignore"

# Singleton instance
try:
    settings = Settings()
except Exception:
    # Fallback for environments without .env
    settings = Settings(
        MT5_ACCOUNT=0,
        MT5_PASSWORD="",
        MT5_SERVER="",
    )
