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
    WEIGHTS_DIR: Path = Path("models/weights")
    
    # Data Config
    SEQUENCE_LENGTH: int = 60
    IMAGE_SIZE: int = 224
    
    # Polling
    TICK_INTERVAL: float = 10.0  # seconds between checks

    @field_validator('CONFIDENCE_THRESHOLD')
    @classmethod
    def validate_threshold(cls, v):
        if not 0.5 <= v <= 0.95:
            raise ValueError("Threshold must be between 0.5 and 0.95")
        return v

    class Config:
        env_file = ".env"
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
