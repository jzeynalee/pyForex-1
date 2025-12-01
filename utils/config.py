import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # MT5 Credentials
    MT5_ACCOUNT: int
    MT5_PASSWORD: str
    MT5_SERVER: str
    MT5_PATH: str = "" # Optional path
    
    # Trading Config
    SYMBOL: str = "EURUSD"
    TIMEFRAME: str = "H1" # Input as string, convert in connector
    LOT_SIZE: float = 0.10
    MAGIC_NUMBER: int = 123456
    
    # Model Config
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    CONFIDENCE_THRESHOLD: float = 0.75

    class Config:
        env_file = ".env"

settings = Settings()