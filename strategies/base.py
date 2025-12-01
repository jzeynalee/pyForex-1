# strategies/base.py
"""
Base strategy interface defining the contract for all strategies.
"""
from abc import ABC, abstractmethod
from typing import Protocol, Optional
import pandas as pd


class DataProvider(Protocol):
    """Protocol for data providers."""
    def get_data(self, n: int = 100) -> pd.DataFrame: ...


class Executor(Protocol):
    """Protocol for trade executors."""
    def entry(self, signal: str, volume: float, sl: float, tp: float): ...
    def get_open_positions(self) -> list: ...


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Strategies receive market data and generate trading decisions.
    Execution is delegated to the executor (live or backtest).
    """
    
    def __init__(
        self,
        data_provider: DataProvider,
        executor: Executor,
        name: str = "BaseStrategy",
    ):
        self.data_provider = data_provider
        self.executor = executor
        self.name = name
        self.is_active = True
    
    @abstractmethod
    def on_bar(self, df: pd.DataFrame) -> Optional[str]:
        """
        Called when a new candle closes.
        
        Args:
            df: DataFrame with recent OHLCV data
        
        Returns:
            Signal generated ('BUY', 'SELL', 'HOLD', or None)
        """
        pass
    
    def on_tick(self, tick_data: dict):
        """
        Called on every tick (optional).
        Override for tick-based strategies.
        
        Args:
            tick_data: Dict with 'bid', 'ask', 'time'
        """
        pass
    
    def on_position_opened(self, position: dict):
        """Called when a position is opened."""
        pass
    
    def on_position_closed(self, position: dict, pnl: float):
        """Called when a position is closed."""
        pass
    
    def activate(self):
        """Enable the strategy."""
        self.is_active = True
    
    def deactivate(self):
        """Disable the strategy."""
        self.is_active = False
