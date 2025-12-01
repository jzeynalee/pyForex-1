# strategies/base.py
# This module defines the strict contract that both your Live Bot and Backtester must follow.

from abc import ABC, abstractmethod

class Strategy(ABC):
    def __init__(self, data_provider, executor):
        self.data_provider = data_provider
        self.executor = executor

    @abstractmethod
    def on_tick(self):
        """Called on every tick (optional)"""
        pass

    @abstractmethod
    def on_bar(self, dataframe):
        """Called when a new candle closes"""
        pass