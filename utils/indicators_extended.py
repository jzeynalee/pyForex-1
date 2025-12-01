# utils/indicators_extended.py
"""
Extended technical indicators for trend detection
Includes ADX, Donchian, EMA slopes, etc.
"""

import pandas as pd
import numpy as np

class TrendIndicators:
    """Collection of trend-specific technical indicators"""
    
    @staticmethod
    def calculate_adx(df, period=14):
        """Calculate Average Directional Index"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smoothed indicators
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def calculate_ema_slope(df, period=20, lookback=5):
        """Calculate EMA slope (rate of change)"""
        ema = df['close'].ewm(span=period, adjust=False).mean()
        slope = (ema - ema.shift(lookback)) / lookback
        
        # Normalize slope
        slope_normalized = slope / df['close'] * 100
        
        return ema, slope_normalized
    
    @staticmethod
    def calculate_donchian(df, period=20):
        """Calculate Donchian Channel"""
        high_channel = df['high'].rolling(window=period).max()
        low_channel = df['low'].rolling(window=period).min()
        mid_channel = (high_channel + low_channel) / 2
        
        # Direction: 1 if price above mid, -1 if below
        direction = np.where(df['close'] > mid_channel, 1, -1)
        
        return high_channel, low_channel, mid_channel, direction
    
    @staticmethod
    def calculate_vwap(df):
        """Calculate Volume Weighted Average Price"""
        # Note: Assumes 'tick_volume' as volume
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['tick_volume']).cumsum() / df['tick_volume'].cumsum()
        
        return vwap
    
    @staticmethod
    def calculate_volatility_compression(df, atr_period=14, compression_window=50):
        """
        Detect volatility compression (potential breakout zones)
        Returns compression ratio (0-1, lower = more compressed)
        """
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=atr_period).mean()
        
        # Compare current ATR to historical ATR
        atr_ma = atr.rolling(window=compression_window).mean()
        compression_ratio = atr / atr_ma
        
        return compression_ratio