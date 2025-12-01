# utils/swing_detector.py
"""
Swing Point Detection using ATR-based ZigZag
Implements Step 1 structural foundation
"""

import pandas as pd
import numpy as np

class SwingDetector:
    """
    Detects swing highs and lows using ATR-based deviation threshold.
    Confirms swings only after N subsequent candles respect the extremum.
    """
    
    def __init__(self, atr_multiplier=3.5, confirmation_candles=2, atr_period=14):
        self.atr_mult = atr_multiplier
        self.confirm_candles = confirmation_candles
        self.atr_period = atr_period
    
    def calculate_atr(self, df):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()
        
        return atr
    
    def detect_swings(self, df):
        """
        Main swing detection algorithm
        Returns DataFrame with swing_high and swing_low columns
        """
        df = df.copy()
        atr = self.calculate_atr(df)
        threshold = atr * self.atr_mult
        
        df['swing_high'] = False
        df['swing_low'] = False
        df['swing_high_price'] = np.nan
        df['swing_low_price'] = np.nan
        
        # Scan for potential swing points
        for i in range(self.confirm_candles, len(df) - self.confirm_candles):
            current_high = df['high'].iloc[i]
            current_low = df['low'].iloc[i]
            
            # Check swing high (local maximum)
            is_swing_high = True
            for j in range(1, self.confirm_candles + 1):
                if df['high'].iloc[i - j] > current_high or \
                   df['high'].iloc[i + j] > current_high:
                    is_swing_high = False
                    break
            
            # Check swing low (local minimum)
            is_swing_low = True
            for j in range(1, self.confirm_candles + 1):
                if df['low'].iloc[i - j] < current_low or \
                   df['low'].iloc[i + j] < current_low:
                    is_swing_low = False
                    break
            
            # Validate against ATR threshold
            if is_swing_high and i > 0:
                prev_swing_idx = df[df['swing_high']].index
                if len(prev_swing_idx) > 0:
                    last_swing = prev_swing_idx[-1]
                    if abs(current_high - df.loc[last_swing, 'high']) >= threshold.iloc[i]:
                        df.loc[df.index[i], 'swing_high'] = True
                        df.loc[df.index[i], 'swing_high_price'] = current_high
                else:
                    df.loc[df.index[i], 'swing_high'] = True
                    df.loc[df.index[i], 'swing_high_price'] = current_high
            
            if is_swing_low and i > 0:
                prev_swing_idx = df[df['swing_low']].index
                if len(prev_swing_idx) > 0:
                    last_swing = prev_swing_idx[-1]
                    if abs(current_low - df.loc[last_swing, 'low']) >= threshold.iloc[i]:
                        df.loc[df.index[i], 'swing_low'] = True
                        df.loc[df.index[i], 'swing_low_price'] = current_low
                else:
                    df.loc[df.index[i], 'swing_low'] = True
                    df.loc[df.index[i], 'swing_low_price'] = current_low
        
        return df
    
    def classify_structure(self, df):
        """
        Classify swing structure into HH/HL/LH/LL patterns
        Returns: 'bullish', 'bearish', 'mixed'
        """
        swing_highs = df[df['swing_high']]['swing_high_price'].values
        swing_lows = df[df['swing_low']]['swing_low_price'].values
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'mixed', 0.0
        
        # Check for Higher Highs and Higher Lows (bullish)
        hh_count = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i] > swing_highs[i-1])
        hl_count = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i] > swing_lows[i-1])
        
        # Check for Lower Lows and Lower Highs (bearish)
        ll_count = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i] < swing_lows[i-1])
        lh_count = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i] < swing_highs[i-1])
        
        total_swings = (len(swing_highs) - 1) + (len(swing_lows) - 1)
        
        if total_swings == 0:
            return 'mixed', 0.0
        
        bullish_score = (hh_count + hl_count) / total_swings
        bearish_score = (ll_count + lh_count) / total_swings
        
        if bullish_score > 0.6:
            return 'bullish', bullish_score
        elif bearish_score > 0.6:
            return 'bearish', bearish_score
        else:
            return 'mixed', max(bullish_score, bearish_score)