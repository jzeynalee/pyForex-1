# ATR (for SL/TP calculation)
# trading/risk_manager.py
import numpy as np
import pandas as pd

class RiskManager:
    def __init__(self, risk_per_trade=0.01, atr_multiplier_sl=1.5, atr_multiplier_tp=3.0):
        """
        :param risk_per_trade: % of equity to risk per trade (default 1%)
        :param atr_multiplier_sl: Multiplier for Stop Loss
        :param atr_multiplier_tp: Multiplier for Take Profit
        """
        self.risk = risk_per_trade
        self.sl_mult = atr_multiplier_sl
        self.tp_mult = atr_multiplier_tp

    def calculate_volatility(self, df, period=14):
        """Calculates Average True Range (ATR)"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(period).mean().iloc[-1]
        return atr

    def get_trade_params(self, balance, price, atr, direction):
        """
        Returns: volume, sl_price, tp_price
        """
        # 1. Calculate SL distance
        sl_dist = atr * self.sl_mult
        tp_dist = atr * self.tp_mult

        # 2. Calculate Prices
        if direction == "BUY":
            sl_price = price - sl_dist
            tp_price = price + tp_dist
        else: # SELL
            sl_price = price + sl_dist
            tp_price = price - tp_dist

        # 3. Calculate Position Size (Volume)
        # Risk Amount = Balance * Risk%
        # Volume = Risk Amount / (SL Distance)
        # Note: This is a simplified calculation. MT5 requires standardization.
        risk_amount = balance * self.risk
        volume = risk_amount / sl_dist 
        
        # Normalize volume to 2 decimal places (standard for many brokers)
        # You often need to divide by contract size (e.g. 100,000 for FX) depending on broker API
        volume = round(volume / 100000, 2) # Assuming Standard Lot sizing for FX
        if volume < 0.01: volume = 0.01

        return volume, sl_price, tp_price