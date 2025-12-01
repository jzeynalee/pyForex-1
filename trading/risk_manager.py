# ATR (for SL/TP calculation)
# trading/risk_manager.py

# trading/risk_manager.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime

class RiskManager:
    def __init__(self, 
                 account_balance,
                 max_daily_loss_pct=0.03,  # Max 3% loss per day
                 risk_per_trade_pct=0.01,  # Max 1% risk per trade
                 max_drawdown_hard_stop=0.10): # Stop bot if 10% down total
        
        self.starting_balance = account_balance
        self.daily_start_balance = account_balance
        self.last_day = datetime.now().day
        
        self.max_daily_loss = max_daily_loss_pct
        self.risk_per_trade = risk_per_trade_pct
        self.max_dd = max_drawdown_hard_stop

    def check_risk_limits(self, current_balance, current_equity):
        """
        Returns True if trading is allowed, False if limits breached.
        """
        # 1. Reset Daily PnL if new day
        if datetime.now().day != self.last_day:
            self.daily_start_balance = current_balance
            self.last_day = datetime.now().day
            print(f"[RISK] New Day. Reset Daily Balance Tracker: {self.daily_start_balance}")

        # 2. Check Daily Loss
        daily_loss = (self.daily_start_balance - current_equity) / self.daily_start_balance
        if daily_loss >= self.max_daily_loss:
            print(f"[RISK ALERT] Daily loss limit hit! ({daily_loss*100:.2f}%)")
            return False

        # 3. Check Total Max Drawdown
        total_dd = (self.starting_balance - current_equity) / self.starting_balance
        if total_dd >= self.max_dd:
            print(f"[RISK ALERT] Max Drawdown hit! Stopping Bot. ({total_dd*100:.2f}%)")
            return False

        return True

    def calculate_volatility(self, df, period=14):
        # (Same ATR Logic as before)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean().iloc[-1]

    def get_trade_params(self, symbol, current_price, atr, direction, current_balance):
        """
        Calculates exact lot size based on Risk % and Stop Loss distance.
        Uses MT5 SymbolInfo for TickValue and ContractSize.
        """
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"[RISK] Symbol {symbol} not found")
            return 0.0, 0.0, 0.0

        # SL / TP Distances
        sl_dist = atr * 1.5
        tp_dist = atr * 3.0

        # Calculate Price Levels
        if direction == "BUY":
            sl_price = current_price - sl_dist
            tp_price = current_price + tp_dist
        else:
            sl_price = current_price + sl_dist
            tp_price = current_price - tp_dist

        # Position Sizing Logic
        risk_amount = current_balance * self.risk_per_trade
        
        # Formula: Volume = Risk / (SL_Points * TickValue)
        # We use TickValue because it accounts for Exchange Rate (e.g. USDJPY) and Contract Size
        # TickSize is the smallest change (e.g. 0.00001)
        
        sl_points = sl_dist / symbol_info.point # Convert price dist to points
        tick_value = symbol_info.trade_tick_value # Value of 1 point change for 1.0 lot
        
        if tick_value == 0:
            volume = 0.01 # Fallback
        else:
            volume = risk_amount / (sl_points * tick_value)

        # Normalize Volume
        step = symbol_info.volume_step
        volume = round(volume / step) * step
        volume = max(volume, symbol_info.volume_min)
        volume = min(volume, symbol_info.volume_max)

        return volume, sl_price, tp_price