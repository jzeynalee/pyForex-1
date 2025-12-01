# trading/mt5_connector.py

import MetaTrader5 as mt5
import pandas as pd
import logging
import time
from utils.config import settings

class MT5Connector:
    """
    Class-based connector to manage MT5 state and connection.
    """
    def __init__(self):
        self.connected = False
        self.tf_map = {
            "H1": mt5.TIMEFRAME_H1,
            "M15": mt5.TIMEFRAME_M15,
            "M5": mt5.TIMEFRAME_M5
        }

    def connect(self):
        """Initializes connection to MT5 terminal"""
        if not mt5.initialize(path=settings.MT5_PATH) if settings.MT5_PATH else mt5.initialize():
            logging.error(f"MT5 Init failed: {mt5.last_error()}")
            return False
            
        authorized = mt5.login(
            login=settings.MT5_ACCOUNT, 
            password=settings.MT5_PASSWORD, 
            server=settings.MT5_SERVER
        )
        
        if authorized:
            self.connected = True
            logging.info(f"Connected to MT5 Account: {settings.MT5_ACCOUNT}")
        else:
            logging.error(f"MT5 Login failed: {mt5.last_error()}")
            
        return self.connected

    def get_data(self, n=100):
        """Fetches latest candles as DataFrame"""
        if not self.connected: 
            if not self.connect(): return pd.DataFrame()
        
        timeframe = self.tf_map.get(settings.TIMEFRAME, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(settings.SYMBOL, timeframe, 0, n)
        
        if rates is None:
            logging.warning(f"Failed to fetch data for {settings.SYMBOL}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df

    def execute_order(self, signal: str, volume: float, sl: float, tp: float):
        """Executes a trade order"""
        if not self.connected: self.connect()

        action = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
        
        # Get current price
        tick = mt5.symbol_info_tick(settings.SYMBOL)
        if not tick:
            logging.error("Failed to get tick data")
            return None
            
        price = tick.ask if signal == "BUY" else tick.bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": settings.SYMBOL,
            "volume": float(volume),
            "type": action,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "magic": settings.MAGIC_NUMBER,
            "comment": "PyForex Hybrid",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        return result