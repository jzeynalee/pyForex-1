# MT5 API connector
# trading/mt5_connector.py

import MetaTrader5 as mt5
from utils.config import *

def mt5_connect():
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize failed")
    if not mt5.login(MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER):
        raise RuntimeError("MT5 login failed")
    print("[MT5] Connected successfully.")

def get_candles(symbol, timeframe, n=200):
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
    }
    data = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, n)
    return data

def send_order(symbol, volume, order_type, sl, tp):
    request = {
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "sl": sl,
        "tp": tp,
        "deviation": 20
    }
    result = mt5.order_send(request)
    print("[ORDER]", result)
    return result
