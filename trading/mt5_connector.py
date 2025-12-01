# trading/mt5_connector.py
import MetaTrader5 as mt5
import time
from utils.config import *

def ensure_connected(func):
    """Decorator to check connection before MT5 calls"""
    def wrapper(*args, **kwargs):
        if not mt5.terminal_info():
            print("[MT5] Connection lost, reconnecting...")
            if not mt5.initialize() or not mt5.login(MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER):
                print("[MT5] Reconnection Failed!")
                return None
        return func(*args, **kwargs)
    return wrapper

@ensure_connected
def get_candles(symbol, timeframe_str, n=100):
    tf_map = {"H1": mt5.TIMEFRAME_H1, "M15": mt5.TIMEFRAME_M15}
    rates = mt5.copy_rates_from_pos(symbol, tf_map.get(timeframe_str, mt5.TIMEFRAME_H1), 0, n)
    if rates is None:
        print(f"[MT5] Failed to get candles for {symbol}")
        return None
    return rates

@ensure_connected
def send_order(symbol, volume, order_type, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid,
        "sl": float(sl),
        "tp": float(tp),
        "deviation": 20,
        "magic": 123456,
        "comment": "PyForex_Bot_V2",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Retry Loop for Execution
    for i in range(3):
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            return result
        elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
            time.sleep(0.5)
            continue # Update price and retry
        else:
            print(f"[MT5 ERROR] Order Failed: {result.comment} ({result.retcode})")
            break
            
    return result