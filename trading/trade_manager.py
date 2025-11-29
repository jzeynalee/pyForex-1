# EXECUTE TRADE
# trading/trade_manager.py

from trading.mt5_connector import send_order
from utils.config import *
import MetaTrader5 as mt5

def execute_trade(signal, sl, tp):
    if signal == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
    else:
        order_type = mt5.ORDER_TYPE_SELL
    
    result = send_order(SYMBOL, LOT_SIZE, order_type, sl, tp)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("[ERROR] Trade execution failed")
    else:
        print(f"[TRADE] {signal} executed at {result.price}")
