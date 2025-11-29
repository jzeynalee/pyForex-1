# COMPLETE LIVE BOT LOOP — FINAL BOT
# trading/live_trading_bot.py

import time
import logging
import pandas as pd
import torch
from datetime import datetime

# Import modules
from inference.build_features import build_features
from trading.signal_engine import generate_signal
from trading.mt5_connector import connect, get_candles, send_order
from trading.risk_manager import RiskManager
from models.fusion import FusionNet
# ... import other models ...

# Config
SYMBOL = "EURUSD"
TIMEFRAME = 16385 # mt5.TIMEFRAME_H1
ACCOUNT = 123456
PASSWORD = "password"
SERVER = "Broker-Server"

# Logging setup
logging.basicConfig(filename='trading_log.log', level=logging.INFO, format='%(asctime)s %(message)s')

def load_production_models():
    # Load your models here (same as backtest)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ... Initialize and load_state_dict ...
    return lstm, vit, yolo, fusion, device

def live_bot():
    print("🚀 Starting Hybrid Trading System...")
    
    # 1. Connect
    if not connect(ACCOUNT, PASSWORD, SERVER):
        print("❌ MT5 Connection Failed")
        return

    # 2. Init Modules
    risk_manager = RiskManager(risk_per_trade=0.01)
    lstm, vit, yolo, fusion, device = load_production_models()
    
    # 3. State Tracking
    current_candle_time = None

    while True:
        try:
            # Check connection
            # (MT5 connection check logic here)

            # Get Data
            candles = get_candles(SYMBOL, TIMEFRAME, n=100)
            df = pd.DataFrame(candles)
            
            # Check for new candle to avoid duplicate trades
            last_time = df['time'].iloc[-1]
            if current_candle_time == last_time:
                time.sleep(1)
                continue
            
            current_candle_time = last_time
            print(f"🔎 Analyzing candle: {datetime.fromtimestamp(last_time)}")

            # Inference
            with torch.no_grad():
                lstm_v, vit_v, yolo_v = build_features(df, lstm, vit, yolo)
                # Move to device
                lstm_v, vit_v, yolo_v = lstm_v.to(device), vit_v.to(device), yolo_v.to(device)
                
                preds = fusion(lstm_v, vit_v, yolo_v)
                probs = preds.softmax(dim=1).cpu().numpy()

            # Signal
            signal = generate_signal(probs, threshold=0.75)
            logging.info(f"Signal: {signal} | Probs: {probs}")

            if signal != "NO_TRADE":
                # Risk Calc
                account_info = mt5.account_info()
                balance = account_info.balance
                current_price = mt5.symbol_info_tick(SYMBOL).ask if signal == "BUY" else mt5.symbol_info_tick(SYMBOL).bid
                atr = risk_manager.calculate_volatility(df)
                
                vol, sl, tp = risk_manager.get_trade_params(balance, current_price, atr, signal)
                
                # Execute
                order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
                result = send_order(SYMBOL, vol, order_type, sl, tp)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"✅ Trade Executed: {signal} @ {current_price}")
                    logging.info(f"Trade Executed: {result}")
                else:
                    print(f"❌ Trade Failed: {result.comment}")

        except Exception as e:
            print(f"⚠️ Error: {e}")
            logging.error(f"Error: {e}")
            time.sleep(5)

        time.sleep(10) # Check freq