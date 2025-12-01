# trading/live_trading_bot.py
# Key changes to integrate trend detection

from fusion_trend_detector import FusionFXTrendDetector
from models.fusion import FusionNet
from models.lstm import LSTMModel
from models.vit import ViTExtractor
from models.yolo_detector import YOLOPatternDetector
from trading.enhanced_signal_engine import EnhancedSignalEngine
from trading.mt5_connector import connect, get_candles, send_order
import pandas as pd
import torch
import time
import logging

def load_production_models():
    """Load all models including trend detector"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Your existing models
    lstm = LSTMModel().to(device).eval()
    vit = ViTExtractor().to(device).eval()
    yolo = YOLOPatternDetector()
    fusion = FusionNet().to(device).eval()
    
    # Load weights
    lstm.load_state_dict(torch.load("models/lstm_best.pt"))
    fusion.load_state_dict(torch.load("models/fusion_best.pt"))
    # ... vit weights ...
    
    # NEW: Initialize trend detector
    trend_detector = FusionFXTrendDetector(ml_model=None)  # Or pass fusion if compatible
    
    # NEW: Initialize enhanced signal engine
    signal_engine = EnhancedSignalEngine(
        pattern_fusion_model=fusion,
        trend_detector=trend_detector
    )
    
    return lstm, vit, yolo, fusion, trend_detector, signal_engine, device

def live_bot():
    print("🚀 Starting Enhanced Hybrid Trading System with FTDM-V1...")
    
    if not connect(ACCOUNT, PASSWORD, SERVER):
        print("❌ MT5 Connection Failed")
        return
    
    risk_manager = RiskManager(risk_per_trade=0.01)
    lstm, vit, yolo, fusion, trend_detector, signal_engine, device = load_production_models()
    
    current_candle_time = None
    
    while True:
        try:
            # Get multi-timeframe data for trend detection
            candles_h4 = get_candles(SYMBOL, "H4", n=200)
            candles_h1 = get_candles(SYMBOL, "H1", n=200)
            candles_m15 = get_candles(SYMBOL, "M15", n=200)
            
            df_h4 = pd.DataFrame(candles_h4)
            df_h1 = pd.DataFrame(candles_h1)
            df_m15 = pd.DataFrame(candles_m15)
            
            # Check for new H1 candle
            last_time = df_h1['time'].iloc[-1]
            if current_candle_time == last_time:
                time.sleep(1)
                continue
            
            current_candle_time = last_time
            print(f"🔎 Analyzing candle: {datetime.fromtimestamp(last_time)}")
            
            # Pattern prediction (your existing models)
            with torch.no_grad():
                lstm_v, vit_v, yolo_v = build_features(df_h1, lstm, vit, yolo)
                lstm_v, vit_v, yolo_v = lstm_v.to(device), vit_v.to(device), yolo_v.to(device)
                
                preds = fusion(lstm_v, vit_v, yolo_v)
                probs = preds.softmax(dim=1).cpu().numpy()[0]
            
            # NEW: Enhanced signal generation with trend context
            dfs_dict = {'H4': df_h4, 'H1': df_h1, 'M15': df_m15}
            signal_result = signal_engine.generate_signal(probs, dfs_dict, threshold=0.70)
            
            signal = signal_result['signal']
            confidence = signal_result['confidence']
            reason = signal_result['reason']
            trend_analysis = signal_result['trend_analysis']
            
            logging.info(f"Signal: {signal} | Confidence: {confidence:.2f} | Reason: {reason}")
            logging.info(f"Trend: {trend_analysis['trend_name']} | Strength: {trend_analysis['trend_strength']:.1f}")
            
            if signal != "NO_TRADE":
                # Risk calculation
                account_info = mt5.account_info()
                balance = account_info.balance
                current_price = mt5.symbol_info_tick(SYMBOL).ask if signal == "BUY" else mt5.symbol_info_tick(SYMBOL).bid
                atr = risk_manager.calculate_volatility(df_h1)
                
                # Adjust position size based on trend strength
                base_risk = 0.01
                if trend_analysis['confidence'] > 0.75:
                    adjusted_risk = base_risk * 1.3  # Increase size for strong trends
                elif trend_analysis['confidence'] < 0.5:
                    adjusted_risk = base_risk * 0.7  # Decrease size for weak trends
                else:
                    adjusted_risk = base_risk
                
                risk_manager.risk = adjusted_risk
                vol, sl, tp = risk_manager.get_trade_params(balance, current_price, atr, signal)
                
                # Execute
                order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
                result = send_order(SYMBOL, vol, order_type, sl, tp)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"✅ Trade Executed: {signal} @ {current_price}")
                    print(f"   Trend Context: {trend_analysis['trend_name']}")
                    print(f"   Confidence: Pattern={signal_result['pattern_prediction']['confidence']:.2f}, Trend={trend_analysis['confidence']:.2f}")
                    logging.info(f"Trade Executed: {result}")
                else:
                    print(f"❌ Trade Failed: {result.comment}")
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            logging.error(f"Error: {e}")
            time.sleep(5)
        
        time.sleep(10)

if __name__ == "__main__":
    live_bot()