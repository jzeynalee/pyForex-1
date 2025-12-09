# trading/live_trading_bot.py
"""
Live trading bot with integrated trend detection.

UPDATED:
- Uses TCN instead of LSTM for sequence modeling
- Full MTF integration with configurable profiles
- Enhanced signal generation with trend context
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging
import time
import pandas as pd
import torch
from typing import Optional, Dict, Any

from utils.config import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
SYMBOL = settings.SYMBOL
ACCOUNT = settings.MT5_ACCOUNT
PASSWORD = settings.MT5_PASSWORD
SERVER = settings.MT5_SERVER
PATH = settings.MT5_PATH

# Model paths - UPDATED: TCN instead of LSTM
MODEL_DIR = Path(__file__).parent.parent / 'models'
TCN_MODEL_PATH = MODEL_DIR / 'tcn_best.pt'
FUSION_MODEL_PATH = MODEL_DIR / 'fusion_best.pt'
TREND_CLASSIFIER_PATH = MODEL_DIR / 'trend_classifier.joblib'


def load_production_models(
    tcn_profile: str = "INTRADAY",
    mtf_profile: str = "SWING",
):
    """
    Load all models including TCN and MTF trend detector.
    
    Args:
        tcn_profile: TCN profile ('SCALP', 'INTRADAY', 'SWING')
        mtf_profile: MTF analysis profile
    
    Returns:
        Dict with all loaded components
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Import model classes
    from models.fusion import FusionNet
    from models.tcn import TCNModel  # UPDATED: TCN instead of LSTM
    from models.vit import ViTExtractor
    from models.yolo_detector import YOLOPatternDetector
    from trading.decision_engine import MTFDecisionEngine
    
    # Initialize models - UPDATED: Use TCN
    tcn = TCNModel.from_profile(tcn_profile).to(device).eval()
    vit = ViTExtractor().to(device).eval()
    yolo = YOLOPatternDetector()
    fusion = FusionNet().to(device).eval()
    
    # Load weights
    if TCN_MODEL_PATH.exists():
        tcn.load_state_dict(torch.load(TCN_MODEL_PATH, map_location=device))
        logger.info(f"Loaded TCN weights from {TCN_MODEL_PATH}")
    else:
        logger.warning(f"TCN weights not found at {TCN_MODEL_PATH}")
    
    if FUSION_MODEL_PATH.exists():
        fusion.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=device))
        logger.info(f"Loaded Fusion weights from {FUSION_MODEL_PATH}")
    
    # Load TrendClassifier for Step 4 (optional)
    trend_classifier = None
    if TREND_CLASSIFIER_PATH.exists():
        try:
            from models.trend_classifier import TrendClassifier
            trend_classifier = TrendClassifier.load(TREND_CLASSIFIER_PATH)
            logger.info(f"Loaded TrendClassifier from {TREND_CLASSIFIER_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load TrendClassifier: {e}")
    else:
        logger.warning(
            f"TrendClassifier not found at {TREND_CLASSIFIER_PATH}. "
            "Step 4 will use neutral defaults."
        )
    
    # Initialize MTF-enabled decision engine
    decision_engine = MTFDecisionEngine(
        profile=mtf_profile,
        ml_model=trend_classifier,
    )
    
    return {
        'tcn': tcn,
        'vit': vit,
        'yolo': yolo,
        'fusion': fusion,
        'decision_engine': decision_engine,
        'device': device,
    }


class RiskManager:
    """Simple risk manager for position sizing."""
    
    def __init__(self, risk_per_trade: float = 0.01):
        self.risk = risk_per_trade
    
    def calculate_volatility(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for volatility."""
        import numpy as np
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        return float(np.mean(tr[-period:]))
    
    def get_trade_params(
        self,
        balance: float,
        current_price: float,
        atr: float,
        direction: str,
        confidence: float = 0.7,
    ) -> tuple:
        """
        Calculate volume, SL, TP.
        
        Args:
            balance: Account balance
            current_price: Current price
            atr: ATR value
            direction: 'BUY' or 'SELL'
            confidence: Signal confidence (adjusts position size)
        
        Returns:
            Tuple of (volume, sl_price, tp_price)
        """
        # Adjust risk based on confidence
        adjusted_risk = self.risk
        if confidence > 0.80:
            adjusted_risk = self.risk * 1.2
        elif confidence < 0.60:
            adjusted_risk = self.risk * 0.8
        
        risk_amount = balance * adjusted_risk
        
        # SL = 1.5 ATR, TP = 3 ATR
        sl_dist = atr * 1.5
        tp_dist = atr * 3.0
        
        if direction == "BUY":
            sl = current_price - sl_dist
            tp = current_price + tp_dist
        else:
            sl = current_price + sl_dist
            tp = current_price - tp_dist
        
        # Calculate volume
        pip_value = 10.0  # $10 per pip per lot
        pips_at_risk = sl_dist / 0.0001
        volume = risk_amount / (pips_at_risk * pip_value) if pips_at_risk > 0 else 0.01
        volume = max(0.01, min(1.0, round(volume, 2)))
        
        return volume, sl, tp


def build_features(df: pd.DataFrame, models: Dict, device: str):
    """
    Build feature vectors from price data.
    
    Args:
        df: Price DataFrame
        models: Dict with tcn, vit, yolo, fusion
        device: Torch device
    
    Returns:
        Tuple of (seq_features, vit_features, yolo_features)
    """
    import numpy as np
    
    # TCN features from price sequence
    seq_len = min(60, len(df))
    cols = ['open', 'high', 'low', 'close', 'tick_volume']
    price_seq = df[cols].tail(seq_len).values.astype(np.float32)
    
    # Normalize
    mean = price_seq.mean(axis=0)
    std = price_seq.std(axis=0) + 1e-8
    price_seq = (price_seq - mean) / std
    
    seq_tensor = torch.tensor(price_seq).float().unsqueeze(0).to(device)
    
    # ViT features from chart image (placeholder if not available)
    try:
        from utils.candle_to_image import candle_image, normalize_for_model
        img = candle_image(df.tail(60), target_size=224)
        img_norm = normalize_for_model(img, use_imagenet_stats=True)
        vit_tensor = torch.tensor(img_norm).float().unsqueeze(0).to(device)
    except ImportError:
        vit_tensor = torch.zeros(1, 3, 224, 224).to(device)
    
    # YOLO features
    yolo_features = models['yolo'].detect(
        np.zeros((224, 224, 3), dtype=np.uint8)
    )
    yolo_tensor = torch.tensor(yolo_features).float().unsqueeze(0).to(device)
    
    return seq_tensor, vit_tensor, yolo_tensor


def get_candles(symbol: str, timeframe: str, n: int = 200) -> list:
    """Fetch candles from MT5."""
    try:
        import MetaTrader5 as mt5
        
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
        }
        
        rates = mt5.copy_rates_from_pos(symbol, tf_map.get(timeframe, mt5.TIMEFRAME_H1), 0, n)
        return rates if rates is not None else []
        
    except ImportError:
        logger.warning("MT5 not available, returning empty data")
        return []


def connect(account: int, password: str, server: str) -> bool:
    """Connect to MT5."""
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            return False
        
        if account and password:
            return mt5.login(account, password=password, server=server)
        
        return True
        
    except ImportError:
        logger.error("MetaTrader5 not installed")
        return False


def send_order(symbol: str, volume: float, order_type: int, sl: float, tp: float):
    """Send order to MT5."""
    import MetaTrader5 as mt5
    
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 123456,
        "comment": "PyForex TCN+MTF",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    return mt5.order_send(request)


def live_bot(
    tcn_profile: str = "INTRADAY",
    mtf_profile: str = "SWING",
):
    """
    Main trading loop.
    
    Args:
        tcn_profile: TCN model profile
        mtf_profile: MTF analysis profile
    """
    print("🚀 Starting Enhanced Trading System (TCN + MTF)...")
    print(f"   TCN Profile: {tcn_profile}")
    print(f"   MTF Profile: {mtf_profile}")
    
    if not connect(ACCOUNT, PASSWORD, SERVER):
        print("❌ MT5 Connection Failed")
        return
    
    import MetaTrader5 as mt5
    
    risk_manager = RiskManager(risk_per_trade=0.01)
    models = load_production_models(tcn_profile, mtf_profile)
    device = models['device']
    
    current_candle_time = None
    
    while True:
        try:
            # Get multi-timeframe data
            if mtf_profile == "SCALP":
                tfs = ["M5", "M15", "H1"]
            elif mtf_profile == "SWING":
                tfs = ["H1", "H4", "D1"]
            else:  # INTRADAY
                tfs = ["M15", "H1", "H4"]
            
            dfs_dict = {}
            for tf in tfs:
                candles = get_candles(SYMBOL, tf, n=200)
                if candles is not None and len(candles) > 0:
                    dfs_dict[tf] = pd.DataFrame(candles)
                    dfs_dict[tf]['time'] = pd.to_datetime(dfs_dict[tf]['time'], unit='s')
            
            # Primary TF for candle check
            primary_tf = tfs[1]  # Middle timeframe
            if primary_tf not in dfs_dict:
                logger.warning(f"No {primary_tf} data received")
                time.sleep(10)
                continue
            
            df_primary = dfs_dict[primary_tf]
            
            # Check for new candle
            last_time = df_primary['time'].iloc[-1]
            if current_candle_time == last_time:
                time.sleep(1)
                continue
            
            current_candle_time = last_time
            print(f"🔎 Analyzing candle: {last_time}")
            
            # Pattern prediction (TCN + ViT + YOLO + Fusion)
            with torch.no_grad():
                seq_v, vit_v, yolo_v = build_features(df_primary, models, device)
                
                # Get TCN features
                tcn_feat = models['tcn'](seq_v, mode='features')
                
                # Get ViT features
                vit_feat = models['vit'](vit_v)
                
                # Fusion
                preds = models['fusion'](tcn_feat, vit_feat, yolo_v)
                probs = preds.softmax(dim=1).cpu().numpy()[0]
            
            # Get recommendation from MTF decision engine
            recommendation = models['decision_engine'].get_recommendation(
                pattern_probs=probs.tolist(),
                dfs_dict=dfs_dict,
            )
            
            signal = recommendation['signal']
            confidence = recommendation['confidence']
            reason = recommendation['reason']
            trend = recommendation['trend']
            
            logger.info(f"Signal: {signal} | Confidence: {confidence:.2f} | Trend: {trend}")
            logger.info(f"Reason: {reason}")
            
            if signal in ("BUY", "SELL"):
                # Risk calculation
                account_info = mt5.account_info()
                balance = account_info.balance
                tick = mt5.symbol_info_tick(SYMBOL)
                current_price = tick.ask if signal == "BUY" else tick.bid
                atr = risk_manager.calculate_volatility(df_primary)
                
                # Get trade parameters
                vol, sl, tp = risk_manager.get_trade_params(
                    balance, current_price, atr, signal, confidence
                )
                
                # Execute order
                order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
                result = send_order(SYMBOL, vol, order_type, sl, tp)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"✅ Trade Executed: {signal} @ {current_price}")
                    print(f"   Trend: {trend}")
                    print(f"   Confidence: {confidence:.2%}")
                    print(f"   Volume: {vol}, SL: {sl:.5f}, TP: {tp:.5f}")
                    logger.info(f"Trade Executed: {result}")
                else:
                    print(f"❌ Trade Failed: {result.comment}")
                    logger.error(f"Trade Failed: {result}")
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            break
            
        except Exception as e:
            print(f"⚠️ Error: {e}")
            logger.error(f"Error: {e}", exc_info=True)
            time.sleep(5)
        
        time.sleep(10)
    
    # Cleanup
    try:
        import MetaTrader5 as mt5
        mt5.shutdown()
    except:
        pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Trading Bot with TCN + MTF")
    parser.add_argument('--tcn-profile', type=str, default='INTRADAY',
                        choices=['SCALP', 'INTRADAY', 'SWING'],
                        help='TCN model profile')
    parser.add_argument('--mtf-profile', type=str, default='SWING',
                        choices=['SCALP', 'INTRADAY', 'SWING'],
                        help='MTF analysis profile')
    
    args = parser.parse_args()
    
    live_bot(
        tcn_profile=args.tcn_profile,
        mtf_profile=args.mtf_profile,
    )