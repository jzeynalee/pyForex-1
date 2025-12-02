# trading/live_trading_bot.py
"""
Live trading bot with integrated trend detection.

Key changes from original:
- Loads TrendClassifier for FTDM-V1 Step 4
- Uses enhanced signal generation with trend context
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging
import time
import pandas as pd
import torch

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

# Model paths
MODEL_DIR = Path(__file__).parent.parent / 'models'
LSTM_MODEL_PATH = MODEL_DIR / 'lstm_best.pt'
FUSION_MODEL_PATH = MODEL_DIR / 'fusion_best.pt'
TREND_CLASSIFIER_PATH = MODEL_DIR / 'trend_classifier.joblib'


def load_production_models():
    """
    Load all models including trend detector with TrendClassifier.
    
    Returns:
        Tuple of (lstm, vit, yolo, fusion, trend_detector, signal_engine, device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Import model classes
    from models.fusion import FusionNet
    from models.lstm import LSTMModel
    from models.vit import ViTExtractor
    from models.yolo_detector import YOLOPatternDetector
    from trading.enhanced_signal_engine import FusionFXTrendDetector
    
    # Initialize models
    lstm = LSTMModel().to(device).eval()
    vit = ViTExtractor().to(device).eval()
    yolo = YOLOPatternDetector()
    fusion = FusionNet().to(device).eval()
    
    # Load weights
    if LSTM_MODEL_PATH.exists():
        lstm.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
        logger.info(f"Loaded LSTM weights from {LSTM_MODEL_PATH}")
    
    if FUSION_MODEL_PATH.exists():
        fusion.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=device))
        logger.info(f"Loaded Fusion weights from {FUSION_MODEL_PATH}")
    
    # Load TrendClassifier for Step 4
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
            "Step 4 will use neutral defaults. "
            "Run 'python scripts/train_trend_classifier.py --synthetic' to create one."
        )
    
    # Initialize trend detector with TrendClassifier
    trend_detector = FusionFXTrendDetector(ml_model=trend_classifier)
    
    # Signal engine wraps trend detector
    signal_engine = EnhancedSignalEngine(
        pattern_fusion_model=fusion,
        trend_detector=trend_detector
    )
    
    return lstm, vit, yolo, fusion, trend_detector, signal_engine, device


class EnhancedSignalEngine:
    """
    Combines pattern predictions with trend context for signal generation.
    """
    
    def __init__(self, pattern_fusion_model, trend_detector):
        self.fusion = pattern_fusion_model
        self.trend_detector = trend_detector
    
    def generate_signal(
        self,
        pattern_probs: list,
        dfs_dict: dict,
        threshold: float = 0.70,
    ) -> dict:
        """
        Generate trading signal from pattern predictions and trend context.
        
        Args:
            pattern_probs: [P(BUY), P(SELL), P(HOLD)] from Fusion model
            dfs_dict: {'H4': df, 'H1': df, 'M15': df}
            threshold: Minimum confidence to trade
        
        Returns:
            Dict with signal, confidence, reason, and trend analysis
        """
        # Get trend analysis
        trend_analysis = self.trend_detector.detect_trend(dfs_dict)
        
        p_buy, p_sell, p_hold = pattern_probs
        
        # Determine pattern signal
        if max(p_buy, p_sell) < threshold:
            pattern_signal = "NO_TRADE"
            pattern_conf = max(p_buy, p_sell)
        elif p_buy > p_sell:
            pattern_signal = "BUY"
            pattern_conf = p_buy
        else:
            pattern_signal = "SELL"
            pattern_conf = p_sell
        
        # Apply trend filter
        trend_direction = trend_analysis['direction']
        trend_conf = trend_analysis['confidence']
        
        # Trading rules based on trend context
        signal = "NO_TRADE"
        reason = ""
        final_confidence = 0.0
        
        if pattern_signal == "NO_TRADE":
            reason = f"Pattern confidence too low ({pattern_conf:.2f} < {threshold})"
            
        elif trend_direction == "SIDEWAYS":
            # In sideways markets, require higher pattern confidence
            if pattern_conf >= 0.80:
                signal = pattern_signal
                final_confidence = pattern_conf * 0.8  # Reduce confidence
                reason = f"Sideways market but strong pattern ({pattern_conf:.2f})"
            else:
                reason = f"Sideways market, pattern not strong enough"
                
        elif pattern_signal == "BUY" and trend_direction == "BULLISH":
            # Aligned: pattern and trend both bullish
            signal = "BUY"
            final_confidence = (pattern_conf + trend_conf) / 2
            reason = f"Aligned bullish (pattern={pattern_conf:.2f}, trend={trend_conf:.2f})"
            
        elif pattern_signal == "SELL" and trend_direction == "BEARISH":
            # Aligned: pattern and trend both bearish
            signal = "SELL"
            final_confidence = (pattern_conf + trend_conf) / 2
            reason = f"Aligned bearish (pattern={pattern_conf:.2f}, trend={trend_conf:.2f})"
            
        elif pattern_signal == "BUY" and trend_direction == "BEARISH":
            # Counter-trend: be cautious
            if pattern_conf >= 0.85 and trend_conf < 0.6:
                signal = "BUY"
                final_confidence = pattern_conf * 0.7
                reason = f"Counter-trend BUY (weak bear trend)"
            else:
                reason = f"Counter-trend BUY rejected (trend too strong)"
                
        elif pattern_signal == "SELL" and trend_direction == "BULLISH":
            # Counter-trend: be cautious
            if pattern_conf >= 0.85 and trend_conf < 0.6:
                signal = "SELL"
                final_confidence = pattern_conf * 0.7
                reason = f"Counter-trend SELL (weak bull trend)"
            else:
                reason = f"Counter-trend SELL rejected (trend too strong)"
        
        return {
            'signal': signal,
            'confidence': final_confidence,
            'reason': reason,
            'trend_analysis': trend_analysis,
            'pattern_prediction': {
                'signal': pattern_signal,
                'confidence': pattern_conf,
                'probabilities': pattern_probs,
            }
        }


class RiskManager:
    """Simple risk manager for position sizing."""
    
    def __init__(self, risk_per_trade: float = 0.01):
        self.risk = risk_per_trade
    
    def calculate_volatility(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR for volatility."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(period).mean().iloc[-1]
    
    def get_trade_params(
        self,
        balance: float,
        current_price: float,
        atr: float,
        direction: str,
    ) -> tuple:
        """
        Calculate volume, SL, TP.
        
        Returns:
            Tuple of (volume, sl_price, tp_price)
        """
        risk_amount = balance * self.risk
        
        # SL = 1.5 ATR, TP = 3 ATR
        sl_dist = atr * 1.5
        tp_dist = atr * 3.0
        
        if direction == "BUY":
            sl = current_price - sl_dist
            tp = current_price + tp_dist
        else:
            sl = current_price + sl_dist
            tp = current_price - tp_dist
        
        # Calculate volume (simplified)
        pip_value = 10.0  # $10 per pip per lot
        pips_at_risk = sl_dist / 0.0001
        volume = risk_amount / (pips_at_risk * pip_value)
        volume = max(0.01, min(1.0, round(volume, 2)))
        
        return volume, sl, tp


def build_features(df, lstm, vit, yolo):
    """
    Build feature vectors from price data.
    
    This is a placeholder - implement based on your actual model inputs.
    """
    import torch
    
    # LSTM features from price sequence
    seq_len = min(60, len(df))
    price_seq = df[['open', 'high', 'low', 'close']].tail(seq_len).values
    price_seq = torch.tensor(price_seq, dtype=torch.float32).unsqueeze(0)
    
    # ViT features from chart image (placeholder)
    vit_features = torch.zeros(1, 768)
    
    # YOLO pattern features (placeholder)
    yolo_features = torch.zeros(1, 256)
    
    return price_seq, vit_features, yolo_features


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
        
        rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, n)
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
        "comment": "PyForex Enhanced",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    return mt5.order_send(request)


def live_bot():
    """Main trading loop."""
    print("🚀 Starting Enhanced Hybrid Trading System with FTDM-V1...")
    
    if not connect(ACCOUNT, PASSWORD, SERVER):
        print("❌ MT5 Connection Failed")
        return
    
    import MetaTrader5 as mt5
    
    risk_manager = RiskManager(risk_per_trade=0.01)
    lstm, vit, yolo, fusion, trend_detector, signal_engine, device = load_production_models()
    
    current_candle_time = None
    
    while True:
        try:
            # Get multi-timeframe data for trend detection
            candles_h4 = get_candles(SYMBOL, "H4", n=200)
            candles_h1 = get_candles(SYMBOL, "H1", n=200)
            candles_m15 = get_candles(SYMBOL, "M15", n=200)
            
            if not candles_h1:
                logger.warning("No H1 data received")
                time.sleep(10)
                continue
            
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
            
            # Pattern prediction (Deep Learning models)
            with torch.no_grad():
                lstm_v, vit_v, yolo_v = build_features(df_h1, lstm, vit, yolo)
                lstm_v = lstm_v.to(device)
                vit_v = vit_v.to(device)
                yolo_v = yolo_v.to(device)
                
                preds = fusion(lstm_v, vit_v, yolo_v)
                probs = preds.softmax(dim=1).cpu().numpy()[0]
            
            # Enhanced signal generation with trend context
            dfs_dict = {'H4': df_h4, 'H1': df_h1, 'M15': df_m15}
            signal_result = signal_engine.generate_signal(probs, dfs_dict, threshold=0.70)
            
            signal = signal_result['signal']
            confidence = signal_result['confidence']
            reason = signal_result['reason']
            trend_analysis = signal_result['trend_analysis']
            
            logger.info(f"Signal: {signal} | Confidence: {confidence:.2f} | Reason: {reason}")
            logger.info(
                f"Trend: {trend_analysis['trend_name']} | "
                f"Strength: {trend_analysis['trend_strength']:.1f} | "
                f"ML Conf: {trend_analysis['details'].get('ml_confidence', 0):.2f}"
            )
            
            if signal != "NO_TRADE":
                # Risk calculation
                account_info = mt5.account_info()
                balance = account_info.balance
                tick = mt5.symbol_info_tick(SYMBOL)
                current_price = tick.ask if signal == "BUY" else tick.bid
                atr = risk_manager.calculate_volatility(df_h1)
                
                # Adjust position size based on trend confidence
                base_risk = 0.01
                trend_conf = trend_analysis['confidence']
                
                if trend_conf > 0.75:
                    adjusted_risk = base_risk * 1.3  # Increase for strong trends
                elif trend_conf < 0.5:
                    adjusted_risk = base_risk * 0.7  # Decrease for weak trends
                else:
                    adjusted_risk = base_risk
                
                risk_manager.risk = adjusted_risk
                vol, sl, tp = risk_manager.get_trade_params(
                    balance, current_price, atr, signal
                )
                
                # Execute order
                order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
                result = send_order(SYMBOL, vol, order_type, sl, tp)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"✅ Trade Executed: {signal} @ {current_price}")
                    print(f"   Trend Context: {trend_analysis['trend_name']}")
                    print(
                        f"   Confidence: Pattern={signal_result['pattern_prediction']['confidence']:.2f}, "
                        f"Trend={trend_conf:.2f}"
                    )
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
    live_bot()