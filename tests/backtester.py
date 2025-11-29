# FULL BACKTESTER (Backtrader)
# File: backtest/backtest_engine.py

import backtrader as bt
import pandas as pd
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.fusion import FusionNet
from models.lstm import LSTMModel
from models.vit import ViTExtractor
from models.yolo_detector import YOLOPatternDetector
from inference.build_features import build_features
from trading.signal_engine import generate_signal

class DLHybridStrategy(bt.Strategy):
    params = (
        ('risk_mgr', None),
    )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.data_open = self.datas[0].open
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low
        self.data_vol = self.datas[0].volume
        
        # Load Models (Move to GPU if available, else CPU for backtest safety)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Backtest running on: {self.device}")

        self.lstm = LSTMModel().to(self.device).eval()
        self.vit = ViTExtractor().to(self.device).eval()
        self.yolo = YOLOPatternDetector() # Ultralytics handles device internally
        self.fusion = FusionNet().to(self.device).eval()
        
        # Load weights (Placeholder paths)
        # self.lstm.load_state_dict(torch.load("models/weights/lstm_best.pt"))
        # self.fusion.load_state_dict(torch.load("models/weights/fusion_best.pt"))

    def next(self):
        # 1. Get History (Last 60 candles for LSTM/ViT)
        if len(self) < 60: return

        # Extract data as DataFrame for the feature builder
        opens = list(self.data_open.get(size=60))
        highs = list(self.data_high.get(size=60))
        lows = list(self.data_low.get(size=60))
        closes = list(self.data_close.get(size=60))
        vols = list(self.data_vol.get(size=60))

        df_window = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 'close': closes, 'tick_volume': vols
        })

        # 2. Inference
        with torch.no_grad():
            lstm_v, vit_v, yolo_v = build_features(df_window, self.lstm, self.vit, self.yolo)
            # Ensure tensors are on correct device
            lstm_v, vit_v = lstm_v.to(self.device), vit_v.to(self.device)
            yolo_v = yolo_v.to(self.device)
            
            preds = self.fusion(lstm_v, vit_v, yolo_v)
            probs = preds.softmax(dim=1).cpu().numpy()

        # 3. Signal
        signal = generate_signal(probs, threshold=0.7)

        # 4. Execution Logic
        if not self.position:
            if signal == "BUY":
                # Calculate size using Risk Manager (Simulated balance)
                # Volatility calculation would go here
                self.buy() # Simplified for backtest
            elif signal == "SELL":
                self.sell()
        else:
            # Simple exit logic (Reverse or TP/SL handled by Backtrader orders)
            if self.position.size > 0 and signal == "SELL":
                self.close()
            elif self.position.size < 0 and signal == "BUY":
                self.close()

def run_backtest():
    cerebro = bt.Cerebro()
    
    # Load Data
    data = bt.feeds.GenericCSVData(
        dataname='data/raw/eurusd.csv',
        dtformat=('%Y-%m-%d %H:%M:%S'),
        openinterest=-1
    )
    cerebro.adddata(data)
    
    cerebro.addstrategy(DLHybridStrategy)
    cerebro.broker.setcash(10000.0)
    
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

if __name__ == '__main__':
    run_backtest()