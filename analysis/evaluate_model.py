# analysis/evaluate_model.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, precision_score
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.feature_adapter import EnhancedDataLoaderV2
from training.train_lstm_enhanced import EnhancedLSTM, TOP_FEATURES

def evaluate_and_optimize():
    # Settings
    SEQ_LEN = 30
    HIDDEN_DIM = 64
    THRESHOLD = 0.05
    DATA_PATH = "data/raw/eurusd_latest.csv"
    MODEL_PATH = "models/weights/lstm_enhanced_best.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Evaluating & Optimizing: {MODEL_PATH}")

    # 1. Load Data
    loader = EnhancedDataLoaderV2(
        sequence_length=SEQ_LEN,
        trend_threshold=THRESHOLD,
        scaler_type='robust'
    )
    df = loader.load_csv(DATA_PATH)
    loader.feature_columns = [f for f in TOP_FEATURES if f in df.columns]
    
    # 2. Scale & Create Sequences
    train_scaled, test_scaled, val_scaled = loader.split_and_scale(df, split_ratio=0.8)
    X_test, y_test = loader.create_sequences(test_scaled, loader.test_close, SEQ_LEN)
    
    # Align Prices for Backtest
    test_close_prices = loader.test_close[SEQ_LEN:]
    test_close_prices = test_close_prices[:len(y_test)]
    market_returns = np.diff(test_close_prices) / test_close_prices[:-1]
    
    # 3. Load Model
    print(f"   Loading weights...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    input_dim = len(loader.feature_columns)
    
    model = EnhancedLSTM(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        num_classes=3,
        bidirectional=False
    ).to(device)
    
    # Robust State Dict Loading
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
    
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # 4. Get Probabilities (The Key Step)
    print("   Generating probabilities...")
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy() # Shape: (N, 3)

    # 5. Threshold Optimization Loop
    print("\n🔍 OPTIMIZING CONFIDENCE THRESHOLD")
    print("-" * 65)
    print(f"{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Return':<10} | {'Precision (Bull/Bear)':<20}")
    print("-" * 65)

    best_return = -999
    best_threshold = 0.0
    SPREAD_COST = 0.0001
    
    # We test thresholds from 0.35 to 0.80
    for threshold in np.arange(0.35, 0.85, 0.05):
        # LOGIC:
        # If prob(Bear) > threshold -> Signal -1
        # If prob(Bull) > threshold -> Signal 1
        # Else -> Signal 0 (Stay Sideways/Cash)
        
        signals = np.zeros(len(probs))
        
        # Iterate through probabilities
        for i, p in enumerate(probs):
            if p[0] > threshold: # Bearish
                signals[i] = -1
            elif p[2] > threshold: # Bullish
                signals[i] = 1
            else:
                signals[i] = 0
                
        # Vectorized Backtest
        # Align signals (predict today for tomorrow)
        trade_signals = signals[:-1]
        
        # Costs: Pay spread only when entering a NEW trade (or flipping)
        # Simple approximation: Pay spread on every non-zero candle (conservative)
        costs = np.abs(trade_signals) * SPREAD_COST
        
        strategy_returns = (market_returns * trade_signals) - costs
        total_return = np.sum(strategy_returns)
        
        # Calculate Win Rate (Only on active trades)
        active_mask = trade_signals != 0
        if active_mask.sum() > 0:
            wins = ((strategy_returns[active_mask] > 0).sum())
            win_rate = wins / active_mask.sum()
        else:
            win_rate = 0.0
            
        print(f"{threshold:.2f}{'':<6} | {active_mask.sum():<8} | {win_rate:.1%}     | {total_return:.4f}     ")
        
        if total_return > best_return:
            best_return = total_return
            best_threshold = threshold
            
    print("-" * 65)
    print(f"🏆 BEST THRESHOLD: {best_threshold:.2f} (Return: {best_return:.4f})")
    
    # 6. Plot Best Result
    threshold = best_threshold
    signals = np.zeros(len(probs))
    for i, p in enumerate(probs):
        if p[0] > threshold: signals[i] = -1
        elif p[2] > threshold: signals[i] = 1
        else: signals[i] = 0
        
    trade_signals = signals[:-1]
    costs = np.abs(trade_signals) * SPREAD_COST
    strategy_returns = (market_returns * trade_signals) - costs
    cum_strategy = np.cumsum(strategy_returns)
    cum_market = np.cumsum(market_returns)
    
    plt.figure(figsize=(12, 6))
    plt.plot(cum_market, label="Market", color='gray', alpha=0.5)
    plt.plot(cum_strategy, label=f"Strategy (Conf > {threshold:.2f})", color='green', linewidth=2)
    plt.title(f"Optimized Strategy (Threshold {threshold:.2f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    evaluate_and_optimize()