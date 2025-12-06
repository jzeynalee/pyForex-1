# analysis/evaluate_horizon.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.feature_adapter import EnhancedDataLoaderV2
from training.train_lstm_enhanced import EnhancedLSTM, TOP_FEATURES

def evaluate_horizon_model():
    # SETTINGS (Must match training)
    SEQ_LEN = 30
    HIDDEN_DIM = 64
    THRESHOLD = 0.05
    HORIZON = 5  # <--- CRITICAL: The model predicts 5 steps ahead
    DATA_PATH = "data/raw/eurusd_latest.csv"
    MODEL_PATH = "models/weights/lstm_enhanced_best.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Evaluating Horizon Model (H={HORIZON}): {MODEL_PATH}")

    # 1. Load Data
    loader = EnhancedDataLoaderV2(sequence_length=SEQ_LEN, trend_threshold=THRESHOLD)
    df = loader.load_csv(DATA_PATH)
    
    # Feature Selection
    loader.feature_columns = [f for f in TOP_FEATURES if f in df.columns]
    
    # 2. Prepare Data (Manual Split to match training)
    # We need the raw Close prices for the test set
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    # We evaluate on the Test chunk (same as training script)
    test_df = df.iloc[train_end:val_end].copy()
    
    # Scale
    from sklearn.preprocessing import RobustScaler
    # Fit scaler on TRAINING chunk to avoid leakage
    scaler = RobustScaler()
    train_chunk = df.iloc[:train_end][loader.feature_columns].values
    scaler.fit(train_chunk)
    
    test_features = scaler.transform(test_df[loader.feature_columns].values)
    
    # Create Sequences
    X_test = []
    # We need to align the "Future Return" with the "Prediction Time"
    # If model predicts at index `t`, the result is `price[t+5] - price[t]`
    # So we need to track the "Entry Price" and "Exit Price" for every sequence
    entry_prices = []
    exit_prices = []
    
    close_prices = test_df['close'].values
    
    limit = len(test_features) - SEQ_LEN - HORIZON
    for i in range(limit):
        X_test.append(test_features[i : i+SEQ_LEN])
        
        # entry_idx is the last candle of the sequence
        entry_idx = i + SEQ_LEN - 1
        entry_prices.append(close_prices[entry_idx])
        exit_prices.append(close_prices[entry_idx + HORIZON])
        
    X_test = np.array(X_test)
    entry_prices = np.array(entry_prices)
    exit_prices = np.array(exit_prices)
    
    print(f"   Test Samples: {len(X_test)}")

    # 3. Load Model
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model = EnhancedLSTM(
        input_dim=len(loader.feature_columns),
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        num_classes=3,
        bidirectional=False
    ).to(device)
    
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint) # Handle simple state dict
        
    model.eval()

    # 4. Predict
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

    # 5. Optimization Loop
    print("\n🔍 OPTIMIZING HORIZON THRESHOLD")
    print("-" * 65)
    print(f"{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Return':<10}")
    print("-" * 65)
    
    # Calculate Returns for every sample (if we bought/sold)
    # Long Return: (Exit - Entry) / Entry
    # Short Return: (Entry - Exit) / Entry
    raw_long_returns = (exit_prices - entry_prices) / entry_prices
    
    best_return = -999
    best_thresh = 0.0
    SPREAD = 0.0001
    
    for thresh in np.arange(0.35, 0.65, 0.02):
        signals = np.zeros(len(probs))
        # 0=Bear, 2=Bull (Indices)
        signals[probs[:, 0] > thresh] = -1 # Short
        signals[probs[:, 2] > thresh] = 1  # Long
        
        # Calculate PnL
        # We assume we hold for exactly 5 candles, then close.
        # This is a simplified "Rolling Backtest"
        
        trade_returns = np.zeros(len(signals))
        
        # Longs
        long_mask = (signals == 1)
        trade_returns[long_mask] = raw_long_returns[long_mask] - SPREAD
        
        # Shorts
        short_mask = (signals == -1)
        trade_returns[short_mask] = -raw_long_returns[short_mask] - SPREAD
        
        total_return = np.sum(trade_returns)
        n_trades = np.sum(signals != 0)
        
        win_rate = 0.0
        if n_trades > 0:
            wins = np.sum(trade_returns > 0)
            win_rate = wins / n_trades
            
        print(f"{thresh:.2f}{'':<6} | {n_trades:<8} | {win_rate:.1%}     | {total_return:.4f}")
        
        if total_return > best_return:
            best_return = total_return
            best_thresh = thresh

    print("-" * 65)
    print(f"🏆 BEST: Threshold {best_thresh:.2f} | Return {best_return:.4f}")
    
    # Plot Cumulative
    signals = np.zeros(len(probs))
    signals[probs[:, 0] > best_thresh] = -1
    signals[probs[:, 2] > best_thresh] = 1
    
    trade_returns = np.zeros(len(signals))
    long_mask = (signals == 1)
    trade_returns[long_mask] = raw_long_returns[long_mask] - SPREAD
    short_mask = (signals == -1)
    trade_returns[short_mask] = -raw_long_returns[short_mask] - SPREAD
    
    plt.figure(figsize=(12, 6))
    plt.plot(np.cumsum(trade_returns), label=f"Horizon Strategy (T={best_thresh:.2f})", color='blue')
    plt.title("Cumulative Return (Fixed 5-Candle Hold)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    evaluate_horizon_model()