# training/train_dynamic.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import RobustScaler

# Custom Modules
from utils.feature_adapter import EnhancedDataLoaderV2
from training.train_lstm_enhanced import EnhancedLSTM, FocalLoss, compute_balanced_weights
from training.feature_selector import DynamicFeatureSelector

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def train_dynamic(data_path, epochs, seq_len, hidden_dim, lr, dropout, horizon, threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 DYNAMIC TRAINING | H={horizon} | Thresh={threshold}% | Device={device}")

    # 1. Load & Engineer (Fast)
    loader = EnhancedDataLoaderV2(sequence_length=seq_len)
    df = loader.load_csv(data_path)
    
    # 2. Horizon Labeling (The Fix)
    # We look 'horizon' steps ahead
    future_pct = df['close'].shift(-horizon) / df['close'] - 1
    thresh_dec = threshold / 100.0
    
    targets = np.ones(len(df), dtype=int) # 1 = Sideways
    targets[future_pct < -thresh_dec] = 0 # 0 = Bearish
    targets[future_pct > thresh_dec] = 2  # 2 = Bullish
    
    df['target'] = targets
    
    # Drop NaN tail from horizon shift
    df = df.iloc[:-horizon].copy()
    
    # 3. Dynamic Selection
    exclude = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'date', 'target']
    selector = DynamicFeatureSelector(n_features=25) # Pick Top 25
    selected_features = selector.select(df, 'target', exclude)
    
    # 4. Scale & Split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    scaler = RobustScaler()
    # Handle Infs/NaNs
    X_train_raw = train_df[selected_features].replace([np.inf, -np.inf], 0).fillna(0).values
    X_test_raw = test_df[selected_features].replace([np.inf, -np.inf], 0).fillna(0).values
    
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    y_train = train_df['target'].values
    y_test = test_df['target'].values

    # 5. Sequences
    def make_seq(data, labels, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            xs.append(data[i : i+seq_len])
            ys.append(labels[i + seq_len]) # Predict state at end of sequence
        return np.array(xs), np.array(ys)
        
    logger.info("   Creating sequences...")
    X_train, y_train_seq = make_seq(X_train_scaled, y_train, seq_len)
    X_test, y_test_seq = make_seq(X_test_scaled, y_test, seq_len)
    
    # 6. Training Setup
    # Sampler
    class_counts = np.bincount(y_train_seq)
    weights = 1. / (class_counts + 1)
    sample_weights = weights[y_train_seq]
    sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), len(sample_weights))
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train_seq).long()), 
                              batch_size=64, sampler=sampler)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test_seq).long()), 
                             batch_size=64, shuffle=False)
    
    model = EnhancedLSTM(len(selected_features), hidden_dim, 1, 3, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_weights = compute_balanced_weights(y_train_seq).to(device)
    criterion = FocalLoss(alpha=loss_weights)
    
    # 7. Loop
    best_acc = 0
    logger.info(f"🧠 Training on {len(X_train)} samples...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                _, p = torch.max(out, 1)
                preds.extend(p.cpu().numpy())
                labs.extend(by.cpu().numpy())
        
        preds = np.array(preds)
        labs = np.array(labs)
        
        # Balanced Accuracy
        accs = [np.mean(preds[labs==i] == i) if np.sum(labs==i) > 0 else 0 for i in range(3)]
        bal_acc = np.mean(accs)
        
        if epoch % 1 == 0:
            logger.info(f"Ep {epoch:3d} | L: {train_loss/len(train_loader):.4f} | BalAcc: {bal_acc:.1%} | B:{accs[0]:.0%} S:{accs[1]:.0%} B:{accs[2]:.0%}")
            
        if bal_acc > best_acc:
            best_acc = bal_acc
            torch.save({
                'model_state': model.state_dict(),
                'features': selected_features,
                'scaler': scaler
            }, "models/weights/lstm_dynamic_best.pt")
            
    logger.info(f"✅ Best Balanced Acc: {best_acc:.1%}")

if __name__ == "__main__":
    train_dynamic("data/raw/eurusd_latest.csv", 50, 30, 64, 1e-3, 0.3, 5, 0.05)