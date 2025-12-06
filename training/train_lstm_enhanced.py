# training/train_lstm_enhanced_fixed.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from collections import Counter
from sklearn.preprocessing import RobustScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.feature_adapter import EnhancedDataLoaderV2

import warnings
# Silence all FutureWarnings (pandas updates, etc.)
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS & CONFIG
# ============================================================
TOP_FEATURES = [
    'volume_zscore', 'extended_bar_ratio', 'volume_roc', 'obv_change', 
    'volume_ma5', 'eom', 'connors_rsi', 'force_index', 
    'ultimate_osc', 'dpo', 'cmf', 'return_gradient', 
    'chaikin_osc', 'local_slope', 'bop', 'macd_hist', 
    'stoch_k', 'gap_intensity', 'atr_percent', 'normalized_variance', 
    'ppo_mean', 'atr_ratio', 'bear_power', 'atr', 'adx'
]

# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

# ============================================================
# MODEL
# ============================================================
class EnhancedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, num_classes=3, dropout=0.2, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        attention_dim = hidden_dim * self.num_directions
        self.attention = nn.Sequential(
            nn.Linear(attention_dim, attention_dim // 2),
            nn.Tanh(),
            nn.Linear(attention_dim // 2, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(attention_dim),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def attention_pooling(self, lstm_out):
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context

    def forward(self, x):
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        context = self.attention_pooling(lstm_out)
        return self.classifier(context)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def compute_balanced_weights(y, max_weight_ratio=3.0):
    class_counts = np.bincount(y)
    weights = 1.0 / np.sqrt(class_counts + 1)
    weights = weights / weights.sum() * len(class_counts)
    
    min_w, max_w = weights.min(), weights.max()
    if max_w / min_w > max_weight_ratio:
        weights = np.clip(weights, min_w, min_w * max_weight_ratio)
        weights = weights / weights.sum() * len(class_counts)
        
    return torch.tensor(weights, dtype=torch.float32)

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_enhanced_lstm(
    data_path="data/raw/eurusd_latest.csv",
    save_dir="models/weights",
    epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    seq_len=30,
    hidden_dim=64,
    dropout=0.2,
    patience=15,
    trend_threshold=0.05,
    device=None
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🚀 Enhanced LSTM (Horizon Labeling + Feature Selection)")
    print("=" * 70)
    
    # 1. Load Data
    loader = EnhancedDataLoaderV2(sequence_length=seq_len, trend_threshold=trend_threshold)
    df = loader.load_csv(data_path)
    
    # === HORIZON LABELING (The Fix) ===
    HORIZON = 5
    future_pct = df['close'].shift(-HORIZON) / df['close'] - 1
    threshold_dec = trend_threshold / 100.0
    
    # 0 = Bearish, 1 = Sideways, 2 = Bullish
    custom_labels = np.ones(len(df), dtype=int)
    custom_labels[future_pct < -threshold_dec] = 0
    custom_labels[future_pct > threshold_dec] = 2
    
    # Handle end of DF NaNs
    custom_labels[-HORIZON:] = 1
    df['custom_target'] = custom_labels
    
    # Feature Selection
    available_features = [f for f in TOP_FEATURES if f in df.columns]
    print(f"✂️  Features Selected: {len(available_features)}")
    loader.feature_columns = available_features
    
    # 2. Split & Scale (Manual to preserve alignment)
    # Split indices
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[train_end:val_end].copy() # Using middle chunk as test for now
    
    # Scale Features
    scaler = RobustScaler()
    train_features = scaler.fit_transform(train_df[available_features].values)
    test_features = scaler.transform(test_df[available_features].values)
    
    train_y_raw = train_df['custom_target'].values
    test_y_raw = test_df['custom_target'].values
    
    # 3. Create Sequences
    def make_seq(data, targets, length):
        xs, ys = [], []
        # Need buffer at end for Horizon
        limit = len(data) - length - HORIZON 
        for i in range(limit):
            xs.append(data[i : i+length])
            # The label for this sequence is the target HORIZON steps after the sequence ends
            # i+length is the index right after the sequence. 
            # We want the trend that STARTS at i+length-1 (last candle) and resolves at i+length-1+HORIZON
            # But simpler: we aligned custom_labels to be "Future return relative to current row".
            # So if we take the row at the END of the sequence (i+length-1), its label describes the future.
            ys.append(targets[i + length - 1]) 
        return np.array(xs), np.array(ys)
        
    X_train, y_train = make_seq(train_features, train_y_raw, seq_len)
    X_test, y_test = make_seq(test_features, test_y_raw, seq_len)
    
    if len(X_train) == 0: raise ValueError("Insufficient data")
    
    # Define n_samples (Fixing the NameError)
    n_samples = len(X_train)
    input_dim = X_train.shape[2]
    
    # 4. Sampler for Imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1. / (class_counts + 1)
    sample_weights = class_weights[y_train]
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print(f"\n📊 DISTRIBUTION (Horizon={HORIZON}):")
    class_names = ['BEARISH', 'SIDEWAYS', 'BULLISH']
    for i, c in enumerate(class_counts):
        if i < len(class_names):
             print(f"  {class_names[i]}: {c} ({c/n_samples:.1%})")

    # 5. Data Loaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 6. Initialize Model
    model = EnhancedLSTM(
        input_dim=input_dim, hidden_dim=hidden_dim, num_layers=1,
        num_classes=3, dropout=dropout, bidirectional=False
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
    
    # Loss Weights
    weights = compute_balanced_weights(y_train).to(device)
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    
    # 7. Training Loop
    best_acc = 0
    patience_counter = 0
    
    print(f"\n🚀 Training on {device}...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Evaluate
        model.eval()
        test_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                
        test_loss /= len(test_loader)
        
        # Metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        per_class_acc = {}
        for i, name in enumerate(class_names):
            mask = all_labels == i
            if mask.sum() > 0:
                per_class_acc[name] = (all_preds[mask] == i).mean()
            else:
                per_class_acc[name] = 0.0
                
        balanced_acc = np.mean(list(per_class_acc.values()))
        
        # Log
        if epoch % 5 == 0 or epoch == 1:
            pc_str = " | ".join([f"{n[0]}:{a:.0%}" for n, a in per_class_acc.items()])
            logger.info(f"Ep {epoch:3d} | L: {train_loss:.3f}/{test_loss:.3f} | BalAcc: {balanced_acc:.1%} | {pc_str}")
            
        # Save Best
        if balanced_acc > best_acc:
            best_acc = balanced_acc
            patience_counter = 0
            # Save simple state dict
            torch.save(model.state_dict(), Path(save_dir) / "lstm_enhanced_best.pt")
            logger.info(f"💾 Best Saved: {best_acc:.1%}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("🛑 Early Stopping")
                break
                
    # Save Scaler for later inference
    import joblib
    joblib.dump(scaler, Path(save_dir) / "scaler_enhanced.joblib")
    print(f"\nFinal Best Balanced Acc: {best_acc:.1%}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="data/raw/eurusd_latest.csv")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--seq-len', type=int, default=30)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.2)
    # NEW: Added learning rate argument
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    train_enhanced_lstm(
        data_path=args.data,
        epochs=args.epochs,
        trend_threshold=args.threshold,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.lr  # Pass it to the trainer
    )

if __name__ == "__main__":
    main()