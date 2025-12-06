# analysis/feature_importance.py
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from utils.feature_adapter import EnhancedDataLoaderV2

def analyze_features():
    print("🚀 Running Feature Importance Analysis...")
    
    # 1. Initialize Loader
    # We use a shorter sequence length for analysis to keep it snappy
    seq_len = 30
    loader = EnhancedDataLoaderV2(
        sequence_length=seq_len,
        label_strategy='ternary',
        scaler_type='robust',
        trend_threshold=0.05 # Matching your current threshold
    )
    
    # 2. Load and Scale Data
    print("   Loading data...")
    df = loader.load_csv("data/raw/eurusd_latest.csv")
    
    # We must scale the data just like in training
    train_scaled, _, _ = loader.split_and_scale(
        df, 
        split_ratio=0.8, 
        validation_ratio=0.1
    )
    
    # 3. Create Sequences (This ensures labels match LSTM exactly)
    # y_train contains the labels (0, 1, 2)
    print("   Creating sequences...")
    X_3d, y = loader.create_sequences(train_scaled, loader.train_close, seq_len)
    
    if len(X_3d) == 0:
        print("❌ Error: No sequences generated. Check threshold or data size.")
        return

    # 4. Flatten for Random Forest
    # X_3d shape is (Samples, TimeSteps, Features)
    # We take the LAST timestep of every sequence ([:, -1, :])
    # This represents the market state right before the prediction is made.
    print(f"   Flattening data: {X_3d.shape} -> (samples, features)")
    X_flat = X_3d[:, -1, :]
    
    # 5. Train Random Forest
    print(f"   Training Random Forest on {len(y)} samples...")
    rf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=12, 
        n_jobs=-1, 
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_flat, y)
    
    # 6. Extract Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n" + "="*50)
    print("🏆 TOP 25 PREDICTIVE FEATURES")
    print("="*50)
    
    feature_names = loader.feature_columns
    top_features = []
    
    for i in range(min(25, len(feature_names))):
        idx = indices[i]
        feat_name = feature_names[idx]
        score = importances[idx]
        top_features.append(feat_name)
        print(f"{i+1:2d}. {feat_name:30s} : {score:.4f}")
        
    # 7. Visualization
    plt.figure(figsize=(12, 10))
    plt.title(f"Feature Importance (Threshold: 0.05%)")
    
    # Plot top 25
    top_n = 25
    plt.barh(range(top_n), importances[indices[:top_n]][::-1], align="center", color='#4a90e2')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel("Relative Importance Score")
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_features()