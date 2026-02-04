"""
Analyze label distribution in training data for MH-TCN models.

This script loads the OHLCV data and generates direction labels to understand
the class imbalance issue.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from collections import Counter


def load_data(path: str) -> pd.DataFrame:
    """Load and validate OHLCV data."""
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values('time', inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    return df


def generate_direction_labels(df: pd.DataFrame, horizon: int, threshold: float) -> np.ndarray:
    """Generate direction labels based on future price movement."""
    close = df['close'].values
    n = len(close)
    labels = np.ones(n, dtype=np.int64)  # Default: Sideways
    
    for i in range(n - horizon):
        future_return = (close[i + horizon] - close[i]) / close[i]
        
        if future_return > threshold:
            labels[i] = 2  # Bull
        elif future_return < -threshold:
            labels[i] = 0  # Bear
        # else: remains 1 (Sideways)
    
    labels[-horizon:] = 1  # Last bars get Sideways
    return labels


def analyze_distribution(labels: np.ndarray, name: str):
    """Analyze and print label distribution."""
    counter = Counter(labels)
    total = len(labels)
    
    print(f"\n{'='*60}")
    print(f"  {name} Label Distribution")
    print(f"{'='*60}")
    
    class_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
    
    for cls in [0, 1, 2]:
        count = counter.get(cls, 0)
        pct = 100 * count / total
        bar = '█' * int(pct / 2)
        print(f"  {class_names[cls]:10s}: {count:6d} ({pct:5.1f}%) {bar}")
    
    print(f"\n  Total samples: {total}")
    
    # Imbalance ratio
    max_count = max(counter.values())
    min_count = min(counter.values()) if min(counter.values()) > 0 else 1
    imbalance_ratio = max_count / min_count
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    return counter


def analyze_with_different_thresholds(df: pd.DataFrame, horizon: int, name: str):
    """Test different thresholds to find optimal balance."""
    print(f"\n{'='*60}")
    print(f"  {name} - Threshold Analysis (horizon={horizon})")
    print(f"{'='*60}")
    
    thresholds = [0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005]
    
    print(f"\n  {'Threshold':>10s} | {'Bear':>8s} | {'Side':>8s} | {'Bull':>8s} | {'Imbalance':>10s}")
    print(f"  {'-'*10} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*10}")
    
    best_threshold = None
    best_balance = float('inf')
    
    for thresh in thresholds:
        labels = generate_direction_labels(df, horizon, thresh)
        counter = Counter(labels)
        
        bear = counter.get(0, 0)
        side = counter.get(1, 0)
        bull = counter.get(2, 0)
        
        max_c = max(counter.values())
        min_c = min(counter.values()) if min(counter.values()) > 0 else 1
        imbalance = max_c / min_c
        
        # Calculate balance score (lower is better)
        # We want roughly equal distribution
        total = len(labels)
        expected = total / 3
        balance_score = abs(bear - expected) + abs(side - expected) + abs(bull - expected)
        
        if balance_score < best_balance and min_c > 100:  # Ensure minimum samples
            best_balance = balance_score
            best_threshold = thresh
        
        print(f"  {thresh:>10.4f} | {bear:>8d} | {side:>8d} | {bull:>8d} | {imbalance:>10.1f}:1")
    
    print(f"\n  Recommended threshold: {best_threshold}")
    return best_threshold


def main():
    # Data paths
    data_files = {
        'H1 (INTRADAY)': 'd:/myBot/data/raw/EURUSD_H1_latest.csv',
        'M5 (SCALP)': 'd:/myBot/data/raw/EURUSD_M5_latest.csv',
        'H4 (SWING)': 'd:/myBot/data/raw/EURUSD_H4_latest.csv',
    }
    
    # Default training parameters
    configs = {
        'H1 (INTRADAY)': {'horizon': 12, 'threshold': 0.001},
        'M5 (SCALP)': {'horizon': 12, 'threshold': 0.001},
        'H4 (SWING)': {'horizon': 12, 'threshold': 0.001},
    }
    
    print("\n" + "="*60)
    print("  MH-TCN LABEL DISTRIBUTION ANALYSIS")
    print("="*60)
    
    recommendations = {}
    
    for name, path in data_files.items():
        try:
            df = load_data(path)
            config = configs[name]
            
            print(f"\n\n{'#'*60}")
            print(f"  Dataset: {name}")
            print(f"  File: {path}")
            print(f"  Rows: {len(df)}")
            print(f"{'#'*60}")
            
            # Current distribution
            print(f"\n  Current settings: horizon={config['horizon']}, threshold={config['threshold']}")
            labels = generate_direction_labels(df, config['horizon'], config['threshold'])
            analyze_distribution(labels, f"{name} (current)")
            
            # Find better threshold
            best_thresh = analyze_with_different_thresholds(df, config['horizon'], name)
            recommendations[name] = best_thresh
            
            # Show improved distribution
            if best_thresh != config['threshold']:
                print(f"\n  With recommended threshold={best_thresh}:")
                labels_new = generate_direction_labels(df, config['horizon'], best_thresh)
                analyze_distribution(labels_new, f"{name} (recommended)")
            
        except Exception as e:
            print(f"\n  ERROR loading {name}: {e}")
    
    # Summary
    print("\n\n" + "="*60)
    print("  SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    print("\n  1. THRESHOLD RECOMMENDATIONS:")
    for name, thresh in recommendations.items():
        print(f"     - {name}: threshold={thresh}")
    
    print("\n  2. ADDITIONAL FIXES TO IMPLEMENT:")
    print("     - Add Focal Loss for direction head (handles class imbalance)")
    print("     - Add class weighting based on inverse frequency")
    print("     - Consider oversampling minority classes")
    print("     - Use stratified sampling in DataLoader")
    
    print("\n  3. CODE CHANGES NEEDED:")
    print("     - training/train_mhtcn.py: Update default threshold")
    print("     - risk_management/phase1_predictive/training.py: Add FocalLoss")
    print("     - Add WeightedRandomSampler for balanced batches")


if __name__ == "__main__":
    main()
