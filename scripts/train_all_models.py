#!/usr/bin/env python
"""
Comprehensive Model Training Script for pyForex

Trains all models in the system:
1. MultiHeadTCN (with risk heads) - Direction, Volatility, Quantiles
2. YOLO - Pattern detection
3. Meta-labeling (LightGBM/RandomForest) - Trade filtering
4. Exit Optimizer (PPO) - RL-based exit timing

Usage Examples:
    python scripts/train_all_models.py --models tcn meta --profiles INTRADAY
    python scripts/train_all_models.py --all
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
import shutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "training_all_models.log")
    ]
)
logger = logging.getLogger(__name__)

from utils.feature_schema import get_feature_schema_version
from utils.training_utils import set_global_seed, copy_schema_tagged

# =============================================================================
# Configuration
# =============================================================================

PROFILES = ['SCALP', 'INTRADAY', 'SWING']

PROFILE_DATA_MAP = {
    'SCALP': {
        'primary': 'data/raw/EURUSD_M5_latest.csv',
        'timeframes': ['M5', 'M15'],
    },
    'INTRADAY': {
        'primary': 'data/raw/EURUSD_H1_latest.csv',
        'timeframes': ['H1', 'H4'],
    },
    'SWING': {
        'primary': 'data/raw/EURUSD_H4_latest.csv',
        'timeframes': ['H4', 'D1'],
    },
}

WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def ensure_dirs():
    """Ensure output directories exist."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    (CHECKPOINTS_DIR / "multihead_tcn").mkdir(exist_ok=True)
    (CHECKPOINTS_DIR / "meta_labeling").mkdir(exist_ok=True)
    (CHECKPOINTS_DIR / "exit_optimizer").mkdir(exist_ok=True)


# =============================================================================
# 1. MultiHeadTCN Training
# =============================================================================

def train_multihead_tcn(
    profile: str,
    data_path: str,
    max_rows: int = 1_000_000,
    max_features: int = 0,
    select_top_features: bool = False,
    epochs: int = 100,
    batch_size: int = 64,
    seq_len: int = 60,
    hidden_dim: int = 128,
    num_layers: int = 4,
    learning_rate: float = 1e-3,
    device: str = 'auto'
) -> Dict[str, Any]:
    """
    Train MultiHeadTCN with risk prediction heads.
    
    Returns training history and metrics.
    """
    logger.info("=" * 60)
    logger.info(f"  TRAINING MultiHeadTCN - {profile}")
    logger.info("=" * 60)
    
    from risk_management.phase1_predictive import (
        create_tcn_for_profile, MultiHeadTCN, TrainingConfig, 
        RiskDataset, MultiHeadTCNTrainer
    )
    from torch.utils.data import DataLoader, Subset
    
    # Device selection
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load and prepare data
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    if len(df) > max_rows:
        logger.info(f"Limiting to {max_rows:,} rows (from {len(df):,})")
        df = df.tail(max_rows)  # Use most recent data
    
    logger.info(f"Data shape: {df.shape}")
    
    # Feature engineering
    logger.info("Engineering features...")
    from utils.features_engineering import FeatureEngineer
    
    engineer = FeatureEngineer()
    features_df = engineer.generate_features(df)
    
    # Remove NaN rows
    features_df = features_df.dropna()
    logger.info(f"Features shape after dropna: {features_df.shape}")
    
    # Get feature columns (exclude OHLCV, time, and non-numeric)
    exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume', 'spread', 'real_volume']
    # Only include numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    if int(max_features or 0) > 0 and len(feature_cols) > int(max_features):
        if bool(select_top_features):
            logger.info(f"Selecting top {int(max_features)} features from {len(feature_cols)}...")
            from sklearn.ensemble import RandomForestClassifier
            
            # Create simple labels for feature selection
            price_change = features_df['close'].pct_change(5).shift(-5)
            labels = (price_change > 0.001).astype(int) - (price_change < -0.001).astype(int) + 1
            labels = labels.dropna()
            
            # Align data
            valid_idx = labels.index.intersection(features_df.index)
            X_sel = features_df.loc[valid_idx, feature_cols].fillna(0)
            y_sel = labels.loc[valid_idx]
            
            # Fit RF for feature importance
            rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
            rf.fit(X_sel[:50000], y_sel[:50000])  # Use subset for speed
            
            importances = pd.Series(rf.feature_importances_, index=feature_cols)
            feature_cols = importances.nlargest(int(max_features)).index.tolist()
            logger.info(f"Selected {len(feature_cols)} features")
        else:
            feature_cols = feature_cols[:int(max_features)]
            logger.info(f"Capped features to first {len(feature_cols)} columns (max_features={int(max_features)})")
    
    # Prepare features and labels
    logger.info("Preparing features and labels...")
    X = features_df[feature_cols].values.astype(np.float32)
    prices = features_df[['open', 'high', 'low', 'close']].values
    
    # Scale features
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    # Create labels for each timestep (RiskDataset will handle sequencing)
    forward_bars = {'SCALP': 6, 'INTRADAY': 10, 'SWING': 20}[profile]
    threshold = {'SCALP': 0.001, 'INTRADAY': 0.002, 'SWING': 0.005}[profile]
    
    n_samples = len(X)
    targets_direction = np.ones(n_samples, dtype=np.int64)  # Default to SIDEWAYS
    targets_volatility = np.zeros(n_samples, dtype=np.float32)
    targets_price_move = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(n_samples - forward_bars):
        # Direction target (3-class)
        future_return = (prices[i + forward_bars, 3] - prices[i, 3]) / prices[i, 3]
        
        if future_return > threshold:
            targets_direction[i] = 2  # BULL
        elif future_return < -threshold:
            targets_direction[i] = 0  # BEAR
        else:
            targets_direction[i] = 1  # SIDEWAYS
        
        # Volatility target (realized volatility over forward period)
        future_prices = prices[i:i+forward_bars, 3]
        targets_volatility[i] = np.std(np.diff(future_prices) / future_prices[:-1]) if len(future_prices) > 1 else 0.001
        
        # Price move target (for quantile regression)
        targets_price_move[i] = future_return
    
    # Trim to valid range (exclude last forward_bars samples)
    valid_end = n_samples - forward_bars
    X = X[:valid_end]
    targets_direction = targets_direction[:valid_end]
    targets_volatility = targets_volatility[:valid_end]
    targets_price_move = targets_price_move[:valid_end]
    
    logger.info(f"Prepared {len(X):,} samples")
    logger.info(f"Direction distribution: BEAR={np.sum(targets_direction==0)}, SIDE={np.sum(targets_direction==1)}, BULL={np.sum(targets_direction==2)}")
    
    # Calculate class weights
    class_counts = np.bincount(targets_direction)
    total_samples = len(targets_direction)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    logger.info(f"Class weights: {class_weights}")

    # Create dataset - RiskDataset creates sequences internally
    dataset = RiskDataset(
        features=X,
        direction_labels=targets_direction,
        volatility_labels=targets_volatility,
        price_move_labels=targets_price_move,
        sequence_length=seq_len
    )
    
    # Custom collate function to handle None vision features
    def collate_fn(batch):
        seqs = torch.stack([item[0] for item in batch])
        targets = {
            'direction': torch.stack([item[1]['direction'] for item in batch]),
            'volatility': torch.stack([item[1]['volatility'] for item in batch]),
            'price_move': torch.stack([item[1]['price_move'] for item in batch])
        }
        # Vision is None for all items, return None
        return seqs, targets, None
    
    # Split train/val chronologically with purge gap to reduce leakage from overlapping sequences
    n_total = len(dataset)
    split_idx = int(0.8 * n_total)
    purge_gap = int(seq_len)
    val_start = min(split_idx + purge_gap, n_total)

    train_dataset = Subset(dataset, range(0, split_idx))
    val_dataset = Subset(dataset, range(val_start, n_total))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    logger.info(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")
    
    # Create model
    model = create_tcn_for_profile(
        profile=profile,
        input_features=len(feature_cols)
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training config
    train_config = TrainingConfig(
        num_epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        warmup_epochs=5,
        patience=15,
        direction_weight=2.0,  # Increase weight for direction loss
        volatility_weight=0.5,
        quantile_weight=0.5
    )
    
    # Train
    trainer = MultiHeadTCNTrainer(model, train_config, device=device, class_weights=class_weights)
    history = trainer.train(train_loader, val_loader)
    
    # Save checkpoint
    checkpoint_path = CHECKPOINTS_DIR / "multihead_tcn" / f"multihead_tcn_{profile}.pth"
    schema_version = get_feature_schema_version()
    torch.save({
        'state_dict': trainer.best_model_state or model.state_dict(),
        'config': {
            'profile': profile,
            'input_dim': len(feature_cols),
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'seq_len': seq_len,
            'feature_schema_version': schema_version,
        },
        'feature_columns': feature_cols,
        'scaler_params': {
            'center': scaler.center_.tolist(),
            'scale': scaler.scale_.tolist()
        },
        'history': history,
        'timestamp': datetime.now().isoformat()
    }, checkpoint_path)

    copy_schema_tagged(checkpoint_path, schema_version)
    
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Also save to weights dir for inference
    weights_path = WEIGHTS_DIR / f"multihead_tcn_{profile}.pth"
    torch.save({
        'state_dict': trainer.best_model_state or model.state_dict(),
        'config': {
            'profile': profile,
            'input_dim': len(feature_cols),
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'seq_len': seq_len,
            'feature_schema_version': schema_version,
        },
        'feature_columns': feature_cols,
        'num_classes': 3,
        'num_directions': 3,
        'num_quantiles': 5
    }, weights_path)

    copy_schema_tagged(weights_path, schema_version)
    
    logger.info(f"Saved weights: {weights_path}")
    
    return {
        'profile': profile,
        'epochs_trained': len(history.get('train_loss', [])),
        'best_val_loss': min(history.get('val_loss', [float('inf')])),
        'checkpoint_path': str(checkpoint_path)
    }


# =============================================================================
# 3. Meta-Labeling Training
# =============================================================================

def train_meta_labeling(
    profile: str,
    data_path: str,
    max_rows: int = 1_000_000,
    device: str = 'auto'
) -> Dict[str, Any]:
    """Train meta-labeling model (LightGBM/RandomForest)."""
    logger.info("=" * 60)
    logger.info(f"  TRAINING Meta-Labeling - {profile}")
    logger.info("=" * 60)
    
    from risk_management.phase3_filtering.meta_labeling import (
        MetaLabelingModel, MetaLabelingConfig
    )
    
    # Load data
    df = pd.read_csv(data_path)
    if len(df) > max_rows:
        df = df.tail(max_rows)
    
    logger.info(f"Data shape: {df.shape}")
    
    # Feature engineering
    from utils.features_engineering import FeatureEngineer
    engineer = FeatureEngineer()
    features_df = engineer.generate_features(df)
    features_df = features_df.dropna()
    
    # Create primary model signals (simulated)
    forward_bars = {'SCALP': 6, 'INTRADAY': 10, 'SWING': 20}[profile]
    threshold = {'SCALP': 0.001, 'INTRADAY': 0.002, 'SWING': 0.005}[profile]
    
    prices = features_df['close'].values
    
    # Primary signals (simulated from simple momentum)
    momentum = features_df['close'].pct_change(5)
    primary_signals = np.where(momentum > 0.001, 'BUY', np.where(momentum < -0.001, 'SELL', 'HOLD'))
    
    # Actual outcomes
    future_returns = features_df['close'].pct_change(forward_bars).shift(-forward_bars)
    
    # Meta-labels: Was the primary signal correct?
    meta_labels = []
    for i, (signal, ret) in enumerate(zip(primary_signals, future_returns)):
        if pd.isna(ret):
            meta_labels.append(np.nan)
        elif signal == 'BUY' and ret > threshold:
            meta_labels.append(1)  # Correct
        elif signal == 'SELL' and ret < -threshold:
            meta_labels.append(1)  # Correct
        elif signal == 'HOLD':
            meta_labels.append(np.nan)  # Skip holds
        else:
            meta_labels.append(0)  # Incorrect
    
    meta_labels = np.array(meta_labels)
    
    # Prepare features for meta-model - only numeric columns
    exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 'tick_volume', 'spread', 'real_volume']
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    X = features_df[feature_cols].values.astype(np.float32)
    y = meta_labels
    
    # Remove NaN labels
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask].astype(int)
    
    logger.info(f"Meta-labeling samples: {len(y):,}")
    logger.info(f"Positive rate: {y.mean():.2%}")
    
    # Train meta-model
    config = MetaLabelingConfig(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05
    )
    
    meta_model = MetaLabelingModel(config)
    
    # Train using the train method (not fit)
    metrics = meta_model.train(X, y, validation_split=0.2)
    
    val_auc = metrics.get('roc_auc', 0.5)
    val_acc = metrics.get('accuracy', 0.0)
    
    logger.info(f"Validation AUC: {val_auc:.4f}")
    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    
    # Save model
    model_path = CHECKPOINTS_DIR / "meta_labeling" / f"meta_model_{profile}.pkl"
    meta_model.save(str(model_path))
    
    logger.info(f"Saved meta-model: {model_path}")
    
    return {
        'profile': profile,
        'val_auc': val_auc,
        'val_acc': val_acc,
        'model_path': str(model_path)
    }


# =============================================================================
# 4. Exit Optimizer Training (PPO)
# =============================================================================

def train_exit_optimizer_script(
    profile: str,
    data_path: str,
    max_rows: int = 1_000_000,
    total_timesteps: int = 100_000,
    device: str = 'auto'
) -> Dict[str, Any]:
    """Train PPO-based exit optimizer."""
    logger.info("=" * 60)
    logger.info(f"  TRAINING Exit Optimizer (PPO) - {profile}")
    logger.info("=" * 60)
    
    try:
        from risk_management.phase4_rl_exit import (
            ExitOptimizer, PPOConfig, create_exit_env, ExitEnvConfig
        )
        from risk_management.phase4_rl_exit.trainer import ExitOptimizerTrainer, TrainingConfig
    except ImportError as e:
        logger.warning(f"Exit optimizer module not available: {e}")
        logger.info("Skipping exit optimizer training.")
        return {'profile': profile, 'status': 'skipped', 'reason': str(e)}
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    df = pd.read_csv(data_path)
    if len(df) > max_rows:
        df = df.tail(max_rows)
    
    # Environment config
    env_config = ExitEnvConfig(
        max_holding_steps=100,
        transaction_cost=0.0001
    )
    
    # Agent config
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01
    )
    
    # Training config
    train_config = TrainingConfig(
        total_timesteps=total_timesteps,
        eval_freq=10000,
        save_freq=25000,
        checkpoint_dir=str(CHECKPOINTS_DIR / "exit_optimizer"),
        device=device
    )
    
    # Train using ExitOptimizerTrainer with correct API
    trainer = ExitOptimizerTrainer(
        config=train_config,
        env_config=env_config,
        agent_config=ppo_config
    )
    history = trainer.train(df)
    
    # Save - trainer creates its own agent internally
    model_path = CHECKPOINTS_DIR / "exit_optimizer" / f"exit_optimizer_{profile}.pth"
    if trainer.agent is not None:
        trainer.agent.save(str(model_path))
    
    logger.info(f"Saved exit optimizer: {model_path}")
    
    return {
        'profile': profile,
        'timesteps_trained': total_timesteps,
        'model_path': str(model_path)
    }


# =============================================================================
# 5. YOLO Training
# =============================================================================

def train_yolo(
    epochs: int = 80,
    batch_size: int = 16,
    device: str = 'auto'
) -> Dict[str, Any]:
    """Train YOLO for pattern detection."""
    logger.info("=" * 60)
    logger.info("  TRAINING YOLO Pattern Detector")
    logger.info("=" * 60)
    
    # Check if YOLO dataset exists
    yolo_yaml = PROJECT_ROOT / "data" / "yolo.yml"
    if not yolo_yaml.exists():
        yolo_yaml = PROJECT_ROOT / "data" / "yolo.yaml"
    if not yolo_yaml.exists():
        logger.warning("YOLO dataset config not found at data/yolo.yml or data/yolo.yaml")
        logger.info("Skipping YOLO training - please prepare dataset first.")
        return {'status': 'skipped', 'reason': 'Dataset not found'}
    
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.warning("ultralytics not installed. Skipping YOLO training.")
        return {'status': 'skipped', 'reason': 'ultralytics not installed'}
    
    if device == 'auto':
        device = 0 if torch.cuda.is_available() else 'cpu'
    
    # Load base model
    model = YOLO("yolov8n.pt")
    
    # Train
    results = model.train(
        data=str(yolo_yaml),
        epochs=epochs,
        imgsz=256,
        device=device,
        batch=batch_size,
        workers=4,
        amp=True,
        cache=True,
        patience=20,
        verbose=True,
        project=str(CHECKPOINTS_DIR / "yolo"),
        name="pattern_detector"
    )
    
    # Export
    model.export(format="pt")
    
    # Copy to weights dir
    import shutil
    best_weights = CHECKPOINTS_DIR / "yolo" / "pattern_detector" / "weights" / "best.pt"
    if best_weights.exists():
        shutil.copy(best_weights, WEIGHTS_DIR / "yolo_patterns.pt")
        logger.info(f"Saved YOLO weights: {WEIGHTS_DIR / 'yolo_patterns.pt'}")
    
    return {
        'status': 'completed',
        'epochs': epochs,
        'results': str(results)
    }


# =============================================================================
# Main Training Orchestrator
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train all pyForex models')
    
    parser.add_argument('--models', nargs='+', 
                        choices=['tcn', 'meta', 'exit', 'yolo', 'all'],
                        default=['all'],
                        help='Models to train')
    parser.add_argument('--profiles', nargs='+',
                        choices=['SCALP', 'INTRADAY', 'SWING', 'all'],
                        default=['all'],
                        help='Trading profiles to train')
    parser.add_argument('--data-rows', type=int, default=1_000_000,
                        help='Maximum data rows to use')
    parser.add_argument('--max-features', type=int, default=0,
                        help='Maximum number of features to use (0 = no cap)')
    parser.add_argument('--select-top-features', action='store_true',
                        help='Explicitly enable RF-based top-N feature selection when --max-features is set')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs for neural networks')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--skip-yolo', action='store_true',
                        help='Skip YOLO training')
    
    args = parser.parse_args()

    set_global_seed(args.seed)
    
    # Resolve 'all' options
    if 'all' in args.models:
        models_to_train = ['tcn', 'meta', 'exit', 'yolo']
    else:
        models_to_train = args.models
    
    if 'all' in args.profiles:
        profiles_to_train = PROFILES
    else:
        profiles_to_train = args.profiles
    
    if args.skip_yolo and 'yolo' in models_to_train:
        models_to_train.remove('yolo')
    
    logger.info("=" * 70)
    logger.info("  pyForex COMPREHENSIVE MODEL TRAINING")
    logger.info("=" * 70)
    logger.info(f"  Models: {models_to_train}")
    logger.info(f"  Profiles: {profiles_to_train}")
    logger.info(f"  Max Data Rows: {args.data_rows:,}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch Size: {args.batch_size}")
    logger.info(f"  Device: {args.device}")
    logger.info("=" * 70)
    
    ensure_dirs()
    
    results = {}
    
    # Train each model for each profile
    for profile in profiles_to_train:
        # Get all timeframes for this profile from MTF config or fallback map
        timeframes = PROFILE_DATA_MAP[profile]['timeframes']
        if profile == 'SCALP':
            timeframes = ['M5', 'M15', 'H1']
        elif profile == 'INTRADAY':
            timeframes = ['M15', 'H1', 'H4']
        elif profile == 'SWING':
            timeframes = ['H1', 'H4', 'D1']

        for timeframe in timeframes:
            # Construct data path dynamically
            data_path = PROJECT_ROOT / f"data/raw/EURUSD_{timeframe}_latest.csv"
            
            if not data_path.exists():
                logger.warning(f"Skipping {profile} {timeframe} - Data not found: {data_path}")
                continue

            if 'tcn' in models_to_train:
                try:
                    # We save with timeframe in filename to support MTF
                    result = train_multihead_tcn(
                        profile=profile,
                        data_path=str(data_path),
                        max_rows=args.data_rows,
                        max_features=args.max_features,
                        select_top_features=args.select_top_features,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        device=args.device
                    )
                    
                    # Save with canonical name that MHTCNFeatureProvider._find_weight_file() expects:
                    #   multihead_tcn_{PROFILE}_{TF}.pth  (primary lookup)
                    # Also keep legacy format for decision_fusion compatibility:
                    #   {profile}_{timeframe}_best.pt
                    
                    src = WEIGHTS_DIR / f"multihead_tcn_{profile}.pth"
                    if src.exists():
                        # Canonical name for 3TF system (MHTCNFeatureProvider)
                        canonical = WEIGHTS_DIR / f"multihead_tcn_{profile}_{timeframe}.pth"
                        shutil.copy(src, canonical)
                        logger.info(f"Saved 3TF weights: {canonical}")
                        
                        # Legacy name for decision_fusion
                        legacy = WEIGHTS_DIR / f"{profile.lower()}_{timeframe.lower()}_best.pt"
                        shutil.copy(src, legacy)
                        logger.info(f"Saved legacy weights: {legacy}")

                    results[f'tcn_{profile}_{timeframe}'] = result
                except Exception as e:
                    logger.error(f"TCN training failed for {profile} {timeframe}: {e}")
                    results[f'tcn_{profile}_{timeframe}'] = {'status': 'failed', 'error': str(e)}
        
        # Meta, Exit, YOLO are usually trained once per profile (dataset aggregation) or on primary TF
        # We keep them running once per profile to avoid redundancy.
        
        # Use primary TF for these
        primary_tf = PROFILE_DATA_MAP[profile]['primary'].split('_')[1].split('_')[0] # e.g. M5 from data/raw/EURUSD_M5_latest.csv
        primary_data_path = PROJECT_ROOT / PROFILE_DATA_MAP[profile]['primary']

        if 'meta' in models_to_train:
            try:
                result = train_meta_labeling(
                    profile=profile,
                    data_path=str(primary_data_path),
                    max_rows=args.data_rows,
                    device=args.device
                )
                results[f'meta_{profile}'] = result
            except Exception as e:
                logger.error(f"Meta-labeling training failed for {profile}: {e}")
                results[f'meta_{profile}'] = {'status': 'failed', 'error': str(e)}
        
        if 'exit' in models_to_train:
            try:
                result = train_exit_optimizer_script(
                    profile=profile,
                    data_path=str(primary_data_path),
                    max_rows=args.data_rows,
                    total_timesteps=100_000,
                    device=args.device
                )
                results[f'exit_{profile}'] = result
            except Exception as e:
                logger.error(f"Exit optimizer training failed for {profile}: {e}")
                results[f'exit_{profile}'] = {'status': 'failed', 'error': str(e)}
    
    # YOLO training (profile-independent)
    if 'yolo' in models_to_train:
        try:
            result = train_yolo(
                epochs=min(args.epochs, 80),
                batch_size=args.batch_size,
                device=args.device
            )
            results['yolo'] = result
        except Exception as e:
            logger.error(f"YOLO training failed: {e}")
            results['yolo'] = {'status': 'failed', 'error': str(e)}
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("  TRAINING SUMMARY")
    logger.info("=" * 70)
    
    for name, result in results.items():
        status = result.get('status', 'completed')
        if status == 'failed':
            logger.info(f"  [FAILED] {name}: {result.get('error', 'Unknown')}")
        elif status == 'skipped':
            logger.info(f"  [SKIPPED] {name}: {result.get('reason', 'Unknown')}")
        else:
            logger.info(f"  [SUCCESS] {name}")
    
    # Save results
    results_path = PROJECT_ROOT / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
