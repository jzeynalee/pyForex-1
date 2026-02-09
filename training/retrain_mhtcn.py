"""
MH-TCN Retraining Pipeline
===========================

Fixes all issues from previous training:
1. Uses full 278-feature engineering (was 64)
2. Weighted sampling for balanced Bear/Side/Bull classes
3. Proper epochs (100+) with patient early stopping
4. Auto-optimal direction threshold per profile
5. Trains all 5 heads: direction, volatility, quantile, outcome
6. Saves to pyForex-assets with validation report

Usage:
    python -m training.retrain_mhtcn --all
    python -m training.retrain_mhtcn --profile SCALP
    python -m training.retrain_mhtcn --profile INTRADAY --epochs 150
"""

import argparse
import gc
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.train_mhtcn import (
    MHTCNDataPreparer,
    MHTCNTrainingConfig,
    compute_optimal_threshold,
    mhtcn_collate_fn,
)
from risk_management.phase1_predictive import (
    MultiHeadTCN,
    MultiHeadTCNTrainer,
    RiskDataset,
    TCNConfig,
    TrainingConfig,
    TradingProfile,
    compute_metrics,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

ASSETS_DIR = Path(r"E:\pyProject\pyForex-assets")
DATA_DIR = ASSETS_DIR / "data" / "mt5" / "EURUSD"
WEIGHTS_DIR = ASSETS_DIR / "models" / "weights"
LOCAL_WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"

PROFILE_DATA_MAP = {
    "SCALP": "M5",
    "INTRADAY": "M15",
    "SWING": "H1",
}

PROFILE_HORIZON_MAP = {
    "SCALP": 12,       # 12 * 5min = 1 hour ahead
    "INTRADAY": 8,     # 8 * 15min = 2 hours ahead
    "SWING": 6,        # 6 * 1H = 6 hours ahead
}

PROFILE_BARRIER_MAP = {
    "SCALP": {"tp_mult": 2.0, "sl_mult": 1.0, "max_bars": 24},
    "INTRADAY": {"tp_mult": 2.0, "sl_mult": 1.0, "max_bars": 16},
    "SWING": {"tp_mult": 2.5, "sl_mult": 1.0, "max_bars": 12},
}


# ============================================================================
# Data loading
# ============================================================================

def find_latest_data(timeframe: str) -> Path:
    """Find the latest CSV for a given timeframe."""
    pattern = f"EURUSD_{timeframe}_*.csv"
    files = sorted(DATA_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files in {DATA_DIR}")
    latest = files[-1]
    logger.info(f"Using data file: {latest.name}")
    return latest


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load CSV and validate OHLCV structure."""
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.sort_values("time", inplace=True)

    if "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df.reset_index(drop=True, inplace=True)
    logger.info(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")
    return df


# ============================================================================
# Feature engineering — full 278 features
# ============================================================================

def generate_full_features(df: pd.DataFrame) -> np.ndarray:
    """Generate ALL features via FeatureEngineerOptimized (278 cols)."""
    from alpha_factory.features_engineering import FeatureEngineerOptimized

    logger.info("Generating full feature set...")
    t0 = time.time()
    eng = FeatureEngineerOptimized()
    featured = eng.generate_features(df.copy(), batch_processing=False)

    # Keep only numeric columns
    feat_cols = [
        c for c in featured.columns
        if c not in ["time", "date", "datetime", "timestamp"]
        and featured[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.int8]
    ]
    features = featured[feat_cols].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    elapsed = time.time() - t0
    logger.info(f"Generated {features.shape[1]} features in {elapsed:.1f}s")
    return features, feat_cols


# ============================================================================
# Label generation (reuse existing logic)
# ============================================================================

def generate_all_labels(
    df: pd.DataFrame, profile: str
) -> Dict[str, np.ndarray]:
    """Generate direction, volatility, price-move, and outcome labels."""
    horizon = PROFILE_HORIZON_MAP[profile]
    barriers = PROFILE_BARRIER_MAP[profile]

    config = MHTCNTrainingConfig(
        profile=profile,
        direction_horizon=horizon,
        auto_threshold=True,
        use_triple_barrier=True,
        tp_multiplier=barriers["tp_mult"],
        sl_multiplier=barriers["sl_mult"],
        max_holding_bars=barriers["max_bars"],
    )

    preparer = MHTCNDataPreparer(config)
    direction = preparer.generate_direction_labels(df)
    volatility = preparer.generate_volatility_labels(df)
    price_move = preparer.generate_price_move_labels(df)
    outcomes = preparer.generate_outcome_labels(df)

    labels = {
        "direction": direction,
        "volatility": volatility,
        "price_move": price_move,
    }
    if outcomes is not None:
        labels["outcomes"] = outcomes

    return labels


# ============================================================================
# Dataset / DataLoader creation with balanced sampling
# ============================================================================

def make_weighted_sampler(
    direction_labels: np.ndarray, seq_len: int, dataset_len: int
) -> WeightedRandomSampler:
    """Create WeightedRandomSampler for balanced classes."""
    # Labels aligned with sequences (label = last timestep of each window)
    aligned = direction_labels[seq_len - 1 :][:dataset_len]
    unique, counts = np.unique(aligned, return_counts=True)
    class_weights = {int(c): len(aligned) / (3 * n) for c, n in zip(unique, counts)}
    sample_weights = np.array([class_weights.get(int(l), 1.0) for l in aligned])
    sample_weights = torch.from_numpy(sample_weights).float()

    logger.info(
        "Class weights: %s",
        {int(c): f"{class_weights.get(int(c), 0):.2f}" for c in sorted(class_weights)},
    )
    return WeightedRandomSampler(sample_weights, num_samples=dataset_len, replacement=True)


def create_loaders(
    features: np.ndarray,
    labels: Dict[str, np.ndarray],
    seq_len: int = 60,
    batch_size: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.10,
    use_weighted: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """Create train/val/test DataLoaders with temporal split."""
    n = len(features)
    test_size = int(n * test_split)
    val_size = int(n * val_split)
    train_size = n - val_size - test_size

    def _slice(start, end):
        sliced = {k: v[start:end] for k, v in labels.items()}
        sliced["features"] = features[start:end]
        return sliced

    train_d = _slice(0, train_size)
    val_d = _slice(train_size, train_size + val_size)
    test_d = _slice(train_size + val_size, n)

    logger.info(f"Split: train={train_size}, val={val_size}, test={test_size}")

    def _make_dataset(d):
        return RiskDataset(
            features=d["features"],
            direction_labels=d["direction"],
            volatility_labels=d["volatility"],
            price_move_labels=d["price_move"],
            sequence_length=seq_len,
            outcome_labels=d.get("outcomes"),
        )

    train_ds = _make_dataset(train_d)
    val_ds = _make_dataset(val_d)
    test_ds = _make_dataset(test_d)

    # Weighted sampler for training
    sampler = None
    shuffle = True
    if use_weighted and len(train_ds) > 0:
        sampler = make_weighted_sampler(
            train_d["direction"], seq_len, len(train_ds)
        )
        shuffle = False

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle,
        sampler=sampler, num_workers=0, pin_memory=pin,
        collate_fn=mhtcn_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=pin, collate_fn=mhtcn_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=pin, collate_fn=mhtcn_collate_fn,
    )

    return train_loader, val_loader, test_loader, test_d


# ============================================================================
# Model creation
# ============================================================================

def build_model(input_dim: int, profile: str, device: torch.device) -> MultiHeadTCN:
    """Create a fresh MH-TCN model with correct input dim."""
    profile_enum = TradingProfile[profile]
    config = TCNConfig(
        input_channels=input_dim,
        hidden_channels=128,
        profile=profile_enum,
    )
    model = MultiHeadTCN(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model: input={input_dim}, hidden=128, profile={profile}, "
        f"receptive_field={config.receptive_field}, params={n_params:,}"
    )
    return model


# ============================================================================
# Training loop
# ============================================================================

def train_profile(
    profile: str,
    epochs: int = 120,
    batch_size: int = 64,
    lr: float = 5e-4,
    seq_len: int = 60,
    patience: int = 20,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """Full training pipeline for one profile."""
    banner = f"  TRAINING MH-TCN: {profile}  "
    logger.info("\n" + "=" * len(banner))
    logger.info(banner)
    logger.info("=" * len(banner))

    device = torch.device(
        "cuda" if device_str == "auto" and torch.cuda.is_available() else
        device_str if device_str != "auto" else "cpu"
    )
    logger.info(f"Device: {device}")

    # 1. Load data
    tf = PROFILE_DATA_MAP[profile]
    data_path = find_latest_data(tf)
    df = load_and_validate(data_path)

    # 2. Full feature engineering
    features, feat_cols = generate_full_features(df)
    input_dim = features.shape[1]

    # 3. Labels
    labels = generate_all_labels(df, profile)

    # Align lengths
    min_len = min(len(features), *(len(v) for v in labels.values()))
    features = features[:min_len]
    labels = {k: v[:min_len] for k, v in labels.items()}

    # 4. Data loaders
    train_loader, val_loader, test_loader, test_data = create_loaders(
        features, labels,
        seq_len=seq_len, batch_size=batch_size,
        val_split=0.15, test_split=0.10, use_weighted=True,
    )

    # 5. Model
    model = build_model(input_dim, profile, device)

    # 6. Class weights for focal loss
    dir_labels = labels["direction"]
    unique, counts = np.unique(dir_labels, return_counts=True)
    cw = np.zeros(3)
    for cls, cnt in zip(unique, counts):
        cw[int(cls)] = len(dir_labels) / (3 * cnt + 1)
    cw = cw / cw.sum() * 3
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)
    logger.info(f"Class weights: Bear={cw[0]:.2f}, Side={cw[1]:.2f}, Bull={cw[2]:.2f}")

    # 7. Trainer
    train_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=lr,
        weight_decay=1e-5,
        num_epochs=epochs,
        patience=patience,
        min_delta=1e-4,
        grad_clip=1.0,
        warmup_epochs=5,
        use_uncertainty_weighting=False,  # Fixed weights — uncertainty suppresses direction
        direction_weight=3.0,             # Boosted to force direction head learning
        volatility_weight=1.0,
        quantile_weight=1.0,
        outcome_weight=1.5,               # Slightly boosted for SL/TP quality
        use_focal_loss=False,             # Disabled — focal down-weights at uniform p≈0.33
        focal_gamma=0.0,
        label_smoothing=0.0,              # Disabled — max gradient signal for direction
    )

    trainer = MultiHeadTCNTrainer(
        model=model, config=train_config,
        device=str(device), class_weights=class_weights,
    )

    # 8. Train
    t0 = time.time()
    history = trainer.train(train_loader, val_loader)
    train_time = time.time() - t0
    logger.info(f"Training complete in {train_time:.0f}s")

    # 9. Evaluate on test set
    test_loss, test_metrics = trainer.validate(test_loader)
    logger.info("=== TEST METRICS ===")
    for k, v in sorted(test_metrics.items()):
        logger.info(f"  {k}: {v:.4f}")

    # 10. Save checkpoint
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_state = trainer.best_model_state or {}

    checkpoint = {
        "state_dict": best_state,
        "config": {
            "profile": profile,
            "input_dim": input_dim,
            "hidden_dim": 128,
            "num_layers": len(TCNConfig(profile=TradingProfile[profile]).dilations),
            "seq_len": seq_len,
            "feature_schema_version": "full_v2",
        },
        "feature_columns": feat_cols,
        "num_classes": 3,
        "num_directions": 3,
        "num_quantiles": 5,
        "training_config": {
            "epochs_run": len(history.get("train_loss", [])),
            "epochs_max": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience,
            "weighted_sampling": True,
            "focal_loss": True,
            "data_file": data_path.name,
            "data_rows": len(df),
            "train_time_s": train_time,
        },
        "test_metrics": test_metrics,
        "history": history,
    }

    # Save locally
    LOCAL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    local_path = LOCAL_WEIGHTS_DIR / f"multihead_tcn_{profile}_{timestamp}.pth"
    torch.save(checkpoint, local_path)
    logger.info(f"Saved: {local_path}")

    # Save as canonical name (overwrite)
    canonical_local = LOCAL_WEIGHTS_DIR / f"multihead_tcn_{profile}.pth"
    torch.save(checkpoint, canonical_local)

    # Copy to assets
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    assets_path = WEIGHTS_DIR / f"multihead_tcn_{profile}.pth"
    assets_pa_v1 = WEIGHTS_DIR / f"multihead_tcn_{profile}_pa_v1.pth"
    shutil.copy2(str(canonical_local), str(assets_path))
    shutil.copy2(str(canonical_local), str(assets_pa_v1))
    logger.info(f"Copied to assets: {assets_path}")

    return {
        "profile": profile,
        "input_dim": input_dim,
        "data_rows": len(df),
        "epochs_run": len(history.get("train_loss", [])),
        "best_val_loss": trainer.best_val_loss,
        "test_metrics": test_metrics,
        "model_path": str(canonical_local),
        "train_time_s": train_time,
    }


# ============================================================================
# Validation report
# ============================================================================

def print_report(results: List[Dict[str, Any]]):
    """Print a summary comparison table."""
    print("\n" + "=" * 80)
    print("  MH-TCN RETRAINING REPORT")
    print("=" * 80)

    for r in results:
        m = r["test_metrics"]
        print(f"\n{'─' * 60}")
        print(f"  Profile: {r['profile']}")
        print(f"  Data: {r['data_rows']:,} rows, {r['input_dim']} features")
        print(f"  Epochs: {r['epochs_run']}, Best val loss: {r['best_val_loss']:.4f}")
        print(f"  Train time: {r['train_time_s']:.0f}s")
        print(f"  Model: {r['model_path']}")
        print(f"  ── Direction ──")
        print(f"    Overall:  {m.get('direction_accuracy', 0):.1%}")
        print(f"    Bear:     {m.get('direction_accuracy_bear', 0):.1%}")
        print(f"    Sideways: {m.get('direction_accuracy_sideways', 0):.1%}")
        print(f"    Bull:     {m.get('direction_accuracy_bull', 0):.1%}")
        print(f"  ── Volatility ──")
        print(f"    MAE:  {m.get('volatility_mae', 0):.6f}")
        print(f"    MAPE: {m.get('volatility_mape', 0):.2f}")
        print(f"  ── Quantiles ──")
        for q in [5, 25, 50, 75, 95]:
            cov = m.get(f"quantile_coverage_q{q}", 0)
            print(f"    Q{q:02d} coverage: {cov:.1%}")
        print(f"  ── Outcome (SL/TP) ──")
        print(f"    Long acc:  {m.get('outcome_accuracy_long', 0):.1%}")
        print(f"    Short acc: {m.get('outcome_accuracy_short', 0):.1%}")

    print(f"\n{'=' * 80}")
    print("  DONE — weights saved to both local and pyForex-assets")
    print("=" * 80)

    # Save JSON report
    report_path = LOCAL_WEIGHTS_DIR / "retrain_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON report: {report_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Retrain MH-TCN with full features")
    parser.add_argument("--profile", "-p", choices=["SCALP", "INTRADAY", "SWING"])
    parser.add_argument("--all", action="store_true", help="Train all 3 profiles")
    parser.add_argument("--epochs", "-e", type=int, default=120)
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%H:%M:%S",
    )

    profiles = ["SCALP", "INTRADAY", "SWING"] if args.all else (
        [args.profile] if args.profile else ["SCALP", "INTRADAY", "SWING"]
    )

    results = []
    for profile in profiles:
        try:
            r = train_profile(
                profile=profile,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seq_len=args.seq_len,
                patience=args.patience,
                device_str=args.device,
            )
            results.append(r)
        except Exception as e:
            logger.error(f"FAILED {profile}: {e}", exc_info=True)
            results.append({"profile": profile, "error": str(e)})

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
