"""
V6 ProbabilisticTCN + AlphaV2 — Train across all trading profiles & timeframes.

Profiles:
    SCALP:     M5, M15, H1
    INTRADAY:  M15, H1, H4
    SWING:     H1, H4, D1

Improvements over original training:
    1. Focal loss with auto pos_weight for class imbalance
    2. LR warmup (5 epochs) to avoid premature early peaks
    3. Adaptive model capacity based on dataset size

Usage:
    python -m research.train_v6_profiles --profile SCALP --device auto
    python -m research.train_v6_profiles --profile ALL --device auto
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.interfaces import RegimeLabel
from research.feature_pipeline import FeaturePipeline, ALL_CATEGORIES
from research.regime_detector import RegimeDetector
from research.alpha_heads import AlphaHeadV2
from research.mhtcn_filters.probabilistic import ProbabilisticTCN
from research.mhtcn_filters.training import (
    TrainingConfig,
    WalkForwardTrainer,
    compute_persistence_labels,
)
from research.train_mhtcn_variants import (
    load_ohlcv,
    generate_features_cached,
    detect_regimes,
    generate_alpha_signals,
    generate_labels,
    build_probability_sequences,
)

logger = logging.getLogger("research.train_v6_profiles")

# ─────────────────────────────────────────────────────────────────────
# Profile definitions
# ─────────────────────────────────────────────────────────────────────

PROFILES = {
    "SCALP": ["M5", "M15", "H1"],
    "INTRADAY": ["M15", "H1", "H4"],
    "SWING": ["H1", "H4", "D1"],
}

DATA_DIR = Path(r"E:\pyProject\data\raw")

TF_DATA_FILES = {
    "M5": DATA_DIR / "EURUSD_M5_latest.csv",
    "M15": DATA_DIR / "EURUSD_M15_latest.csv",
    "H1": DATA_DIR / "EURUSD_H1_latest.csv",
    "H4": DATA_DIR / "EURUSD_H4_latest.csv",
    "D1": DATA_DIR / "EURUSD_D1_latest.csv",
}

# Max rows to load per TF (to avoid OOM on M5/M15)
TF_MAX_ROWS = {
    "M5": 300_000,
    "M15": 300_000,
    "H1": 0,        # 0 = load all
    "H4": 0,
    "D1": 0,
}

# Regime scalar mapping (same as probabilistic.py)
_REGIME_TO_FLOAT = {
    RegimeLabel.TRENDING: 0.0,
    RegimeLabel.RANGING: 0.33,
    RegimeLabel.VOLATILE: 0.67,
    RegimeLabel.TRANSITION: 1.0,
}


# ─────────────────────────────────────────────────────────────────────
# Adaptive model capacity
# ─────────────────────────────────────────────────────────────────────

def get_adaptive_config(n_rows: int, tf: str) -> Dict:
    """Return model hyperparameters scaled to dataset size.

    Small datasets get reduced capacity to prevent overfitting.
    Large datasets get full capacity.
    """
    if n_rows < 20_000:
        # Small: D1 (~14K), H4 when filtered
        return {"hidden": 24, "num_layers": 2, "dropout": 0.25, "batch_size": 32}
    elif n_rows < 60_000:
        # Medium: H4 (~49K)
        return {"hidden": 32, "num_layers": 2, "dropout": 0.20, "batch_size": 64}
    elif n_rows < 200_000:
        # Large: H1 (~175K)
        return {"hidden": 48, "num_layers": 3, "dropout": 0.15, "batch_size": 128}
    else:
        # Very large: M15 (~300K+), M5 (~300K+)
        return {"hidden": 48, "num_layers": 3, "dropout": 0.15, "batch_size": 128}


def get_label_config(tf: str) -> Dict:
    """Return label horizon and persistence threshold tuned per timeframe.

    Shorter TFs need shorter horizons and tighter thresholds.
    """
    configs = {
        "M5":  {"label_horizon": 10, "persistence_threshold": 0.0002},
        "M15": {"label_horizon": 10, "persistence_threshold": 0.0003},
        "H1":  {"label_horizon": 15, "persistence_threshold": 0.0005},
        "H4":  {"label_horizon": 10, "persistence_threshold": 0.0008},
        "D1":  {"label_horizon": 5,  "persistence_threshold": 0.0015},
    }
    return configs.get(tf, {"label_horizon": 10, "persistence_threshold": 0.0005})


# ─────────────────────────────────────────────────────────────────────
# Train one profile/TF combination
# ─────────────────────────────────────────────────────────────────────

def train_v6_single(
    tf: str,
    profile: str,
    output_dir: Path,
    device: str = "cpu",
    epochs: int = 60,
    warmup: int = 200,
) -> Dict:
    """Train V6 ProbabilisticTCN + AlphaV2 for one timeframe."""
    logger.info("=" * 70)
    logger.info(f"  V6 Training: {profile}/{tf}")
    logger.info("=" * 70)

    data_path = TF_DATA_FILES[tf]
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return {"status": "skipped", "reason": f"missing data {data_path}"}

    # Load data (with row limit for large TFs)
    max_rows = TF_MAX_ROWS.get(tf, 0)
    df = load_ohlcv(str(data_path))
    if max_rows > 0 and len(df) > max_rows:
        df = df.iloc[-max_rows:].reset_index(drop=True)
        logger.info(f"Trimmed to last {max_rows:,} rows (from {len(df) + max_rows - max_rows:,})")

    n_rows = len(df)
    logger.info(f"Data: {n_rows:,} bars for {tf}")

    # Adaptive model config
    model_cfg = get_adaptive_config(n_rows, tf)
    label_cfg = get_label_config(tf)
    logger.info(f"Adaptive config: hidden={model_cfg['hidden']}, layers={model_cfg['num_layers']}, "
                f"dropout={model_cfg['dropout']}, batch={model_cfg['batch_size']}")
    logger.info(f"Label config: horizon={label_cfg['label_horizon']}, "
                f"threshold={label_cfg['persistence_threshold']}")

    # Generate features
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"features_{tf}.parquet"
    features = generate_features_cached(df, cache_path)

    # Detect regimes
    regimes = detect_regimes(df, features)
    prices = df["close"].values.astype(np.float64)

    # Generate AlphaV2 signals
    alpha_v2 = AlphaHeadV2()
    signals = generate_alpha_signals(alpha_v2, df, features, regimes, warmup)

    # Generate labels with TF-specific config
    labels = generate_labels(
        prices, signals["directions"],
        horizon=label_cfg["label_horizon"],
        threshold=label_cfg["persistence_threshold"],
    )

    # Build probability sequences (14-channel)
    prob_matrix = build_probability_sequences(
        signals["probabilities"], signals["category_probs"], regimes, df
    )
    logger.info(f"Probability matrix: {prob_matrix.shape}")

    # Training config with focal loss + warmup
    training_config = TrainingConfig(
        sequence_length=64,
        label_horizon=label_cfg["label_horizon"],
        persistence_threshold=label_cfg["persistence_threshold"],
        batch_size=model_cfg["batch_size"],
        learning_rate=5e-4,
        weight_decay=1e-5,
        num_epochs=epochs,
        patience=15,
        grad_clip=1.0,
        use_focal_loss=True,
        focal_gamma=2.0,
        auto_pos_weight=True,
        warmup_epochs=5,
    )

    # Create model with adaptive capacity
    model = ProbabilisticTCN(
        input_channels=14,
        hidden=model_cfg["hidden"],
        num_layers=model_cfg["num_layers"],
        kernel_size=3,
        dropout=model_cfg["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ProbabilisticTCN: {n_params:,} parameters")

    # Walk-forward train
    trainer = WalkForwardTrainer(training_config, device=device)
    train_ds, val_ds, test_ds = trainer.prepare_splits(prob_matrix, labels)

    if len(train_ds) == 0:
        logger.error(f"No training samples for {profile}/{tf}")
        return {"status": "skipped", "reason": "no training samples"}

    history = trainer.train(model, train_ds, val_ds)
    test_metrics = trainer.evaluate_test(model, test_ds)

    # Save weights
    weight_name = f"v6_prob_mhtcn_{profile}_{tf}.pt"
    weight_path = output_dir / weight_name
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model, str(weight_path))

    result = {
        "status": "completed",
        "profile": profile,
        "timeframe": tf,
        "model": "ProbabilisticTCN",
        "weight_path": str(weight_path),
        "n_bars": n_rows,
        "n_params": n_params,
        "hidden": model_cfg["hidden"],
        "num_layers": model_cfg["num_layers"],
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "epochs_trained": len(history.get("train_loss", [])),
        "best_val_loss": min(history.get("val_loss", [float("inf")])),
        "label_horizon": label_cfg["label_horizon"],
        "persistence_threshold": label_cfg["persistence_threshold"],
        "focal_loss": True,
        "warmup_epochs": 5,
        **test_metrics,
    }
    logger.info(f"Result: test_acc={test_metrics.get('test_acc', 'n/a'):.4f}, "
                f"brier={test_metrics.get('test_brier', 'n/a'):.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────

def train_profiles(
    profiles: List[str],
    output_dir: str,
    device: str = "cpu",
    epochs: int = 60,
) -> Dict:
    """Train V6 for all specified profiles and their timeframes."""
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for profile in profiles:
        if profile not in PROFILES:
            logger.error(f"Unknown profile: {profile}")
            continue

        timeframes = PROFILES[profile]
        logger.info("=" * 70)
        logger.info(f"  Profile: {profile} — Timeframes: {timeframes}")
        logger.info("=" * 70)

        for tf in timeframes:
            key = f"{profile}_{tf}"
            try:
                result = train_v6_single(
                    tf=tf,
                    profile=profile,
                    output_dir=out,
                    device=device,
                    epochs=epochs,
                )
                all_results[key] = result
            except Exception as e:
                logger.error(f"Training failed for {key}: {e}", exc_info=True)
                all_results[key] = {"status": "failed", "error": str(e)}

    elapsed = time.time() - t0

    # Summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "profiles": profiles,
        "device": device,
        "epochs": epochs,
        "improvements": [
            "focal_loss_gamma_2.0",
            "auto_pos_weight",
            "lr_warmup_5_epochs",
            "adaptive_model_capacity",
            "tf_specific_label_config",
        ],
        "models": all_results,
    }

    log_path = out / "v6_training_log.json"
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Training log saved to {log_path}")

    # Print summary table
    logger.info("=" * 80)
    logger.info(f"  V6 TRAINING COMPLETE — {len(all_results)} model(s)")
    logger.info("=" * 80)
    for key, res in all_results.items():
        status = res.get("status", "unknown")
        acc = res.get("test_acc", "n/a")
        brier = res.get("test_brier", "n/a")
        n_train = res.get("train_samples", 0)
        epochs_t = res.get("epochs_trained", 0)
        n_params = res.get("n_params", 0)
        if isinstance(acc, float):
            logger.info(
                f"  {key:20s} | {status:9s} | params={n_params:6,d} | "
                f"train={n_train:7,d} | epochs={epochs_t:2d} | "
                f"test_acc={acc:.4f} | brier={brier:.4f}"
            )
        else:
            logger.info(f"  {key:20s} | {status:9s} | {res.get('reason', res.get('error', ''))}")
    logger.info(f"  Total elapsed: {elapsed:.0f}s")
    logger.info("=" * 80)

    return summary


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train V6 ProbabilisticTCN+AlphaV2 across trading profiles"
    )
    parser.add_argument(
        "--profile", default="ALL",
        help="Profile to train: SCALP, INTRADAY, SWING, or ALL"
    )
    parser.add_argument(
        "--output-dir",
        default=r"E:\pyProject\pyForex-assets\models\v6_profiles",
        help="Output directory for weights and logs"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args()

    # Setup logging
    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "training.log", mode="w"),
        ],
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.profile.upper() == "ALL":
        profiles = list(PROFILES.keys())
    else:
        profiles = [p.strip().upper() for p in args.profile.split(",")]

    train_profiles(
        profiles=profiles,
        output_dir=args.output_dir,
        device=device,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
