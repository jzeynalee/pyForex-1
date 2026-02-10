"""
MH-TCN Training Orchestrator for the 6-Variant Research Framework.

Trains 4 MH-TCN models (V1/V3 are no-MH-TCN baselines):
    Model A: ResearchTCN      + AlphaV1 labels  →  V2
    Model B: ResearchTCN      + AlphaV2 labels  →  V4
    Model C: ProbabilisticTCN + AlphaV1 labels  →  V5
    Model D: ProbabilisticTCN + AlphaV2 labels  →  V6

Usage:
    python -m research.train_mhtcn_variants ^
        --data E:/pyProject/data/raw/EURUSD_H1_latest.csv ^
        --output-dir E:/pyProject/pyForex-assets/models/research ^
        --epochs 50 --device auto
"""

import argparse
import json
import logging
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.interfaces import AlphaSignal, Direction, RegimeLabel
from research.feature_pipeline import FeaturePipeline, ALL_CATEGORIES
from research.regime_detector import RegimeDetector
from research.alpha_heads import AlphaHeadV1, AlphaHeadV2
from research.mhtcn_filters.raw_feature import ResearchTCN
from research.mhtcn_filters.probabilistic import ProbabilisticTCN
from research.mhtcn_filters.training import (
    TrainingConfig,
    WalkForwardTrainer,
    MHTCNDataset,
    compute_persistence_labels,
)

# Reuse mature data loading / feature engineering from training/
try:
    from training.retrain_mhtcn import (
        load_and_validate as _load_and_validate,
        generate_full_features as _generate_full_features,
        ASSETS_DIR as RETRAIN_ASSETS_DIR,
    )
    _HAS_RETRAIN = True
except ImportError:
    _HAS_RETRAIN = False

logger = logging.getLogger("research.train_variants")

# Regime to float mapping (mirrored from probabilistic.py)
_REGIME_TO_FLOAT = {
    RegimeLabel.TRENDING: 0.0,
    RegimeLabel.RANGING: 0.33,
    RegimeLabel.VOLATILE: 0.67,
    RegimeLabel.TRANSITION: 1.0,
}


# ─────────────────────────────────────────────────────────────────────
# Phase 0: Data loading & feature generation
# ─────────────────────────────────────────────────────────────────────

def load_ohlcv(path: str) -> pd.DataFrame:
    """Load and validate OHLCV data.

    Delegates to training.retrain_mhtcn.load_and_validate when available.
    """
    if _HAS_RETRAIN:
        logger.info("Using training.retrain_mhtcn.load_and_validate")
        return _load_and_validate(Path(path))

    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df.sort_values("time", inplace=True)

    if "volume" not in df.columns:
        df["volume"] = df.get("tick_volume", 1000)

    df.reset_index(drop=True, inplace=True)
    logger.info(f"Loaded {len(df):,} bars")
    return df


def generate_features_cached(
    df: pd.DataFrame, cache_path: Optional[Path] = None
) -> pd.DataFrame:
    """Generate features with optional disk caching.

    Prefers the full 278-feature engineering from
    training.retrain_mhtcn.generate_full_features when available,
    falling back to research.feature_pipeline.FeaturePipeline.
    """
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    if _HAS_RETRAIN:
        logger.info("Generating features via training.retrain_mhtcn.generate_full_features")
        feat_array, feat_cols = _generate_full_features(df)
        features = pd.DataFrame(feat_array, columns=feat_cols, index=df.index)
    else:
        logger.info("Generating features via research.feature_pipeline.FeaturePipeline")
        pipeline = FeaturePipeline()
        features = pipeline.compute_features(df)

    features = features.fillna(0.0)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(cache_path)
        logger.info(f"Cached features to {cache_path}")

    return features


def detect_regimes(
    df: pd.DataFrame, features: Optional[pd.DataFrame] = None
) -> np.ndarray:
    """Run regime detector on every bar, return RegimeLabel array."""
    logger.info("Detecting regimes ...")
    detector = RegimeDetector()
    regime_series = detector.detect_series(df, features)
    return regime_series.values


# ─────────────────────────────────────────────────────────────────────
# Phase 1: Alpha signal generation
# ─────────────────────────────────────────────────────────────────────

def generate_alpha_signals(
    alpha_head,
    df: pd.DataFrame,
    features: pd.DataFrame,
    regimes: np.ndarray,
    warmup: int = 200,
) -> Dict[str, np.ndarray]:
    """Run an alpha head on every bar after warmup.

    Returns dict with:
        directions: int array (+1, -1, 0)
        probabilities: float array
        category_probs: (n_bars, 6) float array (AlphaV2 only; zeros for V1)
    """
    n = len(df)
    directions = np.zeros(n, dtype=np.int32)
    probabilities = np.zeros(n, dtype=np.float32)
    cat_probs = np.zeros((n, len(ALL_CATEGORIES)), dtype=np.float32)

    alpha_head.reset()

    # AlphaV2 needs category→column mapping to produce directional signals
    if hasattr(alpha_head, "set_category_columns"):
        from .feature_pipeline import CATEGORY_DEFINITIONS, _safe_match
        feat_cols = features.columns.tolist()
        cat_map = {}
        for cat, patterns in CATEGORY_DEFINITIONS.items():
            cat_map[cat] = _safe_match(feat_cols, patterns)
        alpha_head.set_category_columns(cat_map)
        n_mapped = sum(len(v) for v in cat_map.values())
        logger.info(f"Injected category mapping: {n_mapped} features across {len(cat_map)} categories")

    for i in range(warmup, n):
        window_df = df.iloc[max(0, i - 500) : i + 1]
        window_feat = features.iloc[max(0, i - 500) : i + 1]
        regime = regimes[i]

        try:
            signal: AlphaSignal = alpha_head.evaluate(window_df, window_feat, regime)
        except Exception as e:
            logger.debug(f"Alpha eval error at bar {i}: {e}")
            continue

        if signal.direction == Direction.LONG:
            directions[i] = 1
        elif signal.direction == Direction.SHORT:
            directions[i] = -1
        # else HOLD → 0

        probabilities[i] = signal.probability

        for j, cat in enumerate(ALL_CATEGORIES):
            cat_probs[i, j] = signal.category_probs.get(cat, 0.5)

    n_long = int(np.sum(directions == 1))
    n_short = int(np.sum(directions == -1))
    n_hold = int(np.sum(directions == 0))
    logger.info(
        f"{alpha_head.name()} signals: LONG={n_long}, SHORT={n_short}, HOLD={n_hold}"
    )
    return {
        "directions": directions,
        "probabilities": probabilities,
        "category_probs": cat_probs,
    }


# ─────────────────────────────────────────────────────────────────────
# Phase 2: Label generation
# ─────────────────────────────────────────────────────────────────────

def generate_labels(
    prices: np.ndarray,
    directions: np.ndarray,
    horizon: int = 20,
    threshold: float = 0.0005,
) -> np.ndarray:
    """Compute leak-free persistence labels."""
    labels = compute_persistence_labels(prices, directions, horizon, threshold)
    valid = labels >= 0
    pos_rate = labels[valid].mean() if valid.any() else 0.0
    logger.info(
        f"Labels: {int(valid.sum()):,} valid / {len(labels):,} total, "
        f"pos_rate={pos_rate:.3f}"
    )
    return labels


# ─────────────────────────────────────────────────────────────────────
# Phase 3: Train RawFeature ResearchTCN
# ─────────────────────────────────────────────────────────────────────

def train_raw_feature_tcn(
    features: pd.DataFrame,
    labels: np.ndarray,
    config: TrainingConfig,
    output_path: Path,
    device: str = "cpu",
    max_features: int = 64,
) -> Dict:
    """Train a ResearchTCN on raw features + persistence labels."""
    logger.info("=" * 60)
    logger.info("  Training ResearchTCN (RawFeature)")
    logger.info("=" * 60)

    # Prepare numeric feature matrix
    numeric = features.select_dtypes(include=[np.number])
    if numeric.shape[1] > max_features:
        numeric = numeric.iloc[:, :max_features]

    feature_matrix = numeric.values.astype(np.float32)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Z-score per column (full dataset, not per window — that happens inside filter)
    mean = feature_matrix.mean(axis=0, keepdims=True)
    std = feature_matrix.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    feature_matrix = (feature_matrix - mean) / std

    n_features = feature_matrix.shape[1]
    logger.info(f"Feature matrix: {feature_matrix.shape}")

    # Create model
    model = ResearchTCN(
        input_channels=n_features,
        hidden=64,
        num_layers=4,
        kernel_size=3,
        dropout=0.2,
    )

    # Walk-forward split & train
    trainer = WalkForwardTrainer(config, device=device)
    train_ds, val_ds, test_ds = trainer.prepare_splits(feature_matrix, labels)

    if len(train_ds) == 0:
        logger.error("No training samples — skipping ResearchTCN")
        return {"status": "skipped", "reason": "no training samples"}

    history = trainer.train(model, train_ds, val_ds)
    test_metrics = trainer.evaluate_test(model, test_ds)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model, str(output_path))

    result = {
        "status": "completed",
        "model": "ResearchTCN",
        "output_path": str(output_path),
        "n_features": n_features,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "epochs_trained": len(history.get("train_loss", [])),
        "best_val_loss": min(history.get("val_loss", [float("inf")])),
        **test_metrics,
    }
    logger.info(f"ResearchTCN result: {result}")
    return result


# ─────────────────────────────────────────────────────────────────────
# Phase 4: Train ProbabilisticTCN
# ─────────────────────────────────────────────────────────────────────

def build_probability_sequences(
    probabilities: np.ndarray,
    category_probs: np.ndarray,
    regimes: np.ndarray,
    df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Build (n_bars, 14) feature matrix for ProbabilisticTCN.

    Channels:
        0-5:  category probabilities (trend, momentum, oscillator, volatility, volume, structure)
        6:    P_alpha_final
        7:    regime_scalar
        8:    return_1bar
        9:    return_5bar
        10:   return_10bar
        11:   norm_atr
        12:   rsi_norm
        13:   momentum_sign
    """
    n = len(probabilities)
    N_CHANNELS = 14
    matrix = np.full((n, N_CHANNELS), 0.5, dtype=np.float32)

    # Channels 0-5: category probs
    n_cats = min(category_probs.shape[1], 6)
    matrix[:, :n_cats] = category_probs[:, :n_cats]

    # Channel 6: alpha probability
    matrix[:, 6] = np.clip(probabilities, 0.0, 1.0)

    # Channel 7: regime scalar
    for i in range(n):
        r = regimes[i] if i < len(regimes) else RegimeLabel.RANGING
        matrix[i, 7] = _REGIME_TO_FLOAT.get(r, 0.5)

    # Channels 8-13: price-derived features
    if df is not None:
        close_col = "close" if "close" in df.columns else "Close"
        high_col = "high" if "high" in df.columns else "High"
        low_col = "low" if "low" in df.columns else "Low"
        closes = df[close_col].values.astype(np.float64)
        highs = df[high_col].values.astype(np.float64)
        lows = df[low_col].values.astype(np.float64)

        # 1-bar return
        ret1 = np.zeros(n)
        ret1[1:] = (closes[1:] - closes[:-1]) / np.maximum(closes[:-1], 1e-10)
        matrix[:, 8] = np.clip(ret1, -0.05, 0.05).astype(np.float32)

        # 5-bar return
        ret5 = np.zeros(n)
        ret5[5:] = (closes[5:] - closes[:-5]) / np.maximum(closes[:-5], 1e-10)
        matrix[:, 9] = np.clip(ret5, -0.05, 0.05).astype(np.float32)

        # 10-bar return
        ret10 = np.zeros(n)
        ret10[10:] = (closes[10:] - closes[:-10]) / np.maximum(closes[:-10], 1e-10)
        matrix[:, 10] = np.clip(ret10, -0.05, 0.05).astype(np.float32)

        # Normalized ATR (14-bar)
        tr = np.maximum(
            highs - lows,
            np.maximum(
                np.abs(highs - np.roll(closes, 1)),
                np.abs(lows - np.roll(closes, 1)),
            ),
        )
        tr[0] = highs[0] - lows[0]
        atr14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
        norm_atr = atr14 / np.maximum(closes, 1e-10)
        matrix[:, 11] = np.clip(norm_atr, 0, 0.1).astype(np.float32)

        # RSI (14-bar, normalized to [0,1])
        deltas = np.diff(closes, prepend=closes[0])
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        avg_gain = pd.Series(gains).rolling(14, min_periods=1).mean().values
        avg_loss = pd.Series(losses).rolling(14, min_periods=1).mean().values
        rs = avg_gain / np.maximum(avg_loss, 1e-10)
        rsi_norm = rs / (1.0 + rs)
        matrix[:, 12] = rsi_norm.astype(np.float32)

        # Signed momentum (10-bar, scaled)
        mom = np.zeros(n)
        mom[10:] = (closes[10:] - closes[:-10]) / np.maximum(closes[:-10], 1e-10)
        matrix[:, 13] = np.clip(mom * 10, -1.0, 1.0).astype(np.float32)

    return matrix


def train_probabilistic_tcn(
    prob_matrix: np.ndarray,
    labels: np.ndarray,
    config: TrainingConfig,
    output_path: Path,
    device: str = "cpu",
    input_channels: int = 14,
) -> Dict:
    """Train a ProbabilisticTCN on probability sequences + persistence labels."""
    logger.info("=" * 60)
    logger.info("  Training ProbabilisticTCN")
    logger.info("=" * 60)
    logger.info(f"Probability matrix: {prob_matrix.shape}")

    # Use seq_len=64 for probabilistic (matches ProbabilisticMHTCNFilter default)
    config_copy = TrainingConfig(
        sequence_length=64,
        label_horizon=config.label_horizon,
        persistence_threshold=config.persistence_threshold,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_epochs=config.num_epochs,
        patience=max(config.patience, 15),
        grad_clip=config.grad_clip,
        shuffle_labels=config.shuffle_labels,
    )

    model = ProbabilisticTCN(
        input_channels=input_channels,
        hidden=48,
        num_layers=3,
        kernel_size=3,
        dropout=0.15,
    )

    trainer = WalkForwardTrainer(config_copy, device=device)
    train_ds, val_ds, test_ds = trainer.prepare_splits(prob_matrix, labels)

    if len(train_ds) == 0:
        logger.error("No training samples — skipping ProbabilisticTCN")
        return {"status": "skipped", "reason": "no training samples"}

    history = trainer.train(model, train_ds, val_ds)
    test_metrics = trainer.evaluate_test(model, test_ds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(model, str(output_path))

    result = {
        "status": "completed",
        "model": "ProbabilisticTCN",
        "output_path": str(output_path),
        "input_channels": 8,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "epochs_trained": len(history.get("train_loss", [])),
        "best_val_loss": min(history.get("val_loss", [float("inf")])),
        **test_metrics,
    }
    logger.info(f"ProbabilisticTCN result: {result}")
    return result


# ─────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────

def run_all(
    data_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    label_horizon: int = 20,
    persistence_threshold: float = 0.0005,
    max_features: int = 64,
    warmup: int = 200,
    device: str = "cpu",
    variants: Optional[List[str]] = None,
) -> Dict:
    """Training pipeline for MH-TCN research models.

    Args:
        variants: List of variant IDs to train, e.g. ["V6"] or ["V2","V6"].
                  None or empty means train all 4 (V2, V4, V5, V6).
    """
    _ALL_VARIANTS = {"V2", "V4", "V5", "V6"}
    if not variants:
        target_variants = _ALL_VARIANTS
    else:
        target_variants = {v.upper() for v in variants} & _ALL_VARIANTS
        if not target_variants:
            raise ValueError(f"No valid variants in {variants}. Choose from {_ALL_VARIANTS}")
    logger.info(f"Target variants: {sorted(target_variants)}")

    # Determine which alpha heads are needed
    need_v1 = bool(target_variants & {"V2", "V5"})
    need_v2 = bool(target_variants & {"V4", "V6"})
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    training_config = TrainingConfig(
        sequence_length=60,
        label_horizon=label_horizon,
        persistence_threshold=persistence_threshold,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=epochs,
        patience=8,
    )

    # ── Phase 0: Load data & features ──────────────────────────────
    df = load_ohlcv(data_path)
    cache_path = out / "cached_features.parquet"
    features = generate_features_cached(df, cache_path)
    regimes = detect_regimes(df, features)
    prices = df["close"].values.astype(np.float64)

    # ── Phase 1: Generate alpha signals ────────────────────────────
    logger.info("=" * 60)
    logger.info("  Phase 1: Alpha Signal Generation")
    logger.info("=" * 60)

    signals_v1, signals_v2 = None, None

    if need_v1:
        alpha_v1 = AlphaHeadV1()
        signals_v1 = generate_alpha_signals(alpha_v1, df, features, regimes, warmup)
        np.savez_compressed(out / "cached_signals_v1.npz", **signals_v1)
        logger.info("AlphaV1 signals cached.")

    if need_v2:
        alpha_v2 = AlphaHeadV2()
        signals_v2 = generate_alpha_signals(alpha_v2, df, features, regimes, warmup)
        np.savez_compressed(out / "cached_signals_v2.npz", **signals_v2)
        logger.info("AlphaV2 signals cached.")

    # ── Phase 2: Label generation ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Phase 2: Label Generation")
    logger.info("=" * 60)

    labels_v1, labels_v2 = None, None

    if need_v1:
        labels_v1 = generate_labels(
            prices, signals_v1["directions"], label_horizon, persistence_threshold
        )
    if need_v2:
        labels_v2 = generate_labels(
            prices, signals_v2["directions"], label_horizon, persistence_threshold
        )

    # ── Phase 3: Train RawFeature ResearchTCN ──────────────────────
    results = {}

    if target_variants & {"V2", "V4"}:
        logger.info("=" * 60)
        logger.info("  Phase 3: Train ResearchTCN models")
        logger.info("=" * 60)

        if "V2" in target_variants:
            results["V2_raw_mhtcn_alphaV1"] = train_raw_feature_tcn(
                features=features,
                labels=labels_v1,
                config=training_config,
                output_path=out / "raw_mhtcn_alphaV1.pt",
                device=device,
                max_features=max_features,
            )

        if "V4" in target_variants:
            results["V4_raw_mhtcn_alphaV2"] = train_raw_feature_tcn(
                features=features,
                labels=labels_v2,
                config=training_config,
                output_path=out / "raw_mhtcn_alphaV2.pt",
                device=device,
                max_features=max_features,
            )

    # ── Phase 4: Train ProbabilisticTCN ────────────────────────────
    if target_variants & {"V5", "V6"}:
        logger.info("=" * 60)
        logger.info("  Phase 4: Train ProbabilisticTCN models")
        logger.info("=" * 60)

        if "V5" in target_variants:
            prob_matrix_v1 = build_probability_sequences(
                signals_v1["probabilities"], signals_v1["category_probs"], regimes, df
            )
            results["V5_prob_mhtcn_alphaV1"] = train_probabilistic_tcn(
                prob_matrix=prob_matrix_v1,
                labels=labels_v1,
                config=training_config,
                output_path=out / "prob_mhtcn_alphaV1.pt",
                device=device,
            )

        if "V6" in target_variants:
            prob_matrix_v2 = build_probability_sequences(
                signals_v2["probabilities"], signals_v2["category_probs"], regimes, df
            )
            results["V6_prob_mhtcn_alphaV2"] = train_probabilistic_tcn(
                prob_matrix=prob_matrix_v2,
                labels=labels_v2,
                config=training_config,
                output_path=out / "prob_mhtcn_alphaV2.pt",
                device=device,
            )

    # ── Summary ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_path": data_path,
        "n_bars": len(df),
        "n_features": features.shape[1],
        "label_horizon": label_horizon,
        "persistence_threshold": persistence_threshold,
        "epochs": epochs,
        "device": device,
        "elapsed_seconds": round(elapsed, 1),
        "models": results,
    }

    log_path = out / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Training log saved to {log_path}")

    # Print final summary
    logger.info("=" * 70)
    logger.info(f"  TRAINING COMPLETE — {len(results)} MH-TCN model(s) for variants {sorted(target_variants)}")
    logger.info("=" * 70)
    for name, res in results.items():
        status = res.get("status", "unknown")
        acc = res.get("test_acc", "n/a")
        brier = res.get("test_brier", "n/a")
        n_train = res.get("train_samples", 0)
        logger.info(
            f"  {name:35s} | status={status:10s} | "
            f"train={n_train:6,d} | test_acc={acc} | brier={brier}"
        )
    logger.info(f"  Total elapsed: {elapsed:.0f}s")
    logger.info("=" * 70)

    return summary


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train MH-TCN models for the 6-variant research framework"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to OHLCV CSV (e.g. E:\\pyProject\\data\\raw\\EURUSD_H1_latest.csv)"
    )
    parser.add_argument(
        "--output-dir", default="E:\\pyProject\\pyForex-assets\\models\\research",
        help="Directory for trained weights and logs"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label-horizon", type=int, default=20)
    parser.add_argument("--persistence-threshold", type=float, default=0.0005)
    parser.add_argument("--max-features", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--variant", nargs="+", default=None,
        help="Train specific variant(s) only, e.g. --variant V6 or --variant V2 V6"
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "training.log", mode="w"),
        ],
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    run_all(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        label_horizon=args.label_horizon,
        persistence_threshold=args.persistence_threshold,
        max_features=args.max_features,
        warmup=args.warmup,
        device=device,
        variants=args.variant,
    )


if __name__ == "__main__":
    main()
