#!/usr/bin/env python
"""
Train all remaining untrained ML models in pyForex-1.

Models trained:
  1. MetaLabelingModel   (Phase 3 – GBM trade filter)         per profile
  2. TemporalRefinementTCN (Alpha Factory – probability refiner) per profile
  3. PPO ActorCritic     (Phase 4 – RL exit optimizer)         per profile

Usage:
    python -m research.train_remaining_models --device cuda
    python -m research.train_remaining_models --model meta       # just meta-labeling
    python -m research.train_remaining_models --model temporal   # just temporal refinement
    python -m research.train_remaining_models --model ppo        # just PPO
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-45s | %(message)s",
)
logger = logging.getLogger("research.train_remaining_models")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(r"E:\pyProject\data\raw")
ASSETS_DIR = Path(r"E:\pyProject\pyForex-assets\models")
WEIGHTS_DIR = ASSETS_DIR / "weights"

PROFILES = {
    "SCALP":    {"timeframes": ["M5", "M15", "H1"],  "primary_tf": "M15"},
    "INTRADAY": {"timeframes": ["M15", "H1", "H4"],  "primary_tf": "H1"},
    "SWING":    {"timeframes": ["H1", "H4", "D1"],   "primary_tf": "H4"},
}

# Map from config num_layers → correct dilations list for MultiHeadTCN
PROFILE_DILATIONS = {
    "SCALP":    [1, 2, 4, 8],
    "INTRADAY": [1, 2, 4, 8, 16],
    "SWING":    [1, 2, 4, 8, 16, 32, 64],
}


# ===================================================================
# Utility: load OHLCV data
# ===================================================================
def load_ohlcv(timeframe: str, max_rows: int = 300_000) -> pd.DataFrame:
    """Load EURUSD OHLCV CSV for a given timeframe."""
    path = DATA_DIR / f"EURUSD_{timeframe}_latest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_csv(path)
    # Normalize column names
    col_map = {c: c.lower() for c in df.columns}
    df.rename(columns=col_map, inplace=True)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    elif "datetime" in df.columns:
        df.rename(columns={"datetime": "time"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"])
    # Limit rows
    if len(df) > max_rows:
        df = df.iloc[-max_rows:].reset_index(drop=True)
    logger.info(f"Loaded {timeframe}: {len(df)} bars from {path.name}")
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic technical indicators needed for meta-features."""
    c = df["close"]
    h = df["high"]
    l = df["low"]

    # ATR-14
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()

    # Returns
    df["ret_1"] = c.pct_change(1)
    df["ret_5"] = c.pct_change(5)
    df["ret_10"] = c.pct_change(10)

    # RSI-14
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # Momentum-10
    df["momentum"] = c / c.shift(10) - 1.0

    # EMA-20, EMA-50
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()

    # Spread proxy (high - low)
    df["spread"] = h - l

    # Volume (use tick_volume if present)
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ===================================================================
# 1. MetaLabelingModel Training
# ===================================================================
def _generate_direction_labels(close: np.ndarray, horizon: int, threshold: float) -> np.ndarray:
    """Generate 3-class direction labels from forward returns.
    Returns: -1=BEAR, 0=SIDEWAYS, 1=BULL.  NaN-padded at the end."""
    n = len(close)
    labels = np.full(n, np.nan)
    for i in range(n - horizon):
        ret = (close[i + horizon] - close[i]) / close[i]
        if ret > threshold:
            labels[i] = 1
        elif ret < -threshold:
            labels[i] = -1
        else:
            labels[i] = 0
    return labels


def _generate_barrier_outcomes(close: np.ndarray, directions: np.ndarray,
                               horizon: int, tp_mult: float = 1.5,
                               atr: np.ndarray = None) -> np.ndarray:
    """Simplified triple-barrier outcome: 1=WIN, 0=LOSS/TIMEOUT."""
    n = len(close)
    outcomes = np.zeros(n, dtype=np.int32)
    if atr is None:
        atr = np.full(n, np.std(np.diff(close[:min(500, n)])))

    for i in range(n - horizon):
        d = int(directions[i])
        if d == 0:
            continue
        entry = close[i]
        sl_dist = atr[i] * 1.0
        tp_dist = atr[i] * tp_mult
        hit_tp = False
        for j in range(1, horizon + 1):
            pnl = (close[i + j] - entry) * d
            if pnl >= tp_dist:
                hit_tp = True
                break
            if pnl <= -sl_dist:
                break
        outcomes[i] = 1 if hit_tp else 0
    return outcomes


def train_meta_labeling(profile: str, device: str, output_dir: Path) -> Dict:
    """Train MetaLabelingModel for a given profile."""
    from risk_management.phase3_filtering.meta_labeling import (
        MetaLabelingModel, MetaLabelingConfig, MetaFeatureExtractor,
    )

    primary_tf = PROFILES[profile]["primary_tf"]
    logger.info(f"[META] Training MetaLabelingModel for {profile} ({primary_tf})")

    # Load and prepare data
    df = load_ohlcv(primary_tf)
    df = add_technical_indicators(df)
    close = df["close"].values
    atr = df["atr"].values

    # Horizon and threshold per profile
    horizon_map = {"SCALP": 10, "INTRADAY": 15, "SWING": 20}
    thresh_map = {"SCALP": 0.0003, "INTRADAY": 0.0005, "SWING": 0.001}
    horizon = horizon_map[profile]
    threshold = thresh_map[profile]

    # Generate primary direction labels
    dir_labels = _generate_direction_labels(close, horizon, threshold)

    # Generate simulated "primary model" direction probabilities
    # Using momentum + RSI as proxy signals
    rsi = df["rsi"].values / 100.0
    mom = np.clip(df["momentum"].values * 20, -1, 1)
    p_bull = np.clip(0.33 + 0.2 * mom + 0.15 * (rsi - 0.5), 0.05, 0.90)
    p_bear = np.clip(0.33 - 0.2 * mom - 0.15 * (rsi - 0.5), 0.05, 0.90)
    p_side = np.clip(1.0 - p_bull - p_bear, 0.05, 0.90)
    total = p_bull + p_bear + p_side
    p_bull /= total; p_bear /= total; p_side /= total

    direction_probs = np.stack([p_bear, p_side, p_bull], axis=1)

    # Primary predicted direction (argmax → mapped to -1, 0, 1)
    pred_class = np.argmax(direction_probs, axis=1)
    pred_direction = pred_class.astype(float) - 1.0  # 0→-1, 1→0, 2→1

    # Valid mask (where we have labels)
    valid = ~np.isnan(dir_labels)
    idx = np.where(valid)[0]

    # Barrier outcomes
    barrier_outcomes = _generate_barrier_outcomes(
        close, pred_direction, horizon, tp_mult=1.5, atr=atr
    )

    # Create meta-labels: 1 if prediction was correct (matches actual direction and barrier wins)
    meta_labels = np.zeros(len(close), dtype=np.int32)
    for i in idx:
        actual = dir_labels[i]
        pred = pred_direction[i]
        if pred != 0 and pred == actual and barrier_outcomes[i] == 1:
            meta_labels[i] = 1

    # Build meta-features using the built-in extractor
    config = MetaLabelingConfig(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        early_stopping_rounds=20,
    )
    feature_extractor = MetaFeatureExtractor(config)

    primary_predictions = {
        "direction_probs": direction_probs[idx],
        "volatility": atr[idx],
    }

    market_data = df.iloc[idx][["spread", "atr", "volume"]].reset_index(drop=True)
    timestamps = pd.to_datetime(df.iloc[idx]["time"].values) if "time" in df.columns else None

    X = feature_extractor.extract_features(
        primary_predictions=primary_predictions,
        market_data=market_data,
        timestamps=timestamps,
    )
    y = meta_labels[idx]

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    logger.info(f"[META] {profile}: X shape={X.shape}, y mean={y.mean():.3f} "
                f"(pos={y.sum()}, neg={len(y)-y.sum()})")

    # Train model
    model = MetaLabelingModel(config)
    model.feature_extractor = feature_extractor  # Keep feature names consistent
    metrics = model.train(X, y, validation_split=0.2)

    # Cross-validate
    cv_metrics = model.cross_validate(X, y, n_splits=5)

    # Optimize threshold
    split_idx = int(len(X) * 0.8)
    optimal_thresh = model.optimize_threshold(X[split_idx:], y[split_idx:], metric="f1")

    # Save
    save_path = output_dir / f"meta_labeling_{profile}.joblib"
    model.save(str(save_path))

    result = {
        "status": "completed",
        "profile": profile,
        "timeframe": primary_tf,
        "weight_path": str(save_path),
        "n_samples": len(y),
        "pos_rate": float(y.mean()),
        "threshold": float(optimal_thresh),
        **{k: float(v) for k, v in metrics.items()},
        "cv_accuracy_mean": float(np.mean(cv_metrics["accuracy"])),
        "cv_f1_mean": float(np.mean(cv_metrics["f1"])),
    }
    logger.info(f"[META] {profile} done: acc={metrics['accuracy']:.3f}, "
                f"f1={metrics['f1']:.3f}, roc_auc={metrics['roc_auc']:.3f}")
    return result


# ===================================================================
# 2. TemporalRefinementTCN Training
# ===================================================================
def _build_prob_sequences(df: pd.DataFrame, seq_len: int = 20,
                          horizon: int = 10, threshold: float = 0.0005
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Build probability sequences and regime labels from price data.

    Each sample: (seq_len, 4) = [P(bull), P(bear), P(neutral), stability]
    Label: 0=bear, 1=neutral, 2=bull (based on forward return).
    """
    close = df["close"].values
    rsi = df["rsi"].values / 100.0
    mom = np.clip(df["momentum"].values * 20, -1, 1)
    atr = df["atr"].values
    atr_pct = atr / (close + 1e-10)

    # Probability estimates from momentum/RSI
    p_bull = np.clip(0.33 + 0.25 * mom + 0.15 * (rsi - 0.5), 0.05, 0.90)
    p_bear = np.clip(0.33 - 0.25 * mom - 0.15 * (rsi - 0.5), 0.05, 0.90)
    p_neut = np.clip(1.0 - p_bull - p_bear, 0.05, 0.90)
    total = p_bull + p_bear + p_neut
    p_bull /= total; p_bear /= total; p_neut /= total

    # Stability ∈ [0, 1]: lower ATR percentile → higher stability
    atr_roll = pd.Series(atr_pct).rolling(50, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10), raw=False
    ).fillna(0.5).values
    stability = 1.0 - atr_roll

    prob_matrix = np.stack([p_bull, p_bear, p_neut, stability], axis=1)  # (N, 4)

    # Forward-return regime labels
    n = len(close)
    sequences = []
    labels = []
    for i in range(seq_len, n - horizon):
        seq = prob_matrix[i - seq_len:i]
        fwd_ret = (close[i + horizon] - close[i]) / close[i]
        if fwd_ret > threshold:
            label = 2  # bull
        elif fwd_ret < -threshold:
            label = 0  # bear
        else:
            label = 1  # neutral
        sequences.append(seq)
        labels.append(label)

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)


def train_temporal_refinement(profile: str, device: str, output_dir: Path) -> Dict:
    """Train TemporalRefinementTCN for a given profile."""
    from alpha_factory.mhtcn_temporal_refinement import (
        TemporalRefinementTCN, TemporalRefinementConfig,
        TemporalRefinementDataset, TemporalRefinementTrainer,
    )

    primary_tf = PROFILES[profile]["primary_tf"]
    logger.info(f"[TEMPORAL] Training TemporalRefinementTCN for {profile} ({primary_tf})")

    # Load and prepare data
    df = load_ohlcv(primary_tf, max_rows=200_000)
    df = add_technical_indicators(df)

    seq_len = 20
    horizon_map = {"SCALP": 10, "INTRADAY": 15, "SWING": 20}
    thresh_map = {"SCALP": 0.0003, "INTRADAY": 0.0005, "SWING": 0.001}
    horizon = horizon_map[profile]
    threshold = thresh_map[profile]

    sequences, labels = _build_prob_sequences(df, seq_len, horizon, threshold)
    logger.info(f"[TEMPORAL] {profile}: {len(sequences)} samples, "
                f"class dist: {np.bincount(labels, minlength=3)}")

    # Train/val split (chronological)
    n = len(sequences)
    train_end = int(n * 0.8)
    val_start = train_end

    train_ds = TemporalRefinementDataset(
        sequences[:train_end], labels[:train_end], sequence_length=seq_len
    )
    val_ds = TemporalRefinementDataset(
        sequences[val_start:], labels[val_start:], sequence_length=seq_len
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=0
    )

    # Create model
    config = TemporalRefinementConfig(
        sequence_length=seq_len,
        input_channels=4,
        hidden_channels=32,
        num_layers=3,
        kernel_size=3,
        dropout=0.2,
        num_regimes=3,
        output_confidence_adjustment=True,
        learning_rate=1e-3,
        weight_decay=1e-5,
    )
    dev = torch.device(device)
    model = TemporalRefinementTCN(config).to(dev)
    trainer = TemporalRefinementTrainer(model, config, dev)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    patience = 15
    best_state = None
    epochs = 60

    for epoch in range(1, epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        scheduler.step(val_loss)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1

        if epoch <= 5 or epoch % 5 == 0 or improved:
            logger.info(
                f"[TEMPORAL] {profile} ep={epoch:3d} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | patience={patience_counter}/{patience}{improved}"
            )

        if patience_counter >= patience:
            logger.info(f"[TEMPORAL] {profile} early stop at epoch {epoch}")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save
    save_path = output_dir / f"temporal_refinement_{profile}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, save_path)

    result = {
        "status": "completed",
        "profile": profile,
        "timeframe": primary_tf,
        "weight_path": str(save_path),
        "n_train": train_end,
        "n_val": n - val_start,
        "epochs_trained": epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    logger.info(f"[TEMPORAL] {profile} done: val_acc={best_val_acc:.4f}, "
                f"val_loss={best_val_loss:.4f}, epochs={epoch}")
    return result


# ===================================================================
# 3. PPO ActorCritic Training
# ===================================================================
def train_ppo_exit(profile: str, device: str, output_dir: Path) -> Dict:
    """Train PPO exit optimizer for a given profile."""
    from risk_management.phase4_rl_exit.ppo_agent import PPOAgent, PPOConfig
    from risk_management.phase4_rl_exit.environment import ExitTradingEnv, ExitEnvConfig

    primary_tf = PROFILES[profile]["primary_tf"]
    logger.info(f"[PPO] Training PPO ExitOptimizer for {profile} ({primary_tf})")

    # Load data
    df = load_ohlcv(primary_tf, max_rows=100_000)
    # Ensure columns are correct
    required = ["open", "high", "low", "close"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {primary_tf} data")

    # Environment config
    env_config = ExitEnvConfig(
        max_holding_steps={"SCALP": 50, "INTRADAY": 80, "SWING": 120}[profile],
        lookback_bars=20,
        include_market_features=True,
        include_time_features=True,
        trail_stop_atr_mult=0.5,
        sl_hit_penalty=-0.5,
        premature_exit_penalty=-0.1,
        transaction_cost=0.0001,
    )

    # Create environment
    env = ExitTradingEnv(env_config)
    env.set_price_data(df)

    # PPO config
    ppo_config = PPOConfig(
        hidden_sizes=[128, 64],
        clip_epsilon=0.2,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        batch_size=64,
        n_steps=2048,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
    )

    # Create agent
    agent = PPOAgent(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        config=ppo_config,
        device=device,
    )

    # Training loop
    n_episodes = {"SCALP": 1500, "INTRADAY": 1000, "SWING": 800}[profile]
    episode_rewards = []
    episode_lengths = []
    update_metrics = []

    logger.info(f"[PPO] {profile}: obs_dim={env.observation_dim}, "
                f"action_dim={env.action_dim}, episodes={n_episodes}")

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        ep_length = 0

        while True:
            action, log_prob, value = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, value, log_prob, done)
            ep_reward += reward
            ep_length += 1
            state = next_state

            if len(agent.buffer) >= ppo_config.n_steps:
                metrics = agent.update()
                if metrics:
                    update_metrics.append(metrics)

            if done:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

        if episode % 100 == 0:
            recent_r = np.mean(episode_rewards[-100:])
            recent_l = np.mean(episode_lengths[-100:])
            logger.info(
                f"[PPO] {profile} ep={episode:4d}/{n_episodes} | "
                f"avg_reward={recent_r:.3f} | avg_length={recent_l:.1f}"
            )

    # Final update
    if agent.buffer:
        agent.update()

    # Save
    save_path = output_dir / f"ppo_exit_{profile}.pt"
    agent.save(str(save_path))

    avg_last_100 = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0
    result = {
        "status": "completed",
        "profile": profile,
        "timeframe": primary_tf,
        "weight_path": str(save_path),
        "obs_dim": env.observation_dim,
        "action_dim": env.action_dim,
        "n_episodes": n_episodes,
        "avg_reward_last_100": avg_last_100,
        "avg_length_last_100": float(np.mean(episode_lengths[-100:])) if episode_lengths else 0.0,
        "n_updates": len(update_metrics),
        "total_steps": agent._total_steps,
    }
    logger.info(f"[PPO] {profile} done: avg_reward={avg_last_100:.3f}, "
                f"total_steps={agent._total_steps}")
    return result


# ===================================================================
# Master Orchestrator
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Train remaining ML models")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "meta", "temporal", "ppo"],
                        help="Which model to train")
    parser.add_argument("--profile", type=str, default="all",
                        choices=["all", "SCALP", "INTRADAY", "SWING"],
                        help="Which profile to train")
    parser.add_argument("--output-dir", type=str,
                        default=str(ASSETS_DIR / "remaining"),
                        help="Output directory for weights")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = list(PROFILES.keys()) if args.profile == "all" else [args.profile]

    logger.info("=" * 80)
    logger.info(f"Training remaining ML models | device={args.device} | "
                f"models={args.model} | profiles={profiles}")
    logger.info("=" * 80)

    all_results = {}
    t0 = time.time()

    # ---------------------------------------------------------------
    # 1. MetaLabelingModel
    # ---------------------------------------------------------------
    if args.model in ("all", "meta"):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: MetaLabelingModel (GBM trade filter)")
        logger.info("=" * 60)
        for profile in profiles:
            try:
                result = train_meta_labeling(profile, args.device, output_dir)
                all_results[f"meta_{profile}"] = result
            except Exception as e:
                logger.error(f"[META] {profile} FAILED: {e}", exc_info=True)
                all_results[f"meta_{profile}"] = {"status": "failed", "error": str(e)}

    # ---------------------------------------------------------------
    # 2. TemporalRefinementTCN
    # ---------------------------------------------------------------
    if args.model in ("all", "temporal"):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: TemporalRefinementTCN (probability refiner)")
        logger.info("=" * 60)
        for profile in profiles:
            try:
                result = train_temporal_refinement(profile, args.device, output_dir)
                all_results[f"temporal_{profile}"] = result
            except Exception as e:
                logger.error(f"[TEMPORAL] {profile} FAILED: {e}", exc_info=True)
                all_results[f"temporal_{profile}"] = {"status": "failed", "error": str(e)}

    # ---------------------------------------------------------------
    # 3. PPO ActorCritic
    # ---------------------------------------------------------------
    if args.model in ("all", "ppo"):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: PPO ActorCritic (RL exit optimizer)")
        logger.info("=" * 60)
        for profile in profiles:
            try:
                result = train_ppo_exit(profile, args.device, output_dir)
                all_results[f"ppo_{profile}"] = result
            except Exception as e:
                logger.error(f"[PPO] {profile} FAILED: {e}", exc_info=True)
                all_results[f"ppo_{profile}"] = {"status": "failed", "error": str(e)}

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 80)
    logger.info(f"TRAINING COMPLETE | Elapsed: {elapsed:.0f}s")
    logger.info("=" * 80)

    for key, result in all_results.items():
        status = result.get("status", "unknown")
        weight = result.get("weight_path", "N/A")
        extras = ""
        if status == "completed":
            if "accuracy" in result:
                extras = f" | acc={result['accuracy']:.3f} f1={result.get('f1', 0):.3f}"
            elif "best_val_acc" in result:
                extras = f" | val_acc={result['best_val_acc']:.4f}"
            elif "avg_reward_last_100" in result:
                extras = f" | avg_reward={result['avg_reward_last_100']:.3f}"
        logger.info(f"  {key:25s} | {status:9s}{extras}")

    # Save training log
    log_path = output_dir / "remaining_training_log.json"
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "device": args.device,
        "models": all_results,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    logger.info(f"\nTraining log saved: {log_path}")


if __name__ == "__main__":
    main()
