#!/usr/bin/env python
"""
Train DecisionFusionLayer + FusionNet (2-modality: Price Action + TCN).

Both models fuse:
  - TCN sequential features (derived from OHLCV returns/indicators)
  - Price Action pattern features (from PriceActionPatternExtractor)

Usage:
    python -m research.train_fusion_models --device cuda
    python -m research.train_fusion_models --model decision_fusion
    python -m research.train_fusion_models --model fusion_net
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s",
)
logger = logging.getLogger("research.train_fusion_models")

# ---------------------------------------------------------------------------
DATA_DIR = Path(r"E:\pyProject\data\raw")
ASSETS_DIR = Path(r"E:\pyProject\pyForex-assets\models")

PROFILES = {
    "SCALP":    {"primary_tf": "M15", "horizon": 6,  "threshold": 0.15},
    "INTRADAY": {"primary_tf": "H1",  "horizon": 10, "threshold": 0.30},
    "SWING":    {"primary_tf": "H4",  "horizon": 15, "threshold": 0.50},
}

TCN_DIM = 64
PA_DIM = 44  # PriceActionPatternExtractor with extended patterns


# ===================================================================
# Data loading and feature extraction
# ===================================================================
def load_ohlcv(timeframe: str, max_rows: int = 200_000) -> pd.DataFrame:
    path = DATA_DIR / f"EURUSD_{timeframe}_latest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_csv(path)
    col_map = {c: c.lower() for c in df.columns}
    df.rename(columns=col_map, inplace=True)
    if len(df) > max_rows:
        df = df.iloc[-max_rows:].reset_index(drop=True)
    logger.info(f"Loaded {timeframe}: {len(df)} bars")
    return df


def derive_tcn_features(df: pd.DataFrame, dim: int = TCN_DIM) -> np.ndarray:
    """Derive TCN-like features from OHLCV data (returns + indicators).
    
    Returns array of shape (N, dim) where N = len(df).
    """
    c = df["close"].values.astype(np.float64)
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    o = df["open"].values.astype(np.float64)

    feats = []

    # Returns at various horizons
    for lag in [1, 2, 3, 5, 8, 10, 15, 20]:
        ret = np.zeros(len(c))
        ret[lag:] = (c[lag:] - c[:-lag]) / (c[:-lag] + 1e-10)
        feats.append(ret)

    # Log returns
    log_ret = np.zeros(len(c))
    log_ret[1:] = np.log(c[1:] / (c[:-1] + 1e-10))
    feats.append(log_ret)

    # OHLC ratios
    feats.append((c - o) / (h - l + 1e-10))  # body ratio
    feats.append((h - np.maximum(o, c)) / (h - l + 1e-10))  # upper shadow
    feats.append((np.minimum(o, c) - l) / (h - l + 1e-10))  # lower shadow

    # ATR-like
    tr = np.maximum(h - l, np.abs(h - np.roll(c, 1)))
    tr = np.maximum(tr, np.abs(l - np.roll(c, 1)))
    tr[0] = h[0] - l[0]
    atr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    feats.append(atr14 / (c + 1e-10))  # normalized ATR

    # RSI
    delta = np.diff(c, prepend=c[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss_arr = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False).mean().values
    avg_loss = pd.Series(loss_arr).ewm(span=14, adjust=False).mean().values
    rsi = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-10))
    feats.append(rsi / 100.0)

    # EMA crossovers
    ema_fast = pd.Series(c).ewm(span=12, adjust=False).mean().values
    ema_slow = pd.Series(c).ewm(span=26, adjust=False).mean().values
    feats.append((ema_fast - ema_slow) / (c + 1e-10))

    # Momentum
    for period in [5, 10, 20]:
        mom = np.zeros(len(c))
        mom[period:] = c[period:] / (c[:-period] + 1e-10) - 1.0
        feats.append(mom)

    # Volatility (rolling std of returns)
    for win in [5, 10, 20]:
        vol = pd.Series(log_ret).rolling(win, min_periods=1).std().fillna(0).values
        feats.append(vol)

    # Stack and pad/truncate to `dim`
    raw = np.column_stack(feats).astype(np.float32)  # (N, ~22)
    N, D = raw.shape
    if D >= dim:
        result = raw[:, :dim]
    else:
        # Pad with zeros
        result = np.zeros((N, dim), dtype=np.float32)
        result[:, :D] = raw
    
    # Normalize per-feature
    mean = np.nanmean(result, axis=0, keepdims=True)
    std = np.nanstd(result, axis=0, keepdims=True) + 1e-8
    result = (result - mean) / std
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


def extract_pa_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract Price Action pattern features for each bar.
    
    Returns:
        features: (N, pa_dim)
        confidences: (N,)
    """
    from models.price_action_pattern import PriceActionPatternExtractor
    extractor = PriceActionPatternExtractor(
        include_extended_patterns=True,
        include_confidence=False
    )
    
    window_size = 120  # PriceActionPatternExtractor needs >= 100 bars
    N = len(df)
    pa_dim = extractor.get_feature_dim()
    features = np.zeros((N, pa_dim), dtype=np.float32)
    confidences = np.zeros(N, dtype=np.float32)

    for i in range(window_size, N):
        try:
            window = df.iloc[i - window_size:i]
            vec = extractor.extract(window)
            features[i] = vec
            confidences[i] = float(np.mean(np.abs(vec))) if np.any(vec != 0) else 0.0
        except Exception:
            pass  # leave zeros

    return features, confidences


def generate_labels(close: np.ndarray, horizon: int, threshold_pct: float):
    """Generate 3-class direction labels + confidence.
    
    Returns:
        directions: (N,) int64 — 0=BEAR, 1=SIDEWAYS, 2=BULL
        confidence: (N,) float32 — move magnitude relative to threshold
    """
    N = len(close)
    directions = np.ones(N, dtype=np.int64)  # default SIDEWAYS
    confidence = np.zeros(N, dtype=np.float32)

    for i in range(N - horizon):
        pct = (close[i + horizon] - close[i]) / (close[i] + 1e-10) * 100.0
        if pct > threshold_pct:
            directions[i] = 2  # BULL
        elif pct < -threshold_pct:
            directions[i] = 0  # BEAR
        confidence[i] = min(abs(pct) / max(threshold_pct, 1e-6), 1.0)

    return directions, confidence


# ===================================================================
# Dataset
# ===================================================================
class FusionDataset(Dataset):
    """Dataset providing (tcn_features, pa_features, pa_confidence, tcn_stability, direction, confidence)."""

    def __init__(
        self,
        tcn_features: np.ndarray,
        pa_features: np.ndarray,
        pa_confidences: np.ndarray,
        directions: np.ndarray,
        confidences: np.ndarray,
    ):
        assert len(tcn_features) == len(pa_features) == len(directions)
        self.tcn_feat = torch.tensor(tcn_features, dtype=torch.float32)
        self.pa_feat = torch.tensor(pa_features, dtype=torch.float32)
        self.pa_conf = torch.tensor(pa_confidences, dtype=torch.float32)
        self.direction = torch.tensor(directions, dtype=torch.long)
        self.confidence = torch.tensor(confidences, dtype=torch.float32)

        # TCN stability = variance of features per sample
        self.tcn_stability = torch.var(self.tcn_feat, dim=1, keepdim=True)

    def __len__(self):
        return len(self.direction)

    def __getitem__(self, idx):
        return {
            "tcn_features": self.tcn_feat[idx],
            "pa_features": self.pa_feat[idx],
            "pa_confidence": self.pa_conf[idx],
            "tcn_stability": self.tcn_stability[idx],
            "direction": self.direction[idx],
            "confidence": self.confidence[idx],
        }


def build_datasets(
    profile: str, skip_head: int = 50
) -> Tuple[FusionDataset, FusionDataset]:
    """Build train/val datasets for a profile."""
    cfg = PROFILES[profile]
    tf = cfg["primary_tf"]
    horizon = cfg["horizon"]
    threshold = cfg["threshold"]

    df = load_ohlcv(tf)
    close = df["close"].values

    # Extract features
    logger.info(f"[{profile}] Deriving TCN features...")
    tcn_feat = derive_tcn_features(df, dim=TCN_DIM)

    logger.info(f"[{profile}] Extracting Price Action features...")
    pa_feat, pa_conf = extract_pa_features(df)

    logger.info(f"[{profile}] Generating labels (horizon={horizon}, threshold={threshold}%)...")
    directions, confidences = generate_labels(close, horizon, threshold)

    # Trim leading bars (warmup for indicators/PA)
    tcn_feat = tcn_feat[skip_head:]
    pa_feat = pa_feat[skip_head:]
    pa_conf = pa_conf[skip_head:]
    directions = directions[skip_head:]
    confidences = confidences[skip_head:]

    # Remove tail (unlabeled horizon)
    valid = len(directions) - horizon
    tcn_feat = tcn_feat[:valid]
    pa_feat = pa_feat[:valid]
    pa_conf = pa_conf[:valid]
    directions = directions[:valid]
    confidences = confidences[:valid]

    N = len(directions)
    logger.info(f"[{profile}] Total samples: {N}")
    for cls_id, cls_name in enumerate(["BEAR", "SIDEWAYS", "BULL"]):
        cnt = int(np.sum(directions == cls_id))
        logger.info(f"  {cls_name}: {cnt} ({cnt/N:.1%})")

    # Chronological 80/20 split with purge gap
    split = int(0.8 * N)
    purge = horizon + 10
    val_start = min(split + purge, N)

    train_ds = FusionDataset(
        tcn_feat[:split], pa_feat[:split], pa_conf[:split],
        directions[:split], confidences[:split]
    )
    val_ds = FusionDataset(
        tcn_feat[val_start:], pa_feat[val_start:], pa_conf[val_start:],
        directions[val_start:], confidences[val_start:]
    )
    logger.info(f"[{profile}] Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds


# ===================================================================
# Training loops
# ===================================================================
def train_decision_fusion(
    profile: str, device: str, output_dir: Path,
    epochs: int = 60, batch_size: int = 64, lr: float = 5e-4, patience: int = 12,
) -> Dict:
    """Train DecisionFusionLayer for one profile."""
    from models.decision_fusion import DecisionFusionLayer, DecisionOutput

    logger.info("=" * 60)
    logger.info(f"  TRAINING DecisionFusionLayer — {profile}")
    logger.info("=" * 60)

    train_ds, val_ds = build_datasets(profile)
    pa_dim = train_ds.pa_feat.shape[1]

    model = DecisionFusionLayer(
        price_action_dim=pa_dim,
        tcn_dim=TCN_DIM,
        hidden_dim=256,
        num_classes=3,
        use_regime_conditioning=True,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {param_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    direction_loss_fn = nn.CrossEntropyLoss()
    confidence_loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device == "cuda")

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output: DecisionOutput = model(
                price_action_features=batch["pa_features"],
                price_action_confidence=batch["pa_confidence"],
                tcn_features=batch["tcn_features"],
                tcn_stability=batch["tcn_stability"],
            )
            loss_dir = direction_loss_fn(output.direction_logits, batch["direction"])
            loss_conf = confidence_loss_fn(output.confidence.squeeze(), batch["confidence"])
            loss = loss_dir + 0.5 * loss_conf

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * batch["direction"].size(0)
            correct += (output.direction_label == batch["direction"]).sum().item()
            total += batch["direction"].size(0)

        scheduler.step()
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # --- Val ---
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(
                    price_action_features=batch["pa_features"],
                    price_action_confidence=batch["pa_confidence"],
                    tcn_features=batch["tcn_features"],
                    tcn_stability=batch["tcn_stability"],
                )
                loss_dir = direction_loss_fn(output.direction_logits, batch["direction"])
                loss_conf = confidence_loss_fn(output.confidence.squeeze(), batch["confidence"])
                loss = loss_dir + 0.5 * loss_conf
                val_loss_sum += loss.item() * batch["direction"].size(0)
                val_correct += (output.direction_label == batch["direction"]).sum().item()
                val_total += batch["direction"].size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"  Epoch {epoch+1:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    # Save
    save_dir = output_dir / "decision_fusion" / profile.lower()
    save_dir.mkdir(parents=True, exist_ok=True)
    weight_path = save_dir / "best_model.pt"
    torch.save({
        "model_state_dict": best_state or model.state_dict(),
        "profile": profile,
        "pa_dim": pa_dim,
        "tcn_dim": TCN_DIM,
        "best_val_acc": best_val_acc,
        "epochs_trained": len(history["train_loss"]),
        "history": history,
    }, weight_path)
    logger.info(f"  Saved: {weight_path} (val_acc={best_val_acc:.4f})")

    return {
        "status": "completed",
        "profile": profile,
        "best_val_acc": best_val_acc,
        "epochs_trained": len(history["train_loss"]),
        "params": param_count,
        "weight_path": str(weight_path),
    }


def train_fusion_net(
    profile: str, device: str, output_dir: Path,
    epochs: int = 60, batch_size: int = 64, lr: float = 5e-4, patience: int = 12,
) -> Dict:
    """Train FusionNet (gated attention) for one profile."""
    from models.fusion import FusionNet

    logger.info("=" * 60)
    logger.info(f"  TRAINING FusionNet — {profile}")
    logger.info("=" * 60)

    train_ds, val_ds = build_datasets(profile)
    pa_dim = train_ds.pa_feat.shape[1]

    model = FusionNet(
        seq_dim=TCN_DIM,
        price_action_dim=pa_dim,
        hidden_dim=256,
        num_classes=3,
        dropout=0.3,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {param_count:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device == "cuda")
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device == "cuda")

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            seq = batch["tcn_features"].to(device)
            pa = batch["pa_features"].to(device)
            labels = batch["direction"].to(device)

            logits = model(seq, pa)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # --- Val ---
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                seq = batch["tcn_features"].to(device)
                pa = batch["pa_features"].to(device)
                labels = batch["direction"].to(device)

                logits = model(seq, pa)
                loss = criterion(logits, labels)
                val_loss_sum += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            f"  Epoch {epoch+1:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    # Save
    save_dir = output_dir / "fusion_net" / profile.lower()
    save_dir.mkdir(parents=True, exist_ok=True)
    weight_path = save_dir / "best_model.pt"
    torch.save({
        "model_state_dict": best_state or model.state_dict(),
        "profile": profile,
        "seq_dim": TCN_DIM,
        "pa_dim": pa_dim,
        "best_val_acc": best_val_acc,
        "epochs_trained": len(history["train_loss"]),
        "history": history,
    }, weight_path)
    logger.info(f"  Saved: {weight_path} (val_acc={best_val_acc:.4f})")

    # Gate analysis
    model.load_state_dict(best_state or model.state_dict())
    model.to(device).eval()
    with torch.no_grad():
        sample_seq = train_ds.tcn_feat[:100].to(device)
        sample_pa = train_ds.pa_feat[:100].to(device)
        importance = model.get_modality_importance(sample_seq, sample_pa)
        logger.info(f"  Modality importance: {importance}")

    return {
        "status": "completed",
        "profile": profile,
        "best_val_acc": best_val_acc,
        "epochs_trained": len(history["train_loss"]),
        "params": param_count,
        "weight_path": str(weight_path),
        "modality_importance": importance,
    }


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Train fusion models (PA + TCN)")
    parser.add_argument("--model", choices=["decision_fusion", "fusion_net", "all"], default="all")
    parser.add_argument("--profile", choices=["SCALP", "INTRADAY", "SWING", "all"], default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"

    output_dir = ASSETS_DIR / "fusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = list(PROFILES.keys()) if args.profile == "all" else [args.profile]
    models = ["decision_fusion", "fusion_net"] if args.model == "all" else [args.model]

    t0 = time.time()
    all_results = {}

    for profile in profiles:
        if "decision_fusion" in models:
            try:
                result = train_decision_fusion(
                    profile, args.device, output_dir,
                    epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, patience=args.patience,
                )
                all_results[f"decision_fusion_{profile}"] = result
            except Exception as e:
                logger.error(f"[DecisionFusion] {profile} FAILED: {e}", exc_info=True)
                all_results[f"decision_fusion_{profile}"] = {"status": "failed", "error": str(e)}

        if "fusion_net" in models:
            try:
                result = train_fusion_net(
                    profile, args.device, output_dir,
                    epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, patience=args.patience,
                )
                all_results[f"fusion_net_{profile}"] = result
            except Exception as e:
                logger.error(f"[FusionNet] {profile} FAILED: {e}", exc_info=True)
                all_results[f"fusion_net_{profile}"] = {"status": "failed", "error": str(e)}

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 80)
    logger.info(f"FUSION TRAINING COMPLETE | Elapsed: {elapsed:.0f}s")
    logger.info("=" * 80)

    for key, result in all_results.items():
        status = result.get("status", "unknown")
        acc = result.get("best_val_acc", 0)
        params = result.get("params", 0)
        logger.info(f"  {key:35s} | {status:9s} | val_acc={acc:.4f} | params={params:,}")

    # Save log
    log_path = output_dir / "fusion_training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "device": args.device,
            "models": all_results,
        }, f, indent=2, default=str)
    logger.info(f"\nLog saved: {log_path}")


if __name__ == "__main__":
    main()
