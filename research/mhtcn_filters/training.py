"""
MH-TCN Training Protocol — leak-free labels, walk-forward.

Label definitions (NO realized PnL):
    1. Signal persistence: Did the alpha signal direction persist for N bars?
       label = 1 if price moved in signal direction by at least threshold
       label = 0 otherwise
    2. Regime validity: Was the regime classification correct over next N bars?

Walk-forward protocol:
    - Split data chronologically: train (70%) / val (20%) / test (10%)
    - No shuffling — strict temporal ordering
    - Retrain on expanding window or sliding window

Horizon alignment:
    - Label horizon must match alpha horizon (e.g. if alpha targets 20-bar
      moves, labels must measure 20-bar outcomes)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for MH-TCN filter training."""
    # Data
    sequence_length: int = 60       # input window length
    label_horizon: int = 20         # bars forward for label computation
    persistence_threshold: float = 0.0005  # min price move for label=1

    # Walk-forward
    train_ratio: float = 0.70
    val_ratio: float = 0.20
    # test_ratio = 1 - train - val

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 50
    patience: int = 8
    grad_clip: float = 1.0

    # Negative control
    shuffle_labels: bool = False    # set True for shuffled-label control


def compute_persistence_labels(
    prices: np.ndarray,
    directions: np.ndarray,
    horizon: int = 20,
    threshold: float = 0.0005,
) -> np.ndarray:
    """Compute leak-free signal persistence labels.

    Args:
        prices: Close prices array (n_bars,).
        directions: Signal direction per bar (+1=long, -1=short, 0=no signal).
        horizon: Bars forward to check persistence.
        threshold: Minimum |price_change/price| for label=1.

    Returns:
        labels: Binary array (n_bars,). -1 = unlabeled (no signal or
                insufficient forward data).
    """
    n = len(prices)
    labels = np.full(n, -1, dtype=np.int32)

    for i in range(n - horizon):
        if directions[i] == 0:
            continue
        future_price = prices[i + horizon]
        current_price = prices[i]
        if current_price < 1e-10:
            continue

        pct_change = (future_price - current_price) / current_price

        if directions[i] > 0:  # LONG signal
            labels[i] = 1 if pct_change > threshold else 0
        else:  # SHORT signal
            labels[i] = 1 if pct_change < -threshold else 0

    return labels


class MHTCNDataset(Dataset):
    """Dataset for training MH-TCN filters.

    Builds (sequence, label) pairs from feature matrices and persistence labels.
    Only includes samples where label != -1 (i.e., valid signal bars with
    sufficient forward data).
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        seq_len: int = 60,
        shuffle_labels: bool = False,
    ):
        self.seq_len = seq_len
        self.sequences = []
        self.targets = []

        # Find valid indices
        valid_indices = np.where(labels >= 0)[0]
        valid_indices = valid_indices[valid_indices >= seq_len]

        for idx in valid_indices:
            seq = features[idx - seq_len:idx]
            self.sequences.append(seq)
            self.targets.append(labels[idx])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

        # Negative control: shuffle targets
        if shuffle_labels and len(self.targets) > 0:
            np.random.shuffle(self.targets)
            logger.warning("NEGATIVE CONTROL: Labels shuffled for training!")

        logger.info(
            f"MHTCNDataset: {len(self)} samples, "
            f"pos_rate={self.targets.mean():.3f}" if len(self.targets) > 0 else "empty"
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


class WalkForwardTrainer:
    """Walk-forward trainer for MH-TCN filters.

    Splits data chronologically and trains with no future leakage.
    """

    def __init__(self, config: TrainingConfig, device: str = "cpu"):
        self.config = config
        self.device = device

    def prepare_splits(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[MHTCNDataset, MHTCNDataset, MHTCNDataset]:
        """Split data chronologically into train/val/test."""
        n = len(features)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        train_ds = MHTCNDataset(
            features[:train_end], labels[:train_end],
            seq_len=self.config.sequence_length,
            shuffle_labels=self.config.shuffle_labels,
        )
        val_ds = MHTCNDataset(
            features[:val_end], labels[:val_end],  # expanding window
            seq_len=self.config.sequence_length,
        )
        test_ds = MHTCNDataset(
            features[val_end:], labels[val_end:],
            seq_len=self.config.sequence_length,
        )

        logger.info(
            f"Splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )
        return train_ds, val_ds, test_ds

    def train(
        self,
        model: nn.Module,
        train_ds: MHTCNDataset,
        val_ds: MHTCNDataset,
    ) -> Dict[str, List[float]]:
        """Train the model with early stopping.

        Returns training history.
        """
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.num_epochs
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.config.batch_size, shuffle=False
        )

        history = {"train_loss": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.config.num_epochs):
            # Train
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                out = model(x_batch)

                # Handle dict output (ProbabilisticTCN) vs scalar (ResearchTCN)
                if isinstance(out, dict):
                    pred = out["g_factor"]
                else:
                    pred = out

                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(train_loss)

            # Validate
            val_loss, val_acc = self._validate(model, val_loader, criterion)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} — "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"val_acc={val_acc:.3f} patience={patience_counter}/{self.config.patience}"
                )

            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        return history

    def _validate(self, model, loader, criterion):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                out = model(x_batch)
                pred = out["g_factor"] if isinstance(out, dict) else out
                loss = criterion(pred, y_batch)
                total_loss += loss.item()

                preds_binary = (pred > 0.5).float()
                correct += (preds_binary == y_batch).sum().item()
                total += len(y_batch)

        n = max(len(loader), 1)
        acc = correct / max(total, 1)
        return total_loss / n, acc

    def evaluate_test(
        self,
        model: nn.Module,
        test_ds: MHTCNDataset,
    ) -> Dict[str, float]:
        """Evaluate on held-out test set."""
        loader = DataLoader(test_ds, batch_size=self.config.batch_size, shuffle=False)
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                out = model(x_batch)
                pred = out["g_factor"] if isinstance(out, dict) else out
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y_batch.numpy())

        if not all_preds:
            return {"test_acc": 0, "test_brier": 1.0, "n_test": 0}

        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)

        acc = float(np.mean((preds > 0.5).astype(float) == labels))
        brier = float(np.mean((preds - labels) ** 2))

        return {
            "test_acc": acc,
            "test_brier": brier,
            "n_test": len(labels),
            "test_pos_rate": float(labels.mean()),
            "test_pred_mean": float(preds.mean()),
        }

    def save_model(self, model: nn.Module, path: str):
        """Save model weights."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        logger.info(f"Model saved to {path}")
