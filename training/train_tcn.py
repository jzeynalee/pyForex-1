# training/train_tcn.py
"""
TCN model training script - drop-in replacement for train_lstm.py

Key differences from LSTM training:
- Parallelizable (faster training)
- Different learning rate dynamics
- Profile-aware architecture selection
- Optional OneCycle LR scheduler for faster convergence

Usage:
    # Basic training (default profile)
    python training/train_tcn.py --data data/raw/eurusd_latest.csv

    # Profile-specific training
    python training/train_tcn.py --data data/raw/eurusd_latest.csv --profile SCALP
    python training/train_tcn.py --data data/raw/eurusd_latest.csv --profile SWING

    # With attention mechanism
    python training/train_tcn.py --data data/raw/eurusd_latest.csv --attention

    # Multi-scale variant
    python training/train_tcn.py --data data/raw/eurusd_latest.csv --multiscale
"""

import torch
import torch.nn as nn
import logging
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Literal
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tcn import TCNModel, TCNWithAttention, MultiScaleTCN, create_tcn_model
from utils.data_loader import DataLoader as MyDataLoader, DataConfig
from utils.feature_schema import get_feature_schema_version
from utils.training_utils import copy_schema_tagged, set_global_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


class OneCycleLR:
    """
    OneCycle learning rate scheduler for faster convergence.
    Particularly effective for TCN training.
    """
    
    def __init__(
        self,
        optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        self.step_count = 0
        self.warmup_steps = int(total_steps * pct_start)
    
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup: linear increase
            progress = self.step_count / self.warmup_steps
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Annealing: cosine decay
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.final_lr + (self.max_lr - self.final_lr) * (1 + np.cos(np.pi * progress)) / 2
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


def train_tcn_model(
    data_path: str = "data/raw/eurusd_latest.csv",
    save_dir: str = "models/weights",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    seq_len: int = 60,
    hidden_dim: int = 64,
    profile: Optional[Literal['SCALP', 'INTRADAY', 'SWING']] = None,
    variant: Literal['standard', 'attention', 'multiscale'] = 'standard',
    use_onecycle: bool = True,
    device: Optional[str] = None,
    seed: int = 42,
) -> Tuple[nn.Module, Dict]:
    """
    Train TCN model with proper data handling.
    
    Args:
        data_path: Path to training data CSV
        save_dir: Directory to save model weights
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Peak learning rate (for OneCycle) or base LR
        seq_len: Input sequence length
        hidden_dim: Hidden dimension for TCN
        profile: Optional profile preset ('SCALP', 'INTRADAY', 'SWING')
        variant: Model variant ('standard', 'attention', 'multiscale')
        use_onecycle: Use OneCycle LR scheduler
        device: Training device (auto-detect if None)
        seed: Random seed
    
    Returns:
        Tuple of (trained model, training metrics)
    """
    set_global_seed(seed)

    schema_version = get_feature_schema_version()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Training TCN on {device}")
    
    # =========================================================================
    # 1. Load and prepare data
    # =========================================================================
    config = DataConfig(
        sequence_length=seq_len,
        label_strategy='ternary',
        scaler_type='minmax',
    )
    loader = MyDataLoader(config)
    
    df = loader.load_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    
    # Split WITHOUT leakage
    train_scaled, test_scaled, val_scaled = loader.split_and_scale(
        df,
        split_ratio=0.8,
        validation_ratio=0.1,
    )
    
    # Create sequences
    X_train, y_train = loader.create_sequences(train_scaled, seq_len)
    X_test, y_test = loader.create_sequences(test_scaled, seq_len)
    
    if len(X_train) == 0:
        raise ValueError("Insufficient data for training")
    
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Log class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique, counts))}")
    
    # =========================================================================
    # 2. Create data loaders
    # =========================================================================
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if device.type == 'cuda' else False,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # =========================================================================
    # 3. Initialize model
    # =========================================================================
    model_kwargs = {
        'input_dim': 5,
        'hidden_dim': hidden_dim,
        'num_classes': 3,
        'dropout': 0.2,
    }
    
    if profile and variant == 'standard':
        model = TCNModel.from_profile(profile, **model_kwargs)
        logger.info(f"Using TCN with profile: {profile}")
    else:
        model = create_tcn_model(variant, **model_kwargs)
        logger.info(f"Using TCN variant: {variant}")
    
    model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # =========================================================================
    # 4. Training setup
    # =========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
    )
    
    # Scheduler
    total_steps = epochs * len(train_loader)
    if use_onecycle:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.3,
        )
        logger.info(f"Using OneCycle LR (max_lr={learning_rate})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
        logger.info("Using Cosine Annealing LR")
    
    # Loss with class weights for imbalanced data
    class_counts = torch.bincount(torch.tensor(y_train))
    if len(class_counts) == 3 and class_counts.min() > 0:
        weights = 1.0 / class_counts.float()
        weights = weights / weights.sum() * 3  # Normalize
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        logger.info(f"Class weights: {weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # =========================================================================
    # 5. Training loop
    # =========================================================================
    best_acc = 0.0
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'lr': [],
    }
    
    for epoch in range(epochs):
        # ----- Train -----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x, mode='classify')
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # OneCycle steps per batch
            if use_onecycle:
                scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total:.3f}',
            })
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Cosine annealing steps per epoch
        if not use_onecycle:
            scheduler.step()
        
        # ----- Evaluate -----
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x, mode='classify')
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = test_correct / test_total
        
        # Current LR
        current_lr = scheduler.get_lr() if use_onecycle else optimizer.param_groups[0]['lr']
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)
        
        # Save best model
        improved = False
        if test_acc > best_acc:
            best_acc = test_acc
            improved = True
            
        if test_loss < best_loss:
            best_loss = test_loss
            improved = True
            
            save_path = Path(save_dir) / "tcn_best.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'test_acc': test_acc,
                'test_loss': test_loss,
                'profile': profile,
                'variant': variant,
                'hidden_dim': hidden_dim,
                'feature_schema_version': schema_version,
            }, save_path)
            copy_schema_tagged(save_path, schema_version)
            logger.info(f"💾 Saved best model (loss: {best_loss:.4f}, acc: {best_acc:.2%})")
        
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log progress
        logger.info(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2%} | "
            f"LR: {current_lr:.2e}"
        )
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # =========================================================================
    # 6. Save artifacts
    # =========================================================================
    # Save scaler for inference
    scaler_path = Path(save_dir) / "scaler.joblib"
    loader.save_scaler(scaler_path)
    copy_schema_tagged(scaler_path, schema_version)
    
    # Save training history
    history_path = Path(save_dir) / "tcn_training_history.pt"
    torch.save(history, history_path)
    copy_schema_tagged(history_path, schema_version)
    
    logger.info(f"✅ Training complete. Best test acc: {best_acc:.2%}, loss: {best_loss:.4f}")
    
    return model, history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int = 3,
) -> Dict:
    """
    Detailed evaluation with per-class metrics.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x, mode='classify')
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Overall accuracy
    accuracy = (all_preds == all_labels).mean()
    
    # Per-class metrics
    class_names = ['BUY', 'SELL', 'HOLD']
    per_class = {}
    
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).mean()
            class_count = mask.sum()
        else:
            class_acc = 0.0
            class_count = 0
        
        per_class[name] = {
            'accuracy': class_acc,
            'count': int(class_count),
        }
    
    return {
        'accuracy': accuracy,
        'per_class': per_class,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Train TCN model")
    
    # Data arguments
    parser.add_argument('--data', type=str, default="data/raw/eurusd_latest.csv",
                        help="Path to training data CSV")
    parser.add_argument('--save-dir', type=str, default="models/weights",
                        help="Directory to save weights")
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-3,
                        help="Learning rate (peak LR for OneCycle)")
    parser.add_argument('--seq-len', type=int, default=60)
    parser.add_argument('--hidden-dim', type=int, default=64)
    
    # Model architecture
    parser.add_argument('--profile', type=str, choices=['SCALP', 'INTRADAY', 'SWING'],
                        help="Use profile preset")
    parser.add_argument('--attention', action='store_true',
                        help="Use attention variant")
    parser.add_argument('--multiscale', action='store_true',
                        help="Use multi-scale variant")
    
    # Scheduler
    parser.add_argument('--no-onecycle', action='store_true',
                        help="Use cosine annealing instead of OneCycle")
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help="Device (cuda/cpu)")
    
    # Seed
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Determine variant
    if args.attention:
        variant = 'attention'
    elif args.multiscale:
        variant = 'multiscale'
    else:
        variant = 'standard'
    
    print("=" * 60)
    print("TCN Training")
    print("=" * 60)
    print(f"  Data: {args.data}")
    print(f"  Variant: {variant}")
    print(f"  Profile: {args.profile or 'default'}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Scheduler: {'Cosine' if args.no_onecycle else 'OneCycle'}")
    print("=" * 60)
    
    train_tcn_model(
        data_path=args.data,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        profile=args.profile,
        variant=variant,
        use_onecycle=not args.no_onecycle,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()