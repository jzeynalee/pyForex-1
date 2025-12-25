"""
Phase 1: Training Utilities for Multi-Head TCN

This module provides:
- Loss functions for each prediction head
- Combined multi-task loss
- Training loop utilities
- Metrics calculation

This training stack supports both classic prediction heads:
- direction (3-class)
- volatility (regression)
- quantiles (pinball regression)

and trade-objective probability heads:
- outcomes: [p_long, p_short] where each is P(TP hit before SL) within a time barrier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List, Callable
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class DirectionLoss(nn.Module):
    """
    Cross-entropy loss for direction prediction.
    
    Supports class weights for imbalanced data (common in forex).
    """
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.class_weights = class_weights
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(
        self,
        pred_probs: torch.Tensor,
        target_direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_probs: (batch, 3) - probabilities from direction head
            target_direction: (batch,) - class indices (0=Bear, 1=Side, 2=Bull)
        
        Returns:
            Scalar loss
        """
        # Convert probs to logits for numerical stability
        logits = torch.log(pred_probs + 1e-8)
        return self.ce_loss(logits, target_direction)


class VolatilityLoss(nn.Module):
    """
    Loss for volatility prediction.
    
    Uses a combination of:
    - MSE for accuracy
    - Relative error penalty (important for small vs large volatility)
    """
    
    def __init__(self, mse_weight: float = 0.7, relative_weight: float = 0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.relative_weight = relative_weight
    
    def forward(
        self,
        pred_volatility: torch.Tensor,
        target_volatility: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_volatility: (batch,) - predicted σ
            target_volatility: (batch,) - actual realized volatility
        
        Returns:
            Scalar loss
        """
        # MSE loss
        mse = F.mse_loss(pred_volatility, target_volatility)
        
        # Relative error (MAPE-like)
        relative_error = torch.abs(pred_volatility - target_volatility) / (target_volatility + 1e-8)
        relative_loss = relative_error.mean()
        
        return self.mse_weight * mse + self.relative_weight * relative_loss


class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    
    For each quantile τ:
    L_τ(y, ŷ) = τ * max(y - ŷ, 0) + (1-τ) * max(ŷ - y, 0)
    
    This asymmetric loss ensures the model learns the actual quantiles
    of the distribution, not just the mean.
    """
    
    def __init__(self, quantiles: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)):
        super().__init__()
        self.quantiles = quantiles
        self.register_buffer(
            'quantile_tensor',
            torch.tensor(quantiles, dtype=torch.float32)
        )
    
    def forward(
        self,
        pred_quantiles: torch.Tensor,
        target_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_quantiles: (batch, num_quantiles) - predicted quantiles
            target_values: (batch,) - actual price movements
        
        Returns:
            Scalar loss (mean pinball loss across all quantiles)
        """
        # Expand target for each quantile
        target = target_values.unsqueeze(-1)  # (batch, 1)
        
        # Calculate errors
        errors = target - pred_quantiles  # (batch, num_quantiles)
        
        # Pinball loss for each quantile
        quantiles = self.quantile_tensor.to(pred_quantiles.device)
        losses = torch.where(
            errors >= 0,
            quantiles * errors,
            (quantiles - 1) * errors
        )
        
        return losses.mean()


class OutcomeLoss(nn.Module):
    """Binary cross-entropy loss for trade outcome probabilities.

    Targets are dense labels with shape (batch, 2):
    - outcomes[:, 0] == 1 if a long entry would hit TP before SL within the horizon
    - outcomes[:, 1] == 1 if a short entry would hit TP before SL within the horizon

    Predictions are expected to be probabilities in [0, 1].
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_outcomes: torch.Tensor, target_outcomes: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred_outcomes, 1e-6, 1 - 1e-6)
        target = target_outcomes.to(dtype=pred.dtype)
        return F.binary_cross_entropy(pred, target)


class MultiTaskLoss(nn.Module):
    """
    Combined loss for training all heads simultaneously.
    
    Uses uncertainty weighting (Kendall & Gal, 2017) to automatically
    balance the different losses based on their homoscedastic uncertainty.
    """
    
    def __init__(
        self,
        direction_weight: float = 1.0,
        volatility_weight: float = 1.0,
        quantile_weight: float = 1.0,
        outcome_weight: float = 1.0,
        use_uncertainty_weighting: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        quantiles: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
    ):
        super().__init__()
        
        self.direction_loss = DirectionLoss(class_weights)
        self.volatility_loss = VolatilityLoss()
        self.quantile_loss = QuantileLoss(quantiles)
        self.outcome_loss = OutcomeLoss()
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        if use_uncertainty_weighting:
            # Learnable log variances for uncertainty weighting
            self.log_var_direction = nn.Parameter(torch.tensor(0.0))
            self.log_var_volatility = nn.Parameter(torch.tensor(0.0))
            self.log_var_quantile = nn.Parameter(torch.tensor(0.0))
            self.log_var_outcome = nn.Parameter(torch.tensor(0.0))
        else:
            # Fixed weights
            self.direction_weight = direction_weight
            self.volatility_weight = volatility_weight
            self.quantile_weight = quantile_weight
            self.outcome_weight = outcome_weight
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate combined loss.
        
        Args:
            predictions: Dict with 'direction', 'volatility', 'quantiles' and optionally 'outcomes'
            targets: Dict with 'direction', 'volatility', 'price_move' and optionally 'outcomes'
        
        Returns:
            (total_loss, loss_breakdown_dict)
        """
        # Individual losses
        loss_dir = self.direction_loss(predictions['direction'], targets['direction'])
        loss_vol = self.volatility_loss(predictions['volatility'], targets['volatility'])
        loss_quant = self.quantile_loss(predictions['quantiles'], targets['price_move'])

        has_outcomes = ('outcomes' in predictions) and ('outcomes' in targets)
        loss_out = self.outcome_loss(predictions['outcomes'], targets['outcomes']) if has_outcomes else torch.tensor(0.0, device=loss_dir.device)
        
        if self.use_uncertainty_weighting:
            # Uncertainty weighting: L = L_i / (2 * σ_i^2) + log(σ_i)
            total_loss = (
                loss_dir / (2 * torch.exp(self.log_var_direction)) + self.log_var_direction / 2 +
                loss_vol / (2 * torch.exp(self.log_var_volatility)) + self.log_var_volatility / 2 +
                loss_quant / (2 * torch.exp(self.log_var_quantile)) + self.log_var_quantile / 2
            )

            if has_outcomes:
                total_loss = total_loss + (
                    loss_out / (2 * torch.exp(self.log_var_outcome)) + self.log_var_outcome / 2
                )
            
            # Effective weights for logging
            w_dir = 1 / (2 * torch.exp(self.log_var_direction)).item()
            w_vol = 1 / (2 * torch.exp(self.log_var_volatility)).item()
            w_quant = 1 / (2 * torch.exp(self.log_var_quantile)).item()
            w_out = 1 / (2 * torch.exp(self.log_var_outcome)).item() if has_outcomes else 0.0
        else:
            total_loss = (
                self.direction_weight * loss_dir +
                self.volatility_weight * loss_vol +
                self.quantile_weight * loss_quant
            )

            if has_outcomes:
                total_loss = total_loss + (self.outcome_weight * loss_out)

            w_dir, w_vol, w_quant, w_out = self.direction_weight, self.volatility_weight, self.quantile_weight, (self.outcome_weight if has_outcomes else 0.0)
        
        breakdown = {
            'loss_total': total_loss.item(),
            'loss_direction': loss_dir.item(),
            'loss_volatility': loss_vol.item(),
            'loss_quantile': loss_quant.item(),
            'loss_outcome': loss_out.item() if has_outcomes else 0.0,
            'weight_direction': w_dir,
            'weight_volatility': w_vol,
            'weight_quantile': w_quant,
            'weight_outcome': w_out
        }
        
        return total_loss, breakdown


@dataclass
class TrainingConfig:
    """Configuration for training the multi-head TCN."""
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    patience: int = 10           # Early stopping patience
    min_delta: float = 1e-4      # Minimum improvement for early stopping
    grad_clip: float = 1.0       # Gradient clipping
    warmup_epochs: int = 5       # Learning rate warmup
    
    # Loss weights (if not using uncertainty weighting)
    direction_weight: float = 1.0
    volatility_weight: float = 1.0
    quantile_weight: float = 1.0
    outcome_weight: float = 1.0
    use_uncertainty_weighting: bool = True


class RiskDataset(Dataset):
    """
    Dataset for training the multi-head TCN.
    
    Expects preprocessed data with:
    - features: Technical indicators and OHLCV data
    - direction_labels: 0 (Bear), 1 (Sideways), 2 (Bull)
    - volatility_labels: Realized volatility over horizon
    - price_move_labels: Actual price movement for quantile regression
    - outcome_labels: Optional dense labels (n_samples, 2) for [y_long, y_short]
    """
    
    def __init__(
        self,
        features: np.ndarray,
        direction_labels: np.ndarray,
        volatility_labels: np.ndarray,
        price_move_labels: np.ndarray,
        sequence_length: int = 60,
        vision_features: Optional[np.ndarray] = None,
        outcome_labels: Optional[np.ndarray] = None
    ):
        """
        Args:
            features: (num_samples, num_features) raw feature matrix
            direction_labels: (num_samples,) direction class labels
            volatility_labels: (num_samples,) realized volatility
            price_move_labels: (num_samples,) price movement
            sequence_length: Number of timesteps per sequence
            vision_features: Optional (num_samples, vision_dim) vision embeddings
            outcome_labels: Optional (num_samples, 2) dense binary labels [y_long, y_short]
        """
        self.sequence_length = sequence_length
        self.vision_features = vision_features
        
        # Create sequences
        self.sequences = []
        self.targets = []
        self.vision_seqs = []
        
        for i in range(len(features) - sequence_length):
            seq = features[i:i + sequence_length]
            target_idx = i + sequence_length - 1
            
            self.sequences.append(seq)
            self.targets.append({
                'direction': direction_labels[target_idx],
                'volatility': volatility_labels[target_idx],
                'price_move': price_move_labels[target_idx]
            })

            if outcome_labels is not None:
                self.targets[-1]['outcomes'] = outcome_labels[target_idx]
            
            if vision_features is not None:
                self.vision_seqs.append(vision_features[target_idx])
        
        self.sequences = np.array(self.sequences)
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        
        targets = {
            'direction': torch.tensor(self.targets[idx]['direction'], dtype=torch.long),
            'volatility': torch.tensor(self.targets[idx]['volatility'], dtype=torch.float32),
            'price_move': torch.tensor(self.targets[idx]['price_move'], dtype=torch.float32)
        }

        if 'outcomes' in self.targets[idx]:
            targets['outcomes'] = torch.tensor(self.targets[idx]['outcomes'], dtype=torch.float32)
        
        vision = None
        if self.vision_features is not None:
            vision = torch.tensor(self.vision_seqs[idx], dtype=torch.float32)
        
        return seq, targets, vision


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def compute_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute evaluation metrics for all heads.

    If outcome targets are present, includes outcome metrics for
    p_long/p_short training.
    
    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}
    
    # Direction metrics
    pred_dir = predictions['direction'].argmax(dim=-1)
    true_dir = targets['direction']
    metrics['direction_accuracy'] = (pred_dir == true_dir).float().mean().item()
    
    # Per-class accuracy
    for cls in range(3):
        mask = true_dir == cls
        if mask.sum() > 0:
            cls_acc = (pred_dir[mask] == cls).float().mean().item()
            cls_name = ['bear', 'sideways', 'bull'][cls]
            metrics[f'direction_accuracy_{cls_name}'] = cls_acc
    
    # Volatility metrics
    pred_vol = predictions['volatility']
    true_vol = targets['volatility']
    metrics['volatility_mae'] = (pred_vol - true_vol).abs().mean().item()
    metrics['volatility_mape'] = ((pred_vol - true_vol).abs() / (true_vol + 1e-8)).mean().item()
    
    # Quantile metrics
    pred_quant = predictions['quantiles']
    true_move = targets['price_move']
    
    # Coverage (% of true values below each quantile)
    quantile_values = [0.05, 0.25, 0.50, 0.75, 0.95]
    for i, q in enumerate(quantile_values):
        coverage = (true_move < pred_quant[:, i]).float().mean().item()
        metrics[f'quantile_coverage_q{int(q*100)}'] = coverage
    
    # Interval width (Q95 - Q5)
    metrics['prediction_interval_width'] = (pred_quant[:, -1] - pred_quant[:, 0]).mean().item()

    # Outcome metrics (optional)
    if 'outcomes' in predictions and 'outcomes' in targets:
        pred_out = predictions['outcomes']
        true_out = targets['outcomes']
        pred_out = torch.clamp(pred_out, 1e-6, 1 - 1e-6)

        metrics['outcome_bce'] = F.binary_cross_entropy(pred_out, true_out).item()
        pred_bin = (pred_out >= 0.5).to(dtype=true_out.dtype)
        acc_long = (pred_bin[:, 0] == true_out[:, 0]).float().mean().item()
        acc_short = (pred_bin[:, 1] == true_out[:, 1]).float().mean().item()
        metrics['outcome_accuracy_long'] = acc_long
        metrics['outcome_accuracy_short'] = acc_short
        metrics['outcome_accuracy_mean'] = (acc_long + acc_short) / 2
    
    return metrics


class MultiHeadTCNTrainer:
    """
    Trainer for the Multi-Head TCN model.
    
    Handles:
    - Training loop with validation
    - Learning rate scheduling
    - Early stopping
    - Checkpointing
    - Metrics logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = MultiTaskLoss(
            direction_weight=config.direction_weight,
            volatility_weight=config.volatility_weight,
            quantile_weight=config.quantile_weight,
            outcome_weight=config.outcome_weight,
            use_uncertainty_weighting=config.use_uncertainty_weighting,
            class_weights=class_weights
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(self.criterion.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler (cosine with warmup)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs - config.warmup_epochs
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
        
        self.best_model_state = None
        self.best_val_loss = float('inf')
    
    def _warmup_lr(self, epoch: int) -> float:
        """Linear warmup learning rate."""
        if epoch < self.config.warmup_epochs:
            return self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
        return self.config.learning_rate
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = {'direction': [], 'volatility': [], 'quantiles': []}
        all_targets = {'direction': [], 'volatility': [], 'price_move': []}
        
        for batch_idx, (sequences, targets, vision) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            if vision is not None:
                vision = vision.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(sequences, vision, mode='all')

            if 'outcomes' in predictions and 'outcomes' in targets:
                all_preds.setdefault('outcomes', [])
                all_targets.setdefault('outcomes', [])
            
            # Compute loss
            loss, breakdown = self.criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            for key in all_preds:
                all_preds[key].append(predictions[key].detach())
            for key in all_targets:
                all_targets[key].append(targets[key].detach())
        
        # Aggregate predictions
        all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
        all_targets = {k: torch.cat(v, dim=0) for k, v in all_targets.items()}
        
        # Compute metrics
        metrics = compute_metrics(all_preds, all_targets)
        
        return total_loss / len(dataloader), metrics
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = {'direction': [], 'volatility': [], 'quantiles': []}
        all_targets = {'direction': [], 'volatility': [], 'price_move': []}
        
        for sequences, targets, vision in dataloader:
            sequences = sequences.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            if vision is not None:
                vision = vision.to(self.device)
            
            predictions = self.model(sequences, vision, mode='all')

            if 'outcomes' in predictions and 'outcomes' in targets:
                all_preds.setdefault('outcomes', [])
                all_targets.setdefault('outcomes', [])
            loss, _ = self.criterion(predictions, targets)
            
            total_loss += loss.item()
            
            for key in all_preds:
                all_preds[key].append(predictions[key])
            for key in all_targets:
                all_targets[key].append(targets[key])
        
        all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}
        all_targets = {k: torch.cat(v, dim=0) for k, v in all_targets.items()}
        
        metrics = compute_metrics(all_preds, all_targets)
        
        return total_loss / len(dataloader), metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            callbacks: Optional list of callback functions
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            # Warmup learning rate
            if epoch < self.config.warmup_epochs:
                lr = self._warmup_lr(epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update learning rate
            if epoch >= self.config.warmup_epochs:
                self.scheduler.step()
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Dir Acc: {val_metrics['direction_accuracy']:.3f} | "
                f"Vol MAE: {val_metrics['volatility_mae']:.4f}" +
                (f" | Out Acc: {val_metrics.get('outcome_accuracy_mean', 0.0):.3f}" if 'outcome_accuracy_mean' in val_metrics else "")
            )
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['metrics'].append(val_metrics)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics': val_metrics
                }
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, train_loss, val_loss, val_metrics)
            
            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
            logger.info(f"Loaded best model from epoch {self.best_model_state['epoch'] + 1}")
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Checkpoint loaded from {path}")
