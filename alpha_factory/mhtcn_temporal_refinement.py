# alpha_factory/mhtcn_temporal_refinement.py
"""
MH-TCN Temporal Refinement Module
=================================

This module extends MH-TCN to accept probability sequences from the
Probabilistic Alpha Factory and refine them using temporal patterns.

Architecture:
    Probability Sequence (T x 4) → TCN Backbone → Refined Probabilities
    
    Input: [P(bull), P(bear), P(neutral), stability] over T timesteps
    Output: Refined [P(bull), P(bear), P(neutral)] + confidence adjustment

This enables the MH-TCN to learn temporal patterns in the probability
evolution, such as:
- Regime persistence (trends in probabilities)
- Regime transitions (sudden shifts)
- Stability patterns (volatility clustering in confidence)

Training:
    - Supervised on labeled regime outcomes
    - Self-supervised via contrastive learning on sequences
    - Online adaptation via walk-forward updates
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TemporalRefinementConfig:
    """Configuration for temporal refinement model."""
    
    # Input
    sequence_length: int = 20  # Number of timesteps
    input_channels: int = 4   # [P(bull), P(bear), P(neutral), stability]
    
    # Model architecture
    hidden_channels: int = 32
    num_layers: int = 3
    kernel_size: int = 3
    dropout: float = 0.2
    
    # Output
    num_regimes: int = 3  # Bull, Bear, Neutral
    output_confidence_adjustment: bool = True
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Paths
    weights_path: Optional[str] = None


# =============================================================================
# Temporal Convolutional Block
# =============================================================================

class CausalConv1d(nn.Module):
    """Causal 1D convolution with proper padding."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """Residual temporal block with causal convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out + residual


# =============================================================================
# Temporal Refinement Model
# =============================================================================

class TemporalRefinementTCN(nn.Module):
    """
    TCN model for refining probability sequences.
    
    Takes a sequence of probability vectors and outputs refined probabilities
    that account for temporal patterns.
    """
    
    def __init__(self, config: TemporalRefinementConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_channels, config.hidden_channels)
        
        # TCN layers with exponentially increasing dilation
        layers = []
        for i in range(config.num_layers):
            dilation = 2 ** i
            in_ch = config.hidden_channels
            out_ch = config.hidden_channels
            layers.append(TemporalBlock(in_ch, out_ch, config.kernel_size, dilation, config.dropout))
        
        self.tcn = nn.Sequential(*layers)
        
        # Output heads
        self.regime_head = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels // 2, config.num_regimes)
        )
        
        if config.output_confidence_adjustment:
            self.confidence_head = nn.Sequential(
                nn.Linear(config.hidden_channels, config.hidden_channels // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_channels // 2, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )
        else:
            self.confidence_head = None
        
        # Attention for sequence aggregation
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_channels, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_channels)
            return_features: Whether to return intermediate features
        
        Returns:
            Dictionary with:
            - 'regime_probs': Refined regime probabilities (batch, 3)
            - 'confidence_adj': Confidence adjustment factor (batch, 1)
            - 'features': TCN features (optional)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, hidden)
        
        # Transpose for TCN: (batch, hidden, seq_len)
        x = x.transpose(1, 2)
        
        # Apply TCN
        features = self.tcn(x)  # (batch, hidden, seq_len)
        
        # Transpose back: (batch, seq_len, hidden)
        features = features.transpose(1, 2)
        
        # Attention-weighted aggregation
        attn_weights = self.attention(features)  # (batch, seq_len, 1)
        aggregated = (features * attn_weights).sum(dim=1)  # (batch, hidden)
        
        # Regime probabilities
        regime_logits = self.regime_head(aggregated)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        outputs = {'regime_probs': regime_probs}
        
        # Confidence adjustment
        if self.confidence_head is not None:
            confidence_adj = self.confidence_head(aggregated)
            outputs['confidence_adj'] = confidence_adj
        
        if return_features:
            outputs['features'] = aggregated
        
        return outputs
    
    def predict(self, prob_sequence: np.ndarray) -> Dict[str, float]:
        """
        Predict refined probabilities from a probability sequence.
        
        Args:
            prob_sequence: Array of shape (seq_len, 4) with
                          [P(bull), P(bear), P(neutral), stability]
        
        Returns:
            Dictionary with refined probabilities
        """
        self.eval()
        
        # Ensure correct shape
        if prob_sequence.ndim == 2:
            prob_sequence = prob_sequence[np.newaxis, ...]  # Add batch dim
        
        # Pad if needed
        seq_len = prob_sequence.shape[1]
        if seq_len < self.config.sequence_length:
            pad_len = self.config.sequence_length - seq_len
            pad = np.tile(prob_sequence[:, 0:1, :], (1, pad_len, 1))
            prob_sequence = np.concatenate([pad, prob_sequence], axis=1)
        elif seq_len > self.config.sequence_length:
            prob_sequence = prob_sequence[:, -self.config.sequence_length:, :]
        
        # Convert to tensor
        x = torch.tensor(prob_sequence, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.forward(x)
        
        regime_probs = outputs['regime_probs'].cpu().numpy()[0]
        
        result = {
            'bull': float(regime_probs[0]),
            'bear': float(regime_probs[1]),
            'neutral': float(regime_probs[2])
        }
        
        if 'confidence_adj' in outputs:
            result['confidence_adjustment'] = float(outputs['confidence_adj'].cpu().numpy()[0, 0])
        
        return result


# =============================================================================
# Temporal Refinement Provider
# =============================================================================

class TemporalRefinementProvider:
    """
    Provider class for temporal refinement of probability sequences.
    
    Manages model loading, inference, and optional online updates.
    """
    
    def __init__(
        self,
        config: Optional[TemporalRefinementConfig] = None,
        weights_path: Optional[str] = None,
        device: str = 'auto'
    ):
        self.config = config or TemporalRefinementConfig()
        self.device = self._resolve_device(device)
        
        # Initialize model
        self.model = TemporalRefinementTCN(self.config).to(self.device)
        
        # Load weights if available
        if weights_path:
            self._load_weights(weights_path)
        elif self.config.weights_path:
            self._load_weights(self.config.weights_path)
        
        logger.info(f"TemporalRefinementProvider initialized on {self.device}")
    
    def _resolve_device(self, device_str: str) -> torch.device:
        """Resolve device string."""
        if device_str == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device_str)
    
    def _load_weights(self, path: str):
        """Load model weights."""
        path = Path(path)
        if path.exists():
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded temporal refinement weights from {path}")
            except Exception as e:
                logger.warning(f"Could not load weights: {e}")
    
    def refine(self, prob_sequence: np.ndarray) -> Dict[str, float]:
        """
        Refine probability sequence.
        
        Args:
            prob_sequence: Array of shape (seq_len, 4)
        
        Returns:
            Refined probabilities
        """
        return self.model.predict(prob_sequence)
    
    def save_weights(self, path: str):
        """Save model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Saved temporal refinement weights to {path}")


# =============================================================================
# Training Utilities
# =============================================================================

class TemporalRefinementDataset(torch.utils.data.Dataset):
    """Dataset for training temporal refinement model."""
    
    def __init__(
        self,
        prob_sequences: np.ndarray,
        regime_labels: np.ndarray,
        sequence_length: int = 20
    ):
        """
        Args:
            prob_sequences: Array of shape (N, T, 4) with probability sequences
            regime_labels: Array of shape (N,) with regime labels (0=bear, 1=neutral, 2=bull)
            sequence_length: Target sequence length
        """
        self.sequences = prob_sequences
        self.labels = regime_labels
        self.sequence_length = sequence_length
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Pad/truncate to sequence_length
        if len(seq) < self.sequence_length:
            pad_len = self.sequence_length - len(seq)
            pad = np.tile(seq[0:1], (pad_len, 1))
            seq = np.concatenate([pad, seq], axis=0)
        elif len(seq) > self.sequence_length:
            seq = seq[-self.sequence_length:]
        
        return (
            torch.tensor(seq, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


class TemporalRefinementTrainer:
    """Trainer for temporal refinement model."""
    
    def __init__(
        self,
        model: TemporalRefinementTCN,
        config: TemporalRefinementConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (sequences, labels) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(sequences)
            regime_probs = outputs['regime_probs']
            
            loss = self.criterion(regime_probs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Tuple[float, float]:
        """Evaluate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(sequences)
                regime_probs = outputs['regime_probs']
                
                loss = self.criterion(regime_probs, labels)
                total_loss += loss.item()
                
                preds = regime_probs.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        
        return avg_loss, accuracy


# =============================================================================
# Integration with Probabilistic Alpha Factory
# =============================================================================

def create_temporal_refinement_provider(
    weights_dir: Optional[str] = None,
    sequence_length: int = 20
) -> TemporalRefinementProvider:
    """
    Factory function to create temporal refinement provider.
    
    Args:
        weights_dir: Directory containing model weights
        sequence_length: Sequence length for probability history
    
    Returns:
        Configured TemporalRefinementProvider
    """
    config = TemporalRefinementConfig(sequence_length=sequence_length)
    
    weights_path = None
    if weights_dir:
        weights_path = Path(weights_dir) / "temporal_refinement.pth"
        if not weights_path.exists():
            weights_path = None
    
    return TemporalRefinementProvider(
        config=config,
        weights_path=str(weights_path) if weights_path else None
    )


def generate_training_data_from_backtest(
    backtest_results: pd.DataFrame,
    prob_factory: Any,  # ProbabilisticAlphaFactory
    sequence_length: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data for temporal refinement from backtest results.
    
    Args:
        backtest_results: DataFrame with backtest data including regime labels
        prob_factory: ProbabilisticAlphaFactory instance
        sequence_length: Sequence length for training
    
    Returns:
        (prob_sequences, regime_labels) arrays
    """
    sequences = []
    labels = []
    
    # Map regime strings to labels
    regime_map = {'bear': 0, 'neutral': 1, 'bull': 2}
    
    # Generate probability sequences
    prob_history = []
    
    for idx in range(len(backtest_results)):
        row = backtest_results.iloc[idx]
        
        # Get probability vector (would come from prob_factory in real usage)
        prob_vec = [
            row.get('p_bull', 0.33),
            row.get('p_bear', 0.33),
            row.get('p_neutral', 0.34),
            row.get('stability', 0.5)
        ]
        prob_history.append(prob_vec)
        
        # Once we have enough history, create training sample
        if len(prob_history) >= sequence_length:
            seq = np.array(prob_history[-sequence_length:])
            sequences.append(seq)
            
            # Get label (future regime)
            regime_str = row.get('regime', 'neutral')
            label = regime_map.get(regime_str, 1)
            labels.append(label)
    
    return np.array(sequences), np.array(labels)
