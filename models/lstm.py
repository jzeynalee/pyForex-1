# models/lstm.py
"""
LSTM model for time-series feature extraction.
Can output either features (for fusion) or direct predictions.
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM with configurable output mode.
    
    Modes:
        - 'features': Returns hidden state for fusion (default)
        - 'classify': Returns class logits directly
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Feature dimension after LSTM
        self.feature_dim = hidden_dim * self.num_directions
        
        # Layer normalization for stable features
        self.layer_norm = nn.LayerNorm(self.feature_dim)
        
        # Classification head (only used in 'classify' mode)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        mode: str = 'features'
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mode: 'features' returns normalized hidden state,
                  'classify' returns class logits
        
        Returns:
            features: (batch, feature_dim) if mode='features'
            logits: (batch, num_classes) if mode='classify'
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            h_forward = h_n[-2, :, :]  # Last layer forward
            h_backward = h_n[-1, :, :]  # Last layer backward
            features = torch.cat([h_forward, h_backward], dim=1)
        else:
            features = h_n[-1, :, :]  # (batch, hidden_dim)
        
        # Normalize features
        features = self.layer_norm(features)
        
        if mode == 'features':
            return features
        elif mode == 'classify':
            return self.classifier(features)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def get_feature_dim(self) -> int:
        """Returns output feature dimension for fusion layer."""
        return self.feature_dim


class LSTMWithAttention(nn.Module):
    """
    LSTM with temporal attention mechanism.
    Better captures important time steps in the sequence.
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.feature_dim = hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor, mode: str = 'features') -> torch.Tensor:
        # LSTM output: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        
        # Attention weights: (batch, seq_len, 1)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum: (batch, hidden_dim)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        features = self.layer_norm(context)
        
        if mode == 'features':
            return features
        return self.classifier(features)
    
    def get_feature_dim(self) -> int:
        return self.feature_dim
