# models/tcn.py
"""
Temporal Convolutional Network (TCN) for time-series feature extraction.
Drop-in replacement for LSTMModel with same interface.

TCN advantages over LSTM:
- Parallelizable (faster training/inference)
- Stable gradients (no vanishing gradient problem)
- Flexible receptive field via dilations
- Better multi-scale pattern capture

Architecture:
    Input → [TCN Blocks with Residual] → LayerNorm → Features/Classifier

References:
    - "An Empirical Evaluation of Generic Convolutional and Recurrent Networks
       for Sequence Modeling" (Bai et al., 2018)
    - "WaveNet: A Generative Model for Raw Audio" (van den Oord et al., 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal, List, Tuple
import math


class CausalConv1d(nn.Module):
    """
    Causal convolution ensuring output[t] only depends on input[<=t].
    Uses left-padding to maintain causality.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        **kwargs
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Causal padding: (kernel_size - 1) * dilation on the left only
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,  # We'll pad manually for causality
            dilation=dilation,
            **kwargs
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        # Pad only on the left (causal)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """
    Single TCN residual block with:
    - Two causal dilated convolutions
    - Weight normalization
    - Dropout
    - Residual connection (with 1x1 conv if dimensions differ)
    
    Structure:
        Input ──┬── CausalConv → Norm → ReLU → Dropout ──┐
                │                                         │
                │   CausalConv → Norm → ReLU → Dropout ──┤
                │                                         │
                └─────────── [1x1 Conv if needed] ───────┴── (+) → Output
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        
        # First causal conv
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        
        # Second causal conv
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        # Apply weight normalization for stable training
        if use_weight_norm:
            self.conv1.conv = nn.utils.parametrizations.weight_norm(self.conv1.conv)
            self.conv2.conv = nn.utils.parametrizations.weight_norm(self.conv2.conv)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection (1x1 conv if channels differ)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with Kaiming for ReLU activations."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        res = self.residual(x)
        
        return self.relu(out + res)


class TCNBackbone(nn.Module):
    """
    Stack of TCN blocks with exponentially increasing dilations.
    
    Receptive field = 1 + 2 * (kernel_size - 1) * sum(dilations)
    
    Example with kernel_size=3, dilations=[1,2,4,8,16]:
        RF = 1 + 2 * 2 * (1+2+4+8+16) = 1 + 4 * 31 = 125 timesteps
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
        dilation_base: int = 2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Build dilations: [1, 2, 4, 8, ...] or custom
        dilations = [dilation_base ** i for i in range(num_layers)]
        
        # Calculate receptive field
        self.receptive_field = 1 + 2 * (kernel_size - 1) * sum(dilations)
        
        # Build TCN blocks
        layers = []
        for i, dilation in enumerate(dilations):
            in_ch = input_dim if i == 0 else hidden_dim
            out_ch = hidden_dim
            
            layers.append(TCNBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            ))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) - same as LSTM input format
        Returns:
            (batch, seq_len, hidden_dim)
        """
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        
        out = self.network(x)  # (batch, hidden_dim, seq_len)
        
        # Back to (batch, seq_len, hidden_dim)
        return out.transpose(1, 2)


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for time-series classification.
    
    DROP-IN REPLACEMENT for LSTMModel with identical interface:
    - Same __init__ parameters
    - Same forward(x, mode='features'|'classify') signature
    - Same get_feature_dim() method
    
    Usage:
        # Replace LSTM with TCN
        # model = LSTMModel(input_dim=5, hidden_dim=64, num_classes=3)
        model = TCNModel(input_dim=5, hidden_dim=64, num_classes=3)
        
        features = model(x, mode='features')  # (batch, hidden_dim)
        logits = model(x, mode='classify')    # (batch, num_classes)
    
    Profile Presets:
        TCNModel.from_profile('SCALP')   - Fast, short receptive field
        TCNModel.from_profile('INTRADAY') - Balanced
        TCNModel.from_profile('SWING')    - Large receptive field
    """
    
    # Profile configurations
    PROFILES = {
        'SCALP': {
            'num_layers': 4,      # RF ≈ 31 with k=3
            'kernel_size': 3,
            'dilation_base': 2,
        },
        'INTRADAY': {
            'num_layers': 5,      # RF ≈ 63 with k=3
            'kernel_size': 3,
            'dilation_base': 2,
        },
        'SWING': {
            'num_layers': 7,      # RF ≈ 255 with k=3
            'kernel_size': 3,
            'dilation_base': 2,
        },
    }
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 5,
        num_classes: int = 3,
        dropout: float = 0.2,
        kernel_size: int = 3,
        dilation_base: int = 2,
        # Legacy params for LSTM compatibility (ignored)
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.feature_dim = hidden_dim  # Output dimension for fusion
        
        # TCN backbone
        self.tcn = TCNBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            dilation_base=dilation_base,
        )
        
        # Aggregation: we'll use last timestep + global average pooling
        # This captures both final state and overall sequence patterns
        self.agg_weight = nn.Parameter(torch.tensor([0.7, 0.3]))
        
        # Layer normalization for stable features
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head (only used in 'classify' mode)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        
        # Log receptive field
        print(f"TCN receptive field: {self.tcn.receptive_field} timesteps")
    
    @classmethod
    def from_profile(
        cls,
        profile: Literal['SCALP', 'INTRADAY', 'SWING'],
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> 'TCNModel':
        """
        Create TCN with profile-optimized architecture.
        
        Args:
            profile: 'SCALP', 'INTRADAY', or 'SWING'
            input_dim: Number of input features (default 5 for OHLCV)
            hidden_dim: Hidden dimension
            num_classes: Number of output classes
            dropout: Dropout rate
        
        Returns:
            Configured TCNModel instance
        """
        if profile not in cls.PROFILES:
            raise ValueError(f"Unknown profile: {profile}. Use {list(cls.PROFILES.keys())}")
        
        config = cls.PROFILES[profile]
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=config['num_layers'],
            num_classes=num_classes,
            dropout=dropout,
            kernel_size=config['kernel_size'],
            dilation_base=config['dilation_base'],
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
        # TCN forward: (batch, seq_len, input_dim) → (batch, seq_len, hidden_dim)
        tcn_out = self.tcn(x)
        
        # Aggregate sequence to single vector
        # Weighted combination of last timestep and global average
        last_step = tcn_out[:, -1, :]  # (batch, hidden_dim)
        global_avg = tcn_out.mean(dim=1)  # (batch, hidden_dim)
        
        # Learnable weighted combination
        weights = F.softmax(self.agg_weight, dim=0)
        features = weights[0] * last_step + weights[1] * global_avg
        
        # Normalize features
        features = self.layer_norm(features)
        
        if mode == 'features':
            return features
        elif mode == 'classify':
            return self.classifier(features)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'features' or 'classify'")
    
    def get_feature_dim(self) -> int:
        """Returns output feature dimension for fusion layer."""
        return self.feature_dim


class TCNWithAttention(nn.Module):
    """
    TCN with temporal attention mechanism.
    Better captures important time steps (analogous to LSTMWithAttention).
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 5,
        num_classes: int = 3,
        dropout: float = 0.2,
        kernel_size: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.feature_dim = hidden_dim
        
        # TCN backbone
        self.tcn = TCNBackbone(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        
        # Multi-head self-attention over time
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Learned query for aggregation
        self.agg_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x: torch.Tensor, mode: str = 'features') -> torch.Tensor:
        batch_size = x.size(0)
        
        # TCN forward
        tcn_out = self.tcn(x)  # (batch, seq_len, hidden_dim)
        
        # Expand query for batch
        query = self.agg_query.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        
        # Attention: query attends to TCN output
        attn_out, _ = self.attention(query, tcn_out, tcn_out)  # (batch, 1, hidden_dim)
        
        features = attn_out.squeeze(1)  # (batch, hidden_dim)
        features = self.layer_norm(features)
        
        if mode == 'features':
            return features
        return self.classifier(features)
    
    def get_feature_dim(self) -> int:
        return self.feature_dim


class MultiScaleTCN(nn.Module):
    """
    Multi-scale TCN with parallel branches at different resolutions.
    Captures both short-term and long-term patterns simultaneously.
    
    Good for when you need both scalping and swing signals.
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        branch_dim = hidden_dim // 3
        
        # Short-term branch (small receptive field)
        self.short_branch = TCNBackbone(
            input_dim=input_dim,
            hidden_dim=branch_dim,
            num_layers=3,  # RF ≈ 15
            kernel_size=3,
            dropout=dropout,
        )
        
        # Medium-term branch
        self.medium_branch = TCNBackbone(
            input_dim=input_dim,
            hidden_dim=branch_dim,
            num_layers=5,  # RF ≈ 63
            kernel_size=3,
            dropout=dropout,
        )
        
        # Long-term branch (large receptive field)
        self.long_branch = TCNBackbone(
            input_dim=input_dim,
            hidden_dim=branch_dim,
            num_layers=7,  # RF ≈ 255
            kernel_size=3,
            dropout=dropout,
        )
        
        self.feature_dim = branch_dim * 3
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(branch_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        
        # For compatibility
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor, mode: str = 'features') -> torch.Tensor:
        # Run all branches
        short_out = self.short_branch(x)[:, -1, :]   # (batch, branch_dim)
        medium_out = self.medium_branch(x)[:, -1, :]
        long_out = self.long_branch(x)[:, -1, :]
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([short_out, medium_out, long_out], dim=1)
        
        # Fuse
        features = self.fusion(multi_scale)
        features = self.layer_norm(features)
        
        if mode == 'features':
            return features
        return self.classifier(features)
    
    def get_feature_dim(self) -> int:
        return self.hidden_dim


# =============================================================================
# Factory function for easy model selection
# =============================================================================

def create_tcn_model(
    variant: Literal['standard', 'attention', 'multiscale'] = 'standard',
    profile: Optional[Literal['SCALP', 'INTRADAY', 'SWING']] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create TCN variants.
    
    Args:
        variant: 'standard', 'attention', or 'multiscale'
        profile: Optional profile preset ('SCALP', 'INTRADAY', 'SWING')
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        Configured TCN model
    
    Examples:
        # Standard TCN with SCALP profile
        model = create_tcn_model('standard', profile='SCALP')
        
        # Attention TCN with custom config
        model = create_tcn_model('attention', hidden_dim=128, num_layers=6)
        
        # Multi-scale for all timeframes
        model = create_tcn_model('multiscale')
    """
    if variant == 'standard':
        if profile:
            return TCNModel.from_profile(profile, **kwargs)
        return TCNModel(**kwargs)
    
    elif variant == 'attention':
        return TCNWithAttention(**kwargs)
    
    elif variant == 'multiscale':
        return MultiScaleTCN(**kwargs)
    
    else:
        raise ValueError(f"Unknown variant: {variant}")


# =============================================================================
# Backward compatibility alias
# =============================================================================

# This allows: from models.tcn import TCNModel as LSTMModel
# For gradual migration without changing imports everywhere
LSTMModelReplacement = TCNModel


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("TCN Model Test")
    print("=" * 60)
    
    # Test standard TCN
    model = TCNModel(input_dim=5, hidden_dim=64, num_layers=5, num_classes=3)
    x = torch.randn(8, 60, 5)  # (batch=8, seq_len=60, features=5)
    
    features = model(x, mode='features')
    logits = model(x, mode='classify')
    
    print(f"\nInput shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Feature dim: {model.get_feature_dim()}")
    
    # Test profile presets
    print("\n" + "-" * 40)
    print("Profile Presets:")
    for profile in ['SCALP', 'INTRADAY', 'SWING']:
        m = TCNModel.from_profile(profile)
        print(f"  {profile}: RF={m.tcn.receptive_field} timesteps")
    
    # Test attention variant
    print("\n" + "-" * 40)
    print("TCN with Attention:")
    model_attn = TCNWithAttention(input_dim=5, hidden_dim=64)
    feat_attn = model_attn(x, mode='features')
    print(f"  Features shape: {feat_attn.shape}")
    
    # Test multi-scale
    print("\n" + "-" * 40)
    print("Multi-Scale TCN:")
    model_ms = MultiScaleTCN(input_dim=5, hidden_dim=64)
    feat_ms = model_ms(x, mode='features')
    print(f"  Features shape: {feat_ms.shape}")
    
    # Parameter count comparison
    print("\n" + "-" * 40)
    print("Parameter Counts:")
    
    from models.lstm import LSTMModel
    lstm = LSTMModel(input_dim=5, hidden_dim=64, num_layers=2, num_classes=3)
    tcn = TCNModel(input_dim=5, hidden_dim=64, num_layers=5, num_classes=3)
    
    lstm_params = sum(p.numel() for p in lstm.parameters())
    tcn_params = sum(p.numel() for p in tcn.parameters())
    
    print(f"  LSTM: {lstm_params:,} parameters")
    print(f"  TCN:  {tcn_params:,} parameters")
    
    print("\n✅ All tests passed!")