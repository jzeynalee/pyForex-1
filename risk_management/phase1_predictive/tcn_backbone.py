"""
Phase 1: Multi-Head TCN Backbone for Risk Management

This module implements a Temporal Convolutional Network with multiple prediction heads:
- Direction Head: P(Bear), P(Sideways), P(Bull)
- Volatility Head: Predicted volatility (σ) for next N candles
- Quantile Head: Price movement distribution [Q5, Q25, Q50, Q75, Q95]
- Outcome Head: TP-before-SL probabilities [p_long, p_short]

The backbone extracts temporal features that feed all downstream risk calculations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import numpy as np


class TradingProfile(Enum):
    """Trading profile determines TCN receptive field and prediction horizon."""
    SCALP = "SCALP"         # M5-M15, short-term
    INTRADAY = "INTRADAY"   # M30-H1, medium-term
    SWING = "SWING"         # H4-D1, longer-term


@dataclass
class TCNConfig:
    """Configuration for TCN backbone."""
    input_channels: int = 64          # Number of input features
    hidden_channels: int = 128        # Hidden layer size
    num_levels: int = 4               # Number of TCN blocks
    kernel_size: int = 3              # Convolution kernel size
    dropout: float = 0.2              # Dropout rate
    
    # Head-specific configs
    num_direction_classes: int = 3    # Bear, Sideways, Bull
    num_quantiles: int = 5            # Q5, Q25, Q50, Q75, Q95
    quantile_values: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
    
    # Profile-specific dilations
    profile: TradingProfile = TradingProfile.INTRADAY
    
    @property
    def dilations(self) -> List[int]:
        """Get dilation factors based on trading profile."""
        if self.profile == TradingProfile.SCALP:
            return [1, 2, 4, 8]              # ~15 bars receptive field
        elif self.profile == TradingProfile.INTRADAY:
            return [1, 2, 4, 8, 16]          # ~31 bars receptive field
        else:  # SWING
            return [1, 2, 4, 8, 16, 32, 64]  # ~127 bars receptive field
    
    @property
    def receptive_field(self) -> int:
        """Calculate the receptive field size."""
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)


class CausalConv1d(nn.Module):
    """
    Causal convolution that ensures no future information leakage.
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
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            **kwargs
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, channels, seq_len) - same length, causally convolved
        """
        out = self.conv(x)
        # Remove future padding to maintain causality
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    Single TCN block with residual connection.
    
    Structure:
        x -> CausalConv -> BatchNorm -> ReLU -> Dropout ->
             CausalConv -> BatchNorm -> ReLU -> Dropout -> + -> out
        |__________________________________________________|
                        (residual connection)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.conv1 = CausalConv1d(
            in_channels, out_channels, kernel_size, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = CausalConv1d(
            out_channels, out_channels, kernel_size, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (1x1 conv if channel mismatch)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, out_channels, seq_len)
        """
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        return self.relu(out + residual)


class TCNBackbone(nn.Module):
    """
    TCN feature extractor backbone.
    
    Extracts temporal features from input sequences that can be used
    by multiple prediction heads.
    """
    
    def __init__(self, config: TCNConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_channels, config.hidden_channels)
        
        # TCN blocks with increasing dilation
        self.tcn_blocks = nn.ModuleList()
        dilations = config.dilations
        
        for i, dilation in enumerate(dilations):
            in_ch = config.hidden_channels
            out_ch = config.hidden_channels
            
            self.tcn_blocks.append(
                TCNBlock(in_ch, out_ch, config.kernel_size, dilation, config.dropout)
            )
        
        self.output_dim = config.hidden_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract temporal features.
        
        Args:
            x: (batch, seq_len, features) or (batch, features, seq_len)
        Returns:
            (batch, hidden_channels, seq_len) - temporal feature maps
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Expect (batch, seq_len, features), convert to (batch, features, seq_len)
        if x.size(-1) == self.config.input_channels:
            x = x.transpose(1, 2)
        
        # Project input
        # (batch, features, seq_len) -> (batch, seq_len, features) for linear
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x.transpose(1, 2)  # Back to (batch, hidden, seq_len)
        
        # Apply TCN blocks
        for block in self.tcn_blocks:
            x = block(x)
        
        return x


class DirectionHead(nn.Module):
    """
    Predicts market direction probabilities: P(Bear), P(Sideways), P(Bull).
    
    Uses global average pooling + MLP for classification.
    """
    
    def __init__(self, input_dim: int, num_classes: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, channels, seq_len) from backbone
        Returns:
            (batch, 3) - probabilities [P(Bear), P(Sideways), P(Bull)]
        """
        # Global average pooling over time
        pooled = features.mean(dim=-1)  # (batch, channels)
        logits = self.classifier(pooled)
        return logits  # Raw logits — CrossEntropyLoss applies log_softmax internally


class VolatilityHead(nn.Module):
    """
    Predicts future volatility (σ) for risk calculations.
    
    Output is always positive (uses softplus activation).
    Can optionally predict volatility at multiple horizons.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_horizons: int = 1
    ):
        super().__init__()
        self.num_horizons = num_horizons
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_horizons)
        )
        
        # Softplus ensures positive output
        self.softplus = nn.Softplus()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, channels, seq_len) from backbone
        Returns:
            (batch, num_horizons) - predicted volatility values
        """
        # Use last timestep features (most recent)
        last_features = features[:, :, -1]  # (batch, channels)
        raw_output = self.predictor(last_features)
        
        # Ensure positive volatility
        volatility = self.softplus(raw_output)
        
        if self.num_horizons == 1:
            return volatility.squeeze(-1)
        return volatility


class QuantileHead(nn.Module):
    """
    Predicts price movement distribution via quantile regression.
    
    Outputs quantiles [Q5, Q25, Q50, Q75, Q95] representing the
    distribution of price movement over the prediction horizon.
    
    Critical for:
    - Asymmetric SL/TP calculation
    - Confidence intervals
    - Risk-adjusted position sizing
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        quantiles: Tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)
    ):
        super().__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        
        # Separate pathways for lower and upper quantiles
        # This helps learn asymmetric distributions
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.quantile_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            for _ in quantiles
        ])
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, channels, seq_len) from backbone
        Returns:
            (batch, num_quantiles) - quantile predictions
            Ordered: [Q5, Q25, Q50, Q75, Q95]
        """
        # Use last timestep
        last_features = features[:, :, -1]  # (batch, channels)
        shared_features = self.shared(last_features)
        
        # Predict each quantile
        quantile_preds = []
        for predictor in self.quantile_predictors:
            q = predictor(shared_features)
            quantile_preds.append(q)
        
        # Stack and ensure monotonicity (Q5 < Q25 < Q50 < Q75 < Q95)
        raw_quantiles = torch.cat(quantile_preds, dim=-1)
        
        # Enforce monotonicity via cumulative softmax differences
        quantiles = self._enforce_monotonicity(raw_quantiles)
        
        return quantiles
    
    def _enforce_monotonicity(self, raw: torch.Tensor) -> torch.Tensor:
        """
        Ensure quantiles are monotonically increasing.
        
        Uses cumulative sum of positive differences.
        """
        batch_size = raw.size(0)
        
        # Start with Q5 (median minus offset)
        q50_idx = self.num_quantiles // 2
        q50 = raw[:, q50_idx:q50_idx + 1]
        
        # Build lower quantiles (Q5, Q25) by subtracting positive values from Q50
        lower_diffs = F.softplus(raw[:, :q50_idx])  # Ensure positive
        lower_quantiles = q50 - torch.flip(
            torch.cumsum(torch.flip(lower_diffs, [-1]), dim=-1),
            [-1]
        )
        
        # Build upper quantiles (Q75, Q95) by adding positive values to Q50
        upper_diffs = F.softplus(raw[:, q50_idx + 1:])  # Ensure positive
        upper_quantiles = q50 + torch.cumsum(upper_diffs, dim=-1)
        
        # Combine
        return torch.cat([lower_quantiles, q50, upper_quantiles], dim=-1)


class OutcomeHead(nn.Module):
    """Predicts probability of TP being hit before SL within a fixed horizon.

    This head is designed to align the model output with a tradeable objective.
    It outputs two probabilities:
    - p_long:  P(TP_long hits before SL_long | go long now)
    - p_short: P(TP_short hits before SL_short | go short now)

    The head operates on the last-timestep backbone features.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return probabilities for (p_long, p_short).

        Args:
            features: (batch, channels, seq_len) from backbone

        Returns:
            (batch, 2) probabilities in [0, 1] ordered as [p_long, p_short]
        """
        last_features = features[:, :, -1]  # (batch, channels)
        logits = self.predictor(last_features)
        return torch.sigmoid(logits)


class MultiHeadTCN(nn.Module):
    """
    Complete Multi-Head TCN for Risk Management.
    
    Combines:
    - TCN Backbone: Temporal feature extraction
    - Direction Head: Market direction probabilities
    - Volatility Head: Future volatility prediction
    - Quantile Head: Price movement distribution
    - Outcome Head: TP-before-SL probabilities for long/short trades
    
    This is the core predictive model for Phase 1.
    """
    
    def __init__(self, config: TCNConfig, vision_feature_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        
        # Core backbone
        self.backbone = TCNBackbone(config)
        backbone_dim = self.backbone.output_dim
        
        # Optional vision feature fusion
        self.use_vision = vision_feature_dim is not None
        if self.use_vision:
            self.vision_proj = nn.Linear(vision_feature_dim, backbone_dim)
            self.fusion = nn.Sequential(
                nn.Linear(backbone_dim * 2, backbone_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            head_input_dim = backbone_dim
        else:
            head_input_dim = backbone_dim
        
        # Prediction heads
        self.direction_head = DirectionHead(
            head_input_dim,
            num_classes=config.num_direction_classes
        )
        
        self.volatility_head = VolatilityHead(
            head_input_dim,
            num_horizons=1
        )
        
        self.quantile_head = QuantileHead(
            head_input_dim,
            quantiles=config.quantile_values
        )

        self.outcome_head = OutcomeHead(head_input_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        vision_features: Optional[torch.Tensor] = None,
        mode: str = 'all'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional vision feature fusion.
        
        Args:
            x: (batch, seq_len, features) - OHLCV + technical indicators
            vision_features: (batch, vision_dim) - Optional vision features
            mode: 'all' | 'direction' | 'volatility' | 'quantiles' | 'outcomes' | 'features'
        
        Returns:
            Dictionary with requested outputs:
            - direction: (batch, 3) probabilities
            - volatility: (batch,) predicted σ
            - quantiles: (batch, 5) quantile predictions
            - outcomes: (batch, 2) probabilities [p_long, p_short]
            - features: (batch, hidden_dim) extracted features (for downstream)
        """
        # Extract temporal features
        tcn_features = self.backbone(x)  # (batch, channels, seq_len)
        
        # Fuse vision features if provided
        if self.use_vision and vision_features is not None:
            vision_proj = self.vision_proj(vision_features)  # (batch, channels)
            vision_proj = vision_proj.unsqueeze(-1).expand_as(tcn_features)
            
            # Concatenate and fuse
            combined = torch.cat([tcn_features, vision_proj], dim=1)
            # Reshape for fusion layer
            combined = combined.transpose(1, 2)  # (batch, seq_len, channels*2)
            fused = self.fusion(combined)
            features = fused.transpose(1, 2)  # (batch, channels, seq_len)
        else:
            features = tcn_features
        
        # Return based on mode
        if mode == 'features':
            return {'features': features[:, :, -1]}  # Last timestep
        
        output = {}
        
        if mode in ('all', 'direction'):
            output['direction'] = self.direction_head(features)
        
        if mode in ('all', 'volatility'):
            output['volatility'] = self.volatility_head(features)
        
        if mode in ('all', 'quantiles'):
            output['quantiles'] = self.quantile_head(features)

        if mode in ('all', 'outcomes'):
            outcome_probs = self.outcome_head(features)
            output['outcomes'] = outcome_probs
            output['p_long'] = outcome_probs[:, 0]
            output['p_short'] = outcome_probs[:, 1]
        
        # Always include feature vector for downstream use
        if mode == 'all':
            output['features'] = features[:, :, -1]
        
        return output
    
    def predict_risk_params(
        self,
        x: torch.Tensor,
        vision_features: Optional[torch.Tensor] = None
    ) -> 'RiskPrediction':
        """
        Convenience method for getting all risk parameters at once.
        
        Returns a RiskPrediction dataclass with all outputs.
        """
        with torch.no_grad():
            outputs = self.forward(x, vision_features, mode='all')
        
        return RiskPrediction(
            direction_probs=F.softmax(outputs['direction'], dim=-1),
            volatility=outputs['volatility'],
            quantiles=outputs['quantiles'],
            features=outputs['features'],
            p_long=outputs.get('p_long'),
            p_short=outputs.get('p_short')
        )


@dataclass
class RiskPrediction:
    """Container for multi-head TCN predictions.

    The predictive foundation provides both classic market descriptors
    (direction/volatility/quantiles) and trade-outcome probabilities:
    - p_long:  probability that a long trade hits TP before SL within N bars
    - p_short: probability that a short trade hits TP before SL within N bars
    """
    direction_probs: torch.Tensor   # (batch, 3): P(Bear), P(Side), P(Bull)
    volatility: torch.Tensor        # (batch,): Predicted σ
    quantiles: torch.Tensor         # (batch, 5): Q5, Q25, Q50, Q75, Q95
    features: torch.Tensor          # (batch, hidden_dim): For downstream

    p_long: Optional[torch.Tensor] = None   # (batch,): P(TP before SL | long)
    p_short: Optional[torch.Tensor] = None  # (batch,): P(TP before SL | short)
    
    @property
    def predicted_direction(self) -> torch.Tensor:
        """Get predicted direction as class index (0=Bear, 1=Side, 2=Bull)."""
        return self.direction_probs.argmax(dim=-1)
    
    @property
    def direction_confidence(self) -> torch.Tensor:
        """Get confidence of predicted direction."""
        return self.direction_probs.max(dim=-1).values
    
    @property
    def is_bullish(self) -> torch.Tensor:
        """Boolean mask for bullish predictions."""
        return self.predicted_direction == 2
    
    @property
    def is_bearish(self) -> torch.Tensor:
        """Boolean mask for bearish predictions."""
        return self.predicted_direction == 0
    
    @property
    def expected_move(self) -> torch.Tensor:
        """Expected price movement (Q50 / median)."""
        return self.quantiles[:, 2]
    
    @property
    def upside_potential(self) -> torch.Tensor:
        """Upside potential (Q75 - Q50)."""
        return self.quantiles[:, 3] - self.quantiles[:, 2]
    
    @property
    def downside_risk(self) -> torch.Tensor:
        """Downside risk (Q50 - Q25)."""
        return self.quantiles[:, 2] - self.quantiles[:, 1]
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert to numpy dictionary for serialization."""
        out = {
            'direction_probs': self.direction_probs.cpu().numpy(),
            'volatility': self.volatility.cpu().numpy(),
            'quantiles': self.quantiles.cpu().numpy(),
            'features': self.features.cpu().numpy()
        }

        if self.p_long is not None:
            out['p_long'] = self.p_long.cpu().numpy()
        if self.p_short is not None:
            out['p_short'] = self.p_short.cpu().numpy()

        return out


def create_tcn_for_profile(
    profile: str,
    input_features: int = 64,
    vision_features: Optional[int] = None
) -> MultiHeadTCN:
    """
    Factory function to create profile-specific TCN.
    
    Args:
        profile: 'SCALP', 'INTRADAY', or 'SWING'
        input_features: Number of input features (OHLCV + indicators)
        vision_features: Optional vision feature dimension for fusion
    
    Returns:
        Configured MultiHeadTCN instance
    """
    profile_enum = TradingProfile[profile.upper()]
    
    config = TCNConfig(
        input_channels=input_features,
        hidden_channels=128,
        num_levels=len(TCNConfig(profile=profile_enum).dilations),
        kernel_size=3,
        dropout=0.2,
        profile=profile_enum
    )
    
    return MultiHeadTCN(config, vision_feature_dim=vision_features)
