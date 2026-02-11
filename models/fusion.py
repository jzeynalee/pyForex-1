"""
Multi-modal Fusion Networks for pyForex Trading System.

Combines features from multiple modalities:
- Sequential model (TCN/LSTM) for time-series patterns
- Price Action for rule-based candlestick pattern detection

Note: Variable names use 'seq' (sequence) instead of 'lstm' to be
model-agnostic - works with TCN, LSTM, or any sequential encoder.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class FusionNet(nn.Module):
    """
    Late fusion network with gated attention mechanism.
    
    Uses learned gates to dynamically weight each modality's
    contribution based on the input context.
    """
    
    def __init__(
        self,
        seq_dim: int = 64,       # Sequential model (TCN/LSTM) feature dim
        price_action_dim: int = 44,  # Price Action pattern vector dimension
        hidden_dim: int = 256,   # Projection dimension
        num_classes: int = 3,    # BUY, SELL, HOLD
        dropout: float = 0.3,
        # Ignored legacy kwargs
        vit_dim: int = 0, yolo_dim: int = 0,
    ):
        super().__init__()
        
        pa_dim = price_action_dim or yolo_dim or 44
        self.seq_dim = seq_dim
        self.price_action_dim = pa_dim
        self.hidden_dim = hidden_dim
        
        # Per-modality projection to common dimension
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.pa_proj = nn.Sequential(
            nn.Linear(pa_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Gating mechanism for modality weighting
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1),
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(
        self,
        seq_feat: torch.Tensor,
        pa_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with gated fusion.
        
        Args:
            seq_feat: (batch, seq_dim) - Features from TCN/LSTM
            pa_feat: (batch, price_action_dim) - Features from Price Action patterns
        
        Returns:
            logits: (batch, num_classes)
        """
        seq_proj = self.seq_proj(seq_feat)
        pa_proj = self.pa_proj(pa_feat)
        
        concat = torch.cat([seq_proj, pa_proj], dim=1)
        gates = self.gate(concat)  # (batch, 2)
        
        fused = gates[:, 0:1] * seq_proj + gates[:, 1:2] * pa_proj
        return self.classifier(fused)
    
    def forward_with_gates(
        self,
        seq_feat: torch.Tensor,
        pa_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns gate weights for interpretability.
        
        Returns:
            (logits, gate_weights) where gate_weights is (batch, 2)
            with columns [seq_weight, pa_weight]
        """
        seq_proj = self.seq_proj(seq_feat)
        pa_proj = self.pa_proj(pa_feat)
        
        concat = torch.cat([seq_proj, pa_proj], dim=1)
        gates = self.gate(concat)
        
        fused = gates[:, 0:1] * seq_proj + gates[:, 1:2] * pa_proj
        logits = self.classifier(fused)
        return logits, gates
    
    def get_modality_importance(
        self,
        seq_feat: torch.Tensor,
        pa_feat: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Get average modality importance across batch.
        
        Returns:
            {'sequence': weight, 'pattern': weight}
        """
        _, gates = self.forward_with_gates(seq_feat, pa_feat)
        avg_gates = gates.mean(dim=0).detach().cpu().numpy()
        
        return {
            'sequence': float(avg_gates[0]),
            'pattern': float(avg_gates[1]),
        }


class SimpleFusion(nn.Module):
    """
    Simple concatenation-based fusion.
    
    Faster but less sophisticated than gated fusion.
    Good for baseline comparisons.
    """
    
    def __init__(
        self,
        seq_dim: int = 64,
        price_action_dim: int = 44,
        num_classes: int = 3,
        dropout: float = 0.3,
        # Ignored legacy kwargs
        vit_dim: int = 0, yolo_dim: int = 0,
    ):
        super().__init__()
        
        pa_dim = price_action_dim or yolo_dim or 44
        total_dim = seq_dim + pa_dim
        
        self.fc = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )
    
    def forward(
        self,
        seq_feat: torch.Tensor,
        pa_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Simple concatenation fusion."""
        x = torch.cat([seq_feat, pa_feat], dim=1)
        return self.fc(x)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion with cross-modal attention.
    
    Most sophisticated fusion - treats modalities as a sequence
    and applies self-attention to learn cross-modal relationships.
    """
    
    def __init__(
        self,
        seq_dim: int = 64,
        price_action_dim: int = 44,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.1,
        # Ignored legacy kwargs
        vit_dim: int = 0, yolo_dim: int = 0,
    ):
        super().__init__()
        
        pa_dim = price_action_dim or yolo_dim or 44
        self.hidden_dim = hidden_dim
        
        # Project all modalities to same dimension
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.pa_proj = nn.Linear(pa_dim, hidden_dim)
        
        # Learnable modality embeddings (like positional encoding)
        self.modality_embeddings = nn.Parameter(
            torch.randn(2, hidden_dim) * 0.02
        )
        
        # Multi-head self-attention over modalities
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(
        self,
        seq_feat: torch.Tensor,
        pa_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention-based fusion with cross-modal interaction.
        """
        batch_size = seq_feat.size(0)
        
        # Project to common dimension and add modality embeddings
        seq_proj = self.seq_proj(seq_feat) + self.modality_embeddings[0]
        pa_proj = self.pa_proj(pa_feat) + self.modality_embeddings[1]
        
        # Stack as sequence: (batch, 2, hidden_dim)
        x = torch.stack([seq_proj, pa_proj], dim=1)
        
        # Self-attention over modalities
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Flatten and classify
        x = x.view(batch_size, -1)
        return self.classifier(x)


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion that combines seq + price-action features
    with a gated bottleneck.
    """
    
    def __init__(
        self,
        seq_dim: int = 64,
        price_action_dim: int = 44,
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3,
        # Ignored legacy kwargs
        vit_dim: int = 0, yolo_dim: int = 0,
    ):
        super().__init__()
        
        pa_dim = price_action_dim or yolo_dim or 44
        
        # Pattern fusion (sequence + price action)
        self.pattern_fusion = nn.Sequential(
            nn.Linear(seq_dim + pa_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(
        self,
        seq_feat: torch.Tensor,
        pa_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Fusion via concatenation + MLP."""
        pattern_input = torch.cat([seq_feat, pa_feat], dim=1)
        fused = self.pattern_fusion(pattern_input)
        return self.classifier(fused)


# =============================================================================
# Factory Function
# =============================================================================

def create_fusion_model(
    fusion_type: str = "gated",
    seq_dim: int = 64,
    price_action_dim: int = 44,
    num_classes: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create fusion models.
    
    Args:
        fusion_type: One of 'gated', 'simple', 'attention', 'hierarchical'
        seq_dim: Dimension of sequential features (TCN/LSTM output)
        price_action_dim: Dimension of Price Action pattern vector
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to specific model
    
    Returns:
        Fusion model instance
    """
    fusion_classes = {
        'gated': FusionNet,
        'simple': SimpleFusion,
        'attention': AttentionFusion,
        'hierarchical': HierarchicalFusion,
    }
    
    if fusion_type not in fusion_classes:
        raise ValueError(
            f"Unknown fusion type: {fusion_type}. "
            f"Choose from: {list(fusion_classes.keys())}"
        )
    
    return fusion_classes[fusion_type](
        seq_dim=seq_dim,
        price_action_dim=price_action_dim,
        num_classes=num_classes,
        **kwargs
    )


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# For code that still uses old naming
def FusionNet_compat(lstm_dim: int = 64, **kwargs):
    """Backward compatible constructor using old 'lstm_dim' parameter."""
    return FusionNet(seq_dim=lstm_dim, **kwargs)