"""
Multi-modal Fusion Networks for pyForex Trading System.

Combines features from multiple modalities:
- Sequential model (TCN/LSTM) for time-series patterns
- Vision Transformer (ViT) for visual chart patterns  
- YOLO for candlestick pattern detection

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
        vit_dim: int = 768,      # ViT feature dimension
        yolo_dim: int = 25,      # YOLO pattern vector dimension
        hidden_dim: int = 256,   # Projection dimension
        num_classes: int = 3,    # BUY, SELL, HOLD
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.seq_dim = seq_dim
        self.vit_dim = vit_dim
        self.yolo_dim = yolo_dim
        self.hidden_dim = hidden_dim
        
        # Per-modality projection to common dimension
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.yolo_proj = nn.Sequential(
            nn.Linear(yolo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Gating mechanism for modality weighting
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
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
        vit_feat: torch.Tensor,
        yolo_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with gated fusion.
        
        Args:
            seq_feat: (batch, seq_dim) - Features from TCN/LSTM
            vit_feat: (batch, vit_dim) - Features from ViT
            yolo_feat: (batch, yolo_dim) - Features from YOLO
        
        Returns:
            logits: (batch, num_classes)
        """
        # Project each modality to common dimension
        seq_proj = self.seq_proj(seq_feat)    # (batch, hidden_dim)
        vit_proj = self.vit_proj(vit_feat)    # (batch, hidden_dim)
        yolo_proj = self.yolo_proj(yolo_feat) # (batch, hidden_dim)
        
        # Compute gating weights based on concatenated features
        concat = torch.cat([seq_proj, vit_proj, yolo_proj], dim=1)
        gates = self.gate(concat)  # (batch, 3)
        
        # Weighted combination
        fused = (
            gates[:, 0:1] * seq_proj +
            gates[:, 1:2] * vit_proj +
            gates[:, 2:3] * yolo_proj
        )
        
        return self.classifier(fused)
    
    def forward_with_gates(
        self,
        seq_feat: torch.Tensor,
        vit_feat: torch.Tensor,
        yolo_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns gate weights for interpretability.
        
        Returns:
            (logits, gate_weights) where gate_weights is (batch, 3)
            with columns [seq_weight, vit_weight, yolo_weight]
        """
        seq_proj = self.seq_proj(seq_feat)
        vit_proj = self.vit_proj(vit_feat)
        yolo_proj = self.yolo_proj(yolo_feat)
        
        concat = torch.cat([seq_proj, vit_proj, yolo_proj], dim=1)
        gates = self.gate(concat)
        
        fused = (
            gates[:, 0:1] * seq_proj +
            gates[:, 1:2] * vit_proj +
            gates[:, 2:3] * yolo_proj
        )
        
        logits = self.classifier(fused)
        return logits, gates
    
    def get_modality_importance(
        self,
        seq_feat: torch.Tensor,
        vit_feat: torch.Tensor,
        yolo_feat: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Get average modality importance across batch.
        
        Returns:
            {'sequence': weight, 'vision': weight, 'pattern': weight}
        """
        _, gates = self.forward_with_gates(seq_feat, vit_feat, yolo_feat)
        avg_gates = gates.mean(dim=0).detach().cpu().numpy()
        
        return {
            'sequence': float(avg_gates[0]),
            'vision': float(avg_gates[1]),
            'pattern': float(avg_gates[2]),
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
        vit_dim: int = 768,
        yolo_dim: int = 25,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        total_dim = seq_dim + vit_dim + yolo_dim
        
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
        vit_feat: torch.Tensor,
        yolo_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Simple concatenation fusion."""
        x = torch.cat([seq_feat, vit_feat, yolo_feat], dim=1)
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
        vit_dim: int = 768,
        yolo_dim: int = 25,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Project all modalities to same dimension
        self.seq_proj = nn.Linear(seq_dim, hidden_dim)
        self.vit_proj = nn.Linear(vit_dim, hidden_dim)
        self.yolo_proj = nn.Linear(yolo_dim, hidden_dim)
        
        # Learnable modality embeddings (like positional encoding)
        self.modality_embeddings = nn.Parameter(
            torch.randn(3, hidden_dim) * 0.02
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
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(
        self,
        seq_feat: torch.Tensor,
        vit_feat: torch.Tensor,
        yolo_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention-based fusion with cross-modal interaction.
        """
        batch_size = seq_feat.size(0)
        
        # Project to common dimension and add modality embeddings
        seq_proj = self.seq_proj(seq_feat) + self.modality_embeddings[0]
        vit_proj = self.vit_proj(vit_feat) + self.modality_embeddings[1]
        yolo_proj = self.yolo_proj(yolo_feat) + self.modality_embeddings[2]
        
        # Stack as sequence: (batch, 3, hidden_dim)
        x = torch.stack([seq_proj, vit_proj, yolo_proj], dim=1)
        
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
    Hierarchical fusion that first combines related modalities,
    then fuses at a higher level.
    
    Architecture:
    1. Combine seq + yolo (both capture patterns)
    2. Combine result with vit (adds visual context)
    """
    
    def __init__(
        self,
        seq_dim: int = 64,
        vit_dim: int = 768,
        yolo_dim: int = 25,
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        # Level 1: Pattern fusion (sequence + YOLO)
        self.pattern_fusion = nn.Sequential(
            nn.Linear(seq_dim + yolo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Level 2: Visual context projection
        self.visual_proj = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Level 2: Final fusion with gating
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1),
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
        vit_feat: torch.Tensor,
        yolo_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Hierarchical two-stage fusion."""
        # Level 1: Combine pattern-based features
        pattern_input = torch.cat([seq_feat, yolo_feat], dim=1)
        pattern_fused = self.pattern_fusion(pattern_input)
        
        # Level 2: Project visual features
        visual_proj = self.visual_proj(vit_feat)
        
        # Gated combination
        concat = torch.cat([pattern_fused, visual_proj], dim=1)
        gates = self.fusion_gate(concat)
        
        fused = gates[:, 0:1] * pattern_fused + gates[:, 1:2] * visual_proj
        
        return self.classifier(fused)


# =============================================================================
# Factory Function
# =============================================================================

def create_fusion_model(
    fusion_type: str = "gated",
    seq_dim: int = 64,
    vit_dim: int = 768,
    yolo_dim: int = 25,
    num_classes: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create fusion models.
    
    Args:
        fusion_type: One of 'gated', 'simple', 'attention', 'hierarchical'
        seq_dim: Dimension of sequential features (TCN/LSTM output)
        vit_dim: Dimension of ViT features
        yolo_dim: Dimension of YOLO pattern vector
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
        vit_dim=vit_dim,
        yolo_dim=yolo_dim,
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