# models/fusion.py
"""
Multi-modal fusion network combining LSTM, ViT, and YOLO features.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class FusionNet(nn.Module):
    """
    Late fusion network combining features from multiple modalities.
    
    Uses gated fusion mechanism for better modality weighting.
    """
    
    def __init__(
        self,
        lstm_dim: int = 64,
        vit_dim: int = 768,
        yolo_dim: int = 20,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.lstm_dim = lstm_dim
        self.vit_dim = vit_dim
        self.yolo_dim = yolo_dim
        total_dim = lstm_dim + vit_dim + yolo_dim
        
        # Per-modality projection to common dimension
        self.lstm_proj = nn.Sequential(
            nn.Linear(lstm_dim, hidden_dim),
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
            nn.Linear(hidden_dim * 3, 3),
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
        lstm_feat: torch.Tensor,
        vit_feat: torch.Tensor,
        yolo_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            lstm_feat: (batch, lstm_dim)
            vit_feat: (batch, vit_dim)
            yolo_feat: (batch, yolo_dim)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Project each modality
        lstm_proj = self.lstm_proj(lstm_feat)   # (batch, hidden_dim)
        vit_proj = self.vit_proj(vit_feat)      # (batch, hidden_dim)
        yolo_proj = self.yolo_proj(yolo_feat)   # (batch, hidden_dim)
        
        # Compute gating weights
        concat = torch.cat([lstm_proj, vit_proj, yolo_proj], dim=1)
        gates = self.gate(concat)  # (batch, 3)
        
        # Weighted combination
        fused = (
            gates[:, 0:1] * lstm_proj +
            gates[:, 1:2] * vit_proj +
            gates[:, 2:3] * yolo_proj
        )
        
        return self.classifier(fused)
    
    def forward_with_gates(
        self,
        lstm_feat: torch.Tensor,
        vit_feat: torch.Tensor,
        yolo_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns gate weights for interpretability."""
        lstm_proj = self.lstm_proj(lstm_feat)
        vit_proj = self.vit_proj(vit_feat)
        yolo_proj = self.yolo_proj(yolo_feat)
        
        concat = torch.cat([lstm_proj, vit_proj, yolo_proj], dim=1)
        gates = self.gate(concat)
        
        fused = (
            gates[:, 0:1] * lstm_proj +
            gates[:, 1:2] * vit_proj +
            gates[:, 2:3] * yolo_proj
        )
        
        logits = self.classifier(fused)
        return logits, gates


class SimpleFusion(nn.Module):
    """
    Simple concatenation-based fusion.
    Faster but less sophisticated than gated fusion.
    """
    
    def __init__(
        self,
        lstm_dim: int = 64,
        vit_dim: int = 768,
        yolo_dim: int = 20,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        total_dim = lstm_dim + vit_dim + yolo_dim
        
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
        lstm_feat: torch.Tensor,
        vit_feat: torch.Tensor,
        yolo_feat: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([lstm_feat, vit_feat, yolo_feat], dim=1)
        return self.fc(x)


class AttentionFusion(nn.Module):
    """
    Attention-based fusion with cross-modal attention.
    Most sophisticated but slowest.
    """
    
    def __init__(
        self,
        lstm_dim: int = 64,
        vit_dim: int = 768,
        yolo_dim: int = 20,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 3,
    ):
        super().__init__()
        
        # Project all to same dimension
        self.lstm_proj = nn.Linear(lstm_dim, hidden_dim)
        self.vit_proj = nn.Linear(vit_dim, hidden_dim)
        self.yolo_proj = nn.Linear(yolo_dim, hidden_dim)
        
        # Multi-head self-attention over modalities
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(
        self,
        lstm_feat: torch.Tensor,
        vit_feat: torch.Tensor,
        yolo_feat: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = lstm_feat.size(0)
        
        # Project
        lstm_proj = self.lstm_proj(lstm_feat).unsqueeze(1)   # (batch, 1, hidden)
        vit_proj = self.vit_proj(vit_feat).unsqueeze(1)
        yolo_proj = self.yolo_proj(yolo_feat).unsqueeze(1)
        
        # Stack as sequence: (batch, 3, hidden)
        x = torch.cat([lstm_proj, vit_proj, yolo_proj], dim=1)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        
        # Flatten and classify
        x = x.view(batch_size, -1)
        return self.classifier(x)
