"""
Production-Grade Decision/Fusion Layer for YOLO + ViT + TCN

Implements the sophisticated multi-stage fusion architecture with:
1. Feature alignment to shared semantic space
2. Structural attention (YOLO-centric)
3. Confidence-aware gating with market signals
4. Regime conditioning
5. Multi-task decision heads

Integrates with risk_management module for SL/TP and position sizing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np

# Import risk management components
from risk_management.risk_manager import RiskManager, RiskManagerConfig
from risk_management.phase2_risk_calc import SLTPResult, PositionSizeResult, MarketRegime


@dataclass
class DecisionOutput:
    """Complete decision output from fusion layer."""
    # Required fields (no defaults)
    direction_logits: torch.Tensor  # (B, 3) - BEARISH, SIDEWAYS, BULLISH
    direction_probs: torch.Tensor   # (B, 3)
    direction_label: torch.Tensor   # (B,) - argmax
    confidence: torch.Tensor        # (B, 1) - [0, 1]
    gate_weights: torch.Tensor      # (B, 3) - [yolo, vit, tcn]
    regime_probs: torch.Tensor      # (B, 3) - [trend, range, volatile]
    attention_weights: torch.Tensor # (B, 1, N) - YOLO attention
    fused_features: torch.Tensor    # (B, D_f)
    
    # Optional fields (with defaults)
    sl_tp: Optional[List[SLTPResult]] = None
    position_size: Optional[List[PositionSizeResult]] = None


class FeatureAlignment(nn.Module):
    """Stage 1: Project all modalities to shared semantic space."""
    
    def __init__(self, yolo_dim: int, vit_dim: int, tcn_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Projections to common dimension
        self.yolo_proj = nn.Sequential(
            nn.Linear(yolo_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.tcn_proj = nn.Sequential(
            nn.Linear(tcn_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, yolo_feat, vit_feat, tcn_feat):
        """
        Args:
            yolo_feat: (B, N, D_y) or (B, D_y) - YOLO features
            vit_feat: (B, D_v) - ViT CLS token
            tcn_feat: (B, D_t) - TCN features
        
        Returns:
            Tuple of (yolo_proj, vit_proj, tcn_proj) all in (B, *, D_f)
        """
        # Handle different YOLO input shapes
        if yolo_feat.dim() == 3:
            # (B, N, D_y) -> (B, N, D_f)
            yolo_proj = self.yolo_proj(yolo_feat)
        else:
            # (B, D_y) -> (B, 1, D_f) for consistency
            yolo_proj = self.yolo_proj(yolo_feat).unsqueeze(1)
        
        # Project ViT and TCN
        vit_proj = self.vit_proj(vit_feat).unsqueeze(1)  # (B, 1, D_f)
        tcn_proj = self.tcn_proj(tcn_feat).unsqueeze(1)  # (B, 1, D_f)
        
        return yolo_proj, vit_proj, tcn_proj


class StructuralAttention(nn.Module):
    """Stage 2: Cross-attention from ViT/TCN to YOLO structures."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Multi-head attention components
        self.q_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, yolo_feat, vit_feat, tcn_feat):
        """
        Args:
            yolo_feat: (B, N, D_f) - YOLO features (keys/values)
            vit_feat: (B, 1, D_f) - ViT features (query)
            tcn_feat: (B, 1, D_f) - TCN features (query)
        
        Returns:
            attended_vit, attended_tcn: (B, 1, D_f) each
            attention_weights: (B, 1, N) - for explainability
        """
        batch_size, n_yolo, _ = yolo_feat.shape
        
        # Combine ViT and TCN as query
        query = torch.cat([vit_feat.squeeze(1), tcn_feat.squeeze(1)], dim=1)  # (B, 2*D_f)
        query = self.q_proj(query)  # (B, D_f)
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, d)
        
        # YOLO as key/value
        key = self.k_proj(yolo_feat)  # (B, N, D_f)
        key = key.view(batch_size, n_yolo, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        
        value = self.v_proj(yolo_feat)  # (B, N, D_f)
        value = value.view(batch_size, n_yolo, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # (B, H, 1, N)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attended = torch.matmul(attn_weights, value)  # (B, H, 1, d)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_dim)  # (B, 1, D_f)
        attended = self.out_proj(attended)
        
        # Average attention weights across heads for explainability
        avg_attn = attn_weights.mean(dim=1)  # (B, 1, N)
        
        return attended, avg_attn


class ConfidenceAwareGating(nn.Module):
    """Stage 3: Dynamic gating based on quality signals."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),  # 5 quality signals
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3),  # 3 modalities
            nn.Softmax(dim=1)
        )
    
    def forward(self, yolo_conf, vit_entropy, tcn_stability, attn_norm_vit, attn_norm_tcn):
        """
        Args:
            yolo_conf: (B,) or (B, N) - YOLO object confidences
            vit_entropy: (B, 1) - ViT attention entropy
            tcn_stability: (B, 1) - TCN prediction stability
            attn_norm_vit: (B, 1) - L2 norm of attended ViT features
            attn_norm_tcn: (B, 1) - L2 norm of attended TCN features
        
        Returns:
            gate_weights: (B, 3) - [yolo, vit, tcn]
        """
        # Process YOLO confidences
        if yolo_conf.dim() > 1:
            yolo_conf = yolo_conf.mean(dim=1)  # Average over detections
        
        # Normalize inputs
        yolo_conf = torch.clamp(yolo_conf, 0, 1)
        vit_entropy = torch.clamp(vit_entropy / 10, 0, 1)  # Normalize entropy
        tcn_stability = torch.clamp(1 - tcn_stability, 0, 1)  # Invert stability (higher = better)
        attn_norm_vit = torch.clamp(attn_norm_vit / 10, 0, 1)
        attn_norm_tcn = torch.clamp(attn_norm_tcn / 10, 0, 1)
        
        # Concatenate quality signals
        gate_input = torch.stack([
            yolo_conf,
            vit_entropy.squeeze(1),
            tcn_stability.squeeze(1),
            attn_norm_vit.squeeze(1),
            attn_norm_tcn.squeeze(1)
        ], dim=1)
        
        return self.gate_mlp(gate_input)


class RegimeConditioning(nn.Module):
    """Stage 4: Regime-based gating adjustment."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Regime classifier from ViT features
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3),  # Trend, Range, Volatile
            nn.Softmax(dim=1)
        )
        
        # Regime masks (learnable)
        self.register_buffer('trend_mask', torch.tensor([0.2, 0.5, 0.3]))
        self.register_buffer('range_mask', torch.tensor([0.6, 0.2, 0.2]))
        self.register_buffer('volatile_mask', torch.tensor([0.2, 0.2, 0.6]))
    
    def forward(self, vit_features, gate_weights):
        """
        Args:
            vit_features: (B, 1, D_f) - ViT features
            gate_weights: (B, 3) - Original gate weights
        
        Returns:
            adjusted_gates: (B, 3) - Regime-adjusted gates
            regime_probs: (B, 3) - Regime probabilities
        """
        # Classify regime
        regime_probs = self.regime_head(vit_features.squeeze(1))
        
        # Weight regime masks by probabilities
        adjusted_mask = (
            regime_probs[:, 0:1] * self.trend_mask +
            regime_probs[:, 1:2] * self.range_mask +
            regime_probs[:, 2:3] * self.volatile_mask
        )
        
        # Apply regime conditioning
        adjusted_gates = gate_weights * adjusted_mask
        adjusted_gates = F.softmax(adjusted_gates, dim=1)  # Re-normalize
        
        return adjusted_gates, regime_probs


class MultiTaskHeads(nn.Module):
    """Stage 5: Specialized heads for different outputs."""
    
    def __init__(self, hidden_dim: int, num_classes: int = 3):
        super().__init__()
        
        # Direction head
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # SL/TP quantile heads (for risk asymmetry)
        self.sl_quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 5)  # 5 quantiles: 0.1, 0.25, 0.5, 0.75, 0.9
        )
        
        self.tp_quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 5)
        )
        
        # Position sizing head
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, fused_features):
        """
        Args:
            fused_features: (B, D_f) - Fused representation
        
        Returns:
            Dictionary of all head outputs
        """
        direction_logits = self.direction_head(fused_features)
        direction_probs = F.softmax(direction_logits, dim=1)
        direction_label = torch.argmax(direction_probs, dim=1)
        
        confidence = self.confidence_head(fused_features)
        
        sl_quantiles = self.sl_quantile_head(fused_features)
        tp_quantiles = self.tp_quantile_head(fused_features)
        
        position_size = self.position_head(fused_features)
        
        return {
            'direction_logits': direction_logits,
            'direction_probs': direction_probs,
            'direction_label': direction_label,
            'confidence': confidence,
            'sl_quantiles': sl_quantiles,
            'tp_quantiles': tp_quantiles,
            'position_size': position_size
        }


class DecisionFusionLayer(nn.Module):
    """
    Complete Decision/Fusion Layer implementing the production-grade architecture.
    
    Integrates YOLO + ViT + TCN with sophisticated multi-stage fusion,
    confidence-aware gating, regime conditioning, and multi-task outputs.
    """
    
    def __init__(
        self,
        yolo_dim: int = 20,
        vit_dim: int = 768,
        tcn_dim: int = 64,
        hidden_dim: int = 256,
        num_classes: int = 3,
        use_regime_conditioning: bool = True,
        risk_manager_config: Optional[RiskManagerConfig] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_regime_conditioning = use_regime_conditioning
        
        # Stage 1: Feature alignment
        self.alignment = FeatureAlignment(yolo_dim, vit_dim, tcn_dim, hidden_dim)
        
        # Stage 2: Structural attention
        self.attention = StructuralAttention(hidden_dim)
        
        # Stage 3: Confidence-aware gating
        self.gating = ConfidenceAwareGating(hidden_dim)
        
        # Stage 4: Regime conditioning (optional)
        if use_regime_conditioning:
            self.regime = RegimeConditioning(hidden_dim)
        
        # Stage 5: Multi-task heads
        self.heads = MultiTaskHeads(hidden_dim, num_classes)
        
        # Risk manager integration
        self.risk_manager = None
        if risk_manager_config:
            self.risk_manager = RiskManager(risk_manager_config)
    
    def forward(
        self,
        yolo_features: torch.Tensor,
        yolo_confidence: torch.Tensor,
        vit_features: torch.Tensor,
        vit_entropy: torch.Tensor,
        tcn_features: torch.Tensor,
        tcn_stability: torch.Tensor,
        apply_risk_management: bool = False,
        market_data: Optional[Dict] = None
    ) -> DecisionOutput:
        """
        Forward pass through complete fusion pipeline.
        
        Args:
            yolo_features: (B, N, D_y) or (B, D_y) - YOLO pattern features
            yolo_confidence: (B,) or (B, N) - YOLO detection confidences
            vit_features: (B, D_v) - ViT CLS token features
            vit_entropy: (B, 1) - ViT attention entropy
            tcn_features: (B, D_t) - TCN sequential features
            tcn_stability: (B, 1) - TCN prediction stability
            apply_risk_management: Whether to apply risk calculations
            market_data: Dict with market data for risk management
        
        Returns:
            DecisionOutput with all decision components
        """
        batch_size = yolo_features.size(0)
        
        # Stage 1: Feature alignment
        yolo_proj, vit_proj, tcn_proj = self.alignment(yolo_features, vit_features, tcn_features)
        
        # Stage 2: Structural attention
        attended_features, attention_weights = self.attention(yolo_proj, vit_proj, tcn_proj)
        
        # Stage 3: Confidence-aware gating
        attn_norm_vit = torch.norm(attended_features, p=2, dim=-1)
        attn_norm_tcn = torch.norm(attended_features, p=2, dim=-1)
        
        gate_weights = self.gating(
            yolo_confidence,
            vit_entropy,
            tcn_stability,
            attn_norm_vit,
            attn_norm_tcn
        )
        
        # Stage 4: Regime conditioning (optional)
        if self.use_regime_conditioning:
            gate_weights, regime_probs = self.regime(vit_proj, gate_weights)
        else:
            regime_probs = torch.zeros(batch_size, 3, device=yolo_features.device)
            regime_probs[:, 1] = 1.0  # Default to range regime
        
        # Weighted fusion
        # Use confidence-weighted YOLO pooling
        if yolo_confidence.dim() > 1:
            yolo_weights = yolo_confidence.unsqueeze(-1)  # (B, N, 1)
            yolo_weighted = torch.sum(yolo_proj * yolo_weights, dim=1) / torch.sum(yolo_weights, dim=1)
        else:
            yolo_weighted = yolo_proj.mean(dim=1)
        
        fused_features = (
            gate_weights[:, 0:1] * yolo_weighted +
            gate_weights[:, 1:2] * attended_features.squeeze(1) +
            gate_weights[:, 2:3] * tcn_proj.squeeze(1)
        )
        
        # Stage 5: Multi-task heads
        head_outputs = self.heads(fused_features)
        
        # Apply risk management if requested
        sl_tp_results = None
        position_results = None
        
        if apply_risk_management and self.risk_manager and market_data:
            sl_tp_results, position_results = self._apply_risk_management(
                head_outputs, market_data
            )
        
        # Create decision output
        return DecisionOutput(
            direction_logits=head_outputs['direction_logits'],
            direction_probs=head_outputs['direction_probs'],
            direction_label=head_outputs['direction_label'],
            confidence=head_outputs['confidence'],
            sl_tp=sl_tp_results,
            position_size=position_results,
            gate_weights=gate_weights,
            regime_probs=regime_probs,
            attention_weights=attention_weights,
            fused_features=fused_features
        )
    
    def _apply_risk_management(self, head_outputs: Dict, market_data: Dict):
        """Apply risk management calculations using integrated RiskManager."""
        # This would integrate with the risk_management module
        # Implementation depends on specific market_data format
        # Placeholder for now
        return None, None
    
    def get_explanation(self, decision_output: DecisionOutput) -> Dict:
        """
        Generate human-readable explanation for the decision.
        
        Args:
            decision_output: Output from forward pass
        
        Returns:
            Dictionary with explanation components
        """
        batch_size = decision_output.direction_logits.size(0)
        explanations = []
        
        for i in range(batch_size):
            direction = ['BEARISH', 'SIDEWAYS', 'BULLISH'][decision_output.direction_label[i].item()]
            confidence = decision_output.confidence[i].item()
            
            # Modality importance
            yolo_w = decision_output.gate_weights[i, 0].item()
            vit_w = decision_output.gate_weights[i, 1].item()
            tcn_w = decision_output.gate_weights[i, 2].item()
            
            # Regime
            regime = ['TREND', 'RANGE', 'VOLATILE'][torch.argmax(decision_output.regime_probs[i]).item()]
            
            explanation = {
                'decision': direction,
                'confidence': f"{confidence:.3f}",
                'regime': regime,
                'modality_weights': {
                    'YOLO (patterns)': f"{yolo_w:.3f}",
                    'ViT (context)': f"{vit_w:.3f}",
                    'TCN (temporal)': f"{tcn_w:.3f}"
                },
                'dominant_modality': ['YOLO', 'ViT', 'TCN'][torch.argmax(decision_output.gate_weights[i]).item()]
            }
            
            explanations.append(explanation)
        
        return {'batch_explanations': explanations}


# =============================================================================
# Factory Functions
# =============================================================================

def create_decision_fusion(
    fusion_type: str = "production",
    yolo_dim: int = 20,
    vit_dim: int = 768,
    tcn_dim: int = 64,
    hidden_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to create decision fusion models.
    
    Args:
        fusion_type: Type of fusion ('production', 'simple', 'attention')
        yolo_dim: YOLO feature dimension
        vit_dim: ViT feature dimension
        tcn_dim: TCN feature dimension
        hidden_dim: Hidden dimension for projections
        **kwargs: Additional arguments
    
    Returns:
        Decision fusion model
    """
    if fusion_type == "production":
        return DecisionFusionLayer(
            yolo_dim=yolo_dim,
            vit_dim=vit_dim,
            tcn_dim=tcn_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )
    else:
        # Fall back to existing fusion models for comparison
        from .fusion import create_fusion_model
        return create_fusion_model(
            fusion_type=fusion_type,
            seq_dim=tcn_dim,
            vit_dim=vit_dim,
            yolo_dim=yolo_dim,
            **kwargs
        )
