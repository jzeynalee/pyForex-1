"""
Production-Grade Decision/Fusion Layer for Price Action + TCN

Implements the sophisticated multi-stage fusion architecture with:
1. Feature alignment to shared semantic space
2. Structural attention (Price Action-centric)
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
    gate_weights: torch.Tensor      # (B, 2) - [price_action, tcn]
    regime_probs: torch.Tensor      # (B, 3) - [trend, range, volatile]
    attention_weights: torch.Tensor # (B, 1, N) - Price Action attention
    fused_features: torch.Tensor    # (B, D_f)
    
    # Optional fields (with defaults)
    sl_tp: Optional[List[SLTPResult]] = None
    position_size: Optional[List[PositionSizeResult]] = None


class FeatureAlignment(nn.Module):
    """Stage 1: Project all modalities to shared semantic space."""
    
    def __init__(self, price_action_dim: int, tcn_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Projections to common dimension
        self.price_action_proj = nn.Sequential(
            nn.Linear(price_action_dim, hidden_dim),
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
    
    def forward(self, price_action_feat, tcn_feat):
        """
        Args:
            price_action_feat: (B, N, D_p) or (B, D_p) - Price Action features
            tcn_feat: (B, D_t) - TCN features
        
        Returns:
            Tuple of (price_action_proj, tcn_proj) all in (B, *, D_f)
        """
        # Handle different Price Action input shapes
        if price_action_feat.dim() == 3:
            # (B, N, D_p) -> (B, N, D_f)
            price_action_proj = self.price_action_proj(price_action_feat)
        else:
            # (B, D_p) -> (B, 1, D_f) for consistency
            price_action_proj = self.price_action_proj(price_action_feat).unsqueeze(1)
        
        # Project TCN
        tcn_proj = self.tcn_proj(tcn_feat).unsqueeze(1)  # (B, 1, D_f)
        
        return price_action_proj, tcn_proj


class StructuralAttention(nn.Module):
    """Stage 2: Cross-attention from TCN to Price Action structures."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Multi-head attention components
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, price_action_feat, tcn_feat):
        """
        Args:
            price_action_feat: (B, N, D_f) - Price Action features (keys/values)
            tcn_feat: (B, 1, D_f) - TCN features (query)
        
        Returns:
            attended: (B, 1, D_f)
            attention_weights: (B, 1, N) - for explainability
        """
        batch_size, n_price_action, _ = price_action_feat.shape
        
        # TCN as query
        query = tcn_feat.squeeze(1)  # (B, D_f)
        query = self.q_proj(query)  # (B, D_f)
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, d)
        
        # Price Action as key/value
        key = self.k_proj(price_action_feat)  # (B, N, D_f)
        key = key.view(batch_size, n_price_action, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        
        value = self.v_proj(price_action_feat)  # (B, N, D_f)
        value = value.view(batch_size, n_price_action, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        
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
            nn.Linear(3, hidden_dim // 2),  # 3 quality signals
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2),  # 2 modalities: PA + TCN
            nn.Softmax(dim=1)
        )
    
    def forward(self, price_action_conf, tcn_stability, attn_norm_tcn):
        """
        Args:
            price_action_conf: (B,) or (B, N) - Price Action pattern confidences
            tcn_stability: (B, 1) - TCN prediction stability
            attn_norm_tcn: (B, 1) - L2 norm of attended TCN features
        
        Returns:
            gate_weights: (B, 2) - [price_action, tcn]
        """
        # Process Price Action confidences
        if price_action_conf.dim() > 1:
            price_action_conf = price_action_conf.mean(dim=1)  # Average over patterns
        
        # Normalize inputs
        price_action_conf = torch.clamp(price_action_conf, 0, 1)
        tcn_stability = torch.clamp(1 - tcn_stability, 0, 1)  # Invert stability (higher = better)
        attn_norm_tcn = torch.clamp(attn_norm_tcn / 10, 0, 1)
        
        # Concatenate quality signals
        gate_input = torch.stack([
            price_action_conf,
            tcn_stability.squeeze(1),
            attn_norm_tcn.squeeze(1)
        ], dim=1)  # (B, 3)      
        return self.gate_mlp(gate_input)


class RegimeConditioning(nn.Module):
    """Stage 4: Regime-based gating adjustment."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Regime classifier from TCN features
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3),  # Trend, Range, Volatile
            nn.Softmax(dim=1)
        )
        
        # Regime masks (learnable) for 2-modality gating
        self.register_buffer('trend_mask', torch.tensor([0.3, 0.7]))
        self.register_buffer('range_mask', torch.tensor([0.7, 0.3]))
        self.register_buffer('volatile_mask', torch.tensor([0.4, 0.6]))
    
    def forward(self, tcn_features, gate_weights):
        """
        Args:
            tcn_features: (B, 1, D_f) - TCN features
            gate_weights: (B, 2) - Original gate weights
        
        Returns:
            adjusted_gates: (B, 2) - Regime-adjusted gates
            regime_probs: (B, 3) - Regime probabilities
        """
        # Classify regime
        regime_probs = self.regime_head(tcn_features.squeeze(1))
        
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
    
    Integrates Price Action + TCN with sophisticated multi-stage fusion,
    confidence-aware gating, regime conditioning, and multi-task outputs.
    """
    
    def __init__(
        self,
        price_action_dim: int = 44,  # Updated to match PriceActionPatternExtractor
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
        self.alignment = FeatureAlignment(price_action_dim, tcn_dim, hidden_dim)
        
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
        price_action_features: torch.Tensor,
        price_action_confidence: torch.Tensor,
        tcn_features: torch.Tensor,
        tcn_stability: torch.Tensor,
        apply_risk_management: bool = False,
        market_data: Optional[Dict] = None
    ) -> DecisionOutput:
        """
        Forward pass through complete fusion pipeline.
        
        Args:
            price_action_features: (B, N, D_p) or (B, D_p) - Price Action pattern features
            price_action_confidence: (B,) or (B, N) - Price Action pattern confidences
            tcn_features: (B, D_t) - TCN sequential features
            tcn_stability: (B, 1) - TCN prediction stability
            apply_risk_management: Whether to apply risk calculations
            market_data: Dict with market data for risk management
        
        Returns:
            DecisionOutput with all decision components
        """
        batch_size = price_action_features.size(0)
        
        # Stage 1: Feature alignment
        price_action_proj, tcn_proj = self.alignment(price_action_features, tcn_features)
        
        # Stage 2: Structural attention
        attended_features, attention_weights = self.attention(price_action_proj, tcn_proj)
        
        # Stage 3: Confidence-aware gating
        attn_norm_tcn = torch.norm(attended_features, p=2, dim=-1)
        
        gate_weights = self.gating(
            price_action_confidence,
            tcn_stability,
            attn_norm_tcn
        )
        
        # Stage 4: Regime conditioning (optional)
        if self.use_regime_conditioning:
            gate_weights, regime_probs = self.regime(tcn_proj, gate_weights)
        else:
            regime_probs = torch.zeros(batch_size, 3, device=price_action_features.device)
            regime_probs[:, 1] = 1.0  # Default to range regime
        
        # Weighted fusion
        # Use confidence-weighted Price Action pooling
        if price_action_confidence.dim() > 1:
            price_action_weights = price_action_confidence.unsqueeze(-1)  # (B, N, 1)
            price_action_weighted = torch.sum(price_action_proj * price_action_weights, dim=1) / torch.sum(price_action_weights, dim=1)
        else:
            price_action_weighted = price_action_proj.mean(dim=1)
        
        fused_features = (
            gate_weights[:, 0:1] * price_action_weighted +
            gate_weights[:, 1:2] * tcn_proj.squeeze(1)
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
            pa_w = decision_output.gate_weights[i, 0].item()
            tcn_w = decision_output.gate_weights[i, 1].item()
            
            # Regime
            regime = ['TREND', 'RANGE', 'VOLATILE'][torch.argmax(decision_output.regime_probs[i]).item()]
            
            explanation = {
                'decision': direction,
                'confidence': f"{confidence:.3f}",
                'regime': regime,
                'modality_weights': {
                    'PriceAction (patterns)': f"{pa_w:.3f}",
                    'TCN (temporal)': f"{tcn_w:.3f}"
                },
                'dominant_modality': ['PriceAction', 'TCN'][torch.argmax(decision_output.gate_weights[i]).item()]
            }
            
            explanations.append(explanation)
        
        return {'batch_explanations': explanations}


# =============================================================================
# Factory Functions
# =============================================================================

def create_decision_fusion(
    fusion_type: str = "production",
    price_action_dim: int = 44,
    tcn_dim: int = 64,
    hidden_dim: int = 256,
    **kwargs
) -> nn.Module:
    """
    Factory function to create decision fusion models.
    
    Args:
        fusion_type: Type of fusion ('production', 'simple', 'attention')
        price_action_dim: Price Action feature dimension
        tcn_dim: TCN feature dimension
        hidden_dim: Hidden dimension for projections
        **kwargs: Additional arguments
    
    Returns:
        Decision fusion model
    """
    if fusion_type == "production":
        return DecisionFusionLayer(
            price_action_dim=price_action_dim,
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
            price_action_dim=price_action_dim,
            **kwargs
        )
