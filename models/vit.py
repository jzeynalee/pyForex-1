# models/vit.py
"""
Vision Transformer for candlestick chart analysis.
Extracts visual features from chart images.
"""
import torch
import torch.nn as nn
from typing import Optional

try:
    from timm import create_model
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class ViTExtractor(nn.Module):
    """
    Vision Transformer feature extractor using timm library.
    
    Outputs a fixed-size feature vector for fusion with other modalities.
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm library required. Install with: pip install timm")
        
        # Load pretrained ViT
        self.vit = create_model(model_name, pretrained=pretrained)
        
        # Get the original output dimension (768 for base, 1024 for large)
        self.original_dim = self.vit.head.in_features
        
        # Remove classification head - we want features
        self.vit.head = nn.Identity()
        
        # Optional projection to different dimension
        if output_dim and output_dim != self.original_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.original_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
            self.feature_dim = output_dim
        else:
            self.projection = None
            self.feature_dim = self.original_dim
        
        # Optionally freeze backbone for faster training
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all ViT parameters except projection."""
        for param in self.vit.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self, unfreeze_last_n_blocks: int = 2):
        """Unfreeze last N transformer blocks for fine-tuning."""
        # First freeze everything
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Unfreeze last N blocks
        num_blocks = len(self.vit.blocks)
        for i in range(num_blocks - unfreeze_last_n_blocks, num_blocks):
            for param in self.vit.blocks[i].parameters():
                param.requires_grad = True
        
        # Always unfreeze norm
        for param in self.vit.norm.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (batch, 3, 224, 224)
               Values should be normalized to [0, 1] or ImageNet stats
        
        Returns:
            features: (batch, feature_dim)
        """
        # ViT forward (without head returns CLS token features)
        features = self.vit(x)  # (batch, original_dim)
        
        if self.projection is not None:
            features = self.projection(features)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Returns output feature dimension for fusion layer."""
        return self.feature_dim


class LightweightViT(nn.Module):
    """
    Smaller ViT variant for faster inference.
    Uses a tiny/small model instead of base.
    """
    
    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = True,
    ):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm library required")
        
        self.vit = create_model(model_name, pretrained=pretrained)
        self.feature_dim = self.vit.head.in_features  # 192 for tiny
        self.vit.head = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)
    
    def get_feature_dim(self) -> int:
        return self.feature_dim


class ViTChartClassifier(nn.Module):
    """
    Vision Transformer classifier for candlestick chart patterns.
    This is the class that the predictor expects to import.
    """
    
    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = True,
        num_classes: int = 3,  # Bear, Sideways, Bull
        input_size: int = 224,
    ):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm library required. Install with: pip install timm")
        
        # Load ViT model
        self.vit = create_model(model_name, pretrained=pretrained)
        
        # Get feature dimension
        self.feature_dim = self.vit.head.in_features
        
        # Replace head with our classifier
        self.vit.head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        self.num_classes = num_classes
        self.input_size = input_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (batch, 3, input_size, input_size)
        
        Returns:
            logits: (batch, num_classes)
        """
        return self.vit(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
