# models/vit_extractor.py
"""
Vision Transformer feature extractor.
Extracts CLS-token embeddings from a ViT backbone.

Uses timm:
    pip install timm
"""

import torch
import torch.nn as nn
import timm


class ViTExtractor(nn.Module):
    """
    Wrapper around timm ViT models.
    Returns CLS token features of shape [B, hidden_dim].
    """

    def __init__(self, 
                 model_name="vit_base_patch16_224", 
                 pretrained=True, 
                 freeze=True):
        super().__init__()

        # Load timm backbone
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0  # removes classification head
        )

        # Hidden dimension (usually 768 for vit-base)
        self.hidden_dim = self.vit.num_features

        # Optionally freeze backbone
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        Extract CLS token embeddings using forward_features().
        Returns:
            [B, hidden_dim]
        """
        # timm ViT models implement forward_features()
        feats = self.vit.forward_features(x)

        # Some ViT models return dicts with "cls_token"
        if isinstance(feats, dict):
            # For models where forward_features returns a dict
            if "cls_token" in feats:
                return feats["cls_token"]
            if "x" in feats:
                # Last CLS token is feats["x"][:,0]
                return feats["x"][:, 0]

        # Standard timm output: tensor → CLS at index 0
        return feats[:, 0]
