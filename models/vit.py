# models/vit.py
# ViT model (feature extractor)
import torch
import torch.nn as nn
from timm import create_model

class ViTExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = create_model("vit_base_patch16_224", pretrained=True)
        self.vit.head = nn.Identity()   # remove classification head

    def forward(self, x):
        return self.vit(x)
