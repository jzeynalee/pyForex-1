# models/fusion.py
# Fusion model (combines LSTM + ViT + YOLO)

import torch
import torch.nn as nn

class FusionNet(nn.Module):
    def __init__(self, lstm_dim=64, vit_dim=768, yolo_dim=20):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(lstm_dim + vit_dim + yolo_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, lstm_vec, vit_vec, yolo_vec):
        x = torch.cat([lstm_vec, vit_vec, yolo_vec], dim=1)
        return self.fc(x)
