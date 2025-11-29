# models/lstm.py
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim=5, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:,-1]
        return self.fc(out)
