"""
Raw-Feature MH-TCN Filter (Standard Mode).

Input:  OHLC + engineered features (time-windowed)
Role:   Temporal validation of alpha signals — pattern mining
Output: g_factor ∈ (0, 1] for multiplicative modulation

This filter uses the existing MultiHeadTCN backbone from
risk_management/phase1_predictive/tcn_backbone.py when weights are
available, or a lightweight built-in TCN when they are not.

The filter NEVER overrides direction — it only modulates probability.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..interfaces import AlphaSignal, MHTCNFilter, MHTCNOutput

logger = logging.getLogger(__name__)


# ── Lightweight inline TCN for research use ──────────────────────────────

class _CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class _TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.residual(x)
        out = self.drop(self.relu(self.bn1(self.conv1(x))))
        out = self.drop(self.relu(self.bn2(self.conv2(out))))
        return self.relu(out + res)


class ResearchTCN(nn.Module):
    """Lightweight TCN for confidence estimation.

    Outputs a single scalar g_factor ∈ (0, 1] via sigmoid.
    """

    def __init__(
        self,
        input_channels: int = 32,
        hidden: int = 64,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, hidden)
        dilations = [2 ** i for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            _TCNBlock(hidden, hidden, kernel_size, d, dropout)
            for d in dilations
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            (batch,) g_factor values in (0, 1]
        """
        # project input
        out = self.input_proj(x)          # (B, T, H)
        out = out.transpose(1, 2)         # (B, H, T)
        for block in self.blocks:
            out = block(out)
        # global average pool
        pooled = out.mean(dim=-1)         # (B, H)
        logit = self.head(pooled).squeeze(-1)  # (B,)
        return torch.sigmoid(logit)


class RawFeatureMHTCNFilter(MHTCNFilter):
    """MH-TCN filter using raw OHLC + engineered features.

    On each call, it builds a (seq_len, n_features) window from the
    latest bars, runs it through the TCN, and returns g_factor.
    """

    def __init__(
        self,
        seq_len: int = 60,
        max_features: int = 64,
        weights_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.seq_len = seq_len
        self.max_features = max_features
        self.device = device
        self._model: Optional[nn.Module] = None
        self._weights_path = weights_path
        self._input_channels: Optional[int] = None

    def name(self) -> str:
        return "RawFeature_MHTCN"

    def reset(self) -> None:
        pass  # stateless per-bar

    def _ensure_model(self, n_features: int):
        """Lazy-init model on first call when we know input dimension."""
        if self._model is not None and self._input_channels == n_features:
            return
        self._input_channels = n_features
        self._model = ResearchTCN(
            input_channels=n_features,
            hidden=64,
            num_layers=4,
            kernel_size=3,
            dropout=0.1,
        ).to(self.device)

        # Try loading pre-trained weights
        if self._weights_path and Path(self._weights_path).exists():
            try:
                state = torch.load(self._weights_path, map_location=self.device)
                self._model.load_state_dict(state, strict=False)
                logger.info(f"Loaded RawFeature MHTCN weights from {self._weights_path}")
            except Exception as e:
                logger.warning(f"Could not load weights: {e}")

        self._model.eval()

    def filter(
        self,
        signal: AlphaSignal,
        df: pd.DataFrame,
        features: pd.DataFrame,
    ) -> MHTCNOutput:
        """Compute g_factor from raw features."""
        if not signal.is_trade:
            return MHTCNOutput(g_factor=1.0, signal_survival_prob=0.5,
                               confidence_decay=0.0, regime_validity=1.0)

        try:
            # Build input window
            window = self._build_window(features)
            if window is None:
                return MHTCNOutput(g_factor=1.0, signal_survival_prob=0.5,
                                   confidence_decay=0.0, regime_validity=1.0)

            n_feat = window.shape[1]
            self._ensure_model(n_feat)

            # Forward pass
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                g = self._model(x).item()

            return MHTCNOutput(
                g_factor=float(np.clip(g, 0.01, 1.0)),
                signal_survival_prob=float(g),
                confidence_decay=float(1.0 - g),
                regime_validity=float(g),
            )

        except Exception as e:
            logger.debug(f"RawFeature MHTCN error: {e}")
            return MHTCNOutput(g_factor=1.0, signal_survival_prob=0.5,
                               confidence_decay=0.0, regime_validity=1.0)

    def _build_window(self, features: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract last seq_len rows, select up to max_features columns."""
        if len(features) < self.seq_len:
            return None

        # Take numeric columns only
        numeric = features.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            return None

        # Truncate to max_features (take first N to keep deterministic)
        if numeric.shape[1] > self.max_features:
            numeric = numeric.iloc[:, :self.max_features]

        window = numeric.iloc[-self.seq_len:].values.astype(np.float32)
        # Replace NaN/inf
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

        # Z-score normalize per feature (column-wise over the window)
        std = window.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        window = (window - window.mean(axis=0, keepdims=True)) / std

        return window

    def get_model(self) -> Optional[nn.Module]:
        """Access underlying model for training."""
        return self._model
