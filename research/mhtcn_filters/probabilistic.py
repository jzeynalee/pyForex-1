"""
Probabilistic MH-TCN Filter.

Input:  Category-level probabilities + final alpha probability + regime/risk
        flags, windowed over time (e.g. 64 timesteps x 8 features).
Role:   Signal survival probability, confidence decay/amplification,
        regime-conditional validity gate.
Output: g_factor in (0, 1] for multiplicative modulation.

This is the preferred MH-TCN mode -- it operates in probability space
rather than raw feature space, which provides cleaner separation of
concerns and better interpretability.

Training labels must be leak-free: e.g. "did signal persist N bars?"
NOT realized PnL.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..interfaces import AlphaSignal, MHTCNFilter, MHTCNOutput, RegimeLabel
from ..feature_pipeline import ALL_CATEGORIES

logger = logging.getLogger(__name__)


# -- Probability-space TCN ------------------------------------------------

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


class ProbabilisticTCN(nn.Module):
    """TCN operating on probability sequences.

    Input channels:
        - P_trend, P_momentum, P_oscillator, P_volatility, P_volume, P_structure
        - P_alpha_final
        - regime_id (scalar encoding)
    Total: 8 channels over T timesteps.

    Output: g_factor in (0, 1] via sigmoid, plus survival and validity heads.
    """

    def __init__(
        self,
        input_channels: int = 8,
        hidden: int = 32,
        num_layers: int = 3,
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
        # Multi-head output
        self.survival_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.validity_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, channels)
        Returns:
            dict with 'g_factor', 'survival', 'validity' -- all (batch,)
        """
        out = self.input_proj(x)          # (B, T, H)
        out = out.transpose(1, 2)         # (B, H, T)
        for block in self.blocks:
            out = block(out)
        pooled = out.mean(dim=-1)         # (B, H)

        survival = torch.sigmoid(self.survival_head(pooled).squeeze(-1))
        validity = torch.sigmoid(self.validity_head(pooled).squeeze(-1))
        g_factor = survival * validity    # combined gate

        return {
            "g_factor": g_factor,
            "survival": survival,
            "validity": validity,
        }


# -- Regime encoding -------------------------------------------------------

_REGIME_TO_FLOAT = {
    RegimeLabel.TRENDING: 0.0,
    RegimeLabel.RANGING: 0.33,
    RegimeLabel.VOLATILE: 0.67,
    RegimeLabel.TRANSITION: 1.0,
}


class ProbabilisticMHTCNFilter(MHTCNFilter):
    """MH-TCN filter using probability-space inputs.

    Maintains a rolling buffer of per-bar category probabilities so that
    each call can build a (seq_len, n_channels) window.

    Channel layout (8 channels):
        0: P_trend
        1: P_momentum
        2: P_oscillator
        3: P_volatility
        4: P_volume
        5: P_structure
        6: P_alpha_final
        7: regime_scalar
    """

    N_CHANNELS = 8

    def __init__(
        self,
        seq_len: int = 64,
        weights_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.seq_len = seq_len
        self.device = device
        self._weights_path = weights_path

        # Rolling buffer: each entry is a (N_CHANNELS,) vector
        self._buffer: deque = deque(maxlen=seq_len)
        self._model: Optional[ProbabilisticTCN] = None

    def name(self) -> str:
        return "Probabilistic_MHTCN"

    def reset(self) -> None:
        self._buffer.clear()

    def _ensure_model(self):
        if self._model is not None:
            return
        self._model = ProbabilisticTCN(
            input_channels=self.N_CHANNELS,
            hidden=32,
            num_layers=3,
            kernel_size=3,
            dropout=0.1,
        ).to(self.device)

        if self._weights_path and Path(self._weights_path).exists():
            try:
                state = torch.load(self._weights_path, map_location=self.device)
                self._model.load_state_dict(state, strict=False)
                logger.info(f"Loaded Probabilistic MHTCN weights from {self._weights_path}")
            except Exception as e:
                logger.warning(f"Could not load weights: {e}")

        self._model.eval()

    def filter(
        self,
        signal: AlphaSignal,
        df: pd.DataFrame,
        features: pd.DataFrame,
    ) -> MHTCNOutput:
        """Compute g_factor from probability sequence.

        Always appends the current bar's probabilities to the buffer,
        even for HOLD signals, so the temporal context stays current.
        """
        # Build current-bar probability vector and append to buffer
        row = self._build_prob_row(signal)
        self._buffer.append(row)

        # If signal is HOLD, no need to run the model
        if not signal.is_trade:
            return MHTCNOutput(
                g_factor=1.0,
                signal_survival_prob=0.5,
                confidence_decay=0.0,
                regime_validity=1.0,
            )

        # Need enough history to fill the window
        if len(self._buffer) < self.seq_len:
            return MHTCNOutput(
                g_factor=1.0,
                signal_survival_prob=0.5,
                confidence_decay=0.0,
                regime_validity=1.0,
            )

        try:
            self._ensure_model()

            # Build (1, seq_len, N_CHANNELS) tensor
            window = np.array(list(self._buffer), dtype=np.float32)
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self._model(x)

            g = float(out["g_factor"].item())
            surv = float(out["survival"].item())
            valid = float(out["validity"].item())

            return MHTCNOutput(
                g_factor=float(np.clip(g, 0.01, 1.0)),
                signal_survival_prob=surv,
                confidence_decay=float(1.0 - surv),
                regime_validity=valid,
            )

        except Exception as e:
            logger.debug(f"Probabilistic MHTCN error: {e}")
            return MHTCNOutput(
                g_factor=1.0,
                signal_survival_prob=0.5,
                confidence_decay=0.0,
                regime_validity=1.0,
            )

    def _build_prob_row(self, signal: AlphaSignal) -> np.ndarray:
        """Convert an AlphaSignal into a (N_CHANNELS,) probability vector."""
        row = np.full(self.N_CHANNELS, 0.5, dtype=np.float32)

        # Category probabilities (channels 0-5)
        for i, cat in enumerate(ALL_CATEGORIES):
            if cat in signal.category_probs:
                row[i] = float(np.clip(signal.category_probs[cat], 0.0, 1.0))

        # Alpha final probability (channel 6)
        row[6] = float(np.clip(signal.probability, 0.0, 1.0))

        # Regime scalar (channel 7)
        row[7] = _REGIME_TO_FLOAT.get(signal.regime, 0.5)

        return row

    def get_model(self) -> Optional[ProbabilisticTCN]:
        """Access underlying model for training."""
        return self._model
