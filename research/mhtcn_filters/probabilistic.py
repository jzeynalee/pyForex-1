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
    """MH-TCN filter using probability-space + price-derived inputs.

    Maintains a rolling buffer of per-bar feature vectors so that
    each call can build a (seq_len, n_channels) window.

    Channel layout (14 channels):
        0: P_trend
        1: P_momentum
        2: P_oscillator
        3: P_volatility
        4: P_volume
        5: P_structure
        6: P_alpha_final
        7: regime_scalar
        8: return_1bar   (close-to-close pct change, clipped)
        9: return_5bar
       10: return_10bar
       11: norm_atr       (ATR / close, volatility proxy)
       12: rsi_norm       (RSI / 100, bounded [0,1])
       13: momentum_sign  (signed momentum: +1/-1 scaled by strength)
    """

    N_CHANNELS = 14

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
        self._precomputed_rows: Optional[np.ndarray] = None

    def name(self) -> str:
        return "Probabilistic_MHTCN"

    def reset(self) -> None:
        self._buffer.clear()
        self._precomputed_rows = None

    def _ensure_model(self):
        if self._model is not None:
            return
        self._model = ProbabilisticTCN(
            input_channels=self.N_CHANNELS,
            hidden=48,
            num_layers=3,
            kernel_size=3,
            dropout=0.15,
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
        """Compute g_factor from probability + price-derived sequence.

        Always appends the current bar's features to the buffer,
        even for HOLD signals, so the temporal context stays current.
        """
        # Build current-bar feature vector and append to buffer
        row = self._build_prob_row(signal, df)
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

    def precompute_buffer_rows(
        self,
        alpha_head,
        regimes: pd.Series,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> None:
        """Vectorize the entire (n_bars, N_CHANNELS) buffer matrix upfront.

        Channels 0-5: category probs from AlphaV2 precomputed scores
        Channel 6:    alpha probability (max(p_bull, p_bear))
        Channel 7:    regime scalar
        Channels 8-13: price-derived features (returns, ATR, RSI, momentum)
        """
        n = len(closes)
        buf = np.full((n, self.N_CHANNELS), 0.5, dtype=np.float32)

        # Channels 0-5: category probabilities
        from ..feature_pipeline import ALL_CATEGORIES
        if hasattr(alpha_head, '_precomputed_cat_probs') and alpha_head._precomputed_cat_probs is not None:
            for i, cat in enumerate(ALL_CATEGORIES):
                buf[:, i] = alpha_head._precomputed_cat_probs[cat].astype(np.float32)

        # Channel 6: alpha probability = max(p_bull, p_bear)
        if hasattr(alpha_head, '_precomputed_p_bull') and alpha_head._precomputed_p_bull is not None:
            p_bull = alpha_head._precomputed_p_bull
            p_bear = alpha_head._precomputed_p_bear
            buf[:, 6] = np.maximum(p_bull, p_bear).astype(np.float32)

        # Channel 7: regime scalar
        regime_map = {
            RegimeLabel.TRENDING: 0.0,
            RegimeLabel.RANGING: 0.33,
            RegimeLabel.VOLATILE: 0.67,
            RegimeLabel.TRANSITION: 1.0,
        }
        for rl, val in regime_map.items():
            mask = (regimes == rl).values
            buf[mask, 7] = val

        # Channels 8-13: price-derived features (vectorized)
        _RET_SCALE = 0.005

        def _scale_ret(r):
            return np.clip(r / _RET_SCALE, -1.0, 1.0) * 0.5 + 0.5

        # 1-bar return (channel 8)
        ret1 = np.zeros(n, dtype=np.float64)
        ret1[1:] = (closes[1:] - closes[:-1]) / np.maximum(closes[:-1], 1e-10)
        buf[:, 8] = _scale_ret(ret1).astype(np.float32)

        # 5-bar return (channel 9)
        ret5 = np.zeros(n, dtype=np.float64)
        ret5[5:] = (closes[5:] - closes[:-5]) / np.maximum(closes[:-5], 1e-10)
        buf[:, 9] = _scale_ret(ret5).astype(np.float32)

        # 10-bar return (channel 10)
        ret10 = np.zeros(n, dtype=np.float64)
        ret10[10:] = (closes[10:] - closes[:-10]) / np.maximum(closes[:-10], 1e-10)
        buf[:, 10] = _scale_ret(ret10).astype(np.float32)

        # Normalized ATR (channel 11)
        tr = np.maximum(
            highs - lows,
            np.maximum(np.abs(highs - np.roll(closes, 1)),
                       np.abs(lows - np.roll(closes, 1)))
        )
        tr[0] = highs[0] - lows[0]
        atr14 = pd.Series(tr).rolling(14).mean().values
        norm_atr = np.clip(atr14 / np.maximum(closes, 1e-10) * 500.0, 0, 1.0)
        norm_atr = np.nan_to_num(norm_atr, nan=0.5)
        buf[:, 11] = norm_atr.astype(np.float32)

        # RSI (channel 12)
        deltas = np.diff(closes, prepend=closes[0])
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))
        avg_gain = pd.Series(gains).rolling(14).mean().values
        avg_loss = pd.Series(losses).rolling(14).mean().values
        rs = avg_gain / np.maximum(avg_loss, 1e-10)
        rsi = rs / (1.0 + rs)
        rsi = np.nan_to_num(rsi, nan=0.5)
        buf[:, 12] = rsi.astype(np.float32)

        # Momentum sign (channel 13) = 10-bar return scaled
        buf[:, 13] = buf[:, 10]  # same as 10-bar return

        self._precomputed_rows = buf
        logger.info(f"MH-TCN: pre-computed {n:,} buffer rows ({self.N_CHANNELS} channels)")

    def feed_bar(self, bar_idx: int) -> None:
        """O(1) buffer feeding from pre-computed rows."""
        if self._precomputed_rows is not None and bar_idx < len(self._precomputed_rows):
            self._buffer.append(self._precomputed_rows[bar_idx])

    def _build_prob_row(
        self, signal: AlphaSignal, df: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Convert an AlphaSignal + OHLCV into a (N_CHANNELS,) feature vector."""
        row = np.full(self.N_CHANNELS, 0.5, dtype=np.float32)

        # Category probabilities (channels 0-5)
        for i, cat in enumerate(ALL_CATEGORIES):
            if cat in signal.category_probs:
                row[i] = float(np.clip(signal.category_probs[cat], 0.0, 1.0))

        # Alpha final probability (channel 6)
        row[6] = float(np.clip(signal.probability, 0.0, 1.0))

        # Regime scalar (channel 7)
        row[7] = _REGIME_TO_FLOAT.get(signal.regime, 0.5)

        # Price-derived channels (8-13) — scaled to [0, 1] to match training
        _RET_SCALE = 0.005  # 50 pips full scale for EURUSD

        def _sr(r):
            """Scale return to [0, 1]: -50 pips → 0, 0 → 0.5, +50 pips → 1."""
            return float(np.clip(r / _RET_SCALE, -1.0, 1.0) * 0.5 + 0.5)

        if df is not None and len(df) >= 2:
            close_col = "close" if "close" in df.columns else "Close"
            high_col = "high" if "high" in df.columns else "High"
            low_col = "low" if "low" in df.columns else "Low"
            closes = df[close_col].values
            c = closes[-1]
            if c > 1e-10:
                # 1-bar return
                row[8] = _sr((c - closes[-2]) / closes[-2]) if len(closes) >= 2 else 0.5
                # 5-bar return
                idx5 = min(6, len(closes))
                row[9] = _sr((c - closes[-idx5]) / closes[-idx5])
                # 10-bar return
                idx10 = min(11, len(closes))
                row[10] = _sr((c - closes[-idx10]) / closes[-idx10])
                # Normalized ATR (14-bar) — scaled by 500
                if len(df) >= 14:
                    highs = df[high_col].values[-14:]
                    lows = df[low_col].values[-14:]
                    prev_c = closes[-15:-1] if len(closes) >= 15 else closes[-14:]
                    tr = np.maximum(highs - lows, np.maximum(
                        np.abs(highs - prev_c[:len(highs)]),
                        np.abs(lows - prev_c[:len(lows)]))
                    )
                    row[11] = float(np.clip(tr.mean() / c * 500.0, 0, 1.0))
                # RSI (14-bar, normalized to [0,1])
                if len(closes) >= 15:
                    deltas = np.diff(closes[-15:])
                    gains = np.maximum(deltas, 0).mean()
                    losses = np.abs(np.minimum(deltas, 0)).mean()
                    rs = gains / max(losses, 1e-10)
                    row[12] = float(rs / (1.0 + rs))
                # Signed momentum (10-bar, same return scale)
                if len(closes) >= 11:
                    mom = (c - closes[-11]) / closes[-11]
                    row[13] = _sr(mom)

        return row

    def get_model(self) -> Optional[ProbabilisticTCN]:
        """Access underlying model for training."""
        return self._model
