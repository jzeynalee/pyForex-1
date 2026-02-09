"""
Probability calibration for the research framework.

Wraps isotonic regression and Platt scaling with walk-forward updates.
Applied post-aggregation for all variants that produce probabilities.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationState:
    """Rolling buffer of (predicted_prob, actual_outcome) for online calibration."""
    predictions: List[float] = field(default_factory=list)
    outcomes: List[int] = field(default_factory=list)
    max_buffer: int = 2000

    def add(self, pred: float, outcome: int):
        self.predictions.append(pred)
        self.outcomes.append(outcome)
        if len(self.predictions) > self.max_buffer:
            self.predictions = self.predictions[-self.max_buffer:]
            self.outcomes = self.outcomes[-self.max_buffer:]

    @property
    def size(self) -> int:
        return len(self.predictions)


class ProbabilityCalibrator:
    """Walk-forward probability calibrator.

    Supports:
        - isotonic: Non-parametric, requires sklearn
        - platt: Logistic (Platt scaling)
        - passthrough: Returns raw probability (baseline)

    Calibration is updated every `update_freq` samples using the rolling
    buffer of past (prediction, outcome) pairs.
    """

    def __init__(
        self,
        method: str = "isotonic",
        min_samples: int = 100,
        update_freq: int = 50,
        buffer_size: int = 2000,
    ):
        self.method = method
        self.min_samples = min_samples
        self.update_freq = update_freq
        self.state = CalibrationState(max_buffer=buffer_size)
        self._model = None
        self._fitted = False
        self._calls_since_refit = 0

    def calibrate(self, raw_prob: float) -> float:
        """Return calibrated probability. Falls back to raw if not fitted."""
        if not self._fitted or self.method == "passthrough":
            return float(np.clip(raw_prob, 0.0, 1.0))
        try:
            p = np.array([[raw_prob]])
            calibrated = float(self._model.predict(p)[0])
            return float(np.clip(calibrated, 0.01, 0.99))
        except Exception:
            return float(np.clip(raw_prob, 0.0, 1.0))

    def update(self, predicted: float, actual_outcome: int):
        """Record a new (prediction, outcome) pair and optionally refit."""
        self.state.add(predicted, actual_outcome)
        self._calls_since_refit += 1
        if (
            self.state.size >= self.min_samples
            and self._calls_since_refit >= self.update_freq
        ):
            self._fit()
            self._calls_since_refit = 0

    def _fit(self):
        """Fit calibration model on rolling buffer."""
        X = np.array(self.state.predictions).reshape(-1, 1)
        y = np.array(self.state.outcomes)

        if len(np.unique(y)) < 2:
            return  # need both classes

        try:
            if self.method == "isotonic":
                from sklearn.isotonic import IsotonicRegression
                self._model = IsotonicRegression(
                    y_min=0.01, y_max=0.99, out_of_bounds="clip"
                )
                self._model.fit(X.ravel(), y)
            elif self.method == "platt":
                from sklearn.linear_model import LogisticRegression
                self._model = LogisticRegression(C=1.0, max_iter=200)
                self._model.fit(X, y)
            self._fitted = True
        except Exception as e:
            logger.warning(f"Calibration fit failed: {e}")

    def reset(self):
        self.state = CalibrationState(max_buffer=self.state.max_buffer)
        self._model = None
        self._fitted = False
        self._calls_since_refit = 0

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def brier_score(preds: np.ndarray, outcomes: np.ndarray) -> float:
        """Brier score: mean((p - y)^2). Lower is better."""
        return float(np.mean((preds - outcomes) ** 2))

    @staticmethod
    def reliability_curve(
        preds: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """Bin predictions and compute fraction of positives per bin."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_freqs = []
        bin_counts = []
        for i in range(n_bins):
            mask = (preds >= bins[i]) & (preds < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_freqs.append(outcomes[mask].mean())
                bin_counts.append(int(mask.sum()))
        return {
            "bin_centers": np.array(bin_centers),
            "observed_freq": np.array(bin_freqs),
            "counts": np.array(bin_counts),
        }
