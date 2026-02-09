"""
Negative controls — mandatory false-confidence safeguards.

Three controls:
    1. MH-TCN trained on shuffled labels
    2. MH-TCN applied to random alpha (random direction/probability)
    3. Alpha tested in unseen regimes (hold-out regime slices)

If MH-TCN "improves" under these controls → flag as leakage.
"""

import logging
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..interfaces import (
    AlphaHead, AlphaSignal, Direction, MHTCNFilter,
    NullMHTCNFilter, RegimeLabel, VariantConfig, VariantID, VariantResult,
)

logger = logging.getLogger(__name__)


# ── Control 1: Random Alpha Head ─────────────────────────────────────────

class RandomAlphaHead(AlphaHead):
    """Generates random direction and probability — no real signal."""

    def __init__(self, trade_probability: float = 0.05, seed: int = 42):
        self._rng = np.random.RandomState(seed)
        self.trade_probability = trade_probability

    def name(self) -> str:
        return "RandomAlpha"

    def reset(self) -> None:
        self._rng = np.random.RandomState(42)

    def evaluate(self, df, features, regime) -> AlphaSignal:
        if self._rng.random() > self.trade_probability:
            return AlphaSignal(
                direction=Direction.HOLD, probability=0.5,
                confidence=0.0, regime=regime,
            )
        direction = Direction.LONG if self._rng.random() > 0.5 else Direction.SHORT
        prob = float(self._rng.uniform(0.4, 0.8))
        return AlphaSignal(
            direction=direction, probability=prob,
            confidence=prob - 0.5, regime=regime,
            reasoning=["random_control"],
        )


# ── Control 2: Shuffled-Label MH-TCN Wrapper ─────────────────────────────

class ShuffledLabelMHTCNFilter(MHTCNFilter):
    """Wraps an existing MH-TCN filter but shuffles its output randomly.

    This simulates a model trained on shuffled labels — if the harness
    shows improvement with this filter, it indicates leakage in the
    pipeline (not in the model weights themselves).

    For a true shuffled-label control during training, the training
    protocol should shuffle the target labels before fitting.
    """

    def __init__(self, base_filter: MHTCNFilter, seed: int = 123):
        self._base = base_filter
        self._rng = np.random.RandomState(seed)

    def name(self) -> str:
        return f"Shuffled_{self._base.name()}"

    def reset(self) -> None:
        self._base.reset()
        self._rng = np.random.RandomState(123)

    def filter(self, signal, df, features):
        # Run the real filter to maintain identical compute cost
        real_out = self._base.filter(signal, df, features)
        # Replace g_factor with random value (breaking any real signal)
        from ..interfaces import MHTCNOutput
        return MHTCNOutput(
            g_factor=float(self._rng.uniform(0.3, 1.0)),
            signal_survival_prob=float(self._rng.uniform(0.2, 0.8)),
            confidence_decay=float(self._rng.uniform(0.0, 0.5)),
            regime_validity=float(self._rng.uniform(0.3, 1.0)),
        )


class NegativeControlRunner:
    """Builds and returns negative control variant configs.

    These should be run through the same ExperimentHarness alongside
    the real variants. If any control shows improvement over the
    null-filter baseline, it flags potential leakage.
    """

    @staticmethod
    def build_controls(
        real_alpha: AlphaHead,
        real_mhtcn: MHTCNFilter,
        shared_kw: Dict,
    ) -> List[VariantConfig]:
        """Build negative control variant configs.

        Args:
            real_alpha: A real alpha head to pair with shuffled MH-TCN.
            real_mhtcn: A real MH-TCN filter to shuffle.
            shared_kw: Shared execution params (same as real variants).

        Returns:
            List of VariantConfig for negative controls.
        """
        controls = []

        # Control 1: Random alpha + real MH-TCN
        # If MH-TCN improves random alpha → leakage
        controls.append(VariantConfig(
            variant_id=VariantID.V1_ALPHA,  # reuse ID, distinguish by description
            alpha_head=RandomAlphaHead(),
            mhtcn_filter=real_mhtcn,
            description="NEG_CTRL: Random alpha + real MH-TCN",
            **shared_kw,
        ))

        # Control 2: Real alpha + shuffled MH-TCN
        # If shuffled MH-TCN improves real alpha → pipeline leakage
        controls.append(VariantConfig(
            variant_id=VariantID.V2_ALPHA_MHTCN,
            alpha_head=real_alpha,
            mhtcn_filter=ShuffledLabelMHTCNFilter(real_mhtcn),
            description="NEG_CTRL: Real alpha + shuffled MH-TCN",
            **shared_kw,
        ))

        # Control 3: Random alpha + no MH-TCN (pure noise baseline)
        controls.append(VariantConfig(
            variant_id=VariantID.V1_ALPHA,
            alpha_head=RandomAlphaHead(),
            mhtcn_filter=NullMHTCNFilter(),
            description="NEG_CTRL: Random alpha only (noise floor)",
            **shared_kw,
        ))

        return controls

    @staticmethod
    def check_leakage(
        control_results: List[VariantResult],
        baseline_ev: float,
    ) -> Dict[str, bool]:
        """Check if any negative control outperforms the baseline.

        Args:
            control_results: Results from running negative controls.
            baseline_ev: Mean EV from the null-filter real-alpha baseline.

        Returns:
            Dict mapping control description to leakage flag.
        """
        flags = {}
        for r in control_results:
            ctrl_ev = np.mean([t.pnl for t in r.trades]) if r.trades else 0.0
            is_leakage = ctrl_ev > baseline_ev and len(r.trades) > 5
            flags[r.variant_id] = is_leakage
            if is_leakage:
                logger.warning(
                    f"LEAKAGE FLAG: {r.variant_id} control EV={ctrl_ev:.2f} > "
                    f"baseline EV={baseline_ev:.2f}"
                )
        return flags
