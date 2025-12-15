"""Unit tests for `risk_management/phase3_filtering/triple_barrier.py`.

Test Summary:
    Verifies triple-barrier outcome assignment (WIN/LOSS/TIMEOUT) and dataset wiring.

Test Breakdown:
    - TripleBarrierLabeler
        - BUY trade hits TP -> WIN
        - BUY trade hits SL -> LOSS
        - time barrier without sufficient move -> TIMEOUT
    - TripleBarrierDataset
        - class weights sum and keys
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from risk_management.phase3_filtering.triple_barrier import (
    BarrierOutcome,
    TripleBarrierConfig,
    TripleBarrierDataset,
    TripleBarrierLabeler,
)


def _make_price_df(close: np.ndarray) -> pd.DataFrame:
    # Keep high/low slightly around close.
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
        }
    )


class TestTripleBarrierLabeler(unittest.TestCase):
    def test_buy_hits_tp_is_win(self):
        close = np.linspace(1.0, 1.02, 50)
        prices = _make_price_df(close)

        entry_signals = np.zeros(len(prices), dtype=bool)
        entry_signals[10] = True
        directions = np.zeros(len(prices), dtype=int)
        directions[10] = 1

        # Set tp very close so it is hit quickly.
        sl_levels = np.full(len(prices), 0.99)
        tp_levels = np.full(len(prices), 1.001)

        labeler = TripleBarrierLabeler(TripleBarrierConfig(use_dynamic_barriers=True))
        labels, details = labeler.generate_labels(
            prices=prices,
            entry_signals=entry_signals,
            directions=directions,
            sl_levels=sl_levels,
            tp_levels=tp_levels,
            profile="INTRADAY",
        )

        self.assertEqual(labels[10], BarrierOutcome.WIN.value)
        self.assertEqual(details[0].outcome, BarrierOutcome.WIN)

    def test_buy_hits_sl_is_loss(self):
        close = np.linspace(1.0, 0.98, 50)
        prices = _make_price_df(close)

        entry_signals = np.zeros(len(prices), dtype=bool)
        entry_signals[10] = True
        directions = np.zeros(len(prices), dtype=int)
        directions[10] = 1

        sl_levels = np.full(len(prices), 0.999)  # close enough to hit quickly
        tp_levels = np.full(len(prices), 1.10)

        labeler = TripleBarrierLabeler(TripleBarrierConfig(use_dynamic_barriers=True))
        labels, details = labeler.generate_labels(
            prices=prices,
            entry_signals=entry_signals,
            directions=directions,
            sl_levels=sl_levels,
            tp_levels=tp_levels,
            profile="INTRADAY",
        )

        self.assertEqual(labels[10], BarrierOutcome.LOSS.value)
        self.assertEqual(details[0].outcome, BarrierOutcome.LOSS)

    def test_timeout_yields_timeout_when_return_small(self):
        close = np.full(100, 1.0)
        prices = _make_price_df(close)

        entry_signals = np.zeros(len(prices), dtype=bool)
        entry_signals[10] = True
        directions = np.zeros(len(prices), dtype=int)
        directions[10] = 1

        sl_levels = np.full(len(prices), 0.99)
        tp_levels = np.full(len(prices), 1.01)

        cfg = TripleBarrierConfig(use_dynamic_barriers=True)
        cfg.vertical_barrier_periods = 5
        cfg.min_return_threshold = 0.0

        labeler = TripleBarrierLabeler(cfg)
        labels, details = labeler.generate_labels(
            prices=prices,
            entry_signals=entry_signals,
            directions=directions,
            sl_levels=sl_levels,
            tp_levels=tp_levels,
            profile="INTRADAY",
        )

        self.assertEqual(labels[10], BarrierOutcome.TIMEOUT.value)
        self.assertEqual(details[0].barrier_hit, "time")


class TestTripleBarrierDataset(unittest.TestCase):
    def test_class_weights_keys(self):
        labeler = TripleBarrierLabeler()
        features = np.random.randn(200, 10)
        prices = _make_price_df(np.linspace(1.0, 1.01, 200))
        ds = TripleBarrierDataset(features=features, prices=prices, labeler=labeler, sequence_length=10)

        y = np.array([1, 1, 0, -1, -1, -1])
        weights = ds.get_class_weights(y)
        self.assertTrue(set(weights.keys()).issubset({-1, 0, 1}))


if __name__ == "__main__":
    unittest.main()
