import pytest
import pandas as pd
import numpy as np

from strategies.neural_hybrid import NeuralHybridStrategy, StrategyConfig


class SpyPredictor:
    def __init__(self):
        self.called = 0

    def predict(self, *_args, **_kwargs):
        self.called += 1
        raise AssertionError("predict() should not be called when features contain NaNs")


@pytest.mark.integration
def test_no_nans_passed_to_ml(monkeypatch, ohlcv_df):
    df = ohlcv_df.copy()
    df.loc[df.index[-1], "close"] = np.nan

    data_provider = type(
        "DP",
        (),
        {
            "get_ohlcv": lambda *_args, **_kwargs: df,
            "get_spread": lambda *_args, **_kwargs: 1.0,
        },
    )()

    cfg = StrategyConfig(sequence_length=60, use_vision=False, use_yolo=False)
    strat = NeuralHybridStrategy(config=cfg, data_provider=data_provider, executor=None)
    strat._initialized = True
    strat.predictor = SpyPredictor()

    out = strat.evaluate()

    assert out is None
    assert strat.predictor.called == 0
