import pytest
import numpy as np

from strategies.neural_hybrid import NeuralHybridStrategy, StrategyConfig


@pytest.mark.integration
def test_feature_window_correctness(ohlcv_df):
    cfg = StrategyConfig(sequence_length=60, use_vision=False, use_yolo=False)
    strat = NeuralHybridStrategy(config=cfg, data_provider=None, executor=None)
    strat.predictor = object()

    features = strat._prepare_features(ohlcv_df)

    assert isinstance(features, np.ndarray)
    assert features.shape == (cfg.sequence_length, 5)
    assert not np.isnan(features).any()
