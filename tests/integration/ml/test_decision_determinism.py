import pytest
import numpy as np
from datetime import datetime


@pytest.mark.integration
def test_decision_engine_determinism(decision_engine, ohlcv_df):
    predictions = {
        "direction_probs": np.array([0.05, 0.05, 0.90]),
        "volatility": 0.001,
        "quantiles": np.array([-0.0005, -0.0002, 0.0, 0.0004, 0.0008]),
        "features": None,
    }

    entry = float(ohlcv_df["close"].iloc[-1])
    t0 = datetime(2020, 1, 1)

    d1 = decision_engine.evaluate(
        predictions=predictions,
        entry_price=entry,
        pair="EURUSD",
        account_balance=10000.0,
        market_data=ohlcv_df,
        current_spread=1.0,
        current_time=t0,
    )
    d2 = decision_engine.evaluate(
        predictions=predictions,
        entry_price=entry,
        pair="EURUSD",
        account_balance=10000.0,
        market_data=ohlcv_df,
        current_spread=1.0,
        current_time=t0,
    )

    assert d1.to_dict() == d2.to_dict()
