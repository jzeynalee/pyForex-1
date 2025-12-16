import pytest
import numpy as np


@pytest.mark.integration
def test_drawdown_disables_trading(decision_engine, ohlcv_df):
    # Push account into >10% drawdown from 10k peak
    decision_engine.record_trade_result(pnl=-1200.0, is_win=False, size=0.1)

    predictions = {
        "direction_probs": np.array([0.05, 0.05, 0.90]),
        "volatility": 0.001,
        "quantiles": np.array([-0.0005, -0.0002, 0.0, 0.0004, 0.0008]),
        "features": None,
    }

    entry = float(ohlcv_df["close"].iloc[-1])
    d = decision_engine.evaluate(
        predictions=predictions,
        entry_price=entry,
        pair="EURUSD",
        account_balance=8800.0,
        market_data=ohlcv_df,
        current_spread=1.0,
    )

    assert d.should_trade is False
    assert any("Capital protection" in r for r in d.rejection_reasons)
