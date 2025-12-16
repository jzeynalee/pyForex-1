import pytest
import numpy as np

from risk_management.phase2_risk_calc.sl_tp_calculator import TradeDirection


@pytest.mark.integration
def test_sltp_invariants_buy(sltp_calculator):
    entry = 1.1000
    quantiles = np.array([-0.0006, -0.0002, 0.0, 0.0004, 0.0009])

    res = sltp_calculator.calculate(
        entry_price=entry,
        direction=TradeDirection.BUY,
        quantiles=quantiles,
        volatility=0.001,
    )

    assert res.stop_loss < entry
    assert res.take_profit > entry
    assert res.risk_reward_ratio >= sltp_calculator.config.min_risk_reward


@pytest.mark.integration
def test_sltp_invariants_sell(sltp_calculator):
    entry = 1.1000
    quantiles = np.array([-0.0006, -0.0002, 0.0, 0.0004, 0.0009])

    res = sltp_calculator.calculate(
        entry_price=entry,
        direction=TradeDirection.SELL,
        quantiles=quantiles,
        volatility=0.001,
    )

    assert res.stop_loss > entry
    assert res.take_profit < entry
    assert res.risk_reward_ratio >= sltp_calculator.config.min_risk_reward
