import pytest
from datetime import datetime

from trading.position_coordinator import PositionCoordinator, TrackedPosition
from trading.style_config import OrchestratorConfig, TradingStyle


@pytest.mark.integration
def test_trailing_only_tightens_risk_buy():
    config = OrchestratorConfig()
    coord = PositionCoordinator(config)
    coord.initialize(balance=10000.0)

    pos = TrackedPosition(
        ticket=1,
        style=TradingStyle.INTRADAY,
        symbol="EURUSD",
        direction="BUY",
        volume=0.1,
        entry_price=1.1000,
        entry_time=datetime.utcnow(),
        stop_loss=1.0950,
        take_profit=1.1100,
        magic_number=123,
        current_price=1.1080,
        unrealized_pnl=0.0,
        highest_price=1.1090,
        lowest_price=0.0,
        modified_sl=True,
    )

    coord.register_position(pos)

    updates = coord.get_positions_needing_sl_update()
    assert updates
    new_sl = updates[0][1]
    assert new_sl > pos.stop_loss


@pytest.mark.integration
def test_trailing_only_tightens_risk_sell():
    config = OrchestratorConfig()
    coord = PositionCoordinator(config)
    coord.initialize(balance=10000.0)

    pos = TrackedPosition(
        ticket=2,
        style=TradingStyle.INTRADAY,
        symbol="EURUSD",
        direction="SELL",
        volume=0.1,
        entry_price=1.1000,
        entry_time=datetime.utcnow(),
        stop_loss=1.1050,
        take_profit=1.0900,
        magic_number=123,
        current_price=1.0920,
        unrealized_pnl=0.0,
        highest_price=0.0,
        lowest_price=1.0910,
        modified_sl=True,
    )

    coord.register_position(pos)

    updates = coord.get_positions_needing_sl_update()
    assert updates
    new_sl = updates[0][1]
    assert new_sl < pos.stop_loss
