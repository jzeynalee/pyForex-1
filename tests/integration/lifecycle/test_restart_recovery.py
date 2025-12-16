import pytest
from unittest.mock import Mock

from trading.live_trading_bot import LiveTradingBot, BotConfig


@pytest.mark.integration
def test_restart_recovers_open_positions(tmp_path, mt5_executor):
    open_file = tmp_path / "open_positions.json"

    # Bot A executes a trade and persists positions
    strategy_a = Mock()
    strategy_a.initialize.return_value = True

    decision = Mock()
    decision.should_trade = True
    decision.direction = "BUY"
    decision.position_size = 0.1
    decision.stop_loss = 1.095
    decision.take_profit = 1.110
    decision.to_dict.return_value = {
        "direction_confidence": 0.9,
        "meta_score": 0.0,
        "risk_percent": 1.0,
        "risk_reward_ratio": 2.0,
        "regime": "",
    }

    strategy_a.evaluate.return_value = decision
    strategy_a.create_order.side_effect = lambda d: type(
        "O",
        (),
        {
            "symbol": "EURUSD",
            "direction": "BUY",
            "volume": 0.1,
            "price": 0.0,
            "stop_loss": 1.095,
            "take_profit": 1.110,
            "ticket": None,
        },
    )()

    def exec_side(order):
        # Use executor to open a position and set ticket/price
        res = mt5_executor.execute_order(
            symbol=order.symbol,
            order_type="MARKET",
            direction=order.direction,
            volume=order.volume,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            comment="test",
            magic_number=123,
        )
        order.ticket = str(res.get("ticket"))
        order.price = float(res.get("price") or order.price)
        return True

    strategy_a.execute.side_effect = exec_side

    bot_a = LiveTradingBot(
        config=BotConfig(
            dry_run=False,
            log_trades=False,
            persist_open_positions=True,
            open_positions_file=str(open_file),
        ),
        data_provider=Mock(),
        executor=mt5_executor,
        strategy=strategy_a,
    )
    bot_a.initialize(starting_balance=10000.0)

    bot_a._is_market_open = lambda *_args, **_kwargs: True
    bot_a._can_open_new_trade = lambda *_args, **_kwargs: True
    bot_a._evaluate_and_trade(current_time=__import__("datetime").datetime(2020, 1, 1))

    assert bot_a._open_positions

    # Bot B restarts and should recover from file and validate with broker
    strategy_b = Mock()
    strategy_b.initialize.return_value = True

    bot_b = LiveTradingBot(
        config=BotConfig(
            dry_run=False,
            log_trades=False,
            persist_open_positions=True,
            open_positions_file=str(open_file),
        ),
        data_provider=Mock(),
        executor=mt5_executor,
        strategy=strategy_b,
    )
    bot_b.initialize(starting_balance=10000.0)

    assert bot_b._open_positions


@pytest.mark.integration
def test_restart_removes_orphan_positions(tmp_path, mt5_executor):
    open_file = tmp_path / "open_positions.json"

    # Write a position that does not exist in broker state
    open_file.write_text(
        '[{"ticket":"9999","symbol":"EURUSD","direction":1,"entry_price":1.1,'
        '"entry_time":"2020-01-01T00:00:00","volume":0.1,"stop_loss":1.095,"take_profit":1.11}]'
    )

    strategy = Mock()
    strategy.initialize.return_value = True

    bot = LiveTradingBot(
        config=BotConfig(
            dry_run=False,
            log_trades=False,
            persist_open_positions=True,
            open_positions_file=str(open_file),
        ),
        data_provider=Mock(),
        executor=mt5_executor,
        strategy=strategy,
    )
    bot.initialize(starting_balance=10000.0)

    assert bot._open_positions == {}
