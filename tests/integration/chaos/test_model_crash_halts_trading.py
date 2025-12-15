import pytest
from unittest.mock import Mock

from trading.live_trading_bot import LiveTradingBot, BotConfig


@pytest.mark.integration
def test_model_crash_does_not_execute_orders(mt5_executor):
    strategy = Mock()
    strategy.initialize.return_value = True
    strategy.evaluate.side_effect = RuntimeError("model crash")

    bot = LiveTradingBot(
        config=BotConfig(check_interval_seconds=1, dry_run=False, log_trades=False),
        data_provider=Mock(),
        executor=mt5_executor,
        strategy=strategy,
    )
    bot.initialize(starting_balance=10000.0)

    bot._is_market_open = lambda *_args, **_kwargs: True
    bot._can_open_new_trade = lambda *_args, **_kwargs: True

    def _wait_and_stop(_seconds):
        bot._stop_event.set()

    bot._wait = _wait_and_stop

    bot.run()

    assert strategy.execute.call_count == 0
