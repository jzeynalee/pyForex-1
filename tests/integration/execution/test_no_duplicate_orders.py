import pytest
from datetime import datetime

from strategies.neural_hybrid import Order, OrderType
from trading.live_trading_bot import LiveTradingBot, BotConfig


class DummyDecision:
    def __init__(self):
        self.should_trade = True
        self.direction = "BUY"
        self.position_size = 0.10
        self.stop_loss = 1.095
        self.take_profit = 1.110
        self.risk_percent = 1.0
        self.risk_reward_ratio = 2.0
        self.direction_confidence = 0.9
        self.meta_score = 0.8
        self.protection_level = "normal"


class DummyStrategy:
    def __init__(self, executor):
        self._executor = executor

    def initialize(self):
        return True

    def evaluate(self, _current_time):
        return DummyDecision()

    def create_order(self, decision):
        return Order(
            symbol="EURUSD",
            order_type=OrderType.MARKET,
            direction=decision.direction,
            volume=decision.position_size,
            price=0.0,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            comment="test",
        )

    def execute(self, order):
        result = self._executor.execute_order(
            symbol=order.symbol,
            order_type=order.order_type.value,
            direction=order.direction,
            volume=order.volume,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            comment=order.comment,
            magic_number=order.magic_number,
        )
        if result.get("success"):
            order.ticket = str(result.get("ticket"))
            order.price = float(result.get("price") or order.price)
            return True
        return False


@pytest.mark.integration
def test_no_duplicate_orders_within_cooldown(monkeypatch, mt5_executor):
    config = BotConfig(dry_run=False, log_trades=False, min_order_interval_seconds=60)
    bot = LiveTradingBot(config=config, data_provider=None, executor=mt5_executor, strategy=DummyStrategy(mt5_executor))
    bot.initialize(starting_balance=10000.0)

    calls = []
    orig = mt5_executor.execute_order

    def wrapped(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(mt5_executor, "execute_order", wrapped)

    t = datetime(2020, 1, 1)
    bot._evaluate_and_trade(t)
    bot._evaluate_and_trade(t)

    assert len(calls) == 1
    assert bot._trades_executed == 1
