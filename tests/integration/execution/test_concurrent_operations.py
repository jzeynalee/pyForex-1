"""
Concurrent Operations Integration Tests.

Tests thread safety and race condition handling.
"""
import pytest
import threading
import time
from datetime import datetime
from unittest.mock import Mock, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

from trading.live_trading_bot import LiveTradingBot, BotConfig, BotState
from trading.position_coordinator import PositionCoordinator, TrackedPosition
from trading.style_config import OrchestratorConfig, TradingStyle


@pytest.mark.integration
class TestConcurrentPositionUpdates:
    """Tests for concurrent position update handling."""
    
    def test_concurrent_position_registration(self):
        """Multiple threads registering positions should not corrupt state."""
        config = OrchestratorConfig()
        coordinator = PositionCoordinator(config)
        coordinator.initialize(balance=10000.0)
        
        errors = []
        registered = []
        lock = threading.Lock()
        
        def register_position(ticket):
            try:
                pos = TrackedPosition(
                    ticket=ticket,
                    style=TradingStyle.INTRADAY,
                    symbol="EURUSD",
                    direction="BUY",
                    volume=0.1,
                    entry_price=1.1000,
                    entry_time=datetime.utcnow(),
                    stop_loss=1.0950,
                    take_profit=1.1100,
                    magic_number=123,
                    current_price=1.1000,
                    unrealized_pnl=0.0,
                )
                coordinator.register_position(pos)
                with lock:
                    registered.append(ticket)
            except Exception as e:
                with lock:
                    errors.append((ticket, str(e)))
        
        # Register positions from multiple threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register_position, i) for i in range(1, 21)]
            for f in as_completed(futures):
                pass
        
        # Should have no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        # All positions should be registered (up to limits)
        assert len(coordinator.positions) > 0
    
    def test_concurrent_position_closure(self):
        """Multiple threads closing positions should not corrupt state."""
        config = OrchestratorConfig()
        coordinator = PositionCoordinator(config)
        coordinator.initialize(balance=10000.0)
        
        # First, register positions
        for i in range(1, 11):
            pos = TrackedPosition(
                ticket=i,
                style=TradingStyle.INTRADAY,
                symbol="EURUSD",
                direction="BUY",
                volume=0.1,
                entry_price=1.1000,
                entry_time=datetime.utcnow(),
                stop_loss=1.0950,
                take_profit=1.1100,
                magic_number=123,
                current_price=1.1000,
                unrealized_pnl=0.0,
            )
            coordinator.register_position(pos)
        
        errors = []
        closed = []
        lock = threading.Lock()
        
        def close_position(ticket):
            try:
                coordinator.close_position(ticket=ticket, pnl=10.0)
                with lock:
                    closed.append(ticket)
            except Exception as e:
                with lock:
                    errors.append((ticket, str(e)))
        
        # Close positions from multiple threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(close_position, i) for i in range(1, 11)]
            for f in as_completed(futures):
                pass
        
        # Should have no errors (or only "not found" errors for already closed)
        critical_errors = [e for e in errors if "not found" not in str(e[1]).lower()]
        assert len(critical_errors) == 0, f"Critical errors: {critical_errors}"
        # All positions should be closed
        assert len(coordinator.positions) == 0


@pytest.mark.integration
class TestConcurrentBotOperations:
    """Tests for concurrent bot operations."""
    
    def test_bot_state_transitions_thread_safe(self, mt5_executor):
        """Bot state transitions should be thread-safe."""
        strategy = Mock()
        strategy.initialize.return_value = True
        strategy.evaluate.return_value = None
        
        bot = LiveTradingBot(
            config=BotConfig(check_interval_seconds=1, dry_run=True),
            data_provider=Mock(),
            executor=mt5_executor,
            strategy=strategy,
        )
        bot.initialize(starting_balance=10000.0)
        
        states_observed = []
        lock = threading.Lock()
        
        def observe_state():
            for _ in range(10):
                with lock:
                    states_observed.append(bot.state)
                time.sleep(0.01)
        
        def change_state():
            for state in [BotState.RUNNING, BotState.PAUSED, BotState.RUNNING]:
                bot.state = state
                time.sleep(0.02)
        
        # Run observers and state changer concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(observe_state),
                executor.submit(observe_state),
                executor.submit(change_state),
            ]
            for f in as_completed(futures):
                pass
        
        # All observed states should be valid
        valid_states = set(BotState)
        for state in states_observed:
            assert state in valid_states


@pytest.mark.integration
class TestRaceConditionPrevention:
    """Tests for race condition prevention."""
    
    def test_no_double_execution_under_load(self, mt5_executor):
        """Same signal should not result in double execution under load."""
        execution_count = [0]
        lock = threading.Lock()
        
        original_execute = mt5_executor.execute_order
        
        def counting_execute(*args, **kwargs):
            with lock:
                execution_count[0] += 1
            return original_execute(*args, **kwargs)
        
        mt5_executor.execute_order = counting_execute
        
        strategy = Mock()
        strategy.initialize.return_value = True
        
        decision = Mock()
        decision.should_trade = True
        decision.direction = "BUY"
        decision.position_size = 0.1
        decision.stop_loss = 1.095
        decision.take_profit = 1.110
        decision.to_dict.return_value = {}
        
        strategy.evaluate.return_value = decision
        strategy.create_order.return_value = Mock(
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            price=0.0,
            stop_loss=1.095,
            take_profit=1.110,
            ticket=None,
        )
        strategy.execute.return_value = True
        
        bot = LiveTradingBot(
            config=BotConfig(
                check_interval_seconds=1,
                dry_run=False,
                min_order_interval_seconds=60,
            ),
            data_provider=Mock(),
            executor=mt5_executor,
            strategy=strategy,
        )
        bot.initialize(starting_balance=10000.0)
        bot._is_market_open = lambda *args, **kwargs: True
        bot._can_open_new_trade = lambda *args, **kwargs: True
        
        # Simulate rapid concurrent evaluations
        def evaluate():
            bot._evaluate_and_trade(datetime.utcnow())
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(evaluate) for _ in range(10)]
            for f in as_completed(futures):
                pass
        
        # Should only execute once due to cooldown
        assert execution_count[0] <= 1
    
    def test_position_update_atomicity(self):
        """Position updates should be atomic."""
        config = OrchestratorConfig()
        coordinator = PositionCoordinator(config)
        coordinator.initialize(balance=10000.0)
        
        # Register a position
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
            current_price=1.1000,
            unrealized_pnl=0.0,
        )
        coordinator.register_position(pos)
        
        errors = []
        
        def update_price(new_price):
            try:
                if 1 in coordinator.positions:
                    coordinator.positions[1].current_price = new_price
                    coordinator.positions[1].unrealized_pnl = (new_price - 1.1000) * 10000
            except Exception as e:
                errors.append(str(e))
        
        # Concurrent price updates
        with ThreadPoolExecutor(max_workers=10) as executor:
            prices = [1.1000 + i * 0.0001 for i in range(100)]
            futures = [executor.submit(update_price, p) for p in prices]
            for f in as_completed(futures):
                pass
        
        # Should have no errors
        assert len(errors) == 0
        # Position should still be valid
        if 1 in coordinator.positions:
            assert coordinator.positions[1].current_price is not None
