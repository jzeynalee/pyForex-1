# tests/test_trading_position_coordinator.py
"""
Unit tests for trading/position_coordinator.py - Position coordination for multi-style trading.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime, timedelta
from trading.position_coordinator import (
    PositionCoordinator, TrackedPosition, StyleExposure, DailyStats
)
from trading.style_config import TradingStyle, OrchestratorConfig


@pytest.mark.unit
class TestTrackedPosition:
    """Test TrackedPosition dataclass."""

    def test_tracked_position_creation(self):
        """Test creating TrackedPosition."""
        position = TrackedPosition(
            ticket=12345,
            style=TradingStyle.INTRADAY,
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            entry_price=1.1000,
            entry_time=datetime.now(),
            stop_loss=1.0950,
            take_profit=1.1100,
            magic_number=123456
        )

        assert position.ticket == 12345
        assert position.style == TradingStyle.INTRADAY
        assert position.symbol == "EURUSD"
        assert position.direction == "BUY"
        assert position.is_open is True
        assert position.modified_sl is False


@pytest.mark.unit
class TestStyleExposure:
    """Test StyleExposure dataclass."""

    def test_style_exposure_creation(self):
        """Test creating StyleExposure."""
        exposure = StyleExposure(style=TradingStyle.SCALP)

        assert exposure.style == TradingStyle.SCALP
        assert exposure.position_count == 0
        assert exposure.total_volume == 0.0
        assert exposure.unrealized_pnl == 0.0


@pytest.mark.unit
class TestDailyStats:
    """Test DailyStats dataclass."""

    def test_daily_stats_creation(self):
        """Test creating DailyStats."""
        stats = DailyStats(date=datetime.now())

        assert stats.trades_opened == 0
        assert stats.trades_closed == 0
        assert stats.wins == 0
        assert stats.losses == 0


@pytest.mark.unit
class TestPositionCoordinator:
    """Test PositionCoordinator class."""

    @pytest.fixture
    def coordinator(self):
        """Create a PositionCoordinator instance."""
        config = OrchestratorConfig()
        return PositionCoordinator(config)

    def test_init(self, coordinator):
        """Test initialization."""
        assert len(coordinator.positions) == 0
        assert coordinator.starting_balance == 0.0
        assert TradingStyle.SCALP in coordinator.positions_by_style

    def test_initialize(self, coordinator):
        """Test initializing with balance."""
        coordinator.initialize(10000.0)

        assert coordinator.starting_balance == 10000.0
        assert coordinator.current_balance == 10000.0
        assert coordinator.peak_balance == 10000.0

    def test_register_position(self, coordinator):
        """Test registering a position."""
        coordinator.initialize(10000.0)

        position = TrackedPosition(
            ticket=12345,
            style=TradingStyle.INTRADAY,
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            entry_price=1.1000,
            entry_time=datetime.now(),
            stop_loss=1.0950,
            take_profit=1.1100,
            magic_number=123456
        )

        result = coordinator.register_position(position)

        assert result is True
        assert 12345 in coordinator.positions
        assert 12345 in coordinator.positions_by_style[TradingStyle.INTRADAY]
        assert coordinator.daily_stats[TradingStyle.INTRADAY].trades_opened == 1

    def test_register_duplicate_position(self, coordinator):
        """Test registering duplicate position."""
        coordinator.initialize(10000.0)

        position = TrackedPosition(
            ticket=12345,
            style=TradingStyle.INTRADAY,
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            entry_price=1.1000,
            entry_time=datetime.now(),
            stop_loss=1.0950,
            take_profit=1.1100,
            magic_number=123456
        )

        coordinator.register_position(position)
        result = coordinator.register_position(position)  # Duplicate

        assert result is False

    def test_close_position(self, coordinator):
        """Test closing a position."""
        coordinator.initialize(10000.0)

        position = TrackedPosition(
            ticket=12345,
            style=TradingStyle.INTRADAY,
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            entry_price=1.1000,
            entry_time=datetime.now(),
            stop_loss=1.0950,
            take_profit=1.1100,
            magic_number=123456
        )

        coordinator.register_position(position)
        closed = coordinator.close_position(12345, exit_price=1.1050, pnl=50.0)

        assert closed is not None
        assert 12345 not in coordinator.positions
        assert 12345 not in coordinator.positions_by_style[TradingStyle.INTRADAY]
        assert coordinator.daily_stats[TradingStyle.INTRADAY].trades_closed == 1
        assert coordinator.daily_stats[TradingStyle.INTRADAY].wins == 1
        assert coordinator.current_balance == 10050.0

    def test_get_positions_by_style(self, coordinator):
        """Test getting positions by style."""
        coordinator.initialize(10000.0)

        position1 = TrackedPosition(
            ticket=12345,
            style=TradingStyle.INTRADAY,
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            entry_price=1.1000,
            entry_time=datetime.now(),
            stop_loss=1.0950,
            take_profit=1.1100,
            magic_number=123456
        )

        position2 = TrackedPosition(
            ticket=12346,
            style=TradingStyle.SCALP,
            symbol="EURUSD",
            direction="SELL",
            volume=0.05,
            entry_price=1.1000,
            entry_time=datetime.now(),
            stop_loss=1.1050,
            take_profit=1.0950,
            magic_number=123457
        )

        coordinator.register_position(position1)
        coordinator.register_position(position2)

        intraday_positions = coordinator.get_positions_by_style(TradingStyle.INTRADAY)
        scalp_positions = coordinator.get_positions_by_style(TradingStyle.SCALP)

        assert len(intraday_positions) == 1
        assert len(scalp_positions) == 1
        assert intraday_positions[0].ticket == 12345
        assert scalp_positions[0].ticket == 12346

    def test_get_style_exposure(self, coordinator):
        """Test getting exposure metrics."""
        coordinator.initialize(10000.0)

        position = TrackedPosition(
            ticket=12345,
            style=TradingStyle.INTRADAY,
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            entry_price=1.1000,
            entry_time=datetime.now(),
            stop_loss=1.0950,
            take_profit=1.1100,
            magic_number=123456
        )

        coordinator.register_position(position)
        coordinator.update_position(12345, current_price=1.1050, unrealized_pnl=50.0)

        exposure = coordinator.get_style_exposure(TradingStyle.INTRADAY)

        assert exposure.style == TradingStyle.INTRADAY
        assert exposure.position_count == 1
        assert exposure.total_volume == 0.1
        assert exposure.long_volume == 0.1
        assert exposure.unrealized_pnl == 50.0

    def test_get_total_exposure(self, coordinator):
        """Test getting total exposure across all styles."""
        coordinator.initialize(10000.0)

        position1 = TrackedPosition(
            ticket=12345,
            style=TradingStyle.INTRADAY,
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            entry_price=1.1000,
            entry_time=datetime.now(),
            stop_loss=1.0950,
            take_profit=1.1100,
            magic_number=123456
        )

        position2 = TrackedPosition(
            ticket=12346,
            style=TradingStyle.SCALP,
            symbol="EURUSD",
            direction="BUY",
            volume=0.05,
            entry_price=1.1000,
            entry_time=datetime.now(),
            stop_loss=1.0950,
            take_profit=1.1100,
            magic_number=123457
        )

        coordinator.register_position(position1)
        coordinator.register_position(position2)

        total = coordinator.get_total_exposure()

        assert total['total_positions'] == 2
        assert total['total_volume'] == pytest.approx(0.15)
        assert total['long_volume'] == pytest.approx(0.15)

    def test_can_open_position_allowed(self, coordinator):
        """Test checking if position can be opened - allowed."""
        coordinator.initialize(10000.0)

        allowed, reason = coordinator.can_open_position(
            TradingStyle.INTRADAY,
            "BUY",
            0.1
        )

        assert allowed is True
        assert reason == "OK"

    def test_can_open_position_limit_reached(self, coordinator):
        """Test checking if position can be opened - limit reached."""
        coordinator.initialize(10000.0)
        coordinator.config.max_total_positions = 2

        # Add positions up to limit
        for i in range(2):
            position = TrackedPosition(
                ticket=10000 + i,
                style=TradingStyle.INTRADAY,
                symbol="EURUSD",
                direction="BUY",
                volume=0.1,
                entry_price=1.1000,
                entry_time=datetime.now(),
                stop_loss=1.0950,
                take_profit=1.1100,
                magic_number=123456
            )
            coordinator.register_position(position)

        # Check if can add more
        allowed, reason = coordinator.can_open_position(TradingStyle.INTRADAY, "BUY", 0.1)

        assert allowed is False
        assert "limit" in reason.lower()

    def test_update_position(self, coordinator):
        """Test updating position with current price and PnL."""
        coordinator.initialize(10000.0)

        position = TrackedPosition(
            ticket=12345,
            style=TradingStyle.INTRADAY,
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            entry_price=1.1000,
            entry_time=datetime.now(),
            stop_loss=1.0950,
            take_profit=1.1100,
            magic_number=123456
        )

        coordinator.register_position(position)
        coordinator.update_position(12345, current_price=1.1050, unrealized_pnl=50.0)

        updated = coordinator.positions[12345]
        assert updated.current_price == 1.1050
        assert updated.unrealized_pnl == 50.0

    def test_get_daily_stats(self, coordinator):
        """Test getting daily statistics."""
        coordinator.initialize(10000.0)

        position = TrackedPosition(
            ticket=12345,
            style=TradingStyle.INTRADAY,
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
            entry_price=1.1000,
            entry_time=datetime.now(),
            stop_loss=1.0950,
            take_profit=1.1100,
            magic_number=123456
        )

        coordinator.register_position(position)
        coordinator.close_position(12345, exit_price=1.1050, pnl=50.0)

        stats = coordinator.get_daily_stats(TradingStyle.INTRADAY)

        assert stats.trades_opened == 1
        assert stats.trades_closed == 1
        assert stats.wins == 1
        assert stats.losses == 0
        assert stats.realized_pnl == 50.0

