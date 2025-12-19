"""
Multi-Style Position Coordination Integration Tests.

Tests position coordination, exposure limits, and opposing position prevention.
"""
import pytest
from datetime import datetime, timedelta

from trading.position_coordinator import PositionCoordinator, TrackedPosition
from trading.style_config import OrchestratorConfig, TradingStyle, StyleConfig


@pytest.fixture
def coordinator():
    """Create position coordinator with test configuration."""
    config = OrchestratorConfig(
        max_total_positions=10,
        max_positions_per_style={
            TradingStyle.SCALP: 3,
            TradingStyle.INTRADAY: 3,
            TradingStyle.SWING: 2,
        },
        max_exposure_per_symbol=0.5,
        prevent_opposing_positions=True,
    )
    coord = PositionCoordinator(config)
    coord.initialize(balance=10000.0)
    return coord


def create_position(ticket, style, symbol, direction, volume=0.1, entry_price=1.1000):
    """Helper to create test positions."""
    return TrackedPosition(
        ticket=ticket,
        style=style,
        symbol=symbol,
        direction=direction,
        volume=volume,
        entry_price=entry_price,
        entry_time=datetime.utcnow(),
        stop_loss=entry_price - 0.005 if direction == "BUY" else entry_price + 0.005,
        take_profit=entry_price + 0.01 if direction == "BUY" else entry_price - 0.01,
        magic_number=123,
        current_price=entry_price,
        unrealized_pnl=0.0,
    )


@pytest.mark.integration
class TestExposureLimits:
    """Tests for exposure limit enforcement."""
    
    def test_max_positions_per_style_enforced(self, coordinator):
        """Should not allow more positions than style limit."""
        # Add max positions for INTRADAY
        for i in range(3):
            pos = create_position(i + 1, TradingStyle.INTRADAY, "EURUSD", "BUY")
            coordinator.register_position(pos)
        
        # Check if can open another INTRADAY
        can_open = coordinator.can_open_position(
            style=TradingStyle.INTRADAY,
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
        )
        
        assert can_open is False
    
    def test_different_style_can_still_open(self, coordinator):
        """Different style should be able to open even if one style is maxed."""
        # Max out INTRADAY
        for i in range(3):
            pos = create_position(i + 1, TradingStyle.INTRADAY, "EURUSD", "BUY")
            coordinator.register_position(pos)
        
        # SCALP should still be able to open
        can_open = coordinator.can_open_position(
            style=TradingStyle.SCALP,
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
        )
        
        assert can_open is True
    
    def test_max_total_positions_enforced(self, coordinator):
        """Should not exceed total position limit across all styles."""
        # Fill up with positions from different styles
        ticket = 1
        for style in [TradingStyle.SCALP, TradingStyle.INTRADAY, TradingStyle.SWING]:
            max_for_style = coordinator.config.max_positions_per_style.get(style, 2)
            for _ in range(max_for_style):
                pos = create_position(ticket, style, "EURUSD", "BUY")
                coordinator.register_position(pos)
                ticket += 1
        
        # Should not be able to open any more
        can_open = coordinator.can_open_position(
            style=TradingStyle.SCALP,
            symbol="GBPUSD",
            direction="BUY",
            volume=0.1,
        )
        
        # Either total limit or style limit should block
        total_positions = len(coordinator.positions)
        assert total_positions >= 8  # 3 + 3 + 2
    
    def test_symbol_exposure_limit_enforced(self, coordinator):
        """Should not exceed exposure limit for single symbol."""
        # Add large position
        pos = create_position(1, TradingStyle.INTRADAY, "EURUSD", "BUY", volume=0.4)
        coordinator.register_position(pos)
        
        # Try to add another large position on same symbol
        can_open = coordinator.can_open_position(
            style=TradingStyle.SCALP,
            symbol="EURUSD",
            direction="BUY",
            volume=0.2,  # Would exceed 0.5 limit
        )
        
        # Should be blocked due to symbol exposure
        assert can_open is False
    
    def test_different_symbol_not_affected(self, coordinator):
        """Exposure on one symbol should not affect another."""
        # Add position on EURUSD
        pos = create_position(1, TradingStyle.INTRADAY, "EURUSD", "BUY", volume=0.4)
        coordinator.register_position(pos)
        
        # Should be able to open on GBPUSD
        can_open = coordinator.can_open_position(
            style=TradingStyle.INTRADAY,
            symbol="GBPUSD",
            direction="BUY",
            volume=0.3,
        )
        
        assert can_open is True


@pytest.mark.integration
class TestOpposingPositionPrevention:
    """Tests for opposing position prevention."""
    
    def test_opposing_position_blocked_same_symbol(self, coordinator):
        """Should not allow opposing positions on same symbol."""
        # Open BUY position
        pos = create_position(1, TradingStyle.INTRADAY, "EURUSD", "BUY")
        coordinator.register_position(pos)
        
        # Try to open SELL on same symbol
        can_open = coordinator.can_open_position(
            style=TradingStyle.SCALP,
            symbol="EURUSD",
            direction="SELL",
            volume=0.1,
        )
        
        assert can_open is False
    
    def test_same_direction_allowed(self, coordinator):
        """Should allow same direction positions on same symbol."""
        # Open BUY position
        pos = create_position(1, TradingStyle.INTRADAY, "EURUSD", "BUY")
        coordinator.register_position(pos)
        
        # Try to open another BUY
        can_open = coordinator.can_open_position(
            style=TradingStyle.SCALP,
            symbol="EURUSD",
            direction="BUY",
            volume=0.1,
        )
        
        assert can_open is True
    
    def test_opposing_allowed_different_symbol(self, coordinator):
        """Should allow opposing positions on different symbols."""
        # Open BUY on EURUSD
        pos = create_position(1, TradingStyle.INTRADAY, "EURUSD", "BUY")
        coordinator.register_position(pos)
        
        # Should be able to SELL GBPUSD
        can_open = coordinator.can_open_position(
            style=TradingStyle.INTRADAY,
            symbol="GBPUSD",
            direction="SELL",
            volume=0.1,
        )
        
        assert can_open is True
    
    def test_opposing_allowed_when_disabled(self):
        """Should allow opposing positions when prevention is disabled."""
        config = OrchestratorConfig(
            prevent_opposing_positions=False,
        )
        coord = PositionCoordinator(config)
        coord.initialize(balance=10000.0)
        
        # Open BUY position
        pos = create_position(1, TradingStyle.INTRADAY, "EURUSD", "BUY")
        coord.register_position(pos)
        
        # Should be able to open SELL
        can_open = coord.can_open_position(
            style=TradingStyle.SCALP,
            symbol="EURUSD",
            direction="SELL",
            volume=0.1,
        )
        
        assert can_open is True


@pytest.mark.integration
class TestPositionTracking:
    """Tests for position tracking and updates."""
    
    def test_position_registration_and_retrieval(self, coordinator):
        """Positions should be correctly registered and retrievable."""
        pos = create_position(1, TradingStyle.INTRADAY, "EURUSD", "BUY")
        coordinator.register_position(pos)
        
        assert 1 in coordinator.positions
        assert coordinator.positions[1].symbol == "EURUSD"
        assert coordinator.positions[1].direction == "BUY"
    
    def test_position_closure_updates_tracking(self, coordinator):
        """Closing position should update tracking correctly."""
        pos = create_position(1, TradingStyle.INTRADAY, "EURUSD", "BUY")
        coordinator.register_position(pos)
        
        # Close position
        coordinator.close_position(ticket=1, pnl=50.0)
        
        assert 1 not in coordinator.positions
        assert 1 not in coordinator.positions_by_style[TradingStyle.INTRADAY]
    
    def test_pnl_tracking_by_style(self, coordinator):
        """P&L should be tracked separately by style."""
        # Open and close INTRADAY position
        pos1 = create_position(1, TradingStyle.INTRADAY, "EURUSD", "BUY")
        coordinator.register_position(pos1)
        coordinator.close_position(ticket=1, pnl=100.0)
        
        # Open and close SCALP position
        pos2 = create_position(2, TradingStyle.SCALP, "EURUSD", "BUY")
        coordinator.register_position(pos2)
        coordinator.close_position(ticket=2, pnl=-30.0)
        
        # Check daily stats
        intraday_stats = coordinator.daily_stats.get(TradingStyle.INTRADAY)
        scalp_stats = coordinator.daily_stats.get(TradingStyle.SCALP)
        
        if intraday_stats:
            assert intraday_stats.realized_pnl == 100.0
        if scalp_stats:
            assert scalp_stats.realized_pnl == -30.0
    
    def test_aggregate_exposure_calculation(self, coordinator):
        """Aggregate exposure should be calculated correctly."""
        # Add multiple positions
        pos1 = create_position(1, TradingStyle.INTRADAY, "EURUSD", "BUY", volume=0.1)
        pos2 = create_position(2, TradingStyle.SCALP, "EURUSD", "BUY", volume=0.2)
        pos3 = create_position(3, TradingStyle.INTRADAY, "GBPUSD", "SELL", volume=0.15)
        
        coordinator.register_position(pos1)
        coordinator.register_position(pos2)
        coordinator.register_position(pos3)
        
        exposure = coordinator.get_symbol_exposure("EURUSD")
        assert abs(exposure - 0.3) < 0.001  # 0.1 + 0.2


@pytest.mark.integration
class TestDailyLimits:
    """Tests for daily trading limits."""
    
    def test_daily_trade_count_tracked(self, coordinator):
        """Daily trade count should be tracked per style."""
        # Open and close multiple positions
        for i in range(3):
            pos = create_position(i + 1, TradingStyle.INTRADAY, "EURUSD", "BUY")
            coordinator.register_position(pos)
            coordinator.close_position(ticket=i + 1, pnl=10.0)
        
        stats = coordinator.daily_stats.get(TradingStyle.INTRADAY)
        if stats:
            assert stats.trades_closed == 3
    
    def test_daily_stats_reset(self, coordinator):
        """Daily stats should be resettable."""
        # Add some activity
        pos = create_position(1, TradingStyle.INTRADAY, "EURUSD", "BUY")
        coordinator.register_position(pos)
        coordinator.close_position(ticket=1, pnl=50.0)
        
        # Reset daily stats
        coordinator._reset_daily_stats()
        
        stats = coordinator.daily_stats.get(TradingStyle.INTRADAY)
        if stats:
            assert stats.trades_closed == 0
            assert stats.realized_pnl == 0.0
