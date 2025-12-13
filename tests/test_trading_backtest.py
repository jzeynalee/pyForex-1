# tests/test_trading_backtest.py
"""
Unit tests for trading/backtest.py - Backtesting engine.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from trading.backtest import (
    BacktestTrade, Position, BacktestConfig, BacktestExecutor
)


@pytest.mark.unit
class TestBacktestConfig:
    """Test BacktestConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BacktestConfig()

        assert config.initial_balance == 10000.0
        assert config.commission_per_lot == 7.0
        assert config.spread_pips == 1.0
        assert config.pip_value == 10.0

    def test_custom_values(self):
        """Test custom configuration."""
        config = BacktestConfig(
            initial_balance=50000.0,
            commission_per_lot=5.0,
            spread_pips=0.5,
            pip_value=5.0,
        )

        assert config.initial_balance == 50000.0
        assert config.commission_per_lot == 5.0
        assert config.spread_pips == 0.5
        assert config.pip_value == 5.0


@pytest.mark.unit
class TestBacktestExecutor:
    """Test BacktestExecutor class."""

    @pytest.fixture
    def executor(self):
        """Create a backtest executor with default config."""
        config = BacktestConfig(initial_balance=10000.0)
        return BacktestExecutor(config)

    def test_init_default(self):
        """Test executor initialization with default config."""
        executor = BacktestExecutor()

        assert executor.balance == 10000.0
        assert executor.equity == 10000.0
        assert len(executor.positions) == 0
        assert len(executor.trade_history) == 0
        assert executor.ticket_counter == 1000

    def test_init_custom_config(self):
        """Test executor initialization with custom config."""
        config = BacktestConfig(initial_balance=50000.0)
        executor = BacktestExecutor(config)

        assert executor.balance == 50000.0
        assert executor.equity == 50000.0
        assert executor.config.initial_balance == 50000.0

    def test_entry_buy_order(self, executor):
        """Test opening a BUY order."""
        executor.current_price = 1.1000
        executor.current_time = datetime(2024, 1, 1, 12, 0)

        result = executor.entry(
            signal='BUY',
            volume=0.1,
            sl=1.0950,
            tp=1.1100,
        )

        assert result['success'] is True
        assert 'ticket' in result
        assert result['volume'] == 0.1
        assert len(executor.positions) == 1

        # Check position details
        pos = executor.positions[0]
        assert pos.direction == 'BUY'
        assert pos.volume == 0.1
        assert pos.sl == 1.0950
        assert pos.tp == 1.1100

        # Entry price should include spread
        assert pos.entry_price > executor.current_price

    def test_entry_sell_order(self, executor):
        """Test opening a SELL order."""
        executor.current_price = 1.1000

        result = executor.entry(
            signal='SELL',
            volume=0.1,
            sl=1.1050,
            tp=1.0900,
        )

        assert result['success'] is True
        assert len(executor.positions) == 1

        pos = executor.positions[0]
        assert pos.direction == 'SELL'

        # Entry price should include spread
        assert pos.entry_price < executor.current_price

    def test_entry_invalid_signal(self, executor):
        """Test entry with invalid signal."""
        result = executor.entry(
            signal='INVALID',
            volume=0.1,
            sl=1.0950,
            tp=1.1100,
        )

        assert result['success'] is False
        assert 'error' in result
        assert len(executor.positions) == 0

    def test_commission_deduction(self, executor):
        """Test commission is deducted from balance."""
        initial_balance = executor.balance
        executor.current_price = 1.1000

        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)

        # Commission = 7.0 * 0.1 = 0.7
        expected_balance = initial_balance - (7.0 * 0.1)
        assert abs(executor.balance - expected_balance) < 0.01

    def test_update_price_no_trigger(self, executor):
        """Test price update without SL/TP trigger."""
        executor.current_price = 1.1000
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)

        # Update to price that doesn't trigger SL/TP
        executor.update_price(1.1020)

        assert len(executor.positions) == 1
        assert len(executor.trade_history) == 0

    def test_update_price_hit_tp_buy(self, executor):
        """Test BUY order hitting TP."""
        executor.current_price = 1.1000
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)

        initial_balance = executor.balance

        # Hit TP
        executor.update_price(1.1100)

        assert len(executor.positions) == 0
        assert len(executor.trade_history) == 1

        trade = executor.trade_history[0]
        assert trade.status == 'CLOSED_TP'
        assert trade.pnl is not None
        assert trade.pnl > 0
        assert executor.balance > initial_balance

    def test_update_price_hit_sl_buy(self, executor):
        """Test BUY order hitting SL."""
        executor.current_price = 1.1000
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)

        initial_balance = executor.balance

        # Hit SL
        executor.update_price(1.0950)

        assert len(executor.positions) == 0
        assert len(executor.trade_history) == 1

        trade = executor.trade_history[0]
        assert trade.status == 'CLOSED_SL'
        assert trade.pnl is not None
        assert trade.pnl < 0
        assert executor.balance < initial_balance

    def test_update_price_hit_tp_sell(self, executor):
        """Test SELL order hitting TP."""
        executor.current_price = 1.1000
        executor.entry(signal='SELL', volume=0.1, sl=1.1050, tp=1.0900)

        initial_balance = executor.balance

        # Hit TP
        executor.update_price(1.0900)

        assert len(executor.positions) == 0
        assert len(executor.trade_history) == 1

        trade = executor.trade_history[0]
        assert trade.status == 'CLOSED_TP'
        assert trade.pnl > 0
        assert executor.balance > initial_balance

    def test_update_price_hit_sl_sell(self, executor):
        """Test SELL order hitting SL."""
        executor.current_price = 1.1000
        executor.entry(signal='SELL', volume=0.1, sl=1.1050, tp=1.0900)

        initial_balance = executor.balance

        # Hit SL
        executor.update_price(1.1050)

        assert len(executor.positions) == 0
        assert len(executor.trade_history) == 1

        trade = executor.trade_history[0]
        assert trade.status == 'CLOSED_SL'
        assert trade.pnl < 0
        assert executor.balance < initial_balance

    def test_pnl_calculation_buy(self, executor):
        """Test P&L calculation for BUY trade."""
        executor.current_price = 1.1000
        executor.current_time = datetime(2024, 1, 1, 12, 0)

        # Entry with spread
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)

        initial_balance = executor.balance

        # Close at TP: 1.1100
        executor.update_price(1.1100, datetime(2024, 1, 1, 13, 0))

        trade = executor.trade_history[0]

        # Calculate expected P&L
        # Entry at 1.1000 + spread (0.0001) = 1.1001
        # Exit at 1.1100
        # Pips = (1.1100 - 1.1001) / 0.0001 = 99 pips
        # P&L = 99 * 10 * 0.1 = 99
        expected_pnl = 99.0

        assert abs(trade.pnl - expected_pnl) < 1.0  # Allow small rounding
        assert executor.balance == initial_balance + trade.pnl

    def test_pnl_calculation_sell(self, executor):
        """Test P&L calculation for SELL trade."""
        executor.current_price = 1.1000

        executor.entry(signal='SELL', volume=0.1, sl=1.1050, tp=1.0900)

        initial_balance = executor.balance

        # Close at TP: 1.0900
        executor.update_price(1.0900)

        trade = executor.trade_history[0]

        # Entry at 1.1000 - spread = 1.0999
        # Exit at 1.0900
        # Pips = (1.0999 - 1.0900) / 0.0001 = 99 pips
        # P&L = 99 * 10 * 0.1 = 99
        expected_pnl = 99.0

        assert abs(trade.pnl - expected_pnl) < 1.0
        assert executor.balance == initial_balance + trade.pnl

    def test_equity_calculation(self, executor):
        """Test equity calculation with open positions."""
        executor.current_price = 1.1000
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)

        initial_equity = executor.equity

        # Price moves in favor
        executor.update_price(1.1050)

        # Equity should increase with unrealized profit
        assert executor.equity > initial_equity

        # Price moves against
        executor.update_price(1.0980)

        # Equity should decrease
        assert executor.equity < initial_equity

    def test_close_all_positions(self, executor):
        """Test closing all positions manually."""
        executor.current_price = 1.1000

        # Open multiple positions
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)
        executor.entry(signal='SELL', volume=0.05, sl=1.1050, tp=1.0900)

        assert len(executor.positions) == 2

        # Close all
        executor.current_price = 1.1020
        executor.close_all_positions()

        assert len(executor.positions) == 0
        assert len(executor.trade_history) == 2

        # All trades should be CLOSED_MANUAL
        for trade in executor.trade_history:
            assert trade.status == 'CLOSED_MANUAL'

    def test_get_open_positions(self, executor):
        """Test getting open positions."""
        executor.current_price = 1.1000
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)

        positions = executor.get_open_positions()

        assert len(positions) == 1
        pos = positions[0]

        assert pos['type'] == 'BUY'
        assert pos['volume'] == 0.1
        assert pos['sl'] == 1.0950
        assert pos['tp'] == 1.1100
        assert 'profit' in pos

    def test_get_trade_history(self, executor):
        """Test getting trade history."""
        executor.current_price = 1.1000
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)
        executor.update_price(1.1100)  # Close at TP

        history = executor.get_trade_history()

        assert len(history) == 1
        trade = history[0]

        assert trade['direction'] == 'BUY'
        assert trade['volume'] == 0.1
        assert trade['entry_price'] is not None
        assert trade['exit_price'] is not None
        assert trade['pnl'] is not None
        assert trade['status'] == 'CLOSED_TP'

    def test_performance_metrics_no_trades(self, executor):
        """Test performance metrics with no trades."""
        metrics = executor.get_performance_metrics()

        assert metrics['total_trades'] == 0
        assert metrics['win_rate'] == 0
        assert metrics['profit_factor'] == 0
        assert metrics['total_pnl'] == 0

    def test_performance_metrics_with_trades(self, executor):
        """Test performance metrics calculation."""
        executor.current_price = 1.1000

        # Winning trade
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)
        executor.update_price(1.1100)

        # Losing trade
        executor.current_price = 1.1000
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)
        executor.update_price(1.0950)

        # Winning trade
        executor.current_price = 1.1000
        executor.entry(signal='SELL', volume=0.1, sl=1.1050, tp=1.0900)
        executor.update_price(1.0900)

        metrics = executor.get_performance_metrics()

        assert metrics['total_trades'] == 3
        assert metrics['winning_trades'] == 2
        assert metrics['losing_trades'] == 1
        assert metrics['win_rate'] == pytest.approx(2/3, abs=0.01)
        assert metrics['total_pnl'] != 0
        assert 'profit_factor' in metrics
        assert 'average_pnl' in metrics
        assert 'largest_win' in metrics
        assert 'largest_loss' in metrics
        assert 'final_balance' in metrics
        assert 'return_pct' in metrics

    def test_profit_factor_calculation(self, executor):
        """Test profit factor calculation."""
        executor.current_price = 1.1000

        # 2 winners
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)
        executor.update_price(1.1100)

        executor.current_price = 1.1000
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)
        executor.update_price(1.1100)

        # 1 loser
        executor.current_price = 1.1000
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)
        executor.update_price(1.0950)

        metrics = executor.get_performance_metrics()

        # Profit factor = Total wins / Total losses
        assert metrics['profit_factor'] > 1.0

    def test_get_account_info(self, executor):
        """Test getting account info."""
        account_info = executor.get_account_info()

        assert account_info.balance == 10000.0
        assert account_info.equity == 10000.0
        assert account_info.margin == 0
        assert account_info.free_margin == 10000.0
        assert account_info.profit == 0

    def test_get_symbol_info(self, executor):
        """Test getting symbol info."""
        symbol_info = executor.get_symbol_info()

        assert symbol_info['point'] == 0.00001
        assert symbol_info['volume_min'] == 0.01
        assert symbol_info['volume_max'] == 100.0
        assert symbol_info['volume_step'] == 0.01

    def test_multiple_positions(self, executor):
        """Test handling multiple open positions."""
        executor.current_price = 1.1000

        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)
        executor.entry(signal='SELL', volume=0.05, sl=1.1150, tp=1.0900)  # Higher SL to avoid close
        executor.entry(signal='BUY', volume=0.2, sl=1.0950, tp=1.1100)

        assert len(executor.positions) == 3

        # Close BUY positions
        executor.update_price(1.1100)

        # BUY positions should close, SELL remains
        assert len(executor.positions) == 1
        assert len(executor.trade_history) == 2

        # Verify SELL position remains
        assert executor.positions[0].direction == 'SELL'

    def test_ticket_counter_increment(self, executor):
        """Test that ticket numbers increment."""
        executor.current_price = 1.1000

        result1 = executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)
        result2 = executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)

        assert result2['ticket'] > result1['ticket']

    def test_time_tracking(self, executor):
        """Test entry and exit time tracking."""
        entry_time = datetime(2024, 1, 1, 12, 0)
        exit_time = datetime(2024, 1, 1, 13, 0)

        executor.current_price = 1.1000
        executor.current_time = entry_time
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)

        executor.update_price(1.1100, exit_time)

        trade = executor.trade_history[0]
        assert trade.entry_time == entry_time
        assert trade.exit_time == exit_time

    def test_balance_updates_correctly(self, executor):
        """Test that balance updates correctly through trades."""
        initial_balance = executor.balance
        executor.current_price = 1.1000

        # Win 100 (after commission)
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)
        balance_after_entry = executor.balance
        executor.update_price(1.1100)

        # Balance = initial - commission + profit
        assert executor.balance > initial_balance
        assert executor.balance == balance_after_entry + executor.trade_history[0].pnl

    def test_large_volume_trade(self, executor):
        """Test trade with large volume."""
        executor.current_price = 1.1000

        result = executor.entry(signal='BUY', volume=1.0, sl=1.0950, tp=1.1100)

        assert result['success'] is True

        # Commission should be 1.0 * 7.0 = 7.0
        expected_balance = 10000.0 - 7.0
        assert abs(executor.balance - expected_balance) < 0.01

    def test_unrealized_pnl_calculation(self, executor):
        """Test unrealized P&L calculation."""
        executor.current_price = 1.1000
        executor.entry(signal='BUY', volume=0.1, sl=1.0950, tp=1.1100)

        # Price moves up
        executor.update_price(1.1050)

        positions = executor.get_open_positions()
        assert positions[0]['profit'] > 0

        # Price moves down
        executor.update_price(1.0980)

        positions = executor.get_open_positions()
        assert positions[0]['profit'] < 0
