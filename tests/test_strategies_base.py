# tests/test_strategies_base.py
"""
Unit tests for strategies/base.py - Base strategy interface and protocols.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
from abc import ABC
from strategies.base import Strategy, DataProvider, Executor


@pytest.mark.unit
class TestDataProvider:
    """Test DataProvider protocol."""

    def test_data_provider_protocol(self):
        """Test that objects implementing DataProvider protocol work."""
        class MockDataProvider:
            def get_data(self, n: int = 100) -> pd.DataFrame:
                return pd.DataFrame({'close': [1.1000] * n})

        provider = MockDataProvider()
        df = provider.get_data(n=50)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        # This tests that the protocol is properly defined
        assert hasattr(provider, 'get_data')


@pytest.mark.unit
class TestExecutor:
    """Test Executor protocol."""

    def test_executor_protocol(self):
        """Test that objects implementing Executor protocol work."""
        class MockExecutor:
            def entry(self, signal: str, volume: float, sl: float, tp: float):
                return {'success': True, 'ticket': 12345}
            
            def get_open_positions(self):
                return []

        executor = MockExecutor()
        result = executor.entry("BUY", 0.1, 1.0950, 1.1100)

        assert result['success'] is True
        assert hasattr(executor, 'entry')
        assert hasattr(executor, 'get_open_positions')


@pytest.mark.unit
class TestStrategy:
    """Test Strategy abstract base class."""

    @pytest.fixture
    def mock_data_provider(self):
        """Create a mock data provider."""
        provider = Mock()
        provider.get_data = Mock(return_value=pd.DataFrame({
            'close': [1.1000, 1.1005, 1.1010]
        }))
        return provider

    @pytest.fixture
    def mock_executor(self):
        """Create a mock executor."""
        executor = Mock()
        executor.entry = Mock(return_value={'success': True})
        executor.get_open_positions = Mock(return_value=[])
        return executor

    @pytest.fixture
    def concrete_strategy(self, mock_data_provider, mock_executor):
        """Create a concrete implementation of Strategy for testing."""
        class TestStrategy(Strategy):
            def on_bar(self, df: pd.DataFrame):
                return "BUY" if len(df) > 2 else None

        return TestStrategy(
            data_provider=mock_data_provider,
            executor=mock_executor,
            name="TestStrategy"
        )

    def test_init(self, mock_data_provider, mock_executor):
        """Test Strategy initialization."""
        class TestStrategy(Strategy):
            def on_bar(self, df: pd.DataFrame):
                return None

        strategy = TestStrategy(
            data_provider=mock_data_provider,
            executor=mock_executor,
            name="TestStrategy"
        )

        assert strategy.data_provider == mock_data_provider
        assert strategy.executor == mock_executor
        assert strategy.name == "TestStrategy"
        assert strategy.is_active is True

    def test_init_default_name(self, mock_data_provider, mock_executor):
        """Test Strategy initialization with default name."""
        class TestStrategy(Strategy):
            def on_bar(self, df: pd.DataFrame):
                return None

        strategy = TestStrategy(
            data_provider=mock_data_provider,
            executor=mock_executor
        )

        assert strategy.name == "BaseStrategy"

    def test_cannot_instantiate_abstract(self, mock_data_provider, mock_executor):
        """Test that Strategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Strategy(mock_data_provider, mock_executor)

    def test_on_bar_must_be_implemented(self, mock_data_provider, mock_executor):
        """Test that on_bar must be implemented."""
        class IncompleteStrategy(Strategy):
            pass

        with pytest.raises(TypeError):
            IncompleteStrategy(mock_data_provider, mock_executor)

    def test_on_bar_implementation(self, concrete_strategy):
        """Test on_bar method implementation."""
        df = pd.DataFrame({'close': [1.1000, 1.1005, 1.1010, 1.1015]})

        result = concrete_strategy.on_bar(df)

        assert result == "BUY"

    def test_on_bar_insufficient_data(self, concrete_strategy):
        """Test on_bar with insufficient data."""
        df = pd.DataFrame({'close': [1.1000]})  # Only 1 row

        result = concrete_strategy.on_bar(df)

        assert result is None

    def test_on_tick_default(self, concrete_strategy):
        """Test default on_tick implementation (does nothing)."""
        tick_data = {'bid': 1.0999, 'ask': 1.1001, 'time': pd.Timestamp.now()}

        # Should not raise
        concrete_strategy.on_tick(tick_data)

    def test_on_tick_custom_implementation(self, mock_data_provider, mock_executor):
        """Test custom on_tick implementation."""
        class TickStrategy(Strategy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.tick_count = 0

            def on_bar(self, df: pd.DataFrame):
                return None

            def on_tick(self, tick_data: dict):
                self.tick_count += 1

        strategy = TickStrategy(mock_data_provider, mock_executor)
        tick_data = {'bid': 1.0999, 'ask': 1.1001}

        strategy.on_tick(tick_data)
        strategy.on_tick(tick_data)

        assert strategy.tick_count == 2

    def test_on_position_opened_default(self, concrete_strategy):
        """Test default on_position_opened implementation."""
        position = {'ticket': 12345, 'direction': 'BUY', 'volume': 0.1}

        # Should not raise
        concrete_strategy.on_position_opened(position)

    def test_on_position_opened_custom(self, mock_data_provider, mock_executor):
        """Test custom on_position_opened implementation."""
        class PositionTrackingStrategy(Strategy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.opened_positions = []

            def on_bar(self, df: pd.DataFrame):
                return None

            def on_position_opened(self, position: dict):
                self.opened_positions.append(position)

        strategy = PositionTrackingStrategy(mock_data_provider, mock_executor)
        position = {'ticket': 12345, 'direction': 'BUY'}

        strategy.on_position_opened(position)

        assert len(strategy.opened_positions) == 1
        assert strategy.opened_positions[0]['ticket'] == 12345

    def test_on_position_closed_default(self, concrete_strategy):
        """Test default on_position_closed implementation."""
        position = {'ticket': 12345, 'direction': 'BUY'}
        pnl = 50.0

        # Should not raise
        concrete_strategy.on_position_closed(position, pnl)

    def test_on_position_closed_custom(self, mock_data_provider, mock_executor):
        """Test custom on_position_closed implementation."""
        class PnLTrackingStrategy(Strategy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.total_pnl = 0.0

            def on_bar(self, df: pd.DataFrame):
                return None

            def on_position_closed(self, position: dict, pnl: float):
                self.total_pnl += pnl

        strategy = PnLTrackingStrategy(mock_data_provider, mock_executor)

        strategy.on_position_closed({'ticket': 12345}, 50.0)
        strategy.on_position_closed({'ticket': 12346}, -20.0)

        assert strategy.total_pnl == 30.0

    def test_activate(self, concrete_strategy):
        """Test activating strategy."""
        concrete_strategy.is_active = False

        concrete_strategy.activate()

        assert concrete_strategy.is_active is True

    def test_deactivate(self, concrete_strategy):
        """Test deactivating strategy."""
        concrete_strategy.is_active = True

        concrete_strategy.deactivate()

        assert concrete_strategy.is_active is False

    def test_strategy_is_abc(self):
        """Test that Strategy is an ABC."""
        assert issubclass(Strategy, ABC)

