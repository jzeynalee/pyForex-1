# tests/test_strategies_neural_hybrid.py
"""
Unit tests for strategies/neural_hybrid.py - Neural Hybrid Strategy with Risk Management.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from strategies.neural_hybrid import (
    NeuralHybridStrategy, StrategyConfig, Order, OrderType, OpenPosition,
    create_strategy
)
from trading.decision_engine import TradeDecision, Signal


@pytest.mark.unit
class TestOrderType:
    """Test OrderType enum."""

    def test_order_type_values(self):
        """Test order type enum values."""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"


@pytest.mark.unit
class TestOrder:
    """Test Order dataclass."""

    def test_order_creation(self):
        """Test creating Order."""
        order = Order(
            symbol="EURUSD",
            order_type=OrderType.MARKET,
            direction="BUY",
            volume=0.1,
            price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100
        )

        assert order.symbol == "EURUSD"
        assert order.order_type == OrderType.MARKET
        assert order.direction == "BUY"
        assert order.volume == 0.1
        assert order.price == 1.1000
        assert order.stop_loss == 1.0950
        assert order.take_profit == 1.1100

    def test_order_default_values(self):
        """Test Order default values."""
        order = Order(
            symbol="EURUSD",
            order_type=OrderType.MARKET,
            direction="BUY",
            volume=0.1,
            price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100
        )

        assert order.comment == ""
        assert order.magic_number == 123456
        assert order.ticket is None
        assert order.risk_percent == 0.0
        assert order.protection_level == 'normal'


@pytest.mark.unit
class TestOpenPosition:
    """Test OpenPosition dataclass."""

    def test_open_position_creation(self):
        """Test creating OpenPosition."""
        position = OpenPosition(
            ticket="12345",
            direction=1,
            entry_price=1.1000,
            entry_time=datetime.now(),
            volume=0.1,
            stop_loss=1.0950,
            take_profit=1.1100
        )

        assert position.ticket == "12345"
        assert position.direction == 1
        assert position.entry_price == 1.1000
        assert position.volume == 0.1
        assert position.unrealized_pnl == 0.0


@pytest.mark.unit
class TestStrategyConfig:
    """Test StrategyConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StrategyConfig()

        assert config.profile == 'INTRADAY'
        assert config.symbol == 'EURUSD'
        assert config.sequence_length == 60
        assert config.use_vision is True
        assert config.base_risk_percent == 1.0
        assert config.min_direction_confidence == 0.55
        assert config.enable_exit_advisor is True
        assert config.enable_capital_protection is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = StrategyConfig(
            profile='SWING',
            symbol='GBPUSD',
            sequence_length=100,
            use_vision=False,
            base_risk_percent=0.5
        )

        assert config.profile == 'SWING'
        assert config.symbol == 'GBPUSD'
        assert config.sequence_length == 100
        assert config.use_vision is False
        assert config.base_risk_percent == 0.5


@pytest.mark.unit
class TestNeuralHybridStrategy:
    """Test NeuralHybridStrategy class."""

    @pytest.fixture
    def mock_data_provider(self):
        """Create a mock data provider."""
        provider = Mock()
        provider.get_data = Mock(return_value=pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=100, freq='h'),
            'open': [1.1000] * 100,
            'high': [1.1010] * 100,
            'low': [1.0990] * 100,
            'close': [1.1005] * 100,
            'volume': [1000] * 100
        }))
        return provider

    @pytest.fixture
    def mock_executor(self):
        """Create a mock executor."""
        executor = Mock()
        executor.get_account_balance = Mock(return_value=10000.0)
        executor.execute_order = Mock(return_value={
            'success': True,
            'ticket': '12345',
            'price': 1.1000
        })
        executor.get_open_positions = Mock(return_value=[])
        return executor

    @pytest.fixture
    def mock_predictor(self):
        """Create a mock predictor."""
        predictor = Mock()
        prediction_result = Mock()
        prediction_result.probabilities = np.array([0.2, 0.3, 0.5])  # BEAR, SIDEWAYS, BULL
        prediction_result.volatility = 0.001
        prediction_result.quantiles = np.array([-0.002, -0.001, 0.0, 0.001, 0.002])
        prediction_result.features = np.random.randn(60, 64)
        predictor.predict = Mock(return_value=prediction_result)
        return predictor

    def test_init_default(self):
        """Test default initialization."""
        strategy = NeuralHybridStrategy()

        assert strategy.config.profile == 'INTRADAY'
        assert strategy.predictor is None
        assert strategy.decision_engine is None
        assert strategy._initialized is False
        assert len(strategy._open_positions) == 0

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = StrategyConfig(profile='SWING', symbol='GBPUSD')
        strategy = NeuralHybridStrategy(config=config)

        assert strategy.config.profile == 'SWING'
        assert strategy.config.symbol == 'GBPUSD'

    def test_init_with_provider_executor(self, mock_data_provider, mock_executor):
        """Test initialization with data provider and executor."""
        strategy = NeuralHybridStrategy(
            data_provider=mock_data_provider,
            executor=mock_executor
        )

        assert strategy.data_provider == mock_data_provider
        assert strategy.executor == mock_executor

    def test_initialize_success(self, mock_executor):
        """Test successful initialization."""
        with patch('strategies.neural_hybrid.create_predictor') as MockPredictor, \
             patch('strategies.neural_hybrid.EnhancedDecisionEngine') as MockEngine:
            
            mock_predictor_instance = Mock()
            MockPredictor.return_value = mock_predictor_instance
            
            mock_engine_instance = Mock()
            mock_engine_instance.initialize = Mock()
            mock_engine_instance.capital_protector = None
            MockEngine.return_value = mock_engine_instance

            strategy = NeuralHybridStrategy(executor=mock_executor)
            result = strategy.initialize(starting_balance=10000)

            assert result is True
            assert strategy._initialized is True
            assert strategy._starting_balance == 10000.0
            assert strategy.predictor == mock_predictor_instance
            mock_engine_instance.initialize.assert_called_once_with(10000)

    def test_initialize_from_executor(self, mock_executor):
        """Test initialization using executor balance."""
        with patch('strategies.neural_hybrid.create_predictor') as MockPredictor, \
             patch('strategies.neural_hybrid.EnhancedDecisionEngine') as MockEngine:
            
            MockPredictor.return_value = Mock()
            mock_engine_instance = Mock()
            mock_engine_instance.initialize = Mock()
            mock_engine_instance.capital_protector = None
            MockEngine.return_value = mock_engine_instance

            strategy = NeuralHybridStrategy(executor=mock_executor)
            strategy.initialize()

            assert strategy._starting_balance == 10000.0
            mock_executor.get_account_balance.assert_called_once()

    def test_create_order_from_decision(self):
        """Test creating order from trade decision."""
        strategy = NeuralHybridStrategy()
        
        decision = TradeDecision(
            signal=Signal.BULL,
            signal_name="BULL",
            should_trade=True,
            direction="BUY",
            stop_loss=1.0950,
            take_profit=1.1100,
            position_size=0.1,
            risk_percent=1.0,
            risk_reward_ratio=2.0,
            direction_confidence=0.75,
            meta_score=0.8,
            protection_level='normal'
        )

        order = strategy.create_order(decision)

        assert order is not None
        assert order.symbol == 'EURUSD'
        assert order.direction == "BUY"
        assert order.volume == 0.1
        assert order.stop_loss == 1.0950
        assert order.take_profit == 1.1100
        assert order.order_type == OrderType.MARKET

    def test_create_order_no_trade(self):
        """Test creating order when decision says no trade."""
        strategy = NeuralHybridStrategy()
        
        decision = TradeDecision(
            signal=Signal.SIDEWAYS,
            signal_name="SIDEWAYS",
            should_trade=False
        )

        order = strategy.create_order(decision)

        assert order is None

    def test_execute_order_success(self, mock_executor):
        """Test executing order successfully."""
        strategy = NeuralHybridStrategy(executor=mock_executor)
        strategy._initialized = True

        order = Order(
            symbol="EURUSD",
            order_type=OrderType.MARKET,
            direction="BUY",
            volume=0.1,
            price=0.0,
            stop_loss=1.0950,
            take_profit=1.1100
        )

        result = strategy.execute(order)

        assert result is True
        assert order.ticket == '12345'
        assert '12345' in strategy._open_positions
        assert strategy._daily_trades == 1

    def test_execute_order_no_executor(self):
        """Test executing order without executor."""
        strategy = NeuralHybridStrategy()

        order = Order(
            symbol="EURUSD",
            order_type=OrderType.MARKET,
            direction="BUY",
            volume=0.1,
            price=0.0,
            stop_loss=1.0950,
            take_profit=1.1100
        )

        result = strategy.execute(order)

        assert result is False

    def test_execute_order_failure(self, mock_executor):
        """Test executing order when executor returns failure."""
        mock_executor.execute_order.return_value = {
            'success': False,
            'error': 'Insufficient margin'
        }
        strategy = NeuralHybridStrategy(executor=mock_executor)
        strategy._initialized = True

        order = Order(
            symbol="EURUSD",
            order_type=OrderType.MARKET,
            direction="BUY",
            volume=0.1,
            price=0.0,
            stop_loss=1.0950,
            take_profit=1.1100
        )

        result = strategy.execute(order)

        assert result is False
        assert len(strategy._open_positions) == 0

    def test_on_trade_closed(self):
        """Test handling trade close event."""
        strategy = NeuralHybridStrategy()
        strategy._initialized = True
        strategy._current_balance = 10000.0
        
        # Add a position
        position = OpenPosition(
            ticket="12345",
            direction=1,
            entry_price=1.1000,
            entry_time=datetime.now(),
            volume=0.1,
            stop_loss=1.0950,
            take_profit=1.1100
        )
        strategy._open_positions["12345"] = position

        with patch.object(strategy, 'decision_engine') as mock_engine:
            mock_engine.record_trade_result = Mock()
            
            strategy.on_trade_closed("12345", pnl=50.0)

            assert "12345" not in strategy._open_positions
            assert strategy._daily_pnl == 50.0
            assert strategy._current_balance == 10050.0
            assert strategy._daily_wins == 1
            assert strategy._daily_losses == 0
            mock_engine.record_trade_result.assert_called_once_with(50.0, True, 0.1)

    def test_on_trade_closed_loss(self):
        """Test handling losing trade close."""
        strategy = NeuralHybridStrategy()
        strategy._initialized = True
        strategy._current_balance = 10000.0

        with patch.object(strategy, 'decision_engine') as mock_engine:
            mock_engine.record_trade_result = Mock()
            
            strategy.on_trade_closed("12345", pnl=-30.0)

            assert strategy._daily_pnl == -30.0
            assert strategy._current_balance == 9970.0
            assert strategy._daily_wins == 0
            assert strategy._daily_losses == 1

    def test_check_daily_limits(self):
        """Test checking daily limits."""
        strategy = NeuralHybridStrategy()
        strategy._initialized = True
        strategy.config.max_daily_trades = 5
        strategy.config.max_daily_loss = 500.0
        strategy._daily_trades = 4
        strategy._daily_pnl = -400.0

        # Should pass
        assert strategy._check_daily_limits() is True

        # Exceed trade limit
        strategy._daily_trades = 5
        assert strategy._check_daily_limits() is False

        # Exceed loss limit
        strategy._daily_trades = 2
        strategy._daily_pnl = -600.0
        assert strategy._check_daily_limits() is False

    def test_get_stats(self):
        """Test getting strategy statistics."""
        strategy = NeuralHybridStrategy()
        strategy._initialized = True
        strategy._daily_trades = 5
        strategy._daily_pnl = 150.0
        strategy._daily_wins = 3
        strategy._daily_losses = 2
        strategy._current_balance = 10150.0

        stats = strategy.get_stats()

        assert stats['daily_trades'] == 5
        assert stats['daily_pnl'] == 150.0
        assert stats['daily_wins'] == 3
        assert stats['daily_losses'] == 2
        assert stats['win_rate'] == 0.6
        assert stats['current_balance'] == 10150.0

    def test_get_stats_no_trades(self):
        """Test getting stats with no trades."""
        strategy = NeuralHybridStrategy()
        strategy._initialized = True

        stats = strategy.get_stats()

        assert stats['daily_trades'] == 0
        assert stats['win_rate'] == 0.0

    def test_get_open_positions(self):
        """Test getting open positions."""
        strategy = NeuralHybridStrategy()
        strategy._initialized = True
        
        position = OpenPosition(
            ticket="12345",
            direction=1,
            entry_price=1.1000,
            entry_time=datetime.now(),
            volume=0.1,
            stop_loss=1.0950,
            take_profit=1.1100
        )
        strategy._open_positions["12345"] = position

        positions = strategy.get_open_positions()

        assert len(positions) == 1
        assert positions[0]['ticket'] == "12345"


@pytest.mark.unit
class TestCreateStrategy:
    """Test create_strategy factory function."""

    @pytest.fixture
    def mock_data_provider(self):
        """Create a mock data provider."""
        return Mock()

    @pytest.fixture
    def mock_executor(self):
        """Create a mock executor."""
        executor = Mock()
        executor.get_account_balance = Mock(return_value=10000.0)
        return executor

    def test_create_strategy_default(self, mock_data_provider, mock_executor):
        """Test creating strategy with default config."""
        with patch('strategies.neural_hybrid.NeuralHybridStrategy') as MockStrategy:
            mock_instance = Mock()
            MockStrategy.return_value = mock_instance

            result = create_strategy(
                data_provider=mock_data_provider,
                executor=mock_executor
            )

            MockStrategy.assert_called_once()
            assert result == mock_instance

    def test_create_strategy_with_config(self, mock_data_provider, mock_executor):
        """Test creating strategy with custom config."""
        config = StrategyConfig(profile='SWING', symbol='GBPUSD')
        
        with patch('strategies.neural_hybrid.NeuralHybridStrategy') as MockStrategy:
            mock_instance = Mock()
            MockStrategy.return_value = mock_instance

            result = create_strategy(
                config=config,
                data_provider=mock_data_provider,
                executor=mock_executor
            )

            # Check that config was passed
            call_kwargs = MockStrategy.call_args[1]
            assert call_kwargs['config'] == config

