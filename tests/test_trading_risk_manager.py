# tests/test_trading_risk_manager.py
"""
Unit tests for trading/risk_manager.py - Dynamic SL/TP calculation using ML predictions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from trading.risk_manager import (
    RiskManager, RiskMethod, RiskConfig, RiskLevels
)


@pytest.mark.unit
class TestRiskMethod:
    """Test RiskMethod enum."""

    def test_risk_method_values(self):
        """Test risk method enum values."""
        assert RiskMethod.VOLATILITY.value == "volatility"
        assert RiskMethod.QUANTILE.value == "quantile"
        assert RiskMethod.HYBRID.value == "hybrid"
        assert RiskMethod.FIXED_RR.value == "fixed_rr"


@pytest.mark.unit
class TestRiskConfig:
    """Test RiskConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RiskConfig()

        assert config.method == RiskMethod.HYBRID
        assert config.sl_volatility_multiplier == 1.5
        assert config.tp_volatility_multiplier == 2.5
        assert config.min_risk_reward == 1.5
        assert config.risk_per_trade == 0.01
        assert config.pip_value == 0.0001

    def test_custom_values(self):
        """Test custom configuration."""
        config = RiskConfig(
            method=RiskMethod.VOLATILITY,
            sl_volatility_multiplier=2.0,
            min_risk_reward=2.0
        )

        assert config.method == RiskMethod.VOLATILITY
        assert config.sl_volatility_multiplier == 2.0
        assert config.min_risk_reward == 2.0


@pytest.mark.unit
class TestRiskLevels:
    """Test RiskLevels dataclass."""

    def test_risk_levels_creation(self):
        """Test creating RiskLevels."""
        levels = RiskLevels(
            stop_loss=1.0950,
            take_profit=1.1100,
            risk_reward_ratio=2.0,
            sl_pips=50.0,
            tp_pips=100.0,
            predicted_volatility=0.001,
            confidence=0.75,
            method_used="hybrid"
        )

        assert levels.stop_loss == 1.0950
        assert levels.take_profit == 1.1100
        assert levels.risk_reward_ratio == 2.0
        assert levels.sl_pips == 50.0
        assert levels.method_used == "hybrid"

    def test_to_dict(self):
        """Test to_dict method."""
        levels = RiskLevels(
            stop_loss=1.0950,
            take_profit=1.1100,
            risk_reward_ratio=2.0,
            sl_pips=50.0,
            tp_pips=100.0,
            predicted_volatility=0.001,
            confidence=0.75,
            method_used="hybrid"
        )

        result = levels.to_dict()

        assert result['stop_loss'] == 1.0950
        assert result['take_profit'] == 1.1100
        assert result['risk_reward'] == 2.0
        assert result['method'] == "hybrid"


@pytest.mark.unit
class TestRiskManager:
    """Test RiskManager class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock PyTorch model."""
        model = Mock()
        
        # Mock model outputs
        direction_output = Mock()
        direction_output.shape = (1, 3)
        
        volatility_output = Mock()
        volatility_output.shape = (1, 1)
        
        quantiles_output = Mock()
        quantiles_output.shape = (1, 7)
        
        def forward(x):
            return {
                'direction': direction_output,
                'volatility': volatility_output,
                'quantiles': quantiles_output
            }
        
        model.forward = forward
        model.eval = Mock()
        model.to = Mock(return_value=model)
        model.config = Mock()
        model.config.quantiles = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
        
        return model

    @pytest.fixture
    def risk_manager(self, mock_model):
        """Create a RiskManager instance with mocked model."""
        with patch('trading.risk_manager.torch') as mock_torch, \
             patch('trading.risk_manager.F') as mock_F:
            
            # Mock torch operations
            mock_torch.tensor = lambda x, **kwargs: x
            mock_torch.no_grad = lambda: lambda f: f
            mock_torch.device = lambda x: x
            
            # Mock F.softmax
            mock_F.softmax = lambda x, dim: np.array([[0.2, 0.3, 0.5]])  # BEAR, SIDEWAYS, BULL
            
            # Mock model output processing
            direction_output = Mock()
            direction_output.cpu.return_value.numpy.return_value = np.array([[0.2, 0.3, 0.5]])
            
            volatility_output = Mock()
            volatility_output.cpu.return_value.numpy.return_value = np.array([[0.001]])
            
            quantiles_output = Mock()
            quantiles_output.cpu.return_value.numpy.return_value = np.array([
                [-0.002, -0.001, 0.0, 0.001, 0.002, 0.003, 0.004]
            ])
            
            def model_forward(x):
                return {
                    'direction': direction_output,
                    'volatility': volatility_output,
                    'quantiles': quantiles_output
                }
            
            mock_model.forward = model_forward
            
            manager = RiskManager(
                model=mock_model,
                feature_columns=['feature1', 'feature2'],
                config=RiskConfig()
            )
            
            return manager

    def test_init(self, mock_model):
        """Test initialization."""
        with patch('trading.risk_manager.torch') as mock_torch:
            mock_torch.device = lambda x: x
            mock_torch.cuda.is_available.return_value = False
            
            manager = RiskManager(
                model=mock_model,
                feature_columns=['feature1', 'feature2']
            )

            assert manager.model == mock_model
            assert len(manager.feature_columns) == 2
            assert manager.config.method == RiskMethod.HYBRID

    def test_calculate_volatility_based_buy(self, risk_manager):
        """Test volatility-based calculation for BUY."""
        entry_price = 1.1000
        direction = 'BUY'
        volatility = 0.001

        sl, tp = risk_manager._calculate_volatility_based(entry_price, direction, volatility)

        # SL should be below entry, TP should be above
        assert sl < entry_price
        assert tp > entry_price
        assert tp > sl

    def test_calculate_volatility_based_sell(self, risk_manager):
        """Test volatility-based calculation for SELL."""
        entry_price = 1.1000
        direction = 'SELL'
        volatility = 0.001

        sl, tp = risk_manager._calculate_volatility_based(entry_price, direction, volatility)

        # SL should be above entry, TP should be below
        assert sl > entry_price
        assert tp < entry_price
        assert tp < sl

    def test_calculate_position_size(self, risk_manager):
        """Test position size calculation."""
        account_balance = 10000.0
        entry_price = 1.1000
        stop_loss = 1.0950

        position = risk_manager.calculate_position_size(
            account_balance, entry_price, stop_loss
        )

        assert 'lots' in position
        assert 'units' in position
        assert 'risk_amount' in position
        assert position['risk_amount'] > 0

    def test_apply_limits_min_sl(self, risk_manager):
        """Test applying minimum SL limit."""
        entry_price = 1.1000
        direction = 'BUY'
        
        # Create SL that's too tight
        sl = entry_price - 0.00001  # 0.1 pips
        tp = entry_price + 0.001

        sl_limited, tp_limited = risk_manager._apply_limits(entry_price, direction, sl, tp)

        # SL should be adjusted to minimum
        sl_pips = abs(entry_price - sl_limited) / risk_manager.config.pip_value
        assert sl_pips >= risk_manager.config.min_sl_pips - 0.01  # Allow small floating point error

    def test_apply_limits_max_tp(self, risk_manager):
        """Test applying maximum TP limit."""
        entry_price = 1.1000
        direction = 'BUY'
        
        # Create TP that's too wide
        sl = entry_price - 0.001  # 10 pips
        tp = entry_price + 0.1  # 1000 pips

        sl_limited, tp_limited = risk_manager._apply_limits(entry_price, direction, sl, tp)

        # TP should be adjusted to maximum
        tp_pips = abs(tp_limited - entry_price) / risk_manager.config.pip_value
        assert tp_pips <= risk_manager.config.max_tp_pips

    def test_find_quantile_idx(self, risk_manager):
        """Test finding quantile index."""
        idx = risk_manager._find_quantile_idx(0.10)

        assert isinstance(idx, int)
        assert idx >= 0

    def test_analyze_trade(self, risk_manager):
        """Test complete trade analysis."""
        features = np.random.randn(100, 5)  # Sample features
        entry_price = 1.1000
        direction = 'BUY'
        account_balance = 10000.0

        # Mock the calculate_levels method
        with patch.object(risk_manager, 'calculate_levels') as mock_calc:
            mock_levels = RiskLevels(
                stop_loss=1.0950,
                take_profit=1.1100,
                risk_reward_ratio=2.0,
                sl_pips=50.0,
                tp_pips=100.0,
                predicted_volatility=0.001,
                confidence=0.75,
                method_used="hybrid"
            )
            mock_calc.return_value = mock_levels

            result = risk_manager.analyze_trade(
                features, entry_price, direction, account_balance
            )

            assert 'entry' in result
            assert 'direction' in result
            assert 'stop_loss' in result
            assert 'take_profit' in result
            assert 'position_lots' in result
            assert result['entry'] == entry_price
            assert result['direction'] == direction

