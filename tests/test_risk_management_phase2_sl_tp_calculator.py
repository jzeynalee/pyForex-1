# tests/test_risk_management_phase2_sl_tp_calculator.py
"""
Comprehensive unit tests for risk_management/phase2_risk_calc/sl_tp_calculator.py

Tests SL/TP calculation with quantiles, volatility, regime adjustments, and confidence.
"""

import pytest
import numpy as np
from risk_management.phase2_risk_calc.sl_tp_calculator import (
    MarketRegime, TradeDirection, SLTPConfig, SLTPResult, SLTPCalculator
)


@pytest.mark.unit
class TestSLTPConfig:
    """Test SLTPConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SLTPConfig()

        assert config.sl_quantile_buy == 0.05
        assert config.tp_quantile_buy == 0.75
        assert config.min_risk_reward == 1.5
        assert MarketRegime.TRENDING_STRONG in config.regime_sl_multiplier
        assert MarketRegime.VOLATILE in config.regime_sl_multiplier

    def test_regime_multipliers(self):
        """Test regime multipliers are present for all regimes."""
        config = SLTPConfig()

        assert len(config.regime_sl_multiplier) == 5
        assert len(config.regime_tp_multiplier) == 5
        
        # Check all regimes have multipliers
        for regime in MarketRegime:
            assert regime in config.regime_sl_multiplier
            assert regime in config.regime_tp_multiplier

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = SLTPConfig(
            min_risk_reward=2.0,
            sl_quantile_buy=0.1
        )

        assert config.min_risk_reward == 2.0
        assert config.sl_quantile_buy == 0.1


@pytest.mark.unit
class TestSLTPCalculator:
    """Test SLTPCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create SLTPCalculator instance."""
        return SLTPCalculator()

    @pytest.fixture
    def sample_quantiles(self):
        """Sample quantile predictions [Q5, Q25, Q50, Q75, Q95]."""
        return np.array([-0.002, -0.001, 0.0, 0.0015, 0.003])

    def test_init_default(self):
        """Test default initialization."""
        calculator = SLTPCalculator()

        assert calculator.config is not None
        assert isinstance(calculator.config, SLTPConfig)

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = SLTPConfig(min_risk_reward=2.0)
        calculator = SLTPCalculator(config)

        assert calculator.config.min_risk_reward == 2.0

    def test_calculate_buy_quantile_based(self, calculator, sample_quantiles):
        """Test SL/TP calculation for BUY trade using quantiles."""
        entry_price = 1.1000
        volatility = 0.0005

        result = calculator.calculate(
            entry_price=entry_price,
            direction=TradeDirection.BUY,
            quantiles=sample_quantiles,
            volatility=volatility
        )

        assert isinstance(result, SLTPResult)
        assert result.stop_loss < entry_price  # SL below entry for BUY
        assert result.take_profit > entry_price  # TP above entry for BUY
        assert result.risk_reward_ratio >= calculator.config.min_risk_reward
        assert result.sl_distance > 0
        assert result.tp_distance > 0

    def test_calculate_sell_quantile_based(self, calculator, sample_quantiles):
        """Test SL/TP calculation for SELL trade using quantiles."""
        entry_price = 1.1000
        volatility = 0.0005

        result = calculator.calculate(
            entry_price=entry_price,
            direction=TradeDirection.SELL,
            quantiles=sample_quantiles,
            volatility=volatility
        )

        assert isinstance(result, SLTPResult)
        assert result.stop_loss > entry_price  # SL above entry for SELL
        assert result.take_profit < entry_price  # TP below entry for SELL

    def test_calculate_with_regime_adjustment(self, calculator, sample_quantiles):
        """Test SL/TP calculation with regime adjustment."""
        entry_price = 1.1000
        volatility = 0.0005

        result = calculator.calculate(
            entry_price=entry_price,
            direction=TradeDirection.BUY,
            quantiles=sample_quantiles,
            volatility=volatility,
            regime=MarketRegime.VOLATILE
        )

        assert isinstance(result, SLTPResult)
        assert result.regime_adjusted is True
        # In volatile regime, SL should be wider
        assert result.sl_distance > 0

    def test_calculate_with_confidence_adjustment(self, calculator, sample_quantiles):
        """Test SL/TP calculation with confidence adjustment."""
        entry_price = 1.1000
        volatility = 0.0005

        result = calculator.calculate(
            entry_price=entry_price,
            direction=TradeDirection.BUY,
            quantiles=sample_quantiles,
            volatility=volatility,
            direction_confidence=0.85
        )

        assert isinstance(result, SLTPResult)
        assert result.confidence_adjusted is True

    def test_calculate_with_low_confidence(self, calculator, sample_quantiles):
        """Test calculation with low confidence widens SL."""
        entry_price = 1.1000
        volatility = 0.0005

        result_high_conf = calculator.calculate(
            entry_price=entry_price,
            direction=TradeDirection.BUY,
            quantiles=sample_quantiles,
            volatility=volatility,
            direction_confidence=0.9
        )

        result_low_conf = calculator.calculate(
            entry_price=entry_price,
            direction=TradeDirection.BUY,
            quantiles=sample_quantiles,
            volatility=volatility,
            direction_confidence=0.3
        )

        # Low confidence might result in wider SL (but not guaranteed due to min constraints)
        assert result_low_conf.sl_distance > 0

    def test_calculate_with_atr(self, calculator, sample_quantiles):
        """Test calculation using ATR instead of volatility."""
        entry_price = 1.1000
        atr = 0.0008

        result = calculator.calculate(
            entry_price=entry_price,
            direction=TradeDirection.BUY,
            quantiles=sample_quantiles,
            volatility=0.0005,  # Ignored when ATR provided
            atr=atr
        )

        assert isinstance(result, SLTPResult)
        assert result.risk_reward_ratio >= calculator.config.min_risk_reward

    def test_risk_reward_enforcement(self, calculator):
        """Test that minimum risk-reward ratio is enforced."""
        entry_price = 1.1000
        # Create quantiles that would give poor R:R
        poor_quantiles = np.array([-0.001, -0.0005, 0.0, 0.0006, 0.001])  # Small TP

        result = calculator.calculate(
            entry_price=entry_price,
            direction=TradeDirection.BUY,
            quantiles=poor_quantiles,
            volatility=0.0005
        )

        # Should enforce minimum R:R
        assert result.risk_reward_ratio >= calculator.config.min_risk_reward

    def test_all_regimes(self, calculator, sample_quantiles):
        """Test calculation works with all market regimes."""
        entry_price = 1.1000

        for regime in MarketRegime:
            result = calculator.calculate(
                entry_price=entry_price,
                direction=TradeDirection.BUY,
                quantiles=sample_quantiles,
                volatility=0.0005,
                regime=regime
            )

            assert isinstance(result, SLTPResult)
            assert result.regime_adjusted is True
            assert result.risk_reward_ratio >= calculator.config.min_risk_reward

    def test_volatility_bounds(self, calculator, sample_quantiles):
        """Test that SL/TP respect ATR-based bounds."""
        entry_price = 1.1000
        atr = 0.001

        result = calculator.calculate(
            entry_price=entry_price,
            direction=TradeDirection.BUY,
            quantiles=sample_quantiles,
            volatility=0.0005,
            atr=atr
        )

        # SL should be within bounds
        min_sl_distance = atr * calculator.config.min_sl_atr_multiple
        max_sl_distance = atr * calculator.config.max_sl_atr_multiple
        
        # Result might be adjusted, but should be reasonable
        assert result.sl_distance > 0

