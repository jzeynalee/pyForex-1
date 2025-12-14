# tests/test_trading_decision_engine.py
"""
Unit tests for trading/decision_engine.py - Enhanced decision engine with risk management.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from trading.decision_engine import (
    Signal, TradeDecision, DecisionEngineConfig, EnhancedDecisionEngine,
    convert_legacy_predictions, MTFDecisionEngine
)


@pytest.mark.unit
class TestSignal:
    """Test Signal enum."""

    def test_signal_values(self):
        """Test signal enum values."""
        assert Signal.BEAR == 0
        assert Signal.SIDEWAYS == 1
        assert Signal.BULL == 2


@pytest.mark.unit
class TestTradeDecision:
    """Test TradeDecision dataclass."""

    def test_default_values(self):
        """Test default TradeDecision values."""
        decision = TradeDecision(
            signal=Signal.BULL,
            signal_name="BULL",
            should_trade=True
        )

        assert decision.signal == Signal.BULL
        assert decision.signal_name == "BULL"
        assert decision.should_trade is True
        assert decision.rejection_reasons == []
        assert decision.direction == ''
        assert decision.stop_loss == 0.0
        assert decision.position_size == 0.0

    def test_to_dict(self):
        """Test to_dict method."""
        decision = TradeDecision(
            signal=Signal.BULL,
            signal_name="BULL",
            should_trade=True,
            direction="BUY",
            direction_confidence=0.75,
            stop_loss=1.0950,
            take_profit=1.1100,
            sl_pips=50.0,
            tp_pips=100.0,
            risk_reward_ratio=2.0,
            position_size=0.1,
            meta_score=0.8
        )

        result = decision.to_dict()

        assert result['signal'] == Signal.BULL.value
        assert result['signal_name'] == "BULL"
        assert result['should_trade'] is True
        assert result['direction'] == "BUY"
        assert result['direction_confidence'] == 0.75
        assert result['stop_loss'] == 1.0950
        assert result['take_profit'] == 1.1100


@pytest.mark.unit
class TestDecisionEngineConfig:
    """Test DecisionEngineConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DecisionEngineConfig()

        assert config.profile == 'INTRADAY'
        assert config.min_direction_confidence == 0.55
        assert config.min_meta_score == 0.5
        assert config.base_risk_percent == 1.0
        assert config.min_risk_reward == 1.5
        assert config.enable_capital_protection is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = DecisionEngineConfig(
            profile='SWING',
            min_direction_confidence=0.70,
            base_risk_percent=0.5,
            enable_capital_protection=False
        )

        assert config.profile == 'SWING'
        assert config.min_direction_confidence == 0.70
        assert config.base_risk_percent == 0.5
        assert config.enable_capital_protection is False


@pytest.mark.unit
class TestConvertLegacyPredictions:
    """Test convert_legacy_predictions function."""

    def test_convert_with_volatility(self):
        """Test conversion with volatility provided."""
        direction_probs = np.array([0.2, 0.3, 0.5])  # BEAR, SIDEWAYS, BULL
        volatility = 0.002
        entry_price = 1.1000

        result = convert_legacy_predictions(direction_probs, volatility, entry_price)

        assert 'direction_probs' in result
        assert 'volatility' in result
        assert 'quantiles' in result
        assert np.array_equal(result['direction_probs'], direction_probs)
        assert result['volatility'][0] == volatility
        assert len(result['quantiles']) == 5

    def test_convert_without_volatility(self):
        """Test conversion without volatility (uses default)."""
        direction_probs = np.array([0.2, 0.3, 0.5])
        entry_price = 1.1000

        result = convert_legacy_predictions(direction_probs, entry_price=entry_price)

        assert result['volatility'][0] == 0.001  # Default
        assert len(result['quantiles']) == 5

    def test_quantiles_structure(self):
        """Test quantiles have expected structure."""
        direction_probs = np.array([0.2, 0.3, 0.5])
        volatility = 0.001

        result = convert_legacy_predictions(direction_probs, volatility)

        quantiles = result['quantiles']
        # Q5 should be negative, Q95 should be positive
        assert quantiles[0] < 0  # Q5
        assert quantiles[-1] > 0  # Q95
        # Q50 should be near zero
        assert abs(quantiles[2]) < 0.0001  # Q50


@pytest.mark.unit
class TestEnhancedDecisionEngine:
    """Test EnhancedDecisionEngine class."""

    @pytest.fixture
    def mock_risk_components(self):
        """Create mocks for risk management components."""
        with patch('trading.decision_engine.SLTPCalculator') as MockSLTP, \
             patch('trading.decision_engine.PositionSizingCalculator') as MockPos, \
             patch('trading.decision_engine.TradeGatekeeper') as MockGate, \
             patch('trading.decision_engine.RegimeDetector') as MockRegime, \
             patch('trading.decision_engine.CapitalProtector') as MockProtector:
            
            # Setup mocks
            mock_sltp = Mock()
            mock_pos = Mock()
            mock_gate = Mock()
            mock_regime = Mock()
            mock_protector = Mock()
            
            MockSLTP.return_value = mock_sltp
            MockPos.return_value = mock_pos
            MockGate.return_value = mock_gate
            MockRegime.return_value = mock_regime
            MockProtector.return_value = mock_protector
            
            yield {
                'sltp': mock_sltp,
                'pos': mock_pos,
                'gate': mock_gate,
                'regime': mock_regime,
                'protector': mock_protector
            }

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data DataFrame."""
        dates = pd.date_range('2024-01-01', periods=100, freq='h')
        return pd.DataFrame({
            'time': dates,
            'open': [1.1000] * 100,
            'high': [1.1010] * 100,
            'low': [1.0990] * 100,
            'close': [1.1005] * 100,
            'volume': [1000] * 100,
            'atr': [0.001] * 100
        })

    def test_init_default(self, mock_risk_components):
        """Test default initialization."""
        engine = EnhancedDecisionEngine()

        assert engine.config.profile == 'INTRADAY'
        assert engine.sltp_calculator is not None
        assert engine.position_calculator is not None
        assert engine.gatekeeper is not None
        assert engine._initialized is False

    def test_init_with_config(self, mock_risk_components):
        """Test initialization with custom config."""
        config = DecisionEngineConfig(profile='SWING', min_direction_confidence=0.70)
        engine = EnhancedDecisionEngine(config)

        assert engine.config.profile == 'SWING'
        assert engine.config.min_direction_confidence == 0.70

    def test_init_with_capital_protection_disabled(self, mock_risk_components):
        """Test initialization without capital protection."""
        config = DecisionEngineConfig(enable_capital_protection=False)
        engine = EnhancedDecisionEngine(config)

        assert engine.capital_protector is None

    def test_initialize(self, mock_risk_components):
        """Test engine initialization."""
        engine = EnhancedDecisionEngine()
        mock_risk_components['protector'].initialize = Mock()

        engine.initialize(starting_balance=10000)

        assert engine._initialized is True
        assert engine._current_balance == 10000.0
        mock_risk_components['protector'].initialize.assert_called_once_with(10000)

    def test_initialize_without_protection(self, mock_risk_components):
        """Test initialization without capital protection."""
        config = DecisionEngineConfig(enable_capital_protection=False)
        engine = EnhancedDecisionEngine(config)

        engine.initialize(starting_balance=10000)

        assert engine._current_balance == 10000.0

    def test_record_trade_result(self, mock_risk_components):
        """Test recording trade results."""
        engine = EnhancedDecisionEngine()
        engine._current_balance = 10000
        mock_risk_components['protector'].record_trade = Mock()

        engine.record_trade_result(pnl=100, is_win=True, size=0.1)

        assert engine._current_balance == 10100.0
        mock_risk_components['protector'].record_trade.assert_called_once_with(100, True, 0.1)

    def test_evaluate_no_predictions(self, mock_risk_components, sample_market_data):
        """Test evaluation with no predictions."""
        engine = EnhancedDecisionEngine()
        engine.initialize(10000)

        predictions = {}
        decision = engine.evaluate(
            predictions=predictions,
            entry_price=1.1000,
            pair='EURUSD',
            account_balance=10000,
            market_data=sample_market_data
        )

        assert decision.should_trade is False
        assert len(decision.rejection_reasons) > 0
        assert "No direction predictions" in decision.rejection_reasons[0]

    def test_evaluate_low_confidence(self, mock_risk_components, sample_market_data):
        """Test evaluation with low confidence."""
        engine = EnhancedDecisionEngine()
        engine.initialize(10000)
        mock_risk_components['regime'].detect.return_value = Mock(value='NORMAL')

        predictions = {
            'direction_probs': np.array([0.2, 0.3, 0.45])  # Low confidence
        }

        decision = engine.evaluate(
            predictions=predictions,
            entry_price=1.1000,
            pair='EURUSD',
            account_balance=10000,
            market_data=sample_market_data
        )

        assert decision.should_trade is False
        assert any("Low confidence" in r for r in decision.rejection_reasons)

    def test_evaluate_sideways_signal(self, mock_risk_components, sample_market_data):
        """Test evaluation with sideways prediction."""
        engine = EnhancedDecisionEngine()
        engine.initialize(10000)
        mock_risk_components['regime'].detect.return_value = Mock(value='NORMAL')

        predictions = {
            'direction_probs': np.array([0.1, 0.8, 0.1])  # Sideways dominant
        }

        decision = engine.evaluate(
            predictions=predictions,
            entry_price=1.1000,
            pair='EURUSD',
            account_balance=10000,
            market_data=sample_market_data
        )

        assert decision.should_trade is False
        assert any("Sideways" in r for r in decision.rejection_reasons)

    def test_evaluate_capital_protection_block(self, mock_risk_components, sample_market_data):
        """Test evaluation blocked by capital protection."""
        engine = EnhancedDecisionEngine()
        engine.initialize(10000)
        
        # Mock capital protection to block
        mock_risk_components['protector'].check_trade.return_value = {
            'allowed': False,
            'reason': 'Daily loss limit reached'
        }

        predictions = {
            'direction_probs': np.array([0.1, 0.2, 0.7])  # Strong BULL
        }

        decision = engine.evaluate(
            predictions=predictions,
            entry_price=1.1000,
            pair='EURUSD',
            account_balance=10000,
            market_data=sample_market_data
        )

        assert decision.should_trade is False
        assert any("Capital protection" in r for r in decision.rejection_reasons)

    def test_evaluate_full_pipeline_buy(self, mock_risk_components, sample_market_data):
        """Test full evaluation pipeline for BUY signal."""
        engine = EnhancedDecisionEngine()
        engine.initialize(10000)
        
        # Setup mocks
        from risk_management.phase2_risk_calc import MarketRegime, TradeDirection
        
        mock_regime = Mock()
        mock_regime.value = 'NORMAL'
        mock_risk_components['regime'].detect.return_value = mock_regime
        
        # Mock SLTP result
        sltp_result = Mock()
        sltp_result.stop_loss = 1.0950
        sltp_result.take_profit = 1.1100
        sltp_result.sl_pips = 50.0
        sltp_result.tp_pips = 100.0
        sltp_result.risk_reward_ratio = 2.0
        mock_risk_components['sltp'].calculate.return_value = sltp_result
        
        # Mock position sizing result
        pos_result = Mock()
        pos_result.position_size = 0.1
        pos_result.position_units = 10000
        pos_result.risk_amount = 100.0
        pos_result.risk_percent = 1.0
        mock_risk_components['pos'].calculate.return_value = pos_result
        
        # Mock gatekeeper approval
        mock_risk_components['gate'].validate_trade.return_value = {
            'allowed': True,
            'violations': []
        }
        
        # Mock capital protection approval
        mock_risk_components['protector'].check_trade.return_value = {
            'allowed': True,
            'protection_level': 'normal'
        }

        predictions = {
            'direction_probs': np.array([0.1, 0.2, 0.7]),  # Strong BULL
            'volatility': np.array([0.001]),
            'quantiles': np.array([-0.002, -0.001, 0.0, 0.001, 0.002])
        }

        decision = engine.evaluate(
            predictions=predictions,
            entry_price=1.1000,
            pair='EURUSD',
            account_balance=10000,
            market_data=sample_market_data
        )

        assert decision.should_trade is True
        assert decision.direction == 'BUY'
        assert decision.stop_loss == 1.0950
        assert decision.take_profit == 1.1100
        assert decision.position_size == 0.1

    def test_evaluate_gatekeeper_rejection(self, mock_risk_components, sample_market_data):
        """Test evaluation rejected by gatekeeper."""
        engine = EnhancedDecisionEngine()
        engine.initialize(10000)
        
        from risk_management.phase2_risk_calc import MarketRegime
        
        mock_regime = Mock()
        mock_regime.value = 'NORMAL'
        mock_risk_components['regime'].detect.return_value = mock_regime
        
        sltp_result = Mock()
        sltp_result.risk_reward_ratio = 2.0
        sltp_result.stop_loss = 1.0950
        sltp_result.take_profit = 1.1100
        sltp_result.sl_pips = 50.0
        sltp_result.tp_pips = 100.0
        mock_risk_components['sltp'].calculate.return_value = sltp_result
        
        pos_result = Mock()
        pos_result.position_size = 0.1
        pos_result.position_units = 10000
        pos_result.risk_amount = 100.0
        pos_result.risk_percent = 1.0
        mock_risk_components['pos'].calculate.return_value = pos_result
        
        # Gatekeeper rejects
        mock_risk_components['gate'].validate_trade.return_value = {
            'allowed': False,
            'violations': [{'message': 'Spread too wide'}]
        }
        
        mock_risk_components['protector'].check_trade.return_value = {
            'allowed': True
        }

        predictions = {
            'direction_probs': np.array([0.1, 0.2, 0.7])
        }

        decision = engine.evaluate(
            predictions=predictions,
            entry_price=1.1000,
            pair='EURUSD',
            account_balance=10000,
            market_data=sample_market_data
        )

        assert decision.should_trade is False
        assert len(decision.rule_violations) > 0

    def test_get_protection_status(self, mock_risk_components):
        """Test getting protection status."""
        engine = EnhancedDecisionEngine()
        engine.initialize(10000)
        
        mock_state = Mock()
        mock_state.level.value = 'normal'
        mock_state.action.value = 'allow'
        mock_state.size_multiplier = 1.0
        mock_state.trigger_reason = None
        
        mock_metrics = Mock()
        mock_metrics.current_balance = 10000
        mock_metrics.peak_balance = 10500
        mock_metrics.current_drawdown_pct = 0.05
        mock_metrics.daily_pnl = 100
        mock_metrics.weekly_pnl = 200
        mock_metrics.consecutive_losses = 0
        mock_metrics.recent_win_rate = 0.6
        
        mock_risk_components['protector'].get_state.return_value = mock_state
        mock_risk_components['protector'].get_metrics.return_value = mock_metrics

        status = engine.get_protection_status()

        assert status['enabled'] is True
        assert status['level'] == 'normal'
        assert status['metrics']['balance'] == 10000

    def test_get_protection_status_disabled(self, mock_risk_components):
        """Test protection status when disabled."""
        config = DecisionEngineConfig(enable_capital_protection=False)
        engine = EnhancedDecisionEngine(config)

        status = engine.get_protection_status()

        assert status['enabled'] is False

    def test_reset_daily_protection(self, mock_risk_components):
        """Test resetting daily protection."""
        engine = EnhancedDecisionEngine()
        mock_risk_components['protector'].reset_daily = Mock()

        engine.reset_daily_protection()

        mock_risk_components['protector'].reset_daily.assert_called_once()

    def test_reset_weekly_protection(self, mock_risk_components):
        """Test resetting weekly protection."""
        engine = EnhancedDecisionEngine()
        mock_risk_components['protector'].reset_weekly = Mock()

        engine.reset_weekly_protection()

        mock_risk_components['protector'].reset_weekly.assert_called_once()

    def test_estimate_spread(self, mock_risk_components):
        """Test spread estimation."""
        engine = EnhancedDecisionEngine()

        spread_eurusd = engine._estimate_spread('EURUSD')
        spread_gbpusd = engine._estimate_spread('GBPUSD')
        spread_unknown = engine._estimate_spread('UNKNOWN')

        assert spread_eurusd == 1.0
        assert spread_gbpusd == 1.5
        assert spread_unknown == 2.0  # Default


@pytest.mark.unit
class TestMTFDecisionEngine:
    """Test MTFDecisionEngine alias (backward compatibility)."""

    def test_alias_exists(self):
        """Test that MTFDecisionEngine alias exists."""
        assert MTFDecisionEngine == EnhancedDecisionEngine

