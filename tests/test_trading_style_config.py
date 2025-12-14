# tests/test_trading_style_config.py
"""
Unit tests for trading/style_config.py - Trading style configurations.
"""

import pytest
from trading.style_config import (
    TradingStyle, StyleConfig, SCALP_CONFIG, INTRADAY_CONFIG, SWING_CONFIG,
    DEFAULT_STYLE_CONFIGS, OrchestratorConfig,
    get_timeframe_minutes, get_all_required_timeframes, style_for_timeframe
)


@pytest.mark.unit
class TestTradingStyle:
    """Test TradingStyle enum."""

    def test_style_values(self):
        """Test trading style enum values."""
        assert TradingStyle.SCALP.value == "scalp"
        assert TradingStyle.INTRADAY.value == "intraday"
        assert TradingStyle.SWING.value == "swing"


@pytest.mark.unit
class TestStyleConfig:
    """Test StyleConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = StyleConfig(
            name="Test",
            style=TradingStyle.INTRADAY
        )

        assert config.name == "Test"
        assert config.style == TradingStyle.INTRADAY
        assert config.enabled is True
        assert config.primary_timeframe == "H1"
        assert config.min_confidence == 0.65
        assert config.risk_per_trade_pct == 0.01
        assert config.max_positions == 2

    def test_custom_values(self):
        """Test custom configuration."""
        config = StyleConfig(
            name="Custom",
            style=TradingStyle.SCALP,
            enabled=False,
            primary_timeframe="M5",
            min_confidence=0.80,
            risk_per_trade_pct=0.005,
            max_positions=5
        )

        assert config.name == "Custom"
        assert config.style == TradingStyle.SCALP
        assert config.enabled is False
        assert config.primary_timeframe == "M5"
        assert config.min_confidence == 0.80
        assert config.risk_per_trade_pct == 0.005
        assert config.max_positions == 5


@pytest.mark.unit
class TestScalpConfig:
    """Test SCALP_CONFIG preset."""

    def test_scalp_config_values(self):
        """Test scalping configuration values."""
        assert SCALP_CONFIG.name == "Scalper"
        assert SCALP_CONFIG.style == TradingStyle.SCALP
        assert SCALP_CONFIG.enabled is True
        assert SCALP_CONFIG.primary_timeframe == "M15"
        assert "M5" in SCALP_CONFIG.secondary_timeframes
        assert SCALP_CONFIG.min_confidence == 0.75
        assert SCALP_CONFIG.risk_per_trade_pct == 0.005
        assert SCALP_CONFIG.max_positions == 3
        assert SCALP_CONFIG.use_trailing_stop is True
        assert SCALP_CONFIG.allow_counter_trend is False


@pytest.mark.unit
class TestIntradayConfig:
    """Test INTRADAY_CONFIG preset."""

    def test_intraday_config_values(self):
        """Test intraday configuration values."""
        assert INTRADAY_CONFIG.name == "Intraday"
        assert INTRADAY_CONFIG.style == TradingStyle.INTRADAY
        assert INTRADAY_CONFIG.primary_timeframe == "H1"
        assert "M15" in INTRADAY_CONFIG.secondary_timeframes
        assert INTRADAY_CONFIG.min_confidence == 0.65
        assert INTRADAY_CONFIG.risk_per_trade_pct == 0.01
        assert INTRADAY_CONFIG.max_positions == 2
        assert INTRADAY_CONFIG.allow_counter_trend is True


@pytest.mark.unit
class TestSwingConfig:
    """Test SWING_CONFIG preset."""

    def test_swing_config_values(self):
        """Test swing configuration values."""
        assert SWING_CONFIG.name == "Swing"
        assert SWING_CONFIG.style == TradingStyle.SWING
        assert SWING_CONFIG.primary_timeframe == "H4"
        assert "H1" in SWING_CONFIG.secondary_timeframes
        assert SWING_CONFIG.min_confidence == 0.60
        assert SWING_CONFIG.risk_per_trade_pct == 0.015
        assert SWING_CONFIG.max_positions == 2
        assert SWING_CONFIG.min_risk_reward == 2.0


@pytest.mark.unit
class TestDefaultStyleConfigs:
    """Test DEFAULT_STYLE_CONFIGS dictionary."""

    def test_all_styles_present(self):
        """Test all trading styles are present."""
        assert TradingStyle.SCALP in DEFAULT_STYLE_CONFIGS
        assert TradingStyle.INTRADAY in DEFAULT_STYLE_CONFIGS
        assert TradingStyle.SWING in DEFAULT_STYLE_CONFIGS

    def test_config_objects(self):
        """Test config objects are correct instances."""
        assert DEFAULT_STYLE_CONFIGS[TradingStyle.SCALP] == SCALP_CONFIG
        assert DEFAULT_STYLE_CONFIGS[TradingStyle.INTRADAY] == INTRADAY_CONFIG
        assert DEFAULT_STYLE_CONFIGS[TradingStyle.SWING] == SWING_CONFIG


@pytest.mark.unit
class TestOrchestratorConfig:
    """Test OrchestratorConfig dataclass."""

    def test_default_values(self):
        """Test default orchestrator configuration."""
        config = OrchestratorConfig()

        assert config.symbol == "EURUSD"
        assert config.max_total_positions == 5
        assert config.max_daily_loss_pct == 0.03
        assert config.max_drawdown_pct == 0.10
        assert config.prevent_opposing_positions is True

    def test_get_style_config(self):
        """Test getting style configuration."""
        config = OrchestratorConfig()

        scalp_config = config.get_style_config(TradingStyle.SCALP)
        assert scalp_config == SCALP_CONFIG

        intraday_config = config.get_style_config(TradingStyle.INTRADAY)
        assert intraday_config == INTRADAY_CONFIG

    def test_get_enabled_styles(self):
        """Test getting enabled styles."""
        config = OrchestratorConfig()

        enabled = config.get_enabled_styles()
        assert TradingStyle.SCALP in enabled
        assert TradingStyle.INTRADAY in enabled
        assert TradingStyle.SWING in enabled

    def test_get_enabled_styles_with_disabled(self):
        """Test getting enabled styles when some are disabled."""
        config = OrchestratorConfig()
        config.styles[TradingStyle.SCALP].enabled = False

        enabled = config.get_enabled_styles()
        assert TradingStyle.SCALP not in enabled
        assert TradingStyle.INTRADAY in enabled
        assert TradingStyle.SWING in enabled

    def test_get_magic_number(self):
        """Test getting magic numbers for styles."""
        config = OrchestratorConfig()

        scalp_magic = config.get_magic_number(TradingStyle.SCALP)
        intraday_magic = config.get_magic_number(TradingStyle.INTRADAY)
        swing_magic = config.get_magic_number(TradingStyle.SWING)

        assert scalp_magic == config.magic_number_base + 1
        assert intraday_magic == config.magic_number_base + 2
        assert swing_magic == config.magic_number_base + 3
        assert scalp_magic != intraday_magic
        assert intraday_magic != swing_magic


@pytest.mark.unit
class TestTimeframeUtilities:
    """Test timeframe utility functions."""

    def test_get_timeframe_minutes_m1(self):
        """Test M1 timeframe conversion."""
        assert get_timeframe_minutes("M1") == 1

    def test_get_timeframe_minutes_m5(self):
        """Test M5 timeframe conversion."""
        assert get_timeframe_minutes("M5") == 5

    def test_get_timeframe_minutes_h1(self):
        """Test H1 timeframe conversion."""
        assert get_timeframe_minutes("H1") == 60

    def test_get_timeframe_minutes_h4(self):
        """Test H4 timeframe conversion."""
        assert get_timeframe_minutes("H4") == 240

    def test_get_timeframe_minutes_d1(self):
        """Test D1 timeframe conversion."""
        assert get_timeframe_minutes("D1") == 1440

    def test_get_timeframe_minutes_unknown(self):
        """Test unknown timeframe defaults to 60."""
        assert get_timeframe_minutes("UNKNOWN") == 60

    def test_get_timeframe_minutes_case_insensitive(self):
        """Test timeframe is case insensitive."""
        assert get_timeframe_minutes("h1") == 60
        assert get_timeframe_minutes("H1") == 60

    def test_get_all_required_timeframes_default(self):
        """Test getting all required timeframes with default config."""
        config = OrchestratorConfig()

        timeframes = get_all_required_timeframes(config)

        # Should include all timeframes from all enabled styles
        assert "M5" in timeframes  # From SCALP
        assert "M15" in timeframes  # From SCALP and SWING
        assert "H1" in timeframes  # From all styles
        assert "H4" in timeframes  # From SWING
        assert "D1" in timeframes  # From SWING

    def test_get_all_required_timeframes_sorted(self):
        """Test timeframes are sorted by granularity."""
        config = OrchestratorConfig()

        timeframes = get_all_required_timeframes(config)

        # Check sorting (smallest first)
        for i in range(len(timeframes) - 1):
            current_minutes = get_timeframe_minutes(timeframes[i])
            next_minutes = get_timeframe_minutes(timeframes[i + 1])
            assert current_minutes <= next_minutes

    def test_style_for_timeframe_scalp(self):
        """Test style suggestion for scalping timeframes."""
        assert style_for_timeframe("M1") == TradingStyle.SCALP
        assert style_for_timeframe("M5") == TradingStyle.SCALP
        assert style_for_timeframe("M15") == TradingStyle.SCALP

    def test_style_for_timeframe_intraday(self):
        """Test style suggestion for intraday timeframes."""
        assert style_for_timeframe("M30") == TradingStyle.INTRADAY
        assert style_for_timeframe("H1") == TradingStyle.INTRADAY

    def test_style_for_timeframe_swing(self):
        """Test style suggestion for swing timeframes."""
        assert style_for_timeframe("H4") == TradingStyle.SWING
        assert style_for_timeframe("D1") == TradingStyle.SWING

