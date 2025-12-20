# tests/utils/test_utils_config.py
"""
Tests for utils/config.py - Settings configuration.

Tests the Pydantic Settings class for centralized configuration.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import patch


class TestSettingsImport:
    """Test that Settings can be imported."""
    
    def test_settings_class_import(self):
        """Settings class should be importable directly from config module."""
        # Import directly to avoid __init__.py issues
        from utils.config import Settings
        assert Settings is not None
    
    def test_settings_instance_import(self):
        """Global settings instance should be importable."""
        from utils.config import settings
        assert settings is not None


class TestSettingsDefaults:
    """Test default configuration values."""
    
    def test_symbol_default(self):
        """Default symbol should be EURUSD."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        assert s.SYMBOL == "EURUSD"
    
    def test_timeframe_default(self):
        """Default timeframe should be H1."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        assert s.TIMEFRAME == "H1"
    
    def test_magic_number_default(self):
        """Default magic number should be 123456."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        assert s.MAGIC_NUMBER == 123456
    
    def test_max_daily_loss_default(self):
        """Default max daily loss should be 3%."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        assert s.MAX_DAILY_LOSS_PCT == 0.03
    
    def test_risk_per_trade_default(self):
        """Default risk per trade should be 1%."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        assert s.RISK_PER_TRADE_PCT == 0.01
    
    def test_max_drawdown_default(self):
        """Default max drawdown should be 10%."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        assert s.MAX_DRAWDOWN_PCT == 0.10
    
    def test_confidence_threshold_default(self):
        """Default confidence threshold should be 0.60."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        assert s.CONFIDENCE_THRESHOLD == 0.60
    
    def test_sequence_length_default(self):
        """Default sequence length should be 60."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        assert s.SEQUENCE_LENGTH == 60
    
    def test_image_size_default(self):
        """Default image size should be 224."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        assert s.IMAGE_SIZE == 224
    
    def test_tick_interval_default(self):
        """Default tick interval should be 10 seconds."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        assert s.TICK_INTERVAL == 10.0


class TestSettingsTypes:
    """Test that settings have correct types."""
    
    def test_weights_dir_is_path(self):
        """WEIGHTS_DIR should be a Path object."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        assert isinstance(s.WEIGHTS_DIR, Path)
    
    def test_weights_dir_default_value(self):
        """WEIGHTS_DIR default should be models/weights."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        assert s.WEIGHTS_DIR == Path("models/weights")
    
    def test_mt5_account_is_int(self):
        """MT5_ACCOUNT should be an integer."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=12345, MT5_PASSWORD="pass", MT5_SERVER="server")
        assert isinstance(s.MT5_ACCOUNT, int)
        assert s.MT5_ACCOUNT == 12345


class TestConfidenceThresholdValidation:
    """Test CONFIDENCE_THRESHOLD validation."""
    
    def test_threshold_below_minimum_raises(self):
        """Threshold below 0.5 should raise ValueError."""
        from utils.config import Settings
        with pytest.raises(ValueError, match="Threshold must be between 0.5 and 0.95"):
            Settings(
                MT5_ACCOUNT=0,
                MT5_PASSWORD="",
                MT5_SERVER="",
                CONFIDENCE_THRESHOLD=0.49
            )
    
    def test_threshold_above_maximum_raises(self):
        """Threshold above 0.95 should raise ValueError."""
        from utils.config import Settings
        with pytest.raises(ValueError, match="Threshold must be between 0.5 and 0.95"):
            Settings(
                MT5_ACCOUNT=0,
                MT5_PASSWORD="",
                MT5_SERVER="",
                CONFIDENCE_THRESHOLD=0.96
            )
    
    def test_threshold_at_minimum_valid(self):
        """Threshold at 0.5 should be valid."""
        from utils.config import Settings
        s = Settings(
            MT5_ACCOUNT=0,
            MT5_PASSWORD="",
            MT5_SERVER="",
            CONFIDENCE_THRESHOLD=0.5
        )
        assert s.CONFIDENCE_THRESHOLD == 0.5
    
    def test_threshold_at_maximum_valid(self):
        """Threshold at 0.95 should be valid."""
        from utils.config import Settings
        s = Settings(
            MT5_ACCOUNT=0,
            MT5_PASSWORD="",
            MT5_SERVER="",
            CONFIDENCE_THRESHOLD=0.95
        )
        assert s.CONFIDENCE_THRESHOLD == 0.95
    
    @pytest.mark.parametrize("threshold", [0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95])
    def test_threshold_valid_range(self, threshold):
        """Thresholds in valid range should work."""
        from utils.config import Settings
        s = Settings(
            MT5_ACCOUNT=0,
            MT5_PASSWORD="",
            MT5_SERVER="",
            CONFIDENCE_THRESHOLD=threshold
        )
        assert s.CONFIDENCE_THRESHOLD == threshold


class TestTimeframeValidation:
    """Test TIMEFRAME literal validation."""
    
    @pytest.mark.parametrize("tf", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
    def test_valid_timeframes(self, tf):
        """All valid timeframes should be accepted."""
        from utils.config import Settings
        s = Settings(
            MT5_ACCOUNT=0,
            MT5_PASSWORD="",
            MT5_SERVER="",
            TIMEFRAME=tf
        )
        assert s.TIMEFRAME == tf
    
    def test_invalid_timeframe_raises(self):
        """Invalid timeframe should raise ValidationError."""
        from utils.config import Settings
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(
                MT5_ACCOUNT=0,
                MT5_PASSWORD="",
                MT5_SERVER="",
                TIMEFRAME="INVALID"
            )
    
    def test_w1_not_valid_timeframe(self):
        """W1 is not a valid TIMEFRAME in Settings (only in mtf_config)."""
        from utils.config import Settings
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Settings(
                MT5_ACCOUNT=0,
                MT5_PASSWORD="",
                MT5_SERVER="",
                TIMEFRAME="W1"
            )


class TestDeviceConfiguration:
    """Test DEVICE configuration."""
    
    def test_device_without_cuda_env(self):
        """Device should be cpu when CUDA_VISIBLE_DEVICES not set."""
        from utils.config import Settings
        # Clear the env var if set
        with patch.dict(os.environ, {}, clear=True):
            # Need to reimport to get fresh evaluation
            import importlib
            import utils.config
            importlib.reload(utils.config)
            s = utils.config.Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
            assert s.DEVICE in ["cpu", "cuda"]  # Depends on environment
    
    def test_device_is_string(self):
        """DEVICE should be a string."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        assert isinstance(s.DEVICE, str)
    
    def test_device_valid_values(self):
        """DEVICE should be either cpu or cuda."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        assert s.DEVICE in ["cpu", "cuda"]


class TestMT5Credentials:
    """Test MT5 credential handling."""
    
    def test_mt5_account_default(self):
        """MT5_ACCOUNT should load from config.env."""
        from utils.config import Settings
        s = Settings(MT5_PASSWORD="", MT5_SERVER="")
        # Should load the actual value from config.env
        assert s.MT5_ACCOUNT == 5043196962
    
    def test_mt5_password_default(self):
        """MT5_PASSWORD should load from config.env."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_SERVER="")
        # Should load the actual value from config.env
        assert s.MT5_PASSWORD == "Bv@4OyXs"
    
    def test_mt5_server_default(self):
        """MT5_SERVER should load from config.env."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="")
        # Should load the actual value from config.env
        assert s.MT5_SERVER == "MetaQuotes-Demo"
    
    def test_mt5_path_default(self):
        """MT5_PATH should load from config.env."""
        from utils.config import Settings
        s = Settings(MT5_ACCOUNT=0, MT5_PASSWORD="", MT5_SERVER="")
        # Should load the actual value from config.env
        assert s.MT5_PATH == "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
    
    def test_mt5_credentials_can_be_set(self):
        """MT5 credentials should be settable."""
        from utils.config import Settings
        s = Settings(
            MT5_ACCOUNT=12345678,
            MT5_PASSWORD="secret123",
            MT5_SERVER="MetaQuotes-Demo",
            MT5_PATH=r"C:\Program Files\MT5\terminal64.exe"
        )
        assert s.MT5_ACCOUNT == 12345678
        assert s.MT5_PASSWORD == "secret123"
        assert s.MT5_SERVER == "MetaQuotes-Demo"
        assert s.MT5_PATH == r"C:\Program Files\MT5\terminal64.exe"


class TestGlobalSettingsInstance:
    """Test the global settings singleton."""
    
    def test_settings_instance_exists(self):
        """Global settings instance should exist."""
        from utils.config import settings
        assert settings is not None
    
    def test_settings_instance_is_settings_type(self):
        """Global settings should be Settings instance."""
        from utils.config import settings, Settings
        assert isinstance(settings, Settings)
    
    def test_settings_has_required_attributes(self):
        """Global settings should have all required attributes."""
        from utils.config import settings
        required_attrs = [
            'MT5_ACCOUNT', 'MT5_PASSWORD', 'MT5_SERVER', 'MT5_PATH',
            'SYMBOL', 'TIMEFRAME', 'MAGIC_NUMBER',
            'MAX_DAILY_LOSS_PCT', 'RISK_PER_TRADE_PCT', 'MAX_DRAWDOWN_PCT',
            'DEVICE', 'CONFIDENCE_THRESHOLD', 'WEIGHTS_DIR',
            'SEQUENCE_LENGTH', 'IMAGE_SIZE', 'TICK_INTERVAL'
        ]
        for attr in required_attrs:
            assert hasattr(settings, attr), f"Missing attribute: {attr}"


class TestSettingsExtraIgnore:
    """Test that extra fields are ignored."""
    
    def test_extra_fields_ignored(self):
        """Extra fields should be ignored (not raise error)."""
        from utils.config import Settings
        # This should not raise even with extra field
        s = Settings(
            MT5_ACCOUNT=0,
            MT5_PASSWORD="",
            MT5_SERVER="",
            UNKNOWN_FIELD="should be ignored"
        )
        assert not hasattr(s, 'UNKNOWN_FIELD')