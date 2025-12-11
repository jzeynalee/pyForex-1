# tests/utils/test_mtf_config.py
"""
Tests for utils/mtf_config.py - Multi-Timeframe Configuration.

Tests MTFProfile, preset profiles, and utility functions.
"""

import pytest
from typing import Dict


class TestTimeframeEnum:
    """Test Timeframe enumeration."""
    
    def test_timeframe_import(self):
        """Timeframe enum should be importable."""
        from utils.mtf_config import Timeframe
        assert Timeframe is not None
    
    @pytest.mark.parametrize("tf_name,tf_value", [
        ("M1", "M1"),
        ("M5", "M5"),
        ("M15", "M15"),
        ("M30", "M30"),
        ("H1", "H1"),
        ("H4", "H4"),
        ("D1", "D1"),
        ("W1", "W1"),
    ])
    def test_timeframe_values(self, tf_name, tf_value):
        """Each timeframe should have correct value."""
        from utils.mtf_config import Timeframe
        tf = getattr(Timeframe, tf_name)
        assert tf.value == tf_value
    
    def test_timeframe_count(self):
        """Should have exactly 8 timeframes."""
        from utils.mtf_config import Timeframe
        assert len(list(Timeframe)) == 8
    
    def test_timeframe_string_conversion(self):
        """Timeframe value should be string."""
        from utils.mtf_config import Timeframe
        assert Timeframe.H1.value == "H1"
        assert isinstance(Timeframe.H1.value, str)


class TestTFMinutes:
    """Test TF_MINUTES mapping."""
    
    def test_tf_minutes_import(self):
        """TF_MINUTES should be importable."""
        from utils.mtf_config import TF_MINUTES
        assert TF_MINUTES is not None
        assert isinstance(TF_MINUTES, dict)
    
    @pytest.mark.parametrize("tf_name,expected_minutes", [
        ("M1", 1),
        ("M5", 5),
        ("M15", 15),
        ("M30", 30),
        ("H1", 60),
        ("H4", 240),
        ("D1", 1440),
        ("W1", 10080),
    ])
    def test_tf_minutes_values(self, tf_name, expected_minutes):
        """Each timeframe should map to correct minutes."""
        from utils.mtf_config import TF_MINUTES, Timeframe
        tf = Timeframe(tf_name)
        assert TF_MINUTES[tf] == expected_minutes
    
    def test_tf_minutes_covers_all_timeframes(self):
        """TF_MINUTES should have entry for every Timeframe."""
        from utils.mtf_config import TF_MINUTES, Timeframe
        for tf in Timeframe:
            assert tf in TF_MINUTES, f"Missing {tf} in TF_MINUTES"


class TestMTFProfile:
    """Test MTFProfile dataclass."""
    
    def test_mtf_profile_import(self):
        """MTFProfile should be importable."""
        from utils.mtf_config import MTFProfile
        assert MTFProfile is not None
    
    def test_profile_creation_with_defaults(self):
        """Profile can be created with minimal args."""
        from utils.mtf_config import MTFProfile, Timeframe
        profile = MTFProfile(name="TEST")
        assert profile.name == "TEST"
        assert profile.description == ""
        assert profile.primary_tf == Timeframe.H1
    
    def test_profile_default_timeframes(self):
        """Default timeframes should be M15, H1, H4."""
        from utils.mtf_config import MTFProfile, Timeframe
        profile = MTFProfile(name="TEST")
        assert profile.timeframes == (Timeframe.M15, Timeframe.H1, Timeframe.H4)
    
    def test_profile_custom_timeframes(self):
        """Custom timeframes can be specified."""
        from utils.mtf_config import MTFProfile, Timeframe
        profile = MTFProfile(
            name="CUSTOM",
            timeframes=(Timeframe.M5, Timeframe.M15, Timeframe.H1)
        )
        assert profile.timeframes == (Timeframe.M5, Timeframe.M15, Timeframe.H1)
    
    def test_profile_auto_weights(self):
        """Weights should be auto-generated if not provided."""
        from utils.mtf_config import MTFProfile, Timeframe
        profile = MTFProfile(
            name="TEST",
            timeframes=(Timeframe.M15, Timeframe.H1, Timeframe.H4)
        )
        # Should have equal weights for 3 timeframes
        assert len(profile.weights) == 3
        for tf in profile.timeframes:
            assert pytest.approx(profile.weights[tf.value], rel=0.01) == 1/3
    
    def test_profile_custom_weights(self):
        """Custom weights can be specified."""
        from utils.mtf_config import MTFProfile, Timeframe
        custom_weights = {"M15": 0.2, "H1": 0.5, "H4": 0.3}
        profile = MTFProfile(
            name="CUSTOM",
            timeframes=(Timeframe.M15, Timeframe.H1, Timeframe.H4),
            weights=custom_weights
        )
        assert profile.weights == custom_weights
    
    def test_profile_timeframe_strings_property(self):
        """timeframe_strings property should return list of strings."""
        from utils.mtf_config import MTFProfile, Timeframe
        profile = MTFProfile(
            name="TEST",
            timeframes=(Timeframe.M5, Timeframe.M15, Timeframe.H1)
        )
        assert profile.timeframe_strings == ["M5", "M15", "H1"]
    
    def test_profile_higher_tf_property(self):
        """higher_tf property should return highest timeframe."""
        from utils.mtf_config import MTFProfile, Timeframe
        profile = MTFProfile(
            name="TEST",
            timeframes=(Timeframe.M5, Timeframe.M15, Timeframe.H1)
        )
        assert profile.higher_tf == Timeframe.H1
    
    def test_profile_lower_tf_property(self):
        """lower_tf property should return lowest timeframe."""
        from utils.mtf_config import MTFProfile, Timeframe
        profile = MTFProfile(
            name="TEST",
            timeframes=(Timeframe.M5, Timeframe.M15, Timeframe.H1)
        )
        assert profile.lower_tf == Timeframe.M5
    
    def test_profile_get_weight_method(self):
        """get_weight method should return weight for timeframe."""
        from utils.mtf_config import MTFProfile, Timeframe
        profile = MTFProfile(
            name="TEST",
            timeframes=(Timeframe.M15, Timeframe.H1, Timeframe.H4),
            weights={"M15": 0.2, "H1": 0.5, "H4": 0.3}
        )
        assert profile.get_weight("H1") == 0.5
        assert profile.get_weight("M15") == 0.2
        assert profile.get_weight("H4") == 0.3
    
    def test_profile_get_weight_case_insensitive(self):
        """get_weight should be case-insensitive."""
        from utils.mtf_config import MTFProfile, Timeframe
        profile = MTFProfile(
            name="TEST",
            weights={"M15": 0.2, "H1": 0.5, "H4": 0.3}
        )
        assert profile.get_weight("h1") == 0.5
        assert profile.get_weight("H1") == 0.5
    
    def test_profile_get_weight_missing_returns_zero(self):
        """get_weight should return 0 for missing timeframe."""
        from utils.mtf_config import MTFProfile
        profile = MTFProfile(name="TEST")
        assert profile.get_weight("W1") == 0.0
    
    def test_profile_default_candle_counts(self):
        """Candle counts should be auto-generated if not provided."""
        from utils.mtf_config import MTFProfile, Timeframe
        profile = MTFProfile(
            name="TEST",
            timeframes=(Timeframe.M15, Timeframe.H1, Timeframe.H4)
        )
        # Default should be 200 for each
        assert profile.candle_counts == {"M15": 200, "H1": 200, "H4": 200}
    
    def test_profile_custom_candle_counts(self):
        """Custom candle counts can be specified."""
        from utils.mtf_config import MTFProfile, Timeframe
        custom_counts = {"M15": 150, "H1": 100, "H4": 50}
        profile = MTFProfile(
            name="TEST",
            timeframes=(Timeframe.M15, Timeframe.H1, Timeframe.H4),
            candle_counts=custom_counts
        )
        assert profile.candle_counts == custom_counts
    
    def test_profile_analysis_parameters(self):
        """Analysis parameters should have defaults."""
        from utils.mtf_config import MTFProfile
        profile = MTFProfile(name="TEST")
        assert profile.min_bars == 100
        assert profile.min_confluence_score == 0.60
        assert profile.require_higher_tf_alignment is True
    
    def test_profile_signal_parameters(self):
        """Signal parameters should have defaults."""
        from utils.mtf_config import MTFProfile
        profile = MTFProfile(name="TEST")
        assert profile.signal_threshold == 0.65
        assert profile.counter_trend_threshold == 0.85
    
    def test_profile_feature_parameters(self):
        """Feature parameters should have defaults."""
        from utils.mtf_config import MTFProfile
        profile = MTFProfile(name="TEST")
        assert profile.ema_periods == (20, 50, 200)
        assert profile.adx_period == 14
        assert profile.atr_period == 14


class TestPresetProfiles:
    """Test preset profile constants."""
    
    def test_scalp_profile_import(self):
        """SCALP_PROFILE should be importable."""
        from utils.mtf_config import SCALP_PROFILE
        assert SCALP_PROFILE is not None
    
    def test_scalp_profile_name(self):
        """SCALP_PROFILE should have correct name."""
        from utils.mtf_config import SCALP_PROFILE
        assert SCALP_PROFILE.name == "SCALP"
    
    def test_scalp_profile_timeframes(self):
        """SCALP_PROFILE should use M5, M15, H1."""
        from utils.mtf_config import SCALP_PROFILE, Timeframe
        assert SCALP_PROFILE.timeframes == (Timeframe.M5, Timeframe.M15, Timeframe.H1)
    
    def test_scalp_profile_primary_tf(self):
        """SCALP_PROFILE primary should be M15."""
        from utils.mtf_config import SCALP_PROFILE, Timeframe
        assert SCALP_PROFILE.primary_tf == Timeframe.M15
    
    def test_scalp_profile_weights(self):
        """SCALP_PROFILE should have specific weights."""
        from utils.mtf_config import SCALP_PROFILE
        assert SCALP_PROFILE.weights == {"M5": 0.25, "M15": 0.45, "H1": 0.30}
    
    def test_scalp_profile_higher_thresholds(self):
        """SCALP_PROFILE should have higher thresholds for fast trading."""
        from utils.mtf_config import SCALP_PROFILE
        assert SCALP_PROFILE.min_confluence_score == 0.70
        assert SCALP_PROFILE.signal_threshold == 0.70
        assert SCALP_PROFILE.counter_trend_threshold == 0.90
    
    def test_intraday_profile_import(self):
        """INTRADAY_PROFILE should be importable."""
        from utils.mtf_config import INTRADAY_PROFILE
        assert INTRADAY_PROFILE is not None
    
    def test_intraday_profile_name(self):
        """INTRADAY_PROFILE should have correct name."""
        from utils.mtf_config import INTRADAY_PROFILE
        assert INTRADAY_PROFILE.name == "INTRADAY"
    
    def test_intraday_profile_timeframes(self):
        """INTRADAY_PROFILE should use M15, H1, H4."""
        from utils.mtf_config import INTRADAY_PROFILE, Timeframe
        assert INTRADAY_PROFILE.timeframes == (Timeframe.M15, Timeframe.H1, Timeframe.H4)
    
    def test_intraday_profile_primary_tf(self):
        """INTRADAY_PROFILE primary should be H1."""
        from utils.mtf_config import INTRADAY_PROFILE, Timeframe
        assert INTRADAY_PROFILE.primary_tf == Timeframe.H1
    
    def test_intraday_profile_weights(self):
        """INTRADAY_PROFILE should have specific weights."""
        from utils.mtf_config import INTRADAY_PROFILE
        assert INTRADAY_PROFILE.weights == {"M15": 0.20, "H1": 0.45, "H4": 0.35}
    
    def test_swing_profile_import(self):
        """SWING_PROFILE should be importable."""
        from utils.mtf_config import SWING_PROFILE
        assert SWING_PROFILE is not None
    
    def test_swing_profile_name(self):
        """SWING_PROFILE should have correct name."""
        from utils.mtf_config import SWING_PROFILE
        assert SWING_PROFILE.name == "SWING"
    
    def test_swing_profile_timeframes(self):
        """SWING_PROFILE should use H1, H4, D1."""
        from utils.mtf_config import SWING_PROFILE, Timeframe
        assert SWING_PROFILE.timeframes == (Timeframe.H1, Timeframe.H4, Timeframe.D1)
    
    def test_swing_profile_primary_tf(self):
        """SWING_PROFILE primary should be H4."""
        from utils.mtf_config import SWING_PROFILE, Timeframe
        assert SWING_PROFILE.primary_tf == Timeframe.H4
    
    def test_swing_profile_weights(self):
        """SWING_PROFILE should have specific weights."""
        from utils.mtf_config import SWING_PROFILE
        assert SWING_PROFILE.weights == {"H1": 0.20, "H4": 0.45, "D1": 0.35}
    
    def test_swing_profile_lower_thresholds(self):
        """SWING_PROFILE should have lower thresholds for slow trading."""
        from utils.mtf_config import SWING_PROFILE
        assert SWING_PROFILE.min_confluence_score == 0.60
        assert SWING_PROFILE.signal_threshold == 0.60
        assert SWING_PROFILE.counter_trend_threshold == 0.80


class TestProfilesRegistry:
    """Test PROFILES registry dictionary."""
    
    def test_profiles_import(self):
        """PROFILES should be importable."""
        from utils.mtf_config import PROFILES
        assert PROFILES is not None
        assert isinstance(PROFILES, dict)
    
    def test_profiles_contains_all_presets(self):
        """PROFILES should contain all preset profiles."""
        from utils.mtf_config import PROFILES
        assert "SCALP" in PROFILES
        assert "INTRADAY" in PROFILES
        assert "SWING" in PROFILES
    
    def test_profiles_count(self):
        """PROFILES should have exactly 3 entries."""
        from utils.mtf_config import PROFILES
        assert len(PROFILES) == 3
    
    def test_profiles_values_are_mtf_profiles(self):
        """All PROFILES values should be MTFProfile instances."""
        from utils.mtf_config import PROFILES, MTFProfile
        for name, profile in PROFILES.items():
            assert isinstance(profile, MTFProfile), f"{name} is not MTFProfile"


class TestGetProfileFunction:
    """Test get_profile() function."""
    
    def test_get_profile_import(self):
        """get_profile should be importable."""
        from utils.mtf_config import get_profile
        assert callable(get_profile)
    
    def test_get_profile_scalp(self):
        """get_profile('SCALP') should return SCALP_PROFILE."""
        from utils.mtf_config import get_profile, SCALP_PROFILE
        profile = get_profile("SCALP")
        assert profile is SCALP_PROFILE
    
    def test_get_profile_intraday(self):
        """get_profile('INTRADAY') should return INTRADAY_PROFILE."""
        from utils.mtf_config import get_profile, INTRADAY_PROFILE
        profile = get_profile("INTRADAY")
        assert profile is INTRADAY_PROFILE
    
    def test_get_profile_swing(self):
        """get_profile('SWING') should return SWING_PROFILE."""
        from utils.mtf_config import get_profile, SWING_PROFILE
        profile = get_profile("SWING")
        assert profile is SWING_PROFILE
    
    def test_get_profile_case_insensitive(self):
        """get_profile should be case-insensitive."""
        from utils.mtf_config import get_profile, SCALP_PROFILE
        assert get_profile("scalp") is SCALP_PROFILE
        assert get_profile("Scalp") is SCALP_PROFILE
        assert get_profile("SCALP") is SCALP_PROFILE
    
    def test_get_profile_invalid_raises(self):
        """get_profile with invalid name should raise ValueError."""
        from utils.mtf_config import get_profile
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("INVALID")
    
    def test_get_profile_error_message_includes_available(self):
        """ValueError should list available profiles."""
        from utils.mtf_config import get_profile
        with pytest.raises(ValueError) as exc_info:
            get_profile("INVALID")
        error_msg = str(exc_info.value)
        assert "SCALP" in error_msg or "Available" in error_msg


class TestCreateCustomProfileFunction:
    """Test create_custom_profile() function."""
    
    def test_create_custom_profile_import(self):
        """create_custom_profile should be importable."""
        from utils.mtf_config import create_custom_profile
        assert callable(create_custom_profile)
    
    def test_create_custom_profile_basic(self):
        """Basic custom profile creation."""
        from utils.mtf_config import create_custom_profile, Timeframe
        profile = create_custom_profile(
            name="CUSTOM",
            timeframes=["M5", "M15", "H1"],
            primary_tf="M15"
        )
        assert profile.name == "CUSTOM"
        assert profile.timeframes == (Timeframe.M5, Timeframe.M15, Timeframe.H1)
        assert profile.primary_tf == Timeframe.M15
    
    def test_create_custom_profile_with_weights(self):
        """Custom profile with weights."""
        from utils.mtf_config import create_custom_profile
        profile = create_custom_profile(
            name="WEIGHTED",
            timeframes=["H1", "H4", "D1"],
            primary_tf="H4",
            weights={"H1": 0.3, "H4": 0.4, "D1": 0.3}
        )
        assert profile.weights == {"H1": 0.3, "H4": 0.4, "D1": 0.3}
    
    def test_create_custom_profile_case_insensitive_timeframes(self):
        """Timeframe strings should be case-insensitive."""
        from utils.mtf_config import create_custom_profile, Timeframe
        profile = create_custom_profile(
            name="TEST",
            timeframes=["m5", "m15", "h1"],
            primary_tf="h1"
        )
        assert profile.timeframes == (Timeframe.M5, Timeframe.M15, Timeframe.H1)
        assert profile.primary_tf == Timeframe.H1
    
    def test_create_custom_profile_with_kwargs(self):
        """Custom profile with additional kwargs."""
        from utils.mtf_config import create_custom_profile
        profile = create_custom_profile(
            name="CUSTOM",
            timeframes=["M15", "H1", "H4"],
            primary_tf="H1",
            min_bars=200,
            signal_threshold=0.75,
            description="Custom profile for testing"
        )
        assert profile.min_bars == 200
        assert profile.signal_threshold == 0.75
        assert profile.description == "Custom profile for testing"
    
    def test_create_custom_profile_returns_mtf_profile(self):
        """create_custom_profile should return MTFProfile instance."""
        from utils.mtf_config import create_custom_profile, MTFProfile
        profile = create_custom_profile(
            name="TEST",
            timeframes=["M15", "H1"],
            primary_tf="H1"
        )
        assert isinstance(profile, MTFProfile)


class TestGetTFMinutesFunction:
    """Test get_tf_minutes() utility function."""
    
    def test_get_tf_minutes_import(self):
        """get_tf_minutes should be importable."""
        from utils.mtf_config import get_tf_minutes
        assert callable(get_tf_minutes)
    
    @pytest.mark.parametrize("tf,expected", [
        ("M1", 1),
        ("M5", 5),
        ("M15", 15),
        ("M30", 30),
        ("H1", 60),
        ("H4", 240),
        ("D1", 1440),
        ("W1", 10080),
    ])
    def test_get_tf_minutes_values(self, tf, expected):
        """get_tf_minutes should return correct minutes."""
        from utils.mtf_config import get_tf_minutes
        assert get_tf_minutes(tf) == expected
    
    def test_get_tf_minutes_case_insensitive(self):
        """get_tf_minutes should be case-insensitive."""
        from utils.mtf_config import get_tf_minutes
        assert get_tf_minutes("h1") == 60
        assert get_tf_minutes("H1") == 60
        assert get_tf_minutes("h4") == 240
    
    def test_get_tf_minutes_fallback_parsing(self):
        """get_tf_minutes should handle fallback parsing."""
        from utils.mtf_config import get_tf_minutes
        # These are valid patterns that might need fallback parsing
        # Testing that the function doesn't crash
        result = get_tf_minutes("H1")
        assert result == 60


class TestSortTimeframesFunction:
    """Test sort_timeframes() utility function."""
    
    def test_sort_timeframes_import(self):
        """sort_timeframes should be importable."""
        from utils.mtf_config import sort_timeframes
        assert callable(sort_timeframes)
    
    def test_sort_timeframes_ascending(self):
        """sort_timeframes should sort from lowest to highest."""
        from utils.mtf_config import sort_timeframes
        unsorted = ["H4", "M15", "H1", "D1", "M5"]
        expected = ["M5", "M15", "H1", "H4", "D1"]
        assert sort_timeframes(unsorted) == expected
    
    def test_sort_timeframes_already_sorted(self):
        """sort_timeframes on sorted input should return same order."""
        from utils.mtf_config import sort_timeframes
        already_sorted = ["M1", "M5", "M15", "H1"]
        assert sort_timeframes(already_sorted) == already_sorted
    
    def test_sort_timeframes_single_element(self):
        """sort_timeframes with single element."""
        from utils.mtf_config import sort_timeframes
        assert sort_timeframes(["H1"]) == ["H1"]
    
    def test_sort_timeframes_empty_list(self):
        """sort_timeframes with empty list."""
        from utils.mtf_config import sort_timeframes
        assert sort_timeframes([]) == []


class TestGetHigherTimeframeFunction:
    """Test get_higher_timeframe() utility function."""
    
    def test_get_higher_timeframe_import(self):
        """get_higher_timeframe should be importable."""
        from utils.mtf_config import get_higher_timeframe
        assert callable(get_higher_timeframe)
    
    def test_get_higher_timeframe_basic(self):
        """get_higher_timeframe should return next higher TF."""
        from utils.mtf_config import get_higher_timeframe
        available = ["M5", "M15", "H1", "H4", "D1"]
        assert get_higher_timeframe("M15", available) == "H1"
        assert get_higher_timeframe("H1", available) == "H4"
        assert get_higher_timeframe("H4", available) == "D1"
    
    def test_get_higher_timeframe_no_higher(self):
        """get_higher_timeframe returns None when no higher TF."""
        from utils.mtf_config import get_higher_timeframe
        available = ["M5", "M15", "H1"]
        assert get_higher_timeframe("H1", available) is None
    
    def test_get_higher_timeframe_skips_unavailable(self):
        """get_higher_timeframe skips unavailable timeframes."""
        from utils.mtf_config import get_higher_timeframe
        # H1 not in available, should skip to H4
        available = ["M15", "H4", "D1"]
        assert get_higher_timeframe("M15", available) == "H4"


class TestGetLowerTimeframeFunction:
    """Test get_lower_timeframe() utility function."""
    
    def test_get_lower_timeframe_import(self):
        """get_lower_timeframe should be importable."""
        from utils.mtf_config import get_lower_timeframe
        assert callable(get_lower_timeframe)
    
    def test_get_lower_timeframe_basic(self):
        """get_lower_timeframe should return next lower TF."""
        from utils.mtf_config import get_lower_timeframe
        available = ["M5", "M15", "H1", "H4", "D1"]
        assert get_lower_timeframe("H1", available) == "M15"
        assert get_lower_timeframe("H4", available) == "H1"
        assert get_lower_timeframe("D1", available) == "H4"
    
    def test_get_lower_timeframe_no_lower(self):
        """get_lower_timeframe returns None when no lower TF."""
        from utils.mtf_config import get_lower_timeframe
        available = ["M15", "H1", "H4"]
        assert get_lower_timeframe("M15", available) is None
    
    def test_get_lower_timeframe_skips_unavailable(self):
        """get_lower_timeframe skips unavailable timeframes."""
        from utils.mtf_config import get_lower_timeframe
        # H1 not in available, should skip to M15
        available = ["M15", "H4", "D1"]
        assert get_lower_timeframe("H4", available) == "M15"


class TestProfileWeightsSumToOne:
    """Test that profile weights sum to approximately 1.0."""
    
    @pytest.mark.parametrize("profile_name", ["SCALP", "INTRADAY", "SWING"])
    def test_preset_weights_sum_to_one(self, profile_name):
        """Preset profile weights should sum to 1.0."""
        from utils.mtf_config import get_profile
        profile = get_profile(profile_name)
        total = sum(profile.weights.values())
        assert pytest.approx(total, rel=0.01) == 1.0
    
    def test_auto_generated_weights_sum_to_one(self):
        """Auto-generated weights should sum to 1.0."""
        from utils.mtf_config import MTFProfile, Timeframe
        profile = MTFProfile(
            name="TEST",
            timeframes=(Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4)
        )
        total = sum(profile.weights.values())
        assert pytest.approx(total, rel=0.01) == 1.0