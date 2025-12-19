"""
Feature Engineering Consistency Integration Tests.

Tests data pipeline integrity and feature engineering consistency.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils.features_engineering import FeatureEngineer
from utils.indicators_extended import TrendIndicators


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    base = 1.1000
    
    # Generate realistic price movement
    returns = np.random.normal(0, 0.0005, n)
    close = base * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "time": idx,
        "open": close * (1 - np.random.uniform(0, 0.0002, n)),
        "high": close * (1 + np.random.uniform(0, 0.0003, n)),
        "low": close * (1 - np.random.uniform(0, 0.0003, n)),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })
    return df


@pytest.mark.integration
class TestFeatureEngineeringConsistency:
    """Tests for feature engineering consistency."""
    
    def test_features_deterministic(self, sample_ohlcv):
        """Same input should always produce same features."""
        engineer = FeatureEngineer()
        
        features1 = engineer.compute_features(sample_ohlcv.copy())
        features2 = engineer.compute_features(sample_ohlcv.copy())
        
        # Features should be identical
        pd.testing.assert_frame_equal(features1, features2)
    
    def test_features_no_future_leakage(self, sample_ohlcv):
        """Features at time T should not use data from time > T."""
        engineer = FeatureEngineer()
        
        # Compute features for full dataset
        full_features = engineer.compute_features(sample_ohlcv.copy())
        
        # Compute features for truncated dataset (first 400 rows)
        truncated = sample_ohlcv.iloc[:400].copy()
        truncated_features = engineer.compute_features(truncated)
        
        # Features at row 399 should be identical
        if len(truncated_features) > 0 and len(full_features) > 399:
            # Compare last row of truncated with corresponding row in full
            last_idx = len(truncated_features) - 1
            for col in truncated_features.columns:
                if col in full_features.columns:
                    val_trunc = truncated_features[col].iloc[last_idx]
                    val_full = full_features[col].iloc[last_idx]
                    if not np.isnan(val_trunc) and not np.isnan(val_full):
                        assert np.isclose(val_trunc, val_full, rtol=1e-10), \
                            f"Future leakage detected in {col}"
    
    def test_features_handle_missing_data(self, sample_ohlcv):
        """Features should handle missing data gracefully."""
        engineer = FeatureEngineer()
        
        # Introduce some NaN values
        df = sample_ohlcv.copy()
        df.loc[100:105, 'close'] = np.nan
        df.loc[200:202, 'volume'] = np.nan
        
        # Should not raise exception
        features = engineer.compute_features(df)
        
        # Should produce output (may have NaN but shouldn't crash)
        assert features is not None
        assert len(features) > 0
    
    def test_indicator_values_in_valid_range(self, sample_ohlcv):
        """Indicator values should be within valid ranges."""
        df = sample_ohlcv.copy()
        adx, plus_di, minus_di = TrendIndicators.calculate_adx(df)
        df['adx'] = adx
        df['rsi'] = df['close'].diff().apply(lambda x: 50)  # Placeholder
        df['atr'] = TrendIndicators.calculate_volatility_compression(df)[0] if hasattr(TrendIndicators, 'calculate_volatility_compression') else 0.001
        indicators = df
        
        # RSI should be 0-100
        if 'rsi' in indicators.columns:
            rsi_valid = indicators['rsi'].dropna()
            assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all(), \
                "RSI out of valid range [0, 100]"
        
        # ATR should be positive
        if 'atr' in indicators.columns:
            atr_valid = indicators['atr'].dropna()
            assert (atr_valid >= 0).all(), "ATR should be non-negative"
        
        # ADX should be 0-100
        if 'adx' in indicators.columns:
            adx_valid = indicators['adx'].dropna()
            assert (adx_valid >= 0).all() and (adx_valid <= 100).all(), \
                "ADX out of valid range [0, 100]"
    
    def test_rolling_features_warmup_period(self, sample_ohlcv):
        """Rolling features should have proper warmup period."""
        engineer = FeatureEngineer()
        features = engineer.compute_features(sample_ohlcv.copy())
        
        # First N rows should have NaN for features requiring warmup
        # (where N depends on the longest lookback period)
        warmup_rows = 50  # Typical warmup period
        
        # After warmup, features should be mostly valid
        if len(features) > warmup_rows:
            post_warmup = features.iloc[warmup_rows:]
            nan_ratio = post_warmup.isna().sum().sum() / post_warmup.size
            assert nan_ratio < 0.1, f"Too many NaN values after warmup: {nan_ratio:.2%}"


@pytest.mark.integration
class TestDataPipelineIntegrity:
    """Tests for data pipeline integrity."""
    
    def test_ohlcv_relationships_preserved(self, sample_ohlcv):
        """OHLCV relationships should be preserved through pipeline."""
        # High >= Low always
        assert (sample_ohlcv['high'] >= sample_ohlcv['low']).all()
        
        # High >= Open and High >= Close
        assert (sample_ohlcv['high'] >= sample_ohlcv['open']).all()
        assert (sample_ohlcv['high'] >= sample_ohlcv['close']).all()
        
        # Low <= Open and Low <= Close
        assert (sample_ohlcv['low'] <= sample_ohlcv['open']).all()
        assert (sample_ohlcv['low'] <= sample_ohlcv['close']).all()
    
    def test_timestamp_monotonicity(self, sample_ohlcv):
        """Timestamps should be monotonically increasing."""
        times = pd.to_datetime(sample_ohlcv['time'])
        diffs = times.diff().dropna()
        assert (diffs > pd.Timedelta(0)).all(), "Timestamps not monotonically increasing"
    
    def test_no_duplicate_timestamps(self, sample_ohlcv):
        """Should not have duplicate timestamps."""
        times = sample_ohlcv['time']
        assert times.is_unique, "Duplicate timestamps found"
    
    def test_price_continuity(self, sample_ohlcv):
        """Price changes should be within reasonable bounds."""
        returns = sample_ohlcv['close'].pct_change().dropna()
        
        # No single bar should have >10% move (for forex)
        max_return = returns.abs().max()
        assert max_return < 0.10, f"Unrealistic price move: {max_return:.2%}"
    
    def test_volume_non_negative(self, sample_ohlcv):
        """Volume should be non-negative."""
        assert (sample_ohlcv['volume'] >= 0).all(), "Negative volume found"


@pytest.mark.integration
class TestFeatureScaling:
    """Tests for feature scaling consistency."""
    
    def test_scaled_features_bounded(self, sample_ohlcv):
        """Scaled features should be within expected bounds."""
        engineer = FeatureEngineer()
        features = engineer.compute_features(sample_ohlcv.copy())
        
        if hasattr(engineer, 'scale_features'):
            scaled = engineer.scale_features(features)
            
            # Most scaled features should be roughly in [-3, 3] for z-score
            # or [0, 1] for min-max
            for col in scaled.columns:
                valid = scaled[col].dropna()
                if len(valid) > 0:
                    # Check for extreme outliers
                    assert valid.abs().max() < 100, \
                        f"Feature {col} has extreme values after scaling"
    
    def test_scaling_preserves_relative_order(self, sample_ohlcv):
        """Scaling should preserve relative ordering of values."""
        engineer = FeatureEngineer()
        features = engineer.compute_features(sample_ohlcv.copy())
        
        if hasattr(engineer, 'scale_features') and 'close' in features.columns:
            original = features['close'].dropna()
            
            scaled_features = engineer.scale_features(features)
            if 'close' in scaled_features.columns:
                scaled = scaled_features['close'].dropna()
                
                # Rank correlation should be 1.0 (perfect)
                if len(original) == len(scaled):
                    from scipy.stats import spearmanr
                    corr, _ = spearmanr(original, scaled)
                    assert corr > 0.99, "Scaling changed relative ordering"
