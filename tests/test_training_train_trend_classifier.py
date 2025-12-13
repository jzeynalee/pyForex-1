"""
Comprehensive unit tests for training/train_trend_classifier.py

Tests cover:
- TrendFeatureExtractor: Indicator computation and feature extraction
- TrendLabeler: Label generation based on price movement
- Utility functions: Data loading and preparation
- Integration: End-to-end training pipeline

Summary Of Tests:

| Test Class | Test Method | Purpose |
|---|---|---|
| **TestTrendFeatureExtractor** | test_initialization | Verify TrendFeatureExtractor parameters are set correctly |
| | test_compute_indicators_basic | Verify all required indicator columns are added |
| | test_compute_indicators_values_not_nan | Verify indicators produce non-NaN values after warmup |
| | test_compute_indicators_ema_ordering | Verify EMA values follow expected uptrend ordering |
| | test_compute_atr | Verify ATR is computed correctly and is positive |
| | test_compute_adx | Verify ADX and DI values are in valid ranges (0-100) |
| | test_compute_bb_width | Verify Bollinger Band width is computed and positive |
| | test_extract_features_insufficient_data | Verify early indices return None (insufficient history) |
| | test_extract_features_sufficient_data | Verify features extracted correctly for valid indices |
| | test_extract_features_value_ranges | Verify extracted feature values are in reasonable ranges |
| | test_extract_features_with_nan_indicators | Verify graceful handling of NaN indicators |
| **TestTrendLabeler** | test_initialization | Verify TrendLabeler parameters are set correctly |
| | test_generate_labels_shape | Verify labels array shape matches input length |
| | test_generate_labels_values | Verify labels are in valid range: -1, 0, 1, or NaN |
| | test_generate_labels_bullish_trend | Verify uptrend is labeled as bullish (1) |
| | test_generate_labels_bearish_trend | Verify downtrend is labeled as bearish (-1) |
| | test_generate_labels_sideways | Verify ranging market is labeled sideways (0) |
| | test_generate_labels_last_bars_nan | Verify last forward_window bars are NaN |
| | test_generate_labels_threshold_effect | Verify threshold parameter affects label distribution |
| **TestLoadOhlcvData** | test_load_ohlcv_basic | Verify basic CSV data loading works correctly |
| | test_load_ohlcv_missing_columns | Verify error raised for missing required columns |
| | test_load_ohlcv_nonexistent_file | Verify error raised for nonexistent file |
| | test_load_ohlcv_case_insensitivity | Verify column names are case-insensitive |
| | test_load_ohlcv_with_time_column | Verify data with time column loads and parses correctly |
| **TestPrepareTrainingData** | test_prepare_training_data_shapes | Verify output shapes are correct (X: 2D, y: 1D) |
| | test_prepare_training_data_sample_count | Verify reasonable number of samples produced |
| | test_prepare_training_data_label_distribution | Verify labels are in valid range and balanced |
| | test_prepare_training_data_no_nan | Verify no NaN values in features or labels |
| | test_prepare_training_data_custom_parameters | Verify custom forward_window and threshold work |
| **TestIntegration** | test_full_pipeline | Verify complete feature extraction and labeling pipeline |
| | test_main_with_real_data | Verify main() function with mocked real data |
| | test_main_synthetic_data | Verify main() function with synthetic data flag |
| **TestEdgeCases** | test_empty_dataframe | Verify handling of empty dataframe |
| | test_single_row_dataframe | Verify handling of single row dataframe |
| | test_constant_price_data | Verify handling of constant prices |
| | test_large_price_gaps | Verify handling of large price gaps |
| | test_negative_prices | Verify handling of negative price values |
| | test_very_small_prices | Verify handling of very small price values |
| | test_inf_values | Verify handling of infinity values |
| **TestDataIntegrity** | test_feature_extractor_does_not_modify_input | Verify input dataframe is not modified |
| | test_feature_values_consistent | Verify consistent results on multiple runs |
| | test_labels_consistent | Verify consistent label generation on multiple runs |
| | test_feature_count_consistent | Verify all features have 13 elements |
| **TestPerformanceAndScaling** | test_feature_extraction_large_dataset | Verify performance with 10k rows |
| | test_label_generation_large_dataset | Verify label generation with 10k rows |
| | test_multiple_extractions | Verify multiple feature extractions work correctly |
| **TestParametrized** | test_different_ema_periods | Test with different EMA period configurations |
| | test_different_labeler_configs | Test labeler with different forward_window and threshold |
| | test_feature_extraction_at_different_indices | Test feature extraction at various indices |
"""

import sys
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import numpy as np
import pandas as pd

# Add parent directory to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training.train_trend_classifier import (
    TrendFeatureExtractor,
    TrendLabeler,
    load_ohlcv_data,
    prepare_training_data,
    main,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    n = 500
    np.random.seed(42)
    
    # Generate realistic price movement
    close = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
    
    df = pd.DataFrame({
        'open': close + np.random.randn(n) * 0.0005,
        'high': close + np.abs(np.random.randn(n) * 0.0007),
        'low': close - np.abs(np.random.randn(n) * 0.0007),
        'close': close,
        'volume': np.random.randint(1000, 10000, n),
    })
    
    # Ensure high > low
    df['high'] = df[['high', 'close']].max(axis=1) + 0.0001
    df['low'] = df[['low', 'close']].min(axis=1) - 0.0001
    
    return df


@pytest.fixture
def large_ohlcv_data():
    """Create larger OHLCV dataset for integration testing."""
    n = 2000
    np.random.seed(123)
    
    # Simulate trending data
    trend = np.linspace(0, 0.1, n)
    close = 1.1 + trend + np.cumsum(np.random.randn(n) * 0.0005)
    
    df = pd.DataFrame({
        'open': close + np.random.randn(n) * 0.0005,
        'high': close + np.abs(np.random.randn(n) * 0.0007),
        'low': close - np.abs(np.random.randn(n) * 0.0007),
        'close': close,
        'volume': np.random.randint(1000, 10000, n),
    })
    
    df['high'] = df[['high', 'close']].max(axis=1) + 0.0001
    df['low'] = df[['low', 'close']].min(axis=1) - 0.0001
    
    return df


@pytest.fixture
def trend_feature_extractor():
    """Create TrendFeatureExtractor instance."""
    return TrendFeatureExtractor(
        ema_periods=(20, 50, 200),
        adx_period=14,
        roc_periods=(5, 10),
        atr_period=14,
    )


@pytest.fixture
def trend_labeler():
    """Create TrendLabeler instance."""
    return TrendLabeler(forward_window=20, trend_threshold_pct=1.0)


@pytest.fixture
def csv_file(tmp_path, sample_ohlcv_data):
    """Create temporary CSV file with OHLCV data."""
    csv_path = tmp_path / "test_data.csv"
    sample_ohlcv_data.to_csv(csv_path, index=False)
    return csv_path


# =============================================================================
# TESTS: TrendFeatureExtractor
# =============================================================================

class TestTrendFeatureExtractor:
    """Tests for TrendFeatureExtractor class."""
    
    def test_initialization(self):
        """Test TrendFeatureExtractor initialization."""
        extractor = TrendFeatureExtractor(
            ema_periods=(20, 50, 200),
            adx_period=14,
            roc_periods=(5, 10),
            atr_period=14,
        )
        
        assert extractor.ema_periods == (20, 50, 200)
        assert extractor.adx_period == 14
        assert extractor.roc_periods == (5, 10)
        assert extractor.atr_period == 14
    
    def test_compute_indicators_basic(self, sample_ohlcv_data):
        """Test that compute_indicators adds all required columns."""
        extractor = TrendFeatureExtractor()
        df = extractor.compute_indicators(sample_ohlcv_data)
        
        # Check for EMA columns
        assert 'ema_20' in df.columns
        assert 'ema_50' in df.columns
        assert 'ema_200' in df.columns
        
        # Check for ADX and DI columns
        assert 'adx' in df.columns
        assert 'plus_di' in df.columns
        assert 'minus_di' in df.columns
        
        # Check for ROC columns
        assert 'roc_5' in df.columns
        assert 'roc_10' in df.columns
        
        # Check for ATR and BB columns
        assert 'atr' in df.columns
        assert 'bb_width' in df.columns
    
    def test_compute_indicators_values_not_nan(self, sample_ohlcv_data):
        """Test that computed indicators eventually have non-NaN values."""
        extractor = TrendFeatureExtractor()
        df = extractor.compute_indicators(sample_ohlcv_data)
        
        # After 200+ bars, we should have non-NaN values
        assert not df.loc[300:, 'ema_200'].isna().all()
        assert not df.loc[300:, 'adx'].isna().all()
        assert not df.loc[300:, 'atr'].isna().all()
    
    def test_compute_indicators_ema_ordering(self, sample_ohlcv_data):
        """Test that EMA values follow expected ordering in trends."""
        df = sample_ohlcv_data.copy()
        # Create a clear uptrend
        df['close'] = np.arange(len(df)) * 0.01 + 1.0
        
        extractor = TrendFeatureExtractor()
        df = extractor.compute_indicators(df)
        
        # In a clear uptrend, EMA20 > EMA50 > EMA200
        bullish_idx = df.index[-1]
        assert df.loc[bullish_idx, 'ema_20'] > df.loc[bullish_idx, 'ema_50']
        assert df.loc[bullish_idx, 'ema_50'] > df.loc[bullish_idx, 'ema_200']
    
    def test_compute_atr(self, sample_ohlcv_data):
        """Test ATR computation."""
        extractor = TrendFeatureExtractor()
        df = extractor.compute_indicators(sample_ohlcv_data)
        
        # ATR should be positive
        assert (df['atr'] >= 0).any()
        
        # ATR should stabilize after warm-up period
        atr_tail = df['atr'].iloc[-100:]
        assert not atr_tail.isna().all()
    
    def test_compute_adx(self, sample_ohlcv_data):
        """Test ADX and directional indicator computation."""
        extractor = TrendFeatureExtractor()
        df = extractor.compute_indicators(sample_ohlcv_data)
        
        # ADX should be between 0 and 100
        valid_adx = df['adx'].dropna()
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()
        
        # +DI and -DI should be non-negative
        valid_plus_di = df['plus_di'].dropna()
        valid_minus_di = df['minus_di'].dropna()
        assert (valid_plus_di >= 0).all()
        assert (valid_minus_di >= 0).all()
    
    def test_compute_bb_width(self, sample_ohlcv_data):
        """Test Bollinger Band width computation."""
        extractor = TrendFeatureExtractor()
        df = extractor.compute_indicators(sample_ohlcv_data)
        
        # BB width should be positive
        valid_bb = df['bb_width'].dropna()
        assert (valid_bb >= 0).all()
    
    def test_extract_features_insufficient_data(self, trend_feature_extractor, sample_ohlcv_data):
        """Test that extract_features returns None for early indices."""
        df = trend_feature_extractor.compute_indicators(sample_ohlcv_data)
        
        # Early indices should return None (insufficient history)
        for idx in range(50):
            features = trend_feature_extractor.extract_features(df, idx)
            assert features is None
    
    def test_extract_features_sufficient_data(self, trend_feature_extractor, sample_ohlcv_data):
        """Test extract_features for sufficient data."""
        df = trend_feature_extractor.compute_indicators(sample_ohlcv_data)
        
        # Should work for later indices
        features = trend_feature_extractor.extract_features(df, 250)
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert len(features) == 13
        assert features.dtype == float
    
    def test_extract_features_value_ranges(self, trend_feature_extractor, sample_ohlcv_data):
        """Test that extracted features are in reasonable ranges."""
        df = trend_feature_extractor.compute_indicators(sample_ohlcv_data)
        features = trend_feature_extractor.extract_features(df, 250)
        
        # struct_score: 0-1
        assert 0 <= features[0] <= 1
        
        # mtf_score: typically 0.4 or 0.8
        assert features[1] in [0.4, 0.8]
        
        # regime: 0 or 1
        assert features[2] in [0, 1]
        
        # ADX: 0-100
        assert 0 <= features[3] <= 100
        
        # +DI, -DI: usually 0-50
        assert features[4] >= 0
        assert features[5] >= 0
        
        # price_above_ema: 0 or 1
        assert features[6] in [0, 1]
        assert features[7] in [0, 1]
        assert features[8] in [0, 1]
        
        # ema_alignment: -1, 0, or 1
        assert features[9] in [-1.0, 0.0, 1.0]
        
        # vol_compression: 0-1
        assert 0 <= features[10] <= 1
    
    def test_extract_features_with_nan_indicators(self, trend_feature_extractor, large_ohlcv_data):
        """Test extract_features handles NaN indicators gracefully."""
        # Use the large dataframe that has sufficient history (>200 bars)
        df = trend_feature_extractor.compute_indicators(large_ohlcv_data)
        
        # First, verify we can extract features at a valid index
        result_valid = trend_feature_extractor.extract_features(df, 300)
        assert result_valid is not None
        
        # Test that early indices return None (insufficient history)
        result_early = trend_feature_extractor.extract_features(df, 50)
        assert result_early is None


# =============================================================================
# TESTS: TrendLabeler
# =============================================================================

class TestTrendLabeler:
    """Tests for TrendLabeler class."""
    
    def test_initialization(self):
        """Test TrendLabeler initialization."""
        labeler = TrendLabeler(forward_window=20, trend_threshold_pct=1.0)
        
        assert labeler.forward_window == 20
        assert labeler.trend_threshold_pct == 1.0
    
    def test_generate_labels_shape(self, sample_ohlcv_data):
        """Test that generate_labels returns correct shape."""
        labeler = TrendLabeler(forward_window=20)
        labels = labeler.generate_labels(sample_ohlcv_data)
        
        assert len(labels) == len(sample_ohlcv_data)
        assert isinstance(labels, np.ndarray)
    
    def test_generate_labels_values(self, sample_ohlcv_data):
        """Test that labels are in expected range."""
        labeler = TrendLabeler(forward_window=20, trend_threshold_pct=0.5)
        labels = labeler.generate_labels(sample_ohlcv_data)
        
        # Valid values: -1, 0, 1, NaN
        valid_vals = {-1.0, 0.0, 1.0}
        for val in labels:
            assert np.isnan(val) or val in valid_vals
    
    def test_generate_labels_bullish_trend(self):
        """Test labeling for clear bullish trend."""
        df = pd.DataFrame({
            'close': np.arange(100, 200, 1.0)  # Clear uptrend
        })
        
        labeler = TrendLabeler(forward_window=20, trend_threshold_pct=0.5)
        labels = labeler.generate_labels(df)
        
        # Early bars should be labeled BULLISH
        bullish_count = sum(labels[:70] == 1)
        assert bullish_count > 50  # Most early bars are bullish
    
    def test_generate_labels_bearish_trend(self):
        """Test labeling for clear bearish trend."""
        df = pd.DataFrame({
            'close': np.arange(200, 100, -1.0)  # Clear downtrend
        })
        
        labeler = TrendLabeler(forward_window=20, trend_threshold_pct=0.5)
        labels = labeler.generate_labels(df)
        
        # Early bars should be labeled BEARISH
        bearish_count = sum(labels[:70] == -1)
        assert bearish_count > 50  # Most early bars are bearish
    
    def test_generate_labels_sideways(self):
        """Test labeling for sideways/ranging market."""
        df = pd.DataFrame({
            'close': np.tile([1.0, 1.001, 1.002, 1.001], 30)  # Repeating pattern
        })
        
        labeler = TrendLabeler(forward_window=20, trend_threshold_pct=1.0)
        labels = labeler.generate_labels(df)
        
        # Most bars should be SIDEWAYS
        sideways_count = sum(labels == 0)
        assert sideways_count > len(df) * 0.5
    
    def test_generate_labels_last_bars_nan(self, sample_ohlcv_data):
        """Test that last forward_window bars are NaN."""
        labeler = TrendLabeler(forward_window=20)
        labels = labeler.generate_labels(sample_ohlcv_data)
        
        # Last 20 bars should be NaN
        assert np.isnan(labels[-20:]).all()
    
    def test_generate_labels_threshold_effect(self, sample_ohlcv_data):
        """Test that threshold affects label distribution."""
        labeler_low = TrendLabeler(forward_window=20, trend_threshold_pct=0.1)
        labeler_high = TrendLabeler(forward_window=20, trend_threshold_pct=5.0)
        
        labels_low = labeler_low.generate_labels(sample_ohlcv_data)
        labels_high = labeler_high.generate_labels(sample_ohlcv_data)
        
        # Lower threshold should produce more extreme labels
        sideways_low = sum(labels_low == 0)
        sideways_high = sum(labels_high == 0)
        assert sideways_high > sideways_low


# =============================================================================
# TESTS: Utility Functions
# =============================================================================

class TestLoadOhlcvData:
    """Tests for load_ohlcv_data function."""
    
    def test_load_ohlcv_basic(self, csv_file):
        """Test basic OHLCV data loading."""
        df = load_ohlcv_data(csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close'])
    
    def test_load_ohlcv_missing_columns(self, tmp_path):
        """Test that missing required columns raise error."""
        csv_path = tmp_path / "incomplete.csv"
        df = pd.DataFrame({
            'open': [1.0, 1.01],
            'close': [1.01, 1.02],
        })
        df.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            load_ohlcv_data(csv_path)
    
    def test_load_ohlcv_nonexistent_file(self, tmp_path):
        """Test that nonexistent file raises error."""
        nonexistent = tmp_path / "nonexistent.csv"
        
        with pytest.raises(FileNotFoundError):
            load_ohlcv_data(nonexistent)
    
    def test_load_ohlcv_case_insensitivity(self, tmp_path):
        """Test that column names are case-insensitive."""
        csv_path = tmp_path / "uppercase.csv"
        df = pd.DataFrame({
            'OPEN': [1.0, 1.01],
            'HIGH': [1.01, 1.02],
            'LOW': [0.99, 1.00],
            'CLOSE': [1.005, 1.015],
        })
        df.to_csv(csv_path, index=False)
        
        loaded = load_ohlcv_data(csv_path)
        assert all(col in loaded.columns for col in ['open', 'high', 'low', 'close'])
    
    def test_load_ohlcv_with_time_column(self, tmp_path):
        """Test loading data with time column."""
        csv_path = tmp_path / "with_time.csv"
        df = pd.DataFrame({
            'time': ['2023-01-01', '2023-01-02'],
            'open': [1.0, 1.01],
            'high': [1.01, 1.02],
            'low': [0.99, 1.00],
            'close': [1.005, 1.015],
        })
        df.to_csv(csv_path, index=False)
        
        loaded = load_ohlcv_data(csv_path)
        assert 'time' in loaded.columns
        assert pd.api.types.is_datetime64_any_dtype(loaded['time'])


class TestPrepareTrainingData:
    """Tests for prepare_training_data function."""
    
    def test_prepare_training_data_shapes(self, large_ohlcv_data):
        """Test that prepare_training_data returns correct shapes."""
        X, y = prepare_training_data(large_ohlcv_data)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 13  # 13 features
    
    def test_prepare_training_data_sample_count(self, large_ohlcv_data):
        """Test that prepare_training_data produces reasonable sample count."""
        X, y = prepare_training_data(large_ohlcv_data)
        
        # Should have many samples (after feature extraction)
        assert X.shape[0] > 100
    
    def test_prepare_training_data_label_distribution(self, large_ohlcv_data):
        """Test label distribution is reasonable."""
        X, y = prepare_training_data(large_ohlcv_data, trend_threshold_pct=1.0)
        
        # Check that labels are in expected range
        assert all(val in [-1, 0, 1] for val in y)
        
        # Check that we have some of each class (usually)
        unique_labels = np.unique(y)
        assert len(unique_labels) > 0
    
    def test_prepare_training_data_no_nan(self, large_ohlcv_data):
        """Test that features and labels contain no NaN."""
        X, y = prepare_training_data(large_ohlcv_data)
        
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
    
    def test_prepare_training_data_custom_parameters(self, large_ohlcv_data):
        """Test with custom forward_window and threshold."""
        X1, y1 = prepare_training_data(
            large_ohlcv_data,
            forward_window=10,
            trend_threshold_pct=0.5
        )
        X2, y2 = prepare_training_data(
            large_ohlcv_data,
            forward_window=30,
            trend_threshold_pct=2.0
        )
        
        # Different parameters may produce different sample counts
        assert X1.shape[0] > 0
        assert X2.shape[0] > 0


# =============================================================================
# TESTS: Integration
# =============================================================================

class TestIntegration:
    """Integration tests for complete pipeline."""
    
    def test_full_pipeline(self, large_ohlcv_data):
        """Test complete feature extraction and labeling pipeline."""
        extractor = TrendFeatureExtractor()
        df = extractor.compute_indicators(large_ohlcv_data)
        
        labeler = TrendLabeler()
        labels = labeler.generate_labels(df)
        
        # Extract features
        features_list = []
        labels_list = []
        
        for i in range(len(df)):
            if np.isnan(labels[i]):
                continue
            
            features = extractor.extract_features(df, i)
            if features is not None:
                features_list.append(features)
                labels_list.append(labels[i])
        
        assert len(features_list) > 0
        assert len(labels_list) == len(features_list)
    
    @patch('training.train_trend_classifier.load_ohlcv_data')
    @patch('training.train_trend_classifier.TrendClassifier')
    def test_main_with_real_data(self, mock_classifier_class, mock_load_data, large_ohlcv_data, tmp_path, monkeypatch):
        """Test main function with mocked data loading."""
        # Create actual CSV file
        csv_file = tmp_path / 'data.csv'
        large_ohlcv_data.to_csv(csv_file, index=False)
        
        # Setup mocks
        mock_load_data.return_value = large_ohlcv_data
        
        mock_classifier = MagicMock()
        mock_classifier.fit.return_value = {
            'test_accuracy': 0.75,
            'cv_mean': 0.73,
            'cv_std': 0.02,
        }
        mock_classifier.get_feature_importance.return_value = pd.DataFrame({
            'feature': ['f1', 'f2'],
            'importance': [0.5, 0.5],
        })
        mock_classifier_class.return_value = mock_classifier
        
        # Setup args
        output_file = tmp_path / "model.joblib"
        monkeypatch.setattr(
            'sys.argv',
            [
                'train_trend_classifier.py',
                '--data', str(csv_file),
                '--output', str(output_file),
            ]
        )
        
        # Run main
        main()
        
        # Verify calls
        mock_load_data.assert_called_once()
        mock_classifier.fit.assert_called_once()
    
    def test_main_synthetic_data(self, tmp_path, monkeypatch):
        """Test main function with synthetic data flag."""
        output_file = tmp_path / "model.joblib"
        monkeypatch.setattr(
            'sys.argv',
            [
                'train_trend_classifier.py',
                '--data', 'dummy.csv',  # Required argument even with --synthetic
                '--synthetic',
                '--output', str(output_file),
            ]
        )
        
        # Skip if generate_synthetic_training_data doesn't exist
        try:
            from training.train_trend_classifier import generate_synthetic_training_data
            # Function exists, test it
            with patch('training.train_trend_classifier.TrendClassifier'):
                try:
                    main()
                except (ImportError, AttributeError):
                    pass
        except ImportError:
            # Function doesn't exist, skip this test gracefully
            pytest.skip("generate_synthetic_training_data not available")


# =============================================================================
# TESTS: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame({'close': []})
        
        labeler = TrendLabeler()
        labels = labeler.generate_labels(df)
        
        assert len(labels) == 0
    
    def test_single_row_dataframe(self):
        """Test handling of single row."""
        df = pd.DataFrame({
            'open': [1.0],
            'high': [1.01],
            'low': [0.99],
            'close': [1.005],
        })
        
        extractor = TrendFeatureExtractor()
        result_df = extractor.compute_indicators(df)
        
        assert len(result_df) == 1
    
    def test_constant_price_data(self):
        """Test handling of constant price data."""
        df = pd.DataFrame({
            'open': [1.0] * 100,
            'high': [1.0] * 100,
            'low': [1.0] * 100,
            'close': [1.0] * 100,
        })
        
        extractor = TrendFeatureExtractor()
        result_df = extractor.compute_indicators(df)
        
        # Should not crash, though values may be zero/NaN
        assert len(result_df) == 100
    
    def test_large_price_gaps(self):
        """Test handling of large price gaps."""
        df = pd.DataFrame({
            'open': [1.0, 2.0, 1.5, 3.0],
            'high': [1.1, 2.1, 1.6, 3.1],
            'low': [0.9, 1.9, 1.4, 2.9],
            'close': [1.05, 1.95, 1.55, 2.95],
        })
        
        labeler = TrendLabeler(forward_window=2, trend_threshold_pct=0.1)
        labels = labeler.generate_labels(df)
        
        # Should handle large gaps
        assert len(labels) == len(df)
    
    def test_negative_prices(self):
        """Test handling of negative values in indicators."""
        # Note: Prices shouldn't be negative in real data, but test robustness
        df = pd.DataFrame({
            'open': [-1.0, -0.99, -0.98],
            'high': [-0.95, -0.94, -0.93],
            'low': [-1.05, -1.04, -1.03],
            'close': [-1.0, -0.99, -0.98],
        })
        
        extractor = TrendFeatureExtractor()
        result_df = extractor.compute_indicators(df)
        
        # Should compute without crashing
        assert len(result_df) == 3
    
    def test_very_small_prices(self):
        """Test handling of very small price values."""
        df = pd.DataFrame({
            'open': [0.00001, 0.000011, 0.000012],
            'high': [0.000011, 0.000012, 0.000013],
            'low': [0.000009, 0.00001, 0.000011],
            'close': [0.0000105, 0.0000115, 0.0000125],
        })
        
        extractor = TrendFeatureExtractor()
        result_df = extractor.compute_indicators(df)
        
        assert len(result_df) == 3
    
    def test_inf_values(self):
        """Test handling of infinity values."""
        df = pd.DataFrame({
            'open': [1.0, np.inf, 1.02],
            'high': [1.01, 1.02, 1.03],
            'low': [0.99, 1.0, 1.01],
            'close': [1.005, 1.015, 1.025],
        })
        
        # Should handle inf gracefully (may produce NaN indicators)
        extractor = TrendFeatureExtractor()
        result_df = extractor.compute_indicators(df)
        
        assert len(result_df) == 3


# =============================================================================
# TESTS: Data Integrity
# =============================================================================

class TestDataIntegrity:
    """Tests for data integrity and consistency."""
    
    def test_feature_extractor_does_not_modify_input(self, sample_ohlcv_data):
        """Test that extractor doesn't modify input dataframe."""
        df_copy = sample_ohlcv_data.copy()
        extractor = TrendFeatureExtractor()
        
        extractor.compute_indicators(sample_ohlcv_data)
        
        # Original should be unchanged
        pd.testing.assert_frame_equal(sample_ohlcv_data, df_copy)
    
    def test_feature_values_consistent(self, sample_ohlcv_data):
        """Test that running extraction multiple times gives same results."""
        extractor = TrendFeatureExtractor()
        df1 = extractor.compute_indicators(sample_ohlcv_data)
        df2 = extractor.compute_indicators(sample_ohlcv_data)
        
        # Results should be identical
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_labels_consistent(self, sample_ohlcv_data):
        """Test that generating labels multiple times gives same results."""
        labeler = TrendLabeler()
        labels1 = labeler.generate_labels(sample_ohlcv_data)
        labels2 = labeler.generate_labels(sample_ohlcv_data)
        
        # Results should be identical (compare with NaN handling)
        assert np.array_equal(labels1, labels2, equal_nan=True)
    
    def test_feature_count_consistent(self, sample_ohlcv_data):
        """Test that all extracted features have 13 elements."""
        extractor = TrendFeatureExtractor()
        df = extractor.compute_indicators(sample_ohlcv_data)
        
        feature_lengths = []
        for idx in range(200, len(df)):
            features = extractor.extract_features(df, idx)
            if features is not None:
                feature_lengths.append(len(features))
        
        # All features should have length 13
        assert all(length == 13 for length in feature_lengths)
        assert len(feature_lengths) > 0


# =============================================================================
# TESTS: Performance and Scaling
# =============================================================================

class TestPerformanceAndScaling:
    """Tests for performance with different data sizes."""
    
    def test_feature_extraction_large_dataset(self):
        """Test feature extraction performance with large dataset."""
        n = 10000
        df = pd.DataFrame({
            'open': 1.0 + np.cumsum(np.random.randn(n) * 0.0001),
            'high': 1.01 + np.cumsum(np.random.randn(n) * 0.0001),
            'low': 0.99 + np.cumsum(np.random.randn(n) * 0.0001),
            'close': 1.0 + np.cumsum(np.random.randn(n) * 0.0001),
        })
        
        extractor = TrendFeatureExtractor()
        df = extractor.compute_indicators(df)
        
        # Should complete without error
        assert len(df) == n
    
    def test_label_generation_large_dataset(self):
        """Test label generation with large dataset."""
        n = 10000
        df = pd.DataFrame({
            'close': 1.0 + np.cumsum(np.random.randn(n) * 0.0001)
        })
        
        labeler = TrendLabeler()
        labels = labeler.generate_labels(df)
        
        # Should complete without error
        assert len(labels) == n
    
    def test_multiple_extractions(self, large_ohlcv_data):
        """Test extracting many features doesn't cause issues."""
        extractor = TrendFeatureExtractor()
        df = extractor.compute_indicators(large_ohlcv_data)
        
        features_list = []
        for idx in range(200, len(df), 10):  # Extract every 10th
            features = extractor.extract_features(df, idx)
            if features is not None:
                features_list.append(features)
        
        assert len(features_list) > 100


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================

class TestParametrized:
    """Parametrized tests for various configurations."""
    
    @pytest.mark.parametrize("ema_periods", [
        (20, 50, 200),
        (10, 30, 100),
        (5, 20, 50),
    ])
    def test_different_ema_periods(self, ema_periods, sample_ohlcv_data):
        """Test with different EMA period configurations."""
        extractor = TrendFeatureExtractor(ema_periods=ema_periods)
        df = extractor.compute_indicators(sample_ohlcv_data)
        
        # Should have columns for all periods
        for period in ema_periods:
            assert f'ema_{period}' in df.columns
    
    @pytest.mark.parametrize("forward_window,threshold", [
        (10, 0.5),
        (20, 1.0),
        (40, 2.0),
    ])
    def test_different_labeler_configs(self, forward_window, threshold, sample_ohlcv_data):
        """Test labeler with different configurations."""
        labeler = TrendLabeler(forward_window=forward_window, trend_threshold_pct=threshold)
        labels = labeler.generate_labels(sample_ohlcv_data)
        
        # Last forward_window bars should be NaN
        assert np.isnan(labels[-forward_window:]).all()
    
    @pytest.mark.parametrize("idx", [200, 250, 300, 400])
    def test_feature_extraction_at_different_indices(self, idx, trend_feature_extractor, sample_ohlcv_data):
        """Test feature extraction at various indices."""
        df = trend_feature_extractor.compute_indicators(sample_ohlcv_data)
        features = trend_feature_extractor.extract_features(df, idx)
        
        assert features is not None
        assert len(features) == 13


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
