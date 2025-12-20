# tests/test_utils_data_loader.py
"""
Unit tests for utils/data_loader.py - Data loading and preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from utils.data_loader import DataConfig, DataLoader


@pytest.mark.unit
class TestDataConfig:
    """Test DataConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = DataConfig()

        assert config.sequence_length == 60
        assert config.label_strategy == 'ternary'
        assert config.ternary_threshold == 0.001
        assert config.scaler_type == 'minmax'
        assert config.feature_range == (0, 1)

    def test_custom_values(self):
        """Test custom configuration."""
        config = DataConfig(
            sequence_length=120,
            label_strategy='binary',
            scaler_type='standard',
        )

        assert config.sequence_length == 120
        assert config.label_strategy == 'binary'
        assert config.scaler_type == 'standard'


@pytest.mark.unit
class TestDataLoader:
    """Test DataLoader class."""

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file."""
        csv_path = tmp_path / "test_data.csv"

        # Create sample OHLCV data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=200, freq='1h'),
            'open': np.random.randn(200).cumsum() + 1.1,
            'high': np.random.randn(200).cumsum() + 1.11,
            'low': np.random.randn(200).cumsum() + 1.09,
            'close': np.random.randn(200).cumsum() + 1.1,
            'tick_volume': np.random.randint(100, 1000, 200),
        })

        df.to_csv(csv_path, index=False)
        return csv_path

    def test_init_default(self):
        """Test DataLoader initialization with defaults."""
        loader = DataLoader()

        assert loader.config.sequence_length == 60
        assert loader.config.label_strategy == 'ternary'
        assert not loader.is_fitted
        assert loader.scaler is not None

    def test_init_custom_config(self):
        """Test DataLoader initialization with custom config."""
        config = DataConfig(scaler_type='standard')
        loader = DataLoader(config)

        assert loader.config.scaler_type == 'standard'
        assert loader.scaler is not None

    def test_load_csv_success(self, sample_csv):
        """Test successful CSV loading."""
        loader = DataLoader()
        df = loader.load_csv(sample_csv)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200
        assert all(col in df.columns for col in DataLoader.REQUIRED_COLUMNS)

    def test_load_csv_missing_file(self):
        """Test loading non-existent file."""
        loader = DataLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_csv('nonexistent.csv')

    def test_load_csv_missing_columns(self, tmp_path):
        """Test loading CSV with missing required columns."""
        csv_path = tmp_path / "incomplete.csv"
        df = pd.DataFrame({
            'open': [1.1, 1.2],
            'close': [1.15, 1.25],
        })
        df.to_csv(csv_path, index=False)

        loader = DataLoader()

        with pytest.raises(ValueError, match="Missing required columns"):
            loader.load_csv(csv_path)

    def test_load_csv_with_nan_values(self, tmp_path):
        """Test loading CSV with NaN values."""
        csv_path = tmp_path / "with_nan.csv"
        df = pd.DataFrame({
            'open': [1.1, np.nan, 1.3],
            'high': [1.11, 1.21, 1.31],
            'low': [1.09, 1.19, 1.29],
            'close': [1.1, 1.2, 1.3],
            'tick_volume': [100, 200, 300],
        })
        df.to_csv(csv_path, index=False)

        loader = DataLoader()
        result = loader.load_csv(csv_path)

        # NaN row should be dropped
        assert len(result) == 2
        assert not result.isnull().any().any()

    def test_split_and_scale_basic(self, sample_csv):
        """Test basic split and scale operation."""
        loader = DataLoader()
        df = loader.load_csv(sample_csv)

        train, test, val = loader.split_and_scale(df, split_ratio=0.8, validation_ratio=0.0)

        # Returns numpy arrays
        assert isinstance(train, np.ndarray)
        assert isinstance(test, np.ndarray)
        assert len(train) == 160  # 80% of 200
        assert len(test) == 40    # 20% of 200
        assert val is None        # No validation when ratio is 0
        assert loader.is_fitted

    def test_split_and_scale_with_validation(self, sample_csv):
        """Test split with validation set."""
        loader = DataLoader()
        df = loader.load_csv(sample_csv)

        train, test, val = loader.split_and_scale(df, split_ratio=0.7, validation_ratio=0.15)

        # With validation_ratio=0.15, train gets reduced
        # 70% of 200 = 140, then 15% of 140 goes to val = 21, leaving 119 for train
        assert isinstance(train, np.ndarray)
        assert isinstance(val, np.ndarray)
        assert len(train) == 119  # 85% of 140
        assert len(test) == 60    # 30% of 200
        assert len(val) == 21     # 15% of 140
        assert loader.is_fitted

    def test_create_sequences_ternary(self, sample_csv):
        """Test sequence creation with ternary labeling."""
        config = DataConfig(sequence_length=10, label_strategy='ternary')
        loader = DataLoader(config)

        df = loader.load_csv(sample_csv)
        train, test, _ = loader.split_and_scale(df, split_ratio=0.8)

        X, y = loader.create_sequences(train)

        assert X.shape[1] == 10  # sequence_length
        assert X.shape[2] == 5   # OHLCV features
        assert y.ndim == 1
        assert set(np.unique(y)).issubset({0, 1, 2})  # Ternary classes

    def test_create_sequences_binary(self, sample_csv):
        """Test sequence creation with binary labeling."""
        config = DataConfig(sequence_length=10, label_strategy='binary')
        loader = DataLoader(config)

        df = loader.load_csv(sample_csv)
        train, test, _ = loader.split_and_scale(df, split_ratio=0.8)

        X, y = loader.create_sequences(train)

        assert X.shape[1] == 10
        assert set(np.unique(y)).issubset({0, 1})  # Binary classes

    def test_save_and_load_scaler(self, sample_csv, tmp_path):
        """Test scaler persistence."""
        config = DataConfig(scaler_type='minmax')
        loader = DataLoader(config)

        df = loader.load_csv(sample_csv)
        train, test, _ = loader.split_and_scale(df, split_ratio=0.8)

        # Save scaler
        scaler_path = tmp_path / "scaler.joblib"
        loader.save_scaler(scaler_path)

        assert scaler_path.exists()

        # Load scaler in new loader
        new_loader = DataLoader(config)
        new_loader.load_scaler(scaler_path)

        assert new_loader.is_fitted

        # Transform should give same results (test is already numpy array)
        test_transformed = loader.scaler.transform(test)
        new_test_transformed = new_loader.scaler.transform(test)

        np.testing.assert_array_almost_equal(test_transformed, new_test_transformed)

    def test_scaler_types(self, sample_csv):
        """Test different scaler types."""
        df_raw = pd.read_csv(sample_csv)

        for scaler_type in ['minmax', 'standard', 'robust']:
            config = DataConfig(scaler_type=scaler_type)
            loader = DataLoader(config)

            df = loader.load_csv(sample_csv)
            train, test, _ = loader.split_and_scale(df, split_ratio=0.8)

            assert loader.is_fitted
            assert len(train) > 0

    def test_feature_range_minmax(self, sample_csv):
        """Test MinMax scaler with custom feature range."""
        config = DataConfig(scaler_type='minmax', feature_range=(-1, 1))
        loader = DataLoader(config)

        df = loader.load_csv(sample_csv)
        train, test, _ = loader.split_and_scale(df, split_ratio=0.8)

        # train is numpy array, check all values are in range
        assert train.min() >= -1.01  # Allow small floating point error
        assert train.max() <= 1.01

    def test_insufficient_data_for_sequences(self):
        """Test create_sequences with insufficient data."""
        config = DataConfig(sequence_length=100)
        loader = DataLoader(config)

        # Create very small dataframe
        small_df = pd.DataFrame({
            'open': [1.1, 1.2],
            'high': [1.11, 1.21],
            'low': [1.09, 1.19],
            'close': [1.1, 1.2],
            'tick_volume': [100, 200],
        })

        X, y = loader.create_sequences(small_df, seq_len=100)

        # Should return empty arrays when insufficient data
        assert len(X) == 0 or X.shape[1] < 100

    def test_column_normalization(self, sample_csv):
        """Test that column names are normalized to lowercase."""
        # Create CSV with uppercase columns
        csv_with_caps = sample_csv.parent / "caps.csv"
        df = pd.read_csv(sample_csv)
        df.columns = [c.upper() for c in df.columns]
        df.to_csv(csv_with_caps, index=False)

        loader = DataLoader()
        result = loader.load_csv(csv_with_caps)

        # All columns should be lowercase
        assert all(c.islower() for c in result.columns)
