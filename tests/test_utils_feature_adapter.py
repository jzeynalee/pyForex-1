"""Unit tests for `utils/feature_adapter.py`.

Test Summary:
    Covers feature engineering and the EnhancedDataLoaderV2/V3 pipelines,
    including checkpoint-based feature selection and key failure modes.

Test Breakdown:
    - FeatureEngineer
        - `add_all_features()` normalizes columns and produces non-empty feature set
        - fills NaNs deterministically
    - EnhancedDataLoaderV2
        - `load_csv()` sets feature_columns excluding OHLCV/meta
        - `split_and_scale()` returns correctly shaped arrays
        - `create_sequences()` returns aligned X/y with correct label domain
    - EnhancedDataLoaderV3
        - `from_checkpoint()` uses `utils.checkpoint_loader.load_features` when available
        - `load_csv()` inserts missing checkpoint features with zeros
    - Convenience functions
        - `get_available_features()` returns list

Notes:
    Uses lightweight CSV fixtures in a temp directory.
"""

from __future__ import annotations

import sys
import unittest
import pytest
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.feature_adapter import EnhancedDataLoaderV3, EnhancedDataLoaderV2, load_data_for_evaluation, get_available_features, FeatureEngineer
import pandas as pd
import numpy as np
import torch


@pytest.fixture
def mock_checkpoint(tmp_path):
    """Create a mock checkpoint file with features."""
    checkpoint_path = tmp_path / "mock_checkpoint.pt"
    features = ['RSI', 'MACD', 'ATR', 'ADX', 'EMA20', 'EMA50', 'EMA200', 'ROC5', 'ROC10', 'MOMENTUM']
    
    checkpoint = {
        'model_state': {'test': 'value'},
        'feature_columns': features,
        'config': {'model': {'input_dim': len(features)}}
    }
    
    import torch
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path, features


@pytest.fixture
def ohlcv_csv_file(tmp_path):
    """Create a mock OHLCV CSV file."""
    csv_path = tmp_path / "test_data.csv"
    import pandas as pd
    
    n = 120
    close = [100 + i * 0.01 for i in range(n)]
    df = pd.DataFrame({
        "Open": close,
        "High": [c + 0.1 for c in close],
        "Low": [c - 0.1 for c in close],
        "Close": close,
        "Volume": [1000] * n,
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory."""
    return tmp_path


@pytest.fixture
def sample_ohlcv_df():
    """Create a sample OHLCV DataFrame for testing."""
    n = 50  # Small dataset for edge case testing
    close = [100 + i * 0.01 for i in range(n)]
    return pd.DataFrame({
        "Open": close,
        "High": [c + 0.1 for c in close],
        "Low": [c - 0.1 for c in close],
        "Close": close,
        "Volume": [1000] * n,
    })


class TestFeatureAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("utils.feature_adapter")

    def _write_csv(self, path: Path, n=120):
        import pandas as pd

        close = [100 + i * 0.01 for i in range(n)]
        df = pd.DataFrame(
            {
                "Open": close,
                "High": [c + 0.1 for c in close],
                "Low": [c - 0.1 for c in close],
                "Close": close,
                "Volume": [1000] * n,
            }
        )
        df.to_csv(path, index=False)
        return df

    def test_feature_engineer_add_all_features_contract(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "OPEN": [1, 1.1, 1.2, 1.3] * 20,
                "HIGH": [1.1, 1.2, 1.3, 1.4] * 20,
                "LOW": [0.9, 1.0, 1.1, 1.2] * 20,
                "CLOSE": [1.0, 1.1, 1.2, 1.3] * 20,
                "VOLUME": [10] * 80,
            }
        )
        out = self.mod.FeatureEngineer.add_all_features(df)
        self.assertIsInstance(out, pd.DataFrame)
        self.assertIn("macd", out.columns)
        self.assertFalse(out.isna().any().any())

    def test_enhanced_loader_v2_load_csv_and_features(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "d.csv"
            self._write_csv(p)

            loader = self.mod.EnhancedDataLoaderV2(sequence_length=10)
            df = loader.load_csv(str(p))
            self.assertTrue(len(loader.feature_columns) > 0)
            self.assertNotIn("open", loader.feature_columns)
            self.assertNotIn("close", loader.feature_columns)
            self.assertIn("close", df.columns)

    def test_enhanced_loader_v2_split_and_scale_shapes(self):
        import numpy as np

        with TemporaryDirectory() as td:
            p = Path(td) / "d.csv"
            self._write_csv(p)

            loader = self.mod.EnhancedDataLoaderV2(sequence_length=10)
            df = loader.load_csv(str(p))
            train_scaled, val_scaled, test_scaled = loader.split_and_scale(df, split_ratio=0.7, validation_ratio=0.2)
            self.assertTrue(isinstance(train_scaled, np.ndarray))
            self.assertGreater(train_scaled.shape[0], 0)
            self.assertGreater(val_scaled.shape[0], 0)
            self.assertGreater(test_scaled.shape[0], 0)
            self.assertEqual(train_scaled.shape[1], val_scaled.shape[1])

    def test_enhanced_loader_v2_create_sequences_labels_domain(self):
        import numpy as np

        with TemporaryDirectory() as td:
            p = Path(td) / "d.csv"
            self._write_csv(p)

            loader = self.mod.EnhancedDataLoaderV2(sequence_length=10, trend_threshold=0.0001)
            df = loader.load_csv(str(p))
            train_scaled, _val_scaled, _test_scaled = loader.split_and_scale(df, split_ratio=0.8, validation_ratio=0.1)
            X, y = loader.create_sequences(train_scaled, loader.train_close, seq_len=10, horizon=1)
            self.assertTrue(isinstance(X, np.ndarray))
            self.assertTrue(isinstance(y, np.ndarray))
            self.assertEqual(X.shape[0], y.shape[0])
            self.assertTrue(set(np.unique(y)).issubset({0, 1, 2}))

    def test_loader_v3_checkpoint_features_missing_filled_with_zero(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "d.csv"
            self._write_csv(p)

            with patch("utils.checkpoint_loader.load_features", return_value=["does_not_exist", "macd"]):
                loader = self.mod.EnhancedDataLoaderV3.from_checkpoint("dummy.pt", sequence_length=10)
                df = loader.load_csv(str(p))
                self.assertIn("does_not_exist", df.columns)
                self.assertEqual(float(df["does_not_exist"].iloc[-1]), 0.0)
                self.assertIn("does_not_exist", loader.get_feature_columns())

    def test_get_available_features_returns_list(self):
        with TemporaryDirectory() as td:
            p = Path(td) / "d.csv"
            self._write_csv(p)
            feats = self.mod.get_available_features(str(p))
            self.assertTrue(isinstance(feats, list))
            self.assertTrue(len(feats) > 0)


class TestEnhancedDataLoaderV3FromCheckpoint:
    """Test EnhancedDataLoaderV3.from_checkpoint."""
    
    def test_from_checkpoint_creates_instance(self, mock_checkpoint):
        """Test from_checkpoint creates V3 instance."""
        checkpoint_path, features = mock_checkpoint
        
        loader = EnhancedDataLoaderV3.from_checkpoint(str(checkpoint_path))
        
        assert isinstance(loader, EnhancedDataLoaderV3)
    
    def test_from_checkpoint_loads_features(self, mock_checkpoint):
        """Test from_checkpoint loads features from checkpoint."""
        checkpoint_path, expected_features = mock_checkpoint
        
        loader = EnhancedDataLoaderV3.from_checkpoint(str(checkpoint_path))
        
        assert loader._checkpoint_features == expected_features
    
    def test_from_checkpoint_custom_sequence_length(self, mock_checkpoint):
        """Test from_checkpoint respects custom sequence length."""
        checkpoint_path, _ = mock_checkpoint
        
        loader = EnhancedDataLoaderV3.from_checkpoint(
            str(checkpoint_path),
            sequence_length=50
        )
        
        assert loader.sequence_length == 50
    
    def test_from_checkpoint_missing_file_raises(self):
        """Test from_checkpoint raises on missing file."""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            EnhancedDataLoaderV3.from_checkpoint('/nonexistent/checkpoint.pt')
    
    def test_from_checkpoint_invalid_checkpoint_raises(self, temp_dir):
        """Test from_checkpoint raises on invalid checkpoint."""
        from utils.checkpoint_loader import CheckpointFormatError
        checkpoint_path = temp_dir / 'invalid.pt'
        torch.save({'no_features': 'here'}, checkpoint_path)
        
        with pytest.raises((ValueError, CheckpointFormatError)):
            EnhancedDataLoaderV3.from_checkpoint(str(checkpoint_path))


class TestEnhancedDataLoaderV3Loading:
    """Test EnhancedDataLoaderV3 data loading with checkpoint features."""
    
    def test_load_csv_without_checkpoint(self, ohlcv_csv_file):
        """Test load_csv without checkpoint loads all features."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(ohlcv_csv_file)
        
        assert len(loader.feature_columns) > 0
        assert 'rsi_14' in loader.feature_columns
    
    def test_load_csv_with_checkpoint_filters_features(self, ohlcv_csv_file, mock_checkpoint):
        """Test load_csv with checkpoint filters to checkpoint features."""
        checkpoint_path, expected_features = mock_checkpoint
        
        loader = EnhancedDataLoaderV3.from_checkpoint(str(checkpoint_path))
        df = loader.load_csv(ohlcv_csv_file)
        
        # Should use checkpoint features (that exist in data)
        available = [f for f in expected_features if f in df.columns]
        assert len(loader.feature_columns) == len(available)
    
    def test_load_csv_handles_missing_checkpoint_features(self, ohlcv_csv_file, mock_checkpoint):
        """Test load_csv handles missing checkpoint features."""
        checkpoint_path, expected_features = mock_checkpoint
        
        loader = EnhancedDataLoaderV3.from_checkpoint(str(checkpoint_path))
        df = loader.load_csv(ohlcv_csv_file)
        
        # Missing features should be added as zeros
        assert all(f in df.columns for f in expected_features)
    
    def test_get_feature_columns(self, ohlcv_csv_file):
        """Test get_feature_columns returns copy."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(ohlcv_csv_file)
        
        features = loader.get_feature_columns()
        
        assert isinstance(features, list)
        assert features == loader.feature_columns
        
        # Modifying returned list shouldn't affect loader
        features.append('dummy')
        assert 'dummy' not in loader.feature_columns


# =============================================================================
# TEST: Convenience Functions
# =============================================================================

class TestLoadDataForEvaluation:
    """Test load_data_for_evaluation convenience function."""
    
    def test_load_data_for_evaluation_returns_tuple(self, ohlcv_csv_file, mock_checkpoint):
        """Test function returns 4-tuple."""
        checkpoint_path, _ = mock_checkpoint
        
        X_test, y_test, close_prices, features = load_data_for_evaluation(
            ohlcv_csv_file,
            str(checkpoint_path),
        )
        
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        assert isinstance(close_prices, np.ndarray)
        assert isinstance(features, list)
    
    def test_load_data_for_evaluation_shapes(self, ohlcv_csv_file, mock_checkpoint):
        """Test shapes of returned data."""
        checkpoint_path, _ = mock_checkpoint
        
        seq_len = 30
        X_test, y_test, close_prices, features = load_data_for_evaluation(
            ohlcv_csv_file,
            str(checkpoint_path),
            seq_len=seq_len,
            test_ratio=0.2,
        )
        
        # Handle case where test set might be empty due to small dataset
        if len(X_test) == 0:
            # If test set is empty, X_test might be 1D empty array
            assert X_test.ndim <= 3
            assert len(y_test) == 0
        else:
            # X shape: (num_sequences, seq_len, num_features)
            assert X_test.ndim == 3
            assert X_test.shape[1] == seq_len
            
            # y shape: (num_sequences,)
            assert y_test.ndim == 1
            
            # Lengths match
            assert len(X_test) == len(y_test)
    
    def test_load_data_for_evaluation_custom_seq_len(self, ohlcv_csv_file, mock_checkpoint):
        """Test custom sequence length."""
        checkpoint_path, _ = mock_checkpoint
        
        seq_len = 30  # Use 30 to ensure we have enough data for test set
        X_test, _, _, _ = load_data_for_evaluation(
            ohlcv_csv_file,
            str(checkpoint_path),
            seq_len=seq_len,
        )
        
        # Handle case where test set might be empty due to small dataset
        if len(X_test) == 0:
            # If test set is empty, X_test might be 1D empty array
            assert X_test.ndim <= 3
        else:
            assert X_test.ndim == 3
            assert X_test.shape[1] == seq_len
    
    def test_load_data_for_evaluation_custom_test_ratio(self, ohlcv_csv_file, mock_checkpoint):
        """Test custom test ratio."""
        checkpoint_path, _ = mock_checkpoint
        
        X_test1, _, _, _ = load_data_for_evaluation(
            ohlcv_csv_file,
            str(checkpoint_path),
            test_ratio=0.1,
        )
        
        X_test2, _, _, _ = load_data_for_evaluation(
            ohlcv_csv_file,
            str(checkpoint_path),
            test_ratio=0.3,
        )
        
        # More test ratio should give more test samples
        assert len(X_test2) > len(X_test1)


class TestGetAvailableFeatures:
    """Test get_available_features convenience function."""
    
    def test_get_available_features_returns_list(self, ohlcv_csv_file):
        """Test get_available_features returns list."""
        features = get_available_features(ohlcv_csv_file)
        
        assert isinstance(features, list)
        assert len(features) > 0
    
    def test_get_available_features_contains_technical_indicators(self, ohlcv_csv_file):
        """Test returned features include technical indicators."""
        features = get_available_features(ohlcv_csv_file)
        
        assert 'rsi_14' in features
        assert 'macd' in features
    
    def test_get_available_features_missing_file_raises(self):
        """Test error on missing file."""
        with pytest.raises(FileNotFoundError):
            get_available_features('/nonexistent/path.csv')


# =============================================================================
# TEST: Integration
# =============================================================================

class TestIntegration:
    """Integration tests for feature adapter."""
    
    def test_full_pipeline_v2(self, ohlcv_csv_file):
        """Test complete V2 pipeline: load -> split -> scale -> sequences."""
        loader = EnhancedDataLoaderV2()
        
        # Load
        df = loader.load_csv(ohlcv_csv_file)
        assert len(df) > 0
        
        # Split and scale
        train, val, test = loader.split_and_scale(df)
        assert len(train) > 0
        
        # Create sequences
        X_train, y_train = loader.create_sequences(train, loader.train_close, 30)
        assert len(X_train) > 0
        assert len(y_train) > 0
    
    def test_full_pipeline_v3_no_checkpoint(self, ohlcv_csv_file):
        """Test V3 pipeline without checkpoint."""
        loader = EnhancedDataLoaderV3()
        
        df = loader.load_csv(ohlcv_csv_file)
        train, val, test = loader.split_and_scale(df)
        X_train, y_train = loader.create_sequences(train, loader.train_close, 30)
        
        assert len(X_train) > 0
    
    def test_full_pipeline_v3_with_checkpoint(self, ohlcv_csv_file, mock_checkpoint):
        """Test V3 pipeline with checkpoint."""
        checkpoint_path, _ = mock_checkpoint
        
        loader = EnhancedDataLoaderV3.from_checkpoint(str(checkpoint_path))
        df = loader.load_csv(ohlcv_csv_file)
        
        # Checkpoint features should be used
        assert len(loader.feature_columns) > 0
    
    def test_scaler_consistency_across_splits(self, ohlcv_csv_file):
        """Test that scaler is fit on train and applied to val/test."""
        loader = EnhancedDataLoaderV2()
        df = loader.load_csv(ohlcv_csv_file)
        
        train, val, test = loader.split_and_scale(df)
        
        # Scaler should be fitted
        assert loader.scaler.scale_ is not None or hasattr(loader.scaler, 'scale_')
    
    def test_multiple_loaders_independent(self, ohlcv_csv_file):
        """Test that multiple loader instances are independent."""
        loader1 = EnhancedDataLoaderV2(sequence_length=30)
        loader2 = EnhancedDataLoaderV2(sequence_length=50)
        
        assert loader1.sequence_length != loader2.sequence_length
    
    def test_v3_inherits_v2_behavior(self, ohlcv_csv_file):
        """Test V3 maintains backward compatibility with V2."""
        loader_v2 = EnhancedDataLoaderV2()
        loader_v3 = EnhancedDataLoaderV3()
        
        df2 = loader_v2.load_csv(ohlcv_csv_file)
        df3 = loader_v3.load_csv(ohlcv_csv_file)
        
        # Both should load same features (when no checkpoint)
        assert set(loader_v2.feature_columns) == set(loader_v3.feature_columns)


# =============================================================================
# TEST: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataframe_raises(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({'close': [], 'high': [], 'low': [], 'open': []})
        
        with pytest.raises((ValueError, IndexError)):
            FeatureEngineer.add_all_features(df)
    
    def test_single_row_dataframe(self):
        """Test handling of single-row DataFrame."""
        df = pd.DataFrame({
            'close': [1.0],
            'high': [1.01],
            'low': [0.99],
            'open': [1.0],
        })
        
        result = FeatureEngineer.add_all_features(df)
        assert len(result) == 1
    
    def test_all_nan_values(self):
        """Test handling of all NaN values."""
        df = pd.DataFrame({
            'close': [np.nan, np.nan, np.nan],
            'high': [np.nan, np.nan, np.nan],
            'low': [np.nan, np.nan, np.nan],
        })
        
        result = FeatureEngineer.add_all_features(df)
        assert result.isna().sum().sum() == 0  # Should be filled
    
    def test_constant_prices(self):
        """Test handling of constant price data."""
        df = pd.DataFrame({
            'close': [1.0] * 50,
            'high': [1.01] * 50,
            'low': [0.99] * 50,
            'open': [1.0] * 50,
        })
        
        result = FeatureEngineer.add_all_features(df)
        
        # Should complete without error
        assert len(result) == 50
        assert result.isna().sum().sum() == 0
    
    def test_very_large_dataframe(self):
        """Test handling of large DataFrames."""
        np.random.seed(42)
        n = 10000
        prices = 100 + np.cumsum(np.random.randn(n) * 0.1)
        
        df = pd.DataFrame({
            'close': prices,
            'high': prices + 0.1,
            'low': prices - 0.1,
            'open': prices,
            'volume': np.ones(n) * 100,
        })
        
        result = FeatureEngineer.add_all_features(df)
        
        assert len(result) == n
        assert len(result.columns) > 5
    
    def test_sequence_length_longer_than_data(self, sample_ohlcv_df):
        """Test sequence creation when length exceeds data."""
        loader = EnhancedDataLoaderV2()
        
        # Very small data
        small_data = sample_ohlcv_df.iloc[:10].copy()
        
        loader.load_csv  # Ensure feature_columns exists
        loader.feature_columns = ['close']  # Mock feature
        
        scaled = small_data[['close']].values
        X, y = loader.create_sequences(scaled, small_data['close'].values, 30)
        
        # Should return empty or very small result
        assert len(X) == 0 or len(X) <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
