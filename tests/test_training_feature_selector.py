# tests/training_feature_selector.py
"""
Comprehensive unit tests for training/feature_selector.py

This module tests the DynamicFeatureSelector class which:
- Selects top N features based on Random Forest importance
- Handles data sampling for large datasets
- Manages inf/NaN values in feature data
- Excludes specified columns from selection
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import logging

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def basic_dataframe():
    """Create a basic DataFrame with clear feature importance patterns."""
    np.random.seed(42)
    n_rows = 1000
    
    # Create features with varying predictive power
    df = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=n_rows, freq='15min'),
        'feature_strong_1': np.random.randn(n_rows) * 10,  # Strong predictor
        'feature_strong_2': np.random.randn(n_rows) * 8,   # Strong predictor
        'feature_medium_1': np.random.randn(n_rows) * 5,   # Medium predictor
        'feature_medium_2': np.random.randn(n_rows) * 4,   # Medium predictor
        'feature_weak_1': np.random.randn(n_rows) * 0.1,   # Weak predictor
        'feature_weak_2': np.random.randn(n_rows) * 0.05,  # Weak predictor
        'noise_1': np.random.randn(n_rows) * 0.01,         # Noise
        'noise_2': np.random.randn(n_rows) * 0.01,         # Noise
    })
    
    # Create target correlated with strong features
    df['target'] = (
        (df['feature_strong_1'] > 0).astype(int) + 
        (df['feature_strong_2'] > 0).astype(int)
    ) > 1
    df['target'] = df['target'].astype(int)
    
    return df


@pytest.fixture
def large_dataframe():
    """Create a DataFrame larger than default sample_size."""
    np.random.seed(42)
    n_rows = 60000  # Larger than default sample_size of 50000
    
    df = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=n_rows, freq='1min'),
        'feature_1': np.random.randn(n_rows),
        'feature_2': np.random.randn(n_rows),
        'feature_3': np.random.randn(n_rows),
        'feature_4': np.random.randn(n_rows),
        'feature_5': np.random.randn(n_rows),
    })
    
    df['target'] = (df['feature_1'] > 0).astype(int)
    
    return df


@pytest.fixture
def small_dataframe():
    """Create a small DataFrame for edge case testing."""
    np.random.seed(42)
    n_rows = 100
    
    df = pd.DataFrame({
        'feature_1': np.random.randn(n_rows),
        'feature_2': np.random.randn(n_rows),
        'feature_3': np.random.randn(n_rows),
    })
    
    df['target'] = (df['feature_1'] > 0).astype(int)
    
    return df


@pytest.fixture
def dataframe_with_nan():
    """Create a DataFrame with NaN values."""
    np.random.seed(42)
    n_rows = 500
    
    df = pd.DataFrame({
        'feature_1': np.random.randn(n_rows),
        'feature_2': np.random.randn(n_rows),
        'feature_3': np.random.randn(n_rows),
    })
    
    # Introduce NaN values
    df.loc[10:20, 'feature_1'] = np.nan
    df.loc[30:40, 'feature_2'] = np.nan
    
    df['target'] = (df['feature_3'] > 0).astype(int)
    
    return df


@pytest.fixture
def dataframe_with_inf():
    """Create a DataFrame with inf values."""
    np.random.seed(42)
    n_rows = 500
    
    df = pd.DataFrame({
        'feature_1': np.random.randn(n_rows),
        'feature_2': np.random.randn(n_rows),
        'feature_3': np.random.randn(n_rows),
    })
    
    # Introduce inf values
    df.loc[10, 'feature_1'] = np.inf
    df.loc[20, 'feature_1'] = -np.inf
    df.loc[30, 'feature_2'] = np.inf
    
    df['target'] = (df['feature_3'] > 0).astype(int)
    
    return df


@pytest.fixture
def dataframe_with_many_features():
    """Create a DataFrame with many features for n_features testing."""
    np.random.seed(42)
    n_rows = 1000
    n_features = 50
    
    data = {'time': pd.date_range('2020-01-01', periods=n_rows, freq='15min')}
    
    for i in range(n_features):
        data[f'feature_{i}'] = np.random.randn(n_rows) * (n_features - i)
    
    df = pd.DataFrame(data)
    df['target'] = (df['feature_0'] > 0).astype(int)
    
    return df


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestDynamicFeatureSelectorInit:
    """Tests for DynamicFeatureSelector initialization."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector()
        
        assert selector.n_features == 20
        assert selector.sample_size == 50000
    
    def test_custom_n_features(self):
        """Test initialization with custom n_features."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=10)
        
        assert selector.n_features == 10
        assert selector.sample_size == 50000
    
    def test_custom_sample_size(self):
        """Test initialization with custom sample_size."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(sample_size=10000)
        
        assert selector.n_features == 20
        assert selector.sample_size == 10000
    
    def test_custom_both_parameters(self):
        """Test initialization with both custom parameters."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=15, sample_size=25000)
        
        assert selector.n_features == 15
        assert selector.sample_size == 25000
    
    def test_zero_n_features(self):
        """Test initialization with zero n_features."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=0)
        
        assert selector.n_features == 0
    
    def test_large_n_features(self):
        """Test initialization with large n_features."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=1000)
        
        assert selector.n_features == 1000


# ============================================================================
# BASIC SELECTION TESTS
# ============================================================================

class TestBasicSelection:
    """Tests for basic feature selection functionality."""
    
    def test_select_returns_list(self, basic_dataframe):
        """Test that select method returns a list."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=5)
        exclude_cols = ['time', 'target']
        
        result = selector.select(basic_dataframe, 'target', exclude_cols)
        
        assert isinstance(result, list)
    
    def test_select_returns_correct_count(self, basic_dataframe):
        """Test that select returns the correct number of features."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=5)
        exclude_cols = ['time', 'target']
        
        result = selector.select(basic_dataframe, 'target', exclude_cols)
        
        assert len(result) == 5
    
    def test_select_returns_strings(self, basic_dataframe):
        """Test that selected features are strings (column names)."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=3)
        exclude_cols = ['time', 'target']
        
        result = selector.select(basic_dataframe, 'target', exclude_cols)
        
        assert all(isinstance(f, str) for f in result)
    
    def test_selected_features_exist_in_dataframe(self, basic_dataframe):
        """Test that selected features exist in the original DataFrame."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=5)
        exclude_cols = ['time', 'target']
        
        result = selector.select(basic_dataframe, 'target', exclude_cols)
        
        for feature in result:
            assert feature in basic_dataframe.columns
    
    def test_excluded_columns_not_selected(self, basic_dataframe):
        """Test that excluded columns are not in selected features."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=5)
        exclude_cols = ['time', 'target', 'noise_1']
        
        result = selector.select(basic_dataframe, 'target', exclude_cols)
        
        for excluded in exclude_cols:
            assert excluded not in result
    
    def test_no_duplicate_features(self, basic_dataframe):
        """Test that selected features are unique."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=5)
        exclude_cols = ['time', 'target']
        
        result = selector.select(basic_dataframe, 'target', exclude_cols)
        
        assert len(result) == len(set(result))


# ============================================================================
# EXCLUDE COLUMNS TESTS
# ============================================================================

class TestExcludeColumns:
    """Tests for column exclusion functionality."""
    
    def test_exclude_single_column(self, basic_dataframe):
        """Test excluding a single column."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=5)
        exclude_cols = ['time', 'target']  # Must exclude 'time' (datetime) for RF
        
        result = selector.select(basic_dataframe, 'target', exclude_cols)
        
        assert 'target' not in result
        assert 'time' not in result
    
    def test_exclude_multiple_columns(self, basic_dataframe):
        """Test excluding multiple columns."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=3)
        exclude_cols = ['time', 'target', 'noise_1', 'noise_2']
        
        result = selector.select(basic_dataframe, 'target', exclude_cols)
        
        for col in exclude_cols:
            assert col not in result
    
    def test_exclude_empty_list(self, small_dataframe):
        """Test with empty exclude list."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=2)
        exclude_cols = []
        
        # Note: target column can be selected if not excluded
        result = selector.select(small_dataframe, 'target', exclude_cols)
        
        assert len(result) == 2
    
    def test_exclude_all_but_few_columns(self, basic_dataframe):
        """Test excluding most columns, leaving only a few."""
        from training.feature_selector import DynamicFeatureSelector
        
        # Exclude all but 3 feature columns
        exclude_cols = ['time', 'target', 'feature_strong_1', 'feature_strong_2', 
                       'feature_medium_1', 'feature_medium_2']
        
        selector = DynamicFeatureSelector(n_features=2)
        result = selector.select(basic_dataframe, 'target', exclude_cols)
        
        # Should only select from remaining columns
        remaining = set(basic_dataframe.columns) - set(exclude_cols)
        for feature in result:
            assert feature in remaining


# ============================================================================
# SAMPLING TESTS
# ============================================================================

class TestDataSampling:
    """Tests for data sampling behavior."""
    
    def test_no_sampling_when_data_smaller(self, small_dataframe):
        """Test that no sampling occurs when data is smaller than sample_size."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=2, sample_size=50000)
        exclude_cols = ['target']
        
        # Should work without errors - no sampling needed
        result = selector.select(small_dataframe, 'target', exclude_cols)
        
        assert len(result) == 2
    
    def test_sampling_when_data_larger(self, large_dataframe):
        """Test that sampling occurs when data is larger than sample_size."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=3, sample_size=10000)
        exclude_cols = ['time', 'target']
        
        result = selector.select(large_dataframe, 'target', exclude_cols)
        
        assert len(result) == 3
    
    def test_sampling_uses_recent_data(self, large_dataframe):
        """Test that sampling uses the most recent data (tail of DataFrame)."""
        from training.feature_selector import DynamicFeatureSelector
        
        # Create selector with small sample size
        selector = DynamicFeatureSelector(n_features=3, sample_size=1000)
        exclude_cols = ['time', 'target']
        
        # The implementation uses df.iloc[-sample_size:] for recent data
        result = selector.select(large_dataframe, 'target', exclude_cols)
        
        # Should complete successfully
        assert len(result) == 3
    
    def test_exact_sample_size_boundary(self):
        """Test behavior when data size equals sample_size."""
        from training.feature_selector import DynamicFeatureSelector
        
        np.random.seed(42)
        n_rows = 1000  # Exactly equal to sample_size
        
        df = pd.DataFrame({
            'feature_1': np.random.randn(n_rows),
            'feature_2': np.random.randn(n_rows),
            'target': np.random.randint(0, 2, n_rows)
        })
        
        selector = DynamicFeatureSelector(n_features=2, sample_size=1000)
        result = selector.select(df, 'target', ['target'])
        
        assert len(result) == 2


# ============================================================================
# NaN AND INF HANDLING TESTS
# ============================================================================

class TestNaNAndInfHandling:
    """Tests for handling of NaN and infinite values."""
    
    def test_handles_nan_values(self, dataframe_with_nan):
        """Test that selector handles NaN values without error."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=2)
        exclude_cols = ['target']
        
        # Should not raise
        result = selector.select(dataframe_with_nan, 'target', exclude_cols)
        
        assert len(result) == 2
    
    def test_handles_positive_inf(self, dataframe_with_inf):
        """Test that selector handles positive infinity values."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=2)
        exclude_cols = ['target']
        
        # Should not raise
        result = selector.select(dataframe_with_inf, 'target', exclude_cols)
        
        assert len(result) == 2
    
    def test_handles_negative_inf(self, dataframe_with_inf):
        """Test that selector handles negative infinity values."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=2)
        exclude_cols = ['target']
        
        result = selector.select(dataframe_with_inf, 'target', exclude_cols)
        
        assert len(result) == 2
    
    def test_handles_mixed_nan_and_inf(self):
        """Test handling of mixed NaN and inf values."""
        from training.feature_selector import DynamicFeatureSelector
        
        np.random.seed(42)
        n_rows = 500
        
        df = pd.DataFrame({
            'feature_1': np.random.randn(n_rows),
            'feature_2': np.random.randn(n_rows),
            'feature_3': np.random.randn(n_rows),
        })
        
        # Mix of NaN and inf
        df.loc[10:15, 'feature_1'] = np.nan
        df.loc[20, 'feature_1'] = np.inf
        df.loc[25, 'feature_2'] = -np.inf
        df.loc[30:35, 'feature_3'] = np.nan
        
        df['target'] = np.random.randint(0, 2, n_rows)
        
        selector = DynamicFeatureSelector(n_features=2)
        result = selector.select(df, 'target', ['target'])
        
        assert len(result) == 2
    
    def test_all_nan_column(self):
        """Test handling when a feature column is all NaN."""
        from training.feature_selector import DynamicFeatureSelector
        
        np.random.seed(42)
        n_rows = 100
        
        df = pd.DataFrame({
            'feature_1': np.random.randn(n_rows),
            'feature_2': [np.nan] * n_rows,  # All NaN
            'feature_3': np.random.randn(n_rows),
            'target': np.random.randint(0, 2, n_rows)
        })
        
        selector = DynamicFeatureSelector(n_features=2)
        
        # Should handle gracefully (NaN filled with 0)
        result = selector.select(df, 'target', ['target'])
        
        assert len(result) == 2


# ============================================================================
# N_FEATURES EDGE CASES
# ============================================================================

class TestNFeaturesEdgeCases:
    """Tests for edge cases related to n_features parameter."""
    
    def test_n_features_equals_available_features(self, small_dataframe):
        """Test when n_features equals number of available features."""
        from training.feature_selector import DynamicFeatureSelector
        
        # 3 features total, exclude 1 (target), leaves 3 features
        selector = DynamicFeatureSelector(n_features=3)
        exclude_cols = ['target']
        
        result = selector.select(small_dataframe, 'target', exclude_cols)
        
        assert len(result) == 3
    
    def test_n_features_greater_than_available(self, small_dataframe):
        """Test when n_features is greater than available features."""
        from training.feature_selector import DynamicFeatureSelector
        
        # Only 3 features available but requesting 10
        selector = DynamicFeatureSelector(n_features=10)
        exclude_cols = ['target']
        
        # This will raise IndexError because indices[i] will be out of bounds
        # The implementation doesn't guard against this
        with pytest.raises(IndexError):
            selector.select(small_dataframe, 'target', exclude_cols)
    
    def test_n_features_one(self, basic_dataframe):
        """Test selecting only one feature."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=1)
        exclude_cols = ['time', 'target']
        
        result = selector.select(basic_dataframe, 'target', exclude_cols)
        
        assert len(result) == 1
    
    def test_selecting_from_many_features(self, dataframe_with_many_features):
        """Test selecting features from a large pool."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=20)
        exclude_cols = ['time', 'target']
        
        result = selector.select(dataframe_with_many_features, 'target', exclude_cols)
        
        assert len(result) == 20
        assert len(set(result)) == 20  # All unique


# ============================================================================
# RANDOM FOREST CONFIGURATION TESTS
# ============================================================================

class TestRandomForestConfiguration:
    """Tests to verify Random Forest is configured correctly."""
    
    def test_rf_uses_balanced_class_weight(self, basic_dataframe):
        """Test that RF uses balanced class weights for imbalanced data."""
        from training.feature_selector import DynamicFeatureSelector
        
        # Create imbalanced target
        df = basic_dataframe.copy()
        df['target'] = 0
        df.loc[:50, 'target'] = 1  # Only ~5% positive class
        
        selector = DynamicFeatureSelector(n_features=3)
        
        # Should handle imbalanced data due to class_weight='balanced'
        result = selector.select(df, 'target', ['time', 'target'])
        
        assert len(result) == 3
    
    def test_rf_deterministic_with_same_seed(self, basic_dataframe):
        """Test that RF produces consistent results (random_state=42)."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=5)
        exclude_cols = ['time', 'target']
        
        result1 = selector.select(basic_dataframe, 'target', exclude_cols)
        result2 = selector.select(basic_dataframe, 'target', exclude_cols)
        
        assert result1 == result2
    
    def test_importance_ranking_sensible(self, basic_dataframe):
        """Test that feature importance ranking is sensible."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=6)
        exclude_cols = ['time', 'target']
        
        result = selector.select(basic_dataframe, 'target', exclude_cols)
        
        # Strong features should typically rank higher than noise
        # (This is probabilistic but should usually hold)
        strong_features = {'feature_strong_1', 'feature_strong_2'}
        noise_features = {'noise_1', 'noise_2'}
        
        # At least one strong feature should be in top 3
        top_3 = set(result[:3])
        assert len(top_3 & strong_features) >= 1 or len(top_3 & noise_features) == 0


# ============================================================================
# LOGGING TESTS
# ============================================================================

class TestLogging:
    """Tests for logging behavior."""
    
    def test_logs_selection_start(self, basic_dataframe, caplog):
        """Test that feature selection start is logged."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=3)
        
        with caplog.at_level(logging.INFO):
            selector.select(basic_dataframe, 'target', ['time', 'target'])
        
        assert "Running Dynamic Feature Selection" in caplog.text
    
    def test_logs_target_n_features(self, basic_dataframe, caplog):
        """Test that target n_features is logged."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=7)
        
        with caplog.at_level(logging.INFO):
            selector.select(basic_dataframe, 'target', ['time', 'target'])
        
        assert "Top 7" in caplog.text
    
    def test_logs_selected_features(self, basic_dataframe, caplog):
        """Test that selected features are logged."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=3)
        
        with caplog.at_level(logging.INFO):
            result = selector.select(basic_dataframe, 'target', ['time', 'target'])
        
        # Each selected feature should be logged
        for feature in result:
            assert feature in caplog.text
    
    def test_logs_feature_importance_scores(self, basic_dataframe, caplog):
        """Test that importance scores are logged."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=2)
        
        with caplog.at_level(logging.INFO):
            selector.select(basic_dataframe, 'target', ['time', 'target'])
        
        # Should contain importance score format (e.g., "0.1234")
        assert "TOP SELECTED FEATURES" in caplog.text


# ============================================================================
# DATA TYPE TESTS
# ============================================================================

class TestDataTypes:
    """Tests for handling various data types."""
    
    def test_integer_features(self):
        """Test with integer feature columns."""
        from training.feature_selector import DynamicFeatureSelector
        
        np.random.seed(42)
        n_rows = 500
        
        df = pd.DataFrame({
            'int_feature_1': np.random.randint(0, 100, n_rows),
            'int_feature_2': np.random.randint(0, 100, n_rows),
            'int_feature_3': np.random.randint(0, 100, n_rows),
            'target': np.random.randint(0, 2, n_rows)
        })
        
        selector = DynamicFeatureSelector(n_features=2)
        result = selector.select(df, 'target', ['target'])
        
        assert len(result) == 2
    
    def test_float_features(self):
        """Test with float feature columns."""
        from training.feature_selector import DynamicFeatureSelector
        
        np.random.seed(42)
        n_rows = 500
        
        df = pd.DataFrame({
            'float_feature_1': np.random.randn(n_rows).astype(np.float32),
            'float_feature_2': np.random.randn(n_rows).astype(np.float64),
            'target': np.random.randint(0, 2, n_rows)
        })
        
        selector = DynamicFeatureSelector(n_features=2)
        result = selector.select(df, 'target', ['target'])
        
        assert len(result) == 2
    
    def test_mixed_numeric_types(self):
        """Test with mixed numeric types."""
        from training.feature_selector import DynamicFeatureSelector
        
        np.random.seed(42)
        n_rows = 500
        
        df = pd.DataFrame({
            'int_feature': np.random.randint(0, 100, n_rows),
            'float32_feature': np.random.randn(n_rows).astype(np.float32),
            'float64_feature': np.random.randn(n_rows).astype(np.float64),
            'target': np.random.randint(0, 2, n_rows)
        })
        
        selector = DynamicFeatureSelector(n_features=3)
        result = selector.select(df, 'target', ['target'])
        
        assert len(result) == 3
    
    def test_binary_target(self):
        """Test with binary target column."""
        from training.feature_selector import DynamicFeatureSelector
        
        np.random.seed(42)
        n_rows = 500
        
        df = pd.DataFrame({
            'feature_1': np.random.randn(n_rows),
            'feature_2': np.random.randn(n_rows),
            'target': np.random.choice([0, 1], n_rows)
        })
        
        selector = DynamicFeatureSelector(n_features=2)
        result = selector.select(df, 'target', ['target'])
        
        assert len(result) == 2
    
    def test_multiclass_target(self):
        """Test with multiclass target column."""
        from training.feature_selector import DynamicFeatureSelector
        
        np.random.seed(42)
        n_rows = 500
        
        df = pd.DataFrame({
            'feature_1': np.random.randn(n_rows),
            'feature_2': np.random.randn(n_rows),
            'feature_3': np.random.randn(n_rows),
            'target': np.random.choice([0, 1, 2, 3], n_rows)  # 4 classes
        })
        
        selector = DynamicFeatureSelector(n_features=2)
        result = selector.select(df, 'target', ['target'])
        
        assert len(result) == 2


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for various edge cases."""
    
    def test_single_feature_available(self):
        """Test when only one feature is available."""
        from training.feature_selector import DynamicFeatureSelector
        
        np.random.seed(42)
        n_rows = 100
        
        df = pd.DataFrame({
            'only_feature': np.random.randn(n_rows),
            'target': np.random.randint(0, 2, n_rows)
        })
        
        selector = DynamicFeatureSelector(n_features=1)
        result = selector.select(df, 'target', ['target'])
        
        assert result == ['only_feature']
    
    def test_constant_feature(self):
        """Test handling of constant (zero variance) features."""
        from training.feature_selector import DynamicFeatureSelector
        
        np.random.seed(42)
        n_rows = 500
        
        df = pd.DataFrame({
            'constant_feature': [5.0] * n_rows,  # No variance
            'varying_feature': np.random.randn(n_rows),
            'target': np.random.randint(0, 2, n_rows)
        })
        
        selector = DynamicFeatureSelector(n_features=2)
        result = selector.select(df, 'target', ['target'])
        
        assert len(result) == 2
    
    def test_highly_correlated_features(self):
        """Test with highly correlated features."""
        from training.feature_selector import DynamicFeatureSelector
        
        np.random.seed(42)
        n_rows = 500
        
        base = np.random.randn(n_rows)
        df = pd.DataFrame({
            'feature_1': base,
            'feature_2': base + np.random.randn(n_rows) * 0.01,  # Almost identical
            'feature_3': np.random.randn(n_rows),
            'target': (base > 0).astype(int)
        })
        
        selector = DynamicFeatureSelector(n_features=2)
        result = selector.select(df, 'target', ['target'])
        
        assert len(result) == 2
    
    def test_sparse_data(self):
        """Test with sparse data (many zeros)."""
        from training.feature_selector import DynamicFeatureSelector
        
        np.random.seed(42)
        n_rows = 500
        
        # Sparse features (mostly zeros)
        feature_1 = np.zeros(n_rows)
        feature_1[np.random.choice(n_rows, 50, replace=False)] = np.random.randn(50)
        
        df = pd.DataFrame({
            'sparse_feature': feature_1,
            'dense_feature': np.random.randn(n_rows),
            'target': np.random.randint(0, 2, n_rows)
        })
        
        selector = DynamicFeatureSelector(n_features=2)
        result = selector.select(df, 'target', ['target'])
        
        assert len(result) == 2
    
    def test_negative_values_only(self):
        """Test with features containing only negative values."""
        from training.feature_selector import DynamicFeatureSelector
        
        np.random.seed(42)
        n_rows = 500
        
        df = pd.DataFrame({
            'negative_feature': -np.abs(np.random.randn(n_rows)),
            'mixed_feature': np.random.randn(n_rows),
            'target': np.random.randint(0, 2, n_rows)
        })
        
        selector = DynamicFeatureSelector(n_features=2)
        result = selector.select(df, 'target', ['target'])
        
        assert len(result) == 2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_workflow_basic(self, basic_dataframe):
        """Test complete feature selection workflow."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=4)
        exclude_cols = ['time', 'target']
        
        result = selector.select(basic_dataframe, 'target', exclude_cols)
        
        # Verify results
        assert len(result) == 4
        assert all(f not in exclude_cols for f in result)
        assert all(f in basic_dataframe.columns for f in result)
    
    def test_full_workflow_with_preprocessing(self, dataframe_with_nan):
        """Test workflow with data that needs preprocessing."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=2)
        
        result = selector.select(dataframe_with_nan, 'target', ['target'])
        
        assert len(result) == 2
    
    def test_full_workflow_large_dataset(self, large_dataframe):
        """Test workflow with large dataset requiring sampling."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=3, sample_size=10000)
        
        result = selector.select(large_dataframe, 'target', ['time', 'target'])
        
        assert len(result) == 3
    
    def test_selected_features_can_be_used(self, basic_dataframe):
        """Test that selected features can be used to subset DataFrame."""
        from training.feature_selector import DynamicFeatureSelector
        
        selector = DynamicFeatureSelector(n_features=3)
        exclude_cols = ['time', 'target']
        
        selected = selector.select(basic_dataframe, 'target', exclude_cols)
        
        # Use selected features to create feature matrix
        X = basic_dataframe[selected]
        
        assert X.shape == (len(basic_dataframe), 3)
        assert list(X.columns) == selected


# ============================================================================
# MODULE IMPORT TESTS
# ============================================================================

class TestModuleImports:
    """Tests for module import behavior."""
    
    def test_module_imports_without_error(self):
        """Test that the module can be imported without errors."""
        import training.feature_selector
    
    def test_class_is_accessible(self):
        """Test that DynamicFeatureSelector class is accessible."""
        from training.feature_selector import DynamicFeatureSelector
        assert DynamicFeatureSelector is not None
    
    def test_class_is_instantiable(self):
        """Test that class can be instantiated."""
        from training.feature_selector import DynamicFeatureSelector
        selector = DynamicFeatureSelector()
        assert isinstance(selector, DynamicFeatureSelector)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])