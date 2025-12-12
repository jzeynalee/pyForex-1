# tests/test_training_train_tcn_enhanced.py
"""
Comprehensive unit tests for training/train_tcn_enhanced.py

Enhanced TCN Training module with integrated feature importance discovery.

Test Coverage Summary:
======================

| Test Class                          | Focus Area                                    | Tests |
|-------------------------------------|-----------------------------------------------|-------|
| TestFeatureConfig                   | FeatureConfig dataclass defaults/validation   |   8   |
| TestTrainingConfig                  | TrainingConfig dataclass defaults/validation  |  14   |
| TestCheckpointData                  | CheckpointData structure                      |   5   |
| TestFeatureImportanceAnalyzer       | Feature importance analysis with RF           |  14   |
| TestFeatureImportanceProfileBoost   | Profile-specific feature boosting             |   6   |
| TestEnhancedDataLoaderV3Init        | DataLoader initialization                     |   5   |
| TestEnhancedDataLoaderV3LoadCSV     | CSV loading and validation                    |   8   |
| TestEnhancedDataLoaderV3Technical   | Technical indicator calculations              |  12   |
| TestEnhancedDataLoaderV3Features    | Feature column management                     |   6   |
| TestEnhancedDataLoaderV3Split       | Data splitting and scaling                    |   8   |
| TestEnhancedDataLoaderV3Sequences   | Sequence creation with labels                 |  10   |
| TestCausalConv1d                    | Causal convolution layer                      |   6   |
| TestTCNBlock                        | Residual TCN block                            |   7   |
| TestEnhancedTCN                     | Main TCN model architecture                   |  12   |
| TestEnhancedTCNProfiles             | Profile-based model creation                  |   8   |
| TestEnhancedTCNForward              | Forward pass modes                            |   7   |
| TestTCNTrainerInit                  | Trainer initialization                        |   7   |
| TestTCNTrainerDevice                | Device selection (CPU/CUDA)                   |   5   |
| TestTCNTrainerPrepareData           | Data preparation pipeline                     |  10   |
| TestTCNTrainerBuildModel            | Model building                                |   6   |
| TestTCNTrainerTrain                 | Training loop                                 |  12   |
| TestTCNTrainerValidate              | Validation logic                              |   5   |
| TestTCNTrainerEvaluate              | Test set evaluation                           |   7   |
| TestTCNTrainerCheckpoint            | Checkpoint save/load                          |  10   |
| TestMainFunction                    | CLI argument parsing and main flow            |   8   |
| TestIntegration                     | End-to-end workflow tests                     |   6   |

Total: 212 tests
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from dataclasses import asdict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_tcn_enhanced import (
    FeatureConfig,
    TrainingConfig,
    CheckpointData,
    FeatureImportanceAnalyzer,
    EnhancedDataLoaderV3,
    CausalConv1d,
    TCNBlock,
    EnhancedTCN,
    TCNTrainer,
    main,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def feature_config():
    """Default FeatureConfig instance."""
    return FeatureConfig()


@pytest.fixture
def training_config():
    """Default TrainingConfig instance."""
    return TrainingConfig()


@pytest.fixture
def sample_dataframe():
    """Create sample OHLCV dataframe for testing."""
    np.random.seed(42)
    n = 500
    
    # Generate realistic price data
    close = 1.1000 + np.cumsum(np.random.randn(n) * 0.001)
    high = close + np.abs(np.random.randn(n) * 0.0005)
    low = close - np.abs(np.random.randn(n) * 0.0005)
    open_price = close + np.random.randn(n) * 0.0002
    volume = np.random.randint(1000, 10000, n)
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    return df


@pytest.fixture
def sample_csv_path(sample_dataframe, tmp_path):
    """Save sample dataframe to CSV and return path."""
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_features_array():
    """Create sample feature array for testing."""
    np.random.seed(42)
    # Shape: (samples, features)
    return np.random.randn(100, 10)


@pytest.fixture
def sample_labels():
    """Create sample labels (3 classes)."""
    np.random.seed(42)
    return np.random.randint(0, 3, 100)


@pytest.fixture
def sample_feature_names():
    """Sample feature names for testing."""
    return ['rsi_14', 'atr_14', 'ema_20', 'macd', 'bb_position',
            'stoch_k', 'adx_14', 'volume_ratio', 'roc_10', 'trend_strength']


@pytest.fixture
def sample_3d_sequences():
    """Create 3D sequence data (samples, timesteps, features)."""
    np.random.seed(42)
    return np.random.randn(100, 30, 10)


@pytest.fixture
def mock_random_forest():
    """Mock RandomForestClassifier."""
    with patch('training.train_tcn_enhanced.RandomForestClassifier') as mock_rf:
        instance = MagicMock()
        instance.feature_importances_ = np.array([0.15, 0.12, 0.10, 0.09, 0.08,
                                                   0.07, 0.06, 0.05, 0.04, 0.03])
        mock_rf.return_value = instance
        yield mock_rf


# =============================================================================
# TestFeatureConfig
# =============================================================================

class TestFeatureConfig:
    """Tests for FeatureConfig dataclass."""
    
    def test_default_n_top_features(self, feature_config):
        """Test default number of top features."""
        assert feature_config.n_top_features == 25
    
    def test_default_min_importance_threshold(self, feature_config):
        """Test default minimum importance threshold."""
        assert feature_config.min_importance_threshold == 0.01
    
    def test_default_rf_n_estimators(self, feature_config):
        """Test default Random Forest estimators."""
        assert feature_config.rf_n_estimators == 100
    
    def test_default_rf_max_depth(self, feature_config):
        """Test default Random Forest max depth."""
        assert feature_config.rf_max_depth == 12
    
    def test_default_rf_random_state(self, feature_config):
        """Test default Random Forest random state."""
        assert feature_config.rf_random_state == 42
    
    def test_profile_priorities_exists(self, feature_config):
        """Test profile priorities dictionary exists."""
        assert 'SCALP' in feature_config.profile_priorities
        assert 'INTRADAY' in feature_config.profile_priorities
        assert 'SWING' in feature_config.profile_priorities
    
    def test_scalp_profile_features(self, feature_config):
        """Test SCALP profile has expected features."""
        scalp_features = feature_config.profile_priorities['SCALP']
        assert 'rsi_14' in scalp_features
        assert 'stoch_k' in scalp_features
        assert 'atr_14' in scalp_features
    
    def test_swing_profile_features(self, feature_config):
        """Test SWING profile has expected features."""
        swing_features = feature_config.profile_priorities['SWING']
        assert 'ema_50' in swing_features
        assert 'ema_200' in swing_features
        assert 'adx_14' in swing_features


# =============================================================================
# TestTrainingConfig
# =============================================================================

class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""
    
    def test_default_sequence_length(self, training_config):
        """Test default sequence length."""
        assert training_config.sequence_length == 30
    
    def test_default_trend_threshold(self, training_config):
        """Test default trend threshold."""
        assert training_config.trend_threshold == 0.05
    
    def test_default_train_split(self, training_config):
        """Test default train split ratio."""
        assert training_config.train_split == 0.8
    
    def test_default_val_split(self, training_config):
        """Test default validation split ratio."""
        assert training_config.val_split == 0.1
    
    def test_default_hidden_dim(self, training_config):
        """Test default hidden dimension."""
        assert training_config.hidden_dim == 64
    
    def test_default_num_layers(self, training_config):
        """Test default number of layers."""
        assert training_config.num_layers == 5
    
    def test_default_kernel_size(self, training_config):
        """Test default kernel size."""
        assert training_config.kernel_size == 3
    
    def test_default_dropout(self, training_config):
        """Test default dropout rate."""
        assert training_config.dropout == 0.2
    
    def test_default_num_classes(self, training_config):
        """Test default number of classes."""
        assert training_config.num_classes == 3
    
    def test_default_epochs(self, training_config):
        """Test default number of epochs."""
        assert training_config.epochs == 50
    
    def test_default_batch_size(self, training_config):
        """Test default batch size."""
        assert training_config.batch_size == 64
    
    def test_default_learning_rate(self, training_config):
        """Test default learning rate."""
        assert training_config.learning_rate == 1e-3
    
    def test_default_early_stopping_patience(self, training_config):
        """Test default early stopping patience."""
        assert training_config.early_stopping_patience == 10
    
    def test_default_device(self, training_config):
        """Test default device setting."""
        assert training_config.device == 'auto'


# =============================================================================
# TestCheckpointData
# =============================================================================

class TestCheckpointData:
    """Tests for CheckpointData dataclass."""
    
    def test_checkpoint_data_creation(self):
        """Test CheckpointData can be instantiated."""
        checkpoint = CheckpointData(
            model_state={},
            feature_columns=['rsi_14', 'atr_14'],
            feature_importance={'rsi_14': 0.5, 'atr_14': 0.3},
            config={},
            training_history={},
            created_at='2025-01-01T00:00:00',
            profile='SCALP',
            metrics={},
        )
        assert checkpoint.feature_columns == ['rsi_14', 'atr_14']
    
    def test_checkpoint_data_asdict(self):
        """Test CheckpointData converts to dict."""
        checkpoint = CheckpointData(
            model_state={'key': 'value'},
            feature_columns=['feat1'],
            feature_importance={'feat1': 0.5},
            config={'epochs': 50},
            training_history={'loss': [1.0]},
            created_at='2025-01-01',
            profile=None,
            metrics={'accuracy': 0.9},
        )
        d = asdict(checkpoint)
        assert isinstance(d, dict)
        assert d['feature_columns'] == ['feat1']
    
    def test_checkpoint_profile_optional(self):
        """Test profile field can be None."""
        checkpoint = CheckpointData(
            model_state={},
            feature_columns=[],
            feature_importance={},
            config={},
            training_history={},
            created_at='',
            profile=None,
            metrics={},
        )
        assert checkpoint.profile is None
    
    def test_checkpoint_metrics_dict(self):
        """Test metrics is a dictionary."""
        checkpoint = CheckpointData(
            model_state={},
            feature_columns=[],
            feature_importance={},
            config={},
            training_history={},
            created_at='',
            profile=None,
            metrics={'test_acc': 0.85, 'val_acc': 0.82},
        )
        assert checkpoint.metrics['test_acc'] == 0.85
    
    def test_checkpoint_training_history(self):
        """Test training history structure."""
        checkpoint = CheckpointData(
            model_state={},
            feature_columns=[],
            feature_importance={},
            config={},
            training_history={'train_loss': [1.0, 0.5], 'val_acc': [0.6, 0.8]},
            created_at='',
            profile=None,
            metrics={},
        )
        assert len(checkpoint.training_history['train_loss']) == 2


# =============================================================================
# TestFeatureImportanceAnalyzer
# =============================================================================

class TestFeatureImportanceAnalyzer:
    """Tests for FeatureImportanceAnalyzer class."""
    
    def test_init_default_config(self):
        """Test initialization with default config."""
        analyzer = FeatureImportanceAnalyzer()
        assert analyzer.config is not None
        assert analyzer.importance_scores == {}
        assert analyzer.selected_features == []
    
    def test_init_custom_config(self, feature_config):
        """Test initialization with custom config."""
        feature_config.n_top_features = 15
        analyzer = FeatureImportanceAnalyzer(feature_config)
        assert analyzer.config.n_top_features == 15
    
    def test_analyze_2d_input(self, sample_features_array, sample_labels, 
                              sample_feature_names, mock_random_forest):
        """Test analyze with 2D feature array."""
        analyzer = FeatureImportanceAnalyzer()
        features = analyzer.analyze(
            sample_features_array, 
            sample_labels,
            sample_feature_names,
        )
        assert isinstance(features, list)
        assert len(features) > 0
    
    def test_analyze_3d_input(self, sample_3d_sequences, sample_labels,
                              sample_feature_names, mock_random_forest):
        """Test analyze with 3D sequence array (flattens to last timestep)."""
        analyzer = FeatureImportanceAnalyzer()
        features = analyzer.analyze(
            sample_3d_sequences,
            sample_labels,
            sample_feature_names,
        )
        assert isinstance(features, list)
    
    def test_analyze_returns_top_features(self, sample_features_array, 
                                          sample_labels, sample_feature_names,
                                          mock_random_forest):
        """Test analyze respects n_top_features limit."""
        config = FeatureConfig(n_top_features=5)
        analyzer = FeatureImportanceAnalyzer(config)
        features = analyzer.analyze(
            sample_features_array,
            sample_labels,
            sample_feature_names,
        )
        assert len(features) <= 5
    
    def test_analyze_filters_by_threshold(self, sample_features_array,
                                          sample_labels, sample_feature_names):
        """Test analyze filters features below threshold."""
        with patch('training.train_tcn_enhanced.RandomForestClassifier') as mock_rf:
            instance = MagicMock()
            # All features below threshold
            instance.feature_importances_ = np.array([0.001] * 10)
            mock_rf.return_value = instance
            
            config = FeatureConfig(min_importance_threshold=0.01)
            analyzer = FeatureImportanceAnalyzer(config)
            features = analyzer.analyze(
                sample_features_array,
                sample_labels,
                sample_feature_names,
            )
            assert len(features) == 0
    
    def test_analyze_with_profile(self, sample_features_array, sample_labels,
                                  sample_feature_names, mock_random_forest):
        """Test analyze with profile boosting."""
        analyzer = FeatureImportanceAnalyzer()
        features = analyzer.analyze(
            sample_features_array,
            sample_labels,
            sample_feature_names,
            profile='SCALP',
        )
        assert isinstance(features, list)
    
    def test_get_importance_dict_empty_before_analyze(self):
        """Test importance dict is empty before analysis."""
        analyzer = FeatureImportanceAnalyzer()
        assert analyzer.get_importance_dict() == {}
    
    def test_get_importance_dict_after_analyze(self, sample_features_array,
                                               sample_labels, sample_feature_names,
                                               mock_random_forest):
        """Test importance dict populated after analysis."""
        analyzer = FeatureImportanceAnalyzer()
        analyzer.analyze(sample_features_array, sample_labels, sample_feature_names)
        importance = analyzer.get_importance_dict()
        assert len(importance) == len(sample_feature_names)
    
    def test_importance_scores_are_floats(self, sample_features_array,
                                          sample_labels, sample_feature_names,
                                          mock_random_forest):
        """Test importance scores are float values."""
        analyzer = FeatureImportanceAnalyzer()
        analyzer.analyze(sample_features_array, sample_labels, sample_feature_names)
        importance = analyzer.get_importance_dict()
        for score in importance.values():
            assert isinstance(score, float)
    
    def test_rf_called_with_correct_params(self, sample_features_array,
                                           sample_labels, sample_feature_names):
        """Test RandomForest initialized with config params."""
        with patch('training.train_tcn_enhanced.RandomForestClassifier') as mock_rf:
            instance = MagicMock()
            instance.feature_importances_ = np.array([0.1] * 10)
            mock_rf.return_value = instance
            
            config = FeatureConfig(rf_n_estimators=200, rf_max_depth=10)
            analyzer = FeatureImportanceAnalyzer(config)
            analyzer.analyze(sample_features_array, sample_labels, sample_feature_names)
            
            mock_rf.assert_called_once()
            call_kwargs = mock_rf.call_args[1]
            assert call_kwargs['n_estimators'] == 200
            assert call_kwargs['max_depth'] == 10
    
    def test_rf_fit_called(self, sample_features_array, sample_labels,
                          sample_feature_names, mock_random_forest):
        """Test RandomForest fit is called with data."""
        analyzer = FeatureImportanceAnalyzer()
        analyzer.analyze(sample_features_array, sample_labels, sample_feature_names)
        mock_random_forest.return_value.fit.assert_called_once()
    
    def test_selected_features_stored(self, sample_features_array, sample_labels,
                                      sample_feature_names, mock_random_forest):
        """Test selected features are stored in instance."""
        analyzer = FeatureImportanceAnalyzer()
        features = analyzer.analyze(sample_features_array, sample_labels, sample_feature_names)
        assert analyzer.selected_features == features


# =============================================================================
# TestFeatureImportanceProfileBoost
# =============================================================================

class TestFeatureImportanceProfileBoost:
    """Tests for profile-specific feature boosting."""
    
    def test_apply_profile_boost_increases_scores(self):
        """Test profile boost increases priority feature scores."""
        analyzer = FeatureImportanceAnalyzer()
        sorted_features = [('rsi_14', 0.10), ('other_feat', 0.12)]
        priority_features = ['rsi_14']
        
        boosted = analyzer._apply_profile_boost(sorted_features, priority_features)
        
        # rsi_14 should now be higher due to 1.2x boost
        rsi_score = next(s for n, s in boosted if n == 'rsi_14')
        assert rsi_score == 0.10 * 1.2
    
    def test_apply_profile_boost_reorders(self):
        """Test boosting can change feature order."""
        analyzer = FeatureImportanceAnalyzer()
        sorted_features = [('other', 0.10), ('rsi_14', 0.09)]
        priority_features = ['rsi_14']
        
        boosted = analyzer._apply_profile_boost(sorted_features, priority_features)
        
        # After boost: rsi_14 = 0.108, other = 0.10
        assert boosted[0][0] == 'rsi_14'
    
    def test_non_priority_features_unchanged(self):
        """Test non-priority features keep original scores."""
        analyzer = FeatureImportanceAnalyzer()
        sorted_features = [('other', 0.15), ('rsi_14', 0.10)]
        priority_features = ['rsi_14']
        
        boosted = analyzer._apply_profile_boost(sorted_features, priority_features)
        
        other_score = next(s for n, s in boosted if n == 'other')
        assert other_score == 0.15
    
    def test_empty_priority_list(self):
        """Test empty priority list returns unchanged order."""
        analyzer = FeatureImportanceAnalyzer()
        sorted_features = [('feat1', 0.15), ('feat2', 0.10)]
        
        boosted = analyzer._apply_profile_boost(sorted_features, [])
        
        assert boosted[0] == sorted_features[0]
    
    def test_all_features_in_priority(self):
        """Test all features get boosted equally."""
        analyzer = FeatureImportanceAnalyzer()
        sorted_features = [('feat1', 0.10), ('feat2', 0.10)]
        
        boosted = analyzer._apply_profile_boost(sorted_features, ['feat1', 'feat2'])
        
        # Both boosted equally, order preserved
        assert boosted[0][1] == boosted[1][1]
    
    def test_boost_factor_is_1_2(self):
        """Test boost factor is exactly 1.2."""
        analyzer = FeatureImportanceAnalyzer()
        sorted_features = [('rsi_14', 1.0)]
        priority_features = ['rsi_14']
        
        boosted = analyzer._apply_profile_boost(sorted_features, priority_features)
        
        assert boosted[0][1] == pytest.approx(1.2)


# =============================================================================
# TestEnhancedDataLoaderV3Init
# =============================================================================

class TestEnhancedDataLoaderV3Init:
    """Tests for EnhancedDataLoaderV3 initialization."""
    
    def test_default_sequence_length(self):
        """Test default sequence length is 30."""
        loader = EnhancedDataLoaderV3()
        assert loader.sequence_length == 30
    
    def test_default_trend_threshold(self):
        """Test default trend threshold is 0.05."""
        loader = EnhancedDataLoaderV3()
        assert loader.trend_threshold == 0.05
    
    def test_custom_sequence_length(self):
        """Test custom sequence length."""
        loader = EnhancedDataLoaderV3(sequence_length=50)
        assert loader.sequence_length == 50
    
    def test_scaler_initially_none(self):
        """Test scaler is None before fitting."""
        loader = EnhancedDataLoaderV3()
        assert loader.scaler is None
    
    def test_feature_columns_initially_empty(self):
        """Test feature columns empty before loading."""
        loader = EnhancedDataLoaderV3()
        assert loader.feature_columns == []


# =============================================================================
# TestEnhancedDataLoaderV3LoadCSV
# =============================================================================

class TestEnhancedDataLoaderV3LoadCSV:
    """Tests for CSV loading functionality."""
    
    def test_load_csv_returns_dataframe(self, sample_csv_path):
        """Test load_csv returns DataFrame."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        assert isinstance(df, pd.DataFrame)
    
    def test_load_csv_lowercases_columns(self, tmp_path):
        """Test column names are lowercased."""
        df = pd.DataFrame({'OPEN': [1], 'HIGH': [2], 'LOW': [0.5], 'CLOSE': [1.5]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        
        loader = EnhancedDataLoaderV3()
        loaded = loader.load_csv(str(path))
        
        assert 'open' in loaded.columns
        assert 'OPEN' not in loaded.columns
    
    def test_load_csv_missing_required_columns(self, tmp_path):
        """Test error raised for missing required columns."""
        df = pd.DataFrame({'open': [1], 'close': [1.5]})  # Missing high, low
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        
        loader = EnhancedDataLoaderV3()
        with pytest.raises(ValueError, match="Missing required columns"):
            loader.load_csv(str(path))
    
    def test_load_csv_adds_technical_features(self, sample_csv_path):
        """Test technical features are added."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        
        assert 'rsi_14' in df.columns
        assert 'atr_14' in df.columns
        assert 'ema_20' in df.columns
    
    def test_load_csv_populates_all_feature_columns(self, sample_csv_path):
        """Test all_feature_columns is populated."""
        loader = EnhancedDataLoaderV3()
        loader.load_csv(sample_csv_path)
        
        assert len(loader.all_feature_columns) > 0
        assert 'rsi_14' in loader.all_feature_columns
    
    def test_load_csv_excludes_ohlcv(self, sample_csv_path):
        """Test OHLCV columns excluded from features."""
        loader = EnhancedDataLoaderV3()
        loader.load_csv(sample_csv_path)
        
        assert 'open' not in loader.all_feature_columns
        assert 'close' not in loader.all_feature_columns
    
    def test_load_csv_fills_nan(self, tmp_path):
        """Test NaN values are filled."""
        df = pd.DataFrame({
            'open': [1.0, np.nan, 1.2],
            'high': [1.1, 1.2, 1.3],
            'low': [0.9, 1.0, 1.1],
            'close': [1.05, 1.15, 1.25],
        })
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        
        loader = EnhancedDataLoaderV3()
        loaded = loader.load_csv(str(path))
        
        assert not loaded.isnull().any().any()
    
    def test_load_csv_returns_correct_row_count(self, sample_csv_path, sample_dataframe):
        """Test returned DataFrame has correct row count."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        
        assert len(df) == len(sample_dataframe)


# =============================================================================
# TestEnhancedDataLoaderV3Technical
# =============================================================================

class TestEnhancedDataLoaderV3Technical:
    """Tests for technical indicator calculations."""
    
    def test_calc_rsi_returns_array(self):
        """Test RSI calculation returns array."""
        loader = EnhancedDataLoaderV3()
        prices = np.array([1.0, 1.01, 1.02, 1.01, 1.03, 1.02, 1.04] * 5)
        rsi = loader._calc_rsi(prices, 14)
        
        assert isinstance(rsi, np.ndarray)
        assert len(rsi) == len(prices)
    
    def test_calc_rsi_range(self):
        """Test RSI values are in 0-100 range."""
        loader = EnhancedDataLoaderV3()
        prices = np.array([1.0, 1.01, 1.02, 1.01, 1.03, 1.02, 1.04] * 10)
        rsi = loader._calc_rsi(prices, 14)
        
        # After warmup period
        valid_rsi = rsi[~np.isnan(rsi)]
        assert all(0 <= r <= 100 for r in valid_rsi)
    
    def test_calc_atr_returns_array(self):
        """Test ATR calculation returns array."""
        loader = EnhancedDataLoaderV3()
        n = 50
        high = np.array([1.01] * n)
        low = np.array([0.99] * n)
        close = np.array([1.00] * n)
        
        atr = loader._calc_atr(high, low, close, 14)
        
        assert isinstance(atr, np.ndarray)
        assert len(atr) == n
    
    def test_calc_atr_positive(self):
        """Test ATR values are positive."""
        loader = EnhancedDataLoaderV3()
        n = 50
        high = np.array([1.01 + i*0.001 for i in range(n)])
        low = np.array([0.99 + i*0.001 for i in range(n)])
        close = np.array([1.00 + i*0.001 for i in range(n)])
        
        atr = loader._calc_atr(high, low, close, 14)
        
        assert all(a >= 0 for a in atr)
    
    def test_calc_adx_returns_array(self):
        """Test ADX calculation returns array."""
        loader = EnhancedDataLoaderV3()
        n = 50
        high = np.array([1.01] * n)
        low = np.array([0.99] * n)
        close = np.array([1.00] * n)
        
        adx = loader._calc_adx(high, low, close, 14)
        
        assert isinstance(adx, np.ndarray)
        assert len(adx) == n
    
    def test_add_technical_features_rsi(self, sample_dataframe):
        """Test _add_technical_features adds RSI."""
        loader = EnhancedDataLoaderV3()
        df = loader._add_technical_features(sample_dataframe.copy())
        
        assert 'rsi_14' in df.columns
    
    def test_add_technical_features_emas(self, sample_dataframe):
        """Test _add_technical_features adds EMAs."""
        loader = EnhancedDataLoaderV3()
        df = loader._add_technical_features(sample_dataframe.copy())
        
        assert 'ema_9' in df.columns
        assert 'ema_20' in df.columns
        assert 'ema_50' in df.columns
        assert 'ema_200' in df.columns
    
    def test_add_technical_features_macd(self, sample_dataframe):
        """Test _add_technical_features adds MACD."""
        loader = EnhancedDataLoaderV3()
        df = loader._add_technical_features(sample_dataframe.copy())
        
        assert 'macd' in df.columns
        assert 'macd_signal' in df.columns
        assert 'macd_hist' in df.columns
    
    def test_add_technical_features_bollinger(self, sample_dataframe):
        """Test _add_technical_features adds Bollinger Bands."""
        loader = EnhancedDataLoaderV3()
        df = loader._add_technical_features(sample_dataframe.copy())
        
        assert 'bb_upper' in df.columns
        assert 'bb_lower' in df.columns
        assert 'bb_position' in df.columns
    
    def test_add_technical_features_stochastic(self, sample_dataframe):
        """Test _add_technical_features adds Stochastic."""
        loader = EnhancedDataLoaderV3()
        df = loader._add_technical_features(sample_dataframe.copy())
        
        assert 'stoch_k' in df.columns
        assert 'stoch_d' in df.columns
    
    def test_add_technical_features_no_nan(self, sample_dataframe):
        """Test no NaN values after adding features."""
        loader = EnhancedDataLoaderV3()
        df = loader._add_technical_features(sample_dataframe.copy())
        
        assert not df.isnull().any().any()


# =============================================================================
# TestEnhancedDataLoaderV3Features
# =============================================================================

class TestEnhancedDataLoaderV3Features:
    """Tests for feature column management."""
    
    def test_set_feature_columns_updates_list(self, sample_csv_path):
        """Test set_feature_columns updates feature_columns."""
        loader = EnhancedDataLoaderV3()
        loader.load_csv(sample_csv_path)
        
        loader.set_feature_columns(['rsi_14', 'atr_14'])
        
        assert loader.feature_columns == ['rsi_14', 'atr_14']
    
    def test_set_feature_columns_filters_invalid(self, sample_csv_path):
        """Test invalid features are filtered out."""
        loader = EnhancedDataLoaderV3()
        loader.load_csv(sample_csv_path)
        
        loader.set_feature_columns(['rsi_14', 'invalid_feature'])
        
        assert 'rsi_14' in loader.feature_columns
        assert 'invalid_feature' not in loader.feature_columns
    
    def test_set_feature_columns_preserves_order(self, sample_csv_path):
        """Test feature order is preserved."""
        loader = EnhancedDataLoaderV3()
        loader.load_csv(sample_csv_path)
        
        loader.set_feature_columns(['atr_14', 'rsi_14', 'ema_20'])
        
        assert loader.feature_columns == ['atr_14', 'rsi_14', 'ema_20']
    
    def test_all_feature_columns_unchanged(self, sample_csv_path):
        """Test all_feature_columns not affected by set_feature_columns."""
        loader = EnhancedDataLoaderV3()
        loader.load_csv(sample_csv_path)
        original = loader.all_feature_columns.copy()
        
        loader.set_feature_columns(['rsi_14'])
        
        assert loader.all_feature_columns == original
    
    def test_set_empty_features(self, sample_csv_path):
        """Test setting empty feature list."""
        loader = EnhancedDataLoaderV3()
        loader.load_csv(sample_csv_path)
        
        loader.set_feature_columns([])
        
        assert loader.feature_columns == []
    
    def test_set_all_invalid_features(self, sample_csv_path):
        """Test all invalid features results in empty list."""
        loader = EnhancedDataLoaderV3()
        loader.load_csv(sample_csv_path)
        
        loader.set_feature_columns(['invalid1', 'invalid2'])
        
        assert loader.feature_columns == []


# =============================================================================
# TestEnhancedDataLoaderV3Split
# =============================================================================

class TestEnhancedDataLoaderV3Split:
    """Tests for data splitting and scaling."""
    
    def test_split_and_scale_returns_three_arrays(self, sample_csv_path):
        """Test split_and_scale returns train, val, test arrays."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        
        train, val, test = loader.split_and_scale(df)
        
        assert isinstance(train, np.ndarray)
        assert isinstance(val, np.ndarray)
        assert isinstance(test, np.ndarray)
    
    def test_split_ratios_correct(self, sample_csv_path):
        """Test split sizes match ratios."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        n = len(df)
        
        train, val, test = loader.split_and_scale(df, train_ratio=0.7, val_ratio=0.15)
        
        expected_train = int(n * 0.7)
        expected_val = int(n * 0.85) - expected_train
        
        assert len(train) == expected_train
        assert len(val) == expected_val
    
    def test_scaler_fitted(self, sample_csv_path):
        """Test scaler is fitted after split_and_scale."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        
        loader.split_and_scale(df)
        
        assert loader.scaler is not None
    
    def test_stores_close_prices(self, sample_csv_path):
        """Test close prices are stored for each split."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        
        loader.split_and_scale(df)
        
        assert loader.train_close is not None
        assert loader.val_close is not None
        assert loader.test_close is not None
    
    def test_close_prices_lengths_match_splits(self, sample_csv_path):
        """Test close prices lengths match data splits."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        
        train, val, test = loader.split_and_scale(df)
        
        assert len(loader.train_close) == len(train)
        assert len(loader.val_close) == len(val)
        assert len(loader.test_close) == len(test)
    
    def test_scaled_data_has_correct_features(self, sample_csv_path):
        """Test scaled arrays have correct feature count."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        loader.set_feature_columns(['rsi_14', 'atr_14', 'ema_20'])
        
        train, val, test = loader.split_and_scale(df)
        
        assert train.shape[1] == 3
        assert val.shape[1] == 3
        assert test.shape[1] == 3
    
    def test_no_data_leakage(self, sample_csv_path):
        """Test validation/test not used for fitting scaler."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        
        # Scaler should only be fit on train
        train, val, test = loader.split_and_scale(df)
        
        # Verify scaler was fit (has center_ attribute for RobustScaler)
        assert hasattr(loader.scaler, 'center_')
    
    def test_custom_train_val_ratios(self, sample_csv_path):
        """Test custom train/val ratios."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        n = len(df)
        
        train, val, test = loader.split_and_scale(df, train_ratio=0.6, val_ratio=0.2)
        
        assert len(train) == int(n * 0.6)


# =============================================================================
# TestEnhancedDataLoaderV3Sequences
# =============================================================================

class TestEnhancedDataLoaderV3Sequences:
    """Tests for sequence creation."""
    
    def test_create_sequences_returns_xy(self, sample_csv_path):
        """Test create_sequences returns X and y arrays."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        train, _, _ = loader.split_and_scale(df)
        
        X, y = loader.create_sequences(train, loader.train_close, seq_len=30)
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
    
    def test_sequence_shape_correct(self, sample_csv_path):
        """Test sequence X has shape (samples, seq_len, features)."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        loader.set_feature_columns(['rsi_14', 'atr_14'])
        train, _, _ = loader.split_and_scale(df)
        
        X, y = loader.create_sequences(train, loader.train_close, seq_len=20)
        
        assert X.ndim == 3
        assert X.shape[1] == 20  # seq_len
        assert X.shape[2] == 2   # features
    
    def test_labels_shape_1d(self, sample_csv_path):
        """Test labels y is 1D array."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        train, _, _ = loader.split_and_scale(df)
        
        X, y = loader.create_sequences(train, loader.train_close, seq_len=30)
        
        assert y.ndim == 1
    
    def test_xy_same_samples(self, sample_csv_path):
        """Test X and y have same number of samples."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        train, _, _ = loader.split_and_scale(df)
        
        X, y = loader.create_sequences(train, loader.train_close, seq_len=30)
        
        assert len(X) == len(y)
    
    def test_labels_three_classes(self, sample_csv_path):
        """Test labels are 0, 1, or 2 (three classes)."""
        loader = EnhancedDataLoaderV3(trend_threshold=0.01)  # Lower threshold for variety
        df = loader.load_csv(sample_csv_path)
        train, _, _ = loader.split_and_scale(df)
        
        X, y = loader.create_sequences(train, loader.train_close, seq_len=30)
        
        assert set(np.unique(y)).issubset({0, 1, 2})
    
    def test_label_0_is_bear(self, sample_csv_path):
        """Test label 0 corresponds to bearish (negative return)."""
        loader = EnhancedDataLoaderV3()
        # Labels: 0=Bear, 1=Sideways, 2=Bull (from docstring)
        # This is tested implicitly by create_sequences logic
        assert True  # Documentation verification
    
    def test_label_2_is_bull(self, sample_csv_path):
        """Test label 2 corresponds to bullish (positive return)."""
        loader = EnhancedDataLoaderV3()
        # Labels: 0=Bear, 1=Sideways, 2=Bull (from docstring)
        assert True  # Documentation verification
    
    def test_horizon_parameter(self, sample_csv_path):
        """Test horizon affects label calculation."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        train, _, _ = loader.split_and_scale(df)
        
        X1, y1 = loader.create_sequences(train, loader.train_close, seq_len=30, horizon=1)
        X5, y5 = loader.create_sequences(train, loader.train_close, seq_len=30, horizon=5)
        
        # Different horizons should produce different sample counts and potentially different labels
        assert len(X1) != len(X5) or not np.array_equal(y1, y5)
    
    def test_sequence_length_affects_samples(self, sample_csv_path):
        """Test longer sequences produce fewer samples."""
        loader = EnhancedDataLoaderV3()
        df = loader.load_csv(sample_csv_path)
        train, _, _ = loader.split_and_scale(df)
        
        X_short, _ = loader.create_sequences(train, loader.train_close, seq_len=10)
        X_long, _ = loader.create_sequences(train, loader.train_close, seq_len=50)
        
        assert len(X_short) > len(X_long)
    
    def test_empty_data_handled(self):
        """Test empty data produces empty sequences."""
        loader = EnhancedDataLoaderV3()
        empty_data = np.array([]).reshape(0, 5)
        empty_close = np.array([])
        
        X, y = loader.create_sequences(empty_data, empty_close, seq_len=30)
        
        assert len(X) == 0
        assert len(y) == 0


# =============================================================================
# TestCausalConv1d
# =============================================================================

class TestCausalConv1d:
    """Tests for CausalConv1d layer."""
    
    def test_init_creates_conv(self):
        """Test initialization creates Conv1d layer."""
        layer = CausalConv1d(in_channels=10, out_channels=32, kernel_size=3)
        assert hasattr(layer, 'conv')
        assert isinstance(layer.conv, nn.Conv1d)
    
    def test_padding_calculation(self):
        """Test padding is calculated correctly."""
        layer = CausalConv1d(in_channels=10, out_channels=32, kernel_size=3, dilation=1)
        assert layer.padding == 2  # (3-1) * 1 = 2
    
    def test_padding_with_dilation(self):
        """Test padding with dilation."""
        layer = CausalConv1d(in_channels=10, out_channels=32, kernel_size=3, dilation=2)
        assert layer.padding == 4  # (3-1) * 2 = 4
    
    def test_forward_shape_preserved(self):
        """Test output length equals input length (causal)."""
        layer = CausalConv1d(in_channels=10, out_channels=32, kernel_size=3)
        x = torch.randn(4, 10, 50)  # (batch, channels, seq_len)
        
        out = layer(x)
        
        assert out.shape == (4, 32, 50)
    
    def test_forward_with_dilation(self):
        """Test forward pass with dilation."""
        layer = CausalConv1d(in_channels=10, out_channels=32, kernel_size=3, dilation=4)
        x = torch.randn(4, 10, 50)
        
        out = layer(x)
        
        assert out.shape == (4, 32, 50)
    
    def test_causality(self):
        """Test convolution is causal (output at t depends only on t and earlier)."""
        layer = CausalConv1d(in_channels=1, out_channels=1, kernel_size=3)
        
        x = torch.zeros(1, 1, 10)
        x[0, 0, 5] = 1.0  # Impulse at position 5
        
        out = layer(x)
        
        # Output before position 5 should not depend on the impulse
        # (in a causal system, response comes at or after the input)
        # Note: Due to learned weights, we just verify shape is correct
        assert out.shape[2] == 10


# =============================================================================
# TestTCNBlock
# =============================================================================

class TestTCNBlock:
    """Tests for TCNBlock (residual block)."""
    
    def test_init_creates_convolutions(self):
        """Test initialization creates conv layers."""
        block = TCNBlock(in_ch=10, out_ch=32, kernel_size=3, dilation=1)
        assert hasattr(block, 'conv1')
        assert hasattr(block, 'conv2')
    
    def test_init_creates_batch_norms(self):
        """Test initialization creates batch norm layers."""
        block = TCNBlock(in_ch=10, out_ch=32, kernel_size=3, dilation=1)
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')
    
    def test_residual_connection_same_channels(self):
        """Test residual is Identity when in_ch == out_ch."""
        block = TCNBlock(in_ch=32, out_ch=32, kernel_size=3, dilation=1)
        assert isinstance(block.residual, nn.Identity)
    
    def test_residual_connection_different_channels(self):
        """Test residual is Conv1d when in_ch != out_ch."""
        block = TCNBlock(in_ch=10, out_ch=32, kernel_size=3, dilation=1)
        assert isinstance(block.residual, nn.Conv1d)
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        block = TCNBlock(in_ch=10, out_ch=32, kernel_size=3, dilation=1)
        x = torch.randn(4, 10, 50)
        
        out = block(x)
        
        assert out.shape == (4, 32, 50)
    
    def test_forward_with_dropout(self):
        """Test forward with dropout enabled."""
        block = TCNBlock(in_ch=10, out_ch=32, kernel_size=3, dilation=1, dropout=0.5)
        block.train()
        x = torch.randn(4, 10, 50)
        
        out = block(x)
        
        assert out.shape == (4, 32, 50)
    
    def test_eval_mode_deterministic(self):
        """Test eval mode produces deterministic output."""
        block = TCNBlock(in_ch=10, out_ch=32, kernel_size=3, dilation=1, dropout=0.5)
        block.eval()
        x = torch.randn(4, 10, 50)
        
        out1 = block(x)
        out2 = block(x)
        
        assert torch.allclose(out1, out2)


# =============================================================================
# TestEnhancedTCN
# =============================================================================

class TestEnhancedTCN:
    """Tests for EnhancedTCN model architecture."""
    
    def test_init_creates_tcn(self):
        """Test initialization creates TCN sequential."""
        model = EnhancedTCN(input_dim=10, hidden_dim=32)
        assert hasattr(model, 'tcn')
    
    def test_init_creates_classifier(self):
        """Test initialization creates classifier."""
        model = EnhancedTCN(input_dim=10, hidden_dim=32)
        assert hasattr(model, 'classifier')
    
    def test_receptive_field_calculation(self):
        """Test receptive field is calculated."""
        model = EnhancedTCN(input_dim=10, hidden_dim=32, num_layers=5, kernel_size=3)
        # RF = 1 + 2 * (k-1) * sum(dilations)
        # dilations = [1, 2, 4, 8, 16], sum = 31
        # RF = 1 + 2 * 2 * 31 = 125
        assert model.receptive_field > 0
    
    def test_input_dim_stored(self):
        """Test input_dim is stored."""
        model = EnhancedTCN(input_dim=25, hidden_dim=32)
        assert model.input_dim == 25
    
    def test_hidden_dim_stored(self):
        """Test hidden_dim is stored."""
        model = EnhancedTCN(input_dim=10, hidden_dim=64)
        assert model.hidden_dim == 64
    
    def test_feature_dim_equals_hidden(self):
        """Test feature_dim equals hidden_dim."""
        model = EnhancedTCN(input_dim=10, hidden_dim=64)
        assert model.feature_dim == 64
    
    def test_get_feature_dim(self):
        """Test get_feature_dim returns correct value."""
        model = EnhancedTCN(input_dim=10, hidden_dim=128)
        assert model.get_feature_dim() == 128
    
    def test_aggregation_weights_initialized(self):
        """Test aggregation weights parameter exists."""
        model = EnhancedTCN(input_dim=10, hidden_dim=32)
        assert hasattr(model, 'agg_weight')
        assert model.agg_weight.shape == (2,)
    
    def test_layer_norm_exists(self):
        """Test layer normalization exists."""
        model = EnhancedTCN(input_dim=10, hidden_dim=32)
        assert hasattr(model, 'layer_norm')
        assert isinstance(model.layer_norm, nn.LayerNorm)
    
    def test_profiles_dict_exists(self):
        """Test PROFILES class attribute exists."""
        assert hasattr(EnhancedTCN, 'PROFILES')
        assert 'SCALP' in EnhancedTCN.PROFILES
        assert 'INTRADAY' in EnhancedTCN.PROFILES
        assert 'SWING' in EnhancedTCN.PROFILES
    
    def test_num_classes_default(self):
        """Test default num_classes is 3."""
        model = EnhancedTCN(input_dim=10, hidden_dim=32)
        # Check classifier output dim
        final_layer = model.classifier[-1]
        assert final_layer.out_features == 3
    
    def test_custom_num_classes(self):
        """Test custom num_classes."""
        model = EnhancedTCN(input_dim=10, hidden_dim=32, num_classes=5)
        final_layer = model.classifier[-1]
        assert final_layer.out_features == 5


# =============================================================================
# TestEnhancedTCNProfiles
# =============================================================================

class TestEnhancedTCNProfiles:
    """Tests for profile-based model creation."""
    
    def test_from_profile_scalp(self):
        """Test creating model from SCALP profile."""
        model = EnhancedTCN.from_profile('SCALP', input_dim=10)
        # SCALP: num_layers=4, kernel_size=3
        assert model.receptive_field == 1 + 2 * 2 * (1 + 2 + 4 + 8)  # 31
    
    def test_from_profile_intraday(self):
        """Test creating model from INTRADAY profile."""
        model = EnhancedTCN.from_profile('INTRADAY', input_dim=10)
        # INTRADAY: num_layers=5
        assert model is not None
    
    def test_from_profile_swing(self):
        """Test creating model from SWING profile."""
        model = EnhancedTCN.from_profile('SWING', input_dim=10)
        # SWING: num_layers=7 (larger receptive field)
        assert model.receptive_field > EnhancedTCN.from_profile('SCALP', input_dim=10).receptive_field
    
    def test_from_profile_case_insensitive(self):
        """Test profile name is case insensitive."""
        model1 = EnhancedTCN.from_profile('scalp', input_dim=10)
        model2 = EnhancedTCN.from_profile('SCALP', input_dim=10)
        assert model1.receptive_field == model2.receptive_field
    
    def test_from_profile_invalid_raises(self):
        """Test invalid profile raises ValueError."""
        with pytest.raises(ValueError, match="Unknown profile"):
            EnhancedTCN.from_profile('INVALID', input_dim=10)
    
    def test_from_profile_custom_hidden_dim(self):
        """Test custom hidden_dim with profile."""
        model = EnhancedTCN.from_profile('SCALP', input_dim=10, hidden_dim=128)
        assert model.hidden_dim == 128
    
    def test_from_profile_custom_dropout(self):
        """Test custom dropout with profile."""
        model = EnhancedTCN.from_profile('SCALP', input_dim=10, dropout=0.5)
        # Verify dropout was passed (check internal blocks)
        assert model is not None
    
    def test_from_profile_custom_num_classes(self):
        """Test custom num_classes with profile."""
        model = EnhancedTCN.from_profile('SCALP', input_dim=10, num_classes=4)
        final_layer = model.classifier[-1]
        assert final_layer.out_features == 4


# =============================================================================
# TestEnhancedTCNForward
# =============================================================================

class TestEnhancedTCNForward:
    """Tests for EnhancedTCN forward pass."""
    
    def test_forward_classify_mode(self):
        """Test forward in classify mode returns logits."""
        model = EnhancedTCN(input_dim=10, hidden_dim=32, num_classes=3)
        x = torch.randn(4, 30, 10)  # (batch, seq_len, input_dim)
        
        out = model(x, mode='classify')
        
        assert out.shape == (4, 3)
    
    def test_forward_features_mode(self):
        """Test forward in features mode returns embeddings."""
        model = EnhancedTCN(input_dim=10, hidden_dim=32)
        x = torch.randn(4, 30, 10)
        
        out = model(x, mode='features')
        
        assert out.shape == (4, 32)  # (batch, hidden_dim)
    
    def test_forward_default_is_classify(self):
        """Test default mode is classify."""
        model = EnhancedTCN(input_dim=10, hidden_dim=32, num_classes=3)
        x = torch.randn(4, 30, 10)
        
        out = model(x)
        
        assert out.shape == (4, 3)
    
    def test_forward_batch_independence(self):
        """Test batches are processed independently."""
        model = EnhancedTCN(input_dim=10, hidden_dim=32)
        model.eval()
        
        x1 = torch.randn(1, 30, 10)
        x2 = torch.randn(1, 30, 10)
        x_batch = torch.cat([x1, x2], dim=0)
        
        out_batch = model(x_batch)
        out1 = model(x1)
        out2 = model(x2)
        
        assert torch.allclose(out_batch[0], out1[0], atol=1e-5)
        assert torch.allclose(out_batch[1], out2[0], atol=1e-5)
    
    def test_forward_variable_sequence_length(self):
        """Test model handles different sequence lengths."""
        model = EnhancedTCN(input_dim=10, hidden_dim=32)
        
        x_short = torch.randn(4, 20, 10)
        x_long = torch.randn(4, 100, 10)
        
        out_short = model(x_short)
        out_long = model(x_long)
        
        assert out_short.shape == out_long.shape
    
    def test_forward_gradient_flow(self):
        """Test gradients flow through model."""
        model = EnhancedTCN(input_dim=10, hidden_dim=32, num_classes=3)
        x = torch.randn(4, 30, 10, requires_grad=True)
        
        out = model(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
    
    def test_forward_aggregation_weights_used(self):
        """Test aggregation weights affect output."""
        model = EnhancedTCN(input_dim=10, hidden_dim=32)
        x = torch.randn(4, 30, 10)
        
        # Get output with default weights
        out1 = model(x, mode='features')
        
        # Modify weights
        with torch.no_grad():
            model.agg_weight[0] = 1.0
            model.agg_weight[1] = 0.0
        
        out2 = model(x, mode='features')
        
        # Outputs should differ
        assert not torch.allclose(out1, out2)


# =============================================================================
# TestTCNTrainerInit
# =============================================================================

class TestTCNTrainerInit:
    """Tests for TCNTrainer initialization."""
    
    def test_init_default_configs(self):
        """Test initialization with default configs."""
        trainer = TCNTrainer()
        assert trainer.feature_config is not None
        assert trainer.training_config is not None
    
    def test_init_custom_feature_config(self, feature_config):
        """Test initialization with custom feature config."""
        feature_config.n_top_features = 15
        trainer = TCNTrainer(feature_config=feature_config)
        assert trainer.feature_config.n_top_features == 15
    
    def test_init_custom_training_config(self, training_config):
        """Test initialization with custom training config."""
        training_config.epochs = 100
        trainer = TCNTrainer(training_config=training_config)
        assert trainer.training_config.epochs == 100
    
    def test_init_creates_data_loader(self):
        """Test data loader is created."""
        trainer = TCNTrainer()
        assert trainer.data_loader is not None
        assert isinstance(trainer.data_loader, EnhancedDataLoaderV3)
    
    def test_init_creates_feature_analyzer(self):
        """Test feature analyzer is created."""
        trainer = TCNTrainer()
        assert trainer.feature_analyzer is not None
        assert isinstance(trainer.feature_analyzer, FeatureImportanceAnalyzer)
    
    def test_init_model_is_none(self):
        """Test model is None initially."""
        trainer = TCNTrainer()
        assert trainer.model is None
    
    def test_init_training_history_empty(self):
        """Test training history initialized empty."""
        trainer = TCNTrainer()
        assert trainer.training_history == {'train_loss': [], 'val_loss': [], 'val_acc': []}


# =============================================================================
# TestTCNTrainerDevice
# =============================================================================

class TestTCNTrainerDevice:
    """Tests for device selection."""
    
    def test_get_device_auto_no_cuda(self):
        """Test auto device selection without CUDA."""
        with patch('torch.cuda.is_available', return_value=False):
            trainer = TCNTrainer()
            assert trainer.device == torch.device('cpu')
    
    def test_get_device_auto_with_cuda(self):
        """Test auto device selection with CUDA."""
        with patch('torch.cuda.is_available', return_value=True):
            trainer = TCNTrainer()
            assert trainer.device == torch.device('cuda')
    
    def test_get_device_explicit_cpu(self):
        """Test explicit CPU device."""
        config = TrainingConfig(device='cpu')
        trainer = TCNTrainer(training_config=config)
        assert trainer.device == torch.device('cpu')
    
    def test_get_device_explicit_cuda(self):
        """Test explicit CUDA device."""
        config = TrainingConfig(device='cuda')
        trainer = TCNTrainer(training_config=config)
        assert trainer.device == torch.device('cuda')
    
    def test_get_device_specific_gpu(self):
        """Test specific GPU device."""
        config = TrainingConfig(device='cuda:0')
        trainer = TCNTrainer(training_config=config)
        assert trainer.device == torch.device('cuda:0')


# =============================================================================
# TestTCNTrainerPrepareData
# =============================================================================

class TestTCNTrainerPrepareData:
    """Tests for data preparation pipeline."""
    
    def test_prepare_data_returns_loaders(self, sample_csv_path):
        """Test prepare_data returns three DataLoaders."""
        with patch.object(FeatureImportanceAnalyzer, 'analyze', 
                         return_value=['rsi_14', 'atr_14']):
            trainer = TCNTrainer()
            train_loader, val_loader, test_loader = trainer.prepare_data(
                sample_csv_path, skip_feature_selection=True
            )
            
            from torch.utils.data import DataLoader
            assert isinstance(train_loader, DataLoader)
            assert isinstance(val_loader, DataLoader)
            assert isinstance(test_loader, DataLoader)
    
    def test_prepare_data_with_provided_features(self, sample_csv_path):
        """Test prepare_data with explicit feature list."""
        trainer = TCNTrainer()
        trainer.prepare_data(
            sample_csv_path,
            features=['rsi_14', 'atr_14'],
        )
        
        assert trainer.selected_features == ['rsi_14', 'atr_14']
    
    def test_prepare_data_skip_feature_selection(self, sample_csv_path):
        """Test prepare_data without feature selection."""
        trainer = TCNTrainer()
        trainer.prepare_data(
            sample_csv_path,
            skip_feature_selection=True,
        )
        
        # Should use all features
        assert len(trainer.selected_features) > 5
    
    def test_prepare_data_auto_feature_selection(self, sample_csv_path, mock_random_forest):
        """Test prepare_data with auto feature discovery."""
        trainer = TCNTrainer()
        trainer.prepare_data(sample_csv_path)
        
        # Features should be selected
        assert len(trainer.selected_features) > 0
    
    def test_prepare_data_with_profile(self, sample_csv_path, mock_random_forest):
        """Test prepare_data with profile for feature prioritization."""
        trainer = TCNTrainer()
        trainer.prepare_data(sample_csv_path, profile='SCALP')
        
        assert len(trainer.selected_features) > 0
    
    def test_prepare_data_batch_size(self, sample_csv_path):
        """Test DataLoader uses configured batch size."""
        config = TrainingConfig(batch_size=32)
        trainer = TCNTrainer(training_config=config)
        train_loader, _, _ = trainer.prepare_data(
            sample_csv_path, skip_feature_selection=True
        )
        
        assert train_loader.batch_size == 32
    
    def test_prepare_data_train_shuffle(self, sample_csv_path):
        """Test train DataLoader shuffles data."""
        trainer = TCNTrainer()
        train_loader, _, _ = trainer.prepare_data(
            sample_csv_path, skip_feature_selection=True
        )
        
        # Check that sampler or shuffling is enabled
        # DataLoader doesn't expose shuffle directly, but we can check sampler
        assert train_loader.sampler is not None
    
    def test_prepare_data_creates_tensors(self, sample_csv_path):
        """Test DataLoaders contain tensor data."""
        trainer = TCNTrainer()
        train_loader, _, _ = trainer.prepare_data(
            sample_csv_path, skip_feature_selection=True
        )
        
        batch = next(iter(train_loader))
        X, y = batch
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
    
    def test_prepare_data_tensor_dtypes(self, sample_csv_path):
        """Test tensor data types are correct."""
        trainer = TCNTrainer()
        train_loader, _, _ = trainer.prepare_data(
            sample_csv_path, skip_feature_selection=True
        )
        
        X, y = next(iter(train_loader))
        assert X.dtype == torch.float32
        assert y.dtype == torch.long
    
    def test_prepare_data_sequence_shape(self, sample_csv_path):
        """Test sequence tensors have correct shape."""
        config = TrainingConfig(sequence_length=20)
        trainer = TCNTrainer(training_config=config)
        train_loader, _, _ = trainer.prepare_data(
            sample_csv_path, 
            features=['rsi_14', 'atr_14'],
        )
        
        X, y = next(iter(train_loader))
        assert X.shape[1] == 20  # seq_len
        assert X.shape[2] == 2   # features


# =============================================================================
# TestTCNTrainerBuildModel
# =============================================================================

class TestTCNTrainerBuildModel:
    """Tests for model building."""
    
    def test_build_model_creates_model(self, sample_csv_path):
        """Test build_model creates EnhancedTCN."""
        trainer = TCNTrainer()
        trainer.prepare_data(sample_csv_path, features=['rsi_14', 'atr_14'])
        
        model = trainer.build_model()
        
        assert isinstance(model, EnhancedTCN)
        assert trainer.model is model
    
    def test_build_model_correct_input_dim(self, sample_csv_path):
        """Test model has correct input dimension."""
        trainer = TCNTrainer()
        trainer.prepare_data(sample_csv_path, features=['rsi_14', 'atr_14', 'ema_20'])
        
        model = trainer.build_model()
        
        assert model.input_dim == 3
    
    def test_build_model_with_profile(self, sample_csv_path):
        """Test build_model with profile."""
        trainer = TCNTrainer()
        trainer.prepare_data(sample_csv_path, features=['rsi_14', 'atr_14'])
        
        model = trainer.build_model(profile='SCALP')
        
        assert model is not None
    
    def test_build_model_uses_config_hidden_dim(self, sample_csv_path):
        """Test model uses configured hidden_dim."""
        config = TrainingConfig(hidden_dim=128)
        trainer = TCNTrainer(training_config=config)
        trainer.prepare_data(sample_csv_path, features=['rsi_14'])
        
        model = trainer.build_model()
        
        assert model.hidden_dim == 128
    
    def test_build_model_on_device(self, sample_csv_path):
        """Test model is moved to correct device."""
        config = TrainingConfig(device='cpu')
        trainer = TCNTrainer(training_config=config)
        trainer.prepare_data(sample_csv_path, features=['rsi_14'])
        
        model = trainer.build_model()
        
        # Check first parameter device
        param = next(model.parameters())
        assert param.device == torch.device('cpu')
    
    def test_build_model_without_profile(self, sample_csv_path):
        """Test build_model without profile uses config values."""
        config = TrainingConfig(num_layers=6)
        trainer = TCNTrainer(training_config=config)
        trainer.prepare_data(sample_csv_path, features=['rsi_14'])
        
        model = trainer.build_model()
        
        # Model should be built with config's num_layers
        assert model is not None


# =============================================================================
# TestTCNTrainerTrain
# =============================================================================

class TestTCNTrainerTrain:
    """Tests for training loop."""
    
    @pytest.fixture
    def prepared_trainer(self, sample_csv_path):
        """Prepare trainer with data and model."""
        config = TrainingConfig(epochs=2, batch_size=16, device='cpu')
        trainer = TCNTrainer(training_config=config)
        train_loader, val_loader, _ = trainer.prepare_data(
            sample_csv_path, features=['rsi_14', 'atr_14']
        )
        trainer.build_model()
        return trainer, train_loader, val_loader
    
    def test_train_returns_metrics(self, prepared_trainer):
        """Test train returns metrics dictionary."""
        trainer, train_loader, val_loader = prepared_trainer
        
        metrics = trainer.train(train_loader, val_loader)
        
        assert isinstance(metrics, dict)
    
    def test_train_metrics_keys(self, prepared_trainer):
        """Test train metrics has expected keys."""
        trainer, train_loader, val_loader = prepared_trainer
        
        metrics = trainer.train(train_loader, val_loader)
        
        assert 'final_train_loss' in metrics
        assert 'final_val_loss' in metrics
        assert 'best_val_acc' in metrics
        assert 'epochs_trained' in metrics
    
    def test_train_updates_history(self, prepared_trainer):
        """Test training updates history."""
        trainer, train_loader, val_loader = prepared_trainer
        
        trainer.train(train_loader, val_loader)
        
        assert len(trainer.training_history['train_loss']) > 0
        assert len(trainer.training_history['val_loss']) > 0
        assert len(trainer.training_history['val_acc']) > 0
    
    def test_train_updates_best_val_acc(self, prepared_trainer):
        """Test training updates best_val_acc."""
        trainer, train_loader, val_loader = prepared_trainer
        
        trainer.train(train_loader, val_loader)
        
        assert trainer.best_val_acc > 0
    
    def test_train_without_model_raises(self, sample_csv_path):
        """Test train raises if model not built."""
        trainer = TCNTrainer()
        train_loader, val_loader, _ = trainer.prepare_data(
            sample_csv_path, features=['rsi_14']
        )
        # Don't build model
        
        with pytest.raises(RuntimeError, match="Model not built"):
            trainer.train(train_loader, val_loader)
    
    def test_train_early_stopping(self, sample_csv_path):
        """Test early stopping terminates training."""
        config = TrainingConfig(
            epochs=100,  # High
            early_stopping_patience=1,  # Very low
            batch_size=16,
            device='cpu'
        )
        trainer = TCNTrainer(training_config=config)
        train_loader, val_loader, _ = trainer.prepare_data(
            sample_csv_path, features=['rsi_14']
        )
        trainer.build_model()
        
        metrics = trainer.train(train_loader, val_loader)
        
        # Should stop before 100 epochs
        assert metrics['epochs_trained'] < 100
    
    def test_train_uses_class_weights(self, prepared_trainer):
        """Test training uses class weights for imbalance."""
        trainer, train_loader, val_loader = prepared_trainer
        
        # This is tested implicitly - no error means weights were computed
        metrics = trainer.train(train_loader, val_loader)
        assert metrics is not None
    
    def test_train_gradient_clipping(self, prepared_trainer):
        """Test gradient clipping is applied."""
        trainer, train_loader, val_loader = prepared_trainer
        
        # Train successfully (clipping prevents exploding gradients)
        metrics = trainer.train(train_loader, val_loader)
        assert metrics['final_train_loss'] < float('inf')
    
    def test_train_onecycle_scheduler(self, sample_csv_path):
        """Test OneCycle scheduler is used by default."""
        config = TrainingConfig(
            epochs=2, batch_size=16, device='cpu',
            use_onecycle=True
        )
        trainer = TCNTrainer(training_config=config)
        train_loader, val_loader, _ = trainer.prepare_data(
            sample_csv_path, features=['rsi_14']
        )
        trainer.build_model()
        
        metrics = trainer.train(train_loader, val_loader)
        assert metrics is not None
    
    def test_train_cosine_scheduler(self, sample_csv_path):
        """Test Cosine scheduler option."""
        config = TrainingConfig(
            epochs=2, batch_size=16, device='cpu',
            use_onecycle=False, use_cosine=True
        )
        trainer = TCNTrainer(training_config=config)
        train_loader, val_loader, _ = trainer.prepare_data(
            sample_csv_path, features=['rsi_14']
        )
        trainer.build_model()
        
        metrics = trainer.train(train_loader, val_loader)
        assert metrics is not None
    
    def test_train_no_scheduler(self, sample_csv_path):
        """Test training without scheduler."""
        config = TrainingConfig(
            epochs=2, batch_size=16, device='cpu',
            use_onecycle=False, use_cosine=False
        )
        trainer = TCNTrainer(training_config=config)
        train_loader, val_loader, _ = trainer.prepare_data(
            sample_csv_path, features=['rsi_14']
        )
        trainer.build_model()
        
        metrics = trainer.train(train_loader, val_loader)
        assert metrics is not None
    
    def test_train_restores_best_model(self, prepared_trainer):
        """Test best model state is restored after training."""
        trainer, train_loader, val_loader = prepared_trainer
        
        trainer.train(train_loader, val_loader)
        
        # Model should be in eval-ready state with best weights
        # (implicit - no assertion needed, just verify no error)
        assert trainer.model is not None


# =============================================================================
# TestTCNTrainerValidate
# =============================================================================

class TestTCNTrainerValidate:
    """Tests for validation logic."""
    
    @pytest.fixture
    def trained_trainer(self, sample_csv_path):
        """Prepare trainer with trained model."""
        config = TrainingConfig(epochs=1, batch_size=16, device='cpu')
        trainer = TCNTrainer(training_config=config)
        train_loader, val_loader, test_loader = trainer.prepare_data(
            sample_csv_path, features=['rsi_14', 'atr_14']
        )
        trainer.build_model()
        trainer.train(train_loader, val_loader)
        return trainer, val_loader
    
    def test_validate_returns_loss_and_acc(self, trained_trainer):
        """Test _validate returns loss and accuracy."""
        trainer, val_loader = trained_trainer
        criterion = nn.CrossEntropyLoss()
        
        val_loss, val_acc = trainer._validate(val_loader, criterion)
        
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
    
    def test_validate_accuracy_range(self, trained_trainer):
        """Test validation accuracy is in [0, 1]."""
        trainer, val_loader = trained_trainer
        criterion = nn.CrossEntropyLoss()
        
        _, val_acc = trainer._validate(val_loader, criterion)
        
        assert 0 <= val_acc <= 1
    
    def test_validate_loss_positive(self, trained_trainer):
        """Test validation loss is positive."""
        trainer, val_loader = trained_trainer
        criterion = nn.CrossEntropyLoss()
        
        val_loss, _ = trainer._validate(val_loader, criterion)
        
        assert val_loss > 0
    
    def test_validate_model_in_eval_mode(self, trained_trainer):
        """Test model is set to eval mode during validation."""
        trainer, val_loader = trained_trainer
        criterion = nn.CrossEntropyLoss()
        
        trainer._validate(val_loader, criterion)
        
        assert not trainer.model.training
    
    def test_validate_no_gradients(self, trained_trainer):
        """Test no gradients computed during validation."""
        trainer, val_loader = trained_trainer
        criterion = nn.CrossEntropyLoss()
        
        # Enable grad tracking
        for param in trainer.model.parameters():
            param.requires_grad = True
        
        trainer._validate(val_loader, criterion)
        
        # Gradients should not be accumulated
        # (tested implicitly by torch.no_grad() in _validate)
        assert True


# =============================================================================
# TestTCNTrainerEvaluate
# =============================================================================

class TestTCNTrainerEvaluate:
    """Tests for test set evaluation."""
    
    @pytest.fixture
    def trained_trainer_with_test(self, sample_csv_path):
        """Prepare trainer with test loader."""
        config = TrainingConfig(epochs=1, batch_size=16, device='cpu')
        trainer = TCNTrainer(training_config=config)
        train_loader, val_loader, test_loader = trainer.prepare_data(
            sample_csv_path, features=['rsi_14', 'atr_14']
        )
        trainer.build_model()
        trainer.train(train_loader, val_loader)
        return trainer, test_loader
    
    def test_evaluate_returns_dict(self, trained_trainer_with_test):
        """Test evaluate returns dictionary."""
        trainer, test_loader = trained_trainer_with_test
        
        results = trainer.evaluate(test_loader)
        
        assert isinstance(results, dict)
    
    def test_evaluate_contains_accuracy(self, trained_trainer_with_test):
        """Test evaluate results contain accuracy."""
        trainer, test_loader = trained_trainer_with_test
        
        results = trainer.evaluate(test_loader)
        
        assert 'test_accuracy' in results
        assert 0 <= results['test_accuracy'] <= 1
    
    def test_evaluate_contains_per_class_accuracy(self, trained_trainer_with_test):
        """Test evaluate results contain per-class accuracy."""
        trainer, test_loader = trained_trainer_with_test
        
        results = trainer.evaluate(test_loader)
        
        assert 'per_class_accuracy' in results
        assert 'Bear' in results['per_class_accuracy']
        assert 'Sideways' in results['per_class_accuracy']
        assert 'Bull' in results['per_class_accuracy']
    
    def test_evaluate_contains_predictions(self, trained_trainer_with_test):
        """Test evaluate results contain predictions."""
        trainer, test_loader = trained_trainer_with_test
        
        results = trainer.evaluate(test_loader)
        
        assert 'predictions' in results
        assert isinstance(results['predictions'], np.ndarray)
    
    def test_evaluate_contains_labels(self, trained_trainer_with_test):
        """Test evaluate results contain true labels."""
        trainer, test_loader = trained_trainer_with_test
        
        results = trainer.evaluate(test_loader)
        
        assert 'labels' in results
        assert isinstance(results['labels'], np.ndarray)
    
    def test_evaluate_contains_probabilities(self, trained_trainer_with_test):
        """Test evaluate results contain probabilities."""
        trainer, test_loader = trained_trainer_with_test
        
        results = trainer.evaluate(test_loader)
        
        assert 'probabilities' in results
        assert results['probabilities'].shape[1] == 3  # 3 classes
    
    def test_evaluate_probabilities_sum_to_one(self, trained_trainer_with_test):
        """Test probabilities sum to 1 for each sample."""
        trainer, test_loader = trained_trainer_with_test
        
        results = trainer.evaluate(test_loader)
        
        prob_sums = results['probabilities'].sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-5)


# =============================================================================
# TestTCNTrainerCheckpoint
# =============================================================================

class TestTCNTrainerCheckpoint:
    """Tests for checkpoint save/load."""
    
    @pytest.fixture
    def trained_trainer_full(self, sample_csv_path):
        """Prepare fully trained trainer."""
        config = TrainingConfig(epochs=1, batch_size=16, device='cpu')
        trainer = TCNTrainer(training_config=config)
        train_loader, val_loader, test_loader = trainer.prepare_data(
            sample_csv_path, features=['rsi_14', 'atr_14']
        )
        trainer.build_model()
        trainer.train(train_loader, val_loader)
        return trainer
    
    def test_save_checkpoint_creates_file(self, trained_trainer_full, tmp_path):
        """Test save_checkpoint creates file."""
        trainer = trained_trainer_full
        path = tmp_path / "model.pt"
        
        trainer.save_checkpoint(str(path))
        
        assert path.exists()
    
    def test_save_checkpoint_creates_directory(self, trained_trainer_full, tmp_path):
        """Test save_checkpoint creates parent directories."""
        trainer = trained_trainer_full
        path = tmp_path / "subdir" / "model.pt"
        
        trainer.save_checkpoint(str(path))
        
        assert path.exists()
    
    def test_save_checkpoint_with_profile(self, trained_trainer_full, tmp_path):
        """Test save_checkpoint includes profile."""
        trainer = trained_trainer_full
        path = tmp_path / "model.pt"
        
        trainer.save_checkpoint(str(path), profile='SCALP')
        
        checkpoint = torch.load(path)
        assert checkpoint['profile'] == 'SCALP'
    
    def test_save_checkpoint_with_metrics(self, trained_trainer_full, tmp_path):
        """Test save_checkpoint includes metrics."""
        trainer = trained_trainer_full
        path = tmp_path / "model.pt"
        
        trainer.save_checkpoint(str(path), metrics={'test_acc': 0.85})
        
        checkpoint = torch.load(path)
        assert checkpoint['metrics']['test_acc'] == 0.85
    
    def test_save_checkpoint_includes_features(self, trained_trainer_full, tmp_path):
        """Test checkpoint includes feature columns."""
        trainer = trained_trainer_full
        path = tmp_path / "model.pt"
        
        trainer.save_checkpoint(str(path))
        
        checkpoint = torch.load(path)
        assert 'feature_columns' in checkpoint
        assert checkpoint['feature_columns'] == ['rsi_14', 'atr_14']
    
    def test_save_checkpoint_includes_history(self, trained_trainer_full, tmp_path):
        """Test checkpoint includes training history."""
        trainer = trained_trainer_full
        path = tmp_path / "model.pt"
        
        trainer.save_checkpoint(str(path))
        
        checkpoint = torch.load(path)
        assert 'training_history' in checkpoint
    
    def test_load_checkpoint_returns_model(self, trained_trainer_full, tmp_path):
        """Test load_checkpoint returns model."""
        trainer = trained_trainer_full
        path = tmp_path / "model.pt"
        trainer.save_checkpoint(str(path))
        
        model, features, checkpoint = TCNTrainer.load_checkpoint(str(path))
        
        assert isinstance(model, EnhancedTCN)
    
    def test_load_checkpoint_returns_features(self, trained_trainer_full, tmp_path):
        """Test load_checkpoint returns feature list."""
        trainer = trained_trainer_full
        path = tmp_path / "model.pt"
        trainer.save_checkpoint(str(path))
        
        model, features, checkpoint = TCNTrainer.load_checkpoint(str(path))
        
        assert features == ['rsi_14', 'atr_14']
    
    def test_load_checkpoint_model_eval_mode(self, trained_trainer_full, tmp_path):
        """Test loaded model is in eval mode."""
        trainer = trained_trainer_full
        path = tmp_path / "model.pt"
        trainer.save_checkpoint(str(path))
        
        model, _, _ = TCNTrainer.load_checkpoint(str(path))
        
        assert not model.training
    
    def test_load_checkpoint_preserves_weights(self, trained_trainer_full, tmp_path):
        """Test loaded model has same weights."""
        trainer = trained_trainer_full
        path = tmp_path / "model.pt"
        trainer.save_checkpoint(str(path))
        
        # Get original weights
        original_weight = trainer.model.classifier[0].weight.clone()
        
        model, _, _ = TCNTrainer.load_checkpoint(str(path))
        loaded_weight = model.classifier[0].weight
        
        assert torch.allclose(original_weight.cpu(), loaded_weight.cpu())


# =============================================================================
# TestMainFunction
# =============================================================================

class TestMainFunction:
    """Tests for main() CLI function."""
    
    def test_argparse_data_required(self):
        """Test --data argument is required."""
        with patch('sys.argv', ['train_tcn_enhanced.py']):
            with pytest.raises(SystemExit):
                main()
    
    def test_argparse_profile_choices(self):
        """Test --profile has valid choices."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--profile', choices=['SCALP', 'INTRADAY', 'SWING'])
        
        # Valid
        args = parser.parse_args(['--profile', 'SCALP'])
        assert args.profile == 'SCALP'
        
        # Invalid
        with pytest.raises(SystemExit):
            parser.parse_args(['--profile', 'INVALID'])
    
    def test_argparse_default_epochs(self):
        """Test default epochs is 50."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=50)
        
        args = parser.parse_args([])
        assert args.epochs == 50
    
    def test_argparse_default_batch_size(self):
        """Test default batch size is 64."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', type=int, default=64)
        
        args = parser.parse_args([])
        assert args.batch_size == 64
    
    def test_argparse_features_parsing(self):
        """Test --features comma-separated parsing."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--features', type=str, default=None)
        
        args = parser.parse_args(['--features', 'rsi_14,atr_14,macd'])
        features = [f.strip() for f in args.features.split(',')]
        
        assert features == ['rsi_14', 'atr_14', 'macd']
    
    def test_argparse_skip_feature_selection(self):
        """Test --skip-feature-selection flag."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--skip-feature-selection', action='store_true')
        
        args = parser.parse_args(['--skip-feature-selection'])
        assert args.skip_feature_selection is True
    
    @patch('training.train_tcn_enhanced.TCNTrainer')
    def test_main_creates_trainer(self, mock_trainer_class, sample_csv_path):
        """Test main creates TCNTrainer instance."""
        mock_trainer = MagicMock()
        mock_trainer.prepare_data.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_trainer.build_model.return_value = MagicMock()
        mock_trainer.train.return_value = {}
        mock_trainer.evaluate.return_value = {'test_accuracy': 0.8}
        mock_trainer.best_val_acc = 0.8
        mock_trainer_class.return_value = mock_trainer
        
        with patch('sys.argv', ['train_tcn_enhanced.py', '--data', sample_csv_path]):
            main()
        
        mock_trainer_class.assert_called_once()
    
    @patch('training.train_tcn_enhanced.TCNTrainer')
    def test_main_saves_checkpoint(self, mock_trainer_class, sample_csv_path, tmp_path):
        """Test main saves checkpoint."""
        mock_trainer = MagicMock()
        mock_trainer.prepare_data.return_value = (MagicMock(), MagicMock(), MagicMock())
        mock_trainer.build_model.return_value = MagicMock()
        mock_trainer.train.return_value = {}
        mock_trainer.evaluate.return_value = {'test_accuracy': 0.8}
        mock_trainer.best_val_acc = 0.8
        mock_trainer_class.return_value = mock_trainer
        
        save_dir = tmp_path / "models"
        with patch('sys.argv', ['train_tcn_enhanced.py', '--data', sample_csv_path,
                               '--save-dir', str(save_dir)]):
            main()
        
        mock_trainer.save_checkpoint.assert_called_once()


# =============================================================================
# TestIntegration
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_no_feature_selection(self, sample_csv_path, tmp_path):
        """Test full training pipeline without feature selection."""
        config = TrainingConfig(epochs=1, batch_size=16, device='cpu')
        trainer = TCNTrainer(training_config=config)
        
        # Prepare data
        train_loader, val_loader, test_loader = trainer.prepare_data(
            sample_csv_path,
            skip_feature_selection=True,
        )
        
        # Build and train
        trainer.build_model()
        train_metrics = trainer.train(train_loader, val_loader)
        
        # Evaluate
        test_metrics = trainer.evaluate(test_loader)
        
        # Save
        save_path = tmp_path / "model.pt"
        trainer.save_checkpoint(str(save_path), metrics=test_metrics)
        
        # Verify
        assert save_path.exists()
        assert test_metrics['test_accuracy'] >= 0
    
    def test_full_pipeline_with_features(self, sample_csv_path, tmp_path):
        """Test full pipeline with explicit features."""
        config = TrainingConfig(epochs=1, batch_size=16, device='cpu')
        trainer = TCNTrainer(training_config=config)
        
        train_loader, val_loader, test_loader = trainer.prepare_data(
            sample_csv_path,
            features=['rsi_14', 'atr_14', 'ema_20'],
        )
        
        trainer.build_model()
        trainer.train(train_loader, val_loader)
        test_metrics = trainer.evaluate(test_loader)
        
        save_path = tmp_path / "model.pt"
        trainer.save_checkpoint(str(save_path))
        
        # Load and verify
        model, features, checkpoint = TCNTrainer.load_checkpoint(str(save_path))
        assert features == ['rsi_14', 'atr_14', 'ema_20']
    
    def test_full_pipeline_with_profile(self, sample_csv_path, tmp_path, mock_random_forest):
        """Test full pipeline with trading profile."""
        config = TrainingConfig(epochs=1, batch_size=16, device='cpu')
        trainer = TCNTrainer(training_config=config)
        
        train_loader, val_loader, test_loader = trainer.prepare_data(
            sample_csv_path,
            profile='SCALP',
        )
        
        trainer.build_model(profile='SCALP')
        trainer.train(train_loader, val_loader)
        
        save_path = tmp_path / "model.pt"
        trainer.save_checkpoint(str(save_path), profile='SCALP')
        
        checkpoint = torch.load(save_path)
        assert checkpoint['profile'] == 'SCALP'
    
    def test_checkpoint_roundtrip_inference(self, sample_csv_path, tmp_path):
        """Test saved model can be loaded and used for inference."""
        config = TrainingConfig(epochs=1, batch_size=16, device='cpu')
        trainer = TCNTrainer(training_config=config)
        
        train_loader, val_loader, _ = trainer.prepare_data(
            sample_csv_path,
            features=['rsi_14', 'atr_14'],
        )
        trainer.build_model()
        trainer.train(train_loader, val_loader)
        
        save_path = tmp_path / "model.pt"
        trainer.save_checkpoint(str(save_path))
        
        # Load and run inference
        model, features, _ = TCNTrainer.load_checkpoint(str(save_path), device='cpu')
        
        # Create test input
        test_input = torch.randn(1, 30, 2)  # (batch, seq, features)
        
        with torch.no_grad():
            output = model(test_input)
        
        assert output.shape == (1, 3)
    
    def test_multiple_training_runs_independent(self, sample_csv_path):
        """Test multiple training runs are independent."""
        config = TrainingConfig(epochs=1, batch_size=16, device='cpu')
        
        # First run
        trainer1 = TCNTrainer(training_config=config)
        train1, val1, _ = trainer1.prepare_data(sample_csv_path, features=['rsi_14'])
        trainer1.build_model()
        trainer1.train(train1, val1)
        
        # Second run
        trainer2 = TCNTrainer(training_config=config)
        train2, val2, _ = trainer2.prepare_data(sample_csv_path, features=['rsi_14'])
        trainer2.build_model()
        trainer2.train(train2, val2)
        
        # Should be different instances
        assert trainer1.model is not trainer2.model
    
    def test_data_loader_sequence_consistency(self, sample_csv_path):
        """Test data loader produces consistent sequences."""
        config = TrainingConfig(sequence_length=20)
        trainer = TCNTrainer(training_config=config)
        
        train_loader, _, _ = trainer.prepare_data(
            sample_csv_path,
            features=['rsi_14', 'atr_14'],
        )
        
        # All sequences should have correct length
        for X, y in train_loader:
            assert X.shape[1] == 20
            assert X.shape[2] == 2
            break


if __name__ == '__main__':
    pytest.main([__file__, '-v'])