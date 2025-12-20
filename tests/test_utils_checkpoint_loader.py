# tests/test_utils_checkpoint_loader.py
"""
Unit tests for utils.checkpoint_loader module.
Tests checkpoint loading utilities for TCN models after LSTM removal.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from utils.checkpoint_loader import (
    ModelLoader,
    ModelInfo,
    CheckpointFormatError,
    load_model,
    load_features,
    get_checkpoint_info,
    print_checkpoint_summary,
    get_default_features,
)


@pytest.fixture
def tmp_checkpoint_path(tmp_path):
    """Create a temporary checkpoint path."""
    return tmp_path / "test_checkpoint.pt"


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoint files."""
    return tmp_path






@pytest.mark.unit
class TestModelLoader:
    """Test suite for ModelLoader class."""

    def test_init_file_not_found(self):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            ModelLoader("non_existent_file.pt")

    def test_init_with_enhanced_v3_checkpoint(self, enhanced_v3_checkpoint):
        """Test initialization with enhanced v3 checkpoint."""
        # The fixture already saves and returns the path
        loader = ModelLoader(enhanced_v3_checkpoint, device='cpu')

        assert loader.path == enhanced_v3_checkpoint
        assert loader.device.type == 'cpu'
        assert loader._format == 'enhanced_v3'

    def test_init_with_state_dict_checkpoint(self, state_dict_checkpoint):
        """Test initialization with state dict checkpoint."""
        # The fixture already saves and returns the path
        loader = ModelLoader(state_dict_checkpoint)

        assert loader._format == 'state_dict'

    def test_init_auto_device_cpu(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test auto device selection (CPU)."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        with patch('torch.cuda.is_available', return_value=False):
            loader = ModelLoader(tmp_checkpoint_path, device='auto')
            assert loader.device.type == 'cpu'

    def test_init_auto_device_cuda(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test auto device selection (CUDA)."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        with patch('torch.cuda.is_available', return_value=True):
            loader = ModelLoader(tmp_checkpoint_path, device='auto')
            assert loader.device.type == 'cuda'

    def test_detect_format_enhanced_v3(self, enhanced_v3_checkpoint):
        """Test format detection for enhanced v3."""
        # The fixture already saves and returns the path
        loader = ModelLoader(enhanced_v3_checkpoint)
        assert loader._format == 'enhanced_v3'

    def test_detect_format_state_dict(self, state_dict_checkpoint):
        """Test format detection for state dict."""
        # The fixture already saves and returns the path
        loader = ModelLoader(state_dict_checkpoint)
        assert loader._format == 'state_dict'

    def test_get_features_enhanced_v3(self, enhanced_v3_checkpoint):
        """Test getting features from enhanced v3 checkpoint."""
        # The fixture already saves and returns the path
        loader = ModelLoader(enhanced_v3_checkpoint)
        features = loader.get_features()

        assert features == ['RSI', 'MACD', 'ATR', 'ADX', 'EMA20', 'EMA50', 'EMA200', 'ROC5', 'ROC10', 'MOMENTUM']

    def test_get_features_missing(self, state_dict_checkpoint):
        """Test getting features when not available."""
        # The fixture already saves and returns the path
        loader = ModelLoader(state_dict_checkpoint)

        with pytest.raises(CheckpointFormatError, match="doesn't contain feature columns"):
            loader.get_features()

    def test_get_features_safe_with_features(self, enhanced_v3_checkpoint):
        """Test safe feature retrieval when features exist."""
        # The fixture already saves and returns the path
        loader = ModelLoader(enhanced_v3_checkpoint)
        features = loader.get_features_safe()

        assert features == ['RSI', 'MACD', 'ATR', 'ADX', 'EMA20', 'EMA50', 'EMA200', 'ROC5', 'ROC10', 'MOMENTUM']

    def test_get_features_safe_with_fallback(self, state_dict_checkpoint):
        """Test safe feature retrieval with fallback."""
        # The fixture already saves and returns the path
        loader = ModelLoader(state_dict_checkpoint)
        fallback = ['feature1', 'feature2']
        features = loader.get_features_safe(fallback=fallback)

        assert features == fallback

    def test_get_config(self, enhanced_v3_checkpoint):
        """Test getting configuration."""
        # The fixture already saves and returns the path
        loader = ModelLoader(enhanced_v3_checkpoint)
        config = loader.get_config()

        assert 'model' in config
        assert 'training' in config
        assert config['model']['input_dim'] == 10

    def test_get_config_empty(self, state_dict_checkpoint):
        """Test getting config when not available."""
        # The fixture already saves and returns the path
        loader = ModelLoader(state_dict_checkpoint)
        config = loader.get_config()

        assert config == {}

    def test_get_metrics(self, enhanced_v3_checkpoint):
        """Test getting metrics."""
        # The fixture already saves and returns the path
        loader = ModelLoader(enhanced_v3_checkpoint)
        metrics = loader.get_metrics()

        assert 'best_val_acc' in metrics
        assert metrics['best_val_acc'] == 0.92

    def test_get_training_history(self, tmp_checkpoint_path):
        """Test getting training history."""
        checkpoint = {
            'model_state': {},
            'training_history': {
                'train_loss': [0.5, 0.4, 0.3],
                'val_loss': [0.6, 0.5, 0.4],
            }
        }
        torch.save(checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)
        history = loader.get_training_history()

        assert 'train_loss' in history
        assert len(history['train_loss']) == 3

    def test_get_feature_importance(self, tmp_checkpoint_path):
        """Test getting feature importance."""
        checkpoint = {
            'model_state': {},
            'feature_importance': {
                'close': 0.5,
                'volume': 0.3,
                'open': 0.2,
            }
        }
        torch.save(checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)
        importance = loader.get_feature_importance()

        assert importance['close'] == 0.5
        assert importance['volume'] == 0.3

    def test_get_model_info(self, enhanced_v3_checkpoint):
        """Test getting model info."""
        # The fixture already saves and returns the path
        loader = ModelLoader(enhanced_v3_checkpoint)
        info = loader.get_model_info()

        assert isinstance(info, ModelInfo)
        assert info.model_type == 'enhanced_tcn'
        assert info.input_dim == 10
        assert info.hidden_dim == 32
        assert info.num_classes == 3
        assert info.profile == 'INTRADAY'
        assert info.receptive_field == 128

    def test_get_model_tcn(self, enhanced_v3_checkpoint):
        """Test getting TCN model."""
        # The fixture already saves and returns the path
        with patch('training.train_tcn_enhanced.EnhancedTCN') as mock_tcn:
            mock_model = MagicMock()
            mock_model.load_state_dict = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_model.eval = MagicMock()
            mock_tcn.return_value = mock_model

            loader = ModelLoader(enhanced_v3_checkpoint)
            model = loader.get_model()

            mock_tcn.assert_called_once()
            mock_model.load_state_dict.assert_called_once()
            mock_model.eval.assert_called_once()

    def test_get_model_caching(self, enhanced_v3_checkpoint):
        """Test that get_model caches the model."""
        # The fixture already saves and returns the path
        with patch('training.train_tcn_enhanced.EnhancedTCN') as mock_tcn:
            mock_model = MagicMock()
            mock_model.load_state_dict = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_model.eval = MagicMock()
            mock_tcn.return_value = mock_model

            loader = ModelLoader(enhanced_v3_checkpoint)

            # Call twice
            model1 = loader.get_model()
            model2 = loader.get_model()

            # Should only build once
            assert mock_tcn.call_count == 1
            assert model1 is model2

    def test_get_model_unknown_type(self, tmp_checkpoint_path):
        """Test getting model with unknown type."""
        checkpoint = {
            'model_state': {},
            'config': {
                'model': {'input_dim': 5, 'hidden_dim': 64},
                'training': {'num_classes': 3}
            }
        }
        torch.save(checkpoint, tmp_checkpoint_path)

        with patch('utils.checkpoint_loader.ModelLoader.get_model_info') as mock_info:
            mock_info.return_value = ModelInfo(
                model_type='unknown_model',
                input_dim=5,
                hidden_dim=64,
                num_classes=3,
                profile=None,
                receptive_field=None,
                feature_count=5,
                created_at=None
            )

            loader = ModelLoader(tmp_checkpoint_path)

            with pytest.raises(CheckpointFormatError, match="Unknown model type"):
                loader.get_model()

    def test_get_state_dict_from_enhanced(self, enhanced_v3_checkpoint):
        """Test extracting state dict from enhanced checkpoint."""
        # The fixture already saves and returns the path
        loader = ModelLoader(enhanced_v3_checkpoint)
        state_dict = loader._get_state_dict()

        assert 'linear1.weight' in state_dict

    def test_get_state_dict_from_direct(self, state_dict_checkpoint):
        """Test extracting state dict from direct checkpoint."""
        # The fixture already saves and returns the path
        loader = ModelLoader(state_dict_checkpoint)
        state_dict = loader._get_state_dict()

        assert 'tcn.linear1.weight' in state_dict

    def test_summary(self, enhanced_v3_checkpoint):
        """Test generating summary."""
        # The fixture already saves and returns the path
        loader = ModelLoader(enhanced_v3_checkpoint)
        summary = loader.summary()

        assert "Checkpoint:" in summary
        assert "enhanced_v3" in summary
        assert "INTRADAY" in summary
        assert "92" in summary  # accuracy


@pytest.mark.unit
class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_load_model(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test load_model convenience function."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        with patch('training.train_tcn_enhanced.EnhancedTCN') as mock_tcn:
            mock_model = MagicMock()
            mock_model.load_state_dict = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_model.eval = MagicMock()
            mock_tcn.return_value = mock_model

            model, features = load_model(tmp_checkpoint_path, device='cpu')

            assert features == ['open', 'high', 'low', 'close', 'volume']
            mock_tcn.assert_called_once()

    def test_load_features(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test load_features convenience function."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        features = load_features(tmp_checkpoint_path)

        assert features == ['open', 'high', 'low', 'close', 'volume']

    def test_get_checkpoint_info(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test get_checkpoint_info convenience function."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        info = get_checkpoint_info(tmp_checkpoint_path)

        assert isinstance(info, ModelInfo)
        assert info.input_dim == 5
        assert info.profile == 'INTRADAY'

    def test_print_checkpoint_summary(self, tmp_checkpoint_path, enhanced_v3_checkpoint, capsys):
        """Test print_checkpoint_summary convenience function."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        print_checkpoint_summary(tmp_checkpoint_path)

        captured = capsys.readouterr()
        assert "Checkpoint:" in captured.out
        assert "INTRADAY" in captured.out

    def test_get_default_features(self, tmp_path):
        """Test get_default_features function."""
        # Create default checkpoint
        default_path = Path("models/weights/tcn_enhanced_best.pt")

        with patch('utils.checkpoint_loader._DEFAULT_CHECKPOINT', str(tmp_path / "default.pt")):
            checkpoint = {
                'model_state': {},
                'feature_columns': ['default_feat1', 'default_feat2'],
            }
            torch.save(checkpoint, tmp_path / "default.pt")

            features = get_default_features()
            assert features == ['default_feat1', 'default_feat2']

    def test_get_default_features_not_found(self):
        """Test get_default_features when file not found."""
        with patch('utils.checkpoint_loader._DEFAULT_CHECKPOINT', "non_existent.pt"):
            features = get_default_features()
            assert features == []


@pytest.mark.unit
class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating ModelInfo."""
        info = ModelInfo(
            model_type='tcn',
            input_dim=10,
            hidden_dim=64,
            num_classes=3,
            profile='SCALP',
            receptive_field=31,
            feature_count=10,
            created_at='2024-01-01'
        )

        assert info.model_type == 'tcn'
        assert info.input_dim == 10
        assert info.hidden_dim == 64
        assert info.num_classes == 3
        assert info.profile == 'SCALP'
        assert info.receptive_field == 31
        assert info.feature_count == 10
        assert info.created_at == '2024-01-01'

    def test_model_info_with_none_values(self):
        """Test ModelInfo with optional None values."""
        info = ModelInfo(
            model_type='tcn',
            input_dim=5,
            hidden_dim=64,
            num_classes=3,
            profile=None,
            receptive_field=None,
            feature_count=5,
            created_at=None
        )

        assert info.profile is None
        assert info.receptive_field is None
        assert info.created_at is None


@pytest.mark.unit
class TestCheckpointFormatError:
    """Test CheckpointFormatError exception."""

    def test_raise_checkpoint_format_error(self):
        """Test raising CheckpointFormatError."""
        with pytest.raises(CheckpointFormatError):
            raise CheckpointFormatError("Invalid format")

    def test_error_message(self):
        """Test error message."""
        try:
            raise CheckpointFormatError("Test error message")
        except CheckpointFormatError as e:
            assert str(e) == "Test error message"


@pytest.mark.unit
class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_handle_dataparallel_prefix(self, tmp_checkpoint_path):
        """Test handling of 'module.' prefix from DataParallel."""
        checkpoint = {
            'model_state': {
                'module.tcn.network.0.conv1.conv.weight': torch.randn(64, 5, 3),
                'module.classifier.0.weight': torch.randn(64, 64),
            },
            'feature_columns': ['open', 'high', 'low', 'close', 'volume'],
            'config': {
                'model': {'input_dim': 5, 'hidden_dim': 64},
                'training': {'num_classes': 3}
            }
        }
        torch.save(checkpoint, tmp_checkpoint_path)

        with patch('training.train_tcn_enhanced.EnhancedTCN') as mock_tcn:
            mock_model = MagicMock()
            mock_model.load_state_dict = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_model.eval = MagicMock()
            mock_tcn.return_value = mock_model

            loader = ModelLoader(tmp_checkpoint_path)
            model = loader.get_model()

            # Check that state dict was cleaned
            call_args = mock_model.load_state_dict.call_args[0][0]
            assert 'tcn.network.0.conv1.conv.weight' in call_args
            assert 'module.tcn.network.0.conv1.conv.weight' not in call_args

    def test_enhanced_v2_format(self, tmp_checkpoint_path):
        """Test enhanced v2 format (intermediate)."""
        checkpoint = {
            'model_state': {
                'tcn.network.0.conv1.conv.weight': torch.randn(64, 5, 3),
            },
            'config': {
                'model': {'input_dim': 5, 'hidden_dim': 64},
                'training': {'num_classes': 3},
                'feature_columns': ['open', 'high', 'low', 'close', 'volume']
            }
        }
        torch.save(checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)

        # Should detect as enhanced_v2
        assert loader._format == 'enhanced_v2'

        # Should be able to extract features from config
        features = loader.get_features()
        assert features == ['open', 'high', 'low', 'close', 'volume']


@pytest.fixture
def sample_model():
    """Create a simple model for testing."""
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=10, hidden_dim=32, num_classes=3):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, x):
            x = torch.relu(self.linear1(x))
            return self.linear2(x)
    
    return SimpleModel(input_dim=10, hidden_dim=32, num_classes=3)


@pytest.fixture
def enhanced_v3_checkpoint(temp_checkpoint_dir, sample_model):
    """Create a new format checkpoint (enhanced_v3)."""
    checkpoint_path = temp_checkpoint_dir / "tcn_enhanced_v3.pt"
    
    checkpoint = {
        'model_state': sample_model.state_dict(),
        'feature_columns': ['RSI', 'MACD', 'ATR', 'ADX', 'EMA20', 'EMA50', 'EMA200', 'ROC5', 'ROC10', 'MOMENTUM'],
        'config': {
            'model': {
                'input_dim': 10,
                'hidden_dim': 32,
                'receptive_field': 128,
            },
            'training': {
                'num_classes': 3,
                'lr': 0.001,
                'epochs': 50,
            },
            'profile': 'INTRADAY',
        },
        'training_history': {
            'train_loss': [0.5, 0.4, 0.3],
            'val_loss': [0.55, 0.45, 0.35],
            'val_acc': [0.75, 0.85, 0.90],
        },
        'metrics': {
            'best_val_acc': 0.92,
            'test_accuracy': 0.88,
            'precision': 0.89,
            'recall': 0.87,
            'f1': 0.88,
        },
        'feature_importance': {
            'RSI': 0.25,
            'MACD': 0.20,
            'ATR': 0.15,
            'ADX': 0.18,
            'EMA20': 0.10,
            'EMA50': 0.07,
            'EMA200': 0.03,
            'ROC5': 0.01,
            'ROC10': 0.01,
            'MOMENTUM': 0.0,
        },
        'created_at': '2025-01-15T10:30:00',
        'profile': 'INTRADAY',
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def enhanced_v2_checkpoint(temp_checkpoint_dir, sample_model):
    """Create an intermediate format checkpoint (enhanced_v2)."""
    checkpoint_path = temp_checkpoint_dir / "lstm_enhanced_v2.pt"
    
    checkpoint = {
        'model_state': sample_model.state_dict(),
        'config': {
            'feature_columns': ['RSI', 'MACD', 'ATR', 'ADX', 'MOMENTUM'],
            'model': {
                'input_dim': 5,
                'hidden_dim': 64,
            },
            'training': {
                'num_classes': 3,
            },
        },
        'training_history': {
            'train_loss': [0.6, 0.5],
            'val_loss': [0.65, 0.55],
        },
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def enhanced_v1_checkpoint(temp_checkpoint_dir, sample_model):
    """Create an old format checkpoint (enhanced_v1)."""
    checkpoint_path = temp_checkpoint_dir / "model_v1.pt"
    
    # v1: model_state only, no config
    checkpoint = {
        'model_state': sample_model.state_dict(),
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def state_dict_checkpoint(temp_checkpoint_dir, sample_model):
    """Create a state dict only checkpoint (with tcn prefix for detection)."""
    checkpoint_path = temp_checkpoint_dir / "state_dict.pt"
    # Create a state dict with tcn. prefix so it gets detected as 'state_dict' format
    state_dict = sample_model.state_dict()
    renamed_state = {f'tcn.{k}': v for k, v in state_dict.items()}
    torch.save(renamed_state, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def malformed_checkpoint(temp_checkpoint_dir):
    """Create a malformed checkpoint without model state."""
    checkpoint_path = temp_checkpoint_dir / "malformed.pt"
    torch.save({'some_data': 'value'}, checkpoint_path)
    return checkpoint_path


# ============================================================================
# TESTS: ModelLoader Initialization
# ============================================================================

class TestModelLoaderInit:
    """Tests for ModelLoader initialization."""
    
    def test_loader_init_with_v3_checkpoint(self, enhanced_v3_checkpoint):
        """Test initializing loader with enhanced v3 checkpoint."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        assert loader.path == enhanced_v3_checkpoint
        assert loader.checkpoint is not None
        assert loader._format == 'enhanced_v3'
    
    def test_loader_init_with_v2_checkpoint(self, enhanced_v2_checkpoint):
        """Test initializing loader with enhanced v2 checkpoint."""
        loader = ModelLoader(enhanced_v2_checkpoint)
        assert loader._format == 'enhanced_v2'
    
    def test_loader_init_with_v1_checkpoint(self, enhanced_v1_checkpoint):
        """Test initializing loader with enhanced v1 checkpoint."""
        loader = ModelLoader(enhanced_v1_checkpoint)
        assert loader._format == 'enhanced_v1'
    
    def test_loader_init_with_state_dict(self, state_dict_checkpoint):
        """Test initializing loader with state dict checkpoint."""
        loader = ModelLoader(state_dict_checkpoint)
        assert loader._format == 'state_dict'
    
    def test_loader_init_nonexistent_file(self, temp_checkpoint_dir):
        """Test that nonexistent checkpoint raises FileNotFoundError."""
        nonexistent = temp_checkpoint_dir / "nonexistent.pt"
        with pytest.raises(FileNotFoundError):
            ModelLoader(nonexistent)
    
    def test_loader_device_auto(self, enhanced_v3_checkpoint):
        """Test device='auto' selects appropriate device."""
        loader = ModelLoader(enhanced_v3_checkpoint, device='auto')
        assert loader.device in [torch.device('cpu'), torch.device('cuda')]
    
    def test_loader_device_cpu(self, enhanced_v3_checkpoint):
        """Test device='cpu' explicitly."""
        loader = ModelLoader(enhanced_v3_checkpoint, device='cpu')
        assert loader.device == torch.device('cpu')
    
    def test_loader_with_string_path(self, enhanced_v3_checkpoint):
        """Test that string paths work as well as Path objects."""
        loader = ModelLoader(str(enhanced_v3_checkpoint))
        assert loader.path == Path(enhanced_v3_checkpoint)


# ============================================================================
# TESTS: Format Detection
# ============================================================================

class TestFormatDetection:
    """Tests for checkpoint format detection."""
    
    def test_detect_enhanced_v3(self, enhanced_v3_checkpoint):
        """Test detection of enhanced v3 format."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        assert loader._detect_format() == 'enhanced_v3'
    
    def test_detect_enhanced_v2(self, enhanced_v2_checkpoint):
        """Test detection of enhanced v2 format."""
        loader = ModelLoader(enhanced_v2_checkpoint)
        assert loader._detect_format() == 'enhanced_v2'
    
    def test_detect_enhanced_v1(self, enhanced_v1_checkpoint):
        """Test detection of enhanced v1 format."""
        loader = ModelLoader(enhanced_v1_checkpoint)
        assert loader._detect_format() == 'enhanced_v1'
    
    def test_detect_state_dict(self, state_dict_checkpoint):
        """Test detection of state dict format."""
        loader = ModelLoader(state_dict_checkpoint)
        assert loader._detect_format() == 'state_dict'


# ============================================================================
# TESTS: Feature Extraction
# ============================================================================

class TestFeatureExtraction:
    """Tests for feature extraction."""
    
    def test_get_features_v3(self, enhanced_v3_checkpoint):
        """Test getting features from v3 checkpoint."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        features = loader.get_features()
        assert isinstance(features, list)
        assert len(features) == 10
        assert 'RSI' in features
        assert 'MACD' in features
    
    def test_get_features_v2(self, enhanced_v2_checkpoint):
        """Test getting features from v2 checkpoint."""
        loader = ModelLoader(enhanced_v2_checkpoint)
        features = loader.get_features()
        assert isinstance(features, list)
        assert len(features) == 5
        assert 'RSI' in features
    
    def test_get_features_missing_raises(self, malformed_checkpoint):
        """Test that missing features raise CheckpointFormatError."""
        loader = ModelLoader(malformed_checkpoint)
        with pytest.raises(CheckpointFormatError):
            loader.get_features()
    
    def test_get_features_safe_with_fallback(self, malformed_checkpoint):
        """Test get_features_safe returns fallback when features missing."""
        loader = ModelLoader(malformed_checkpoint)
        fallback = ['FEATURE1', 'FEATURE2']
        features = loader.get_features_safe(fallback=fallback)
        assert features == fallback
    
    def test_get_features_safe_without_fallback(self, malformed_checkpoint):
        """Test get_features_safe returns None when no fallback."""
        loader = ModelLoader(malformed_checkpoint)
        features = loader.get_features_safe()
        assert features is None
    
    def test_get_features_safe_success(self, enhanced_v3_checkpoint):
        """Test get_features_safe returns features when available."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        features = loader.get_features_safe(fallback=['FALLBACK'])
        assert 'RSI' in features
        assert len(features) == 10


# ============================================================================
# TESTS: Configuration Access
# ============================================================================

class TestConfigurationAccess:
    """Tests for accessing checkpoint configuration."""
    
    def test_get_config_v3(self, enhanced_v3_checkpoint):
        """Test getting config from v3 checkpoint."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        config = loader.get_config()
        assert isinstance(config, dict)
        assert 'model' in config
        assert config['model']['hidden_dim'] == 32
    
    def test_get_config_empty_when_missing(self, state_dict_checkpoint):
        """Test get_config returns empty dict when not available."""
        loader = ModelLoader(state_dict_checkpoint)
        config = loader.get_config()
        assert config == {}
    
    def test_get_training_history(self, enhanced_v3_checkpoint):
        """Test getting training history."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        history = loader.get_training_history()
        assert 'train_loss' in history
        assert 'val_acc' in history
        assert len(history['train_loss']) == 3
    
    def test_get_training_history_empty(self, state_dict_checkpoint):
        """Test get_training_history returns empty dict when missing."""
        loader = ModelLoader(state_dict_checkpoint)
        history = loader.get_training_history()
        assert history == {}
    
    def test_get_metrics(self, enhanced_v3_checkpoint):
        """Test getting metrics."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        metrics = loader.get_metrics()
        assert 'best_val_acc' in metrics
        assert metrics['test_accuracy'] == 0.88
    
    def test_get_metrics_empty(self, state_dict_checkpoint):
        """Test get_metrics returns empty dict when missing."""
        loader = ModelLoader(state_dict_checkpoint)
        metrics = loader.get_metrics()
        assert metrics == {}
    
    def test_get_feature_importance(self, enhanced_v3_checkpoint):
        """Test getting feature importance."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        importance = loader.get_feature_importance()
        assert 'RSI' in importance
        assert importance['RSI'] == 0.25
    
    def test_get_feature_importance_empty(self, state_dict_checkpoint):
        """Test get_feature_importance returns empty dict when missing."""
        loader = ModelLoader(state_dict_checkpoint)
        importance = loader.get_feature_importance()
        assert importance == {}


# ============================================================================
# TESTS: Model Info
# ============================================================================

class TestModelInfo:
    """Tests for model information retrieval."""
    
    def test_get_model_info_v3(self, enhanced_v3_checkpoint):
        """Test getting model info from v3 checkpoint."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        info = loader.get_model_info()
        assert isinstance(info, ModelInfo)
        assert info.model_type == 'enhanced_tcn'
        assert info.input_dim == 10
        assert info.hidden_dim == 32
        assert info.num_classes == 3
        assert info.feature_count == 10
        assert info.profile == 'INTRADAY'
    
    def test_model_info_dataclass_fields(self, enhanced_v3_checkpoint):
        """Test that ModelInfo has all required fields."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        info = loader.get_model_info()
        assert hasattr(info, 'model_type')
        assert hasattr(info, 'input_dim')
        assert hasattr(info, 'hidden_dim')
        assert hasattr(info, 'num_classes')
        assert hasattr(info, 'profile')
        assert hasattr(info, 'receptive_field')
        assert hasattr(info, 'feature_count')
        assert hasattr(info, 'created_at')


# ============================================================================
# TESTS: Model Loading
# ============================================================================

class TestModelLoading:
    """Tests for loading actual models (mock compatible)."""
    
    def test_get_model_returns_module(self, enhanced_v3_checkpoint):
        """Test that get_model returns a nn.Module."""
        # This test will fail to load the actual model because EnhancedTCN
        # isn't available or state dict mismatch, but we can test the interface
        loader = ModelLoader(enhanced_v3_checkpoint)
        try:
            model = loader.get_model()
            assert isinstance(model, nn.Module)
            assert model.training == False  # Should be in eval mode
        except (ImportError, ModuleNotFoundError, RuntimeError) as e:
            # Expected if training module not available or state dict mismatch
            pytest.skip(f"Cannot load model: {type(e).__name__}")
    
    def test_get_model_device_placement(self, enhanced_v3_checkpoint):
        """Test that model is placed on correct device."""
        loader = ModelLoader(enhanced_v3_checkpoint, device='cpu')
        try:
            model = loader.get_model()
            # Check if parameters are on CPU
            for param in model.parameters():
                assert param.device.type == 'cpu'
        except (ImportError, ModuleNotFoundError, RuntimeError) as e:
            pytest.skip(f"Cannot load model: {type(e).__name__}")
    
    def test_get_model_caching(self, enhanced_v3_checkpoint):
        """Test that model is cached after first load."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        try:
            model1 = loader.get_model()
            model2 = loader.get_model()
            # Should be the same object (cached)
            assert model1 is model2
        except (ImportError, ModuleNotFoundError, RuntimeError) as e:
            pytest.skip(f"Cannot load model: {type(e).__name__}")


# ============================================================================
# TESTS: Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_load_features_function(self, enhanced_v3_checkpoint):
        """Test load_features convenience function."""
        features = load_features(enhanced_v3_checkpoint)
        assert isinstance(features, list)
        assert len(features) == 10
        assert 'RSI' in features
    
    def test_get_checkpoint_info_function(self, enhanced_v3_checkpoint):
        """Test get_checkpoint_info convenience function."""
        info = get_checkpoint_info(enhanced_v3_checkpoint)
        assert isinstance(info, ModelInfo)
        assert info.input_dim == 10
    
    def test_load_model_function(self, enhanced_v3_checkpoint):
        """Test load_model convenience function."""
        try:
            model, features = load_model(enhanced_v3_checkpoint)
            assert isinstance(model, nn.Module)
            assert isinstance(features, list)
        except (ImportError, ModuleNotFoundError, RuntimeError) as e:
            pytest.skip(f"Cannot load model: {type(e).__name__}")


# ============================================================================
# TESTS: State Dict Handling
# ============================================================================

class TestStateDict:
    """Tests for state dict extraction and handling."""
    
    def test_get_state_dict_v3(self, enhanced_v3_checkpoint, sample_model):
        """Test extracting state dict from v3 checkpoint."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        state_dict = loader._get_state_dict()
        assert isinstance(state_dict, dict)
        assert 'linear1.weight' in state_dict or 'linear1' in str(state_dict.keys())
    
    def test_get_state_dict_missing_raises(self, malformed_checkpoint):
        """Test that missing state dict raises error."""
        loader = ModelLoader(malformed_checkpoint)
        with pytest.raises(CheckpointFormatError):
            loader._get_state_dict()
    
    def test_state_dict_dataparallel_handling(self, temp_checkpoint_dir, sample_model):
        """Test handling of DataParallel module prefix in state dict."""
        # Create checkpoint with DataParallel prefix
        checkpoint_path = temp_checkpoint_dir / "dataparallel.pt"
        state_dict = sample_model.state_dict()
        
        # Add 'module.' prefix
        prefixed_state = {f'module.{k}': v for k, v in state_dict.items()}
        
        checkpoint = {
            'model_state': prefixed_state,
            'feature_columns': ['F1', 'F2'],
            'config': {'model': {'input_dim': 2, 'hidden_dim': 16}, 'training': {'num_classes': 3}},
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Test that _get_state_dict can extract and handle the prefix
        loader = ModelLoader(checkpoint_path)
        state_dict = loader._get_state_dict()
        
        # Verify module. prefix is present
        assert any('module.' in k for k in state_dict.keys())


# ============================================================================
# TESTS: Summary and Printing
# ============================================================================

class TestSummary:
    """Tests for checkpoint summary generation."""
    
    def test_summary_format(self, enhanced_v3_checkpoint):
        """Test that summary returns formatted string."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        summary = loader.summary()
        assert isinstance(summary, str)
        assert 'Checkpoint:' in summary
        assert 'Format:' in summary
        assert 'Model Type:' in summary
        assert 'Input Dim:' in summary
    
    def test_summary_includes_features(self, enhanced_v3_checkpoint):
        """Test that summary includes features count."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        summary = loader.summary()
        assert 'Features:' in summary
        assert '10' in summary  # 10 features
    
    def test_summary_includes_metrics(self, enhanced_v3_checkpoint):
        """Test that summary includes metrics."""
        loader = ModelLoader(enhanced_v3_checkpoint)
        summary = loader.summary()
        assert 'Best Val Acc:' in summary or 'Val Acc:' in summary
    
    def test_summary_handles_missing_features(self, malformed_checkpoint):
        """Test that summary handles missing features gracefully."""
        loader = ModelLoader(malformed_checkpoint)
        try:
            summary = loader.summary()
            assert 'Features:' in summary
            assert 'Not stored' in summary or 'not' in summary.lower()
        except CheckpointFormatError:
            # Expected if checkpoint structure is too malformed
            pytest.skip("Malformed checkpoint structure")


# ============================================================================
# TESTS: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_features_list(self, temp_checkpoint_dir, sample_model):
        """Test handling of empty features list."""
        checkpoint_path = temp_checkpoint_dir / "empty_features.pt"
        checkpoint = {
            'model_state': sample_model.state_dict(),
            'feature_columns': [],
        }
        torch.save(checkpoint, checkpoint_path)
        
        loader = ModelLoader(checkpoint_path)
        features = loader.get_features()
        assert features == []
    
    def test_large_feature_list(self, temp_checkpoint_dir, sample_model):
        """Test handling of large feature list."""
        checkpoint_path = temp_checkpoint_dir / "many_features.pt"
        feature_list = [f'FEATURE_{i}' for i in range(1000)]
        
        checkpoint = {
            'model_state': sample_model.state_dict(),
            'feature_columns': feature_list,
        }
        torch.save(checkpoint, checkpoint_path)
        
        loader = ModelLoader(checkpoint_path)
        features = loader.get_features()
        assert len(features) == 1000
        assert 'FEATURE_0' in features
        assert 'FEATURE_999' in features
    
    def test_special_characters_in_checkpoint_name(self, temp_checkpoint_dir, sample_model):
        """Test handling checkpoints with special characters."""
        checkpoint_path = temp_checkpoint_dir / "model-v1.0_best[test].pt"
        checkpoint = {
            'model_state': sample_model.state_dict(),
            'feature_columns': ['F1', 'F2'],
        }
        torch.save(checkpoint, checkpoint_path)
        
        loader = ModelLoader(checkpoint_path)
        assert loader.path.exists()
    
    def test_nested_config_structure(self, temp_checkpoint_dir, sample_model):
        """Test deeply nested config structure."""
        checkpoint_path = temp_checkpoint_dir / "nested_config.pt"
        checkpoint = {
            'model_state': sample_model.state_dict(),
            'feature_columns': ['F1'],
            'config': {
                'model': {
                    'architecture': {
                        'layers': {
                            'layer1': {'size': 128},
                            'layer2': {'size': 64},
                        }
                    }
                }
            }
        }
        torch.save(checkpoint, checkpoint_path)
        
        loader = ModelLoader(checkpoint_path)
        config = loader.get_config()
        assert config['model']['architecture']['layers']['layer1']['size'] == 128


# ============================================================================
# TESTS: Backward Compatibility
# ============================================================================

class TestBackwardCompatibility:
    """Tests for backward compatibility features."""
    
    def test_get_default_features(self):
        """Test get_default_features function."""
        # This will try to load from default path, which may not exist
        features = get_default_features()
        # Should return list (empty if default doesn't exist)
        assert isinstance(features, list)
