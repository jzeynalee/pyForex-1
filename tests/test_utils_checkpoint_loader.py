# tests/test_utils_checkpoint_loader.py
"""
Unit tests for utils.checkpoint_loader module.
Tests checkpoint loading utilities for TCN models after LSTM removal.
"""
import pytest
import torch
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
def enhanced_v3_checkpoint():
    """Create enhanced v3 format checkpoint (latest)."""
    return {
        'model_state': {
            'tcn.network.0.conv1.conv.weight': torch.randn(64, 5, 3),
            'tcn.network.0.conv1.conv.bias': torch.randn(64),
        },
        'feature_columns': ['open', 'high', 'low', 'close', 'volume'],
        'config': {
            'model': {
                'input_dim': 5,
                'hidden_dim': 64,
                'num_layers': 5,
                'receptive_field': 63,
            },
            'training': {
                'num_classes': 3,
                'epochs': 50,
                'batch_size': 64,
            }
        },
        'metrics': {
            'best_val_acc': 0.75,
            'test_accuracy': 0.73,
        },
        'profile': 'INTRADAY',
        'created_at': '2024-01-01T12:00:00',
    }


@pytest.fixture
def state_dict_checkpoint():
    """Create direct state dict checkpoint (legacy)."""
    return {
        'tcn.network.0.conv1.conv.weight': torch.randn(64, 5, 3),
        'tcn.network.0.conv1.conv.bias': torch.randn(64),
        'classifier.0.weight': torch.randn(64, 64),
    }


@pytest.mark.unit
class TestModelLoader:
    """Test suite for ModelLoader class."""

    def test_init_file_not_found(self):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            ModelLoader("non_existent_file.pt")

    def test_init_with_enhanced_v3_checkpoint(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test initialization with enhanced v3 checkpoint."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path, device='cpu')

        assert loader.path == tmp_checkpoint_path
        assert loader.device.type == 'cpu'
        assert loader._format == 'enhanced_v3'

    def test_init_with_state_dict_checkpoint(self, tmp_checkpoint_path, state_dict_checkpoint):
        """Test initialization with state dict checkpoint."""
        torch.save(state_dict_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)

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

    def test_detect_format_enhanced_v3(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test format detection for enhanced v3."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)
        assert loader._format == 'enhanced_v3'

    def test_detect_format_state_dict(self, tmp_checkpoint_path, state_dict_checkpoint):
        """Test format detection for state dict."""
        torch.save(state_dict_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)
        assert loader._format == 'state_dict'

    def test_get_features_enhanced_v3(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test getting features from enhanced v3 checkpoint."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)
        features = loader.get_features()

        assert features == ['open', 'high', 'low', 'close', 'volume']

    def test_get_features_missing(self, tmp_checkpoint_path, state_dict_checkpoint):
        """Test getting features when not available."""
        torch.save(state_dict_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)

        with pytest.raises(CheckpointFormatError, match="doesn't contain feature columns"):
            loader.get_features()

    def test_get_features_safe_with_features(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test safe feature retrieval when features exist."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)
        features = loader.get_features_safe()

        assert features == ['open', 'high', 'low', 'close', 'volume']

    def test_get_features_safe_with_fallback(self, tmp_checkpoint_path, state_dict_checkpoint):
        """Test safe feature retrieval with fallback."""
        torch.save(state_dict_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)
        fallback = ['feature1', 'feature2']
        features = loader.get_features_safe(fallback=fallback)

        assert features == fallback

    def test_get_config(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test getting configuration."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)
        config = loader.get_config()

        assert 'model' in config
        assert 'training' in config
        assert config['model']['input_dim'] == 5

    def test_get_config_empty(self, tmp_checkpoint_path, state_dict_checkpoint):
        """Test getting config when not available."""
        torch.save(state_dict_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)
        config = loader.get_config()

        assert config == {}

    def test_get_metrics(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test getting metrics."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)
        metrics = loader.get_metrics()

        assert 'best_val_acc' in metrics
        assert metrics['best_val_acc'] == 0.75

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

    def test_get_model_info(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test getting model info."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)
        info = loader.get_model_info()

        assert isinstance(info, ModelInfo)
        assert info.model_type == 'enhanced_tcn'
        assert info.input_dim == 5
        assert info.hidden_dim == 64
        assert info.num_classes == 3
        assert info.profile == 'INTRADAY'
        assert info.receptive_field == 63

    def test_get_model_tcn(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test getting TCN model."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        with patch('training.train_tcn_enhanced.EnhancedTCN') as mock_tcn:
            mock_model = MagicMock()
            mock_model.load_state_dict = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_model.eval = MagicMock()
            mock_tcn.return_value = mock_model

            loader = ModelLoader(tmp_checkpoint_path)
            model = loader.get_model()

            mock_tcn.assert_called_once()
            mock_model.load_state_dict.assert_called_once()
            mock_model.eval.assert_called_once()

    def test_get_model_caching(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test that get_model caches the model."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        with patch('training.train_tcn_enhanced.EnhancedTCN') as mock_tcn:
            mock_model = MagicMock()
            mock_model.load_state_dict = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_model.eval = MagicMock()
            mock_tcn.return_value = mock_model

            loader = ModelLoader(tmp_checkpoint_path)

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

    def test_get_state_dict_from_enhanced(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test extracting state dict from enhanced checkpoint."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)
        state_dict = loader._get_state_dict()

        assert 'tcn.network.0.conv1.conv.weight' in state_dict

    def test_get_state_dict_from_direct(self, tmp_checkpoint_path, state_dict_checkpoint):
        """Test extracting state dict from direct checkpoint."""
        torch.save(state_dict_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)
        state_dict = loader._get_state_dict()

        assert 'tcn.network.0.conv1.conv.weight' in state_dict

    def test_summary(self, tmp_checkpoint_path, enhanced_v3_checkpoint):
        """Test generating summary."""
        torch.save(enhanced_v3_checkpoint, tmp_checkpoint_path)

        loader = ModelLoader(tmp_checkpoint_path)
        summary = loader.summary()

        assert "Checkpoint:" in summary
        assert "enhanced_v3" in summary
        assert "INTRADAY" in summary
        assert "75" in summary  # accuracy


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
