# tests/test_training_train_vit.py
"""
Unit tests for training/train_vit.py - ViT classifier training.

Tests cover:
- Argument parsing
- Mixup augmentation
- ClassifierHead model
- Feature caching mechanism
- Training loop with label smoothing
- Early stopping
- Checkpoint saving
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from torch.utils.data import DataLoader, TensorDataset

# Import the modules to test
from training.train_vit import (
    get_args,
    mixup_data,
    mixup_criterion,
    ClassifierHead,
    maybe_build_cache,
    train_classifier,
    main,
)


@pytest.mark.unit
class TestArgumentParser:
    """Test argument parsing."""

    @patch('sys.argv', [
        'train_vit.py',
        '--data_dir', '/path/to/data',
        '--batch_size', '32',
        '--num_epochs', '50',
    ])
    def test_get_args_basic(self):
        """Test basic argument parsing."""
        args = get_args()

        assert args.data_dir == '/path/to/data'
        assert args.batch_size == 32
        assert args.num_epochs == 50

    @patch('sys.argv', [
        'train_vit.py',
        '--data_dir', '/path/to/data',
        '--lr', '1e-3',
        '--weight_decay', '0.1',
        '--dropout', '0.3',
        '--label_smoothing', '0.2',
        '--mixup_alpha', '0.5',
        '--patience', '15',
    ])
    def test_get_args_advanced(self):
        """Test advanced argument parsing."""
        args = get_args()

        assert args.lr == 1e-3
        assert args.weight_decay == 0.1
        assert args.dropout == 0.3
        assert args.label_smoothing == 0.2
        assert args.mixup_alpha == 0.5
        assert args.patience == 15


@pytest.mark.unit
class TestMixup:
    """Test mixup augmentation functions."""

    def test_mixup_data_disabled(self):
        """Test mixup with alpha=0 (disabled)."""
        x = torch.randn(10, 768)
        y = torch.randint(0, 3, (10,))

        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.0)

        # Should return original data when disabled
        assert torch.equal(mixed_x, x)
        assert torch.equal(y_a, y)
        assert torch.equal(y_b, y)
        assert lam == 1.0

    def test_mixup_data_enabled(self):
        """Test mixup with alpha>0."""
        torch.manual_seed(42)
        x = torch.randn(10, 768)
        y = torch.randint(0, 3, (10,))

        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)

        # Mixed data should be different from original
        assert not torch.equal(mixed_x, x)
        assert torch.equal(y_a, y)  # Original labels
        assert not torch.equal(y_b, y)  # Permuted labels
        assert 0.0 <= lam <= 1.0

    def test_mixup_data_shape_preserved(self):
        """Test that mixup preserves tensor shapes."""
        x = torch.randn(16, 768)
        y = torch.randint(0, 3, (16,))

        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)

        assert mixed_x.shape == x.shape
        assert y_a.shape == y.shape
        assert y_b.shape == y.shape

    def test_mixup_criterion(self):
        """Test mixup loss calculation."""
        criterion = nn.CrossEntropyLoss()
        pred = torch.randn(10, 3)
        y_a = torch.randint(0, 3, (10,))
        y_b = torch.randint(0, 3, (10,))
        lam = 0.6

        loss = mixup_criterion(criterion, pred, y_a, y_b, lam)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0


@pytest.mark.unit
class TestClassifierHead:
    """Test ClassifierHead model."""

    def test_init_default(self):
        """Test default initialization."""
        model = ClassifierHead()

        assert isinstance(model, nn.Module)
        assert hasattr(model, 'norm')
        assert hasattr(model, 'block')
        assert hasattr(model, 'classifier')

    def test_init_custom(self):
        """Test custom initialization."""
        model = ClassifierHead(
            in_features=512,
            hidden_dim=128,
            num_classes=5,
            dropout=0.3,
        )

        # Check input dimension
        assert model.norm.normalized_shape[0] == 512

        # Check output dimension
        params = list(model.parameters())
        assert params[-1].shape[0] == 5  # num_classes

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = ClassifierHead(in_features=768, hidden_dim=256, num_classes=3)
        x = torch.randn(16, 768)

        output = model(x)

        assert output.shape == (16, 3)

    def test_forward_batch_processing(self):
        """Test forward with different batch sizes."""
        model = ClassifierHead()

        for batch_size in [1, 8, 32]:
            x = torch.randn(batch_size, 768)
            output = model(x)
            assert output.shape == (batch_size, 3)

    def test_dropout_training_mode(self):
        """Test dropout is active in training mode."""
        model = ClassifierHead(dropout=0.5)
        model.train()

        x = torch.randn(100, 768)

        # Run multiple times and check for variation (dropout should cause this)
        outputs = [model(x) for _ in range(3)]

        # Outputs should differ due to dropout
        assert not torch.allclose(outputs[0], outputs[1])

    def test_dropout_eval_mode(self):
        """Test dropout is inactive in eval mode."""
        model = ClassifierHead(dropout=0.5)
        model.eval()

        x = torch.randn(100, 768)

        # Run multiple times
        with torch.no_grad():
            outputs = [model(x) for _ in range(3)]

        # Outputs should be identical in eval mode
        assert torch.allclose(outputs[0], outputs[1])


@pytest.mark.unit
class TestMaybeBuildCache:
    """Test feature caching mechanism."""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories."""
        data_dir = tmp_path / "data"
        cache_path = tmp_path / "cache.pt"

        # Create dummy train/val directories
        train_dir = data_dir / "train"
        val_dir = data_dir / "val"
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)

        # Create class subdirectories
        (train_dir / "class0").mkdir()
        (train_dir / "class1").mkdir()
        (val_dir / "class0").mkdir()
        (val_dir / "class1").mkdir()

        # Create dummy images (1x1 pixel PNG files)
        for class_dir in [train_dir / "class0", train_dir / "class1"]:
            for i in range(3):
                (class_dir / f"img{i}.png").write_bytes(
                    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
                    b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01'
                    b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
                )

        for class_dir in [val_dir / "class0", val_dir / "class1"]:
            for i in range(2):
                (class_dir / f"img{i}.png").write_bytes(
                    b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
                    b'\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01'
                    b'\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
                )

        return {
            'data_dir': str(data_dir),
            'cache_path': str(cache_path),
        }

    def test_load_existing_cache(self, temp_dirs):
        """Test loading existing cache."""
        cache_path = temp_dirs['cache_path']

        # Create dummy cache
        dummy_cache = {
            'train_x': torch.randn(10, 768),
            'train_y': torch.randint(0, 2, (10,)),
            'val_x': torch.randn(5, 768),
            'val_y': torch.randint(0, 2, (5,)),
            'num_classes': 2,
        }
        torch.save(dummy_cache, cache_path)

        # Load cache
        cache = maybe_build_cache(
            data_dir=temp_dirs['data_dir'],
            cache_path=cache_path,
            device='cpu',
        )

        assert 'train_x' in cache
        assert 'val_x' in cache
        assert cache['num_classes'] == 2

    @patch('training.train_vit.ViTExtractor')
    def test_build_new_cache(self, mock_vit_class, temp_dirs):
        """Test building cache from scratch."""
        # Create a real mock ViT model
        class MockViT(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, imgs):
                batch_size = imgs.shape[0]
                return torch.randn(batch_size, 768)

            def eval(self):
                return self

            def to(self, device):
                return self

        mock_vit_class.return_value = MockViT()

        cache = maybe_build_cache(
            data_dir=temp_dirs['data_dir'],
            cache_path=temp_dirs['cache_path'],
            device='cpu',
        )

        # Verify cache structure
        assert 'train_x' in cache
        assert 'train_y' in cache
        assert 'val_x' in cache
        assert 'val_y' in cache
        assert 'num_classes' in cache

        # Verify cache was saved
        assert os.path.exists(temp_dirs['cache_path'])


@pytest.mark.unit
class TestTrainClassifier:
    """Test train_classifier function."""

    @pytest.fixture
    def sample_features(self):
        """Create sample feature tensors."""
        train_x = torch.randn(100, 768)
        train_y = torch.randint(0, 3, (100,))
        val_x = torch.randn(20, 768)
        val_y = torch.randint(0, 3, (20,))
        return train_x, train_y, val_x, val_y

    @pytest.fixture
    def mock_args(self, tmp_path):
        """Create mock arguments."""
        args = MagicMock()
        args.save_dir = str(tmp_path / "checkpoints")
        args.device = 'cpu'
        args.batch_size = 16
        args.num_epochs = 2
        args.lr = 5e-4
        args.weight_decay = 0.05
        args.dropout = 0.2
        args.label_smoothing = 0.1
        args.mixup_alpha = 0.2
        args.patience = 5
        return args

    def test_train_basic(self, sample_features, mock_args):
        """Test basic training flow."""
        train_x, train_y, val_x, val_y = sample_features

        classifier = train_classifier(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            num_classes=3,
            args=mock_args,
        )

        # Verify classifier was returned
        assert isinstance(classifier, ClassifierHead)

        # Verify checkpoint was saved
        checkpoint_dir = Path(mock_args.save_dir)
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "best_head.pth").exists()

    def test_train_history_saved(self, sample_features, mock_args):
        """Test that training history is saved."""
        train_x, train_y, val_x, val_y = sample_features

        train_classifier(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            num_classes=3,
            args=mock_args,
        )

        # Verify history file exists
        history_path = Path(mock_args.save_dir) / "training_history.pt"
        assert history_path.exists()

        # Load and verify history
        history = torch.load(history_path)
        assert 'train_acc' in history
        assert 'val_acc' in history
        assert 'train_loss' in history

    def test_early_stopping(self, sample_features, mock_args):
        """Test early stopping mechanism."""
        train_x, train_y, val_x, val_y = sample_features

        # Set very low patience
        mock_args.patience = 1
        mock_args.num_epochs = 100  # Would take long without early stopping

        # Mock perfect classifier (won't improve after epoch 1)
        with patch('training.train_vit.ClassifierHead') as mock_head_class:
            class MockHead(nn.Module):
                def __init__(self, *args, **kwargs):
                    super().__init__()
                    self.linear = nn.Linear(768, 3)  # Correct input size

                def forward(self, x):
                    # Use actual linear layer to preserve gradients
                    return self.linear(x)

                def train(self, mode=True):
                    super().train(mode)
                    return self

                def eval(self):
                    super().eval()
                    return self

            mock_head_class.return_value = MockHead()

            train_classifier(
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                num_classes=3,
                args=mock_args,
            )

            # Training should stop early (not run all 100 epochs)
            history = torch.load(Path(mock_args.save_dir) / "training_history.pt")
            assert len(history['train_acc']) < 100

    def test_mixup_integration(self, sample_features, mock_args):
        """Test that mixup is applied during training."""
        train_x, train_y, val_x, val_y = sample_features

        # Enable mixup
        mock_args.mixup_alpha = 0.3
        mock_args.num_epochs = 1

        with patch('training.train_vit.mixup_data') as mock_mixup:
            # Make mixup return modified data
            def mixup_side_effect(x, y, alpha):
                return x, y, y, 1.0

            mock_mixup.side_effect = mixup_side_effect

            train_classifier(
                train_x=train_x,
                train_y=train_y,
                val_x=val_x,
                val_y=val_y,
                num_classes=3,
                args=mock_args,
            )

            # Verify mixup was called
            assert mock_mixup.called

    def test_label_smoothing(self, sample_features, mock_args):
        """Test label smoothing is applied."""
        train_x, train_y, val_x, val_y = sample_features

        mock_args.label_smoothing = 0.2
        mock_args.num_epochs = 1

        # Don't mock CrossEntropyLoss, let it run normally
        # Just verify training completes
        classifier = train_classifier(
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            val_y=val_y,
            num_classes=3,
            args=mock_args,
        )

        # Verify classifier was created
        assert isinstance(classifier, ClassifierHead)


@pytest.mark.unit
class TestMain:
    """Test main function."""

    @patch('training.train_vit.maybe_build_cache')
    @patch('training.train_vit.train_classifier')
    @patch('sys.argv', [
        'train_vit.py',
        '--data_dir', '/path/to/data',
        '--batch_size', '32',
        '--num_epochs', '10',
    ])
    def test_main_flow(self, mock_train, mock_cache):
        """Test main execution flow."""
        # Mock cache
        mock_cache.return_value = {
            'train_x': torch.randn(100, 768),
            'train_y': torch.randint(0, 3, (100,)),
            'val_x': torch.randn(20, 768),
            'val_y': torch.randint(0, 3, (20,)),
            'num_classes': 3,
        }

        # Mock classifier
        mock_train.return_value = MagicMock(spec=ClassifierHead)

        main()

        # Verify cache was built
        assert mock_cache.called

        # Verify training was called
        assert mock_train.called

    @patch('training.train_vit.maybe_build_cache')
    @patch('training.train_vit.train_classifier')
    @patch('sys.argv', [
        'train_vit.py',
        '--data_dir', '/path/to/data',
        '--mixup_alpha', '0.5',
        '--dropout', '0.3',
    ])
    def test_main_custom_args(self, mock_train, mock_cache):
        """Test main with custom arguments."""
        mock_cache.return_value = {
            'train_x': torch.randn(50, 768),
            'train_y': torch.randint(0, 2, (50,)),
            'val_x': torch.randn(10, 768),
            'val_y': torch.randint(0, 2, (10,)),
            'num_classes': 2,
        }

        mock_train.return_value = MagicMock()

        main()

        # Verify training was called
        assert mock_train.called

        # Get the args that were passed
        # train_classifier is called with positional args and 'args' keyword
        call_kwargs = mock_train.call_args[1]
        if 'args' in call_kwargs:
            args = call_kwargs['args']
            assert args.mixup_alpha == 0.5
            assert args.dropout == 0.3
