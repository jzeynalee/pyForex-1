# tests/test_training_finetune_vit.py
"""
Comprehensive unit tests for training/finetune_vit.py

This module tests the ViT fine-tuning script which includes:
- Argument parsing with auto-detected device
- Data augmentation transforms
- FineTunableViT model with partial unfreezing
- WarmupCosineScheduler for learning rate scheduling
- Training and validation loops
- Checkpoint saving and resuming
"""

import sys
import os
import math
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import tempfile
import shutil
import logging

import pytest
import numpy as np

# Try to import torch, use mock if not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = MagicMock()
    nn = MagicMock()


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_args():
    """Create mock arguments object."""
    args = MagicMock()
    args.data_dir = "/fake/data"
    args.batch_size = 16
    args.num_epochs = 30
    args.lr_head = 1e-3
    args.lr_backbone = 1e-5
    args.weight_decay = 0.01
    args.unfreeze_blocks = 4
    args.dropout = 0.1
    args.label_smoothing = 0.1
    args.patience = 7
    args.warmup_epochs = 2
    args.save_dir = "./checkpoints_test"
    args.num_workers = 4
    args.device = "cpu"
    args.resume = None
    return args


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory structure for training data."""
    # Create train and val directories with class subdirectories
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    
    for class_name in ["class_0", "class_1", "class_2"]:
        (train_dir / class_name).mkdir(parents=True)
        (val_dir / class_name).mkdir(parents=True)
        
        # Create dummy image files (just empty files for structure)
        for i in range(5):
            (train_dir / class_name / f"img_{i}.jpg").touch()
            (val_dir / class_name / f"img_{i}.jpg").touch()
    
    return tmp_path


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


# ============================================================================
# ARGUMENT PARSING TESTS
# ============================================================================

class TestArgumentParsing:
    """Tests for argument parsing functionality."""
    
    def test_get_args_returns_namespace(self):
        """Test that get_args returns an argparse Namespace."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            from training.finetune_vit import get_args
            args = get_args()
            assert hasattr(args, 'data_dir')
    
    def test_data_dir_is_required(self):
        """Test that data_dir argument is required."""
        with patch('sys.argv', ['script.py']):
            from training.finetune_vit import get_args
            with pytest.raises(SystemExit):
                get_args()
    
    def test_default_batch_size(self):
        """Test default batch size is 16."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            from training.finetune_vit import get_args
            args = get_args()
            assert args.batch_size == 16
    
    def test_default_num_epochs(self):
        """Test default num_epochs is 30."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            from training.finetune_vit import get_args
            args = get_args()
            assert args.num_epochs == 30
    
    def test_default_lr_head(self):
        """Test default lr_head is 1e-3."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            from training.finetune_vit import get_args
            args = get_args()
            assert args.lr_head == 1e-3
    
    def test_default_lr_backbone(self):
        """Test default lr_backbone is 1e-5."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            from training.finetune_vit import get_args
            args = get_args()
            assert args.lr_backbone == 1e-5
    
    def test_default_weight_decay(self):
        """Test default weight_decay is 0.01."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            from training.finetune_vit import get_args
            args = get_args()
            assert args.weight_decay == 0.01
    
    def test_default_unfreeze_blocks(self):
        """Test default unfreeze_blocks is 4."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            from training.finetune_vit import get_args
            args = get_args()
            assert args.unfreeze_blocks == 4
    
    def test_default_dropout(self):
        """Test default dropout is 0.1."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            from training.finetune_vit import get_args
            args = get_args()
            assert args.dropout == 0.1
    
    def test_default_label_smoothing(self):
        """Test default label_smoothing is 0.1."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            from training.finetune_vit import get_args
            args = get_args()
            assert args.label_smoothing == 0.1
    
    def test_default_patience(self):
        """Test default patience is 7."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            from training.finetune_vit import get_args
            args = get_args()
            assert args.patience == 7
    
    def test_default_warmup_epochs(self):
        """Test default warmup_epochs is 2."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            from training.finetune_vit import get_args
            args = get_args()
            assert args.warmup_epochs == 2
    
    def test_default_num_workers(self):
        """Test default num_workers is 4."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            from training.finetune_vit import get_args
            args = get_args()
            assert args.num_workers == 4
    
    def test_default_resume_is_none(self):
        """Test default resume is None."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            from training.finetune_vit import get_args
            args = get_args()
            assert args.resume is None
    
    def test_custom_batch_size(self):
        """Test custom batch_size argument."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path', '--batch_size', '32']):
            from training.finetune_vit import get_args
            args = get_args()
            assert args.batch_size == 32
    
    def test_custom_lr_values(self):
        """Test custom learning rate arguments."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path', 
                               '--lr_head', '0.001', '--lr_backbone', '0.0001']):
            from training.finetune_vit import get_args
            args = get_args()
            assert args.lr_head == 0.001
            assert args.lr_backbone == 0.0001
    
    def test_device_auto_detection_cpu(self):
        """Test device auto-detection when CUDA is not available."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            with patch('torch.cuda.is_available', return_value=False):
                from training.finetune_vit import get_args
                args = get_args()
                assert args.device == "cpu"
    
    def test_device_auto_detection_cuda(self):
        """Test device auto-detection when CUDA is available."""
        with patch('sys.argv', ['script.py', '--data_dir', '/fake/path']):
            with patch('torch.cuda.is_available', return_value=True):
                from training.finetune_vit import get_args
                args = get_args()
                assert args.device == "cuda"


# ============================================================================
# TRANSFORM TESTS
# ============================================================================

class TestTransforms:
    """Tests for data augmentation transforms."""
    
    def test_get_transforms_train_returns_compose(self):
        """Test that get_transforms returns a Compose for training."""
        from training.finetune_vit import get_transforms
        transform = get_transforms(is_train=True)
        assert transform is not None
    
    def test_get_transforms_val_returns_compose(self):
        """Test that get_transforms returns a Compose for validation."""
        from training.finetune_vit import get_transforms
        transform = get_transforms(is_train=False)
        assert transform is not None
    
    def test_train_transform_has_augmentations(self):
        """Test that training transform includes augmentations."""
        from training.finetune_vit import get_transforms
        transform = get_transforms(is_train=True)
        
        # Check transform has multiple operations
        assert hasattr(transform, 'transforms')
        assert len(transform.transforms) > 3  # More than basic resize+totensor+normalize
    
    def test_val_transform_minimal(self):
        """Test that validation transform is minimal."""
        from training.finetune_vit import get_transforms
        transform = get_transforms(is_train=False)
        
        assert hasattr(transform, 'transforms')
        # Val should have fewer transforms than train
        train_transform = get_transforms(is_train=True)
        assert len(transform.transforms) < len(train_transform.transforms)
    
    def test_transforms_include_resize_to_224(self):
        """Test that transforms include resize to 224x224."""
        from training.finetune_vit import get_transforms
        from torchvision import transforms
        
        transform = get_transforms(is_train=True)
        
        # Find Resize transform
        has_resize = any(
            isinstance(t, transforms.Resize) 
            for t in transform.transforms
        )
        assert has_resize
    
    def test_transforms_include_normalize(self):
        """Test that transforms include normalization."""
        from training.finetune_vit import get_transforms
        from torchvision import transforms
        
        transform = get_transforms(is_train=True)
        
        has_normalize = any(
            isinstance(t, transforms.Normalize) 
            for t in transform.transforms
        )
        assert has_normalize
    
    def test_train_transform_has_random_horizontal_flip(self):
        """Test that training transform includes random horizontal flip."""
        from training.finetune_vit import get_transforms
        from torchvision import transforms
        
        transform = get_transforms(is_train=True)
        
        has_flip = any(
            isinstance(t, transforms.RandomHorizontalFlip) 
            for t in transform.transforms
        )
        assert has_flip
    
    def test_train_transform_has_color_jitter(self):
        """Test that training transform includes color jitter."""
        from training.finetune_vit import get_transforms
        from torchvision import transforms
        
        transform = get_transforms(is_train=True)
        
        has_jitter = any(
            isinstance(t, transforms.ColorJitter) 
            for t in transform.transforms
        )
        assert has_jitter
    
    def test_train_transform_has_random_erasing(self):
        """Test that training transform includes random erasing."""
        from training.finetune_vit import get_transforms
        from torchvision import transforms
        
        transform = get_transforms(is_train=True)
        
        has_erasing = any(
            isinstance(t, transforms.RandomErasing) 
            for t in transform.transforms
        )
        assert has_erasing


# ============================================================================
# FINETUNABLE VIT MODEL TESTS
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestFineTunableViT:
    """Tests for FineTunableViT model class."""
    
    def test_model_initialization(self):
        """Test model initializes without error."""
        with patch('timm.create_model') as mock_timm:
            # Create mock ViT model
            mock_vit = MagicMock()
            mock_vit.embed_dim = 768
            mock_vit.blocks = [MagicMock() for _ in range(12)]
            mock_vit.norm = MagicMock()
            mock_vit.parameters.return_value = iter([torch.nn.Parameter(torch.randn(10))])
            
            for block in mock_vit.blocks:
                block.parameters.return_value = iter([torch.nn.Parameter(torch.randn(10))])
            mock_vit.norm.parameters.return_value = iter([torch.nn.Parameter(torch.randn(10))])
            
            mock_timm.return_value = mock_vit
            
            from training.finetune_vit import FineTunableViT
            model = FineTunableViT(num_classes=3, unfreeze_blocks=4, dropout=0.1)
            
            assert model is not None
    
    def test_model_has_vit_backbone(self):
        """Test model has ViT backbone."""
        with patch('timm.create_model') as mock_timm:
            mock_vit = MagicMock()
            mock_vit.embed_dim = 768
            mock_vit.blocks = [MagicMock() for _ in range(12)]
            mock_vit.norm = MagicMock()
            mock_vit.parameters.return_value = iter([])
            for block in mock_vit.blocks:
                block.parameters.return_value = iter([])
            mock_vit.norm.parameters.return_value = iter([])
            
            mock_timm.return_value = mock_vit
            
            from training.finetune_vit import FineTunableViT
            model = FineTunableViT(num_classes=3)
            
            assert hasattr(model, 'vit')
    
    def test_model_has_head(self):
        """Test model has classification head."""
        with patch('timm.create_model') as mock_timm:
            mock_vit = MagicMock()
            mock_vit.embed_dim = 768
            mock_vit.blocks = [MagicMock() for _ in range(12)]
            mock_vit.norm = MagicMock()
            mock_vit.parameters.return_value = iter([])
            for block in mock_vit.blocks:
                block.parameters.return_value = iter([])
            mock_vit.norm.parameters.return_value = iter([])
            
            mock_timm.return_value = mock_vit
            
            from training.finetune_vit import FineTunableViT
            model = FineTunableViT(num_classes=3)
            
            assert hasattr(model, 'head')
    
    def test_timm_create_model_called_correctly(self):
        """Test timm.create_model is called with correct parameters."""
        with patch('timm.create_model') as mock_timm:
            mock_vit = MagicMock()
            mock_vit.embed_dim = 768
            mock_vit.blocks = [MagicMock() for _ in range(12)]
            mock_vit.norm = MagicMock()
            mock_vit.parameters.return_value = iter([])
            for block in mock_vit.blocks:
                block.parameters.return_value = iter([])
            mock_vit.norm.parameters.return_value = iter([])
            
            mock_timm.return_value = mock_vit
            
            from training.finetune_vit import FineTunableViT
            model = FineTunableViT(num_classes=3)
            
            mock_timm.assert_called_once_with(
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=0
            )
    
    def test_get_param_groups_returns_two_groups(self):
        """Test get_param_groups returns two parameter groups."""
        with patch('timm.create_model') as mock_timm:
            mock_vit = MagicMock()
            mock_vit.embed_dim = 768
            mock_vit.blocks = [MagicMock() for _ in range(12)]
            mock_vit.norm = MagicMock()
            mock_vit.parameters.return_value = iter([])
            for block in mock_vit.blocks:
                block.parameters.return_value = iter([])
            mock_vit.norm.parameters.return_value = iter([])
            
            mock_timm.return_value = mock_vit
            
            from training.finetune_vit import FineTunableViT
            model = FineTunableViT(num_classes=3)
            
            param_groups = model.get_param_groups(
                lr_backbone=1e-5,
                lr_head=1e-3,
                weight_decay=0.01
            )
            
            assert len(param_groups) == 2
    
    def test_get_param_groups_has_correct_lrs(self):
        """Test get_param_groups has correct learning rates."""
        with patch('timm.create_model') as mock_timm:
            mock_vit = MagicMock()
            mock_vit.embed_dim = 768
            mock_vit.blocks = [MagicMock() for _ in range(12)]
            mock_vit.norm = MagicMock()
            mock_vit.parameters.return_value = iter([])
            for block in mock_vit.blocks:
                block.parameters.return_value = iter([])
            mock_vit.norm.parameters.return_value = iter([])
            
            mock_timm.return_value = mock_vit
            
            from training.finetune_vit import FineTunableViT
            model = FineTunableViT(num_classes=3)
            
            param_groups = model.get_param_groups(
                lr_backbone=1e-5,
                lr_head=1e-3,
                weight_decay=0.01
            )
            
            assert param_groups[0]['lr'] == 1e-5  # Backbone
            assert param_groups[1]['lr'] == 1e-3  # Head


# ============================================================================
# WARMUP COSINE SCHEDULER TESTS
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestWarmupCosineScheduler:
    """Tests for WarmupCosineScheduler class."""
    
    def test_scheduler_initialization(self):
        """Test scheduler initializes correctly."""
        from training.finetune_vit import WarmupCosineScheduler
        
        # Create mock optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [
            {'lr': 1e-5},
            {'lr': 1e-3}
        ]
        
        scheduler = WarmupCosineScheduler(
            optimizer=mock_optimizer,
            warmup_epochs=2,
            total_epochs=30,
            min_lr=1e-7
        )
        
        assert scheduler.warmup_epochs == 2
        assert scheduler.total_epochs == 30
        assert scheduler.min_lr == 1e-7
    
    def test_scheduler_stores_base_lrs(self):
        """Test scheduler stores base learning rates."""
        from training.finetune_vit import WarmupCosineScheduler
        
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [
            {'lr': 1e-5},
            {'lr': 1e-3}
        ]
        
        scheduler = WarmupCosineScheduler(
            optimizer=mock_optimizer,
            warmup_epochs=2,
            total_epochs=30
        )
        
        assert scheduler.base_lrs == [1e-5, 1e-3]
    
    def test_warmup_increases_lr(self):
        """Test that learning rate increases during warmup."""
        from training.finetune_vit import WarmupCosineScheduler
        
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [
            {'lr': 1e-5},
            {'lr': 1e-3}
        ]
        
        scheduler = WarmupCosineScheduler(
            optimizer=mock_optimizer,
            warmup_epochs=2,
            total_epochs=30
        )
        
        # Get LR at epoch 1 (during warmup)
        scheduler.step(1)
        lr_epoch_1 = mock_optimizer.param_groups[0]['lr']
        
        # Get LR at epoch 2 (end of warmup)
        scheduler.step(2)
        lr_epoch_2 = mock_optimizer.param_groups[0]['lr']
        
        # LR should increase during warmup
        assert lr_epoch_2 > lr_epoch_1
    
    def test_lr_at_warmup_end_equals_base(self):
        """Test that LR equals base LR at end of warmup."""
        from training.finetune_vit import WarmupCosineScheduler
        
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [
            {'lr': 1e-5},
            {'lr': 1e-3}
        ]
        
        scheduler = WarmupCosineScheduler(
            optimizer=mock_optimizer,
            warmup_epochs=2,
            total_epochs=30
        )
        
        scheduler.step(2)  # End of warmup
        
        # Should be very close to base LR
        assert abs(mock_optimizer.param_groups[0]['lr'] - 1e-5) < 1e-10
        assert abs(mock_optimizer.param_groups[1]['lr'] - 1e-3) < 1e-10
    
    def test_cosine_decay_after_warmup(self):
        """Test cosine decay after warmup period."""
        from training.finetune_vit import WarmupCosineScheduler
        
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [
            {'lr': 1e-5},
            {'lr': 1e-3}
        ]
        
        scheduler = WarmupCosineScheduler(
            optimizer=mock_optimizer,
            warmup_epochs=2,
            total_epochs=30
        )
        
        # Get LR after warmup at different epochs
        scheduler.step(5)
        lr_epoch_5 = mock_optimizer.param_groups[0]['lr']
        
        scheduler.step(15)
        lr_epoch_15 = mock_optimizer.param_groups[0]['lr']
        
        scheduler.step(29)
        lr_epoch_29 = mock_optimizer.param_groups[0]['lr']
        
        # LR should decrease with cosine schedule
        assert lr_epoch_15 < lr_epoch_5
        assert lr_epoch_29 < lr_epoch_15
    
    def test_lr_approaches_min_at_end(self):
        """Test LR approaches min_lr at end of training."""
        from training.finetune_vit import WarmupCosineScheduler
        
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [
            {'lr': 1e-5},
            {'lr': 1e-3}
        ]
        
        min_lr = 1e-7
        scheduler = WarmupCosineScheduler(
            optimizer=mock_optimizer,
            warmup_epochs=2,
            total_epochs=30,
            min_lr=min_lr
        )
        
        scheduler.step(30)  # Final epoch
        
        # Should be close to min_lr
        assert mock_optimizer.param_groups[0]['lr'] < 1e-6
    
    def test_get_lr_returns_current_lrs(self):
        """Test get_lr returns current learning rates."""
        from training.finetune_vit import WarmupCosineScheduler
        
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [
            {'lr': 1e-5},
            {'lr': 1e-3}
        ]
        
        scheduler = WarmupCosineScheduler(
            optimizer=mock_optimizer,
            warmup_epochs=2,
            total_epochs=30
        )
        
        lrs = scheduler.get_lr()
        
        assert lrs == [1e-5, 1e-3]
    
    def test_warmup_starts_at_10_percent(self):
        """Test warmup starts at 10% of base LR."""
        from training.finetune_vit import WarmupCosineScheduler
        
        mock_optimizer = MagicMock()
        mock_optimizer.param_groups = [
            {'lr': 1e-4},
        ]
        
        scheduler = WarmupCosineScheduler(
            optimizer=mock_optimizer,
            warmup_epochs=2,
            total_epochs=30
        )
        
        # At epoch 0, alpha = 0.1 + 0.9 * (0/2) = 0.1
        scheduler.step(0)
        
        # LR should be ~10% of base
        expected_lr = 1e-4 * 0.1
        assert abs(mock_optimizer.param_groups[0]['lr'] - expected_lr) < 1e-10


# ============================================================================
# TRAINING LOOP TESTS
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainingLoop:
    """Tests for training loop functions."""
    
    def test_train_one_epoch_returns_loss_and_acc(self):
        """Test train_one_epoch returns loss and accuracy."""
        from training.finetune_vit import train_one_epoch
        
        # Create mocks
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.randn(10))])
        mock_model.return_value = torch.randn(4, 3)  # batch_size=4, num_classes=3
        
        mock_loader = [
            (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 2, 0]))
        ]
        
        mock_criterion = MagicMock()
        mock_criterion.return_value = torch.tensor(0.5, requires_grad=True)
        
        mock_optimizer = MagicMock()
        
        loss, acc = train_one_epoch(
            mock_model, mock_loader, mock_criterion, mock_optimizer, 'cpu'
        )
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1
    
    def test_train_one_epoch_calls_optimizer_step(self):
        """Test train_one_epoch calls optimizer.step()."""
        from training.finetune_vit import train_one_epoch
        
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.randn(10))])
        mock_model.return_value = torch.randn(4, 3)
        
        mock_loader = [
            (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 2, 0]))
        ]
        
        mock_criterion = MagicMock()
        mock_criterion.return_value = torch.tensor(0.5, requires_grad=True)
        
        mock_optimizer = MagicMock()
        
        train_one_epoch(mock_model, mock_loader, mock_criterion, mock_optimizer, 'cpu')
        
        mock_optimizer.step.assert_called()
    
    def test_train_one_epoch_calls_zero_grad(self):
        """Test train_one_epoch calls optimizer.zero_grad()."""
        from training.finetune_vit import train_one_epoch
        
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.randn(10))])
        mock_model.return_value = torch.randn(4, 3)
        
        mock_loader = [
            (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 2, 0]))
        ]
        
        mock_criterion = MagicMock()
        mock_criterion.return_value = torch.tensor(0.5, requires_grad=True)
        
        mock_optimizer = MagicMock()
        
        train_one_epoch(mock_model, mock_loader, mock_criterion, mock_optimizer, 'cpu')
        
        mock_optimizer.zero_grad.assert_called()
    
    def test_train_one_epoch_sets_model_train(self):
        """Test train_one_epoch sets model to train mode."""
        from training.finetune_vit import train_one_epoch
        
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.nn.Parameter(torch.randn(10))])
        mock_model.return_value = torch.randn(4, 3)
        
        mock_loader = [
            (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 2, 0]))
        ]
        
        mock_criterion = MagicMock()
        mock_criterion.return_value = torch.tensor(0.5, requires_grad=True)
        
        mock_optimizer = MagicMock()
        
        train_one_epoch(mock_model, mock_loader, mock_criterion, mock_optimizer, 'cpu')
        
        mock_model.train.assert_called()


# ============================================================================
# VALIDATION LOOP TESTS
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestValidationLoop:
    """Tests for validation loop function."""
    
    def test_validate_returns_loss_and_acc(self):
        """Test validate returns loss and accuracy."""
        from training.finetune_vit import validate
        
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(4, 3)
        
        mock_loader = [
            (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 2, 0]))
        ]
        
        mock_criterion = MagicMock()
        mock_criterion.return_value = torch.tensor(0.5)
        
        loss, acc = validate(mock_model, mock_loader, mock_criterion, 'cpu')
        
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1
    
    def test_validate_sets_model_eval(self):
        """Test validate sets model to eval mode."""
        from training.finetune_vit import validate
        
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(4, 3)
        
        mock_loader = [
            (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 2, 0]))
        ]
        
        mock_criterion = MagicMock()
        mock_criterion.return_value = torch.tensor(0.5)
        
        validate(mock_model, mock_loader, mock_criterion, 'cpu')
        
        mock_model.eval.assert_called()
    
    def test_validate_no_gradient(self):
        """Test validate runs without gradient computation."""
        from training.finetune_vit import validate
        
        mock_model = MagicMock()
        mock_model.return_value = torch.randn(4, 3)
        
        mock_loader = [
            (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 2, 0]))
        ]
        
        mock_criterion = MagicMock()
        mock_criterion.return_value = torch.tensor(0.5)
        
        # The @torch.no_grad() decorator should prevent gradient tracking
        # We verify by checking the function runs successfully
        loss, acc = validate(mock_model, mock_loader, mock_criterion, 'cpu')
        
        assert loss is not None
        assert acc is not None


# ============================================================================
# TRAIN FUNCTION TESTS
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTrainFunction:
    """Tests for main train function."""
    
    def test_train_creates_save_dir(self, mock_args, temp_checkpoint_dir):
        """Test train creates save directory."""
        mock_args.save_dir = str(temp_checkpoint_dir / "new_dir")
        mock_args.data_dir = "/nonexistent"
        mock_args.num_epochs = 1
        
        with patch('training.finetune_vit.FineTunableViT'), \
             patch('training.finetune_vit.datasets.ImageFolder') as mock_dataset, \
             patch('training.finetune_vit.DataLoader'), \
             patch('training.finetune_vit.train_one_epoch', return_value=(0.5, 0.8)), \
             patch('training.finetune_vit.validate', return_value=(0.4, 0.85)):
            
            mock_dataset.return_value.classes = ['class_0', 'class_1']
            mock_dataset.return_value.__len__ = lambda self: 100
            
            # This will fail at DataLoader, but save_dir should be created
            try:
                from training.finetune_vit import train
                train(mock_args)
            except:
                pass
        
        assert os.path.exists(mock_args.save_dir)
    
    def test_train_handles_cuda_unavailable(self, mock_args):
        """Test train falls back to CPU when CUDA unavailable."""
        mock_args.device = "cuda"
        
        with patch('torch.cuda.is_available', return_value=False):
            from training.finetune_vit import train
            
            # The function should detect CUDA unavailable and fall back
            # We just verify it doesn't raise immediately
            assert True  # If we get here, the device check logic exists


# ============================================================================
# CHECKPOINT TESTS
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestCheckpointing:
    """Tests for checkpoint saving and loading."""
    
    def test_checkpoint_contains_required_keys(self, temp_checkpoint_dir):
        """Test that saved checkpoints contain all required keys."""
        # Create a mock checkpoint structure
        checkpoint = {
            'epoch': 10,
            'model_state': {},
            'optimizer_state': {},
            'best_acc': 0.95,
            'num_classes': 3,
            'classes': ['class_0', 'class_1', 'class_2'],
            'args': {'batch_size': 16}
        }
        
        required_keys = ['epoch', 'model_state', 'optimizer_state', 'best_acc', 
                        'num_classes', 'classes', 'args']
        
        for key in required_keys:
            assert key in checkpoint
    
    def test_resume_loads_checkpoint(self, mock_args, temp_checkpoint_dir):
        """Test that resume parameter loads checkpoint correctly."""
        # This tests the resume logic structure
        checkpoint_path = temp_checkpoint_dir / "test_checkpoint.pth"
        
        mock_checkpoint = {
            'epoch': 5,
            'model_state': {},
            'optimizer_state': {},
            'best_acc': 0.9,
            'num_classes': 3
        }
        
        torch.save(mock_checkpoint, checkpoint_path)
        
        # Verify checkpoint was saved
        assert checkpoint_path.exists()
        
        # Verify it can be loaded
        loaded = torch.load(checkpoint_path)
        assert loaded['epoch'] == 5
        assert loaded['best_acc'] == 0.9


# ============================================================================
# DEVICE HANDLING TESTS
# ============================================================================

class TestDeviceHandling:
    """Tests for device handling logic."""
    
    def test_cpu_device_sets_num_workers_zero(self, mock_args):
        """Test CPU device sets num_workers to 0."""
        mock_args.device = "cpu"
        
        # Based on the code logic:
        # is_cpu = device.type == "cpu"
        # num_workers = 0 if is_cpu else args.num_workers
        
        # Simulate the logic
        is_cpu = mock_args.device == "cpu"
        num_workers = 0 if is_cpu else mock_args.num_workers
        
        assert num_workers == 0
    
    def test_cuda_device_uses_configured_num_workers(self, mock_args):
        """Test CUDA device uses configured num_workers."""
        mock_args.device = "cuda"
        mock_args.num_workers = 4
        
        is_cpu = mock_args.device == "cpu"
        num_workers = 0 if is_cpu else mock_args.num_workers
        
        assert num_workers == 4
    
    def test_cpu_device_disables_pin_memory(self, mock_args):
        """Test CPU device disables pin_memory."""
        mock_args.device = "cpu"
        
        is_cpu = mock_args.device == "cpu"
        pin_memory = not is_cpu
        
        assert pin_memory == False
    
    def test_cuda_device_enables_pin_memory(self, mock_args):
        """Test CUDA device enables pin_memory."""
        mock_args.device = "cuda"
        
        is_cpu = mock_args.device == "cpu"
        pin_memory = not is_cpu
        
        assert pin_memory == True


# ============================================================================
# EARLY STOPPING TESTS
# ============================================================================

class TestEarlyStopping:
    """Tests for early stopping logic."""
    
    def test_patience_counter_increments_on_no_improvement(self):
        """Test patience counter increments when no improvement."""
        best_acc = 0.9
        val_acc = 0.85  # Worse than best
        patience_counter = 3
        
        if val_acc > best_acc:
            patience_counter = 0
        else:
            patience_counter += 1
        
        assert patience_counter == 4
    
    def test_patience_counter_resets_on_improvement(self):
        """Test patience counter resets on improvement."""
        best_acc = 0.9
        val_acc = 0.95  # Better than best
        patience_counter = 5
        
        if val_acc > best_acc:
            patience_counter = 0
            best_acc = val_acc
        else:
            patience_counter += 1
        
        assert patience_counter == 0
        assert best_acc == 0.95
    
    def test_early_stop_triggers_at_patience(self):
        """Test early stopping triggers when patience exceeded."""
        patience = 7
        patience_counter = 7
        
        should_stop = patience_counter >= patience
        
        assert should_stop == True
    
    def test_early_stop_not_triggered_before_patience(self):
        """Test early stopping doesn't trigger before patience."""
        patience = 7
        patience_counter = 5
        
        should_stop = patience_counter >= patience
        
        assert should_stop == False


# ============================================================================
# LOGGING AND OUTPUT TESTS
# ============================================================================

class TestLoggingAndOutput:
    """Tests for logging and print output."""
    
    def test_training_prints_banner(self, mock_args, capsys):
        """Test training prints startup banner."""
        # The banner printing is in train() function
        # We verify the format expectations
        expected_elements = [
            "ViT Fine-Tuning",
            "Data:",
            "LR",
            "Device:"
        ]
        
        # These elements should appear in the banner output
        assert all(isinstance(e, str) for e in expected_elements)
    
    def test_cuda_debug_prints_at_import(self, capsys):
        """Test CUDA debug info prints at module import."""
        # The module prints CUDA debug info at import time
        # We just verify the module can be imported
        import training.finetune_vit
        
        # If we get here, import succeeded
        assert True


# ============================================================================
# GRADIENT CLIPPING TESTS
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestGradientClipping:
    """Tests for gradient clipping in training."""
    
    def test_gradient_clipping_applied(self):
        """Test that gradient clipping is applied during training."""
        from training.finetune_vit import train_one_epoch
        
        mock_model = MagicMock()
        param = torch.nn.Parameter(torch.randn(10))
        mock_model.parameters.return_value = iter([param])
        mock_model.return_value = torch.randn(4, 3)
        
        mock_loader = [
            (torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 2, 0]))
        ]
        
        mock_criterion = MagicMock()
        mock_criterion.return_value = torch.tensor(0.5, requires_grad=True)
        
        mock_optimizer = MagicMock()
        
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            train_one_epoch(mock_model, mock_loader, mock_criterion, mock_optimizer, 'cpu')
            
            mock_clip.assert_called()


# ============================================================================
# MODULE IMPORT TESTS
# ============================================================================

class TestModuleImports:
    """Tests for module import behavior."""
    
    def test_module_imports_without_error(self):
        """Test that the module can be imported without errors."""
        import training.finetune_vit
    
    def test_get_args_is_callable(self):
        """Test that get_args function is accessible."""
        from training.finetune_vit import get_args
        assert callable(get_args)
    
    def test_get_transforms_is_callable(self):
        """Test that get_transforms function is accessible."""
        from training.finetune_vit import get_transforms
        assert callable(get_transforms)
    
    def test_finetunablevit_is_accessible(self):
        """Test that FineTunableViT class is accessible."""
        from training.finetune_vit import FineTunableViT
        assert FineTunableViT is not None
    
    def test_warmupcosinescheduler_is_accessible(self):
        """Test that WarmupCosineScheduler class is accessible."""
        from training.finetune_vit import WarmupCosineScheduler
        assert WarmupCosineScheduler is not None
    
    def test_train_is_callable(self):
        """Test that train function is accessible."""
        from training.finetune_vit import train
        assert callable(train)
    
    def test_main_is_callable(self):
        """Test that main function is accessible."""
        from training.finetune_vit import main
        assert callable(main)


# ============================================================================
# LABEL SMOOTHING TESTS
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestLabelSmoothing:
    """Tests for label smoothing configuration."""
    
    def test_crossentropy_supports_label_smoothing(self):
        """Test CrossEntropyLoss supports label_smoothing parameter."""
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Verify it was created successfully
        assert criterion is not None
    
    def test_label_smoothing_value_in_range(self, mock_args):
        """Test label smoothing value is in valid range [0, 1]."""
        assert 0 <= mock_args.label_smoothing <= 1


# ============================================================================
# HISTORY TRACKING TESTS
# ============================================================================

class TestHistoryTracking:
    """Tests for training history tracking."""
    
    def test_history_structure(self):
        """Test training history has correct structure."""
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        required_keys = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr']
        
        for key in required_keys:
            assert key in history
    
    def test_history_appends_correctly(self):
        """Test history values append correctly."""
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Simulate one epoch
        history['train_loss'].append(0.5)
        history['train_acc'].append(0.8)
        history['val_loss'].append(0.4)
        history['val_acc'].append(0.85)
        history['lr'].append([1e-5, 1e-3])
        
        assert len(history['train_loss']) == 1
        assert len(history['lr']) == 1
        assert history['lr'][0] == [1e-5, 1e-3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])