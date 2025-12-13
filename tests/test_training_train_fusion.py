# tests/test_training_train_fusion.py
"""
Unit tests for training/train_fusion.py - Fusion model training.

Tests cover:
- FeatureDataset initialization and feature extraction
- train_fusion_model function
- Pre-trained model loading
- Feature fusion and training loop
- Model saving and checkpointing
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from torch.utils.data import DataLoader

# Import the modules to test
from training.train_fusion import (
    FeatureDataset,
    train_fusion_model,
    main,
)


@pytest.mark.unit
class TestFeatureDataset:
    """Test FeatureDataset class."""

    @pytest.fixture
    def mock_models(self):
        """Create mock TCN, ViT, and YOLO models."""
        tcn = MagicMock(spec=nn.Module)
        tcn.eval.return_value = tcn
        tcn.return_value = torch.randn(1, 64)  # TCN features

        vit = MagicMock(spec=nn.Module)
        vit.eval.return_value = vit
        vit.return_value = torch.randn(1, 768)  # ViT features

        yolo = MagicMock()
        yolo.detect.return_value = np.random.randn(20)  # YOLO features

        return tcn, vit, yolo

    @pytest.fixture
    def sample_data(self):
        """Create sample data and labels."""
        # 10 samples, 60 timesteps, 5 features
        data = np.random.randn(10, 60, 5).astype(np.float32)
        labels = np.random.randint(0, 3, size=10)
        return data, labels

    def test_init(self, mock_models, sample_data):
        """Test FeatureDataset initialization."""
        tcn, vit, yolo = mock_models
        data, labels = sample_data

        with patch('training.train_fusion.candle_image') as mock_candle, \
             patch('training.train_fusion.normalize_for_model') as mock_normalize:

            # Mock image generation
            mock_candle.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            mock_normalize.return_value = np.random.randn(224, 224, 3).astype(np.float32)

            dataset = FeatureDataset(
                data=data,
                labels=labels,
                tcn_model=tcn,
                vit_model=vit,
                yolo_detector=yolo,
                device=torch.device('cpu'),
                seq_len=60,
                img_size=224,
            )

            assert len(dataset) == len(labels)
            assert 'tcn' in dataset.features
            assert 'vit' in dataset.features
            assert 'yolo' in dataset.features

    def test_extract_all_features(self, mock_models, sample_data):
        """Test feature extraction from all models."""
        tcn, vit, yolo = mock_models
        data, labels = sample_data

        with patch('training.train_fusion.candle_image') as mock_candle, \
             patch('training.train_fusion.normalize_for_model') as mock_normalize:

            mock_candle.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            mock_normalize.return_value = np.random.randn(224, 224, 3).astype(np.float32)

            dataset = FeatureDataset(
                data=data,
                labels=labels,
                tcn_model=tcn,
                vit_model=vit,
                yolo_detector=yolo,
                device=torch.device('cpu'),
            )

            # Check feature shapes
            assert dataset.features['tcn'].shape[0] == len(data)
            assert dataset.features['vit'].shape[0] == len(data)
            assert dataset.features['yolo'].shape[0] == len(data)

    def test_getitem(self, mock_models, sample_data):
        """Test __getitem__ returns correct tuple."""
        tcn, vit, yolo = mock_models
        data, labels = sample_data

        with patch('training.train_fusion.candle_image') as mock_candle, \
             patch('training.train_fusion.normalize_for_model') as mock_normalize:

            mock_candle.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            mock_normalize.return_value = np.random.randn(224, 224, 3).astype(np.float32)

            dataset = FeatureDataset(
                data=data,
                labels=labels,
                tcn_model=tcn,
                vit_model=vit,
                yolo_detector=yolo,
                device=torch.device('cpu'),
            )

            tcn_feat, vit_feat, yolo_feat, label = dataset[0]

            assert isinstance(tcn_feat, torch.Tensor)
            assert isinstance(vit_feat, torch.Tensor)
            assert isinstance(yolo_feat, torch.Tensor)
            assert isinstance(label, (int, np.integer))


@pytest.mark.unit
class TestTrainFusionModel:
    """Test train_fusion_model function."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary directory structure."""
        data_dir = tmp_path / "data"
        weights_dir = tmp_path / "weights"
        data_dir.mkdir()
        weights_dir.mkdir()

        # Create dummy CSV
        csv_path = data_dir / "test.csv"
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=200, freq='1h'),
            'open': np.random.randn(200).cumsum() + 1.1,
            'high': np.random.randn(200).cumsum() + 1.11,
            'low': np.random.randn(200).cumsum() + 1.09,
            'close': np.random.randn(200).cumsum() + 1.1,
            'tick_volume': np.random.randint(100, 1000, 200),
        })
        df.to_csv(csv_path, index=False)

        return {
            'csv_path': str(csv_path),
            'weights_dir': str(weights_dir),
        }

    @patch('training.train_fusion.YOLOPatternDetector')
    @patch('training.train_fusion.MockYOLODetector')
    @patch('training.train_fusion.ViTExtractor')
    @patch('training.train_fusion.TCNModel')
    @patch('training.train_fusion.FusionNet')
    @patch('training.train_fusion.MyDataLoader')
    def test_train_fusion_basic(
        self,
        mock_data_loader_class,
        mock_fusion_class,
        mock_tcn_class,
        mock_vit_class,
        mock_mock_yolo_class,
        mock_yolo_class,
        temp_data_dir,
    ):
        """Test basic training flow."""
        # Mock data loader
        mock_loader = MagicMock()
        mock_data_loader_class.return_value = mock_loader

        mock_df = pd.DataFrame({
            'open': np.random.randn(200),
            'high': np.random.randn(200),
            'low': np.random.randn(200),
            'close': np.random.randn(200),
            'tick_volume': np.random.randint(100, 1000, 200),
        })
        mock_loader.load_csv.return_value = mock_df

        train_scaled = mock_df.iloc[:160]
        test_scaled = mock_df.iloc[160:]
        mock_loader.split_and_scale.return_value = (train_scaled, test_scaled, None)

        X_train = np.random.randn(100, 60, 5).astype(np.float32)
        y_train = np.random.randint(0, 3, size=100)
        X_test = np.random.randn(20, 60, 5).astype(np.float32)
        y_test = np.random.randint(0, 3, size=20)

        mock_loader.create_sequences.side_effect = [
            (X_train, y_train),
            (X_test, y_test),
        ]

        # Mock models
        class MockTCN(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.linear = nn.Linear(10, 64)

            def forward(self, x, mode='features'):
                return torch.randn(x.shape[0], 64)

            def get_feature_dim(self):
                return 64

            def eval(self):
                return self

            def to(self, device):
                return self

        mock_tcn_class.return_value = MockTCN()

        class MockViT(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.linear = nn.Linear(10, 768)

            def forward(self, x):
                return torch.randn(x.shape[0], 768)

            def get_feature_dim(self):
                return 768

            def eval(self):
                return self

            def to(self, device):
                return self

        mock_vit_class.return_value = MockViT()

        mock_yolo = MagicMock()
        mock_yolo.get_feature_dim.return_value = 20
        mock_mock_yolo_class.return_value = mock_yolo

        # Create a real fusion model mock that behaves properly
        class MockFusion(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.linear = nn.Linear(64 + 768 + 20, 3)  # Combined feature size

            def forward(self, tcn_f, vit_f, yolo_f):
                # Concatenate features and use linear layer to preserve gradients
                combined = torch.cat([tcn_f, vit_f, yolo_f], dim=1)
                return self.linear(combined)

            def to(self, device):
                super().to(device)
                return self

        mock_fusion_class.return_value = MockFusion()

        # Patch FeatureDataset to avoid actual feature extraction
        with patch('training.train_fusion.FeatureDataset') as mock_dataset_class:
            # Create mock datasets
            def create_mock_dataset(data, labels, *args, **kwargs):
                dataset = MagicMock()
                dataset.__len__ = lambda self: len(labels)

                def getitem(self, idx):
                    return (
                        torch.randn(64),   # TCN features
                        torch.randn(768),  # ViT features
                        torch.randn(20),   # YOLO features
                        int(labels[idx]),  # Convert to Python int
                    )
                dataset.__getitem__ = getitem
                return dataset

            mock_dataset_class.side_effect = create_mock_dataset

            # Patch torch.save to avoid file I/O
            with patch('torch.save') as mock_save:
                # Run training with 1 epoch
                model, history = train_fusion_model(
                    data_path=temp_data_dir['csv_path'],
                    weights_dir=temp_data_dir['weights_dir'],
                    epochs=1,
                    batch_size=8,
                    learning_rate=1e-3,
                    device='cpu',
                )

                # Verify history structure
                assert 'train_loss' in history
                assert 'test_loss' in history
                assert 'test_acc' in history
                assert len(history['train_loss']) == 1

    @patch('training.train_fusion.torch.load')
    @patch('training.train_fusion.YOLOPatternDetector')
    @patch('training.train_fusion.MockYOLODetector')
    @patch('training.train_fusion.ViTExtractor')
    @patch('training.train_fusion.TCNModel')
    @patch('training.train_fusion.FusionNet')
    @patch('training.train_fusion.MyDataLoader')
    def test_pretrained_loading(
        self,
        mock_data_loader_class,
        mock_fusion_class,
        mock_tcn_class,
        mock_vit_class,
        mock_mock_yolo_class,
        mock_yolo_class,
        mock_torch_load,
        temp_data_dir,
    ):
        """Test that pre-trained weights are loaded if available."""
        # Create dummy weight files
        weights_dir = Path(temp_data_dir['weights_dir'])
        (weights_dir / "tcn_best.pt").touch()
        (weights_dir / "vit_best.pt").touch()
        (weights_dir / "yolo_best.pt").touch()

        # Mock torch.load to return dummy state dicts
        mock_torch_load.return_value = {}

        # Mock data loader
        mock_loader = MagicMock()
        mock_data_loader_class.return_value = mock_loader

        mock_df = pd.DataFrame({
            'open': np.random.randn(200),
            'high': np.random.randn(200),
            'low': np.random.randn(200),
            'close': np.random.randn(200),
            'tick_volume': np.random.randint(100, 1000, 200),
        })
        mock_loader.load_csv.return_value = mock_df
        mock_loader.split_and_scale.return_value = (mock_df.iloc[:160], mock_df.iloc[160:], None)

        X_train = np.random.randn(50, 60, 5).astype(np.float32)
        y_train = np.random.randint(0, 3, size=50)
        X_test = np.random.randn(10, 60, 5).astype(np.float32)
        y_test = np.random.randint(0, 3, size=10)
        mock_loader.create_sequences.side_effect = [(X_train, y_train), (X_test, y_test)]

        # Mock models
        class MockTCN(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.linear = nn.Linear(10, 64)

            def forward(self, x, mode='features'):
                return torch.randn(x.shape[0], 64)

            def get_feature_dim(self):
                return 64

            def eval(self):
                return self

            def to(self, device):
                return self

        class MockViT(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.linear = nn.Linear(10, 768)

            def forward(self, x):
                return torch.randn(x.shape[0], 768)

            def get_feature_dim(self):
                return 768

            def eval(self):
                return self

            def to(self, device):
                return self

        mock_tcn = MockTCN()
        mock_tcn.load_state_dict = MagicMock()
        mock_tcn_class.return_value = mock_tcn

        mock_vit = MockViT()
        mock_vit.load_state_dict = MagicMock()
        mock_vit_class.return_value = mock_vit

        mock_yolo_instance = MagicMock()
        mock_yolo_instance.get_feature_dim.return_value = 20
        mock_yolo_class.return_value = mock_yolo_instance

        # Create a real fusion model mock
        class MockFusion(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.linear = nn.Linear(64 + 768 + 20, 3)

            def forward(self, tcn_f, vit_f, yolo_f):
                combined = torch.cat([tcn_f, vit_f, yolo_f], dim=1)
                return self.linear(combined)

            def to(self, device):
                super().to(device)
                return self

        mock_fusion_class.return_value = MockFusion()

        with patch('training.train_fusion.FeatureDataset') as mock_dataset_class, \
             patch('torch.save'):

            def create_mock_dataset(data, labels, *args, **kwargs):
                dataset = MagicMock()
                dataset.__len__ = lambda self: len(labels)

                def getitem(self, idx):
                    return (
                        torch.randn(64),
                        torch.randn(768),
                        torch.randn(20),
                        int(labels[idx]),  # Convert to Python int
                    )
                dataset.__getitem__ = getitem
                return dataset

            mock_dataset_class.side_effect = create_mock_dataset

            train_fusion_model(
                data_path=temp_data_dir['csv_path'],
                weights_dir=temp_data_dir['weights_dir'],
                epochs=1,
                batch_size=8,
                device='cpu',
            )

            # Verify load_state_dict was called for TCN and ViT
            assert mock_tcn.load_state_dict.called
            assert mock_vit.load_state_dict.called
            # YOLO uses YOLOPatternDetector which loads via constructor
            assert mock_yolo_class.called


@pytest.mark.unit
class TestMain:
    """Test main function and argument parsing."""

    @patch('training.train_fusion.train_fusion_model')
    @patch('sys.argv', ['train_fusion.py', '--data', 'data.csv', '--epochs', '5'])
    def test_main_default_args(self, mock_train):
        """Test main with default arguments."""
        mock_train.return_value = (MagicMock(), {})

        main()

        mock_train.assert_called_once()
        call_args = mock_train.call_args
        assert call_args[1]['data_path'] == 'data.csv'
        assert call_args[1]['epochs'] == 5

    @patch('training.train_fusion.train_fusion_model')
    @patch('sys.argv', [
        'train_fusion.py',
        '--data', 'custom.csv',
        '--weights-dir', 'custom_weights',
        '--epochs', '10',
        '--batch-size', '16',
        '--lr', '1e-4',
    ])
    def test_main_custom_args(self, mock_train):
        """Test main with custom arguments."""
        mock_train.return_value = (MagicMock(), {})

        main()

        call_args = mock_train.call_args
        assert call_args[1]['data_path'] == 'custom.csv'
        assert call_args[1]['weights_dir'] == 'custom_weights'
        assert call_args[1]['epochs'] == 10
        assert call_args[1]['batch_size'] == 16
        assert call_args[1]['learning_rate'] == 1e-4
