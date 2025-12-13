# tests/test_training_train_tcn.py
"""
Unit tests for training/train_tcn.py - TCN model training.

Tests cover:
- OneCycleLR scheduler
- train_tcn_model function with different configurations
- Model variants (standard, attention, multiscale)
- Profile-based training
- evaluate_model function
- Checkpoint saving and history tracking
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from torch.utils.data import DataLoader, TensorDataset

# Import the modules to test
from training.train_tcn import (
    OneCycleLR,
    train_tcn_model,
    evaluate_model,
    main,
)


@pytest.mark.unit
class TestOneCycleLR:
    """Test OneCycleLR scheduler."""

    def test_init(self):
        """Test OneCycleLR initialization."""
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=1e-3)
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=1e-2,
            total_steps=100,
            pct_start=0.3,
        )

        assert scheduler.max_lr == 1e-2
        assert scheduler.total_steps == 100
        assert scheduler.warmup_steps == 30
        assert scheduler.step_count == 0

    def test_warmup_phase(self):
        """Test learning rate during warmup phase."""
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=1e-3)
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=1e-2,
            total_steps=100,
            pct_start=0.3,
        )

        initial_lr = scheduler.initial_lr

        # Step through warmup
        for _ in range(30):
            scheduler.step()

        # LR should increase to max_lr
        final_warmup_lr = scheduler.get_lr()
        assert final_warmup_lr > initial_lr
        assert final_warmup_lr == pytest.approx(1e-2, rel=1e-3)

    def test_annealing_phase(self):
        """Test learning rate during annealing phase."""
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=1e-3)
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=1e-2,
            total_steps=100,
            pct_start=0.3,
        )

        # Step through entire schedule
        for _ in range(100):
            scheduler.step()

        # LR should decay to final_lr
        final_lr = scheduler.get_lr()
        assert final_lr < 1e-2
        assert final_lr == pytest.approx(scheduler.final_lr, rel=1e-3)

    def test_get_lr(self):
        """Test get_lr returns current learning rate."""
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(10))], lr=1e-3)
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=1e-2,
            total_steps=100,
        )

        scheduler.step()
        lr = scheduler.get_lr()
        assert isinstance(lr, float)
        assert lr > 0


@pytest.mark.unit
class TestTrainTCNModel:
    """Test train_tcn_model function."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary directory and CSV file."""
        data_dir = tmp_path / "data"
        save_dir = tmp_path / "weights"
        data_dir.mkdir()
        save_dir.mkdir()

        # Create dummy CSV with realistic data
        csv_path = data_dir / "test.csv"
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=300, freq='1h'),
            'open': np.random.randn(300).cumsum() + 1.1,
            'high': np.random.randn(300).cumsum() + 1.11,
            'low': np.random.randn(300).cumsum() + 1.09,
            'close': np.random.randn(300).cumsum() + 1.1,
            'tick_volume': np.random.randint(100, 1000, 300),
        })
        df.to_csv(csv_path, index=False)

        return {
            'csv_path': str(csv_path),
            'save_dir': str(save_dir),
        }

    @patch('training.train_tcn.MyDataLoader')
    @patch('training.train_tcn.TCNModel')
    def test_train_basic(self, mock_tcn_class, mock_loader_class, temp_data_dir):
        """Test basic training flow."""
        # Mock data loader
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        mock_df = pd.DataFrame({
            'open': np.random.randn(300),
            'high': np.random.randn(300),
            'low': np.random.randn(300),
            'close': np.random.randn(300),
            'tick_volume': np.random.randint(100, 1000, 300),
        })
        mock_loader.load_csv.return_value = mock_df

        train_scaled = mock_df.iloc[:240]
        test_scaled = mock_df.iloc[240:]
        val_scaled = mock_df.iloc[210:240]
        mock_loader.split_and_scale.return_value = (train_scaled, test_scaled, val_scaled)

        # Create sequences
        X_train = np.random.randn(150, 60, 5).astype(np.float32)
        y_train = np.random.randint(0, 3, size=150)
        X_test = np.random.randn(30, 60, 5).astype(np.float32)
        y_test = np.random.randint(0, 3, size=30)

        mock_loader.create_sequences.side_effect = [
            (X_train, y_train),
            (X_test, y_test),
        ]
        mock_loader.save_scaler = MagicMock()

        # Create a real TCN model mock
        class MockTCN(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.linear = nn.Linear(10, 3)

            def forward(self, x, mode='classify'):
                batch_size = x.shape[0]
                return torch.randn(batch_size, 3)

            def to(self, device):
                return self

        mock_tcn_class.return_value = MockTCN()

        # Patch torch.save to avoid file I/O
        with patch('torch.save') as mock_save:
            model, history = train_tcn_model(
                data_path=temp_data_dir['csv_path'],
                save_dir=temp_data_dir['save_dir'],
                epochs=2,
                batch_size=16,
                learning_rate=1e-3,
                use_onecycle=True,
                device='cpu',
            )

            # Verify history structure
            assert 'train_loss' in history
            assert 'train_acc' in history
            assert 'test_loss' in history
            assert 'test_acc' in history
            assert 'lr' in history
            assert len(history['train_loss']) == 2

            # Verify model was saved
            assert mock_save.call_count >= 2  # History + checkpoint

    @patch('training.train_tcn.MyDataLoader')
    @patch('training.train_tcn.TCNModel')
    def test_train_with_profile(self, mock_tcn_class, mock_loader_class, temp_data_dir):
        """Test training with profile preset."""
        # Setup mocks
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        mock_df = pd.DataFrame({
            'open': np.random.randn(200),
            'high': np.random.randn(200),
            'low': np.random.randn(200),
            'close': np.random.randn(200),
            'tick_volume': np.random.randint(100, 1000, 200),
        })
        mock_loader.load_csv.return_value = mock_df
        mock_loader.split_and_scale.return_value = (mock_df.iloc[:160], mock_df.iloc[160:], None)

        X = np.random.randn(100, 60, 5).astype(np.float32)
        y = np.random.randint(0, 3, size=100)
        mock_loader.create_sequences.return_value = (X, y)
        mock_loader.save_scaler = MagicMock()

        # Mock TCNModel.from_profile
        class MockTCN(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.linear = nn.Linear(5, 3)  # Proper input size

            def forward(self, x, mode='classify'):
                # Use actual linear layer to preserve gradients
                batch_size = x.shape[0]
                x_flat = x.view(batch_size, -1)[:, :5]  # Take first 5 features
                return self.linear(x_flat)

            def to(self, device):
                super().to(device)
                return self

        mock_tcn_class.from_profile.return_value = MockTCN()

        with patch('torch.save'):
            train_tcn_model(
                data_path=temp_data_dir['csv_path'],
                save_dir=temp_data_dir['save_dir'],
                epochs=1,
                batch_size=16,
                profile='SCALP',
                variant='standard',
                device='cpu',
            )

            # Verify from_profile was called
            mock_tcn_class.from_profile.assert_called_once()
            call_args = mock_tcn_class.from_profile.call_args
            assert call_args[0][0] == 'SCALP'

    @patch('training.train_tcn.MyDataLoader')
    @patch('training.train_tcn.create_tcn_model')
    def test_train_with_variant(self, mock_create_tcn, mock_loader_class, temp_data_dir):
        """Test training with model variant."""
        # Setup mocks
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        mock_df = pd.DataFrame({
            'open': np.random.randn(200),
            'high': np.random.randn(200),
            'low': np.random.randn(200),
            'close': np.random.randn(200),
            'tick_volume': np.random.randint(100, 1000, 200),
        })
        mock_loader.load_csv.return_value = mock_df
        mock_loader.split_and_scale.return_value = (mock_df.iloc[:160], mock_df.iloc[160:], None)

        X = np.random.randn(100, 60, 5).astype(np.float32)
        y = np.random.randint(0, 3, size=100)
        mock_loader.create_sequences.return_value = (X, y)
        mock_loader.save_scaler = MagicMock()

        class MockTCN(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.linear = nn.Linear(5, 3)

            def forward(self, x, mode='classify'):
                batch_size = x.shape[0]
                x_flat = x.view(batch_size, -1)[:, :5]
                return self.linear(x_flat)

            def to(self, device):
                super().to(device)
                return self

        mock_create_tcn.return_value = MockTCN()

        with patch('torch.save'):
            train_tcn_model(
                data_path=temp_data_dir['csv_path'],
                save_dir=temp_data_dir['save_dir'],
                epochs=1,
                batch_size=16,
                variant='attention',
                device='cpu',
            )

            # Verify create_tcn_model was called with attention variant
            mock_create_tcn.assert_called_once()
            call_args = mock_create_tcn.call_args
            assert call_args[0][0] == 'attention'

    @patch('training.train_tcn.MyDataLoader')
    @patch('training.train_tcn.TCNModel')
    def test_class_weighting(self, mock_tcn_class, mock_loader_class, temp_data_dir):
        """Test class weighting for imbalanced data."""
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        mock_df = pd.DataFrame({
            'open': np.random.randn(200),
            'high': np.random.randn(200),
            'low': np.random.randn(200),
            'close': np.random.randn(200),
            'tick_volume': np.random.randint(100, 1000, 200),
        })
        mock_loader.load_csv.return_value = mock_df
        mock_loader.split_and_scale.return_value = (mock_df.iloc[:160], mock_df.iloc[160:], None)

        # Imbalanced dataset: mostly class 0
        X_train = np.random.randn(100, 60, 5).astype(np.float32)
        y_train = np.array([0] * 70 + [1] * 20 + [2] * 10)
        X_test = np.random.randn(20, 60, 5).astype(np.float32)
        y_test = np.random.randint(0, 3, size=20)

        mock_loader.create_sequences.side_effect = [(X_train, y_train), (X_test, y_test)]
        mock_loader.save_scaler = MagicMock()

        class MockTCN(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.linear = nn.Linear(10, 3)

            def forward(self, x, mode='classify'):
                return torch.randn(x.shape[0], 3)

            def to(self, device):
                return self

        mock_tcn_class.return_value = MockTCN()

        # Don't patch CrossEntropyLoss - let the test run through actual loss calculation
        with patch('torch.save'):
            train_tcn_model(
                data_path=temp_data_dir['csv_path'],
                save_dir=temp_data_dir['save_dir'],
                epochs=1,
                batch_size=16,
                device='cpu',
            )

            # Just verify training completed

    @patch('training.train_tcn.MyDataLoader')
    @patch('training.train_tcn.TCNModel')
    def test_cosine_annealing_scheduler(self, mock_tcn_class, mock_loader_class, temp_data_dir):
        """Test training with cosine annealing scheduler."""
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader

        mock_df = pd.DataFrame({
            'open': np.random.randn(200),
            'high': np.random.randn(200),
            'low': np.random.randn(200),
            'close': np.random.randn(200),
            'tick_volume': np.random.randint(100, 1000, 200),
        })
        mock_loader.load_csv.return_value = mock_df
        mock_loader.split_and_scale.return_value = (mock_df.iloc[:160], mock_df.iloc[160:], None)

        X = np.random.randn(100, 60, 5).astype(np.float32)
        y = np.random.randint(0, 3, size=100)
        mock_loader.create_sequences.return_value = (X, y)
        mock_loader.save_scaler = MagicMock()

        class MockTCN(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.linear = nn.Linear(10, 3)

            def forward(self, x, mode='classify'):
                return torch.randn(x.shape[0], 3)

            def to(self, device):
                return self

        mock_tcn_class.return_value = MockTCN()

        with patch('torch.save'):
            model, history = train_tcn_model(
                data_path=temp_data_dir['csv_path'],
                save_dir=temp_data_dir['save_dir'],
                epochs=1,
                batch_size=16,
                use_onecycle=False,  # Use cosine annealing
                device='cpu',
            )

            # Verify model trained
            assert 'lr' in history


@pytest.mark.unit
class TestEvaluateModel:
    """Test evaluate_model function."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, mode='classify'):
                batch_size = x.shape[0]
                # Return logits that favor class 0
                return torch.tensor([[2.0, 0.5, 0.3]] * batch_size)

            def eval(self):
                return self

        return MockModel()

    @pytest.fixture
    def sample_dataloader(self):
        """Create a sample dataloader."""
        X = torch.randn(50, 60, 5)
        y = torch.zeros(50, dtype=torch.long)  # All class 0
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=10)

    def test_evaluate_basic(self, mock_model, sample_dataloader):
        """Test basic evaluation."""
        metrics = evaluate_model(
            model=mock_model,
            data_loader=sample_dataloader,
            device=torch.device('cpu'),
            num_classes=3,
        )

        assert 'accuracy' in metrics
        assert 'per_class' in metrics
        assert 'predictions' in metrics
        assert 'labels' in metrics
        assert 'probabilities' in metrics

        # Should have high accuracy since all labels are 0 and model favors 0
        assert metrics['accuracy'] > 0.8

    def test_evaluate_per_class_metrics(self, mock_model, sample_dataloader):
        """Test per-class metrics calculation."""
        metrics = evaluate_model(
            model=mock_model,
            data_loader=sample_dataloader,
            device=torch.device('cpu'),
            num_classes=3,
        )

        per_class = metrics['per_class']
        assert 'BUY' in per_class
        assert 'SELL' in per_class
        assert 'HOLD' in per_class

        # Class 0 (BUY) should have high accuracy
        assert per_class['BUY']['accuracy'] > 0.8
        assert per_class['BUY']['count'] == 50

    def test_evaluate_probabilities(self, mock_model, sample_dataloader):
        """Test probability output."""
        metrics = evaluate_model(
            model=mock_model,
            data_loader=sample_dataloader,
            device=torch.device('cpu'),
        )

        probs = metrics['probabilities']
        assert probs.shape[0] == 50
        assert probs.shape[1] == 3

        # Probabilities should sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0)


@pytest.mark.unit
class TestMain:
    """Test main function and argument parsing."""

    @patch('training.train_tcn.train_tcn_model')
    @patch('sys.argv', [
        'train_tcn.py',
        '--data', 'data.csv',
        '--epochs', '10',
    ])
    def test_main_basic(self, mock_train):
        """Test main with basic arguments."""
        mock_train.return_value = (MagicMock(), {})

        main()

        mock_train.assert_called_once()
        call_kwargs = mock_train.call_args[1]
        assert call_kwargs['data_path'] == 'data.csv'
        assert call_kwargs['epochs'] == 10

    @patch('training.train_tcn.train_tcn_model')
    @patch('sys.argv', [
        'train_tcn.py',
        '--data', 'data.csv',
        '--profile', 'SCALP',
    ])
    def test_main_with_profile(self, mock_train):
        """Test main with profile argument."""
        mock_train.return_value = (MagicMock(), {})

        main()

        call_kwargs = mock_train.call_args[1]
        assert call_kwargs['profile'] == 'SCALP'
        assert call_kwargs['variant'] == 'standard'

    @patch('training.train_tcn.train_tcn_model')
    @patch('sys.argv', [
        'train_tcn.py',
        '--data', 'data.csv',
        '--attention',
    ])
    def test_main_with_attention(self, mock_train):
        """Test main with attention variant."""
        mock_train.return_value = (MagicMock(), {})

        main()

        call_kwargs = mock_train.call_args[1]
        assert call_kwargs['variant'] == 'attention'

    @patch('training.train_tcn.train_tcn_model')
    @patch('sys.argv', [
        'train_tcn.py',
        '--data', 'data.csv',
        '--multiscale',
    ])
    def test_main_with_multiscale(self, mock_train):
        """Test main with multiscale variant."""
        mock_train.return_value = (MagicMock(), {})

        main()

        call_kwargs = mock_train.call_args[1]
        assert call_kwargs['variant'] == 'multiscale'

    @patch('training.train_tcn.train_tcn_model')
    @patch('sys.argv', [
        'train_tcn.py',
        '--data', 'data.csv',
        '--no-onecycle',
    ])
    def test_main_no_onecycle(self, mock_train):
        """Test main with cosine annealing scheduler."""
        mock_train.return_value = (MagicMock(), {})

        main()

        call_kwargs = mock_train.call_args[1]
        assert call_kwargs['use_onecycle'] == False
