# tests/test_analysis_evaluate_tcn_horizon.py
"""
Comprehensive unit tests for analysis/evaluate_tcn_horizon.py

Tests horizon-based evaluation for TCN models.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.unit
class TestLoadCheckpointWithFeatures:
    """Test checkpoint loading with features."""

    @patch('analysis.evaluate_tcn_horizon.torch.load')
    @patch('training.train_tcn_enhanced.EnhancedTCN')
    def test_load_checkpoint_with_features(self, mock_enhanced_tcn, mock_torch_load):
        """Test loading checkpoint with feature columns."""
        mock_checkpoint = {
            'feature_columns': ['close', 'high', 'low'],
            'config': {
                'model': {'input_dim': 3, 'hidden_dim': 64},
                'training': {'num_classes': 3, 'dropout': 0.2}
            },
            'model_state': {}
        }
        mock_torch_load.return_value = mock_checkpoint

        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_enhanced_tcn.return_value = mock_model

        from analysis.evaluate_tcn_horizon import load_checkpoint_with_features
        model, features, checkpoint = load_checkpoint_with_features('model.pt')

        assert len(features) == 3
        assert model is not None

    @patch('analysis.evaluate_tcn_horizon.torch.load')
    def test_load_checkpoint_missing_features_raises_error(self, mock_torch_load):
        """Test that missing feature_columns raises error."""
        mock_checkpoint = {'model_state': {}}
        mock_torch_load.return_value = mock_checkpoint

        from analysis.evaluate_tcn_horizon import load_checkpoint_with_features
        
        with pytest.raises(ValueError, match="feature_columns"):
            load_checkpoint_with_features('model.pt')


@pytest.mark.unit
class TestHorizonEvaluator:
    """Test HorizonEvaluator class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.eval.return_value = model
        model.to.return_value = model
        return model

    @pytest.fixture
    def evaluator(self, mock_model):
        """Create HorizonEvaluator instance."""
        from analysis.evaluate_tcn_horizon import HorizonEvaluator
        feature_columns = ['close', 'high', 'low']
        return HorizonEvaluator(mock_model, feature_columns, device='cpu')

    def test_init(self, mock_model):
        """Test HorizonEvaluator initialization."""
        from analysis.evaluate_tcn_horizon import HorizonEvaluator
        feature_columns = ['close', 'high', 'low']
        
        evaluator = HorizonEvaluator(mock_model, feature_columns, device='cpu')

        assert evaluator.model == mock_model
        assert evaluator.feature_columns == feature_columns
        mock_model.eval.assert_called_once()

    def test_get_probabilities(self, evaluator):
        """Test getting probabilities from model."""
        mock_logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        evaluator.model.return_value = mock_logits
        
        X_test = np.random.randn(2, 30, 3)
        
        with patch('torch.from_numpy') as mock_from_numpy:
            mock_tensor = torch.FloatTensor(X_test)
            mock_from_numpy.return_value = mock_tensor
            
            probs = evaluator.get_probabilities(X_test)
            
            assert probs.shape[0] == 2
            assert probs.shape[1] == 3  # 3 classes

    @patch('analysis.evaluate_tcn_horizon.pd.read_csv')
    def test_prepare_horizon_data(self, mock_read_csv, evaluator):
        """Test preparing horizon-based data."""
        # Create real DataFrame with enough rows for seq_len + horizon + test split
        import pandas as pd
        n_rows = 200  # Enough for 80% train, 20% test with seq_len=30 and horizon=5
        mock_df = pd.DataFrame({
            'close': np.random.randn(n_rows) + 1.1,  # Positive prices
            'high': np.random.randn(n_rows) + 1.11,
            'low': np.random.randn(n_rows) + 1.09
        })
        mock_read_csv.return_value = mock_df

        # Mock _ensure_features to return the same df
        evaluator._ensure_features = Mock(return_value=mock_df)
        
        X, entry_prices, exit_prices = evaluator.prepare_horizon_data(
            'data.csv', seq_len=30, horizon=5
        )

        mock_read_csv.assert_called_once()
        assert len(X) > 0
        assert len(entry_prices) == len(X)
        assert len(exit_prices) == len(X)

    def test_evaluate_horizon_strategy(self, evaluator):
        """Test horizon strategy evaluation."""
        probs = np.array([
            [0.1, 0.2, 0.7],  # Strong BULL
            [0.7, 0.2, 0.1],  # Strong BEAR
            [0.33, 0.34, 0.33]  # SIDEWAYS
        ])
        entry_prices = np.array([1.1000, 1.1050, 1.1100])
        exit_prices = np.array([1.1020, 1.1030, 1.1110])  # Mixed results

        # Suppress print output during test
        with patch('builtins.print'):
            result = evaluator.evaluate_horizon_strategy(
                probs, entry_prices, exit_prices, spread_cost=0.0001
            )

        assert 'best' in result
        assert 'threshold' in result['best']
        assert 'total_return' in result['best']
        assert 'all_results' in result

