# tests/test_analysis_evaluate_tcn_model.py
"""
Comprehensive unit tests for analysis/evaluate_tcn_model.py

Tests TCN model evaluation functionality.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


@pytest.mark.unit
class TestLoadTCNCheckpoint:
    """Test TCN checkpoint loading."""

    @patch('analysis.evaluate_tcn_model.torch.load')
    @patch('analysis.evaluate_tcn_model.EnhancedTCN')
    def test_load_checkpoint_with_feature_columns(self, mock_enhanced_tcn, mock_torch_load):
        """Test loading checkpoint with feature_columns."""
        mock_checkpoint = {
            'feature_columns': ['close', 'high', 'low', 'volume'],
            'config': {
                'model': {'input_dim': 4, 'hidden_dim': 64, 'num_layers': 3},
                'training': {'num_classes': 3, 'dropout': 0.2}
            },
            'model_state': {}
        }
        mock_torch_load.return_value = mock_checkpoint

        mock_model = MagicMock()
        mock_enhanced_tcn.return_value = mock_model

        from analysis.evaluate_tcn_model import load_tcn_checkpoint
        model, features, checkpoint = load_tcn_checkpoint('model.pt')

        assert len(features) == 4
        assert features == ['close', 'high', 'low', 'volume']
        assert model is not None

    @patch('analysis.evaluate_tcn_model.torch.load')
    @patch('analysis.evaluate_tcn_model.EnhancedTCN')
    def test_load_checkpoint_missing_features(self, mock_enhanced_tcn, mock_torch_load):
        """Test loading checkpoint without feature_columns raises error."""
        mock_checkpoint = {'model_state': {}}
        mock_torch_load.return_value = mock_checkpoint

        from analysis.evaluate_tcn_model import load_tcn_checkpoint
        
        with pytest.raises(ValueError, match="feature_columns"):
            load_tcn_checkpoint('model.pt')

    @patch('analysis.evaluate_tcn_model.torch.load')
    @patch('analysis.evaluate_tcn_model.EnhancedTCN')
    def test_load_checkpoint_with_config(self, mock_enhanced_tcn, mock_torch_load):
        """Test loading checkpoint with full config."""
        mock_checkpoint = {
            'feature_columns': ['close', 'high', 'low'],
            'config': {
                'model': {'input_dim': 3, 'hidden_dim': 128, 'num_layers': 4},
                'training': {'num_classes': 3, 'dropout': 0.3}
            },
            'model_state': {'layer.weight': torch.randn(3, 128)}
        }
        mock_torch_load.return_value = mock_checkpoint

        mock_model = MagicMock()
        mock_enhanced_tcn.return_value = mock_model

        from analysis.evaluate_tcn_model import load_tcn_checkpoint
        model, features, checkpoint = load_tcn_checkpoint('model.pt')

        # Should use config values
        mock_enhanced_tcn.assert_called()
        assert len(features) == 3


@pytest.mark.unit
class TestTCNEvaluator:
    """Test TCNEvaluator class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock TCN model."""
        model = MagicMock()
        model.eval.return_value = model
        model.to.return_value = model
        return model

    @pytest.fixture
    def evaluator(self, mock_model):
        """Create TCNEvaluator instance."""
        from analysis.evaluate_tcn_model import TCNEvaluator
        feature_columns = ['close', 'high', 'low']
        return TCNEvaluator(mock_model, feature_columns, device='cpu')

    def test_init(self, mock_model):
        """Test TCNEvaluator initialization."""
        from analysis.evaluate_tcn_model import TCNEvaluator
        feature_columns = ['close', 'high', 'low']
        
        evaluator = TCNEvaluator(mock_model, feature_columns, device='cpu')

        assert evaluator.model == mock_model
        assert evaluator.feature_columns == feature_columns
        mock_model.eval.assert_called_once()

    def test_init_auto_device(self, mock_model):
        """Test initialization with auto device selection."""
        from analysis.evaluate_tcn_model import TCNEvaluator
        feature_columns = ['close', 'high', 'low']
        
        with patch('torch.cuda.is_available', return_value=False):
            evaluator = TCNEvaluator(mock_model, feature_columns, device='auto')
            assert evaluator.device.type == 'cpu'

    @patch('analysis.evaluate_tcn_model.pd.read_csv')
    def test_prepare_data(self, mock_read_csv, evaluator):
        """Test data preparation."""
        mock_df = MagicMock()
        mock_df.columns = ['close', 'high', 'low', 'volume']
        mock_df.iloc.__getitem__.return_value = mock_df
        mock_read_csv.return_value = mock_df

        # Mock _ensure_features
        evaluator._ensure_features = Mock(return_value=mock_df)

        X_test, y_test, close_prices, entry_indices = evaluator.prepare_data('data.csv')

        assert mock_read_csv.called

    def test_get_probabilities(self, evaluator):
        """Test getting probabilities from model."""
        # Mock model forward
        mock_logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        evaluator.model.return_value = mock_logits
        
        X_test = np.random.randn(2, 30, 3)
        
        with patch('torch.from_numpy') as mock_from_numpy:
            mock_tensor = torch.FloatTensor(X_test)
            mock_from_numpy.return_value = mock_tensor
            
            probs = evaluator.get_probabilities(X_test)
            
            assert probs.shape[0] == 2
            assert probs.shape[1] == 3  # 3 classes

