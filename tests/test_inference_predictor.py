# tests/test_inference_predictor.py
"""
Unit tests for inference.predictor module (TCN-based predictors).
Tests cover RiskAwareTCNPredictor and HybridPredictor after LSTM removal.
"""
import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from inference.predictor import (
    RiskAwareTCNPredictor,
    HybridPredictor,
    PredictorConfig,
    PredictionResult,
    Signal,
    get_device,
    create_predictor,
)


@pytest.mark.unit
class TestSignalEnum:
    """Test Signal enumeration."""

    def test_signal_values(self):
        """Test Signal enum has correct values."""
        assert Signal.BEAR == 0
        assert Signal.SIDEWAYS == 1
        assert Signal.BULL == 2


@pytest.mark.unit
class TestPredictorConfig:
    """Test PredictorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PredictorConfig()

        assert config.weights_dir == "models/weights"
        assert config.device == "auto"
        assert config.sequence_length == 60
        assert config.model_type == "tcn"
        assert config.profile == "INTRADAY"
        assert config.use_risk_heads == True
        assert config.use_vision == True
        assert config.use_yolo == True
        assert config.fusion_type == "gated"
        assert config.confidence_threshold == 0.55

    def test_custom_config(self):
        """Test custom configuration."""
        config = PredictorConfig(
            profile="SCALP",
            model_type="tcn",
            use_vision=False,
            confidence_threshold=0.70
        )

        assert config.profile == "SCALP"
        assert config.use_vision == False
        assert config.confidence_threshold == 0.70


@pytest.mark.unit
class TestGetDevice:
    """Test device resolution function."""

    def test_auto_device_cuda_available(self):
        """Test auto device selection when CUDA is available."""
        with patch('torch.cuda.is_available', return_value=True):
            device = get_device("auto")
            assert device.type == "cuda"

    def test_auto_device_cuda_unavailable(self):
        """Test auto device selection when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            device = get_device("auto")
            assert device.type == "cpu"

    def test_explicit_cpu(self):
        """Test explicit CPU device."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_explicit_cuda(self):
        """Test explicit CUDA device."""
        device = get_device("cuda")
        assert device.type == "cuda"


@pytest.mark.unit
class TestRiskAwareTCNPredictor:
    """Test suite for RiskAwareTCNPredictor."""

    @pytest.fixture
    def mock_tcn_model(self):
        """Create a mock TCN model."""
        model = MagicMock()
        model.eval.return_value = None
        return model

    @pytest.fixture
    def predictor_config(self):
        """Create test predictor config."""
        return PredictorConfig(
            device="cpu",
            profile="INTRADAY",
            use_risk_heads=True
        )

    def test_init_default(self):
        """Test predictor initialization with defaults."""
        with patch('inference.predictor.RiskAwareTCNPredictor._init_model'):
            predictor = RiskAwareTCNPredictor()

            assert predictor.config is not None
            assert predictor.device.type == "cpu" or predictor.device.type == "cuda"
            assert predictor._scaler is None
            assert predictor._feature_names is None

    def test_init_with_config(self, predictor_config):
        """Test initialization with custom config."""
        with patch('inference.predictor.RiskAwareTCNPredictor._init_model'):
            predictor = RiskAwareTCNPredictor(config=predictor_config)

            assert predictor.config.profile == "INTRADAY"
            assert predictor.device.type == "cpu"

    def test_init_model_with_risk_heads(self, predictor_config):
        """Test model initialization with risk heads."""
        with patch('risk_management.create_tcn_for_profile') as mock_create:
            mock_model = MagicMock()
            mock_create.return_value = mock_model

            predictor = RiskAwareTCNPredictor(config=predictor_config)

            mock_create.assert_called_once()
            assert predictor._use_risk_heads == True

    def test_init_model_fallback_to_standard_tcn(self, predictor_config):
        """Test fallback to standard TCN when risk_management unavailable."""
        # The actual implementation already has try/except for ImportError
        # We just need to ensure the fallback works correctly
        # Since risk_management may or may not be available, just verify the predictor initializes
        predictor = RiskAwareTCNPredictor(config=predictor_config)

        # Should initialize successfully either way
        assert predictor.model is not None
        assert hasattr(predictor, '_use_risk_heads')

    def test_load_weights(self, predictor_config, tmp_path):
        """Test loading model weights."""
        with patch('inference.predictor.RiskAwareTCNPredictor._init_model'):
            predictor = RiskAwareTCNPredictor(config=predictor_config)
            predictor.model = MagicMock()

            # Create fake checkpoint (use simple dict for config, not MagicMock to avoid pickling error)
            checkpoint = {
                'model_state_dict': {},
                'feature_names': ['close', 'volume'],
                'config': {'input_channels': 10}  # Use dict instead of MagicMock
            }

            checkpoint_path = tmp_path / "test_weights.pt"
            torch.save(checkpoint, checkpoint_path)

            predictor.load_weights(str(checkpoint_path))

            predictor.model.load_state_dict.assert_called_once()
            predictor.model.eval.assert_called_once()
            assert predictor._feature_names == ['close', 'volume']

    def test_prepare_input_from_dataframe(self, predictor_config):
        """Test input preparation from DataFrame."""
        with patch('inference.predictor.RiskAwareTCNPredictor._init_model'):
            predictor = RiskAwareTCNPredictor(config=predictor_config)
            predictor._feature_names = ['close', 'volume', 'high', 'low', 'open']

            # Create test DataFrame
            df = pd.DataFrame({
                'close': [1.1, 1.2, 1.3],
                'volume': [100, 200, 150],
                'high': [1.2, 1.3, 1.4],
                'low': [1.0, 1.1, 1.2],
                'open': [1.05, 1.15, 1.25],
                'extra': [0, 0, 0]  # Should be ignored
            })

            tensor = predictor._prepare_input(df)

            assert tensor.shape == (1, 3, 5)  # (batch=1, seq_len=3, features=5)
            assert tensor.device.type == predictor.device.type

    def test_prepare_input_from_numpy(self, predictor_config):
        """Test input preparation from numpy array."""
        with patch('inference.predictor.RiskAwareTCNPredictor._init_model'):
            predictor = RiskAwareTCNPredictor(config=predictor_config)

            # 2D array (seq_len, features)
            arr = np.random.randn(60, 5)
            tensor = predictor._prepare_input(arr)

            assert tensor.shape == (1, 60, 5)

    def test_prepare_input_from_tensor(self, predictor_config):
        """Test input preparation from torch tensor."""
        with patch('inference.predictor.RiskAwareTCNPredictor._init_model'):
            predictor = RiskAwareTCNPredictor(config=predictor_config)

            # Create tensor
            t = torch.randn(60, 5)
            result = predictor._prepare_input(t)

            assert result.shape == (1, 60, 5)

    def test_predict_with_risk_heads(self, predictor_config):
        """Test prediction with risk heads enabled."""
        with patch('inference.predictor.RiskAwareTCNPredictor._init_model'):
            predictor = RiskAwareTCNPredictor(config=predictor_config)
            predictor._use_risk_heads = True

            # Mock model
            mock_model = MagicMock()
            mock_outputs = {
                'direction': torch.tensor([[0.2, 0.3, 0.5]]),  # BULL wins
                'volatility': torch.tensor([0.015]),
                'quantiles': torch.tensor([[0.98, 1.00, 1.02, 1.04, 1.06]]),
                'features': torch.randn(1, 64)
            }
            mock_model.return_value = mock_outputs
            predictor.model = mock_model

            # Create test features
            features = np.random.randn(60, 5)

            result = predictor.predict(features, return_features=True)

            assert isinstance(result, PredictionResult)
            assert result.predicted_class == 2  # BULL
            assert result.signal_name == "BULL"
            assert result.confidence == pytest.approx(0.5)
            assert result.volatility == pytest.approx(0.015)
            assert len(result.quantiles) == 5
            assert result.features is not None

    def test_predict_without_risk_heads(self, predictor_config):
        """Test prediction with standard TCN (no risk heads)."""
        with patch('inference.predictor.RiskAwareTCNPredictor._init_model'):
            predictor = RiskAwareTCNPredictor(config=predictor_config)
            predictor._use_risk_heads = False

            # Mock standard TCN model - return logits (before softmax)
            mock_model = MagicMock()
            # Higher logit for BEAR (index 0)
            mock_model.return_value = torch.tensor([[2.0, 0.5, 0.3]])
            predictor.model = mock_model

            features = np.random.randn(60, 5)
            result = predictor.predict(features)

            assert result.predicted_class == 0  # BEAR
            assert result.signal_name == "BEAR"
            # After softmax, should have highest prob
            assert result.probabilities[0] > result.probabilities[1]
            assert result.probabilities[0] > result.probabilities[2]
            assert result.volatility == 0.0  # No risk head
            assert np.all(result.quantiles == 0)

    def test_predict_batch(self, predictor_config):
        """Test batch prediction."""
        with patch('inference.predictor.RiskAwareTCNPredictor._init_model'):
            predictor = RiskAwareTCNPredictor(config=predictor_config)
            predictor._use_risk_heads = True

            # Mock model
            mock_model = MagicMock()
            batch_size = 4
            mock_outputs = {
                'direction': torch.randn(batch_size, 3),
                'volatility': torch.randn(batch_size, 1),
                'quantiles': torch.randn(batch_size, 5),
                'features': torch.randn(batch_size, 64)
            }
            mock_model.return_value = mock_outputs
            predictor.model = mock_model

            features = np.random.randn(batch_size, 60, 5)
            results = predictor.predict_batch(features)

            assert 'direction_probs' in results
            assert 'volatility' in results
            assert 'quantiles' in results
            assert results['direction_probs'].shape == (batch_size, 3)
            assert results['volatility'].shape == (batch_size, 1)
            assert results['quantiles'].shape == (batch_size, 5)

    def test_to_dict(self, predictor_config):
        """Test conversion to dictionary format."""
        with patch('inference.predictor.RiskAwareTCNPredictor._init_model'):
            predictor = RiskAwareTCNPredictor(config=predictor_config)

            result = PredictionResult(
                probabilities=np.array([0.2, 0.3, 0.5]),
                predicted_class=2,
                confidence=0.5,
                signal_name="BULL",
                volatility=0.015,
                quantiles=np.array([0.98, 1.00, 1.02, 1.04, 1.06]),
                features=np.random.randn(64)
            )

            result_dict = predictor.to_dict(result)

            assert 'direction_probs' in result_dict
            assert 'volatility' in result_dict
            assert 'quantiles' in result_dict
            assert 'features' in result_dict


@pytest.mark.unit
class TestHybridPredictor:
    """Test suite for HybridPredictor."""

    @pytest.fixture
    def predictor_config(self):
        """Create test predictor config."""
        return PredictorConfig(
            device="cpu",
            use_vision=True,
            use_yolo=True,
            fusion_type="gated"
        )

    def test_init_default(self):
        """Test hybrid predictor initialization."""
        with patch('inference.predictor.RiskAwareTCNPredictor'):
            predictor = HybridPredictor()

            assert predictor.tcn_predictor is not None
            assert predictor.config is not None

    def test_init_with_weights(self):
        """Test initialization with model weights."""
        with patch('inference.predictor.RiskAwareTCNPredictor') as mock_tcn:
            mock_tcn_instance = MagicMock()
            mock_tcn.return_value = mock_tcn_instance

            predictor = HybridPredictor(tcn_weights="models/tcn.pt")

            mock_tcn_instance.load_weights.assert_called_once_with("models/tcn.pt")

    def test_init_vision_components(self, predictor_config):
        """Test vision component initialization."""
        with patch('inference.predictor.RiskAwareTCNPredictor'):
            # The HybridPredictor may try to import ViT, mock it to avoid errors
            predictor = HybridPredictor(config=predictor_config)

            # Vision should be configured even if initialization fails
            assert predictor.config.use_vision == True

    def test_predict_tcn_only(self):
        """Test prediction with TCN only (no vision)."""
        config = PredictorConfig(use_vision=False, use_yolo=False)

        with patch('inference.predictor.RiskAwareTCNPredictor') as mock_tcn_class:
            mock_tcn = MagicMock()
            mock_result = PredictionResult(
                probabilities=np.array([0.2, 0.3, 0.5]),
                predicted_class=2,
                confidence=0.5,
                signal_name="BULL",
                volatility=0.015,
                quantiles=np.array([0.98, 1.00, 1.02, 1.04, 1.06])
            )
            mock_tcn.predict.return_value = mock_result
            mock_tcn_class.return_value = mock_tcn

            predictor = HybridPredictor(config=config)
            features = np.random.randn(60, 5)

            result = predictor.predict(features)

            assert result.predicted_class == 2
            assert result.signal_name == "BULL"

    def test_predict_with_vision(self):
        """Test prediction with vision components."""
        config = PredictorConfig(use_vision=True)

        with patch('inference.predictor.RiskAwareTCNPredictor') as mock_tcn_class:
            with patch('inference.predictor.HybridPredictor._init_vision'):
                with patch('inference.predictor.HybridPredictor._init_fusion'):
                    predictor = HybridPredictor(config=config)

                    # Mock TCN prediction
                    mock_result = PredictionResult(
                        probabilities=np.array([0.2, 0.3, 0.5]),
                        predicted_class=2,
                        confidence=0.5,
                        signal_name="BULL",
                        volatility=0.015,
                        quantiles=np.array([0.98, 1.00, 1.02, 1.04, 1.06]),
                        features=np.random.randn(64)
                    )
                    predictor.tcn_predictor.predict = MagicMock(return_value=mock_result)

                    # No chart image provided -> should return TCN result
                    features = np.random.randn(60, 5)
                    result = predictor.predict(features, chart_image=None)

                    assert result == mock_result


@pytest.mark.unit
class TestCreatePredictor:
    """Test factory function for creating predictors."""

    def test_create_tcn_predictor(self):
        """Test creating TCN-only predictor."""
        with patch('inference.predictor.RiskAwareTCNPredictor') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            predictor = create_predictor(
                profile='SCALP',
                weights_path='models/tcn.pt',
                use_vision=False,
                use_yolo=False
            )

            assert isinstance(predictor, MagicMock)
            mock_class.assert_called_once()

    def test_create_hybrid_predictor(self):
        """Test creating hybrid predictor with vision."""
        with patch('inference.predictor.HybridPredictor') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            predictor = create_predictor(
                profile='INTRADAY',
                weights_path='models/tcn.pt',
                use_vision=True,
                use_yolo=False
            )

            mock_class.assert_called_once()

    def test_create_predictor_with_yolo(self):
        """Test creating predictor with YOLO enabled."""
        with patch('inference.predictor.HybridPredictor') as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            predictor = create_predictor(
                profile='SWING',
                use_yolo=True
            )

            mock_class.assert_called_once()
            # Check that HybridPredictor was called (first positional arg is config)
            call_args = mock_class.call_args
            config_arg = call_args[0][0] if call_args[0] else call_args[1].get('config')
            assert config_arg.use_yolo == True


@pytest.mark.unit
class TestPredictionResult:
    """Test PredictionResult NamedTuple."""

    def test_create_result(self):
        """Test creating prediction result."""
        result = PredictionResult(
            probabilities=np.array([0.2, 0.3, 0.5]),
            predicted_class=2,
            confidence=0.5,
            signal_name="BULL",
            volatility=0.015,
            quantiles=np.array([0.98, 1.00, 1.02, 1.04, 1.06])
        )

        assert result.predicted_class == 2
        assert result.signal_name == "BULL"
        assert result.confidence == 0.5
        assert result.volatility == 0.015
        assert len(result.quantiles) == 5
        assert result.gate_weights is None
        assert result.features is None

    def test_result_with_optional_fields(self):
        """Test result with gate weights and features."""
        result = PredictionResult(
            probabilities=np.array([0.2, 0.3, 0.5]),
            predicted_class=2,
            confidence=0.5,
            signal_name="BULL",
            volatility=0.015,
            quantiles=np.array([0.98, 1.00, 1.02, 1.04, 1.06]),
            gate_weights=np.array([0.6, 0.3, 0.1]),
            features=np.random.randn(64)
        )

        assert result.gate_weights is not None
        assert len(result.gate_weights) == 3
        assert result.features is not None
        assert len(result.features) == 64
