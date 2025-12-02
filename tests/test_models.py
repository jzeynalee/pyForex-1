# tests/test_models.py
"""
Unit tests for ML model modules.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch
from models.lstm import LSTMModel, LSTMWithAttention
from models.fusion import FusionNet, SimpleFusion, AttentionFusion
from models.trend_classifier import (
    TrendClassifier, TrendClassifierConfig, 
    generate_synthetic_training_data
)
from models.yolo_detector import MockYOLODetector

@pytest.mark.unit
class TestLSTMModel:
    """Test suite for LSTM model."""
    
    def test_init_default(self):
        """Test LSTM initialization with defaults."""
        model = LSTMModel()
        
        assert model.hidden_dim == 64
        assert model.num_layers == 2
        assert model.feature_dim == 64
    
    def test_init_custom(self):
        """Test LSTM initialization with custom params."""
        model = LSTMModel(
            input_dim=10,
            hidden_dim=128,
            num_layers=3,
            num_classes=5,
            bidirectional=True,
        )
        
        assert model.hidden_dim == 128
        assert model.feature_dim == 256  # bidirectional doubles it
    
    def test_forward_features_mode(self):
        """Test LSTM forward pass in features mode."""
        model = LSTMModel()
        model.eval()
        
        batch_size = 4
        seq_len = 60
        input_dim = 5
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        with torch.no_grad():
            features = model(x, mode='features')
        
        assert features.shape == (batch_size, 64)
    
    def test_forward_classify_mode(self):
        """Test LSTM forward pass in classify mode."""
        model = LSTMModel(num_classes=3)
        model.eval()
        
        x = torch.randn(4, 60, 5)
        
        with torch.no_grad():
            logits = model(x, mode='classify')
        
        assert logits.shape == (4, 3)
    
    def test_invalid_mode(self):
        """Test that invalid mode raises error."""
        model = LSTMModel()
        x = torch.randn(1, 60, 5)
        
        with pytest.raises(ValueError, match="Unknown mode"):
            model(x, mode='invalid')
    
    def test_get_feature_dim(self):
        """Test feature dimension getter."""
        model = LSTMModel(hidden_dim=128)
        assert model.get_feature_dim() == 128
        
        model_bi = LSTMModel(hidden_dim=128, bidirectional=True)
        assert model_bi.get_feature_dim() == 256

@pytest.mark.unit
class TestLSTMWithAttention:
    """Test suite for LSTM with attention model."""
    
    def test_forward(self):
        """Test forward pass."""
        model = LSTMWithAttention()
        model.eval()
        
        x = torch.randn(4, 60, 5)
        
        with torch.no_grad():
            features = model(x, mode='features')
        
        assert features.shape == (4, 64)
    
    def test_classify_mode(self):
        """Test classification mode."""
        model = LSTMWithAttention(num_classes=3)
        model.eval()
        
        x = torch.randn(4, 60, 5)
        
        with torch.no_grad():
            logits = model(x, mode='classify')
        
        assert logits.shape == (4, 3)

@pytest.mark.unit
class TestFusionNet:
    """Test suite for FusionNet model."""
    
    def test_init(self):
        """Test FusionNet initialization."""
        model = FusionNet()
        
        assert model.lstm_dim == 64
        assert model.vit_dim == 768
        assert model.yolo_dim == 20
    
    def test_forward(self):
        """Test FusionNet forward pass."""
        model = FusionNet()
        model.eval()
        
        batch_size = 4
        lstm_feat = torch.randn(batch_size, 64)
        vit_feat = torch.randn(batch_size, 768)
        yolo_feat = torch.randn(batch_size, 20)
        
        with torch.no_grad():
            logits = model(lstm_feat, vit_feat, yolo_feat)
        
        assert logits.shape == (batch_size, 3)
    
    def test_forward_with_gates(self):
        """Test forward pass that returns gate weights."""
        model = FusionNet()
        model.eval()
        
        lstm_feat = torch.randn(4, 64)
        vit_feat = torch.randn(4, 768)
        yolo_feat = torch.randn(4, 20)
        
        with torch.no_grad():
            logits, gates = model.forward_with_gates(lstm_feat, vit_feat, yolo_feat)
        
        assert logits.shape == (4, 3)
        assert gates.shape == (4, 3)
        
        # Gates should sum to 1 (softmax output)
        gate_sums = gates.sum(dim=1)
        assert torch.allclose(gate_sums, torch.ones(4), atol=1e-5)

@pytest.mark.unit
class TestSimpleFusion:
    """Test suite for SimpleFusion model."""
    
    def test_forward(self):
        """Test SimpleFusion forward pass."""
        model = SimpleFusion()
        model.eval()
        
        lstm_feat = torch.randn(4, 64)
        vit_feat = torch.randn(4, 768)
        yolo_feat = torch.randn(4, 20)
        
        with torch.no_grad():
            logits = model(lstm_feat, vit_feat, yolo_feat)
        
        assert logits.shape == (4, 3)

@pytest.mark.unit
class TestAttentionFusion:
    """Test suite for AttentionFusion model."""
    
    def test_forward(self):
        """Test AttentionFusion forward pass."""
        model = AttentionFusion()
        model.eval()
        
        lstm_feat = torch.randn(4, 64)
        vit_feat = torch.randn(4, 768)
        yolo_feat = torch.randn(4, 20)
        
        with torch.no_grad():
            logits = model(lstm_feat, vit_feat, yolo_feat)
        
        assert logits.shape == (4, 3)

@pytest.mark.unit
class TestTrendClassifier:
    """Test suite for TrendClassifier."""
    
    def test_init_default(self):
        """Test TrendClassifier initialization."""
        classifier = TrendClassifier()
        
        assert classifier.is_fitted == False
        assert classifier.model is None
    
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = TrendClassifierConfig(n_estimators=50, max_depth=3)
        classifier = TrendClassifier(config)
        
        assert classifier.config.n_estimators == 50
        assert classifier.config.max_depth == 3
    
    def test_generate_synthetic_data(self):
        """Test synthetic training data generation."""
        X, y = generate_synthetic_training_data(n_samples=100)
        
        assert X.shape == (100, 13)
        assert y.shape == (100,)
        assert set(np.unique(y)).issubset({-1, 0, 1})
    
    def test_fit(self):
        """Test model fitting."""
        X, y = generate_synthetic_training_data(n_samples=500)
        
        classifier = TrendClassifier()
        metrics = classifier.fit(X, y, validate=False)
        
        assert classifier.is_fitted == True
        assert 'test_accuracy' in metrics
    
    def test_predict_proba(self):
        """Test probability prediction."""
        X, y = generate_synthetic_training_data(n_samples=500)
        
        classifier = TrendClassifier()
        classifier.fit(X, y, validate=False)
        
        probs = classifier.predict_proba(X[:10])
        
        assert probs.shape == (10, 3)
        # Probabilities should sum to 1
        assert np.allclose(probs.sum(axis=1), 1.0)
    
    def test_predict(self):
        """Test class prediction."""
        X, y = generate_synthetic_training_data(n_samples=500)
        
        classifier = TrendClassifier()
        classifier.fit(X, y, validate=False)
        
        predictions = classifier.predict(X[:10])
        
        assert len(predictions) == 10
        # Predictions should be -1, 0, or 1
        assert all(p in [-1, 0, 1] for p in predictions)
    
    def test_predict_without_fit(self):
        """Test that prediction fails without fitting."""
        classifier = TrendClassifier()
        
        with pytest.raises(RuntimeError, match="Model not fitted"):
            classifier.predict_proba([[0] * 13])
    
    def test_save_load(self, tmp_path):
        """Test model persistence."""
        X, y = generate_synthetic_training_data(n_samples=500)
        
        classifier = TrendClassifier()
        classifier.fit(X, y, validate=False)
        
        model_path = tmp_path / "test_model.joblib"
        classifier.save(model_path)
        
        loaded = TrendClassifier.load(model_path)
        
        assert loaded.is_fitted == True
        
        # Predictions should match
        orig_pred = classifier.predict(X[:5])
        loaded_pred = loaded.predict(X[:5])
        
        assert np.array_equal(orig_pred, loaded_pred)
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        X, y = generate_synthetic_training_data(n_samples=500)
        
        classifier = TrendClassifier()
        classifier.fit(X, y, validate=False)
        
        importance = classifier.get_feature_importance()
        
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert len(importance) == 13

    @pytest.mark.parametrize("n_samples,expected_shape", [
        (100, (100, 13)),
        (1000, (1000, 13)),
        (1, (1, 13)),
    ])
    def test_synthetic_data_shapes(self, n_samples, expected_shape):
        """Test synthetic data generation with various sizes."""
        X, y = generate_synthetic_training_data(n_samples=n_samples)
        assert X.shape == expected_shape
        assert y.shape == (n_samples,)
    
    @pytest.mark.parametrize("probs,expected_direction", [
        ([0.1, 0.2, 0.7], 1),   # Bullish
        ([0.7, 0.2, 0.1], -1),  # Bearish
        ([0.3, 0.4, 0.3], 0),   # Sideways (argmax=1, 1-1=0)
    ])

    def test_predict_direction_mapping(self, probs, expected_direction):
        """Test that prediction correctly maps to directions."""
        X, y = generate_synthetic_training_data(n_samples=500)
        classifier = TrendClassifier()
        classifier.fit(X, y, validate=False)        
        
        with patch.object(classifier.model, 'predict_proba', return_value=np.array([probs])):
            # Re-scale since scaler expects scaled input
            with patch.object(classifier.scaler, 'transform', return_value=np.array([[0]*13])):
                prediction = classifier.predict([[0]*13])
                assert prediction[0] == expected_direction

@pytest.mark.unit
class TestMockYOLODetector:
    """Test suite for MockYOLODetector."""
    
    def test_init(self):
        """Test MockYOLODetector initialization."""
        detector = MockYOLODetector()
        
        assert detector.num_classes == 20
        assert detector.feature_dim == 20
    
    def test_detect(self):
        """Test detection output."""
        detector = MockYOLODetector()
        
        # Create dummy image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        features = detector.detect(img)
        
        assert features.shape == (20,)
        assert features.dtype == np.float32
        
        # Values should be binary (0 or 1)
        assert all(v in [0.0, 1.0] for v in features)
    
    def test_reproducibility(self):
        """Test that detection is reproducible for same input."""
        detector = MockYOLODetector()
        
        img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        
        feat1 = detector.detect(img)
        feat2 = detector.detect(img)
        
        assert np.array_equal(feat1, feat2)
    
    def test_get_feature_dim(self):
        """Test feature dimension getter."""
        detector = MockYOLODetector(num_classes=30)
        assert detector.get_feature_dim() == 30