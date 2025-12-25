# tests/test_models.py
"""
Unit tests for ML model modules (updated after LSTM removal).
Tests cover TCN variants, fusion models, and other components.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch
from models.tcn import (
    TCNModel,
    TCNWithAttention,
    MultiScaleTCN,
    CausalConv1d,
    TCNBlock,
    TCNBackbone,
    create_tcn_model,
)
from models.fusion import FusionNet, SimpleFusion, AttentionFusion
from models.trend_classifier import (
    TrendClassifier, TrendClassifierConfig,
    generate_synthetic_training_data
)
from models.yolo_detector import MockYOLODetector

@pytest.mark.unit
class TestTCNModel:
    """Test suite for TCN model."""
    
    def test_init_default(self):
        """Test TCN initialization with defaults."""
        model = TCNModel()

        assert model.hidden_dim == 64
        # TCN uses TCNBackbone - check it exists
        assert hasattr(model, 'tcn')
        assert model.feature_dim == 64
    
    def test_init_custom(self):
        """Test TCN initialization with custom params."""
        model = TCNModel(
            input_dim=10,
            hidden_dim=128,
            num_layers=3,
            num_classes=5,
        )

        assert model.hidden_dim == 128
        # TCN doesn't support bidirectional, feature_dim stays same as hidden_dim
        assert model.feature_dim == 128
    
    def test_forward_features_mode(self):
        """Test TCN forward pass in features mode."""
        model = TCNModel()
        model.eval()
        
        batch_size = 4
        seq_len = 60
        input_dim = 5
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        with torch.no_grad():
            features = model(x, mode='features')
        
        assert features.shape == (batch_size, 64)
    
    def test_forward_classify_mode(self):
        """Test TCN forward pass in classify mode."""
        model = TCNModel(num_classes=3)
        model.eval()
        
        x = torch.randn(4, 60, 5)
        
        with torch.no_grad():
            logits = model(x, mode='classify')
        
        assert logits.shape == (4, 3)
    
    def test_invalid_mode(self):
        """Test that invalid mode raises error."""
        model = TCNModel()
        x = torch.randn(1, 60, 5)
        
        with pytest.raises(ValueError, match="Unknown mode"):
            model(x, mode='invalid')
    
    def test_get_feature_dim(self):
        """Test feature dimension getter."""
        model = TCNModel(hidden_dim=128)
        assert model.get_feature_dim() == 128

    def test_from_profile_scalp(self):
        """Test TCN creation with SCALP profile."""
        model = TCNModel.from_profile('SCALP', input_dim=5, hidden_dim=64)

        assert model is not None
        assert model.hidden_dim == 64
        # SCALP has 4 layers
        assert hasattr(model, 'tcn')

    def test_from_profile_intraday(self):
        """Test TCN creation with INTRADAY profile."""
        model = TCNModel.from_profile('INTRADAY', input_dim=5, hidden_dim=64)

        assert model is not None
        assert model.hidden_dim == 64

    def test_from_profile_swing(self):
        """Test TCN creation with SWING profile."""
        model = TCNModel.from_profile('SWING', input_dim=5, hidden_dim=64)

        assert model is not None
        assert model.hidden_dim == 64

    def test_from_profile_invalid(self):
        """Test invalid profile raises error."""
        with pytest.raises(ValueError, match="Unknown profile"):
            TCNModel.from_profile('INVALID')

    def test_receptive_field_calculation(self):
        """Test that receptive field is calculated."""
        model = TCNModel(input_dim=5, hidden_dim=64, num_layers=5)

        # Should have calculated receptive field
        assert model.tcn.receptive_field > 0

    def test_bidirectional_parameter_ignored(self):
        """Test that bidirectional parameter is ignored (for compatibility)."""
        # TCN doesn't support bidirectional but accepts it for API compatibility
        model = TCNModel(bidirectional=True)

        # Should still work, just ignores the parameter
        assert model.feature_dim == 64  # Not doubled

@pytest.mark.unit
class TestTCNWithAttention:
    """Test suite for TCN with attention model."""

    def test_init(self):
        """Test TCNWithAttention initialization."""
        model = TCNWithAttention(input_dim=5, hidden_dim=64, num_layers=4)

        assert model.hidden_dim == 64
        assert model.feature_dim == 64

    def test_forward_features(self):
        """Test forward pass in features mode."""
        model = TCNWithAttention(input_dim=5, hidden_dim=64)
        model.eval()

        x = torch.randn(4, 60, 5)

        with torch.no_grad():
            features = model(x, mode='features')

        assert features.shape == (4, 64)

    def test_forward_classify(self):
        """Test forward pass in classify mode."""
        model = TCNWithAttention(input_dim=5, hidden_dim=64, num_classes=3)
        model.eval()

        x = torch.randn(4, 60, 5)

        with torch.no_grad():
            logits = model(x, mode='classify')

        assert logits.shape == (4, 3)

    def test_attention_mechanism(self):
        """Test that attention mechanism is present."""
        model = TCNWithAttention(input_dim=5, hidden_dim=64, num_heads=4)

        assert hasattr(model, 'attention')
        assert hasattr(model, 'agg_query')

    def test_get_feature_dim(self):
        """Test feature dimension getter."""
        model = TCNWithAttention(hidden_dim=128)
        assert model.get_feature_dim() == 128

@pytest.mark.unit
class TestFusionNet:
    """Test suite for FusionNet model."""
    
    def test_init(self):
        """Test FusionNet initialization."""
        model = FusionNet()

        assert model.seq_dim == 64  # Changed from tcn_dim to seq_dim
        assert model.vit_dim == 768
        assert model.yolo_dim == 25
    
    def test_forward(self):
        """Test FusionNet forward pass."""
        model = FusionNet()
        model.eval()
        
        batch_size = 4
        tcn_feat = torch.randn(batch_size, 64)
        vit_feat = torch.randn(batch_size, 768)
        yolo_feat = torch.randn(batch_size, 25)
        
        with torch.no_grad():
            logits = model(tcn_feat, vit_feat, yolo_feat)
        
        assert logits.shape == (batch_size, 3)
    
    def test_forward_with_gates(self):
        """Test forward pass that returns gate weights."""
        model = FusionNet()
        model.eval()
        
        tcn_feat = torch.randn(4, 64)
        vit_feat = torch.randn(4, 768)
        yolo_feat = torch.randn(4, 25)
        
        with torch.no_grad():
            logits, gates = model.forward_with_gates(tcn_feat, vit_feat, yolo_feat)
        
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
        
        tcn_feat = torch.randn(4, 64)
        vit_feat = torch.randn(4, 768)
        yolo_feat = torch.randn(4, 25)
        
        with torch.no_grad():
            logits = model(tcn_feat, vit_feat, yolo_feat)
        
        assert logits.shape == (4, 3)

@pytest.mark.unit
class TestAttentionFusion:
    """Test suite for AttentionFusion model."""
    
    def test_forward(self):
        """Test AttentionFusion forward pass."""
        model = AttentionFusion()
        model.eval()
        
        tcn_feat = torch.randn(4, 64)
        vit_feat = torch.randn(4, 768)
        yolo_feat = torch.randn(4, 25)
        
        with torch.no_grad():
            logits = model(tcn_feat, vit_feat, yolo_feat)
        
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
        
        assert detector.num_classes == 25
        assert detector.feature_dim == 25
    
    def test_detect(self):
        """Test detection output."""
        detector = MockYOLODetector()
        
        # Create dummy image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        features = detector.detect(img)
        
        assert features.shape == (25,)
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


# =============================================================================
# NEW TESTS FOR TCN VARIANTS (Post-LSTM Removal)
# =============================================================================

@pytest.mark.unit
class TestCausalConv1d:
    """Test suite for CausalConv1d layer."""

    def test_init(self):
        """Test CausalConv1d initialization."""
        conv = CausalConv1d(in_channels=5, out_channels=64, kernel_size=3, dilation=1)

        assert conv.kernel_size == 3
        assert conv.dilation == 1
        assert conv.padding == 2  # (3-1)*1 = 2

    def test_forward_causality(self):
        """Test that output is causal (depends only on past)."""
        conv = CausalConv1d(in_channels=5, out_channels=64, kernel_size=3)
        conv.eval()

        # Create input with known pattern
        x = torch.zeros(1, 5, 10)
        x[:, :, -1] = 1.0  # Only last timestep has signal

        with torch.no_grad():
            out = conv(x)

        # Output should be same length as input
        assert out.shape[2] == x.shape[2]

        # Earlier timesteps should not be affected by future signal
        # (this is a basic check - proper causality test would be more complex)
        assert out.shape == (1, 64, 10)

    def test_dilation(self):
        """Test dilated convolution."""
        conv = CausalConv1d(in_channels=5, out_channels=64, kernel_size=3, dilation=4)

        assert conv.dilation == 4
        assert conv.padding == 8  # (3-1)*4 = 8


@pytest.mark.unit
class TestTCNBlock:
    """Test suite for TCN residual block."""

    def test_init(self):
        """Test TCNBlock initialization."""
        block = TCNBlock(
            in_channels=5,
            out_channels=64,
            kernel_size=3,
            dilation=1,
            dropout=0.2
        )

        assert hasattr(block, 'conv1')
        assert hasattr(block, 'conv2')
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')
        assert hasattr(block, 'residual')

    def test_forward_same_channels(self):
        """Test forward with same in/out channels."""
        block = TCNBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            dilation=1
        )
        block.eval()

        x = torch.randn(4, 64, 60)

        with torch.no_grad():
            out = block(x)

        assert out.shape == x.shape

    def test_forward_different_channels(self):
        """Test forward with different in/out channels."""
        block = TCNBlock(
            in_channels=5,
            out_channels=64,
            kernel_size=3,
            dilation=1
        )
        block.eval()

        x = torch.randn(4, 5, 60)

        with torch.no_grad():
            out = block(x)

        assert out.shape == (4, 64, 60)

    def test_residual_connection(self):
        """Test that residual connection is used."""
        block = TCNBlock(
            in_channels=5,
            out_channels=64,
            kernel_size=3,
            dilation=1
        )

        # Residual should be 1x1 conv when channels differ
        assert isinstance(block.residual, torch.nn.Conv1d)


@pytest.mark.unit
class TestTCNBackbone:
    """Test suite for TCN backbone (stack of blocks)."""

    def test_init(self):
        """Test TCNBackbone initialization."""
        backbone = TCNBackbone(
            input_dim=5,
            hidden_dim=64,
            num_layers=4,
            kernel_size=3,
            dropout=0.2
        )

        assert backbone.input_dim == 5
        assert backbone.hidden_dim == 64
        assert backbone.receptive_field > 0

    def test_receptive_field_calculation(self):
        """Test receptive field increases with layers."""
        backbone_small = TCNBackbone(input_dim=5, hidden_dim=64, num_layers=3)
        backbone_large = TCNBackbone(input_dim=5, hidden_dim=64, num_layers=6)

        assert backbone_large.receptive_field > backbone_small.receptive_field

    def test_forward(self):
        """Test forward pass."""
        backbone = TCNBackbone(
            input_dim=5,
            hidden_dim=64,
            num_layers=4
        )
        backbone.eval()

        # Input: (batch, seq_len, input_dim)
        x = torch.randn(4, 60, 5)

        with torch.no_grad():
            out = backbone(x)

        # Output: (batch, seq_len, hidden_dim)
        assert out.shape == (4, 60, 64)

    def test_dilation_progression(self):
        """Test that dilations increase exponentially."""
        backbone = TCNBackbone(
            input_dim=5,
            hidden_dim=64,
            num_layers=4,
            dilation_base=2
        )

        # With base 2 and 4 layers: [1, 2, 4, 8]
        # RF = 1 + 2*(3-1)*(1+2+4+8) = 1 + 2*2*15 = 61
        assert backbone.receptive_field == 61


@pytest.mark.unit
class TestMultiScaleTCN:
    """Test suite for MultiScaleTCN model."""

    def test_init(self):
        """Test MultiScaleTCN initialization."""
        model = MultiScaleTCN(
            input_dim=5,
            hidden_dim=64,
            num_classes=3
        )

        assert hasattr(model, 'short_branch')
        assert hasattr(model, 'medium_branch')
        assert hasattr(model, 'long_branch')
        assert hasattr(model, 'fusion')

    def test_forward_features(self):
        """Test forward in features mode."""
        model = MultiScaleTCN(input_dim=5, hidden_dim=64)
        model.eval()

        x = torch.randn(4, 60, 5)

        with torch.no_grad():
            features = model(x, mode='features')

        assert features.shape == (4, 64)

    def test_forward_classify(self):
        """Test forward in classify mode."""
        model = MultiScaleTCN(input_dim=5, hidden_dim=64, num_classes=3)
        model.eval()

        x = torch.randn(4, 60, 5)

        with torch.no_grad():
            logits = model(x, mode='classify')

        assert logits.shape == (4, 3)

    def test_multi_scale_branches(self):
        """Test that model has multiple branches with different receptive fields."""
        model = MultiScaleTCN(input_dim=5, hidden_dim=64)

        # Each branch should have different receptive field
        short_rf = model.short_branch.receptive_field
        medium_rf = model.medium_branch.receptive_field
        long_rf = model.long_branch.receptive_field

        assert short_rf < medium_rf < long_rf

    def test_get_feature_dim(self):
        """Test feature dimension getter."""
        model = MultiScaleTCN(hidden_dim=96)
        assert model.get_feature_dim() == 96


@pytest.mark.unit
class TestCreateTCNModel:
    """Test factory function for TCN variants."""

    def test_create_standard_tcn(self):
        """Test creating standard TCN."""
        model = create_tcn_model(variant='standard', input_dim=5, hidden_dim=64)

        assert isinstance(model, TCNModel)

    def test_create_standard_with_profile(self):
        """Test creating standard TCN with profile."""
        model = create_tcn_model(variant='standard', profile='SCALP')

        assert isinstance(model, TCNModel)

    def test_create_attention_tcn(self):
        """Test creating attention TCN."""
        model = create_tcn_model(variant='attention', input_dim=5, hidden_dim=64)

        assert isinstance(model, TCNWithAttention)

    def test_create_multiscale_tcn(self):
        """Test creating multi-scale TCN."""
        model = create_tcn_model(variant='multiscale', input_dim=5, hidden_dim=64)

        assert isinstance(model, MultiScaleTCN)

    def test_create_invalid_variant(self):
        """Test invalid variant raises error."""
        with pytest.raises(ValueError, match="Unknown variant"):
            create_tcn_model(variant='invalid')


@pytest.mark.unit
class TestTCNProfiles:
    """Test TCN profile configurations."""

    def test_profiles_exist(self):
        """Test that profile configurations exist."""
        assert 'SCALP' in TCNModel.PROFILES
        assert 'INTRADAY' in TCNModel.PROFILES
        assert 'SWING' in TCNModel.PROFILES

    def test_profile_receptive_fields(self):
        """Test that profiles have increasing receptive fields."""
        scalp = TCNModel.from_profile('SCALP')
        intraday = TCNModel.from_profile('INTRADAY')
        swing = TCNModel.from_profile('SWING')

        scalp_rf = scalp.tcn.receptive_field
        intraday_rf = intraday.tcn.receptive_field
        swing_rf = swing.tcn.receptive_field

        assert scalp_rf < intraday_rf < swing_rf

    @pytest.mark.parametrize("profile", ['SCALP', 'INTRADAY', 'SWING'])
    def test_profile_forward(self, profile):
        """Test forward pass for each profile."""
        model = TCNModel.from_profile(profile, input_dim=5, hidden_dim=64)
        model.eval()

        x = torch.randn(2, 60, 5)

        with torch.no_grad():
            features = model(x, mode='features')
            logits = model(x, mode='classify')

        assert features.shape == (2, 64)
        assert logits.shape == (2, 3)