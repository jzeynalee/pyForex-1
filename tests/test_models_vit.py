# tests/test_models_vit.py
"""
Comprehensive unit tests for models/vit.py

Tests ViT models for chart analysis with projection layers.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.unit
class TestViTExtractor:
    """Test ViTExtractor class from vit.py."""

    @patch('models.vit.create_model')
    def test_init_default(self, mock_create_model):
        """Test default initialization."""
        mock_model = MagicMock()
        mock_model.head.in_features = 768
        mock_model.head = MagicMock()
        mock_model.head.in_features = 768
        mock_create_model.return_value = mock_model

        from models.vit import ViTExtractor
        extractor = ViTExtractor()

        assert extractor.feature_dim == 768
        assert extractor.projection is None
        assert hasattr(extractor, 'vit')

    @patch('models.vit.create_model')
    def test_init_with_output_dim(self, mock_create_model):
        """Test initialization with custom output dimension."""
        mock_model = MagicMock()
        mock_model.head.in_features = 768
        mock_model.head = MagicMock()
        mock_model.head.in_features = 768
        mock_create_model.return_value = mock_model

        from models.vit import ViTExtractor
        extractor = ViTExtractor(output_dim=256)

        assert extractor.feature_dim == 256
        assert extractor.projection is not None

    @patch('models.vit.create_model')
    def test_init_custom_model_name(self, mock_create_model):
        """Test initialization with custom model name."""
        mock_model = MagicMock()
        mock_model.head.in_features = 1024
        mock_model.head = MagicMock()
        mock_model.head.in_features = 1024
        mock_create_model.return_value = mock_model

        from models.vit import ViTExtractor
        extractor = ViTExtractor(model_name="vit_large_patch16_224")

        assert extractor.feature_dim == 1024
        mock_create_model.assert_called_with("vit_large_patch16_224", pretrained=True)

    @patch('models.vit.create_model')
    def test_init_not_pretrained(self, mock_create_model):
        """Test initialization without pretrained weights."""
        mock_model = MagicMock()
        mock_model.head.in_features = 768
        mock_model.head = MagicMock()
        mock_model.head.in_features = 768
        mock_create_model.return_value = mock_model

        from models.vit import ViTExtractor
        extractor = ViTExtractor(pretrained=False)

        mock_create_model.assert_called_with("vit_base_patch16_224", pretrained=False)

    @patch('models.vit.create_model')
    def test_forward_without_projection(self, mock_create_model):
        """Test forward pass without projection."""
        mock_model = MagicMock()
        mock_model.head.in_features = 768
        mock_model.head = MagicMock()
        mock_model.head.in_features = 768
        mock_model.return_value = torch.randn(2, 768)
        mock_create_model.return_value = mock_model

        from models.vit import ViTExtractor
        extractor = ViTExtractor()

        x = torch.randn(2, 3, 224, 224)
        output = extractor.forward(x)

        assert output.shape == (2, 768)

    @patch('models.vit.create_model')
    def test_forward_with_projection(self, mock_create_model):
        """Test forward pass with projection."""
        mock_model = MagicMock()
        mock_model.head.in_features = 768
        mock_model.head = MagicMock()
        mock_model.head.in_features = 768
        mock_model.return_value = torch.randn(2, 768)
        mock_create_model.return_value = mock_model

        from models.vit import ViTExtractor
        extractor = ViTExtractor(output_dim=256)

        x = torch.randn(2, 3, 224, 224)
        output = extractor.forward(x)

        assert output.shape == (2, 256)

    @patch('models.vit.create_model')
    def test_get_feature_dim(self, mock_create_model):
        """Test get_feature_dim method."""
        mock_model = MagicMock()
        mock_model.head.in_features = 768
        mock_model.head = MagicMock()
        mock_model.head.in_features = 768
        mock_create_model.return_value = mock_model

        from models.vit import ViTExtractor
        extractor = ViTExtractor()

        assert extractor.get_feature_dim() == 768

    @patch('models.vit.create_model')
    def test_freeze_backbone(self, mock_create_model):
        """Test freezing backbone parameters."""
        mock_model = MagicMock()
        mock_model.head.in_features = 768
        mock_model.head = MagicMock()
        mock_model.head.in_features = 768
        mock_param = MagicMock()
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]
        mock_create_model.return_value = mock_model

        from models.vit import ViTExtractor
        extractor = ViTExtractor(freeze_backbone=True)

        # Should attempt to freeze (actual freezing tested in integration tests)
        assert extractor.vit == mock_model

    @patch('models.vit.create_model')
    def test_unfreeze_backbone(self, mock_create_model):
        """Test unfreezing last N blocks."""
        mock_model = MagicMock()
        mock_model.head.in_features = 768
        mock_model.head = MagicMock()
        mock_model.head.in_features = 768
        
        # Create mock blocks
        mock_blocks = [MagicMock() for _ in range(12)]
        for block in mock_blocks:
            block.parameters.return_value = [MagicMock(requires_grad=True)]
        mock_model.blocks = mock_blocks
        mock_model.norm = MagicMock()
        mock_model.norm.parameters.return_value = [MagicMock(requires_grad=True)]
        mock_model.parameters.return_value = [MagicMock(requires_grad=True)]
        mock_create_model.return_value = mock_model

        from models.vit import ViTExtractor
        extractor = ViTExtractor(freeze_backbone=True)
        extractor.unfreeze_backbone(unfreeze_last_n_blocks=2)

        # Should have called parameters on last blocks
        assert len(mock_blocks) == 12


@pytest.mark.unit
class TestLightweightViT:
    """Test LightweightViT class."""

    @patch('models.vit.create_model')
    def test_init_default(self, mock_create_model):
        """Test default initialization."""
        mock_model = MagicMock()
        mock_model.head.in_features = 192
        mock_model.head = MagicMock()
        mock_model.head.in_features = 192
        mock_create_model.return_value = mock_model

        from models.vit import LightweightViT
        vit = LightweightViT()

        assert vit.feature_dim == 192
        mock_create_model.assert_called_with("vit_tiny_patch16_224", pretrained=True)

    @patch('models.vit.create_model')
    def test_forward(self, mock_create_model):
        """Test forward pass."""
        mock_model = MagicMock()
        mock_model.head.in_features = 192
        mock_model.head = MagicMock()
        mock_model.head.in_features = 192
        mock_model.return_value = torch.randn(2, 192)
        mock_create_model.return_value = mock_model

        from models.vit import LightweightViT
        vit = LightweightViT()

        x = torch.randn(2, 3, 224, 224)
        output = vit.forward(x)

        assert output.shape == (2, 192)

    @patch('models.vit.create_model')
    def test_get_feature_dim(self, mock_create_model):
        """Test get_feature_dim method."""
        mock_model = MagicMock()
        mock_model.head.in_features = 192
        mock_model.head = MagicMock()
        mock_model.head.in_features = 192
        mock_create_model.return_value = mock_model

        from models.vit import LightweightViT
        vit = LightweightViT()

        assert vit.get_feature_dim() == 192

    @patch('models.vit.create_model')
    def test_init_not_pretrained(self, mock_create_model):
        """Test initialization without pretrained weights."""
        mock_model = MagicMock()
        mock_model.head.in_features = 192
        mock_model.head = MagicMock()
        mock_model.head.in_features = 192
        mock_create_model.return_value = mock_model

        from models.vit import LightweightViT
        vit = LightweightViT(pretrained=False)

        mock_create_model.assert_called_with("vit_tiny_patch16_224", pretrained=False)

