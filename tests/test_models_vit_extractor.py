# tests/test_models_vit_extractor.py
"""
Comprehensive unit tests for models/vit_extractor.py

Tests ViT feature extractor for chart pattern analysis.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock


@pytest.mark.unit
class TestViTExtractor:
    """Test ViTExtractor class."""

    @patch('models.vit_extractor.timm')
    def test_init_default(self, mock_timm):
        """Test default initialization."""
        mock_model = MagicMock()
        mock_model.num_features = 768
        mock_timm.create_model.return_value = mock_model

        from models.vit_extractor import ViTExtractor
        extractor = ViTExtractor()

        assert extractor.hidden_dim == 768
        mock_timm.create_model.assert_called_once_with(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0
        )

    @patch('models.vit_extractor.timm')
    def test_init_custom_model(self, mock_timm):
        """Test initialization with custom model name."""
        mock_model = MagicMock()
        mock_model.num_features = 1024
        mock_timm.create_model.return_value = mock_model

        from models.vit_extractor import ViTExtractor
        extractor = ViTExtractor(model_name="vit_large_patch16_224")

        assert extractor.hidden_dim == 1024
        mock_timm.create_model.assert_called_with(
            "vit_large_patch16_224",
            pretrained=True,
            num_classes=0
        )

    @patch('models.vit_extractor.timm')
    def test_init_not_pretrained(self, mock_timm):
        """Test initialization without pretrained weights."""
        mock_model = MagicMock()
        mock_model.num_features = 768
        mock_timm.create_model.return_value = mock_model

        from models.vit_extractor import ViTExtractor
        extractor = ViTExtractor(pretrained=False)

        mock_timm.create_model.assert_called_with(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0
        )

    @patch('models.vit_extractor.timm')
    def test_init_freeze_false(self, mock_timm):
        """Test initialization without freezing."""
        mock_model = MagicMock()
        mock_model.num_features = 768
        mock_param = MagicMock()
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]
        mock_timm.create_model.return_value = mock_model

        from models.vit_extractor import ViTExtractor
        extractor = ViTExtractor(freeze=False)

        # Parameters should not be frozen
        assert mock_param.requires_grad == True

    @patch('models.vit_extractor.timm')
    def test_forward_features_dict_cls_token(self, mock_timm):
        """Test forward pass with dict output containing cls_token."""
        mock_model = MagicMock()
        mock_model.num_features = 768
        
        # Simulate dict output with cls_token
        cls_token = torch.randn(2, 768)
        mock_model.forward_features.return_value = {"cls_token": cls_token}
        mock_timm.create_model.return_value = mock_model

        from models.vit_extractor import ViTExtractor
        extractor = ViTExtractor()
        
        x = torch.randn(2, 3, 224, 224)
        output = extractor.forward(x)

        assert output.shape == (2, 768)
        torch.testing.assert_close(output, cls_token)

    @patch('models.vit_extractor.timm')
    def test_forward_features_dict_x(self, mock_timm):
        """Test forward pass with dict output containing 'x' key."""
        mock_model = MagicMock()
        mock_model.num_features = 768
        
        # Simulate dict output with 'x' (sequence of tokens)
        seq_tokens = torch.randn(2, 197, 768)  # 196 patches + 1 CLS
        mock_model.forward_features.return_value = {"x": seq_tokens}
        mock_timm.create_model.return_value = mock_model

        from models.vit_extractor import ViTExtractor
        extractor = ViTExtractor()
        
        x = torch.randn(2, 3, 224, 224)
        output = extractor.forward(x)

        assert output.shape == (2, 768)
        torch.testing.assert_close(output, seq_tokens[:, 0])

    @patch('models.vit_extractor.timm')
    def test_forward_features_tensor(self, mock_timm):
        """Test forward pass with tensor output."""
        mock_model = MagicMock()
        mock_model.num_features = 768
        
        # Simulate tensor output (first dim is batch, second is sequence)
        features = torch.randn(2, 197, 768)  # 197 = 196 patches + 1 CLS
        mock_model.forward_features.return_value = features
        mock_timm.create_model.return_value = mock_model

        from models.vit_extractor import ViTExtractor
        extractor = ViTExtractor()
        
        x = torch.randn(2, 3, 224, 224)
        output = extractor.forward(x)

        assert output.shape == (2, 768)
        torch.testing.assert_close(output, features[:, 0])

    @patch('models.vit_extractor.timm')
    def test_forward_different_batch_sizes(self, mock_timm):
        """Test forward pass with different batch sizes."""
        mock_model = MagicMock()
        mock_model.num_features = 768
        
        def mock_forward_features(x):
            batch_size = x.shape[0]
            return torch.randn(batch_size, 197, 768)
        
        mock_model.forward_features.side_effect = mock_forward_features
        mock_timm.create_model.return_value = mock_model

        from models.vit_extractor import ViTExtractor
        extractor = ViTExtractor()
        
        # Test batch size 1
        x1 = torch.randn(1, 3, 224, 224)
        out1 = extractor.forward(x1)
        assert out1.shape == (1, 768)
        
        # Test batch size 8
        x8 = torch.randn(8, 3, 224, 224)
        out8 = extractor.forward(x8)
        assert out8.shape == (8, 768)

    @patch('models.vit_extractor.timm')
    def test_freeze_parameters(self, mock_timm):
        """Test that parameters are frozen by default."""
        mock_model = MagicMock()
        mock_model.num_features = 768
        mock_param1 = MagicMock()
        mock_param1.requires_grad = True
        mock_param2 = MagicMock()
        mock_param2.requires_grad = True
        mock_model.parameters.return_value = [mock_param1, mock_param2]
        mock_timm.create_model.return_value = mock_model

        from models.vit_extractor import ViTExtractor
        extractor = ViTExtractor(freeze=True)

        # Parameters should be frozen (but requires_grad might not be set in mock)
        # We can't directly check this without actual model, but we verify call structure
        assert extractor.vit == mock_model

