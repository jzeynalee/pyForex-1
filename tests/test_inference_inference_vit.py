# tests/test_inference_inference_vit.py
"""
Comprehensive unit tests for inference/inference_vit.py

Tests ViT inference script functionality.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import json


@pytest.mark.unit
class TestGetArgs:
    """Test argument parsing."""

    @patch('inference.inference_vit.argparse.ArgumentParser')
    def test_get_args(self, mock_parser_class):
        """Test argument parser setup."""
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_args = Mock()
        mock_args.image = "test.jpg"
        mock_args.class_map = "map.json"
        mock_args.checkpoint = "model.pth"
        mock_args.device = "cpu"
        mock_parser.parse_args.return_value = mock_args

        from inference.inference_vit import get_args
        args = get_args()

        assert args.image == "test.jpg"
        assert args.checkpoint == "model.pth"
        assert args.device == "cpu"


@pytest.mark.unit
class TestLoadModel:
    """Test model loading function."""

    @patch('inference.inference_vit.ViTExtractor')
    @patch('inference.inference_vit.torch.load')
    @patch('inference.inference_vit.nn.Sequential')
    def test_load_model(self, mock_sequential, mock_torch_load, mock_vit_class):
        """Test loading ViT model and classifier."""
        mock_vit = MagicMock()
        mock_vit_class.return_value = mock_vit
        
        mock_classifier = MagicMock()
        mock_sequential.return_value = mock_classifier
        
        mock_state = {"0.weight": torch.randn(10, 768), "0.bias": torch.randn(10)}
        mock_torch_load.return_value = mock_state

        from inference.inference_vit import load_model
        vit, classifier = load_model(num_classes=10, checkpoint="model.pth", device="cpu")

        assert vit is not None
        assert classifier is not None
        mock_vit_class.assert_called_once()
        mock_sequential.assert_called_once()

    @patch('inference.inference_vit.ViTExtractor')
    @patch('inference.inference_vit.torch.load')
    @patch('inference.inference_vit.nn.Sequential')
    def test_load_model_cuda(self, mock_sequential, mock_torch_load, mock_vit_class):
        """Test loading model on CUDA device."""
        mock_vit = MagicMock()
        mock_vit.to.return_value = mock_vit
        mock_vit_class.return_value = mock_vit
        
        mock_classifier = MagicMock()
        mock_classifier.to.return_value = mock_classifier
        mock_sequential.return_value = mock_classifier
        
        mock_state = {"0.weight": torch.randn(3, 768)}
        mock_torch_load.return_value = mock_state

        from inference.inference_vit import load_model
        vit, classifier = load_model(num_classes=3, checkpoint="model.pth", device="cuda")

        mock_vit.to.assert_called()
        mock_classifier.to.assert_called()


@pytest.mark.unit
class TestPreprocess:
    """Test image preprocessing."""

    @patch('inference.inference_vit.Image.open')
    @patch('inference.inference_vit.transforms.Compose')
    def test_preprocess(self, mock_compose, mock_image_open):
        """Test image preprocessing."""
        mock_img = Mock()
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value = mock_img
        
        mock_transform = Mock()
        mock_transform.return_value = torch.randn(1, 3, 224, 224)
        mock_compose.return_value = mock_transform

        from inference.inference_vit import preprocess
        result = preprocess("test.jpg")

        assert result.shape == (1, 3, 224, 224) or result.shape == (1, 1, 3, 224, 224)
        mock_image_open.assert_called_once_with("test.jpg")

    @patch('inference.inference_vit.Image.open')
    @patch('inference.inference_vit.transforms.Compose')
    def test_preprocess_converts_rgb(self, mock_compose, mock_image_open):
        """Test that image is converted to RGB."""
        mock_img = Mock()
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value = mock_img
        
        mock_transform = Mock()
        mock_transform.return_value = torch.randn(1, 3, 224, 224)
        mock_compose.return_value = mock_transform

        from inference.inference_vit import preprocess
        preprocess("test.jpg")

        mock_img.convert.assert_called_once_with("RGB")


@pytest.mark.unit
class TestPredict:
    """Test prediction function."""

    def test_predict(self):
        """Test prediction with mocked models."""
        mock_vit = MagicMock()
        mock_vit.return_value = torch.randn(1, 768)
        mock_vit.eval.return_value = mock_vit
        
        mock_classifier = MagicMock()
        logits = torch.randn(1, 3)
        logits[0, 1] = 5.0  # Make class 1 the highest
        mock_classifier.return_value = logits
        
        class_names = {"0": "bear", "1": "sideways", "2": "bull"}

        from inference.inference_vit import predict
        class_name, prob = predict(mock_vit, mock_classifier, torch.randn(1, 3, 224, 224), 
                                   class_names, "cpu")

        assert class_name in ["bear", "sideways", "bull"]
        assert 0 <= prob <= 1

    def test_predict_with_softmax(self):
        """Test prediction applies softmax correctly."""
        mock_vit = MagicMock()
        mock_vit.return_value = torch.randn(1, 768)
        
        mock_classifier = MagicMock()
        # Raw logits
        raw_logits = torch.tensor([[2.0, 5.0, 1.0]])
        mock_classifier.return_value = raw_logits
        
        class_names = {"0": "bear", "1": "sideways", "2": "bull"}

        from inference.inference_vit import predict
        class_name, prob = predict(mock_vit, mock_classifier, torch.randn(1, 3, 224, 224), 
                                   class_names, "cpu")

        # Should predict class 1 (sideways) with highest probability
        assert class_name == "sideways"
        assert prob > 0.5  # Should be the highest probability


@pytest.mark.unit
class TestMain:
    """Test main function."""

    @patch('inference.inference_vit.get_args')
    @patch('inference.inference_vit.load_model')
    @patch('inference.inference_vit.preprocess')
    @patch('inference.inference_vit.predict')
    @patch('builtins.open', new_callable=mock_open, read_data='{"0": "bear", "1": "sideways", "2": "bull"}')
    def test_main(self, mock_file, mock_predict, mock_preprocess, mock_load_model, mock_get_args):
        """Test main execution flow."""
        mock_args = Mock()
        mock_args.image = "test.jpg"
        mock_args.class_map = "map.json"
        mock_args.checkpoint = "model.pth"
        mock_args.device = "cpu"
        mock_get_args.return_value = mock_args
        
        mock_vit = Mock()
        mock_classifier = Mock()
        mock_load_model.return_value = (mock_vit, mock_classifier)
        
        mock_img_tensor = torch.randn(1, 3, 224, 224)
        mock_preprocess.return_value = mock_img_tensor
        
        mock_predict.return_value = ("sideways", 0.85)

        from inference.inference_vit import main
        main()

        mock_get_args.assert_called_once()
        mock_load_model.assert_called_once()
        mock_preprocess.assert_called_once_with("test.jpg")
        mock_predict.assert_called_once()

