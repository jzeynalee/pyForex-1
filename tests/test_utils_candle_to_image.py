#!/usr/bin/env python3
"""
Tests for utils/candle_to_image.py - Candlestick image rendering.
"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image


class TestCandlestickRenderer:
    """Tests for CandlestickRenderer class."""

    def test_renderer_creation(self):
        """Test creating a renderer."""
        from utils.candle_to_image import CandlestickRenderer
        
        renderer = CandlestickRenderer()
        
        assert renderer.image_size == (224, 224)
        assert renderer.background_color == (255, 255, 255)

    def test_renderer_custom_size(self):
        """Test renderer with custom image size."""
        from utils.candle_to_image import CandlestickRenderer
        
        renderer = CandlestickRenderer(image_size=(256, 256))
        
        assert renderer.image_size == (256, 256)

    def test_renderer_custom_colors(self):
        """Test renderer with custom colors."""
        from utils.candle_to_image import CandlestickRenderer
        
        renderer = CandlestickRenderer(
            background_color=(0, 0, 0),
            bull_color=(0, 255, 0),
            bear_color=(255, 0, 0),
        )
        
        assert renderer.background_color == (0, 0, 0)
        assert renderer.bull_color == (0, 255, 0)
        assert renderer.bear_color == (255, 0, 0)

    def test_render_basic(self, small_ohlcv_df):
        """Test basic rendering."""
        from utils.candle_to_image import CandlestickRenderer
        
        renderer = CandlestickRenderer(image_size=(224, 224))
        img = renderer.render(small_ohlcv_df)
        
        assert isinstance(img, np.ndarray)
        assert img.shape == (224, 224, 3)
        assert img.dtype == np.uint8

    def test_render_empty_df(self):
        """Test rendering with empty DataFrame."""
        from utils.candle_to_image import CandlestickRenderer
        
        renderer = CandlestickRenderer()
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
        
        img = renderer.render(df)
        
        assert img.shape == (224, 224, 3)

    def test_render_single_candle(self):
        """Test rendering with single candle."""
        from utils.candle_to_image import CandlestickRenderer
        
        renderer = CandlestickRenderer()
        df = pd.DataFrame({
            'open': [100.0],
            'high': [101.0],
            'low': [99.0],
            'close': [100.5],
        })
        
        img = renderer.render(df)
        
        assert img.shape == (224, 224, 3)

    def test_render_with_volume(self, small_ohlcv_df):
        """Test rendering with volume bars."""
        from utils.candle_to_image import CandlestickRenderer
        
        renderer = CandlestickRenderer()
        img = renderer.render(small_ohlcv_df, include_volume=True)
        
        assert img.shape == (224, 224, 3)

    def test_render_flat_prices(self):
        """Test rendering when all prices are the same."""
        from utils.candle_to_image import CandlestickRenderer
        
        renderer = CandlestickRenderer()
        df = pd.DataFrame({
            'open': [100.0] * 10,
            'high': [100.0] * 10,
            'low': [100.0] * 10,
            'close': [100.0] * 10,
        })
        
        img = renderer.render(df)
        
        assert img.shape == (224, 224, 3)

    def test_render_with_annotations(self, small_ohlcv_df):
        """Test render_with_annotations method."""
        from utils.candle_to_image import CandlestickRenderer
        
        renderer = CandlestickRenderer()
        annotations = [
            {'start_idx': 5, 'end_idx': 7, 'class_id': 0},
            {'start_idx': 15, 'end_idx': 17, 'class_id': 1},
        ]
        
        img, bboxes = renderer.render_with_annotations(small_ohlcv_df, annotations)
        
        assert img.shape == (224, 224, 3)
        assert len(bboxes) == 2
        
        # Check bbox structure
        for bbox in bboxes:
            assert 'class_id' in bbox
            assert 'x_center' in bbox
            assert 'y_center' in bbox
            assert 'width' in bbox
            assert 'height' in bbox
            # Values should be normalized (0-1)
            assert 0 <= bbox['x_center'] <= 1
            assert 0 <= bbox['y_center'] <= 1

    def test_render_different_sizes(self, small_ohlcv_df):
        """Test rendering at different image sizes."""
        from utils.candle_to_image import CandlestickRenderer
        
        for size in [64, 128, 224, 256, 512]:
            renderer = CandlestickRenderer(image_size=(size, size))
            img = renderer.render(small_ohlcv_df)
            
            assert img.shape == (size, size, 3)


class TestCandleImage:
    """Tests for candle_image convenience function."""

    def test_candle_image_basic(self, small_ohlcv_df):
        """Test basic candle_image function."""
        from utils.candle_to_image import candle_image
        
        img = candle_image(small_ohlcv_df)
        
        assert isinstance(img, np.ndarray)
        assert img.shape == (224, 224, 3)
        assert img.dtype == np.uint8

    def test_candle_image_custom_size(self, small_ohlcv_df):
        """Test candle_image with custom size."""
        from utils.candle_to_image import candle_image
        
        img = candle_image(small_ohlcv_df, target_size=256)
        
        assert img.shape == (256, 256, 3)

    def test_candle_image_with_volume(self, small_ohlcv_df):
        """Test candle_image with volume."""
        from utils.candle_to_image import candle_image
        
        img = candle_image(small_ohlcv_df, include_volume=True)
        
        assert img.shape == (224, 224, 3)


class TestNormalizeForModel:
    """Tests for normalize_for_model function."""

    def test_normalize_basic(self, small_ohlcv_df):
        """Test basic normalization."""
        from utils.candle_to_image import candle_image, normalize_for_model
        
        img = candle_image(small_ohlcv_df)
        normalized = normalize_for_model(img)
        
        # Should be CHW format
        assert normalized.shape == (3, 224, 224)
        assert normalized.dtype == np.float32

    def test_normalize_imagenet_stats(self, small_ohlcv_df):
        """Test normalization with ImageNet stats."""
        from utils.candle_to_image import candle_image, normalize_for_model
        
        img = candle_image(small_ohlcv_df)
        normalized = normalize_for_model(img, use_imagenet_stats=True)
        
        # Values should be roughly centered around 0
        # (exact range depends on image content)
        assert normalized.min() < 0 or normalized.max() > 1

    def test_normalize_without_imagenet(self, small_ohlcv_df):
        """Test normalization without ImageNet stats."""
        from utils.candle_to_image import candle_image, normalize_for_model
        
        img = candle_image(small_ohlcv_df)
        normalized = normalize_for_model(img, use_imagenet_stats=False)
        
        # Should be in [0, 1] range
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_normalize_preserves_channels(self):
        """Test that normalization preserves color channels."""
        from utils.candle_to_image import normalize_for_model
        
        # Create a simple RGB image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # Red channel
        
        normalized = normalize_for_model(img, use_imagenet_stats=False)
        
        # Red channel should have higher values
        assert normalized[0].mean() > normalized[1].mean()
        assert normalized[0].mean() > normalized[2].mean()


class TestRendererEdgeCases:
    """Tests for edge cases in rendering."""

    def test_very_small_price_range(self):
        """Test rendering with very small price differences."""
        from utils.candle_to_image import CandlestickRenderer
        
        renderer = CandlestickRenderer()
        df = pd.DataFrame({
            'open': [1.10000, 1.10001, 1.10002],
            'high': [1.10003, 1.10004, 1.10005],
            'low': [0.99997, 0.99998, 0.99999],
            'close': [1.10001, 1.10002, 1.10003],
        })
        
        img = renderer.render(df)
        assert img.shape == (224, 224, 3)

    def test_negative_prices(self):
        """Test rendering with negative prices (edge case)."""
        from utils.candle_to_image import CandlestickRenderer
        
        renderer = CandlestickRenderer()
        df = pd.DataFrame({
            'open': [-10, -9, -8],
            'high': [-8, -7, -6],
            'low': [-12, -11, -10],
            'close': [-9, -8, -7],
        })
        
        img = renderer.render(df)
        assert img.shape == (224, 224, 3)

    def test_large_price_range(self):
        """Test rendering with large price range."""
        from utils.candle_to_image import CandlestickRenderer
        
        renderer = CandlestickRenderer()
        df = pd.DataFrame({
            'open': [100, 500, 1000],
            'high': [200, 600, 1500],
            'low': [50, 400, 800],
            'close': [150, 550, 1200],
        })
        
        img = renderer.render(df)
        assert img.shape == (224, 224, 3)

    def test_many_candles(self):
        """Test rendering with many candles."""
        from utils.candle_to_image import CandlestickRenderer
        
        renderer = CandlestickRenderer()
        n = 500
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        
        df = pd.DataFrame({
            'open': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices + np.random.randn(n) * 0.2,
        })
        
        img = renderer.render(df)
        assert img.shape == (224, 224, 3)