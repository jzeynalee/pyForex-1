# candle_to_image.py
"""
Candlestick chart image generator for ML training.
Converts OHLCV DataFrames to chart images suitable for ViT/YOLO.
"""

import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Optional, List, Dict
import pandas as pd


class CandlestickRenderer:
    """
    Renders OHLCV data as candlestick chart images.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        background_color: Tuple[int, int, int] = (255, 255, 255),
        bull_color: Tuple[int, int, int] = (0, 180, 0),
        bear_color: Tuple[int, int, int] = (220, 0, 0),
        wick_color: Tuple[int, int, int] = (60, 60, 60),
        padding: float = 0.1,
        candle_width_ratio: float = 0.7,
    ):
        self.image_size = image_size
        self.background_color = background_color
        self.bull_color = bull_color
        self.bear_color = bear_color
        self.wick_color = wick_color
        self.padding = padding
        self.candle_width_ratio = candle_width_ratio
    
    def render(
        self, 
        df: pd.DataFrame,
        include_volume: bool = False,
    ) -> np.ndarray:
        """
        Render candlestick chart from OHLCV DataFrame.
        
        Args:
            df: DataFrame with 'open', 'high', 'low', 'close' columns
            include_volume: Whether to render volume bars at bottom
        
        Returns:
            numpy array of shape (H, W, 3) in RGB uint8 format
        """
        width, height = self.image_size
        
        img = Image.new('RGB', (width, height), self.background_color)
        draw = ImageDraw.Draw(img)
        
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        n_candles = len(df)
        if n_candles == 0:
            return np.array(img)
        
        price_min = lows.min()
        price_max = highs.max()
        price_range = price_max - price_min
        
        if price_range == 0:
            price_range = price_max * 0.01
        
        price_min -= price_range * self.padding
        price_max += price_range * self.padding
        price_range = price_max - price_min
        
        chart_height = height * (0.8 if include_volume else 1.0)
        candle_spacing = width / n_candles
        candle_width = candle_spacing * self.candle_width_ratio
        
        def price_to_y(price):
            return int(chart_height - ((price - price_min) / price_range * chart_height))
        
        for i in range(n_candles):
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]
            
            x_center = candle_spacing * (i + 0.5)
            x_left = x_center - candle_width / 2
            x_right = x_center + candle_width / 2
            
            y_high = price_to_y(h)
            y_low = price_to_y(l)
            y_open = price_to_y(o)
            y_close = price_to_y(c)
            
            is_bullish = c >= o
            body_color = self.bull_color if is_bullish else self.bear_color
            
            wick_x = int(x_center)
            draw.line([(wick_x, y_high), (wick_x, y_low)], fill=self.wick_color, width=1)
            
            body_top = min(y_open, y_close)
            body_bottom = max(y_open, y_close)
            
            if body_bottom - body_top < 1:
                body_bottom = body_top + 1
            
            draw.rectangle(
                [int(x_left), body_top, int(x_right), body_bottom],
                fill=body_color,
                outline=self.wick_color,
            )
        
        if include_volume and 'tick_volume' in df.columns:
            self._draw_volume(draw, df, width, height, chart_height, candle_spacing, candle_width)
        
        return np.array(img)
    
    def _draw_volume(self, draw, df, width, height, chart_height, candle_spacing, candle_width):
        """Draw volume bars at the bottom of the chart."""
        volumes = df['tick_volume'].values if 'tick_volume' in df.columns else df.get('volume', pd.Series([0]*len(df))).values
        
        vol_max = volumes.max() if volumes.max() > 0 else 1
        vol_height = height - chart_height
        
        for i, vol in enumerate(volumes):
            x_center = candle_spacing * (i + 0.5)
            x_left = x_center - candle_width / 2
            x_right = x_center + candle_width / 2
            
            bar_height = (vol / vol_max) * vol_height * 0.8
            y_bottom = height
            y_top = height - bar_height
            
            is_bullish = df['close'].iloc[i] >= df['open'].iloc[i]
            color = self.bull_color if is_bullish else self.bear_color
            
            light_color = tuple(min(255, c + 100) for c in color)
            draw.rectangle([int(x_left), int(y_top), int(x_right), int(y_bottom)], fill=light_color)
    
    def render_with_annotations(
        self,
        df: pd.DataFrame,
        annotations: List[Dict],
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Render chart with pattern annotations for YOLO training.
        
        Args:
            df: OHLCV DataFrame
            annotations: List of dicts with 'start_idx', 'end_idx', 'pattern_class'
        
        Returns:
            (image_array, bounding_boxes)
        """
        img = self.render(df)
        width, height = self.image_size
        
        n_candles = len(df)
        candle_spacing = width / n_candles
        
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_range = price_max - price_min
        price_min -= price_range * self.padding
        price_max += price_range * self.padding
        price_range = price_max - price_min
        
        bboxes = []
        for ann in annotations:
            start_idx = ann['start_idx']
            end_idx = ann['end_idx']
            class_id = ann['class_id']
            
            x_left = (start_idx / n_candles)
            x_right = ((end_idx + 1) / n_candles)
            
            pattern_df = df.iloc[start_idx:end_idx+1]
            y_high = (price_max - pattern_df['high'].max()) / price_range
            y_low = (price_max - pattern_df['low'].min()) / price_range
            
            x_center = (x_left + x_right) / 2
            y_center = (y_high + y_low) / 2
            box_width = x_right - x_left
            box_height = y_low - y_high
            
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            box_width = max(0.01, min(1, box_width))
            box_height = max(0.01, min(1, box_height))
            
            bboxes.append({
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': box_width,
                'height': box_height,
            })
        
        return img, bboxes


def candle_image(
    df: pd.DataFrame,
    target_size: int = 224,
    include_volume: bool = False,
) -> np.ndarray:
    """
    Convenience function to generate candlestick image.
    
    Args:
        df: OHLCV DataFrame
        target_size: Output image size (square)
        include_volume: Include volume bars
    
    Returns:
        numpy array (H, W, 3) uint8 RGB
    """
    renderer = CandlestickRenderer(image_size=(target_size, target_size))
    return renderer.render(df, include_volume=include_volume)


def normalize_for_model(
    img: np.ndarray,
    use_imagenet_stats: bool = True,
) -> np.ndarray:
    """
    Normalize image for neural network input.
    
    Args:
        img: HWC uint8 image
        use_imagenet_stats: Use ImageNet mean/std
    
    Returns:
        CHW float32 normalized array
    """
    img_float = img.astype(np.float32) / 255.0
    img_chw = img_float.transpose(2, 0, 1)
    
    if use_imagenet_stats:
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_chw = (img_chw - mean) / std
    
    return img_chw


if __name__ == "__main__":
    np.random.seed(42)
    n = 60
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(n) * 0.3),
        'low': prices - np.abs(np.random.randn(n) * 0.3),
        'close': prices + np.random.randn(n) * 0.2,
        'tick_volume': np.random.randint(100, 1000, n),
    })
    
    renderer = CandlestickRenderer()
    img = renderer.render(df, include_volume=True)
    
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    
    Image.fromarray(img).save("test_candle.png")
    print("Saved test_candle.png")