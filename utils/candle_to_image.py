# utils/candle_to_image.py
"""
High-performance candlestick chart image generation using OpenCV.
"""
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def candle_image(
    df_window: pd.DataFrame,
    target_size: int = 224,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    bullish_color: Tuple[int, int, int] = (0, 200, 0),
    bearish_color: Tuple[int, int, int] = (200, 0, 0),
    wick_color: Tuple[int, int, int] = (50, 50, 50),
    body_width_ratio: float = 0.6,
    padding_pct: float = 0.05,
    add_volume: bool = False,
) -> np.ndarray:
    """
    Generates a candlestick chart image using OpenCV.
    
    Args:
        df_window: DataFrame with 'open', 'high', 'low', 'close' columns
        target_size: Output image dimensions (square)
        background_color: RGB background color
        bullish_color: RGB color for bullish candles
        bearish_color: RGB color for bearish candles
        wick_color: RGB color for wicks
        body_width_ratio: Width of candle body relative to slot width
        padding_pct: Padding percentage for price range
        add_volume: If True, adds volume bars at bottom
    
    Returns:
        RGB image array of shape (target_size, target_size, 3)
    """
    # Initialize canvas
    img = np.full((target_size, target_size, 3), background_color, dtype=np.uint8)
    
    if df_window.empty:
        return img
    
    # Extract OHLC data
    opens = df_window['open'].values.astype(np.float64)
    highs = df_window['high'].values.astype(np.float64)
    lows = df_window['low'].values.astype(np.float64)
    closes = df_window['close'].values.astype(np.float64)
    
    num_candles = len(df_window)
    
    # Calculate price range with padding
    min_price = lows.min()
    max_price = highs.max()
    price_range = max_price - min_price
    
    if price_range == 0:
        price_range = min_price * 0.01 if min_price > 0 else 1.0
    
    padding = price_range * padding_pct
    y_min = min_price - padding
    y_max = max_price + padding
    y_range = y_max - y_min
    
    # Chart area (leave room for volume if enabled)
    chart_height = int(target_size * 0.8) if add_volume else target_size
    volume_height = target_size - chart_height if add_volume else 0
    
    # Calculate candle dimensions
    candle_width = target_size / num_candles
    body_half_width = max(1, int(candle_width * body_width_ratio / 2))
    
    def price_to_y(price: float) -> int:
        """Convert price to Y coordinate (inverted: high price = low Y)."""
        normalized = (price - y_min) / y_range
        return int(chart_height - (normalized * chart_height))
    
    # Draw candles
    for i in range(num_candles):
        cx = int((i + 0.5) * candle_width)  # Center X
        
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        
        y_open = price_to_y(o)
        y_close = price_to_y(c)
        y_high = price_to_y(h)
        y_low = price_to_y(l)
        
        # Determine direction
        is_bullish = c >= o
        color = bullish_color if is_bullish else bearish_color
        
        # Body coordinates
        body_top = min(y_open, y_close)
        body_bottom = max(y_open, y_close)
        
        # Ensure minimum body height of 1 pixel
        if body_bottom == body_top:
            body_bottom += 1
        
        # Draw wick (vertical line from low to high)
        cv2.line(img, (cx, y_high), (cx, y_low), wick_color, thickness=1)
        
        # Draw body (filled rectangle)
        x1 = cx - body_half_width
        x2 = cx + body_half_width
        cv2.rectangle(img, (x1, body_top), (x2, body_bottom), color, thickness=-1)
    
    # Optionally add volume bars
    if add_volume and 'tick_volume' in df_window.columns:
        volumes = df_window['tick_volume'].values.astype(np.float64)
        max_vol = volumes.max()
        if max_vol > 0:
            for i in range(num_candles):
                cx = int((i + 0.5) * candle_width)
                vol_height = int((volumes[i] / max_vol) * volume_height * 0.9)
                
                is_bullish = closes[i] >= opens[i]
                color = bullish_color if is_bullish else bearish_color
                
                x1 = cx - body_half_width
                x2 = cx + body_half_width
                y1 = target_size - vol_height
                y2 = target_size
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
    
    return img


def candle_image_with_indicators(
    df_window: pd.DataFrame,
    target_size: int = 224,
    ma_periods: Optional[list] = None,
) -> np.ndarray:
    """
    Generates candlestick chart with optional moving averages overlaid.
    
    Args:
        df_window: DataFrame with OHLC data
        target_size: Output image size
        ma_periods: List of MA periods to plot (e.g., [20, 50])
    
    Returns:
        RGB image array
    """
    # Start with base candle image
    img = candle_image(df_window, target_size)
    
    if ma_periods is None or df_window.empty:
        return img
    
    closes = df_window['close'].values.astype(np.float64)
    lows = df_window['low'].values.astype(np.float64)
    highs = df_window['high'].values.astype(np.float64)
    
    # Price scaling (same as candle_image)
    min_price = lows.min()
    max_price = highs.max()
    price_range = max_price - min_price
    if price_range == 0:
        price_range = 1.0
    padding = price_range * 0.05
    y_min = min_price - padding
    y_max = max_price + padding
    y_range = y_max - y_min
    
    num_candles = len(df_window)
    candle_width = target_size / num_candles
    
    def price_to_y(price: float) -> int:
        normalized = (price - y_min) / y_range
        return int(target_size - (normalized * target_size))
    
    # MA colors (blue shades)
    ma_colors = [
        (255, 100, 100),  # Light blue
        (255, 50, 50),    # Darker blue
        (200, 50, 50),    # Even darker
    ]
    
    for idx, period in enumerate(ma_periods):
        if period > len(closes):
            continue
            
        # Calculate SMA
        ma = pd.Series(closes).rolling(period).mean().values
        
        color = ma_colors[idx % len(ma_colors)]
        
        # Draw MA line
        points = []
        for i in range(period - 1, num_candles):
            cx = int((i + 0.5) * candle_width)
            cy = price_to_y(ma[i])
            points.append((cx, cy))
        
        if len(points) > 1:
            points_array = np.array(points, dtype=np.int32)
            cv2.polylines(img, [points_array], False, color, thickness=1)
    
    return img


def normalize_for_model(
    img: np.ndarray,
    use_imagenet_stats: bool = True,
) -> np.ndarray:
    """
    Normalize image for neural network input.
    
    Args:
        img: RGB image in uint8 [0, 255]
        use_imagenet_stats: If True, normalize with ImageNet mean/std
    
    Returns:
        Normalized float32 array in CHW format
    """
    # Convert to float and scale to [0, 1]
    img_float = img.astype(np.float32) / 255.0
    
    if use_imagenet_stats:
        # ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_float = (img_float - mean) / std
    
    # HWC -> CHW
    img_float = np.transpose(img_float, (2, 0, 1))
    
    return img_float
