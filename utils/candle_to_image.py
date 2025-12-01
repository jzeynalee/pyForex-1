# utils/candle_to_image.py
# utils/candle_to_image.py
import cv2
import numpy as np
import pandas as pd

def candle_image(df_window, target_size=224):
    """
    Generates a candlestick chart image using OpenCV for high performance.
    
    Args:
        df_window (pd.DataFrame): DataFrame containing 'open', 'high', 'low', 'close'
        target_size (int): Height and Width of the output image (square)
    
    Returns:
        np.array: RGB Image of shape (target_size, target_size, 3)
    """
    # 1. Initialize Canvas (White Background, RGB)
    # Shape: (Height, Width, Channels)
    img = np.zeros((target_size, target_size, 3), dtype=np.uint8) + 255 

    if df_window.empty:
        return img

    # 2. Extract Data & Normalize
    # Ensure we strictly use numpy arrays for speed
    opens = df_window['open'].values
    highs = df_window['high'].values
    lows = df_window['low'].values
    closes = df_window['close'].values

    # Determine scale with a small padding (5%) so wicks don't touch edges
    min_price = lows.min()
    max_price = highs.max()
    padding = (max_price - min_price) * 0.05
    if padding == 0: padding = 1e-5 # Avoid division by zero for flat lines
    
    y_min = min_price - padding
    y_max = max_price + padding
    y_range = y_max - y_min

    # 3. Calculate Coordinates
    # X-axis: distribute candles evenly
    num_candles = len(df_window)
    candle_width = target_size / num_candles
    
    # Half-width for the candle body (e.g., 80% of the allocated slot)
    body_w_half = max(1, int((candle_width * 0.4))) 

    # Colors (RGB)
    color_bullish = (0, 200, 0)  # Green
    color_bearish = (200, 0, 0)  # Red
    color_wick = (0, 0, 0)       # Black wicks often provide better contrast

    # Vectorized normalization would be faster, but loop is sufficient for 60 candles
    # Coordinate system: Y=0 is top, Y=224 is bottom.
    # So High Price -> Low Y value.
    
    def get_y(price):
        return int(target_size - ((price - y_min) / y_range) * target_size)

    for i in range(num_candles):
        # Center X of the candle
        cx = int((i + 0.5) * candle_width)
        
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        
        y_open = get_y(o)
        y_close = get_y(c)
        y_high = get_y(h)
        y_low = get_y(l)

        # Determine Bullish/Bearish
        if c >= o:
            color = color_bullish
            top, bottom = y_close, y_open
        else:
            color = color_bearish
            top, bottom = y_open, y_close

        # Draw Wick (Line from Low to High)
        cv2.line(img, (cx, y_high), (cx, y_low), color_wick, thickness=1)

        # Draw Body (Rectangle)
        # Ensure body has at least 1px height
        if top == bottom:
            bottom += 1
            
        cv2.rectangle(img, (cx - body_w_half, top), (cx + body_w_half, bottom), color, thickness=-1)

    return img