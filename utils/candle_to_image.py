# utils/candle_to_image.py
import mplfinance as mpf
import pandas as pd
import numpy as np
from PIL import Image
import io

def candle_image(df_window):
    fig = mpf.figure(style='charles', figsize=(3,3))
    ax = fig.add_subplot(1,1,1)
    mpf.plot(df_window, type='candle', ax=ax)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    img = img.resize((224,224))   
    return np.array(img)
