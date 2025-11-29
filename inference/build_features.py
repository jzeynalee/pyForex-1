# Feature Builder (LSTM + ViT + YOLO)
# inference/build_features.py

import torch
from utils.candle_to_image import candle_image
import numpy as np

def build_features(df_window, lstm_model, vit_model, yolo_detector):

    # ---- LSTM vector ----
    seq = df_window[['open','high','low','close','tick_volume']].values
    lstm_in = torch.tensor(seq[-60:]).float().unsqueeze(0)
    lstm_vec = lstm_model(lstm_in).detach()

    # ---- ViT vector ----
    img = candle_image(df_window)
    img = torch.tensor(img).permute(2,0,1).float()/255.
    img = img.unsqueeze(0)
    vit_vec = vit_model(img).detach()

    # ---- YOLO vector ----
    yolo_vec = torch.tensor(
        yolo_detector.detect(img[0].permute(1,2,0).numpy())
    ).float().unsqueeze(0)

    return lstm_vec, vit_vec, yolo_vec
