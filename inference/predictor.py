# Combines load_models and build_features
# inference/predictor.py

import torch
import numpy as np
from models.lstm import LSTMModel
from models.vit import ViTExtractor
from models.fusion import FusionNet
from models.yolo_detector import YOLOPatternDetector
from utils.candle_to_image import candle_image
from utils.config import settings

class HybridPredictor:
    def __init__(self, weights_dir="models/weights/"):
        self.device = torch.device(settings.DEVICE)
        self.load_models(weights_dir)
    
    def load_models(self, weights_dir):
        # Initialize Architecture
        self.lstm = LSTMModel().to(self.device).eval()
        self.vit = ViTExtractor().to(self.device).eval()
        self.fusion = FusionNet().to(self.device).eval()
        self.yolo = YOLOPatternDetector(f"{weights_dir}/yolo_best.pt") # Ultralytics handles device internally
        
        # Load State Dicts
        self.lstm.load_state_dict(torch.load(f"{weights_dir}/lstm_best.pt", map_location=self.device))
        self.vit.load_state_dict(torch.load(f"{weights_dir}/vit_best.pt", map_location=self.device))
        self.fusion.load_state_dict(torch.load(f"{weights_dir}/fusion_best.pt", map_location=self.device))

    def predict(self, df_window):
        """
        Takes raw DataFrame, handles all transformations and inference
        """
        # 1. LSTM Input Prep
        seq = df_window[['open','high','low','close','tick_volume']].values
        lstm_in = torch.tensor(seq[-60:]).float().unsqueeze(0).to(self.device)

        # 2. Vision Input Prep (Bottleneck: Image Generation)
        img_array = candle_image(df_window)
        # Permute for Torch (C, H, W)
        img_tensor = torch.tensor(img_array).permute(2,0,1).float()/255.
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # 3. Inference
        with torch.no_grad():
            lstm_feat = self.lstm(lstm_in)
            vit_feat = self.vit(img_tensor)
            
            # YOLO needs numpy array in HWC
            yolo_vec_np = self.yolo.detect(img_array) 
            yolo_feat = torch.tensor(yolo_vec_np).float().unsqueeze(0).to(self.device)
            
            # Fusion
            logits = self.fusion(lstm_feat, vit_feat, yolo_feat)
            probs = logits.softmax(dim=1).cpu().numpy()[0]
            
        return probs # [Buy_Prob, Sell_Prob, Hold_Prob]