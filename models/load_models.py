# LOAD ALL MODELS
# models/load_models.py

import torch
from models.lstm import LSTMModel
from models.vit import ViTExtractor
from models.fusion import FusionNet
from models.yolo_detector import YOLOPatternDetector

def load_all():
    lstm = LSTMModel()
    lstm.load_state_dict(torch.load("models/lstm_best.pt"))
    lstm.eval()

    vit = ViTExtractor()
    vit.load_state_dict(torch.load("models/vit_best.pt"))
    vit.eval()

    fusion = FusionNet()
    fusion.load_state_dict(torch.load("models/fusion_best.pt"))
    fusion.eval()

    yolo = YOLOPatternDetector("models/yolo_best.pt")

    return lstm, vit, fusion, yolo
