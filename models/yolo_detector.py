# models/yolo_detector.py
# YOLO candlestick pattern detector

from ultralytics import YOLO
import numpy as np

class YOLOPatternDetector:
    def __init__(self, model_path="yolo_best.pt"):
        self.model = YOLO(model_path)

    def detect(self, img):
        results = self.model(img, verbose=False)[0]
        vector = [0]*20
        for b in results.boxes:
            cls = int(b.cls)
            vector[cls] = 1
        return np.array(vector)
