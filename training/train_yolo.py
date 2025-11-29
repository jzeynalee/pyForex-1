# training/train_yolo.py
# This script trains a YOLOv8 model using the Ultralytics library.
# It uses a pre-trained YOLOv8n model, trains it on a specified dataset for 80 epochs with image size 256, and then exports the trained model in PyTorch format.
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data/yolo.yaml", epochs=80, imgsz=256)
model.export(format="pt")