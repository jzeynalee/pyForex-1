# training/train_yolo.py
"""
YOLOv8 training script with GPU acceleration.
"""
from ultralytics import YOLO
import torch

# Verify CUDA is available
print("=" * 50)
print("GPU CHECK")
print("=" * 50)
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
else:
    print("⚠️  WARNING: CUDA not available! Training will be slow on CPU.")
    print("   Install CUDA PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
print("=" * 50)

model = YOLO("yolov8n.pt")

# Train with GPU acceleration
model.train(
    data="data/yolo.yaml",
    epochs=80,
    imgsz=256,
    device=0,           # Force GPU (use 'cpu' to force CPU, or [0,1] for multi-GPU)
    batch=16,           # Increase to 32 or 64 if you have enough VRAM
    workers=4,          # Parallel data loading threads
    amp=True,           # Mixed precision (FP16) - 2x faster, less VRAM
    cache=True,         # Cache images in RAM for faster training
    patience=20,        # Early stopping patience
    verbose=True,
)

# Export trained model
model.export(format="pt")
print("✅ Training complete! Model exported.")