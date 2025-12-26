# training/train_yolo.py
"""
YOLOv8 training script with GPU acceleration.
"""
from ultralytics import YOLO
import os
import torch
from multiprocessing import freeze_support

def run_training():
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
    data_cfg = os.environ.get("YOLO_DATA_CFG") or os.environ.get("YOLO_DATA")
    if not data_cfg:
        data_cfg = "data/yolo.yml" if os.path.exists("data/yolo.yml") else "data/yolo.yaml"

    device_cfg = os.environ.get("YOLO_DEVICE")
    if device_cfg is None or str(device_cfg).strip() == "":
        device_cfg = 0 if torch.cuda.is_available() else "cpu"
    else:
        device_cfg = str(device_cfg).strip()
        if device_cfg.isdigit():
            device_cfg = int(device_cfg)

    workers_cfg = int(os.environ.get("YOLO_WORKERS", "4"))

    epochs_cfg = int(os.environ.get("YOLO_EPOCHS", "80"))
    imgsz_cfg = int(os.environ.get("YOLO_IMGSZ", "256"))
    batch_cfg = int(os.environ.get("YOLO_BATCH", "16"))

    cache_env = os.environ.get("YOLO_CACHE")
    if cache_env is None or str(cache_env).strip() == "":
        cache_cfg = True
    else:
        cache_cfg = str(cache_env).strip().lower() in {"1", "true", "yes", "y", "on"}

    model.train(
        data=data_cfg,
        epochs=epochs_cfg,
        imgsz=imgsz_cfg,
        device=device_cfg,           # Force GPU (use 'cpu' to force CPU, or [0,1] for multi-GPU)
        batch=batch_cfg,           # Increase to 32 or 64 if you have enough VRAM
        workers=workers_cfg,          # Parallel data loading threads
        amp=True,           # Mixed precision (FP16) - 2x faster, less VRAM
        cache=cache_cfg,         # Cache images in RAM for faster training
        patience=20,        # Early stopping patience
        verbose=True,
    )

    # Export trained model
    model.export(format="pt")
    print("✅ Training complete! Model exported.")


if __name__ == "__main__":
    freeze_support()
    run_training()