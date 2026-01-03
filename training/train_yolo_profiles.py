# training/train_yolo_profiles.py
"""
YOLOv8 training script for multiple trading profiles.
Trains separate models for SCALP, INTRADAY, and SWING trading.
"""
from ultralytics import YOLO
import torch
import argparse
from pathlib import Path


def check_gpu():
    """Verify CUDA availability and print GPU info."""
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
    print("=" * 50)
    return torch.cuda.is_available()


# Profile configurations
PROFILES = {
    'SCALP': {
        'data_yaml': r'datasets\EURUSD_M5\yolo\data.yaml',
        'epochs': 50,
        'imgsz': 256,
        'batch': 16,
        'patience': 15,
        'description': 'Short-term scalping patterns (M5 timeframe)',
    },
    'INTRADAY': {
        'data_yaml': r'datasets\EURUSD_H1\yolo\data.yaml',
        'epochs': 60,
        'imgsz': 256,
        'batch': 16,
        'patience': 20,
        'description': 'Intraday trading patterns (H1 timeframe)',
    },
    'SWING': {
        'data_yaml': r'datasets\EURUSD_H4\yolo\data.yaml',
        'epochs': 80,
        'imgsz': 256,
        'batch': 16,
        'patience': 25,
        'description': 'Swing trading patterns (H4 timeframe)',
    },
}


def train_profile(profile: str, device: int = 0, workers: int = 4):
    """Train YOLO model for a specific trading profile."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}. Use {list(PROFILES.keys())}")
    
    config = PROFILES[profile]
    
    print("\n" + "=" * 60)
    print(f"🎯 TRAINING YOLO MODEL: {profile}")
    print(f"   {config['description']}")
    print("=" * 60)
    
    # Check if dataset exists
    data_yaml = Path(config['data_yaml'])
    if not data_yaml.exists():
        print(f"❌ Dataset not found: {data_yaml}")
        print("   Run dataset generation first!")
        return None
    
    # Initialize model
    model = YOLO("yolov8n.pt")
    
    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        device=device,
        batch=config['batch'],
        workers=workers,
        amp=True,
        cache=True,
        patience=config['patience'],
        verbose=True,
        project='runs/yolo',
        name=f'{profile.lower()}_train',
        exist_ok=True,
    )
    
    # Export model
    output_dir = Path('models/yolo')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best model
    best_model_path = Path(f'runs/yolo/{profile.lower()}_train/weights/best.pt')
    if best_model_path.exists():
        import shutil
        dest = output_dir / f'yolo_{profile.lower()}.pt'
        shutil.copy(best_model_path, dest)
        print(f"✅ Model saved: {dest}")
    
    return results


def train_all_profiles(device: int = 0, workers: int = 4):
    """Train YOLO models for all trading profiles."""
    results = {}
    
    for profile in PROFILES:
        try:
            result = train_profile(profile, device, workers)
            results[profile] = 'SUCCESS' if result else 'FAILED'
        except Exception as e:
            print(f"❌ Error training {profile}: {e}")
            results[profile] = f'ERROR: {e}'
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    for profile, status in results.items():
        emoji = "✅" if status == 'SUCCESS' else "❌"
        print(f"   {emoji} {profile}: {status}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO models for trading profiles")
    parser.add_argument('--profile', type=str, choices=['SCALP', 'INTRADAY', 'SWING', 'ALL'],
                        default='ALL', help='Trading profile to train')
    parser.add_argument('--device', type=int, default=0, help='GPU device (0, 1, etc.) or -1 for CPU')
    parser.add_argument('--workers', type=int, default=4, help='Data loader workers')
    
    args = parser.parse_args()
    
    has_gpu = check_gpu()
    device = args.device if has_gpu and args.device >= 0 else 'cpu'
    
    if args.profile == 'ALL':
        train_all_profiles(device, args.workers)
    else:
        train_profile(args.profile, device, args.workers)


if __name__ == "__main__":
    main()
