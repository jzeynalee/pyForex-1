# training/train_vit_profiles.py
"""
ViT training script for multiple trading profiles.
Trains separate models for SCALP, INTRADAY, and SWING trading.

Workflow:
1. Generate ViT datasets (ImageFolder format) for each profile
2. Extract features using pretrained ViT
3. Train classifier head for each profile
"""
import argparse
import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.vit_extractor import ViTExtractor
from utils.vit_dataset_generator import ViTDatasetGenerator, FuturePriceLabeler


# Profile configurations
PROFILES = {
    'SCALP': {
        'data_csv': 'data/raw/EURUSD_M5_latest.csv',
        'dataset_dir': 'datasets/vit_scalp',
        'cache_path': 'datasets/vit_scalp/cache.pt',
        'window_size': 30,
        'stride': 5,
        'forward_bars': 6,
        'threshold_pct': 0.15,
        'epochs': 50,
        'patience': 10,
        'description': 'Short-term scalping (M5 timeframe)',
    },
    'INTRADAY': {
        'data_csv': 'data/raw/EURUSD_H1_latest.csv',
        'dataset_dir': 'datasets/vit_intraday',
        'cache_path': 'datasets/vit_intraday/cache.pt',
        'window_size': 60,
        'stride': 10,
        'forward_bars': 10,
        'threshold_pct': 0.3,
        'epochs': 60,
        'patience': 15,
        'description': 'Intraday trading (H1 timeframe)',
    },
    'SWING': {
        'data_csv': 'data/raw/EURUSD_H4_latest.csv',
        'dataset_dir': 'datasets/vit_swing',
        'cache_path': 'datasets/vit_swing/cache.pt',
        'window_size': 90,
        'stride': 15,
        'forward_bars': 15,
        'threshold_pct': 0.5,
        'epochs': 80,
        'patience': 20,
        'description': 'Swing trading (H4 timeframe)',
    },
}


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


class ClassifierHead(nn.Module):
    """Two-layer classifier head for ViT features."""
    
    def __init__(self, in_features=768, hidden_dim=256, num_classes=3, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.block(x)
        return self.classifier(x)


def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def generate_dataset(profile: str, max_samples: int = None) -> bool:
    """Generate ViT dataset for a profile."""
    config = PROFILES[profile]
    
    print(f"\n📊 Generating ViT dataset for {profile}...")
    print(f"   CSV: {config['data_csv']}")
    print(f"   Output: {config['dataset_dir']}")
    
    csv_path = Path(config['data_csv'])
    if not csv_path.exists():
        print(f"❌ Data file not found: {csv_path}")
        return False
    
    labeler = FuturePriceLabeler(
        forward_bars=config['forward_bars'],
        threshold_pct=config['threshold_pct']
    )
    
    generator = ViTDatasetGenerator(
        output_dir=config['dataset_dir'],
        image_size=224,
        window_size=config['window_size'],
        stride=config['stride'],
        val_split=0.2,
        labeler=labeler,
    )
    
    try:
        stats = generator.generate_from_csv(str(csv_path), symbol=f"EURUSD_{profile}", max_samples=max_samples)
        print(f"✅ Dataset generated: {stats['train']['total']} train, {stats['val']['total']} val")
        return True
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        return False


def build_feature_cache(profile: str, device: str) -> dict:
    """Extract ViT features and cache them."""
    config = PROFILES[profile]
    cache_path = Path(config['cache_path'])
    
    if cache_path.exists():
        print(f"⚡ Loading cached features: {cache_path}")
        return torch.load(cache_path)
    
    print(f"⚡ Building feature cache for {profile}...")
    
    data_dir = Path(config['dataset_dir'])
    if not (data_dir / 'train').exists():
        raise FileNotFoundError(f"Dataset not found at {data_dir}. Run dataset generation first.")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = datasets.ImageFolder(str(data_dir / 'train'), transform)
    val_ds = datasets.ImageFolder(str(data_dir / 'val'), transform)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    vit = ViTExtractor().to(device)
    vit.eval()
    
    def extract_features(loader, desc):
        feats, labels = [], []
        with torch.no_grad():
            for imgs, y in tqdm(loader, desc=desc):
                imgs = imgs.to(device)
                f = vit(imgs)
                feats.append(f.cpu())
                labels.append(y)
        return torch.cat(feats), torch.cat(labels)
    
    train_x, train_y = extract_features(train_loader, "Extracting train features")
    val_x, val_y = extract_features(val_loader, "Extracting val features")
    
    cache = {
        'train_x': train_x,
        'train_y': train_y,
        'val_x': val_x,
        'val_y': val_y,
        'num_classes': len(train_ds.classes),
        'class_names': train_ds.classes,
    }
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"💾 Saved cache → {cache_path}")
    
    return cache


def train_classifier(profile: str, cache: dict, device: str, lr: float = 5e-4) -> dict:
    """Train classifier head for a profile."""
    config = PROFILES[profile]
    
    train_x, train_y = cache['train_x'], cache['train_y']
    val_x, val_y = cache['val_x'], cache['val_y']
    num_classes = cache['num_classes']
    
    print(f"\n🎯 Training ViT classifier for {profile}")
    print(f"   Train: {len(train_x)}, Val: {len(val_x)}, Classes: {num_classes}")
    
    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    classifier = ClassifierHead(
        in_features=768,
        hidden_dim=256,
        num_classes=num_classes,
        dropout=0.2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)
    
    best_acc = 0
    patience_counter = 0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': []}
    
    for epoch in range(1, config['epochs'] + 1):
        # Training
        classifier.train()
        total_loss, correct, total = 0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
            
            optimizer.zero_grad()
            out = classifier(mixed_x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            with torch.no_grad():
                clean_out = classifier(x)
                correct += (clean_out.argmax(1) == y).sum().item()
                total += y.size(0)
        
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        classifier.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = classifier(x)
                correct += (out.argmax(1) == y).sum().item()
        
        val_acc = correct / len(val_ds)
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_loss'].append(avg_loss)
        
        scheduler.step()
        
        print(f"  Epoch {epoch:3d}/{config['epochs']} | Loss: {avg_loss:.4f} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            
            # Save best model
            save_dir = Path('models/vit')
            save_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state': classifier.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'num_classes': num_classes,
                'class_names': cache.get('class_names', ['BEARISH', 'SIDEWAYS', 'BULLISH']),
                'profile': profile,
            }, save_dir / f'vit_{profile.lower()}.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"  🛑 Early stopping at epoch {epoch}")
            break
    
    print(f"✅ {profile} training complete! Best Val Acc: {best_acc:.4f}")
    return {'best_acc': best_acc, 'history': history}


def train_profile(profile: str, device: str, skip_dataset: bool = False, max_samples: int = None):
    """Train ViT model for a specific profile."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}. Use {list(PROFILES.keys())}")
    
    config = PROFILES[profile]
    
    print("\n" + "=" * 60)
    print(f"🎯 TRAINING ViT MODEL: {profile}")
    print(f"   {config['description']}")
    print("=" * 60)
    
    # Step 1: Generate dataset if needed
    dataset_dir = Path(config['dataset_dir'])
    if not skip_dataset and not (dataset_dir / 'train').exists():
        if not generate_dataset(profile, max_samples):
            return None
    elif not (dataset_dir / 'train').exists():
        print(f"❌ Dataset not found at {dataset_dir}")
        return None
    
    # Step 2: Build feature cache
    cache = build_feature_cache(profile, device)
    
    # Step 3: Train classifier
    result = train_classifier(profile, cache, device)
    
    return result


def train_all_profiles(device: str, skip_dataset: bool = False, max_samples: int = None):
    """Train ViT models for all profiles."""
    results = {}
    
    for profile in PROFILES:
        try:
            result = train_profile(profile, device, skip_dataset, max_samples)
            results[profile] = result['best_acc'] if result else 'FAILED'
        except Exception as e:
            print(f"❌ Error training {profile}: {e}")
            results[profile] = f'ERROR: {e}'
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VIT TRAINING SUMMARY")
    print("=" * 60)
    for profile, result in results.items():
        if isinstance(result, float):
            print(f"   ✅ {profile}: {result:.4f} val accuracy")
        else:
            print(f"   ❌ {profile}: {result}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train ViT models for trading profiles")
    parser.add_argument('--profile', type=str, choices=['SCALP', 'INTRADAY', 'SWING', 'ALL'],
                        default='ALL', help='Trading profile to train')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--skip-dataset', action='store_true', help='Skip dataset generation')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples per dataset')
    
    args = parser.parse_args()
    
    has_gpu = check_gpu()
    device = args.device if has_gpu else 'cpu'
    
    if args.profile == 'ALL':
        train_all_profiles(device, args.skip_dataset, args.max_samples)
    else:
        train_profile(args.profile, device, args.skip_dataset, args.max_samples)


if __name__ == "__main__":
    main()
