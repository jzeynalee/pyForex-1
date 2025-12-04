# Train ViT (fine-tuning) - Enhanced Version
# training/train_vit.py

"""
Training script for ViT with feature caching.
First run:   extracts ViT embeddings for train/val → saves dataset_cache.pt
Next runs:   loads cached tensors instantly.

Enhancements:
- Deeper classifier head (768 → 256 → num_classes)
- Label smoothing
- Feature-level Mixup augmentation
- Cosine annealing with warm restarts
- Gradient clipping
- Early stopping with patience
"""

import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from models.vit_extractor import ViTExtractor


# -----------------------------------------------------
# Argument Parser
# -----------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train ViT using cached features")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup_alpha", type=float, default=0.2,
                        help="Mixup alpha (0 to disable)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_vit_cached")
    parser.add_argument("--cache_path", type=str, default="./dataset_cache.pt")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


# -----------------------------------------------------
# Mixup augmentation for features
# -----------------------------------------------------
def mixup_data(x, y, alpha=0.2):
    """Apply mixup to feature vectors and labels."""
    if alpha <= 0:
        return x, y, y, 1.0
    
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# -----------------------------------------------------
# Deeper Classifier Head
# -----------------------------------------------------
class ClassifierHead(nn.Module):
    """
    Two-layer classifier with residual-like structure.
    768 → 256 → num_classes
    """
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
        
        # Initialize weights
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


# -----------------------------------------------------
# Extract or Load Cached Features
# -----------------------------------------------------
def maybe_build_cache(data_dir, cache_path, device):
    if os.path.exists(cache_path):
        print(f"⚡ Loading cached features: {cache_path}")
        return torch.load(cache_path)

    print("⚡ Building feature cache (first run)...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # datasets
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Feature extractor
    vit = ViTExtractor().to(device)
    vit.eval()

    def build_features(loader):
        feats = []
        labels = []
        with torch.no_grad():
            p = tqdm(loader)
            for imgs, y in p:
                imgs = imgs.to(device)
                f = vit(imgs)           # shape: [B, 768]
                feats.append(f.cpu())
                labels.append(y)
        return torch.cat(feats), torch.cat(labels)

    train_x, train_y = build_features(train_loader)
    val_x, val_y = build_features(val_loader)

    num_classes = len(train_ds.classes)

    cache = {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "num_classes": num_classes
    }

    torch.save(cache, cache_path)
    print(f"💾 Saved cache → {cache_path}")

    return cache


# -----------------------------------------------------
# Training loop
# -----------------------------------------------------
def train_classifier(train_x, train_y, val_x, val_y, num_classes, args):
    os.makedirs(args.save_dir, exist_ok=True)

    device = args.device

    # Create TensorDatasets + loaders
    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              drop_last=True)  # drop_last for stable mixup
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Deeper classifier head
    classifier = ClassifierHead(
        in_features=768,
        hidden_dim=256,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)
    
    print(f"🧠 Classifier params: {sum(p.numel() for p in classifier.parameters()):,}")

    # Label smoothing cross entropy
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    optimizer = torch.optim.AdamW(
        classifier.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,      # restart every 10 epochs
        T_mult=1,
        eta_min=1e-6
    )

    best_acc = 0
    patience_counter = 0
    history = {"train_acc": [], "val_acc": [], "train_loss": []}

    for epoch in range(1, args.num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n📘 Epoch {epoch}/{args.num_epochs} (lr: {current_lr:.2e})")
        
        classifier.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # Apply mixup
            mixed_x, y_a, y_b, lam = mixup_data(x, y, args.mixup_alpha)

            optimizer.zero_grad()
            out = classifier(mixed_x)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            
            # For accuracy, use original (non-mixed) predictions
            with torch.no_grad():
                clean_out = classifier(x)
                correct += (clean_out.argmax(1) == y).sum().item()
                total += y.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"  🔹 Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f}")

        # Validation (no mixup)
        classifier.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = classifier(x)
                correct += (out.argmax(1) == y).sum().item()

        val_acc = correct / len(val_ds)
        print(f"  🔹 Val Acc: {val_acc:.4f}")

        # Track history
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(avg_loss)

        # Step scheduler
        scheduler.step()

        # Check for improvement
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            ckpt = os.path.join(args.save_dir, "best_head.pth")
            torch.save({
                "model_state": classifier.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
                "num_classes": num_classes,
                "args": vars(args)
            }, ckpt)
            print(f"  💾 Saved best classifier → {ckpt} (acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{args.patience})")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n🛑 Early stopping at epoch {epoch} (best val acc: {best_acc:.4f})")
            break

    print(f"\n✅ Training complete! Best Val Acc: {best_acc:.4f}")
    
    # Save training history
    history_path = os.path.join(args.save_dir, "training_history.pt")
    torch.save(history, history_path)
    print(f"📊 Saved training history → {history_path}")
    
    return classifier


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    args = get_args()
    
    print("=" * 50)
    print("🚀 ViT Classifier Training (Enhanced)")
    print("=" * 50)
    print(f"  📁 Data: {args.data_dir}")
    print(f"  🎯 LR: {args.lr}, Weight Decay: {args.weight_decay}")
    print(f"  🎲 Mixup α: {args.mixup_alpha}, Dropout: {args.dropout}")
    print(f"  🏷️  Label Smoothing: {args.label_smoothing}")
    print(f"  ⏱️  Patience: {args.patience}")
    print("=" * 50)

    cache = maybe_build_cache(args.data_dir, args.cache_path, args.device)

    train_x = cache["train_x"]
    train_y = cache["train_y"]
    val_x = cache["val_x"]
    val_y = cache["val_y"]
    num_classes = cache["num_classes"]
    
    print(f"📊 Dataset: {len(train_x)} train / {len(val_x)} val / {num_classes} classes")

    train_classifier(train_x, train_y, val_x, val_y, num_classes, args)


if __name__ == "__main__":
    main()