# Train ViT (fine-tuning)
# training/train_vit.py

"""
Training script for ViT with feature caching.
First run:   extracts ViT embeddings for train/val → saves dataset_cache.pt
Next runs:   loads cached tensors instantly.
"""

import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm
from models.vit_extractor import ViTExtractor


# -----------------------------------------------------
# Argument Parser
# -----------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train ViT using cached features")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_vit_cached")
    parser.add_argument("--cache_path", type=str, default="./dataset_cache.pt")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Simple classifier head
    classifier = nn.Sequential(
        nn.LayerNorm(768),
        nn.Linear(768, num_classes)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr)

    best_acc = 0

    for epoch in range(1, args.num_epochs + 1):
        print(f"\n📘 Epoch {epoch}/{args.num_epochs}")
        classifier.train()
        total_loss = 0
        correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = classifier(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()

        acc = correct / len(train_ds)
        print(f"  🔹 Train Loss: {total_loss:.4f} | Train Acc: {acc:.4f}")

        # Validation
        classifier.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = classifier(x)
                correct += (out.argmax(1) == y).sum().item()

        val_acc = correct / len(val_ds)
        print(f"  🔹 Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = os.path.join(args.save_dir, "best_head.pth")
            torch.save(classifier.state_dict(), ckpt)
            print(f"  💾 Saved best classifier → {ckpt}")

    return classifier


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    args = get_args()

    cache = maybe_build_cache(args.data_dir, args.cache_path, args.device)

    train_x = cache["train_x"]
    train_y = cache["train_y"]
    val_x = cache["val_x"]
    val_y = cache["val_y"]
    num_classes = cache["num_classes"]

    train_classifier(train_x, train_y, val_x, val_y, num_classes, args)


if __name__ == "__main__":
    main()
