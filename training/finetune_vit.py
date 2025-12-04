# Fine-tune ViT (end-to-end)
# training/finetune_vit.py

"""
Fine-tuning script for ViT on chart classification.
Unfreezes the last N transformer blocks + classifier head.
Uses differential learning rates.
"""

import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import timm

# ============================================
# CUDA DEBUG - Check at import time
# ============================================
print(f"[DEBUG] torch.cuda.is_available() = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[DEBUG] torch.cuda.device_count() = {torch.cuda.device_count()}")
    print(f"[DEBUG] torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
# ============================================


# -----------------------------------------------------
# Argument Parser
# -----------------------------------------------------
def get_args():
    # Detect device BEFORE argparse
    detected_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    parser = argparse.ArgumentParser(description="Fine-tune ViT end-to-end")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr_head", type=float, default=1e-3,
                        help="Learning rate for classifier head")
    parser.add_argument("--lr_backbone", type=float, default=1e-5,
                        help="Learning rate for unfrozen backbone layers")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--unfreeze_blocks", type=int, default=4,
                        help="Number of transformer blocks to unfreeze (from end)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_vit_finetuned")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=detected_device,
                        help="Device to use (auto-detected)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    return parser.parse_args()


# -----------------------------------------------------
# Data Augmentation
# -----------------------------------------------------
def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


# -----------------------------------------------------
# ViT Model with Partial Unfreezing
# -----------------------------------------------------
class FineTunableViT(nn.Module):
    def __init__(self, num_classes, unfreeze_blocks=4, dropout=0.1):
        super().__init__()
        
        self.vit = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0
        )
        
        self.embed_dim = self.vit.embed_dim
        
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._init_head()
        self._configure_freezing(unfreeze_blocks)
        
    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _configure_freezing(self, unfreeze_blocks):
        for param in self.vit.parameters():
            param.requires_grad = False
        
        blocks = self.vit.blocks
        num_blocks = len(blocks)
        
        for i in range(num_blocks - unfreeze_blocks, num_blocks):
            for param in blocks[i].parameters():
                param.requires_grad = True
        
        for param in self.vit.norm.parameters():
            param.requires_grad = True
        
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        print(f"🔓 Unfroze last {unfreeze_blocks}/{num_blocks} transformer blocks")
        print(f"📊 Parameters: {trainable:,} trainable / {frozen:,} frozen / {total:,} total")
    
    def forward(self, x):
        features = self.vit(x)
        return self.head(features)
    
    def get_param_groups(self, lr_backbone, lr_head, weight_decay):
        backbone_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        return [
            {'params': backbone_params, 'lr': lr_backbone, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': lr_head, 'weight_decay': weight_decay}
        ]


# -----------------------------------------------------
# Learning Rate Scheduler with Warmup
# -----------------------------------------------------
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch <= self.warmup_epochs:
            alpha = 0.1 + 0.9 * (epoch / self.warmup_epochs)
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i] * alpha
        else:
            import math
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            alpha = 0.5 * (1 + math.cos(progress * math.pi))
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.min_lr + (self.base_lrs[i] - self.min_lr) * alpha
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# -----------------------------------------------------
# Training Loop
# -----------------------------------------------------
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), correct / total


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(val_loader, desc="Validation"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), correct / total


# -----------------------------------------------------
# Main Training Function
# -----------------------------------------------------
def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Force CUDA check
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")
    
    is_cpu = device.type == "cpu"
    
    print("=" * 60)
    print("🚀 ViT Fine-Tuning (End-to-End)")
    print("=" * 60)
    print(f"  📁 Data: {args.data_dir}")
    print(f"  🎯 LR (backbone): {args.lr_backbone}, LR (head): {args.lr_head}")
    print(f"  🔓 Unfreezing: {args.unfreeze_blocks} blocks")
    print(f"  🎲 Label Smoothing: {args.label_smoothing}, Dropout: {args.dropout}")
    print(f"  🔥 Device: {device}")
    if is_cpu:
        print("  ⚠️  CPU training detected - this will be slow!")
        print("  💡 Consider using Google Colab or a GPU machine")
    print("=" * 60)
    
    # Data loaders
    train_transform = get_transforms(is_train=True)
    val_transform = get_transforms(is_train=False)
    
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "train"), 
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, "val"), 
        transform=val_transform
    )
    
    num_workers = 0 if is_cpu else args.num_workers
    pin_memory = not is_cpu
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    num_classes = len(train_dataset.classes)
    print(f"📊 Dataset: {len(train_dataset)} train / {len(val_dataset)} val / {num_classes} classes")
    print(f"📋 Classes: {train_dataset.classes}")
    
    # Model
    model = FineTunableViT(
        num_classes=num_classes,
        unfreeze_blocks=args.unfreeze_blocks,
        dropout=args.dropout
    ).to(device)
    
    # Optimizer with differential LR
    param_groups = model.get_param_groups(
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups)
    
    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.num_epochs
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Resume from checkpoint
    start_epoch = 1
    best_acc = 0
    if args.resume:
        print(f"📂 Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"   Resumed at epoch {start_epoch}, best acc: {best_acc:.4f}")
    
    # Training loop
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        scheduler.step(epoch)
        lrs = scheduler.get_lr()
        
        print(f"\n📘 Epoch {epoch}/{args.num_epochs} (lr_backbone: {lrs[0]:.2e}, lr_head: {lrs[1]:.2e})")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"  🔹 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  🔹 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(lrs)
        
        # Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_acc': best_acc,
                'num_classes': num_classes,
                'classes': train_dataset.classes,
                'args': vars(args)
            }
            
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(checkpoint, ckpt_path)
            print(f"  💾 Saved best model → {ckpt_path} (acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{args.patience})")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n🛑 Early stopping at epoch {epoch} (best val acc: {best_acc:.4f})")
            break
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            periodic_ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_acc': best_acc,
                'num_classes': num_classes,
                'args': vars(args)
            }
            torch.save(periodic_ckpt, os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth"))
    
    print(f"\n✅ Training complete! Best Val Acc: {best_acc:.4f}")
    
    # Save history
    history_path = os.path.join(args.save_dir, "training_history.pt")
    torch.save(history, history_path)
    print(f"📊 Saved training history → {history_path}")
    
    return model, best_acc


# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    args = get_args()
    print(f"[DEBUG] args.device = {args.device}")
    train(args)


if __name__ == "__main__":
    main()