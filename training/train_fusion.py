# training/train_fusion.py
"""
Fusion model training - combines features from LSTM, ViT, and YOLO.
"""
import torch
import torch.nn as nn
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.lstm import LSTMModel
from models.vit import ViTExtractor
from models.fusion import FusionNet, SimpleFusion
from models.yolo_detector import YOLOPatternDetector, MockYOLODetector
from utils.data_loader import DataLoader as MyDataLoader, DataConfig
from utils.candle_to_image import candle_image, normalize_for_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureDataset(torch.utils.data.Dataset):
    """Dataset that extracts features from pre-trained models."""
    
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        lstm_model: nn.Module,
        vit_model: nn.Module,
        yolo_detector,
        device: torch.device,
        seq_len: int = 60,
        img_size: int = 224,
    ):
        self.data = data
        self.labels = labels
        self.lstm_model = lstm_model
        self.vit_model = vit_model
        self.yolo_detector = yolo_detector
        self.device = device
        self.seq_len = seq_len
        self.img_size = img_size
        
        # Pre-extract features for efficiency
        self.features = self._extract_all_features()
    
    def _extract_all_features(self) -> Dict[str, torch.Tensor]:
        """Extract features from all samples."""
        logger.info("Extracting features from all samples...")
        
        lstm_feats = []
        vit_feats = []
        yolo_feats = []
        
        self.lstm_model.eval()
        self.vit_model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(len(self.data)), desc="Extracting"):
                # LSTM features
                seq = torch.tensor(self.data[i]).float().unsqueeze(0).to(self.device)
                lstm_feat = self.lstm_model(seq, mode='features')
                lstm_feats.append(lstm_feat.cpu())
                
                # Generate image from sequence
                # Convert scaled data back to approximate OHLCV for image
                import pandas as pd
                df = pd.DataFrame(
                    self.data[i],
                    columns=['open', 'high', 'low', 'close', 'tick_volume']
                )
                img = candle_image(df, target_size=self.img_size)
                img_norm = normalize_for_model(img)
                img_tensor = torch.tensor(img_norm).float().unsqueeze(0).to(self.device)
                
                # ViT features
                vit_feat = self.vit_model(img_tensor)
                vit_feats.append(vit_feat.cpu())
                
                # YOLO features
                yolo_vec = self.yolo_detector.detect(img)
                yolo_feats.append(torch.tensor(yolo_vec).float())
        
        return {
            'lstm': torch.cat(lstm_feats, dim=0),
            'vit': torch.cat(vit_feats, dim=0),
            'yolo': torch.stack(yolo_feats, dim=0),
        }
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            self.features['lstm'][idx],
            self.features['vit'][idx],
            self.features['yolo'][idx],
            self.labels[idx],
        )


def train_fusion_model(
    data_path: str = "data/raw/eurusd_latest.csv",
    weights_dir: str = "models/weights",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    device: Optional[str] = None,
) -> Tuple[nn.Module, dict]:
    """
    Train fusion model using pre-trained component models.
    
    Args:
        data_path: Path to training data
        weights_dir: Directory with pre-trained weights
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Training device
    
    Returns:
        Trained fusion model and training history
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    weights_dir = Path(weights_dir)
    
    logger.info(f"Training fusion model on {device}")
    
    # 1. Load data
    config = DataConfig(sequence_length=60, label_strategy='ternary')
    loader = MyDataLoader(config)
    
    df = loader.load_csv(data_path)
    train_scaled, test_scaled, _ = loader.split_and_scale(df, split_ratio=0.8)
    
    X_train, y_train = loader.create_sequences(train_scaled)
    X_test, y_test = loader.create_sequences(test_scaled)
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 2. Load pre-trained models (frozen)
    lstm = LSTMModel().to(device).eval()
    vit = ViTExtractor().to(device).eval()
    
    lstm_path = weights_dir / "lstm_best.pt"
    if lstm_path.exists():
        lstm.load_state_dict(torch.load(lstm_path, map_location=device))
        logger.info("Loaded pre-trained LSTM")
    
    vit_path = weights_dir / "vit_best.pt"
    if vit_path.exists():
        vit.load_state_dict(torch.load(vit_path, map_location=device))
        logger.info("Loaded pre-trained ViT")
    
    # YOLO (or mock)
    yolo_path = weights_dir / "yolo_best.pt"
    if yolo_path.exists():
        yolo = YOLOPatternDetector(str(yolo_path))
    else:
        logger.warning("YOLO weights not found, using mock detector")
        yolo = MockYOLODetector()
    
    # Freeze feature extractors
    for param in lstm.parameters():
        param.requires_grad = False
    for param in vit.parameters():
        param.requires_grad = False
    
    # 3. Create feature datasets
    logger.info("Creating training dataset...")
    train_dataset = FeatureDataset(
        X_train, y_train, lstm, vit, yolo, device
    )
    
    logger.info("Creating test dataset...")
    test_dataset = FeatureDataset(
        X_test, y_test, lstm, vit, yolo, device
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Initialize fusion model
    fusion = FusionNet(
        lstm_dim=lstm.get_feature_dim(),
        vit_dim=vit.get_feature_dim(),
        yolo_dim=yolo.get_feature_dim() if hasattr(yolo, 'get_feature_dim') else 20,
        num_classes=3,
    ).to(device)
    
    # 5. Training setup
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # 6. Training loop
    best_acc = 0.0
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # Train
        fusion.train()
        train_loss = 0.0
        
        for lstm_f, vit_f, yolo_f, labels in train_loader:
            lstm_f = lstm_f.to(device)
            vit_f = vit_f.to(device)
            yolo_f = yolo_f.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = fusion(lstm_f, vit_f, yolo_f)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Evaluate
        fusion.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for lstm_f, vit_f, yolo_f, labels in test_loader:
                lstm_f = lstm_f.to(device)
                vit_f = vit_f.to(device)
                yolo_f = yolo_f.to(device)
                labels = labels.to(device)
                
                outputs = fusion(lstm_f, vit_f, yolo_f)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = weights_dir / "fusion_best.pt"
            torch.save(fusion.state_dict(), save_path)
            logger.info(f"Saved best fusion model (acc: {best_acc:.2%})")
        
        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.2%}"
            )
    
    logger.info(f"Training complete. Best accuracy: {best_acc:.2%}")
    return fusion, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data/raw/eurusd_latest.csv")
    parser.add_argument('--weights-dir', default="models/weights")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()
    
    train_fusion_model(
        data_path=args.data,
        weights_dir=args.weights_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
