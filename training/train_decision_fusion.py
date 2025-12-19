"""
Training script for the Decision Fusion Layer

Trains the production-grade multi-modal fusion network that combines
YOLO + ViT + TCN features with sophisticated gating and multi-task outputs.

Usage:
    python train_decision_fusion.py --profile INTRADAY --epochs 100
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.decision_fusion import DecisionFusionLayer, DecisionOutput
from models.vit_extractor import ViTExtractor
from training.train_tcn_enhanced import EnhancedTCN
from utils.candle_to_image import CandlestickRenderer
from utils.pattern_detector import CandlestickPatternDetector
from risk_management.risk_manager import RiskManager, RiskManagerConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """
    Dataset for multi-modal training with YOLO, ViT, and TCN features.
    """
    
    def __init__(
        self,
        data_csv: str,
        sequence_length: int = 60,
        window_size: int = 224,
        stride: int = 10,
        forward_bars: int = 10,
        threshold_pct: float = 0.3,
        mode: str = 'train'
    ):
        self.data = pd.read_csv(data_csv)
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.stride = stride
        self.forward_bars = forward_bars
        self.threshold_pct = threshold_pct
        self.mode = mode
        
        # Initialize components
        self.vit_extractor = ViTExtractor()
        self.tcn_model = EnhancedTCN.from_profile('INTRADAY', input_dim=4)
        self.pattern_detector = CandlestickPatternDetector()
        self.renderer = CandlestickRenderer(image_size=(window_size, window_size))
        
        # Generate samples
        self.samples = self._generate_samples()
        
        logger.info(f"Created {mode} dataset with {len(self.samples)} samples")
    
    def _generate_samples(self) -> List[Dict]:
        """Generate training samples from OHLCV data."""
        samples = []
        
        for i in range(0, len(self.data) - self.window_size - self.forward_bars, self.stride):
            # Extract window
            window = self.data.iloc[i:i + self.window_size].copy()
            future = self.data.iloc[i + self.window_size:i + self.window_size + self.forward_bars]
            
            # Skip if insufficient future data
            if len(future) < self.forward_bars:
                continue
            
            # Calculate labels
            current_price = window['close'].iloc[-1]
            future_prices = future['close'].values
            
            # Direction label (3-class)
            price_change = (future_prices[-1] - current_price) / current_price
            
            if price_change > self.threshold_pct:
                direction = 2  # BULLISH
            elif price_change < -self.threshold_pct:
                direction = 0  # BEARISH
            else:
                direction = 1  # SIDEWAYS
            
            # Confidence based on price move magnitude
            confidence = min(abs(price_change) / self.threshold_pct, 1.0)
            
            # TCN stability (prediction variance)
            tcn_features = self._extract_tcn_features(window)
            tcn_stability = torch.var(tcn_features).unsqueeze(0)
            
            # ViT entropy (attention entropy)
            chart_image = self._render_chart(window)
            vit_features, vit_entropy = self._extract_vit_features(chart_image)
            
            # YOLO patterns and confidences
            yolo_features, yolo_confidence = self._extract_yolo_features(window)
            
            sample = {
                'tcn_features': tcn_features,
                'tcn_stability': tcn_stability,
                'vit_features': vit_features,
                'vit_entropy': vit_entropy,
                'yolo_features': yolo_features,
                'yolo_confidence': yolo_confidence,
                'direction': torch.tensor(direction, dtype=torch.long),
                'confidence': torch.tensor(confidence, dtype=torch.float32),
                'price_change': torch.tensor(price_change, dtype=torch.float32)
            }
            
            samples.append(sample)
        
        return samples
    
    def _extract_tcn_features(self, window: pd.DataFrame) -> torch.Tensor:
        """Extract TCN features from OHLCV data."""
        # Normalize OHLCV (use available columns)
        available_cols = ['open', 'high', 'low', 'close']
        if 'volume' in window.columns:
            available_cols.append('volume')
        
        features = window[available_cols].values
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)  # (1, seq_len, 5)
        
        # Extract features using TCN
        with torch.no_grad():
            tcn_out = self.tcn_model(features_tensor, mode='features')
            tcn_features = tcn_out.squeeze(0)  # (hidden_dim,)
        
        return tcn_features
    
    def _render_chart(self, window: pd.DataFrame) -> np.ndarray:
        """Render candlestick chart image."""
        return self.renderer.render(window)
    
    def _extract_vit_features(self, chart_image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract ViT features and compute attention entropy."""
        with torch.no_grad():
            # Convert image to tensor
            image_tensor = torch.FloatTensor(chart_image).permute(2, 0, 1).unsqueeze(0)
            
            # Extract features
            vit_features = self.vit_extractor(image_tensor)  # (1, 768)
            
            # Compute attention entropy (simplified)
            # In practice, you'd extract attention weights from ViT
            attention_weights = torch.softmax(torch.randn(1, 12, 197, 197), dim=-1)
            entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1).mean()
            entropy = entropy.unsqueeze(0)  # (1, 1)
        
        return vit_features.squeeze(0), entropy
    
    def _extract_yolo_features(self, window: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract YOLO pattern features and confidences."""
        # Detect patterns
        patterns = self.pattern_detector.detect_all_patterns(window)
        
        # Convert to feature vector
        pattern_vector = torch.zeros(20)  # 20 pattern types
        confidences = []
        
        for pattern in patterns:
            if pattern.pattern_class < 20:
                pattern_vector[pattern.pattern_class] = pattern.confidence
                confidences.append(pattern.confidence)
        
        if not confidences:
            confidences = [0.0]
        
        return pattern_vector, torch.tensor(confidences, dtype=torch.float32).mean()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class DecisionFusionTrainer:
    """Trainer for the Decision Fusion Layer."""
    
    def __init__(
        self,
        model: DecisionFusionLayer,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.direction_criterion = nn.CrossEntropyLoss()
        self.confidence_criterion = nn.MSELoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training")
        
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            output: DecisionOutput = self.model(
                yolo_features=batch['yolo_features'],
                yolo_confidence=batch['yolo_confidence'],
                vit_features=batch['vit_features'],
                vit_entropy=batch['vit_entropy'],
                tcn_features=batch['tcn_features'],
                tcn_stability=batch['tcn_stability']
            )
            
            # Compute losses
            direction_loss = self.direction_criterion(
                output.direction_logits,
                batch['direction']
            )
            
            confidence_loss = self.confidence_criterion(
                output.confidence.squeeze(),
                batch['confidence']
            )
            
            # Total loss
            total_loss_batch = direction_loss + 0.5 * confidence_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_loss_batch.item()
            correct += (output.direction_label == batch['direction']).sum().item()
            total += batch['direction'].size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss_batch.item():.4f}",
                'acc': f"{correct/total:.4f}"
            })
        
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = correct / total
        
        return {'loss': epoch_loss, 'acc': epoch_acc}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                output: DecisionOutput = self.model(
                    yolo_features=batch['yolo_features'],
                    yolo_confidence=batch['yolo_confidence'],
                    vit_features=batch['vit_features'],
                    vit_entropy=batch['vit_entropy'],
                    tcn_features=batch['tcn_features'],
                    tcn_stability=batch['tcn_stability']
                )
                
                # Compute losses
                direction_loss = self.direction_criterion(
                    output.direction_logits,
                    batch['direction']
                )
                
                confidence_loss = self.confidence_criterion(
                    output.confidence.squeeze(),
                    batch['confidence']
                )
                
                total_loss_batch = direction_loss + 0.5 * confidence_loss
                
                # Update metrics
                total_loss += total_loss_batch.item()
                correct += (output.direction_label == batch['direction']).sum().item()
                total += batch['direction'].size(0)
        
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = correct / total
        
        return {'loss': epoch_loss, 'acc': epoch_acc}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_dir: str,
        patience: int = 15
    ):
        """Train the model."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save metrics
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            
            # Log results
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['acc']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['acc']:.4f}"
            )
            
            # Save best model
            if val_metrics['acc'] > best_val_acc:
                best_val_acc = val_metrics['acc']
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_metrics['acc'],
                    'history': self.history
                }, save_dir / 'best_model.pt')
                
                logger.info(f"New best model saved with val acc: {best_val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Save final model
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_metrics['acc'],
            'history': self.history
        }, save_dir / 'final_model.pt')
        
        # Save training history
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train Decision Fusion Layer")
    parser.add_argument('--profile', type=str, default='INTRADAY', 
                       choices=['SCALP', 'INTRADAY', 'SWING'])
    parser.add_argument('--data_csv', type=str, 
                       default='data/raw/EURUSD_H1_latest.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='models/decision_fusion')
    parser.add_argument('--patience', type=int, default=15)
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    logger.info(f"Training Decision Fusion on {args.device}")
    
    # Profile-specific configurations
    profile_configs = {
        'SCALP': {
            'sequence_length': 30,
            'window_size': 224,
            'stride': 5,
            'forward_bars': 6,
            'threshold_pct': 0.15
        },
        'INTRADAY': {
            'sequence_length': 60,
            'window_size': 224,
            'stride': 10,
            'forward_bars': 10,
            'threshold_pct': 0.3
        },
        'SWING': {
            'sequence_length': 90,
            'window_size': 224,
            'stride': 15,
            'forward_bars': 15,
            'threshold_pct': 0.5
        }
    }
    
    config = profile_configs[args.profile]
    
    # Create model
    model = DecisionFusionLayer(
        yolo_dim=20,
        vit_dim=768,
        tcn_dim=64,
        hidden_dim=256,
        num_classes=3,
        use_regime_conditioning=True
    )
    
    logger.info(f"Created Decision Fusion model for {args.profile}")
    
    # Create datasets
    train_dataset = MultiModalDataset(
        data_csv=args.data_csv,
        mode='train',
        **config
    )
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    # Create trainer
    trainer = DecisionFusionTrainer(
        model=model,
        device=args.device,
        learning_rate=args.learning_rate
    )
    
    # Train
    best_val_acc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_dir=f"{args.save_dir}/{args.profile.lower()}",
        patience=args.patience
    )
    
    logger.info(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
