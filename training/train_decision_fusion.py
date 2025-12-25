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
from typing import Dict, List, Optional, Tuple, Union
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.decision_fusion import DecisionFusionLayer, DecisionOutput
from models.vit_extractor import ViTExtractor
from training.train_tcn_enhanced import EnhancedTCN
from utils.candle_to_image import CandlestickRenderer
from models.yolo_detector import YOLOPatternDetector
from utils.checkpoint_loader import ModelLoader
from utils.mtf_config import Timeframe, get_profile
from utils.feature_schema import get_feature_schema_version
from utils.training_utils import copy_schema_tagged, set_global_seed
# RiskManager is used by DecisionFusionLayer internally, not needed here

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _ViTHead(torch.nn.Module):
    def __init__(self, in_features: int = 768, hidden_dim: int = 256, num_classes: int = 3, dropout: float = 0.2):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.block(x)
        return self.classifier(x)


def _load_vit_head(vit_checkpoint: Optional[str], device: str) -> Optional[torch.nn.Module]:
    if not vit_checkpoint:
        return None
    p = Path(vit_checkpoint)
    if not p.exists():
        logger.warning(f"ViT checkpoint not found: {p}")
        return None

    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state') if isinstance(ckpt, dict) else None
    if not isinstance(state, dict):
        logger.warning(f"Unexpected ViT checkpoint format: {p}")
        return None

    head = _ViTHead().to(device)
    head.load_state_dict(state)
    head.eval()
    return head


def _load_tcn_checkpoint(tcn_checkpoint: Optional[str], device: str) -> Tuple[EnhancedTCN, List[str]]:
    if tcn_checkpoint:
        p = Path(tcn_checkpoint)
        if p.exists():
            loader = ModelLoader(p, device=device)
            model = loader.get_model(device=device)
            features = loader.get_features_safe(fallback=[])
            return model, (features or [])
        logger.warning(f"TCN checkpoint not found: {p}")

    model = EnhancedTCN.from_profile('INTRADAY', input_dim=4).to(device).eval()
    return model, ['open', 'high', 'low', 'close']


def _load_tcn_checkpoint_for_profile(
    tcn_checkpoint: Optional[str],
    profile: str,
    device: str,
) -> Tuple[EnhancedTCN, List[str]]:
    model, features = _load_tcn_checkpoint(tcn_checkpoint, device=device)
    if tcn_checkpoint and Path(tcn_checkpoint).exists():
        return model, features

    # If checkpoint missing, at least match profile architecture
    fallback = EnhancedTCN.from_profile(profile.upper(), input_dim=max(4, len(features) or 4)).to(device).eval()
    return fallback, (features or ['open', 'high', 'low', 'close'])


def _load_yolo_detector(yolo_checkpoint: Optional[str], include_confidence: bool = False) -> YOLOPatternDetector:
    if yolo_checkpoint:
        p = Path(yolo_checkpoint)
        if p.exists():
            return YOLOPatternDetector(model_path=str(p), include_confidence=include_confidence)
        logger.warning(f"YOLO checkpoint not found: {p}")

    return YOLOPatternDetector(include_confidence=include_confidence)


def _extract_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    cols = []
    for c in feature_cols:
        cc = c
        if cc == 'volume' and 'volume' not in df.columns and 'tick_volume' in df.columns:
            cc = 'tick_volume'
        cols.append(cc)

    mat = []
    for cc in cols:
        if cc in df.columns:
            mat.append(df[cc].to_numpy(dtype=np.float32))
        else:
            mat.append(np.zeros((len(df),), dtype=np.float32))

    if not mat:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.stack(mat, axis=1)


def _resolve_data_csv(timeframe: str) -> str:
    return f"data/raw/EURUSD_{timeframe.upper()}_latest.csv"


def _resolve_tcn_checkpoint(profile: str, timeframe: str) -> str:
    return f"models/weights/{profile.lower()}_{timeframe.lower()}_best.pt"


def _resolve_yolo_checkpoint(profile: str) -> str:
    run_path = Path(f"runs/yolo/{profile.lower()}_train/weights/best.pt")
    if run_path.exists():
        return str(run_path)
    return f"models/yolo/yolo_{profile.lower()}.pt"


def _resolve_vit_checkpoint(profile: str) -> str:
    return f"models/vit/vit_{profile.lower()}.pt"


def _default_hparams_for_timeframe(timeframe: str) -> Dict[str, Union[int, float]]:
    tf = timeframe.upper()
    if tf == 'M5':
        return {'sequence_length': 30, 'window_bars': 30, 'stride': 5, 'forward_bars': 6, 'threshold_pct': 0.15}
    if tf == 'M15':
        return {'sequence_length': 60, 'window_bars': 60, 'stride': 10, 'forward_bars': 10, 'threshold_pct': 0.3}
    if tf == 'H1':
        return {'sequence_length': 60, 'window_bars': 60, 'stride': 10, 'forward_bars': 10, 'threshold_pct': 0.3}
    if tf == 'H4':
        return {'sequence_length': 90, 'window_bars': 90, 'stride': 15, 'forward_bars': 15, 'threshold_pct': 0.5}
    if tf == 'D1':
        return {'sequence_length': 120, 'window_bars': 120, 'stride': 20, 'forward_bars': 20, 'threshold_pct': 1.0}
    return {'sequence_length': 60, 'window_bars': 60, 'stride': 10, 'forward_bars': 10, 'threshold_pct': 0.3}


class MultiModalDataset(Dataset):
    """
    Dataset for multi-modal training with YOLO, ViT, and TCN features.
    """
    
    def __init__(
        self,
        data_csv: str,
        profile: str,
        timeframe: str,
        device: str = 'cpu',
        image_size: int = 224,
        sequence_length: int = 60,
        window_bars: int = 60,
        stride: int = 10,
        forward_bars: int = 10,
        threshold_pct: float = 0.3,
        mode: str = 'train',
        max_samples: Optional[int] = None,
        vit_checkpoint: Optional[str] = None,
        tcn_checkpoint: Optional[str] = None,
        yolo_checkpoint: Optional[str] = None,
        yolo_include_confidence: bool = False,
    ):
        self.data = pd.read_csv(data_csv)
        self.profile = profile.upper()
        self.timeframe = timeframe.upper()
        self.device = device

        self.image_size = image_size
        self.sequence_length = sequence_length
        self.window_bars = window_bars
        self.stride = stride
        self.forward_bars = forward_bars
        self.threshold_pct = threshold_pct
        self.mode = mode
        self.max_samples = max_samples
        
        # Initialize components
        self.vit_extractor = ViTExtractor(pretrained=True, freeze=True).eval()
        self.vit_head = _load_vit_head(vit_checkpoint, device=self.device)

        self.tcn_model, self.tcn_feature_columns = _load_tcn_checkpoint_for_profile(
            tcn_checkpoint,
            profile=self.profile,
            device=self.device,
        )

        self.yolo_detector = _load_yolo_detector(yolo_checkpoint, include_confidence=yolo_include_confidence)
        self.renderer = CandlestickRenderer(image_size=(image_size, image_size))
        
        # Generate samples
        self.samples = self._generate_samples()
        
        logger.info(f"Created {mode} dataset with {len(self.samples)} samples")
    
    def _generate_samples(self) -> List[Dict]:
        """Generate training samples from OHLCV data."""
        samples = []
        
        for i in range(0, len(self.data) - self.window_bars - self.forward_bars, self.stride):
            # Extract window
            window = self.data.iloc[i:i + self.window_bars].copy()
            future = self.data.iloc[i + self.window_bars:i + self.window_bars + self.forward_bars]
            
            # Skip if insufficient future data
            if len(future) < self.forward_bars:
                continue
            
            # Calculate labels
            current_price = window['close'].iloc[-1]
            future_prices = future['close'].values
            
            # Direction label (3-class)
            pct_change = (future_prices[-1] - current_price) / current_price * 100.0
            
            if pct_change > self.threshold_pct:
                direction = 2  # BULLISH
            elif pct_change < -self.threshold_pct:
                direction = 0  # BEARISH
            else:
                direction = 1  # SIDEWAYS
            
            # Confidence based on price move magnitude
            confidence = min(abs(pct_change) / self.threshold_pct, 1.0) if self.threshold_pct > 0 else 0.0
            
            # TCN stability (prediction variance)
            tcn_features = self._extract_tcn_features(window)
            tcn_stability = torch.var(tcn_features).unsqueeze(0)
            
            # ViT entropy (attention entropy)
            chart_image = self._render_chart(window)
            vit_features, vit_entropy = self._extract_vit_features(chart_image)
            
            # YOLO patterns and confidences
            yolo_features, yolo_confidence = self._extract_yolo_features(chart_image)
            
            sample = {
                'tcn_features': tcn_features,
                'tcn_stability': tcn_stability,
                'vit_features': vit_features,
                'vit_entropy': vit_entropy,
                'yolo_features': yolo_features,
                'yolo_confidence': yolo_confidence,
                'direction': torch.tensor(direction, dtype=torch.long),
                'confidence': torch.tensor(confidence, dtype=torch.float32),
                'price_change': torch.tensor(pct_change, dtype=torch.float32)
            }
            
            samples.append(sample)

            if self.max_samples is not None and len(samples) >= self.max_samples:
                break
        
        return samples

    def _extract_tcn_features(self, window: pd.DataFrame) -> torch.Tensor:
        """Extract TCN features from OHLCV data."""
        feature_cols = self.tcn_feature_columns
        if not feature_cols:
            feature_cols = ['open', 'high', 'low', 'close']

        w = window.tail(self.sequence_length)
        features = _extract_feature_matrix(w, feature_cols)
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Extract features using TCN
        with torch.no_grad():
            tcn_out = self.tcn_model(features_tensor.to(self.device), mode='features')
            tcn_features = tcn_out.squeeze(0).detach().cpu()
        
        return tcn_features
    
    def _render_chart(self, window: pd.DataFrame) -> np.ndarray:
        """Render candlestick chart image."""
        return self.renderer.render(window)
    
    def _extract_vit_features(self, chart_image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract ViT features and compute attention entropy."""
        with torch.no_grad():
            # Convert image to tensor
            image_tensor = torch.FloatTensor(chart_image).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            # Extract features
            vit_features = self.vit_extractor(image_tensor)  # (1, 768)
            
            # Compute entropy from profile-specific ViT classifier head probabilities
            if self.vit_head is not None:
                logits = self.vit_head(vit_features)
                probs = torch.softmax(logits, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
            else:
                entropy = torch.zeros((1, 1), device=vit_features.device)
        
        return vit_features.squeeze(0).detach().cpu(), entropy.detach().cpu()
    
    def _extract_yolo_features(self, chart_image: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract YOLO pattern features and confidences."""
        vec = self.yolo_detector.detect(chart_image)
        vec_t = torch.tensor(vec, dtype=torch.float32)

        if self.yolo_detector.include_confidence:
            num = self.yolo_detector.num_classes
            conf_part = vec_t[num:]
            conf = conf_part[conf_part > 0].mean() if (conf_part > 0).any() else torch.tensor(0.0)
        else:
            conf = vec_t.mean() if vec_t.numel() else torch.tensor(0.0)

        return vec_t, conf
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def _log_class_distribution(name: str, samples: List[Dict]):
    counts = {0: 0, 1: 0, 2: 0}
    for s in samples:
        counts[int(s['direction'].item())] += 1
    total = max(sum(counts.values()), 1)
    logger.info(
        f"{name} class distribution | "
        f"BEARISH={counts[0]} ({counts[0]/total:.2%}), "
        f"SIDEWAYS={counts[1]} ({counts[1]/total:.2%}), "
        f"BULLISH={counts[2]} ({counts[2]/total:.2%})"
    )


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

        schema_version = get_feature_schema_version()
        
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
                    'history': self.history,
                    'feature_schema_version': schema_version,
                }, save_dir / 'best_model.pt')

                copy_schema_tagged(save_dir / 'best_model.pt', schema_version)
                
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
            'history': self.history,
            'feature_schema_version': schema_version,
        }, save_dir / 'final_model.pt')

        copy_schema_tagged(save_dir / 'final_model.pt', schema_version)
        
        # Save training history
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        copy_schema_tagged(save_dir / 'training_history.json', schema_version)
        
        return best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train Decision Fusion Layer")
    parser.add_argument('--profile', type=str, default='INTRADAY', 
                       choices=['SCALP', 'INTRADAY', 'SWING'])
    parser.add_argument('--timeframe', type=str, default=None)
    parser.add_argument('--data_csv', type=str, 
                       default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='models/decision_fusion')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vit_checkpoint', type=str, default=None)
    parser.add_argument('--tcn_checkpoint', type=str, default=None)
    parser.add_argument('--yolo_checkpoint', type=str, default=None)
    parser.add_argument('--yolo_include_confidence', action='store_true')
    
    args = parser.parse_args()

    set_global_seed(args.seed)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    
    logger.info(f"Training Decision Fusion on {args.device}")
    
    mtf_profile = get_profile(args.profile)
    timeframe = (args.timeframe or mtf_profile.primary_tf.value).upper()
    if timeframe not in mtf_profile.timeframe_strings:
        raise ValueError(
            f"Timeframe {timeframe} not allowed for profile {args.profile}. "
            f"Allowed: {mtf_profile.timeframe_strings}"
        )

    data_csv = args.data_csv or _resolve_data_csv(timeframe)
    if not Path(data_csv).exists():
        raise FileNotFoundError(f"Data CSV not found: {data_csv}")

    vit_checkpoint = args.vit_checkpoint or _resolve_vit_checkpoint(args.profile)

    tcn_checkpoint = args.tcn_checkpoint or _resolve_tcn_checkpoint(args.profile, timeframe)
    if not Path(tcn_checkpoint).exists():
        fallback = Path('models/weights/tcn_enhanced_best.pt')
        if fallback.exists():
            tcn_checkpoint = str(fallback)

    yolo_checkpoint = args.yolo_checkpoint or _resolve_yolo_checkpoint(args.profile)

    config = _default_hparams_for_timeframe(timeframe)

    # Create model
    yolo_dim = 40 if args.yolo_include_confidence else 20
    model = DecisionFusionLayer(
        yolo_dim=yolo_dim,
        vit_dim=768,
        tcn_dim=64,
        hidden_dim=256,
        num_classes=3,
        use_regime_conditioning=True
    )

    logger.info(f"Created Decision Fusion model for {args.profile} ({timeframe})")
    
    # Create dataset
    train_dataset = MultiModalDataset(
        data_csv=data_csv,
        profile=args.profile,
        timeframe=timeframe,
        device=args.device,
        image_size=224,
        mode='train',
        max_samples=args.max_samples,
        vit_checkpoint=vit_checkpoint,
        tcn_checkpoint=tcn_checkpoint,
        yolo_checkpoint=yolo_checkpoint,
        yolo_include_confidence=args.yolo_include_confidence,
        **config,
    )

    # Chronological split with purge gap to reduce leakage from overlapping windows
    n = len(train_dataset)
    split_idx = int(0.8 * n)
    purge_gap = int((config['window_bars'] + config['forward_bars']) / max(config['stride'], 1)) + 1
    val_start = min(split_idx + purge_gap, n)

    train_samples = train_dataset.samples[:split_idx]
    val_samples = train_dataset.samples[val_start:]
    _log_class_distribution("TRAIN", train_samples)
    _log_class_distribution("VAL", val_samples)

    # Wrap as datasets
    train_dataset = torch.utils.data.Subset(train_dataset, range(0, split_idx))
    val_dataset = torch.utils.data.Subset(train_dataset.dataset, range(val_start, n))
    
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
        save_dir=f"{args.save_dir}/{args.profile.lower()}/{timeframe.lower()}",
        patience=args.patience
    )
    
    logger.info(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
