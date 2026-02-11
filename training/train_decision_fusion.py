"""
Training script for the Decision Fusion Layer

Trains the production-grade multi-modal fusion network that combines
Price Action + TCN features with sophisticated gating and multi-task outputs.

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
import hashlib
from typing import Dict, List, Optional, Tuple, Union
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.decision_fusion import DecisionFusionLayer, DecisionOutput
from models.price_action_pattern import PriceActionPatternExtractor
from utils.checkpoint_loader import ModelLoader
from utils.mtf_config import Timeframe, get_profile
from utils.feature_schema import get_feature_schema_version
from utils.training_utils import copy_schema_tagged, set_global_seed

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_tcn_checkpoint(tcn_checkpoint: Optional[str], device: str):
    """Load a TCN checkpoint via ModelLoader. Returns (model, feature_columns)."""
    if tcn_checkpoint:
        p = Path(tcn_checkpoint)
        if p.exists():
            loader = ModelLoader(p, device=device)
            model = loader.get_model(device=device)
            features = loader.get_features_safe(fallback=[])
            return model, (features or [])
        logger.warning(f"TCN checkpoint not found: {p}")
    return None, ['open', 'high', 'low', 'close']


def _load_price_action_extractor(include_extended: bool = True, include_confidence: bool = False) -> PriceActionPatternExtractor:
    """Load the Price Action pattern extractor."""
    return PriceActionPatternExtractor(
        include_extended_patterns=include_extended,
        include_confidence=include_confidence
    )


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


def _resolve_price_action_checkpoint(profile: str) -> str:
    """Price Action doesn't need checkpoints - return empty string."""
    return ""


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
    Dataset for multi-modal training with Price Action + TCN features.
    """
    
    def __init__(
        self,
        profile: str,
        timeframe: str,
        csv_path: str,
        sequence_length: int = 60,
        window_bars: int = 30,
        stride: int = 5,
        forward_bars: int = 6,
        threshold_pct: float = 0.15,
        max_samples: Optional[int] = None,
        tcn_checkpoint: Optional[str] = None,
        price_action_include_extended: bool = True,
        price_action_include_confidence: bool = False,
        cache_dir: Optional[str] = None,
        use_cache: bool = False,
        refresh_cache: bool = False,
        device: str = 'auto',
        seed: int = 42,
        **_kwargs,
    ):
        self.profile = profile
        self.timeframe = timeframe
        self.data_csv = csv_path
        self.sequence_length = sequence_length
        self.window_bars = window_bars
        self.stride = stride
        self.forward_bars = forward_bars
        self.threshold_pct = threshold_pct
        self.max_samples = max_samples
        self.cache_dir = cache_dir
        self.use_cache = bool(use_cache)
        self.refresh_cache = bool(refresh_cache)
        self.tcn_checkpoint = tcn_checkpoint

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load data
        self.data = pd.read_csv(csv_path)
        for col in ['open', 'high', 'low', 'close']:
            if col not in self.data.columns:
                raise ValueError(f"CSV missing required column: {col}")
        
        # Initialize components
        self.tcn_model, self.tcn_feature_columns = _load_tcn_checkpoint(
            tcn_checkpoint, device=self.device
        )
        if self.tcn_model is not None:
            self.tcn_model.eval()

        self.price_action_extractor = _load_price_action_extractor(
            include_extended=price_action_include_extended,
            include_confidence=price_action_include_confidence
        )

        # Generate samples (optionally cached)
        self.samples = self._load_or_generate_samples()
        
        logger.info(f"Created dataset with {len(self.samples)} samples")

    def _cache_key_dict(self) -> Dict[str, Union[str, int, float, bool]]:
        def _mtime(p: Optional[str]) -> int:
            if not p:
                return 0
            try:
                return int(Path(p).stat().st_mtime)
            except Exception:
                return 0

        try:
            data_mtime = int(Path(self.data_csv).stat().st_mtime)
        except Exception:
            data_mtime = 0

        return {
            'data_csv': str(self.data_csv),
            'data_mtime': data_mtime,
            'profile': str(self.profile),
            'timeframe': str(self.timeframe),
            'sequence_length': int(self.sequence_length),
            'window_bars': int(self.window_bars),
            'stride': int(self.stride),
            'forward_bars': int(self.forward_bars),
            'threshold_pct': float(self.threshold_pct),
            'max_samples': int(self.max_samples) if self.max_samples is not None else -1,
            'tcn_checkpoint': str(self.tcn_checkpoint or ''),
            'tcn_checkpoint_mtime': _mtime(self.tcn_checkpoint),
        }

    def _cache_path(self) -> Optional[Path]:
        if not (self.use_cache and self.cache_dir):
            return None

        key_dict = self._cache_key_dict()
        payload = json.dumps(key_dict, sort_keys=True, ensure_ascii=True).encode('utf-8')
        key = hashlib.sha256(payload).hexdigest()[:16]

        root = Path(self.cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        return root / f"fusion_samples_{self.profile.lower()}_{self.timeframe.lower()}_{key}.pt"

    def _load_or_generate_samples(self) -> List[Dict]:
        cache_path = self._cache_path()
        if cache_path is not None and cache_path.exists() and (not self.refresh_cache):
            try:
                samples = torch.load(cache_path, map_location='cpu', weights_only=False)
                if isinstance(samples, list) and samples:
                    logger.info(f"Loaded cached fusion samples: {cache_path}")
                    return samples
                logger.warning(f"Cache exists but invalid/empty, regenerating: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load cache ({cache_path}): {e}. Regenerating...")

        samples = self._generate_samples()

        if cache_path is not None:
            try:
                torch.save(samples, cache_path)
                logger.info(f"Saved cached fusion samples: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache ({cache_path}): {e}")

        return samples
    
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
            
            # Price Action patterns and confidences
            price_action_features, price_action_confidence = self._extract_price_action_features(window)
            
            sample = {
                'tcn_features': tcn_features,
                'tcn_stability': tcn_stability,
                'price_action_features': price_action_features,
                'price_action_confidence': price_action_confidence,
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
        if self.tcn_model is None:
            # No TCN checkpoint — derive simple features from OHLCV
            w = window.tail(self.sequence_length)
            closes = w['close'].values.astype(np.float32)
            returns = np.diff(closes) / (closes[:-1] + 1e-8)
            # Pad to fixed 64-dim feature vector
            feat = np.zeros(64, dtype=np.float32)
            feat[:min(len(returns), 64)] = returns[:64]
            return torch.tensor(feat, dtype=torch.float32)

        feature_cols = self.tcn_feature_columns
        if not feature_cols:
            feature_cols = ['open', 'high', 'low', 'close']

        w = window.tail(self.sequence_length)
        features = _extract_feature_matrix(w, feature_cols)
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
        
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        with torch.no_grad():
            tcn_out = self.tcn_model(features_tensor.to(self.device), mode='features')
            tcn_features = tcn_out.squeeze(0).detach().cpu()
        
        return tcn_features
    
    def _extract_price_action_features(self, window: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract Price Action pattern features and confidences."""
        vec = self.price_action_extractor.extract(window)
        vec_t = torch.tensor(vec, dtype=torch.float32)

        if self.price_action_extractor.include_confidence:
            num = self.price_action_extractor.num_classes
            conf_part = vec_t[num:]
            conf = conf_part[conf_part > 0].mean() if (conf_part > 0).any() else torch.tensor(0.0)
        else:
            conf = torch.tensor(vec_t.mean().item(), dtype=torch.float32)

        return vec_t, conf
    
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
                price_action_features=batch['price_action_features'],
                price_action_confidence=batch['price_action_confidence'],
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
                    price_action_features=batch['price_action_features'],
                    price_action_confidence=batch['price_action_confidence'],
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
    parser.add_argument('--tcn_checkpoint', type=str, default=None)
    parser.add_argument('--price_action_include_extended', action='store_true', help='Include extended price action patterns')
    parser.add_argument('--price_action_include_confidence', action='store_true', help='Include confidence scores')
    parser.add_argument('--use_cache', action='store_true', help='Cache extracted fusion samples to disk')
    parser.add_argument('--refresh_cache', action='store_true', help='Rebuild cache even if present')
    parser.add_argument('--cache_dir', type=str, default='cache/decision_fusion', help='Cache directory for fusion samples')
    
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

    tcn_checkpoint = args.tcn_checkpoint or _resolve_tcn_checkpoint(args.profile, timeframe)
    if not Path(tcn_checkpoint).exists():
        fallback = Path('models/weights/tcn_enhanced_best.pt')
        if fallback.exists():
            tcn_checkpoint = str(fallback)

    price_action_checkpoint = _resolve_price_action_checkpoint(args.profile)

    config = _default_hparams_for_timeframe(timeframe)

    # Create model
    price_action_dim = 44 if args.price_action_include_extended else 25  # 25 primary + 19 extended
    if args.price_action_include_confidence:
        price_action_dim *= 2  # Double for confidence scores
    
    model = DecisionFusionLayer(
        price_action_dim=price_action_dim,
        tcn_dim=64,
        hidden_dim=256,
        num_classes=3,
        use_regime_conditioning=True
    ).to(args.device)

    logger.info(f"Created Decision Fusion model for {args.profile} ({timeframe})")
    
    # Create dataset
    train_dataset = MultiModalDataset(
        profile=args.profile,
        timeframe=timeframe,
        csv_path=data_csv,
        sequence_length=config['sequence_length'],
        window_bars=config['window_bars'],
        stride=config['stride'],
        forward_bars=config['forward_bars'],
        threshold_pct=config['threshold_pct'],
        device=args.device,
        max_samples=args.max_samples,
        tcn_checkpoint=tcn_checkpoint,
        price_action_include_extended=args.price_action_include_extended,
        price_action_include_confidence=args.price_action_include_confidence,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        refresh_cache=args.refresh_cache,
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
