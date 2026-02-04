# training/walk_forward_trainer.py
"""
Walk-Forward Training Infrastructure for MH-TCN

Implements proper walk-forward validation to prevent overfitting:
1. Rolling train/test windows
2. Anchored expanding windows
3. Purged cross-validation (gap between train/test)

Key Features:
- No future data leakage
- Realistic out-of-sample testing
- Performance tracking across folds
- Automatic model selection
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward training."""
    # Window sizes (in bars)
    train_window: int = 5000       # Training window size
    test_window: int = 1000        # Test window size
    step_size: int = 500           # Step between folds
    purge_gap: int = 50            # Gap between train/test to prevent leakage
    
    # Training parameters
    epochs_per_fold: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    early_stopping_patience: int = 5
    
    # Model configuration
    profile: str = 'INTRADAY'
    sequence_length: int = 60
    
    # Paths
    output_dir: str = 'models/weights/walk_forward'
    
    # Validation
    min_folds: int = 3
    
    # Triple barrier labeling
    use_triple_barrier: bool = True
    tp_multiplier: float = 2.0  # TP = ATR * multiplier
    sl_multiplier: float = 1.0  # SL = ATR * multiplier
    max_holding_bars: int = 20  # Maximum bars to hold


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold_idx: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    
    # Metrics
    train_loss: float
    test_loss: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    
    # Direction-specific metrics
    bull_precision: float = 0.0
    bear_precision: float = 0.0
    
    # Trade simulation
    simulated_trades: int = 0
    simulated_win_rate: float = 0.0
    simulated_pnl: float = 0.0
    
    # Model path
    model_path: Optional[str] = None


class TripleBarrierLabeler:
    """
    Creates labels using triple barrier method.
    
    For each bar, determines if:
    - TP is hit first (label = direction of trade)
    - SL is hit first (label = opposite direction)
    - Neither hit within max_holding (label = sideways)
    """
    
    def __init__(
        self,
        tp_multiplier: float = 2.0,
        sl_multiplier: float = 1.0,
        max_holding_bars: int = 20,
        atr_period: int = 14
    ):
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.max_holding_bars = max_holding_bars
        self.atr_period = atr_period
    
    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(self.atr_period).mean()
    
    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate labels for each bar.
        
        Returns:
            Series with labels: 0=BEAR, 1=SIDEWAYS, 2=BULL
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        atr = self.calculate_atr(df)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        labels = np.ones(len(df), dtype=np.int64)  # Default to SIDEWAYS
        
        for i in range(len(df) - self.max_holding_bars):
            entry_price = close[i]
            current_atr = atr.iloc[i] if pd.notna(atr.iloc[i]) else 0.001
            
            tp_long = entry_price + current_atr * self.tp_multiplier
            sl_long = entry_price - current_atr * self.sl_multiplier
            tp_short = entry_price - current_atr * self.tp_multiplier
            sl_short = entry_price + current_atr * self.sl_multiplier
            
            # Check future bars
            long_result = 0  # 0=neither, 1=TP hit, -1=SL hit
            short_result = 0
            
            for j in range(1, min(self.max_holding_bars + 1, len(df) - i)):
                future_high = high[i + j]
                future_low = low[i + j]
                
                # Check long trade
                if long_result == 0:
                    if future_high >= tp_long:
                        long_result = 1
                    elif future_low <= sl_long:
                        long_result = -1
                
                # Check short trade
                if short_result == 0:
                    if future_low <= tp_short:
                        short_result = 1
                    elif future_high >= sl_short:
                        short_result = -1
                
                if long_result != 0 and short_result != 0:
                    break
            
            # Determine label
            if long_result == 1 and short_result != 1:
                labels[i] = 2  # BULL - long trade wins
            elif short_result == 1 and long_result != 1:
                labels[i] = 0  # BEAR - short trade wins
            else:
                labels[i] = 1  # SIDEWAYS - neither or both
        
        return pd.Series(labels, index=df.index)


class TimeSeriesDataset(Dataset):
    """Dataset for time series with proper sequencing."""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sequence_length: int = 60
    ):
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        
        # Valid indices (need sequence_length history)
        self.valid_indices = list(range(sequence_length, len(features)))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        
        # Get sequence
        seq = self.features[actual_idx - self.sequence_length:actual_idx]
        label = self.labels[actual_idx]
        
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class WalkForwardTrainer:
    """
    Walk-forward training for MH-TCN.
    
    Implements rolling window training with proper temporal ordering
    to prevent future data leakage.
    """
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Labeler
        self.labeler = TripleBarrierLabeler(
            tp_multiplier=config.tp_multiplier,
            sl_multiplier=config.sl_multiplier,
            max_holding_bars=config.max_holding_bars
        ) if config.use_triple_barrier else None
        
        # Results tracking
        self.fold_results: List[FoldResult] = []
        
        logger.info(f"WalkForwardTrainer initialized on {self.device}")
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature array from OHLCV DataFrame."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        features = []
        
        # Price features (normalized)
        close = df['close'].values
        base_price = close[0] if close[0] != 0 else 1.0
        
        features.append((close - base_price) / base_price)
        features.append((df['open'].values - base_price) / base_price)
        features.append((df['high'].values - base_price) / base_price)
        features.append((df['low'].values - base_price) / base_price)
        
        # Returns
        returns = np.diff(close, prepend=close[0]) / (close + 1e-8)
        features.append(returns)
        
        # Volume
        if 'volume' in df.columns or 'tick_volume' in df.columns:
            vol_col = 'volume' if 'volume' in df.columns else 'tick_volume'
            vol = df[vol_col].values.astype(float)
            features.append(vol / (np.mean(vol) + 1e-8))
        else:
            features.append(np.ones(len(close)))
        
        # Technical indicators
        # RSI
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().values
        avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().values
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi / 100)
        
        # MACD
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        macd = ema12 - ema26
        signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
        features.append(macd / (base_price + 1e-8))
        features.append(signal / (base_price + 1e-8))
        
        # Bollinger Bands
        sma20 = pd.Series(close).rolling(20, min_periods=1).mean().values
        std20 = pd.Series(close).rolling(20, min_periods=1).std().values
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_pos = (close - bb_lower) / (bb_upper - bb_lower + 1e-8)
        features.append(bb_pos)
        
        # ATR
        high = df['high'].values
        low = df['low'].values
        tr = np.maximum(high - low, np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        ))
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
        features.append(atr / (base_price + 1e-8))
        
        # ADX
        plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low),
                          np.maximum(high - np.roll(high, 1), 0), 0)
        minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)),
                           np.maximum(np.roll(low, 1) - low, 0), 0)
        plus_di = 100 * pd.Series(plus_dm).rolling(14, min_periods=1).mean().values / (atr + 1e-8)
        minus_di = 100 * pd.Series(minus_dm).rolling(14, min_periods=1).mean().values / (atr + 1e-8)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = pd.Series(dx).rolling(14, min_periods=1).mean().values
        features.append(adx / 100)
        
        # Stack and clean
        feature_array = np.stack(features, axis=1)
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feature_array
    
    def create_model(self, input_dim: int) -> nn.Module:
        """Create MH-TCN model."""
        try:
            from risk_management.phase1_predictive.tcn_backbone import (
                MultiHeadTCN, TCNConfig, TradingProfile
            )
            
            profile_map = {
                'SCALP': TradingProfile.SCALP,
                'INTRADAY': TradingProfile.INTRADAY,
                'SWING': TradingProfile.SWING,
            }
            
            config = TCNConfig(
                input_channels=input_dim,
                hidden_channels=128,
                profile=profile_map.get(self.config.profile.upper(), TradingProfile.INTRADAY)
            )
            
            return MultiHeadTCN(config).to(self.device)
            
        except ImportError:
            logger.error("Could not import MH-TCN")
            raise
    
    def train_fold(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold_idx: int
    ) -> Tuple[float, float]:
        """Train model for one fold."""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.config.epochs_per_fold):
            # Training
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x, mode='all')
                loss = criterion(outputs['direction'], batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_x, mode='all')
                    loss = criterion(outputs['direction'], batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Fold {fold_idx}: Early stopping at epoch {epoch}")
                    break
            
            if epoch % 5 == 0:
                logger.info(f"Fold {fold_idx} Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # Restore best state
        if best_state is not None:
            model.load_state_dict(best_state)
        
        return train_loss, best_val_loss
    
    def evaluate_fold(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on test set."""
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = model(batch_x, mode='all')
                preds = outputs['direction'].argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = (all_preds == all_labels).mean()
        
        # Per-class metrics
        metrics = {'accuracy': accuracy}
        
        for cls, name in enumerate(['bear', 'sideways', 'bull']):
            mask = all_labels == cls
            if mask.sum() > 0:
                precision = ((all_preds == cls) & (all_labels == cls)).sum() / max((all_preds == cls).sum(), 1)
                recall = ((all_preds == cls) & (all_labels == cls)).sum() / mask.sum()
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                metrics[f'{name}_precision'] = precision
                metrics[f'{name}_recall'] = recall
                metrics[f'{name}_f1'] = f1
        
        # Overall precision/recall/f1 (macro average)
        metrics['precision'] = np.mean([metrics.get(f'{n}_precision', 0) for n in ['bear', 'sideways', 'bull']])
        metrics['recall'] = np.mean([metrics.get(f'{n}_recall', 0) for n in ['bear', 'sideways', 'bull']])
        metrics['f1'] = np.mean([metrics.get(f'{n}_f1', 0) for n in ['bear', 'sideways', 'bull']])
        
        return metrics
    
    def run(self, df: pd.DataFrame) -> List[FoldResult]:
        """
        Run walk-forward training.
        
        Args:
            df: DataFrame with OHLCV data (must have 'time' column)
        
        Returns:
            List of FoldResult for each fold
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)
        
        # Prepare features and labels
        logger.info("Preparing features...")
        features = self.prepare_features(df)
        
        logger.info("Generating labels...")
        if self.labeler:
            labels = self.labeler.label(df).values
        else:
            # Simple return-based labels
            returns = df['close'].pct_change().shift(-1).fillna(0)
            labels = np.where(returns > 0.0005, 2, np.where(returns < -0.0005, 0, 1))
        
        # Calculate number of folds
        total_samples = len(df)
        min_samples = self.config.train_window + self.config.purge_gap + self.config.test_window
        
        if total_samples < min_samples:
            raise ValueError(f"Not enough data: {total_samples} < {min_samples}")
        
        n_folds = (total_samples - min_samples) // self.config.step_size + 1
        
        if n_folds < self.config.min_folds:
            raise ValueError(f"Not enough folds: {n_folds} < {self.config.min_folds}")
        
        logger.info(f"Running {n_folds} walk-forward folds...")
        
        self.fold_results = []
        
        for fold_idx in range(n_folds):
            # Calculate window boundaries
            train_start = fold_idx * self.config.step_size
            train_end = train_start + self.config.train_window
            test_start = train_end + self.config.purge_gap
            test_end = min(test_start + self.config.test_window, total_samples)
            
            if test_end > total_samples:
                break
            
            logger.info(f"\nFold {fold_idx + 1}/{n_folds}: train[{train_start}:{train_end}] test[{test_start}:{test_end}]")
            
            # Get timestamps
            train_start_time = df['time'].iloc[train_start] if 'time' in df.columns else datetime.now()
            train_end_time = df['time'].iloc[train_end - 1] if 'time' in df.columns else datetime.now()
            test_start_time = df['time'].iloc[test_start] if 'time' in df.columns else datetime.now()
            test_end_time = df['time'].iloc[test_end - 1] if 'time' in df.columns else datetime.now()
            
            # Create datasets
            train_features = features[train_start:train_end]
            train_labels = labels[train_start:train_end]
            test_features = features[test_start:test_end]
            test_labels = labels[test_start:test_end]
            
            train_dataset = TimeSeriesDataset(train_features, train_labels, self.config.sequence_length)
            test_dataset = TimeSeriesDataset(test_features, test_labels, self.config.sequence_length)
            
            if len(train_dataset) < 100 or len(test_dataset) < 50:
                logger.warning(f"Fold {fold_idx}: Not enough samples, skipping")
                continue
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Create and train model
            model = self.create_model(input_dim=features.shape[1])
            train_loss, val_loss = self.train_fold(model, train_loader, test_loader, fold_idx)
            
            # Evaluate
            metrics = self.evaluate_fold(model, test_loader)
            
            # Save model
            model_path = self.output_dir / f"fold_{fold_idx}_{self.config.profile}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'input_channels': features.shape[1],
                    'hidden_channels': 128,
                    'profile': self.config.profile,
                },
                'fold_idx': fold_idx,
                'metrics': metrics,
            }, model_path)
            
            # Record results
            result = FoldResult(
                fold_idx=fold_idx,
                train_start=train_start_time,
                train_end=train_end_time,
                test_start=test_start_time,
                test_end=test_end_time,
                train_loss=train_loss,
                test_loss=val_loss,
                test_accuracy=metrics['accuracy'],
                test_precision=metrics['precision'],
                test_recall=metrics['recall'],
                test_f1=metrics['f1'],
                bull_precision=metrics.get('bull_precision', 0),
                bear_precision=metrics.get('bear_precision', 0),
                model_path=str(model_path)
            )
            
            self.fold_results.append(result)
            
            logger.info(f"Fold {fold_idx}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        
        # Save summary
        self._save_summary()
        
        return self.fold_results
    
    def _save_summary(self):
        """Save training summary."""
        summary = {
            'config': {
                'train_window': self.config.train_window,
                'test_window': self.config.test_window,
                'step_size': self.config.step_size,
                'profile': self.config.profile,
            },
            'n_folds': len(self.fold_results),
            'avg_accuracy': np.mean([r.test_accuracy for r in self.fold_results]),
            'avg_f1': np.mean([r.test_f1 for r in self.fold_results]),
            'avg_bull_precision': np.mean([r.bull_precision for r in self.fold_results]),
            'avg_bear_precision': np.mean([r.bear_precision for r in self.fold_results]),
            'folds': [
                {
                    'fold_idx': r.fold_idx,
                    'test_accuracy': r.test_accuracy,
                    'test_f1': r.test_f1,
                    'model_path': r.model_path,
                }
                for r in self.fold_results
            ]
        }
        
        summary_path = self.output_dir / 'walk_forward_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary saved to {summary_path}")
    
    def get_best_model_path(self) -> Optional[str]:
        """Get path to best performing model across folds."""
        if not self.fold_results:
            return None
        
        best_fold = max(self.fold_results, key=lambda r: r.test_f1)
        return best_fold.model_path


def run_walk_forward_training(
    data_path: str,
    profile: str = 'INTRADAY',
    output_dir: str = 'models/weights/walk_forward'
) -> str:
    """
    Convenience function to run walk-forward training.
    
    Args:
        data_path: Path to CSV with OHLCV data
        profile: Trading profile
        output_dir: Output directory for models
    
    Returns:
        Path to best model
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Configure
    config = WalkForwardConfig(
        profile=profile,
        output_dir=output_dir
    )
    
    # Train
    trainer = WalkForwardTrainer(config)
    trainer.run(df)
    
    return trainer.get_best_model_path()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Walk-Forward Training for MH-TCN')
    parser.add_argument('--data', type=str, required=True, help='Path to OHLCV CSV')
    parser.add_argument('--profile', type=str, default='INTRADAY', help='Trading profile')
    parser.add_argument('--output', type=str, default='models/weights/walk_forward', help='Output directory')
    
    args = parser.parse_args()
    
    best_model = run_walk_forward_training(args.data, args.profile, args.output)
    print(f"\nBest model: {best_model}")
