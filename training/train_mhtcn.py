"""
Multi-Head TCN Training Module
==============================

Dedicated training script for the Multi-Head TCN model with full support for:
- Direction prediction (3-class: Bear, Sideways, Bull)
- Volatility prediction (regression)
- Quantile prediction (pinball loss for Q5, Q25, Q50, Q75, Q95)
- Outcome prediction (optional: P(TP before SL) for long/short)

This module wraps the existing infrastructure from risk_management.phase1_predictive
with proper data preparation and label generation.

Usage:
    python -m training.train_mhtcn --data data/EURUSD_H1.csv --profile INTRADAY
    python -m training.train_mhtcn --data data/EURUSD_M5.csv --profile SCALP --epochs 50

Or via main.py:
    python main.py train mhtcn --data data/EURUSD_H1.csv --profile INTRADAY
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def mhtcn_collate_fn(batch: List) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Custom collate function that handles None vision features.
    
    The RiskDataset returns (sequence, targets_dict, vision) where vision can be None.
    Standard collate_fn fails on None, so we handle it here.
    """
    sequences = torch.stack([item[0] for item in batch])
    
    # Collate target dictionaries
    targets = {}
    first_targets = batch[0][1]
    for key in first_targets.keys():
        targets[key] = torch.stack([item[1][key] for item in batch])
    
    # Handle vision (may be None)
    vision = None
    if batch[0][2] is not None:
        vision = torch.stack([item[2] for item in batch])
    
    return sequences, targets, vision

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def compute_optimal_threshold(
    close_prices: np.ndarray,
    horizon: int,
    target_imbalance_ratio: float = 1.5,
    search_range: Tuple[float, float] = (0.0001, 0.01),
    num_steps: int = 20
) -> float:
    """
    Dynamically compute the optimal direction threshold for balanced classes.
    
    This function searches for a threshold that produces the most balanced
    distribution of Bear/Sideways/Bull labels.
    
    Args:
        close_prices: Array of close prices
        horizon: Number of bars ahead for direction calculation
        target_imbalance_ratio: Maximum acceptable imbalance ratio (default 1.5:1)
        search_range: (min_threshold, max_threshold) to search
        num_steps: Number of threshold values to test
    
    Returns:
        Optimal threshold value that minimizes class imbalance
    """
    n = len(close_prices)
    
    # Calculate future returns for all samples
    future_returns = np.zeros(n)
    for i in range(n - horizon):
        future_returns[i] = (close_prices[i + horizon] - close_prices[i]) / close_prices[i]
    
    # Search for optimal threshold
    min_thresh, max_thresh = search_range
    thresholds = np.linspace(min_thresh, max_thresh, num_steps)
    
    best_threshold = 0.001  # Default fallback
    best_balance_score = float('inf')
    
    for thresh in thresholds:
        # Generate labels with this threshold
        labels = np.ones(n, dtype=np.int64)  # Default: Sideways
        labels[future_returns > thresh] = 2   # Bull
        labels[future_returns < -thresh] = 0  # Bear
        labels[-horizon:] = 1  # Last bars get Sideways
        
        # Count classes
        bear_count = np.sum(labels == 0)
        side_count = np.sum(labels == 1)
        bull_count = np.sum(labels == 2)
        
        # Skip if any class has too few samples
        min_count = min(bear_count, side_count, bull_count)
        if min_count < 100:
            continue
        
        # Calculate imbalance ratio
        max_count = max(bear_count, side_count, bull_count)
        imbalance_ratio = max_count / min_count
        
        # Calculate balance score (deviation from equal distribution)
        expected = n / 3
        balance_score = (
            abs(bear_count - expected) + 
            abs(side_count - expected) + 
            abs(bull_count - expected)
        ) / n
        
        # Prefer thresholds with low imbalance and good balance
        if imbalance_ratio <= target_imbalance_ratio and balance_score < best_balance_score:
            best_balance_score = balance_score
            best_threshold = thresh
    
    logger.info(f"Computed optimal threshold: {best_threshold:.6f} (balance_score={best_balance_score:.4f})")
    return best_threshold


@dataclass
class MHTCNTrainingConfig:
    """Configuration for MH-TCN training."""
    # Data
    data_path: str = ""
    profile: str = "INTRADAY"  # SCALP, INTRADAY, SWING
    
    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 15
    grad_clip: float = 1.0
    
    # Model
    sequence_length: int = 60
    hidden_channels: int = 128
    num_layers: int = 4
    dropout: float = 0.2
    
    # Labels
    direction_horizon: int = 12  # Bars ahead for direction label
    volatility_horizon: int = 12  # Bars for realized volatility
    direction_threshold: float = -1.0  # -1 means auto-compute from data
    auto_threshold: bool = True  # Automatically compute optimal threshold
    
    # Triple barrier (optional)
    use_triple_barrier: bool = True
    tp_multiplier: float = 2.0  # TP = ATR * multiplier
    sl_multiplier: float = 1.0  # SL = ATR * multiplier
    max_holding_bars: int = 24  # Max bars before timeout
    
    # Validation
    val_split: float = 0.15
    test_split: float = 0.10
    
    # Output
    output_dir: str = "models/weights"
    save_best: bool = True
    
    # Device
    device: str = "auto"
    
    # Class balancing
    use_weighted_sampler: bool = True  # Use WeightedRandomSampler for balanced batches


class MHTCNDataPreparer:
    """
    Prepares data for MH-TCN training with proper label generation.
    
    Generates:
    - Direction labels: 0 (Bear), 1 (Sideways), 2 (Bull)
    - Volatility labels: Realized volatility over horizon
    - Price move labels: Actual price movement for quantile regression
    - Outcome labels (optional): [y_long, y_short] for TP-before-SL
    """
    
    def __init__(self, config: MHTCNTrainingConfig):
        self.config = config
        self._feature_engineer = None
    
    def load_data(self, path: str) -> pd.DataFrame:
        """Load and validate OHLCV data."""
        logger.info(f"Loading data from {path}")
        
        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Validate required columns
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Parse time if present
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.sort_values('time', inplace=True)
        
        # Add volume if missing
        if 'volume' not in df.columns:
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
            else:
                df['volume'] = 1000  # Placeholder
        
        df.reset_index(drop=True, inplace=True)
        logger.info(f"Loaded {len(df)} rows")
        
        return df
    
    def generate_features(self, df: pd.DataFrame) -> np.ndarray:
        """Generate technical features for model input."""
        logger.info("Generating features...")
        
        try:
            from alpha_factory.features_engineering import FeatureEngineerOptimized
            self._feature_engineer = FeatureEngineerOptimized()
            featured_df = self._feature_engineer.generate_features(df.copy())
        except ImportError:
            logger.warning("FeatureEngineerOptimized not available, using basic features")
            featured_df = self._generate_basic_features(df)
        
        # Select numeric columns only (exclude time, etc.)
        feature_cols = [c for c in featured_df.columns 
                       if c not in ['time', 'date', 'datetime'] 
                       and featured_df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        
        features = featured_df[feature_cols].values.astype(np.float32)
        
        # Handle NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        logger.info(f"Generated {features.shape[1]} features")
        return features
    
    def _generate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback basic feature generation."""
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_mid'] + 1e-10)
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Price position
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        return df.fillna(0)
    
    def generate_direction_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate direction labels based on future price movement.
        
        Labels:
            0 = Bear (price drops > threshold)
            1 = Sideways (price within threshold)
            2 = Bull (price rises > threshold)
        
        If auto_threshold is enabled, the optimal threshold is computed
        dynamically from the data to achieve balanced class distribution.
        """
        horizon = self.config.direction_horizon
        close = df['close'].values
        n = len(close)
        
        # Compute optimal threshold if auto_threshold is enabled
        if self.config.auto_threshold or self.config.direction_threshold < 0:
            # Profile-specific search ranges based on typical price movements
            profile_search_ranges = {
                'SCALP': (0.00005, 0.0005),   # M5 data: smaller moves
                'INTRADAY': (0.0001, 0.005),  # M15/H1 data: medium moves
                'SWING': (0.001, 0.02),       # H4/D1 data: larger moves
            }
            search_range = profile_search_ranges.get(self.config.profile.upper(), (0.0001, 0.01))
            
            threshold = compute_optimal_threshold(
                close_prices=close,
                horizon=horizon,
                target_imbalance_ratio=1.5,
                search_range=search_range,
                num_steps=30
            )
            # Update config with computed threshold for reference
            self.config.direction_threshold = threshold
        else:
            threshold = self.config.direction_threshold
        
        logger.info(f"Using direction threshold: {threshold:.6f}")
        
        labels = np.ones(n, dtype=np.int64)  # Default: Sideways
        
        for i in range(n - horizon):
            future_return = (close[i + horizon] - close[i]) / close[i]
            
            if future_return > threshold:
                labels[i] = 2  # Bull
            elif future_return < -threshold:
                labels[i] = 0  # Bear
            # else: remains 1 (Sideways)
        
        # Last `horizon` bars get Sideways label (no future data)
        labels[-horizon:] = 1
        
        # Log distribution
        bear_count = np.sum(labels == 0)
        side_count = np.sum(labels == 1)
        bull_count = np.sum(labels == 2)
        total = len(labels)
        logger.info(
            f"Direction labels: Bear={bear_count} ({100*bear_count/total:.1f}%), "
            f"Sideways={side_count} ({100*side_count/total:.1f}%), "
            f"Bull={bull_count} ({100*bull_count/total:.1f}%)"
        )
        return labels
    
    def generate_volatility_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate realized volatility labels.
        
        Volatility = std(returns) over the next `horizon` bars.
        """
        horizon = self.config.volatility_horizon
        returns = df['close'].pct_change().values
        n = len(returns)
        
        volatility = np.zeros(n, dtype=np.float32)
        
        for i in range(n - horizon):
            future_returns = returns[i+1:i+1+horizon]
            volatility[i] = np.std(future_returns) if len(future_returns) > 0 else 0.0
        
        # Fill last bars with rolling historical volatility
        hist_vol = np.std(returns[-horizon:]) if horizon > 0 else 0.01
        volatility[-horizon:] = hist_vol
        
        # Clip extreme values
        volatility = np.clip(volatility, 1e-6, 0.1)
        
        logger.info(f"Volatility labels: mean={volatility.mean():.6f}, std={volatility.std():.6f}")
        return volatility
    
    def generate_price_move_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate price movement labels for quantile regression.
        
        This is the actual price change over the horizon, used to train
        the quantile head to predict the distribution of future moves.
        """
        horizon = self.config.direction_horizon
        close = df['close'].values
        n = len(close)
        
        price_moves = np.zeros(n, dtype=np.float32)
        
        for i in range(n - horizon):
            price_moves[i] = (close[i + horizon] - close[i]) / close[i]
        
        # Last bars: use 0 (no future data)
        price_moves[-horizon:] = 0.0
        
        logger.info(f"Price move labels: mean={price_moves.mean():.6f}, std={price_moves.std():.6f}")
        return price_moves
    
    def generate_outcome_labels(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Generate triple-barrier outcome labels for p_long/p_short training.
        
        For each bar, determines if a long or short entry would hit TP before SL
        within the max holding period.
        
        Returns:
            (n, 2) array where [:, 0] = y_long, [:, 1] = y_short
        """
        if not self.config.use_triple_barrier:
            return None
        
        logger.info("Generating triple-barrier outcome labels...")
        
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # Calculate ATR for dynamic barriers
        atr = self._calculate_atr(df, period=14)
        
        n = len(close)
        outcomes = np.zeros((n, 2), dtype=np.float32)
        
        tp_mult = self.config.tp_multiplier
        sl_mult = self.config.sl_multiplier
        max_bars = self.config.max_holding_bars
        
        for i in range(n - max_bars):
            entry_price = close[i]
            current_atr = atr[i] if atr[i] > 0 else 0.001
            
            tp_distance = current_atr * tp_mult
            sl_distance = current_atr * sl_mult
            
            # Long position
            long_tp = entry_price + tp_distance
            long_sl = entry_price - sl_distance
            
            # Short position
            short_tp = entry_price - tp_distance
            short_sl = entry_price + sl_distance
            
            # Check future bars
            long_hit_tp = False
            long_hit_sl = False
            short_hit_tp = False
            short_hit_sl = False
            
            for j in range(1, min(max_bars + 1, n - i)):
                future_high = high[i + j]
                future_low = low[i + j]
                
                # Long: check TP (high >= tp) and SL (low <= sl)
                if not long_hit_tp and not long_hit_sl:
                    if future_high >= long_tp:
                        long_hit_tp = True
                    if future_low <= long_sl:
                        long_hit_sl = True
                
                # Short: check TP (low <= tp) and SL (high >= sl)
                if not short_hit_tp and not short_hit_sl:
                    if future_low <= short_tp:
                        short_hit_tp = True
                    if future_high >= short_sl:
                        short_hit_sl = True
                
                if (long_hit_tp or long_hit_sl) and (short_hit_tp or short_hit_sl):
                    break
            
            # Label: 1 if TP hit before SL, 0 otherwise
            outcomes[i, 0] = 1.0 if (long_hit_tp and not long_hit_sl) else 0.0
            outcomes[i, 1] = 1.0 if (short_hit_tp and not short_hit_sl) else 0.0
        
        logger.info(f"Outcome labels: long_wins={outcomes[:, 0].sum():.0f}, short_wins={outcomes[:, 1].sum():.0f}")
        return outcomes
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate Average True Range."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        n = len(close)
        tr = np.zeros(n)
        
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        # Simple moving average of TR
        atr = np.zeros(n)
        for i in range(period, n):
            atr[i] = np.mean(tr[i-period+1:i+1])
        
        # Fill initial values
        atr[:period] = atr[period] if period < n else 0.001
        
        return atr
    
    def prepare_dataset(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prepare complete dataset with features and all labels.
        
        Returns:
            Dictionary with 'features', 'direction', 'volatility', 'price_move', 
            and optionally 'outcomes'
        """
        features = self.generate_features(df)
        direction = self.generate_direction_labels(df)
        volatility = self.generate_volatility_labels(df)
        price_move = self.generate_price_move_labels(df)
        outcomes = self.generate_outcome_labels(df)
        
        # Ensure all arrays have same length
        min_len = min(len(features), len(direction), len(volatility), len(price_move))
        if outcomes is not None:
            min_len = min(min_len, len(outcomes))
        
        dataset = {
            'features': features[:min_len],
            'direction': direction[:min_len],
            'volatility': volatility[:min_len],
            'price_move': price_move[:min_len],
        }
        
        if outcomes is not None:
            dataset['outcomes'] = outcomes[:min_len]
        
        logger.info(f"Dataset prepared: {min_len} samples, {features.shape[1]} features")
        return dataset
    
    def split_data(self, dataset: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict]:
        """
        Split dataset into train/val/test sets (temporal split, no shuffling).
        """
        n = len(dataset['features'])
        
        test_size = int(n * self.config.test_split)
        val_size = int(n * self.config.val_split)
        train_size = n - val_size - test_size
        
        def slice_dict(d: Dict, start: int, end: int) -> Dict:
            return {k: v[start:end] for k, v in d.items()}
        
        train = slice_dict(dataset, 0, train_size)
        val = slice_dict(dataset, train_size, train_size + val_size)
        test = slice_dict(dataset, train_size + val_size, n)
        
        logger.info(f"Split: train={train_size}, val={val_size}, test={test_size}")
        return train, val, test


class MHTCNTrainer:
    """
    High-level trainer for MH-TCN that wraps the existing infrastructure.
    """
    
    def __init__(self, config: MHTCNTrainingConfig):
        self.config = config
        self.device = self._get_device()
        self.model = None
        self.trainer = None
        self.history = None
    
    def _get_device(self) -> torch.device:
        """Resolve device."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)
    
    def create_model(self, input_dim: int) -> torch.nn.Module:
        """Create MH-TCN model."""
        from risk_management.phase1_predictive import (
            create_tcn_for_profile, TCNConfig, TradingProfile
        )
        
        profile_map = {
            'SCALP': TradingProfile.SCALP,
            'INTRADAY': TradingProfile.INTRADAY,
            'SWING': TradingProfile.SWING,
        }
        
        profile = profile_map.get(self.config.profile.upper(), TradingProfile.INTRADAY)
        
        model = create_tcn_for_profile(
            profile=self.config.profile,
            input_features=input_dim
        )
        
        logger.info(f"Created MH-TCN model: input_dim={input_dim}, profile={self.config.profile}")
        return model.to(self.device)
    
    def create_dataloaders(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray]
    ) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders with optional weighted sampling."""
        from risk_management.phase1_predictive import RiskDataset
        from torch.utils.data import WeightedRandomSampler
        
        train_dataset = RiskDataset(
            features=train_data['features'],
            direction_labels=train_data['direction'],
            volatility_labels=train_data['volatility'],
            price_move_labels=train_data['price_move'],
            sequence_length=self.config.sequence_length,
            outcome_labels=train_data.get('outcomes')
        )
        
        val_dataset = RiskDataset(
            features=val_data['features'],
            direction_labels=val_data['direction'],
            volatility_labels=val_data['volatility'],
            price_move_labels=val_data['price_move'],
            sequence_length=self.config.sequence_length,
            outcome_labels=val_data.get('outcomes')
        )
        
        # Create weighted sampler for balanced batches
        sampler = None
        shuffle = True
        if self.config.use_weighted_sampler:
            # Get direction labels for each sample in the dataset
            # The dataset creates sequences, so we need to get the label for each sequence
            seq_len = self.config.sequence_length
            direction_labels = train_data['direction'][seq_len-1:]  # Labels aligned with sequences
            direction_labels = direction_labels[:len(train_dataset)]  # Trim to dataset size
            
            # Compute sample weights (inverse of class frequency)
            unique, counts = np.unique(direction_labels, return_counts=True)
            class_weights = {cls: len(direction_labels) / (3 * count) for cls, count in zip(unique, counts)}
            
            # Assign weight to each sample based on its class
            sample_weights = np.array([class_weights.get(int(lbl), 1.0) for lbl in direction_labels])
            sample_weights = torch.from_numpy(sample_weights).float()
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True
            )
            shuffle = False  # Sampler handles shuffling
            logger.info(f"Using WeightedRandomSampler for balanced batches")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=mhtcn_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=mhtcn_collate_fn
        )
        
        logger.info(f"Created dataloaders: train={len(train_dataset)}, val={len(val_dataset)}")
        return train_loader, val_loader
    
    def compute_class_weights(self, direction_labels: np.ndarray) -> torch.Tensor:
        """Compute class weights for imbalanced direction labels."""
        unique, counts = np.unique(direction_labels, return_counts=True)
        total = len(direction_labels)
        
        weights = np.zeros(3)
        for cls, count in zip(unique, counts):
            weights[int(cls)] = total / (3 * count + 1)
        
        # Normalize
        weights = weights / weights.sum() * 3
        
        logger.info(f"Class weights: Bear={weights[0]:.2f}, Sideways={weights[1]:.2f}, Bull={weights[2]:.2f}")
        return torch.tensor(weights, dtype=torch.float32).to(self.device)
    
    def train(
        self,
        train_data: Dict[str, np.ndarray],
        val_data: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Train the MH-TCN model.
        
        Returns:
            Training history and metrics
        """
        from risk_management.phase1_predictive import (
            MultiHeadTCNTrainer, TrainingConfig
        )
        
        # Create model
        input_dim = train_data['features'].shape[1]
        self.model = self.create_model(input_dim)
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(train_data, val_data)
        
        # Compute class weights
        class_weights = self.compute_class_weights(train_data['direction'])
        
        # Training config
        train_config = TrainingConfig(
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            num_epochs=self.config.epochs,
            patience=self.config.patience,
            grad_clip=self.config.grad_clip,
            use_uncertainty_weighting=True
        )
        
        # Create trainer
        self.trainer = MultiHeadTCNTrainer(
            model=self.model,
            config=train_config,
            device=str(self.device),
            class_weights=class_weights
        )
        
        # Train
        logger.info(f"Starting training for {self.config.epochs} epochs on {self.device}")
        self.history = self.trainer.train(train_loader, val_loader)
        
        return self.history
    
    def evaluate(self, test_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate model on test set."""
        from risk_management.phase1_predictive import RiskDataset, compute_metrics
        
        test_dataset = RiskDataset(
            features=test_data['features'],
            direction_labels=test_data['direction'],
            volatility_labels=test_data['volatility'],
            price_move_labels=test_data['price_move'],
            sequence_length=self.config.sequence_length,
            outcome_labels=test_data.get('outcomes')
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=mhtcn_collate_fn
        )
        
        _, metrics = self.trainer.validate(test_loader)
        
        logger.info("Test metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        
        return metrics
    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save trained model."""
        if path is None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(output_dir / f"multihead_tcn_{self.config.profile}_{timestamp}.pth")
        
        # Get input dimension from model
        input_dim = self.model.config.input_channels if hasattr(self.model, 'config') else 64
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': {
                'input_channels': input_dim,
                'hidden_channels': self.config.hidden_channels,
                'num_layers': self.config.num_layers,
                'profile': self.config.profile,
                'sequence_length': self.config.sequence_length,
            },
            'training_config': {
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
            },
            'history': self.history,
            'best_val_loss': self.trainer.best_val_loss if self.trainer else None,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
        
        # Also save as the standard name for the profile
        standard_path = Path(self.config.output_dir) / f"multihead_tcn_{self.config.profile}.pth"
        torch.save(checkpoint, standard_path)
        logger.info(f"Model also saved to {standard_path}")
        
        return path


def train_mhtcn(
    data_path: str,
    profile: str = "INTRADAY",
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    output_dir: str = "models/weights",
    use_triple_barrier: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Main training function for MH-TCN.
    
    Args:
        data_path: Path to OHLCV CSV file
        profile: Trading profile (SCALP, INTRADAY, SWING)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Directory to save model
        use_triple_barrier: Whether to train outcome heads
        **kwargs: Additional config overrides
    
    Returns:
        Dictionary with training results
    """
    # Create config
    config = MHTCNTrainingConfig(
        data_path=data_path,
        profile=profile.upper(),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir,
        use_triple_barrier=use_triple_barrier,
        **{k: v for k, v in kwargs.items() if hasattr(MHTCNTrainingConfig, k)}
    )
    
    # Prepare data
    preparer = MHTCNDataPreparer(config)
    df = preparer.load_data(data_path)
    dataset = preparer.prepare_dataset(df)
    train_data, val_data, test_data = preparer.split_data(dataset)
    
    # Train
    trainer = MHTCNTrainer(config)
    history = trainer.train(train_data, val_data)
    
    # Evaluate
    test_metrics = trainer.evaluate(test_data)
    
    # Save
    model_path = trainer.save_model()
    
    return {
        'model_path': model_path,
        'history': history,
        'test_metrics': test_metrics,
        'config': config
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train Multi-Head TCN model for forex prediction"
    )
    
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to OHLCV CSV file"
    )
    parser.add_argument(
        "--profile", "-p",
        default="INTRADAY",
        choices=["SCALP", "INTRADAY", "SWING"],
        help="Trading profile (default: INTRADAY)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="models/weights",
        help="Output directory for model (default: models/weights)"
    )
    parser.add_argument(
        "--sequence-length", "-s",
        type=int,
        default=60,
        help="Sequence length for TCN input (default: 60)"
    )
    parser.add_argument(
        "--no-triple-barrier",
        action="store_true",
        help="Disable triple-barrier outcome labels"
    )
    parser.add_argument(
        "--direction-horizon",
        type=int,
        default=12,
        help="Bars ahead for direction label (default: 12)"
    )
    parser.add_argument(
        "--direction-threshold",
        type=float,
        default=0.001,
        help="Min price move for bull/bear label (default: 0.001 = 0.1%%)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Run training
    try:
        results = train_mhtcn(
            data_path=args.data,
            profile=args.profile,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            sequence_length=args.sequence_length,
            use_triple_barrier=not args.no_triple_barrier,
            direction_horizon=args.direction_horizon,
            direction_threshold=args.direction_threshold,
        )
        
        print("\n" + "=" * 60)
        print("  MH-TCN TRAINING COMPLETE")
        print("=" * 60)
        print(f"\n  Model saved to: {results['model_path']}")
        print(f"\n  Test Metrics:")
        for k, v in results['test_metrics'].items():
            print(f"    {k}: {v:.4f}")
        print("\n" + "=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
