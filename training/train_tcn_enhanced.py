# training/train_tcn_enhanced.py
"""
Enhanced TCN Training with Integrated Feature Importance Discovery.

This script provides end-to-end training that:
1. Auto-discovers top predictive features using Random Forest
2. Trains TCN model using selected features
3. Saves feature configuration WITH the model checkpoint
4. Supports profile-based training (SCALP, INTRADAY, SWING)

Key improvements over separate feature_importance.py + train scripts:
- Unified pipeline: No manual copying of TOP_FEATURES
- Checkpoint includes features: Evaluation scripts load from checkpoint
- Profile-aware feature selection: Different features for different strategies
- Reproducible: Feature discovery uses same data splits as training

Usage:
    # Auto-discover features and train
    python training/train_tcn_enhanced.py --data data/raw/eurusd_latest.csv

    # Use specific profile
    python training/train_tcn_enhanced.py --data data/raw/eurusd_latest.csv --profile SCALP

    # Skip feature discovery, use all features
    python training/train_tcn_enhanced.py --data data/raw/eurusd_latest.csv --skip-feature-selection

    # Use custom feature list
    python training/train_tcn_enhanced.py --data data/raw/eurusd_latest.csv --features "close,rsi_14,atr_14"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Literal, Union
from dataclasses import dataclass, field, asdict
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm

# Sklearn for feature importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tcn import CausalConv1d as _CausalConv1d, TCNBlock as _TCNBlock, TCNModel
from utils.feature_schema import get_feature_schema_version
from utils.training_utils import set_global_seed, copy_schema_tagged

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

class TCNBlock(_TCNBlock):
    def __init__(
        self,
        in_ch: Optional[int] = None,
        out_ch: Optional[int] = None,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
        use_weight_norm: bool = True,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
    ):
        if in_channels is None:
            in_channels = in_ch
        if out_channels is None:
            out_channels = out_ch
        super().__init__(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            dropout=float(dropout),
            use_weight_norm=bool(use_weight_norm),
        )

CausalConv1d = _CausalConv1d

class EnhancedTCN(nn.Module):
    PROFILES = TCNModel.PROFILES

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_classes: int = 3,
        dropout: float = 0.2,
        kernel_size: int = 3,
        dilation_base: int = 2,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)

        self._tcn = TCNModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=int(num_layers),
            num_classes=self.num_classes,
            dropout=float(dropout),
            kernel_size=int(kernel_size),
            dilation_base=int(dilation_base),
        )

        self.tcn = self._tcn.tcn
        self.classifier = self._tcn.classifier
        self.layer_norm = self._tcn.layer_norm
        self.feature_dim = self._tcn.feature_dim
        self.receptive_field = int(getattr(self.tcn, 'receptive_field', 0) or 0)

    @classmethod
    def from_profile(
        cls,
        profile: Literal['SCALP', 'INTRADAY', 'SWING'],
        input_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> 'EnhancedTCN':
        profile = profile.upper()
        if profile not in TCNModel.PROFILES:
            raise ValueError(f"Unknown profile: {profile}. Use {list(TCNModel.PROFILES.keys())}")

        cfg = TCNModel.PROFILES[profile]
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=cfg['num_layers'],
            num_classes=num_classes,
            dropout=dropout,
            kernel_size=cfg['kernel_size'],
            dilation_base=cfg.get('dilation_base', 2),
        )

    def forward(self, x: torch.Tensor, mode: str = 'classify') -> torch.Tensor:
        return self._tcn(x, mode=mode)

    def get_feature_dim(self) -> int:
        return int(self.feature_dim)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FeatureConfig:
    """Configuration for feature selection."""
    n_top_features: int = 25
    min_importance_threshold: float = 0.01
    rf_n_estimators: int = 100
    rf_max_depth: int = 12
    rf_random_state: int = 42

    # Profile-specific feature preferences
    # These features are prioritized for each profile
    profile_priorities: Dict[str, List[str]] = field(default_factory=lambda: {
        'SCALP': [
            'rsi_14', 'stoch_k', 'stoch_d', 'bb_position', 'atr_14',
            'volume_ratio', 'price_momentum_5', 'ema_cross_signal',
        ],
        'INTRADAY': [
            'adx_14', 'rsi_14', 'macd_hist', 'ema_20', 'ema_50',
            'atr_14', 'volume_sma_ratio', 'trend_strength',
        ],
        'SWING': [
            'ema_50', 'ema_200', 'adx_14', 'weekly_trend', 'roc_20',
            'support_distance', 'resistance_distance', 'donchian_position',
        ],
    })


@dataclass
class TrainingConfig:
    """Configuration for TCN training."""
    # Data
    sequence_length: int = 30  # Shorter sequences work better
    trend_threshold: float = 0.001  # 0.1% base threshold
    train_split: float = 0.7
    val_split: float = 0.15

    # Model - SIMPLIFIED
    hidden_dim: int = 64  # Smaller for less overfitting
    num_layers: int = 4  # Fewer layers
    kernel_size: int = 3
    dropout: float = 0.2  # Less dropout
    num_classes: int = 3  # Bear=0, Sideways=1, Bull=2

    # Training
    epochs: int = 50
    batch_size: int = 64  # Smaller batches
    learning_rate: float = 1e-3  # Standard LR
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10

    # Scheduler
    use_onecycle: bool = True  # OneCycle works well
    use_cosine: bool = False

    # Misc
    device: str = 'auto'
    seed: int = 42


@dataclass
class CheckpointData:
    """Data structure for model checkpoints."""
    model_state: dict
    feature_columns: List[str]
    feature_importance: Dict[str, float]
    config: Dict
    training_history: Dict
    created_at: str
    profile: Optional[str]
    metrics: Dict


# =============================================================================
# Feature Importance Discovery
# =============================================================================

class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance using Random Forest.

    Integrated into training pipeline to ensure consistency between
    feature selection and model training.
    """

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.importance_scores: Dict[str, float] = {}
        self.selected_features: List[str] = []

    def analyze(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        profile: Optional[str] = None,
    ) -> List[str]:
        """
        Analyze feature importance and return top features.

        Args:
            X: Feature matrix (samples, timesteps, features) - 3D for sequences
               or (samples, features) - 2D for flat
            y: Labels
            feature_names: List of feature names
            profile: Optional profile for prioritizing certain features

        Returns:
            List of top feature names
        """
        logger.info("🔍 Running Feature Importance Analysis...")

        # Flatten if 3D (take last timestep of each sequence)
        if X.ndim == 3:
            X_flat = X[:, -1, :]  # Shape: (samples, features)
            logger.info(f"   Flattened 3D data: {X.shape} -> {X_flat.shape}")
        else:
            X_flat = X

        # Train Random Forest
        logger.info(f"   Training Random Forest on {len(y)} samples...")
        rf = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            n_jobs=-1,
            random_state=self.config.rf_random_state,
            class_weight='balanced'
        )
        rf.fit(X_flat, y)

        # Extract importance scores
        importances = rf.feature_importances_

        # Build importance dictionary
        # Ensure we don't exceed the number of importances
        n_features = min(len(feature_names), len(importances))
        self.importance_scores = {
            name: float(importances[i])
            for i, name in enumerate(feature_names[:n_features])
        }

        # Sort by importance
        sorted_features = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Apply profile prioritization if specified
        if profile and profile.upper() in self.config.profile_priorities:
            priority_features = self.config.profile_priorities[profile.upper()]
            sorted_features = self._apply_profile_boost(sorted_features, priority_features)

        # Select top N features above threshold
        self.selected_features = []
        for name, score in sorted_features:
            if len(self.selected_features) >= self.config.n_top_features:
                break
            if score >= self.config.min_importance_threshold:
                self.selected_features.append(name)

        # SAFETY FALLBACK: If no features selected, use core OHLCV features
        if len(self.selected_features) == 0:
            logger.warning("⚠️ No features met importance threshold. Using OHLCV fallback.")
            # Try OHLCV first
            core_features = ['open', 'high', 'low', 'close', 'volume']
            available_core = [f for f in core_features if f in feature_names]

            if available_core:
                self.selected_features = available_core
                logger.info(f"   Using core features: {self.selected_features}")
            else:
                # Use technical indicators as fallback
                tech_features = ['rsi_14', 'atr_14', 'ema_20', 'ema_50', 'macd']
                available_tech = [f for f in tech_features if f in feature_names]

                if available_tech:
                    self.selected_features = available_tech[:5]
                    logger.info(f"   Using technical features: {self.selected_features}")
                else:
                    # Last resort: use first 5 available
                    self.selected_features = feature_names[:5]
                    logger.warning(f"   Using first 5 features: {self.selected_features}")

        # Ensure we have at least 3 features minimum
        while len(self.selected_features) < 3 and len(self.selected_features) < len(feature_names):
            remaining = [f for f in feature_names if f not in self.selected_features]
            if remaining:
                self.selected_features.append(remaining[0])
            else:
                break

        # Log results
        logger.info(f"\n{'='*50}")
        logger.info(f"🏆 TOP {len(self.selected_features)} PREDICTIVE FEATURES")
        logger.info(f"{'='*50}")
        for i, name in enumerate(self.selected_features[:15], 1):
            score = self.importance_scores[name]
            logger.info(f"{i:2d}. {name:30s} : {score:.4f}")
        if len(self.selected_features) > 15:
            logger.info(f"   ... and {len(self.selected_features) - 15} more")

        return self.selected_features

    def _apply_profile_boost(
        self,
        sorted_features: List[Tuple[str, float]],
        priority_features: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Boost priority features for specific trading profiles.

        This ensures profile-relevant features are more likely to be selected
        even if they're slightly lower in raw importance.
        """
        BOOST_FACTOR = 1.2

        boosted = []
        for name, score in sorted_features:
            if name in priority_features:
                boosted.append((name, score * BOOST_FACTOR))
            else:
                boosted.append((name, score))

        # Re-sort after boosting
        return sorted(boosted, key=lambda x: x[1], reverse=True)

    def get_importance_dict(self) -> Dict[str, float]:
        """Return full importance scores dictionary."""
        return self.importance_scores.copy()


# =============================================================================
# Enhanced Data Loader (Unified with feature selection)
# =============================================================================

class SequenceWindowDataset(Dataset):
    def __init__(self, data_2d: np.ndarray, labels: np.ndarray, seq_len: int):
        if data_2d.ndim != 2:
            raise ValueError(f"data_2d must be 2D, got shape={getattr(data_2d, 'shape', None)}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got shape={getattr(labels, 'shape', None)}")
        if len(labels) + seq_len > len(data_2d) + 1:
            raise ValueError("labels length is inconsistent with data_2d and seq_len")

        self.data_2d = np.asarray(data_2d, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        start = int(idx)
        end = start + self.seq_len
        x = torch.from_numpy(self.data_2d[start:end])
        y = torch.tensor(self.labels[start], dtype=torch.long)
        return x, y


class EnhancedDataLoaderV3:
    """
    Data loader with integrated feature selection support.

    Improvements over V2:
    - Accepts feature subset for training
    - Returns feature names with data
    - Consistent train/val/test splits
    """

    def __init__(
        self,
        sequence_length: int = 30,
        trend_threshold: float = 0.001,  # Legacy param, now using percentile-based thresholds
        scaler_type: str = 'robust',
    ):
        self.sequence_length = sequence_length
        self.trend_threshold = trend_threshold
        self.scaler_type = scaler_type

        self.scaler = None
        self.feature_columns: List[str] = []
        self.all_feature_columns: List[str] = []

        # Store split data
        self.train_close: np.ndarray = None
        self.val_close: np.ndarray = None
        self.test_close: np.ndarray = None

    def load_csv(self, path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load and prepare data from CSV."""
        logger.info(f"📂 Loading data from {path}")
        df = pd.read_csv(path)

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()

        if max_rows is not None and max_rows > 0 and len(df) > max_rows:
            df = df.tail(max_rows).reset_index(drop=True)

        # Ensure required columns
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure a usable volume column for feature engineering
        if 'volume' not in df.columns:
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
            elif 'real_volume' in df.columns:
                df['volume'] = df['real_volume']
            else:
                df['volume'] = 0.0

        # Add enriched features (includes pa_* and all engineered columns)
        try:
            from utils.features_engineering import FeatureEngineer
            fe = FeatureEngineer()
            df = fe.generate_features(df, batch_processing=False)
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
        except Exception as e:
            logger.warning(f"FeatureEngineer failed ({e}), falling back to basic technical features")
            df = self._add_technical_features(df)

        # Store all available feature columns (excluding OHLCV and labels)
        exclude = ['open', 'high', 'low', 'close', 'volume', 'time', 'date',
                   'datetime', 'timestamp', 'label', 'target']
        self.all_feature_columns = [c for c in df.columns if c not in exclude]
        self.feature_columns = self.all_feature_columns.copy()

        logger.info(f"   Loaded {len(df)} rows, {len(self.all_feature_columns)} features")
        return df

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        if 'rsi_14' not in df.columns:
            df['rsi_14'] = self._calc_rsi(close, 14)

        if 'atr_14' not in df.columns:
            df['atr_14'] = self._calc_atr(high, low, close, 14)

        for period in [9, 20, 50, 200]:
            col = f'ema_{period}'
            if col not in df.columns:
                df[col] = pd.Series(close).ewm(span=period, adjust=False).mean()

        if 'macd' not in df.columns:
            ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
            ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

        if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns or 'bb_position' not in df.columns:
            close_s = pd.Series(close)
            ma = close_s.rolling(window=20).mean()
            std = close_s.rolling(window=20).std(ddof=0)
            upper = ma + 2 * std
            lower = ma - 2 * std
            df['bb_upper'] = upper
            df['bb_lower'] = lower
            width = (upper - lower).replace(0, np.nan)
            df['bb_position'] = ((close_s - lower) / width).fillna(0.5).clip(0.0, 1.0)

        if 'stoch_k' not in df.columns or 'stoch_d' not in df.columns:
            low_s = pd.Series(low)
            high_s = pd.Series(high)
            close_s = pd.Series(close)
            ll = low_s.rolling(window=14).min()
            hh = high_s.rolling(window=14).max()
            denom = (hh - ll).replace(0, np.nan)
            stoch_k = ((close_s - ll) / denom * 100.0).fillna(50.0)
            stoch_d = stoch_k.rolling(window=3).mean().fillna(50.0)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d

        if 'adx_14' not in df.columns:
            df['adx_14'] = self._calc_adx(high, low, close, 14)

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return df

    def _calc_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        high_s = pd.Series(high)
        low_s = pd.Series(low)
        close_s = pd.Series(close)

        up_move = high_s.diff()
        down_move = -low_s.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        prev_close = close_s.shift(1)
        tr = pd.concat(
            [
                (high_s - low_s).abs(),
                (high_s - prev_close).abs(),
                (low_s - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100.0 * (pd.Series(plus_dm).rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100.0 * (pd.Series(minus_dm).rolling(window=period).mean() / (atr + 1e-10))

        dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)).fillna(0.0)
        adx = dx.rolling(window=period).mean().fillna(0.0)
        return adx.values

    def _calc_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = pd.Series(gains).rolling(window=period).mean().values
        avg_loss = pd.Series(losses).rolling(window=period).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate([[50], rsi])

    def _calc_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - prev_close),
                np.abs(low - prev_close),
            ),
        )
        return pd.Series(tr).rolling(window=period).mean().fillna(0).values

    def set_feature_columns(self, features: List[str]):
        available = set(self.all_feature_columns)
        valid = [f for f in features if f in available]
        if len(valid) < len(features):
            missing = [f for f in features if f not in available]
            logger.warning(f"Features not found (skipped): {missing}")
        self.feature_columns = valid
        logger.info(f"   Using {len(self.feature_columns)} features")

    def split_and_scale(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        self.train_close = df['close'].iloc[:train_end].values
        self.val_close = df['close'].iloc[train_end:val_end].values
        self.test_close = df['close'].iloc[val_end:].values

        train_df = df.iloc[:train_end][self.feature_columns]
        val_df = df.iloc[train_end:val_end][self.feature_columns]
        test_df = df.iloc[val_end:][self.feature_columns]

        self.scaler = RobustScaler()
        train_scaled = self.scaler.fit_transform(train_df.values).astype(np.float32, copy=False)
        val_scaled = self.scaler.transform(val_df.values).astype(np.float32, copy=False)
        test_scaled = self.scaler.transform(test_df.values).astype(np.float32, copy=False)

        logger.info(f"   Train: {len(train_scaled)}, Val: {len(val_scaled)}, Test: {len(test_scaled)}")
        return train_scaled, val_scaled, test_scaled

    def create_sequences(
        self,
        data: np.ndarray,
        close_prices: np.ndarray,
        seq_len: int,
        horizon: int = 3,
        timeframe_hint: str = None,
        max_sequences: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        limit = len(data) - seq_len - horizon + 1
        if limit <= 0:
            return np.array([]), np.array([])

        if max_sequences is not None and limit > max_sequences:
            start = limit - max_sequences
            data = data[start:]
            close_prices = close_prices[start:]
            limit = len(data) - seq_len - horizon + 1

        if limit > 250000:
            raise MemoryError("create_sequences would allocate a very large array; use SequenceWindowDataset instead")

        entry_idx = np.arange(limit, dtype=np.int64) + (seq_len - 1)
        exit_idx = entry_idx + horizon
        entry_price = close_prices[entry_idx]
        exit_price = close_prices[exit_idx]
        returns = (exit_price - entry_price) / entry_price

        p33 = np.percentile(returns, 33)
        p67 = np.percentile(returns, 67)
        y = np.where(returns <= p33, 0, np.where(returns >= p67, 2, 1)).astype(np.int64)

        X = np.empty((limit, seq_len, data.shape[1]), dtype=np.float32)
        for i in range(limit):
            X[i] = data[i:i + seq_len]

        return X, y


# =============================================================================
# Training Pipeline
# =============================================================================

class TCNTrainer:
    """
    End-to-end TCN training with feature discovery.
    """

    def __init__(
        self,
        feature_config: FeatureConfig = None,
        training_config: TrainingConfig = None,
    ):
        self.feature_config = feature_config or FeatureConfig()
        self.training_config = training_config or TrainingConfig()

        self.device = self._get_device()
        self.model: Optional[EnhancedTCN] = None
        self.feature_analyzer = FeatureImportanceAnalyzer(self.feature_config)
        self.data_loader = EnhancedDataLoaderV3(
            sequence_length=self.training_config.sequence_length,
            trend_threshold=self.training_config.trend_threshold,
        )

        # Training state
        self.selected_features: List[str] = []
        self.training_history: Dict = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.best_val_acc: float = 0.0

    def _get_device(self) -> torch.device:
        if self.training_config.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.training_config.device)

    def prepare_data(
        self,
        data_path: str,
        features: Optional[List[str]] = None,
        skip_feature_selection: bool = False,
        profile: Optional[str] = None,
        max_rows: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load data, optionally discover features, prepare loaders.

        Args:
            data_path: Path to CSV data
            features: Optional list of features (skips discovery if provided)
            skip_feature_selection: If True, use all available features
            profile: Trading profile for feature prioritization
        """
        # Load data
        df = self.data_loader.load_csv(data_path, max_rows=max_rows)

        # Feature selection
        if features:
            # Use provided features
            self.selected_features = features
            self.data_loader.set_feature_columns(features)
            logger.info(f"Using {len(features)} provided features")

        elif skip_feature_selection:
            # Use all features
            self.selected_features = self.data_loader.all_feature_columns.copy()
            logger.info(f"Using all {len(self.selected_features)} features (no selection)")
            self.data_loader.set_feature_columns(self.selected_features)

        else:
            train_scaled, _, _ = self.data_loader.split_and_scale(
                df,
                train_ratio=self.training_config.train_split,
                val_ratio=self.training_config.val_split,
            )

            seq_len = self.training_config.sequence_length
            horizon = 3
            limit = len(train_scaled) - seq_len - horizon + 1
            if limit <= 0:
                raise ValueError("Insufficient data for feature selection")

            entry_idx = np.arange(limit, dtype=np.int64) + (seq_len - 1)
            exit_idx = entry_idx + horizon
            entry_price = self.data_loader.train_close[entry_idx]
            exit_price = self.data_loader.train_close[exit_idx]
            returns = (exit_price - entry_price) / entry_price

            p33 = np.percentile(returns, 33)
            p67 = np.percentile(returns, 67)
            y_fs = np.where(returns <= p33, 0, np.where(returns >= p67, 2, 1)).astype(np.int64)
            X_fs = train_scaled[entry_idx]

            max_fs_samples = 50000
            if len(y_fs) > max_fs_samples:
                start = len(y_fs) - max_fs_samples
                X_fs = X_fs[start:]
                y_fs = y_fs[start:]

            self.selected_features = self.feature_analyzer.analyze(
                X_fs,
                y_fs,
                self.data_loader.feature_columns,
                profile=profile,
            )
            self.data_loader.set_feature_columns(self.selected_features)

        train_scaled, val_scaled, test_scaled = self.data_loader.split_and_scale(
            df,
            train_ratio=self.training_config.train_split,
            val_ratio=self.training_config.val_split,
        )

        seq_len = self.training_config.sequence_length
        horizon = 3

        train_returns_limit = len(self.data_loader.train_close) - seq_len - horizon + 1
        if train_returns_limit <= 0:
            raise ValueError("Insufficient data for sequence labeling")
        train_entry_idx = np.arange(train_returns_limit, dtype=np.int64) + (seq_len - 1)
        train_exit_idx = train_entry_idx + horizon
        train_entry = self.data_loader.train_close[train_entry_idx]
        train_exit = self.data_loader.train_close[train_exit_idx]
        train_returns = (train_exit - train_entry) / train_entry
        p33 = np.percentile(train_returns, 33)
        p67 = np.percentile(train_returns, 67)

        def _make_labels(close_prices: np.ndarray) -> np.ndarray:
            limit_local = len(close_prices) - seq_len - horizon + 1
            if limit_local <= 0:
                return np.array([], dtype=np.int64)
            entry_idx = np.arange(limit_local, dtype=np.int64) + (seq_len - 1)
            exit_idx = entry_idx + horizon
            entry_price = close_prices[entry_idx]
            exit_price = close_prices[exit_idx]
            returns = (exit_price - entry_price) / entry_price
            return np.where(returns <= p33, 0, np.where(returns >= p67, 2, 1)).astype(np.int64)

        y_train = _make_labels(self.data_loader.train_close)
        y_val = _make_labels(self.data_loader.val_close)
        y_test = _make_labels(self.data_loader.test_close)

        def _label_summary(y: np.ndarray) -> Dict[str, str]:
            if len(y) == 0:
                return {'Bear': '0', 'Sideways': '0', 'Bull': '0'}
            counts = np.bincount(y, minlength=3)
            total = int(len(y))
            return {
                'Bear': f"{int(counts[0])} ({counts[0] / total:.1%})",
                'Sideways': f"{int(counts[1])} ({counts[1] / total:.1%})",
                'Bull': f"{int(counts[2])} ({counts[2] / total:.1%})",
            }

        logger.info(f"   3-class labels (horizon={horizon}): {_label_summary(y_train)}")

        train_ds = SequenceWindowDataset(train_scaled, y_train, seq_len)
        val_ds = SequenceWindowDataset(val_scaled, y_val, seq_len)
        test_ds = SequenceWindowDataset(test_scaled, y_test, seq_len)

        train_loader = DataLoader(train_ds, batch_size=self.training_config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.training_config.batch_size)
        test_loader = DataLoader(test_ds, batch_size=self.training_config.batch_size)

        return train_loader, val_loader, test_loader

    def build_model(self, profile: Optional[str] = None) -> EnhancedTCN:
        """Build TCN model with appropriate architecture."""
        input_dim = len(self.selected_features)

        if profile:
            self.model = EnhancedTCN.from_profile(
                profile,
                input_dim=input_dim,
                hidden_dim=self.training_config.hidden_dim,
                num_classes=self.training_config.num_classes,
                dropout=self.training_config.dropout,
            )
            logger.info(f"🏗️ Built TCN for {profile} profile (RF={self.model.receptive_field})")
        else:
            self.model = EnhancedTCN(
                input_dim=input_dim,
                hidden_dim=self.training_config.hidden_dim,
                num_layers=self.training_config.num_layers,
                num_classes=self.training_config.num_classes,
                dropout=self.training_config.dropout,
                kernel_size=self.training_config.kernel_size,
            )
            logger.info(f"🏗️ Built TCN (RF={self.model.receptive_field})")

        self.model = self.model.to(self.device)
        return self.model

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict:
        """
        Train the model with early stopping.

        Returns:
            Training metrics dictionary
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        # Class weights for imbalanced data
        ds_labels = getattr(getattr(train_loader, 'dataset', None), 'labels', None)
        if ds_labels is not None:
            class_counts = np.bincount(np.asarray(ds_labels, dtype=np.int64), minlength=3)
        else:
            all_labels = []
            for _, y in train_loader:
                all_labels.extend(y.numpy())
            class_counts = np.bincount(all_labels, minlength=3)
        class_weights = 1.0 / (class_counts + 1e-10)
        class_weights = class_weights / class_weights.sum() * 3
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

        # Scheduler
        if self.training_config.use_onecycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.training_config.learning_rate * 10,
                epochs=self.training_config.epochs,
                steps_per_epoch=len(train_loader),
            )
        elif self.training_config.use_cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.training_config.epochs,
            )
        else:
            scheduler = None

        # Training loop
        best_model_state = None
        patience_counter = 0

        logger.info(f"\n🚀 Starting training on {self.device}")
        logger.info(f"   Epochs: {self.training_config.epochs}, Batch size: {self.training_config.batch_size}")

        for epoch in range(self.training_config.epochs):
            # Train
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.training_config.epochs}")
            for X_batch, y_batch in pbar:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                if self.training_config.use_onecycle and scheduler:
                    scheduler.step()

                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_loss /= len(train_loader)

            # Validate
            val_loss, val_acc = self._validate(val_loader, criterion)

            # Step scheduler (if not OneCycle)
            if scheduler and not self.training_config.use_onecycle:
                scheduler.step()

            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)

            logger.info(
                f"   Epoch {epoch+1}: Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2%}"
            )

            # Early stopping check
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.training_config.early_stopping_patience:
                    logger.info(f"   Early stopping at epoch {epoch+1}")
                    break

        # Restore best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return {
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'best_val_acc': self.best_val_acc,
            'epochs_trained': epoch + 1,
        }

    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate model and return loss and accuracy."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()

        return val_loss / len(val_loader), correct / total

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate on test set."""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)

                outputs = self.model(X_batch)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = (all_preds == all_labels).mean()

        # Per-class accuracy
        class_names = ['Bear', 'Sideways', 'Bull']
        per_class_acc = {}
        for i, name in enumerate(class_names):
            mask = all_labels == i
            if mask.sum() > 0:
                per_class_acc[name] = (all_preds[mask] == i).mean()
            else:
                per_class_acc[name] = 0.0

        logger.info(f"\n📊 Test Results:")
        logger.info(f"   Overall Accuracy: {accuracy:.2%}")
        for name, acc in per_class_acc.items():
            logger.info(f"   {name} Accuracy: {acc:.2%}")

        return {
            'test_accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': np.array(all_probs),
        }

    def save_checkpoint(
        self,
        path: str,
        profile: Optional[str] = None,
        metrics: Dict = None,
    ):
        """
        Save model with feature configuration.

        The checkpoint includes:
        - model_state: Model weights
        - feature_columns: Selected features (for evaluation)
        - feature_importance: Full importance scores
        - config: Training configuration
        - training_history: Loss/accuracy curves
        - metrics: Final evaluation metrics
        """
        checkpoint = CheckpointData(
            model_state=self.model.state_dict(),
            feature_columns=self.selected_features,
            feature_importance=self.feature_analyzer.get_importance_dict(),
            config={
                'training': asdict(self.training_config),
                'feature': asdict(self.feature_config),
                'model': {
                    'input_dim': self.model.input_dim,
                    'hidden_dim': self.model.hidden_dim,
                    'receptive_field': self.model.receptive_field,
                },
                'feature_schema_version': get_feature_schema_version(),
            },
            training_history=self.training_history,
            created_at=datetime.now().isoformat(),
            profile=profile,
            metrics=metrics or {},
        )

        # Save as dict (for torch.load compatibility)
        save_dict = asdict(checkpoint)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, path)

        logger.info(f"💾 Checkpoint saved to {path}")
        logger.info(f"   Features: {len(self.selected_features)}")
        logger.info(f"   Best Val Acc: {self.best_val_acc:.2%}")

    @staticmethod
    def load_checkpoint(path: str, device: str = 'auto') -> Tuple['EnhancedTCN', List[str], Dict]:
        """
        Load model from checkpoint.

        Returns:
            model: Loaded EnhancedTCN model
            features: List of feature columns
            checkpoint: Full checkpoint dict
        """
        checkpoint = torch.load(path, map_location='cpu')

        # Get model config
        model_config = checkpoint['config']['model']
        training_config = checkpoint['config']['training']

        # Rebuild model
        model = EnhancedTCN(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_classes=training_config['num_classes'],
            dropout=training_config['dropout'],
        )
        model.load_state_dict(checkpoint['model_state'])

        # Move to device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

        features = checkpoint['feature_columns']

        logger.info(f"✅ Loaded checkpoint from {path}")
        logger.info(f"   Features: {len(features)}")
        logger.info(f"   Profile: {checkpoint.get('profile', 'N/A')}")

        return model, features, checkpoint

# Main Entry Point
# =============================================================================

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Train Enhanced TCN with Feature Discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover features and train
  python train_tcn_enhanced.py --data data/raw/eurusd_latest.csv

  # Use SCALP profile (faster trading)
  python train_tcn_enhanced.py --data data/raw/eurusd_latest.csv --profile SCALP

  # Skip feature discovery, use all
  python train_tcn_enhanced.py --data data/raw/eurusd_latest.csv --skip-feature-selection

  # Use specific features
  python train_tcn_enhanced.py --data data/raw/eurusd_latest.csv --features "rsi_14,atr_14,macd"
        """
    )

    # Required
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data')

    # Feature selection
    parser.add_argument('--skip-feature-selection', action='store_true',
                        help='Skip feature discovery, use all features')
    parser.add_argument('--features', type=str, default=None,
                        help='Comma-separated list of features to use')
    parser.add_argument('--n-features', type=int, default=25,
                        help='Number of top features to select (default: 25)')

    # Model/Profile
    parser.add_argument('--profile', type=str, choices=['SCALP', 'INTRADAY', 'SWING'],
                        default=None, help='Trading profile (affects architecture and features)')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=5, help='Number of TCN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seq-len', type=int, default=30, help='Sequence length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max-rows', type=int, default=None, help='Limit dataset rows (use most recent)')

    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Trend threshold (default: 0.05 = 5 pips)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    # Scheduler
    parser.add_argument('--use-cosine', action='store_true', help='Use cosine scheduler')
    parser.add_argument('--no-onecycle', action='store_true', help='Disable OneCycle scheduler')

    # Output
    parser.add_argument('--save-dir', type=str, default='models/weights',
                        help='Directory to save model')
    parser.add_argument('--name', type=str, default='tcn_enhanced',
                        help='Model name for checkpoint')

    # Use provided args or parse from command line
    if args is None:
        args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    print("=" * 60)
    print("🚀 Enhanced TCN Training with Feature Discovery")
    print("=" * 60)
    print(f"   Data: {args.data}")
    print(f"   Profile: {args.profile or 'default'}")
    print(f"   Features: {'auto-discover' if not args.features else 'provided'}")
    print(f"   Hidden dim: {args.hidden_dim}")
    print(f"   Epochs: {args.epochs}")
    print("=" * 60)

    # Build configs
    feature_config = FeatureConfig(n_top_features=args.n_features)

    training_config = TrainingConfig(
        sequence_length=args.seq_len,
        trend_threshold=args.threshold,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
        use_onecycle=not args.no_onecycle,
        use_cosine=args.use_cosine,
        seed=args.seed,
    )

    set_global_seed(training_config.seed)

    # Create trainer
    trainer = TCNTrainer(
        feature_config=feature_config,
        training_config=training_config,
    )

    # Parse features if provided
    features = None
    if args.features:
        features = [f.strip() for f in args.features.split(',')]

    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(
        data_path=args.data,
        features=features,
        skip_feature_selection=args.skip_feature_selection,
        profile=args.profile,
        max_rows=args.max_rows,
    )

    # Build model
    trainer.build_model(profile=args.profile)

    # Train
    train_metrics = trainer.train(train_loader, val_loader)

    # Evaluate
    test_metrics = trainer.evaluate(test_loader)

    # Save checkpoint
    save_path = Path(args.save_dir) / f"{args.name}_best.pt"
    trainer.save_checkpoint(
        path=str(save_path),
        profile=args.profile,
        metrics={**train_metrics, **test_metrics},
    )

    copy_schema_tagged(save_path, get_feature_schema_version())

    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"   Best Val Accuracy: {trainer.best_val_acc:.2%}")
    print(f"   Test Accuracy: {test_metrics['test_accuracy']:.2%}")
    print(f"   Checkpoint: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()