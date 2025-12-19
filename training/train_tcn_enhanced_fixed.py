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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Sklearn for feature importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

class EnhancedDataLoaderV3:
    def __init__(
        self,
        sequence_length: int = 30,
        trend_threshold: float = 0.05,
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
    
    def load_csv(self, path: str) -> pd.DataFrame:
        """Load and prepare data from CSV."""
        logger.info(f"📂 Loading data from {path}")
        df = pd.read_csv(path)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Add technical features if not present
        df = self._add_technical_features(df)
        
        # Store all available feature columns (excluding OHLCV and labels)
        exclude = ['open', 'high', 'low', 'close', 'volume', 'time', 'date', 
                   'datetime', 'timestamp', 'label', 'target']
        self.all_feature_columns = [c for c in df.columns if c not in exclude]
        self.feature_columns = self.all_feature_columns.copy()
        
        logger.info(f"   Loaded {len(df)} rows, {len(self.all_feature_columns)} features")
        if self.all_feature_columns:
            logger.info(f"   Sample features: {self.all_feature_columns[:5]}")
        else:
            logger.warning("   No technical features found!")
        return df
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators if not present."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # RSI
        if 'rsi_14' not in df.columns:
            df['rsi_14'] = self._calc_rsi(close, 14)
        
        # ATR
        if 'atr_14' not in df.columns:
            df['atr_14'] = self._calc_atr(high, low, close, 14)
        
        # EMAs
        for period in [9, 20, 50, 200]:
            col = f'ema_{period}'
            if col not in df.columns:
                df[col] = pd.Series(close).ewm(span=period, adjust=False).mean()
        
        # MACD
        if 'macd' not in df.columns:
            ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
            ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def _calc_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).rolling(window=period).mean().values
        avg_loss = pd.Series(losses).rolling(window=period).mean().values
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([[50], rsi])  # Pad first value
    
    def _calc_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate ATR."""
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - prev_close),
                np.abs(low - prev_close)
            )
        )
        
        return pd.Series(tr).rolling(window=period).mean().fillna(0).values
    
    def set_feature_columns(self, features: List[str]):
        """Set specific features to use."""
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
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data and scale features.
        
        Returns:
            train_scaled, val_scaled, test_scaled
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Store close prices for label generation
        self.train_close = df['close'].iloc[:train_end].values
        self.val_close = df['close'].iloc[train_end:val_end].values
        self.test_close = df['close'].iloc[val_end:].values
        
        # Extract features
        train_df = df.iloc[:train_end][self.feature_columns]
        val_df = df.iloc[train_end:val_end][self.feature_columns]
        test_df = df.iloc[val_end:][self.feature_columns]
        
        # Scale
        self.scaler = RobustScaler()
        train_scaled = self.scaler.fit_transform(train_df.values)
        val_scaled = self.scaler.transform(val_df.values)
        test_scaled = self.scaler.transform(test_df.values)
        
        logger.info(f"   Train: {len(train_scaled)}, Val: {len(val_scaled)}, Test: {len(test_scaled)}")
        
        return train_scaled, val_scaled, test_scaled
    
    def create_sequences(
        self,
        data: np.ndarray,
        close_prices: np.ndarray,
        seq_len: int,
        horizon: int = 1,
        timeframe_hint: str = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences with labels based on future returns.
        """
        X, y = []
        
        # Adaptive trend threshold based on timeframe
        adaptive_threshold = self._get_adaptive_threshold(timeframe_hint, seq_len, horizon)
        logger.info(f"   Using adaptive trend threshold: {adaptive_threshold:.3f} ({timeframe_hint or 'default'})")
        
        max_start = len(data) - seq_len - horizon
        logger.info(f"   Sequence creation: data_len={len(data)}, seq_len={seq_len}, horizon={horizon}")
        logger.info(f"   Max start index: {max_start}")
        
        if max_start <= 0:
            logger.warning(f"⚠️ Insufficient data: need {seq_len + horizon} bars, got {len(data)}")
            return np.array([]), np.array([])
        
        for i in range(max_start + 1):
            X.append(data[i:i+seq_len])
            
            # Calculate future return
            entry_price = close_prices[i + seq_len - 1]
            exit_price = close_prices[i + seq_len - 1 + horizon]
            pct_return = (exit_price - entry_price) / entry_price
            
            # Classify with adaptive threshold
            if pct_return > adaptive_threshold:
                y.append(2)  # Bull
            elif pct_return < -adaptive_threshold:
                y.append(0)  # Bear
            else:
                y.append(1)  # Sideways
        
        return np.array(X), np.array(y)
    
    def _get_adaptive_threshold(self, timeframe_hint: str, seq_len: int, horizon: int) -> float:
        """Get adaptive trend threshold based on timeframe characteristics."""
        # Base thresholds for different timeframes (more lenient for longer timeframes)
        timeframe_thresholds = {
            'M1': 0.002,   # 0.2% for 1-minute
            'M5': 0.005,   # 0.5% for 5-minute  
            'M15': 0.008,  # 0.8% for 15-minute
            'M30': 0.012,  # 1.2% for 30-minute
            'H1': 0.015,   # 1.5% for 1-hour
            'H4': 0.025,   # 2.5% for 4-hour
            'D1': 0.04,    # 4.0% for daily
        }
        
        # Use provided hint or default to base threshold
        if timeframe_hint and timeframe_hint.upper() in timeframe_thresholds:
            return timeframe_thresholds[timeframe_hint.upper()]
        
        # Fallback: calculate based on sequence length
        # Longer sequences should have higher thresholds
        base_threshold = self.trend_threshold
        if seq_len <= 30:
            return base_threshold
        elif seq_len <= 60:
            return base_threshold * 1.5
        else:
            return base_threshold * 2.0

# Test the fixed create_sequences method
if __name__ == "__main__":
    print("Testing fixed create_sequences method...")
    
    loader = EnhancedDataLoaderV3(sequence_length=60, trend_threshold=0.05)
    df = loader.load_csv('data/raw/EURUSD_M5_latest.csv')
    
    # Test with some feature columns
    loader.feature_columns = ['rsi_14', 'atr_14', 'ema_20', 'ema_50', 'macd']
    
    train_scaled, val_scaled, test_scaled = loader.split_and_scale(df, train_ratio=0.8, val_ratio=0.1)
    
    print(f"Train data shape: {train_scaled.shape}")
    print(f"Close prices length: {len(loader.train_close)}")
    
    # Test create_sequences
    X, y = loader.create_sequences(train_scaled, loader.train_close, 60, timeframe_hint='M5')
    print(f"Created sequences: X shape={X.shape}, y shape={y.shape}")
    print(f"Number of sequences: {len(X)}")
    print(f"Success!")
