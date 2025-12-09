# utils/feature_adapter.py
"""
Feature Adapter: Unified data loading and feature engineering.

UPDATED: Now integrates with checkpoint_loader for automatic feature selection.
Provides backward compatibility with older code while supporting the new
checkpoint-based feature management.

Classes:
- EnhancedDataLoaderV2: Original data loader (kept for compatibility)
- EnhancedDataLoaderV3: New loader with checkpoint integration
- FeatureEngineer: Standalone feature computation
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Engineering
# =============================================================================

class FeatureEngineer:
    """
    Computes technical indicators and derived features.
    
    Can be used standalone or as part of data loading pipeline.
    """
    
    @staticmethod
    def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical features to DataFrame."""
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()
        
        close = df['close'].values
        high = df['high'].values if 'high' in df.columns else close
        low = df['low'].values if 'low' in df.columns else close
        volume = df['volume'].values if 'volume' in df.columns else np.ones_like(close)
        
        # Price-based features
        df = FeatureEngineer._add_momentum_features(df, close)
        df = FeatureEngineer._add_volatility_features(df, close, high, low)
        df = FeatureEngineer._add_trend_features(df, close, high, low)
        df = FeatureEngineer._add_volume_features(df, volume)
        df = FeatureEngineer._add_oscillators(df, close, high, low)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    @staticmethod
    def _add_momentum_features(df: pd.DataFrame, close: np.ndarray) -> pd.DataFrame:
        """Add momentum indicators."""
        # Rate of Change
        for period in [5, 10, 20]:
            col = f'roc_{period}'
            if col not in df.columns:
                shifted = np.roll(close, period)
                shifted[:period] = close[:period]
                df[col] = (close - shifted) / (shifted + 1e-10) * 100
        
        # Price momentum
        for period in [5, 10, 20]:
            col = f'price_momentum_{period}'
            if col not in df.columns:
                shifted = np.roll(close, period)
                shifted[:period] = close[:period]
                df[col] = (close - shifted) / (shifted + 1e-10)
        
        # MACD
        if 'macd' not in df.columns:
            ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
            ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def _add_volatility_features(
        df: pd.DataFrame, 
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> pd.DataFrame:
        """Add volatility indicators."""
        # ATR
        if 'atr_14' not in df.columns:
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            tr = np.maximum(
                high - low,
                np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
            )
            df['atr_14'] = pd.Series(tr).rolling(14).mean()
        
        # Bollinger Bands
        if 'bb_position' not in df.columns:
            sma20 = pd.Series(close).rolling(20).mean()
            std20 = pd.Series(close).rolling(20).std()
            df['bb_upper'] = sma20 + 2 * std20
            df['bb_lower'] = sma20 - 2 * std20
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma20 + 1e-10)
            df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Standard deviation
        for period in [10, 20]:
            col = f'std_{period}'
            if col not in df.columns:
                df[col] = pd.Series(close).rolling(period).std()
        
        return df
    
    @staticmethod
    def _add_trend_features(
        df: pd.DataFrame,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> pd.DataFrame:
        """Add trend indicators."""
        # EMAs
        for period in [9, 20, 50, 100, 200]:
            col = f'ema_{period}'
            if col not in df.columns:
                df[col] = pd.Series(close).ewm(span=period, adjust=False).mean()
        
        # SMAs
        for period in [10, 20, 50]:
            col = f'sma_{period}'
            if col not in df.columns:
                df[col] = pd.Series(close).rolling(period).mean()
        
        # ADX
        if 'adx_14' not in df.columns:
            period = 14
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            prev_high = np.roll(high, 1)
            prev_high[0] = high[0]
            prev_low = np.roll(low, 1)
            prev_low[0] = low[0]
            
            tr = np.maximum(
                high - low,
                np.maximum(np.abs(high - prev_close), np.abs(low - prev_close))
            )
            
            plus_dm = np.where(
                (high - prev_high) > (prev_low - low),
                np.maximum(high - prev_high, 0),
                0
            )
            minus_dm = np.where(
                (prev_low - low) > (high - prev_high),
                np.maximum(prev_low - low, 0),
                0
            )
            
            atr = pd.Series(tr).rolling(period).mean()
            plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / (atr + 1e-10)
            minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / (atr + 1e-10)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
            df['adx_14'] = pd.Series(dx).rolling(period).mean()
            df['plus_di_14'] = plus_di
            df['minus_di_14'] = minus_di
        
        # Donchian Channels
        for period in [20]:
            if f'donchian_high_{period}' not in df.columns:
                df[f'donchian_high_{period}'] = pd.Series(high).rolling(period).max()
                df[f'donchian_low_{period}'] = pd.Series(low).rolling(period).min()
                df[f'donchian_position_{period}'] = (
                    (close - df[f'donchian_low_{period}']) / 
                    (df[f'donchian_high_{period}'] - df[f'donchian_low_{period}'] + 1e-10)
                )
        
        return df
    
    @staticmethod
    def _add_volume_features(df: pd.DataFrame, volume: np.ndarray) -> pd.DataFrame:
        """Add volume-based indicators."""
        if 'volume_sma_ratio' not in df.columns:
            vol_sma = pd.Series(volume).rolling(20).mean()
            df['volume_sma_ratio'] = volume / (vol_sma + 1e-10)
        
        if 'volume_roc' not in df.columns:
            shifted = np.roll(volume, 5)
            shifted[:5] = volume[:5]
            df['volume_roc'] = (volume - shifted) / (shifted + 1e-10)
        
        return df
    
    @staticmethod
    def _add_oscillators(
        df: pd.DataFrame,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> pd.DataFrame:
        """Add oscillator indicators."""
        # RSI
        if 'rsi_14' not in df.columns:
            deltas = np.diff(close)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = pd.Series(gains).rolling(14).mean()
            avg_loss = pd.Series(losses).rolling(14).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            df['rsi_14'] = np.concatenate([[50], rsi.values])
        
        # Stochastic
        if 'stoch_k' not in df.columns:
            low_14 = pd.Series(low).rolling(14).min()
            high_14 = pd.Series(high).rolling(14).max()
            df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # Williams %R
        if 'williams_r' not in df.columns:
            high_14 = pd.Series(high).rolling(14).max()
            low_14 = pd.Series(low).rolling(14).min()
            df['williams_r'] = -100 * (high_14 - close) / (high_14 - low_14 + 1e-10)
        
        # CCI
        if 'cci_20' not in df.columns:
            tp = (high + low + close) / 3
            sma_tp = pd.Series(tp).rolling(20).mean()
            mad = pd.Series(tp).rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
            df['cci_20'] = (tp - sma_tp) / (0.015 * mad + 1e-10)
        
        return df


# =============================================================================
# Enhanced Data Loader V2 (Original - Backward Compatible)
# =============================================================================

class EnhancedDataLoaderV2:
    """
    Original data loader with feature engineering.
    
    Kept for backward compatibility with existing code.
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        label_strategy: str = 'ternary',
        scaler_type: str = 'robust',
        trend_threshold: float = 0.05,
    ):
        self.sequence_length = sequence_length
        self.label_strategy = label_strategy
        self.scaler_type = scaler_type
        self.trend_threshold = trend_threshold
        
        self.scaler = None
        self.feature_columns: List[str] = []
        
        # Store close prices for label generation
        self.train_close: np.ndarray = None
        self.val_close: np.ndarray = None
        self.test_close: np.ndarray = None
    
    def load_csv(self, path: str) -> pd.DataFrame:
        """Load and prepare data from CSV."""
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()
        
        # Add technical features
        df = FeatureEngineer.add_all_features(df)
        
        # Set feature columns (exclude OHLCV and meta)
        exclude = ['open', 'high', 'low', 'close', 'volume', 'time', 'date', 
                   'datetime', 'timestamp', 'label', 'target', 'tick_volume']
        self.feature_columns = [c for c in df.columns if c not in exclude]
        
        logger.info(f"Loaded {len(df)} rows, {len(self.feature_columns)} features")
        return df
    
    def split_and_scale(
        self,
        df: pd.DataFrame,
        split_ratio: float = 0.8,
        validation_ratio: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split data and scale features."""
        n = len(df)
        train_end = int(n * split_ratio)
        val_end = int(n * (split_ratio + validation_ratio))
        
        # Store close prices
        self.train_close = df['close'].iloc[:train_end].values
        self.val_close = df['close'].iloc[train_end:val_end].values
        self.test_close = df['close'].iloc[val_end:].values
        
        # Extract features
        train_df = df.iloc[:train_end][self.feature_columns]
        val_df = df.iloc[train_end:val_end][self.feature_columns]
        test_df = df.iloc[val_end:][self.feature_columns]
        
        # Create and fit scaler
        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        train_scaled = self.scaler.fit_transform(train_df.values)
        val_scaled = self.scaler.transform(val_df.values)
        test_scaled = self.scaler.transform(test_df.values)
        
        return train_scaled, val_scaled, test_scaled
    
    def create_sequences(
        self,
        data: np.ndarray,
        close_prices: np.ndarray,
        seq_len: int,
        horizon: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences with labels."""
        X, y = [], []
        
        limit = len(data) - seq_len - horizon
        for i in range(limit):
            X.append(data[i:i+seq_len])
            
            # Calculate return and create label
            entry = close_prices[i + seq_len - 1]
            exit_price = close_prices[i + seq_len - 1 + horizon]
            pct_return = (exit_price - entry) / entry
            
            if pct_return > self.trend_threshold:
                y.append(2)  # Bull
            elif pct_return < -self.trend_threshold:
                y.append(0)  # Bear
            else:
                y.append(1)  # Sideways
        
        return np.array(X), np.array(y)


# =============================================================================
# Enhanced Data Loader V3 (New - Checkpoint Integration)
# =============================================================================

class EnhancedDataLoaderV3(EnhancedDataLoaderV2):
    """
    Data loader with checkpoint-based feature selection.
    
    Can automatically load feature columns from a model checkpoint,
    ensuring consistency between training and inference.
    
    Usage:
        # Auto-load features from checkpoint
        loader = EnhancedDataLoaderV3.from_checkpoint(
            "models/weights/tcn_enhanced_best.pt"
        )
        
        # Or use all features
        loader = EnhancedDataLoaderV3(sequence_length=30)
    """
    
    def __init__(
        self,
        sequence_length: int = 30,
        trend_threshold: float = 0.05,
        scaler_type: str = 'robust',
    ):
        super().__init__(
            sequence_length=sequence_length,
            label_strategy='ternary',
            scaler_type=scaler_type,
            trend_threshold=trend_threshold,
        )
        
        self._checkpoint_features: Optional[List[str]] = None
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        sequence_length: int = 30,
        trend_threshold: float = 0.05,
    ) -> 'EnhancedDataLoaderV3':
        """
        Create loader with features from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            sequence_length: Sequence length
            trend_threshold: Label threshold
        
        Returns:
            Configured loader with checkpoint's features
        """
        instance = cls(
            sequence_length=sequence_length,
            trend_threshold=trend_threshold,
        )
        
        # Load features from checkpoint
        try:
            from utils.checkpoint_loader import load_features
            instance._checkpoint_features = load_features(checkpoint_path)
            logger.info(f"Loaded {len(instance._checkpoint_features)} features from checkpoint")
        except ImportError:
            # Fallback: manual loading
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'feature_columns' in checkpoint:
                instance._checkpoint_features = checkpoint['feature_columns']
                logger.info(f"Loaded {len(instance._checkpoint_features)} features (manual)")
            else:
                raise ValueError("Checkpoint doesn't contain feature_columns")
        
        return instance
    
    def load_csv(self, path: str) -> pd.DataFrame:
        """Load data and apply checkpoint features if available."""
        df = super().load_csv(path)
        
        # Override features with checkpoint features if available
        if self._checkpoint_features:
            # Verify features exist in data
            available = set(df.columns)
            valid_features = [f for f in self._checkpoint_features if f in available]
            
            missing = set(self._checkpoint_features) - set(valid_features)
            if missing:
                logger.warning(f"Missing features (will use zeros): {missing}")
                for f in missing:
                    df[f] = 0
                valid_features = self._checkpoint_features
            
            self.feature_columns = valid_features
            logger.info(f"Using {len(self.feature_columns)} checkpoint features")
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get current feature columns."""
        return self.feature_columns.copy()


# =============================================================================
# Convenience Functions
# =============================================================================

def load_data_for_evaluation(
    data_path: str,
    checkpoint_path: str,
    seq_len: int = 30,
    test_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load data for model evaluation using checkpoint's features.
    
    Args:
        data_path: Path to CSV data
        checkpoint_path: Path to model checkpoint
        seq_len: Sequence length
        test_ratio: Ratio of data for testing
    
    Returns:
        X_test, y_test, close_prices, feature_columns
    """
    loader = EnhancedDataLoaderV3.from_checkpoint(
        checkpoint_path,
        sequence_length=seq_len,
    )
    
    df = loader.load_csv(data_path)
    
    # Split (only need test portion)
    n = len(df)
    test_start = int(n * (1 - test_ratio))
    
    train_df = df.iloc[:test_start]
    test_df = df.iloc[test_start:]
    
    # Scale
    scaler = RobustScaler()
    scaler.fit(train_df[loader.feature_columns].values)
    test_scaled = scaler.transform(test_df[loader.feature_columns].values)
    
    close_prices = test_df['close'].values
    
    # Create sequences
    X_test, y_test = loader.create_sequences(
        test_scaled,
        close_prices,
        seq_len,
    )
    
    # Align close prices
    close_aligned = close_prices[seq_len-1:-1][:len(y_test)]
    
    return X_test, y_test, close_aligned, loader.feature_columns


def get_available_features(data_path: str) -> List[str]:
    """Get all available features from a data file."""
    loader = EnhancedDataLoaderV2()
    df = loader.load_csv(data_path)
    return loader.feature_columns