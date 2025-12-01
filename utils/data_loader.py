# utils/data_loader.py
"""
Robust data loader for financial time series with leak-free scaling.
"""
import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Tuple, Optional, List, Union, Literal
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    sequence_length: int = 60
    label_strategy: Literal['binary', 'ternary', 'regression'] = 'ternary'
    ternary_threshold: float = 0.001  # 0.1% move for ternary classification
    scaler_type: Literal['minmax', 'standard', 'robust'] = 'minmax'
    feature_range: Tuple[float, float] = (0, 1)


class DataLoader:
    """
    Production-grade data loader with:
    - Leak-free train/test splitting
    - Scaler persistence for inference
    - Multiple labeling strategies
    - Robust error handling
    """
    
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'tick_volume']
    
    SCALERS = {
        'minmax': MinMaxScaler,
        'standard': StandardScaler,
        'robust': RobustScaler,
    }
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.scaler = self._create_scaler()
        self.is_fitted = False
        self._feature_columns = None
    
    def _create_scaler(self):
        """Initialize scaler based on config."""
        scaler_cls = self.SCALERS.get(self.config.scaler_type, MinMaxScaler)
        if self.config.scaler_type == 'minmax':
            return scaler_cls(feature_range=self.config.feature_range)
        return scaler_cls()
    
    def load_csv(
        self, 
        path: Union[str, Path],
        additional_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load and validate CSV data.
        
        Args:
            path: Path to CSV file
            additional_columns: Extra columns to include beyond OHLCV
        
        Returns:
            Cleaned DataFrame
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        try:
            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Validate required columns
            required = set(self.REQUIRED_COLUMNS)
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            # Select columns
            columns = self.REQUIRED_COLUMNS.copy()
            if additional_columns:
                for col in additional_columns:
                    if col.lower() in df.columns and col.lower() not in columns:
                        columns.append(col.lower())
            
            df = df[columns].copy()
            
            # Handle missing values
            initial_len = len(df)
            df.dropna(inplace=True)
            dropped = initial_len - len(df)
            if dropped > 0:
                logger.warning(f"Dropped {dropped} rows with NaN values")
            
            # Validate data types
            for col in columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(inplace=True)
            
            self._feature_columns = columns
            logger.info(f"Loaded {len(df)} rows from {path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
    
    def split_and_scale(
        self,
        df: pd.DataFrame,
        split_ratio: float = 0.8,
        validation_ratio: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Split data chronologically and scale WITHOUT data leakage.
        
        CRITICAL: Scaler is fitted ONLY on training data.
        
        Args:
            df: Input DataFrame
            split_ratio: Fraction for training
            validation_ratio: Fraction for validation (from train portion)
        
        Returns:
            Tuple of (train_scaled, test_scaled, val_scaled or None)
        """
        n = len(df)
        train_end = int(n * split_ratio)
        
        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:].copy()
        
        # Optional validation split from training data
        val_scaled = None
        if validation_ratio > 0:
            val_start = int(len(train_df) * (1 - validation_ratio))
            val_df = train_df.iloc[val_start:].copy()
            train_df = train_df.iloc[:val_start].copy()
        
        # FIT SCALER ON TRAIN DATA ONLY
        self.scaler.fit(train_df)
        self.is_fitted = True
        
        # Transform all splits
        train_scaled = self.scaler.transform(train_df)
        test_scaled = self.scaler.transform(test_df)
        
        if validation_ratio > 0:
            val_scaled = self.scaler.transform(val_df)
        
        logger.info(
            f"Split: {len(train_df)} train, {len(test_df)} test" +
            (f", {len(val_df)} val" if validation_ratio > 0 else "")
        )
        
        return train_scaled, test_scaled, val_scaled
    
    def scale(self, df: pd.DataFrame) -> np.ndarray:
        """
        Scale data using fitted scaler (for inference).
        
        Args:
            df: DataFrame to scale
        
        Returns:
            Scaled numpy array
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call split_and_scale() or load_scaler() first.")
        return self.scaler.transform(df)
    
    def inverse_scale(self, data: np.ndarray) -> np.ndarray:
        """Convert scaled values back to original scale."""
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted")
        return self.scaler.inverse_transform(data)
    
    def create_sequences(
        self,
        data: np.ndarray,
        seq_len: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences and labels for training.
        
        Sequences X[i] = data[i:i+seq_len]
        Labels y[i] = label computed from data[i+seq_len]
        
        Args:
            data: Scaled numpy array of shape (n_samples, n_features)
            seq_len: Sequence length (uses config default if None)
        
        Returns:
            X: (n_sequences, seq_len, n_features)
            y: (n_sequences,)
        """
        seq_len = seq_len or self.config.sequence_length
        
        if len(data) < seq_len + 1:
            logger.warning(f"Insufficient data: {len(data)} < {seq_len + 1}")
            return np.array([]), np.array([])
        
        X, y = [], []
        close_idx = 3  # Index of close price in OHLCV
        
        for i in range(len(data) - seq_len):
            # Input sequence
            X.append(data[i:i + seq_len])
            
            # Label: predict movement from seq end to next bar
            prev_close = data[i + seq_len - 1, close_idx]
            curr_close = data[i + seq_len, close_idx]
            
            label = self._compute_label(curr_close, prev_close)
            y.append(label)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64 if self.config.label_strategy != 'regression' else np.float32)
        
        logger.info(f"Created {len(X)} sequences of length {seq_len}")
        return X, y
    
    def _compute_label(self, curr: float, prev: float) -> Union[int, float]:
        """Compute label based on configured strategy."""
        if prev == 0:
            return 2 if self.config.label_strategy == 'ternary' else 0
        
        pct_change = (curr - prev) / prev
        
        if self.config.label_strategy == 'binary':
            return 1 if curr > prev else 0
        
        elif self.config.label_strategy == 'ternary':
            threshold = self.config.ternary_threshold
            if pct_change > threshold:
                return 0  # BUY
            elif pct_change < -threshold:
                return 1  # SELL
            else:
                return 2  # HOLD
        
        elif self.config.label_strategy == 'regression':
            return pct_change
        
        return 0
    
    def save_scaler(self, path: Union[str, Path]):
        """Save fitted scaler to disk."""
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        logger.info(f"Scaler saved to {path}")
    
    def load_scaler(self, path: Union[str, Path]):
        """Load scaler from disk (for inference)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler not found: {path}")
        
        self.scaler = joblib.load(path)
        self.is_fitted = True
        logger.info(f"Scaler loaded from {path}")


def create_inference_window(
    df: pd.DataFrame,
    seq_len: int = 60,
) -> pd.DataFrame:
    """
    Extract the last `seq_len` rows for inference.
    
    Args:
        df: Full DataFrame
        seq_len: Number of rows needed
    
    Returns:
        Last seq_len rows as DataFrame
    """
    if len(df) < seq_len:
        raise ValueError(f"DataFrame has {len(df)} rows, need {seq_len}")
    
    return df.tail(seq_len).copy()
