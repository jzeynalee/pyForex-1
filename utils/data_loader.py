# utils/data_loader.py

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, List, Union
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Setup logger
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Robust data loader for financial time series.
    Combines production-grade engineering with leakage-free scaling.
    """
    
    VALID_SCALERS = {
        'minmax': MinMaxScaler,
        'standard': StandardScaler,
        'robust': RobustScaler
    }
    
    REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'tick_volume']
    
    def __init__(self, scaler_type: str = 'minmax', feature_range: Tuple[int, int] = (0, 1)):
        self.scaler_type = scaler_type
        self.scaler = self._initialize_scaler(scaler_type, feature_range)
        self.is_fitted = False

    def _initialize_scaler(self, scaler_type, feature_range):
        if scaler_type == 'minmax':
            return MinMaxScaler(feature_range=feature_range)
        elif scaler_type in self.VALID_SCALERS:
            return self.VALID_SCALERS[scaler_type]()
        else:
            raise ValueError(f"Invalid scaler: {scaler_type}")

    def load_csv(self, path: Union[str, Path]) -> pd.DataFrame:
        """Loads and cleans CSV data."""
        try:
            df = pd.read_csv(path)
            # Normalize columns
            df.columns = [c.lower() for c in df.columns]
            
            # Validate
            missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns: {missing}")
                
            df = df[self.REQUIRED_COLUMNS].copy()
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

    def split_and_scale(self, df: pd.DataFrame, split_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        [CRITICAL FIX] Splits data FIRST, then scales.
        Prevents data leakage by fitting scaler ONLY on training data.
        """
        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # Fit on Train ONLY
        self.scaler.fit(train_df)
        self.is_fitted = True
        
        train_scaled = self.scaler.transform(train_df)
        test_scaled = self.scaler.transform(test_df)
        
        logger.info(f"Split data: {len(train_df)} Train, {len(test_df)} Test")
        return train_scaled, test_scaled

    def inverse_scale(self, data: np.ndarray) -> np.ndarray:
        """Crucial for converting model predictions back to real prices."""
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted")
        return self.scaler.inverse_transform(data)

    def create_sequences(
        self, 
        data: np.ndarray, 
        seq_len: int = 60, 
        label_strategy: str = 'binary_direction'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates sequences X (0 to t-1) and targets y (t).
        """
        if len(data) < seq_len + 1:
             return np.array([]), np.array([])

        X, y = [], []
        
        # Loop aligns correctly to predict the IMMEDIATE next candle
        for i in range(seq_len, len(data)):
            # Input: Sequence from i-seq_len to i-1
            X.append(data[i-seq_len:i])
            
            # Label Generation
            current_close = data[i][3]     # Close at t
            prev_close = data[i-1][3]      # Close at t-1
            
            label = self._compute_label(current_close, prev_close, label_strategy)
            y.append(label)
            
        return np.array(X), np.array(y)

    def _compute_label(self, current, prev, strategy):
        if strategy == 'binary_direction':
            return 1 if current > prev else 0
        elif strategy == 'regression':
            return (current - prev) / prev if prev != 0 else 0
        # Add ternary/threshold logic here if needed
        return 0