# training/train_trend_classifier.py
"""
Training script for Trend Classifier (FTDM-V1 Step 4).

This script:
1. Loads historical OHLCV data
2. Computes features using the same pipeline as FusionFXTrendDetector
3. Generates labels based on future price movement
4. Trains and saves the XGBoost classifier

Usage:
    python scripts/train_trend_classifier.py --data data/EURUSD_H1.csv --output models/trend_classifier.joblib
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.trend_classifier import TrendClassifier, TrendClassifierConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrendFeatureExtractor:
    """
    Extracts the 13 features expected by TrendClassifier.
    
    Mirrors the features from FusionFXTrendDetector._prepare_ml_input()
    """
    
    def __init__(
        self,
        ema_periods: Tuple[int, int, int] = (20, 50, 200),
        adx_period: int = 14,
        roc_periods: Tuple[int, int] = (5, 10),
        atr_period: int = 14,
    ):
        self.ema_periods = ema_periods
        self.adx_period = adx_period
        self.roc_periods = roc_periods
        self.atr_period = atr_period
    
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all required indicators on the dataframe.
        
        Args:
            df: DataFrame with OHLCV columns
        
        Returns:
            DataFrame with added indicator columns
        """
        df = df.copy()
        
        # EMAs
        for period in self.ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # ADX and DI
        df = self._compute_adx(df)
        
        # ROC
        for period in self.roc_periods:
            df[f'roc_{period}'] = df['close'].pct_change(period) * 100
        
        # ATR for volatility
        df['atr'] = self._compute_atr(df)
        
        # Bollinger Band width for compression
        df['bb_width'] = self._compute_bb_width(df)
        
        return df
    
    def _compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Compute ADX, +DI, -DI."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        df['adx'] = adx.values
        df['plus_di'] = plus_di.values
        df['minus_di'] = minus_di.values
        
        return df
    
    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        """Compute ATR."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.ewm(span=self.atr_period, adjust=False).mean()
    
    def _compute_bb_width(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Compute Bollinger Band width (proxy for volatility compression)."""
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        
        upper = sma + 2 * std
        lower = sma - 2 * std
        
        # Normalized width
        width = (upper - lower) / sma
        
        return width
    
    def extract_features(self, df: pd.DataFrame, idx: int) -> Optional[np.ndarray]:
        """
        Extract the 13 features for a single bar.
        
        Args:
            df: DataFrame with computed indicators
            idx: Index of the bar to extract features for
        
        Returns:
            Array of 13 features or None if insufficient data
        """
        if idx < 200:  # Need enough history for EMA200
            return None
        
        row = df.iloc[idx]
        
        # Check for NaN in critical columns
        required_cols = ['adx', 'plus_di', 'minus_di', 'ema_20', 'ema_50', 'ema_200']
        if any(pd.isna(row[col]) for col in required_cols):
            return None
        
        close = row['close']
        
        # 1. struct_score: Simplified - based on price vs EMAs
        above_count = sum([
            close > row['ema_20'],
            close > row['ema_50'],
            close > row['ema_200'],
        ])
        struct_score = above_count / 3.0
        
        # 2. mtf_score: Simplified - use EMA alignment strength
        ema_order_bull = (row['ema_20'] > row['ema_50'] > row['ema_200'])
        ema_order_bear = (row['ema_20'] < row['ema_50'] < row['ema_200'])
        mtf_score = 0.8 if (ema_order_bull or ema_order_bear) else 0.4
        
        # 3. regime: 1 if trending (ADX > 25), 0 if ranging
        regime = 1 if row['adx'] > 25 else 0
        
        # 4-6. ADX, +DI, -DI
        adx = row['adx']
        plus_di = row['plus_di']
        minus_di = row['minus_di']
        
        # 7-9. Price above EMAs (binary)
        price_above_ema20 = 1 if close > row['ema_20'] else 0
        price_above_ema50 = 1 if close > row['ema_50'] else 0
        price_above_ema200 = 1 if close > row['ema_200'] else 0
        
        # 10. EMA alignment: -1 (bearish) to +1 (bullish)
        if ema_order_bull:
            ema_alignment = 1.0
        elif ema_order_bear:
            ema_alignment = -1.0
        else:
            ema_alignment = 0.0
        
        # 11. Volatility compression (normalized BB width)
        # Lower = more compressed
        bb_width = row.get('bb_width', 0.02)
        vol_compression = 1.0 - min(bb_width / 0.05, 1.0)  # Normalize
        
        # 12-13. ROC
        roc_5 = row.get('roc_5', 0)
        roc_10 = row.get('roc_10', 0)
        
        # Handle NaN in ROC
        roc_5 = 0 if pd.isna(roc_5) else roc_5
        roc_10 = 0 if pd.isna(roc_10) else roc_10
        
        features = np.array([
            struct_score,
            mtf_score,
            regime,
            adx,
            plus_di,
            minus_di,
            price_above_ema20,
            price_above_ema50,
            price_above_ema200,
            ema_alignment,
            vol_compression,
            roc_5,
            roc_10,
        ])
        
        return features


class TrendLabeler:
    """
    Generates trend labels based on future price movement.
    """
    
    def __init__(
        self,
        forward_window: int = 20,
        trend_threshold_pct: float = 1.0,
    ):
        """
        Args:
            forward_window: Number of bars to look ahead
            trend_threshold_pct: Minimum % move to classify as trend
        """
        self.forward_window = forward_window
        self.trend_threshold_pct = trend_threshold_pct
    
    def generate_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate trend labels for each bar.
        
        Args:
            df: DataFrame with 'close' column
        
        Returns:
            Array of labels: -1 (BEAR), 0 (SIDEWAYS), 1 (BULL)
        """
        closes = df['close'].values
        n = len(closes)
        labels = np.zeros(n)
        
        for i in range(n - self.forward_window):
            current = closes[i]
            future = closes[i + self.forward_window]
            
            pct_change = (future - current) / current * 100
            
            if pct_change >= self.trend_threshold_pct:
                labels[i] = 1  # BULLISH
            elif pct_change <= -self.trend_threshold_pct:
                labels[i] = -1  # BEARISH
            else:
                labels[i] = 0  # SIDEWAYS
        
        # Last forward_window bars get NaN (no future data)
        labels[-self.forward_window:] = np.nan
        
        return labels


def load_ohlcv_data(path: Path) -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    df = pd.read_csv(path)
    
    # Normalize column names
    df.columns = df.columns.str.lower()
    
    # Ensure required columns
    required = ['open', 'high', 'low', 'close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Parse time if present
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    elif 'date' in df.columns:
        df['time'] = pd.to_datetime(df['date'])
    
    logger.info(f"Loaded {len(df)} rows from {path}")
    
    return df


def prepare_training_data(
    df: pd.DataFrame,
    forward_window: int = 20,
    trend_threshold_pct: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features and labels for training.
    
    Args:
        df: Raw OHLCV DataFrame
        forward_window: Bars to look ahead for labeling
        trend_threshold_pct: Threshold for trend classification
    
    Returns:
        X: Feature matrix
        y: Labels
    """
    # Compute indicators
    extractor = TrendFeatureExtractor()
    df = extractor.compute_indicators(df)
    
    # Generate labels
    labeler = TrendLabeler(forward_window, trend_threshold_pct)
    labels = labeler.generate_labels(df)
    
    # Extract features for each bar
    features_list = []
    labels_list = []
    
    for i in range(len(df)):
        if np.isnan(labels[i]):
            continue
        
        features = extractor.extract_features(df, i)
        if features is not None:
            features_list.append(features)
            labels_list.append(labels[i])
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    logger.info(f"Prepared {len(X)} samples")
    logger.info(f"Class distribution: BEAR={sum(y==-1)}, SIDEWAYS={sum(y==0)}, BULL={sum(y==1)}")
    
    return X, y


def main():
    parser = argparse.ArgumentParser(description='Train Trend Classifier')
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to OHLCV CSV file'
    )
    parser.add_argument(
        '--output', type=str, default='models/trend_classifier.joblib',
        help='Output path for trained model'
    )
    parser.add_argument(
        '--forward-window', type=int, default=20,
        help='Bars to look ahead for labeling'
    )
    parser.add_argument(
        '--threshold', type=float, default=1.0,
        help='Percentage threshold for trend classification'
    )
    parser.add_argument(
        '--n-estimators', type=int, default=100,
        help='Number of trees in the ensemble'
    )
    parser.add_argument(
        '--max-depth', type=int, default=5,
        help='Maximum tree depth'
    )
    parser.add_argument(
        '--synthetic', action='store_true',
        help='Use synthetic data for demo (ignores --data)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Trend Classifier Training")
    print("=" * 60)
    
    if args.synthetic:
        # Use synthetic data
        from models.trend_classifier import generate_synthetic_training_data
        logger.info("Using synthetic training data")
        X, y = generate_synthetic_training_data(n_samples=5000)
    else:
        # Load real data
        data_path = Path(args.data)
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            sys.exit(1)
        
        df = load_ohlcv_data(data_path)
        X, y = prepare_training_data(
            df,
            forward_window=args.forward_window,
            trend_threshold_pct=args.threshold,
        )
    
    if len(X) < 100:
        logger.error(f"Insufficient training data: {len(X)} samples")
        sys.exit(1)
    
    # Configure and train
    config = TrendClassifierConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    
    classifier = TrendClassifier(config)
    metrics = classifier.fit(X, y, validate=True)
    
    # Print feature importance
    print("\nFeature Importance:")
    print(classifier.get_feature_importance().to_string(index=False))
    
    # Save model
    output_path = Path(args.output)
    classifier.save(output_path)
    
    print(f"\n✅ Model saved to {output_path}")
    print(f"   Test Accuracy: {metrics['test_accuracy']:.4f}")
    if 'cv_mean' in metrics:
        print(f"   CV Accuracy: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")


if __name__ == "__main__":
    main()