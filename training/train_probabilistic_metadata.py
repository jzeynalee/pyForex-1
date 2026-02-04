# training/train_probabilistic_metadata.py
"""
Walk-Forward Training for Probabilistic Alpha Factory Metadata
==============================================================

This script performs walk-forward backtesting to populate the feature
metadata store with:
- Regime-specific quantiles for each feature
- Hit rates (conditional accuracy)
- Regime specificity (KL divergence)
- Sharpe contribution

It also trains the temporal refinement model on probability sequences.

Usage:
    python -m training.train_probabilistic_metadata --data data/EURUSD_H1.csv --profile INTRADAY
    python -m training.train_probabilistic_metadata --data data/EURUSD_M5.csv --profile SCALP --folds 5
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alpha_factory.probabilistic_alpha_factory import (
    ProbabilisticAlphaFactory,
    ProbabilisticConfig,
    FeatureMetadataStore,
    RegimePriorCalculator
)
from alpha_factory.mhtcn_temporal_refinement import (
    TemporalRefinementTCN,
    TemporalRefinementConfig,
    TemporalRefinementDataset,
    TemporalRefinementTrainer
)

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward training."""
    
    # Data
    data_path: str = ""
    profile: str = "INTRADAY"
    
    # Walk-forward
    n_folds: int = 5
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Regime labeling
    direction_horizon: int = 12
    direction_threshold: float = 0.001
    
    # Feature selection
    key_features: List[str] = None
    
    # Temporal refinement training
    train_temporal: bool = True
    temporal_epochs: int = 50
    temporal_batch_size: int = 32
    temporal_sequence_length: int = 20
    
    # Output
    output_dir: str = "alpha_factory/metadata"
    save_temporal_weights: bool = True
    
    def __post_init__(self):
        if self.key_features is None:
            self.key_features = [
                "rsi", "macd", "macd_histogram", "adx", "bb_position",
                "atr_ratio", "momentum", "trend_strength", "volatility_ratio",
                "sma_20", "sma_50", "ema_12", "ema_26", "roc", "williams_r",
                "stoch_k", "stoch_d", "cci", "mfi", "obv_ratio"
            ]


class WalkForwardMetadataTrainer:
    """
    Walk-forward trainer for populating feature metadata.
    
    Performs rolling window backtests to compute:
    - Feature statistics per regime
    - Hit rates and specificity
    - Probability sequences for temporal model training
    """
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.metadata_store = FeatureMetadataStore(
            Path(config.output_dir) / "feature_stats.json"
        )
        self.regime_calculator = RegimePriorCalculator(ProbabilisticConfig())
        
        # Storage for temporal training data
        self.prob_sequences: List[np.ndarray] = []
        self.regime_labels: List[int] = []
        
        # Results storage
        self.fold_results: List[Dict] = []
    
    def load_data(self, path: str) -> pd.DataFrame:
        """Load and prepare OHLCV data."""
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
                df['volume'] = 1000
        
        df.reset_index(drop=True, inplace=True)
        logger.info(f"Loaded {len(df)} rows")
        
        return df
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate technical features."""
        logger.info("Generating features...")
        
        try:
            from alpha_factory.features_engineering import FeatureEngineerOptimized
            engineer = FeatureEngineerOptimized()
            featured_df = engineer.generate_features(df.copy())
        except ImportError:
            logger.warning("FeatureEngineerOptimized not available, using basic features")
            featured_df = self._generate_basic_features(df)
        
        logger.info(f"Generated {len(featured_df.columns)} features")
        return featured_df
    
    def _generate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback basic feature generation."""
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma20 + 2 * std20
        df['bb_lower'] = sma20 - 2 * std20
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # ADX (simplified)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        plus_di = 100 * plus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        minus_di = 100 * minus_dm.rolling(14).mean() / (df['atr'] + 1e-10)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Momentum
        df['momentum'] = df['close'].pct_change(10)
        df['roc'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        
        # Volatility ratio
        df['volatility_ratio'] = df['returns'].rolling(10).std() / df['returns'].rolling(50).std()
        
        # Trend strength
        df['trend_strength'] = abs(df['close'] - df['sma_20']) / (df['atr'] + 1e-10)
        
        return df.fillna(0)
    
    def generate_regime_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Generate regime labels based on future price movement."""
        horizon = self.config.direction_horizon
        threshold = self.config.direction_threshold
        
        close = df['close'].values
        n = len(close)
        labels = np.ones(n, dtype=np.int64)  # Default: neutral (1)
        
        for i in range(n - horizon):
            future_return = (close[i + horizon] - close[i]) / close[i]
            
            if future_return > threshold:
                labels[i] = 2  # Bull
            elif future_return < -threshold:
                labels[i] = 0  # Bear
        
        labels[-horizon:] = 1  # Last bars get neutral
        
        return labels
    
    def run_walk_forward(self, df: pd.DataFrame, features_df: pd.DataFrame):
        """Run walk-forward analysis."""
        n = len(df)
        fold_size = n // self.config.n_folds
        
        logger.info(f"Running {self.config.n_folds}-fold walk-forward analysis")
        
        # Generate regime labels
        regime_labels = self.generate_regime_labels(df)
        regime_strings = {0: 'bear', 1: 'neutral', 2: 'bull'}
        
        all_feature_stats = {feat: {'bull': [], 'bear': [], 'neutral': []} 
                            for feat in self.config.key_features}
        all_hit_rates = {feat: [] for feat in self.config.key_features}
        
        for fold_idx in range(self.config.n_folds):
            logger.info(f"Processing fold {fold_idx + 1}/{self.config.n_folds}")
            
            # Define fold boundaries
            test_start = fold_idx * fold_size
            test_end = min((fold_idx + 1) * fold_size, n)
            
            # Training data: everything before test
            train_end = test_start
            train_start = max(0, train_end - int(fold_size * 3))  # Use 3x fold size for training
            
            if train_end - train_start < 100:
                logger.warning(f"Fold {fold_idx}: insufficient training data, skipping")
                continue
            
            train_features = features_df.iloc[train_start:train_end]
            train_labels = regime_labels[train_start:train_end]
            
            test_features = features_df.iloc[test_start:test_end]
            test_labels = regime_labels[test_start:test_end]
            
            # Compute feature statistics per regime
            for feat in self.config.key_features:
                if feat not in train_features.columns:
                    continue
                
                for regime_idx, regime_name in regime_strings.items():
                    mask = train_labels == regime_idx
                    if mask.sum() < 10:
                        continue
                    
                    feat_values = train_features.loc[mask.astype(bool), feat].dropna()
                    if len(feat_values) < 10:
                        continue
                    
                    quantiles = {
                        'q10': float(feat_values.quantile(0.1)),
                        'q25': float(feat_values.quantile(0.25)),
                        'q50': float(feat_values.quantile(0.5)),
                        'q75': float(feat_values.quantile(0.75)),
                        'q90': float(feat_values.quantile(0.9))
                    }
                    all_feature_stats[feat][regime_name].append(quantiles)
            
            # Compute hit rates on test set
            for feat in self.config.key_features:
                if feat not in test_features.columns:
                    continue
                
                feat_values = test_features[feat].values
                
                # Simple hit rate: correlation with regime direction
                # Bull (2) = positive, Bear (0) = negative
                direction = (test_labels - 1).astype(float)  # -1, 0, 1
                
                valid_mask = np.isfinite(feat_values) & np.isfinite(direction)
                if valid_mask.sum() < 10:
                    continue
                
                corr = np.corrcoef(feat_values[valid_mask], direction[valid_mask])[0, 1]
                if np.isfinite(corr):
                    hit_rate = (abs(corr) + 1) / 2  # Map to [0, 1]
                    all_hit_rates[feat].append(hit_rate)
            
            # Collect probability sequences for temporal training
            self._collect_prob_sequences(
                df.iloc[test_start:test_end],
                features_df.iloc[test_start:test_end],
                test_labels
            )
            
            # Store fold results
            self.fold_results.append({
                'fold': fold_idx,
                'train_size': train_end - train_start,
                'test_size': test_end - test_start
            })
        
        # Aggregate statistics across folds
        self._aggregate_and_save_stats(all_feature_stats, all_hit_rates)
    
    def _collect_prob_sequences(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        labels: np.ndarray
    ):
        """Collect probability sequences for temporal model training."""
        seq_len = self.config.temporal_sequence_length
        
        # Generate probability vectors using regime calculator
        prob_history = []
        
        for i in range(len(df)):
            # Get regime prior
            if i >= 20:
                window_df = df.iloc[max(0, i-50):i+1]
                regime_prior = self.regime_calculator.calculate(window_df)
                
                prob_vec = [
                    regime_prior.p_bull,
                    regime_prior.p_bear,
                    regime_prior.p_neutral,
                    1.0 - regime_prior.p_volatile  # Stability
                ]
            else:
                prob_vec = [0.33, 0.33, 0.34, 0.5]
            
            prob_history.append(prob_vec)
            
            # Once we have enough history, create training sample
            if len(prob_history) >= seq_len:
                seq = np.array(prob_history[-seq_len:], dtype=np.float32)
                self.prob_sequences.append(seq)
                self.regime_labels.append(int(labels[i]))
    
    def _aggregate_and_save_stats(
        self,
        all_feature_stats: Dict,
        all_hit_rates: Dict
    ):
        """Aggregate statistics across folds and save to metadata store."""
        logger.info("Aggregating feature statistics...")
        
        for feat in self.config.key_features:
            # Aggregate quantiles
            quantiles_bull = self._aggregate_quantiles(all_feature_stats[feat]['bull'])
            quantiles_bear = self._aggregate_quantiles(all_feature_stats[feat]['bear'])
            quantiles_neutral = self._aggregate_quantiles(all_feature_stats[feat]['neutral'])
            
            # Aggregate hit rate
            hit_rates = all_hit_rates.get(feat, [])
            hit_rate = float(np.mean(hit_rates)) if hit_rates else 0.5
            
            # Calculate specificity (variance of means across regimes)
            means = []
            for regime_name in ['bull', 'bear', 'neutral']:
                regime_quantiles = all_feature_stats[feat][regime_name]
                if regime_quantiles:
                    regime_means = [q['q50'] for q in regime_quantiles]
                    means.append(np.mean(regime_means))
            
            if len(means) >= 2:
                specificity = float(np.std(means) / (np.mean(np.abs(means)) + 1e-10))
                specificity = np.clip(specificity, 0, 1)
            else:
                specificity = 0.5
            
            # Update metadata store
            if quantiles_bull or quantiles_bear or quantiles_neutral:
                self.metadata_store.update_feature_stats(
                    feat,
                    quantiles_bull,
                    quantiles_bear,
                    quantiles_neutral,
                    hit_rate,
                    specificity
                )
        
        # Save metadata
        self.metadata_store.save()
        logger.info(f"Saved feature statistics for {len(self.config.key_features)} features")
    
    def _aggregate_quantiles(self, quantile_list: List[Dict]) -> Dict:
        """Aggregate quantiles across folds."""
        if not quantile_list:
            return {}
        
        result = {}
        for key in ['q10', 'q25', 'q50', 'q75', 'q90']:
            values = [q[key] for q in quantile_list if key in q]
            if values:
                result[key] = float(np.mean(values))
        
        return result
    
    def train_temporal_model(self) -> Optional[TemporalRefinementTCN]:
        """Train temporal refinement model on collected sequences."""
        if not self.prob_sequences:
            logger.warning("No probability sequences collected, skipping temporal training")
            return None
        
        logger.info(f"Training temporal refinement model on {len(self.prob_sequences)} sequences")
        
        # Prepare data
        sequences = np.array(self.prob_sequences)
        labels = np.array(self.regime_labels)
        
        # Split train/val
        n = len(sequences)
        train_size = int(n * 0.8)
        
        train_sequences = sequences[:train_size]
        train_labels = labels[:train_size]
        val_sequences = sequences[train_size:]
        val_labels = labels[train_size:]
        
        # Create datasets
        train_dataset = TemporalRefinementDataset(
            train_sequences, train_labels, self.config.temporal_sequence_length
        )
        val_dataset = TemporalRefinementDataset(
            val_sequences, val_labels, self.config.temporal_sequence_length
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.temporal_batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.temporal_batch_size, shuffle=False
        )
        
        # Create model
        config = TemporalRefinementConfig(
            sequence_length=self.config.temporal_sequence_length
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TemporalRefinementTCN(config).to(device)
        
        # Create trainer
        trainer = TemporalRefinementTrainer(model, config, device)
        
        # Training loop
        best_val_acc = 0.0
        best_state = None
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.config.temporal_epochs):
            train_loss = trainer.train_epoch(train_loader)
            val_loss, val_acc = trainer.evaluate(val_loader)
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
        
        logger.info(f"Temporal model training complete. Best val accuracy: {best_val_acc:.4f}")
        
        # Save weights
        if self.config.save_temporal_weights:
            weights_path = Path(self.config.output_dir) / "temporal_refinement.pth"
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'best_val_acc': best_val_acc
            }, weights_path)
            logger.info(f"Saved temporal refinement weights to {weights_path}")
        
        return model
    
    def run(self, data_path: str) -> Dict:
        """Run full walk-forward training pipeline."""
        # Load data
        df = self.load_data(data_path)
        
        # Generate features
        features_df = self.generate_features(df)
        
        # Run walk-forward analysis
        self.run_walk_forward(df, features_df)
        
        # Train temporal model
        temporal_model = None
        if self.config.train_temporal:
            temporal_model = self.train_temporal_model()
        
        return {
            'n_folds': len(self.fold_results),
            'n_features': len(self.config.key_features),
            'n_prob_sequences': len(self.prob_sequences),
            'temporal_model_trained': temporal_model is not None
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Walk-forward training for Probabilistic Alpha Factory metadata"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to OHLCV CSV file"
    )
    parser.add_argument(
        "--profile", type=str, default="INTRADAY",
        choices=["SCALP", "INTRADAY", "SWING"],
        help="Trading profile"
    )
    parser.add_argument(
        "--folds", type=int, default=5,
        help="Number of walk-forward folds"
    )
    parser.add_argument(
        "--temporal-epochs", type=int, default=50,
        help="Epochs for temporal model training"
    )
    parser.add_argument(
        "--output-dir", type=str, default="alpha_factory/metadata",
        help="Output directory for metadata"
    )
    parser.add_argument(
        "--no-temporal", action="store_true",
        help="Skip temporal model training"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create config
    config = WalkForwardConfig(
        data_path=args.data,
        profile=args.profile,
        n_folds=args.folds,
        temporal_epochs=args.temporal_epochs,
        output_dir=args.output_dir,
        train_temporal=not args.no_temporal
    )
    
    # Run training
    trainer = WalkForwardMetadataTrainer(config)
    results = trainer.run(args.data)
    
    print("\n" + "=" * 60)
    print("Walk-Forward Training Complete")
    print("=" * 60)
    print(f"Folds processed: {results['n_folds']}")
    print(f"Features analyzed: {results['n_features']}")
    print(f"Probability sequences: {results['n_prob_sequences']}")
    print(f"Temporal model trained: {results['temporal_model_trained']}")
    print(f"Metadata saved to: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
