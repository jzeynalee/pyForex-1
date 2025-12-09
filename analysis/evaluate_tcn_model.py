# analysis/evaluate_tcn_model.py
"""
Evaluation script for Enhanced TCN models.

Key improvement: Loads feature configuration FROM the checkpoint,
eliminating the need for hardcoded TOP_FEATURES imports.

Usage:
    python analysis/evaluate_tcn_model.py --model models/weights/tcn_enhanced_best.pt
    python analysis/evaluate_tcn_model.py --model models/weights/tcn_enhanced_best.pt --data data/raw/eurusd_latest.csv
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix


def load_tcn_checkpoint(model_path: str, device: str = 'auto') -> Tuple[torch.nn.Module, List[str], Dict]:
    """
    Load TCN model and feature configuration from checkpoint.
    
    Returns:
        model: Loaded model ready for inference
        feature_columns: List of features used during training
        checkpoint: Full checkpoint dictionary
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Handle both old and new checkpoint formats
    if 'feature_columns' in checkpoint:
        feature_columns = checkpoint['feature_columns']
    elif 'config' in checkpoint and 'features' in checkpoint['config']:
        feature_columns = checkpoint['config']['features']
    else:
        raise ValueError(
            "Checkpoint doesn't contain feature_columns. "
            "Was this model trained with train_tcn_enhanced.py?"
        )
    
    # Get model config
    if 'config' in checkpoint and 'model' in checkpoint['config']:
        model_config = checkpoint['config']['model']
        training_config = checkpoint['config'].get('training', {})
    else:
        # Fallback for older checkpoints
        model_config = {
            'input_dim': len(feature_columns),
            'hidden_dim': 64,
        }
        training_config = {'num_classes': 3, 'dropout': 0.2}
    
    # Import and rebuild model
    from training.train_tcn_enhanced import EnhancedTCN
    
    model = EnhancedTCN(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_classes=training_config.get('num_classes', 3),
        dropout=training_config.get('dropout', 0.2),
    )
    
    # Load weights
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move to device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    print(f"✅ Loaded model from {model_path}")
    print(f"   Features: {len(feature_columns)}")
    print(f"   Input dim: {model_config['input_dim']}")
    if 'profile' in checkpoint and checkpoint['profile']:
        print(f"   Profile: {checkpoint['profile']}")
    
    return model, feature_columns, checkpoint


class TCNEvaluator:
    """
    Comprehensive evaluator for TCN models.
    
    Features:
    - Threshold optimization for trading signals
    - Backtesting with spread costs
    - Per-class metrics
    - Visualization
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        feature_columns: List[str],
        device: str = 'auto',
    ):
        self.model = model
        self.feature_columns = feature_columns
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def prepare_data(
        self,
        data_path: str,
        seq_len: int = 30,
        trend_threshold: float = 0.05,
        test_ratio: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare test data using same features as training.
        
        Returns:
            X_test: Test sequences
            y_test: Test labels
            close_prices: Close prices for backtesting
            entry_indices: Indices for each prediction
        """
        # Load data
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.lower().str.strip()
        
        # Add any missing technical features
        df = self._ensure_features(df)
        
        # Check feature availability
        available = set(df.columns)
        missing = [f for f in self.feature_columns if f not in available]
        if missing:
            print(f"⚠️ Missing features (will use zeros): {missing}")
            for f in missing:
                df[f] = 0
        
        # Split data (use last test_ratio for evaluation)
        n = len(df)
        test_start = int(n * (1 - test_ratio))
        
        train_df = df.iloc[:test_start]
        test_df = df.iloc[test_start:]
        
        # Scale using training data
        scaler = RobustScaler()
        scaler.fit(train_df[self.feature_columns].values)
        
        test_scaled = scaler.transform(test_df[self.feature_columns].values)
        close_prices = test_df['close'].values
        
        # Create sequences
        X_test = []
        y_test = []
        entry_indices = []
        
        limit = len(test_scaled) - seq_len - 1
        for i in range(limit):
            X_test.append(test_scaled[i:i+seq_len])
            
            # Label based on next candle return
            entry_price = close_prices[i + seq_len - 1]
            exit_price = close_prices[i + seq_len]
            pct_return = (exit_price - entry_price) / entry_price
            
            if pct_return > trend_threshold:
                y_test.append(2)  # Bull
            elif pct_return < -trend_threshold:
                y_test.append(0)  # Bear
            else:
                y_test.append(1)  # Sideways
            
            entry_indices.append(i + seq_len - 1)
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        close_prices = close_prices[seq_len-1:-1][:len(y_test)]
        
        print(f"   Test samples: {len(X_test)}")
        print(f"   Class distribution: Bear={sum(y_test==0)}, Side={sum(y_test==1)}, Bull={sum(y_test==2)}")
        
        return X_test, y_test, close_prices, np.array(entry_indices)
    
    def _ensure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add commonly needed technical features if missing."""
        close = df['close'].values
        high = df['high'].values if 'high' in df.columns else close
        low = df['low'].values if 'low' in df.columns else close
        
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
        
        # ATR
        if 'atr_14' not in df.columns:
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
            df['atr_14'] = pd.Series(tr).rolling(14).mean().fillna(0)
        
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
        
        # Bollinger Bands
        if 'bb_position' not in df.columns:
            sma20 = pd.Series(close).rolling(20).mean()
            std20 = pd.Series(close).rolling(20).std()
            df['bb_upper'] = sma20 + 2 * std20
            df['bb_lower'] = sma20 - 2 * std20
            df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # Stochastic
        if 'stoch_k' not in df.columns:
            low_14 = pd.Series(low).rolling(14).min()
            high_14 = pd.Series(high).rolling(14).max()
            df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # ROC
        for period in [5, 10, 20]:
            col = f'roc_{period}'
            if col not in df.columns:
                df[col] = (close - np.roll(close, period)) / (np.roll(close, period) + 1e-10) * 100
        
        # Fill NaN
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def get_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities for all samples."""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
        
        return probs
    
    def optimize_threshold(
        self,
        probs: np.ndarray,
        y_true: np.ndarray,
        close_prices: np.ndarray,
        spread_cost: float = 0.0001,
        threshold_range: Tuple[float, float] = (0.35, 0.80),
        step: float = 0.02,
    ) -> Dict:
        """
        Find optimal confidence threshold for trading signals.
        
        Returns:
            Dict with best threshold and metrics
        """
        print("\n🔍 OPTIMIZING CONFIDENCE THRESHOLD")
        print("-" * 70)
        print(f"{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Return':<10} | {'Sharpe':<8}")
        print("-" * 70)
        
        # Calculate raw returns
        returns = np.diff(close_prices) / close_prices[:-1]
        
        best_result = {'threshold': 0.5, 'return': -999, 'sharpe': -999}
        all_results = []
        
        for thresh in np.arange(threshold_range[0], threshold_range[1], step):
            # Generate signals based on threshold
            signals = np.zeros(len(probs))
            signals[probs[:, 0] > thresh] = -1  # Short (Bear)
            signals[probs[:, 2] > thresh] = 1   # Long (Bull)
            
            # Align signals with returns
            trade_signals = signals[:-1]
            
            # Calculate strategy returns with spread cost
            costs = np.abs(trade_signals) * spread_cost
            strategy_returns = (returns * trade_signals) - costs
            
            total_return = np.sum(strategy_returns)
            n_trades = np.sum(trade_signals != 0)
            
            # Win rate
            active_mask = trade_signals != 0
            if active_mask.sum() > 0:
                wins = (strategy_returns[active_mask] > 0).sum()
                win_rate = wins / active_mask.sum()
                
                # Sharpe (annualized, assuming hourly data)
                if strategy_returns[active_mask].std() > 0:
                    sharpe = (strategy_returns[active_mask].mean() / 
                             strategy_returns[active_mask].std() * np.sqrt(252 * 24))
                else:
                    sharpe = 0
            else:
                win_rate = 0
                sharpe = 0
            
            result = {
                'threshold': thresh,
                'n_trades': int(n_trades),
                'win_rate': win_rate,
                'total_return': total_return,
                'sharpe': sharpe,
            }
            all_results.append(result)
            
            print(f"{thresh:.2f}{'':<6} | {n_trades:<8} | {win_rate:.1%}     | {total_return:.4f}     | {sharpe:.2f}")
            
            if total_return > best_result['return']:
                best_result = result
        
        print("-" * 70)
        print(f"🏆 BEST: Threshold {best_result['threshold']:.2f} | "
              f"Return {best_result['total_return']:.4f} | "
              f"Win Rate {best_result['win_rate']:.1%}")
        
        return {
            'best': best_result,
            'all_results': all_results,
        }
    
    def plot_results(
        self,
        probs: np.ndarray,
        close_prices: np.ndarray,
        threshold: float,
        spread_cost: float = 0.0001,
        save_path: Optional[str] = None,
    ):
        """Plot cumulative returns with optimized threshold."""
        returns = np.diff(close_prices) / close_prices[:-1]
        
        # Generate signals
        signals = np.zeros(len(probs))
        signals[probs[:, 0] > threshold] = -1
        signals[probs[:, 2] > threshold] = 1
        
        trade_signals = signals[:-1]
        costs = np.abs(trade_signals) * spread_cost
        strategy_returns = (returns * trade_signals) - costs
        
        # Cumulative returns
        cum_strategy = np.cumsum(strategy_returns)
        cum_market = np.cumsum(returns)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Cumulative returns
        ax1 = axes[0, 0]
        ax1.plot(cum_market, label='Market (Buy & Hold)', color='gray', alpha=0.7)
        ax1.plot(cum_strategy, label=f'Strategy (Conf > {threshold:.2f})', color='blue', linewidth=2)
        ax1.set_title('Cumulative Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Return')
        
        # 2. Signal distribution
        ax2 = axes[0, 1]
        signal_counts = [
            np.sum(signals == -1),
            np.sum(signals == 0),
            np.sum(signals == 1)
        ]
        ax2.bar(['Short', 'Hold', 'Long'], signal_counts, color=['red', 'gray', 'green'])
        ax2.set_title('Signal Distribution')
        ax2.set_ylabel('Count')
        
        # 3. Probability distribution
        ax3 = axes[1, 0]
        ax3.hist(probs[:, 0], bins=50, alpha=0.5, label='Bear', color='red')
        ax3.hist(probs[:, 1], bins=50, alpha=0.5, label='Sideways', color='gray')
        ax3.hist(probs[:, 2], bins=50, alpha=0.5, label='Bull', color='green')
        ax3.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.2f})')
        ax3.set_title('Probability Distribution')
        ax3.legend()
        ax3.set_xlabel('Probability')
        
        # 4. Rolling performance
        ax4 = axes[1, 1]
        window = min(100, len(strategy_returns) // 10)
        if window > 1:
            rolling_return = pd.Series(strategy_returns).rolling(window).sum()
            ax4.plot(rolling_return, color='blue')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_title(f'Rolling {window}-Period Return')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Return')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()
    
    def classification_report(self, probs: np.ndarray, y_true: np.ndarray, threshold: float = 0.5):
        """Print detailed classification metrics."""
        # Get predictions at given threshold
        preds = np.argmax(probs, axis=1)
        
        # For trading signals with threshold
        signals = np.ones(len(probs))  # Default to sideways
        signals[probs[:, 0] > threshold] = 0  # Bear
        signals[probs[:, 2] > threshold] = 2  # Bull
        
        print("\n📋 CLASSIFICATION REPORT (argmax)")
        print("-" * 50)
        print(classification_report(
            y_true, preds,
            target_names=['Bear', 'Sideways', 'Bull'],
            digits=3
        ))
        
        print("\n📋 CLASSIFICATION REPORT (threshold={:.2f})".format(threshold))
        print("-" * 50)
        print(classification_report(
            y_true, signals.astype(int),
            target_names=['Bear', 'Sideways', 'Bull'],
            digits=3
        ))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, preds)
        print("\n📊 CONFUSION MATRIX:")
        print(pd.DataFrame(
            cm,
            index=['True: Bear', 'True: Side', 'True: Bull'],
            columns=['Pred: Bear', 'Pred: Side', 'Pred: Bull']
        ))


def main():
    parser = argparse.ArgumentParser(description="Evaluate TCN Model")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/raw/eurusd_latest.csv', help='Path to data')
    parser.add_argument('--seq-len', type=int, default=30, help='Sequence length')
    parser.add_argument('--threshold', type=float, default=0.05, help='Trend threshold')
    parser.add_argument('--spread', type=float, default=0.0001, help='Spread cost')
    parser.add_argument('--save-plot', type=str, default=None, help='Path to save plot')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 TCN Model Evaluation")
    print("=" * 60)
    
    # Load model and features from checkpoint
    model, feature_columns, checkpoint = load_tcn_checkpoint(args.model)
    
    print(f"\n📊 Features loaded from checkpoint:")
    for i, f in enumerate(feature_columns[:10], 1):
        print(f"   {i}. {f}")
    if len(feature_columns) > 10:
        print(f"   ... and {len(feature_columns) - 10} more")
    
    # Create evaluator
    evaluator = TCNEvaluator(model, feature_columns)
    
    # Prepare data
    X_test, y_test, close_prices, _ = evaluator.prepare_data(
        args.data,
        seq_len=args.seq_len,
        trend_threshold=args.threshold,
    )
    
    # Get probabilities
    probs = evaluator.get_probabilities(X_test)
    
    # Optimize threshold
    optimization_result = evaluator.optimize_threshold(
        probs, y_test, close_prices,
        spread_cost=args.spread,
    )
    
    best_threshold = optimization_result['best']['threshold']
    
    # Classification report
    evaluator.classification_report(probs, y_test, threshold=best_threshold)
    
    # Plot
    evaluator.plot_results(
        probs, close_prices, best_threshold,
        spread_cost=args.spread,
        save_path=args.save_plot,
    )


if __name__ == "__main__":
    main()