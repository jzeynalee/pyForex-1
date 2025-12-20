# analysis/evaluate_tcn_horizon.py
"""
Horizon-based evaluation for TCN models.

Evaluates model performance for fixed holding periods (e.g., 5 candles ahead).
Useful for strategies that hold positions for a predetermined duration.

Usage:
    python analysis/evaluate_tcn_horizon.py --model models/weights/tcn_enhanced_best.pt --horizon 5
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


def load_checkpoint_with_features(model_path: str, device: str = 'auto'):
    """Load model and features from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract features
    if 'feature_columns' in checkpoint:
        feature_columns = checkpoint['feature_columns']
    elif 'features' in checkpoint:
        feature_columns = checkpoint['features']
    else:
        raise ValueError("Checkpoint doesn't contain feature_columns")
    
    # Get model config
    model_config = checkpoint.get('config', {}).get('model', {})
    training_config = checkpoint.get('config', {}).get('training', {})
    
    # Rebuild model
    try:
        from training.train_tcn_enhanced import EnhancedTCN
    except ImportError as e:
        # Fallback: create a dummy TCN model for testing
        class DummyTCN(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim=64, num_classes=3, dropout=0.2):
                super().__init__()
                self.fc = torch.nn.Linear(input_dim, num_classes)
            
            def forward(self, x):
                return self.fc(x)
        
        EnhancedTCN = DummyTCN
    
    model = EnhancedTCN(
        input_dim=model_config.get('input_dim', len(feature_columns)),
        hidden_dim=model_config.get('hidden_dim', 64),
        num_classes=training_config.get('num_classes', 3),
        dropout=training_config.get('dropout', 0.2),
    )
    
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    return model, feature_columns, checkpoint


class HorizonEvaluator:
    """
    Evaluates model with fixed holding period strategy.
    
    Instead of exiting when signal changes, holds for exactly N candles.
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
        
        self.model.to(self.device)
        self.model.eval()
    
    def prepare_horizon_data(
        self,
        data_path: str,
        seq_len: int = 30,
        horizon: int = 5,
        test_ratio: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for horizon-based evaluation.
        
        Returns:
            X_test: Test sequences
            entry_prices: Price at entry (last candle of sequence)
            exit_prices: Price at exit (horizon candles later)
        """
        df = pd.read_csv(data_path)
        df.columns = df.columns.str.lower().str.strip()
        df = self._ensure_features(df)
        
        # Handle missing features
        for f in self.feature_columns:
            if f not in df.columns:
                df[f] = 0
        
        # Split
        n = len(df)
        test_start = int(n * (1 - test_ratio))
        
        train_df = df.iloc[:test_start]
        test_df = df.iloc[test_start:]
        
        # Scale
        scaler = RobustScaler()
        scaler.fit(train_df[self.feature_columns].values)
        test_scaled = scaler.transform(test_df[self.feature_columns].values)
        
        close_prices = test_df['close'].values
        
        # Create sequences with horizon alignment
        X_test = []
        entry_prices = []
        exit_prices = []
        
        limit = len(test_scaled) - seq_len - horizon
        for i in range(limit):
            X_test.append(test_scaled[i:i+seq_len])
            
            entry_idx = i + seq_len - 1
            entry_prices.append(close_prices[entry_idx])
            exit_prices.append(close_prices[entry_idx + horizon])
        
        X_test = np.array(X_test)
        entry_prices = np.array(entry_prices)
        exit_prices = np.array(exit_prices)
        
        print(f"   Test samples: {len(X_test)}")
        print(f"   Horizon: {horizon} candles")
        
        return X_test, entry_prices, exit_prices
    
    def _ensure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical features if missing."""
        close = df['close'].values
        high = df['high'].values if 'high' in df.columns else close
        low = df['low'].values if 'low' in df.columns else close
        
        # Basic features
        if 'rsi_14' not in df.columns:
            deltas = np.diff(close)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = pd.Series(gains).rolling(14).mean()
            avg_loss = pd.Series(losses).rolling(14).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            df['rsi_14'] = np.concatenate([[50], rsi.values])
        
        if 'atr_14' not in df.columns:
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
            df['atr_14'] = pd.Series(tr).rolling(14).mean().fillna(0)
        
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
        
        if 'bb_position' not in df.columns:
            sma20 = pd.Series(close).rolling(20).mean()
            std20 = pd.Series(close).rolling(20).std()
            df['bb_upper'] = sma20 + 2 * std20
            df['bb_lower'] = sma20 - 2 * std20
            df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        if 'stoch_k' not in df.columns:
            low_14 = pd.Series(low).rolling(14).min()
            high_14 = pd.Series(high).rolling(14).max()
            df['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
            df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        for period in [5, 10, 20]:
            col = f'roc_{period}'
            if col not in df.columns:
                df[col] = (close - np.roll(close, period)) / (np.roll(close, period) + 1e-10) * 100
        
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return df
    
    def get_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
        
        return probs
    
    def evaluate_horizon_strategy(
        self,
        probs: np.ndarray,
        entry_prices: np.ndarray,
        exit_prices: np.ndarray,
        spread_cost: float = 0.0001,
        threshold_range: Tuple[float, float] = (0.35, 0.70),
        step: float = 0.02,
    ) -> Dict:
        """
        Evaluate fixed-horizon strategy at different thresholds.
        
        Strategy:
        - If prob(Bull) > threshold: Go long, hold for horizon candles
        - If prob(Bear) > threshold: Go short, hold for horizon candles
        - Otherwise: Stay flat
        """
        print("\n🔍 OPTIMIZING HORIZON THRESHOLD")
        print("-" * 70)
        print(f"{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Return':<10} | {'Avg Trade':<10}")
        print("-" * 70)
        
        # Raw returns for each position
        raw_long_returns = (exit_prices - entry_prices) / entry_prices
        
        best_result = {'threshold': 0.5, 'return': -999}
        all_results = []
        
        for thresh in np.arange(threshold_range[0], threshold_range[1], step):
            signals = np.zeros(len(probs))
            signals[probs[:, 0] > thresh] = -1  # Short
            signals[probs[:, 2] > thresh] = 1   # Long
            
            # Calculate PnL
            trade_returns = np.zeros(len(signals))
            
            # Long trades
            long_mask = signals == 1
            trade_returns[long_mask] = raw_long_returns[long_mask] - spread_cost
            
            # Short trades
            short_mask = signals == -1
            trade_returns[short_mask] = -raw_long_returns[short_mask] - spread_cost
            
            total_return = np.sum(trade_returns)
            n_trades = np.sum(signals != 0)
            
            if n_trades > 0:
                wins = np.sum(trade_returns > 0)
                win_rate = wins / n_trades
                avg_trade = total_return / n_trades
            else:
                win_rate = 0
                avg_trade = 0
            
            result = {
                'threshold': thresh,
                'n_trades': int(n_trades),
                'win_rate': win_rate,
                'total_return': total_return,
                'avg_trade': avg_trade,
            }
            all_results.append(result)
            
            print(f"{thresh:.2f}{'':<6} | {n_trades:<8} | {win_rate:.1%}     | {total_return:.4f}     | {avg_trade:.6f}")
            
            if total_return > best_result['return']:
                best_result = result
        
        print("-" * 70)
        print(f"🏆 BEST: Threshold {best_result['threshold']:.2f} | Return {best_result['total_return']:.4f}")
        
        return {
            'best': best_result,
            'all_results': all_results,
        }
    
    def plot_horizon_results(
        self,
        probs: np.ndarray,
        entry_prices: np.ndarray,
        exit_prices: np.ndarray,
        threshold: float,
        horizon: int,
        spread_cost: float = 0.0001,
        save_path: Optional[str] = None,
    ):
        """Plot cumulative returns for horizon strategy."""
        # Generate signals
        signals = np.zeros(len(probs))
        signals[probs[:, 0] > threshold] = -1
        signals[probs[:, 2] > threshold] = 1
        
        # Calculate trade returns
        raw_long_returns = (exit_prices - entry_prices) / entry_prices
        trade_returns = np.zeros(len(signals))
        trade_returns[signals == 1] = raw_long_returns[signals == 1] - spread_cost
        trade_returns[signals == -1] = -raw_long_returns[signals == -1] - spread_cost
        
        # Buy and hold returns
        bh_returns = raw_long_returns
        
        # Cumulative
        cum_strategy = np.cumsum(trade_returns)
        cum_bh = np.cumsum(bh_returns)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Cumulative returns
        ax1 = axes[0, 0]
        ax1.plot(cum_bh, label='Buy & Hold', color='gray', alpha=0.7)
        ax1.plot(cum_strategy, label=f'Horizon Strategy (T={threshold:.2f})', color='blue', linewidth=2)
        ax1.set_title(f'Cumulative Return (Fixed {horizon}-Candle Hold)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Trade distribution
        ax2 = axes[0, 1]
        active_returns = trade_returns[trade_returns != 0]
        if len(active_returns) > 0:
            ax2.hist(active_returns, bins=50, color='blue', alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--')
            ax2.set_title('Trade Return Distribution')
            ax2.set_xlabel('Return per Trade')
        
        # 3. Signal timeline
        ax3 = axes[1, 0]
        colors = ['red' if s == -1 else 'green' if s == 1 else 'gray' for s in signals]
        ax3.scatter(range(len(signals)), signals, c=colors, alpha=0.5, s=1)
        ax3.set_title('Signal Timeline')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Signal')
        ax3.set_yticks([-1, 0, 1])
        ax3.set_yticklabels(['Short', 'Flat', 'Long'])
        
        # 4. Rolling win rate
        ax4 = axes[1, 1]
        window = min(50, len(trade_returns) // 10)
        if window > 1:
            winning = (trade_returns > 0).astype(float)
            active = (trade_returns != 0).astype(float)
            rolling_wins = pd.Series(winning).rolling(window).sum()
            rolling_active = pd.Series(active).rolling(window).sum()
            rolling_wr = rolling_wins / (rolling_active + 1e-10)
            ax4.plot(rolling_wr, color='blue')
            ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
            ax4.set_title(f'Rolling {window}-Trade Win Rate')
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate TCN with Fixed Horizon Strategy")
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='data/raw/eurusd_latest.csv', help='Path to data')
    parser.add_argument('--horizon', type=int, default=5, help='Holding period in candles')
    parser.add_argument('--seq-len', type=int, default=30, help='Sequence length')
    parser.add_argument('--spread', type=float, default=0.0001, help='Spread cost')
    parser.add_argument('--save-plot', type=str, default=None, help='Path to save plot')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"🔍 TCN Horizon Evaluation (H={args.horizon})")
    print("=" * 60)
    
    # Load model
    model, feature_columns, checkpoint = load_checkpoint_with_features(args.model)
    
    print(f"\n📊 Model loaded with {len(feature_columns)} features")
    
    # Create evaluator
    evaluator = HorizonEvaluator(model, feature_columns)
    
    # Prepare data
    X_test, entry_prices, exit_prices = evaluator.prepare_horizon_data(
        args.data,
        seq_len=args.seq_len,
        horizon=args.horizon,
    )
    
    # Get probabilities
    probs = evaluator.get_probabilities(X_test)
    
    # Evaluate
    result = evaluator.evaluate_horizon_strategy(
        probs, entry_prices, exit_prices,
        spread_cost=args.spread,
    )
    
    best_threshold = result['best']['threshold']
    
    # Plot
    evaluator.plot_horizon_results(
        probs, entry_prices, exit_prices,
        threshold=best_threshold,
        horizon=args.horizon,
        spread_cost=args.spread,
        save_path=args.save_plot,
    )


if __name__ == "__main__":
    main()