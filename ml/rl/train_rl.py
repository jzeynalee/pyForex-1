# ml/rl/train_rl.py
"""
Training script for RL Exit Optimization Agent.

Trains a PPO agent to learn optimal exit strategies using:
- Real market data
- TCN risk model for state features
- Multiple reward functions (Sharpe, Sortino, custom)

Usage:
    # Train with default settings
    python ml/rl/train_rl.py --data data/raw/eurusd_latest.csv
    
    # Use pre-trained TCN for features
    python ml/rl/train_rl.py --data data/raw/eurusd_latest.csv --tcn-model models/weights/tcn_risk_best.pt
    
    # Custom reward function
    python ml/rl/train_rl.py --data data/raw/eurusd_latest.csv --reward sharpe
"""

import torch
import numpy as np
import pandas as pd
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.rl.environment import TradingExitEnv, EnvConfig
from ml.rl.agent import ExitAgent, PPOConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)


# =============================================================================
# Reward Functions
# =============================================================================

class RewardShaper:
    """
    Shapes rewards for RL training.
    
    Different reward functions optimize for different objectives:
    - PnL: Raw profit/loss
    - Sharpe: Risk-adjusted returns
    - Sortino: Downside risk-adjusted returns
    - Custom: Blended objectives
    """
    
    @staticmethod
    def pnl_reward(pnl: float, scale: float = 100.0) -> float:
        """Simple PnL-based reward."""
        return pnl * scale
    
    @staticmethod
    def sharpe_component(
        returns: List[float],
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Sharpe ratio component for reward.
        
        Used as a running reward component during episodes.
        """
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        excess_returns = returns - risk_free_rate
        
        if returns.std() == 0:
            return 0.0
        
        return excess_returns.mean() / returns.std()
    
    @staticmethod
    def sortino_component(
        returns: List[float],
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Sortino ratio component (penalizes downside volatility only).
        """
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        excess_returns = returns - risk_free_rate
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            downside_std = 1e-6
        else:
            downside_std = np.sqrt(np.mean(negative_returns ** 2))
        
        if downside_std == 0:
            downside_std = 1e-6
        
        return excess_returns.mean() / downside_std
    
    @staticmethod
    def drawdown_penalty(
        equity_curve: List[float],
        max_dd_threshold: float = 0.1,
    ) -> float:
        """
        Penalty for drawdown.
        
        Returns negative value if drawdown exceeds threshold.
        """
        if len(equity_curve) < 2:
            return 0.0
        
        equity = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max
        max_dd = drawdown.max()
        
        if max_dd > max_dd_threshold:
            return -(max_dd - max_dd_threshold) * 10
        
        return 0.0
    
    @staticmethod
    def time_penalty(
        time_in_trade: int,
        max_time: int = 100,
        penalty_rate: float = 0.001,
    ) -> float:
        """
        Small penalty for holding too long.
        
        Encourages decisive action.
        """
        if time_in_trade > max_time * 0.5:
            return -penalty_rate * (time_in_trade - max_time * 0.5)
        return 0.0
    
    @staticmethod
    def win_bonus(won: bool, amount: float = 0.1) -> float:
        """Bonus for winning trade."""
        return amount if won else 0.0


class CustomRewardEnv(TradingExitEnv):
    """
    Trading environment with customizable reward shaping.
    """
    
    def __init__(
        self,
        price_data: np.ndarray,
        features_data: np.ndarray,
        volatility_data: Optional[np.ndarray] = None,
        config: Optional[EnvConfig] = None,
        reward_type: str = 'sharpe',
    ):
        super().__init__(price_data, features_data, volatility_data, config)
        
        self.reward_type = reward_type
        self.episode_returns: List[float] = []
        self.episode_equity: List[float] = []
    
    def reset(self, seed=None, options=None):
        state, info = super().reset(seed, options)
        self.episode_returns = []
        self.episode_equity = [self.config.initial_balance]
        return state, info
    
    def step(self, action: int):
        state, reward, terminated, truncated, info = super().step(action)
        
        # Track returns
        if reward != 0:
            self.episode_returns.append(reward / self.config.reward_scale)
            self.episode_equity.append(self.episode_equity[-1] + reward)
        
        # Shape reward based on type
        shaped_reward = self._shape_reward(reward, terminated or truncated, info)
        
        return state, shaped_reward, terminated, truncated, info
    
    def _shape_reward(
        self, 
        base_reward: float, 
        done: bool,
        info: Dict
    ) -> float:
        """Apply reward shaping."""
        if self.reward_type == 'pnl':
            return base_reward
        
        elif self.reward_type == 'sharpe':
            # Add Sharpe component at end of episode
            if done and len(self.episode_returns) > 1:
                sharpe = RewardShaper.sharpe_component(self.episode_returns)
                return base_reward + sharpe * 10
            return base_reward
        
        elif self.reward_type == 'sortino':
            if done and len(self.episode_returns) > 1:
                sortino = RewardShaper.sortino_component(self.episode_returns)
                return base_reward + sortino * 10
            return base_reward
        
        elif self.reward_type == 'custom':
            shaped = base_reward
            
            # Time penalty
            shaped += RewardShaper.time_penalty(
                self.current_step, 
                self.config.max_steps
            )
            
            # Drawdown penalty at end
            if done:
                shaped += RewardShaper.drawdown_penalty(self.episode_equity)
                
                # Win bonus
                if base_reward > 0:
                    shaped += RewardShaper.win_bonus(True)
            
            return shaped
        
        return base_reward


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_training_data(
    data_path: str,
    tcn_checkpoint: Optional[str] = None,
    seq_len: int = 30,
) -> Dict[str, np.ndarray]:
    """
    Prepare data for RL training.
    
    Args:
        data_path: Path to CSV data
        tcn_checkpoint: Optional TCN model for feature extraction
        seq_len: Sequence length for features
    
    Returns:
        Dict with price_data, features_data, volatility_data
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower().str.strip()
    
    # OHLC data
    price_data = df[['open', 'high', 'low', 'close']].values
    
    # Features
    if tcn_checkpoint and Path(tcn_checkpoint).exists():
        logger.info(f"Extracting features using TCN: {tcn_checkpoint}")
        features_data, volatility_data = extract_tcn_features(
            df, tcn_checkpoint, seq_len
        )
    else:
        logger.info("Using basic features")
        features_data = create_basic_features(df)
        volatility_data = create_volatility_estimate(df)
    
    logger.info(f"Data prepared: {len(price_data)} samples, {features_data.shape[1]} features")
    
    return {
        'price_data': price_data.astype(np.float32),
        'features_data': features_data.astype(np.float32),
        'volatility_data': volatility_data.astype(np.float32),
    }


def extract_tcn_features(
    df: pd.DataFrame,
    checkpoint_path: str,
    seq_len: int,
) -> tuple:
    """Extract features using trained TCN model."""
    from models.tcn_risk import load_risk_model_checkpoint
    from utils.feature_adapter import FeatureEngineer
    from sklearn.preprocessing import RobustScaler
    
    # Load model
    model, feature_columns, _ = load_risk_model_checkpoint(checkpoint_path)
    model.eval()
    
    # Prepare features
    df = FeatureEngineer.add_all_features(df)
    
    # Handle missing features
    for f in feature_columns:
        if f not in df.columns:
            df[f] = 0
    
    # Scale
    scaler = RobustScaler()
    features = scaler.fit_transform(df[feature_columns].values)
    
    # Extract TCN features for each position
    n = len(features)
    tcn_features = []
    volatility = []
    
    device = next(model.parameters()).device
    
    for i in range(seq_len, n):
        seq = features[i-seq_len:i]
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(seq_tensor, return_features=True)
            tcn_features.append(outputs['features'].cpu().numpy().flatten())
            volatility.append(outputs['volatility'].cpu().numpy().item())
    
    # Pad beginning with zeros
    feature_dim = tcn_features[0].shape[0]
    padding = [np.zeros(feature_dim) for _ in range(seq_len)]
    vol_padding = [0.0] * seq_len
    
    tcn_features = padding + tcn_features
    volatility = vol_padding + volatility
    
    return np.array(tcn_features), np.array(volatility)


def create_basic_features(df: pd.DataFrame) -> np.ndarray:
    """Create basic features when TCN not available."""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    features = []
    
    # Returns
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0], returns])
    features.append(returns)
    
    # Volatility (rolling std)
    vol = pd.Series(returns).rolling(14).std().fillna(0).values
    features.append(vol)
    
    # RSI
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = pd.Series(gains).rolling(14).mean().fillna(0)
    avg_loss = pd.Series(losses).rolling(14).mean().fillna(0)
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[50], rsi.values]) / 100  # Normalize
    features.append(rsi)
    
    # ATR
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(14).mean().fillna(0).values
    atr_norm = atr / close  # Normalize
    features.append(atr_norm)
    
    # Price position relative to recent range
    high_20 = pd.Series(high).rolling(20).max().fillna(high)
    low_20 = pd.Series(low).rolling(20).min().fillna(low)
    position = (close - low_20) / (high_20 - low_20 + 1e-10)
    features.append(position.values)
    
    return np.column_stack(features)


def create_volatility_estimate(df: pd.DataFrame) -> np.ndarray:
    """Create simple volatility estimate."""
    close = df['close'].values
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0], returns])
    
    vol = pd.Series(returns).rolling(14).std().fillna(0).values
    return vol


# =============================================================================
# Training
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    total_timesteps: int = 100000
    eval_freq: int = 10000
    save_freq: int = 25000
    n_eval_episodes: int = 10
    
    # Environment
    max_episode_steps: int = 100
    reward_type: str = 'sharpe'
    
    # PPO
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    
    # Output
    save_dir: str = 'models/rl'
    name: str = 'exit_agent'


def train_exit_agent(
    data: Dict[str, np.ndarray],
    config: TrainConfig,
) -> ExitAgent:
    """
    Train RL exit agent.
    
    Args:
        data: Dict with price_data, features_data, volatility_data
        config: Training configuration
    
    Returns:
        Trained ExitAgent
    """
    # Environment config
    env_config = EnvConfig(max_steps=config.max_episode_steps)
    
    # Create environments
    logger.info("Creating training environment...")
    train_env = CustomRewardEnv(
        price_data=data['price_data'],
        features_data=data['features_data'],
        volatility_data=data['volatility_data'],
        config=env_config,
        reward_type=config.reward_type,
    )
    
    # Evaluation environment (different data split)
    n = len(data['price_data'])
    eval_start = int(n * 0.8)
    
    eval_env = CustomRewardEnv(
        price_data=data['price_data'][eval_start:],
        features_data=data['features_data'][eval_start:],
        volatility_data=data['volatility_data'][eval_start:] if data['volatility_data'] is not None else None,
        config=env_config,
        reward_type='pnl',  # Use simple PnL for evaluation
    )
    
    # PPO config
    ppo_config = PPOConfig(
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
    )
    
    # Create agent
    logger.info("Creating agent...")
    agent = ExitAgent(train_env, ppo_config, use_sb3=False)
    
    # Training
    logger.info(f"Starting training for {config.total_timesteps} steps...")
    save_path = Path(config.save_dir) / f"{config.name}.pt"
    
    training_stats = agent.train(
        total_timesteps=config.total_timesteps,
        eval_env=eval_env,
        save_path=str(save_path),
    )
    
    # Save final model
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(save_path))
    
    # Final evaluation
    logger.info("Final evaluation...")
    final_reward = agent.agent.evaluate(eval_env, n_episodes=config.n_eval_episodes)
    logger.info(f"Final evaluation reward: {final_reward:.4f}")
    
    return agent


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train RL Exit Agent")
    
    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data')
    parser.add_argument('--tcn-model', type=str, default=None,
                        help='Path to TCN model for feature extraction')
    
    # Training
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--reward', type=str, default='sharpe',
                        choices=['pnl', 'sharpe', 'sortino', 'custom'],
                        help='Reward function type')
    
    # PPO
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048, help='Steps per update')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    
    # Output
    parser.add_argument('--save-dir', type=str, default='models/rl',
                        help='Directory to save model')
    parser.add_argument('--name', type=str, default='exit_agent',
                        help='Model name')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 RL Exit Agent Training")
    print("=" * 60)
    print(f"   Data: {args.data}")
    print(f"   TCN Model: {args.tcn_model or 'None (basic features)'}")
    print(f"   Timesteps: {args.timesteps}")
    print(f"   Reward: {args.reward}")
    print("=" * 60)
    
    # Prepare data
    data = prepare_training_data(
        args.data,
        tcn_checkpoint=args.tcn_model,
    )
    
    # Config
    config = TrainConfig(
        total_timesteps=args.timesteps,
        reward_type=args.reward,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        name=args.name,
    )
    
    # Train
    agent = train_exit_agent(data, config)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"   Model saved to: {config.save_dir}/{config.name}.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()