# risk_management/phase4_rl_exit/ppo_agent.py
"""
Phase 4: PPO Agent for Exit Timing Optimization

Proximal Policy Optimization (PPO) agent that learns when to exit trades.
Uses actor-critic architecture with shared feature extraction.

Features:
- Clipped surrogate objective for stable training
- Generalized Advantage Estimation (GAE)
- Value function clipping
- Entropy bonus for exploration
- Gradient clipping

The agent is specifically designed for the exit timing problem:
- Discrete action space (HOLD, EXIT, TRAIL, PARTIAL)
- Sequential decision making
- Risk-sensitive reward structure
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from collections import deque
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RolloutBuffer(NamedTuple):
    """Single rollout step data."""
    state: np.ndarray
    action: int
    reward: float
    value: float
    log_prob: float
    done: bool


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    # Network architecture
    hidden_sizes: List[int] = None  # Default: [128, 64]
    activation: str = 'relu'
    
    # PPO hyperparameters
    clip_epsilon: float = 0.2       # PPO clipping parameter
    value_clip: float = 0.2         # Value function clipping
    
    # Training
    learning_rate: float = 3e-4
    gamma: float = 0.99             # Discount factor
    gae_lambda: float = 0.95        # GAE parameter
    
    # Epochs and batches
    n_epochs: int = 10              # PPO epochs per update
    batch_size: int = 64
    n_steps: int = 2048             # Steps before update
    
    # Regularization
    entropy_coef: float = 0.01      # Entropy bonus
    value_coef: float = 0.5         # Value loss coefficient
    max_grad_norm: float = 0.5      # Gradient clipping
    
    # Target KL for early stopping
    target_kl: Optional[float] = 0.01
    
    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 64]


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Shared feature extraction followed by separate policy and value heads.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = None,
        activation: str = 'relu'
    ):
        super().__init__()
        
        hidden_sizes = hidden_sizes or [128, 64]
        
        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU
        else:
            act_fn = nn.ReLU
        
        # Shared feature extractor
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes[:-1]:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                act_fn(),
                nn.LayerNorm(hidden_size)
            ])
            prev_size = hidden_size
        
        self.shared = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            act_fn(),
            nn.Linear(hidden_sizes[-1], action_dim)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            act_fn(),
            nn.Linear(hidden_sizes[-1], 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
        
        # Smaller init for output layers
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value."""
        features = self.shared(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value.squeeze(-1)
    
    def get_action(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action from policy.
        
        Returns:
            (action, log_prob, value)
        """
        action_logits, value = self.forward(x)
        
        dist = Categorical(logits=action_logits)
        
        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            (log_probs, values, entropy)
        """
        action_logits, values = self.forward(states)
        
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy


class PPOAgent:
    """
    PPO Agent for exit timing optimization.
    
    Usage:
        agent = PPOAgent(obs_dim=22, action_dim=6)
        
        # Collect experience
        for _ in range(n_steps):
            action, log_prob, value = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.store(state, action, reward, value, log_prob, done)
            state = next_state
        
        # Update policy
        agent.update()
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[PPOConfig] = None,
        device: str = 'auto'
    ):
        self.config = config or PPOConfig()
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create actor-critic network
        self.policy = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate
        )
        
        # Experience buffer
        self.buffer: List[RolloutBuffer] = []
        
        # Tracking
        self._update_count = 0
        self._total_steps = 0
        
        logger.info(
            f"PPOAgent initialized: obs_dim={obs_dim}, action_dim={action_dim}, "
            f"device={self.device}"
        )
    
    def act(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action given state.
        
        Args:
            state: Observation from environment
            deterministic: If True, select greedy action
        
        Returns:
            (action, log_prob, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(
                state_tensor, deterministic
            )
        
        return (
            action.cpu().item(),
            log_prob.cpu().item(),
            value.cpu().item()
        )
    
    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store transition in buffer."""
        self.buffer.append(RolloutBuffer(
            state=state,
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done
        ))
        self._total_steps += 1
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using collected experience.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < self.config.batch_size:
            return {}
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae()
        
        # Convert buffer to tensors
        states = torch.FloatTensor(
            np.array([b.state for b in self.buffer])
        ).to(self.device)
        actions = torch.LongTensor(
            [b.action for b in self.buffer]
        ).to(self.device)
        old_log_probs = torch.FloatTensor(
            [b.log_prob for b in self.buffer]
        ).to(self.device)
        old_values = torch.FloatTensor(
            [b.value for b in self.buffer]
        ).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        n_updates = 0
        
        # PPO epochs
        n_samples = len(self.buffer)
        indices = np.arange(n_samples)
        
        for epoch in range(self.config.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n_samples, self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate current policy
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )
                
                # Policy loss (clipped surrogate)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                if self.config.value_clip > 0:
                    values_clipped = batch_old_values + torch.clamp(
                        values - batch_old_values,
                        -self.config.value_clip,
                        self.config.value_clip
                    )
                    value_loss1 = F.mse_loss(values, batch_returns)
                    value_loss2 = F.mse_loss(values_clipped, batch_returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                
                # Approximate KL
                with torch.no_grad():
                    kl = (batch_old_log_probs - log_probs).mean().item()
                total_kl += kl
                n_updates += 1
            
            # Early stopping on KL divergence
            if self.config.target_kl is not None:
                avg_kl = total_kl / n_updates
                if avg_kl > self.config.target_kl:
                    logger.debug(f"Early stopping at epoch {epoch+1} due to KL={avg_kl:.4f}")
                    break
        
        # Clear buffer
        self.buffer = []
        self._update_count += 1
        
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'kl_divergence': total_kl / n_updates,
            'n_updates': n_updates
        }
    
    def _compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and GAE advantages.
        
        Returns:
            (returns, advantages)
        """
        n = len(self.buffer)
        returns = np.zeros(n)
        advantages = np.zeros(n)
        
        last_gae = 0
        last_return = 0
        
        # Bootstrap from last value if episode not done
        if not self.buffer[-1].done:
            last_return = self.buffer[-1].value
            last_value = self.buffer[-1].value
        else:
            last_value = 0
        
        for t in reversed(range(n)):
            if self.buffer[t].done:
                last_gae = 0
                last_return = 0
                next_value = 0
            else:
                if t == n - 1:
                    next_value = last_value
                else:
                    next_value = self.buffer[t + 1].value
            
            delta = (
                self.buffer[t].reward +
                self.config.gamma * next_value -
                self.buffer[t].value
            )
            
            last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae
            advantages[t] = last_gae
            
            last_return = self.buffer[t].reward + self.config.gamma * last_return
            returns[t] = last_return
        
        return returns, advantages
    
    def save(self, path: str):
        """Save agent to file."""
        save_dict = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'update_count': self._update_count,
            'total_steps': self._total_steps
        }
        torch.save(save_dict, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._update_count = checkpoint.get('update_count', 0)
        self._total_steps = checkpoint.get('total_steps', 0)
        logger.info(f"Agent loaded from {path}")
    
    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        obs_dim: int,
        action_dim: int,
        device: str = 'auto'
    ) -> 'PPOAgent':
        """Load agent from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint.get('config', PPOConfig())
        
        agent = cls(obs_dim, action_dim, config, device)
        agent.load(path)
        
        return agent


class ExitOptimizer:
    """
    High-level interface for exit optimization.
    
    Wraps PPOAgent and environment for easy training and inference.
    
    Usage:
        optimizer = ExitOptimizer()
        optimizer.train(price_data, n_episodes=1000)
        
        # At inference
        action = optimizer.recommend_action(position, market_state)
    """
    
    def __init__(
        self,
        env_config: Optional['ExitEnvConfig'] = None,
        agent_config: Optional[PPOConfig] = None,
        device: str = 'auto'
    ):
        from .environment import ExitTradingEnv, ExitEnvConfig
        
        self.env_config = env_config or ExitEnvConfig()
        self.agent_config = agent_config or PPOConfig()
        
        # Create environment (data set later)
        self.env = ExitTradingEnv(self.env_config)
        
        # Create agent
        self.agent = PPOAgent(
            obs_dim=self.env.observation_dim,
            action_dim=self.env.action_dim,
            config=self.agent_config,
            device=device
        )
        
        # Training stats
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
    
    def train(
        self,
        price_data: 'pd.DataFrame',
        n_episodes: int = 1000,
        log_interval: int = 100,
        save_path: Optional[str] = None,
        save_interval: int = 500
    ) -> Dict[str, List[float]]:
        """
        Train the exit optimizer.
        
        Args:
            price_data: OHLC price data
            n_episodes: Number of training episodes
            log_interval: Episodes between logging
            save_path: Path to save checkpoints
            save_interval: Episodes between saves
        
        Returns:
            Training history
        """
        self.env.set_price_data(price_data)
        
        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_loss': [],
            'value_loss': []
        }
        
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                # Select action
                action, log_prob, value = self.agent.act(state)
                
                # Step environment
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition
                self.agent.store(state, action, reward, value, log_prob, done)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                # Update if buffer full
                if len(self.agent.buffer) >= self.agent_config.n_steps:
                    metrics = self.agent.update()
                    if metrics:
                        history['policy_loss'].append(metrics['policy_loss'])
                        history['value_loss'].append(metrics['value_loss'])
                
                if done:
                    break
            
            history['episode_rewards'].append(episode_reward)
            history['episode_lengths'].append(episode_length)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Logging
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(history['episode_rewards'][-log_interval:])
                avg_length = np.mean(history['episode_lengths'][-log_interval:])
                logger.info(
                    f"Episode {episode+1}/{n_episodes} | "
                    f"Avg Reward: {avg_reward:.3f} | "
                    f"Avg Length: {avg_length:.1f}"
                )
            
            # Save checkpoint
            if save_path and (episode + 1) % save_interval == 0:
                self.save(f"{save_path}_ep{episode+1}.pt")
        
        # Final update with remaining buffer
        if self.agent.buffer:
            self.agent.update()
        
        # Final save
        if save_path:
            self.save(f"{save_path}_final.pt")
        
        return history
    
    def recommend_action(
        self,
        state: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, str, float]:
        """
        Get exit recommendation.
        
        Args:
            state: Current state observation
            deterministic: Use greedy action selection
        
        Returns:
            (action_id, action_name, confidence)
        """
        from .environment import ExitAction
        
        action, log_prob, value = self.agent.act(state, deterministic)
        
        # Calculate confidence from log_prob
        confidence = np.exp(log_prob)
        
        return action, ExitAction(action).name, confidence
    
    def save(self, path: str):
        """Save optimizer state."""
        self.agent.save(path)
    
    def load(self, path: str):
        """Load optimizer state."""
        self.agent.load(path)
