# risk_management/phase4_rl_exit/trainer.py
"""
Phase 4: Training Utilities for Exit Optimizer

Provides:
- Parallel environment training
- Curriculum learning
- Evaluation utilities
- Integration with live trading
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import json

from .environment import (
    ExitTradingEnv, ExitEnvConfig, ExitAction, Position, create_exit_env
)
from .ppo_agent import PPOAgent, PPOConfig, ExitOptimizer

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for exit optimizer training."""
    # Training parameters
    total_timesteps: int = 500_000
    n_envs: int = 4                     # Parallel environments
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: int = 3
    
    # Evaluation
    eval_freq: int = 10_000
    eval_episodes: int = 50
    
    # Checkpointing
    save_freq: int = 50_000
    checkpoint_dir: str = 'checkpoints/exit_optimizer'
    
    # Logging
    log_freq: int = 1000
    tensorboard: bool = False
    
    # Early stopping
    patience: int = 10                  # Eval rounds without improvement
    min_improvement: float = 0.01


class CurriculumScheduler:
    """
    Curriculum learning for exit optimization.
    
    Stages:
    1. Easy: Wide SL/TP, trending markets
    2. Medium: Normal SL/TP, mixed conditions  
    3. Hard: Tight SL/TP, choppy markets
    """
    
    def __init__(
        self,
        n_stages: int = 3,
        progress_threshold: float = 0.6
    ):
        self.n_stages = n_stages
        self.progress_threshold = progress_threshold
        self.current_stage = 0
        self.stage_performance: List[List[float]] = [[] for _ in range(n_stages)]
    
    def get_env_params(self) -> Dict:
        """Get environment parameters for current stage."""
        if self.current_stage == 0:
            # Easy: Wide stops, long time
            return {
                'sl_multiplier': (2.0, 3.5),
                'tp_multiplier': (3.0, 5.0),
                'max_holding_steps': 150
            }
        elif self.current_stage == 1:
            # Medium: Normal conditions
            return {
                'sl_multiplier': (1.5, 2.5),
                'tp_multiplier': (2.0, 4.0),
                'max_holding_steps': 100
            }
        else:
            # Hard: Tight stops
            return {
                'sl_multiplier': (1.0, 2.0),
                'tp_multiplier': (1.5, 3.0),
                'max_holding_steps': 75
            }
    
    def update(self, episode_reward: float) -> bool:
        """
        Update curriculum based on performance.
        
        Returns:
            True if stage advanced
        """
        self.stage_performance[self.current_stage].append(episode_reward)
        
        # Check for stage advancement
        if len(self.stage_performance[self.current_stage]) >= 100:
            recent = self.stage_performance[self.current_stage][-100:]
            win_rate = sum(1 for r in recent if r > 0) / len(recent)
            
            if win_rate >= self.progress_threshold and self.current_stage < self.n_stages - 1:
                logger.info(
                    f"Advancing curriculum: Stage {self.current_stage} -> {self.current_stage + 1} "
                    f"(win_rate={win_rate:.2%})"
                )
                self.current_stage += 1
                return True
        
        return False


class ExitOptimizerTrainer:
    """
    Full training pipeline for exit optimizer.
    
    Features:
    - Multi-environment training
    - Curriculum learning
    - Periodic evaluation
    - Checkpointing
    - Early stopping
    
    Usage:
        trainer = ExitOptimizerTrainer(config)
        trainer.train(train_data, eval_data)
    """
    
    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        env_config: Optional[ExitEnvConfig] = None,
        agent_config: Optional[PPOConfig] = None
    ):
        self.config = config or TrainingConfig()
        self.env_config = env_config or ExitEnvConfig()
        self.agent_config = agent_config or PPOConfig()
        
        # Will be initialized in train()
        self.envs: List[ExitTradingEnv] = []
        self.agent: Optional[PPOAgent] = None
        self.curriculum: Optional[CurriculumScheduler] = None
        
        # Tracking
        self.training_history: Dict[str, List] = {
            'timesteps': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'eval_rewards': [],
            'policy_loss': [],
            'value_loss': []
        }
        
        self.best_eval_reward = float('-inf')
        self.patience_counter = 0
    
    def train(
        self,
        train_data: pd.DataFrame,
        eval_data: Optional[pd.DataFrame] = None,
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        Train the exit optimizer.
        
        Args:
            train_data: Training price data
            eval_data: Evaluation price data (optional)
            callback: Optional callback(timestep, metrics)
        
        Returns:
            Training history
        """
        # Setup
        self._setup_training(train_data)
        
        if eval_data is not None:
            self.eval_env = create_exit_env(eval_data, self.env_config)
        else:
            self.eval_env = None
        
        # Training loop
        timestep = 0
        episode_count = 0
        
        states = [env.reset() for env in self.envs]
        episode_rewards = [0.0] * self.config.n_envs
        episode_lengths = [0] * self.config.n_envs
        
        logger.info(f"Starting training for {self.config.total_timesteps} timesteps")
        
        while timestep < self.config.total_timesteps:
            # Collect experience from all environments
            for i, (env, state) in enumerate(zip(self.envs, states)):
                action, log_prob, value = self.agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                self.agent.store(state, action, reward, value, log_prob, done)
                
                episode_rewards[i] += reward
                episode_lengths[i] += 1
                states[i] = next_state
                timestep += 1
                
                if done:
                    self.training_history['episode_rewards'].append(episode_rewards[i])
                    self.training_history['episode_lengths'].append(episode_lengths[i])
                    
                    # Curriculum update
                    if self.curriculum:
                        self.curriculum.update(episode_rewards[i])
                    
                    # Reset
                    states[i] = env.reset()
                    episode_rewards[i] = 0.0
                    episode_lengths[i] = 0
                    episode_count += 1
            
            # Update policy
            if len(self.agent.buffer) >= self.agent_config.n_steps:
                metrics = self.agent.update()
                if metrics:
                    self.training_history['policy_loss'].append(metrics['policy_loss'])
                    self.training_history['value_loss'].append(metrics['value_loss'])
                    self.training_history['timesteps'].append(timestep)
            
            # Logging
            if timestep % self.config.log_freq == 0:
                self._log_progress(timestep, episode_count)
            
            # Evaluation
            if self.eval_env and timestep % self.config.eval_freq == 0:
                eval_reward = self._evaluate()
                self.training_history['eval_rewards'].append(eval_reward)
                
                # Check for improvement
                if eval_reward > self.best_eval_reward + self.config.min_improvement:
                    self.best_eval_reward = eval_reward
                    self.patience_counter = 0
                    self._save_checkpoint(timestep, is_best=True)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at timestep {timestep}")
                    break
            
            # Checkpointing
            if timestep % self.config.save_freq == 0:
                self._save_checkpoint(timestep)
            
            # Callback
            if callback:
                callback(timestep, self.training_history)
        
        # Final save
        self._save_checkpoint(timestep, is_final=True)
        
        return self.training_history
    
    def _setup_training(self, train_data: pd.DataFrame):
        """Initialize training components."""
        # Create environments
        self.envs = [
            create_exit_env(train_data, self.env_config)
            for _ in range(self.config.n_envs)
        ]
        
        # Create agent
        self.agent = PPOAgent(
            obs_dim=self.envs[0].observation_dim,
            action_dim=self.envs[0].action_dim,
            config=self.agent_config
        )
        
        # Curriculum
        if self.config.use_curriculum:
            self.curriculum = CurriculumScheduler(
                n_stages=self.config.curriculum_stages
            )
        
        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _log_progress(self, timestep: int, episode_count: int):
        """Log training progress."""
        recent_rewards = self.training_history['episode_rewards'][-100:]
        recent_lengths = self.training_history['episode_lengths'][-100:]
        
        if recent_rewards:
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            win_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
            
            logger.info(
                f"Timestep {timestep:,} | Episodes: {episode_count} | "
                f"Avg Reward: {avg_reward:.3f} | Win Rate: {win_rate:.1%} | "
                f"Avg Length: {avg_length:.1f}"
            )
    
    def _evaluate(self) -> float:
        """Run evaluation episodes."""
        if self.eval_env is None:
            return 0.0
        
        rewards = []
        
        for _ in range(self.config.eval_episodes):
            state = self.eval_env.reset()
            episode_reward = 0
            
            while True:
                action, _, _ = self.agent.act(state, deterministic=True)
                next_state, reward, done, _ = self.eval_env.step(action)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            rewards.append(episode_reward)
        
        avg_reward = np.mean(rewards)
        win_rate = sum(1 for r in rewards if r > 0) / len(rewards)
        
        logger.info(
            f"Evaluation: Avg Reward = {avg_reward:.3f}, Win Rate = {win_rate:.1%}"
        )
        
        return avg_reward
    
    def _save_checkpoint(
        self,
        timestep: int,
        is_best: bool = False,
        is_final: bool = False
    ):
        """Save training checkpoint."""
        if is_best:
            path = Path(self.config.checkpoint_dir) / 'best_model.pt'
        elif is_final:
            path = Path(self.config.checkpoint_dir) / 'final_model.pt'
        else:
            path = Path(self.config.checkpoint_dir) / f'checkpoint_{timestep}.pt'
        
        self.agent.save(str(path))
        
        # Save training history
        history_path = Path(self.config.checkpoint_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy types for JSON serialization
            history = {
                k: [float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for v in vals]
                for k, vals in self.training_history.items()
            }
            json.dump(history, f)


class ExitAdvisor:
    """
    Production interface for exit recommendations.
    
    Wraps the trained model for live trading integration.
    
    Usage:
        advisor = ExitAdvisor.load('best_model.pt')
        
        # In trading loop
        recommendation = advisor.get_recommendation(
            position=position,
            market_data=df
        )
        
        if recommendation['action'] == 'EXIT':
            close_position()
    """
    
    def __init__(
        self,
        agent: PPOAgent,
        env_config: Optional[ExitEnvConfig] = None
    ):
        self.agent = agent
        self.env_config = env_config or ExitEnvConfig()
        
        # For state construction
        self._temp_env: Optional[ExitTradingEnv] = None
    
    @classmethod
    def load(
        cls,
        model_path: str,
        env_config: Optional[ExitEnvConfig] = None,
        device: str = 'auto'
    ) -> 'ExitAdvisor':
        """Load advisor from saved model."""
        env_config = env_config or ExitEnvConfig()
        
        # Calculate dimensions
        temp_env = ExitTradingEnv(env_config)
        
        agent = PPOAgent.from_checkpoint(
            model_path,
            obs_dim=temp_env.observation_dim,
            action_dim=temp_env.action_dim,
            device=device
        )
        
        return cls(agent, env_config)
    
    def get_recommendation(
        self,
        position: Position,
        market_data: pd.DataFrame,
        deterministic: bool = True
    ) -> Dict:
        """
        Get exit recommendation for current position.
        
        Args:
            position: Current trading position
            market_data: Recent market data (OHLCV)
            deterministic: Use greedy action selection
        
        Returns:
            Dict with 'action', 'action_name', 'confidence', 'value'
        """
        # Create temporary environment for state construction
        if self._temp_env is None:
            self._temp_env = ExitTradingEnv(self.env_config)
        
        self._temp_env.set_price_data(market_data)
        
        # Reset with the current position
        state = self._temp_env.reset(
            position=position,
            start_idx=len(market_data) - 1
        )
        
        # Get recommendation
        action, log_prob, value = self.agent.act(state, deterministic)
        
        return {
            'action': action,
            'action_name': ExitAction(action).name,
            'confidence': np.exp(log_prob),
            'value': value,
            'recommended': action != ExitAction.HOLD
        }
    
    def should_exit(
        self,
        position: Position,
        market_data: pd.DataFrame,
        confidence_threshold: float = 0.5
    ) -> Tuple[bool, str]:
        """
        Simple interface: should we exit?
        
        Returns:
            (should_exit: bool, reason: str)
        """
        rec = self.get_recommendation(position, market_data)
        
        if rec['action'] == ExitAction.EXIT and rec['confidence'] >= confidence_threshold:
            return True, f"RL recommends EXIT (confidence={rec['confidence']:.2%})"
        
        if rec['action'] in [ExitAction.PARTIAL_25, ExitAction.PARTIAL_50, ExitAction.PARTIAL_75]:
            return True, f"RL recommends {rec['action_name']} (confidence={rec['confidence']:.2%})"
        
        return False, f"RL recommends HOLD (confidence={rec['confidence']:.2%})"


def train_exit_optimizer(
    train_data: pd.DataFrame,
    eval_data: Optional[pd.DataFrame] = None,
    total_timesteps: int = 500_000,
    checkpoint_dir: str = 'checkpoints/exit_optimizer',
    **kwargs
) -> Tuple[ExitAdvisor, Dict]:
    """
    Convenience function to train exit optimizer.
    
    Args:
        train_data: Training price data
        eval_data: Evaluation price data
        total_timesteps: Total training timesteps
        checkpoint_dir: Directory for checkpoints
        **kwargs: Additional config options
    
    Returns:
        (trained_advisor, training_history)
    """
    config = TrainingConfig(
        total_timesteps=total_timesteps,
        checkpoint_dir=checkpoint_dir,
        **kwargs
    )
    
    trainer = ExitOptimizerTrainer(config)
    history = trainer.train(train_data, eval_data)
    
    # Load best model
    best_path = Path(checkpoint_dir) / 'best_model.pt'
    if best_path.exists():
        advisor = ExitAdvisor.load(str(best_path))
    else:
        advisor = ExitAdvisor(trainer.agent)
    
    return advisor, history
