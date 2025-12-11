# risk_management/phase4_rl_exit/__init__.py
"""
Phase 4: RL-Based Exit Optimization

Reinforcement learning system for optimizing trade exit timing.
Uses PPO (Proximal Policy Optimization) with a focused action space.

Components:
- ExitTradingEnv: Gym-compatible environment for exit decisions
- PPOAgent: Actor-critic agent with clipped surrogate objective
- ExitOptimizer: High-level training interface
- ExitAdvisor: Production inference interface

The RL agent is narrow-scoped to exit timing ONLY:
- NOT responsible for entry decisions
- NOT responsible for position sizing
- NOT responsible for SL/TP levels (those come from Phase 2)

Action Space:
- HOLD: Keep position open
- EXIT: Close entire position
- TRAIL_STOP: Tighten stop loss
- PARTIAL_25/50/75: Close partial position

State Space:
- Position info (direction, PnL, time in trade)
- Market features (volatility, momentum, trend)
- Risk info (distance to SL/TP)

Usage:
    # Training
    from risk_management.phase4_rl_exit import train_exit_optimizer
    
    advisor, history = train_exit_optimizer(
        train_data=price_df,
        eval_data=eval_df,
        total_timesteps=500_000
    )
    
    # Inference
    from risk_management.phase4_rl_exit import ExitAdvisor
    
    advisor = ExitAdvisor.load('best_model.pt')
    recommendation = advisor.get_recommendation(position, market_data)
    
    if recommendation['action_name'] == 'EXIT':
        close_position()
"""

from .environment import (
    ExitTradingEnv,
    ExitEnvConfig,
    ExitAction,
    Position,
    create_exit_env
)

from .ppo_agent import (
    PPOAgent,
    PPOConfig,
    ActorCritic,
    ExitOptimizer
)

from .trainer import (
    ExitOptimizerTrainer,
    TrainingConfig,
    CurriculumScheduler,
    ExitAdvisor,
    train_exit_optimizer
)

__all__ = [
    # Environment
    'ExitTradingEnv',
    'ExitEnvConfig',
    'ExitAction',
    'Position',
    'create_exit_env',
    # Agent
    'PPOAgent',
    'PPOConfig',
    'ActorCritic',
    'ExitOptimizer',
    # Training
    'ExitOptimizerTrainer',
    'TrainingConfig',
    'CurriculumScheduler',
    # Inference
    'ExitAdvisor',
    'train_exit_optimizer'
]
