#!/usr/bin/env python3
"""
Test script to verify the exit optimizer fix.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_exit_optimizer_fix():
    """Test that the exit optimizer can be created and initialized without errors."""
    
    print("Testing Exit Optimizer Fix...")
    
    try:
        from risk_management.phase4_rl_exit import (
            ExitOptimizer, PPOConfig, create_exit_env, ExitEnvConfig
        )
        from risk_management.phase4_rl_exit.trainer import ExitOptimizerTrainer, TrainingConfig
        
        print("✅ Imports successful")
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        data = pd.DataFrame({
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 102,
            'low': np.random.randn(n_samples).cumsum() + 98,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        print("✅ Sample data created")
        
        # Test configuration creation
        env_config = ExitEnvConfig(
            max_holding_steps=50,
            transaction_cost=0.0001
        )
        
        ppo_config = PPOConfig(
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01
        )
        
        train_config = TrainingConfig(
            total_timesteps=1000,  # Small for testing
            eval_freq=500,
            save_freq=1000,
            checkpoint_dir='test_checkpoints',
            device='cpu'  # Force CPU for testing
        )
        
        print("✅ Configurations created")
        
        # Test trainer creation
        trainer = ExitOptimizerTrainer(
            config=train_config,
            env_config=env_config,
            agent_config=ppo_config
        )
        
        print("✅ Trainer created successfully")
        
        # Test environment creation
        env = create_exit_env(data, env_config)
        print(f"✅ Environment created: obs_dim={env.observation_dim}, action_dim={env.action_dim}")
        
        # Test agent creation
        from risk_management.phase4_rl_exit.ppo_agent import PPOAgent
        agent = PPOAgent(
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            config=ppo_config,
            device='cpu'
        )
        
        print("✅ Agent created successfully")
        
        # Test one step
        state = env.reset()
        action, log_prob, value = agent.act(state)
        
        print(f"✅ Agent step successful: action={action}, log_prob={log_prob:.4f}, value={value:.4f}")
        
        # Test environment step
        next_state, reward, done, info = env.step(action)
        
        print(f"✅ Environment step successful: reward={reward:.4f}, done={done}")
        
        print("\n🎉 All tests passed! Exit optimizer fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_exit_optimizer_fix()
    sys.exit(0 if success else 1)
