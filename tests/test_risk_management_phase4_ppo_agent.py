"""Unit tests for `risk_management/phase4_rl_exit/ppo_agent.py`.

Test Summary:
    Provides lightweight behavioral tests for PPO components.

Test Breakdown:
    - PPOConfig
        - default hidden_sizes initialization
    - ActorCritic
        - forward output shapes
    - PPOAgent
        - act returns (action:int, log_prob:float, value:float)

Notes:
    These tests are skipped if `torch` is not available.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestPPOComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.mod = import_module("risk_management.phase4_rl_exit.ppo_agent")
            cls.torch = import_module("torch")
            # If tests are running with the lightweight fake torch stub, skip.
            if getattr(cls.torch, "__version__", "").startswith("0.0-fake"):
                raise unittest.SkipTest("Skipping PPO tests (fake torch stub)")
            if not hasattr(getattr(cls.torch, "nn", None), "ReLU"):
                raise unittest.SkipTest("Skipping PPO tests (torch.nn.ReLU unavailable)")
            if not hasattr(cls.torch, "zeros"):
                raise unittest.SkipTest("Skipping PPO tests (torch.zeros unavailable)")
        except Exception as e:
            raise unittest.SkipTest(f"Skipping PPO tests (missing torch/ppo deps): {e}")

    def test_ppoconfig_default_hidden_sizes(self):
        PPOConfig = self.mod.PPOConfig
        cfg = PPOConfig(hidden_sizes=None)
        self.assertEqual(cfg.hidden_sizes, [128, 64])

    def test_actorcritic_forward_shapes(self):
        ActorCritic = self.mod.ActorCritic
        net = ActorCritic(obs_dim=10, action_dim=6)
        x = self.torch.zeros((4, 10))
        logits, value = net(x)
        self.assertEqual(tuple(logits.shape), (4, 6))
        self.assertEqual(tuple(value.shape), (4,))

    def test_agent_act_returns_types(self):
        PPOAgent = self.mod.PPOAgent
        agent = PPOAgent(obs_dim=10, action_dim=6, device="cpu")
        state = np.zeros(10, dtype=np.float32)
        action, log_prob, value = agent.act(state)
        self.assertIsInstance(action, int)
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)


if __name__ == "__main__":
    unittest.main()
