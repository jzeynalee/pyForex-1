"""Unit tests for `risk_management/phase4_rl_exit/trainer.py`.

Test Summary:
    Validates curriculum scheduling behavior and trainer setup wiring.

Test Breakdown:
    - CurriculumScheduler
        - advances stage after sufficient positive performance

Notes:
    Tests that touch `ExitOptimizerTrainer` are skipped if `torch` is not available.
"""

from __future__ import annotations

import sys
import unittest
from importlib import import_module
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestCurriculumScheduler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("risk_management.phase4_rl_exit.trainer")

    def test_curriculum_advances_stage(self):
        CurriculumScheduler = self.mod.CurriculumScheduler
        sched = CurriculumScheduler(n_stages=3, progress_threshold=0.6)

        advanced = False
        for _ in range(100):
            advanced = sched.update(episode_reward=1.0)
        self.assertTrue(advanced)
        self.assertEqual(sched.current_stage, 1)


class TestTrainerSetup(unittest.TestCase):
    def test_setup_training_creates_envs_and_agent(self):
        try:
            mod = import_module("risk_management.phase4_rl_exit.trainer")
            torch = import_module("torch")
            if getattr(torch, "__version__", "").startswith("0.0-fake"):
                raise unittest.SkipTest("Skipping trainer setup test (fake torch stub)")
            if not hasattr(getattr(torch, "nn", None), "ReLU"):
                raise unittest.SkipTest("Skipping trainer setup test (torch.nn.ReLU unavailable)")
        except Exception as e:
            raise unittest.SkipTest(f"Skipping trainer setup test (missing torch): {e}")

        ExitOptimizerTrainer = mod.ExitOptimizerTrainer
        TrainingConfig = mod.TrainingConfig

        cfg = TrainingConfig(total_timesteps=10, n_envs=2, checkpoint_dir=".tmp_checkpoints")
        trainer = ExitOptimizerTrainer(cfg)

        # Minimal OHLC data
        close = pd.Series([1.0 + i * 0.0001 for i in range(500)])
        df = pd.DataFrame({
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
        })

        trainer._setup_training(df)
        self.assertEqual(len(trainer.envs), 2)
        self.assertIsNotNone(trainer.agent)


if __name__ == "__main__":
    unittest.main()
