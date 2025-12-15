"""Unit tests for `risk_management/phase4_rl_exit/environment.py`.

Test Summary:
    Validates the RL exit environment mechanics (reset/step), SL/TP checks,
    and basic observation sizing.

Test Breakdown:
    - ExitTradingEnv
        - set_price_data validates required columns
        - reset returns observation of configured dimension
        - EXIT action closes position and ends episode
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from risk_management.phase4_rl_exit.environment import (
    ExitAction,
    ExitEnvConfig,
    ExitTradingEnv,
    Position,
)


class TestExitTradingEnv(unittest.TestCase):
    def _make_price_data(self, n=200):
        close = np.linspace(1.0, 1.01, n)
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
            }
        )

    def test_set_price_data_requires_ohlc(self):
        env = ExitTradingEnv(ExitEnvConfig())
        with self.assertRaises(ValueError):
            env.set_price_data(pd.DataFrame({"close": [1.0]}))

    def test_reset_observation_dim(self):
        cfg = ExitEnvConfig(max_holding_steps=10)
        env = ExitTradingEnv(cfg)
        env.set_price_data(self._make_price_data())
        obs = env.reset(seed=0)
        self.assertEqual(obs.shape[0], env.observation_dim)

    def test_exit_action_ends_episode(self):
        cfg = ExitEnvConfig(max_holding_steps=10)
        env = ExitTradingEnv(cfg)
        df = self._make_price_data()
        env.set_price_data(df)

        pos = Position(
            direction=1,
            entry_price=float(df["close"].iloc[50]),
            entry_time=0,
            initial_size=1.0,
            current_size=1.0,
            stop_loss=float(df["close"].iloc[50]) * 0.99,
            take_profit=float(df["close"].iloc[50]) * 1.02,
            initial_sl=float(df["close"].iloc[50]) * 0.99,
            initial_tp=float(df["close"].iloc[50]) * 1.02,
        )

        env.reset(position=pos, start_idx=50)
        obs, reward, done, info = env.step(int(ExitAction.EXIT))
        self.assertTrue(done)
        self.assertTrue(info.get("position_closed") or info.get("close_reason") == "exit_action")


if __name__ == "__main__":
    unittest.main()
