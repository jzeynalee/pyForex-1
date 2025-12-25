# risk_management/phase4_rl_exit/environment.py
"""
Phase 4: RL Exit Optimization Environment

Gym-compatible environment for training an exit timing agent.
The agent learns WHEN to exit, not position sizing or entry decisions.

State space:
    - Position info (direction, entry price, current price, unrealized PnL)
    - Time info (holding time, time to SL/TP, market session)
    - Market features (volatility, trend strength, momentum)
    - Risk info (distance to SL, distance to TP, risk-reward position)

Action space:
    - HOLD: Keep position open
    - EXIT: Close entire position at market
    - TRAIL_STOP: Tighten stop loss
    - PARTIAL_CLOSE: Close portion of position (25%, 50%, 75%)

Reward:
    - Risk-adjusted returns with penalties for:
        - Premature exits (leaving profit on table)
        - Late exits (giving back gains)
        - Hitting stop loss
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
import logging
from collections import deque

logger = logging.getLogger(__name__)


class ExitAction(IntEnum):
    """Exit action space."""
    HOLD = 0
    EXIT = 1
    TRAIL_STOP = 2
    PARTIAL_25 = 3
    PARTIAL_50 = 4
    PARTIAL_75 = 5
    PARTIAL_CLOSE = PARTIAL_50
    TIGHTEN_STOP = TRAIL_STOP


@dataclass
class RLPosition:
    """Represents an open trading position."""
    direction: int          # 1 = long, -1 = short
    entry_price: float
    entry_time: int         # Step index when opened
    initial_size: float     # Original position size
    current_size: float     # Current size (may be reduced by partials)
    stop_loss: float
    take_profit: float
    initial_sl: float       # Original SL for reference
    initial_tp: float       # Original TP for reference
    
    @property
    def is_long(self) -> bool:
        return self.direction == 1
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L in price units."""
        return (current_price - self.entry_price) * self.direction
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L as percentage."""
        return self.unrealized_pnl(current_price) / self.entry_price * 100
    
    def risk_reward_position(self, current_price: float) -> float:
        """Calculate position within SL-TP range (-1 at SL, +1 at TP)."""
        if self.is_long:
            sl_distance = self.entry_price - self.stop_loss
            tp_distance = self.take_profit - self.entry_price
        else:
            sl_distance = self.stop_loss - self.entry_price
            tp_distance = self.entry_price - self.take_profit
        
        pnl = self.unrealized_pnl(current_price)
        
        if pnl >= 0:
            return pnl / tp_distance if tp_distance > 0 else 0
        else:
            return pnl / sl_distance if sl_distance > 0 else 0


class Position:
    def __init__(
        self,
        direction: int = 1,
        entry_price: float = 0.0,
        entry_time: Union[int, datetime, None] = 0,
        initial_size: Optional[float] = None,
        current_size: Optional[float] = None,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        initial_sl: Optional[float] = None,
        initial_tp: Optional[float] = None,
        ticket: Optional[str] = None,
        symbol: Optional[str] = None,
        volume: float = 0.0,
        current_price: float = 0.0,
        unrealized_pnl: float = 0.0,
        **_kwargs,
    ):
        self.ticket = ticket
        self.symbol = symbol

        self.direction = int(direction)
        self.entry_price = float(entry_price)
        self.entry_time = entry_time
        self.volume = float(volume)
        self.stop_loss = float(stop_loss)
        self.take_profit = float(take_profit)

        if initial_size is None or float(initial_size) == 0.0:
            initial_size = float(self.volume) if self.volume else 1.0
        if current_size is None or float(current_size) == 0.0:
            current_size = float(self.volume) if self.volume else float(initial_size)
        if initial_sl is None or float(initial_sl) == 0.0:
            initial_sl = float(self.stop_loss)
        if initial_tp is None or float(initial_tp) == 0.0:
            initial_tp = float(self.take_profit)

        self.initial_size = float(initial_size)
        self.current_size = float(current_size)
        self.initial_sl = float(initial_sl)
        self.initial_tp = float(initial_tp)

        self.current_price = float(current_price)
        self.unrealized_pnl = float(unrealized_pnl)

    @property
    def is_long(self) -> bool:
        return int(self.direction) == 1

    def unrealized_pnl_pct(self, current_price: float) -> float:
        entry = float(self.entry_price) if self.entry_price else 1.0
        pnl = (float(current_price) - float(self.entry_price)) * float(self.direction)
        return pnl / entry * 100

    def risk_reward_position(self, current_price: float) -> float:
        return RLPosition(
            direction=int(self.direction),
            entry_price=float(self.entry_price),
            entry_time=0,
            initial_size=float(self.initial_size),
            current_size=float(self.current_size),
            stop_loss=float(self.stop_loss),
            take_profit=float(self.take_profit),
            initial_sl=float(self.initial_sl),
            initial_tp=float(self.initial_tp),
        ).risk_reward_position(float(current_price))


@dataclass 
class ExitEnvConfig:
    """Configuration for exit environment."""
    # Episode settings
    max_holding_steps: int = 100        # Max steps per episode
    
    # State features
    lookback_bars: int = 20             # Technical indicator lookback
    include_market_features: bool = True
    include_time_features: bool = True
    
    # Action settings
    trail_stop_atr_mult: float = 0.5    # ATR multiplier for trailing
    partial_close_fractions: List[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75]
    )
    
    # Reward settings
    risk_free_rate: float = 0.0
    sharpe_window: int = 20             # Window for Sharpe calculation
    
    # Penalties
    sl_hit_penalty: float = -0.5        # Extra penalty for hitting SL
    premature_exit_penalty: float = -0.1  # Penalty for exiting in profit zone early
    transaction_cost: float = 0.0001    # Cost per trade (as fraction)
    
    # Normalization
    normalize_state: bool = True
    price_scale: float = 10000          # For forex (1 pip = 0.0001)


class ExitTradingEnv:
    """
    Gym-compatible environment for exit timing optimization.
    
    The environment simulates holding a position and deciding when to exit.
    It does NOT handle entry decisions - those are made by the primary model.
    
    Usage:
        env = ExitTradingEnv(config, price_data)
        state = env.reset(position)
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
    """
    
    # Gym-like interface
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        config: Optional[ExitEnvConfig] = None,
        price_data: Optional[pd.DataFrame] = None
    ):
        self.config = config or ExitEnvConfig()
        self._price_data = price_data
        
        # State dimensions
        self._position_features = 8      # Position-related features
        self._market_features = 10       # Market indicators
        self._time_features = 4          # Time-related features
        
        # Calculate observation space size
        self.observation_dim = self._position_features
        if self.config.include_market_features:
            self.observation_dim += self._market_features
        if self.config.include_time_features:
            self.observation_dim += self._time_features
        
        self.action_dim = len(ExitAction)
        
        # Episode state
        self._position: Optional[Position] = None
        self._current_step = 0
        self._start_idx = 0
        self._episode_returns: List[float] = []
        self._done = False
        
        # Running statistics for normalization
        self._obs_mean = np.zeros(self.observation_dim)
        self._obs_std = np.ones(self.observation_dim)
        self._obs_count = 0
        
        logger.info(
            f"ExitTradingEnv initialized: obs_dim={self.observation_dim}, "
            f"action_dim={self.action_dim}"
        )
    
    def set_price_data(self, data: pd.DataFrame):
        """Set or update price data for the environment."""
        required_cols = ['open', 'high', 'low', 'close']
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        self._price_data = data.copy()
        self._precompute_indicators()
    
    def _precompute_indicators(self):
        """Precompute technical indicators for efficiency."""
        if self._price_data is None:
            return
        
        df = self._price_data
        
        # ATR
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.ewm(span=14, adjust=False).mean()
        
        # Returns
        df['returns'] = close.pct_change()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Momentum
        df['momentum'] = close.pct_change(10)
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Trend strength (ADX approximation)
        df['trend_strength'] = abs(df['momentum']) / (df['volatility'] + 1e-10)
        
        # Bollinger position
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        df['bb_position'] = (close - sma) / (2 * std + 1e-10)
        
        self._price_data = df.fillna(0)
    
    def reset(
        self,
        position: Optional[Position] = None,
        start_idx: Optional[int] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Reset environment for new episode.
        
        Args:
            position: Position to manage (if None, create random)
            start_idx: Starting index in price data
            seed: Random seed
        
        Returns:
            Initial observation
        """
        if seed is not None:
            np.random.seed(seed)
        
        if self._price_data is None:
            raise RuntimeError("Price data not set. Call set_price_data() first.")
        
        # Set start index
        max_start = len(self._price_data) - self.config.max_holding_steps - 1
        if start_idx is not None:
            self._start_idx = min(start_idx, max_start)
        else:
            self._start_idx = np.random.randint(
                self.config.lookback_bars,
                max(self.config.lookback_bars + 1, max_start)
            )
        
        self._current_step = 0
        self._episode_returns = []
        self._done = False
        
        # Set or create position
        if position is not None:
            self._position = position
        else:
            self._position = self._create_random_position()
        
        return self._get_observation()
    
    def _create_random_position(self) -> Position:
        """Create a random position for training."""
        idx = self._start_idx
        price = self._price_data['close'].iloc[idx]
        atr = self._price_data['atr'].iloc[idx]
        
        direction = np.random.choice([1, -1])
        
        # Random SL/TP based on ATR
        sl_mult = np.random.uniform(1.0, 2.5)
        tp_mult = np.random.uniform(1.5, 4.0)
        
        if direction == 1:  # Long
            sl = price - sl_mult * atr
            tp = price + tp_mult * atr
        else:  # Short
            sl = price + sl_mult * atr
            tp = price - tp_mult * atr
        
        return Position(
            direction=direction,
            entry_price=price,
            entry_time=0,
            initial_size=1.0,
            current_size=1.0,
            stop_loss=sl,
            take_profit=tp,
            initial_sl=sl,
            initial_tp=tp
        )
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action in environment.
        
        Args:
            action: ExitAction value
        
        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode finished. Call reset().")
        
        action = ExitAction(action)
        current_idx = self._start_idx + self._current_step
        current_price = self._price_data['close'].iloc[current_idx]
        
        # Process action
        info = {
            'action': action.name,
            'price': current_price,
            'pnl': 0.0,
            'position_closed': False,
            'close_reason': None
        }
        
        reward = 0.0
        position_closed = False
        close_reason = None
        
        if action == ExitAction.HOLD:
            # Just advance time
            pass
        
        elif action == ExitAction.EXIT:
            # Close entire position
            reward, info = self._close_position(current_price, 1.0, 'exit_action')
            position_closed = True
            close_reason = 'exit_action'
        
        elif action == ExitAction.TRAIL_STOP:
            # Tighten stop loss
            self._trail_stop(current_idx)
        
        elif action in [ExitAction.PARTIAL_25, ExitAction.PARTIAL_50, ExitAction.PARTIAL_75]:
            # Partial close
            fraction = self.config.partial_close_fractions[action.value - 3]
            if self._position.current_size > 0.1:  # Min remaining size
                partial_reward, partial_info = self._close_position(
                    current_price, fraction, f'partial_{int(fraction*100)}'
                )
                reward += partial_reward
                info.update(partial_info)
        
        # Check SL/TP hit (if position still open)
        if not position_closed and self._position.current_size > 0:
            sl_tp_result = self._check_sl_tp(current_idx)
            if sl_tp_result is not None:
                reward, info = sl_tp_result
                position_closed = True
        
        # Advance step
        self._current_step += 1
        self._episode_returns.append(reward)
        
        # Check done conditions
        self._done = (
            position_closed or
            self._current_step >= self.config.max_holding_steps or
            self._position.current_size <= 0.01
        )
        
        # Timeout penalty
        if self._done and not position_closed:
            timeout_reward, timeout_info = self._close_position(
                current_price, 1.0, 'timeout'
            )
            reward += timeout_reward
            info.update(timeout_info)
        
        obs = self._get_observation() if not self._done else np.zeros(self.observation_dim)
        
        return obs, reward, self._done, info
    
    def _close_position(
        self,
        price: float,
        fraction: float,
        reason: str
    ) -> Tuple[float, Dict]:
        """Close position (fully or partially) and calculate reward."""
        close_size = self._position.current_size * fraction
        pnl = self._position.unrealized_pnl(price) * close_size
        pnl_pct = self._position.unrealized_pnl_pct(price)
        
        # Transaction cost
        cost = close_size * price * self.config.transaction_cost
        pnl -= cost
        
        # Risk-adjusted reward
        reward = self._calculate_reward(pnl_pct, reason)
        
        # Update position
        self._position.current_size -= close_size
        
        info = {
            'position_closed': fraction >= 0.99,
            'close_reason': reason,
            'close_price': price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'close_size': close_size,
            'remaining_size': self._position.current_size
        }
        
        return reward, info
    
    def _trail_stop(self, current_idx: int):
        """Tighten stop loss based on current price."""
        price = self._price_data['close'].iloc[current_idx]
        atr = self._price_data['atr'].iloc[current_idx]
        trail_distance = atr * self.config.trail_stop_atr_mult
        
        if self._position.is_long:
            new_sl = price - trail_distance
            if new_sl > self._position.stop_loss:
                self._position.stop_loss = new_sl
        else:
            new_sl = price + trail_distance
            if new_sl < self._position.stop_loss:
                self._position.stop_loss = new_sl
    
    def _check_sl_tp(self, current_idx: int) -> Optional[Tuple[float, Dict]]:
        """Check if SL or TP was hit during this bar."""
        high = self._price_data['high'].iloc[current_idx]
        low = self._price_data['low'].iloc[current_idx]
        
        if self._position.is_long:
            # Check SL (low touches SL)
            if low <= self._position.stop_loss:
                return self._close_position(
                    self._position.stop_loss, 1.0, 'stop_loss'
                )
            # Check TP (high touches TP)
            if high >= self._position.take_profit:
                return self._close_position(
                    self._position.take_profit, 1.0, 'take_profit'
                )
        else:
            # Check SL (high touches SL)
            if high >= self._position.stop_loss:
                return self._close_position(
                    self._position.stop_loss, 1.0, 'stop_loss'
                )
            # Check TP (low touches TP)
            if low <= self._position.take_profit:
                return self._close_position(
                    self._position.take_profit, 1.0, 'take_profit'
                )
        
        return None
    
    def _calculate_reward(self, pnl_pct: float, reason: str) -> float:
        """
        Calculate risk-adjusted reward.
        
        Reward structure:
        - Base: PnL percentage
        - Bonus: Extra reward for hitting TP
        - Penalty: Extra penalty for hitting SL
        - Penalty: Small penalty for premature profitable exits
        """
        reward = pnl_pct / 100  # Normalize to [-1, 1] range roughly
        
        if reason == 'stop_loss':
            reward += self.config.sl_hit_penalty
        
        elif reason == 'take_profit':
            reward += 0.3  # Bonus for clean exit
        
        elif reason == 'exit_action' and pnl_pct > 0:
            # Check if exiting early in profit
            rr_pos = self._position.risk_reward_position(
                self._price_data['close'].iloc[self._start_idx + self._current_step]
            )
            if rr_pos < 0.5:  # Less than halfway to TP
                reward += self.config.premature_exit_penalty
        
        # Sharpe-like adjustment
        if len(self._episode_returns) >= self.config.sharpe_window:
            recent = np.array(self._episode_returns[-self.config.sharpe_window:])
            if np.std(recent) > 0:
                sharpe_adj = np.mean(recent) / (np.std(recent) + 1e-8)
                reward += 0.1 * np.clip(sharpe_adj, -1, 1)
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        current_idx = self._start_idx + self._current_step
        current_price = self._price_data['close'].iloc[current_idx]
        
        obs = []
        
        # Position features
        pos = self._position
        obs.extend([
            pos.direction,                                    # Direction
            pos.unrealized_pnl_pct(current_price) / 10,      # PnL (normalized)
            pos.risk_reward_position(current_price),          # RR position
            (current_price - pos.stop_loss) / pos.entry_price * 100,  # SL distance
            (pos.take_profit - current_price) / pos.entry_price * 100,  # TP distance
            pos.current_size / pos.initial_size,             # Remaining size
            self._current_step / self.config.max_holding_steps,  # Time in trade
            (pos.stop_loss - pos.initial_sl) / (pos.entry_price + 1e-10) * 100  # SL tightened
        ])
        
        # Market features
        if self.config.include_market_features:
            df = self._price_data
            obs.extend([
                df['returns'].iloc[current_idx] * 100,        # Current return
                df['volatility'].iloc[current_idx] * 100,     # Volatility
                df['momentum'].iloc[current_idx] * 100,       # Momentum
                df['rsi'].iloc[current_idx] / 100 - 0.5,      # RSI (centered)
                df['trend_strength'].iloc[current_idx],       # Trend strength
                df['bb_position'].iloc[current_idx],          # BB position
                df['atr'].iloc[current_idx] / current_price * 1000,  # ATR normalized
                # Recent price action
                (current_price - df['close'].iloc[current_idx-5]) / current_price * 100,
                (df['high'].iloc[current_idx-5:current_idx+1].max() - current_price) / current_price * 100,
                (current_price - df['low'].iloc[current_idx-5:current_idx+1].min()) / current_price * 100
            ])
        
        # Time features
        if self.config.include_time_features:
            # Assuming datetime index
            if hasattr(self._price_data.index, 'hour'):
                hour = self._price_data.index[current_idx].hour
                day = self._price_data.index[current_idx].dayofweek
            else:
                hour = (current_idx % 24)
                day = (current_idx // 24) % 5
            
            obs.extend([
                np.sin(2 * np.pi * hour / 24),                # Hour (cyclical)
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * day / 5),                  # Day (cyclical)
                np.cos(2 * np.pi * day / 5)
            ])
        
        obs = np.array(obs, dtype=np.float32)
        
        # Normalize
        if self.config.normalize_state:
            obs = self._normalize_obs(obs)
        
        return obs
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Running normalization of observations."""
        self._obs_count += 1
        
        # Update running statistics
        delta = obs - self._obs_mean
        self._obs_mean += delta / self._obs_count
        delta2 = obs - self._obs_mean
        self._obs_std = np.sqrt(
            (self._obs_std ** 2 * (self._obs_count - 1) + delta * delta2) / self._obs_count
        )
        
        # Normalize
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)
    
    def render(self, mode: str = 'human'):
        """Render current state."""
        if mode != 'human':
            return
        
        current_idx = self._start_idx + self._current_step
        price = self._price_data['close'].iloc[current_idx]
        pos = self._position
        
        print(f"\n=== Step {self._current_step} ===")
        print(f"Price: {price:.5f}")
        print(f"Position: {'LONG' if pos.is_long else 'SHORT'}")
        print(f"Entry: {pos.entry_price:.5f}")
        print(f"SL: {pos.stop_loss:.5f} | TP: {pos.take_profit:.5f}")
        print(f"PnL: {pos.unrealized_pnl_pct(price):.2f}%")
        print(f"Size: {pos.current_size:.2f}")
    
    @property
    def observation_space(self) -> Dict:
        """Return observation space info."""
        return {
            'shape': (self.observation_dim,),
            'dtype': np.float32,
            'low': -np.inf,
            'high': np.inf
        }
    
    @property
    def action_space(self) -> Dict:
        """Return action space info."""
        return {
            'n': self.action_dim,
            'dtype': np.int64
        }


def create_exit_env(
    price_data: pd.DataFrame,
    config: Optional[ExitEnvConfig] = None
) -> ExitTradingEnv:
    """Factory function to create exit environment."""
    env = ExitTradingEnv(config)
    env.set_price_data(price_data)
    return env
