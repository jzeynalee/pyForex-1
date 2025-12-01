# trading/risk_manager.py
"""
Risk management for trading operations.
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Dict, NamedTuple
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


class TradeParams(NamedTuple):
    """Calculated trade parameters."""
    volume: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    atr: float


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_daily_loss_pct: float = 0.03     # 3% max daily loss
    risk_per_trade_pct: float = 0.01      # 1% risk per trade
    max_drawdown_pct: float = 0.10        # 10% max total drawdown
    atr_period: int = 14                   # ATR lookback period
    sl_atr_multiplier: float = 1.5         # SL = ATR * multiplier
    tp_atr_multiplier: float = 3.0         # TP = ATR * multiplier
    max_positions: int = 3                 # Max concurrent positions
    max_daily_trades: int = 10             # Max trades per day


class RiskManager:
    """
    Comprehensive risk management system.
    
    Features:
    - Daily loss limits
    - Max drawdown protection
    - ATR-based position sizing
    - Trade frequency limits
    """
    
    def __init__(
        self,
        account_balance: float,
        config: Optional[RiskConfig] = None,
    ):
        self.starting_balance = account_balance
        self.daily_start_balance = account_balance
        self.config = config or RiskConfig()
        
        # Daily tracking
        self.last_reset_day = datetime.now().day
        self.daily_trades = 0
        self.daily_pnl = 0.0
        
        # State flags
        self.trading_allowed = True
        self.halt_reason: Optional[str] = None
    
    def check_risk_limits(
        self,
        current_balance: float,
        current_equity: float,
        open_positions: int = 0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if trading is allowed based on risk limits.
        
        Args:
            current_balance: Current account balance
            current_equity: Current account equity (balance + unrealized P&L)
            open_positions: Number of currently open positions
        
        Returns:
            Tuple of (is_allowed, reason_if_not_allowed)
        """
        # Check for new day reset
        self._check_daily_reset(current_balance)
        
        # Check 1: Daily loss limit
        daily_loss_pct = (self.daily_start_balance - current_equity) / self.daily_start_balance
        if daily_loss_pct >= self.config.max_daily_loss_pct:
            reason = f"Daily loss limit ({daily_loss_pct:.2%} >= {self.config.max_daily_loss_pct:.2%})"
            logger.warning(f"[RISK] {reason}")
            return False, reason
        
        # Check 2: Total drawdown
        total_dd = (self.starting_balance - current_equity) / self.starting_balance
        if total_dd >= self.config.max_drawdown_pct:
            reason = f"Max drawdown reached ({total_dd:.2%} >= {self.config.max_drawdown_pct:.2%})"
            logger.error(f"[RISK] {reason}")
            self.trading_allowed = False
            self.halt_reason = reason
            return False, reason
        
        # Check 3: Daily trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            reason = f"Daily trade limit ({self.daily_trades} >= {self.config.max_daily_trades})"
            logger.info(f"[RISK] {reason}")
            return False, reason
        
        # Check 4: Max positions
        if open_positions >= self.config.max_positions:
            reason = f"Max positions ({open_positions} >= {self.config.max_positions})"
            return False, reason
        
        return True, None
    
    def _check_daily_reset(self, current_balance: float):
        """Reset daily counters at start of new trading day."""
        today = datetime.now().day
        if today != self.last_reset_day:
            logger.info(f"[RISK] New day - resetting counters. Previous balance: {self.daily_start_balance:.2f}")
            self.daily_start_balance = current_balance
            self.last_reset_day = today
            self.daily_trades = 0
            self.daily_pnl = 0.0
    
    def calculate_atr(
        self,
        df: pd.DataFrame,
        period: Optional[int] = None,
    ) -> float:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with 'high', 'low', 'close' columns
            period: Lookback period (uses config default if None)
        
        Returns:
            Current ATR value
        """
        period = period or self.config.atr_period
        
        if len(df) < period + 1:
            logger.warning(f"Insufficient data for ATR ({len(df)} < {period + 1})")
            return 0.0
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # True Range components
        tr1 = high[1:] - low[1:]                      # High - Low
        tr2 = np.abs(high[1:] - close[:-1])           # |High - Prev Close|
        tr3 = np.abs(low[1:] - close[:-1])            # |Low - Prev Close|
        
        true_range = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Simple moving average of TR
        atr = np.mean(true_range[-period:])
        
        return float(atr)
    
    def get_params(
        self,
        df: pd.DataFrame,
        signal: str,
        current_balance: Optional[float] = None,
        symbol_info: Optional[Dict] = None,
    ) -> TradeParams:
        """
        Calculate trade parameters (volume, SL, TP).
        
        This is the main interface for strategy classes.
        
        Args:
            df: Recent price data
            signal: 'BUY' or 'SELL'
            current_balance: Account balance (uses stored if None)
            symbol_info: MT5 symbol info for lot sizing
        
        Returns:
            TradeParams with calculated values
        """
        balance = current_balance or self.daily_start_balance
        
        # Calculate ATR
        atr = self.calculate_atr(df)
        if atr <= 0:
            # Fallback to percentage-based
            current_price = df['close'].iloc[-1]
            atr = current_price * 0.002  # 0.2% as fallback
        
        current_price = df['close'].iloc[-1]
        
        # SL/TP distances
        sl_dist = atr * self.config.sl_atr_multiplier
        tp_dist = atr * self.config.tp_atr_multiplier
        
        # Calculate price levels
        if signal.upper() == 'BUY':
            sl_price = current_price - sl_dist
            tp_price = current_price + tp_dist
        else:
            sl_price = current_price + sl_dist
            tp_price = current_price - tp_dist
        
        # Calculate position size
        risk_amount = balance * self.config.risk_per_trade_pct
        volume = self._calculate_volume(
            risk_amount=risk_amount,
            sl_distance=sl_dist,
            symbol_info=symbol_info,
        )
        
        return TradeParams(
            volume=volume,
            stop_loss=round(sl_price, 5),
            take_profit=round(tp_price, 5),
            risk_amount=risk_amount,
            atr=atr,
        )
    
    # Alias for backwards compatibility
    def get_trade_params(
        self,
        symbol: str,
        current_price: float,
        atr: float,
        direction: str,
        current_balance: float,
    ) -> Tuple[float, float, float]:
        """
        Legacy interface for trade parameter calculation.
        
        Returns:
            Tuple of (volume, sl_price, tp_price)
        """
        sl_dist = atr * self.config.sl_atr_multiplier
        tp_dist = atr * self.config.tp_atr_multiplier
        
        if direction.upper() == 'BUY':
            sl_price = current_price - sl_dist
            tp_price = current_price + tp_dist
        else:
            sl_price = current_price + sl_dist
            tp_price = current_price - tp_dist
        
        risk_amount = current_balance * self.config.risk_per_trade_pct
        
        # Get symbol info if MT5 available
        symbol_info = None
        if MT5_AVAILABLE:
            info = mt5.symbol_info(symbol)
            if info:
                symbol_info = {
                    'point': info.point,
                    'trade_tick_value': info.trade_tick_value,
                    'volume_min': info.volume_min,
                    'volume_max': info.volume_max,
                    'volume_step': info.volume_step,
                }
        
        volume = self._calculate_volume(risk_amount, sl_dist, symbol_info)
        
        return volume, sl_price, tp_price
    
    def _calculate_volume(
        self,
        risk_amount: float,
        sl_distance: float,
        symbol_info: Optional[Dict] = None,
    ) -> float:
        """
        Calculate position size based on risk.
        
        Formula: Volume = Risk / (SL_Points * TickValue)
        """
        if symbol_info is None:
            # Fallback: assume standard forex lot
            # 1 pip = 0.0001, 1 lot = $10/pip
            pips = sl_distance / 0.0001
            pip_value = 10.0  # $10 per pip per lot
            
            if pips * pip_value == 0:
                return 0.01
            
            volume = risk_amount / (pips * pip_value)
            return max(0.01, min(1.0, round(volume, 2)))
        
        # Use symbol info for accurate calculation
        point = symbol_info.get('point', 0.00001)
        tick_value = symbol_info.get('trade_tick_value', 1.0)
        vol_min = symbol_info.get('volume_min', 0.01)
        vol_max = symbol_info.get('volume_max', 100.0)
        vol_step = symbol_info.get('volume_step', 0.01)
        
        if tick_value == 0 or point == 0:
            return vol_min
        
        sl_points = sl_distance / point
        volume = risk_amount / (sl_points * tick_value)
        
        # Normalize to lot step
        volume = round(volume / vol_step) * vol_step
        volume = max(vol_min, min(vol_max, volume))
        
        return round(volume, 2)
    
    def record_trade(self, pnl: float = 0):
        """Record a trade execution."""
        self.daily_trades += 1
        self.daily_pnl += pnl
        logger.debug(f"[RISK] Trade recorded. Daily: {self.daily_trades}, P&L: {self.daily_pnl:.2f}")
    
    def update_balance(self, new_balance: float):
        """Update stored balance (e.g., after reconnection)."""
        self.daily_start_balance = new_balance
        logger.info(f"[RISK] Balance updated: {new_balance:.2f}")
    
    def get_status(self) -> Dict:
        """Get current risk manager status."""
        return {
            'trading_allowed': self.trading_allowed,
            'halt_reason': self.halt_reason,
            'starting_balance': self.starting_balance,
            'daily_start_balance': self.daily_start_balance,
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'config': {
                'max_daily_loss': self.config.max_daily_loss_pct,
                'risk_per_trade': self.config.risk_per_trade_pct,
                'max_drawdown': self.config.max_drawdown_pct,
            }
        }
