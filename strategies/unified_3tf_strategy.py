# strategies/unified_3tf_strategy.py
"""
Unified 3TF Strategy using MH-TCN + Alpha Factory Integration

This is the new primary strategy that replaces the fragmented pipeline with:
1. MH-TCN for predictions (direction, volatility, quantiles)
2. Alpha Factory 3TF logic for decision cascade (HTF → MTF → LTF)
3. Simplified risk management with higher thresholds

Key Improvements:
- Single unified pipeline (no fragmentation)
- Higher confidence thresholds (trade less, trade better)
- Proper 3TF cascade (HTF veto → MTF validate → LTF trigger)
- Walk-forward trained models
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class Unified3TFConfig:
    """Configuration for unified 3TF strategy."""
    # Profile
    profile: str = 'INTRADAY'
    symbol: str = 'EURUSD'
    
    # Timeframes (set based on profile)
    htf: str = 'H4'
    mtf: str = 'H1'
    ltf: str = 'M15'
    
    # Confidence thresholds - HIGH for quality trades
    min_htf_confidence: float = 0.60
    min_mtf_confidence: float = 0.65
    min_ltf_confidence: float = 0.70
    min_stability: float = 0.50
    min_directional_score: float = 0.30
    
    # Risk management - CONSERVATIVE
    base_risk_percent: float = 0.5
    min_risk_reward: float = 2.0
    max_open_trades: int = 2
    max_daily_trades: int = 5
    max_daily_loss_pct: float = 2.0
    
    # Model paths
    weights_dir: str = 'models/weights'
    
    # Sequence length for MH-TCN
    sequence_length: int = 60
    
    def __post_init__(self):
        """Set timeframes based on profile."""
        profile_tfs = {
            'SCALP': ('H1', 'M15', 'M5'),
            'INTRADAY': ('H4', 'H1', 'M15'),
            'SWING': ('D1', 'H4', 'H1'),
        }
        
        if self.profile.upper() in profile_tfs:
            self.htf, self.mtf, self.ltf = profile_tfs[self.profile.upper()]


@dataclass
class TradeSignal:
    """Trade signal from unified strategy."""
    direction: str  # 'BUY', 'SELL', or 'HOLD'
    confidence: float
    stop_loss: float
    take_profit: float
    size_multiplier: float
    
    # Metadata
    htf_bias: str
    mtf_aligned: bool
    ltf_triggered: bool
    logic_path: str
    timestamp: datetime


class Unified3TFStrategy(Strategy):
    """
    Unified 3TF Strategy using MH-TCN + Alpha Factory.
    
    This strategy implements the simplified, high-quality trading approach:
    1. HTF (Governor): Determines if we should trade and in which direction
    2. MTF (Validator): Confirms structure alignment
    3. LTF (Trigger): Provides precise entry timing
    
    All predictions come from MH-TCN with proper walk-forward training.
    """
    
    def __init__(
        self,
        config: Optional[Unified3TFConfig] = None,
        data_provider=None,
        executor=None,
        **kwargs
    ):
        self.config = config or Unified3TFConfig()
        self.data_provider = data_provider
        self.executor = executor
        self.name = f"Unified3TF_{self.config.profile}"
        
        # Initialize 3TF engine
        self._engine = None
        self._initialized = False
        
        # State tracking
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._last_trade_date = None
        self._open_positions: Dict[str, dict] = {}
        
        logger.info(f"Unified3TFStrategy created for {self.config.symbol} ({self.config.profile})")
    
    def _get_engine(self):
        """Lazy load the 3TF engine."""
        if self._engine is None:
            try:
                from alpha_factory.mhtcn_integration import UnifiedThreeTFEngine

                profile_type = str(self.config.profile or 'INTRADAY').upper()
                if profile_type == 'SCALP':
                    profile_type = 'SCALPING'

                weights_dir = str(getattr(self.config, 'weights_dir', '') or '')
                if not weights_dir or weights_dir.replace('\\', '/').lower().startswith('models/weights'):
                    try:
                        from utils.config import settings
                        weights_dir = str(getattr(settings, 'WEIGHTS_DIR', weights_dir) or weights_dir)
                    except Exception:
                        pass
                self._engine = UnifiedThreeTFEngine(
                    symbol=self.config.symbol,
                    profile_type=profile_type,
                    weights_dir=weights_dir
                )
                logger.info("UnifiedThreeTFEngine loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load UnifiedThreeTFEngine: {e}")
                self._engine = None
        return self._engine
    
    def initialize(self, starting_balance: float = 10000.0) -> bool:
        """Initialize the strategy."""
        try:
            engine = self._get_engine()
            if engine is None:
                logger.error("Could not initialize 3TF engine")
                return False
            
            self._initialized = True
            logger.info(f"Unified3TFStrategy initialized with balance: {starting_balance}")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def on_bar(self, df: pd.DataFrame) -> Optional[str]:
        """
        Process a new bar and generate trading signal.
        
        This method is called by the trading bot when a new bar closes.
        It fetches MTF data and runs the 3TF cascade.
        
        Args:
            df: Recent OHLCV data for the LTF
        
        Returns:
            'BUY', 'SELL', or None
        """
        if not self._initialized:
            if not self.initialize():
                return None
        
        # Reset daily stats if new day
        self._check_daily_reset()
        
        # Check daily limits
        if not self._check_limits():
            return None
        
        # Get current time
        current_time = None
        if 'time' in df.columns:
            try:
                current_time = pd.to_datetime(df['time'].iloc[-1]).to_pydatetime()
            except Exception:
                current_time = datetime.utcnow()
        else:
            current_time = datetime.utcnow()
        
        # Fetch MTF data
        data_htf, data_mtf, data_ltf = self._fetch_mtf_data(df)
        
        if any(d is None or len(d) < self.config.sequence_length for d in [data_htf, data_mtf, data_ltf]):
            try:
                engine = self._get_engine()
                if engine is not None:
                    engine.last_rejection_stage = "DATA"
                    engine.last_rejection_reason = (
                        f"insufficient bars htf={0 if data_htf is None else len(data_htf)} "
                        f"mtf={0 if data_mtf is None else len(data_mtf)} "
                        f"ltf={0 if data_ltf is None else len(data_ltf)} "
                        f"need>={int(self.config.sequence_length)}"
                    )
            except Exception:
                pass
            logger.debug("Insufficient MTF data")
            return None
        
        # Run 3TF evaluation
        engine = self._get_engine()
        if engine is None:
            return None
        
        try:
            instruction = engine.evaluate(
                data_htf=data_htf,
                data_mtf=data_mtf,
                data_ltf=data_ltf,
                current_time=current_time
            )
            
            if instruction is None:
                return None
            
            # Convert instruction to signal
            direction = 'BUY' if instruction.direction == 'LONG' else 'SELL'
            
            # Execute if we have an executor
            if self.executor is not None:
                self._execute_trade(instruction, data_ltf)
            
            # Track trade
            self._daily_trades += 1
            
            logger.info(
                f"Trade signal: {direction} {self.config.symbol} "
                f"(conf={instruction.confidence:.2f})"
            )
            
            return direction
            
        except Exception as e:
            logger.error(f"3TF evaluation error: {e}")
            return None
    
    def _fetch_mtf_data(self, ltf_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fetch data for all timeframes."""
        data_ltf = ltf_df
        data_mtf = None
        data_htf = None
        
        if self.data_provider is not None:
            try:
                # Fetch MTF data
                data_mtf = self.data_provider.get_ohlcv(
                    self.config.symbol,
                    timeframe=self.config.mtf,
                    count=self.config.sequence_length + 50
                )
                
                # Fetch HTF data
                data_htf = self.data_provider.get_ohlcv(
                    self.config.symbol,
                    timeframe=self.config.htf,
                    count=self.config.sequence_length + 50
                )
            except Exception as e:
                logger.warning(f"Could not fetch MTF data: {e}")
        
        # Fallback: resample from LTF if MTF data not available
        if data_mtf is None or data_mtf.empty:
            data_mtf = self._resample_data(ltf_df, self.config.mtf)
        
        if data_htf is None or data_htf.empty:
            data_htf = self._resample_data(ltf_df, self.config.htf)
        
        return data_htf, data_mtf, data_ltf
    
    def _resample_data(self, df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """Resample OHLCV data to a higher timeframe."""
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        
        if 'time' not in df.columns:
            return df
        
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        
        # Map timeframe to pandas resample rule
        tf_map = {
            'M5': '5T', 'M15': '15T', 'M30': '30T',
            'H1': '1h', 'H4': '4h', 'D1': '1D',
        }
        
        rule = tf_map.get(target_tf.upper(), '1h')
        
        try:
            resampled = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
            })
            
            if 'volume' in df.columns:
                resampled['volume'] = df['volume'].resample(rule).sum()
            elif 'tick_volume' in df.columns:
                resampled['tick_volume'] = df['tick_volume'].resample(rule).sum()
            
            resampled = resampled.dropna().reset_index()
            return resampled
            
        except Exception as e:
            logger.warning(f"Resample failed: {e}")
            return df.reset_index()
    
    def _execute_trade(self, instruction, df: pd.DataFrame):
        """Execute trade based on instruction."""
        if self.executor is None:
            return
        
        try:
            entry_price = float(df['close'].iloc[-1])
            
            # Get SL/TP from engine
            engine = self._get_engine()
            if engine is not None:
                # Get prediction for SL/TP calculation
                prediction = engine.feature_provider.predict(df, self.config.ltf)
                if prediction is not None:
                    sl, tp = engine.get_sltp_from_quantiles(
                        prediction, entry_price, instruction.direction
                    )
                else:
                    sl, tp = self._fallback_sltp(entry_price, instruction.direction)
            else:
                sl, tp = self._fallback_sltp(entry_price, instruction.direction)
            
            # Calculate position size
            volume = self._calculate_position_size(entry_price, sl, instruction.size_multiplier)
            
            # Execute
            direction = 'BUY' if instruction.direction == 'LONG' else 'SELL'
            
            if hasattr(self.executor, 'entry'):
                self.executor.entry(
                    signal=direction,
                    volume=volume,
                    sl=sl,
                    tp=tp
                )
            elif hasattr(self.executor, 'open_position'):
                self.executor.open_position(
                    direction=direction,
                    volume=volume,
                    stop_loss=sl,
                    take_profit=tp
                )
                
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
    
    def _fallback_sltp(self, entry_price: float, direction: str) -> Tuple[float, float]:
        """Fallback SL/TP calculation."""
        sl_pips = 30
        tp_pips = 60  # 2:1 R:R
        pip_value = 0.0001
        
        if direction == 'LONG':
            return entry_price - sl_pips * pip_value, entry_price + tp_pips * pip_value
        else:
            return entry_price + sl_pips * pip_value, entry_price - tp_pips * pip_value
    
    def _calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        size_multiplier: float = 1.0
    ) -> float:
        """Calculate position size based on risk."""
        # Get account balance
        balance = 10000.0
        if self.executor is not None:
            if hasattr(self.executor, 'balance'):
                balance = float(self.executor.balance)
            elif hasattr(self.executor, 'get_account_balance'):
                balance = float(self.executor.get_account_balance())
        
        # Calculate risk amount
        risk_amount = balance * (self.config.base_risk_percent / 100) * size_multiplier
        
        # Calculate SL distance in pips
        sl_distance = abs(entry_price - stop_loss)
        pip_value = 0.0001 if 'JPY' not in self.config.symbol else 0.01
        sl_pips = sl_distance / pip_value
        
        # Calculate lot size (assuming $10 per pip for 1 lot)
        pip_value_per_lot = 10.0
        if sl_pips > 0:
            lots = risk_amount / (sl_pips * pip_value_per_lot)
        else:
            lots = 0.01
        
        # Clamp to reasonable range
        lots = max(0.01, min(lots, 1.0))
        
        return round(lots, 2)
    
    def _check_limits(self) -> bool:
        """Check if trading is allowed based on limits."""
        # Check max daily trades
        if self._daily_trades >= self.config.max_daily_trades:
            logger.debug("Max daily trades reached")
            return False
        
        # Check max open trades
        if len(self._open_positions) >= self.config.max_open_trades:
            logger.debug("Max open trades reached")
            return False
        
        # Check daily loss limit
        if self.executor is not None:
            balance = 10000.0
            if hasattr(self.executor, 'balance'):
                balance = float(self.executor.balance)
            
            max_loss = balance * (self.config.max_daily_loss_pct / 100)
            if self._daily_pnl < -max_loss:
                logger.debug("Daily loss limit reached")
                return False
        
        return True
    
    def _check_daily_reset(self):
        """Reset daily stats if new day."""
        today = datetime.utcnow().date()
        
        if self._last_trade_date != today:
            self._daily_trades = 0
            self._daily_pnl = 0.0
            self._last_trade_date = today
    
    def get_stats(self) -> Dict:
        """Get strategy statistics."""
        return {
            'name': self.name,
            'profile': self.config.profile,
            'symbol': self.config.symbol,
            'daily_trades': self._daily_trades,
            'daily_pnl': self._daily_pnl,
            'open_positions': len(self._open_positions),
            'initialized': self._initialized,
        }


def create_unified_strategy(
    profile: str = 'INTRADAY',
    symbol: str = 'EURUSD',
    data_provider=None,
    executor=None,
    **kwargs
) -> Unified3TFStrategy:
    """
    Factory function to create unified 3TF strategy.
    
    Args:
        profile: Trading profile ('SCALP', 'INTRADAY', 'SWING')
        symbol: Trading symbol
        data_provider: Data provider instance
        executor: Trade executor instance
    
    Returns:
        Configured Unified3TFStrategy
    """
    config = Unified3TFConfig(
        profile=profile.upper(),
        symbol=symbol.upper(),
        **{k: v for k, v in kwargs.items() if hasattr(Unified3TFConfig, k)}
    )
    
    return Unified3TFStrategy(
        config=config,
        data_provider=data_provider,
        executor=executor
    )
