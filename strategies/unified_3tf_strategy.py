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

from alpha_factory.features_engineering import FeatureEngineerOptimized
from alpha_factory.market_data import MarketData
from alpha_factory.probabilistic_alpha_factory import create_probabilistic_alpha_factory

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

    relaxed_alignment: bool = False
    
    # Risk management - CONSERVATIVE
    base_risk_percent: float = 0.5
    min_risk_reward: float = 2.0
    max_open_trades: int = 2
    max_daily_trades: int = 5
    max_daily_loss_pct: float = 2.0
    min_sl_pips: float = 8.0
    max_lot: float = 1.0
    
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

        if self.profile.upper() == 'SCALP':
            self.max_lot = 0.3
            self.relaxed_alignment = True
            self.min_htf_confidence = 0.52
            self.min_mtf_confidence = 0.55
            self.min_ltf_confidence = 0.58
            self.min_stability = 0.40


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
        
        # Authoritative decision engine (probabilistic alpha factory)
        self._engine = None
        self._feature_engineer: Optional[FeatureEngineerOptimized] = None
        self._initialized = False

        # Backtest diagnostics (kept for compatibility with main.py backtest reporting)
        self.last_rejection_stage: str = ""
        self.last_rejection_reason: str = ""
        
        # State tracking
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._last_trade_date = None
        self._open_positions: Dict[str, dict] = {}
        self._last_entry_time: Optional[datetime] = None
        
        logger.info(f"Unified3TFStrategy created for {self.config.symbol} ({self.config.profile})")

    def _sync_open_positions(self):
        """Sync internal open-position view from executor (backtest/live)."""
        self._open_positions = {}
        if self.executor is None:
            return
        try:
            if hasattr(self.executor, 'get_open_positions'):
                try:
                    pos = self.executor.get_open_positions(symbol=getattr(self.config, 'symbol', None))
                except TypeError:
                    pos = self.executor.get_open_positions()
            elif hasattr(self.executor, 'positions'):
                pos = list(getattr(self.executor, 'positions') or [])
            else:
                pos = []
        except Exception:
            pos = []
        try:
            for p in (pos or []):
                if isinstance(p, dict):
                    ticket = p.get('ticket')
                    if ticket is None:
                        continue
                    self._open_positions[str(ticket)] = p
                else:
                    ticket = getattr(p, 'ticket', None)
                    if ticket is None:
                        continue
                    self._open_positions[str(ticket)] = {
                        'ticket': ticket,
                        'type': getattr(p, 'direction', None),
                        'volume': getattr(p, 'volume', None),
                        'price_open': getattr(p, 'entry_price', None),
                        'sl': getattr(p, 'sl', None),
                        'tp': getattr(p, 'tp', None),
                        'entry_time': getattr(p, 'entry_time', None),
                    }
        except Exception:
            self._open_positions = {}

    def _has_open_direction(self, direction: str) -> bool:
        d = str(direction or '').upper()
        for p in (self._open_positions or {}).values():
            try:
                pdir = str(p.get('type') or p.get('direction') or '').upper()
                if pdir == d:
                    return True
            except Exception:
                continue
        return False
    
    def _get_engine(self):
        """Lazy load the authoritative decision engine."""
        if self._engine is None:
            weights_dir = str(getattr(self.config, 'weights_dir', '') or '')
            if not weights_dir or weights_dir.replace('\\', '/').lower().startswith('models/weights'):
                try:
                    from utils.config import settings
                    weights_dir = str(getattr(settings, 'WEIGHTS_DIR', weights_dir) or weights_dir)
                except Exception:
                    pass

            self._engine = create_probabilistic_alpha_factory(
                mhtcn_weights_dir=weights_dir,
                profile=str(self.config.profile or 'INTRADAY').upper(),
            )
            logger.info("ProbabilisticAlphaFactory loaded successfully")
        return self._engine
    
    def initialize(self, starting_balance: float = 10000.0) -> bool:
        """Initialize the strategy."""
        try:
            engine = self._get_engine()
            if engine is None:
                logger.error("Could not initialize 3TF engine")
                return False

            self._feature_engineer = FeatureEngineerOptimized()
            
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

        self.last_rejection_stage = ""
        self.last_rejection_reason = ""
        
        # Get current time
        current_time = None
        if df is not None and not df.empty and 'time' in df.columns:
            try:
                current_time = pd.to_datetime(df['time'].iloc[-1]).to_pydatetime()
            except Exception:
                current_time = datetime.utcnow()
        else:
            current_time = datetime.utcnow()

        # Reset daily stats if new day (based on bar time)
        self._check_daily_reset(current_time)

        # Sync open positions from executor before enforcing limits
        self._sync_open_positions()
        
        # Check daily limits
        if not self._check_limits():
            return None
        
        # Fetch MTF data
        data_htf, data_mtf, data_ltf = self._fetch_mtf_data(df)
        
        if any(d is None or len(d) < self.config.sequence_length for d in [data_htf, data_mtf, data_ltf]):
            try:
                engine = self._get_engine()
                if engine is not None:
                    self.last_rejection_stage = "DATA"
                    self.last_rejection_reason = (
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
            balance = 10000.0
            if self.executor is not None:
                try:
                    if hasattr(self.executor, 'balance'):
                        balance = float(self.executor.balance)
                    elif hasattr(self.executor, 'get_account_balance'):
                        balance = float(self.executor.get_account_balance())
                except Exception:
                    balance = 10000.0

            htf_out = self._evaluate_timeframe(data_htf, timeframe=self.config.htf, equity=balance, signal_id="htf")
            if not self._passes_gate(htf_out, min_conf=self.config.min_htf_confidence, min_stability=self.config.min_stability):
                if not bool(getattr(self.config, 'relaxed_alignment', False)):
                    self.last_rejection_stage = "HTF"
                    self.last_rejection_reason = "htf gate failed"
                    return None

            mtf_out = self._evaluate_timeframe(data_mtf, timeframe=self.config.mtf, equity=balance, signal_id="mtf")
            if not self._passes_gate(mtf_out, min_conf=self.config.min_mtf_confidence, min_stability=self.config.min_stability):
                self.last_rejection_stage = "MTF"
                self.last_rejection_reason = "mtf gate failed"
                return None

            if str(getattr(mtf_out, 'direction', 'HOLD')) == 'HOLD':
                self.last_rejection_stage = "MTF"
                self.last_rejection_reason = "mtf hold"
                return None

            if not bool(getattr(self.config, 'relaxed_alignment', False)):
                if str(mtf_out.direction) != str(htf_out.direction):
                    self.last_rejection_stage = "MTF"
                    self.last_rejection_reason = f"mtf dir {mtf_out.direction} != htf dir {htf_out.direction}"
                    return None

            ltf_out = self._evaluate_timeframe(data_ltf, timeframe=self.config.ltf, equity=balance, signal_id="ltf")
            if not self._passes_gate(ltf_out, min_conf=self.config.min_ltf_confidence, min_stability=self.config.min_stability):
                self.last_rejection_stage = "LTF"
                self.last_rejection_reason = "ltf gate failed"
                return None

            if ltf_out.direction == 'HOLD':
                self.last_rejection_stage = "LTF"
                self.last_rejection_reason = "ltf hold"
                return None

            if bool(getattr(self.config, 'relaxed_alignment', False)):
                if str(ltf_out.direction) != str(mtf_out.direction):
                    self.last_rejection_stage = "MTF"
                    self.last_rejection_reason = f"mtf dir {mtf_out.direction} != ltf dir {ltf_out.direction}"
                    return None
            else:
                if str(ltf_out.direction) != str(htf_out.direction):
                    self.last_rejection_stage = "LTF"
                    self.last_rejection_reason = f"ltf dir {ltf_out.direction} != htf dir {htf_out.direction}"
                    return None

            direction = 'BUY' if ltf_out.direction == 'LONG' else 'SELL'

            # Enforce position limits using actual open positions (avoid stacking entries)
            self._sync_open_positions()
            if len(self._open_positions) >= int(self.config.max_open_trades or 0):
                self.last_rejection_stage = "LIMIT"
                self.last_rejection_reason = "max open trades"
                return None
            if self._has_open_direction(direction):
                self.last_rejection_stage = "LIMIT"
                self.last_rejection_reason = f"{direction} already open"
                return None

            if self.executor is not None:
                self._execute_trade_probabilistic(direction, ltf_out, data_ltf)

            self._last_entry_time = current_time
            self._sync_open_positions()

            self._daily_trades += 1

            logger.info(
                f"Trade signal: {direction} {self.config.symbol} "
                f"(conf={ltf_out.confidence:.2f})"
            )

            return direction

        except Exception as e:
            logger.error(f"Probabilistic 3TF evaluation error: {e}")
            self.last_rejection_stage = "ERROR"
            self.last_rejection_reason = str(e)[:240]
            return None

    def _evaluate_timeframe(self, df: pd.DataFrame, timeframe: str, equity: float, signal_id: str):
        if df is None or df.empty:
            raise ValueError("empty dataframe")
        if self._feature_engineer is None:
            self._feature_engineer = FeatureEngineerOptimized()

        d0 = df.copy()
        d0.columns = [str(c).lower().strip() for c in d0.columns]
        if 'volume' not in d0.columns:
            if 'tick_volume' in d0.columns:
                d0['volume'] = d0['tick_volume']
            else:
                d0['volume'] = 0.0

        feats = self._feature_engineer.generate_features(d0, batch_processing=False)

        swing_points = None
        try:
            if feats is not None and not feats.empty and 'swing_high' in feats.columns and 'swing_low' in feats.columns:
                from alpha_factory.market_data import SwingPoint

                base = d0.reset_index(drop=True)
                f0 = feats.reset_index(drop=True)

                if 'time' in base.columns:
                    times = pd.to_datetime(base['time'], errors='coerce')
                else:
                    times = pd.Series([pd.NaT] * len(base))

                sh = f0['swing_high'].fillna(0).astype(int)
                sl = f0['swing_low'].fillna(0).astype(int)

                swing_points = []
                for i in range(len(base)):
                    if int(sh.iloc[i]) == 1:
                        t = times.iloc[i]
                        swing_points.append(SwingPoint(
                            index=int(i),
                            time=t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t,
                            price=float(base['high'].iloc[i]),
                            point_type='high',
                            strength=0.5,
                            confirmed=True,
                            confidence=1.0,
                        ))
                    elif int(sl.iloc[i]) == 1:
                        t = times.iloc[i]
                        swing_points.append(SwingPoint(
                            index=int(i),
                            time=t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t,
                            price=float(base['low'].iloc[i]),
                            point_type='low',
                            strength=0.5,
                            confirmed=True,
                            confidence=1.0,
                        ))

                if len(swing_points) > 200:
                    swing_points = swing_points[-200:]
            else:
                md = MarketData(d0, handle_splits=False)
                swing_points = md.extract_swings(lookback=5, strength_threshold=0.3)
        except Exception:
            swing_points = None

        engine = self._get_engine()
        out = engine.evaluate(
            df=d0,
            features=feats,
            timeframe=str(timeframe or "H1").upper(),
            swing_points=swing_points,
            causality_results=None,
            current_equity=float(equity or 1.0),
            signal_id=str(signal_id or "default"),
        )
        return out

    @staticmethod
    def _passes_gate(out, min_conf: float, min_stability: float) -> bool:
        try:
            if out is None:
                return False
            if float(out.confidence) < float(min_conf):
                return False
            if float(getattr(out, 'stability_score', 1.0)) < float(min_stability):
                return False
            return True
        except Exception:
            return False
    
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
    
    def _execute_trade_probabilistic(self, direction: str, decision_out, df: pd.DataFrame):
        """Execute trade based on probabilistic decision output."""
        if self.executor is None:
            return
        
        try:
            entry_price = float(df['close'].iloc[-1])

            sl, tp = self._atr_sltp(df, entry_price, direction)
            
            # Calculate position size
            size_mult = float(getattr(decision_out, 'size_multiplier', 1.0) or 1.0)
            volume = self._calculate_position_size(entry_price, sl, size_mult)
            
            # Execute
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

    def _atr_sltp(self, df: pd.DataFrame, entry_price: float, direction: str) -> Tuple[float, float]:
        try:
            d0 = df.copy()
            d0.columns = [str(c).lower().strip() for c in d0.columns]
            high = d0['high']
            low = d0['low']
            close = d0['close']
            prev_close = close.shift(1)
            tr = pd.concat([
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
            if not np.isfinite(atr) or atr <= 0:
                return self._fallback_sltp(entry_price, 'LONG' if direction == 'BUY' else 'SHORT')

            sl_mult = 1.5
            sl_dist = atr * sl_mult
            rr = float(getattr(self.config, 'min_risk_reward', 2.0) or 2.0)
            tp_dist = sl_dist * rr
            if direction == 'BUY':
                return entry_price - sl_dist, entry_price + tp_dist
            return entry_price + sl_dist, entry_price - tp_dist
        except Exception:
            return self._fallback_sltp(entry_price, 'LONG' if direction == 'BUY' else 'SHORT')
    
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

        min_sl_pips = float(getattr(self.config, 'min_sl_pips', 8.0) or 8.0)
        effective_sl_pips = max(float(sl_pips), float(min_sl_pips))
        
        # Calculate lot size (assuming $10 per pip for 1 lot)
        pip_value_per_lot = 10.0
        if effective_sl_pips > 0:
            lots = risk_amount / (effective_sl_pips * pip_value_per_lot)
        else:
            lots = 0.01
        
        # Clamp to reasonable range
        max_lot = float(getattr(self.config, 'max_lot', 1.0) or 1.0)
        lots = max(0.01, min(lots, max_lot))
        
        return round(lots, 2)
    
    def _check_limits(self) -> bool:
        """Check if trading is allowed based on limits."""
        # Check max daily trades
        if self._daily_trades >= self.config.max_daily_trades:
            self.last_rejection_stage = "LIMIT"
            self.last_rejection_reason = "max daily trades"
            return False
        
        # Check max open trades
        if len(self._open_positions) >= self.config.max_open_trades:
            self.last_rejection_stage = "LIMIT"
            self.last_rejection_reason = "max open trades"
            return False
        
        # Check daily loss limit
        if self.executor is not None:
            balance = 10000.0
            if hasattr(self.executor, 'balance'):
                balance = float(self.executor.balance)
            
            max_loss = balance * (self.config.max_daily_loss_pct / 100)
            if self._daily_pnl < -max_loss:
                self.last_rejection_stage = "LIMIT"
                self.last_rejection_reason = "daily loss limit"
                return False
        
        return True
    
    def _check_daily_reset(self, current_time: Optional[datetime] = None):
        """Reset daily stats if new day."""
        try:
            today = (current_time or datetime.utcnow()).date()
        except Exception:
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
