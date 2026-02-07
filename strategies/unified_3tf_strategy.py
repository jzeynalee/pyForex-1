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
from utils.config import settings

from alpha_factory.features_engineering import FeatureEngineerOptimized
from alpha_factory.market_data import MarketData
from alpha_factory.probabilistic_alpha_factory import create_probabilistic_alpha_factory, ProbabilisticConfig

try:
    from risk_management.phase2_risk_calc.sl_tp_calculator import (
        SLTPCalculator, SLTPConfig, TradeDirection,
    )
    from risk_management.phase2_risk_calc.position_sizing import (
        PositionSizingCalculator, PositionSizingConfig,
    )
    _PHASE2_AVAILABLE = True
except ImportError:
    _PHASE2_AVAILABLE = False

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
    
    # Confidence thresholds
    min_htf_confidence: float = 0.62
    min_mtf_confidence: float = 0.66
    min_ltf_confidence: float = 0.72
    min_stability: float = 0.55
    min_directional_score: float = 0.32

    relaxed_alignment: bool = False
    
    # Risk management
    base_risk_percent: float = 0.5
    min_risk_reward: float = 2.0
    max_open_trades: int = 2
    max_daily_trades: int = 5
    max_daily_loss_pct: float = 2.0
    min_sl_pips: float = 10.0
    max_lot: float = 1.0
    cooldown: float = 30.0  # Minutes between trades
    atr_sl_mult: float = 2.0
    atr_period: int = 14
    
    # Model paths
    weights_dir: str = 'models/weights'
    
    # Sequence length for MH-TCN
    sequence_length: int = 60

    fast_backtest: bool = False
    
    def __post_init__(self):
        """Set parameters based on profile from settings."""
        p = self.profile.upper()
        
        # Load profile-specific settings from centralized config
        if p == 'SCALP':
            self.htf = settings.SCALP_HTF
            self.mtf = settings.SCALP_MTF
            self.ltf = settings.SCALP_LTF
            
            self.min_htf_confidence = settings.SCALP_MIN_HTF_CONF
            self.min_mtf_confidence = settings.SCALP_MIN_MTF_CONF
            self.min_ltf_confidence = settings.SCALP_MIN_LTF_CONF
            self.min_stability = settings.SCALP_MIN_STABILITY
            self.min_directional_score = settings.SCALP_MIN_DIR_SCORE
            self.relaxed_alignment = settings.SCALP_RELAXED
            
            self.base_risk_percent = settings.SCALP_BASE_RISK
            self.min_risk_reward = settings.SCALP_MIN_RR
            self.max_open_trades = settings.SCALP_MAX_OPEN
            self.max_daily_trades = settings.SCALP_MAX_DAILY
            self.max_daily_loss_pct = settings.SCALP_MAX_LOSS
            self.min_sl_pips = settings.SCALP_MIN_SL_PIPS
            self.max_lot = settings.SCALP_MAX_LOT
            self.cooldown = settings.SCALP_COOLDOWN
            self.atr_sl_mult = settings.SCALP_ATR_SL_MULT
            self.atr_period = settings.SCALP_ATR_PERIOD
            
        elif p == 'SWING':
            self.htf = settings.SWING_HTF
            self.mtf = settings.SWING_MTF
            self.ltf = settings.SWING_LTF
            
            self.min_htf_confidence = settings.SWING_MIN_HTF_CONF
            self.min_mtf_confidence = settings.SWING_MIN_MTF_CONF
            self.min_ltf_confidence = settings.SWING_MIN_LTF_CONF
            self.min_stability = settings.SWING_MIN_STABILITY
            self.min_directional_score = settings.SWING_MIN_DIR_SCORE
            self.relaxed_alignment = settings.SWING_RELAXED
            
            self.base_risk_percent = settings.SWING_BASE_RISK
            self.min_risk_reward = settings.SWING_MIN_RR
            self.max_open_trades = settings.SWING_MAX_OPEN
            self.max_daily_trades = settings.SWING_MAX_DAILY
            self.max_daily_loss_pct = settings.SWING_MAX_LOSS
            self.min_sl_pips = settings.SWING_MIN_SL_PIPS
            self.max_lot = settings.SWING_MAX_LOT
            self.cooldown = settings.SWING_COOLDOWN
            self.atr_sl_mult = settings.SWING_ATR_SL_MULT
            self.atr_period = settings.SWING_ATR_PERIOD
            
        else: # Default to INTRADAY
            self.htf = settings.INTRADAY_HTF
            self.mtf = settings.INTRADAY_MTF
            self.ltf = settings.INTRADAY_LTF
            
            self.min_htf_confidence = settings.INTRADAY_MIN_HTF_CONF
            self.min_mtf_confidence = settings.INTRADAY_MIN_MTF_CONF
            self.min_ltf_confidence = settings.INTRADAY_MIN_LTF_CONF
            self.min_stability = settings.INTRADAY_MIN_STABILITY
            self.min_directional_score = settings.INTRADAY_MIN_DIR_SCORE
            self.relaxed_alignment = settings.INTRADAY_RELAXED
            
            self.base_risk_percent = settings.INTRADAY_BASE_RISK
            self.min_risk_reward = settings.INTRADAY_MIN_RR
            self.max_open_trades = settings.INTRADAY_MAX_OPEN
            self.max_daily_trades = settings.INTRADAY_MAX_DAILY
            self.max_daily_loss_pct = settings.INTRADAY_MAX_LOSS
            self.min_sl_pips = settings.INTRADAY_MIN_SL_PIPS
            self.max_lot = settings.INTRADAY_MAX_LOT
            self.cooldown = settings.INTRADAY_COOLDOWN
            self.atr_sl_mult = settings.INTRADAY_ATR_SL_MULT
            self.atr_period = settings.INTRADAY_ATR_PERIOD


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

        self._tf_eval_cache: Dict[str, Dict[str, object]] = {}

        # Backtest diagnostics (kept for compatibility with main.py backtest reporting)
        self.last_rejection_stage: str = ""
        self.last_rejection_reason: str = ""
        
        # State tracking
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._last_trade_date = None
        self._open_positions: Dict[str, dict] = {}
        self._last_entry_time: Optional[datetime] = None
        self._current_time: Optional[datetime] = None
        
        # Trend stability tracking
        self._tf_direction_history: Dict[str, List[str]] = {
            self.config.htf: [],
            self.config.mtf: [],
            self.config.ltf: []
        }
        self._max_history = 5
        
        logger.info(f"Unified3TFStrategy created for {self.config.symbol} ({self.config.profile})")

    def _sync_open_positions(self):
        """Sync internal open-position view from executor (backtest/live)."""
        self._open_positions = {}
        if self.executor is None:
            return
        
        # Update daily PnL from closed trades if in backtest/sim mode
        try:
            th = getattr(self.executor, 'trade_history', [])
            if th:
                today = (self._current_time or datetime.utcnow()).date()
                if self._last_trade_date:
                    today = self._last_trade_date
                
                daily_pnl = 0.0
                for trade in th:
                    exit_time = getattr(trade, 'exit_time', None)
                    if exit_time:
                        if isinstance(exit_time, str):
                            exit_time = pd.to_datetime(exit_time)
                        if hasattr(exit_time, 'date') and exit_time.date() == today:
                            daily_pnl += float(getattr(trade, 'pnl', 0.0))
                self._daily_pnl = daily_pnl
        except Exception as e:
            logger.debug(f"Could not sync daily PnL: {e}")

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

            cfg = None
            try:
                p = str(self.config.profile or 'INTRADAY').upper()
                
                # Extract decision params based on profile
                if p == 'SCALP':
                    agg = settings.SCALP_AGG_METHOD
                    mhtcn_w = settings.SCALP_MHTCN_WEIGHT
                    stab_w = settings.SCALP_STABILITY_WEIGHT
                    reg_scale = settings.SCALP_REGIME_SCALE
                    ent_w = settings.SCALP_ENTROPY_WEIGHT
                    calib = settings.SCALP_CALIB_METHOD
                    decay = settings.SCALP_DECAY_RATE
                    key_only = settings.SCALP_KEY_FEATS_ONLY
                elif p == 'SWING':
                    agg = settings.SWING_AGG_METHOD
                    mhtcn_w = settings.SWING_MHTCN_WEIGHT
                    stab_w = settings.SWING_STABILITY_WEIGHT
                    reg_scale = settings.SWING_REGIME_SCALE
                    ent_w = settings.SWING_ENTROPY_WEIGHT
                    calib = settings.SWING_CALIB_METHOD
                    decay = settings.SWING_DECAY_RATE
                    key_only = settings.SWING_KEY_FEATS_ONLY
                else:
                    agg = settings.INTRADAY_AGG_METHOD
                    mhtcn_w = settings.INTRADAY_MHTCN_WEIGHT
                    stab_w = settings.INTRADAY_STABILITY_WEIGHT
                    reg_scale = settings.INTRADAY_REGIME_SCALE
                    ent_w = settings.INTRADAY_ENTROPY_WEIGHT
                    calib = settings.INTRADAY_CALIB_METHOD
                    decay = settings.INTRADAY_DECAY_RATE
                    key_only = settings.INTRADAY_KEY_FEATS_ONLY

                cfg = ProbabilisticConfig(
                    key_features_only=key_only or bool(getattr(self.config, 'fast_backtest', False)),
                    aggregation_method=agg,
                    mhtcn_weight=mhtcn_w,
                    stability_weight=stab_w,
                    regime_scale_factor=reg_scale,
                    entropy_weight=ent_w,
                    calibration_method=calib,
                    alpha_decay_rate=decay
                )
            except Exception as e:
                logger.warning(f"Could not initialize custom ProbabilisticConfig for profile {p}: {e}")
                cfg = None

            self._engine = create_probabilistic_alpha_factory(
                config=cfg,
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

    def _directional_score_ok(self, out) -> bool:
        try:
            if out is None:
                return False
            pb = float(getattr(out, 'final_p_bull', 0.0) or 0.0)
            pr = float(getattr(out, 'final_p_bear', 0.0) or 0.0)
            score = abs(pb - pr)
            return score >= float(getattr(self.config, 'min_directional_score', 0.0) or 0.0)
        except Exception:
            return True
    
    def _update_direction_history(self, timeframe: str, direction: str):
        """Update historical directions for stability checks."""
        tf = str(timeframe).upper()
        if tf not in self._tf_direction_history:
            self._tf_direction_history[tf] = []
        
        self._tf_direction_history[tf].append(str(direction).upper())
        if len(self._tf_direction_history[tf]) > self._max_history:
            self._tf_direction_history[tf].pop(0)

    def _is_trend_stable(self, timeframe: str, required_bars: int = 2) -> bool:
        """Check if the trend has been stable for the required number of bars."""
        tf = str(timeframe).upper()
        history = self._tf_direction_history.get(tf, [])
        if len(history) < required_bars:
            return False
        
        recent = history[-required_bars:]
        # Check if all recent bars have the same non-HOLD direction
        first = recent[0]
        if first == 'HOLD':
            return False
        return all(d == first for d in recent)

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
        
        self._current_time = current_time

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

            # HTF Evaluation (The Governor)
            htf_out = self._evaluate_timeframe(data_htf, timeframe=self.config.htf, equity=balance, signal_id="htf")
            htf_pass = self._passes_gate(htf_out, min_conf=self.config.min_htf_confidence, min_stability=self.config.min_stability)
            
            # MTF Evaluation (The Validator)
            mtf_out = self._evaluate_timeframe(data_mtf, timeframe=self.config.mtf, equity=balance, signal_id="mtf")
            mtf_pass = self._passes_gate(mtf_out, min_conf=self.config.min_mtf_confidence, min_stability=self.config.min_stability)
            
            # LTF Evaluation (The Trigger)
            ltf_out = self._evaluate_timeframe(data_ltf, timeframe=self.config.ltf, equity=balance, signal_id="ltf")
            ltf_pass = self._passes_gate(ltf_out, min_conf=self.config.min_ltf_confidence, min_stability=self.config.min_stability)

            # Update direction histories
            self._update_direction_history(self.config.htf, htf_out.direction)
            self._update_direction_history(self.config.mtf, mtf_out.direction)
            self._update_direction_history(self.config.ltf, ltf_out.direction)

            # Alignment logic based on profile
            if bool(getattr(self.config, 'relaxed_alignment', False)):
                # Relaxed: MTF + LTF must agree. HTF is optional but cannot oppose.
                if not mtf_pass:
                    self.last_rejection_stage = "MTF"
                    mtf_conf = float(getattr(mtf_out, 'confidence', 0))
                    mtf_stab = float(getattr(mtf_out, 'stability_score', 0))
                    self.last_rejection_reason = (
                        f"mtf gate failed conf={mtf_conf:.3f}<{self.config.min_mtf_confidence:.2f} "
                        f"stab={mtf_stab:.3f}<{self.config.min_stability:.2f} "
                        f"dir={getattr(mtf_out, 'direction', '?')}"
                    )
                    return None
                
                if mtf_out.direction == 'HOLD':
                    self.last_rejection_stage = "MTF"
                    self.last_rejection_reason = "mtf hold"
                    return None

                if not ltf_pass:
                    self.last_rejection_stage = "LTF"
                    self.last_rejection_reason = "ltf gate failed"
                    return None

                if ltf_out.direction == 'HOLD':
                    self.last_rejection_stage = "LTF"
                    self.last_rejection_reason = "ltf hold"
                    return None

                if str(ltf_out.direction) != str(mtf_out.direction):
                    self.last_rejection_stage = "ALIGN"
                    self.last_rejection_reason = f"mtf dir {mtf_out.direction} != ltf dir {ltf_out.direction}"
                    return None

                # Trend Stability Check (SCALP specific)
                # Ensure MTF trend is stable for at least 2 bars
                if not self._is_trend_stable(self.config.mtf, required_bars=2):
                    self.last_rejection_stage = "STABILITY"
                    self.last_rejection_reason = f"mtf trend unstable: {self._tf_direction_history.get(self.config.mtf)}"
                    return None

                # HTF Veto: If HTF is confident, it must not oppose LTF
                if htf_pass and htf_out.direction != 'HOLD' and str(htf_out.direction) != str(ltf_out.direction):
                    self.last_rejection_stage = "ALIGN"
                    self.last_rejection_reason = f"htf dir {htf_out.direction} opposes ltf dir {ltf_out.direction}"
                    return None
            else:
                # Strict: All 3 TFs must agree and pass gates
                if not htf_pass or htf_out.direction == 'HOLD':
                    self.last_rejection_stage = "HTF"
                    self.last_rejection_reason = "htf gate failed or hold"
                    return None
                
                if not mtf_pass or mtf_out.direction == 'HOLD':
                    self.last_rejection_stage = "MTF"
                    self.last_rejection_reason = "mtf gate failed or hold"
                    return None
                
                if not ltf_pass or ltf_out.direction == 'HOLD':
                    self.last_rejection_stage = "LTF"
                    self.last_rejection_reason = "ltf gate failed or hold"
                    return None

                if not (str(htf_out.direction) == str(mtf_out.direction) == str(ltf_out.direction)):
                    self.last_rejection_stage = "ALIGN"
                    self.last_rejection_reason = f"3TF mismatch: h={htf_out.direction}, m={mtf_out.direction}, l={ltf_out.direction}"
                    return None

            # Directional Score Check
            if not self._directional_score_ok(ltf_out):
                self.last_rejection_stage = "LTF"
                self.last_rejection_reason = "ltf directional score too low"
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

            logger.debug(
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

        tf_key = str(timeframe or '').upper()
        cache_key = None
        try:
            if 'time' in df.columns:
                cache_key = str(pd.to_datetime(df['time'].iloc[-1]))
        except Exception:
            cache_key = None

        if cache_key is not None:
            cached = self._tf_eval_cache.get(tf_key)
            if cached and str(cached.get('t')) == str(cache_key):
                return cached.get('out')

        d0 = df.copy()
        d0.columns = [str(c).lower().strip() for c in d0.columns]
        if 'volume' not in d0.columns:
            if 'tick_volume' in d0.columns:
                d0['volume'] = d0['tick_volume']
            else:
                d0['volume'] = 0.0

        if bool(getattr(self.config, 'fast_backtest', False)):
            feats = self._generate_fast_features(d0)
        else:
            if self._feature_engineer is None:
                self._feature_engineer = FeatureEngineerOptimized()
            feats = self._feature_engineer.generate_features(d0, batch_processing=False)

        swing_points = None
        if not bool(getattr(self.config, 'fast_backtest', False)):
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

        # Compute lightweight causality: rolling feature-return correlations
        causality_results = self._compute_fast_causality(d0, feats)

        engine = self._get_engine()
        out = engine.evaluate(
            df=d0,
            features=feats,
            timeframe=str(timeframe or "H1").upper(),
            swing_points=swing_points,
            causality_results=causality_results,
            current_equity=float(equity or 1.0),
            signal_id=str(signal_id or "default"),
        )

        if cache_key is not None:
            self._tf_eval_cache[tf_key] = {'t': cache_key, 'out': out}
        return out

    @staticmethod
    def _compute_fast_causality(df: pd.DataFrame, feats: pd.DataFrame) -> dict:
        """Compute lightweight feature-return correlations as fast causality proxy.
        
        Instead of full Granger/Transfer-Entropy analysis (too expensive per-bar),
        compute rolling correlation between each feature and forward returns.
        Returns a dict compatible with CausalConfidenceCalculator.
        """
        try:
            if feats is None or feats.empty or df is None or len(df) < 30:
                return None
            
            close = df['close'] if 'close' in df.columns else None
            if close is None or len(close) < 30:
                return None
            
            # Forward returns (1-bar and 3-bar)
            fwd_ret_1 = close.pct_change().shift(-1)
            fwd_ret_3 = close.pct_change(3).shift(-3)
            
            correlations = {}
            causal_ranking = {}
            
            lookback = min(100, len(feats) - 5)
            if lookback < 20:
                return None
            
            for col in feats.columns:
                try:
                    feat_vals = feats[col].iloc[-lookback:]
                    ret1 = fwd_ret_1.iloc[-lookback:]
                    ret3 = fwd_ret_3.iloc[-lookback:]
                    
                    # Drop NaN pairs
                    mask1 = feat_vals.notna() & ret1.notna()
                    mask3 = feat_vals.notna() & ret3.notna()
                    
                    if mask1.sum() < 15 or mask3.sum() < 15:
                        continue
                    
                    corr1 = float(feat_vals[mask1].corr(ret1[mask1]))
                    corr3 = float(feat_vals[mask3].corr(ret3[mask3]))
                    
                    if not np.isfinite(corr1):
                        corr1 = 0.0
                    if not np.isfinite(corr3):
                        corr3 = 0.0
                    
                    # Combined score: blend of 1-bar and 3-bar correlations
                    combined = abs(corr1) * 0.4 + abs(corr3) * 0.6
                    
                    correlations[col] = combined
                    causal_ranking[col] = {
                        'combined_score': float(combined),
                        'p_value': max(0.01, 1.0 - combined * 2),  # Approximate
                        'f_statistic': float(combined * 10),
                    }
                except Exception:
                    continue
            
            if not correlations:
                return None
            
            return {
                'causal_ranking': causal_ranking,
                'correlations': correlations,
            }
        except Exception:
            return None

    @staticmethod
    def _ema(s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=int(span), adjust=False, min_periods=int(span)).mean()

    def _generate_fast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        d0 = df.copy()
        d0.columns = [str(c).lower().strip() for c in d0.columns]
        close = pd.to_numeric(d0.get('close'), errors='coerce')
        high = pd.to_numeric(d0.get('high'), errors='coerce')
        low = pd.to_numeric(d0.get('low'), errors='coerce')

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / (avg_loss + 1e-12)
        rsi = 100 - (100 / (1 + rs))

        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        macd = ema12 - ema26
        macd_signal = self._ema(macd, 9)
        macd_hist = macd - macd_signal

        ma20 = close.rolling(20, min_periods=20).mean()
        sd20 = close.rolling(20, min_periods=20).std()
        bb_upper = ma20 + 2.0 * sd20
        bb_lower = ma20 - 2.0 * sd20
        bb_pos = (close - bb_lower) / ((bb_upper - bb_lower) + 1e-12)

        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=14).mean()
        atr_ratio = atr / (close.abs() + 1e-12)

        up_move = high.diff()
        down_move = (-low.diff())
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        tr14 = tr.rolling(14, min_periods=14).sum()
        plus_di = 100.0 * (plus_dm.rolling(14, min_periods=14).sum() / (tr14 + 1e-12))
        minus_di = 100.0 * (minus_dm.rolling(14, min_periods=14).sum() / (tr14 + 1e-12))
        dx = ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12)) * 100.0
        adx = dx.rolling(14, min_periods=14).mean()

        momentum = close.pct_change(10)
        vol20 = close.pct_change().rolling(20, min_periods=20).std()
        vol100 = close.pct_change().rolling(100, min_periods=100).std()
        volatility_ratio = vol20 / (vol100 + 1e-12)

        ema20 = self._ema(close, 20)
        ema50 = self._ema(close, 50)
        trend_strength = ((ema20 - ema50).abs() / (atr + 1e-12)).clip(0, 10)

        out = pd.DataFrame(index=d0.index)
        out['rsi'] = rsi
        out['macd'] = macd
        out['macd_histogram'] = macd_hist
        out['adx'] = adx
        out['bb_position'] = bb_pos
        out['atr_ratio'] = atr_ratio
        out['momentum'] = momentum
        out['trend_strength'] = trend_strength
        out['volatility_ratio'] = volatility_ratio
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
        """Execute trade using Phase 2 risk management with MH-TCN quantile predictions."""
        if self.executor is None:
            return
        
        try:
            entry_price = float(df['close'].iloc[-1])
            atr = self._compute_atr(df)
            confidence = float(getattr(decision_out, 'confidence', 0.5) or 0.5)
            size_mult = float(getattr(decision_out, 'size_multiplier', 1.0) or 1.0)
            
            # Extract MH-TCN quantiles and volatility from prediction
            quantiles_5 = None
            model_volatility = None
            mhtcn_pred = getattr(decision_out, 'mhtcn_prediction', None)
            if mhtcn_pred is not None:
                raw_q = getattr(mhtcn_pred, 'quantiles', None)
                if raw_q is not None and len(raw_q) >= 5:
                    quantiles_5 = self._map_quantiles_to_5(raw_q)
                    if quantiles_5 is not None:
                        if not all(quantiles_5[i] <= quantiles_5[i + 1] for i in range(4)):
                            quantiles_5 = None
                        elif np.max(np.abs(quantiles_5)) > 0.05:
                            quantiles_5 = None
                mv = getattr(mhtcn_pred, 'volatility', None)
                if mv is not None:
                    mv = float(mv)
                    if 0.00005 < mv < 0.01:
                        model_volatility = mv
            
            volatility = model_volatility if model_volatility is not None else atr
            
            # Phase 2: Calculate SL/TP using SLTPCalculator
            sl, tp = self._atr_sltp(df, entry_price, direction)  # fallback
            if _PHASE2_AVAILABLE:
                try:
                    trade_dir = TradeDirection.BUY if direction == 'BUY' else TradeDirection.SELL
                    sl_mult = float(getattr(self.config, 'atr_sl_mult', 2.0) or 2.0)
                    sltp_config = SLTPConfig(
                        min_risk_reward=float(getattr(self.config, 'min_risk_reward', 1.5) or 1.5),
                        min_sl_atr_multiple=max(1.0, sl_mult * 0.6),
                        max_sl_atr_multiple=sl_mult * 1.5,
                        min_tp_atr_multiple=1.5,
                        max_tp_atr_multiple=sl_mult * 3.0,
                    )
                    sltp_calc = SLTPCalculator(sltp_config)
                    sltp_result = sltp_calc.calculate(
                        entry_price=entry_price,
                        direction=trade_dir,
                        quantiles=quantiles_5,
                        volatility=volatility,
                        regime=None,
                        direction_confidence=confidence,
                        atr=atr,
                    )
                    sl = sltp_result.stop_loss
                    tp = sltp_result.take_profit
                except Exception as e:
                    logger.debug(f"Phase 2 SL/TP failed ({e}), using ATR fallback")
            
            # Phase 2: Calculate position size
            volume = self._calculate_position_size_v2(
                entry_price, sl, confidence, volatility, size_mult
            )
            
            # Execute
            if hasattr(self.executor, 'entry'):
                self.executor.entry(signal=direction, volume=volume, sl=sl, tp=tp)
            elif hasattr(self.executor, 'open_position'):
                self.executor.open_position(
                    direction=direction, volume=volume, stop_loss=sl, take_profit=tp
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
            atr_period = int(getattr(self.config, 'atr_period', 14))
            atr = float(tr.rolling(atr_period, min_periods=1).mean().iloc[-1])
            if not np.isfinite(atr) or atr <= 0:
                return self._fallback_sltp(entry_price, 'LONG' if direction == 'BUY' else 'SHORT')

            sl_mult = float(getattr(self.config, 'atr_sl_mult', 1.5))
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
    
    def _compute_atr(self, df: pd.DataFrame) -> float:
        """Compute ATR from OHLCV data, returns raw ATR value."""
        try:
            d0 = df.copy()
            d0.columns = [str(c).lower().strip() for c in d0.columns]
            high, low, close = d0['high'], d0['low'], d0['close']
            prev_close = close.shift(1)
            tr = pd.concat([
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr_period = int(getattr(self.config, 'atr_period', 14))
            atr = float(tr.rolling(atr_period, min_periods=1).mean().iloc[-1])
            if np.isfinite(atr) and atr > 0:
                return atr
        except Exception:
            pass
        return 0.002  # fallback ~20 pips
    
    @staticmethod
    def _map_quantiles_to_5(raw_quantiles) -> Optional[np.ndarray]:
        """Map raw MH-TCN quantiles to [Q5, Q25, Q50, Q75, Q95] format."""
        try:
            raw = np.asarray(raw_quantiles, dtype=float).flatten()
            n = len(raw)
            if n == 5:
                return raw
            elif n == 7:
                # [Q5, Q10, Q25, Q50, Q75, Q90, Q95] → [Q5, Q25, Q50, Q75, Q95]
                return np.array([raw[0], raw[2], raw[3], raw[4], raw[6]])
            elif n >= 5:
                indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
                return raw[indices]
        except Exception:
            pass
        return None
    
    def _calculate_position_size_v2(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        volatility: float,
        size_multiplier: float = 1.0,
    ) -> float:
        """Calculate position size using Phase 2 PositionSizingCalculator."""
        balance = 10000.0
        if self.executor is not None:
            if hasattr(self.executor, 'balance'):
                balance = float(self.executor.balance)
            elif hasattr(self.executor, 'get_account_balance'):
                balance = float(self.executor.get_account_balance())
        
        max_lot = float(getattr(self.config, 'max_lot', 1.0) or 1.0)
        base_risk = float(self.config.base_risk_percent)
        
        if _PHASE2_AVAILABLE:
            try:
                ps_config = PositionSizingConfig(
                    base_risk_percent=base_risk * max(size_multiplier, 0.1),
                    max_risk_percent=base_risk * 2.0,
                    min_risk_percent=0.05,
                    confidence_scaling=True,
                    min_confidence_for_full_size=0.6,
                    low_confidence_risk_reduction=0.5,
                    volatility_scaling=True,
                    high_vol_risk_reduction=0.6,
                    low_vol_risk_increase=1.1,
                    use_streak_adjustment=False,
                    use_kelly=False,
                )
                ps_calc = PositionSizingCalculator(ps_config)
                result = ps_calc.calculate(
                    account_balance=balance,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    pair=self.config.symbol,
                    direction_confidence=confidence,
                    volatility=volatility,
                )
                lots = result.position_size
                lots = max(0.01, min(lots, max_lot))
                return round(lots, 2)
            except Exception as e:
                logger.debug(f"Phase 2 position sizing failed ({e}), using fallback")
        
        # Fallback: simple risk-based calculation
        risk_amount = balance * (base_risk / 100) * size_multiplier
        sl_distance = abs(entry_price - stop_loss)
        pip_value = 0.0001 if 'JPY' not in self.config.symbol else 0.01
        sl_pips = sl_distance / pip_value
        min_sl_pips = float(getattr(self.config, 'min_sl_pips', 8.0) or 8.0)
        effective_sl_pips = max(float(sl_pips), float(min_sl_pips))
        lots = risk_amount / (effective_sl_pips * 10.0) if effective_sl_pips > 0 else 0.01
        lots = max(0.01, min(lots, max_lot))
        return round(lots, 2)
    
    def _check_limits(self) -> bool:
        """Check if trading is allowed based on limits."""
        # Check cooldown between trades
        cooldown_minutes = float(getattr(self.config, 'cooldown', 30.0) or 30.0)
        if self._last_entry_time is not None and self._current_time is not None:
            elapsed = (self._current_time - self._last_entry_time).total_seconds() / 60.0
            if elapsed < cooldown_minutes:
                self.last_rejection_stage = "COOLDOWN"
                self.last_rejection_reason = f"cooldown {elapsed:.1f}m<{cooldown_minutes:.1f}m"
                return False

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
