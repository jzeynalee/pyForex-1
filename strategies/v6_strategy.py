# strategies/v6_strategy.py
"""
V6 Strategy — Production-grade AlphaV2 + ProbabilisticTCN with full 3TF cascade.

Architecture (Phase 2 — model-driven direction):
    Each timeframe (HTF, TTF, LTF) gets its own V6TFEngine containing:
        - FeaturePipeline → 200+ indicators
        - RegimeDetector → TRENDING/RANGING/VOLATILE/TRANSITION
        - AlphaHeadV2 → FEATURE ENGINE (category probs feed MH-TCN ch 0-7)
        - ProbabilisticMHTCNFilter → g_factor drives BOTH direction AND confidence

    Phase 2 key change: MH-TCN's g_factor determines trade DIRECTION:
        g_factor > buy_threshold  → BUY  (model predicts price UP)
        g_factor < sell_threshold → SELL (model predicts price DOWN)
        between thresholds        → HOLD (model uncertain)
    AlphaV2 no longer decides direction — it's a feature engine only.

    Separate MH-TCN weights per profile×TF from v6_profiles/:
        SCALP:     LTF=M5  (SCALP_M5.pt),  TTF=M15 (SCALP_M15.pt),  HTF=H1  (SCALP_H1.pt)
        INTRADAY:  LTF=M15 (INTRADAY_M15.pt), TTF=H1 (INTRADAY_H1.pt), HTF=H4 (INTRADAY_H4.pt)
        SWING:     LTF=H1  (SWING_H1.pt),  TTF=H4  (SWING_H4.pt),  HTF=D1  (SWING_D1.pt)

    3TF Cascade:
        1. HTF (Governor)  — model-driven directional bias; vetoes opposing trades
        2. TTF (Decision)  — primary signal; g_factor → direction + confidence
        3. LTF (Trigger)   — model confirms direction alignment for entry

Usage:
    python main.py live --strategy v6 --symbol EURUSD
    python main.py backtest --strategy v6 --data <CSV> --profile INTRADAY
"""

import logging
import time as _time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from strategies.base import Strategy
except ImportError:
    from abc import ABC, abstractmethod
    class Strategy(ABC):
        @abstractmethod
        def on_bar(self, df): pass

logger = logging.getLogger(__name__)


# ── Profile Definitions ─────────────────────────────────────────────────────

# Weight file naming: v6_prob_mhtcn_{PROFILE}_{TF}.pt
_WEIGHT_TEMPLATE = "v6_prob_mhtcn_{profile}_{tf}.pt"

@dataclass
class V6ProfileConfig:
    """Per-profile configuration for V6 strategy."""
    # Timeframe hierarchy
    ltf: str = "M15"
    ttf: str = "H1"       # main trading frame (base_tf)
    htf: str = "H4"

    # AlphaHeadV2 parameters (feature engine — no longer decides direction)
    lookback: int = 100
    min_alpha_probability: float = 0.40
    directional_edge_min: float = 0.03

    # Phase 2: MH-TCN model-driven direction thresholds
    # g_factor > buy_threshold  → BUY  (model confident price goes up)
    # g_factor < sell_threshold → SELL (model confident price goes down)
    # between thresholds        → HOLD (model uncertain)
    buy_threshold_ttf:  float = 0.55   # TTF: primary decision
    sell_threshold_ttf: float = 0.45
    buy_threshold_htf:  float = 0.53   # HTF: governor (slightly looser)
    sell_threshold_htf: float = 0.47
    buy_threshold_ltf:  float = 0.52   # LTF: trigger confirmation
    sell_threshold_ltf: float = 0.48

    # MH-TCN model
    seq_len: int = 64

    # Risk management
    base_risk_percent: float = 0.5
    min_risk_reward: float = 2.0
    max_open_trades: int = 2
    max_daily_trades: int = 5
    max_daily_loss_pct: float = 2.0
    cooldown_bars: int = 4
    atr_sl_mult: float = 2.0
    atr_period: int = 14
    min_sl_pips: float = 10.0
    max_lot: float = 1.0

    def weights_for_tf(self, profile: str, tf: str) -> str:
        """Return weight filename for a given profile and TF."""
        return _WEIGHT_TEMPLATE.format(profile=profile.upper(), tf=tf.upper())


PROFILE_CONFIGS: Dict[str, V6ProfileConfig] = {
    "SCALP": V6ProfileConfig(
        ltf="M5", ttf="M15", htf="H1",
        lookback=80,
        min_alpha_probability=0.40,
        directional_edge_min=0.03,
        buy_threshold_ttf=0.55, sell_threshold_ttf=0.45,
        buy_threshold_htf=0.53, sell_threshold_htf=0.47,
        buy_threshold_ltf=0.52, sell_threshold_ltf=0.48,
        cooldown_bars=4,
        base_risk_percent=0.25,
        min_risk_reward=1.5,
        max_open_trades=1,
        max_daily_trades=10,
        max_daily_loss_pct=1.5,
        min_sl_pips=6.0,
        max_lot=0.3,
        atr_sl_mult=2.8,
        atr_period=20,
    ),
    "INTRADAY": V6ProfileConfig(
        ltf="M15", ttf="H1", htf="H4",
        lookback=100,
        min_alpha_probability=0.40,
        directional_edge_min=0.03,
        buy_threshold_ttf=0.55, sell_threshold_ttf=0.45,
        buy_threshold_htf=0.53, sell_threshold_htf=0.47,
        buy_threshold_ltf=0.52, sell_threshold_ltf=0.48,
        cooldown_bars=4,
        base_risk_percent=0.5,
        min_risk_reward=2.0,
        max_open_trades=2,
        max_daily_trades=5,
        max_daily_loss_pct=2.0,
        min_sl_pips=10.0,
        max_lot=1.0,
        atr_sl_mult=2.0,
        atr_period=14,
    ),
    "SWING": V6ProfileConfig(
        ltf="H1", ttf="H4", htf="D1",
        lookback=80,
        min_alpha_probability=0.40,
        directional_edge_min=0.03,
        buy_threshold_ttf=0.55, sell_threshold_ttf=0.45,
        buy_threshold_htf=0.53, sell_threshold_htf=0.47,
        buy_threshold_ltf=0.52, sell_threshold_ltf=0.48,
        cooldown_bars=3,
        base_risk_percent=1.0,
        min_risk_reward=3.0,
        max_open_trades=3,
        max_daily_trades=2,
        max_daily_loss_pct=3.0,
        min_sl_pips=30.0,
        max_lot=2.0,
        atr_sl_mult=2.5,
        atr_period=14,
    ),
}


@dataclass
class V6StrategyConfig:
    """Top-level configuration for V6 strategy."""
    profile: str = "INTRADAY"
    symbol: str = "EURUSD"
    device: str = "cpu"
    weights_dir: str = ""
    fast_backtest: bool = False

    def __post_init__(self):
        self.profile = self.profile.upper()
        if self.profile not in PROFILE_CONFIGS:
            logger.warning(f"Unknown profile '{self.profile}', defaulting to INTRADAY")
            self.profile = "INTRADAY"

    @property
    def profile_cfg(self) -> V6ProfileConfig:
        return PROFILE_CONFIGS[self.profile]


# ── Per-TF Evaluation Result ────────────────────────────────────────────────

@dataclass
class TFEvalResult:
    """Result of evaluating one timeframe through the V6 pipeline.

    Phase 2: direction is derived from MH-TCN g_factor, NOT AlphaV2.
    confidence = distance from 0.5 (how sure the model is about direction).
    """
    direction: str = "HOLD"   # "LONG", "SHORT", "HOLD"
    g_factor: float = 0.5
    confidence: float = 0.0   # abs(g_factor - 0.5), range [0, 0.5]
    p_alpha: float = 0.5     # AlphaV2 prob (diagnostic only)
    regime: str = "TRANSITION"
    is_trade: bool = False


# ── Per-TF Engine ────────────────────────────────────────────────────────────

class V6TFEngine:
    """Self-contained V6 evaluation engine for a single timeframe.

    Each TF gets its own AlphaHeadV2, ProbabilisticMHTCNFilter (with
    separate weights), FeaturePipeline, and RegimeDetector.
    """

    def __init__(
        self,
        role: str,          # "HTF", "TTF", "LTF"
        tf_label: str,      # e.g. "H4", "H1", "M15"
        profile: str,
        lookback: int = 100,
        seq_len: int = 64,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        min_alpha_probability: float = 0.40,
        directional_edge_min: float = 0.03,
    ):
        self.role = role
        self.tf_label = tf_label
        self.profile = profile
        self._lookback = lookback
        self._seq_len = seq_len
        self._weights_path = weights_path
        self._device = device
        self._min_alpha_prob = min_alpha_probability
        self._dir_edge_min = directional_edge_min

        # Lazy-loaded
        self._feature_pipeline = None
        self._regime_detector = None
        self._alpha_head = None
        self._mhtcn_filter = None
        self._ready = False

    def init(self) -> bool:
        """Initialize all V6 components for this TF."""
        if self._ready:
            return True
        try:
            from research.feature_pipeline import FeaturePipeline
            from research.regime_detector import RegimeDetector
            from research.alpha_heads import AlphaHeadV2
            from research.mhtcn_filters.probabilistic import ProbabilisticMHTCNFilter

            self._feature_pipeline = FeaturePipeline(max_lookback=500)
            self._regime_detector = RegimeDetector()
            self._alpha_head = AlphaHeadV2(
                lookback=self._lookback,
                min_probability=self._min_alpha_prob,
                directional_edge_min=self._dir_edge_min,
            )
            self._mhtcn_filter = ProbabilisticMHTCNFilter(
                seq_len=self._seq_len,
                weights_path=self._weights_path,
                device=self._device,
            )
            self._ready = True
            logger.info(
                f"  V6TFEngine[{self.role}/{self.tf_label}] ready | "
                f"weights={self._weights_path or 'NONE'}"
            )
            return True
        except Exception as e:
            logger.error(f"V6TFEngine[{self.role}/{self.tf_label}] init failed: {e}")
            return False

    def evaluate(
        self,
        df: pd.DataFrame,
        buy_threshold: float = 0.55,
        sell_threshold: float = 0.45,
    ) -> TFEvalResult:
        """Run the full V6 Phase 2 pipeline on a DataFrame for this TF.

        Phase 2 key change: DIRECTION comes from MH-TCN g_factor, not AlphaV2.
        AlphaV2 still runs — its 6 category probabilities are channels 0-5
        of the MH-TCN's 14-channel input. But direction is now model-driven:
            g_factor > buy_threshold  → LONG
            g_factor < sell_threshold → SHORT
            else                      → HOLD

        Returns TFEvalResult with model-driven direction and confidence.
        """
        if not self._ready and not self.init():
            return TFEvalResult()

        if df is None or df.empty or len(df) < 30:
            return TFEvalResult()

        try:
            # Compute features (200+ indicators)
            features = self._feature_pipeline.compute_features(df)

            # Inject category mapping on first call
            cat_map = self._feature_pipeline.get_category_columns()
            if cat_map and not self._alpha_head._category_columns:
                self._alpha_head.set_category_columns(cat_map)

            # Detect regime
            regime = self._regime_detector.detect(df, features)

            # AlphaV2 evaluation — FEATURE ENGINE role only
            # Its category probs + regime feed into MH-TCN channels 0-7
            signal = self._alpha_head.evaluate(df, features, regime)

            # Feed MH-TCN buffer and get g_factor
            # The 14-channel input: [6 category probs, alpha_prob, regime,
            #                        ret_1, ret_5, ret_10, ATR, RSI, momentum]
            mhtcn_out = self._mhtcn_filter.filter(signal, df, features)
            g_factor = float(mhtcn_out.g_factor)

            # ── Phase 2: MH-TCN drives direction ────────────────────────
            # The model was trained: label=1 means "price goes UP",
            # label=0 means "price goes DOWN". sigmoid output > 0.5 = UP.
            confidence = abs(g_factor - 0.5)

            if g_factor > buy_threshold:
                direction = "LONG"
                is_trade = True
            elif g_factor < sell_threshold:
                direction = "SHORT"
                is_trade = True
            else:
                direction = "HOLD"
                is_trade = False

            return TFEvalResult(
                direction=direction,
                g_factor=g_factor,
                confidence=confidence,
                p_alpha=float(signal.probability),
                regime=str(regime),
                is_trade=is_trade,
            )

        except Exception as e:
            logger.debug(f"V6TFEngine[{self.role}/{self.tf_label}] eval error: {e}")
            return TFEvalResult()


# ── V6 Strategy ─────────────────────────────────────────────────────────────

class V6Strategy(Strategy):
    """
    Production V6 strategy: AlphaV2 + ProbabilisticTCN with full 3TF cascade.

    3TF Cascade Logic:
        1. HTF Governor  — direction bias; vetoes opposing TTF signals
        2. TTF Decision   — primary signal; g_factor IS trade probability
        3. LTF Trigger    — confirms direction alignment for entry

    Graceful degradation: if HTF/LTF data is unavailable, falls back
    to TTF-only mode (same as previous single-TF implementation).

    Phase 1 Fix: g_factor from ProbabilisticTCN used directly as trade
    probability — no softening formula.
    """

    def __init__(
        self,
        config: Optional[V6StrategyConfig] = None,
        data_provider=None,
        executor=None,
        **kwargs,
    ):
        self.config = config or V6StrategyConfig()
        self.data_provider = data_provider
        self.executor = executor
        self.name = f"V6_3TF_{self.config.profile}"

        self._pcfg = self.config.profile_cfg

        # 3TF engines (lazy-loaded)
        self._htf_engine: Optional[V6TFEngine] = None
        self._ttf_engine: Optional[V6TFEngine] = None
        self._ltf_engine: Optional[V6TFEngine] = None
        self._initialized = False

        # Direction history for stability checks
        self._htf_dir_history: List[str] = []
        self._ttf_dir_history: List[str] = []
        self._max_history = 10

        # Preloaded TF data for rolling windows
        self._htf_data: Optional[pd.DataFrame] = None
        self._ttf_data: Optional[pd.DataFrame] = None
        self._ltf_data: Optional[pd.DataFrame] = None
        self._htf_window_size: int = 150  # ensure 100+ bars after lookback
        self._ttf_window_size: int = 150
        self._ltf_window_size: int = 150

        # State
        self._bar_count = 0
        self._last_trade_bar = -999
        self._daily_trades = 0
        self._daily_pnl = 0.0
        self._last_trade_date = None
        self._open_positions: Dict[str, dict] = {}
        self._current_time: Optional[datetime] = None

        # Diagnostics
        self.last_rejection_stage: str = ""
        self.last_rejection_reason: str = ""
        self._trade_log: List[dict] = []

        logger.info(
            f"V6Strategy created: profile={self.config.profile} "
            f"LTF={self._pcfg.ltf} TTF={self._pcfg.ttf} HTF={self._pcfg.htf}"
        )

    # ── Initialization ───────────────────────────────────────────────────

    def _lazy_init(self) -> bool:
        """Create 3 V6TFEngines with separate weights per TF."""
        if self._initialized:
            return True
        try:
            profile = self.config.profile
            pcfg = self._pcfg

            # Resolve weight paths for each TF
            htf_w = self._resolve_weights(pcfg.weights_for_tf(profile, pcfg.htf))
            ttf_w = self._resolve_weights(pcfg.weights_for_tf(profile, pcfg.ttf))
            ltf_w = self._resolve_weights(pcfg.weights_for_tf(profile, pcfg.ltf))

            common = dict(
                profile=profile,
                lookback=pcfg.lookback,
                seq_len=pcfg.seq_len,
                device=self.config.device,
                min_alpha_probability=pcfg.min_alpha_probability,
                directional_edge_min=pcfg.directional_edge_min,
            )

            self._htf_engine = V6TFEngine(
                role="HTF", tf_label=pcfg.htf, weights_path=htf_w, **common
            )
            self._ttf_engine = V6TFEngine(
                role="TTF", tf_label=pcfg.ttf, weights_path=ttf_w, **common
            )
            self._ltf_engine = V6TFEngine(
                role="LTF", tf_label=pcfg.ltf, weights_path=ltf_w, **common
            )

            # Initialize all engines
            ok = (
                self._htf_engine.init()
                and self._ttf_engine.init()
                and self._ltf_engine.init()
            )
            if not ok:
                logger.warning("Some V6TFEngines failed to init; will degrade gracefully")

            self._initialized = True
            logger.info(
                f"V6 3TF initialized: HTF={pcfg.htf}({htf_w}) "
                f"TTF={pcfg.ttf}({ttf_w}) LTF={pcfg.ltf}({ltf_w})"
            )
            return True

        except Exception as e:
            logger.error(f"V6Strategy initialization failed: {e}", exc_info=True)
            return False

    def _resolve_weights(self, filename: str) -> Optional[str]:
        """Resolve weight file path for a given filename."""
        # Explicit config dir
        if self.config.weights_dir:
            p = Path(self.config.weights_dir) / filename
            if p.exists():
                return str(p)

        # Settings ASSETS_DIR
        try:
            from utils.config import settings
            assets_dir = Path(getattr(settings, "ASSETS_DIR", ""))
            for subdir in ("models/v6_profiles", "models/weights"):
                p = assets_dir / subdir / filename
                if p.exists():
                    return str(p)
            # Also check V6_WEIGHTS_DIR setting
            v6_dir = Path(str(getattr(settings, "V6_WEIGHTS_DIR", "") or ""))
            if v6_dir.exists():
                p = v6_dir / filename
                if p.exists():
                    return str(p)
        except Exception:
            pass

        # Local fallback
        local = Path("models/weights") / filename
        if local.exists():
            return str(local)

        logger.warning(f"MH-TCN weights not found: {filename}")
        return None

    def initialize(self, starting_balance: float = 10000.0) -> bool:
        """Explicit initialization (called by backtest framework)."""
        return self._lazy_init()

    def _preload_tf_data(self, base_df: pd.DataFrame) -> bool:
        """Preload full datasets for all 3 TFs to avoid resampling per bar."""
        try:
            pcfg = self._pcfg
            base_df = self._prep_df(base_df)
            if base_df is None or len(base_df) < 200:
                logger.warning("Insufficient base data for TF preloading")
                return False

            # For backtest: try data_provider first, fall back to resampling
            if self.data_provider is not None:
                try:
                    self._htf_data = self._fetch_tf_data(pcfg.htf, base_df, count=500)
                    self._ttf_data = self._fetch_tf_data(pcfg.ttf, base_df, count=500)
                    self._ltf_data = self._fetch_tf_data(pcfg.ltf, base_df, count=500)
                except Exception as e:
                    logger.debug(f"Provider preload failed: {e}")

            # Fallback: resample from base_df
            if self._htf_data is None:
                self._htf_data = self._resample_data(base_df, pcfg.htf)
            if self._ttf_data is None:
                self._ttf_data = self._resample_data(base_df, pcfg.ttf)
            if self._ltf_data is None:
                self._ltf_data = self._resample_data(base_df, pcfg.ltf)

            # Validate we have enough data
            for tf_name, tf_data in [("HTF", self._htf_data), ("TTF", self._ttf_data), ("LTF", self._ltf_data)]:
                if tf_data is None or len(tf_data) < 100:
                    logger.warning(f"Insufficient {tf_name} data: {0 if tf_data is None else len(tf_data)} bars")
                    return False

            logger.info(
                f"Preloaded TF data: HTF={len(self._htf_data)} "
                f"TTF={len(self._ttf_data)} LTF={len(self._ltf_data)}"
            )
            return True

        except Exception as e:
            logger.error(f"TF data preload failed: {e}")
            return False

    # ── Data Fetching ────────────────────────────────────────────────────

    def _fetch_tf_data(self, tf: str, fallback_df: pd.DataFrame, count: int = 200) -> Optional[pd.DataFrame]:
        """Fetch data for a specific TF from data_provider, with resampling fallback."""
        # Try data_provider first
        if self.data_provider is not None:
            try:
                data = self.data_provider.get_ohlcv(
                    self.config.symbol, timeframe=tf, count=count
                )
                if data is not None and not data.empty and len(data) >= 30:
                    return self._prep_df(data)
            except Exception as e:
                logger.debug(f"Could not fetch {tf} data from provider: {e}")

        # Fallback: resample from the input (LTF) data
        resampled = self._resample_data(fallback_df, tf)
        if resampled is not None and not resampled.empty and len(resampled) >= 30:
            return resampled

        return None

    def _get_rolling_window(self, tf_data: pd.DataFrame, window_size: int, current_time: datetime) -> Optional[pd.DataFrame]:
        """Extract a rolling window ending at current_time from preloaded TF data."""
        if tf_data is None or tf_data.empty:
            return None

        try:
            # Ensure time column exists and is datetime
            df = tf_data.copy()
            if 'time' not in df.columns:
                return None
            df['time'] = pd.to_datetime(df['time'])

            # Filter to bars up to current_time
            mask = df['time'] <= current_time
            recent = df[mask].tail(window_size)

            if len(recent) < 100:  # ensure minimum for indicators
                return None

            return recent.reset_index(drop=True)

        except Exception as e:
            logger.debug(f"Rolling window extraction failed: {e}")
            return None

    @staticmethod
    def _resample_data(df: pd.DataFrame, target_tf: str) -> Optional[pd.DataFrame]:
        """Resample OHLCV data to a higher timeframe."""
        if df is None or df.empty:
            return None
        d0 = df.copy()
        d0.columns = [str(c).lower().strip() for c in d0.columns]
        if "time" not in d0.columns:
            return d0

        try:
            d0["time"] = pd.to_datetime(d0["time"])
            d0 = d0.set_index("time")
            tf_map = {
                "M5": "5min", "M15": "15min", "M30": "30min",
                "H1": "1h", "H4": "4h", "D1": "1D",
            }
            rule = tf_map.get(target_tf.upper(), "1h")
            resampled = d0.resample(rule).agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
            })
            if "volume" in d0.columns:
                resampled["volume"] = d0["volume"].resample(rule).sum()
            elif "tick_volume" in d0.columns:
                resampled["volume"] = d0["tick_volume"].resample(rule).sum()
            else:
                resampled["volume"] = 0.0
            resampled = resampled.dropna().reset_index()
            return resampled if len(resampled) >= 20 else None
        except Exception as e:
            logger.debug(f"Resample to {target_tf} failed: {e}")
            return None

    # ── Main Bar Handler ─────────────────────────────────────────────────

    def on_bar(self, df: pd.DataFrame) -> Optional[str]:
        """
        Process a new bar through the full 3TF V6 cascade.

        Cascade:
            1. Fetch HTF, TTF, LTF data
            2. HTF → AlphaV2 direction + MH-TCN g_factor (Governor)
            3. TTF → AlphaV2 direction + MH-TCN g_factor (Decision)
            4. LTF → AlphaV2 direction + MH-TCN g_factor (Trigger)
            5. Alignment check: TTF+LTF must agree; HTF must not oppose
            6. Phase 1: TTF g_factor IS the trade probability
            7. If g_factor > threshold → execute trade

        Graceful degradation: if HTF/LTF data unavailable, falls back
        to TTF-only mode.
        """
        if not self._lazy_init():
            return None

        self.last_rejection_stage = ""
        self.last_rejection_reason = ""
        self._bar_count += 1

        # Parse current time
        self._current_time = self._extract_time(df)
        self._check_daily_reset()
        self._sync_open_positions()

        # ── Gate checks ──────────────────────────────────────────────────

        if len(self._open_positions) >= self._pcfg.max_open_trades:
            self.last_rejection_stage = "LIMIT"
            self.last_rejection_reason = "max open trades"
            return None

        if self._daily_trades >= self._pcfg.max_daily_trades:
            self.last_rejection_stage = "LIMIT"
            self.last_rejection_reason = "max daily trades"
            return None

        bars_since_trade = self._bar_count - self._last_trade_bar
        if bars_since_trade < self._pcfg.cooldown_bars:
            self.last_rejection_stage = "COOLDOWN"
            self.last_rejection_reason = f"cooldown {bars_since_trade}/{self._pcfg.cooldown_bars}"
            return None

        # ── Data prep and TF window extraction ────────────────────────────────
        d0 = self._prep_df(df)
        if d0 is None or len(d0) < 50:
            self.last_rejection_stage = "DATA"
            self.last_rejection_reason = f"insufficient bars ({0 if d0 is None else len(d0)})"
            return None

        # Preload TF data on first bar if not already done
        if self._htf_data is None and self._bar_count == 1:
            if not self._preload_tf_data(d0):
                logger.warning("TF data preload failed, falling back to TTF-only mode")
                self._htf_data = None
                self._ltf_data = None

        # Extract rolling windows from preloaded data
        pcfg = self._pcfg
        htf_data = self._get_rolling_window(self._htf_data, self._htf_window_size, self._current_time) if self._htf_data is not None else None
        ttf_data = self._get_rolling_window(self._ttf_data, self._ttf_window_size, self._current_time) if self._ttf_data is not None else d0
        ltf_data = self._get_rolling_window(self._ltf_data, self._ltf_window_size, self._current_time) if self._ltf_data is not None else None

        # Validate we have sufficient data for each TF
        if ttf_data is None or len(ttf_data) < 100:
            self.last_rejection_stage = "DATA"
            self.last_rejection_reason = f"insufficient TTF data ({0 if ttf_data is None else len(ttf_data)})"
            return None

        # HTF/LTF are optional - degrade gracefully if insufficient
        if htf_data is not None and len(htf_data) < 100:
            htf_data = None
        if ltf_data is not None and len(ltf_data) < 100:
            ltf_data = None

        # ── Phase 2: 3TF Model-Driven Cascade ─────────────────────────

        # TTF Decision: model determines direction + confidence
        ttf_result = self._ttf_engine.evaluate(
            ttf_data,
            buy_threshold=pcfg.buy_threshold_ttf,
            sell_threshold=pcfg.sell_threshold_ttf,
        )
        if not ttf_result.is_trade:
            self.last_rejection_stage = "TTF_MODEL"
            self.last_rejection_reason = (
                f"TTF HOLD: g={ttf_result.g_factor:.3f} "
                f"(buy>{pcfg.buy_threshold_ttf}, sell<{pcfg.sell_threshold_ttf})"
            )
            return None

        # HTF Governor: model-driven direction must not oppose TTF
        htf_result = None
        if htf_data is not None and self._htf_engine._ready:
            htf_result = self._htf_engine.evaluate(
                htf_data,
                buy_threshold=pcfg.buy_threshold_htf,
                sell_threshold=pcfg.sell_threshold_htf,
            )
            self._update_dir_history(self._htf_dir_history, htf_result.direction)

            if htf_result.is_trade and htf_result.direction != "HOLD":
                # HTF model has a directional opinion — must not oppose TTF
                if htf_result.direction != ttf_result.direction:
                    self.last_rejection_stage = "HTF_VETO"
                    self.last_rejection_reason = (
                        f"HTF model={htf_result.direction}(g={htf_result.g_factor:.3f}) "
                        f"opposes TTF={ttf_result.direction}(g={ttf_result.g_factor:.3f})"
                    )
                    return None

        # HTF trend stability: require consistent direction for 2+ bars
        if not self._is_htf_stable(ttf_result.direction):
            if len(self._htf_dir_history) >= 2:
                self.last_rejection_stage = "HTF_STABILITY"
                self.last_rejection_reason = (
                    f"HTF trend unstable: {self._htf_dir_history[-3:]}"
                )
                return None

        # LTF Trigger: model-driven confirmation of TTF direction
        ltf_result = None
        if ltf_data is not None and self._ltf_engine._ready:
            ltf_result = self._ltf_engine.evaluate(
                ltf_data,
                buy_threshold=pcfg.buy_threshold_ltf,
                sell_threshold=pcfg.sell_threshold_ltf,
            )

            if ltf_result.is_trade and ltf_result.direction != "HOLD":
                # LTF model has a signal — must align with TTF
                if ltf_result.direction != ttf_result.direction:
                    self.last_rejection_stage = "LTF_ALIGN"
                    self.last_rejection_reason = (
                        f"LTF model={ltf_result.direction}(g={ltf_result.g_factor:.3f}) "
                        f"!= TTF={ttf_result.direction}(g={ttf_result.g_factor:.3f})"
                    )
                    return None

        # Update TTF direction history
        self._update_dir_history(self._ttf_dir_history, ttf_result.direction)

        # ── Determine trade direction ────────────────────────────────────

        if ttf_result.direction == "LONG":
            direction = "BUY"
        elif ttf_result.direction == "SHORT":
            direction = "SELL"
        else:
            self.last_rejection_stage = "DIRECTION"
            self.last_rejection_reason = "no clear TTF direction"
            return None

        if self._has_open_direction(direction):
            self.last_rejection_stage = "LIMIT"
            self.last_rejection_reason = f"{direction} already open"
            return None

        # ── Execute trade ────────────────────────────────────────────────

        # Confidence = TTF model's distance from 0.5
        # HTF alignment bonus: if HTF model agrees, boost confidence
        g_final = ttf_result.g_factor
        if htf_result is not None and htf_result.is_trade:
            if htf_result.direction == ttf_result.direction:
                # HTF confirms → boost by HTF confidence (max +5%)
                g_final = min(1.0, g_final + htf_result.confidence * 0.1)

        if self.executor is not None:
            self._execute_trade(direction, g_final, ttf_data)

        self._last_trade_bar = self._bar_count
        self._daily_trades += 1

        # Log trade
        self._trade_log.append({
            "bar": self._bar_count,
            "time": self._current_time,
            "direction": direction,
            "g_ttf": ttf_result.g_factor,
            "g_htf": htf_result.g_factor if htf_result else None,
            "g_ltf": ltf_result.g_factor if ltf_result else None,
            "g_final": g_final,
            "p_alpha_ttf": ttf_result.p_alpha,
            "regime_ttf": ttf_result.regime,
            "htf_dir": htf_result.direction if htf_result else "N/A",
            "ltf_dir": ltf_result.direction if ltf_result else "N/A",
        })

        _g_htf = f"{htf_result.g_factor:.3f}" if htf_result else "N/A"
        _g_ltf = f"{ltf_result.g_factor:.3f}" if ltf_result else "N/A"
        logger.info(
            f"V6 3TF TRADE: {direction} | "
            f"g_ttf={ttf_result.g_factor:.3f} "
            f"g_htf={_g_htf} g_ltf={_g_ltf} | "
            f"regime={ttf_result.regime}"
        )

        return direction

    # ── Direction History / Stability ─────────────────────────────────────

    @staticmethod
    def _update_dir_history(history: list, direction: str, max_len: int = 10):
        history.append(str(direction).upper())
        while len(history) > max_len:
            history.pop(0)

    def _is_htf_stable(self, expected_dir: str, required: int = 2) -> bool:
        """Check HTF has been consistent with expected direction."""
        hist = self._htf_dir_history
        if len(hist) < required:
            return True  # not enough data — don't block
        recent = hist[-required:]
        return all(d == expected_dir or d == "HOLD" for d in recent)

    # ── Trade Execution ──────────────────────────────────────────────────

    def _execute_trade(self, direction: str, g_factor: float, df: pd.DataFrame):
        """Execute trade with ATR-based SL/TP and risk-based sizing."""
        if self.executor is None:
            return
        try:
            entry_price = float(df["close"].iloc[-1])
            atr = self._compute_atr(df)
            if atr < 1e-8:
                logger.warning("ATR too small, skipping trade")
                return

            sl_dist = atr * self._pcfg.atr_sl_mult
            rr = self._pcfg.min_risk_reward
            tp_dist = sl_dist * rr

            min_sl_dist = self._pcfg.min_sl_pips * 0.0001
            if sl_dist < min_sl_dist:
                sl_dist = min_sl_dist
                tp_dist = sl_dist * rr

            if direction == "BUY":
                sl = entry_price - sl_dist
                tp = entry_price + tp_dist
            else:
                sl = entry_price + sl_dist
                tp = entry_price - tp_dist

            volume = self._calculate_position_size(entry_price, sl, g_factor)

            if hasattr(self.executor, "entry"):
                self.executor.entry(signal=direction, volume=volume, sl=sl, tp=tp)
            elif hasattr(self.executor, "open_position"):
                self.executor.open_position(
                    direction=direction, volume=volume, stop_loss=sl, take_profit=tp
                )
        except Exception as e:
            logger.error(f"V6 trade execution failed: {e}")

    def _calculate_position_size(
        self, entry_price: float, stop_loss: float, g_factor: float
    ) -> float:
        """Risk-based position sizing with g_factor confidence scaling."""
        balance = 10000.0
        if self.executor is not None:
            if hasattr(self.executor, "balance"):
                try:
                    balance = float(self.executor.balance)
                except Exception:
                    pass
            elif hasattr(self.executor, "get_account_info"):
                try:
                    info = self.executor.get_account_info()
                    balance = float(getattr(info, "balance", 10000.0))
                except Exception:
                    pass

        risk_pct = self._pcfg.base_risk_percent / 100.0
        risk_amount = balance * risk_pct

        sl_distance = abs(entry_price - stop_loss)
        if sl_distance < 1e-8:
            return 0.01

        pip_value = 10.0  # $10 per pip per lot for EURUSD
        sl_pips = sl_distance / 0.0001
        volume = risk_amount / (sl_pips * pip_value)

        # Scale by g_factor confidence (higher g → larger position)
        confidence_scale = 0.5 + g_factor  # g=0.52 → 1.02x, g=0.65 → 1.15x
        volume *= confidence_scale

        volume = max(0.01, min(volume, self._pcfg.max_lot))
        return round(volume, 2)

    # ── Helper Methods ───────────────────────────────────────────────────

    def _compute_atr(self, df: pd.DataFrame, period: int = None) -> float:
        period = period or self._pcfg.atr_period
        try:
            h = df["high"] if "high" in df.columns else df["High"]
            l = df["low"] if "low" in df.columns else df["Low"]
            c = df["close"] if "close" in df.columns else df["Close"]
            tr = pd.concat([
                h - l,
                (h - c.shift(1)).abs(),
                (l - c.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_val = float(tr.rolling(period, min_periods=1).mean().iloc[-1])
            return atr_val if np.isfinite(atr_val) else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _prep_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        d0 = df.copy()
        d0.columns = [str(c).lower().strip() for c in d0.columns]
        if "volume" not in d0.columns:
            if "tick_volume" in d0.columns:
                d0["volume"] = d0["tick_volume"]
            else:
                d0["volume"] = 0.0
        return d0

    @staticmethod
    def _extract_time(df: pd.DataFrame) -> Optional[datetime]:
        try:
            if df is not None and not df.empty:
                cols = [c.lower().strip() for c in df.columns]
                if "time" in cols:
                    col = df.columns[cols.index("time")]
                    return pd.to_datetime(df[col].iloc[-1]).to_pydatetime()
        except Exception:
            pass
        return datetime.utcnow()

    def _check_daily_reset(self):
        if self._current_time is None:
            return
        today = self._current_time.date()
        if self._last_trade_date != today:
            self._daily_trades = 0
            self._daily_pnl = 0.0
            self._last_trade_date = today

    def _sync_open_positions(self):
        self._open_positions = {}
        if self.executor is None:
            return
        try:
            if hasattr(self.executor, "get_open_positions"):
                try:
                    pos = self.executor.get_open_positions(
                        symbol=getattr(self.config, "symbol", None)
                    )
                except TypeError:
                    pos = self.executor.get_open_positions()
            elif hasattr(self.executor, "positions"):
                pos = list(getattr(self.executor, "positions") or [])
            else:
                pos = []
            for p in pos or []:
                if isinstance(p, dict):
                    ticket = p.get("ticket")
                    if ticket is not None:
                        self._open_positions[str(ticket)] = p
                else:
                    ticket = getattr(p, "ticket", None)
                    if ticket is not None:
                        self._open_positions[str(ticket)] = {
                            "ticket": ticket,
                            "direction": getattr(p, "direction", None),
                            "volume": getattr(p, "volume", None),
                        }
        except Exception:
            self._open_positions = {}

    def _has_open_direction(self, direction: str) -> bool:
        d = str(direction).upper()
        for p in (self._open_positions or {}).values():
            try:
                pdir = str(p.get("type") or p.get("direction") or "").upper()
                if pdir == d:
                    return True
            except Exception:
                continue
        return False

    # ── Backtest Support ─────────────────────────────────────────────────

    def precompute_features(self, data_by_tf: dict) -> None:
        """Pre-compute features for all TFs for backtest speedup."""
        if not self._lazy_init():
            return
        for tf_label, df_tf in data_by_tf.items():
            if df_tf is None or df_tf.empty:
                continue
            d0 = self._prep_df(df_tf)
            if d0 is None:
                continue
            for engine in (self._htf_engine, self._ttf_engine, self._ltf_engine):
                if engine is not None and engine._ready and engine.tf_label.upper() == str(tf_label).upper():
                    try:
                        t0 = _time.time()
                        engine._feature_pipeline.compute_features(d0)
                        elapsed = _time.time() - t0
                        logger.info(
                            f"V6: Pre-computed features for {engine.role}/{engine.tf_label} "
                            f"({len(d0)} bars) in {elapsed:.1f}s"
                        )
                    except Exception as e:
                        logger.warning(f"V6: Feature precompute failed for {tf_label}: {e}")
