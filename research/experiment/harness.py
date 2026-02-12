"""
Experiment Harness — runs all 6 variants through identical backtest execution.

Core loop per variant:
    for each bar:
        1. Shared regime detection
        2. Alpha head evaluation -> AlphaSignal
        3. MH-TCN filter (if present) -> g_factor
        4. P_final = P_alpha * g_factor
        5. Trade decision (P_final > min_probability gate)
        6. Execution with shared SL/TP/sizing logic
        7. Position management (SL/TP checks on open trades)

All variants use identical:
    - Data (same OHLCV bars)
    - Feature engineering (same pipeline)
    - Execution costs (spread, commission, slippage)
    - Risk sizing (ATR-based SL, fixed % risk)
    - Position management logic
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..interfaces import (
    AlphaSignal, Direction, MHTCNOutput, TradeRecord,
    VariantConfig, VariantResult,
)
from ..feature_pipeline import FeaturePipeline
from ..regime_detector import RegimeDetector
from ..calibrator import ProbabilityCalibrator

logger = logging.getLogger(__name__)


@dataclass
class _OpenPosition:
    """Internal position tracker."""
    ticket: int
    bar_index: int
    entry_time: Optional[pd.Timestamp]
    direction: Direction
    entry_price: float
    sl: float
    tp: float
    volume: float
    p_alpha: float
    p_final: float
    g_factor: float
    regime: str


class _VariantRunner:
    """Runs a single variant through the backtest loop."""

    def __init__(self, config: VariantConfig, pip_size: float = 0.0001):
        self.cfg = config
        self.pip_size = pip_size
        self.balance = config.initial_balance
        self.equity_curve: List[float] = []
        self.trades: List[TradeRecord] = []
        self.positions: List[_OpenPosition] = []
        self._ticket = 1000
        self._last_trade_bar = -999
        self._calibrator = ProbabilityCalibrator(method="passthrough")

    def reset(self):
        self.balance = self.cfg.initial_balance
        self.equity_curve = []
        self.trades = []
        self.positions = []
        self._ticket = 1000
        self._last_trade_bar = -999
        self.cfg.alpha_head.reset()
        self.cfg.mhtcn_filter.reset()
        self._calibrator.reset()

    def step(
        self,
        bar_idx: int,
        df_full: pd.DataFrame,
        feat_full: pd.DataFrame,
        regime,
        current_bar,
        *,
        high: float = 0.0,
        low: float = 0.0,
        close: float = 0.0,
        ts=None,
        atr_val: float = 0.0,
        _win: int = 100,
    ):
        """Process one bar: manage positions, evaluate signal, maybe open."""
        if close == 0.0:
            close = self._get_close(current_bar)
            high = self._get_high(current_bar)
            low = self._get_low(current_bar)
            ts = current_bar.name if isinstance(current_bar.name, pd.Timestamp) else None

        # 1. Check existing positions for SL/TP hit
        self._manage_positions(bar_idx, high, low, close, ts)

        # 2. Record equity
        unrealized = self._unrealized_pnl(close)
        self.equity_curve.append(self.balance + unrealized)

        # 3. Skip if at max open trades or in cooldown
        if len(self.positions) >= self.cfg.max_open_trades:
            return
        if (bar_idx - self._last_trade_bar) < self.cfg.cooldown_bars:
            return

        # 4. Alpha evaluation (pass bar_idx for vectorized pre-computed path)
        _has_precomputed = (hasattr(self.cfg.alpha_head, '_precomputed_p_bull')
                           and self.cfg.alpha_head._precomputed_p_bull is not None)
        if _has_precomputed:
            signal = self.cfg.alpha_head.evaluate(None, None, regime, bar_idx=bar_idx)
        else:
            s = max(0, bar_idx - _win)
            signal = self.cfg.alpha_head.evaluate(
                df_full.iloc[s:bar_idx + 1], feat_full.iloc[s:bar_idx + 1], regime)

        if not signal.is_trade:
            return

        # 5. MH-TCN filter
        mf = self.cfg.mhtcn_filter
        _mhtcn_precomputed = (hasattr(mf, '_precomputed_rows')
                              and mf._precomputed_rows is not None)
        if _mhtcn_precomputed:
            # Buffer already fed by harness loop; run model directly
            mhtcn_out = self._run_mhtcn_from_buffer(mf)
        else:
            s = max(0, bar_idx - _win)
            df_window = df_full.iloc[s:bar_idx + 1]
            feat_window = feat_full.iloc[s:bar_idx + 1]
            mhtcn_out = mf.filter(signal, df_window, feat_window)

        # 6. Compute final probability
        p_alpha = signal.probability
        g = mhtcn_out.g_factor
        # Softened g: floor at 0.3 so MH-TCN modulates but doesn't kill signals
        g_soft = 0.3 + 0.7 * g
        p_final = p_alpha * g_soft
        p_final = self._calibrator.calibrate(p_final)
        # Track g_factor diagnostics
        if not hasattr(self, '_g_diag'):
            self._g_diag = []
        self._g_diag.append((g, g_soft, p_alpha, p_final))

        if p_final < self.cfg.min_probability:
            return

        # 7. Compute SL/TP from ATR
        atr = atr_val if atr_val > 1e-8 else self._compute_atr(df_window)
        if atr < 1e-8:
            return

        sl_dist = atr * self.cfg.atr_sl_mult
        tp_dist = sl_dist * self.cfg.min_rr  # minimum R:R

        if signal.direction == Direction.LONG:
            sl_price = close - sl_dist
            tp_price = close + tp_dist
        else:
            sl_price = close + sl_dist
            tp_price = close - tp_dist

        # 8. Position sizing (fixed fractional risk)
        risk_amount = self.balance * self.cfg.risk_per_trade
        sl_pips = sl_dist / self.pip_size
        if sl_pips < 1:
            return
        volume = risk_amount / (sl_pips * self.cfg.pip_value)
        volume = round(max(0.01, min(volume, 1.0)), 2)

        # 9. Apply spread cost at entry
        spread_cost = self.cfg.spread_pips * self.pip_size
        if signal.direction == Direction.LONG:
            entry_price = close + spread_cost / 2
        else:
            entry_price = close - spread_cost / 2

        # 10. Commission
        commission = volume * self.cfg.commission_per_lot
        self.balance -= commission

        # 11. Open position
        self._ticket += 1
        pos = _OpenPosition(
            ticket=self._ticket,
            bar_index=bar_idx,
            entry_time=ts,
            direction=signal.direction,
            entry_price=entry_price,
            sl=sl_price,
            tp=tp_price,
            volume=volume,
            p_alpha=p_alpha,
            p_final=p_final,
            g_factor=g,
            regime=regime.value,
        )
        self.positions.append(pos)
        self._last_trade_bar = bar_idx

    def _manage_positions(self, bar_idx, high, low, close, ts):
        """Check SL/TP hits and close positions."""
        closed = []
        for pos in self.positions:
            hit_sl = False
            hit_tp = False
            exit_price = close

            if pos.direction == Direction.LONG:
                if low <= pos.sl:
                    hit_sl = True
                    exit_price = pos.sl
                elif high >= pos.tp:
                    hit_tp = True
                    exit_price = pos.tp
            else:  # SHORT
                if high >= pos.sl:
                    hit_sl = True
                    exit_price = pos.sl
                elif low <= pos.tp:
                    hit_tp = True
                    exit_price = pos.tp

            if hit_sl or hit_tp:
                pnl = self._calc_pnl(pos, exit_price)
                self.balance += pnl
                forward_label = 1 if pnl > 0 else 0

                self.trades.append(TradeRecord(
                    variant_id=self.cfg.variant_id.value,
                    bar_index=pos.bar_index,
                    entry_time=pos.entry_time,
                    exit_time=ts,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    sl=pos.sl,
                    tp=pos.tp,
                    volume=pos.volume,
                    pnl=pnl,
                    p_alpha=pos.p_alpha,
                    p_final=pos.p_final,
                    g_factor=pos.g_factor,
                    regime=pos.regime,
                    forward_label=forward_label,
                ))

                # Update calibrator with outcome
                self._calibrator.update(pos.p_final, forward_label)
                closed.append(pos)

        for pos in closed:
            self.positions.remove(pos)

    def _calc_pnl(self, pos: _OpenPosition, exit_price: float) -> float:
        if pos.direction == Direction.LONG:
            pips = (exit_price - pos.entry_price) / self.pip_size
        else:
            pips = (pos.entry_price - exit_price) / self.pip_size
        return pips * pos.volume * self.cfg.pip_value

    def _unrealized_pnl(self, current_price: float) -> float:
        total = 0.0
        for pos in self.positions:
            total += self._calc_pnl(pos, current_price)
        return total

    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        h = df["high"] if "high" in df.columns else df["High"]
        l = df["low"] if "low" in df.columns else df["Low"]
        c = df["close"] if "close" in df.columns else df["Close"]
        if len(h) < period + 1:
            return 0.0
        tr = pd.concat([
            h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if not np.isnan(atr) else 0.0

    def _run_mhtcn_from_buffer(self, mf) -> MHTCNOutput:
        """Run MH-TCN model directly from pre-fed buffer (no DataFrame needed)."""
        if len(mf._buffer) < mf.seq_len:
            return MHTCNOutput(
                g_factor=1.0,
                signal_survival_prob=0.5,
                confidence_decay=0.0,
                regime_validity=1.0,
            )
        try:
            mf._ensure_model()
            import torch
            window = np.array(list(mf._buffer), dtype=np.float32)
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(mf.device)
            with torch.no_grad():
                out = mf._model(x)
            g = float(out["g_factor"].item())
            surv = float(out["survival"].item())
            valid = float(out["validity"].item())
            return MHTCNOutput(
                g_factor=float(np.clip(g, 0.01, 1.0)),
                signal_survival_prob=surv,
                confidence_decay=float(1.0 - surv),
                regime_validity=valid,
            )
        except Exception as e:
            logger.debug(f"MH-TCN buffer run error: {e}")
            return MHTCNOutput(
                g_factor=1.0,
                signal_survival_prob=0.5,
                confidence_decay=0.0,
                regime_validity=1.0,
            )

    def _get_close(self, bar):
        return float(bar.get("close", bar.get("Close", 0)))

    def _get_high(self, bar):
        return float(bar.get("high", bar.get("High", 0)))

    def _get_low(self, bar):
        return float(bar.get("low", bar.get("Low", 0)))


class ExperimentHarness:
    """Runs multiple variants through the same data with identical execution.

    Usage:
        harness = ExperimentHarness(df_ohlcv)
        results = harness.run(variants, warmup=200)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pip_size: float = 0.0001,
        feature_pipeline: Optional[FeaturePipeline] = None,
        regime_detector: Optional[RegimeDetector] = None,
    ):
        self.df = df.copy()
        self.pip_size = pip_size
        self.feature_pipeline = feature_pipeline or FeaturePipeline()
        self.regime_detector = regime_detector or RegimeDetector()

        # Normalize column names to lowercase
        self.df.columns = [c.lower() for c in self.df.columns]

        # Pre-compute shared features once
        logger.info("Computing shared features...")
        t0 = time.time()
        self.features = self.feature_pipeline.compute_features(self.df)
        logger.info(f"Features computed: {self.features.shape[1]} cols in {time.time()-t0:.1f}s")

        # Inject category mapping into any Alpha2 heads later
        self._cat_map = self.feature_pipeline.get_category_columns()

    def run(
        self,
        variants: List[VariantConfig],
        warmup: int = 200,
        progress_interval: int = 1000,
    ) -> List[VariantResult]:
        """Execute all variants bar-by-bar on the same data.

        Args:
            variants: List of VariantConfig to evaluate.
            warmup: Number of initial bars to skip (feature lookback).
            progress_interval: Log progress every N bars.

        Returns:
            List of VariantResult, one per variant.
        """
        n_bars = len(self.df)
        logger.info(f"Running {len(variants)} variants on {n_bars} bars (warmup={warmup})")

        # Initialize runners
        runners: List[_VariantRunner] = []
        for vc in variants:
            runner = _VariantRunner(vc, pip_size=self.pip_size)
            runner.reset()
            # Inject category column mapping for Alpha2
            if hasattr(vc.alpha_head, 'set_category_columns'):
                vc.alpha_head.set_category_columns(self._cat_map)
            runners.append(runner)

        # Pre-compute regime for each bar
        logger.info("Pre-computing regimes...")
        regimes = self.regime_detector.detect_series(self.df, self.features)

        # Pre-compute AlphaV2 scores vectorially if applicable
        for runner in runners:
            ah = runner.cfg.alpha_head
            if hasattr(ah, 'precompute_all_scores') and hasattr(ah, '_precomputed_p_bull'):
                if ah._precomputed_p_bull is None:
                    ah.precompute_all_scores(self.features, regimes)

        # Pre-extract numpy arrays for fast per-bar access
        _highs = self.df["high"].values if "high" in self.df.columns else self.df["High"].values
        _lows = self.df["low"].values if "low" in self.df.columns else self.df["Low"].values
        _closes = self.df["close"].values if "close" in self.df.columns else self.df["Close"].values
        _has_dt_idx = isinstance(self.df.index, pd.DatetimeIndex)

        # Pre-compute ATR as numpy array
        _tr = np.maximum(
            _highs - _lows,
            np.maximum(np.abs(_highs - np.roll(_closes, 1)),
                       np.abs(_lows - np.roll(_closes, 1)))
        )
        _tr[0] = _highs[0] - _lows[0]
        _atr = pd.Series(_tr).rolling(14).mean().values

        # Pre-compute MH-TCN buffer rows for each runner
        for runner in runners:
            mf = runner.cfg.mhtcn_filter
            ah = runner.cfg.alpha_head
            if hasattr(mf, 'precompute_buffer_rows') and hasattr(ah, '_precomputed_p_bull'):
                if ah._precomputed_p_bull is not None:
                    mf.precompute_buffer_rows(ah, regimes, _closes, _highs, _lows)

        # Feed MH-TCN buffer during warmup so it has full context at trading start
        for runner in runners:
            mf = runner.cfg.mhtcn_filter
            if hasattr(mf, 'feed_bar') and hasattr(mf, '_precomputed_rows') and mf._precomputed_rows is not None:
                for i in range(warmup):
                    mf.feed_bar(i)
                logger.info(f"MH-TCN buffer pre-filled with {min(warmup, mf.seq_len)} warmup bars")

        # Main loop
        t0 = time.time()
        _win = 100  # reduced window for MH-TCN price-derived features

        # Store references for lazy window builder
        _df_ref = self.df
        _feat_ref = self.features

        for i in range(warmup, n_bars):
            regime = regimes.iloc[i]
            h_i = float(_highs[i])
            l_i = float(_lows[i])
            c_i = float(_closes[i])
            ts_i = _df_ref.index[i] if _has_dt_idx else None
            atr_i = float(_atr[i]) if not np.isnan(_atr[i]) else 0.0

            for runner in runners:
                # Feed MH-TCN buffer every bar (O(1) from pre-computed rows)
                mf = runner.cfg.mhtcn_filter
                if hasattr(mf, 'feed_bar') and hasattr(mf, '_precomputed_rows') and mf._precomputed_rows is not None:
                    mf.feed_bar(i)

                runner.step(
                    i, _df_ref, _feat_ref, regime, None,
                    high=h_i, low=l_i, close=c_i, ts=ts_i, atr_val=atr_i,
                    _win=_win,
                )

            if (i - warmup) % progress_interval == 0 and i > warmup:
                elapsed = time.time() - t0
                bars_done = i - warmup
                rate = bars_done / max(elapsed, 0.01)
                logger.info(
                    f"  bar {i}/{n_bars} ({bars_done} processed, {rate:.0f} bars/s)"
                )

        elapsed = time.time() - t0
        logger.info(f"Backtest complete in {elapsed:.1f}s")

        # Log g_factor diagnostics
        for runner in runners:
            if hasattr(runner, '_g_diag') and runner._g_diag:
                diag = runner._g_diag
                g_raw = [d[0] for d in diag]
                g_soft = [d[1] for d in diag]
                p_alphas = [d[2] for d in diag]
                p_finals = [d[3] for d in diag]
                logger.info(
                    f"  g_factor diagnostics ({len(diag)} trade signals): "
                    f"g_raw=[{np.min(g_raw):.3f}, {np.mean(g_raw):.3f}, {np.max(g_raw):.3f}] "
                    f"g_soft=[{np.min(g_soft):.3f}, {np.mean(g_soft):.3f}, {np.max(g_soft):.3f}] "
                    f"p_alpha=[{np.min(p_alphas):.3f}, {np.mean(p_alphas):.3f}, {np.max(p_alphas):.3f}] "
                    f"p_final=[{np.min(p_finals):.3f}, {np.mean(p_finals):.3f}, {np.max(p_finals):.3f}]"
                )

        # Collect results
        results = []
        for runner in runners:
            # Close any remaining open positions at last close
            last_close = self._get_close(self.df.iloc[-1])
            last_ts = self.df.index[-1] if isinstance(self.df.index, pd.DatetimeIndex) else None
            for pos in list(runner.positions):
                pnl = runner._calc_pnl(pos, last_close)
                runner.balance += pnl
                runner.trades.append(TradeRecord(
                    variant_id=runner.cfg.variant_id.value,
                    bar_index=pos.bar_index,
                    entry_time=pos.entry_time,
                    exit_time=last_ts,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=last_close,
                    sl=pos.sl, tp=pos.tp, volume=pos.volume,
                    pnl=pnl, p_alpha=pos.p_alpha, p_final=pos.p_final,
                    g_factor=pos.g_factor, regime=pos.regime,
                    forward_label=1 if pnl > 0 else 0,
                ))
            runner.positions.clear()
            runner.equity_curve.append(runner.balance)

            results.append(VariantResult(
                variant_id=runner.cfg.variant_id.value,
                trades=runner.trades,
                equity_curve=np.array(runner.equity_curve),
            ))

        return results

    def _get_close(self, bar):
        return float(bar.get("close", bar.get("Close", 0)))
