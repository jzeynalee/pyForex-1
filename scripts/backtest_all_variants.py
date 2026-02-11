#!/usr/bin/env python
"""
Comprehensive Backtesting Script for All Model Variants
========================================================

Tests all trained model variants across all profiles (SCALP, INTRADAY, SWING)
and generates a comparison report with key metrics for variant selection.

Variants tested:
    1. Unified3TF          — ProbabilisticAlphaFactory + MH-TCN (primary pipeline)
    2. Unified3TF_fast     — Same but with fast_backtest=True (key features only)
    3. NeuralHybrid        — EnhancedDecisionEngine + TCN + Meta-labeling
    4. DecisionFusion      — DecisionFusionLayer (PA + TCN fusion)
    5. FusionNet           — FusionNet gated late fusion (PA + TCN)

Usage:
    python -m scripts.backtest_all_variants --profiles INTRADAY
    python -m scripts.backtest_all_variants --profiles SCALP INTRADAY SWING
    python -m scripts.backtest_all_variants --variants Unified3TF NeuralHybrid
    python -m scripts.backtest_all_variants --max-bars 5000
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from trading.bot import BacktestBot
from trading.backtest import BacktestExecutor, BacktestConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s"
logging.basicConfig(level=logging.WARNING, format=LOG_FMT, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("backtest_all")
logger.setLevel(logging.INFO)

# Suppress noisy sub-loggers during batch runs
for noisy in (
    "alpha_factory", "strategies", "trading", "inference",
    "risk_management", "trend_detection", "models",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ASSETS_DIR = Path("E:/pyProject/pyForex-assets")
DATA_DIR = Path("E:/pyProject/data/raw")

PROFILE_DATA_MAP = {
    "SCALP": "EURUSD_M5_latest.csv",
    "INTRADAY": "EURUSD_H1_latest.csv",
    "SWING": "EURUSD_H1_latest.csv",
}

PROFILE_BASE_TF = {
    "SCALP": "M5",
    "INTRADAY": "H1",
    "SWING": "H1",
}

ALL_PROFILES = ["SCALP", "INTRADAY", "SWING"]
ALL_VARIANTS = [
    "Unified3TF",
    "Unified3TF_fast",
    "NeuralHybrid",
    "DecisionFusion",
    "FusionNet",
]

INITIAL_BALANCE = 10_000.0


# ---------------------------------------------------------------------------
# Metrics calculator
# ---------------------------------------------------------------------------
@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics for comparison."""
    variant: str = ""
    profile: str = ""
    initial_balance: float = 10_000.0
    final_balance: float = 10_000.0
    total_return_pct: float = 0.0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_amt: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_rr_ratio: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_pnl: float = 0.0
    buy_signals: int = 0
    sell_signals: int = 0
    hold_signals: int = 0
    bars_processed: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""


def compute_metrics(
    variant: str,
    profile: str,
    result: dict,
    signals: list,
    elapsed: float,
) -> BacktestMetrics:
    """Compute comprehensive metrics from backtest result."""
    m = BacktestMetrics(variant=variant, profile=profile)
    m.initial_balance = INITIAL_BALANCE
    m.final_balance = float(result.get("final_balance", INITIAL_BALANCE))
    m.total_pnl = m.final_balance - m.initial_balance
    m.total_return_pct = (m.total_pnl / m.initial_balance) * 100.0
    m.elapsed_seconds = elapsed

    # Signal counts
    for s in signals:
        sig = s.get("signal")
        if sig == "BUY":
            m.buy_signals += 1
        elif sig == "SELL":
            m.sell_signals += 1
        else:
            m.hold_signals += 1
    m.bars_processed = len(signals)

    # Trade-level metrics
    trades = result.get("trades", [])
    m.total_trades = len(trades)

    if trades:
        pnls = [float(t.get("pnl", 0) or 0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        m.winning_trades = len(wins)
        m.losing_trades = len(losses)
        m.win_rate = m.winning_trades / m.total_trades if m.total_trades else 0.0

        total_win_amt = sum(wins) if wins else 0.0
        total_loss_amt = abs(sum(losses)) if losses else 0.0
        m.profit_factor = total_win_amt / total_loss_amt if total_loss_amt > 0 else float("inf")

        m.avg_win = np.mean(wins) if wins else 0.0
        m.avg_loss = np.mean(losses) if losses else 0.0
        m.avg_rr_ratio = abs(m.avg_win / m.avg_loss) if m.avg_loss != 0 else 0.0
        m.largest_win = max(wins) if wins else 0.0
        m.largest_loss = min(losses) if losses else 0.0
        m.avg_trade_pnl = np.mean(pnls) if pnls else 0.0

    # Equity curve metrics
    equity_curve = _build_equity_curve(signals, trades)
    if len(equity_curve) > 1:
        eq = np.array(equity_curve)
        returns = np.diff(eq) / eq[:-1]
        returns = returns[np.isfinite(returns)]

        if len(returns) > 1:
            # Sharpe (annualized, assume hourly bars as proxy)
            bars_per_year = 252 * 24
            mean_r = float(np.mean(returns))
            std_r = float(np.std(returns))
            m.sharpe_ratio = (mean_r * bars_per_year - 0.02) / (std_r * np.sqrt(bars_per_year)) if std_r > 1e-10 else 0.0

            # Sortino
            downside = returns[returns < 0]
            downside_std = float(np.std(downside)) if len(downside) > 1 else 1e-10
            m.sortino_ratio = (mean_r * bars_per_year - 0.02) / (downside_std * np.sqrt(bars_per_year)) if downside_std > 1e-10 else 0.0

        # Max Drawdown
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        m.max_drawdown_pct = float(np.min(dd)) * 100.0
        m.max_drawdown_amt = float(np.min(eq - peak))

        # Calmar
        years = max(len(equity_curve) / (252 * 24), 1e-6)
        annual_return = m.total_return_pct / years
        m.calmar_ratio = annual_return / abs(m.max_drawdown_pct) if abs(m.max_drawdown_pct) > 0.01 else 0.0

    return m


def _build_equity_curve(signals: list, trades: list) -> List[float]:
    """Reconstruct approximate equity curve from trade history."""
    curve = [INITIAL_BALANCE]
    balance = INITIAL_BALANCE
    for t in trades:
        pnl = float(t.get("pnl", 0) or 0)
        balance += pnl
        curve.append(balance)
    return curve


# ---------------------------------------------------------------------------
# Strategy factories
# ---------------------------------------------------------------------------

# Cache for pre-computed stores so we don't rebuild for Unified3TF vs Unified3TF_fast
_precomputed_cache: Dict[str, type] = {}


def _create_unified3tf_strategy_class(
    profile: str, fast: bool = False, full_df: pd.DataFrame = None,
):
    """Create a Unified3TFStrategy class preconfigured for the profile.

    When *full_df* is supplied, uses the fast-backtest variant that
    pre-computes features once for the entire dataset (100-300x speedup).
    """
    cache_key = f"{profile}_{'fast' if fast else 'full'}_{id(full_df)}"
    if cache_key in _precomputed_cache:
        return _precomputed_cache[cache_key]

    if full_df is not None:
        from strategies.unified_3tf_fast_backtest import create_fast_backtest_strategy
        cls = create_fast_backtest_strategy(
            full_df=full_df,
            profile=profile,
            use_fast_features=fast,
        )
        _precomputed_cache[cache_key] = cls
        return cls

    # Fallback: original per-bar strategy (very slow for backtests)
    from strategies.unified_3tf_strategy import Unified3TFStrategy, Unified3TFConfig

    class _Configured(Unified3TFStrategy):
        def __init__(self, data_provider=None, executor=None, **kwargs):
            cfg = Unified3TFConfig(profile=profile)
            cfg.fast_backtest = fast
            super().__init__(config=cfg, data_provider=data_provider, executor=executor)

    _Configured.__name__ = f"Unified3TF_{'fast_' if fast else ''}{profile}"
    return _Configured


def _create_neural_hybrid_strategy_class(profile: str):
    """Create a NeuralHybridStrategy class preconfigured for the profile."""
    from strategies.neural_hybrid import NeuralHybridStrategy, StrategyConfig

    class _Configured(NeuralHybridStrategy):
        def __init__(self, data_provider=None, executor=None, **kwargs):
            cfg = StrategyConfig(profile=profile)
            super().__init__(config=cfg, data_provider=data_provider, executor=executor)

    _Configured.__name__ = f"NeuralHybrid_{profile}"
    return _Configured


def _create_fusion_strategy_class(profile: str, fusion_type: str):
    """Create a strategy wrapper around DecisionFusion or FusionNet models."""
    import torch
    from strategies.base import Strategy

    model_dir_map = {
        "DecisionFusion": ASSETS_DIR / "models" / "fusion" / "decision_fusion",
        "FusionNet": ASSETS_DIR / "models" / "fusion" / "fusion_net",
    }

    class FusionStrategy(Strategy):
        """Lightweight strategy that loads a fusion model and generates signals."""

        def __init__(self, data_provider=None, executor=None, **kwargs):
            self.data_provider = data_provider
            self.executor = executor
            self.name = f"{fusion_type}_{profile}"
            self._model = None
            self._device = "cpu"
            self._profile = profile
            self._fusion_type = fusion_type
            self._initialized = False
            self._feature_engineer = None
            self._pa_extractor = None
            self._seq_length = 60

        def _load_model(self):
            """Load the fusion model weights."""
            base_dir = model_dir_map[self._fusion_type]
            weights_path = base_dir / profile.lower() / "best_model.pt"

            if not weights_path.exists():
                logger.warning(f"Weights not found: {weights_path}")
                return False

            try:
                state = torch.load(weights_path, map_location=self._device, weights_only=False)

                if self._fusion_type == "DecisionFusion":
                    from models.decision_fusion import DecisionFusionLayer
                    self._model = DecisionFusionLayer(
                        price_action_dim=44, tcn_dim=64,
                        hidden_dim=256, num_classes=3,
                    )
                else:
                    from models.fusion import FusionNet
                    self._model = FusionNet(
                        seq_dim=64, price_action_dim=44,
                        hidden_dim=256, num_classes=3,
                    )

                self._model.load_state_dict(state, strict=False)
                self._model.eval()
                return True
            except Exception as e:
                logger.error(f"Failed to load {self._fusion_type} model: {e}")
                return False

        def _get_tcn_features(self, df: pd.DataFrame) -> Optional[torch.Tensor]:
            """Extract TCN features from data using feature engineering."""
            try:
                if self._feature_engineer is None:
                    from alpha_factory.features_engineering import FeatureEngineerOptimized
                    self._feature_engineer = FeatureEngineerOptimized()

                d = df.copy()
                d.columns = [str(c).lower().strip() for c in d.columns]
                if "volume" not in d.columns:
                    d["volume"] = d.get("tick_volume", 0.0)

                feats = self._feature_engineer.generate_features(d, batch_processing=False)
                if feats is None or feats.empty:
                    return None

                numeric = feats.select_dtypes(include=[np.number])
                vals = numeric.iloc[-1].values.astype(np.float32)
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

                # Project to 64-dim via simple truncation/padding
                if len(vals) >= 64:
                    tcn_vec = vals[:64]
                else:
                    tcn_vec = np.zeros(64, dtype=np.float32)
                    tcn_vec[: len(vals)] = vals

                return torch.tensor(tcn_vec, dtype=torch.float32).unsqueeze(0)
            except Exception as e:
                logger.debug(f"TCN feature extraction error: {e}")
                return None

        def _get_pa_features(self, df: pd.DataFrame) -> Optional[torch.Tensor]:
            """Extract Price Action features."""
            try:
                if self._pa_extractor is None:
                    from models.price_action_pattern import PriceActionPatternExtractor
                    self._pa_extractor = PriceActionPatternExtractor(window_size=120)

                d = df.copy()
                d.columns = [str(c).lower().strip() for c in d.columns]
                if "volume" not in d.columns:
                    d["volume"] = d.get("tick_volume", 0.0)

                pa_vec = self._pa_extractor.extract_features(d)
                if pa_vec is None:
                    return None

                if isinstance(pa_vec, np.ndarray):
                    pa_vec = np.nan_to_num(pa_vec.astype(np.float32), nan=0.0)
                else:
                    pa_vec = np.zeros(44, dtype=np.float32)

                if len(pa_vec) < 44:
                    padded = np.zeros(44, dtype=np.float32)
                    padded[: len(pa_vec)] = pa_vec
                    pa_vec = padded

                return torch.tensor(pa_vec[:44], dtype=torch.float32).unsqueeze(0)
            except Exception as e:
                logger.debug(f"PA feature extraction error: {e}")
                return None

        def on_bar(self, df: pd.DataFrame) -> Optional[str]:
            if df is None or len(df) < 120:
                return None

            if not self._initialized:
                if not self._load_model():
                    return None
                self._initialized = True

            if self._model is None:
                return None

            tcn_feat = self._get_tcn_features(df)
            pa_feat = self._get_pa_features(df)

            if tcn_feat is None or pa_feat is None:
                return None

            try:
                with torch.no_grad():
                    if self._fusion_type == "DecisionFusion":
                        pa_conf = torch.tensor([0.7], dtype=torch.float32)
                        tcn_stab = torch.tensor([[0.7]], dtype=torch.float32)
                        output = self._model(
                            price_action_features=pa_feat.unsqueeze(1),
                            price_action_confidence=pa_conf,
                            tcn_features=tcn_feat,
                            tcn_stability=tcn_stab,
                        )
                        logits = output.direction_logits
                    else:
                        logits = self._model(tcn_feat, pa_feat)

                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                    pred = int(torch.argmax(probs).item())
                    conf = float(probs[pred].item())

                    # pred: 0=BEAR, 1=NEUTRAL, 2=BULL
                    if conf < 0.55:
                        return None

                    direction = None
                    if pred == 2:
                        direction = "BUY"
                    elif pred == 0:
                        direction = "SELL"
                    else:
                        return None

                    # Execute trade via executor
                    if self.executor is not None:
                        price = float(df["close"].iloc[-1])
                        atr = self._estimate_atr(df)
                        sl_dist = max(atr * 2.0, 0.0010)

                        if direction == "BUY":
                            sl = price - sl_dist
                            tp = price + sl_dist * 2.0
                        else:
                            sl = price + sl_dist
                            tp = price - sl_dist * 2.0

                        self.executor.entry(
                            signal=direction,
                            volume=0.1,
                            sl=sl,
                            tp=tp,
                        )

                    return direction
            except Exception as e:
                logger.debug(f"Fusion forward error: {e}")
                return None

        def _estimate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
            """Simple ATR estimation."""
            try:
                d = df.tail(period + 1).copy()
                d.columns = [str(c).lower().strip() for c in d.columns]
                highs = d["high"].values
                lows = d["low"].values
                closes = d["close"].values
                tr = np.maximum(
                    highs[1:] - lows[1:],
                    np.maximum(
                        np.abs(highs[1:] - closes[:-1]),
                        np.abs(lows[1:] - closes[:-1]),
                    ),
                )
                return float(np.mean(tr)) if len(tr) > 0 else 0.0020
            except Exception:
                return 0.0020

    FusionStrategy.__name__ = f"{fusion_type}_{profile}"
    return FusionStrategy


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(profile: str, max_bars: Optional[int] = None) -> pd.DataFrame:
    """Load CSV data for the given profile."""
    csv_name = PROFILE_DATA_MAP[profile]
    csv_path = DATA_DIR / csv_name

    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    logger.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [str(c).lower().strip() for c in df.columns]

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

    if "volume" not in df.columns:
        if "tick_volume" in df.columns:
            df["volume"] = df["tick_volume"]
        else:
            df["volume"] = 100

    if max_bars and len(df) > max_bars:
        df = df.tail(max_bars).reset_index(drop=True)

    logger.info(f"  Loaded {len(df):,} bars for {profile}")
    return df


# ---------------------------------------------------------------------------
# Single backtest runner
# ---------------------------------------------------------------------------

def run_single_backtest(
    variant: str,
    profile: str,
    df: pd.DataFrame,
    max_bars: Optional[int] = None,
) -> BacktestMetrics:
    """Run a single backtest for a variant+profile combination."""
    logger.info(f"  Running {variant} / {profile} ({len(df):,} bars)...")

    try:
        # Create strategy class
        if variant == "Unified3TF":
            strategy_cls = _create_unified3tf_strategy_class(profile, fast=False, full_df=df)
        elif variant == "Unified3TF_fast":
            strategy_cls = _create_unified3tf_strategy_class(profile, fast=True, full_df=df)
        elif variant == "NeuralHybrid":
            strategy_cls = _create_neural_hybrid_strategy_class(profile)
        elif variant == "DecisionFusion":
            strategy_cls = _create_fusion_strategy_class(profile, "DecisionFusion")
        elif variant == "FusionNet":
            strategy_cls = _create_fusion_strategy_class(profile, "FusionNet")
        else:
            m = BacktestMetrics(variant=variant, profile=profile, error=f"Unknown variant: {variant}")
            return m

        base_tf = PROFILE_BASE_TF[profile]

        t0 = time.time()
        bot = BacktestBot(
            data=df,
            strategy_class=strategy_cls,
            initial_balance=INITIAL_BALANCE,
            profile=profile,
            base_timeframe=base_tf,
        )
        result = bot.run()
        elapsed = time.time() - t0

        signals = result.get("signals", [])
        metrics = compute_metrics(variant, profile, result, signals, elapsed)

        # Rejection diagnostics
        if metrics.total_trades == 0 and hasattr(bot, 'strategy'):
            strat = bot.strategy
            rej_stage = getattr(strat, 'last_rejection_stage', '')
            rej_reason = getattr(strat, 'last_rejection_reason', '')
            sig_counts = {}
            for s in signals:
                sig_val = s.get('signal')
                sig_counts[sig_val] = sig_counts.get(sig_val, 0) + 1
            logger.warning(
                f"    0 trades! Last rejection: [{rej_stage}] {rej_reason[:120]}  "
                f"signal_distribution={sig_counts}"
            )

        logger.info(
            f"    Done in {elapsed:.1f}s | trades={metrics.total_trades} | "
            f"return={metrics.total_return_pct:+.2f}% | WR={metrics.win_rate:.1%} | "
            f"PF={metrics.profit_factor:.2f} | DD={metrics.max_drawdown_pct:.2f}%"
        )
        return metrics

    except Exception as e:
        logger.error(f"    ERROR in {variant}/{profile}: {e}", exc_info=False)
        m = BacktestMetrics(variant=variant, profile=profile, error=str(e)[:200])
        return m


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(all_metrics: List[BacktestMetrics], output_dir: Path):
    """Generate comparison report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Console report
    print("\n" + "=" * 120)
    print("  COMPREHENSIVE BACKTEST COMPARISON REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)

    # Group by profile
    profiles_seen = sorted(set(m.profile for m in all_metrics))

    for profile in profiles_seen:
        profile_metrics = [m for m in all_metrics if m.profile == profile]
        print(f"\n{'─' * 120}")
        print(f"  PROFILE: {profile}")
        print(f"{'─' * 120}")

        header = (
            f"  {'Variant':<22s} │ {'Return%':>8s} │ {'MaxDD%':>8s} │ {'Sharpe':>7s} │ "
            f"{'Sortino':>7s} │ {'Trades':>6s} │ {'WinRate':>7s} │ {'PF':>7s} │ "
            f"{'AvgWin':>8s} │ {'AvgLoss':>8s} │ {'AvgRR':>5s} │ {'Time(s)':>7s} │ {'Status':>10s}"
        )
        print(header)
        print(f"  {'─' * 21}─┼{'─' * 9}─┼{'─' * 9}─┼{'─' * 8}─┼{'─' * 8}─┼{'─' * 7}─┼{'─' * 8}─┼{'─' * 8}─┼{'─' * 9}─┼{'─' * 9}─┼{'─' * 6}─┼{'─' * 8}─┼{'─' * 10}─")

        for m in sorted(profile_metrics, key=lambda x: x.total_return_pct, reverse=True):
            status = "OK" if not m.error else f"ERR"
            pf_str = f"{m.profit_factor:.2f}" if m.profit_factor < 1000 else "inf"
            print(
                f"  {m.variant:<22s} │ {m.total_return_pct:>+8.2f} │ {m.max_drawdown_pct:>8.2f} │ "
                f"{m.sharpe_ratio:>7.2f} │ {m.sortino_ratio:>7.2f} │ {m.total_trades:>6d} │ "
                f"{m.win_rate:>6.1%} │ {pf_str:>7s} │ {m.avg_win:>8.2f} │ {m.avg_loss:>8.2f} │ "
                f"{m.avg_rr_ratio:>5.2f} │ {m.elapsed_seconds:>7.1f} │ {status:>10s}"
            )

    # Best variant per profile
    print(f"\n{'=' * 120}")
    print("  BEST VARIANT PER PROFILE (by Return%)")
    print(f"{'=' * 120}")
    for profile in profiles_seen:
        valid = [m for m in all_metrics if m.profile == profile and not m.error]
        if valid:
            best = max(valid, key=lambda x: x.total_return_pct)
            print(f"  {profile:<12s}: {best.variant:<22s} ({best.total_return_pct:+.2f}%  WR={best.win_rate:.1%}  PF={best.profit_factor:.2f}  Sharpe={best.sharpe_ratio:.2f})")
        else:
            print(f"  {profile:<12s}: No valid results")

    # Risk-adjusted ranking (composite score)
    print(f"\n{'=' * 120}")
    print("  COMPOSITE RANKING (weighted: 30% Return + 25% Sharpe + 20% PF + 15% WinRate + 10% -MaxDD)")
    print(f"{'=' * 120}")

    scored = []
    valid_all = [m for m in all_metrics if not m.error and m.total_trades > 0]
    if valid_all:
        # Normalize metrics to [0, 1] range
        returns = [m.total_return_pct for m in valid_all]
        sharpes = [m.sharpe_ratio for m in valid_all]
        pfs = [min(m.profit_factor, 10.0) for m in valid_all]
        wrs = [m.win_rate for m in valid_all]
        dds = [-m.max_drawdown_pct for m in valid_all]  # Less DD = better

        def _norm(vals):
            vmin, vmax = min(vals), max(vals)
            rng = vmax - vmin
            if rng < 1e-10:
                return [0.5] * len(vals)
            return [(v - vmin) / rng for v in vals]

        n_ret = _norm(returns)
        n_sha = _norm(sharpes)
        n_pf = _norm(pfs)
        n_wr = _norm(wrs)
        n_dd = _norm(dds)

        for i, m in enumerate(valid_all):
            composite = (
                0.30 * n_ret[i]
                + 0.25 * n_sha[i]
                + 0.20 * n_pf[i]
                + 0.15 * n_wr[i]
                + 0.10 * n_dd[i]
            )
            scored.append((composite, m))

        scored.sort(key=lambda x: x[0], reverse=True)

        print(f"  {'Rank':<5s} {'Variant':<22s} {'Profile':<12s} {'Score':>6s} {'Return%':>9s} {'Sharpe':>7s} {'PF':>7s} {'WR':>7s} {'MaxDD%':>8s}")
        print(f"  {'─' * 5} {'─' * 22} {'─' * 12} {'─' * 6} {'─' * 9} {'─' * 7} {'─' * 7} {'─' * 7} {'─' * 8}")
        for rank, (score, m) in enumerate(scored, 1):
            pf_str = f"{m.profit_factor:.2f}" if m.profit_factor < 1000 else "inf"
            print(
                f"  {rank:<5d} {m.variant:<22s} {m.profile:<12s} {score:>6.3f} "
                f"{m.total_return_pct:>+8.2f}% {m.sharpe_ratio:>7.2f} {pf_str:>7s} "
                f"{m.win_rate:>6.1%} {m.max_drawdown_pct:>8.2f}%"
            )

    print(f"\n{'=' * 120}\n")

    # Save JSON report
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "initial_balance": INITIAL_BALANCE,
        "results": [],
    }
    for m in all_metrics:
        json_data["results"].append({
            "variant": m.variant,
            "profile": m.profile,
            "final_balance": round(m.final_balance, 2),
            "total_return_pct": round(m.total_return_pct, 4),
            "max_drawdown_pct": round(m.max_drawdown_pct, 4),
            "sharpe_ratio": round(m.sharpe_ratio, 4),
            "sortino_ratio": round(m.sortino_ratio, 4),
            "calmar_ratio": round(m.calmar_ratio, 4),
            "total_trades": m.total_trades,
            "win_rate": round(m.win_rate, 4),
            "profit_factor": round(min(m.profit_factor, 9999.0), 4),
            "avg_win": round(m.avg_win, 2),
            "avg_loss": round(m.avg_loss, 2),
            "avg_rr_ratio": round(m.avg_rr_ratio, 4),
            "largest_win": round(m.largest_win, 2),
            "largest_loss": round(m.largest_loss, 2),
            "buy_signals": m.buy_signals,
            "sell_signals": m.sell_signals,
            "hold_signals": m.hold_signals,
            "bars_processed": m.bars_processed,
            "elapsed_seconds": round(m.elapsed_seconds, 1),
            "error": m.error,
        })

    if scored:
        json_data["composite_ranking"] = [
            {"rank": i + 1, "variant": m.variant, "profile": m.profile, "score": round(s, 4)}
            for i, (s, m) in enumerate(scored)
        ]

    json_path = output_dir / "backtest_report.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    logger.info(f"JSON report saved to {json_path}")

    # Save CSV summary
    csv_rows = []
    for m in all_metrics:
        csv_rows.append({
            "variant": m.variant,
            "profile": m.profile,
            "return_pct": round(m.total_return_pct, 4),
            "max_dd_pct": round(m.max_drawdown_pct, 4),
            "sharpe": round(m.sharpe_ratio, 4),
            "sortino": round(m.sortino_ratio, 4),
            "trades": m.total_trades,
            "win_rate": round(m.win_rate, 4),
            "profit_factor": round(min(m.profit_factor, 9999.0), 4),
            "avg_rr": round(m.avg_rr_ratio, 4),
            "elapsed_s": round(m.elapsed_seconds, 1),
            "error": m.error[:50] if m.error else "",
        })
    csv_path = output_dir / "backtest_summary.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logger.info(f"CSV summary saved to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backtest all model variants")
    parser.add_argument(
        "--profiles", nargs="+", default=ALL_PROFILES,
        choices=ALL_PROFILES, help="Profiles to test",
    )
    parser.add_argument(
        "--variants", nargs="+", default=ALL_VARIANTS,
        help=f"Variants to test. Choices: {ALL_VARIANTS}",
    )
    parser.add_argument("--max-bars", type=int, default=None, help="Max bars per data file")
    parser.add_argument(
        "--output-dir", type=str,
        default=str(ASSETS_DIR / "backtest_results"),
        help="Output directory for reports",
    )
    args = parser.parse_args()

    profiles = [p.upper() for p in args.profiles]
    variants = args.variants
    output_dir = Path(args.output_dir)
    max_bars = args.max_bars

    logger.info("=" * 80)
    logger.info("  BACKTEST ALL VARIANTS")
    logger.info(f"  Profiles: {profiles}")
    logger.info(f"  Variants: {variants}")
    logger.info(f"  Max bars: {max_bars or 'all'}")
    logger.info(f"  Output:   {output_dir}")
    logger.info("=" * 80)

    # Pre-load data per profile
    data_cache: Dict[str, pd.DataFrame] = {}
    for profile in profiles:
        try:
            data_cache[profile] = load_data(profile, max_bars)
        except FileNotFoundError as e:
            logger.error(str(e))

    all_metrics: List[BacktestMetrics] = []

    total_runs = len(profiles) * len(variants)
    run_idx = 0

    for profile in profiles:
        if profile not in data_cache:
            logger.warning(f"Skipping {profile} — no data")
            continue

        df = data_cache[profile]

        for variant in variants:
            run_idx += 1
            logger.info(f"\n[{run_idx}/{total_runs}] {variant} / {profile}")

            metrics = run_single_backtest(variant, profile, df, max_bars)
            all_metrics.append(metrics)

    # Generate report
    if all_metrics:
        generate_report(all_metrics, output_dir)
    else:
        logger.error("No backtest results to report.")


if __name__ == "__main__":
    main()
