"""
Run V6 (Alpha2 + ProbabilisticMHTCN) through the research harness
for each profile (SCALP, INTRADAY, SWING) with profile-specific weights.

Usage:
    python scripts/run_v6_profiles.py
    python scripts/run_v6_profiles.py --profiles SCALP,INTRADAY
    python scripts/run_v6_profiles.py --max-bars 50000
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from research.interfaces import VariantID, VariantConfig
from research.feature_pipeline import FeaturePipeline
from research.regime_detector import RegimeDetector
from research.experiment.harness import ExperimentHarness
from research.experiment.metrics import MetricsCollector
from research.alpha_heads import AlphaHeadV2
from research.mhtcn_filters.probabilistic import ProbabilisticMHTCNFilter

logger = logging.getLogger("v6_profiles")

# ── Profile definitions ─────────────────────────────────────────────────────
# Each profile maps to:
#   base_tf   – the LTF used as the harness data source
#   weights   – V6 ProbabilisticTCN weights for that TF
PROFILE_CONFIG = {
    "SCALP": {
        "base_tf": "M5",
        "weights": "v6_prob_mhtcn_SCALP_M5.pt",
    },
    "INTRADAY": {
        "base_tf": "M15",
        "weights": "v6_prob_mhtcn_INTRADAY_M15.pt",
    },
    "SWING": {
        "base_tf": "H1",
        "weights": "v6_prob_mhtcn_SWING_H1.pt",
    },
}

# Data directory
DATA_DIR = Path("E:/pyProject/pyForex-assets/data/mt5/EURUSD")
WEIGHTS_DIR = Path("E:/pyProject/pyForex-assets/models/v6_profiles")
OUTPUT_DIR = Path("E:/pyProject/pyForex-assets/backtests/v6_research")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def find_latest_csv(tf: str) -> Path:
    """Find the most recent CSV for a given timeframe."""
    pattern = f"EURUSD_{tf}_*.csv"
    files = sorted(DATA_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No CSV found for {tf} in {DATA_DIR}")
    return files[0]


def load_data(path: Path, max_bars: int = 0) -> pd.DataFrame:
    """Load OHLCV CSV with datetime index."""
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.set_index("time")
    # Normalize columns
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("open", "high", "low", "close", "volume", "tick_volume"):
            col_map[c] = cl
    if col_map:
        df = df.rename(columns=col_map)
    if "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]

    df = df.dropna(subset=["open", "high", "low", "close"])

    if max_bars > 0 and len(df) > max_bars:
        df = df.iloc[-max_bars:]

    return df


def build_v6_variant(weights_path: str, device: str = "cpu") -> VariantConfig:
    """Build a single V6 variant config with profile-specific weights."""
    alpha = AlphaHeadV2(
        lookback=100,
        min_probability=0.40,
        directional_edge_min=0.04,
    )
    mhtcn_filter = ProbabilisticMHTCNFilter(
        seq_len=64,
        weights_path=weights_path,
        device=device,
    )
    return VariantConfig(
        variant_id=VariantID.V6_ALPHA2_PROB_MHTCN,
        alpha_head=alpha,
        mhtcn_filter=mhtcn_filter,
        description="Alpha2 + ProbabilisticMHTCN (profile-specific)",
        initial_balance=10_000.0,
        risk_per_trade=0.01,
        commission_per_lot=7.0,
        spread_pips=1.0,
        pip_value=10.0,
        min_rr=1.5,
        atr_sl_mult=2.0,
        max_open_trades=1,
        cooldown_bars=6,
        min_probability=0.55,
    )


def run_profile(
    profile: str,
    max_bars: int,
    device: str,
) -> dict:
    """Run V6 on one profile and return metrics."""
    cfg = PROFILE_CONFIG[profile]
    tf = cfg["base_tf"]
    weights_file = WEIGHTS_DIR / cfg["weights"]

    logger.info("=" * 70)
    logger.info(f"  PROFILE: {profile}  |  TF: {tf}  |  Weights: {weights_file.name}")
    logger.info("=" * 70)

    # Load data
    csv_path = find_latest_csv(tf)
    logger.info(f"Loading {csv_path.name}...")
    df = load_data(csv_path, max_bars=max_bars)
    logger.info(f"  {len(df):,} bars  |  {df.index[0]} → {df.index[-1]}")

    # Build variant
    variant = build_v6_variant(str(weights_file), device=device)

    # Run harness
    t0 = time.time()
    harness = ExperimentHarness(
        df,
        pip_size=0.0001,
        feature_pipeline=FeaturePipeline(),
        regime_detector=RegimeDetector(),
    )
    results = harness.run([variant], warmup=200)
    elapsed = time.time() - t0

    # Compute metrics
    collector = MetricsCollector(total_bars=len(df))
    all_metrics = collector.compare_variants(results)
    r = results[0]
    m = all_metrics[r.variant_id]

    # Print summary
    logger.info(f"\n  {profile} V6 Results ({len(df):,} bars, {elapsed:.1f}s):")
    logger.info(f"    Trades:      {m['n_trades']:.0f}")
    logger.info(f"    Win Rate:    {m['win_rate']*100:.1f}%")
    logger.info(f"    Total PnL:   ${m['total_pnl']:.2f}")
    logger.info(f"    Return:      {m['total_return_pct']:.2f}%")
    logger.info(f"    Sharpe:      {m['sharpe']:.2f}")
    logger.info(f"    Max DD:      {m['max_drawdown_pct']:.1f}%")
    logger.info(f"    Profit Fac:  {m['profit_factor']:.2f}")
    logger.info(f"    Brier:       {m['brier_score']:.4f}")
    logger.info(f"    Mean g:      {m['mean_g_factor']:.3f}")
    logger.info(f"    Trades/1K:   {m['trades_per_1k_bars']:.1f}")

    # Save trades CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if r.trades:
        rows = []
        for t in r.trades:
            rows.append({
                "profile": profile,
                "variant": t.variant_id,
                "bar_index": t.bar_index,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "direction": t.direction.value,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "sl": t.sl,
                "tp": t.tp,
                "volume": t.volume,
                "pnl": t.pnl,
                "p_alpha": t.p_alpha,
                "p_final": t.p_final,
                "g_factor": t.g_factor,
                "regime": t.regime,
            })
        trade_df = pd.DataFrame(rows)
        out_path = OUTPUT_DIR / f"v6_trades_{profile}.csv"
        trade_df.to_csv(out_path, index=False)
        logger.info(f"    Trades CSV:  {out_path}")

    # Save equity curve
    eq_path = OUTPUT_DIR / f"v6_equity_{profile}.csv"
    eq_df = pd.DataFrame({"equity": r.equity_curve})
    eq_df.to_csv(eq_path, index=False)

    return {
        "profile": profile,
        "tf": tf,
        "bars": len(df),
        "elapsed_s": elapsed,
        **m,
    }


def main():
    parser = argparse.ArgumentParser(description="Run V6 across profiles")
    parser.add_argument("--profiles", default="SCALP,INTRADAY,SWING",
                        help="Comma-separated profiles to run")
    parser.add_argument("--max-bars", type=int, default=0,
                        help="Max bars per profile (0 = all)")
    parser.add_argument("--device", default="cpu", help="torch device")
    args = parser.parse_args()

    setup_logging()
    profiles = [p.strip().upper() for p in args.profiles.split(",")]

    all_results = []
    for profile in profiles:
        if profile not in PROFILE_CONFIG:
            logger.warning(f"Unknown profile: {profile}, skipping")
            continue
        result = run_profile(profile, args.max_bars, args.device)
        all_results.append(result)

    # ── Comparison table ─────────────────────────────────────────────────
    if all_results:
        logger.info("\n" + "=" * 90)
        logger.info("  V6 CROSS-PROFILE COMPARISON")
        logger.info("=" * 90)
        header = (
            f"{'Profile':<12} {'TF':<5} {'Bars':>10} "
            f"{'Trades':>7} {'WR%':>6} {'PnL':>10} {'Ret%':>7} "
            f"{'Sharpe':>7} {'MaxDD%':>7} {'PF':>6} {'g_avg':>6}"
        )
        logger.info(header)
        logger.info("-" * 90)
        for r in all_results:
            row = (
                f"{r['profile']:<12} {r['tf']:<5} {r['bars']:>10,} "
                f"{r['n_trades']:>7.0f} {r['win_rate']*100:>5.1f}% "
                f"${r['total_pnl']:>9.2f} {r['total_return_pct']:>6.2f}% "
                f"{r['sharpe']:>7.2f} {r['max_drawdown_pct']:>6.1f}% "
                f"{r['profit_factor']:>5.2f} {r['mean_g_factor']:>5.3f}"
            )
            logger.info(row)
        logger.info("=" * 90)

        # Save summary
        summary_df = pd.DataFrame(all_results)
        summary_path = OUTPUT_DIR / "v6_profile_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
