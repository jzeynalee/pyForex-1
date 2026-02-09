"""
Main entry point for the Multi-Alpha + MH-TCN Research Framework.

Usage:
    python -m research.run_experiment --data path/to/ohlcv.csv
    python -m research.run_experiment --data path/to/ohlcv.csv --variants V1,V3,V6
    python -m research.run_experiment --data path/to/ohlcv.csv --neg-controls

The script:
    1. Loads OHLCV data
    2. Builds all 6 variants (or a subset)
    3. Runs them through the shared harness
    4. Computes metrics and generates attribution report
    5. Optionally runs negative controls
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.interfaces import VariantID
from research.feature_pipeline import FeaturePipeline
from research.regime_detector import RegimeDetector
from research.experiment.variants import build_all_variants
from research.experiment.harness import ExperimentHarness
from research.experiment.metrics import MetricsCollector
from research.experiment.attribution import AttributionReporter
from research.experiment.negative_controls import NegativeControlRunner

logger = logging.getLogger("research")


def setup_logging(log_file: str = "research_experiment.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w"),
        ],
    )


def load_data(path: str) -> pd.DataFrame:
    """Load OHLCV CSV with datetime index."""
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("open", "high", "low", "close", "volume", "tick_volume"):
            col_map[c] = cl
    if col_map:
        df = df.rename(columns=col_map)

    required = {"open", "high", "low", "close"}
    present = set(df.columns)
    missing = required - present
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with NaN in OHLC
    df = df.dropna(subset=["open", "high", "low", "close"])
    logger.info(f"Loaded {len(df)} bars from {path}")
    logger.info(f"  Range: {df.index[0]} -> {df.index[-1]}")
    return df


def parse_variant_filter(s: str):
    """Parse comma-separated variant IDs like 'V1,V3,V6'."""
    if not s:
        return None
    ids = set()
    for part in s.split(","):
        part = part.strip().upper()
        for vid in VariantID:
            if part in vid.value.upper() or part in vid.name.upper():
                ids.add(vid)
    return ids if ids else None


def main():
    parser = argparse.ArgumentParser(description="Multi-Alpha + MH-TCN Research")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV")
    parser.add_argument("--variants", default="", help="Comma-separated variant filter (e.g. V1,V3)")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup bars")
    parser.add_argument("--neg-controls", action="store_true", help="Run negative controls")
    parser.add_argument("--output-dir", default="research_output", help="Output directory")
    parser.add_argument("--pip-size", type=float, default=0.0001, help="Pip size")
    parser.add_argument("--initial-balance", type=float, default=10000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--min-probability", type=float, default=0.55)
    parser.add_argument("--device", default="cpu", help="torch device")
    args = parser.parse_args()

    setup_logging(str(Path(args.output_dir) / "research_experiment.log"))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_data(args.data)

    # 2. Build variants
    all_variants = build_all_variants(
        device=args.device,
        initial_balance=args.initial_balance,
        risk_per_trade=args.risk_per_trade,
        min_probability=args.min_probability,
    )

    # Filter if requested
    variant_filter = parse_variant_filter(args.variants)
    if variant_filter:
        all_variants = [v for v in all_variants if v.variant_id in variant_filter]
        logger.info(f"Filtered to {len(all_variants)} variants: "
                     f"{[v.variant_id.value for v in all_variants]}")

    # 3. Run harness
    harness = ExperimentHarness(
        df, pip_size=args.pip_size,
        feature_pipeline=FeaturePipeline(),
        regime_detector=RegimeDetector(),
    )
    results = harness.run(all_variants, warmup=args.warmup)

    # 4. Compute metrics
    collector = MetricsCollector(total_bars=len(df))
    all_metrics = collector.compare_variants(results)

    # 5. Print summary
    logger.info("=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    for r in results:
        m = all_metrics[r.variant_id]
        logger.info(
            f"  {r.variant_id:30s} | "
            f"trades={m['n_trades']:3.0f} | "
            f"WR={m['win_rate']*100:5.1f}% | "
            f"PnL=${m['total_pnl']:8.2f} | "
            f"Sharpe={m['sharpe']:6.2f} | "
            f"MaxDD={m['max_drawdown_pct']:5.1f}%"
        )

    # 6. Negative controls
    neg_flags = None
    if args.neg_controls and all_variants:
        logger.info("Running negative controls...")
        # Use first real alpha and first real MHTCN for controls
        real_alpha = all_variants[0].alpha_head
        real_mhtcn = None
        for v in all_variants:
            if v.mhtcn_filter.name() != "NullFilter":
                real_mhtcn = v.mhtcn_filter
                break
        if real_mhtcn:
            shared_kw = dict(
                initial_balance=args.initial_balance,
                risk_per_trade=args.risk_per_trade,
                min_probability=args.min_probability,
                commission_per_lot=7.0, spread_pips=1.0,
                pip_value=10.0, min_rr=1.5, atr_sl_mult=2.0,
                max_open_trades=1, cooldown_bars=6,
            )
            controls = NegativeControlRunner.build_controls(
                real_alpha, real_mhtcn, shared_kw
            )
            ctrl_results = harness.run(controls, warmup=args.warmup)
            baseline_ev = all_metrics.get(
                results[0].variant_id, {}
            ).get("mean_ev", 0.0)
            neg_flags = NegativeControlRunner.check_leakage(ctrl_results, baseline_ev)
        else:
            logger.info("No MH-TCN filter found in variants — skipping neg controls")

    # 7. Generate report
    reporter = AttributionReporter(output_dir=args.output_dir)
    data_desc = f"{len(df)} bars from {df.index[0]} to {df.index[-1]}"
    report = reporter.generate_report(
        results, all_metrics,
        negative_control_flags=neg_flags,
        data_description=data_desc,
    )
    logger.info(f"Report saved to {args.output_dir}/")

    # 8. Save trade logs
    for r in results:
        if r.trades:
            trade_rows = []
            for t in r.trades:
                trade_rows.append({
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
            trade_df = pd.DataFrame(trade_rows)
            safe_name = r.variant_id.replace("+", "_").replace(" ", "_")
            path = Path(args.output_dir) / f"trades_{safe_name}.csv"
            trade_df.to_csv(path, index=False)

    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
