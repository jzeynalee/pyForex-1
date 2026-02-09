"""
Attribution report generator.

Produces a structured Markdown report comparing all 6 variants with:
    - Variant summary table
    - Metric comparison (Alpha Quality, Risk, Efficiency)
    - Interaction effects (delta-EV, prob sharpening)
    - Negative control results
    - Production recommendation
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..interfaces import VariantResult

logger = logging.getLogger(__name__)


class AttributionReporter:
    """Generates comparison reports from variant results."""

    def __init__(self, output_dir: str = "research_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        results: List[VariantResult],
        metrics: Dict[str, Dict[str, float]],
        negative_control_flags: Optional[Dict[str, bool]] = None,
        data_description: str = "",
    ) -> str:
        """Generate full attribution report as Markdown string.

        Also saves to file.
        """
        lines: List[str] = []

        # Header
        lines.append("# Multi-Alpha + MH-TCN Research Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if data_description:
            lines.append(f"\nData: {data_description}")
        lines.append("")

        # Summary table
        lines.append("## Variant Summary")
        lines.append("")
        lines.append(self._summary_table(results, metrics))
        lines.append("")

        # Alpha quality
        lines.append("## Alpha Quality")
        lines.append("")
        lines.append(self._quality_table(metrics))
        lines.append("")

        # Risk metrics
        lines.append("## Risk Metrics")
        lines.append("")
        lines.append(self._risk_table(metrics))
        lines.append("")

        # Efficiency
        lines.append("## Efficiency")
        lines.append("")
        lines.append(self._efficiency_table(metrics))
        lines.append("")

        # Interaction effects
        lines.append("## Interaction Effects (MH-TCN Impact)")
        lines.append("")
        lines.append(self._interaction_table(metrics))
        lines.append("")

        # Negative controls
        if negative_control_flags:
            lines.append("## Negative Control Results")
            lines.append("")
            lines.append(self._negative_controls_section(negative_control_flags))
            lines.append("")

        # Recommendation
        lines.append("## Production Recommendation")
        lines.append("")
        lines.append(self._recommendation(results, metrics))
        lines.append("")

        report = "\n".join(lines)

        # Save
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"research_report_{ts}.md"
        path.write_text(report, encoding="utf-8")
        logger.info(f"Report saved to {path}")

        return report

    # ------------------------------------------------------------------
    # Table builders
    # ------------------------------------------------------------------

    def _summary_table(self, results, metrics) -> str:
        header = "| Variant | Trades | Win Rate | Total PnL | Sharpe | Max DD% |"
        sep =    "|---------|--------|----------|-----------|--------|---------|"
        rows = [header, sep]
        for r in results:
            m = metrics.get(r.variant_id, {})
            rows.append(
                f"| {r.variant_id} "
                f"| {m.get('n_trades', 0):.0f} "
                f"| {m.get('win_rate', 0)*100:.1f}% "
                f"| ${m.get('total_pnl', 0):.2f} "
                f"| {m.get('sharpe', 0):.2f} "
                f"| {m.get('max_drawdown_pct', 0):.1f}% |"
            )
        return "\n".join(rows)

    def _quality_table(self, metrics) -> str:
        header = "| Variant | Mean EV | Brier | IC | Profit Factor |"
        sep =    "|---------|---------|-------|----|---------------|"
        rows = [header, sep]
        for vid, m in metrics.items():
            rows.append(
                f"| {vid} "
                f"| ${m.get('mean_ev', 0):.2f} "
                f"| {m.get('brier_score', 1):.4f} "
                f"| {m.get('ic', 0):.3f} "
                f"| {m.get('profit_factor', 0):.2f} |"
            )
        return "\n".join(rows)

    def _risk_table(self, metrics) -> str:
        header = "| Variant | Max DD | Max DD% | CVaR 95% | Clustering Risk |"
        sep =    "|---------|--------|---------|----------|-----------------|"
        rows = [header, sep]
        for vid, m in metrics.items():
            rows.append(
                f"| {vid} "
                f"| ${m.get('max_drawdown', 0):.2f} "
                f"| {m.get('max_drawdown_pct', 0):.1f}% "
                f"| ${m.get('cvar_95', 0):.2f} "
                f"| {m.get('trade_clustering_risk', 0):.1f} |"
            )
        return "\n".join(rows)

    def _efficiency_table(self, metrics) -> str:
        header = "| Variant | Trades/1K bars | Avg Duration | Capital Util |"
        sep =    "|---------|----------------|--------------|--------------|"
        rows = [header, sep]
        for vid, m in metrics.items():
            rows.append(
                f"| {vid} "
                f"| {m.get('trades_per_1k_bars', 0):.1f} "
                f"| {m.get('avg_trade_duration_bars', 0):.0f} bars "
                f"| {m.get('total_return_pct', 0):.1f}% |"
            )
        return "\n".join(rows)

    def _interaction_table(self, metrics) -> str:
        header = "| Variant | Mean g | Mean P_alpha | Mean P_final | Sharpening | Delta-EV |"
        sep =    "|---------|--------|--------------|--------------|------------|----------|"
        rows = [header, sep]
        for vid, m in metrics.items():
            delta_keys = [k for k in m if k.startswith("delta_ev")]
            delta_ev = m[delta_keys[0]] if delta_keys else 0.0
            rows.append(
                f"| {vid} "
                f"| {m.get('mean_g_factor', 1):.3f} "
                f"| {m.get('mean_p_alpha', 0):.3f} "
                f"| {m.get('mean_p_final', 0):.3f} "
                f"| {m.get('prob_sharpening', 0):+.3f} "
                f"| ${delta_ev:+.2f} |"
            )
        return "\n".join(rows)

    def _negative_controls_section(self, flags: Dict[str, bool]) -> str:
        lines = []
        any_leakage = False
        for ctrl_id, is_leak in flags.items():
            status = "**LEAKAGE DETECTED**" if is_leak else "PASS"
            lines.append(f"- {ctrl_id}: {status}")
            if is_leak:
                any_leakage = True
        if any_leakage:
            lines.append("")
            lines.append(
                "> **WARNING**: One or more negative controls show improvement. "
                "Investigate label leakage, data snooping, or pipeline contamination."
            )
        else:
            lines.append("")
            lines.append("> All negative controls passed. No leakage detected.")
        return "\n".join(lines)

    def _recommendation(self, results, metrics) -> str:
        """Select best variant based on composite score."""
        if not metrics:
            return "No results to evaluate."

        # Score: weighted combination of Sharpe, -Brier, -MaxDD%, profitFactor
        scores = {}
        for vid, m in metrics.items():
            score = (
                0.35 * m.get("sharpe", 0)
                + 0.20 * (1.0 - m.get("brier_score", 1))
                + 0.25 * (1.0 - m.get("max_drawdown_pct", 100) / 100)
                + 0.20 * min(m.get("profit_factor", 0), 5.0) / 5.0
            )
            scores[vid] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        lines = ["### Ranking (composite score)\n"]
        for i, (vid, score) in enumerate(ranked):
            m = metrics[vid]
            marker = " **<-- RECOMMENDED**" if i == 0 else ""
            lines.append(
                f"{i+1}. **{vid}** — score={score:.3f} "
                f"(Sharpe={m.get('sharpe',0):.2f}, "
                f"Brier={m.get('brier_score',1):.3f}, "
                f"PF={m.get('profit_factor',0):.2f}){marker}"
            )

        best = ranked[0][0]
        lines.append(f"\n### Selected: **{best}**")
        lines.append("")
        lines.append("Selection criteria: Robust EV, stable calibration, ")
        lines.append("survivability under stress, clean attribution.")

        return "\n".join(lines)
