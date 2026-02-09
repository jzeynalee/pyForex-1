"""
Evaluation metrics for the research framework.

Computes per-variant:
    Alpha Quality:  Mean EV, Brier score, Reliability curve, IC
    Risk:           Max drawdown, CVaR, Trade clustering risk
    Efficiency:     Trades/1000 bars, Capital utilization, Turnover
    Interaction:    Delta-EV (with/without MHTCN), Trade survival, Prob sharpening
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..interfaces import TradeRecord, VariantResult
from ..calibrator import ProbabilityCalibrator

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Compute all required metrics for a set of variant results."""

    def __init__(self, total_bars: int = 1, risk_free_rate: float = 0.0):
        self.total_bars = total_bars
        self.risk_free_rate = risk_free_rate

    def compute_all(self, result: VariantResult) -> Dict[str, float]:
        """Compute full metric suite for one variant."""
        m: Dict[str, float] = {}
        trades = result.trades
        eq = result.equity_curve

        # -- Alpha Quality --
        m["n_trades"] = len(trades)
        m["mean_ev"] = self._mean_ev(trades)
        m["win_rate"] = self._win_rate(trades)
        m["brier_score"] = self._brier_score(trades)
        m["ic"] = self._information_coefficient(trades)

        # -- Risk --
        m["total_pnl"] = sum(t.pnl for t in trades)
        m["total_return_pct"] = (eq[-1] / eq[0] - 1) * 100 if len(eq) > 1 and eq[0] > 0 else 0.0
        m["max_drawdown"] = self._max_drawdown(eq)
        m["max_drawdown_pct"] = self._max_drawdown_pct(eq)
        m["cvar_95"] = self._cvar(trades, alpha=0.05)
        m["trade_clustering_risk"] = self._clustering_risk(trades)
        m["sharpe"] = self._sharpe(eq)

        # -- Efficiency --
        m["trades_per_1k_bars"] = len(trades) / max(self.total_bars / 1000, 0.001)
        m["avg_trade_duration_bars"] = self._avg_duration(trades)
        m["profit_factor"] = self._profit_factor(trades)

        # -- Interaction (populated externally via compare_variants) --
        m["mean_g_factor"] = self._mean_g_factor(trades)
        m["mean_p_alpha"] = np.mean([t.p_alpha for t in trades]) if trades else 0.0
        m["mean_p_final"] = np.mean([t.p_final for t in trades]) if trades else 0.0
        m["prob_sharpening"] = m["mean_p_final"] - m["mean_p_alpha"]

        return m

    def compare_variants(
        self,
        results: List[VariantResult],
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics for all variants and interaction effects.

        Returns dict[variant_id -> metrics_dict].
        """
        all_metrics: Dict[str, Dict[str, float]] = {}
        for r in results:
            r.metrics = self.compute_all(r)
            all_metrics[r.variant_id] = r.metrics

        # Interaction: delta-EV between paired variants
        pairs = [
            ("V1_Alpha", "V2_Alpha+MHTCN"),
            ("V3_Alpha2", "V4_Alpha2+MHTCN"),
            ("V1_Alpha", "V5_Alpha+ProbMHTCN"),
            ("V3_Alpha2", "V6_Alpha2+ProbMHTCN"),
        ]
        for base_id, enhanced_id in pairs:
            if base_id in all_metrics and enhanced_id in all_metrics:
                delta = all_metrics[enhanced_id]["mean_ev"] - all_metrics[base_id]["mean_ev"]
                all_metrics[enhanced_id][f"delta_ev_vs_{base_id}"] = delta

        return all_metrics

    # ------------------------------------------------------------------
    # Alpha Quality
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_ev(trades: List[TradeRecord]) -> float:
        if not trades:
            return 0.0
        return float(np.mean([t.pnl for t in trades]))

    @staticmethod
    def _win_rate(trades: List[TradeRecord]) -> float:
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.pnl > 0)
        return wins / len(trades)

    @staticmethod
    def _brier_score(trades: List[TradeRecord]) -> float:
        """Brier score: mean((p_final - outcome)^2)."""
        valid = [t for t in trades if t.forward_label is not None]
        if not valid:
            return 1.0
        preds = np.array([t.p_final for t in valid])
        outcomes = np.array([t.forward_label for t in valid])
        return float(np.mean((preds - outcomes) ** 2))

    @staticmethod
    def _information_coefficient(trades: List[TradeRecord]) -> float:
        """Rank correlation between predicted probability and realized PnL."""
        if len(trades) < 5:
            return 0.0
        preds = np.array([t.p_final for t in trades])
        pnls = np.array([t.pnl for t in trades])
        if np.std(preds) < 1e-10 or np.std(pnls) < 1e-10:
            return 0.0
        try:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(preds, pnls)
            return float(corr) if not np.isnan(corr) else 0.0
        except ImportError:
            # Fallback: Pearson
            corr = np.corrcoef(preds, pnls)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0

    # ------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------

    @staticmethod
    def _max_drawdown(eq: np.ndarray) -> float:
        if len(eq) < 2:
            return 0.0
        peak = np.maximum.accumulate(eq)
        dd = peak - eq
        return float(np.max(dd))

    @staticmethod
    def _max_drawdown_pct(eq: np.ndarray) -> float:
        if len(eq) < 2:
            return 0.0
        peak = np.maximum.accumulate(eq)
        dd_pct = (peak - eq) / (peak + 1e-10)
        return float(np.max(dd_pct) * 100)

    @staticmethod
    def _cvar(trades: List[TradeRecord], alpha: float = 0.05) -> float:
        """Conditional Value at Risk (Expected Shortfall)."""
        if not trades:
            return 0.0
        pnls = sorted([t.pnl for t in trades])
        n_tail = max(1, int(len(pnls) * alpha))
        return float(np.mean(pnls[:n_tail]))

    @staticmethod
    def _clustering_risk(trades: List[TradeRecord]) -> float:
        """Measure trade clustering: std of inter-trade bar gaps.
        Higher = more clustered = riskier."""
        if len(trades) < 3:
            return 0.0
        bars = sorted([t.bar_index for t in trades])
        gaps = np.diff(bars)
        if len(gaps) == 0:
            return 0.0
        return float(np.std(gaps))

    @staticmethod
    def _sharpe(eq: np.ndarray) -> float:
        if len(eq) < 10:
            return 0.0
        returns = np.diff(eq) / (eq[:-1] + 1e-10)
        if np.std(returns) < 1e-10:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------

    @staticmethod
    def _avg_duration(trades: List[TradeRecord]) -> float:
        """Average trade duration in bars (using bar_index delta)."""
        if not trades:
            return 0.0
        durations = []
        for t in trades:
            if t.entry_time and t.exit_time:
                dt = (t.exit_time - t.entry_time).total_seconds() / 300  # ~M5 bars
                durations.append(dt)
        return float(np.mean(durations)) if durations else 0.0

    @staticmethod
    def _profit_factor(trades: List[TradeRecord]) -> float:
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        if gross_loss < 1e-10:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_g_factor(trades: List[TradeRecord]) -> float:
        if not trades:
            return 1.0
        return float(np.mean([t.g_factor for t in trades]))
