"""Unit tests for `ml/performance_monitor.py`.

Test Summary:
    Comprehensive coverage of the performance monitoring subsystem:
    thresholds, trade record math, snapshot/alert generation, custom metrics,
    export/reset utilities, and retraining recommendation logic.

Test Breakdown:
    - Threshold evaluation
        - `MetricThreshold.check()` for both "above" and "below" directions
    - Trade record behavior
        - `TradeRecord.is_winner`, `TradeRecord.return_pct`, and `to_dict()`
    - Monitor state transitions
        - `set_initial_equity()` updates equity state and curve
        - `add_trade()` updates equity only when a trade is closed
        - `close_trade()` closes an open trade and can trigger a snapshot
    - Metric computation
        - `update_metrics()` produces expected core metrics (win_rate, profit_factor)
        - `get_metric_history()` returns recorded history
        - custom metrics are included in snapshots when registered
    - Retraining recommendation
        - `needs_retraining()` returns True when critical conditions are present
    - Utilities
        - `export_history()` writes JSON
        - `reset()` clears state
"""

from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timedelta
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _mk_closed_trade(mod, trade_id: str, pnl: float, direction: int = 1):
    now = datetime.now()
    entry_price = 1.0
    exit_price = 1.0 + (0.01 if pnl > 0 else -0.01)
    return mod.TradeRecord(
        trade_id=trade_id,
        entry_time=now - timedelta(minutes=10),
        exit_time=now,
        symbol="EURUSD",
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
    )


class TestMetricThreshold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("ml.performance_monitor")

    def test_check_above_direction(self):
        mod = self.mod
        thr = mod.MetricThreshold(mod.MetricType.WIN_RATE, warning_threshold=0.6, critical_threshold=0.5, direction="above")

        self.assertEqual(thr.check(0.7), mod.AlertLevel.INFO)
        self.assertEqual(thr.check(0.55), mod.AlertLevel.WARNING)
        self.assertEqual(thr.check(0.49), mod.AlertLevel.CRITICAL)

    def test_check_below_direction(self):
        mod = self.mod
        thr = mod.MetricThreshold(mod.MetricType.MAX_DRAWDOWN, warning_threshold=0.15, critical_threshold=0.25, direction="below")

        self.assertEqual(thr.check(0.10), mod.AlertLevel.INFO)
        self.assertEqual(thr.check(0.20), mod.AlertLevel.WARNING)
        self.assertEqual(thr.check(0.30), mod.AlertLevel.CRITICAL)


class TestTradeRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("ml.performance_monitor")

    def test_is_winner_and_return_pct(self):
        mod = self.mod

        t = _mk_closed_trade(mod, "t", pnl=5.0, direction=1)
        self.assertTrue(t.is_winner)
        self.assertGreater(t.return_pct, 0.0)

        t2 = _mk_closed_trade(mod, "t2", pnl=-5.0, direction=1)
        self.assertFalse(t2.is_winner)
        self.assertLess(t2.return_pct, 0.0)

        t3 = mod.TradeRecord(
            trade_id="t3",
            entry_time=datetime.now() - timedelta(minutes=10),
            exit_time=datetime.now(),
            symbol="EURUSD",
            direction=-1,
            entry_price=1.0,
            exit_price=0.99,
            pnl=5.0,
        )
        self.assertTrue(t3.is_winner)
        self.assertGreater(t3.return_pct, 0.0)

    def test_to_dict_keys(self):
        mod = self.mod
        t = _mk_closed_trade(mod, "t", pnl=1.0)
        d = t.to_dict()
        for key in ["trade_id", "entry_time", "exit_time", "symbol", "direction", "entry_price", "exit_price", "pnl"]:
            self.assertIn(key, d)


class TestPerformanceMonitorComprehensive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = import_module("ml.performance_monitor")

    def test_set_initial_equity_sets_curve(self):
        mod = self.mod
        pm = mod.PerformanceMonitor(mod.MonitorConfig())
        pm.set_initial_equity(12345.0)

        self.assertEqual(pm.initial_equity, 12345.0)
        self.assertEqual(pm.current_equity, 12345.0)
        self.assertEqual(pm.peak_equity, 12345.0)
        self.assertTrue(len(pm.equity_curve) >= 1)

    def test_add_trade_open_then_close_trade(self):
        mod = self.mod
        cfg = mod.MonitorConfig()
        cfg.update_interval_trades = 9999
        cfg.update_interval_minutes = 9999
        pm = mod.PerformanceMonitor(cfg)
        pm.last_update_time = datetime.now()

        now = datetime.now()
        open_trade = mod.TradeRecord(
            trade_id="open",
            entry_time=now - timedelta(minutes=5),
            exit_time=None,
            symbol="EURUSD",
            direction=1,
            entry_price=1.0,
            exit_price=None,
            pnl=0.0,
        )
        snap = pm.add_trade(open_trade)
        self.assertIsNone(snap)
        self.assertEqual(pm.current_equity, pm.initial_equity)

        snap2 = pm.close_trade("open", exit_time=now, exit_price=1.01, pnl=10.0)
        self.assertTrue(snap2 is None or hasattr(snap2, "metrics"))
        self.assertEqual(len(pm.closed_trades), 1)
        self.assertGreater(pm.current_equity, pm.initial_equity)

    def test_update_metrics_core_values(self):
        mod = self.mod
        cfg = mod.MonitorConfig()
        cfg.update_interval_trades = 9999
        cfg.update_interval_minutes = 9999
        pm = mod.PerformanceMonitor(cfg)
        pm.last_update_time = datetime.now()

        pm.add_trade(_mk_closed_trade(mod, "w", pnl=10.0))
        pm.add_trade(_mk_closed_trade(mod, "l", pnl=-5.0))

        snap = pm.update_metrics()
        self.assertIn(mod.MetricType.WIN_RATE.value, snap.metrics)
        self.assertIn(mod.MetricType.PROFIT_FACTOR.value, snap.metrics)
        self.assertAlmostEqual(snap.metrics[mod.MetricType.WIN_RATE.value], 0.5)
        self.assertGreater(snap.metrics[mod.MetricType.PROFIT_FACTOR.value], 0.0)

        hist = pm.get_metric_history(mod.MetricType.WIN_RATE, lookback=10)
        self.assertTrue(isinstance(hist, list))

    def test_register_custom_metric_included(self):
        mod = self.mod
        cfg = mod.MonitorConfig()
        cfg.update_interval_trades = 9999
        cfg.update_interval_minutes = 9999
        pm = mod.PerformanceMonitor(cfg)
        pm.last_update_time = datetime.now()

        pm.add_trade(_mk_closed_trade(mod, "t1", pnl=1.0))
        pm.add_trade(_mk_closed_trade(mod, "t2", pnl=-1.0))

        def custom_avg_pnl(trades, _equity_curve):
            if not trades:
                return 0.0
            return sum(t.pnl for t in trades) / len(trades)

        pm.register_custom_metric("avg_pnl", custom_avg_pnl)
        snap = pm.update_metrics()
        self.assertIn("avg_pnl", snap.metrics)

    def test_needs_retraining_true_when_critical_conditions_present(self):
        mod = self.mod
        pm = mod.PerformanceMonitor(mod.MonitorConfig())

        metrics = {
            mod.MetricType.WIN_RATE.value: 0.0,
            mod.MetricType.PROFIT_FACTOR.value: 0.0,
            mod.MetricType.SHARPE_RATIO.value: -1.0,
            mod.MetricType.MAX_DRAWDOWN.value: 0.5,
        }
        snap = mod.PerformanceSnapshot(
            timestamp=datetime.now(),
            metrics=metrics,
            alerts=[],
            trade_count=100,
            equity=9000.0,
            drawdown=0.1,
        )
        pm.snapshots.append(snap)

        pm.alerts.extend(
            [
                mod.PerformanceAlert(
                    timestamp=datetime.now(),
                    metric_type=mod.MetricType.PROFIT_FACTOR,
                    alert_level=mod.AlertLevel.CRITICAL,
                    current_value=0.0,
                    threshold_value=0.9,
                    message="critical profit factor",
                    recommendation="retrain",
                ),
                mod.PerformanceAlert(
                    timestamp=datetime.now(),
                    metric_type=mod.MetricType.WIN_RATE,
                    alert_level=mod.AlertLevel.CRITICAL,
                    current_value=0.0,
                    threshold_value=0.35,
                    message="critical win rate",
                    recommendation="retrain",
                ),
            ]
        )

        should, reason = pm.needs_retraining()
        self.assertTrue(should)
        self.assertIn("Critical metrics", reason)

    def test_export_history_writes_json(self):
        mod = self.mod
        cfg = mod.MonitorConfig()
        cfg.update_interval_trades = 9999
        cfg.update_interval_minutes = 9999
        pm = mod.PerformanceMonitor(cfg)
        pm.last_update_time = datetime.now()
        pm.add_trade(_mk_closed_trade(mod, "t1", pnl=1.0))
        pm.add_trade(_mk_closed_trade(mod, "t2", pnl=-1.0))
        pm.update_metrics()

        with TemporaryDirectory() as td:
            out = Path(td) / "perf.json"
            pm.export_history(str(out))
            self.assertTrue(out.exists())
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("trades", data)
            self.assertIn("snapshots", data)

    def test_reset_clears_state(self):
        mod = self.mod
        pm = mod.PerformanceMonitor(mod.MonitorConfig())
        pm.last_update_time = datetime.now()
        pm.add_trade(_mk_closed_trade(mod, "t1", pnl=1.0))
        pm.update_metrics()

        self.assertTrue(len(pm.closed_trades) >= 1)
        pm.reset()
        self.assertEqual(len(pm.closed_trades), 0)
        self.assertEqual(len(pm.alerts), 0)
        self.assertEqual(len(pm.snapshots), 0)


if __name__ == "__main__":
    unittest.main()
