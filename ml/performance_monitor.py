"""
Performance Monitor Module for pyForex ML System.

Tracks model performance metrics in real-time and triggers alerts
when performance degrades below acceptable thresholds.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import logging
import json

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    # Classification metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    
    # Trading metrics
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    
    # Signal quality
    SIGNAL_ACCURACY = "signal_accuracy"
    PREDICTION_CORRELATION = "prediction_correlation"
    
    # Custom
    CUSTOM = "custom"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


@dataclass
class MetricThreshold:
    """Threshold configuration for a metric."""
    metric_type: MetricType
    warning_threshold: float      # Trigger warning below this
    critical_threshold: float     # Trigger critical below this
    direction: str = "above"      # "above" = higher is better, "below" = lower is better
    lookback_periods: int = 20    # Periods to average for smoothing
    
    def check(self, value: float) -> AlertLevel:
        """Check value against thresholds."""
        if self.direction == "above":
            if value < self.critical_threshold:
                return AlertLevel.CRITICAL
            elif value < self.warning_threshold:
                return AlertLevel.WARNING
        else:  # below is better (e.g., drawdown)
            if value > self.critical_threshold:
                return AlertLevel.CRITICAL
            elif value > self.warning_threshold:
                return AlertLevel.WARNING
        return AlertLevel.INFO


@dataclass
class PerformanceAlert:
    """Performance alert notification."""
    timestamp: datetime
    metric_type: MetricType
    alert_level: AlertLevel
    current_value: float
    threshold_value: float
    message: str
    recommendation: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'metric_type': self.metric_type.value,
            'alert_level': self.alert_level.name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'message': self.message,
            'recommendation': self.recommendation
        }


@dataclass
class PerformanceSnapshot:
    """Snapshot of all performance metrics at a point in time."""
    timestamp: datetime
    metrics: Dict[str, float]
    alerts: List[PerformanceAlert]
    trade_count: int
    equity: float
    drawdown: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'alerts': [a.to_dict() for a in self.alerts],
            'trade_count': self.trade_count,
            'equity': self.equity,
            'drawdown': self.drawdown
        }


@dataclass
class MonitorConfig:
    """Configuration for performance monitoring."""
    # Default thresholds
    default_thresholds: Dict[MetricType, MetricThreshold] = field(default_factory=dict)
    
    # Update frequency
    update_interval_trades: int = 10    # Update every N trades
    update_interval_minutes: int = 60   # Or every N minutes
    
    # History
    history_length: int = 1000          # Keep last N snapshots
    
    # Alert settings
    alert_cooldown_minutes: int = 30    # Min time between same alerts
    consecutive_alerts_threshold: int = 3  # Alerts before escalation
    
    def __post_init__(self):
        """Set default thresholds if not provided."""
        if not self.default_thresholds:
            self.default_thresholds = {
                MetricType.WIN_RATE: MetricThreshold(
                    MetricType.WIN_RATE, 0.45, 0.35, "above"
                ),
                MetricType.PROFIT_FACTOR: MetricThreshold(
                    MetricType.PROFIT_FACTOR, 1.2, 0.9, "above"
                ),
                MetricType.SHARPE_RATIO: MetricThreshold(
                    MetricType.SHARPE_RATIO, 0.5, 0.0, "above"
                ),
                MetricType.MAX_DRAWDOWN: MetricThreshold(
                    MetricType.MAX_DRAWDOWN, 0.15, 0.25, "below"
                ),
                MetricType.SIGNAL_ACCURACY: MetricThreshold(
                    MetricType.SIGNAL_ACCURACY, 0.50, 0.40, "above"
                ),
            }


class TradeRecord:
    """Record of a single trade for performance tracking."""
    
    def __init__(
        self,
        trade_id: str,
        entry_time: datetime,
        exit_time: Optional[datetime],
        symbol: str,
        direction: int,  # 1 = long, -1 = short
        entry_price: float,
        exit_price: Optional[float],
        pnl: float = 0.0,
        predicted_direction: Optional[int] = None,
        confidence: Optional[float] = None
    ):
        self.trade_id = trade_id
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.pnl = pnl
        self.predicted_direction = predicted_direction
        self.confidence = confidence
        self.is_closed = exit_time is not None
    
    @property
    def is_winner(self) -> bool:
        return self.pnl > 0
    
    @property
    def return_pct(self) -> float:
        if self.exit_price and self.entry_price:
            return (self.exit_price - self.entry_price) / self.entry_price * self.direction
        return 0.0
    
    def to_dict(self) -> Dict:
        return {
            'trade_id': self.trade_id,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'pnl': self.pnl,
            'predicted_direction': self.predicted_direction,
            'confidence': self.confidence
        }


class PerformanceMonitor:
    """
    Main performance monitoring class.
    
    Tracks trading performance metrics in real-time and generates
    alerts when performance degrades.
    """
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        
        # Trade history
        self.trades: List[TradeRecord] = []
        self.closed_trades: List[TradeRecord] = []
        
        # Equity tracking
        self.initial_equity: float = 10000.0
        self.current_equity: float = self.initial_equity
        self.peak_equity: float = self.initial_equity
        self.equity_curve: List[tuple] = []  # (timestamp, equity)
        
        # Metric history
        self.metric_history: Dict[MetricType, deque] = {
            mt: deque(maxlen=self.config.history_length)
            for mt in MetricType
        }
        
        # Snapshots
        self.snapshots: deque = deque(maxlen=self.config.history_length)
        
        # Alerts
        self.alerts: List[PerformanceAlert] = []
        self.alert_timestamps: Dict[MetricType, datetime] = {}
        self.consecutive_alerts: Dict[MetricType, int] = {}
        
        # Custom metrics
        self.custom_metrics: Dict[str, Callable] = {}
        self.custom_thresholds: Dict[str, MetricThreshold] = {}
        
        # Counters
        self.trades_since_update: int = 0
        self.last_update_time: Optional[datetime] = None
        
        logger.info("PerformanceMonitor initialized")
    
    def set_initial_equity(self, equity: float) -> None:
        """Set initial equity for tracking."""
        self.initial_equity = equity
        self.current_equity = equity
        self.peak_equity = equity
        self.equity_curve = [(datetime.now(), equity)]
    
    def add_trade(self, trade: TradeRecord) -> Optional[PerformanceSnapshot]:
        """
        Add a trade and update metrics.
        Returns snapshot if update interval reached.
        """
        self.trades.append(trade)
        
        if trade.is_closed:
            self.closed_trades.append(trade)
            self.current_equity += trade.pnl
            self.peak_equity = max(self.peak_equity, self.current_equity)
            self.equity_curve.append((datetime.now(), self.current_equity))
        
        self.trades_since_update += 1
        
        # Check if update needed
        should_update = (
            self.trades_since_update >= self.config.update_interval_trades or
            self._time_since_update() >= timedelta(minutes=self.config.update_interval_minutes)
        )
        
        if should_update:
            return self.update_metrics()
        
        return None
    
    def close_trade(
        self,
        trade_id: str,
        exit_time: datetime,
        exit_price: float,
        pnl: float
    ) -> Optional[PerformanceSnapshot]:
        """Close an open trade."""
        for trade in self.trades:
            if trade.trade_id == trade_id and not trade.is_closed:
                trade.exit_time = exit_time
                trade.exit_price = exit_price
                trade.pnl = pnl
                trade.is_closed = True
                self.closed_trades.append(trade)
                
                self.current_equity += pnl
                self.peak_equity = max(self.peak_equity, self.current_equity)
                self.equity_curve.append((datetime.now(), self.current_equity))
                
                self.trades_since_update += 1
                
                if self.trades_since_update >= self.config.update_interval_trades:
                    return self.update_metrics()
                
                return None
        
        logger.warning(f"Trade {trade_id} not found for closing")
        return None
    
    def _time_since_update(self) -> timedelta:
        """Time since last metric update."""
        if self.last_update_time is None:
            return timedelta(hours=24)  # Force first update
        return datetime.now() - self.last_update_time
    
    def update_metrics(self) -> PerformanceSnapshot:
        """Calculate all metrics and check thresholds."""
        self.trades_since_update = 0
        self.last_update_time = datetime.now()
        
        metrics = self._calculate_metrics()
        alerts = self._check_thresholds(metrics)
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            metrics=metrics,
            alerts=alerts,
            trade_count=len(self.closed_trades),
            equity=self.current_equity,
            drawdown=self._calculate_drawdown()
        )
        
        self.snapshots.append(snapshot)
        
        # Store metrics in history
        for metric_type, value in metrics.items():
            try:
                mt = MetricType(metric_type)
                self.metric_history[mt].append((datetime.now(), value))
            except ValueError:
                pass  # Custom metric
        
        if alerts:
            for alert in alerts:
                self.alerts.append(alert)
                logger.warning(f"Performance Alert: {alert.message}")
        
        return snapshot
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate all performance metrics."""
        metrics = {}
        
        if not self.closed_trades:
            return metrics
        
        trades = self.closed_trades
        pnls = [t.pnl for t in trades]
        returns = [t.return_pct for t in trades if t.return_pct != 0]
        
        # Win rate
        winners = [t for t in trades if t.is_winner]
        metrics[MetricType.WIN_RATE.value] = len(winners) / len(trades) if trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        metrics[MetricType.PROFIT_FACTOR.value] = (
            gross_profit / gross_loss if gross_loss > 0 else float('inf')
        )
        
        # Sharpe ratio (simplified, using trade returns)
        if len(returns) > 1:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            metrics[MetricType.SHARPE_RATIO.value] = (
                mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
            )
        else:
            metrics[MetricType.SHARPE_RATIO.value] = 0
        
        # Sortino ratio
        if len(returns) > 1:
            negative_returns = [r for r in returns if r < 0]
            if negative_returns:
                downside_std = np.std(negative_returns)
                metrics[MetricType.SORTINO_RATIO.value] = (
                    mean_return / downside_std * np.sqrt(252) if downside_std > 0 else 0
                )
            else:
                metrics[MetricType.SORTINO_RATIO.value] = float('inf')
        else:
            metrics[MetricType.SORTINO_RATIO.value] = 0
        
        # Max drawdown
        metrics[MetricType.MAX_DRAWDOWN.value] = self._calculate_max_drawdown()
        
        # Calmar ratio
        max_dd = metrics[MetricType.MAX_DRAWDOWN.value]
        if max_dd > 0 and len(returns) > 0:
            annual_return = np.mean(returns) * 252  # Approximate
            metrics[MetricType.CALMAR_RATIO.value] = annual_return / max_dd
        else:
            metrics[MetricType.CALMAR_RATIO.value] = 0
        
        # Signal accuracy (if predictions available)
        trades_with_pred = [t for t in trades if t.predicted_direction is not None]
        if trades_with_pred:
            correct = sum(
                1 for t in trades_with_pred
                if (t.pnl > 0 and t.predicted_direction == t.direction) or
                   (t.pnl < 0 and t.predicted_direction != t.direction)
            )
            metrics[MetricType.SIGNAL_ACCURACY.value] = correct / len(trades_with_pred)
        
        # Custom metrics
        for name, func in self.custom_metrics.items():
            try:
                metrics[name] = func(trades, self.equity_curve)
            except Exception as e:
                logger.warning(f"Error calculating custom metric {name}: {e}")
        
        return metrics
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0
        
        equities = [e for _, e in self.equity_curve]
        peak = equities[0]
        max_dd = 0.0
        
        for equity in equities:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _check_thresholds(self, metrics: Dict[str, float]) -> List[PerformanceAlert]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        now = datetime.now()
        
        for metric_type, threshold in self.config.default_thresholds.items():
            value = metrics.get(metric_type.value)
            if value is None:
                continue
            
            alert_level = threshold.check(value)
            
            if alert_level in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
                # Check cooldown
                last_alert = self.alert_timestamps.get(metric_type)
                if last_alert:
                    if (now - last_alert).seconds < self.config.alert_cooldown_minutes * 60:
                        continue
                
                # Update consecutive count
                self.consecutive_alerts[metric_type] = self.consecutive_alerts.get(metric_type, 0) + 1
                
                # Escalate if consecutive
                if self.consecutive_alerts[metric_type] >= self.config.consecutive_alerts_threshold:
                    alert_level = AlertLevel.CRITICAL
                
                threshold_val = (
                    threshold.critical_threshold if alert_level == AlertLevel.CRITICAL
                    else threshold.warning_threshold
                )
                
                alert = PerformanceAlert(
                    timestamp=now,
                    metric_type=metric_type,
                    alert_level=alert_level,
                    current_value=value,
                    threshold_value=threshold_val,
                    message=self._generate_alert_message(metric_type, value, threshold_val, alert_level),
                    recommendation=self._generate_alert_recommendation(metric_type, alert_level)
                )
                
                alerts.append(alert)
                self.alert_timestamps[metric_type] = now
            else:
                # Reset consecutive counter if metric is healthy
                self.consecutive_alerts[metric_type] = 0
        
        # Check custom thresholds
        for name, threshold in self.custom_thresholds.items():
            value = metrics.get(name)
            if value is not None:
                alert_level = threshold.check(value)
                if alert_level in [AlertLevel.WARNING, AlertLevel.CRITICAL]:
                    alerts.append(PerformanceAlert(
                        timestamp=now,
                        metric_type=MetricType.CUSTOM,
                        alert_level=alert_level,
                        current_value=value,
                        threshold_value=threshold.warning_threshold,
                        message=f"Custom metric '{name}' at {value:.3f}",
                        recommendation="Review custom metric performance."
                    ))
        
        return alerts
    
    def _generate_alert_message(
        self,
        metric_type: MetricType,
        value: float,
        threshold: float,
        level: AlertLevel
    ) -> str:
        """Generate human-readable alert message."""
        level_str = "WARNING" if level == AlertLevel.WARNING else "CRITICAL"
        return (
            f"{level_str}: {metric_type.value} is {value:.3f} "
            f"(threshold: {threshold:.3f})"
        )
    
    def _generate_alert_recommendation(
        self,
        metric_type: MetricType,
        level: AlertLevel
    ) -> str:
        """Generate recommendation based on metric and alert level."""
        recommendations = {
            MetricType.WIN_RATE: {
                AlertLevel.WARNING: "Review entry signals and filters.",
                AlertLevel.CRITICAL: "Consider pausing trading. Evaluate signal quality."
            },
            MetricType.PROFIT_FACTOR: {
                AlertLevel.WARNING: "Review risk/reward ratios and exit rules.",
                AlertLevel.CRITICAL: "Trading system unprofitable. Immediate review needed."
            },
            MetricType.SHARPE_RATIO: {
                AlertLevel.WARNING: "Returns inconsistent. Review position sizing.",
                AlertLevel.CRITICAL: "Risk-adjusted returns poor. Consider model retraining."
            },
            MetricType.MAX_DRAWDOWN: {
                AlertLevel.WARNING: "Drawdown elevated. Consider reducing position size.",
                AlertLevel.CRITICAL: "Severe drawdown. Pause trading and review strategy."
            },
            MetricType.SIGNAL_ACCURACY: {
                AlertLevel.WARNING: "Signal accuracy declining. Monitor feature drift.",
                AlertLevel.CRITICAL: "Model predictions unreliable. Immediate retraining needed."
            }
        }
        
        return recommendations.get(metric_type, {}).get(
            level, "Review performance and consider adjustments."
        )
    
    def register_custom_metric(
        self,
        name: str,
        calculator: Callable,
        threshold: Optional[MetricThreshold] = None
    ) -> None:
        """Register a custom metric calculator."""
        self.custom_metrics[name] = calculator
        if threshold:
            self.custom_thresholds[name] = threshold
        logger.info(f"Registered custom metric: {name}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get latest metric values."""
        if self.snapshots:
            return self.snapshots[-1].metrics
        return self._calculate_metrics()
    
    def get_metric_history(
        self,
        metric_type: MetricType,
        lookback: int = 50
    ) -> List[tuple]:
        """Get metric history as list of (timestamp, value) tuples."""
        history = self.metric_history.get(metric_type, deque())
        return list(history)[-lookback:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        metrics = self.get_current_metrics()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_trades': len(self.closed_trades),
            'open_trades': len([t for t in self.trades if not t.is_closed]),
            'initial_equity': self.initial_equity,
            'current_equity': self.current_equity,
            'total_pnl': self.current_equity - self.initial_equity,
            'total_return_pct': (self.current_equity - self.initial_equity) / self.initial_equity * 100,
            'current_drawdown': self._calculate_drawdown(),
            'max_drawdown': self._calculate_max_drawdown(),
            'metrics': metrics,
            'active_alerts': len([a for a in self.alerts[-10:] if a.alert_level != AlertLevel.INFO]),
            'needs_retraining': self._should_retrain()
        }
    
    def _should_retrain(self) -> bool:
        """Determine if model retraining is recommended based on alerts."""
        if not self.alerts:
            return False
        
        recent_alerts = [a for a in self.alerts[-20:]]
        critical_count = sum(1 for a in recent_alerts if a.alert_level == AlertLevel.CRITICAL)
        
        return critical_count >= 2
    
    def needs_retraining(self) -> tuple[bool, str]:
        """Check if retraining is needed based on performance."""
        should_retrain = self._should_retrain()
        
        if should_retrain:
            reasons = []
            metrics = self.get_current_metrics()
            
            for metric_type, threshold in self.config.default_thresholds.items():
                value = metrics.get(metric_type.value)
                if value is not None:
                    if threshold.check(value) == AlertLevel.CRITICAL:
                        reasons.append(f"{metric_type.value}={value:.3f}")
            
            return True, f"Critical metrics: {', '.join(reasons)}"
        
        return False, "Performance within acceptable bounds"
    
    def export_history(self, filepath: str) -> None:
        """Export performance history to JSON file."""
        data = {
            'initial_equity': self.initial_equity,
            'trades': [t.to_dict() for t in self.closed_trades],
            'snapshots': [s.to_dict() for s in self.snapshots],
            'alerts': [a.to_dict() for a in self.alerts],
            'equity_curve': [(t.isoformat(), e) for t, e in self.equity_curve]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Performance history exported to {filepath}")
    
    def reset(self) -> None:
        """Reset monitor state."""
        self.trades = []
        self.closed_trades = []
        self.current_equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.equity_curve = [(datetime.now(), self.initial_equity)]
        self.alerts = []
        self.alert_timestamps = {}
        self.consecutive_alerts = {}
        self.trades_since_update = 0
        
        for history in self.metric_history.values():
            history.clear()
        
        self.snapshots.clear()
        
        logger.info("PerformanceMonitor reset")
