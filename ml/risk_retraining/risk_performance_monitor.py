# ml/risk_retraining/risk_performance_monitor.py
"""
Performance Monitor for Risk Management Models.

Tracks specialized metrics for:
- TCN Risk Model: Direction accuracy, volatility MAE, quantile calibration
- GBM Meta-Labeling: Precision, recall, filter improvement
- RL Exit Optimizer: Sharpe improvement, exit timing metrics

Extends the base PerformanceMonitor with risk-specific metrics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import json

from .risk_retraining_config import (
    RiskModelType, RiskRetrainingConfig,
    TCNRiskMetrics, GBMMetaMetrics, RLExitMetrics
)

logger = logging.getLogger(__name__)


# =============================================================================
# Metric Types
# =============================================================================

class MetricStatus(Enum):
    """Status of a metric relative to its threshold."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class MetricSnapshot:
    """Point-in-time metric measurement."""
    name: str
    value: float
    threshold: float
    status: MetricStatus
    timestamp: datetime
    is_minimum: bool = True     # True if value should be >= threshold
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'threshold': self.threshold,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'is_minimum': self.is_minimum,
        }


@dataclass
class ModelHealth:
    """Overall health status of a model."""
    model_type: RiskModelType
    status: MetricStatus
    metrics: List[MetricSnapshot]
    last_updated: datetime
    needs_retraining: bool
    reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'model_type': self.model_type.name,
            'status': self.status.value,
            'metrics': [m.to_dict() for m in self.metrics],
            'last_updated': self.last_updated.isoformat(),
            'needs_retraining': self.needs_retraining,
            'reason': self.reason,
        }


# =============================================================================
# Metric Calculators
# =============================================================================

class TCNRiskMetricCalculator:
    """Calculates metrics for TCN Risk Model."""
    
    @staticmethod
    def direction_accuracy(
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Calculate direction prediction accuracy."""
        if len(predictions) == 0:
            return 0.0
        pred_classes = np.argmax(predictions, axis=1)
        return np.mean(pred_classes == targets)
    
    @staticmethod
    def direction_f1(
        predictions: np.ndarray,
        targets: np.ndarray,
        average: str = 'weighted'
    ) -> float:
        """Calculate F1 score for direction prediction."""
        from sklearn.metrics import f1_score
        if len(predictions) == 0:
            return 0.0
        pred_classes = np.argmax(predictions, axis=1)
        return f1_score(targets, pred_classes, average=average, zero_division=0)
    
    @staticmethod
    def volatility_mae(
        predicted_vol: np.ndarray,
        realized_vol: np.ndarray
    ) -> float:
        """Calculate MAE for volatility predictions."""
        if len(predicted_vol) == 0:
            return float('inf')
        return np.mean(np.abs(predicted_vol - realized_vol))
    
    @staticmethod
    def volatility_mape(
        predicted_vol: np.ndarray,
        realized_vol: np.ndarray,
        epsilon: float = 1e-8
    ) -> float:
        """Calculate MAPE for volatility predictions."""
        if len(predicted_vol) == 0:
            return float('inf')
        return np.mean(np.abs((predicted_vol - realized_vol) / (realized_vol + epsilon)))
    
    @staticmethod
    def volatility_correlation(
        predicted_vol: np.ndarray,
        realized_vol: np.ndarray
    ) -> float:
        """Calculate correlation between predicted and realized volatility."""
        if len(predicted_vol) < 2:
            return 0.0
        corr = np.corrcoef(predicted_vol, realized_vol)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    
    @staticmethod
    def quantile_pinball_loss(
        predicted_quantiles: np.ndarray,
        actual_values: np.ndarray,
        quantile_levels: List[float]
    ) -> float:
        """
        Calculate average pinball loss across all quantiles.
        
        Pinball loss = tau * max(y - q, 0) + (1-tau) * max(q - y, 0)
        """
        if len(predicted_quantiles) == 0:
            return float('inf')
        
        total_loss = 0.0
        for i, tau in enumerate(quantile_levels):
            q = predicted_quantiles[:, i]
            y = actual_values
            
            errors = y - q
            loss = np.where(
                errors >= 0,
                tau * errors,
                (tau - 1) * errors
            )
            total_loss += np.mean(loss)
        
        return total_loss / len(quantile_levels)
    
    @staticmethod
    def quantile_coverage(
        predicted_quantiles: np.ndarray,
        actual_values: np.ndarray,
        quantile_idx: int
    ) -> float:
        """
        Calculate empirical coverage for a specific quantile.
        
        Coverage = fraction of actual values below the predicted quantile.
        """
        if len(predicted_quantiles) == 0:
            return 0.0
        q = predicted_quantiles[:, quantile_idx]
        return np.mean(actual_values <= q)
    
    @staticmethod
    def quantile_crossing_rate(predicted_quantiles: np.ndarray) -> float:
        """
        Calculate rate of quantile crossing violations.
        
        Quantiles should be monotonically increasing: Q5 < Q25 < Q50 < Q75 < Q95
        """
        if len(predicted_quantiles) == 0:
            return 0.0
        
        violations = 0
        total = 0
        
        for i in range(predicted_quantiles.shape[1] - 1):
            lower = predicted_quantiles[:, i]
            upper = predicted_quantiles[:, i + 1]
            violations += np.sum(lower > upper)
            total += len(lower)
        
        return violations / max(total, 1)
    
    @staticmethod
    def sl_hit_before_tp_rate(
        trade_outcomes: pd.DataFrame
    ) -> float:
        """
        Calculate rate of SL being hit before TP.
        
        trade_outcomes should have columns: 'exit_reason' with values 'sl', 'tp', 'timeout'
        """
        if len(trade_outcomes) == 0:
            return 0.0
        
        sl_exits = (trade_outcomes['exit_reason'] == 'sl').sum()
        tp_exits = (trade_outcomes['exit_reason'] == 'tp').sum()
        
        if sl_exits + tp_exits == 0:
            return 0.5  # No data
        
        return sl_exits / (sl_exits + tp_exits)


class GBMMetaMetricCalculator:
    """Calculates metrics for GBM Meta-Labeling Model."""
    
    @staticmethod
    def precision(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate precision (when we predict 'take trade', how often correct)."""
        from sklearn.metrics import precision_score
        if len(predictions) == 0:
            return 0.0
        pred_binary = (predictions >= 0.5).astype(int)
        return precision_score(targets, pred_binary, zero_division=0)
    
    @staticmethod
    def recall(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate recall (of winning trades, how many did we catch)."""
        from sklearn.metrics import recall_score
        if len(predictions) == 0:
            return 0.0
        pred_binary = (predictions >= 0.5).astype(int)
        return recall_score(targets, pred_binary, zero_division=0)
    
    @staticmethod
    def f1(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate F1 score."""
        from sklearn.metrics import f1_score
        if len(predictions) == 0:
            return 0.0
        pred_binary = (predictions >= 0.5).astype(int)
        return f1_score(targets, pred_binary, zero_division=0)
    
    @staticmethod
    def filter_rate(predictions: np.ndarray, threshold: float = 0.5) -> float:
        """Calculate what fraction of trades are filtered out."""
        if len(predictions) == 0:
            return 0.0
        return np.mean(predictions < threshold)
    
    @staticmethod
    def filtered_win_rate(
        predictions: np.ndarray,
        trade_results: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """Calculate win rate of trades that pass the filter."""
        mask = predictions >= threshold
        if mask.sum() == 0:
            return 0.0
        return np.mean(trade_results[mask] > 0)
    
    @staticmethod
    def filter_improvement(
        predictions: np.ndarray,
        trade_results: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """Calculate improvement in win rate from filtering."""
        if len(predictions) == 0:
            return 0.0
        
        baseline_win_rate = np.mean(trade_results > 0)
        filtered_win_rate = GBMMetaMetricCalculator.filtered_win_rate(
            predictions, trade_results, threshold
        )
        
        return filtered_win_rate - baseline_win_rate
    
    @staticmethod
    def profit_factor(
        predictions: np.ndarray,
        trade_results: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """Calculate profit factor of filtered trades."""
        mask = predictions >= threshold
        if mask.sum() == 0:
            return 0.0
        
        filtered_results = trade_results[mask]
        wins = filtered_results[filtered_results > 0].sum()
        losses = abs(filtered_results[filtered_results < 0].sum())
        
        if losses == 0:
            return float('inf') if wins > 0 else 0.0
        return wins / losses


class RLExitMetricCalculator:
    """Calculates metrics for RL Exit Optimizer."""
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio of returns."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - risk_free
        std = np.std(excess_returns)
        if std == 0:
            return 0.0
        return np.mean(excess_returns) / std * np.sqrt(252)  # Annualized
    
    @staticmethod
    def sharpe_improvement_vs_fixed(
        rl_returns: np.ndarray,
        fixed_returns: np.ndarray,
        risk_free: float = 0.0
    ) -> float:
        """Calculate Sharpe improvement over fixed exit strategy."""
        rl_sharpe = RLExitMetricCalculator.sharpe_ratio(rl_returns, risk_free)
        fixed_sharpe = RLExitMetricCalculator.sharpe_ratio(fixed_returns, risk_free)
        return rl_sharpe - fixed_sharpe
    
    @staticmethod
    def average_reward(rewards: np.ndarray) -> float:
        """Calculate average episode reward."""
        if len(rewards) == 0:
            return 0.0
        return np.mean(rewards)
    
    @staticmethod
    def policy_entropy(action_probs: np.ndarray, epsilon: float = 1e-10) -> float:
        """Calculate entropy of action distribution (higher = more exploration)."""
        if len(action_probs) == 0:
            return 0.0
        # Clip to avoid log(0)
        probs = np.clip(action_probs, epsilon, 1.0)
        return -np.mean(np.sum(probs * np.log(probs), axis=-1))
    
    @staticmethod
    def premature_exit_rate(
        exit_times: np.ndarray,
        optimal_exit_times: np.ndarray,
        threshold_ratio: float = 0.5
    ) -> float:
        """Calculate rate of exiting too early (before X% of optimal time)."""
        if len(exit_times) == 0:
            return 0.0
        early_exits = exit_times < (optimal_exit_times * threshold_ratio)
        return np.mean(early_exits)
    
    @staticmethod
    def profitable_exit_ratio(exit_pnls: np.ndarray) -> float:
        """Calculate ratio of profitable exits."""
        if len(exit_pnls) == 0:
            return 0.0
        return np.mean(exit_pnls > 0)
    
    @staticmethod
    def exit_drawdown_from_optimal(
        actual_pnl: np.ndarray,
        optimal_pnl: np.ndarray
    ) -> float:
        """Calculate average drawdown from optimal exit point."""
        if len(actual_pnl) == 0:
            return 0.0
        drawdown = optimal_pnl - actual_pnl
        return np.mean(np.maximum(drawdown, 0))


# =============================================================================
# Performance Monitor
# =============================================================================

@dataclass
class PerformanceWindow:
    """Rolling window for performance data."""
    
    max_size: int = 1000
    
    # Data storage
    direction_preds: deque = field(default_factory=lambda: deque(maxlen=1000))
    direction_targets: deque = field(default_factory=lambda: deque(maxlen=1000))
    volatility_preds: deque = field(default_factory=lambda: deque(maxlen=1000))
    volatility_realized: deque = field(default_factory=lambda: deque(maxlen=1000))
    quantile_preds: deque = field(default_factory=lambda: deque(maxlen=1000))
    quantile_actuals: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Meta-labeling data
    meta_preds: deque = field(default_factory=lambda: deque(maxlen=1000))
    meta_targets: deque = field(default_factory=lambda: deque(maxlen=1000))
    trade_results: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # RL data
    rl_rewards: deque = field(default_factory=lambda: deque(maxlen=1000))
    rl_returns: deque = field(default_factory=lambda: deque(maxlen=1000))
    fixed_returns: deque = field(default_factory=lambda: deque(maxlen=1000))
    action_probs: deque = field(default_factory=lambda: deque(maxlen=1000))
    exit_pnls: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    # Trade outcomes
    trade_outcomes: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        # Recreate deques with correct maxlen
        self.direction_preds = deque(maxlen=self.max_size)
        self.direction_targets = deque(maxlen=self.max_size)
        self.volatility_preds = deque(maxlen=self.max_size)
        self.volatility_realized = deque(maxlen=self.max_size)
        self.quantile_preds = deque(maxlen=self.max_size)
        self.quantile_actuals = deque(maxlen=self.max_size)
        self.meta_preds = deque(maxlen=self.max_size)
        self.meta_targets = deque(maxlen=self.max_size)
        self.trade_results = deque(maxlen=self.max_size)
        self.rl_rewards = deque(maxlen=self.max_size)
        self.rl_returns = deque(maxlen=self.max_size)
        self.fixed_returns = deque(maxlen=self.max_size)
        self.action_probs = deque(maxlen=self.max_size)
        self.exit_pnls = deque(maxlen=self.max_size)


class RiskPerformanceMonitor:
    """
    Performance monitor for Risk Management models.
    
    Tracks metrics for all risk model types and triggers retraining
    when performance degrades below thresholds.
    """
    
    def __init__(
        self,
        config: RiskRetrainingConfig,
        window_size: int = 1000,
        grace_period_hours: int = 24,
    ):
        self.config = config
        self.window_size = window_size
        self.grace_period = timedelta(hours=grace_period_hours)
        
        # Performance windows per model
        self.windows: Dict[RiskModelType, PerformanceWindow] = {
            RiskModelType.TCN_RISK: PerformanceWindow(max_size=window_size),
            RiskModelType.GBM_META: PerformanceWindow(max_size=window_size),
            RiskModelType.RL_EXIT: PerformanceWindow(max_size=window_size),
        }
        
        # Last retraining times
        self.last_retrained: Dict[RiskModelType, Optional[datetime]] = {
            model_type: None for model_type in RiskModelType
        }
        
        # Metric history
        self.metric_history: Dict[RiskModelType, List[Dict]] = {
            model_type: [] for model_type in RiskModelType
        }
        
        # Calculators
        self.tcn_calc = TCNRiskMetricCalculator()
        self.gbm_calc = GBMMetaMetricCalculator()
        self.rl_calc = RLExitMetricCalculator()
        
        # Callbacks
        self.on_health_change: Optional[Callable] = None
        
        logger.info(f"RiskPerformanceMonitor initialized for profile: {config.profile_name}")
    
    # =========================================================================
    # Data Recording
    # =========================================================================
    
    def record_tcn_prediction(
        self,
        direction_pred: np.ndarray,
        direction_target: int,
        volatility_pred: float,
        volatility_realized: float,
        quantile_pred: Optional[np.ndarray] = None,
        quantile_actual: Optional[float] = None,
    ):
        """Record a TCN Risk Model prediction."""
        window = self.windows[RiskModelType.TCN_RISK]
        
        window.direction_preds.append(direction_pred)
        window.direction_targets.append(direction_target)
        window.volatility_preds.append(volatility_pred)
        window.volatility_realized.append(volatility_realized)
        
        if quantile_pred is not None:
            window.quantile_preds.append(quantile_pred)
        if quantile_actual is not None:
            window.quantile_actuals.append(quantile_actual)
    
    def record_meta_prediction(
        self,
        prediction: float,
        target: int,
        trade_result: float,
    ):
        """Record a GBM Meta-Labeling prediction."""
        window = self.windows[RiskModelType.GBM_META]
        
        window.meta_preds.append(prediction)
        window.meta_targets.append(target)
        window.trade_results.append(trade_result)
    
    def record_rl_episode(
        self,
        reward: float,
        returns: float,
        fixed_returns: float,
        action_probs: Optional[np.ndarray] = None,
        exit_pnl: Optional[float] = None,
    ):
        """Record an RL Exit episode result."""
        window = self.windows[RiskModelType.RL_EXIT]
        
        window.rl_rewards.append(reward)
        window.rl_returns.append(returns)
        window.fixed_returns.append(fixed_returns)
        
        if action_probs is not None:
            window.action_probs.append(action_probs)
        if exit_pnl is not None:
            window.exit_pnls.append(exit_pnl)
    
    def record_trade_outcome(
        self,
        entry_time: datetime,
        exit_time: datetime,
        exit_reason: str,  # 'sl', 'tp', 'timeout', 'rl_exit'
        pnl: float,
        direction: str,
    ):
        """Record a trade outcome for SL/TP analysis."""
        window = self.windows[RiskModelType.TCN_RISK]
        window.trade_outcomes.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'exit_reason': exit_reason,
            'pnl': pnl,
            'direction': direction,
        })
        
        # Keep only recent outcomes
        if len(window.trade_outcomes) > self.window_size:
            window.trade_outcomes = window.trade_outcomes[-self.window_size:]
    
    # =========================================================================
    # Metric Calculation
    # =========================================================================
    
    def calculate_tcn_metrics(self) -> Dict[str, MetricSnapshot]:
        """Calculate all TCN Risk Model metrics."""
        window = self.windows[RiskModelType.TCN_RISK]
        metrics_config = self.config.tcn_metrics
        now = datetime.now()
        snapshots = {}
        
        # Direction metrics
        if len(window.direction_preds) >= 50:
            preds = np.array(list(window.direction_preds))
            targets = np.array(list(window.direction_targets))
            
            acc = self.tcn_calc.direction_accuracy(preds, targets)
            snapshots['direction_accuracy'] = MetricSnapshot(
                name='direction_accuracy',
                value=acc,
                threshold=metrics_config.min_direction_accuracy,
                status=self._get_status(acc, metrics_config.min_direction_accuracy, is_minimum=True),
                timestamp=now,
                is_minimum=True,
            )
            
            f1 = self.tcn_calc.direction_f1(preds, targets)
            snapshots['direction_f1'] = MetricSnapshot(
                name='direction_f1',
                value=f1,
                threshold=metrics_config.min_direction_f1,
                status=self._get_status(f1, metrics_config.min_direction_f1, is_minimum=True),
                timestamp=now,
                is_minimum=True,
            )
        
        # Volatility metrics
        if len(window.volatility_preds) >= 50:
            vol_preds = np.array(list(window.volatility_preds))
            vol_realized = np.array(list(window.volatility_realized))
            
            mae = self.tcn_calc.volatility_mae(vol_preds, vol_realized)
            snapshots['volatility_mae'] = MetricSnapshot(
                name='volatility_mae',
                value=mae,
                threshold=metrics_config.max_volatility_mae,
                status=self._get_status(mae, metrics_config.max_volatility_mae, is_minimum=False),
                timestamp=now,
                is_minimum=False,
            )
            
            mape = self.tcn_calc.volatility_mape(vol_preds, vol_realized)
            snapshots['volatility_mape'] = MetricSnapshot(
                name='volatility_mape',
                value=mape,
                threshold=metrics_config.max_volatility_mape,
                status=self._get_status(mape, metrics_config.max_volatility_mape, is_minimum=False),
                timestamp=now,
                is_minimum=False,
            )
            
            corr = self.tcn_calc.volatility_correlation(vol_preds, vol_realized)
            snapshots['volatility_correlation'] = MetricSnapshot(
                name='volatility_correlation',
                value=corr,
                threshold=metrics_config.min_volatility_correlation,
                status=self._get_status(corr, metrics_config.min_volatility_correlation, is_minimum=True),
                timestamp=now,
                is_minimum=True,
            )
        
        # Quantile metrics
        if len(window.quantile_preds) >= 50:
            q_preds = np.array(list(window.quantile_preds))
            q_actuals = np.array(list(window.quantile_actuals))
            quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
            
            pinball = self.tcn_calc.quantile_pinball_loss(q_preds, q_actuals, quantile_levels)
            snapshots['quantile_pinball_loss'] = MetricSnapshot(
                name='quantile_pinball_loss',
                value=pinball,
                threshold=metrics_config.max_quantile_pinball_loss,
                status=self._get_status(pinball, metrics_config.max_quantile_pinball_loss, is_minimum=False),
                timestamp=now,
                is_minimum=False,
            )
            
            # Q5 coverage
            q5_cov = self.tcn_calc.quantile_coverage(q_preds, q_actuals, 0)
            q5_status = MetricStatus.HEALTHY
            if q5_cov < metrics_config.min_quantile_coverage_q5 or q5_cov > metrics_config.max_quantile_coverage_q5:
                q5_status = MetricStatus.WARNING
            snapshots['quantile_coverage_q5'] = MetricSnapshot(
                name='quantile_coverage_q5',
                value=q5_cov,
                threshold=0.05,  # Target
                status=q5_status,
                timestamp=now,
            )
            
            # Q95 coverage
            q95_cov = self.tcn_calc.quantile_coverage(q_preds, q_actuals, 4)
            q95_status = MetricStatus.HEALTHY
            if q95_cov < metrics_config.min_quantile_coverage_q95 or q95_cov > metrics_config.max_quantile_coverage_q95:
                q95_status = MetricStatus.WARNING
            snapshots['quantile_coverage_q95'] = MetricSnapshot(
                name='quantile_coverage_q95',
                value=q95_cov,
                threshold=0.95,  # Target
                status=q95_status,
                timestamp=now,
            )
            
            crossing = self.tcn_calc.quantile_crossing_rate(q_preds)
            snapshots['quantile_crossing_rate'] = MetricSnapshot(
                name='quantile_crossing_rate',
                value=crossing,
                threshold=metrics_config.max_quantile_crossing_rate,
                status=self._get_status(crossing, metrics_config.max_quantile_crossing_rate, is_minimum=False),
                timestamp=now,
                is_minimum=False,
            )
        
        # Trade outcome metrics
        if len(window.trade_outcomes) >= 20:
            outcomes_df = pd.DataFrame(window.trade_outcomes)
            sl_rate = self.tcn_calc.sl_hit_before_tp_rate(outcomes_df)
            snapshots['sl_hit_before_tp_rate'] = MetricSnapshot(
                name='sl_hit_before_tp_rate',
                value=sl_rate,
                threshold=metrics_config.max_sl_hit_before_tp_rate,
                status=self._get_status(sl_rate, metrics_config.max_sl_hit_before_tp_rate, is_minimum=False),
                timestamp=now,
                is_minimum=False,
            )
        
        return snapshots
    
    def calculate_gbm_metrics(self) -> Dict[str, MetricSnapshot]:
        """Calculate all GBM Meta-Labeling metrics."""
        window = self.windows[RiskModelType.GBM_META]
        metrics_config = self.config.gbm_metrics
        now = datetime.now()
        snapshots = {}
        
        if len(window.meta_preds) < 50:
            return snapshots
        
        preds = np.array(list(window.meta_preds))
        targets = np.array(list(window.meta_targets))
        results = np.array(list(window.trade_results))
        
        # Core metrics
        prec = self.gbm_calc.precision(preds, targets)
        snapshots['precision'] = MetricSnapshot(
            name='precision',
            value=prec,
            threshold=metrics_config.min_precision,
            status=self._get_status(prec, metrics_config.min_precision, is_minimum=True),
            timestamp=now,
            is_minimum=True,
        )
        
        rec = self.gbm_calc.recall(preds, targets)
        snapshots['recall'] = MetricSnapshot(
            name='recall',
            value=rec,
            threshold=metrics_config.min_recall,
            status=self._get_status(rec, metrics_config.min_recall, is_minimum=True),
            timestamp=now,
            is_minimum=True,
        )
        
        f1 = self.gbm_calc.f1(preds, targets)
        snapshots['f1'] = MetricSnapshot(
            name='f1',
            value=f1,
            threshold=metrics_config.min_f1,
            status=self._get_status(f1, metrics_config.min_f1, is_minimum=True),
            timestamp=now,
            is_minimum=True,
        )
        
        # Filter metrics
        filter_rate = self.gbm_calc.filter_rate(preds)
        fr_status = MetricStatus.HEALTHY
        if filter_rate < metrics_config.min_filter_rate or filter_rate > metrics_config.max_filter_rate:
            fr_status = MetricStatus.WARNING
        snapshots['filter_rate'] = MetricSnapshot(
            name='filter_rate',
            value=filter_rate,
            threshold=metrics_config.target_filter_rate,
            status=fr_status,
            timestamp=now,
        )
        
        filtered_wr = self.gbm_calc.filtered_win_rate(preds, results)
        snapshots['filtered_win_rate'] = MetricSnapshot(
            name='filtered_win_rate',
            value=filtered_wr,
            threshold=metrics_config.min_filtered_win_rate,
            status=self._get_status(filtered_wr, metrics_config.min_filtered_win_rate, is_minimum=True),
            timestamp=now,
            is_minimum=True,
        )
        
        improvement = self.gbm_calc.filter_improvement(preds, results)
        snapshots['filter_improvement'] = MetricSnapshot(
            name='filter_improvement',
            value=improvement,
            threshold=metrics_config.min_filter_improvement,
            status=self._get_status(improvement, metrics_config.min_filter_improvement, is_minimum=True),
            timestamp=now,
            is_minimum=True,
        )
        
        pf = self.gbm_calc.profit_factor(preds, results)
        snapshots['profit_factor'] = MetricSnapshot(
            name='profit_factor',
            value=pf,
            threshold=metrics_config.min_filtered_profit_factor,
            status=self._get_status(pf, metrics_config.min_filtered_profit_factor, is_minimum=True),
            timestamp=now,
            is_minimum=True,
        )
        
        return snapshots
    
    def calculate_rl_metrics(self) -> Dict[str, MetricSnapshot]:
        """Calculate all RL Exit Optimizer metrics."""
        window = self.windows[RiskModelType.RL_EXIT]
        metrics_config = self.config.rl_metrics
        now = datetime.now()
        snapshots = {}
        
        if len(window.rl_returns) < 30:
            return snapshots
        
        rl_returns = np.array(list(window.rl_returns))
        fixed_returns = np.array(list(window.fixed_returns))
        rewards = np.array(list(window.rl_rewards))
        
        # Sharpe improvement
        sharpe_imp = self.rl_calc.sharpe_improvement_vs_fixed(rl_returns, fixed_returns)
        snapshots['sharpe_vs_fixed'] = MetricSnapshot(
            name='sharpe_vs_fixed',
            value=sharpe_imp,
            threshold=metrics_config.min_sharpe_vs_fixed,
            status=self._get_status(sharpe_imp, metrics_config.min_sharpe_vs_fixed, is_minimum=True),
            timestamp=now,
            is_minimum=True,
        )
        
        # Average reward
        avg_reward = self.rl_calc.average_reward(rewards)
        snapshots['average_reward'] = MetricSnapshot(
            name='average_reward',
            value=avg_reward,
            threshold=metrics_config.min_average_reward,
            status=self._get_status(avg_reward, metrics_config.min_average_reward, is_minimum=True),
            timestamp=now,
            is_minimum=True,
        )
        
        # Exit metrics
        if len(window.exit_pnls) >= 30:
            exit_pnls = np.array(list(window.exit_pnls))
            profitable_ratio = self.rl_calc.profitable_exit_ratio(exit_pnls)
            snapshots['profitable_exit_ratio'] = MetricSnapshot(
                name='profitable_exit_ratio',
                value=profitable_ratio,
                threshold=metrics_config.min_profitable_exit_ratio,
                status=self._get_status(profitable_ratio, metrics_config.min_profitable_exit_ratio, is_minimum=True),
                timestamp=now,
                is_minimum=True,
            )
        
        # Policy entropy
        if len(window.action_probs) >= 30:
            action_probs = np.array(list(window.action_probs))
            entropy = self.rl_calc.policy_entropy(action_probs)
            # Higher entropy is generally better (more exploration)
            # We check for entropy collapse
            snapshots['policy_entropy'] = MetricSnapshot(
                name='policy_entropy',
                value=entropy,
                threshold=0.5,  # Minimum acceptable entropy
                status=MetricStatus.WARNING if entropy < 0.3 else MetricStatus.HEALTHY,
                timestamp=now,
                is_minimum=True,
            )
        
        return snapshots
    
    # =========================================================================
    # Health Assessment
    # =========================================================================
    
    def _get_status(
        self,
        value: float,
        threshold: float,
        is_minimum: bool
    ) -> MetricStatus:
        """Determine metric status based on threshold."""
        if is_minimum:
            if value >= threshold:
                return MetricStatus.HEALTHY
            elif value >= threshold * 0.9:
                return MetricStatus.WARNING
            else:
                return MetricStatus.CRITICAL
        else:
            if value <= threshold:
                return MetricStatus.HEALTHY
            elif value <= threshold * 1.1:
                return MetricStatus.WARNING
            else:
                return MetricStatus.CRITICAL
    
    def get_model_health(self, model_type: RiskModelType) -> ModelHealth:
        """Get overall health status for a model."""
        now = datetime.now()
        
        # Calculate metrics based on model type
        if model_type == RiskModelType.TCN_RISK:
            metrics = self.calculate_tcn_metrics()
        elif model_type == RiskModelType.GBM_META:
            metrics = self.calculate_gbm_metrics()
        elif model_type == RiskModelType.RL_EXIT:
            metrics = self.calculate_rl_metrics()
        else:
            return ModelHealth(
                model_type=model_type,
                status=MetricStatus.UNKNOWN,
                metrics=[],
                last_updated=now,
                needs_retraining=False,
                reason="Unknown model type",
            )
        
        if not metrics:
            return ModelHealth(
                model_type=model_type,
                status=MetricStatus.UNKNOWN,
                metrics=[],
                last_updated=now,
                needs_retraining=False,
                reason="Insufficient data",
            )
        
        # Check grace period
        last_retrained = self.last_retrained.get(model_type)
        in_grace_period = (
            last_retrained is not None and
            (now - last_retrained) < self.grace_period
        )
        
        # Determine overall status
        snapshots = list(metrics.values())
        statuses = [s.status for s in snapshots]
        
        critical_count = sum(1 for s in statuses if s == MetricStatus.CRITICAL)
        warning_count = sum(1 for s in statuses if s == MetricStatus.WARNING)
        
        if critical_count > 0:
            overall_status = MetricStatus.CRITICAL
        elif warning_count > len(statuses) * 0.3:
            overall_status = MetricStatus.WARNING
        else:
            overall_status = MetricStatus.HEALTHY
        
        # Determine if retraining needed
        needs_retraining = (
            overall_status == MetricStatus.CRITICAL and
            not in_grace_period
        )
        
        # Build reason
        reason = None
        if needs_retraining:
            critical_metrics = [s.name for s in snapshots if s.status == MetricStatus.CRITICAL]
            reason = f"Critical metrics: {', '.join(critical_metrics)}"
        elif in_grace_period:
            reason = "In grace period after retraining"
        
        return ModelHealth(
            model_type=model_type,
            status=overall_status,
            metrics=snapshots,
            last_updated=now,
            needs_retraining=needs_retraining,
            reason=reason,
        )
    
    def get_all_health(self) -> Dict[RiskModelType, ModelHealth]:
        """Get health status for all models."""
        return {
            model_type: self.get_model_health(model_type)
            for model_type in RiskModelType
        }
    
    def check_retraining_needed(self) -> List[Tuple[RiskModelType, str]]:
        """Check which models need retraining and why."""
        results = []
        
        for model_type in RiskModelType:
            health = self.get_model_health(model_type)
            if health.needs_retraining:
                results.append((model_type, health.reason or "Performance degradation"))
        
        return results
    
    def mark_retrained(self, model_type: RiskModelType):
        """Mark a model as having been retrained."""
        self.last_retrained[model_type] = datetime.now()
        logger.info(f"Marked {model_type.name} as retrained")
    
    # =========================================================================
    # History and Export
    # =========================================================================
    
    def save_metrics_snapshot(self, model_type: RiskModelType):
        """Save current metrics to history."""
        if model_type == RiskModelType.TCN_RISK:
            metrics = self.calculate_tcn_metrics()
        elif model_type == RiskModelType.GBM_META:
            metrics = self.calculate_gbm_metrics()
        elif model_type == RiskModelType.RL_EXIT:
            metrics = self.calculate_rl_metrics()
        else:
            return
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {name: m.to_dict() for name, m in metrics.items()},
        }
        
        self.metric_history[model_type].append(snapshot)
        
        # Keep only last 1000 snapshots
        if len(self.metric_history[model_type]) > 1000:
            self.metric_history[model_type] = self.metric_history[model_type][-1000:]
    
    def export_history(self, filepath: str):
        """Export metric history to JSON."""
        export_data = {
            model_type.name: history
            for model_type, history in self.metric_history.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported metric history to {filepath}")
    
    def get_metric_trend(
        self,
        model_type: RiskModelType,
        metric_name: str,
        window: int = 20
    ) -> Optional[List[float]]:
        """Get recent trend for a specific metric."""
        history = self.metric_history.get(model_type, [])
        
        values = []
        for snapshot in history[-window:]:
            metrics = snapshot.get('metrics', {})
            if metric_name in metrics:
                values.append(metrics[metric_name].get('value', 0))
        
        return values if values else None