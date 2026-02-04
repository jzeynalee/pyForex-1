# alpha_factory/calibration_monitor.py
"""
Calibration Monitoring for Probabilistic Alpha Factory
=======================================================

This module provides tools for monitoring and evaluating the calibration
quality of the probabilistic predictions:

1. Brier Score: Measures accuracy of probabilistic predictions
2. Reliability Diagrams: Visual calibration assessment
3. Expected Calibration Error (ECE): Quantifies miscalibration
4. Feature Ablation: Identifies feature contribution to performance

Usage:
    monitor = CalibrationMonitor()
    monitor.record_prediction(probs, actual_regime)
    report = monitor.generate_report()
"""

import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Container for calibration metrics."""
    brier_score: float
    brier_score_bull: float
    brier_score_bear: float
    brier_score_neutral: float
    expected_calibration_error: float
    max_calibration_error: float
    accuracy: float
    n_samples: int
    reliability_data: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)


class CalibrationMonitor:
    """
    Monitors and evaluates calibration quality of probabilistic predictions.
    
    Tracks:
    - Predicted probabilities vs actual outcomes
    - Per-regime calibration
    - Feature-level contribution analysis
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize calibration monitor.
        
        Args:
            n_bins: Number of bins for reliability diagrams
        """
        self.n_bins = n_bins
        
        # Storage for predictions and outcomes
        self._predictions: List[Dict[str, float]] = []
        self._actuals: List[int] = []  # 0=bear, 1=neutral, 2=bull
        self._timestamps: List[datetime] = []
        
        # Feature contribution tracking
        self._feature_contributions: Dict[str, List[float]] = defaultdict(list)
        self._feature_outcomes: Dict[str, List[int]] = defaultdict(list)
        
        # Rolling metrics
        self._rolling_brier: List[float] = []
        self._rolling_accuracy: List[float] = []
    
    def record_prediction(
        self,
        probs: Dict[str, float],
        actual_regime: int,
        timestamp: Optional[datetime] = None,
        feature_contributions: Optional[Dict[str, float]] = None
    ):
        """
        Record a prediction and its outcome.
        
        Args:
            probs: Dictionary with 'bull', 'bear', 'neutral' probabilities
            actual_regime: Actual regime (0=bear, 1=neutral, 2=bull)
            timestamp: Optional timestamp
            feature_contributions: Optional feature contribution scores
        """
        self._predictions.append(probs)
        self._actuals.append(actual_regime)
        self._timestamps.append(timestamp or datetime.utcnow())
        
        # Track feature contributions
        if feature_contributions:
            for feat, contrib in feature_contributions.items():
                self._feature_contributions[feat].append(contrib)
                self._feature_outcomes[feat].append(actual_regime)
        
        # Update rolling metrics
        self._update_rolling_metrics()
    
    def _update_rolling_metrics(self, window: int = 100):
        """Update rolling metrics."""
        if len(self._predictions) < 10:
            return
        
        # Get recent predictions
        recent_preds = self._predictions[-window:]
        recent_actuals = self._actuals[-window:]
        
        # Calculate Brier score
        brier = self._calculate_brier_score(recent_preds, recent_actuals)
        self._rolling_brier.append(brier)
        
        # Calculate accuracy
        correct = 0
        for pred, actual in zip(recent_preds, recent_actuals):
            predicted = max(pred, key=pred.get)
            regime_map = {'bear': 0, 'neutral': 1, 'bull': 2}
            if regime_map.get(predicted, -1) == actual:
                correct += 1
        
        accuracy = correct / len(recent_preds)
        self._rolling_accuracy.append(accuracy)
    
    def _calculate_brier_score(
        self,
        predictions: List[Dict[str, float]],
        actuals: List[int]
    ) -> float:
        """Calculate Brier score for predictions."""
        if not predictions:
            return 0.0
        
        total_brier = 0.0
        
        for pred, actual in zip(predictions, actuals):
            # Convert actual to one-hot
            actual_probs = [0.0, 0.0, 0.0]
            actual_probs[actual] = 1.0
            
            # Get predicted probs
            pred_probs = [
                pred.get('bear', 0.33),
                pred.get('neutral', 0.34),
                pred.get('bull', 0.33)
            ]
            
            # Brier score = mean squared error
            brier = sum((p - a) ** 2 for p, a in zip(pred_probs, actual_probs)) / 3
            total_brier += brier
        
        return total_brier / len(predictions)
    
    def calculate_metrics(self) -> CalibrationMetrics:
        """Calculate comprehensive calibration metrics."""
        if len(self._predictions) < 10:
            return CalibrationMetrics(
                brier_score=0.0,
                brier_score_bull=0.0,
                brier_score_bear=0.0,
                brier_score_neutral=0.0,
                expected_calibration_error=0.0,
                max_calibration_error=0.0,
                accuracy=0.0,
                n_samples=len(self._predictions)
            )
        
        # Overall Brier score
        brier_score = self._calculate_brier_score(self._predictions, self._actuals)
        
        # Per-regime Brier scores
        brier_bull = self._calculate_regime_brier('bull', 2)
        brier_bear = self._calculate_regime_brier('bear', 0)
        brier_neutral = self._calculate_regime_brier('neutral', 1)
        
        # Expected Calibration Error
        ece, mce, reliability_data = self._calculate_calibration_error()
        
        # Accuracy
        correct = 0
        for pred, actual in zip(self._predictions, self._actuals):
            predicted = max(pred, key=pred.get)
            regime_map = {'bear': 0, 'neutral': 1, 'bull': 2}
            if regime_map.get(predicted, -1) == actual:
                correct += 1
        
        accuracy = correct / len(self._predictions)
        
        return CalibrationMetrics(
            brier_score=brier_score,
            brier_score_bull=brier_bull,
            brier_score_bear=brier_bear,
            brier_score_neutral=brier_neutral,
            expected_calibration_error=ece,
            max_calibration_error=mce,
            accuracy=accuracy,
            n_samples=len(self._predictions),
            reliability_data=reliability_data
        )
    
    def _calculate_regime_brier(self, regime_name: str, regime_idx: int) -> float:
        """Calculate Brier score for a specific regime."""
        total_brier = 0.0
        
        for pred, actual in zip(self._predictions, self._actuals):
            pred_prob = pred.get(regime_name, 0.33)
            actual_prob = 1.0 if actual == regime_idx else 0.0
            
            brier = (pred_prob - actual_prob) ** 2
            total_brier += brier
        
        return total_brier / len(self._predictions) if self._predictions else 0.0
    
    def _calculate_calibration_error(self) -> Tuple[float, float, Dict]:
        """
        Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        
        Returns:
            (ECE, MCE, reliability_data)
        """
        reliability_data = {'bull': [], 'bear': [], 'neutral': []}
        
        for regime_name, regime_idx in [('bull', 2), ('bear', 0), ('neutral', 1)]:
            # Get probabilities and outcomes for this regime
            probs = [pred.get(regime_name, 0.33) for pred in self._predictions]
            outcomes = [1 if actual == regime_idx else 0 for actual in self._actuals]
            
            # Bin probabilities
            bin_edges = np.linspace(0, 1, self.n_bins + 1)
            
            for i in range(self.n_bins):
                bin_mask = (np.array(probs) >= bin_edges[i]) & (np.array(probs) < bin_edges[i + 1])
                
                if bin_mask.sum() > 0:
                    bin_probs = np.array(probs)[bin_mask]
                    bin_outcomes = np.array(outcomes)[bin_mask]
                    
                    mean_prob = float(np.mean(bin_probs))
                    mean_outcome = float(np.mean(bin_outcomes))
                    
                    reliability_data[regime_name].append((mean_prob, mean_outcome))
        
        # Calculate ECE and MCE
        ece = 0.0
        mce = 0.0
        total_samples = 0
        
        for regime_name in ['bull', 'bear', 'neutral']:
            for mean_prob, mean_outcome in reliability_data[regime_name]:
                error = abs(mean_prob - mean_outcome)
                ece += error
                mce = max(mce, error)
                total_samples += 1
        
        ece = ece / total_samples if total_samples > 0 else 0.0
        
        return ece, mce, reliability_data
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance based on contribution to correct predictions.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance = {}
        
        for feat, contributions in self._feature_contributions.items():
            outcomes = self._feature_outcomes[feat]
            
            if len(contributions) < 10:
                continue
            
            # Calculate correlation between contribution and outcome direction
            # Positive contribution should correlate with bull (2), negative with bear (0)
            direction = [(o - 1) for o in outcomes]  # -1, 0, 1
            
            corr = np.corrcoef(contributions, direction)[0, 1]
            if np.isfinite(corr):
                importance[feat] = abs(corr)
        
        # Normalize
        if importance:
            max_imp = max(importance.values())
            if max_imp > 0:
                importance = {k: v / max_imp for k, v in importance.items()}
        
        return importance
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive calibration report."""
        metrics = self.calculate_metrics()
        feature_importance = self.get_feature_importance()
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'n_samples': metrics.n_samples,
            'metrics': {
                'brier_score': metrics.brier_score,
                'brier_score_bull': metrics.brier_score_bull,
                'brier_score_bear': metrics.brier_score_bear,
                'brier_score_neutral': metrics.brier_score_neutral,
                'expected_calibration_error': metrics.expected_calibration_error,
                'max_calibration_error': metrics.max_calibration_error,
                'accuracy': metrics.accuracy
            },
            'feature_importance': feature_importance,
            'rolling_brier': self._rolling_brier[-100:] if self._rolling_brier else [],
            'rolling_accuracy': self._rolling_accuracy[-100:] if self._rolling_accuracy else [],
            'reliability_data': metrics.reliability_data
        }
        
        return report
    
    def get_calibration_summary(self) -> str:
        """Get human-readable calibration summary."""
        metrics = self.calculate_metrics()
        
        summary = f"""
Calibration Summary
==================
Samples: {metrics.n_samples}

Brier Scores (lower is better, 0 = perfect):
  Overall: {metrics.brier_score:.4f}
  Bull:    {metrics.brier_score_bull:.4f}
  Bear:    {metrics.brier_score_bear:.4f}
  Neutral: {metrics.brier_score_neutral:.4f}

Calibration Error:
  ECE: {metrics.expected_calibration_error:.4f}
  MCE: {metrics.max_calibration_error:.4f}

Accuracy: {metrics.accuracy:.2%}
"""
        
        # Add interpretation
        if metrics.brier_score < 0.1:
            summary += "\n✓ Excellent calibration"
        elif metrics.brier_score < 0.2:
            summary += "\n✓ Good calibration"
        elif metrics.brier_score < 0.3:
            summary += "\n⚠ Moderate calibration - consider recalibration"
        else:
            summary += "\n✗ Poor calibration - recalibration needed"
        
        return summary
    
    def save_report(self, path: str):
        """Save calibration report to JSON file."""
        import json
        
        report = self.generate_report()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved calibration report to {path}")
    
    def reset(self):
        """Reset all tracked data."""
        self._predictions.clear()
        self._actuals.clear()
        self._timestamps.clear()
        self._feature_contributions.clear()
        self._feature_outcomes.clear()
        self._rolling_brier.clear()
        self._rolling_accuracy.clear()


class FeatureAblationAnalyzer:
    """
    Analyzes feature contribution through ablation studies.
    
    Measures how removing each feature affects prediction quality.
    """
    
    def __init__(self):
        self._baseline_metrics: Optional[CalibrationMetrics] = None
        self._ablation_results: Dict[str, CalibrationMetrics] = {}
    
    def set_baseline(self, metrics: CalibrationMetrics):
        """Set baseline metrics (with all features)."""
        self._baseline_metrics = metrics
    
    def record_ablation(self, feature_name: str, metrics: CalibrationMetrics):
        """Record metrics with a feature removed."""
        self._ablation_results[feature_name] = metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance based on ablation impact.
        
        Higher score = more important (removing it hurts performance more).
        """
        if self._baseline_metrics is None:
            return {}
        
        importance = {}
        baseline_brier = self._baseline_metrics.brier_score
        
        for feat, metrics in self._ablation_results.items():
            # Importance = increase in Brier score when feature is removed
            delta = metrics.brier_score - baseline_brier
            importance[feat] = max(0, delta)  # Only positive deltas matter
        
        # Normalize
        if importance:
            max_imp = max(importance.values())
            if max_imp > 0:
                importance = {k: v / max_imp for k, v in importance.items()}
        
        return importance
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate ablation analysis report."""
        importance = self.get_feature_importance()
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'baseline_brier': self._baseline_metrics.brier_score if self._baseline_metrics else None,
            'baseline_accuracy': self._baseline_metrics.accuracy if self._baseline_metrics else None,
            'feature_importance': dict(sorted_features),
            'ablation_details': {
                feat: {
                    'brier_score': metrics.brier_score,
                    'accuracy': metrics.accuracy,
                    'delta_brier': metrics.brier_score - self._baseline_metrics.brier_score
                }
                for feat, metrics in self._ablation_results.items()
            } if self._baseline_metrics else {}
        }


class OnlineCalibrationTracker:
    """
    Tracks calibration quality in real-time during live trading.
    
    Provides alerts when calibration degrades.
    """
    
    def __init__(
        self,
        alert_threshold_brier: float = 0.25,
        alert_threshold_ece: float = 0.15,
        window_size: int = 100
    ):
        self.alert_threshold_brier = alert_threshold_brier
        self.alert_threshold_ece = alert_threshold_ece
        self.window_size = window_size
        
        self.monitor = CalibrationMonitor()
        self._alerts: List[Dict] = []
    
    def update(
        self,
        probs: Dict[str, float],
        actual_regime: int,
        feature_contributions: Optional[Dict[str, float]] = None
    ) -> Optional[Dict]:
        """
        Update tracker with new prediction.
        
        Returns alert dict if calibration has degraded, None otherwise.
        """
        self.monitor.record_prediction(probs, actual_regime, feature_contributions=feature_contributions)
        
        # Check calibration every window_size predictions
        if len(self.monitor._predictions) % self.window_size == 0:
            metrics = self.monitor.calculate_metrics()
            
            alert = None
            
            if metrics.brier_score > self.alert_threshold_brier:
                alert = {
                    'type': 'brier_degradation',
                    'timestamp': datetime.utcnow().isoformat(),
                    'brier_score': metrics.brier_score,
                    'threshold': self.alert_threshold_brier,
                    'message': f"Brier score {metrics.brier_score:.4f} exceeds threshold {self.alert_threshold_brier}"
                }
            
            if metrics.expected_calibration_error > self.alert_threshold_ece:
                alert = {
                    'type': 'ece_degradation',
                    'timestamp': datetime.utcnow().isoformat(),
                    'ece': metrics.expected_calibration_error,
                    'threshold': self.alert_threshold_ece,
                    'message': f"ECE {metrics.expected_calibration_error:.4f} exceeds threshold {self.alert_threshold_ece}"
                }
            
            if alert:
                self._alerts.append(alert)
                logger.warning(f"Calibration alert: {alert['message']}")
                return alert
        
        return None
    
    def get_current_metrics(self) -> CalibrationMetrics:
        """Get current calibration metrics."""
        return self.monitor.calculate_metrics()
    
    def get_alerts(self) -> List[Dict]:
        """Get all alerts."""
        return self._alerts.copy()
    
    def clear_alerts(self):
        """Clear alert history."""
        self._alerts.clear()


def calculate_brier_skill_score(
    forecast_brier: float,
    reference_brier: float = 0.25  # Climatological reference (random guess)
) -> float:
    """
    Calculate Brier Skill Score (BSS).
    
    BSS = 1 - (Brier_forecast / Brier_reference)
    
    BSS > 0: Better than reference
    BSS = 0: Same as reference
    BSS < 0: Worse than reference
    BSS = 1: Perfect forecast
    
    Args:
        forecast_brier: Brier score of the forecast
        reference_brier: Brier score of reference forecast (default: random guess)
    
    Returns:
        Brier Skill Score
    """
    if reference_brier <= 0:
        return 0.0
    
    return 1.0 - (forecast_brier / reference_brier)
