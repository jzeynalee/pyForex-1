"""
Drift Detection Module for pyForex ML System.

Detects data drift (feature distribution changes) and concept drift
(relationship between features and target changes) to trigger model retraining.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import logging
from scipy import stats
from collections import deque

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift that can be detected."""
    NONE = "none"
    DATA_DRIFT = "data_drift"           # Feature distribution changed
    CONCEPT_DRIFT = "concept_drift"     # Feature-target relationship changed
    GRADUAL = "gradual"                 # Slow drift over time
    SUDDEN = "sudden"                   # Abrupt change
    RECURRING = "recurring"             # Seasonal/cyclical patterns


class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class DriftResult:
    """Result of drift detection analysis."""
    timestamp: datetime
    drift_detected: bool
    drift_type: DriftType
    severity: DriftSeverity
    overall_score: float  # 0-1, higher = more drift
    feature_scores: Dict[str, float]  # Per-feature drift scores
    drifted_features: List[str]  # Features that exceeded threshold
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'drift_detected': self.drift_detected,
            'drift_type': self.drift_type.value,
            'severity': self.severity.name,
            'overall_score': self.overall_score,
            'feature_scores': self.feature_scores,
            'drifted_features': self.drifted_features,
            'details': self.details,
            'recommendation': self.recommendation
        }


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    # Thresholds for statistical tests
    ks_threshold: float = 0.1          # KS test p-value threshold
    psi_threshold: float = 0.2         # PSI threshold (>0.2 = significant drift)
    js_threshold: float = 0.1          # Jensen-Shannon divergence threshold
    
    # Window sizes
    reference_window_size: int = 1000  # Baseline data window
    detection_window_size: int = 200   # Recent data window for comparison
    
    # Feature-level settings
    min_features_drifted: int = 3      # Min features to flag overall drift
    feature_drift_ratio: float = 0.2   # Ratio of features drifted to flag
    
    # Severity thresholds (based on overall score)
    low_threshold: float = 0.15
    medium_threshold: float = 0.30
    high_threshold: float = 0.50
    critical_threshold: float = 0.70
    
    # Monitoring settings
    check_interval_bars: int = 50      # Check every N new bars
    history_length: int = 100          # Keep last N drift results


class StatisticalTests:
    """Statistical tests for drift detection."""
    
    @staticmethod
    def ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distribution difference.
        Returns (statistic, p-value).
        """
        if len(reference) < 10 or len(current) < 10:
            return 0.0, 1.0
        
        try:
            stat, p_value = stats.ks_2samp(reference, current)
            return float(stat), float(p_value)
        except Exception as e:
            logger.warning(f"KS test failed: {e}")
            return 0.0, 1.0
    
    @staticmethod
    def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Population Stability Index.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change
        """
        if len(reference) < 10 or len(current) < 10:
            return 0.0
        
        try:
            # Create bins from reference distribution
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())
            
            if min_val == max_val:
                return 0.0
            
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            # Calculate proportions
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            cur_counts, _ = np.histogram(current, bins=bin_edges)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            ref_props = (ref_counts + epsilon) / (len(reference) + bins * epsilon)
            cur_props = (cur_counts + epsilon) / (len(current) + bins * epsilon)
            
            # Calculate PSI
            psi_value = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
            return float(psi_value)
            
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def js_divergence(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """
        Jensen-Shannon divergence (symmetric version of KL divergence).
        Returns value in [0, 1] where 0 = identical distributions.
        """
        if len(reference) < 10 or len(current) < 10:
            return 0.0
        
        try:
            min_val = min(reference.min(), current.min())
            max_val = max(reference.max(), current.max())
            
            if min_val == max_val:
                return 0.0
            
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
            cur_hist, _ = np.histogram(current, bins=bin_edges, density=True)
            
            # Normalize
            epsilon = 1e-10
            ref_hist = ref_hist + epsilon
            cur_hist = cur_hist + epsilon
            ref_hist = ref_hist / ref_hist.sum()
            cur_hist = cur_hist / cur_hist.sum()
            
            # JS divergence
            m = 0.5 * (ref_hist + cur_hist)
            js = 0.5 * (stats.entropy(ref_hist, m) + stats.entropy(cur_hist, m))
            
            return float(np.sqrt(js))  # Square root for [0,1] range
            
        except Exception as e:
            logger.warning(f"JS divergence calculation failed: {e}")
            return 0.0
    
    @staticmethod
    def mean_shift(reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate normalized mean shift."""
        if len(reference) < 5 or len(current) < 5:
            return 0.0
        
        ref_mean = np.mean(reference)
        cur_mean = np.mean(current)
        ref_std = np.std(reference)
        
        if ref_std < 1e-10:
            return 0.0
        
        return abs(cur_mean - ref_mean) / ref_std
    
    @staticmethod
    def variance_ratio(reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate variance ratio (current / reference)."""
        if len(reference) < 5 or len(current) < 5:
            return 1.0
        
        ref_var = np.var(reference)
        cur_var = np.var(current)
        
        if ref_var < 1e-10:
            return 1.0
        
        return cur_var / ref_var


class DriftDetector:
    """
    Main drift detector class.
    
    Monitors feature distributions and model performance to detect
    when retraining might be necessary.
    """
    
    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()
        
        # Reference data (baseline)
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_stats: Dict[str, Dict] = {}
        
        # Current window
        self.current_buffer: deque = deque(maxlen=self.config.detection_window_size)
        
        # History
        self.drift_history: deque = deque(maxlen=self.config.history_length)
        self.bars_since_check: int = 0
        
        # Feature importance (optional, for weighted drift)
        self.feature_importance: Dict[str, float] = {}
        
        logger.info(f"DriftDetector initialized with config: {self.config}")
    
    def set_reference(self, data: pd.DataFrame) -> None:
        """
        Set reference (baseline) data for drift comparison.
        Should be called with training data after model is trained.
        """
        self.reference_data = data.copy()
        self._compute_reference_stats()
        logger.info(f"Reference data set with {len(data)} samples, {len(data.columns)} features")
    
    def _compute_reference_stats(self) -> None:
        """Pre-compute statistics for reference data."""
        if self.reference_data is None:
            return
        
        self.reference_stats = {}
        for col in self.reference_data.columns:
            values = self.reference_data[col].dropna().values
            if len(values) > 0:
                self.reference_stats[col] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75),
                }
    
    def set_feature_importance(self, importance: Dict[str, float]) -> None:
        """Set feature importance for weighted drift calculation."""
        self.feature_importance = importance
        logger.info(f"Feature importance set for {len(importance)} features")
    
    def add_sample(self, sample: pd.Series) -> Optional[DriftResult]:
        """
        Add a new sample to the detection buffer.
        Returns drift result if check interval reached.
        """
        self.current_buffer.append(sample)
        self.bars_since_check += 1
        
        if self.bars_since_check >= self.config.check_interval_bars:
            self.bars_since_check = 0
            return self.check_drift()
        
        return None
    
    def add_batch(self, data: pd.DataFrame) -> Optional[DriftResult]:
        """Add batch of samples and check for drift."""
        for _, row in data.iterrows():
            self.current_buffer.append(row)
        
        return self.check_drift()
    
    def check_drift(self) -> DriftResult:
        """
        Check for drift between reference and current data.
        """
        timestamp = datetime.now()
        
        if self.reference_data is None:
            return DriftResult(
                timestamp=timestamp,
                drift_detected=False,
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                overall_score=0.0,
                feature_scores={},
                drifted_features=[],
                recommendation="No reference data set. Call set_reference() first."
            )
        
        if len(self.current_buffer) < self.config.detection_window_size * 0.5:
            return DriftResult(
                timestamp=timestamp,
                drift_detected=False,
                drift_type=DriftType.NONE,
                severity=DriftSeverity.NONE,
                overall_score=0.0,
                feature_scores={},
                drifted_features=[],
                recommendation=f"Insufficient data. Need at least {int(self.config.detection_window_size * 0.5)} samples."
            )
        
        # Convert buffer to DataFrame
        current_df = pd.DataFrame(list(self.current_buffer))
        
        # Analyze each feature
        feature_scores = {}
        drifted_features = []
        details = {}
        
        common_cols = set(self.reference_data.columns) & set(current_df.columns)
        
        for col in common_cols:
            ref_values = self.reference_data[col].dropna().values
            cur_values = current_df[col].dropna().values
            
            if len(ref_values) < 10 or len(cur_values) < 10:
                continue
            
            # Run statistical tests
            ks_stat, ks_pval = StatisticalTests.ks_test(ref_values, cur_values)
            psi = StatisticalTests.psi(ref_values, cur_values)
            js_div = StatisticalTests.js_divergence(ref_values, cur_values)
            mean_shift = StatisticalTests.mean_shift(ref_values, cur_values)
            var_ratio = StatisticalTests.variance_ratio(ref_values, cur_values)
            
            # Composite score for this feature
            feature_score = self._compute_feature_drift_score(
                ks_pval, psi, js_div, mean_shift, var_ratio
            )
            feature_scores[col] = feature_score
            
            # Check if feature has drifted
            if (ks_pval < self.config.ks_threshold or 
                psi > self.config.psi_threshold or 
                js_div > self.config.js_threshold):
                drifted_features.append(col)
                details[col] = {
                    'ks_pval': ks_pval,
                    'psi': psi,
                    'js_divergence': js_div,
                    'mean_shift': mean_shift,
                    'variance_ratio': var_ratio
                }
        
        # Calculate overall drift score
        overall_score = self._compute_overall_score(feature_scores, drifted_features)
        
        # Determine drift type and severity
        drift_detected, drift_type, severity = self._classify_drift(
            overall_score, drifted_features, len(common_cols)
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            drift_detected, drift_type, severity, drifted_features
        )
        
        result = DriftResult(
            timestamp=timestamp,
            drift_detected=drift_detected,
            drift_type=drift_type,
            severity=severity,
            overall_score=overall_score,
            feature_scores=feature_scores,
            drifted_features=drifted_features,
            details=details,
            recommendation=recommendation
        )
        
        self.drift_history.append(result)
        
        if drift_detected:
            logger.warning(
                f"Drift detected! Type: {drift_type.value}, Severity: {severity.name}, "
                f"Score: {overall_score:.3f}, Drifted features: {len(drifted_features)}"
            )
        
        return result
    
    def _compute_feature_drift_score(
        self,
        ks_pval: float,
        psi: float,
        js_div: float,
        mean_shift: float,
        var_ratio: float
    ) -> float:
        """Compute composite drift score for a single feature."""
        # Convert p-value to score (lower p-value = higher score)
        ks_score = 1 - ks_pval
        
        # Normalize PSI (cap at 1)
        psi_score = min(psi / 0.5, 1.0)
        
        # JS divergence already in [0,1]
        js_score = js_div
        
        # Mean shift score (normalize using threshold)
        mean_score = min(mean_shift / 3.0, 1.0)
        
        # Variance ratio score (deviation from 1)
        var_score = min(abs(var_ratio - 1) / 2.0, 1.0)
        
        # Weighted combination
        weights = [0.25, 0.25, 0.20, 0.15, 0.15]
        scores = [ks_score, psi_score, js_score, mean_score, var_score]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    def _compute_overall_score(
        self,
        feature_scores: Dict[str, float],
        drifted_features: List[str]
    ) -> float:
        """Compute overall drift score from feature scores."""
        if not feature_scores:
            return 0.0
        
        # Use feature importance if available
        if self.feature_importance:
            weighted_sum = 0.0
            weight_sum = 0.0
            for feat, score in feature_scores.items():
                weight = self.feature_importance.get(feat, 1.0)
                weighted_sum += score * weight
                weight_sum += weight
            return weighted_sum / weight_sum if weight_sum > 0 else 0.0
        else:
            return np.mean(list(feature_scores.values()))
    
    def _classify_drift(
        self,
        overall_score: float,
        drifted_features: List[str],
        total_features: int
    ) -> Tuple[bool, DriftType, DriftSeverity]:
        """Classify drift type and severity."""
        # Check if drift detected
        n_drifted = len(drifted_features)
        drift_ratio = n_drifted / total_features if total_features > 0 else 0
        
        drift_detected = (
            n_drifted >= self.config.min_features_drifted or
            drift_ratio >= self.config.feature_drift_ratio or
            overall_score >= self.config.medium_threshold
        )
        
        if not drift_detected:
            return False, DriftType.NONE, DriftSeverity.NONE
        
        # Determine severity
        if overall_score >= self.config.critical_threshold:
            severity = DriftSeverity.CRITICAL
        elif overall_score >= self.config.high_threshold:
            severity = DriftSeverity.HIGH
        elif overall_score >= self.config.medium_threshold:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW
        
        # Determine drift type (simplified - could be enhanced with trend analysis)
        drift_type = DriftType.DATA_DRIFT
        
        # Check history for patterns
        if len(self.drift_history) >= 5:
            recent_scores = [r.overall_score for r in list(self.drift_history)[-5:]]
            score_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            if score_trend > 0.05:
                drift_type = DriftType.GRADUAL
            elif overall_score - np.mean(recent_scores[:-1]) > 0.3:
                drift_type = DriftType.SUDDEN
        
        return True, drift_type, severity
    
    def _generate_recommendation(
        self,
        drift_detected: bool,
        drift_type: DriftType,
        severity: DriftSeverity,
        drifted_features: List[str]
    ) -> str:
        """Generate actionable recommendation based on drift analysis."""
        if not drift_detected:
            return "No significant drift detected. Continue monitoring."
        
        recommendations = []
        
        if severity == DriftSeverity.CRITICAL:
            recommendations.append("URGENT: Immediate model retraining recommended.")
            recommendations.append("Consider pausing live trading until model is updated.")
        elif severity == DriftSeverity.HIGH:
            recommendations.append("Schedule model retraining within 24 hours.")
            recommendations.append("Increase monitoring frequency.")
        elif severity == DriftSeverity.MEDIUM:
            recommendations.append("Plan model retraining within the week.")
            recommendations.append("Review feature distributions for anomalies.")
        else:
            recommendations.append("Monitor drift trends over next few sessions.")
        
        if drift_type == DriftType.SUDDEN:
            recommendations.append("Sudden drift detected - check for market regime change or data issues.")
        elif drift_type == DriftType.GRADUAL:
            recommendations.append("Gradual drift trend observed - consider adaptive learning strategy.")
        
        if len(drifted_features) > 0:
            top_drifted = drifted_features[:5]
            recommendations.append(f"Most affected features: {', '.join(top_drifted)}")
        
        return " ".join(recommendations)
    
    def get_drift_trend(self, lookback: int = 10) -> Dict[str, Any]:
        """Analyze drift trend over recent history."""
        if len(self.drift_history) < 2:
            return {'trend': 'insufficient_data', 'direction': 0}
        
        recent = list(self.drift_history)[-lookback:]
        scores = [r.overall_score for r in recent]
        
        # Linear regression for trend
        x = np.arange(len(scores))
        slope, intercept = np.polyfit(x, scores, 1)
        
        # Classify trend
        if slope > 0.02:
            trend = 'increasing'
            direction = 1
        elif slope < -0.02:
            trend = 'decreasing'
            direction = -1
        else:
            trend = 'stable'
            direction = 0
        
        return {
            'trend': trend,
            'direction': direction,
            'slope': slope,
            'current_score': scores[-1],
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'n_detections': sum(1 for r in recent if r.drift_detected)
        }
    
    def reset(self) -> None:
        """Reset detector state (keeps reference data)."""
        self.current_buffer.clear()
        self.bars_since_check = 0
        logger.info("Drift detector reset")
    
    def full_reset(self) -> None:
        """Full reset including reference data."""
        self.reset()
        self.reference_data = None
        self.reference_stats = {}
        self.drift_history.clear()
        logger.info("Drift detector fully reset")


class ConceptDriftDetector(DriftDetector):
    """
    Extended drift detector that also monitors concept drift
    (changes in the feature-target relationship).
    """
    
    def __init__(self, config: Optional[DriftConfig] = None):
        super().__init__(config)
        self.reference_predictions: Optional[np.ndarray] = None
        self.reference_targets: Optional[np.ndarray] = None
        self.prediction_buffer: deque = deque(maxlen=self.config.detection_window_size)
        self.target_buffer: deque = deque(maxlen=self.config.detection_window_size)
    
    def set_reference_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> None:
        """Set reference predictions and targets for concept drift detection."""
        self.reference_predictions = predictions
        self.reference_targets = targets
        logger.info(f"Reference predictions set with {len(predictions)} samples")
    
    def add_prediction(self, prediction: float, target: float) -> None:
        """Add prediction-target pair for concept drift monitoring."""
        self.prediction_buffer.append(prediction)
        self.target_buffer.append(target)
    
    def check_concept_drift(self) -> Optional[DriftResult]:
        """Check for concept drift using prediction error distribution."""
        if (self.reference_predictions is None or 
            len(self.prediction_buffer) < self.config.detection_window_size * 0.5):
            return None
        
        # Calculate reference errors
        ref_errors = self.reference_predictions - self.reference_targets
        
        # Calculate current errors
        cur_preds = np.array(list(self.prediction_buffer))
        cur_targets = np.array(list(self.target_buffer))
        cur_errors = cur_preds - cur_targets
        
        # Compare error distributions
        ks_stat, ks_pval = StatisticalTests.ks_test(ref_errors, cur_errors)
        psi = StatisticalTests.psi(ref_errors, cur_errors)
        
        # Check for concept drift
        concept_drift_detected = (
            ks_pval < self.config.ks_threshold or
            psi > self.config.psi_threshold * 1.5  # Stricter threshold for concept drift
        )
        
        if concept_drift_detected:
            return DriftResult(
                timestamp=datetime.now(),
                drift_detected=True,
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=DriftSeverity.HIGH if psi > 0.3 else DriftSeverity.MEDIUM,
                overall_score=psi,
                feature_scores={'prediction_error': psi},
                drifted_features=['prediction_error'],
                details={
                    'ks_pval': ks_pval,
                    'psi': psi,
                    'ref_error_mean': float(np.mean(ref_errors)),
                    'cur_error_mean': float(np.mean(cur_errors)),
                    'ref_error_std': float(np.std(ref_errors)),
                    'cur_error_std': float(np.std(cur_errors))
                },
                recommendation="Concept drift detected. Model-target relationship has changed. Retraining strongly recommended."
            )
        
        return None
