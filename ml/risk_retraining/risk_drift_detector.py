# ml/risk_retraining/risk_drift_detector.py
"""
Drift Detector for Risk Management Models.

Extends standard drift detection with specialized checks for:
- Volatility regime changes
- Quantile distribution shifts
- Meta-feature distribution changes
- RL state/action distribution changes

Uses multiple detection methods:
- Population Stability Index (PSI)
- Kolmogorov-Smirnov test
- Jensen-Shannon divergence
- Mean/variance shift detection
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
from scipy import stats
import warnings

from .risk_retraining_config import (
    RiskModelType, RiskRetrainingConfig,
    TCNRiskDriftConfig, GBMMetaDriftConfig, RLExitDriftConfig
)

logger = logging.getLogger(__name__)


# =============================================================================
# Drift Types
# =============================================================================

class DriftType(Enum):
    """Types of drift detected."""
    NO_DRIFT = "no_drift"
    DATA_DRIFT = "data_drift"              # Feature distributions changed
    CONCEPT_DRIFT = "concept_drift"        # Feature-target relationship changed
    VOLATILITY_REGIME = "volatility_regime"  # Market volatility regime changed
    QUANTILE_CALIBRATION = "quantile_calibration"  # Quantile coverage drifted
    PREDICTION_DRIFT = "prediction_drift"  # Model output distribution changed
    GRADUAL_DRIFT = "gradual_drift"        # Slow drift over time
    SUDDEN_DRIFT = "sudden_drift"          # Abrupt change


class DriftSeverity(Enum):
    """Severity of detected drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    """Result of drift detection."""
    drift_detected: bool
    drift_type: DriftType
    severity: DriftSeverity
    score: float
    threshold: float
    details: Dict[str, Any]
    timestamp: datetime
    features_affected: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'drift_detected': self.drift_detected,
            'drift_type': self.drift_type.value,
            'severity': self.severity.value,
            'score': self.score,
            'threshold': self.threshold,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'features_affected': self.features_affected,
        }


@dataclass
class VolatilityRegimeResult:
    """Result of volatility regime detection."""
    regime_changed: bool
    current_regime: str  # 'low', 'normal', 'high', 'extreme'
    previous_regime: str
    volatility_ratio: float  # Current / historical
    z_score: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'regime_changed': self.regime_changed,
            'current_regime': self.current_regime,
            'previous_regime': self.previous_regime,
            'volatility_ratio': self.volatility_ratio,
            'z_score': self.z_score,
            'timestamp': self.timestamp.isoformat(),
        }


# =============================================================================
# Drift Detection Methods
# =============================================================================

class DriftDetectionMethods:
    """Collection of drift detection algorithms."""
    
    @staticmethod
    def population_stability_index(
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures the shift in distribution between two samples.
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Slight change
        - PSI >= 0.2: Significant change
        """
        # Create bins from reference distribution
        _, bin_edges = np.histogram(reference, bins=n_bins)
        
        # Calculate proportions
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Avoid division by zero
        ref_pct = (ref_counts + 1) / (len(reference) + n_bins)
        curr_pct = (curr_counts + 1) / (len(current) + n_bins)
        
        # PSI formula
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        
        return psi
    
    @staticmethod
    def ks_test(
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test.
        
        Returns (statistic, p_value).
        Lower p-value indicates significant difference.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, p_value = stats.ks_2samp(reference, current)
        return statistic, p_value
    
    @staticmethod
    def jensen_shannon_divergence(
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 50
    ) -> float:
        """
        Calculate Jensen-Shannon divergence.
        
        JS divergence is symmetric and bounded [0, 1].
        """
        # Create common bins
        all_data = np.concatenate([reference, current])
        _, bin_edges = np.histogram(all_data, bins=n_bins)
        
        # Calculate histograms
        ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
        curr_hist, _ = np.histogram(current, bins=bin_edges, density=True)
        
        # Normalize
        ref_hist = ref_hist / (ref_hist.sum() + 1e-10)
        curr_hist = curr_hist / (curr_hist.sum() + 1e-10)
        
        # JS divergence
        m = 0.5 * (ref_hist + curr_hist)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            js = 0.5 * (stats.entropy(ref_hist, m) + stats.entropy(curr_hist, m))
        
        return js if not np.isnan(js) else 0.0
    
    @staticmethod
    def mean_shift_test(
        reference: np.ndarray,
        current: np.ndarray,
        threshold_std: float = 2.0
    ) -> Tuple[bool, float]:
        """
        Test for significant mean shift.
        
        Returns (shifted, z_score).
        """
        ref_mean = np.mean(reference)
        ref_std = np.std(reference)
        curr_mean = np.mean(current)
        
        if ref_std == 0:
            return False, 0.0
        
        z_score = abs(curr_mean - ref_mean) / ref_std
        shifted = z_score > threshold_std
        
        return shifted, z_score
    
    @staticmethod
    def variance_ratio_test(
        reference: np.ndarray,
        current: np.ndarray,
        threshold: float = 2.0
    ) -> Tuple[bool, float]:
        """
        Test for significant variance change.
        
        Returns (changed, ratio).
        """
        ref_var = np.var(reference)
        curr_var = np.var(current)
        
        if ref_var == 0:
            return False, 1.0
        
        ratio = curr_var / ref_var
        changed = ratio > threshold or ratio < (1 / threshold)
        
        return changed, ratio


# =============================================================================
# Feature Drift Detector
# =============================================================================

class FeatureDriftDetector:
    """Detects drift in feature distributions."""
    
    def __init__(
        self,
        reference_window_size: int = 5000,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.1,
        js_threshold: float = 0.15,
    ):
        self.reference_window_size = reference_window_size
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.js_threshold = js_threshold
        
        self.reference_data: Dict[str, np.ndarray] = {}
        self.methods = DriftDetectionMethods()
    
    def set_reference(self, feature_name: str, data: np.ndarray):
        """Set reference distribution for a feature."""
        self.reference_data[feature_name] = data[-self.reference_window_size:]
    
    def detect_drift(
        self,
        feature_name: str,
        current_data: np.ndarray
    ) -> DriftResult:
        """Detect drift for a single feature."""
        now = datetime.now()
        
        if feature_name not in self.reference_data:
            return DriftResult(
                drift_detected=False,
                drift_type=DriftType.NO_DRIFT,
                severity=DriftSeverity.NONE,
                score=0.0,
                threshold=0.0,
                details={'error': 'No reference data'},
                timestamp=now,
            )
        
        reference = self.reference_data[feature_name]
        
        # Calculate metrics
        psi = self.methods.population_stability_index(reference, current_data)
        ks_stat, ks_pval = self.methods.ks_test(reference, current_data)
        js_div = self.methods.jensen_shannon_divergence(reference, current_data)
        mean_shifted, mean_z = self.methods.mean_shift_test(reference, current_data)
        var_changed, var_ratio = self.methods.variance_ratio_test(reference, current_data)
        
        # Determine drift
        drift_detected = (
            psi >= self.psi_threshold or
            ks_pval < self.ks_threshold or
            js_div >= self.js_threshold
        )
        
        # Determine severity
        if not drift_detected:
            severity = DriftSeverity.NONE
        elif psi >= self.psi_threshold * 2 or js_div >= self.js_threshold * 2:
            severity = DriftSeverity.CRITICAL
        elif psi >= self.psi_threshold * 1.5 or js_div >= self.js_threshold * 1.5:
            severity = DriftSeverity.HIGH
        elif psi >= self.psi_threshold or js_div >= self.js_threshold:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW
        
        # Determine drift type
        if not drift_detected:
            drift_type = DriftType.NO_DRIFT
        elif mean_shifted and var_changed:
            drift_type = DriftType.SUDDEN_DRIFT
        elif var_changed:
            drift_type = DriftType.DATA_DRIFT
        else:
            drift_type = DriftType.GRADUAL_DRIFT
        
        return DriftResult(
            drift_detected=drift_detected,
            drift_type=drift_type,
            severity=severity,
            score=psi,
            threshold=self.psi_threshold,
            details={
                'psi': psi,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'js_divergence': js_div,
                'mean_z_score': mean_z,
                'variance_ratio': var_ratio,
            },
            timestamp=now,
            features_affected=[feature_name],
        )
    
    def detect_multi_feature_drift(
        self,
        current_data: Dict[str, np.ndarray]
    ) -> DriftResult:
        """Detect drift across multiple features."""
        results = []
        drifted_features = []
        
        for feature_name, data in current_data.items():
            if feature_name in self.reference_data:
                result = self.detect_drift(feature_name, data)
                results.append(result)
                if result.drift_detected:
                    drifted_features.append(feature_name)
        
        if not results:
            return DriftResult(
                drift_detected=False,
                drift_type=DriftType.NO_DRIFT,
                severity=DriftSeverity.NONE,
                score=0.0,
                threshold=0.0,
                details={'error': 'No features to check'},
                timestamp=datetime.now(),
            )
        
        # Aggregate results
        drift_detected = len(drifted_features) > 0
        avg_score = np.mean([r.score for r in results])
        max_severity = max(r.severity.value for r in results)
        
        severity_map = {s.value: s for s in DriftSeverity}
        
        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.DATA_DRIFT if drift_detected else DriftType.NO_DRIFT,
            severity=severity_map.get(max_severity, DriftSeverity.NONE),
            score=avg_score,
            threshold=self.psi_threshold,
            details={
                'features_checked': len(results),
                'features_drifted': len(drifted_features),
                'drift_ratio': len(drifted_features) / len(results),
                'per_feature': {r.features_affected[0]: r.to_dict() for r in results},
            },
            timestamp=datetime.now(),
            features_affected=drifted_features,
        )


# =============================================================================
# Volatility Regime Detector
# =============================================================================

class VolatilityRegimeDetector:
    """Detects changes in market volatility regime."""
    
    REGIMES = {
        'low': (0.0, 0.7),
        'normal': (0.7, 1.3),
        'high': (1.3, 2.0),
        'extreme': (2.0, float('inf')),
    }
    
    def __init__(
        self,
        lookback_periods: int = 100,
        regime_change_threshold: float = 0.5,
        z_score_threshold: float = 2.0,
    ):
        self.lookback_periods = lookback_periods
        self.regime_change_threshold = regime_change_threshold
        self.z_score_threshold = z_score_threshold
        
        self.historical_volatility: deque = deque(maxlen=5000)
        self.current_regime: str = 'normal'
    
    def update(self, volatility: float):
        """Update with new volatility observation."""
        self.historical_volatility.append(volatility)
    
    def _classify_regime(self, ratio: float) -> str:
        """Classify volatility ratio to regime."""
        for regime, (low, high) in self.REGIMES.items():
            if low <= ratio < high:
                return regime
        return 'extreme'
    
    def detect_regime_change(
        self,
        current_volatility: Optional[float] = None
    ) -> VolatilityRegimeResult:
        """Detect if volatility regime has changed."""
        now = datetime.now()
        
        if len(self.historical_volatility) < self.lookback_periods:
            return VolatilityRegimeResult(
                regime_changed=False,
                current_regime=self.current_regime,
                previous_regime=self.current_regime,
                volatility_ratio=1.0,
                z_score=0.0,
                timestamp=now,
            )
        
        hist_vol = np.array(list(self.historical_volatility))
        historical_mean = np.mean(hist_vol[:-self.lookback_periods]) if len(hist_vol) > self.lookback_periods else np.mean(hist_vol)
        historical_std = np.std(hist_vol[:-self.lookback_periods]) if len(hist_vol) > self.lookback_periods else np.std(hist_vol)
        
        # Use provided volatility or recent average
        if current_volatility is not None:
            current_vol = current_volatility
        else:
            current_vol = np.mean(hist_vol[-self.lookback_periods:])
        
        # Calculate ratio and z-score
        if historical_mean == 0:
            ratio = 1.0
            z_score = 0.0
        else:
            ratio = current_vol / historical_mean
            z_score = (current_vol - historical_mean) / (historical_std + 1e-10)
        
        # Classify regime
        new_regime = self._classify_regime(ratio)
        previous_regime = self.current_regime
        
        # Detect change
        regime_changed = (
            new_regime != previous_regime and
            (abs(ratio - 1.0) > self.regime_change_threshold or
             abs(z_score) > self.z_score_threshold)
        )
        
        if regime_changed:
            self.current_regime = new_regime
            logger.info(f"Volatility regime changed: {previous_regime} -> {new_regime}")
        
        return VolatilityRegimeResult(
            regime_changed=regime_changed,
            current_regime=new_regime,
            previous_regime=previous_regime,
            volatility_ratio=ratio,
            z_score=z_score,
            timestamp=now,
        )


# =============================================================================
# Quantile Calibration Drift Detector
# =============================================================================

class QuantileCalibrationDetector:
    """Detects drift in quantile prediction calibration."""
    
    def __init__(
        self,
        quantile_levels: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95],
        drift_threshold: float = 0.1,
        window_size: int = 500,
    ):
        self.quantile_levels = quantile_levels
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        
        self.predictions: deque = deque(maxlen=window_size)
        self.actuals: deque = deque(maxlen=window_size)
    
    def update(self, predicted_quantiles: np.ndarray, actual_value: float):
        """Update with new prediction."""
        self.predictions.append(predicted_quantiles)
        self.actuals.append(actual_value)
    
    def calculate_coverage(self, quantile_idx: int) -> float:
        """Calculate empirical coverage for a quantile."""
        if len(self.predictions) < 50:
            return self.quantile_levels[quantile_idx]
        
        preds = np.array(list(self.predictions))
        actuals = np.array(list(self.actuals))
        
        return np.mean(actuals <= preds[:, quantile_idx])
    
    def detect_calibration_drift(self) -> DriftResult:
        """Detect if quantile calibration has drifted."""
        now = datetime.now()
        
        if len(self.predictions) < 100:
            return DriftResult(
                drift_detected=False,
                drift_type=DriftType.NO_DRIFT,
                severity=DriftSeverity.NONE,
                score=0.0,
                threshold=self.drift_threshold,
                details={'error': 'Insufficient data'},
                timestamp=now,
            )
        
        # Calculate coverage drift for each quantile
        coverage_drifts = {}
        max_drift = 0.0
        
        for i, level in enumerate(self.quantile_levels):
            empirical = self.calculate_coverage(i)
            drift = abs(empirical - level)
            coverage_drifts[f'q{int(level*100)}'] = {
                'target': level,
                'empirical': empirical,
                'drift': drift,
            }
            max_drift = max(max_drift, drift)
        
        # Determine drift
        drift_detected = max_drift > self.drift_threshold
        
        if not drift_detected:
            severity = DriftSeverity.NONE
        elif max_drift > self.drift_threshold * 2:
            severity = DriftSeverity.CRITICAL
        elif max_drift > self.drift_threshold * 1.5:
            severity = DriftSeverity.HIGH
        else:
            severity = DriftSeverity.MEDIUM
        
        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.QUANTILE_CALIBRATION if drift_detected else DriftType.NO_DRIFT,
            severity=severity,
            score=max_drift,
            threshold=self.drift_threshold,
            details={
                'coverage_analysis': coverage_drifts,
                'max_drift': max_drift,
                'samples': len(self.predictions),
            },
            timestamp=now,
        )


# =============================================================================
# Main Risk Drift Detector
# =============================================================================

class RiskDriftDetector:
    """
    Main drift detector for Risk Management models.
    
    Combines:
    - Feature drift detection (for TCN inputs)
    - Volatility regime detection
    - Quantile calibration drift
    - Prediction distribution drift (for GBM/RL)
    """
    
    def __init__(self, config: RiskRetrainingConfig):
        self.config = config
        
        # Initialize detectors for each model type
        self._init_tcn_detectors()
        self._init_gbm_detectors()
        self._init_rl_detectors()
        
        # Last check times
        self.last_checked: Dict[RiskModelType, datetime] = {}
        
        # Drift history
        self.drift_history: Dict[RiskModelType, List[DriftResult]] = {
            model_type: [] for model_type in RiskModelType
        }
        
        logger.info("RiskDriftDetector initialized")
    
    def _init_tcn_detectors(self):
        """Initialize TCN-specific drift detectors."""
        tcn_config = self.config.tcn_drift
        
        self.tcn_feature_detector = FeatureDriftDetector(
            reference_window_size=tcn_config.reference_window_size,
            psi_threshold=tcn_config.psi_threshold,
            ks_threshold=tcn_config.ks_threshold,
            js_threshold=tcn_config.js_threshold,
        )
        
        self.volatility_regime_detector = VolatilityRegimeDetector(
            lookback_periods=tcn_config.volatility_mean_shift_periods,
            regime_change_threshold=tcn_config.volatility_regime_change_threshold,
        )
        
        self.quantile_calibration_detector = QuantileCalibrationDetector(
            drift_threshold=tcn_config.quantile_calibration_drift,
        )
        
        self.tcn_monitor_features = tcn_config.monitor_features
    
    def _init_gbm_detectors(self):
        """Initialize GBM-specific drift detectors."""
        gbm_config = self.config.gbm_drift
        
        self.gbm_feature_detector = FeatureDriftDetector(
            reference_window_size=gbm_config.reference_window_size,
            psi_threshold=gbm_config.psi_threshold,
            ks_threshold=gbm_config.ks_threshold,
        )
        
        # Prediction distribution detector
        self.gbm_pred_detector = FeatureDriftDetector(
            reference_window_size=gbm_config.reference_window_size,
            psi_threshold=gbm_config.prediction_distribution_shift,
        )
    
    def _init_rl_detectors(self):
        """Initialize RL-specific drift detectors."""
        rl_config = self.config.rl_drift
        
        self.rl_state_detector = FeatureDriftDetector(
            psi_threshold=rl_config.state_psi_threshold,
        )
        
        self.rl_action_detector = FeatureDriftDetector(
            psi_threshold=rl_config.action_distribution_shift,
        )
        
        self.rl_reward_detector = FeatureDriftDetector(
            psi_threshold=rl_config.reward_distribution_shift,
        )
    
    # =========================================================================
    # Reference Data Setting
    # =========================================================================
    
    def set_tcn_reference(
        self,
        feature_data: Dict[str, np.ndarray],
        volatility_data: np.ndarray,
    ):
        """Set reference data for TCN drift detection."""
        for feature_name, data in feature_data.items():
            if feature_name in self.tcn_monitor_features:
                self.tcn_feature_detector.set_reference(feature_name, data)
        
        # Initialize volatility regime
        for v in volatility_data:
            self.volatility_regime_detector.update(v)
        
        logger.info(f"TCN reference set with {len(feature_data)} features")
    
    def set_gbm_reference(
        self,
        meta_features: Dict[str, np.ndarray],
        predictions: np.ndarray,
    ):
        """Set reference data for GBM drift detection."""
        for feature_name, data in meta_features.items():
            self.gbm_feature_detector.set_reference(feature_name, data)
        
        self.gbm_pred_detector.set_reference('predictions', predictions)
        
        logger.info("GBM reference set")
    
    def set_rl_reference(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ):
        """Set reference data for RL drift detection."""
        # States - assume flattened
        self.rl_state_detector.set_reference('state', states.flatten())
        self.rl_action_detector.set_reference('action', actions.flatten())
        self.rl_reward_detector.set_reference('reward', rewards)
        
        logger.info("RL reference set")
    
    # =========================================================================
    # Drift Detection
    # =========================================================================
    
    def check_tcn_drift(
        self,
        feature_data: Dict[str, np.ndarray],
        volatility: Optional[float] = None,
        quantile_preds: Optional[np.ndarray] = None,
        quantile_actual: Optional[float] = None,
    ) -> Dict[str, DriftResult]:
        """Check for drift in TCN Risk Model."""
        results = {}
        
        # Feature drift
        feature_results = self.tcn_feature_detector.detect_multi_feature_drift(feature_data)
        results['feature_drift'] = feature_results
        
        # Volatility regime
        if volatility is not None:
            self.volatility_regime_detector.update(volatility)
        regime_result = self.volatility_regime_detector.detect_regime_change(volatility)
        
        if regime_result.regime_changed:
            results['volatility_regime'] = DriftResult(
                drift_detected=True,
                drift_type=DriftType.VOLATILITY_REGIME,
                severity=DriftSeverity.HIGH,
                score=abs(regime_result.volatility_ratio - 1.0),
                threshold=self.config.tcn_drift.volatility_regime_change_threshold,
                details=regime_result.to_dict(),
                timestamp=datetime.now(),
            )
        
        # Quantile calibration
        if quantile_preds is not None and quantile_actual is not None:
            self.quantile_calibration_detector.update(quantile_preds, quantile_actual)
            calibration_result = self.quantile_calibration_detector.detect_calibration_drift()
            if calibration_result.drift_detected:
                results['quantile_calibration'] = calibration_result
        
        self.last_checked[RiskModelType.TCN_RISK] = datetime.now()
        
        # Store in history
        for result in results.values():
            if result.drift_detected:
                self.drift_history[RiskModelType.TCN_RISK].append(result)
        
        return results
    
    def check_gbm_drift(
        self,
        meta_features: Dict[str, np.ndarray],
        predictions: np.ndarray,
    ) -> Dict[str, DriftResult]:
        """Check for drift in GBM Meta-Labeling Model."""
        results = {}
        
        # Meta-feature drift
        feature_results = self.gbm_feature_detector.detect_multi_feature_drift(meta_features)
        results['feature_drift'] = feature_results
        
        # Prediction distribution drift
        pred_result = self.gbm_pred_detector.detect_drift('predictions', predictions)
        if pred_result.drift_detected:
            pred_result.drift_type = DriftType.PREDICTION_DRIFT
            results['prediction_drift'] = pred_result
        
        self.last_checked[RiskModelType.GBM_META] = datetime.now()
        
        # Store in history
        for result in results.values():
            if result.drift_detected:
                self.drift_history[RiskModelType.GBM_META].append(result)
        
        return results
    
    def check_rl_drift(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
    ) -> Dict[str, DriftResult]:
        """Check for drift in RL Exit Optimizer."""
        results = {}
        
        # State distribution drift
        state_result = self.rl_state_detector.detect_drift('state', states.flatten())
        if state_result.drift_detected:
            results['state_drift'] = state_result
        
        # Action distribution drift
        action_result = self.rl_action_detector.detect_drift('action', actions.flatten())
        if action_result.drift_detected:
            results['action_drift'] = action_result
        
        # Reward distribution drift
        reward_result = self.rl_reward_detector.detect_drift('reward', rewards)
        if reward_result.drift_detected:
            results['reward_drift'] = reward_result
        
        self.last_checked[RiskModelType.RL_EXIT] = datetime.now()
        
        # Store in history
        for result in results.values():
            if result.drift_detected:
                self.drift_history[RiskModelType.RL_EXIT].append(result)
        
        return results
    
    # =========================================================================
    # Aggregated Checks
    # =========================================================================
    
    def should_check(self, model_type: RiskModelType) -> bool:
        """Check if enough time has passed since last drift check."""
        last = self.last_checked.get(model_type)
        if last is None:
            return True
        
        config = self.config.get_drift_config_for_model(model_type)
        interval = timedelta(minutes=config.check_interval_minutes)
        
        return datetime.now() - last >= interval
    
    def check_retraining_needed(
        self,
        model_type: RiskModelType,
        drift_results: Dict[str, DriftResult]
    ) -> Tuple[bool, Optional[str]]:
        """Determine if retraining is needed based on drift results."""
        if not drift_results:
            return False, None
        
        # Check for any critical drift
        critical_drifts = [
            name for name, result in drift_results.items()
            if result.severity == DriftSeverity.CRITICAL
        ]
        
        if critical_drifts:
            return True, f"Critical drift in: {', '.join(critical_drifts)}"
        
        # Check for multiple high-severity drifts
        high_drifts = [
            name for name, result in drift_results.items()
            if result.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
        ]
        
        if len(high_drifts) >= 2:
            return True, f"Multiple high-severity drifts: {', '.join(high_drifts)}"
        
        # Check for volatility regime change (always triggers)
        if 'volatility_regime' in drift_results:
            return True, "Volatility regime changed"
        
        return False, None
    
    def get_drift_summary(self, model_type: RiskModelType) -> Dict:
        """Get summary of recent drift for a model."""
        history = self.drift_history.get(model_type, [])
        recent = [d for d in history if (datetime.now() - d.timestamp).total_seconds() < 86400]  # Last 24h
        
        return {
            'model_type': model_type.name,
            'total_drifts_24h': len(recent),
            'drift_types': list(set(d.drift_type.value for d in recent)),
            'max_severity': max((d.severity.value for d in recent), default='none'),
            'last_checked': self.last_checked.get(model_type, 'never'),
            'features_affected': list(set(f for d in recent for f in d.features_affected)),
        }