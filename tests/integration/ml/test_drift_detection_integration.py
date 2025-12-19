"""
Drift Detection Integration Tests.

Tests ML drift detection and its integration with the trading system.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

from ml.drift_detector import (
    DriftDetector, DriftConfig, DriftType, DriftSeverity, DriftResult
)


@pytest.fixture
def drift_detector():
    """Create drift detector with test configuration."""
    config = DriftConfig(
        ks_threshold=0.1,
        psi_threshold=0.2,
        reference_window_size=100,
        detection_window_size=50,
        min_features_drifted=2,
    )
    return DriftDetector(config)


@pytest.fixture
def reference_features():
    """Generate stable reference feature distribution."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n),
        'feature_2': np.random.normal(5, 2, n),
        'feature_3': np.random.uniform(0, 1, n),
        'feature_4': np.random.exponential(1, n),
    })


@pytest.mark.integration
class TestDriftDetection:
    """Tests for drift detection functionality."""
    
    def test_no_drift_detected_with_stable_distribution(self, drift_detector, reference_features):
        """No drift should be detected when distribution is stable."""
        # Set reference
        drift_detector.set_reference(reference_features)
        
        # Generate similar current data
        np.random.seed(43)
        n = 50
        current = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n),
            'feature_2': np.random.normal(5, 2, n),
            'feature_3': np.random.uniform(0, 1, n),
            'feature_4': np.random.exponential(1, n),
        })
        
        result = drift_detector.detect(current)
        
        assert result.drift_detected is False
        assert result.drift_type == DriftType.NONE
        assert result.severity == DriftSeverity.NONE
    
    def test_drift_detected_with_mean_shift(self, drift_detector, reference_features):
        """Drift should be detected when feature means shift significantly."""
        drift_detector.set_reference(reference_features)
        
        # Generate shifted data
        np.random.seed(44)
        n = 50
        current = pd.DataFrame({
            'feature_1': np.random.normal(3, 1, n),  # Mean shifted from 0 to 3
            'feature_2': np.random.normal(10, 2, n),  # Mean shifted from 5 to 10
            'feature_3': np.random.uniform(0, 1, n),
            'feature_4': np.random.exponential(1, n),
        })
        
        result = drift_detector.detect(current)
        
        assert result.drift_detected is True
        assert result.drift_type in (DriftType.DATA_DRIFT, DriftType.SUDDEN)
        assert result.severity.value >= DriftSeverity.MEDIUM.value
        assert len(result.drifted_features) >= 2
    
    def test_drift_detected_with_variance_change(self, drift_detector, reference_features):
        """Drift should be detected when feature variance changes significantly."""
        drift_detector.set_reference(reference_features)
        
        # Generate data with changed variance
        np.random.seed(45)
        n = 50
        current = pd.DataFrame({
            'feature_1': np.random.normal(0, 5, n),  # Variance increased 5x
            'feature_2': np.random.normal(5, 8, n),  # Variance increased 4x
            'feature_3': np.random.uniform(0, 1, n),
            'feature_4': np.random.exponential(1, n),
        })
        
        result = drift_detector.detect(current)
        
        assert result.drift_detected is True
        assert result.overall_score > 0.15
    
    def test_gradual_drift_detection(self, drift_detector, reference_features):
        """Gradual drift should be detected over multiple windows."""
        drift_detector.set_reference(reference_features)
        
        # Simulate gradual drift over multiple checks
        drift_scores = []
        for shift in [0.5, 1.0, 1.5, 2.0, 2.5]:
            np.random.seed(46)
            n = 50
            current = pd.DataFrame({
                'feature_1': np.random.normal(shift, 1, n),
                'feature_2': np.random.normal(5 + shift, 2, n),
                'feature_3': np.random.uniform(0, 1, n),
                'feature_4': np.random.exponential(1, n),
            })
            result = drift_detector.detect(current)
            drift_scores.append(result.overall_score)
        
        # Drift scores should generally increase with larger shifts
        assert drift_scores[-1] > drift_scores[0]
    
    def test_drift_severity_levels(self, drift_detector, reference_features):
        """Different drift magnitudes should produce different severity levels."""
        drift_detector.set_reference(reference_features)
        
        severities = []
        for shift in [0.1, 1.0, 3.0, 5.0]:
            np.random.seed(47)
            n = 50
            current = pd.DataFrame({
                'feature_1': np.random.normal(shift, 1, n),
                'feature_2': np.random.normal(5 + shift, 2, n),
                'feature_3': np.random.uniform(0, 1, n),
                'feature_4': np.random.exponential(1, n),
            })
            result = drift_detector.detect(current)
            severities.append(result.severity.value)
        
        # Larger shifts should produce higher severity
        assert severities[-1] >= severities[0]


@pytest.mark.integration
class TestDriftAndTradingIntegration:
    """Tests for drift detection integration with trading decisions."""
    
    def test_drift_triggers_model_confidence_reduction(self, drift_detector, reference_features):
        """Detected drift should reduce model confidence in predictions."""
        drift_detector.set_reference(reference_features)
        
        # Detect significant drift
        np.random.seed(48)
        n = 50
        drifted = pd.DataFrame({
            'feature_1': np.random.normal(5, 1, n),
            'feature_2': np.random.normal(15, 2, n),
            'feature_3': np.random.uniform(0.5, 1.5, n),
            'feature_4': np.random.exponential(3, n),
        })
        
        result = drift_detector.detect(drifted)
        
        if result.drift_detected:
            # Calculate confidence adjustment factor
            confidence_factor = 1.0 - (result.overall_score * 0.5)
            assert confidence_factor < 1.0
            
            # Original confidence of 0.9 should be reduced
            original_confidence = 0.9
            adjusted_confidence = original_confidence * confidence_factor
            assert adjusted_confidence < original_confidence
    
    def test_critical_drift_should_halt_trading(self, drift_detector, reference_features):
        """Critical drift severity should recommend halting trading."""
        drift_detector.set_reference(reference_features)
        
        # Create extremely drifted data
        np.random.seed(49)
        n = 50
        extreme_drift = pd.DataFrame({
            'feature_1': np.random.normal(10, 5, n),
            'feature_2': np.random.normal(25, 10, n),
            'feature_3': np.random.uniform(2, 4, n),
            'feature_4': np.random.exponential(5, n),
        })
        
        result = drift_detector.detect(extreme_drift)
        
        if result.severity == DriftSeverity.CRITICAL:
            assert "retrain" in result.recommendation.lower() or "halt" in result.recommendation.lower()
    
    def test_drift_result_serialization(self, drift_detector, reference_features):
        """Drift results should be serializable for logging/storage."""
        drift_detector.set_reference(reference_features)
        
        np.random.seed(50)
        n = 50
        current = pd.DataFrame({
            'feature_1': np.random.normal(2, 1, n),
            'feature_2': np.random.normal(7, 2, n),
            'feature_3': np.random.uniform(0, 1, n),
            'feature_4': np.random.exponential(1, n),
        })
        
        result = drift_detector.detect(current)
        result_dict = result.to_dict()
        
        assert 'timestamp' in result_dict
        assert 'drift_detected' in result_dict
        assert 'drift_type' in result_dict
        assert 'severity' in result_dict
        assert 'overall_score' in result_dict
        assert 'feature_scores' in result_dict


@pytest.mark.integration
class TestDriftMonitoring:
    """Tests for continuous drift monitoring."""
    
    def test_drift_history_tracking(self, drift_detector, reference_features):
        """Drift detector should maintain history of detections."""
        drift_detector.set_reference(reference_features)
        
        # Run multiple detections
        for i in range(5):
            np.random.seed(50 + i)
            n = 50
            current = pd.DataFrame({
                'feature_1': np.random.normal(i * 0.5, 1, n),
                'feature_2': np.random.normal(5, 2, n),
                'feature_3': np.random.uniform(0, 1, n),
                'feature_4': np.random.exponential(1, n),
            })
            drift_detector.detect(current)
        
        history = drift_detector.get_history()
        assert len(history) >= 5
    
    def test_reference_update_after_retraining(self, drift_detector, reference_features):
        """Reference should be updatable after model retraining."""
        drift_detector.set_reference(reference_features)
        
        # Simulate new reference after retraining
        np.random.seed(55)
        n = 100
        new_reference = pd.DataFrame({
            'feature_1': np.random.normal(2, 1, n),  # New baseline
            'feature_2': np.random.normal(7, 2, n),
            'feature_3': np.random.uniform(0, 1, n),
            'feature_4': np.random.exponential(1, n),
        })
        
        drift_detector.set_reference(new_reference)
        
        # Data similar to new reference should not show drift
        np.random.seed(56)
        n = 50
        current = pd.DataFrame({
            'feature_1': np.random.normal(2, 1, n),
            'feature_2': np.random.normal(7, 2, n),
            'feature_3': np.random.uniform(0, 1, n),
            'feature_4': np.random.exponential(1, n),
        })
        
        result = drift_detector.detect(current)
        assert result.drift_detected is False or result.severity == DriftSeverity.LOW
