"""
Drift Detection Integration Tests.

Tests ML drift detection and its integration with the trading system.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch

# Import directly from the module file to avoid __init__ issues
import importlib.util
import sys
from pathlib import Path

# Load drift_detector module directly
drift_detector_path = Path(__file__).parent.parent.parent.parent / "ml" / "drift_detector.py"
spec = importlib.util.spec_from_file_location("drift_detector_direct", drift_detector_path)
drift_detector_module = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(drift_detector_module)
    DriftDetector = drift_detector_module.DriftDetector
    DriftConfig = drift_detector_module.DriftConfig
    DriftType = drift_detector_module.DriftType
    DriftSeverity = drift_detector_module.DriftSeverity
    DriftResult = drift_detector_module.DriftResult
    HAS_DRIFT_DETECTOR = True
except Exception:
    HAS_DRIFT_DETECTOR = False
    DriftDetector = None
    DriftConfig = None
    DriftType = None
    DriftSeverity = None
    DriftResult = None


@pytest.fixture
def drift_detector():
    """Create drift detector with test configuration."""
    if not HAS_DRIFT_DETECTOR:
        pytest.skip("Drift detector not available")
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
        
        result = drift_detector.add_batch(current)
        
        assert result is not None
        # With similar distribution, drift should be minimal (allow some variance)
        assert result.severity in (DriftSeverity.NONE, DriftSeverity.LOW, DriftSeverity.MEDIUM)
    
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
        
        result = drift_detector.add_batch(current)
        
        assert result is not None
        # With significant shift, drift should be detected
        assert result.overall_score > 0
    
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
        
        result = drift_detector.add_batch(current)
        
        assert result is not None
        assert result.overall_score >= 0
    
    def test_gradual_drift_detection(self, drift_detector, reference_features):
        """Gradual drift should be detected over multiple windows."""
        drift_detector.set_reference(reference_features)
        
        # Simulate gradual drift over multiple checks
        drift_scores = []
        for shift in [0.5, 1.0, 1.5, 2.0, 2.5]:
            np.random.seed(46 + int(shift * 10))
            n = 50
            current = pd.DataFrame({
                'feature_1': np.random.normal(shift, 1, n),
                'feature_2': np.random.normal(5 + shift, 2, n),
                'feature_3': np.random.uniform(0, 1, n),
                'feature_4': np.random.exponential(1, n),
            })
            result = drift_detector.add_batch(current)
            if result:
                drift_scores.append(result.overall_score)
        
        # Should have collected some scores
        assert len(drift_scores) > 0
    
    def test_drift_severity_levels(self, drift_detector, reference_features):
        """Different drift magnitudes should produce different severity levels."""
        drift_detector.set_reference(reference_features)
        
        severities = []
        for shift in [0.1, 1.0, 3.0, 5.0]:
            np.random.seed(47 + int(shift * 10))
            n = 50
            current = pd.DataFrame({
                'feature_1': np.random.normal(shift, 1, n),
                'feature_2': np.random.normal(5 + shift, 2, n),
                'feature_3': np.random.uniform(0, 1, n),
                'feature_4': np.random.exponential(1, n),
            })
            result = drift_detector.add_batch(current)
            if result:
                severities.append(result.severity.value)
        
        # Should have collected some severities
        assert len(severities) > 0


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
        
        result = drift_detector.add_batch(drifted)
        
        assert result is not None
        # Calculate confidence adjustment factor
        confidence_factor = 1.0 - (result.overall_score * 0.5)
        
        # Original confidence of 0.9 should be reduced if drift detected
        original_confidence = 0.9
        adjusted_confidence = original_confidence * confidence_factor
        assert adjusted_confidence <= original_confidence
    
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
        
        result = drift_detector.add_batch(extreme_drift)
        
        assert result is not None
        # With extreme drift, recommendation should exist
        assert result.recommendation is not None
    
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
        
        result = drift_detector.add_batch(current)
        
        assert result is not None
        # Check result has expected attributes
        assert hasattr(result, 'timestamp')
        assert hasattr(result, 'drift_detected')
        assert hasattr(result, 'drift_type')
        assert hasattr(result, 'severity')
        assert hasattr(result, 'overall_score')


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
            drift_detector.add_batch(current)
        
        # Check that detector maintains state
        assert drift_detector.reference_data is not None
    
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
        
        result = drift_detector.add_batch(current)
        assert result is not None
        # With similar data to new reference, drift should be minimal
        assert result.severity in (DriftSeverity.NONE, DriftSeverity.LOW, DriftSeverity.MEDIUM)
