"""
Risk Management Model Retraining System
========================================

Automated retraining system for Risk Management ML models including:
- TCN Multi-Head (direction + volatility + quantiles)
- GBM Meta-Labeling (trade filtering)
- RL Exit Optimizer (exit timing)

Features:
- Multi-task performance monitoring
- Drift detection (feature, volatility regime, quantile calibration)
- Dependency chain management (TCN → GBM → RL)
- Champion/challenger model comparison
- Blackout periods and cooldown management
- Profile-based presets (SCALP, SWING, INTRADAY)

Quick Start:
-----------
    from ml.risk_retraining import create_swing_scheduler, RiskModelType
    
    # Create and start scheduler
    scheduler = create_swing_scheduler()
    scheduler.start()
    
    # Record predictions for monitoring
    scheduler.record_prediction(
        RiskModelType.TCN_RISK,
        direction_pred=pred_probs,
        direction_target=actual_direction,
        volatility_pred=vol_pred,
        volatility_realized=realized_vol
    )
    
    # Check status
    status = scheduler.get_status()
    print(status)

Manual Retraining:
-----------------
    scheduler.trigger_manual_retraining(
        RiskModelType.TCN_RISK,
        reason="Quarterly model review"
    )

Custom Configuration:
--------------------
    from ml.risk_retraining import (
        RiskRetrainingScheduler,
        RiskRetrainingConfig,
        TCNRiskRetrainingConfig,
        GBMMetaRetrainingConfig
    )
    
    # Build custom scheduler
    scheduler = (
        RiskRetrainingScheduler.builder()
        .with_tcn_config(TCNRiskRetrainingConfig(
            direction_accuracy_threshold=0.58,
            volatility_mae_threshold=0.003
        ))
        .with_gbm_config(GBMMetaRetrainingConfig(
            precision_threshold=0.62
        ))
        .with_cascade_retraining(True)
        .build()
    )
"""

__version__ = "1.0.0"
__author__ = "pyForex"

# =============================================================================
# Configuration
# =============================================================================
from .risk_retraining_config import (
    # Enums
    RiskModelType,
    TriggerType,
    RetrainingStatus,
    
    # Metric configs
    TCNRiskMetrics,
    GBMMetaMetrics,
    RLExitMetrics,
    
    # Drift configs
    TCNRiskDriftConfig,
    GBMMetaDriftConfig,
    RLExitDriftConfig,
    
    # Schedule configs
    ScheduleConfig,
    BlackoutPeriod,
    
    # Dependency config
    ModelDependencyConfig,
    
    # Main retraining configs
    TCNRiskRetrainingConfig,
    GBMMetaRetrainingConfig,
    RLExitRetrainingConfig,
    RiskRetrainingConfig,
    
    # Profile presets
    get_scalp_config,
    get_swing_config,
    get_intraday_config,
)

# =============================================================================
# Performance Monitoring
# =============================================================================
from .risk_performance_monitor import (
    # Data classes
    MetricSnapshot,
    ModelHealth,
    PerformanceWindow,
    
    # Metric calculators
    TCNRiskMetricCalculator,
    GBMMetaMetricCalculator,
    RLExitMetricCalculator,
    
    # Main monitor
    RiskPerformanceMonitor,
)

# =============================================================================
# Drift Detection
# =============================================================================
from .risk_drift_detector import (
    # Enums
    DriftSeverity,
    VolatilityRegime,
    
    # Data classes
    DriftResult,
    FeatureDriftResult,
    RegimeChangeResult,
    CalibrationDriftResult,
    
    # Detectors
    FeatureDriftDetector,
    VolatilityRegimeDetector,
    QuantileCalibrationDetector,
    
    # Main detector
    RiskDriftDetector,
)

# =============================================================================
# Retraining Pipelines
# =============================================================================
from .risk_retraining_pipeline import (
    # Enums
    PipelineStage,
    
    # Data classes
    PipelineResult,
    
    # Data provider
    RiskDataProvider,
    
    # Individual pipelines
    TCNRiskTrainingPipeline,
    GBMMetaTrainingPipeline,
    RLExitTrainingPipeline,
    
    # Pipeline manager
    RiskRetrainingPipelineManager,
)

# =============================================================================
# Scheduler
# =============================================================================
from .risk_retraining_scheduler import (
    # Data classes
    RetrainingEvent,
    SchedulerStatus,
    
    # Managers
    BlackoutPeriodManager,
    CooldownManager,
    ScheduleManager,
    
    # Main scheduler
    RiskRetrainingScheduler,
    
    # Convenience constructors
    create_scalp_scheduler,
    create_swing_scheduler,
    create_intraday_scheduler,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version
    "__version__",
    
    # === Configuration ===
    "RiskModelType",
    "TriggerType",
    "RetrainingStatus",
    "TCNRiskMetrics",
    "GBMMetaMetrics",
    "RLExitMetrics",
    "TCNRiskDriftConfig",
    "GBMMetaDriftConfig",
    "RLExitDriftConfig",
    "ScheduleConfig",
    "BlackoutPeriod",
    "ModelDependencyConfig",
    "TCNRiskRetrainingConfig",
    "GBMMetaRetrainingConfig",
    "RLExitRetrainingConfig",
    "RiskRetrainingConfig",
    "get_scalp_config",
    "get_swing_config",
    "get_intraday_config",
    
    # === Performance Monitoring ===
    "MetricSnapshot",
    "ModelHealth",
    "PerformanceWindow",
    "TCNRiskMetricCalculator",
    "GBMMetaMetricCalculator",
    "RLExitMetricCalculator",
    "RiskPerformanceMonitor",
    
    # === Drift Detection ===
    "DriftSeverity",
    "VolatilityRegime",
    "DriftResult",
    "FeatureDriftResult",
    "RegimeChangeResult",
    "CalibrationDriftResult",
    "FeatureDriftDetector",
    "VolatilityRegimeDetector",
    "QuantileCalibrationDetector",
    "RiskDriftDetector",
    
    # === Pipelines ===
    "PipelineStage",
    "PipelineResult",
    "RiskDataProvider",
    "TCNRiskTrainingPipeline",
    "GBMMetaTrainingPipeline",
    "RLExitTrainingPipeline",
    "RiskRetrainingPipelineManager",
    
    # === Scheduler ===
    "RetrainingEvent",
    "SchedulerStatus",
    "BlackoutPeriodManager",
    "CooldownManager",
    "ScheduleManager",
    "RiskRetrainingScheduler",
    "create_scalp_scheduler",
    "create_swing_scheduler",
    "create_intraday_scheduler",
]


def get_version() -> str:
    """Return the package version."""
    return __version__


def get_available_models() -> list[RiskModelType]:
    """Return list of supported model types."""
    return list(RiskModelType)


def get_available_profiles() -> list[str]:
    """Return list of available profile presets."""
    return ["SCALP", "SWING", "INTRADAY"]