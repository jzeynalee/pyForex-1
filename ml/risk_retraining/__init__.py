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
    RetrainingTriggerType,
    
    # Metric configs
    TCNRiskMetrics,
    GBMMetaMetrics,
    RLExitMetrics,
    
    # Drift configs
    TCNRiskDriftConfig,
    GBMMetaDriftConfig,
    RLExitDriftConfig,
    
    # Schedule configs
    TCNRiskScheduleConfig,
    GBMMetaScheduleConfig,
    RLExitScheduleConfig,
    
    # Dependency config
    ModelDependencyConfig,
    
    # Main retraining config
    RiskRetrainingConfig,
    
    # Profile presets
    get_config_for_profile,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    "__version__",
    "RiskModelType",
    "RetrainingTriggerType",
    "TCNRiskMetrics",
    "GBMMetaMetrics",
    "RLExitMetrics",
    "TCNRiskDriftConfig",
    "GBMMetaDriftConfig",
    "RLExitDriftConfig",
    "TCNRiskScheduleConfig",
    "GBMMetaScheduleConfig",
    "RLExitScheduleConfig",
    "ModelDependencyConfig",
    "RiskRetrainingConfig",
    "get_config_for_profile",
]

try:
    from .risk_performance_monitor import (
        MetricSnapshot,
        ModelHealth,
        PerformanceWindow,
        TCNRiskMetricCalculator,
        GBMMetaMetricCalculator,
        RLExitMetricCalculator,
        RiskPerformanceMonitor,
    )

    __all__.extend(
        [
            "MetricSnapshot",
            "ModelHealth",
            "PerformanceWindow",
            "TCNRiskMetricCalculator",
            "GBMMetaMetricCalculator",
            "RLExitMetricCalculator",
            "RiskPerformanceMonitor",
        ]
    )
except Exception:
    pass

try:
    from .risk_drift_detector import (
        DriftSeverity,
        VolatilityRegime,
        DriftResult,
        FeatureDriftResult,
        RegimeChangeResult,
        CalibrationDriftResult,
        FeatureDriftDetector,
        VolatilityRegimeDetector,
        QuantileCalibrationDetector,
        RiskDriftDetector,
    )

    __all__.extend(
        [
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
        ]
    )
except Exception:
    pass

try:
    from .risk_retraining_pipeline import (
        PipelineStage,
        PipelineResult,
        RiskDataProvider,
        TCNRiskTrainingPipeline,
        GBMMetaTrainingPipeline,
        RLExitTrainingPipeline,
        RiskRetrainingPipelineManager,
    )

    __all__.extend(
        [
            "PipelineStage",
            "PipelineResult",
            "RiskDataProvider",
            "TCNRiskTrainingPipeline",
            "GBMMetaTrainingPipeline",
            "RLExitTrainingPipeline",
            "RiskRetrainingPipelineManager",
        ]
    )
except Exception:
    pass

try:
    from .risk_retraining_scheduler import (
        RetrainingEvent,
        SchedulerStatus,
        BlackoutPeriodManager,
        CooldownManager,
        ScheduleManager,
        RiskRetrainingScheduler,
        create_scalp_scheduler,
        create_swing_scheduler,
        create_intraday_scheduler,
    )

    __all__.extend(
        [
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
    )
except Exception:
    pass


def get_version() -> str:
    """Return the package version."""
    return __version__


def get_available_models() -> list[RiskModelType]:
    """Return list of supported model types."""
    return list(RiskModelType)


def get_available_profiles() -> list[str]:
    """Return list of available profile presets."""
    return ["SCALP", "SWING", "INTRADAY"]