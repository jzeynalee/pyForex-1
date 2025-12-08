"""
pyForex ML Module

This module provides automated model lifecycle management including:
- Drift detection (data and concept drift)
- Performance monitoring with alerts
- Model versioning and rollback
- Automated retraining pipelines
- Scheduling with market-aware timing

Quick Start:
    from ml import RetrainingScheduler, RetrainingConfig
    
    config = RetrainingConfig.for_swing()
    scheduler = RetrainingScheduler(config)
    scheduler.start_monitoring()

For profile-specific setup:
    from ml import create_scheduler_for_profile
    
    scheduler = create_scheduler_for_profile("SCALP")
"""

# =============================================================================
# Configuration (import first - no dependencies)
# =============================================================================

from .retraining_config import (
    # Enums
    RetrainingTrigger,
    ScheduleType,
    MarketSession,
    ModelType,
    TradingProfile,
    
    # Threshold configs
    PerformanceThresholds,
    DriftThresholds,
    
    # Component configs  
    ScheduleConfig,
    ValidationConfig,
    DataConfig,
    ModelConfig,
    VersioningConfig,
    NotificationConfig,
    
    # Master config
    RetrainingConfig,
)

# =============================================================================
# Monitoring Components
# =============================================================================

from .drift_detector import (
    DriftDetector,
    ConceptDriftDetector,
    DriftConfig,
    DriftResult,
    DriftType,
    DriftSeverity,
    StatisticalTests,
)

from .performance_monitor import (
    PerformanceMonitor,
    MonitorConfig,
    TradeRecord,
    MetricType,
    AlertLevel,
    MetricThreshold,
    PerformanceAlert,
    PerformanceSnapshot,
)

# =============================================================================
# Model Management
# =============================================================================

from .model_manager import (
    ModelManager,
    ManagerConfig,
    ModelMetadata,
    ValidationResult,
)

# =============================================================================
# Training Pipeline
# =============================================================================

from .retraining_pipeline import (
    RetrainingPipeline,
    PipelineStage,
    PipelineResult,
    DataSplit,
)

# =============================================================================
# Scheduler
# =============================================================================

from .retraining_scheduler import (
    RetrainingScheduler,
    RetrainingStatus,
    RetrainingEvent,
    TriggerType,
    DataPreparer,
    ModelTrainer,
    create_scheduler,
    create_scheduler_for_profile,
)

# =============================================================================
# Version Info
# =============================================================================

__version__ = "2.0.0"
__author__ = "pyForex Team"

# =============================================================================
# Convenience Exports
# =============================================================================

__all__ = [
    # Configuration
    'RetrainingConfig',
    'RetrainingTrigger',
    'ScheduleType',
    'ScheduleConfig',
    'PerformanceThresholds',
    'DriftThresholds',
    'TradingProfile',
    
    # Drift Detection
    'DriftDetector',
    'ConceptDriftDetector', 
    'DriftConfig',
    'DriftResult',
    'DriftType',
    'DriftSeverity',
    
    # Performance Monitoring
    'PerformanceMonitor',
    'MonitorConfig',
    'TradeRecord',
    'MetricType',
    'AlertLevel',
    
    # Model Management
    'ModelManager',
    'ManagerConfig',
    'ModelMetadata',
    
    # Pipeline
    'RetrainingPipeline',
    'PipelineStage',
    'PipelineResult',
    
    # Scheduler
    'RetrainingScheduler',
    'RetrainingStatus',
    'RetrainingEvent',
    'DataPreparer',
    'ModelTrainer',
    
    # Factory functions
    'create_scheduler',
    'create_scheduler_for_profile',
]