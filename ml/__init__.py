"""
ML Module for pyForex Trading System.

This module provides:
- Drift detection (data drift, concept drift)
- Performance monitoring and alerting
- Model versioning and lifecycle management
- Automated retraining scheduling

Usage:
    from ml import RetrainingScheduler, create_scheduler_for_profile
    
    # Create scheduler for SWING trading
    scheduler = create_scheduler_for_profile("SWING")
    
    # Start background monitoring
    scheduler.start_monitoring()
    
    # Or trigger manually
    scheduler.trigger_manual_retraining("Quarterly review")
"""

from .drift_detector import (
    DriftDetector,
    ConceptDriftDetector,
    DriftConfig,
    DriftResult,
    DriftType,
    DriftSeverity,
    StatisticalTests
)

from .performance_monitor import (
    PerformanceMonitor,
    MonitorConfig,
    TradeRecord,
    MetricType,
    MetricThreshold,
    AlertLevel,
    PerformanceAlert,
    PerformanceSnapshot
)

from .model_manager import (
    ModelManager,
    ManagerConfig,
    ModelMetadata,
    ValidationResult
)

from .retraining_scheduler import (
    RetrainingScheduler,
    RetrainingConfig,
    ScheduleConfig,
    RetrainingEvent,
    RetrainingStatus,
    TriggerType,
    DataPreparer,
    ModelTrainer,
    create_scheduler,
    create_scheduler_for_profile
)

__all__ = [
    # Drift Detection
    'DriftDetector',
    'ConceptDriftDetector',
    'DriftConfig',
    'DriftResult',
    'DriftType',
    'DriftSeverity',
    'StatisticalTests',
    
    # Performance Monitoring
    'PerformanceMonitor',
    'MonitorConfig',
    'TradeRecord',
    'MetricType',
    'MetricThreshold',
    'AlertLevel',
    'PerformanceAlert',
    'PerformanceSnapshot',
    
    # Model Management
    'ModelManager',
    'ManagerConfig',
    'ModelMetadata',
    'ValidationResult',
    
    # Retraining Scheduler
    'RetrainingScheduler',
    'RetrainingConfig',
    'ScheduleConfig',
    'RetrainingEvent',
    'RetrainingStatus',
    'TriggerType',
    'DataPreparer',
    'ModelTrainer',
    'create_scheduler',
    'create_scheduler_for_profile'
]
