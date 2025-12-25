"""
Retraining Scheduler Module for pyForex ML System.

Orchestrates automatic model retraining based on:
- Time-based schedules
- Performance degradation
- Data/concept drift detection
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import threading
import time
import json
from pathlib import Path

from utils.feature_schema import get_feature_schema_version

from .drift_detector import (
    DriftDetector, ConceptDriftDetector, DriftConfig, 
    DriftResult, DriftSeverity, DriftType
)
from .performance_monitor import (
    PerformanceMonitor, MonitorConfig, TradeRecord,
    MetricType, AlertLevel, PerformanceSnapshot
)
from .model_manager import (
    ModelManager, ManagerConfig, ModelMetadata, ValidationResult
)

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of retraining triggers."""
    SCHEDULED = "scheduled"           # Time-based schedule
    PERFORMANCE = "performance"       # Performance degradation
    DRIFT = "drift"                   # Data/concept drift
    MANUAL = "manual"                 # Manual trigger
    REGIME_CHANGE = "regime_change"   # Market regime change


class RetrainingStatus(Enum):
    """Status of retraining process."""
    IDLE = "idle"
    TRIGGERED = "triggered"
    PREPARING_DATA = "preparing_data"
    TRAINING = "training"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class RetrainingEvent:
    """Record of a retraining event."""
    event_id: str
    timestamp: datetime
    trigger_type: TriggerType
    trigger_reason: str
    profile_name: str
    status: RetrainingStatus
    old_model_id: Optional[str] = None
    new_model_id: Optional[str] = None
    training_duration_seconds: float = 0
    validation_result: Optional[Dict] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'trigger_type': self.trigger_type.value,
            'trigger_reason': self.trigger_reason,
            'profile_name': self.profile_name,
            'status': self.status.value,
            'old_model_id': self.old_model_id,
            'new_model_id': self.new_model_id,
            'training_duration_seconds': self.training_duration_seconds,
            'validation_result': self.validation_result,
            'error_message': self.error_message
        }


@dataclass
class ScheduleConfig:
    """Configuration for scheduled retraining."""
    enabled: bool = True
    
    # Time-based schedule
    schedule_type: str = "weekly"  # "daily", "weekly", "monthly", "cron"
    schedule_day: int = 6          # Day of week (0=Mon, 6=Sun) for weekly
    schedule_hour: int = 2         # Hour of day (0-23)
    schedule_minute: int = 0       # Minute of hour
    
    # Forex market awareness
    avoid_market_hours: bool = True
    market_close_hours: List[int] = field(default_factory=lambda: [21, 22, 23, 0, 1, 2])  # UTC
    avoid_high_impact_news: bool = True
    news_blackout_hours: int = 2   # Hours before/after high-impact news
    
    # Session preferences
    preferred_session: str = "weekend"  # "weekend", "asian", "european", "american"


@dataclass
class RetrainingConfig:
    """Configuration for retraining scheduler."""
    # General settings
    profile_name: str = "SWING"
    models_dir: str = "./models"
    data_dir: str = "./data"
    logs_dir: str = "./logs/retraining"
    
    # Schedule configuration
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    
    # Performance-based triggers
    performance_trigger_enabled: bool = True
    min_trades_before_trigger: int = 50
    
    # Drift-based triggers
    drift_trigger_enabled: bool = True
    drift_check_interval_minutes: int = 60
    
    # Training settings
    min_training_samples: int = 5000
    max_training_samples: int = 50000
    validation_split: float = 0.2
    
    # Safety settings
    require_validation: bool = True
    auto_rollback_enabled: bool = True
    rollback_grace_period_hours: int = 24
    max_consecutive_failures: int = 3
    
    # Data window (for training data selection)
    lookback_days: int = 90         # Use last N days of data
    min_lookback_days: int = 30     # Minimum required
    
    # Notification settings
    notify_on_trigger: bool = True
    notify_on_completion: bool = True
    notify_on_failure: bool = True


class DataPreparer:
    """
    Handles data preparation for retraining.
    This is a base class - implement specific logic in subclass.
    """
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
    
    def prepare_training_data(
        self,
        profile_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> tuple[Any, Any, Any, Any]:
        """
        Prepare training and validation data.
        
        Returns:
            (X_train, y_train, X_val, y_val)
        """
        raise NotImplementedError("Implement in subclass")
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        raise NotImplementedError("Implement in subclass")


class ModelTrainer:
    """
    Handles model training.
    This is a base class - implement specific logic in subclass.
    """
    
    def __init__(self, config: RetrainingConfig):
        self.config = config
    
    def train(
        self,
        X_train: Any,
        y_train: Any,
        hyperparameters: Optional[Dict] = None
    ) -> tuple[Any, Dict[str, float]]:
        """
        Train a model.
        
        Returns:
            (trained_model, training_metrics)
        """
        raise NotImplementedError("Implement in subclass")
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        raise NotImplementedError("Implement in subclass")
    
    def get_model_type(self) -> str:
        """Get model type string."""
        return "unknown"


class RetrainingScheduler:
    """
    Main retraining scheduler that orchestrates:
    - Time-based scheduled retraining
    - Performance-triggered retraining
    - Drift-triggered retraining
    """
    
    def __init__(
        self,
        config: RetrainingConfig,
        data_preparer: Optional[DataPreparer] = None,
        model_trainer: Optional[ModelTrainer] = None,
        drift_config: Optional[DriftConfig] = None,
        monitor_config: Optional[MonitorConfig] = None,
        manager_config: Optional[ManagerConfig] = None
    ):
        self.config = config
        
        # Initialize components
        self.drift_detector = ConceptDriftDetector(drift_config or DriftConfig())
        self.performance_monitor = PerformanceMonitor(monitor_config or MonitorConfig())
        self.model_manager = ModelManager(
            manager_config or ManagerConfig(models_dir=config.models_dir)
        )
        
        # Pluggable components
        self.data_preparer = data_preparer
        self.model_trainer = model_trainer
        
        # State
        self.status = RetrainingStatus.IDLE
        self.current_event: Optional[RetrainingEvent] = None
        self.retraining_history: List[RetrainingEvent] = []
        self.consecutive_failures = 0
        
        # Last check timestamps
        self.last_schedule_check = datetime.now()
        self.last_drift_check = datetime.now()
        self.last_performance_check = datetime.now()
        
        # Threading
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Callbacks
        self.on_trigger_callbacks: List[Callable] = []
        self.on_complete_callbacks: List[Callable] = []
        self.on_failure_callbacks: List[Callable] = []
        
        # Setup logging directory
        Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"RetrainingScheduler initialized for profile: {config.profile_name}")
    
    def set_data_preparer(self, preparer: DataPreparer) -> None:
        """Set the data preparer component."""
        self.data_preparer = preparer
    
    def set_model_trainer(self, trainer: ModelTrainer) -> None:
        """Set the model trainer component."""
        self.model_trainer = trainer
    
    def add_trigger_callback(self, callback: Callable) -> None:
        """Add callback for retraining trigger events."""
        self.on_trigger_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable) -> None:
        """Add callback for retraining completion events."""
        self.on_complete_callbacks.append(callback)
    
    def add_failure_callback(self, callback: Callable) -> None:
        """Add callback for retraining failure events."""
        self.on_failure_callbacks.append(callback)
    
    # ==================== Trade & Feature Recording ====================
    
    def record_trade(self, trade: TradeRecord) -> Optional[RetrainingEvent]:
        """
        Record a trade and check for performance-based retraining trigger.
        """
        snapshot = self.performance_monitor.add_trade(trade)
        
        if snapshot and self.config.performance_trigger_enabled:
            needs_retrain, reason = self.performance_monitor.needs_retraining()
            if needs_retrain:
                return self.trigger_retraining(TriggerType.PERFORMANCE, reason)
        
        return None
    
    def record_features(self, features: Any) -> Optional[RetrainingEvent]:
        """
        Record feature data and check for drift-based retraining trigger.
        """
        import pandas as pd
        
        if isinstance(features, pd.Series):
            drift_result = self.drift_detector.add_sample(features)
        else:
            drift_result = self.drift_detector.add_batch(features)
        
        if drift_result and drift_result.drift_detected:
            if drift_result.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                return self.trigger_retraining(
                    TriggerType.DRIFT,
                    drift_result.recommendation
                )
        
        return None
    
    def record_prediction(self, prediction: float, actual: float) -> None:
        """Record prediction for concept drift detection."""
        self.drift_detector.add_prediction(prediction, actual)
    
    # ==================== Trigger Checking ====================
    
    def check_scheduled_trigger(self) -> Optional[RetrainingEvent]:
        """Check if scheduled retraining should be triggered."""
        if not self.config.schedule.enabled:
            return None
        
        now = datetime.now()
        schedule = self.config.schedule
        
        # Check based on schedule type
        should_trigger = False
        
        if schedule.schedule_type == "daily":
            # Trigger if it's the scheduled hour and we haven't triggered today
            if (now.hour == schedule.schedule_hour and 
                now.minute >= schedule.schedule_minute and
                (now - self.last_schedule_check).days >= 1):
                should_trigger = True
                
        elif schedule.schedule_type == "weekly":
            # Trigger if it's the scheduled day and hour
            if (now.weekday() == schedule.schedule_day and
                now.hour == schedule.schedule_hour and
                now.minute >= schedule.schedule_minute and
                (now - self.last_schedule_check).days >= 7):
                should_trigger = True
                
        elif schedule.schedule_type == "monthly":
            # Trigger on first of month at scheduled hour
            if (now.day == 1 and
                now.hour == schedule.schedule_hour and
                now.minute >= schedule.schedule_minute and
                (now - self.last_schedule_check).days >= 28):
                should_trigger = True
        
        if should_trigger:
            # Check market hours
            if schedule.avoid_market_hours:
                if not self._is_safe_time_for_retraining():
                    logger.info("Scheduled retraining postponed - market hours")
                    return None
            
            self.last_schedule_check = now
            return self.trigger_retraining(
                TriggerType.SCHEDULED,
                f"Scheduled {schedule.schedule_type} retraining"
            )
        
        return None
    
    def check_drift_trigger(self) -> Optional[RetrainingEvent]:
        """Check drift detector and trigger if needed."""
        if not self.config.drift_trigger_enabled:
            return None
        
        now = datetime.now()
        if (now - self.last_drift_check).seconds < self.config.drift_check_interval_minutes * 60:
            return None
        
        self.last_drift_check = now
        
        # Check data drift
        drift_result = self.drift_detector.check_drift()
        if drift_result.drift_detected and drift_result.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            return self.trigger_retraining(TriggerType.DRIFT, drift_result.recommendation)
        
        # Check concept drift
        concept_result = self.drift_detector.check_concept_drift()
        if concept_result and concept_result.drift_detected:
            return self.trigger_retraining(TriggerType.DRIFT, concept_result.recommendation)
        
        return None
    
    def check_performance_trigger(self) -> Optional[RetrainingEvent]:
        """Check performance monitor and trigger if needed."""
        if not self.config.performance_trigger_enabled:
            return None
        
        # Ensure minimum trades
        if len(self.performance_monitor.closed_trades) < self.config.min_trades_before_trigger:
            return None
        
        needs_retrain, reason = self.performance_monitor.needs_retraining()
        if needs_retrain:
            return self.trigger_retraining(TriggerType.PERFORMANCE, reason)
        
        return None
    
    def _is_safe_time_for_retraining(self) -> bool:
        """Check if current time is safe for retraining (market closed)."""
        now = datetime.utcnow()
        
        # Check if it's weekend (Saturday or Sunday)
        if now.weekday() in [5, 6]:  # Saturday = 5, Sunday = 6
            return True
        
        # Check if in safe hours
        if now.hour in self.config.schedule.market_close_hours:
            return True
        
        return False
    
    # ==================== Retraining Process ====================
    
    def trigger_retraining(
        self,
        trigger_type: TriggerType,
        reason: str
    ) -> Optional[RetrainingEvent]:
        """
        Trigger a retraining event.
        """
        with self._lock:
            if self.status != RetrainingStatus.IDLE:
                logger.warning(f"Retraining already in progress: {self.status.value}")
                return None
            
            # Check consecutive failures
            if self.consecutive_failures >= self.config.max_consecutive_failures:
                logger.error("Max consecutive failures reached. Manual intervention required.")
                return None
            
            # Create event
            event = RetrainingEvent(
                event_id=self._generate_event_id(),
                timestamp=datetime.now(),
                trigger_type=trigger_type,
                trigger_reason=reason,
                profile_name=self.config.profile_name,
                status=RetrainingStatus.TRIGGERED,
                old_model_id=self.model_manager.active_models.get(self.config.profile_name)
            )
            
            self.current_event = event
            self.status = RetrainingStatus.TRIGGERED
            
            # Notify
            self._notify_trigger(event)
            
            logger.info(f"Retraining triggered: {trigger_type.value} - {reason}")
            
            return event
    
    def execute_retraining(self, event: Optional[RetrainingEvent] = None) -> RetrainingEvent:
        """
        Execute the retraining process.
        """
        event = event or self.current_event
        if not event:
            raise ValueError("No retraining event to execute")
        
        if not self.data_preparer or not self.model_trainer:
            raise ValueError("DataPreparer and ModelTrainer must be set")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Prepare data
            self._update_status(RetrainingStatus.PREPARING_DATA, event)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.lookback_days)
            
            X_train, y_train, X_val, y_val = self.data_preparer.prepare_training_data(
                self.config.profile_name,
                start_date,
                end_date
            )
            
            if len(X_train) < self.config.min_training_samples:
                raise ValueError(
                    f"Insufficient training data: {len(X_train)} < {self.config.min_training_samples}"
                )
            
            # Step 2: Train model
            self._update_status(RetrainingStatus.TRAINING, event)
            
            hyperparameters = self.model_trainer.get_default_hyperparameters()
            model, training_metrics = self.model_trainer.train(X_train, y_train, hyperparameters)
            
            # Step 3: Save model
            new_model_id = self.model_manager.save_model(
                model=model,
                profile_name=self.config.profile_name,
                version=self._generate_version(),
                model_type=self.model_trainer.get_model_type(),
                hyperparameters=hyperparameters,
                feature_names=self.data_preparer.get_feature_names(),
                feature_schema_version=get_feature_schema_version(),
                training_data=X_train,
                training_start=start_date,
                training_end=end_date,
                validation_metrics=training_metrics,
                notes=f"Auto-retrained: {event.trigger_type.value}"
            )
            
            event.new_model_id = new_model_id
            
            # Step 4: Validate
            self._update_status(RetrainingStatus.VALIDATING, event)
            
            if self.config.require_validation:
                validation_result = self.model_manager.validate_model(
                    candidate_id=new_model_id,
                    validation_data=X_val,
                    validation_targets=y_val,
                    predict_fn=lambda m, x: m.predict(x),
                    metric_calculators=self._get_metric_calculators()
                )
                
                event.validation_result = validation_result.to_dict()
                
                if not validation_result.passed:
                    raise ValueError(f"Validation failed: {validation_result.recommendation}")
            
            # Step 5: Deploy
            self._update_status(RetrainingStatus.DEPLOYING, event)
            
            success = self.model_manager.activate_model(
                new_model_id,
                force=not self.config.require_validation
            )
            
            if not success:
                raise ValueError("Failed to activate new model")
            
            # Update reference data for drift detector
            import pandas as pd
            if isinstance(X_train, pd.DataFrame):
                self.drift_detector.set_reference(X_train)
            
            # Complete
            event.status = RetrainingStatus.COMPLETED
            event.training_duration_seconds = (datetime.now() - start_time).total_seconds()
            
            self.consecutive_failures = 0
            self._notify_completion(event)
            
            logger.info(f"Retraining completed successfully: {new_model_id}")
            
        except Exception as e:
            event.status = RetrainingStatus.FAILED
            event.error_message = str(e)
            event.training_duration_seconds = (datetime.now() - start_time).total_seconds()
            
            self.consecutive_failures += 1
            self._notify_failure(event)
            
            logger.error(f"Retraining failed: {e}")
            
            # Auto-rollback if enabled
            if self.config.auto_rollback_enabled and event.old_model_id:
                self._attempt_rollback(event)
        
        finally:
            self.retraining_history.append(event)
            self._save_event_log(event)
            self.status = RetrainingStatus.IDLE
            self.current_event = None
        
        return event
    
    def _attempt_rollback(self, event: RetrainingEvent) -> None:
        """Attempt to rollback to previous model."""
        try:
            rolled_back_id = self.model_manager.rollback(self.config.profile_name, steps=1)
            if rolled_back_id:
                event.status = RetrainingStatus.ROLLED_BACK
                logger.info(f"Rolled back to: {rolled_back_id}")
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
    
    def _update_status(self, status: RetrainingStatus, event: RetrainingEvent) -> None:
        """Update retraining status."""
        self.status = status
        event.status = status
        logger.info(f"Retraining status: {status.value}")
    
    def _get_metric_calculators(self) -> Dict[str, Callable]:
        """Get metric calculator functions for validation."""
        import numpy as np
        
        def accuracy(preds, targets):
            return np.mean((preds > 0.5) == (targets > 0.5))
        
        def directional_accuracy(preds, targets):
            return np.mean(np.sign(preds) == np.sign(targets))
        
        def mse(preds, targets):
            return np.mean((preds - targets) ** 2)
        
        return {
            'accuracy': accuracy,
            'directional_accuracy': directional_accuracy,
            'mse': mse
        }
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        return f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _generate_version(self) -> str:
        """Generate version string."""
        return f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _save_event_log(self, event: RetrainingEvent) -> None:
        """Save event to log file."""
        log_file = Path(self.config.logs_dir) / f"{event.event_id}.json"
        with open(log_file, 'w') as f:
            json.dump(event.to_dict(), f, indent=2)
    
    # ==================== Notifications ====================
    
    def _notify_trigger(self, event: RetrainingEvent) -> None:
        """Notify trigger callbacks."""
        if self.config.notify_on_trigger:
            for callback in self.on_trigger_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Trigger callback error: {e}")
    
    def _notify_completion(self, event: RetrainingEvent) -> None:
        """Notify completion callbacks."""
        if self.config.notify_on_completion:
            for callback in self.on_complete_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Completion callback error: {e}")
    
    def _notify_failure(self, event: RetrainingEvent) -> None:
        """Notify failure callbacks."""
        if self.config.notify_on_failure:
            for callback in self.on_failure_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Failure callback error: {e}")
    
    # ==================== Background Monitoring ====================
    
    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self._running:
            logger.warning("Monitoring already running")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Background monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Background monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                # Check scheduled trigger
                event = self.check_scheduled_trigger()
                if event:
                    self.execute_retraining(event)
                
                # Check drift trigger (less frequently)
                if self.config.drift_trigger_enabled:
                    event = self.check_drift_trigger()
                    if event:
                        self.execute_retraining(event)
                
                # Check performance trigger
                if self.config.performance_trigger_enabled:
                    event = self.check_performance_trigger()
                    if event:
                        self.execute_retraining(event)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
            
            time.sleep(60)  # Check every minute
    
    # ==================== Manual Operations ====================
    
    def trigger_manual_retraining(self, reason: str = "Manual trigger") -> RetrainingEvent:
        """Manually trigger retraining."""
        event = self.trigger_retraining(TriggerType.MANUAL, reason)
        if event:
            return self.execute_retraining(event)
        raise ValueError("Failed to trigger retraining")
    
    def force_rollback(self, steps: int = 1) -> Optional[str]:
        """Force rollback to previous model version."""
        return self.model_manager.rollback(self.config.profile_name, steps)
    
    # ==================== Status & Reporting ====================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            'status': self.status.value,
            'profile': self.config.profile_name,
            'current_event': self.current_event.to_dict() if self.current_event else None,
            'consecutive_failures': self.consecutive_failures,
            'active_model': self.model_manager.active_models.get(self.config.profile_name),
            'total_retraining_events': len(self.retraining_history),
            'last_retraining': self.retraining_history[-1].to_dict() if self.retraining_history else None,
            'monitoring_active': self._running,
            'drift_trend': self.drift_detector.get_drift_trend(),
            'performance_summary': self.performance_monitor.get_performance_summary()
        }
    
    def get_retraining_history(self, limit: int = 10) -> List[Dict]:
        """Get recent retraining history."""
        return [e.to_dict() for e in self.retraining_history[-limit:]]
    
    def get_next_scheduled_time(self) -> Optional[datetime]:
        """Get next scheduled retraining time."""
        if not self.config.schedule.enabled:
            return None
        
        now = datetime.now()
        schedule = self.config.schedule
        
        if schedule.schedule_type == "daily":
            next_time = now.replace(
                hour=schedule.schedule_hour,
                minute=schedule.schedule_minute,
                second=0,
                microsecond=0
            )
            if next_time <= now:
                next_time += timedelta(days=1)
            return next_time
        
        elif schedule.schedule_type == "weekly":
            days_ahead = schedule.schedule_day - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_time = now + timedelta(days=days_ahead)
            return next_time.replace(
                hour=schedule.schedule_hour,
                minute=schedule.schedule_minute,
                second=0,
                microsecond=0
            )
        
        return None


# ==================== Factory Functions ====================

def create_scheduler(
    profile_name: str = "SWING",
    models_dir: str = "./models",
    schedule_type: str = "weekly",
    enable_drift_detection: bool = True,
    enable_performance_monitoring: bool = True
) -> RetrainingScheduler:
    """
    Factory function to create a configured RetrainingScheduler.
    """
    config = RetrainingConfig(
        profile_name=profile_name,
        models_dir=models_dir,
        schedule=ScheduleConfig(
            enabled=True,
            schedule_type=schedule_type
        ),
        drift_trigger_enabled=enable_drift_detection,
        performance_trigger_enabled=enable_performance_monitoring
    )
    
    return RetrainingScheduler(config=config)


def create_scheduler_for_profile(profile_name: str) -> RetrainingScheduler:
    """
    Create scheduler with profile-specific defaults.
    """
    if profile_name == "SCALP":
        config = RetrainingConfig(
            profile_name="SCALP",
            schedule=ScheduleConfig(
                schedule_type="daily",
                schedule_hour=2
            ),
            min_trades_before_trigger=100,  # More trades for scalping
            lookback_days=30  # Shorter lookback for fast markets
        )
    else:  # SWING
        config = RetrainingConfig(
            profile_name="SWING",
            schedule=ScheduleConfig(
                schedule_type="weekly",
                schedule_day=6,  # Sunday
                schedule_hour=2
            ),
            min_trades_before_trigger=30,
            lookback_days=90
        )
    
    return RetrainingScheduler(config=config)
