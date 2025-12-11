# ml/risk_retraining/risk_retraining_scheduler.py
"""
Retraining Scheduler for Risk Management Models.

Orchestrates retraining triggers for:
- TCN Risk Model
- GBM Meta-Labeling  
- RL Exit Optimizer

Features:
- Multiple trigger types (scheduled, performance, drift, dependency)
- Model dependency chain handling
- Cooldown management
- Blackout period enforcement
- Callback hooks for notifications
"""

import threading
import logging
from typing import Dict, Optional, List, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from enum import Enum
import time as time_module
import json
from pathlib import Path

from .risk_retraining_config import (
    RiskModelType, RiskRetrainingConfig, RetrainingTriggerType,
    get_config_for_profile
)
from .risk_performance_monitor import RiskPerformanceMonitor, ModelHealth, MetricStatus
from .risk_drift_detector import RiskDriftDetector, DriftResult, DriftSeverity
from .risk_retraining_pipeline import (
    RiskRetrainingPipelineManager, PipelineResult, PipelineStage
)

logger = logging.getLogger(__name__)


# =============================================================================
# Scheduler State
# =============================================================================

class SchedulerState(Enum):
    """State of the retraining scheduler."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    RETRAINING = "retraining"


@dataclass
class RetrainingEvent:
    """Record of a retraining event."""
    model_type: RiskModelType
    trigger: RetrainingTriggerType
    timestamp: datetime
    success: bool
    duration_seconds: float
    metrics_before: Optional[Dict[str, float]]
    metrics_after: Optional[Dict[str, float]]
    reason: str
    
    def to_dict(self) -> Dict:
        return {
            'model_type': self.model_type.name,
            'trigger': self.trigger.name,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'duration_seconds': self.duration_seconds,
            'metrics_before': self.metrics_before,
            'metrics_after': self.metrics_after,
            'reason': self.reason,
        }


# =============================================================================
# Blackout Period Manager
# =============================================================================

class BlackoutPeriodManager:
    """Manages blackout periods when retraining should not occur."""
    
    def __init__(self, config: RiskRetrainingConfig):
        self.config = config
        self.schedule = config.tcn_schedule
    
    def is_in_blackout(self) -> Tuple[bool, Optional[str]]:
        """Check if current time is in a blackout period."""
        now = datetime.utcnow()
        hour = now.hour
        
        # London open (7-9 UTC)
        if self.schedule.avoid_london_open and 7 <= hour < 9:
            return True, "London open (7-9 UTC)"
        
        # NY open (13-15 UTC)
        if self.schedule.avoid_ny_open and 13 <= hour < 15:
            return True, "NY open (13-15 UTC)"
        
        # Tokyo open (23-01 UTC)
        if self.schedule.avoid_tokyo_open and (hour >= 23 or hour < 1):
            return True, "Tokyo open (23-01 UTC)"
        
        # NFP day (first Friday of month)
        if self.schedule.avoid_nfp_day:
            if now.weekday() == 4 and now.day <= 7:  # Friday, first week
                if 12 <= hour < 15:  # NFP release window
                    return True, "NFP release window"
        
        return False, None
    
    def get_next_available_window(self) -> datetime:
        """Get next time when retraining is allowed."""
        now = datetime.utcnow()
        
        # Try each hour for next 24 hours
        for hours_ahead in range(1, 25):
            candidate = now + timedelta(hours=hours_ahead)
            # Temporarily update time for check
            in_blackout, _ = self._check_time(candidate)
            if not in_blackout:
                return candidate
        
        # Default to 24 hours from now
        return now + timedelta(hours=24)
    
    def _check_time(self, dt: datetime) -> Tuple[bool, Optional[str]]:
        """Check if a specific time is in blackout."""
        hour = dt.hour
        
        if self.schedule.avoid_london_open and 7 <= hour < 9:
            return True, "London open"
        if self.schedule.avoid_ny_open and 13 <= hour < 15:
            return True, "NY open"
        if self.schedule.avoid_tokyo_open and (hour >= 23 or hour < 1):
            return True, "Tokyo open"
        
        return False, None


# =============================================================================
# Cooldown Manager
# =============================================================================

class CooldownManager:
    """Manages cooldown periods between retraining runs."""
    
    def __init__(self, config: RiskRetrainingConfig):
        self.config = config
        self.last_retrained: Dict[RiskModelType, datetime] = {}
    
    def is_in_cooldown(self, model_type: RiskModelType) -> Tuple[bool, Optional[timedelta]]:
        """Check if model is in cooldown period."""
        last = self.last_retrained.get(model_type)
        if last is None:
            return False, None
        
        cooldown = timedelta(hours=self.config.cooldown_hours)
        elapsed = datetime.now() - last
        
        if elapsed < cooldown:
            remaining = cooldown - elapsed
            return True, remaining
        
        return False, None
    
    def mark_retrained(self, model_type: RiskModelType):
        """Mark model as having been retrained."""
        self.last_retrained[model_type] = datetime.now()
    
    def get_next_available(self, model_type: RiskModelType) -> datetime:
        """Get next time when model can be retrained."""
        last = self.last_retrained.get(model_type)
        if last is None:
            return datetime.now()
        
        cooldown = timedelta(hours=self.config.cooldown_hours)
        return last + cooldown


# =============================================================================
# Schedule Manager
# =============================================================================

class ScheduleManager:
    """Manages scheduled retraining times."""
    
    def __init__(self, config: RiskRetrainingConfig):
        self.config = config
        self.last_scheduled_run: Dict[RiskModelType, datetime] = {}
    
    def is_scheduled_now(self, model_type: RiskModelType) -> bool:
        """Check if scheduled retraining is due now."""
        schedule = self.config.get_schedule_config_for_model(model_type)
        
        if not schedule.enabled:
            return False
        
        now = datetime.utcnow()
        last_run = self.last_scheduled_run.get(model_type)
        
        # Check based on schedule type
        if schedule.schedule_type == 'daily':
            if now.hour == schedule.schedule_hour:
                if last_run is None or (now - last_run) > timedelta(hours=20):
                    return True
        
        elif schedule.schedule_type == 'weekly':
            if now.weekday() == schedule.schedule_day_of_week and now.hour == schedule.schedule_hour:
                if last_run is None or (now - last_run) > timedelta(days=5):
                    return True
        
        elif schedule.schedule_type == 'monthly':
            if now.day == schedule.schedule_day_of_month and now.hour == schedule.schedule_hour:
                if last_run is None or (now - last_run) > timedelta(days=25):
                    return True
        
        return False
    
    def mark_scheduled_run(self, model_type: RiskModelType):
        """Mark that scheduled run occurred."""
        self.last_scheduled_run[model_type] = datetime.now()
    
    def get_next_scheduled(self, model_type: RiskModelType) -> Optional[datetime]:
        """Get next scheduled retraining time."""
        schedule = self.config.get_schedule_config_for_model(model_type)
        
        if not schedule.enabled:
            return None
        
        now = datetime.utcnow()
        
        if schedule.schedule_type == 'daily':
            next_run = now.replace(hour=schedule.schedule_hour, minute=0, second=0)
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run
        
        elif schedule.schedule_type == 'weekly':
            days_ahead = schedule.schedule_day_of_week - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_run = now + timedelta(days=days_ahead)
            next_run = next_run.replace(hour=schedule.schedule_hour, minute=0, second=0)
            return next_run
        
        return None


# =============================================================================
# Main Scheduler
# =============================================================================

class RiskRetrainingScheduler:
    """
    Main scheduler for Risk Management model retraining.
    
    Coordinates:
    - Performance monitoring
    - Drift detection
    - Scheduled retraining
    - Dependency chain execution
    - Callbacks and notifications
    """
    
    def __init__(
        self,
        config: Optional[RiskRetrainingConfig] = None,
        profile: Optional[str] = None,
    ):
        # Configuration
        if config is not None:
            self.config = config
        elif profile is not None:
            self.config = get_config_for_profile(profile)
        else:
            self.config = RiskRetrainingConfig()
        
        # Managers
        self.blackout_manager = BlackoutPeriodManager(self.config)
        self.cooldown_manager = CooldownManager(self.config)
        self.schedule_manager = ScheduleManager(self.config)
        
        # Monitors
        self.performance_monitor = RiskPerformanceMonitor(self.config)
        self.drift_detector = RiskDriftDetector(self.config)
        
        # Pipeline manager
        self.pipeline_manager = RiskRetrainingPipelineManager(self.config)
        
        # State
        self.state = SchedulerState.STOPPED
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # History
        self.event_history: List[RetrainingEvent] = []
        self.pending_retraining: List[Tuple[RiskModelType, RetrainingTriggerType, str]] = []
        
        # Callbacks
        self.on_retraining_start: Optional[Callable[[RiskModelType, str], None]] = None
        self.on_retraining_complete: Optional[Callable[[PipelineResult], None]] = None
        self.on_retraining_failed: Optional[Callable[[RiskModelType, str], None]] = None
        self.on_drift_detected: Optional[Callable[[RiskModelType, DriftResult], None]] = None
        
        # Check interval
        self.check_interval_seconds = 60
        
        logger.info(f"RiskRetrainingScheduler initialized for profile: {self.config.profile_name}")
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    def start(self):
        """Start the scheduler in background thread."""
        if self.state == SchedulerState.RUNNING:
            logger.warning("Scheduler already running")
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.state = SchedulerState.RUNNING
        
        logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self.state = SchedulerState.STOPPED
        
        logger.info("Scheduler stopped")
    
    def pause(self):
        """Pause the scheduler temporarily."""
        self.state = SchedulerState.PAUSED
        logger.info("Scheduler paused")
    
    def resume(self):
        """Resume paused scheduler."""
        if self.state == SchedulerState.PAUSED:
            self.state = SchedulerState.RUNNING
            logger.info("Scheduler resumed")
    
    # =========================================================================
    # Main Loop
    # =========================================================================
    
    def _run_loop(self):
        """Main scheduler loop."""
        while not self._stop_event.is_set():
            try:
                if self.state == SchedulerState.RUNNING:
                    self._check_all_triggers()
                    self._process_pending_retraining()
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
            
            # Wait for next check
            self._stop_event.wait(self.check_interval_seconds)
    
    def _check_all_triggers(self):
        """Check all retraining triggers for all models."""
        for model_type in RiskModelType:
            self._check_triggers_for_model(model_type)
    
    def _check_triggers_for_model(self, model_type: RiskModelType):
        """Check all triggers for a specific model."""
        # Check cooldown first
        in_cooldown, remaining = self.cooldown_manager.is_in_cooldown(model_type)
        if in_cooldown:
            return
        
        # Check blackout
        in_blackout, reason = self.blackout_manager.is_in_blackout()
        if in_blackout:
            return
        
        # Check scheduled retraining
        if self.schedule_manager.is_scheduled_now(model_type):
            self._queue_retraining(
                model_type,
                RetrainingTriggerType.SCHEDULED,
                "Scheduled retraining"
            )
            return
        
        # Check performance-based triggers
        health = self.performance_monitor.get_model_health(model_type)
        if health.needs_retraining:
            self._queue_retraining(
                model_type,
                RetrainingTriggerType.PERFORMANCE,
                health.reason or "Performance degradation"
            )
            return
        
        # Check drift-based triggers
        if self.drift_detector.should_check(model_type):
            drift_triggered, drift_reason = self._check_drift_trigger(model_type)
            if drift_triggered:
                self._queue_retraining(
                    model_type,
                    RetrainingTriggerType.DRIFT,
                    drift_reason
                )
    
    def _check_drift_trigger(self, model_type: RiskModelType) -> Tuple[bool, Optional[str]]:
        """Check if drift triggers retraining."""
        # This requires recent data - in production this would come from
        # the live data feed. For now, return False.
        return False, None
    
    def _queue_retraining(
        self,
        model_type: RiskModelType,
        trigger: RetrainingTriggerType,
        reason: str
    ):
        """Queue a retraining request."""
        # Check if already queued
        for queued in self.pending_retraining:
            if queued[0] == model_type:
                return
        
        self.pending_retraining.append((model_type, trigger, reason))
        logger.info(f"Queued retraining: {model_type.name} ({trigger.name}) - {reason}")
    
    def _process_pending_retraining(self):
        """Process queued retraining requests."""
        if not self.pending_retraining:
            return
        
        # Process one at a time
        model_type, trigger, reason = self.pending_retraining.pop(0)
        self._execute_retraining(model_type, trigger, reason)
    
    # =========================================================================
    # Retraining Execution
    # =========================================================================
    
    def _execute_retraining(
        self,
        model_type: RiskModelType,
        trigger: RetrainingTriggerType,
        reason: str
    ):
        """Execute retraining for a model."""
        self.state = SchedulerState.RETRAINING
        start_time = datetime.now()
        
        # Get metrics before
        health_before = self.performance_monitor.get_model_health(model_type)
        metrics_before = {m.name: m.value for m in health_before.metrics}
        
        # Notify start
        if self.on_retraining_start:
            try:
                self.on_retraining_start(model_type, reason)
            except Exception as e:
                logger.error(f"Callback error: {e}")
        
        logger.info(f"Starting retraining: {model_type.name} ({trigger.name}) - {reason}")
        
        try:
            # Run pipeline with dependencies
            results = self.pipeline_manager.run_with_dependencies(
                model_type,
                trigger
            )
            
            # Get primary result
            primary_result = results[0]
            
            if primary_result.success:
                # Update managers
                self.cooldown_manager.mark_retrained(model_type)
                self.performance_monitor.mark_retrained(model_type)
                
                if trigger == RetrainingTriggerType.SCHEDULED:
                    self.schedule_manager.mark_scheduled_run(model_type)
                
                # Notify completion
                if self.on_retraining_complete:
                    try:
                        self.on_retraining_complete(primary_result)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                logger.info(f"Retraining completed: {model_type.name}")
            else:
                # Notify failure
                if self.on_retraining_failed:
                    try:
                        self.on_retraining_failed(model_type, primary_result.error or "Unknown error")
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
                
                logger.error(f"Retraining failed: {model_type.name} - {primary_result.error}")
            
            # Record event
            event = RetrainingEvent(
                model_type=model_type,
                trigger=trigger,
                timestamp=start_time,
                success=primary_result.success,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                metrics_before=metrics_before,
                metrics_after=primary_result.metrics,
                reason=reason,
            )
            self.event_history.append(event)
            
        except Exception as e:
            logger.error(f"Retraining error: {e}")
            
            if self.on_retraining_failed:
                try:
                    self.on_retraining_failed(model_type, str(e))
                except:
                    pass
        
        finally:
            self.state = SchedulerState.RUNNING
    
    # =========================================================================
    # Manual Triggers
    # =========================================================================
    
    def trigger_manual_retraining(
        self,
        model_type: RiskModelType,
        reason: str = "Manual trigger"
    ) -> bool:
        """Manually trigger retraining for a model."""
        # Check cooldown
        in_cooldown, remaining = self.cooldown_manager.is_in_cooldown(model_type)
        if in_cooldown:
            logger.warning(f"Model in cooldown, {remaining} remaining")
            return False
        
        self._queue_retraining(model_type, RetrainingTriggerType.MANUAL, reason)
        
        # Process immediately if not already retraining
        if self.state != SchedulerState.RETRAINING:
            self._process_pending_retraining()
        
        return True
    
    def force_retraining(
        self,
        model_type: RiskModelType,
        reason: str = "Forced retraining"
    ):
        """Force retraining, bypassing cooldown."""
        logger.warning(f"Forcing retraining for {model_type.name}")
        self._execute_retraining(model_type, RetrainingTriggerType.MANUAL, reason)
    
    # =========================================================================
    # Data Recording
    # =========================================================================
    
    def record_prediction(
        self,
        model_type: RiskModelType,
        **kwargs
    ):
        """Record a prediction for performance monitoring."""
        if model_type == RiskModelType.TCN_RISK:
            self.performance_monitor.record_tcn_prediction(**kwargs)
        elif model_type == RiskModelType.GBM_META:
            self.performance_monitor.record_meta_prediction(**kwargs)
        elif model_type == RiskModelType.RL_EXIT:
            self.performance_monitor.record_rl_episode(**kwargs)
    
    def update_drift_reference(
        self,
        model_type: RiskModelType,
        **kwargs
    ):
        """Update drift detector reference data."""
        if model_type == RiskModelType.TCN_RISK:
            self.drift_detector.set_tcn_reference(**kwargs)
        elif model_type == RiskModelType.GBM_META:
            self.drift_detector.set_gbm_reference(**kwargs)
        elif model_type == RiskModelType.RL_EXIT:
            self.drift_detector.set_rl_reference(**kwargs)
    
    # =========================================================================
    # Status and Reporting
    # =========================================================================
    
    def get_status(self) -> Dict:
        """Get current scheduler status."""
        return {
            'state': self.state.value,
            'profile': self.config.profile_name,
            'pending_retraining': len(self.pending_retraining),
            'total_events': len(self.event_history),
            'models': {
                model_type.name: {
                    'health': self.performance_monitor.get_model_health(model_type).to_dict(),
                    'in_cooldown': self.cooldown_manager.is_in_cooldown(model_type)[0],
                    'next_scheduled': str(self.schedule_manager.get_next_scheduled(model_type)),
                }
                for model_type in RiskModelType
            },
            'blackout': {
                'in_blackout': self.blackout_manager.is_in_blackout()[0],
                'reason': self.blackout_manager.is_in_blackout()[1],
            },
        }
    
    def get_event_history(self, limit: int = 20) -> List[Dict]:
        """Get recent retraining events."""
        return [e.to_dict() for e in self.event_history[-limit:]]
    
    def export_status(self, filepath: str):
        """Export status to JSON file."""
        status = self.get_status()
        status['event_history'] = self.get_event_history(50)
        status['timestamp'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(status, f, indent=2)
        
        logger.info(f"Exported status to {filepath}")


# =============================================================================
# Builder Pattern
# =============================================================================

class RiskRetrainingSchedulerBuilder:
    """Builder for RiskRetrainingScheduler."""
    
    def __init__(self):
        self._config: Optional[RiskRetrainingConfig] = None
        self._callbacks: Dict[str, Callable] = {}
    
    def with_config(self, config: RiskRetrainingConfig) -> 'RiskRetrainingSchedulerBuilder':
        self._config = config
        return self
    
    def with_profile(self, profile: str) -> 'RiskRetrainingSchedulerBuilder':
        self._config = get_config_for_profile(profile)
        return self
    
    def on_start(self, callback: Callable) -> 'RiskRetrainingSchedulerBuilder':
        self._callbacks['start'] = callback
        return self
    
    def on_complete(self, callback: Callable) -> 'RiskRetrainingSchedulerBuilder':
        self._callbacks['complete'] = callback
        return self
    
    def on_failed(self, callback: Callable) -> 'RiskRetrainingSchedulerBuilder':
        self._callbacks['failed'] = callback
        return self
    
    def on_drift(self, callback: Callable) -> 'RiskRetrainingSchedulerBuilder':
        self._callbacks['drift'] = callback
        return self
    
    def build(self) -> RiskRetrainingScheduler:
        scheduler = RiskRetrainingScheduler(config=self._config)
        
        if 'start' in self._callbacks:
            scheduler.on_retraining_start = self._callbacks['start']
        if 'complete' in self._callbacks:
            scheduler.on_retraining_complete = self._callbacks['complete']
        if 'failed' in self._callbacks:
            scheduler.on_retraining_failed = self._callbacks['failed']
        if 'drift' in self._callbacks:
            scheduler.on_drift_detected = self._callbacks['drift']
        
        return scheduler


# =============================================================================
# Convenience Functions
# =============================================================================

def create_scheduler_for_profile(
    profile: str,
    symbols: Optional[List[str]] = None
) -> RiskRetrainingScheduler:
    """Create scheduler for a trading profile."""
    config = get_config_for_profile(profile, symbols)
    return RiskRetrainingScheduler(config=config)


def create_scalp_scheduler() -> RiskRetrainingScheduler:
    """Create scheduler optimized for scalping."""
    return create_scheduler_for_profile('SCALP')


def create_swing_scheduler() -> RiskRetrainingScheduler:
    """Create scheduler optimized for swing trading."""
    return create_scheduler_for_profile('SWING')


def create_intraday_scheduler() -> RiskRetrainingScheduler:
    """Create scheduler optimized for intraday trading."""
    return create_scheduler_for_profile('INTRADAY')