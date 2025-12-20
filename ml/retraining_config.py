"""
Consolidated Retraining Configuration for pyForex ML System.

This module unifies all configuration classes for the retraining system,
eliminating duplicates that existed across multiple files.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable
from enum import Enum, auto
from datetime import time, timedelta, datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class RetrainingTrigger(Enum):
    """Types of triggers that can initiate retraining."""
    SCHEDULED = "scheduled"           # Time-based (cron-like)
    PERFORMANCE = "performance"       # Metrics dropped below threshold
    DRIFT = "drift"                   # Feature/concept drift detected
    MANUAL = "manual"                 # User-initiated
    REGIME_CHANGE = "regime_change"   # Market regime shifted


class ScheduleType(Enum):
    """Types of time-based schedules."""
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class MarketSession(Enum):
    """Forex market sessions with UTC times."""
    SYDNEY = "sydney"      # 22:00 - 07:00 UTC
    TOKYO = "tokyo"        # 00:00 - 09:00 UTC
    LONDON = "london"      # 08:00 - 17:00 UTC
    NEW_YORK = "new_york"  # 13:00 - 22:00 UTC
    WEEKEND = "weekend"    # Saturday 21:00 - Sunday 21:00 UTC


class ModelType(Enum):
    """Supported model types."""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    TCN = "tcn"
    ENSEMBLE = "ensemble"


class TradingProfile(Enum):
    """Trading strategy profiles."""
    SCALP = "SCALP"
    INTRADAY = "INTRADAY"
    SWING = "SWING"


# =============================================================================
# THRESHOLD CONFIGURATIONS
# =============================================================================

@dataclass
class PerformanceThresholds:
    """Thresholds for performance-based retraining triggers."""
    
    # Accuracy metrics (minimum acceptable values)
    min_accuracy: float = 0.55
    min_precision: float = 0.50
    min_recall: float = 0.50
    min_f1_score: float = 0.52
    
    # Trading metrics
    min_win_rate: float = 0.45
    min_profit_factor: float = 1.2
    max_drawdown: float = 0.15
    min_sharpe_ratio: float = 0.5
    min_sortino_ratio: float = 0.5
    
    # Model confidence
    min_avg_confidence: float = 0.55
    max_uncertainty: float = 0.30
    
    # Evaluation windows
    evaluation_window_trades: int = 50
    evaluation_window_days: int = 7
    grace_period_hours: int = 24
    
    def check_thresholds(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check which thresholds are violated. Returns {metric: is_violated}."""
        threshold_map = {
            'accuracy': ('min', self.min_accuracy),
            'precision': ('min', self.min_precision),
            'recall': ('min', self.min_recall),
            'f1_score': ('min', self.min_f1_score),
            'win_rate': ('min', self.min_win_rate),
            'profit_factor': ('min', self.min_profit_factor),
            'drawdown': ('max', self.max_drawdown),
            'sharpe_ratio': ('min', self.min_sharpe_ratio),
            'sortino_ratio': ('min', self.min_sortino_ratio),
            'avg_confidence': ('min', self.min_avg_confidence),
            'uncertainty': ('max', self.max_uncertainty),
        }
        
        violations = {}
        for metric_name, (direction, threshold) in threshold_map.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if direction == 'min':
                    violations[metric_name] = value < threshold
                else:
                    violations[metric_name] = value > threshold
        
        return violations
    
    def get_violated(self, metrics: Dict[str, float]) -> List[str]:
        """Get list of violated threshold names."""
        violations = self.check_thresholds(metrics)
        return [k for k, v in violations.items() if v]


@dataclass
class DriftThresholds:
    """Thresholds for drift detection."""
    
    # Statistical test thresholds
    ks_threshold: float = 0.1          # KS test p-value threshold
    psi_threshold: float = 0.2         # Population Stability Index
    js_threshold: float = 0.1          # Jensen-Shannon divergence
    
    # Window sizes
    reference_window_size: int = 1000
    detection_window_size: int = 200
    
    # Feature-level settings
    min_features_drifted: int = 3
    feature_drift_ratio: float = 0.2
    
    # Severity thresholds
    low_threshold: float = 0.15
    medium_threshold: float = 0.30
    high_threshold: float = 0.50
    critical_threshold: float = 0.70
    
    # Monitoring
    check_interval_bars: int = 50
    history_length: int = 100


# =============================================================================
# SCHEDULE CONFIGURATION
# =============================================================================

@dataclass
class ScheduleConfig:
    """Configuration for time-based retraining schedule."""
    
    enabled: bool = True
    schedule_type: ScheduleType = ScheduleType.WEEKLY
    
    # Time settings
    training_days: List[int] = field(default_factory=lambda: [5])  # Saturday
    training_hour: int = 2   # 2 AM UTC
    training_minute: int = 0
    timezone: str = "UTC"
    
    # For custom interval-based schedules
    custom_interval_hours: int = 168  # 1 week
    
    # Market awareness
    avoid_market_hours: bool = True
    market_close_hours: List[int] = field(
        default_factory=lambda: [21, 22, 23, 0, 1, 2]
    )
    blackout_sessions: List[MarketSession] = field(default_factory=list)
    
    # News awareness
    avoid_high_impact_news: bool = True
    news_buffer_hours: int = 4
    
    # Session preference
    preferred_session: MarketSession = MarketSession.WEEKEND
    
    def get_next_scheduled_time(self, from_time: Optional[datetime] = None) -> datetime:
        """Calculate next scheduled retraining time."""
        now = from_time or datetime.now()
        
        if self.schedule_type == ScheduleType.DAILY:
            next_time = now.replace(
                hour=self.training_hour,
                minute=self.training_minute,
                second=0, microsecond=0
            )
            if next_time <= now:
                next_time += timedelta(days=1)
            return next_time
        
        elif self.schedule_type == ScheduleType.WEEKLY:
            # Find next matching day
            target_day = self.training_days[0] if self.training_days else 5
            days_ahead = target_day - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_time = now + timedelta(days=days_ahead)
            return next_time.replace(
                hour=self.training_hour,
                minute=self.training_minute,
                second=0, microsecond=0
            )
        
        elif self.schedule_type == ScheduleType.CUSTOM:
            return now + timedelta(hours=self.custom_interval_hours)
        
        return now + timedelta(days=7)  # Default fallback


# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuration for model validation before deployment."""
    
    # Sample requirements
    min_validation_samples: int = 500
    validation_split: float = 0.2
    
    # Champion/Challenger comparison
    require_improvement: bool = True
    min_improvement_pct: float = 2.0
    
    # Statistical significance
    require_statistical_significance: bool = True
    significance_level: float = 0.05
    
    # Stability checks (cross-validation)
    require_stability_check: bool = True
    stability_folds: int = 5
    max_std_across_folds: float = 0.05
    
    # Out-of-time validation
    use_oot_validation: bool = True
    oot_window_days: int = 14
    
    # Primary metric for comparison
    primary_metric: str = "sharpe_ratio"
    secondary_metrics: List[str] = field(
        default_factory=lambda: ["win_rate", "profit_factor"]
    )
    max_secondary_degradation_pct: float = 10.0


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for training data management."""
    
    # Data windows
    max_training_samples: int = 1000000
    min_training_samples: int = 100000
    lookback_days: int = 90
    
    # Feature selection
    feature_selection_enabled: bool = True
    max_features: int = 100
    min_feature_importance: float = 0.01
    
    # Data quality
    max_missing_ratio: float = 0.05
    max_outlier_ratio: float = 0.02
    
    # Class balancing
    balance_classes: bool = True
    balance_method: str = "smote"  # 'smote', 'undersample', 'oversample'


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for model training."""
    
    model_type: ModelType = ModelType.LIGHTGBM
    
    # Hyperparameter optimization
    tune_hyperparameters: bool = True
    tuning_method: str = "optuna"
    tuning_trials: int = 50
    tuning_timeout_minutes: int = 30
    
    # Training settings
    early_stopping_rounds: int = 50
    n_estimators: int = 1000
    learning_rate: float = 0.05
    
    # Ensemble settings
    ensemble_models: List[ModelType] = field(
        default_factory=lambda: [ModelType.LIGHTGBM, ModelType.XGBOOST]
    )
    ensemble_weights: Optional[List[float]] = None


# =============================================================================
# VERSIONING CONFIGURATION
# =============================================================================

@dataclass
class VersioningConfig:
    """Configuration for model versioning and rollback."""
    
    model_dir: str = "models"
    max_versions_to_keep: int = 10
    
    # Naming
    version_prefix: str = "v"
    include_timestamp: bool = True
    include_metrics: bool = True
    
    # Rollback settings
    enable_auto_rollback: bool = True
    rollback_threshold_trades: int = 20
    rollback_degradation_pct: float = 10.0


# =============================================================================
# NOTIFICATION CONFIGURATION
# =============================================================================

@dataclass
class NotificationConfig:
    """Configuration for retraining notifications."""
    
    enable_logging: bool = True
    enable_file_logging: bool = True
    log_file_path: str = "logs/retraining.log"
    
    # Event notifications
    notify_on_start: bool = True
    notify_on_complete: bool = True
    notify_on_failure: bool = True
    notify_on_validation_fail: bool = True
    notify_on_rollback: bool = True
    
    # Optional webhook
    webhook_url: Optional[str] = None


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================

@dataclass
class RetrainingConfig:
    """
    Master configuration for the retraining system.
    
    Consolidates all sub-configurations into a single manageable object.
    """
    
    # Sub-configurations
    performance: PerformanceThresholds = field(default_factory=PerformanceThresholds)
    drift: DriftThresholds = field(default_factory=DriftThresholds)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    versioning: VersioningConfig = field(default_factory=VersioningConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    
    # Enabled triggers
    enabled_triggers: Set[RetrainingTrigger] = field(
        default_factory=lambda: {
            RetrainingTrigger.SCHEDULED,
            RetrainingTrigger.PERFORMANCE,
            RetrainingTrigger.DRIFT,
            RetrainingTrigger.MANUAL
        }
    )
    
    # General settings
    profile: TradingProfile = TradingProfile.SWING
    symbols: List[str] = field(default_factory=lambda: ["EURUSD"])
    cooldown_hours: int = 12
    monitor_interval_seconds: int = 300
    max_consecutive_failures: int = 3
    
    # Directory structure
    models_dir: str = "./models"
    data_dir: str = "./data"
    logs_dir: str = "./logs/retraining"
    
    def is_trigger_enabled(self, trigger: RetrainingTrigger) -> bool:
        """Check if a specific trigger is enabled."""
        return trigger in self.enabled_triggers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'profile': self.profile.value,
            'symbols': self.symbols,
            'cooldown_hours': self.cooldown_hours,
            'enabled_triggers': [t.value for t in self.enabled_triggers],
            'schedule_type': self.schedule.schedule_type.value,
            'model_type': self.model.model_type.value,
            'min_training_samples': self.data.min_training_samples,
            'lookback_days': self.data.lookback_days,
        }
    
    # =========================================================================
    # FACTORY METHODS
    # =========================================================================
    
    @classmethod
    def for_scalping(cls) -> 'RetrainingConfig':
        """Create config optimized for scalping strategies."""
        config = cls()
        config.profile = TradingProfile.SCALP
        config.schedule.schedule_type = ScheduleType.DAILY
        config.schedule.training_days = list(range(7))
        config.performance.evaluation_window_trades = 100
        config.performance.evaluation_window_days = 3
        config.data.lookback_days = 30
        config.cooldown_hours = 6
        return config
    
    @classmethod
    def for_intraday(cls) -> 'RetrainingConfig':
        """Create config optimized for intraday trading."""
        config = cls()
        config.profile = TradingProfile.INTRADAY
        config.schedule.schedule_type = ScheduleType.WEEKLY
        config.schedule.training_days = [2, 5]  # Wed, Sat
        config.performance.evaluation_window_trades = 50
        config.performance.evaluation_window_days = 5
        config.data.lookback_days = 60
        config.cooldown_hours = 12
        return config
    
    @classmethod
    def for_swing(cls) -> 'RetrainingConfig':
        """Create config optimized for swing trading."""
        config = cls()
        config.profile = TradingProfile.SWING
        config.schedule.schedule_type = ScheduleType.WEEKLY
        config.performance.evaluation_window_trades = 30
        config.performance.evaluation_window_days = 14
        config.data.lookback_days = 180
        config.cooldown_hours = 48
        return config
    
    @classmethod
    def conservative(cls) -> 'RetrainingConfig':
        """Create conservative config with strict validation."""
        config = cls()
        config.validation.min_improvement_pct = 5.0
        config.validation.require_statistical_significance = True
        config.validation.stability_folds = 10
        config.performance.grace_period_hours = 48
        config.cooldown_hours = 72
        return config
    
    @classmethod
    def aggressive(cls) -> 'RetrainingConfig':
        """Create aggressive config for fast adaptation."""
        config = cls()
        config.schedule.schedule_type = ScheduleType.DAILY
        config.validation.min_improvement_pct = 0.5
        config.validation.require_statistical_significance = False
        config.performance.grace_period_hours = 6
        config.cooldown_hours = 6
        config.enabled_triggers.add(RetrainingTrigger.REGIME_CHANGE)
        return config
    
    @classmethod
    def from_profile(cls, profile: str) -> 'RetrainingConfig':
        """Create config from profile name string."""
        profile_upper = profile.upper()
        if profile_upper == "SCALP":
            return cls.for_scalping()
        elif profile_upper == "INTRADAY":
            return cls.for_intraday()
        elif profile_upper == "SWING":
            return cls.for_swing()
        else:
            logger.warning(f"Unknown profile '{profile}', using SWING defaults")
            return cls.for_swing()