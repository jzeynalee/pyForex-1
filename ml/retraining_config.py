"""
Retraining Configuration Module for pyForex ML System.

Defines all configuration options for the retraining scheduler including
time-based schedules, performance thresholds, and forex-specific settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
from datetime import time, timedelta
import logging

logger = logging.getLogger(__name__)


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
    """Forex market sessions."""
    SYDNEY = "sydney"      # 22:00 - 07:00 UTC
    TOKYO = "tokyo"        # 00:00 - 09:00 UTC
    LONDON = "london"      # 08:00 - 17:00 UTC
    NEW_YORK = "new_york"  # 13:00 - 22:00 UTC


@dataclass
class PerformanceThresholds:
    """Thresholds for performance-based retraining triggers."""
    # Accuracy metrics
    min_accuracy: float = 0.55          # Minimum classification accuracy
    min_precision: float = 0.50         # Minimum precision
    min_recall: float = 0.50            # Minimum recall
    min_f1_score: float = 0.52          # Minimum F1 score
    
    # Trading metrics
    min_win_rate: float = 0.45          # Minimum win rate
    min_profit_factor: float = 1.2      # Minimum profit factor
    max_drawdown: float = 0.15          # Maximum allowed drawdown
    min_sharpe_ratio: float = 0.5       # Minimum Sharpe ratio
    min_sortino_ratio: float = 0.5      # Minimum Sortino ratio
    
    # Model confidence metrics
    min_avg_confidence: float = 0.55    # Average prediction confidence
    max_uncertainty: float = 0.30       # Maximum prediction uncertainty
    
    # Rolling window for metric calculation
    evaluation_window_trades: int = 50  # Number of trades for evaluation
    evaluation_window_days: int = 7     # Days for time-based evaluation
    
    # Grace period (don't trigger immediately after retraining)
    grace_period_hours: int = 24
    
    def check_thresholds(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check which thresholds are violated."""
        violations = {}
        
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
        
        for metric_name, (direction, threshold) in threshold_map.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if direction == 'min':
                    violations[metric_name] = value < threshold
                else:
                    violations[metric_name] = value > threshold
        
        return violations


@dataclass
class ScheduleConfig:
    """Configuration for time-based retraining schedule."""
    schedule_type: ScheduleType = ScheduleType.WEEKLY
    
    # For WEEKLY schedule
    training_days: List[int] = field(default_factory=lambda: [5])  # Saturday (5)
    training_time: time = field(default_factory=lambda: time(2, 0))  # 2 AM UTC
    
    # For CUSTOM schedule (interval-based)
    custom_interval_hours: int = 168  # 1 week
    
    # Timezone
    timezone: str = "UTC"
    
    # Blackout periods (don't retrain during these times)
    blackout_sessions: List[MarketSession] = field(default_factory=list)
    
    # Economic calendar awareness
    avoid_high_impact_news: bool = True
    news_buffer_hours: int = 4  # Hours before/after high-impact news


@dataclass
class ValidationConfig:
    """Configuration for model validation before deployment."""
    # Validation requirements
    min_validation_samples: int = 500
    validation_split: float = 0.2
    
    # Champion/Challenger comparison
    require_improvement: bool = True
    min_improvement_pct: float = 2.0    # New model must be X% better
    
    # Statistical significance
    require_statistical_significance: bool = True
    significance_level: float = 0.05
    
    # Stability checks
    require_stability_check: bool = True
    stability_folds: int = 5            # Cross-validation folds
    max_std_across_folds: float = 0.05  # Max std deviation across folds
    
    # Out-of-time validation
    use_oot_validation: bool = True
    oot_window_days: int = 14           # Use last N days as OOT set


@dataclass
class DataConfig:
    """Configuration for training data management."""
    # Data windows
    max_training_samples: int = 50000
    min_training_samples: int = 5000
    lookback_days: int = 90             # Use last N days of data
    
    # Feature selection
    feature_selection_enabled: bool = True
    max_features: int = 100
    min_feature_importance: float = 0.01
    
    # Data quality
    max_missing_ratio: float = 0.05
    max_outlier_ratio: float = 0.02
    
    # Class balancing
    balance_classes: bool = True
    balance_method: str = "smote"       # 'smote', 'undersample', 'oversample'


@dataclass
class ModelConfig:
    """Configuration for model training."""
    # Model type
    model_type: str = "lightgbm"        # 'lightgbm', 'xgboost', 'random_forest', 'ensemble'
    
    # Hyperparameter optimization
    tune_hyperparameters: bool = True
    tuning_method: str = "optuna"       # 'optuna', 'grid', 'random'
    tuning_trials: int = 50
    tuning_timeout_minutes: int = 30
    
    # Training settings
    early_stopping_rounds: int = 50
    n_estimators: int = 1000
    learning_rate: float = 0.05
    
    # Ensemble settings (if model_type == 'ensemble')
    ensemble_models: List[str] = field(default_factory=lambda: ['lightgbm', 'xgboost'])
    ensemble_weights: Optional[List[float]] = None


@dataclass
class VersioningConfig:
    """Configuration for model versioning and rollback."""
    # Storage
    model_dir: str = "models"
    max_versions_to_keep: int = 10
    
    # Naming
    version_prefix: str = "v"
    include_timestamp: bool = True
    include_metrics: bool = True
    
    # Rollback
    enable_auto_rollback: bool = True
    rollback_threshold_trades: int = 20  # Min trades before rollback decision
    rollback_degradation_pct: float = 10.0  # Rollback if X% worse than previous


@dataclass 
class NotificationConfig:
    """Configuration for retraining notifications."""
    # Notification channels
    enable_logging: bool = True
    enable_file_logging: bool = True
    log_file_path: str = "logs/retraining.log"
    
    # Notification levels
    notify_on_start: bool = True
    notify_on_complete: bool = True
    notify_on_failure: bool = True
    notify_on_validation_fail: bool = True
    notify_on_rollback: bool = True
    
    # Webhook (optional - for Slack, Discord, etc.)
    webhook_url: Optional[str] = None
    

@dataclass
class RetrainingConfig:
    """Master configuration for the retraining system."""
    # Component configurations
    performance: PerformanceThresholds = field(default_factory=PerformanceThresholds)
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
    
    # Cooldown (minimum time between retraining)
    cooldown_hours: int = 12
    
    # Forex-specific
    symbols: List[str] = field(default_factory=lambda: ["EURUSD"])
    timeframe_profile: str = "SWING"  # or "SCALP"
    
    # Monitoring
    monitor_interval_seconds: int = 300  # Check triggers every 5 minutes
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'enabled_triggers': [t.value for t in self.enabled_triggers],
            'cooldown_hours': self.cooldown_hours,
            'symbols': self.symbols,
            'timeframe_profile': self.timeframe_profile,
            'monitor_interval_seconds': self.monitor_interval_seconds,
            'schedule_type': self.schedule.schedule_type.value,
            'training_days': self.schedule.training_days,
        }
    
    @classmethod
    def for_scalping(cls) -> 'RetrainingConfig':
        """Create config optimized for scalping strategies."""
        config = cls()
        config.timeframe_profile = "SCALP"
        config.schedule.schedule_type = ScheduleType.DAILY
        config.schedule.training_days = list(range(7))  # Every day
        config.performance.evaluation_window_trades = 100
        config.performance.evaluation_window_days = 3
        config.data.lookback_days = 30
        config.cooldown_hours = 6
        return config
    
    @classmethod
    def for_swing_trading(cls) -> 'RetrainingConfig':
        """Create config optimized for swing trading strategies."""
        config = cls()
        config.timeframe_profile = "SWING"
        config.schedule.schedule_type = ScheduleType.WEEKLY
        config.performance.evaluation_window_trades = 30
        config.performance.evaluation_window_days = 14
        config.data.lookback_days = 180
        config.cooldown_hours = 48
        return config
    
    @classmethod
    def conservative(cls) -> 'RetrainingConfig':
        """Create conservative config with stricter validation."""
        config = cls()
        config.validation.min_improvement_pct = 5.0
        config.validation.require_statistical_significance = True
        config.validation.stability_folds = 10
        config.performance.grace_period_hours = 48
        config.cooldown_hours = 72
        return config
    
    @classmethod
    def aggressive(cls) -> 'RetrainingConfig':
        """Create aggressive config with faster adaptation."""
        config = cls()
        config.schedule.schedule_type = ScheduleType.DAILY
        config.validation.min_improvement_pct = 0.5
        config.validation.require_statistical_significance = False
        config.performance.grace_period_hours = 6
        config.cooldown_hours = 6
        config.enabled_triggers.add(RetrainingTrigger.REGIME_CHANGE)
        return config
