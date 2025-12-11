# ml/risk_retraining/risk_retraining_config.py
"""
Configuration for Risk Management Model Retraining.

Defines configurations for:
- TCN Risk Model (volatility + quantile heads)
- GBM Meta-Labeling Model
- RL Exit Optimizer (PPO)

Each model type has specific:
- Performance metrics and thresholds
- Drift detection parameters
- Retraining triggers
- Dependency chains
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Literal, Tuple
from enum import Enum, auto
from datetime import timedelta


# =============================================================================
# Model Types
# =============================================================================

class RiskModelType(Enum):
    """Types of risk management models."""
    TCN_RISK = auto()       # Multi-head TCN (direction + volatility + quantiles)
    GBM_META = auto()       # GBM meta-labeling for trade filtering
    RL_EXIT = auto()        # PPO/SAC for exit optimization
    

class RetrainingTriggerType(Enum):
    """Types of retraining triggers."""
    SCHEDULED = auto()          # Time-based (daily, weekly, etc.)
    PERFORMANCE = auto()        # Metrics dropped below threshold
    DRIFT = auto()              # Data distribution changed
    MANUAL = auto()             # User-initiated
    DEPENDENCY = auto()         # Upstream model was retrained
    REGIME_CHANGE = auto()      # Market regime shift detected


# =============================================================================
# Metric Configurations
# =============================================================================

@dataclass
class TCNRiskMetrics:
    """Performance metrics for TCN Risk Model."""
    
    # Direction metrics (existing)
    min_direction_accuracy: float = 0.52
    min_direction_f1: float = 0.50
    
    # Volatility metrics (Phase 1)
    max_volatility_mae: float = 0.002       # Max acceptable MAE for vol prediction
    max_volatility_mape: float = 0.25       # Max 25% MAPE
    min_volatility_correlation: float = 0.3  # Min correlation with realized vol
    
    # Quantile metrics (Phase 2)
    max_quantile_pinball_loss: float = 0.015    # Max average pinball loss
    min_quantile_coverage_q5: float = 0.03      # Q5 should cover ~5% (tolerance: 3-7%)
    max_quantile_coverage_q5: float = 0.07
    min_quantile_coverage_q95: float = 0.93     # Q95 should cover ~95% (tolerance: 93-97%)
    max_quantile_coverage_q95: float = 0.97
    max_quantile_crossing_rate: float = 0.02    # Max rate of quantile crossing violations
    
    # Trading performance
    min_risk_adjusted_sharpe: float = 0.8
    max_sl_hit_before_tp_rate: float = 0.55     # SL shouldn't hit too often before TP
    
    def to_dict(self) -> Dict:
        return {
            'direction_accuracy': self.min_direction_accuracy,
            'direction_f1': self.min_direction_f1,
            'volatility_mae': self.max_volatility_mae,
            'volatility_mape': self.max_volatility_mape,
            'volatility_correlation': self.min_volatility_correlation,
            'quantile_pinball_loss': self.max_quantile_pinball_loss,
            'quantile_coverage_q5_min': self.min_quantile_coverage_q5,
            'quantile_coverage_q5_max': self.max_quantile_coverage_q5,
            'quantile_coverage_q95_min': self.min_quantile_coverage_q95,
            'quantile_coverage_q95_max': self.max_quantile_coverage_q95,
            'quantile_crossing_rate': self.max_quantile_crossing_rate,
            'risk_adjusted_sharpe': self.min_risk_adjusted_sharpe,
            'sl_hit_before_tp_rate': self.max_sl_hit_before_tp_rate,
        }


@dataclass 
class GBMMetaMetrics:
    """Performance metrics for GBM Meta-Labeling Model."""
    
    # Core meta-labeling metrics
    min_precision: float = 0.65             # When we say "take trade", should be right 65%+
    min_recall: float = 0.40                # Don't filter too many good trades
    min_f1: float = 0.50
    
    # Trade filtering metrics
    min_filtered_win_rate: float = 0.55     # Win rate of trades that pass filter
    min_filter_improvement: float = 0.05    # Should improve win rate by at least 5%
    target_filter_rate: float = 0.35        # Target ~35% of trades filtered
    min_filter_rate: float = 0.15           # Filter at least 15%
    max_filter_rate: float = 0.60           # Don't filter more than 60%
    
    # Profit metrics
    min_filtered_profit_factor: float = 1.3
    
    def to_dict(self) -> Dict:
        return {
            'precision': self.min_precision,
            'recall': self.min_recall,
            'f1': self.min_f1,
            'filtered_win_rate': self.min_filtered_win_rate,
            'filter_improvement': self.min_filter_improvement,
            'target_filter_rate': self.target_filter_rate,
            'min_filter_rate': self.min_filter_rate,
            'max_filter_rate': self.max_filter_rate,
            'filtered_profit_factor': self.min_filtered_profit_factor,
        }


@dataclass
class RLExitMetrics:
    """Performance metrics for RL Exit Optimizer."""
    
    # RL-specific metrics
    min_sharpe_vs_fixed: float = 0.1        # Sharpe improvement over fixed exits
    min_average_reward: float = 0.0         # Average episode reward
    max_policy_entropy_drop: float = 0.5    # Entropy shouldn't collapse too much
    
    # Exit timing metrics
    max_premature_exit_rate: float = 0.25   # Exiting before trade reaches potential
    min_profitable_exit_ratio: float = 0.55 # Exits should be profitable more often
    
    # Drawdown metrics
    max_exit_drawdown: float = 0.05         # Max drawdown from optimal exit point
    
    def to_dict(self) -> Dict:
        return {
            'sharpe_vs_fixed': self.min_sharpe_vs_fixed,
            'average_reward': self.min_average_reward,
            'policy_entropy_drop': self.max_policy_entropy_drop,
            'premature_exit_rate': self.max_premature_exit_rate,
            'profitable_exit_ratio': self.min_profitable_exit_ratio,
            'exit_drawdown': self.max_exit_drawdown,
        }


# =============================================================================
# Drift Detection Configurations
# =============================================================================

@dataclass
class TCNRiskDriftConfig:
    """Drift detection config for TCN Risk Model."""
    
    enabled: bool = True
    check_interval_minutes: int = 60
    
    # Feature drift thresholds
    psi_threshold: float = 0.2              # Population Stability Index
    ks_threshold: float = 0.1               # KS test p-value
    js_threshold: float = 0.15              # Jensen-Shannon divergence
    
    # Volatility-specific drift
    volatility_regime_change_threshold: float = 0.5  # Realized vol changed by 50%+
    volatility_mean_shift_periods: int = 50          # Rolling window for detection
    
    # Quantile drift
    quantile_calibration_drift: float = 0.1  # Coverage drift threshold
    
    # Features to monitor
    monitor_features: List[str] = field(default_factory=lambda: [
        'atr_14', 'volatility_20', 'bb_width', 'returns_std',
        'volume_ma_ratio', 'spread_pips', 'rsi_14', 'adx_14'
    ])
    
    # Minimum samples for drift detection
    min_samples: int = 500
    reference_window_size: int = 5000


@dataclass
class GBMMetaDriftConfig:
    """Drift detection config for GBM Meta-Labeling."""
    
    enabled: bool = True
    check_interval_minutes: int = 120       # Less frequent than TCN
    
    # Feature drift (meta-features from TCN)
    psi_threshold: float = 0.25
    ks_threshold: float = 0.15
    
    # Prediction drift
    prediction_distribution_shift: float = 0.2  # GBM output distribution changed
    confidence_calibration_drift: float = 0.15
    
    # Minimum samples
    min_samples: int = 200
    reference_window_size: int = 2000


@dataclass
class RLExitDriftConfig:
    """Drift detection config for RL Exit Optimizer."""
    
    enabled: bool = True
    check_interval_minutes: int = 240       # Less frequent (RL is more stable)
    
    # State distribution drift
    state_psi_threshold: float = 0.3
    
    # Reward distribution drift
    reward_distribution_shift: float = 0.25
    
    # Policy drift (action distribution changed)
    action_distribution_shift: float = 0.2
    
    min_episodes: int = 100
    reference_episodes: int = 1000


# =============================================================================
# Schedule Configurations
# =============================================================================

@dataclass
class TCNRiskScheduleConfig:
    """Schedule config for TCN Risk Model retraining."""
    
    enabled: bool = True
    schedule_type: Literal['daily', 'weekly', 'monthly', 'custom'] = 'weekly'
    
    # For daily/weekly
    schedule_hour: int = 3          # 3 AM UTC
    schedule_day_of_week: int = 6   # Sunday (0=Monday, 6=Sunday)
    
    # For monthly
    schedule_day_of_month: int = 1
    
    # Custom interval
    custom_interval_hours: Optional[int] = None
    
    # Blackout periods (avoid retraining during volatile sessions)
    avoid_london_open: bool = True      # 7-9 UTC
    avoid_ny_open: bool = True          # 13-15 UTC
    avoid_tokyo_open: bool = True       # 23-01 UTC
    avoid_nfp_day: bool = True          # First Friday of month
    
    # Profile-based defaults
    @classmethod
    def for_scalp(cls) -> 'TCNRiskScheduleConfig':
        return cls(
            schedule_type='daily',
            schedule_hour=2,
            avoid_london_open=True,
            avoid_ny_open=True,
        )
    
    @classmethod
    def for_swing(cls) -> 'TCNRiskScheduleConfig':
        return cls(
            schedule_type='weekly',
            schedule_hour=3,
            schedule_day_of_week=6,
        )


@dataclass
class GBMMetaScheduleConfig:
    """Schedule config for GBM Meta-Labeling retraining."""
    
    enabled: bool = True
    schedule_type: Literal['weekly', 'monthly', 'after_tcn'] = 'after_tcn'
    
    # If scheduled independently
    schedule_hour: int = 4
    schedule_day_of_week: int = 6
    
    # Delay after TCN retraining (if after_tcn mode)
    delay_after_tcn_hours: int = 2


@dataclass
class RLExitScheduleConfig:
    """Schedule config for RL Exit Optimizer updates."""
    
    enabled: bool = True
    schedule_type: Literal['weekly', 'monthly', 'continuous'] = 'continuous'
    
    # For continuous learning
    update_every_n_episodes: int = 50
    
    # For scheduled full retraining
    schedule_hour: int = 5
    schedule_day_of_week: int = 0   # Monday


# =============================================================================
# Model Dependency Configuration
# =============================================================================

@dataclass
class ModelDependencyConfig:
    """Defines dependencies between models for cascading retraining."""
    
    # When TCN is retrained, what else needs retraining?
    tcn_triggers_gbm: bool = True
    tcn_triggers_rl: bool = False   # RL can adapt without full retraining
    
    # Delay between dependent retraining
    dependency_delay_minutes: int = 30
    
    # Validation requirements before cascading
    require_tcn_validation_before_gbm: bool = True
    require_gbm_validation_before_rl: bool = True
    
    # Max cascade depth (prevent infinite loops)
    max_cascade_depth: int = 3


# =============================================================================
# Main Configuration
# =============================================================================

@dataclass
class RiskRetrainingConfig:
    """
    Complete configuration for Risk Management model retraining.
    
    Combines all model-specific configs and global settings.
    """
    
    # Profile name
    profile_name: str = "INTRADAY"
    symbols: List[str] = field(default_factory=lambda: ["EURUSD"])
    
    # Model-specific metrics
    tcn_metrics: TCNRiskMetrics = field(default_factory=TCNRiskMetrics)
    gbm_metrics: GBMMetaMetrics = field(default_factory=GBMMetaMetrics)
    rl_metrics: RLExitMetrics = field(default_factory=RLExitMetrics)
    
    # Drift detection configs
    tcn_drift: TCNRiskDriftConfig = field(default_factory=TCNRiskDriftConfig)
    gbm_drift: GBMMetaDriftConfig = field(default_factory=GBMMetaDriftConfig)
    rl_drift: RLExitDriftConfig = field(default_factory=RLExitDriftConfig)
    
    # Schedule configs
    tcn_schedule: TCNRiskScheduleConfig = field(default_factory=TCNRiskScheduleConfig)
    gbm_schedule: GBMMetaScheduleConfig = field(default_factory=GBMMetaScheduleConfig)
    rl_schedule: RLExitScheduleConfig = field(default_factory=RLExitScheduleConfig)
    
    # Dependencies
    dependencies: ModelDependencyConfig = field(default_factory=ModelDependencyConfig)
    
    # Global settings
    cooldown_hours: int = 4                 # Min time between retraining same model
    max_retraining_attempts: int = 3        # Max retries on failure
    validation_holdout_ratio: float = 0.2   # Data held out for validation
    champion_challenger_enabled: bool = True
    auto_rollback_enabled: bool = True
    rollback_grace_period_hours: int = 24
    
    # Paths
    models_dir: str = "models/risk"
    data_dir: str = "data/processed"
    logs_dir: str = "logs/retraining"
    
    # Notifications
    notify_on_start: bool = True
    notify_on_complete: bool = True
    notify_on_failure: bool = True
    notify_on_rollback: bool = True
    
    @classmethod
    def for_scalp(cls, symbols: Optional[List[str]] = None) -> 'RiskRetrainingConfig':
        """Configuration optimized for scalping strategies."""
        return cls(
            profile_name="SCALP",
            symbols=symbols or ["EURUSD", "GBPUSD"],
            tcn_metrics=TCNRiskMetrics(
                max_volatility_mae=0.001,   # Tighter for scalping
                min_direction_accuracy=0.54,
            ),
            tcn_schedule=TCNRiskScheduleConfig.for_scalp(),
            cooldown_hours=2,               # More frequent updates OK
        )
    
    @classmethod
    def for_swing(cls, symbols: Optional[List[str]] = None) -> 'RiskRetrainingConfig':
        """Configuration optimized for swing trading."""
        return cls(
            profile_name="SWING",
            symbols=symbols or ["EURUSD"],
            tcn_metrics=TCNRiskMetrics(
                max_volatility_mae=0.003,   # More tolerance for swing
                min_direction_accuracy=0.50,
            ),
            tcn_schedule=TCNRiskScheduleConfig.for_swing(),
            cooldown_hours=12,              # Less frequent updates
        )
    
    @classmethod
    def for_intraday(cls, symbols: Optional[List[str]] = None) -> 'RiskRetrainingConfig':
        """Configuration optimized for intraday trading."""
        return cls(
            profile_name="INTRADAY",
            symbols=symbols or ["EURUSD", "GBPUSD", "USDJPY"],
            tcn_schedule=TCNRiskScheduleConfig(
                schedule_type='weekly',
                schedule_day_of_week=5,     # Saturday
            ),
            cooldown_hours=6,
        )
    
    def get_metrics_for_model(self, model_type: RiskModelType) -> Dict:
        """Get metrics dict for a specific model type."""
        if model_type == RiskModelType.TCN_RISK:
            return self.tcn_metrics.to_dict()
        elif model_type == RiskModelType.GBM_META:
            return self.gbm_metrics.to_dict()
        elif model_type == RiskModelType.RL_EXIT:
            return self.rl_metrics.to_dict()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_drift_config_for_model(self, model_type: RiskModelType):
        """Get drift config for a specific model type."""
        if model_type == RiskModelType.TCN_RISK:
            return self.tcn_drift
        elif model_type == RiskModelType.GBM_META:
            return self.gbm_drift
        elif model_type == RiskModelType.RL_EXIT:
            return self.rl_drift
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_schedule_config_for_model(self, model_type: RiskModelType):
        """Get schedule config for a specific model type."""
        if model_type == RiskModelType.TCN_RISK:
            return self.tcn_schedule
        elif model_type == RiskModelType.GBM_META:
            return self.gbm_schedule
        elif model_type == RiskModelType.RL_EXIT:
            return self.rl_schedule
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Presets
# =============================================================================

PROFILE_CONFIGS = {
    'SCALP': RiskRetrainingConfig.for_scalp,
    'INTRADAY': RiskRetrainingConfig.for_intraday,
    'SWING': RiskRetrainingConfig.for_swing,
}


def get_config_for_profile(
    profile: str,
    symbols: Optional[List[str]] = None
) -> RiskRetrainingConfig:
    """Get retraining config for a trading profile."""
    if profile not in PROFILE_CONFIGS:
        raise ValueError(f"Unknown profile: {profile}. Available: {list(PROFILE_CONFIGS.keys())}")
    return PROFILE_CONFIGS[profile](symbols)