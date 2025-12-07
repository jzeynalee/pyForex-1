# pyForex ML Retraining System

## Overview

The ML Retraining System provides automated model lifecycle management for the pyForex trading system. It monitors model performance and data quality in real-time, triggering retraining when necessary.

## Components

### 1. Drift Detector (`drift_detector.py`)

Detects when input data distributions change (data drift) or when the relationship between features and targets changes (concept drift).

```python
from ml import DriftDetector, DriftConfig, ConceptDriftDetector

# Basic usage
config = DriftConfig(
    ks_threshold=0.1,      # KS test p-value threshold
    psi_threshold=0.2,     # Population Stability Index threshold
    detection_window_size=200
)
detector = DriftDetector(config)

# Set baseline from training data
detector.set_reference(training_features_df)

# Monitor new data
result = detector.add_batch(new_features_df)
if result.drift_detected:
    print(f"Drift detected! Severity: {result.severity.name}")
    print(f"Drifted features: {result.drifted_features}")
```

**Key Features:**
- Statistical tests: KS test, PSI, Jensen-Shannon divergence
- Feature-level drift scoring
- Severity classification (LOW, MEDIUM, HIGH, CRITICAL)
- Drift trend analysis

### 2. Performance Monitor (`performance_monitor.py`)

Tracks trading performance metrics and triggers alerts when performance degrades.

```python
from ml import PerformanceMonitor, TradeRecord, MonitorConfig

# Setup
config = MonitorConfig(
    update_interval_trades=10,
    alert_cooldown_minutes=30
)
monitor = PerformanceMonitor(config)
monitor.set_initial_equity(10000)

# Record trades
trade = TradeRecord(
    trade_id="trade_001",
    entry_time=entry_dt,
    exit_time=exit_dt,
    symbol="EURUSD",
    direction=1,  # Long
    entry_price=1.1000,
    exit_price=1.1050,
    pnl=50.0
)
monitor.add_trade(trade)

# Check if retraining needed
needs_retrain, reason = monitor.needs_retraining()
```

**Tracked Metrics:**
- Win rate
- Profit factor
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Calmar ratio
- Signal accuracy

### 3. Model Manager (`model_manager.py`)

Handles model versioning, storage, validation, and deployment.

```python
from ml import ModelManager, ManagerConfig

config = ManagerConfig(
    models_dir="./models",
    max_versions=10,
    validation_metric="sharpe_ratio",
    min_improvement_pct=5.0
)
manager = ModelManager(config)

# Save a trained model
model_id = manager.save_model(
    model=trained_model,
    profile_name="SWING",
    version="v1.0",
    model_type="gradient_boosting",
    hyperparameters={...},
    feature_names=[...],
    training_data=X_train,
    training_start=start_date,
    training_end=end_date,
    validation_metrics={"accuracy": 0.65}
)

# Validate against current model
result = manager.validate_model(
    candidate_id=model_id,
    validation_data=X_val,
    validation_targets=y_val,
    predict_fn=lambda m, x: m.predict(x),
    metric_calculators={...}
)

# Activate if validation passed
if result.passed:
    manager.activate_model(model_id)

# Rollback if needed
manager.rollback("SWING", steps=1)
```

### 4. Retraining Scheduler (`retraining_scheduler.py`)

Orchestrates the entire retraining process with multiple trigger types.

```python
from ml import RetrainingScheduler, RetrainingConfig, ScheduleConfig

config = RetrainingConfig(
    profile_name="SWING",
    models_dir="./models",
    schedule=ScheduleConfig(
        enabled=True,
        schedule_type="weekly",
        schedule_day=6,  # Sunday
        schedule_hour=2,
        avoid_market_hours=True
    ),
    performance_trigger_enabled=True,
    drift_trigger_enabled=True
)

scheduler = RetrainingScheduler(config)

# Set custom components
scheduler.set_data_preparer(my_data_preparer)
scheduler.set_model_trainer(my_trainer)

# Add callbacks
scheduler.add_completion_callback(lambda e: send_notification(e))

# Start background monitoring
scheduler.start_monitoring()

# Or trigger manually
scheduler.trigger_manual_retraining("Quarterly review")
```

## Trigger Types

| Trigger | Description | When to Use |
|---------|-------------|-------------|
| **SCHEDULED** | Time-based (daily/weekly/monthly) | Regular model refresh |
| **PERFORMANCE** | Metrics fall below thresholds | Model degradation |
| **DRIFT** | Data distribution changes | Market regime change |
| **MANUAL** | User-initiated | Ad-hoc updates |
| **REGIME_CHANGE** | Market structure changes | Volatility shifts |

## Configuration Guide

### Schedule Configuration

```python
schedule = ScheduleConfig(
    enabled=True,
    schedule_type="weekly",     # daily, weekly, monthly
    schedule_day=6,             # 0=Mon, 6=Sun
    schedule_hour=2,            # UTC hour
    avoid_market_hours=True,    # Only retrain when market closed
    preferred_session="weekend" # Best time for retraining
)
```

### Profile-Specific Settings

**SCALP Profile:**
- Daily retraining schedule
- 100+ trades before performance trigger
- 30-day lookback for training data
- Higher drift sensitivity

**SWING Profile:**
- Weekly retraining schedule
- 30+ trades before performance trigger
- 90-day lookback for training data
- Standard drift sensitivity

## Integration Example

```python
class MyTradingBot:
    def __init__(self):
        # Setup retraining
        self.scheduler = create_scheduler_for_profile("SWING")
        self.scheduler.set_data_preparer(MyDataPreparer())
        self.scheduler.set_model_trainer(MyTrainer())
        
    def start(self):
        # Initialize and start monitoring
        self.scheduler.start_monitoring()
        
    def on_trade_closed(self, trade):
        # Record trade - may trigger retraining
        event = self.scheduler.record_trade(trade)
        if event:
            print(f"Retraining triggered: {event.trigger_reason}")
            
    def on_new_features(self, features):
        # Check for drift - may trigger retraining
        event = self.scheduler.record_features(features)
```

## Custom DataPreparer

Implement the `DataPreparer` base class:

```python
class MyDataPreparer(DataPreparer):
    def prepare_training_data(
        self,
        profile_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[X_train, y_train, X_val, y_val]:
        # Fetch historical data
        # Build features
        # Split train/validation
        return X_train, y_train, X_val, y_val
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names
```

## Custom ModelTrainer

Implement the `ModelTrainer` base class:

```python
class MyTrainer(ModelTrainer):
    def train(
        self,
        X_train, y_train,
        hyperparameters: Dict
    ) -> Tuple[model, metrics]:
        # Train your model
        # Calculate metrics
        return model, {"accuracy": 0.7, "sharpe": 1.2}
    
    def get_default_hyperparameters(self) -> Dict:
        return {"n_estimators": 100, "max_depth": 5}
```

## Safety Features

1. **Validation Gate**: New models must outperform current model
2. **Auto-Rollback**: Automatic rollback if performance drops
3. **Consecutive Failure Limit**: Stops after N failures
4. **Market Hours Awareness**: Avoid retraining during live trading
5. **News Blackout**: Optional pause around high-impact events

## Monitoring Dashboard

Get comprehensive status:

```python
status = scheduler.get_status()
# Returns:
# {
#   'status': 'idle',
#   'profile': 'SWING',
#   'active_model': 'SWING_v1_20250607_...',
#   'consecutive_failures': 0,
#   'drift_trend': {'trend': 'stable', 'score': 0.12},
#   'performance_summary': {...}
# }
```

## Best Practices

1. **Start Conservative**: Begin with weekly scheduled retraining
2. **Monitor First**: Run in monitoring-only mode before enabling auto-retrain
3. **Set Appropriate Thresholds**: Tune based on your strategy's characteristics
4. **Keep History**: Maintain model versions for analysis and rollback
5. **Test Thoroughly**: Validate on out-of-sample data before deployment

## File Structure

```
ml/
├── __init__.py              # Module exports
├── drift_detector.py        # Data/concept drift detection
├── performance_monitor.py   # Trading performance tracking
├── model_manager.py         # Model versioning & deployment
└── retraining_scheduler.py  # Main orchestration

examples/
└── retraining_example.py    # Full integration example
```

## Dependencies

- numpy
- pandas
- scipy (for statistical tests)
- scikit-learn (optional, for model training)

## Changelog

### v1.0.0
- Initial release
- Drift detection (KS, PSI, JS divergence)
- Performance monitoring with alerts
- Model versioning and validation
- Scheduled and triggered retraining
- Integration with MTF system
