# pyForex Risk Management Integration Guide v2

## Overview

This package integrates a comprehensive 5-phase risk management system with pyForex-1.

### Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE RISK MANAGEMENT PIPELINE                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ENTRY PATH:                                                           │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Phase 1  │─►│ Phase 2   │─►│ Phase 3  │─►│ Phase 5  │─►│ Execute │ │
│  │ TCN      │  │ SL/TP     │  │ Meta     │  │ Capital  │  │         │ │
│  │ Predict  │  │ Sizing    │  │ Filter   │  │ Protect  │  │         │ │
│  └──────────┘  └───────────┘  └──────────┘  └──────────┘  └─────────┘ │
│                                                                        │
│  EXIT PATH (Active Position Management):                               │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  Open Position ──► Phase 4 (RL Exit Advisor) ──► Exit Decision   │ │
│  │                    HOLD / EXIT / TRAIL / PARTIAL                  │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  SAFETY LAYER (Always Active):                                         │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  Phase 5: Capital Protection                                      │ │
│  │  • Daily/Weekly/Monthly Loss Limits                               │ │
│  │  • Drawdown Protection (Graduated Size Reduction)                 │ │
│  │  • Losing Streak Cooldown                                         │ │
│  │  • Kill Switch (Emergency Halt)                                   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Installation

### 1. Extract Package

```bash
unzip pyforex_risk_integration_v2.zip -d /path/to/pyForex-1/
```

### 2. Copy Files

```bash
# New folders (copy entirely)
cp -r risk_management/ /path/to/pyForex-1/

# Updated files (backup and replace)
cp trading/decision_engine.py /path/to/pyForex-1/trading/
cp trading/live_trading_bot.py /path/to/pyForex-1/trading/
cp strategies/neural_hybrid.py /path/to/pyForex-1/strategies/
cp inference/predictor.py /path/to/pyForex-1/inference/
```

### 3. Install Dependencies

```bash
pip install torch lightgbm
```

## Quick Start

### Basic Usage (All 5 Phases)

```python
from strategies.neural_hybrid import create_strategy
from trading.live_trading_bot import create_bot

# Create strategy with all phases
strategy = create_strategy(
    profile='INTRADAY',
    symbol='EURUSD',
    data_provider=data_provider,
    executor=executor,
    tcn_weights='models/weights/tcn_best.pt',
    meta_model_path='models/weights/meta_model.joblib',
    exit_model_path='models/weights/exit_advisor.pt',  # Phase 4
    starting_balance=10000
)

# Or create full bot
bot = create_bot(
    symbol='EURUSD',
    profile='INTRADAY',
    data_provider=data_provider,
    executor=executor,
    starting_balance=10000,
    enable_exit_advisor=True,      # Phase 4
    enable_capital_protection=True, # Phase 5
    dry_run=True  # Paper trading
)

bot.run()
```

## Phase 4: RL Exit Optimization

### What It Does

The Exit Advisor uses PPO (Proximal Policy Optimization) to learn optimal exit timing. It monitors open positions and recommends:

| Action | Description |
|--------|-------------|
| HOLD | Keep position open |
| EXIT | Close entire position |
| TRAIL_STOP | Tighten stop loss |
| PARTIAL_25/50/75 | Close partial position |

### Training the Exit Advisor

```python
from risk_management import train_exit_optimizer

# Prepare price data (OHLCV DataFrame)
train_data = pd.read_csv('data/eurusd_train.csv')
eval_data = pd.read_csv('data/eurusd_eval.csv')

# Train
advisor, history = train_exit_optimizer(
    train_data=train_data,
    eval_data=eval_data,
    total_timesteps=500_000,
    checkpoint_dir='models/exit_optimizer'
)

# Model saved to: models/exit_optimizer/best_model.pt
```

### Using the Exit Advisor

```python
from risk_management import ExitAdvisor, Position

# Load trained model
advisor = ExitAdvisor.load('models/weights/exit_advisor.pt')

# Check a position
position = Position(
    direction=1,  # Long
    entry_price=1.1000,
    entry_time=0,
    initial_size=1.0,
    current_size=1.0,
    stop_loss=1.0950,
    take_profit=1.1100,
    initial_sl=1.0950,
    initial_tp=1.1100
)

# Get recommendation
rec = advisor.get_recommendation(position, market_data)
print(f"Action: {rec['action_name']}, Confidence: {rec['confidence']:.2%}")

# Simple interface
should_exit, reason = advisor.should_exit(position, market_data)
if should_exit:
    close_position()
```

### Integration with Strategy

```python
# Strategy automatically checks positions when exit_model_path is set
strategy = create_strategy(
    profile='INTRADAY',
    symbol='EURUSD',
    exit_model_path='models/weights/exit_advisor.pt',
    exit_confidence_threshold=0.6,  # Only exit if confidence > 60%
    ...
)

# Manual position check
market_data = data_provider.get_ohlcv('EURUSD', count=50)
recommendations = strategy.check_all_positions(market_data)

for rec in recommendations:
    if rec['action'] == 'EXIT':
        executor.close_position(rec['ticket'])
```

## Phase 5: Capital Protection

### Protection Rules

| Rule | Default | Action |
|------|---------|--------|
| Daily Loss Limit | 3% | Block new trades |
| Weekly Loss Limit | 6% | Block new trades |
| Monthly Loss Limit | 10% | Block new trades |
| Max Drawdown | 10% | Graduate size reduction |
| Losing Streak | 5 losses | 30-min cooldown |
| Kill Switch | Equity < 85% peak | Halt all trading |

### Protection Levels

```
NORMAL     → Full trading, no restrictions
    ↓
CAUTION    → Reduced position sizes
    ↓
WARNING    → Significantly reduced sizes
    ↓
CRITICAL   → No new trades allowed
    ↓
KILLED     → Complete trading halt
```

### Using Capital Protection

```python
from risk_management import CapitalProtector, ProtectionConfig

# Create protector
protector = CapitalProtector(ProtectionConfig(
    max_daily_loss_pct=3.0,
    max_drawdown_pct=10.0,
    max_consecutive_losses=5,
    losing_streak_cooldown_minutes=30
))

protector.initialize(balance=10000)

# Before each trade
check = protector.check_trade(
    proposed_size=0.5,
    account_balance=10000
)

if check['allowed']:
    execute_trade(size=check['adjusted_size'])
    # Note: adjusted_size may be smaller than proposed
else:
    print(f"Blocked: {check['reason']}")

# After each trade closes
protector.record_trade(pnl=150, is_win=True)

# Check status
state = protector.get_state()
print(f"Level: {state.level.value}, Multiplier: {state.size_multiplier}")
```

### Integration with Decision Engine

Capital protection is automatically integrated into the decision engine:

```python
from trading.decision_engine import EnhancedDecisionEngine, DecisionEngineConfig

config = DecisionEngineConfig(
    profile='INTRADAY',
    enable_capital_protection=True,
    max_daily_loss_pct=3.0,
    max_drawdown_pct=10.0
)

engine = EnhancedDecisionEngine(config)
engine.initialize(starting_balance=10000)

# Decision automatically includes protection checks
decision = engine.evaluate(predictions, entry_price, ...)

# decision.protection_level - Current protection level
# decision.protection_warnings - Any active warnings
# decision.size_adjusted_by_protection - True if size was reduced

# Record results for protection tracking
engine.record_trade_result(pnl=150, is_win=True)

# Check status
status = engine.get_protection_status()
```

### Using with Live Bot

```python
from trading.live_trading_bot import LiveTradingBot, BotConfig

config = BotConfig(
    symbol='EURUSD',
    profile='INTRADAY',
    
    # Phase 4: Exit Advisor
    enable_exit_advisor=True,
    exit_model_path='models/weights/exit_advisor.pt',
    exit_confidence_threshold=0.6,
    
    # Phase 5: Capital Protection
    enable_capital_protection=True,
    max_daily_loss_percent=3.0,
    max_weekly_loss_percent=6.0,
    max_drawdown_percent=10.0,
    max_consecutive_losses=5,
    cooldown_minutes=30
)

bot = LiveTradingBot(config, data_provider, executor)
bot.initialize(starting_balance=10000)

# Protection event callback
def on_protection_event(event_type, data):
    if event_type == 'kill_switch':
        send_alert("KILL SWITCH ACTIVATED!")
    elif event_type == 'trade_blocked':
        log_blocked_trade(data)

bot.on_protection_event = on_protection_event

# Run
bot.run()

# Check status
status = bot.get_status()
print(f"Protection Level: {status['protection']['level']}")
print(f"Exits by Advisor: {status['exits_by_advisor']}")
print(f"Protection Blocks: {status['protection_blocks']}")
```

## Training Pipeline

### Train Multi-Head TCN (Phase 1)

```python
from risk_management import create_tcn_for_profile, MultiHeadTCNTrainer, TrainingConfig

# Create model
model = create_tcn_for_profile('INTRADAY', input_features=64)

# Configure training
config = TrainingConfig(
    batch_size=32,
    learning_rate=1e-4,
    epochs=100
)

# Train
trainer = MultiHeadTCNTrainer(model, config)
history = trainer.train(train_loader, val_loader)
trainer.save('models/weights/tcn_best.pt')
```

### Train Meta-Labeling Model (Phase 3)

```python
from risk_management import TripleBarrierLabeler, MetaLabelingModel

# Generate labels
labeler = TripleBarrierLabeler()
labels = labeler.generate_labels(
    prices=close_prices,
    signals=signal_times,
    directions=signal_directions,
    sl_levels=stop_losses,
    tp_levels=take_profits
)

# Train meta-model
meta_model = MetaLabelingModel()
meta_model.train(meta_features, labels)
meta_model.save('models/weights/meta_model.joblib')
```

### Train Exit Advisor (Phase 4)

```python
from risk_management import train_exit_optimizer

advisor, history = train_exit_optimizer(
    train_data=price_df,
    eval_data=eval_df,
    total_timesteps=500_000,
    checkpoint_dir='models/exit_optimizer'
)
```

## File Reference

### New Folders
```
risk_management/
├── __init__.py                    # v2.0.0 with all phases
├── risk_manager.py
├── phase1_predictive/             # Multi-head TCN
├── phase2_risk_calc/              # SL/TP, Sizing, Rules
├── phase3_filtering/              # Triple Barrier, Meta-labeling
├── phase4_rl_exit/                # NEW: RL Exit Optimizer
│   ├── __init__.py
│   ├── environment.py             # Gym-compatible env
│   ├── ppo_agent.py               # PPO implementation
│   └── trainer.py                 # Training utilities
├── phase5_capital_protection/     # NEW: Capital Protection
│   ├── __init__.py
│   ├── protection_rules.py        # Core rules
│   └── integration.py             # Integration helpers
└── utils/
```

### Updated Files
```
trading/
├── decision_engine.py             # v2: +Phase 5 integration
└── live_trading_bot.py            # v2: +Phase 4 & 5

strategies/
└── neural_hybrid.py               # v2: +Phase 4 & 5 hooks

inference/
└── predictor.py                   # Risk-aware predictions
```

## Backward Compatibility

### Aliases Maintained
```python
# Old names still work
from trading.decision_engine import MTFDecisionEngine  # → EnhancedDecisionEngine
from inference.predictor import TCNPredictor           # → RiskAwareTCNPredictor
from inference.predictor import SimpleLSTMPredictor    # → RiskAwareTCNPredictor
```

### Legacy Prediction Conversion
```python
from trading.decision_engine import convert_legacy_predictions

# Convert old format to new
new_preds = convert_legacy_predictions(
    direction_probs=old_probs,
    volatility=0.001,
    entry_price=1.1000
)
```

## Configuration Reference

### BotConfig (Live Trading)

```python
BotConfig(
    # Trading
    symbol='EURUSD',
    profile='INTRADAY',
    
    # Model Paths
    tcn_weights='models/weights/tcn_best.pt',
    meta_model_path='models/weights/meta_model.joblib',
    exit_model_path='models/weights/exit_advisor.pt',
    
    # Phase 4: Exit Advisor
    enable_exit_advisor=True,
    exit_confidence_threshold=0.6,
    
    # Phase 5: Capital Protection
    enable_capital_protection=True,
    max_daily_loss_percent=3.0,
    max_weekly_loss_percent=6.0,
    max_monthly_loss_percent=10.0,
    max_drawdown_percent=10.0,
    max_consecutive_losses=5,
    cooldown_minutes=30,
    
    # Safety
    dry_run=True,
    max_open_trades=3
)
```

### DecisionEngineConfig

```python
DecisionEngineConfig(
    profile='INTRADAY',
    
    # Thresholds
    min_direction_confidence=0.55,
    min_meta_score=0.5,
    min_risk_reward=1.5,
    
    # Phase 5
    enable_capital_protection=True,
    max_daily_loss_pct=3.0,
    max_weekly_loss_pct=6.0,
    max_drawdown_pct=10.0
)
```

## Testing

```bash
# Run with dry_run mode first
python -c "
from trading.live_trading_bot import create_bot
from your_data_provider import DataProvider
from your_executor import Executor

bot = create_bot(
    symbol='EURUSD',
    profile='INTRADAY',
    data_provider=DataProvider(),
    executor=Executor(),
    dry_run=True,
    enable_exit_advisor=True,
    enable_capital_protection=True
)

# Run for limited time
import time
bot.start()
time.sleep(300)  # 5 minutes
bot.stop()

print(bot.get_status())
"
```

## Summary

| Phase | Component | Purpose |
|-------|-----------|---------|
| 1 | Multi-Head TCN | Direction, volatility, quantile predictions |
| 2 | Risk Calculations | SL/TP, position sizing, hard rules |
| 3 | Trade Filtering | Triple barrier labeling, meta-labeling |
| 4 | **Exit Advisor** | RL-based exit timing optimization |
| 5 | **Capital Protection** | Rule-based safety limits with kill switch |

**Total: ~6,500 lines of integrated risk management code**
