# pyForex-1 Risk Management Integration Guide

## Overview

This update integrates the comprehensive ML-based risk management system (Phases 1-3) with the existing pyForex-1 codebase.

## What's New

### Risk Management System (Phases 1-3)

| Phase | Component | Purpose |
|-------|-----------|---------|
| **Phase 1** | Multi-Head TCN | Direction, Volatility, Quantile predictions |
| **Phase 2** | SL/TP Calculator | ML-based stop-loss and take-profit |
| **Phase 2** | Position Sizing | Risk-adjusted position sizes |
| **Phase 2** | Hard Rules | Leverage, exposure, session filters |
| **Phase 3** | Triple Barrier | Supervised learning labels |
| **Phase 3** | Meta-Labeling | GBM trade filter |

### Updated Components

| File | Status | Changes |
|------|--------|---------|
| `risk_management/` | **NEW** | Complete risk management package |
| `trading/decision_engine.py` | **REPLACED** | Full Phase 1-3 integration |
| `inference/predictor.py` | **REPLACED** | Risk-aware predictions |
| `strategies/neural_hybrid.py` | **REPLACED** | ML-based SL/TP and sizing |
| `trading/live_trading_bot.py` | **REPLACED** | Production bot with risk mgmt |

---

## Installation

### 1. Copy Risk Management Package

Copy the entire `risk_management/` directory to your pyForex-1 root:

```
pyForex-1/
├── risk_management/           # NEW
│   ├── __init__.py
│   ├── risk_manager.py
│   ├── phase1_predictive/
│   │   ├── __init__.py
│   │   ├── tcn_backbone.py
│   │   └── training.py
│   ├── phase2_risk_calc/
│   │   ├── __init__.py
│   │   ├── sl_tp_calculator.py
│   │   ├── position_sizing.py
│   │   └── hard_rules.py
│   ├── phase3_filtering/
│   │   ├── __init__.py
│   │   ├── triple_barrier.py
│   │   └── meta_labeling.py
│   └── utils/
│       ├── __init__.py
│       └── indicators.py
├── trading/
│   ├── decision_engine.py     # REPLACE
│   └── live_trading_bot.py    # REPLACE
├── inference/
│   └── predictor.py           # REPLACE
├── strategies/
│   └── neural_hybrid.py       # REPLACE
└── ...
```

### 2. Update Requirements

Add to `requirements.txt`:

```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
lightgbm>=3.3.0  # For meta-labeling
```

### 3. Replace Files

Replace these existing files with the new versions:
- `trading/decision_engine.py`
- `trading/live_trading_bot.py`
- `inference/predictor.py`
- `strategies/neural_hybrid.py`

---

## Usage

### Basic Prediction with Risk Parameters

```python
from inference.predictor import RiskAwareTCNPredictor

# Create predictor
predictor = RiskAwareTCNPredictor(
    config=PredictorConfig(profile='INTRADAY'),
    weights_path='models/weights/tcn_best.pt'
)

# Get prediction with risk parameters
result = predictor.predict(features)

print(f"Direction: {result.signal_name} ({result.confidence:.1%})")
print(f"Volatility: {result.volatility:.6f}")
print(f"Quantiles: {result.quantiles}")
```

### Enhanced Decision Engine

```python
from trading.decision_engine import EnhancedDecisionEngine, DecisionEngineConfig

# Configure engine
config = DecisionEngineConfig(
    profile='INTRADAY',
    min_direction_confidence=0.55,
    base_risk_percent=1.0,
    min_risk_reward=1.5
)

engine = EnhancedDecisionEngine(config)

# Evaluate trade
decision = engine.evaluate(
    predictions=predictor.to_dict(result),
    entry_price=1.1234,
    pair='EURUSD',
    account_balance=10000,
    market_data=df
)

if decision.should_trade:
    print(f"Direction: {decision.direction}")
    print(f"Position Size: {decision.position_size} lots")
    print(f"SL: {decision.stop_loss:.5f} ({decision.sl_pips:.1f} pips)")
    print(f"TP: {decision.take_profit:.5f} ({decision.tp_pips:.1f} pips)")
    print(f"Risk: {decision.risk_percent:.1f}%")
    print(f"R:R: {decision.risk_reward_ratio:.2f}")
else:
    print(f"Trade rejected: {decision.rejection_reasons}")
```

### Full Strategy

```python
from strategies.neural_hybrid import NeuralHybridStrategy, StrategyConfig

config = StrategyConfig(
    profile='INTRADAY',
    symbol='EURUSD',
    tcn_weights='models/weights/tcn_best.pt',
    base_risk_percent=1.0,
    min_confidence=0.55
)

strategy = NeuralHybridStrategy(config, data_provider, executor)
strategy.initialize()

# Evaluate market
decision = strategy.evaluate()

if decision and decision.should_trade:
    order = strategy.create_order(decision)
    strategy.execute(order)
```

### Live Trading Bot

```python
from trading.live_trading_bot import LiveTradingBot, BotConfig

config = BotConfig(
    symbol='EURUSD',
    profile='INTRADAY',
    tcn_weights='models/weights/tcn_best.pt',
    base_risk_percent=1.0,
    max_daily_loss_percent=3.0,
    dry_run=True  # Paper trading first!
)

bot = LiveTradingBot(config, data_provider, executor)
bot.initialize()
bot.run()  # Start trading
```

---

## Training the Risk Management Model

### Train Multi-Head TCN

```python
from risk_management import (
    create_tcn_for_profile,
    MultiHeadTCNTrainer,
    TrainingConfig,
    RiskDataset
)

# Create model
model = create_tcn_for_profile('INTRADAY', input_features=64)

# Prepare data
dataset = RiskDataset(
    features=X,
    direction_labels=y_direction,
    volatility_labels=y_volatility,
    price_move_labels=y_price_move,
    sequence_length=60
)

# Train
config = TrainingConfig(
    batch_size=64,
    learning_rate=1e-3,
    num_epochs=100
)

trainer = MultiHeadTCNTrainer(model, config)
history = trainer.train(train_loader, val_loader)
trainer.save_checkpoint('models/weights/tcn_risk_best.pt')
```

### Train Meta-Labeling Model

```python
from risk_management import (
    TripleBarrierLabeler,
    MetaLabelingModel,
    MetaLabelingConfig
)

# Generate triple barrier labels
labeler = TripleBarrierLabeler()
labels, detailed = labeler.generate_labels(
    prices=df,
    entry_signals=signals,
    directions=directions,
    sl_levels=sl_levels,
    tp_levels=tp_levels
)

# Create meta-labels
meta_model = MetaLabelingModel()
meta_labels = meta_model.create_meta_labels(
    primary_directions=directions,
    barrier_outcomes=labels
)

# Extract features and train
meta_features = meta_model.feature_extractor.extract_features(
    primary_predictions=predictions,
    market_data=df
)

metrics = meta_model.train(meta_features, meta_labels)
meta_model.save('models/weights/meta_model.joblib')
```

---

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Market Data → Feature Engineering → RiskAwareTCNPredictor      │
│                                              ↓                  │
│                              ┌───────────────┴───────────────┐  │
│                              │      Multi-Head Outputs       │  │
│                              │  • Direction: P(Bear/Side/Bull)│  │
│                              │  • Volatility: σ predicted     │  │
│                              │  • Quantiles: [Q5..Q95]        │  │
│                              └───────────────┬───────────────┘  │
│                                              ↓                  │
├─────────────────────────────────────────────────────────────────┤
│                      ENHANCED DECISION ENGINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌────────────┐  │
│  │ Phase 2: SL/TP  │    │ Phase 2: Sizing │    │Phase 2:    │  │
│  │ Calculator      │    │ Calculator      │    │Hard Rules  │  │
│  │                 │    │                 │    │            │  │
│  │ SL = f(Q5/Q95,  │    │ Size = f(σ,     │    │• Leverage  │  │
│  │       σ, regime)│    │   confidence,   │    │• Exposure  │  │
│  │ TP = f(Q25/Q75, │    │   account_risk) │    │• Sessions  │  │
│  │       σ, conf)  │    │                 │    │• Spread    │  │
│  └────────┬────────┘    └────────┬────────┘    └─────┬──────┘  │
│           └──────────────────────┼──────────────────┘          │
│                                  ↓                              │
│                    ┌─────────────────────────┐                  │
│                    │ Phase 3: Meta-Labeling  │                  │
│                    │ Filter                  │                  │
│                    │                         │                  │
│                    │ P(correct) >= threshold │                  │
│                    └───────────┬─────────────┘                  │
│                                ↓                                │
│                    ┌─────────────────────────┐                  │
│                    │    TRADE DECISION       │                  │
│                    │                         │                  │
│                    │ • should_trade: bool    │                  │
│                    │ • direction: BUY/SELL   │                  │
│                    │ • position_size: lots   │                  │
│                    │ • stop_loss: price      │                  │
│                    │ • take_profit: price    │                  │
│                    │ • risk_percent: %       │                  │
│                    │ • risk_reward: ratio    │                  │
│                    └─────────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Migration Notes

### Breaking Changes

1. **Prediction Format**: Predictions now include `volatility` and `quantiles` in addition to `direction_probs`

2. **Decision Engine**: `MTFDecisionEngine` renamed to `EnhancedDecisionEngine` (alias kept for compatibility)

3. **Position Sizing**: No longer hardcoded - calculated from predictions

4. **SL/TP**: No longer ATR-only - uses quantile regression

### Backward Compatibility

The following aliases are maintained:
- `SimpleLSTMPredictor` → `RiskAwareTCNPredictor`
- `TCNPredictor` → `RiskAwareTCNPredictor`
- `MTFDecisionEngine` → `EnhancedDecisionEngine`

### Legacy Prediction Conversion

If using old prediction format:

```python
from trading.decision_engine import convert_legacy_predictions

# Old format: [P(BUY), P(SELL), P(HOLD)]
legacy_probs = model.predict(features)

# Convert to new format: [P(BEAR), P(SIDEWAYS), P(BULL)]
new_predictions = convert_legacy_predictions(legacy_probs)

# Use with decision engine
decision = engine.evaluate(predictions=new_predictions, ...)
```

---

## File Checksums

```
risk_management/__init__.py         - Package exports
risk_management/risk_manager.py     - Unified RiskManager class
risk_management/phase1_predictive/  - Multi-head TCN (Direction/Vol/Quantile)
risk_management/phase2_risk_calc/   - SL/TP, Position Sizing, Hard Rules
risk_management/phase3_filtering/   - Triple Barrier, Meta-Labeling
trading/decision_engine.py          - Enhanced decision engine
trading/live_trading_bot.py         - Production trading bot
inference/predictor.py              - Risk-aware predictors
strategies/neural_hybrid.py         - Full strategy with risk mgmt
```

---

## Support

If you encounter issues during integration:
1. Check import paths match your project structure
2. Verify all dependencies are installed
3. Test with `dry_run=True` before live trading
4. Review logs for rejection reasons
