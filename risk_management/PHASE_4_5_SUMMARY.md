# Phase 4 & 5 Implementation Summary

## Phase 4: RL Exit Optimization

### Purpose
Narrow-scope reinforcement learning agent for optimizing **exit timing only**.
NOT responsible for entry decisions, position sizing, or SL/TP levels.

### Architecture

```
State Space (22 features):
├── Position Features (8)
│   ├── Direction (long/short)
│   ├── Unrealized PnL %
│   ├── Risk-Reward Position (-1=SL, 0=entry, +1=TP)
│   ├── Distance to SL
│   ├── Distance to TP
│   ├── Remaining size %
│   ├── Time in trade %
│   └── SL tightened amount
├── Market Features (10)
│   ├── Current return
│   ├── Volatility
│   ├── Momentum
│   ├── RSI
│   ├── Trend strength
│   ├── Bollinger position
│   ├── ATR normalized
│   └── Recent price action (3)
└── Time Features (4)
    ├── Hour (cyclical sin/cos)
    └── Day (cyclical sin/cos)

Action Space (6 actions):
├── HOLD       - Keep position open
├── EXIT       - Close entire position
├── TRAIL_STOP - Tighten stop loss
├── PARTIAL_25 - Close 25% of position
├── PARTIAL_50 - Close 50% of position
└── PARTIAL_75 - Close 75% of position

Reward:
├── Base: PnL percentage
├── Bonus: +0.3 for hitting take profit
├── Penalty: -0.5 for hitting stop loss
├── Penalty: -0.1 for premature profitable exits
└── Sharpe adjustment for risk-adjusted returns
```

### Files

| File | Lines | Description |
|------|-------|-------------|
| `environment.py` | ~450 | Gym-compatible exit decision environment |
| `ppo_agent.py` | ~400 | Actor-critic PPO agent with GAE |
| `trainer.py` | ~350 | Training pipeline with curriculum learning |
| `__init__.py` | ~60 | Module exports |

### Usage

```python
# Training
from risk_management.phase4_rl_exit import train_exit_optimizer

advisor, history = train_exit_optimizer(
    train_data=price_df,
    eval_data=eval_df,
    total_timesteps=500_000
)

# Inference
from risk_management.phase4_rl_exit import ExitAdvisor, Position

advisor = ExitAdvisor.load('best_model.pt')

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

recommendation = advisor.get_recommendation(position, market_data)
# {'action': 0, 'action_name': 'HOLD', 'confidence': 0.73, 'value': 0.15}

should_exit, reason = advisor.should_exit(position, market_data)
```

### Key Features

- **Curriculum Learning**: Starts with easy conditions, progressively harder
- **GAE (Generalized Advantage Estimation)**: Reduced variance in policy gradients
- **PPO Clipping**: Stable training with trust region
- **Early Stopping**: Based on KL divergence or validation performance

---

## Phase 5: Capital Protection

### Purpose
Rule-based safety overlays that **override all other decisions**.
These are deterministic hard limits, NOT learned.

### Protection Rules

| Rule | Trigger | Action |
|------|---------|--------|
| Daily Loss Limit | Loss > 3% of balance | Block new trades |
| Weekly Loss Limit | Loss > 6% of balance | Block new trades |
| Monthly Loss Limit | Loss > 10% of balance | Block new trades |
| Drawdown Protection | DD > 5% | Reduce position sizes |
| Max Drawdown | DD > 10% | Significantly reduced sizes |
| Losing Streak | 5 consecutive losses | 30-minute cooldown |
| Low Win Rate | Win rate < 35% | Reduce sizes by 25% |
| Equity Kill Switch | Equity < 85% of peak | All trading stopped |

### Protection Levels

```
NORMAL     → Full trading, no restrictions
    ↓
CAUTION    → Reduced position sizes (warning thresholds)
    ↓
WARNING    → Significantly reduced sizes (serious concern)
    ↓
CRITICAL   → No new trades, manage existing only
    ↓
KILLED     → Complete trading halt (kill switch)
```

### Files

| File | Lines | Description |
|------|-------|-------------|
| `protection_rules.py` | ~500 | Core protection logic |
| `integration.py` | ~300 | Integration with trading system |
| `__init__.py` | ~70 | Module exports |

### Usage

```python
# Basic Usage
from risk_management.phase5_capital_protection import (
    CapitalProtector, ProtectionConfig
)

protector = CapitalProtector(ProtectionConfig(
    max_daily_loss_pct=3.0,
    max_drawdown_pct=10.0,
    max_consecutive_losses=5
))

protector.initialize(balance=10000)

# Before each trade
check = protector.check_trade(
    proposed_size=0.5,
    account_balance=10000,
    current_exposure=2.5
)

if check['allowed']:
    execute_trade(size=check['adjusted_size'])
else:
    print(f"Blocked: {check['reason']}")

# After each trade
protector.record_trade(pnl=150.0, is_win=True)

# Using ProtectionManager (simpler)
from risk_management.phase5_capital_protection import ProtectionManager

manager = ProtectionManager()
manager.start_session(balance=10000)

check = manager.pre_trade_check(size=0.5, balance=10000)
manager.post_trade_update(pnl=-50, is_win=False)

status = manager.get_status()
# {
#     'protection_level': 'normal',
#     'size_multiplier': 1.0,
#     'metrics': {
#         'drawdown_pct': 0.5,
#         'consecutive_losses': 1,
#         'win_rate': 0.55
#     }
# }

# Context Manager
from risk_management.phase5_capital_protection import ProtectedTradingSession

with ProtectedTradingSession(balance=10000) as session:
    while session.can_trade():
        result = session.get_adjusted_size(proposed_size=0.5)
        if result.allowed:
            # Execute trade with result.adjusted_size
            session.record_result(pnl=100, is_win=True)

# Decorator-based Protection
from risk_management.phase5_capital_protection import TradingGuard

guard = TradingGuard()
guard.initialize(balance=10000)

@guard.protect_entry
def open_trade(size, **kwargs):
    # kwargs['adjusted_size'] contains protection-adjusted size
    return executor.open(size=kwargs['adjusted_size'])

@guard.protect_exit
def close_trade(ticket, **kwargs):
    result = executor.close(ticket)
    return {'success': True, 'pnl': result.pnl}
```

### Integration with RiskManager

```python
from risk_management import RiskManager
from risk_management.phase5_capital_protection import integrate_with_risk_manager

manager = RiskManager.create_for_profile('INTRADAY')
manager = integrate_with_risk_manager(manager)

# Now all trades are automatically protected
decision = manager.evaluate_trade_opportunity(...)
# Decision will be blocked/adjusted if protection rules triggered

# Record trade results
manager.record_trade_result(pnl=150, is_win=True)

# Check protection status
status = manager.get_protection_status()
```

---

## Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPLETE RISK MANAGEMENT                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ENTRY DECISION PATH:                                           │
│  ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌────────────┐  │
│  │ Phase 1  │──►│ Phase 2   │──►│ Phase 3  │──►│ Phase 5    │  │
│  │ TCN      │   │ SL/TP     │   │ Meta-    │   │ Capital    │  │
│  │ Predict  │   │ Sizing    │   │ Label    │   │ Protection │  │
│  └──────────┘   └───────────┘   └──────────┘   └────────────┘  │
│       │              │               │               │          │
│       ▼              ▼               ▼               ▼          │
│   Direction      Risk Params    Filter Signal   Final Check     │
│   Vol/Quant      Position Size  Quality Score   Allow/Block     │
│                                                                 │
│  EXIT DECISION PATH:                                            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  Market Data ──► Phase 4 (RL Exit) ──► Action            │  │
│  │       │                                   │               │  │
│  │       │          HOLD / EXIT / TRAIL     │               │  │
│  │       │          PARTIAL_25/50/75        │               │  │
│  │       │                                   │               │  │
│  │       └──────────────────────────────────┘               │  │
│  │                                                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Summary

### Phase 4: ~1,260 lines
```
phase4_rl_exit/
├── __init__.py           (60 lines)
├── environment.py        (450 lines)
├── ppo_agent.py          (400 lines)
└── trainer.py            (350 lines)
```

### Phase 5: ~870 lines
```
phase5_capital_protection/
├── __init__.py           (70 lines)
├── protection_rules.py   (500 lines)
└── integration.py        (300 lines)
```

### Total All Phases: ~5,700 lines
```
Phase 1: ~1,050 lines (TCN + training)
Phase 2: ~1,550 lines (SL/TP + sizing + rules)
Phase 3: ~1,000 lines (triple barrier + meta-labeling)
Phase 4: ~1,260 lines (RL exit)
Phase 5: ~870 lines (capital protection)
```
