# Forex Risk Management System - Architecture Review

**Date:** December 10, 2025  
**Project:** pyForex Risk Management Extension  
**Status:** Architecture Approved, Implementation In Progress

---

## Executive Summary

This document captures the architectural review and redesign of a comprehensive Forex risk management system. The goal is to extend the existing Vision Models (YOLO/ViT) + TCN architecture to cover all aspects of risk management: position sizing, leverage, SL/TP, trade invalidation, regime adaptation, exposure controls, and capital preservation.

---

## Original Proposed Architecture

The initial proposal suggested the following pipeline:

```
Vision Models → Feature Extraction → Triple Barrier → GBM → Quantile Regression → Meta-Labeling → RL (Adaptive Risk Layer)
```

### Components in Original Plan

| Component | Purpose |
|-----------|---------|
| Vision Models | YOLO/ViT for chart pattern recognition |
| Feature Extraction | Convert vision outputs to features |
| Triple Barrier | Label trades as WIN/LOSS/TIMEOUT |
| GBM | Primary directional prediction |
| Quantile Regression | Price distribution modeling |
| Meta-Labeling | Trade filtering |
| RL | Adaptive control of all risk parameters |

---

## Critical Issues Identified

### Issue 1: Circular Dependency

**Problem:** Triple Barrier requires SL/TP levels to label outcomes, but SL/TP calculation was positioned AFTER Triple Barrier in the pipeline (coming from Quantile Regression).

```
❌ Triple Barrier (needs SL/TP) → ... → Quantile Regression (produces SL/TP)
```

**Solution:** Calculate SL/TP BEFORE Triple Barrier labeling.

### Issue 2: GBM Misplacement

**Problem:** GBM was positioned as the primary directional predictor, but the existing codebase already has a working TCN backbone for direction prediction.

**Solution:** Use TCN as primary predictor, repurpose GBM for meta-labeling (trade filtering).

### Issue 3: RL Scope Too Broad

**Problem:** Single RL agent controlling:
- Position sizing
- Leverage
- SL/TP placement
- Trade avoidance
- Exposure management
- Capital preservation

This creates:
- Massive action space (combinatorial explosion)
- Credit assignment problems (which decision caused the outcome?)
- Enormous training data requirements
- Difficult debugging and interpretability

**Solution:** Narrow RL scope to exit timing only. Use formulas and rules for other risk parameters.

### Issue 4: Vision Integration Unclear

**Problem:** How YOLO/ViT features flow into risk calculations was not specified.

**Solution:** Add optional vision feature fusion to TCN backbone.

---

## Revised Architecture

### 5-Phase Implementation Plan

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: PREDICTIVE FOUNDATION               │
│  ┌─────────────────┐                                            │
│  │ OHLCV Data      │──┐                                         │
│  └─────────────────┘  │    ┌──────────────┐                     │
│                       ├───►│ TCN Backbone │                     │
│  ┌─────────────────┐  │    └──────┬───────┘                     │
│  │ Vision Features │──┘           │                             │
│  │ (YOLO/ViT)      │              ▼                             │
│  └─────────────────┘    ┌─────────┴─────────┐                   │
│                         │   Three Heads:     │                   │
│                         │ • Direction        │                   │
│                         │ • Volatility       │                   │
│                         │ • Quantiles        │                   │
│                         └─────────┬─────────┘                   │
└───────────────────────────────────┼─────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2: RISK CALCULATIONS                   │
│                                                                 │
│  Deterministic formulas using Phase 1 outputs:                  │
│                                                                 │
│  • SL = entry ± f(Q5/Q95, σ, regime)                           │
│  • TP = entry ± f(Q25/Q75, σ, direction_confidence)            │
│  • Position Size = f(σ, account_risk%, confidence)             │
│                                                                 │
│  Hard Rules:                                                    │
│  • Max leverage per regime                                      │
│  • Max exposure per pair/correlation group                      │
│  • Session filters (spread, liquidity thresholds)              │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 3: TRADE FILTERING                     │
│                                                                 │
│  Triple Barrier Labeling:                                       │
│  • Uses SL/TP from Phase 2                                      │
│  • Labels historical trades as WIN/LOSS/TIMEOUT                 │
│                                                                 │
│  Meta-Labeling GBM:                                             │
│  • Input: primary predictions + market features                 │
│  • Output: P(primary_prediction_correct)                        │
│  • Action: filter trades where P < threshold                    │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 4: ADAPTIVE EXIT (RL)                  │
│                                                                 │
│  Narrow-scope PPO for exit timing ONLY:                         │
│                                                                 │
│  State:  position info, unrealized PnL, market features, time   │
│  Actions: HOLD | EXIT | TRAIL_STOP | PARTIAL_CLOSE              │
│  Reward: Risk-adjusted returns (Sharpe-like)                    │
│                                                                 │
│  NOT responsible for:                                           │
│  • Position sizing                                              │
│  • Leverage                                                     │
│  • Entry decisions                                              │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 5: CAPITAL PROTECTION                  │
│                                                                 │
│  Rule-based overlays (NOT learned):                             │
│                                                                 │
│  • Daily loss limit → stop trading                              │
│  • Drawdown threshold → reduce position sizes                   │
│  • Losing streak → cooldown period                              │
│  • Equity curve monitoring → kill switch                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Phase Specifications

### Phase 1: Predictive Foundation

**Purpose:** Generate all predictions needed for downstream risk calculations.

**Components:**
| Component | Output | Shape |
|-----------|--------|-------|
| Direction Head | P(Bear), P(Sideways), P(Bull) | (batch, 3) |
| Volatility Head | Predicted σ | (batch, 1) |
| Quantile Head | Q5, Q25, Q50, Q75, Q95 | (batch, 5) |

**Implementation:**
- ~500 lines
- 2-3 files
- Extends existing TCN backbone

**Key Design Decisions:**
- Multi-task learning with uncertainty weighting
- Quantile regression uses pinball loss
- Monotonicity constraint on quantiles (Q5 < Q25 < Q50 < Q75 < Q95)
- Optional vision feature fusion via projection + attention

---

### Phase 2: Risk Calculations

**Purpose:** Convert predictions into actionable trade parameters.

**Components:**

#### SL/TP Calculator
```python
# For BUY:
SL = entry + Q5 (adjusted by regime, confidence)
TP = entry + Q75 (adjusted by regime, confidence)

# For SELL:
SL = entry + Q95
TP = entry + Q25
```

#### Position Sizing
```python
risk_amount = account_balance × risk_percent
position_size = risk_amount / (SL_distance × pip_value)

# Adjustments:
- Confidence scaling (low confidence → smaller size)
- Volatility scaling (high vol → smaller size)
- Kelly criterion (optional cap)
- Streak adjustment (losing streak → smaller size)
```

#### Hard Rules Engine
| Rule | Trigger | Action |
|------|---------|--------|
| Max Spread | spread > threshold | Block trade |
| Session Filter | outside allowed sessions | Block trade |
| Rollover Avoidance | near rollover time | Block trade |
| Max Pair Exposure | exposure > 5% | Reduce/block |
| Max Total Exposure | exposure > 20% | Block trade |
| Max Leverage | leverage > regime_limit | Cap leverage |
| News Blackout | near high-impact news | Block trade |

**Implementation:**
- ~800 lines
- 3-4 files

---

### Phase 3: Trade Filtering

**Purpose:** Filter out low-quality signals before execution.

#### Triple Barrier Labeling

Creates supervised learning targets:
```
For each entry signal:
1. Set upper barrier (TP)
2. Set lower barrier (SL)  
3. Set time barrier (max holding period)
4. Walk forward through prices
5. Label based on which barrier hit first:
   - Upper barrier → WIN (+1)
   - Lower barrier → LOSS (-1)
   - Time barrier → TIMEOUT (0)
```

#### Meta-Labeling GBM

Predicts: P(primary_model_correct | features)

**Features:**
- Primary model outputs (direction probs, confidence, volatility, quantiles)
- Market conditions (spread, ATR, volume)
- Time features (hour, day, session)
- Derived features (entropy, margin, asymmetry)

**Training:**
- LightGBM or sklearn GradientBoosting
- Time series cross-validation
- Threshold optimization for precision/recall tradeoff

**Implementation:**
- ~600 lines
- 2-3 files

---

### Phase 4: Adaptive Exit (RL)

**Purpose:** Optimize exit timing for open positions.

**Scope:** Exit decisions ONLY (not entry, sizing, or SL/TP placement)

**MDP Definition:**
```
State Space:
- Position info (direction, entry price, current price)
- Unrealized PnL (absolute and as % of SL/TP)
- Time in trade
- Market features (volatility, momentum, regime)
- Distance to SL/TP

Action Space (discrete):
- HOLD: Maintain current SL/TP
- EXIT: Close position now
- TRAIL_STOP: Tighten SL to lock profits
- PARTIAL_CLOSE: Close portion of position

Reward:
- Risk-adjusted returns (Sharpe-like)
- Penalize excessive holding
- Bonus for optimal exit timing
```

**Algorithm:** PPO (Proximal Policy Optimization)

**Implementation:**
- ~700 lines
- 3-4 files
- Uses Stable-Baselines3

---

### Phase 5: Capital Protection

**Purpose:** Prevent catastrophic losses through deterministic rules.

**Rules:**
| Rule | Condition | Action |
|------|-----------|--------|
| Daily Loss Limit | daily_loss > 3% | Stop trading for day |
| Drawdown Circuit Breaker | drawdown > 10% | Halt all trading |
| Position Size Scaling | drawdown > 5% | Reduce sizes by 50% |
| Losing Streak Cooldown | 5+ consecutive losses | Pause for 4 hours |
| Equity Curve Kill Switch | below 200-day MA | Manual review required |

**Why Rules, Not RL:**
- Safety-critical decisions need predictable behavior
- Regulatory requirements may mandate specific limits
- Easier to audit and explain
- No training data needed for edge cases

**Implementation:**
- ~400 lines
- 1-2 files

---

## Architecture Comparison

| Aspect | Original Plan | Revised Architecture |
|--------|---------------|----------------------|
| Primary predictor | GBM | TCN (already exists) |
| GBM role | Directional forecast | Meta-labeling filter |
| Triple Barrier | Before GBM (circular dependency) | After SL/TP calculation |
| RL scope | Controls everything | Exit timing only |
| Capital protection | RL-learned | Rule-based (safer) |
| Position sizing | RL-learned | Formula-based with ML inputs |
| SL/TP calculation | Post-GBM | Pre-Triple Barrier |

---

## Implementation Priority

### Immediate (Phases 1-3)
These provide immediate value with lower complexity:
- ✅ Phase 1: Multi-head TCN backbone
- ✅ Phase 2: SL/TP Calculator, Position Sizing, Hard Rules
- ✅ Phase 3: Triple Barrier Labeling, Meta-Labeling GBM

### Deferred (Phases 4-5)
Add after validating Phases 1-3:
- ⏳ Phase 4: RL Exit Optimization
- ⏳ Phase 5: Capital Protection (can implement in parallel)

---

## File Structure

```
risk_management/
├── __init__.py
├── phase1_predictive/
│   ├── __init__.py
│   ├── tcn_backbone.py      # Multi-head TCN model
│   └── training.py          # Loss functions, trainer
├── phase2_risk_calc/
│   ├── __init__.py
│   ├── sl_tp_calculator.py  # SL/TP calculation
│   ├── position_sizing.py   # Position size calculation
│   └── hard_rules.py        # Deterministic rules engine
├── phase3_filtering/
│   ├── __init__.py
│   ├── triple_barrier.py    # Triple barrier labeling
│   └── meta_labeling.py     # GBM meta-labeling
├── phase4_rl_exit/          # Future
│   └── ...
├── phase5_capital/          # Future
│   └── ...
└── utils/
    └── __init__.py
```

---

## Integration with Existing Codebase

### Existing Components to Leverage:
- TCN backbone architecture (extend with new heads)
- Retraining system (~4,100 lines) with scheduler, drift detection
- Multi-timeframe data support
- FTDM-V1 trend detection
- Vision models (YOLO/ViT)

### Integration Points:
1. **TCN Backbone:** Add volatility and quantile heads to existing direction head
2. **Retraining System:** Extend to handle multi-task loss and meta-model updates
3. **Vision Features:** Optional fusion layer in TCN backbone
4. **Data Pipeline:** Leverage existing MTF data loading

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Direction Accuracy | > 55% | Held-out test set |
| Quantile Coverage | Within 5% of target | Q50 calibration |
| Meta-Label Precision | > 70% | Filtered trade win rate |
| Risk-Adjusted Return | Sharpe > 1.5 | Backtest |
| Max Drawdown | < 15% | Backtest |
| Trade Filter Rate | 30-50% | Signals filtered by meta-model |

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Kendall, A., & Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*
- Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*

---

## Appendix: Key Formulas

### Pinball Loss (Quantile Regression)
```
L_τ(y, ŷ) = τ × max(y - ŷ, 0) + (1-τ) × max(ŷ - y, 0)
```

### Kelly Criterion
```
f* = W - (1-W)/R

Where:
- W = win rate
- R = average win / average loss
- f* = optimal fraction of capital to risk
```

### Position Sizing
```
Position Size = Risk Amount / (SL Distance × Pip Value)

Risk Amount = Account Balance × Risk Percentage × Adjustment Factors
```

### Uncertainty Weighting (Multi-Task Learning)
```
L_total = Σ (L_i / (2σ_i²)) + log(σ_i)

Where σ_i are learnable task uncertainties
```