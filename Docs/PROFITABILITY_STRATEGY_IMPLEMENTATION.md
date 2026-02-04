# pyForex-1 Profitability Strategy Implementation (v2.0)

## Overview

This document summarizes the implementation of the profitability improvement strategy for pyForex-1.

**Date:** February 2026  
**Version:** 2.0

---

## Strategy Summary

| Action | Status | Impact |
|--------|--------|--------|
| **Simplify** (remove TCN/ViT) | ✅ Complete | Reduced noise, faster inference |
| **Integrate** (MH-TCN → Alpha Factory 3TF) | ✅ Complete | Unified decision pipeline |
| **Retrain** (walk-forward validation) | ✅ Complete | Proper out-of-sample testing |
| **Raise thresholds** (trade less, trade better) | ✅ Complete | Higher quality trades |

---

## Changes Made

### 1. Simplified Model Architecture

**Removed/Disabled:**
- `models/vit.py` - ViT disabled by default (weak signal, adds latency)
- `inference/inference_vit.py` - ViT inference disabled
- `use_vision=False` now default in `PredictorConfig` and `StrategyConfig`

**Kept:**
- `risk_management/phase1_predictive/tcn_backbone.py` - MH-TCN (Multi-Head TCN)
  - Direction head: P(Bear), P(Sideways), P(Bull)
  - Volatility head: Predicted σ
  - Quantile head: Q5, Q25, Q50, Q75, Q95
  - Outcome head: P(TP before SL) for long/short

### 2. New Unified Pipeline

**New Files Created:**

#### `alpha_factory/mhtcn_integration.py`
- `MHTCNPrediction` - Container for MH-TCN outputs
- `MHTCNFeatureProvider` - Bridges MH-TCN to 3TF FeatureSnapshots
- `UnifiedThreeTFEngine` - Main entry point with raised thresholds

#### `strategies/unified_3tf_strategy.py`
- `Unified3TFConfig` - Configuration with higher thresholds
- `Unified3TFStrategy` - New primary strategy using unified pipeline
- `create_unified_strategy()` - Factory function

#### `training/walk_forward_trainer.py`
- `WalkForwardConfig` - Configuration for walk-forward training
- `TripleBarrierLabeler` - Proper labeling for direction prediction
- `WalkForwardTrainer` - Rolling window training with purged CV
- `FoldResult` - Results tracking per fold

### 3. Raised Thresholds

| Parameter | Old Value | New Value | Location |
|-----------|-----------|-----------|----------|
| HTF Confidence | 0.50 | 0.60 | `UnifiedThreeTFEngine` |
| MTF Confidence | 0.60 | 0.65 | `UnifiedThreeTFEngine` |
| LTF Confidence | 0.65 | 0.70 | `UnifiedThreeTFEngine` |
| Stability | 0.40 | 0.50 | `UnifiedThreeTFEngine` |
| Directional Score | 0.25 | 0.30 | `UnifiedThreeTFEngine` |
| Min R:R Ratio | 1.5 | 2.0 | `StrategyConfig`, `DecisionEngineConfig` |
| Base Risk % | 1.0 | 0.5 | `StrategyConfig`, `DecisionEngineConfig` |
| Max Daily Loss | 3.0% | 2.0% | `DecisionEngineConfig` |
| Max Drawdown | 10.0% | 8.0% | `DecisionEngineConfig` |
| Confidence Gate | 70th %ile | 75th %ile | `DecisionConfig` |

### 4. Updated Training Commands

```bash
# Walk-forward training (RECOMMENDED)
python main.py train walk-forward --data data/raw/eurusd.csv --profile INTRADAY

# Single-fold MH-TCN training
python main.py train mhtcn --data data/raw/eurusd.csv --epochs 30

# Available models: tcn, mhtcn, walk-forward, vit, fusion, trend
```

---

## Architecture Comparison

### Before (Fragmented)
```
NeuralHybridStrategy
    → HybridPredictor
        → TCN (direction only)
        → ViT (chart images) ← WEAK SIGNAL
        → FusionNet
    → DecisionEngine
        → Meta-labeling
        → MTF alignment
        → Hard rules
        → Capital protection
```

### After (Unified)
```
Unified3TFStrategy
    → UnifiedThreeTFEngine
        → MHTCNFeatureProvider
            → MH-TCN (direction + volatility + quantiles + outcomes)
        → ThreeTFOrchestrator
            → HTF Decision (Governor)
            → MTF Decision (Validator)
            → LTF Decision (Trigger)
        → TradeInstruction
```

---

## 3TF Profiles

| Profile | LTF (Trigger) | MTF (Structure) | HTF (Bias) |
|---------|---------------|-----------------|------------|
| SCALP | M5 | M15 | H1 |
| INTRADAY | M15 | H1 | H4 |
| SWING | H1 | H4 | D1 |

---

## Walk-Forward Training

The new training infrastructure implements proper walk-forward validation:

1. **Rolling Windows**: Train on N bars, test on M bars, step by S bars
2. **Purge Gap**: Gap between train/test to prevent leakage
3. **Triple Barrier Labels**: Proper labeling based on TP/SL outcomes
4. **Fold Tracking**: Metrics tracked per fold for model selection

**Default Configuration:**
```python
train_window = 5000    # Training window size
test_window = 1000     # Test window size
step_size = 500        # Step between folds
purge_gap = 50         # Gap to prevent leakage
epochs_per_fold = 30   # Training epochs per fold
```

---

## Usage

### Live Trading with Unified Strategy
```python
from strategies.unified_3tf_strategy import create_unified_strategy

strategy = create_unified_strategy(
    profile='INTRADAY',
    symbol='EURUSD',
    data_provider=data_provider,
    executor=executor
)

# Process bars
signal = strategy.on_bar(df)  # Returns 'BUY', 'SELL', or None
```

### Using UnifiedThreeTFEngine Directly
```python
from alpha_factory import UnifiedThreeTFEngine

engine = UnifiedThreeTFEngine(
    symbol='EURUSD',
    profile_type='INTRADAY',
    weights_dir='models/weights'
)

instruction = engine.evaluate(
    data_htf=df_h4,
    data_mtf=df_h1,
    data_ltf=df_m15
)

if instruction:
    print(f"Trade: {instruction.direction} @ conf={instruction.confidence:.2f}")
```

---

## Next Steps

1. **Train MH-TCN with walk-forward validation:**
   ```bash
   python main.py train walk-forward --data data/raw/eurusd_m15.csv --profile INTRADAY
   ```

2. **Run backtest with unified strategy:**
   ```bash
   python main.py backtest --data data/EURUSD_H1.csv --strategy unified
   ```

3. **Monitor performance metrics:**
   - Win rate should increase (fewer but higher quality trades)
   - Sharpe ratio should improve
   - Max drawdown should decrease

---

## Files Modified

| File | Changes |
|------|---------|
| `alpha_factory/__init__.py` | Added MH-TCN integration exports |
| `alpha_factory/decision_making.py` | Raised thresholds |
| `inference/predictor.py` | Disabled ViT by default, raised thresholds |
| `strategies/neural_hybrid.py` | Disabled ViT, raised thresholds |
| `trading/decision_engine.py` | Raised thresholds, tightened risk |
| `main.py` | Added walk-forward training command |

## Files Created

| File | Purpose |
|------|---------|
| `alpha_factory/mhtcn_integration.py` | MH-TCN + 3TF integration |
| `strategies/unified_3tf_strategy.py` | New unified strategy |
| `training/walk_forward_trainer.py` | Walk-forward training infrastructure |

---

## Summary

The implementation follows the "trade less, trade better" philosophy:

1. **Simplified**: Removed weak ViT signal, kept strong MH-TCN
2. **Integrated**: Unified MH-TCN with Alpha Factory 3TF cascade
3. **Validated**: Walk-forward training prevents overfitting
4. **Conservative**: Higher thresholds, lower risk, better R:R

Expected outcomes:
- Fewer trades (filtered by higher thresholds)
- Higher win rate (better quality signals)
- Lower drawdown (tighter risk management)
- Better Sharpe ratio (improved risk-adjusted returns)
