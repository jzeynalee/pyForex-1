# pyForex v2.0 - Training & Backtesting Workflow

## Overview

This document outlines the complete workflow for training and backtesting the v2.0 system with MH-TCN + Alpha Factory integration.

---

## Current Model Architecture (v2.0)

### Active Models

| Model | Location | Purpose | Training |
|-------|----------|---------|----------|
| **MH-TCN** | `risk_management/phase1_predictive/tcn_backbone.py` | Direction, volatility, quantiles, outcomes | `walk-forward` |
| **Price Action** | `models/price_action_pattern.py` | Pattern detection (no training needed) | N/A |
| **Trend Classifier** | `models/trend_classifier.py` | Trend classification | `trend` |
| **Decision Fusion** | `models/decision_fusion.py` | Multi-signal fusion | `train_decision_fusion.py` |

### Removed Models (v2.0)
- ~~ViT (Vision Transformer)~~ - Weak signal, high latency
- ~~TCN (Simple)~~ - Replaced by MH-TCN
- ~~YOLO~~ - Replaced by Price Action patterns

---

## Training Workflow

### Step 1: Fetch Data

```powershell
# Fetch data for all timeframes
python main.py fetch-data --symbol EURUSD --timeframes M5,M15,H1,H4 --bars 50000
```

Data will be saved to: `E:\pyProject\pyForex-assets\data\mt5\EURUSD\`

### Step 2: Train MH-TCN (Walk-Forward)

**Recommended for all profiles:**

```powershell
# SCALP profile (M5/M15/H1)
python main.py train walk-forward --data "E:\pyProject\pyForex-assets\data\mt5\EURUSD\EURUSD_M5_*.csv" --profile SCALP --epochs 30

# INTRADAY profile (M15/H1/H4)
python main.py train walk-forward --data "E:\pyProject\pyForex-assets\data\mt5\EURUSD\EURUSD_M15_*.csv" --profile INTRADAY --epochs 30

# SWING profile (H1/H4/D1)
python main.py train walk-forward --data "E:\pyProject\pyForex-assets\data\mt5\EURUSD\EURUSD_H1_*.csv" --profile SWING --epochs 30
```

**Output:**
- Models saved to: `models/weights/walk_forward/fold_*_<PROFILE>.pth`
- Summary: `models/weights/walk_forward/walk_forward_summary.json`

### Step 3: Train Trend Classifier (Optional)

```powershell
python main.py train trend --data "E:\pyProject\pyForex-assets\data\mt5\EURUSD\EURUSD_H1_*.csv" --save-dir models/weights
```

### Step 4: Copy Best Models to Production

```powershell
# Copy best walk-forward model to production weights
Copy-Item "models\weights\walk_forward\fold_16_INTRADAY.pth" "E:\pyProject\pyForex-assets\models\weights\multihead_tcn_INTRADAY.pth"
```

---

## Backtesting Workflow

### Option 1: Simple Backtest

```powershell
python main.py backtest --data "E:\pyProject\pyForex-assets\data\mt5\EURUSD\EURUSD_H1_*.csv" --strategy neural --balance 10000
```

### Option 2: Alpha Factory Backtest (Probabilistic Engine)

```powershell
python main.py alpha-backtest --data "E:\pyProject\pyForex-assets\data\mt5\EURUSD\EURUSD_H1_*.csv" --engine decision --window 300 --balance 10000
```

---

## Alpha Factory Integration

### 3TF System Flow

```
HTF (Governor)     →  MTF (Validator)    →  LTF (Trigger)
   ↓                      ↓                     ↓
Allow/Deny trade    Confirm structure     Execute entry
Set bias (LONG/SHORT)  Boost confidence    Final timing
```

### Using UnifiedThreeTFEngine

```python
from alpha_factory import UnifiedThreeTFEngine

engine = UnifiedThreeTFEngine(
    symbol='EURUSD',
    profile_type='INTRADAY',
    weights_dir='E:/pyProject/pyForex-assets/models/weights'
)

# Evaluate with MTF data
instruction = engine.evaluate(
    data_htf=df_h4,  # H4 data
    data_mtf=df_h1,  # H1 data
    data_ltf=df_m15  # M15 data
)

if instruction:
    print(f"Trade: {instruction.direction}")
    print(f"Confidence: {instruction.confidence:.2f}")
    print(f"Size multiplier: {instruction.size_multiplier:.2f}")
```

---

## Confidence Thresholds (v2.0)

| Threshold | Value | Description |
|-----------|-------|-------------|
| HTF Confidence | 0.60 | Minimum to allow trading |
| MTF Confidence | 0.65 | Minimum for structure validation |
| LTF Confidence | 0.70 | Minimum for entry trigger |
| Stability | 0.50 | Minimum feature stability |
| Directional Score | 0.30 | Minimum for bias determination |
| Min R:R Ratio | 2.0 | Minimum risk-reward |
| Base Risk % | 0.5% | Per-trade risk |

---

## Complete Training Pipeline

### Full Automated Training

```powershell
# 1. Fetch fresh data
python main.py fetch-data --symbol EURUSD --timeframes M5,M15,H1,H4,D1 --bars 50000

# 2. Train MH-TCN for all profiles
python main.py train walk-forward --data "E:\pyProject\pyForex-assets\data\mt5\EURUSD\EURUSD_M5_*.csv" --profile SCALP
python main.py train walk-forward --data "E:\pyProject\pyForex-assets\data\mt5\EURUSD\EURUSD_M15_*.csv" --profile INTRADAY
python main.py train walk-forward --data "E:\pyProject\pyForex-assets\data\mt5\EURUSD\EURUSD_H1_*.csv" --profile SWING

# 3. Train trend classifier
python main.py train trend --data "E:\pyProject\pyForex-assets\data\mt5\EURUSD\EURUSD_H1_*.csv"

# 4. Run backtests
python main.py backtest --data "E:\pyProject\pyForex-assets\data\mt5\EURUSD\EURUSD_H1_*.csv" --strategy neural
```

---

## Model Files Summary

### Required Files (v2.0)

```
E:\pyProject\pyForex-assets\models\weights\
├── multihead_tcn_SCALP.pth      # MH-TCN for SCALP
├── multihead_tcn_INTRADAY.pth   # MH-TCN for INTRADAY
├── multihead_tcn_SWING.pth      # MH-TCN for SWING
├── trend_classifier.joblib      # Trend classifier (optional)
└── walk_forward/                # Walk-forward training outputs
    ├── fold_*_SCALP.pth
    ├── fold_*_INTRADAY.pth
    ├── fold_*_SWING.pth
    └── walk_forward_summary.json
```

### Not Required (removed in v2.0)
- ~~vit_*.pth~~ - ViT removed
- ~~tcn_*.pth~~ - Old TCN removed
- ~~yolo_*.pt~~ - YOLO removed

---

## Performance Expectations

Based on walk-forward training results:

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Direction Accuracy | 40-50% | 3-class (BEAR/SIDEWAYS/BULL) |
| F1 Score | 0.35-0.45 | Macro average |
| Win Rate (trading) | 45-55% | With 2:1 R:R |
| Sharpe Ratio | 0.5-1.5 | Depends on market conditions |
| Max Drawdown | < 10% | With capital protection |

---

## Next Steps

1. **Improve Feature Engineering**: Add more technical indicators to `walk_forward_trainer.py`
2. **Tune Hyperparameters**: Adjust training window, epochs, learning rate
3. **Add More Data**: Train on longer history for better generalization
4. **Cross-Pair Training**: Train on multiple currency pairs
5. **Live Paper Trading**: Test with mock connector before live
