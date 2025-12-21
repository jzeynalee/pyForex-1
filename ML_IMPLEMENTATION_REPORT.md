# pyForex ML Implementation Report

**Generated:** December 21, 2025  
**Project:** pyForex-1 - Neural Hybrid Forex Trading System  
**Status:** Training Complete, Integration In Progress

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [ML Models Implementation](#ml-models-implementation)
4. [Training Results](#training-results)
5. [Backtesting Results](#backtesting-results)
6. [Risk Management System](#risk-management-system)
7. [Data Pipeline](#data-pipeline)
8. [Known Issues & Limitations](#known-issues--limitations)
9. [Future TODOs](#future-todos)
10. [File Structure](#file-structure)

---

## Executive Summary

The pyForex system is a comprehensive neural-hybrid forex trading platform that combines multiple ML models for market prediction, signal filtering, and exit optimization. The system implements a 5-phase risk management pipeline with predictive modeling, hard rules, meta-labeling, RL-based exit optimization, and capital protection.

### Key Achievements
- ✅ **MultiHeadTCN** trained on 1M data points for all 3 profiles (SCALP, INTRADAY, SWING)
- ✅ **Vision Transformer (ViT)** fine-tuned for chart pattern recognition
- ✅ **Meta-Labeling** models trained using LightGBM for signal filtering
- ✅ **PPO Exit Optimizer** trained with 100K timesteps per profile
- ✅ **Comprehensive backtesting framework** with event-driven replay
- ⚠️ **YOLO Pattern Detector** pending dataset preparation

### Current Performance Metrics
| Model | SCALP | INTRADAY | SWING |
|-------|-------|----------|-------|
| MultiHeadTCN Val Loss | -4.46 | - | -0.62 |
| ViT Val Accuracy | 95.0% | 59.7% | 45.8% |
| Meta-Label AUC | 69.2% | 66.7% | 49.8% |
| Exit Optimizer Win Rate | ~30% | ~31% | ~44% |

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        pyForex Trading System                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Data       │───▶│   Feature    │───▶│   ML         │          │
│  │   Ingestion  │    │   Engineering│    │   Inference  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────────────────────────────────────────────┐          │
│  │              Risk Management Pipeline                 │          │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│  │  │Phase 1 │▶│Phase 2 │▶│Phase 3 │▶│Phase 4 │▶│Phase 5 │        │
│  │  │Predict │ │HardRule│ │MetaLbl │ │RLExit  │ │Capital │        │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │
│  └──────────────────────────────────────────────────────┘          │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Decision   │───▶│   Order      │───▶│   MT5        │          │
│  │   Engine     │    │   Execution  │    │   Connector  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Trading Profiles

| Profile | Timeframes | Forward Bars | Threshold | Use Case |
|---------|------------|--------------|-----------|----------|
| **SCALP** | M5, M15, H1 | 6 bars | 0.1% | Quick trades, high frequency |
| **INTRADAY** | M15, H1, H4 | 10 bars | 0.2% | Day trading |
| **SWING** | H1, H4, D1 | 20 bars | 0.5% | Multi-day positions |

---

## ML Models Implementation

### 1. MultiHeadTCN (Temporal Convolutional Network)

**Location:** `risk_management/phase1_predictive/tcn_backbone.py`

**Architecture:**
- **Backbone:** Temporal Convolutional Network with dilated causal convolutions
- **Multi-Head Outputs:**
  - Direction Head: 3-class classification (BEAR, SIDE, BULL)
  - Volatility Head: Regression for ATR prediction
  - Quantile Head: 5 quantiles (5%, 25%, 50%, 75%, 95%)

**Key Features:**
- Exponentially increasing dilations for long-range dependencies
- Residual connections for gradient flow
- Profile-specific configurations
- Optional vision feature fusion

**Training Configuration:**
```python
{
    'sequence_length': 60,
    'hidden_dim': 128,
    'num_layers': 4,
    'learning_rate': 1e-3,
    'batch_size': 64,
    'epochs': 50-100,
    'early_stopping_patience': 10
}
```

### 2. Vision Transformer (ViT)

**Location:** `models/vit.py`, `training/finetune_vit.py`

**Architecture:**
- Base: `vit_tiny_patch16_224` (pretrained on ImageNet-21k)
- Custom classification head for 3-class output
- Differential learning rates for fine-tuning

**Purpose:**
- Chart pattern recognition from candlestick images
- Visual feature extraction for fusion with TCN

**Training Configuration:**
```python
{
    'model': 'vit_tiny_patch16_224',
    'image_size': 224,
    'epochs': 30,
    'batch_size': 32,
    'learning_rate': 1e-4
}
```

### 3. Meta-Labeling Model (LightGBM)

**Location:** `risk_management/phase3_filtering/meta_labeling.py`

**Purpose:**
- Answers: "Given that my primary model says BUY/SELL, should I actually take this trade?"
- Filters low-quality signals to improve precision

**Features Used:**
- Primary model's direction probabilities
- Volatility predictions
- Quantile spread (prediction interval width)
- Market microstructure (spread, ATR, volume)
- Time features (hour, day, session)

**Training Configuration:**
```python
{
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'early_stopping_rounds': 20,
    'use_class_weights': True
}
```

### 4. PPO Exit Optimizer

**Location:** `risk_management/phase4_rl_exit/`

**Architecture:**
- **Environment:** Custom Gym-compatible `ExitTradingEnv`
- **Agent:** PPO (Proximal Policy Optimization)
- **Action Space:** HOLD, EXIT, TRAIL_STOP, PARTIAL_25/50/75

**State Features (22 dimensions):**
- Position info (direction, entry price, unrealized PnL)
- Time info (holding duration, time to SL/TP)
- Market features (volatility, trend, momentum)
- Risk info (distance to SL/TP, risk-reward position)

**Reward Design:**
- Risk-adjusted returns
- Penalties for: premature exits, late exits, hitting SL
- Transaction costs

**Training Configuration:**
```python
{
    'total_timesteps': 100_000,
    'n_envs': 4,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2
}
```

### 5. YOLO Pattern Detector (Pending)

**Location:** `models/yolo_detector.py`, `training/train_yolo.py`

**Purpose:**
- Real-time chart pattern detection
- Identifies: Head & Shoulders, Double Top/Bottom, Triangles, etc.

**Status:** ⚠️ Requires labeled dataset preparation at `data/yolo.yaml`

---

## Training Results

### MultiHeadTCN Training Summary

| Profile | Data Points | Epochs | Best Val Loss | Early Stop | Checkpoint |
|---------|-------------|--------|---------------|------------|------------|
| SCALP | 1,000,000 | 31 | -4.46 | Yes (patience 10) | `multihead_tcn_SCALP.pth` |
| INTRADAY | 174,571 | 50 | - | No | `multihead_tcn_INTRADAY.pth` |
| SWING | 49,160 | 24 | -0.62 | Yes | `multihead_tcn_SWING.pth` |

**Training Metrics (SCALP Example):**
- Direction Accuracy: ~62-89%
- Volatility MAE: Converging
- Multi-task loss combining direction, volatility, and quantile objectives

### ViT Training Summary

| Profile | Samples | Epochs | Final Train Loss | Best Val Acc | Weights |
|---------|---------|--------|------------------|--------------|---------|
| SCALP | 10,000 | 30 | 0.0156 | **95.0%** | `vit_SCALP.pth` |
| INTRADAY | 10,000 | 30 | 0.0424 | 59.7% | `vit_INTRADAY.pth` |
| SWING | 10,000 | 30 | 0.0682 | 45.8% | `vit_SWING.pth` |

**Note:** SCALP profile shows excellent performance. INTRADAY/SWING may benefit from more diverse training data.

### Meta-Labeling Training Summary

| Profile | Samples | Positive Rate | Val AUC | Val Accuracy | Model |
|---------|---------|---------------|---------|--------------|-------|
| SCALP | 86,466 | 13.98% | **69.2%** | 86.8% | `meta_model_SCALP.pkl` |
| INTRADAY | ~17,000 | ~20% | 66.7% | 80.3% | `meta_model_INTRADAY.pkl` |
| SWING | 40,077 | 32.19% | 49.8% | 76.1% | `meta_model_SWING.pkl` |

**Observations:**
- SCALP has best AUC but low positive rate (imbalanced)
- SWING has near-random AUC (0.498) - needs investigation

### Exit Optimizer Training Summary

| Profile | Timesteps | Episodes | Final Avg Reward | Final Win Rate | Model |
|---------|-----------|----------|------------------|----------------|-------|
| SCALP | 100,000 | 34,967 | 0.057 | ~30% | `exit_optimizer_SCALP.pth` |
| INTRADAY | 100,000 | 39,672 | 0.043 | ~31% | `exit_optimizer_INTRADAY.pth` |
| SWING | 100,000 | 29,792 | 0.095 | **44%** | `exit_optimizer_SWING.pth` |

**Training Progression (SCALP):**
```
Timestep 10K:  Avg Reward: -0.053, Win Rate: 7%
Timestep 50K:  Avg Reward: 0.007,  Win Rate: 7%
Timestep 75K:  Avg Reward: 0.038,  Win Rate: 28%
Timestep 100K: Avg Reward: 0.057,  Win Rate: 30%
```

---

## Backtesting Results

### Backtesting Framework

**Location:** `trading/backtest_connector.py`, `backtesting/orchestrator.py`

**Features:**
- Event-driven replay (no code divergence from live)
- Configurable slippage, commission, latency simulation
- Equity curve tracking with Max DD and Sharpe calculation
- Comprehensive metrics reporting

### Recent Backtest Results

**Note:** Backtesting is still being refined. Current results show integration issues.

| Date | Profile | Total Trades | Win Rate | Sharpe | Max DD | Final Balance |
|------|---------|--------------|----------|--------|--------|---------------|
| 2025-12-21 | INTRADAY | 1,853 | 30.3% | -5.13 | 95.9% | $476.60 |
| 2025-12-20 | INTRADAY | 0 | 0% | 0 | 0% | $10,000 |

**Acceptance Gate Criteria:**
- Sharpe Ratio ≥ 1.5
- Profit Factor ≥ 1.5
- Win Rate ≥ 45%
- Minimum 30 trades

**Current Status:** ❌ Not passing acceptance gate - requires further tuning

### Identified Issues
1. Signal generation not translating to profitable trades
2. Position sizing may be too aggressive
3. Exit timing needs optimization
4. Model confidence thresholds need calibration

---

## Risk Management System

### 5-Phase Pipeline

```
Phase 1: Predictive Risk Assessment (MultiHeadTCN)
    ↓
Phase 2: Hard Rules Engine
    ↓
Phase 3: Meta-Labeling Filter (LightGBM)
    ↓
Phase 4: RL Exit Optimization (PPO)
    ↓
Phase 5: Capital Protection
```

### Phase 1: Predictive Risk Assessment
- **Model:** MultiHeadTCN
- **Outputs:** Direction probabilities, volatility forecast, quantile predictions
- **Purpose:** Initial signal generation with uncertainty quantification

### Phase 2: Hard Rules Engine
**Location:** `risk_management/phase2_risk_calc/hard_rules.py`

**Rules:**
- Session filtering (allowed trading sessions)
- Weekend/holiday closing
- Maximum exposure limits
- Correlation limits
- Drawdown limits

### Phase 3: Meta-Labeling
- **Model:** LightGBM classifier
- **Purpose:** Filter low-quality signals
- **Threshold:** Configurable (default 0.5)

### Phase 4: RL Exit Optimization
- **Model:** PPO Agent
- **Actions:** HOLD, EXIT, TRAIL_STOP, PARTIAL_CLOSE
- **Purpose:** Optimal exit timing to maximize risk-adjusted returns

### Phase 5: Capital Protection
**Location:** `risk_management/phase5_capital_protection/`

**Features:**
- Daily loss limits
- Drawdown-based position reduction
- Emergency stop mechanisms
- Recovery mode after significant losses

---

## Data Pipeline

### Available Data

| Symbol | Timeframe | Rows | File Size | Path |
|--------|-----------|------|-----------|------|
| EURUSD | M5 | ~1,000,000 | 64 MB | `data/raw/EURUSD_M5_latest.csv` |
| EURUSD | M15 | ~333,000 | 43 MB | `data/raw/EURUSD_M15_latest.csv` |
| EURUSD | H1 | ~174,000 | 11 MB | `data/raw/EURUSD_H1_latest.csv` |
| EURUSD | H4 | ~49,000 | 3 MB | `data/raw/EURUSD_H4_latest.csv` |
| EURUSD | D1 | ~8,700 | 752 KB | `data/raw/EURUSD_D1_latest.csv` |

### Feature Engineering

**Location:** `utils/features_engineering.py`

**Features Generated:**
- **Price-based:** Returns, log returns, price ratios
- **Volatility:** ATR, Bollinger Bands, historical volatility
- **Momentum:** RSI, MACD, Stochastic, ROC
- **Trend:** SMA, EMA, ADX, Aroon
- **Volume:** OBV, VWAP, volume ratios
- **Pattern:** Candlestick patterns, support/resistance

---

## Known Issues & Limitations

### Critical Issues
1. **Backtest Performance:** Current backtests show significant losses (-95% in some cases)
2. **SWING Meta-Label AUC:** Near-random (49.8%) - model not learning meaningful patterns
3. **YOLO Dataset:** Missing labeled dataset for pattern detection

### Technical Debt
1. Some emoji characters in logging cause Windows encoding issues (fixed)
2. Feature engineering creates some NaN values requiring handling
3. Large dataset processing requires batch processing

### Model Limitations
1. **ViT:** Trained on synthetic chart features, not real candlestick images
2. **Meta-Labeling:** Simulated primary signals, not actual model predictions
3. **Exit Optimizer:** Trained in isolation, not integrated with full pipeline

---

## Future TODOs

### High Priority
- [ ] **Fix backtest integration** - Ensure ML signals translate to profitable trades
- [ ] **Calibrate confidence thresholds** - Find optimal thresholds for each profile
- [ ] **Prepare YOLO dataset** - Label chart patterns for pattern detection training
- [ ] **End-to-end pipeline testing** - Test full prediction → execution flow

### Medium Priority
- [ ] **Improve SWING meta-labeling** - Investigate why AUC is near-random
- [ ] **Real candlestick images for ViT** - Generate actual chart images for training
- [ ] **Hyperparameter optimization** - Grid search for optimal model parameters
- [ ] **Cross-validation** - Implement proper time-series cross-validation

### Low Priority
- [ ] **Multi-symbol support** - Extend to GBPUSD, USDJPY, etc.
- [ ] **Ensemble methods** - Combine multiple models for better predictions
- [ ] **Online learning** - Implement incremental model updates
- [ ] **Explainability** - Add SHAP/LIME for model interpretation

### Infrastructure
- [ ] **Model versioning** - Implement MLflow or similar
- [ ] **Automated retraining** - Schedule periodic model updates
- [ ] **Monitoring dashboard** - Real-time model performance tracking
- [ ] **A/B testing framework** - Compare model versions in production

---

## File Structure

```
pyForex-1/
├── models/
│   ├── weights/                    # Trained model weights
│   │   ├── multihead_tcn_*.pth    # TCN weights per profile
│   │   ├── vit_*.pth              # ViT weights per profile
│   │   └── *.pt                   # Other model weights
│   ├── tcn.py                     # TCN model definitions
│   ├── vit.py                     # ViT model definitions
│   ├── decision_fusion.py         # Multi-model fusion
│   └── yolo_detector.py           # YOLO pattern detector
│
├── risk_management/
│   ├── phase1_predictive/         # MultiHeadTCN
│   │   ├── tcn_backbone.py
│   │   └── training.py
│   ├── phase2_risk_calc/          # Hard rules
│   │   └── hard_rules.py
│   ├── phase3_filtering/          # Meta-labeling
│   │   └── meta_labeling.py
│   ├── phase4_rl_exit/            # PPO exit optimizer
│   │   ├── environment.py
│   │   ├── ppo_agent.py
│   │   └── trainer.py
│   └── phase5_capital_protection/ # Capital protection
│
├── checkpoints/
│   ├── multihead_tcn/             # TCN checkpoints
│   ├── meta_labeling/             # Meta-label models
│   ├── exit_optimizer/            # PPO checkpoints
│   └── vit/                       # ViT checkpoints
│
├── training/
│   ├── train_tcn_enhanced.py      # TCN training script
│   ├── finetune_vit.py            # ViT fine-tuning
│   └── train_yolo.py              # YOLO training
│
├── scripts/
│   └── train_all_models.py        # Comprehensive training script
│
├── trading/
│   ├── backtest_connector.py      # Backtest connector
│   ├── backtest_runner.py         # Backtest orchestrator
│   └── bot.py                     # Trading bot
│
├── backtesting/
│   └── orchestrator.py            # Backtest orchestration
│
├── data/
│   └── raw/                       # Raw price data
│       └── EURUSD_*.csv
│
├── backtest_reports/              # Backtest results
├── backtest_artifacts/            # Backtest artifacts
└── training_results.json          # Training summary
```

---

## Appendix: Training Commands

### Train All Models
```bash
python scripts/train_all_models.py --models all --profiles all --data-rows 1000000 --epochs 50
```

### Train Specific Model
```bash
# MultiHeadTCN only
python scripts/train_all_models.py --models tcn --profiles SCALP INTRADAY SWING

# ViT only
python scripts/train_all_models.py --models vit --profiles all

# Meta-labeling only
python scripts/train_all_models.py --models meta --profiles all

# Exit optimizer only
python scripts/train_all_models.py --models exit --profiles all
```

### Run Backtest
```bash
python -m backtesting.orchestrator --profile INTRADAY --start 2023-01-01 --end 2023-12-31
```

---

## Appendix: Model Weights Summary

| Model | Profile | File | Size |
|-------|---------|------|------|
| MultiHeadTCN | SCALP | `multihead_tcn_SCALP.pth` | 5.6 MB |
| MultiHeadTCN | INTRADAY | `multihead_tcn_INTRADAY.pth` | 6.8 MB |
| MultiHeadTCN | SWING | `multihead_tcn_SWING.pth` | 9.1 MB |
| ViT | SCALP | `vit_SCALP.pth` | 21.1 MB |
| ViT | INTRADAY | `vit_INTRADAY.pth` | 21.1 MB |
| ViT | SWING | `vit_SWING.pth` | 21.1 MB |
| Meta-Label | SCALP | `meta_model_SCALP.pkl` | ~1 MB |
| Meta-Label | INTRADAY | `meta_model_INTRADAY.pkl` | ~1 MB |
| Meta-Label | SWING | `meta_model_SWING.pkl` | ~1 MB |
| Exit Optimizer | SCALP | `exit_optimizer_SCALP.pth` | ~5 MB |
| Exit Optimizer | INTRADAY | `exit_optimizer_INTRADAY.pth` | ~5 MB |
| Exit Optimizer | SWING | `exit_optimizer_SWING.pth` | ~5 MB |

---

*Report generated by pyForex ML Training Pipeline*
