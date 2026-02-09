# MH-TCN Training Plan — Complete

## Overview

MH-TCN serves **two distinct purposes** in pyForex-1. This plan covers both.

| # | Purpose | Architecture | Heads | Models | Script |
|---|---------|-------------|-------|--------|--------|
| **A** | **Risk Management** — 3TF decision cascade, SL/TP (quantiles), position sizing (outcome), direction confidence | `MultiHeadTCN` (128 hidden, 5 heads) | Direction (3-class), Volatility, Quantile (5), Outcome (p_long/p_short) | **9** (3 profiles × 3 TFs) | `scripts/train_all_models.py` |
| **B** | **Research Variants** — confidence modulation (g_factor) for the 6-variant experiment | `ResearchTCN` (64 hidden) + `ProbabilisticTCN` (32 hidden) | Single sigmoid g_factor ∈ (0,1] | **4** (per alpha×filter combo) | `research/train_mhtcn_variants.py` |

**Total: 13 models to train.**

---

## Data

Source: `E:\pyProject\data\raw\`  
Output: `E:\pyProject\pyForex-assets\`

| File | Rows | Used By (Part A 3TF) | Part B |
|------|------|----------------------|--------|
| `EURUSD_M5_latest.csv` | ~1,000,000 | SCALP LTF | — |
| `EURUSD_M15_latest.csv` | ~676,000 | SCALP MTF, INTRADAY LTF | — |
| `EURUSD_H1_latest.csv` | ~175,000 | SCALP HTF, INTRADAY MTF, SWING LTF | Research variants |
| `EURUSD_H4_latest.csv` | ~49,000 | INTRADAY HTF, SWING MTF | — |
| `EURUSD_D1_latest.csv` | ~14,000 | SWING HTF | — |

⚠ **Note**: `pyForex-assets/data/mt5/EURUSD/` currently has M5, M15, H1 only.
H4 and D1 are available at `E:\pyProject\data\raw\` and need to be copied to
assets or the training scripts need to be configured to use that path for H4/D1.

---

# PART A — Risk Management MultiHeadTCN (9 models: 3 profiles × 3 TFs)

## A.1 Purpose

The full `MultiHeadTCN` from `risk_management/phase1_predictive/tcn_backbone.py`
is the backbone of the risk management pipeline. It feeds into:

- **`phase2_risk_calc/sl_tp_calculator.py`** — Uses **quantile head** predictions
  (Q5, Q25, Q50, Q75, Q95 of price distribution) to set data-driven SL/TP levels,
  adjusted by regime and volatility.
- **`phase2_risk_calc/position_sizing.py`** — Uses **direction confidence** and
  **volatility head** to scale position size (confidence-weighted Kelly/fixed-fraction).
- **`risk_management/risk_manager.py`** — Orchestrates Phase 1→2→3: prediction →
  SL/TP + position sizing → hard rules + meta-labeling filter.
- **`alpha_factory/mhtcn_integration.py`** — `MHTCNFeatureProvider` loads **per-TF**
  models, generates `MHTCNPrediction` (direction, volatility, quantiles, outcome)
  for the `UnifiedThreeTFEngine`.

## A.2 Why 9 Models (3TF Architecture)

The decision-making process uses **3TF (Three-Timeframe) analysis** as defined in
`alpha_factory/trading_profiles.py`. Every trade decision requires MH-TCN predictions
from **all three timeframes** in a profile:

```
alpha_factory/trading_profiles.py:
  SCALPING  → LTF: M5   MTF: M15  HTF: H1
  INTRADAY  → LTF: M15  MTF: H1   HTF: H4
  SWING     → LTF: H1   MTF: H4   HTF: D1
```

`UnifiedThreeTFEngine.evaluate()` calls `feature_provider.predict()` for each TF:
```
snapshot_htf = self.feature_provider.create_snapshot(data_htf, htf_str, ...)
snapshot_mtf = self.feature_provider.create_snapshot(data_mtf, mtf_str, ...)
snapshot_ltf = self.feature_provider.create_snapshot(data_ltf, ltf_str, ...)
```

Each TF produces a `FeatureSnapshot` with `directional_score`, `confidence`,
`stability`, and `regime_flags` (quantiles, p_long, p_short). The 3TF cascade
(HTF bias → MTF validation → LTF trigger) requires all three to agree.

`MHTCNFeatureProvider._find_weight_file()` searches for per-TF weights:
`multihead_tcn_{PROFILE}_{TF}.pth` (preferred) → `multihead_tcn_{PROFILE}.pth` (fallback).

## A.3 Full Model Matrix (9 models)

| # | Profile | TF Role | Data TF | Data Source | Weights Filename |
|---|---------|---------|---------|-------------|------------------|
| 1 | SCALP | LTF | M5 | `pyForex-assets/.../EURUSD_M5_*.csv` | `multihead_tcn_SCALP_M5.pth` |
| 2 | SCALP | MTF | M15 | `pyForex-assets/.../EURUSD_M15_*.csv` | `multihead_tcn_SCALP_M15.pth` |
| 3 | SCALP | HTF | H1 | `pyForex-assets/.../EURUSD_H1_*.csv` | `multihead_tcn_SCALP_H1.pth` |
| 4 | INTRADAY | LTF | M15 | `pyForex-assets/.../EURUSD_M15_*.csv` | `multihead_tcn_INTRADAY_M15.pth` |
| 5 | INTRADAY | MTF | H1 | `pyForex-assets/.../EURUSD_H1_*.csv` | `multihead_tcn_INTRADAY_H1.pth` |
| 6 | INTRADAY | HTF | H4 | `data/raw/EURUSD_H4_latest.csv` ⚠ | `multihead_tcn_INTRADAY_H4.pth` |
| 7 | SWING | LTF | H1 | `pyForex-assets/.../EURUSD_H1_*.csv` | `multihead_tcn_SWING_H1.pth` |
| 8 | SWING | MTF | H4 | `data/raw/EURUSD_H4_latest.csv` ⚠ | `multihead_tcn_SWING_H4.pth` |
| 9 | SWING | HTF | D1 | `data/raw/EURUSD_D1_latest.csv` ⚠ | `multihead_tcn_SWING_D1.pth` |

⚠ **Data gap**: H4 and D1 data not yet in `pyForex-assets/data/mt5/EURUSD/`.
Available at `E:\pyProject\data\raw\` (H4 ~49K rows, D1 ~14K rows).
**Action**: Copy or symlink H4/D1 CSVs into assets, or update training scripts
to also search `E:\pyProject\data\raw\`.

### Head outputs per timeframe role:

| Head | HTF Role | MTF Role | LTF Role |
|------|----------|----------|----------|
| **Direction** | Participation bias (Bull/Bear gate) | Structure validation | Entry trigger |
| **Volatility** | Regime stability check | ATR-based SL sizing | Timing precision |
| **Quantile** | Wide distribution for regime context | SL/TP level calculation | Tight entry zones |
| **Outcome** | Macro trade viability (p_long/p_short) | Trade quality filter | Final confidence |

## A.4 Training Pipeline

**Primary script**: `scripts/train_all_models.py` — iterates all 3 TFs per profile.
**Per-profile script**: `training/retrain_mhtcn.py` — trains primary TF only (LTF).
Both use the same core: `MultiHeadTCNTrainer` + `MHTCNDataPreparer` + 278-feature engineering.

Per-model training steps:
1. **Load data** — CSV for the target timeframe
2. **Feature engineering** — `FeatureEngineerOptimized` → 278 numeric features
3. **Multi-label generation** — `MHTCNDataPreparer`:
   - Direction labels (3-class: Bear/Sideways/Bull) with auto-optimal threshold
   - Volatility labels (regression)
   - Price move labels (quantile targets)
   - Outcome labels via triple barrier (TP-before-SL probabilities)
4. **Weighted sampling** — `WeightedRandomSampler` to balance Bear/Side/Bull classes
5. **Train** — `MultiHeadTCNTrainer`:
   - Epochs: 120, patience: 20, lr: 5e-4
   - AdamW, grad clip 1.0, warmup 5 epochs
   - Direction weight: 3.0, outcome weight: 1.5, vol/quant: 1.0
   - Temporal split: 75% train / 15% val / 10% test
6. **Save** — `multihead_tcn_{PROFILE}_{TF}.pth` to local + `pyForex-assets`

## A.5 Execution

```powershell
& d:\myBot\.venv312\scripts\activate.ps1

# Option 1: Train all 9 models via train_all_models.py (recommended)
python scripts/train_all_models.py --models tcn --profiles all --epochs 120 --device auto

# Option 2: Train primary-TF only via retrain_mhtcn.py (3 models — fallback mode)
python -m training.retrain_mhtcn --all --epochs 120 --device auto
```

## A.6 Outputs

```
E:\pyProject\pyForex-assets\models\weights\
├── multihead_tcn_SCALP_M5.pth        # SCALP LTF
├── multihead_tcn_SCALP_M15.pth       # SCALP MTF
├── multihead_tcn_SCALP_H1.pth        # SCALP HTF
├── multihead_tcn_INTRADAY_M15.pth    # INTRADAY LTF
├── multihead_tcn_INTRADAY_H1.pth     # INTRADAY MTF
├── multihead_tcn_INTRADAY_H4.pth     # INTRADAY HTF
├── multihead_tcn_SWING_H1.pth        # SWING LTF
├── multihead_tcn_SWING_H4.pth        # SWING MTF
├── multihead_tcn_SWING_D1.pth        # SWING HTF
└── retrain_report.json
```

Each checkpoint contains: `state_dict`, `config`, `feature_columns`, `training_config`, `test_metrics`, `history`.

### What each head provides to risk management:

| Head | Output | Consumed By |
|------|--------|-------------|
| **Direction** | P(Bear), P(Sideways), P(Bull) | `UnifiedThreeTFEngine` → 3TF cascade (HTF bias → MTF validation → LTF trigger) |
| **Volatility** | σ_predicted (regression) | `PositionSizingCalculator` → volatility scaling; `SLTPCalculator` → ATR adjustment |
| **Quantile** | Q5, Q25, Q50, Q75, Q95 of price Δ | `SLTPCalculator.calculate()` → data-driven SL (Q5/Q95) and TP (Q75/Q25) |
| **Outcome** | p_long, p_short (TP-before-SL prob) | `MHTCNPrediction.confidence` → final trade gating; Kelly input for position sizing |

---

# PART B — Research Variant MH-TCN (4 models)

## B.1 Purpose

The 6-variant research framework (`research/experiment/variants.py`) compares
alpha heads with/without MH-TCN confidence modulation. The MH-TCN **never
overrides direction** — it only scales alpha probability:
**P_final = P_alpha × g_mhtcn**.

## B.2 Variant Matrix

| ID | Alpha Head | MH-TCN Filter | Model to Train | Weights Output |
|----|-----------|---------------|---------------|----------------|
| V1 | AlphaV1 (rule-based) | NullFilter | — (baseline) | — |
| V2 | AlphaV1 | RawFeatureMHTCNFilter | `ResearchTCN` w/ AlphaV1 labels | `raw_mhtcn_alphaV1.pt` |
| V3 | AlphaV2 (category-prob) | NullFilter | — (baseline) | — |
| V4 | AlphaV2 | RawFeatureMHTCNFilter | `ResearchTCN` w/ AlphaV2 labels | `raw_mhtcn_alphaV2.pt` |
| V5 | AlphaV1 | ProbabilisticMHTCNFilter | `ProbabilisticTCN` w/ AlphaV1 labels | `prob_mhtcn_alphaV1.pt` |
| V6 | AlphaV2 | ProbabilisticMHTCNFilter | `ProbabilisticTCN` w/ AlphaV2 labels | `prob_mhtcn_alphaV2.pt` |

**4 models** trained (V1 and V3 are no-MH-TCN baselines).

## B.3 Model Architectures

**ResearchTCN** (`research/mhtcn_filters/raw_feature.py`):
- Input: `(batch, 60, n_features)` — raw OHLC + engineered features
- Output: single `g_factor ∈ (0, 1]` via sigmoid
- Hidden: 64, 4 layers, kernel_size=3, dilations=[1,2,4,8]

**ProbabilisticTCN** (`research/mhtcn_filters/probabilistic.py`):
- Input: `(batch, 64, 8)` — 8 probability channels over 64 timesteps
  - Channels 0–5: P_trend, P_momentum, P_oscillator, P_volatility, P_volume, P_structure
  - Channel 6: P_alpha_final
  - Channel 7: regime_scalar
- Output: `g_factor`, `survival`, `validity` — all via sigmoid
- Hidden: 32, 3 layers, kernel_size=3, dilations=[1,2,4]

## B.4 Training Pipeline

Uses `research/train_mhtcn_variants.py → run_all()`:

1. **Phase 0** — Load OHLCV, generate 278 features (reuses `training/retrain_mhtcn`), detect regimes
2. **Phase 1** — Run AlphaHeadV1 + AlphaHeadV2 on every bar → direction arrays
3. **Phase 2** — `compute_persistence_labels()` → leak-free binary labels (signal persisted for 20 bars?)
4. **Phase 3** — Train 2× `ResearchTCN` (raw features + labels_v1/v2) → V2, V4
5. **Phase 4** — Build probability sequences, train 2× `ProbabilisticTCN` → V5, V6

Walk-forward protocol (`research/mhtcn_filters/training.py`):
- Split: 70/20/10 chronological
- BCELoss, AdamW (lr=1e-3), CosineAnnealing, patience=8, epochs=50

## B.5 Execution

```powershell
& d:\myBot\.venv312\scripts\activate.ps1

# Train all 4 research MH-TCN models
python -m research.train_mhtcn_variants `
    --data E:\pyProject\data\raw\EURUSD_H1_latest.csv `
    --output-dir E:\pyProject\pyForex-assets\models\research `
    --epochs 50 --device auto

# Run the full 6-variant experiment
python -m research.run_experiment `
    --data E:\pyProject\data\raw\EURUSD_H1_latest.csv `
    --output-dir E:\pyProject\pyForex-assets\models\research `
    --neg-controls --device auto
```

## B.6 Outputs

```
E:\pyProject\pyForex-assets\models\research\
├── raw_mhtcn_alphaV1.pt          # ResearchTCN for V2
├── raw_mhtcn_alphaV2.pt          # ResearchTCN for V4
├── prob_mhtcn_alphaV1.pt         # ProbabilisticTCN for V5
├── prob_mhtcn_alphaV2.pt         # ProbabilisticTCN for V6
├── training_log.json             # Metrics / history for all 4 runs
├── cached_features.parquet       # Pre-computed feature matrix
├── cached_signals_v1.npz         # AlphaV1 signals per bar
├── cached_signals_v2.npz         # AlphaV2 signals per bar
└── research_report_*.md          # Experiment comparison report
```

## B.7 Validation & Negative Controls

1. **Key comparisons**:
   - V1 vs V2 → Does raw MH-TCN add value to AlphaV1?
   - V3 vs V4 → Does raw MH-TCN add value to AlphaV2?
   - V1 vs V5 → Does probabilistic MH-TCN add value to AlphaV1?
   - V3 vs V6 → Does probabilistic MH-TCN add value to AlphaV2?
   - V2 vs V5 → Raw vs probabilistic MH-TCN (same alpha)?
   - V4 vs V6 → Raw vs probabilistic MH-TCN (same alpha)?

2. **Negative controls**: shuffled labels, random filter, inverted filter.

3. **Metrics**: Sharpe, max drawdown, win rate, total PnL, Brier score, trade count.

---

# Full Execution Order

```powershell
& d:\myBot\.venv312\scripts\activate.ps1

# ──────────────────────────────────────────────────────
# PART A: Risk Management MultiHeadTCN (9 models)
#   3 profiles × 3 TFs each for the 3TF decision cascade
# ──────────────────────────────────────────────────────
python scripts/train_all_models.py --models tcn --profiles all --epochs 120 --device auto

# ──────────────────────────────────────────────────────
# PART B: Research Variant MH-TCN (4 models)
# ──────────────────────────────────────────────────────
python -m research.train_mhtcn_variants `
    --data E:\pyProject\data\raw\EURUSD_H1_latest.csv `
    --output-dir E:\pyProject\pyForex-assets\models\research `
    --epochs 50 --device auto

# ──────────────────────────────────────────────────────
# PART B validation: 6-variant experiment + neg controls
# ──────────────────────────────────────────────────────
python -m research.run_experiment `
    --data E:\pyProject\data\raw\EURUSD_H1_latest.csv `
    --output-dir E:\pyProject\pyForex-assets\models\research `
    --neg-controls --device auto
```

**Recommended order**: Part A first (risk management models are needed by the
live 3TF system), then Part B (research variants can reference Part A weights
if needed).

---

# Complete Output Map

```
E:\pyProject\pyForex-assets\
├── models/
│   ├── weights/                              # PART A — Risk Management (9 models)
│   │   ├── multihead_tcn_SCALP_M5.pth       #   SCALP LTF
│   │   ├── multihead_tcn_SCALP_M15.pth      #   SCALP MTF
│   │   ├── multihead_tcn_SCALP_H1.pth       #   SCALP HTF
│   │   ├── multihead_tcn_INTRADAY_M15.pth   #   INTRADAY LTF
│   │   ├── multihead_tcn_INTRADAY_H1.pth    #   INTRADAY MTF
│   │   ├── multihead_tcn_INTRADAY_H4.pth    #   INTRADAY HTF
│   │   ├── multihead_tcn_SWING_H1.pth       #   SWING LTF
│   │   ├── multihead_tcn_SWING_H4.pth       #   SWING MTF
│   │   ├── multihead_tcn_SWING_D1.pth       #   SWING HTF
│   │   └── retrain_report.json
│   └── research/                             # PART B — Research Variants (4 models)
│       ├── raw_mhtcn_alphaV1.pt              #   ResearchTCN for V2
│       ├── raw_mhtcn_alphaV2.pt              #   ResearchTCN for V4
│       ├── prob_mhtcn_alphaV1.pt             #   ProbabilisticTCN for V5
│       ├── prob_mhtcn_alphaV2.pt             #   ProbabilisticTCN for V6
│       ├── training_log.json
│       └── cached_*.parquet / *.npz
└── data/
    └── mt5/EURUSD/                           # Source data (M5, M15, H1 present; H4, D1 need import)
```
