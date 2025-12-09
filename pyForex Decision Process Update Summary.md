# pyForex Decision Process Update Summary

## Overview

This update consolidates the decision-making process to properly integrate:
- **TCN (Temporal Convolutional Network)** replacing LSTM
- **MTF (Multi-Timeframe) Analysis** for trend context
- **Enhanced Decision Engine** with MTF awareness

## Files Updated/Created

### 1. NEW: `utils/mtf_config.py`
**Purpose**: Configuration profiles for MTF analysis

Contains:
- `MTFProfile` dataclass for MTF configuration
- Preset profiles: `SCALP_PROFILE`, `INTRADAY_PROFILE`, `SWING_PROFILE`
- `Timeframe` enum and utility functions
- `get_profile()` factory function

**Usage**:
```python
from utils.mtf_config import get_profile, SCALP_PROFILE

profile = get_profile("SWING")
print(profile.timeframes)  # (H1, H4, D1)
print(profile.weights)     # {'H1': 0.20, 'H4': 0.45, 'D1': 0.35}
```

---

### 2. NEW: `utils/mtf_features.py`
**Purpose**: Feature engineering for MTF analysis

Contains:
- `MTFFeatureBuilder` class for comprehensive feature extraction
- Per-timeframe features: EMA, ADX, RSI, ATR, momentum
- Cross-timeframe confluence features
- `MTFFeatureSet` dataclass for structured output

**Usage**:
```python
from utils.mtf_features import MTFFeatureBuilder

builder = MTFFeatureBuilder()
feature_set = builder.build_features(dfs_dict, primary_tf="H1")
```

---

### 3. UPDATED: `inference/predictor.py`
**Changes**:
- ✅ TCN is now the default sequence model (was LSTM)
- ✅ `SimpleLSTMPredictor` → `TCNPredictor` (with backward-compatible alias)
- ✅ Weight loading updated: `tcn_best.pt` instead of `lstm_best.pt`
- ✅ Added deprecation warnings for LSTM usage

**Key Changes**:
```python
# OLD:
from models.lstm import LSTMModel
self.seq_model = LSTMModel(...)

# NEW:
from models.tcn import TCNModel
self.seq_model = TCNModel.from_profile(self.config.tcn_profile)
```

---

### 4. UPDATED: `trading/decision_engine.py`
**Changes**:
- ✅ Now integrates with `MTFTrendResult` from MTFTrendDetector
- ✅ Added `MTFDecisionEngine` class with built-in trend detector
- ✅ Enhanced decision logic with:
  - MTF alignment checks
  - Regime-aware filtering
  - Higher timeframe confirmation
  - Confluence scoring
- ✅ `DecisionConfig` dataclass for configuration
- ✅ Added `get_recommendation()` for comprehensive output

**New Classes**:
```python
# Standard decision engine
engine = DecisionEngine(config=DecisionConfig())
result = engine.decide(pattern_probs, trend_analysis=trend_dict)

# MTF-enabled decision engine (recommended)
engine = MTFDecisionEngine(profile="SWING")
recommendation = engine.get_recommendation(pattern_probs, dfs_dict)
```

---

### 5. UPDATED: `strategies/neural_hybrid.py`
**Changes**:
- ✅ Uses `MTFDecisionEngine` instead of simple `DecisionEngine`
- ✅ `SimpleLSTMStrategy` → `SimpleTCNStrategy` (with alias)
- ✅ Integrated MTF data preparation
- ✅ Confidence-based position sizing
- ✅ Enhanced logging with trend context

**Key Changes**:
```python
# OLD:
from inference.predictor import SimpleLSTMPredictor
self.trend_detector = FusionFXTrendDetector(ml_model=None)
self.decision_engine = DecisionEngine(threshold=0.65)

# NEW:
from inference.predictor import TCNPredictor
self.decision_engine = MTFDecisionEngine(
    profile=mtf_profile,
    config=decision_config,
)
```

---

### 6. UPDATED: `trading/live_trading_bot.py`
**Changes**:
- ✅ Loads TCN model instead of LSTM
- ✅ Uses `MTFDecisionEngine` for signal decisions
- ✅ Configurable profiles via CLI arguments
- ✅ Multi-timeframe data fetching
- ✅ Enhanced logging with trend information

**New CLI**:
```bash
python trading/live_trading_bot.py --tcn-profile INTRADAY --mtf-profile SWING
```

---

## Decision Flow (Updated)

```
┌─────────────────────────────────────────────────────────────────┐
│                    UPDATED DECISION PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. DATA COLLECTION                                             │
│     └── MTFDataProvider fetches M15, H1, H4 (or configured TFs) │
│                                                                 │
│  2. PATTERN RECOGNITION (Fast Brain)                            │
│     ├── TCNModel extracts sequence features (replaces LSTM)     │
│     ├── ViTExtractor extracts visual features                   │
│     ├── YOLODetector detects candlestick patterns               │
│     └── FusionNet combines → [P(BUY), P(SELL), P(HOLD)]         │
│                                                                 │
│  3. TREND ANALYSIS (Slow Brain)                                 │
│     ├── StructuralAnalyzer: Swing-based trend detection         │
│     ├── MTFAnalyzerV2: Cross-timeframe confluence               │
│     ├── RegimeClassifier: TRENDING/RANGING/VOLATILE             │
│     └── TrendClassifier (ML): Probabilistic confirmation        │
│                                                                 │
│  4. DECISION ENGINE (Prefrontal Cortex)                         │
│     ├── Pattern signal evaluation                               │
│     ├── Trend alignment check                                   │
│     ├── Regime filtering                                        │
│     ├── Higher TF confirmation                                  │
│     └── Final signal with confidence                            │
│                                                                 │
│  5. RISK MANAGEMENT                                             │
│     ├── Position sizing based on confidence                     │
│     ├── ATR-based SL/TP                                         │
│     └── Daily loss limits                                       │
│                                                                 │
│  6. EXECUTION                                                   │
│     └── Order sent to MT5                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Migration Guide

### If you're using `SimpleLSTMPredictor`:
```python
# Old (still works via alias)
from inference.predictor import SimpleLSTMPredictor
predictor = SimpleLSTMPredictor(weights_path="model.pt")

# New (recommended)
from inference.predictor import TCNPredictor
predictor = TCNPredictor(weights_path="tcn_best.pt", profile="INTRADAY")
```

### If you're using `DecisionEngine` directly:
```python
# Old
engine = DecisionEngine(threshold=0.65)
result = engine.decide(probs, trend_analysis)

# New (with MTF integration)
engine = MTFDecisionEngine(profile="SWING")
result = engine.decide(probs, mtf_result=mtf_trend_result)
# Or use get_recommendation() for full context
```

### If you're using `NeuralHybridStrategy`:
```python
# Old
strategy = NeuralHybridStrategy(data_provider, executor, risk_manager)

# New (specify MTF profile)
strategy = NeuralHybridStrategy(
    data_provider, executor, risk_manager,
    mtf_profile="SWING",  # NEW: MTF profile
)
```

---

## Profile Recommendations

| Trading Style | TCN Profile | MTF Profile | Timeframes |
|--------------|-------------|-------------|------------|
| Scalping     | SCALP       | SCALP       | M5, M15, H1 |
| Day Trading  | INTRADAY    | INTRADAY    | M15, H1, H4 |
| Swing        | SWING       | SWING       | H1, H4, D1 |

---

## Files to Add to Your Repo

Copy these files to your pyForex repository:

1. `utils/mtf_config.py` → NEW
2. `utils/mtf_features.py` → NEW  
3. `inference/predictor.py` → REPLACE existing
4. `trading/decision_engine.py` → REPLACE existing
5. `strategies/neural_hybrid.py` → REPLACE existing
6. `trading/live_trading_bot.py` → REPLACE existing

---

## Testing

After adding files, test with:

```python
# Test MTF config
from utils.mtf_config import get_profile
profile = get_profile("SWING")
print(f"Timeframes: {profile.timeframe_strings}")

# Test TCN predictor
from inference.predictor import TCNPredictor
predictor = TCNPredictor(profile="INTRADAY")
# result = predictor.predict(df)  # with real data

# Test decision engine
from trading.decision_engine import MTFDecisionEngine
engine = MTFDecisionEngine(profile="SWING")
# recommendation = engine.get_recommendation(probs, dfs_dict)
```