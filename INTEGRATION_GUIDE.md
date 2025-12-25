# pyForex TCN Feature Integration - Migration Guide

## Overview

This update integrates feature importance discovery with TCN training, ensuring features are stored WITH model checkpoints. This eliminates the need for hardcoded `TOP_FEATURES` imports.

## Files to Update/Add

### New Files (Add to project)
| File | Purpose |
|------|---------|
| `training/train_tcn_enhanced.py` | Training with auto feature discovery |
| `analysis/evaluate_tcn_model.py` | Evaluation (loads features from checkpoint) |
| `analysis/evaluate_tcn_horizon.py` | Horizon-based evaluation |
| `utils/checkpoint_loader.py` | Unified checkpoint loading utility |

### Updated Files (Replace existing)
| File | Changes |
|------|---------|
| `utils/feature_adapter.py` | Added V3 loader with checkpoint integration |
| `inference/predictor.py` | Uses checkpoint_loader, removes hardcoded features |

---

## Migration Steps

### Step 1: Add New Files
Copy these files to your project:
```
training/train_tcn_enhanced.py
analysis/evaluate_tcn_model.py
analysis/evaluate_tcn_horizon.py
utils/checkpoint_loader.py
```

### Step 2: Update Existing Files
Replace these files in your project:
```
utils/feature_adapter.py
inference/predictor.py
```

### Step 3: Retrain Model
Train a new model to get checkpoint with features:
```bash
python training/train_tcn_enhanced.py \
    --data data/raw/eurusd_latest.csv \
    --profile INTRADAY
```

### Step 4: Update Imports (if needed)
Old code:
```python
from training.train_lstm_enhanced import EnhancedLSTM, TOP_FEATURES
```

New code:
```python
from utils.checkpoint_loader import load_model, load_features
model, features = load_model("models/weights/tcn_enhanced_best.pt")
```

---

## Usage Examples

### Training
```bash
# Auto-discover features and train
python training/train_tcn_enhanced.py --data data/raw/eurusd_latest.csv

# With profile (affects architecture and feature priorities)
python training/train_tcn_enhanced.py --data data/raw/eurusd_latest.csv --profile SCALP
python training/train_tcn_enhanced.py --data data/raw/eurusd_latest.csv --profile SWING

# Use all features (skip selection)
python training/train_tcn_enhanced.py --data data/raw/eurusd_latest.csv --skip-feature-selection

# Custom features
python training/train_tcn_enhanced.py --data data/raw/eurusd_latest.csv --features "rsi_14,atr_14,macd"
```

### Evaluation
```bash
# Standard evaluation
python analysis/evaluate_tcn_model.py --model models/weights/tcn_enhanced_best.pt

# Horizon-based (fixed holding period)
python analysis/evaluate_tcn_horizon.py --model models/weights/tcn_enhanced_best.pt --horizon 5
```

### Inference
```python
from inference.predictor import TCNPredictor, create_predictor

# Create predictor (features loaded automatically)
predictor = TCNPredictor(checkpoint_path="models/weights/tcn_enhanced_best.pt")

# Make prediction
result = predictor.predict(df_window)
print(f"Signal: {result.signal_name}, Confidence: {result.confidence:.2%}")

# Get trading signal with threshold
signal = predictor.get_signal(df_window, threshold=0.6)
print(f"Trade: {signal}")  # 'BUY', 'SELL', or 'HOLD'
```

### Data Loading
```python
from utils.feature_adapter import EnhancedDataLoaderV3, load_data_for_evaluation

# Load with checkpoint's features
loader = EnhancedDataLoaderV3.from_checkpoint("models/weights/tcn_enhanced_best.pt")
df = loader.load_csv("data/raw/eurusd_latest.csv")

# Or use convenience function
X_test, y_test, close_prices, features = load_data_for_evaluation(
    data_path="data/raw/eurusd_latest.csv",
    checkpoint_path="models/weights/tcn_enhanced_best.pt",
)
```

### Checkpoint Inspection
```python
from utils.checkpoint_loader import ModelLoader, print_checkpoint_summary

# Quick summary
print_checkpoint_summary("models/weights/tcn_enhanced_best.pt")

# Detailed access
loader = ModelLoader("models/weights/tcn_enhanced_best.pt")
print(loader.summary())

features = loader.get_features()
config = loader.get_config()
importance = loader.get_feature_importance()
metrics = loader.get_metrics()
```

---

## Checkpoint Structure

New checkpoints include:
```python
{
    'model_state': {...},           # Model weights
    'feature_columns': [...],       # ⭐ Features used (NEW!)
    'feature_importance': {...},    # Full importance scores
    'config': {
        'training': {...},          # Hyperparameters
        'feature': {...},           # Feature selection config
        'model': {...},             # Architecture config
        'feature_schema_version': 'pa_v1',  # Feature generator version
    },
    'training_history': {...},      # Loss/accuracy curves
    'profile': 'SCALP',             # Trading profile
    'metrics': {...},               # Evaluation metrics
    'created_at': '...',            # Timestamp
}
```

### Artifact Naming (Option A)

To keep compatibility with existing consumers while retaining traceability, training may write both:

- Legacy filename: `*_best.pt` / `*.pth`
- Schema-tagged copy: `*_{feature_schema_version}_best.pt` / `*_{feature_schema_version}.pth`

Example:

- `models/weights/scalp_m5_best.pt`
- `models/weights/scalp_m5_pa_v1_best.pt`

---

## Backward Compatibility

### Old Checkpoints
If you have old checkpoints without `feature_columns`, you'll see:
```
CheckpointFormatError: Checkpoint doesn't contain feature_columns.
Please retrain with train_tcn_enhanced.py
```

### Legacy Code
These aliases are provided:
```python
# Old name still works
from inference.predictor import SimpleLSTMPredictor  # -> TCNPredictor

# Old data loader still works
from utils.feature_adapter import EnhancedDataLoaderV2
```

---

## Integration Points

### With Decision Engine
```python
from inference.predictor import TCNPredictor
from trading.decision_engine import DecisionEngine

predictor = TCNPredictor(checkpoint_path="models/weights/tcn_enhanced_best.pt")
engine = DecisionEngine()

# Get prediction
result = predictor.predict(df_window)

# Feed to decision engine
decision = engine.decide(
    pattern_probs=result.probabilities.tolist(),
    mtf_result=mtf_detector.detect(dfs_dict),
)
```

### With MTF Analysis
The predictor integrates seamlessly with the MTF system since both use the same checkpoint-based feature management.

---

## Troubleshooting

### "Checkpoint doesn't contain feature_columns"
→ Retrain model using `train_tcn_enhanced.py`

### "Feature 'xxx' not found"
→ Feature will be set to 0. This may affect accuracy. Consider retraining.

### Import errors
→ Ensure all files are in correct directories and `__init__.py` files exist.

---

## File Checksums (for verification)
After adding files, verify:
- `training/train_tcn_enhanced.py` - ~800 lines
- `analysis/evaluate_tcn_model.py` - ~400 lines
- `analysis/evaluate_tcn_horizon.py` - ~350 lines
- `utils/checkpoint_loader.py` - ~350 lines
- `utils/feature_adapter.py` - ~400 lines
- `inference/predictor.py` - ~550 lines
