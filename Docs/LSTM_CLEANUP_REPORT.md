# LSTM to TCN Migration - Cleanup Report
**Project:** pyForex-1
**Generated:** 2025-12-12
**Status:** ⚠️ INCOMPLETE - Legacy LSTM files still present

---

## Executive Summary

The project has **partially migrated** from LSTM to TCN models, but **LSTM files were NOT fully removed**. There are:

- ✅ **Backward compatibility aliases** in place (SimpleLSTMPredictor → RiskAwareTCNPredictor)
- ⚠️ **3 LSTM implementation files** still in the codebase (742 lines total)
- ⚠️ **Broken imports** in main.py trying to use non-existent classes
- ⚠️ **Multiple references** to LSTM in CLI, documentation, and training scripts

---

## Current State

### ✅ What Works (TCN with LSTM Aliases)

The following **legacy names** are **aliases to TCN** and work correctly:

1. **`SimpleLSTMPredictor`** → `RiskAwareTCNPredictor`
   - Defined in: `inference/predictor.py:516`
   - Exported from: `inference/__init__.py`
   - Status: ✅ **Working alias** - no actual LSTM code

2. **`TCNPredictor`** → `RiskAwareTCNPredictor`
   - Defined in: `inference/predictor.py:515`
   - Status: ✅ **Working alias**

### ❌ What's Broken

1. **`SimpleLSTMStrategy`** in `main.py:445, 460`
   - **ERROR:** This class does NOT exist
   - Imported from: `strategies.neural_hybrid`
   - Actual exports: Only `NeuralHybridStrategy` (which uses TCN)
   - Impact: Backtest command with `--strategy lstm` will FAIL

### ⚠️ Legacy LSTM Files Still Present

| File | Size | Status | Action Needed |
|------|------|--------|---------------|
| `models/lstm.py` | 161 lines | Still imported in some places | ⚠️ EVALUATE |
| `training/train_lstm.py` | 223 lines | Referenced in main.py | ⚠️ EVALUATE |
| `training/train_lstm_enhanced.py` | 358 lines | Imported by analysis scripts | ⚠️ EVALUATE |
| **Total** | **742 lines** | | |

---

## Detailed Analysis

### 1. `models/lstm.py` (161 lines)

**Description:** LSTM model implementation with:
- `LSTMModel` - Bidirectional LSTM
- `LSTMWithAttention` - LSTM with attention mechanism

**Still imported by:**
- `models/tcn.py:636` - Fallback import (commented usage)
- `training/train_lstm.py:17` - Training script
- `utils/checkpoint_loader.py:154` - Legacy checkpoint loading

**Usage context:**
```python
# models/tcn.py:636 (fallback for old checkpoints)
from models.lstm import LSTMModel

# utils/checkpoint_loader.py:154 (loading old LSTM checkpoints)
from models.lstm import LSTMModel
```

**Recommendation:** ⚠️ **KEEP FOR NOW**
- Needed for loading old LSTM checkpoints
- Used as fallback in checkpoint loader
- Remove only if all old checkpoints are converted to TCN

---

### 2. `training/train_lstm.py` (223 lines)

**Description:** Training script for LSTM model

**Still referenced by:**
- `main.py:547-549` - CLI train command for LSTM
  ```python
  from training.train_lstm import train_lstm_model
  train_lstm_model(...)
  ```

**Imports:**
- `models.lstm.LSTMModel`
- `models.lstm.LSTMWithAttention`

**CLI Usage:**
```bash
python main.py train lstm --epochs 50  # This still works
```

**Recommendation:** ⚠️ **REMOVE** (if no longer training LSTM models)
- The project uses TCN now
- Training new LSTM models is not needed
- Only keep if intentionally maintaining dual model support

---

### 3. `training/train_lstm_enhanced.py` (358 lines)

**Description:** Enhanced LSTM training with feature engineering

**Still imported by:**
- `analysis/evaluate_horizon.py:13`
- `analysis/evaluate_model.py:16`
- `training/auto_retrain.py:12`
- `training/train_dynamic.py:16`
- `utils/checkpoint_loader.py:147`

**Exports:**
- `EnhancedLSTM` - Model class
- `TOP_FEATURES` - Feature list constant
- `train_enhanced_lstm()` - Training function

**Usage examples:**
```python
# analysis/evaluate_horizon.py:13
from training.train_lstm_enhanced import EnhancedLSTM, TOP_FEATURES

# training/auto_retrain.py:12
from training.train_lstm_enhanced import train_enhanced_lstm
```

**Recommendation:** ⚠️ **EVALUATE**
- Used by analysis scripts (some of which are already marked as potentially unused)
- Used by auto_retrain.py for LSTM retraining
- Check if `TOP_FEATURES` is still needed elsewhere

---

### 4. Analysis Scripts Using LSTM

Legacy analysis scripts still reference LSTM:

| Script | Status | LSTM Import |
|--------|--------|-------------|
| `analysis/evaluate_horizon.py` | Legacy/unused | `EnhancedLSTM, TOP_FEATURES` |
| `analysis/evaluate_model.py` | Legacy/unused | `EnhancedLSTM, TOP_FEATURES` |
| `analysis/feature_importance.py` | Superseded by train_tcn_enhanced | N/A |

**Recommendation:** These scripts were already flagged as potentially unused in the previous analysis.

---

### 5. Documentation References

LSTM is still mentioned in:

| File | Reference |
|------|-----------|
| `main.py:11` | "Model training (LSTM, ViT, Fusion...)" |
| `main.py:927-928` | CLI args for LSTM (--seq-len, --attention) |
| `main.py:943` | `--simple` flag for "simple LSTM predictor" |
| `INTEGRATION_GUIDE.md:54` | Import example using `train_lstm_enhanced` |
| `pyForex Decision Process Update Summary.md` | Migration notes |
| Various risk_management guides | Migration examples |

---

## Issues Found

### 🔴 CRITICAL: Broken Import in main.py

**Location:** `main.py:445`
```python
from strategies.neural_hybrid import NeuralHybridStrategy, SimpleLSTMStrategy
```

**Problem:**
- `SimpleLSTMStrategy` does **NOT exist** in `strategies/neural_hybrid.py`
- Only `NeuralHybridStrategy` is exported
- Used in backtest command: `strategy_map["lstm"] = SimpleLSTMStrategy`

**Impact:**
- Import will fail when trying to run backtests
- `python main.py backtest --strategy lstm` will crash

**Fix needed:** Remove or create alias

---

## Recommendations

### Immediate Actions (Fix Broken Code)

1. **Fix main.py broken import:**
   ```python
   # Option A: Remove SimpleLSTMStrategy entirely
   from strategies.neural_hybrid import NeuralHybridStrategy
   strategy_map = {
       "neural": NeuralHybridStrategy,
       # "lstm": removed - use "neural" instead
   }

   # Option B: Create alias to NeuralHybridStrategy
   from strategies.neural_hybrid import NeuralHybridStrategy
   SimpleLSTMStrategy = NeuralHybridStrategy  # Alias
   ```

### Decision Point: Complete LSTM Removal?

You need to decide: **Keep LSTM support or remove completely?**

#### Option A: 🗑️ Complete LSTM Removal

**Remove these files:**
- `models/lstm.py` (161 lines)
- `training/train_lstm.py` (223 lines)
- `training/train_lstm_enhanced.py` (358 lines)
- `analysis/evaluate_horizon.py` (if not needed)
- `analysis/evaluate_model.py` (if not needed)

**Update these files:**
- `main.py` - Remove LSTM train command and SimpleLSTMStrategy
- `training/auto_retrain.py` - Remove LSTM retraining
- `utils/checkpoint_loader.py` - Remove LSTM checkpoint loading (or keep as legacy)

**Benefits:**
- Cleaner codebase (-742 lines)
- No confusion about which model to use
- Reduced maintenance burden

**Risks:**
- Can't load old LSTM checkpoints
- Can't retrain LSTM models if needed
- Breaking change for any external code using LSTM

#### Option B: 🔧 Keep LSTM as Legacy Support

**Keep the files but:**
- Document clearly that TCN is the primary model
- Mark LSTM as deprecated/legacy
- Fix broken imports
- Ensure all LSTM paths still work

**Benefits:**
- Backward compatibility with old checkpoints
- Can still load/use LSTM models if needed
- Gradual migration path

**Risks:**
- More code to maintain
- Confusion about which model is current

---

## Migration Checklist

If you choose **Option A (Complete Removal)**, follow this checklist:

### Phase 1: Fix Immediate Issues
- [ ] Fix `main.py:445` - Remove or alias `SimpleLSTMStrategy`
- [ ] Test that main.py imports work
- [ ] Update CLI help text to remove LSTM references

### Phase 2: Remove LSTM Training
- [ ] Remove `training/train_lstm.py`
- [ ] Remove `training/train_lstm_enhanced.py`
- [ ] Remove LSTM training from `main.py` train command
- [ ] Update `training/auto_retrain.py` to remove LSTM retraining
- [ ] Remove `training/train_dynamic.py` if it's LSTM-specific

### Phase 3: Remove LSTM Model
- [ ] Decide: Keep `models/lstm.py` for legacy checkpoint loading?
- [ ] If removing: Update `utils/checkpoint_loader.py`
- [ ] If removing: Test that no code breaks
- [ ] Remove LSTM imports from `models/tcn.py`

### Phase 4: Remove LSTM Analysis Scripts
- [ ] Remove `analysis/evaluate_horizon.py`
- [ ] Remove `analysis/evaluate_model.py`
- [ ] Check if `TOP_FEATURES` is used elsewhere

### Phase 5: Update Documentation
- [ ] Update `main.py` docstring (remove LSTM mention)
- [ ] Update CLI help text
- [ ] Update `INTEGRATION_GUIDE.md`
- [ ] Add migration note in README.md
- [ ] Archive old LSTM documentation

### Phase 6: Remove Compatibility Aliases (Optional)
- [ ] Remove `SimpleLSTMPredictor` alias (breaking change)
- [ ] Update any code using the old name
- [ ] Update documentation

---

## Testing After Cleanup

After removing LSTM files, test:

```bash
# Test CLI imports
python main.py --help
python main.py train --help
python main.py backtest --help

# Test actual functionality
python main.py train tcn --data data/sample.csv --epochs 1
python main.py predict --model models/weights/tcn_best.pt --data data/sample.csv

# Test imports in Python
python -c "from inference import RiskAwareTCNPredictor; print('OK')"
python -c "from strategies import NeuralHybridStrategy; print('OK')"
```

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| LSTM source files | 3 | ⚠️ Still present (742 lines) |
| Working aliases | 2 | ✅ SimpleLSTMPredictor, TCNPredictor |
| Broken imports | 1 | 🔴 SimpleLSTMStrategy in main.py |
| Files importing LSTM | 10+ | ⚠️ Various scripts |
| Documentation refs | 5+ | ⚠️ Need updating |

**Recommendation:**
1. **Immediate:** Fix the broken `SimpleLSTMStrategy` import in main.py
2. **Decision:** Choose between complete removal or legacy support
3. **If removing:** Follow the migration checklist above
4. **If keeping:** Document as deprecated and ensure all paths work

---

## Appendix: All LSTM References

### Files with LSTM in code
```
./analysis/evaluate_horizon.py - import EnhancedLSTM, TOP_FEATURES
./analysis/evaluate_model.py - import EnhancedLSTM, TOP_FEATURES
./inference/predictor.py - SimpleLSTMPredictor alias (works)
./main.py - Multiple references (some broken)
./models/fusion.py - May reference LSTM
./models/lstm.py - Implementation
./models/tcn.py - Fallback import, compatibility comment
./strategies/style_strategies.py - Import SimpleLSTMPredictor (works via alias)
./tests/conftest.py - Test fixtures
./tests/test_main.py - Tests
./training/auto_retrain.py - Import train_enhanced_lstm
./training/train_dynamic.py - Import EnhancedLSTM
./training/train_lstm.py - Implementation
./training/train_lstm_enhanced.py - Implementation
./training/train_tcn.py - Legacy comparison
./trend_detection/trend_features.py - Reference in comments/logic
./utils/checkpoint_loader.py - Load LSTM checkpoints
```

### Commands to find all LSTM references
```bash
# Find files mentioning LSTM
find . -name "*.py" -not -path "./*venv*" | xargs grep -l "lstm" -i

# Find actual imports
grep -r "from.*lstm\|import.*lstm" --include="*.py" -i

# Find class definitions
grep -r "class.*LSTM" --include="*.py"
```
