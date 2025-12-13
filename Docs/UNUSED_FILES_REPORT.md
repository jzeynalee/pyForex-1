# Unused Files Analysis Report
**Project:** pyForex-1
**Generated:** 2025-12-12
**Total Python Files:** 138
**Potentially Unused:** 9 files

---

## Summary

This report identifies Python files in the pyForex-1 project that appear to be unused based on import analysis. Files are categorized by their likely status and whether they can be safely removed.

---

## ✅ SAFE TO REMOVE (4 files)

These files are clearly unused and can be safely deleted:

### 1. `tests/conftest_01.py` (5,070 bytes)
- **Status:** Duplicate/legacy test configuration
- **Reason:** The project uses `tests/conftest.py` as the main test configuration. This appears to be an old version.
- **Action:** ✅ DELETE

### 2. `tests/utils_conftest.py` (7,720 bytes)
- **Status:** Unused test utilities
- **Reason:** Not imported anywhere in the test suite
- **Action:** ✅ DELETE

### 3. `analyze_unused.py` (6,375 bytes)
- **Status:** Temporary analysis script (created during this analysis)
- **Reason:** Not part of the original project
- **Action:** ✅ DELETE

### 4. `trend_detection/mtf_analyzer.py` (104 lines, ~3KB)
- **Status:** Superseded by newer version
- **Reason:**
  - `mtf_analyzer_v2.py` exists (493 lines) and is actively imported
  - `trend_detection/__init__.py` imports from `mtf_analyzer_v2`
  - `mtf_trend_detector.py` imports from `mtf_analyzer_v2`
  - The v1 file is not imported anywhere
- **Action:** ✅ DELETE (v1 is superseded by v2)

**Total space saved: ~22 KB**

---

## ⚠️ KEEP - STANDALONE SCRIPTS (5 files)

These files are **NOT imported** but serve as **standalone CLI tools** or **examples**. They should be kept.

### 1. `analysis/evaluate_horizon.py` (5,929 bytes)
- **Status:** Standalone evaluation script
- **Purpose:** CLI tool for horizon-based model evaluation
- **References:** Not mentioned in documentation
- **Action:** ⚠️ KEEP (standalone tool)
- **Note:** Consider documenting usage in README

### 2. `analysis/evaluate_model.py` (5,589 bytes)
- **Status:** Standalone evaluation script
- **Purpose:** Generic model evaluation tool
- **References:** Not mentioned in documentation
- **Action:** ⚠️ KEEP (standalone tool)
- **Note:** May be legacy - verify if still functional

### 3. `analysis/evaluate_tcn_horizon.py` (15,733 bytes)
- **Status:** Standalone evaluation script
- **Purpose:** TCN-specific horizon evaluation
- **References:**
  - **Documented** in `INTEGRATION_GUIDE.md`
  - Example usage: `python analysis/evaluate_tcn_horizon.py --model models/weights/tcn_enhanced_best.pt --horizon 5`
- **Action:** ✅ KEEP (documented in integration guide)

### 4. `analysis/evaluate_tcn_model.py` (19,456 bytes)
- **Status:** Standalone evaluation script
- **Purpose:** TCN model evaluation with feature loading from checkpoint
- **References:**
  - **Documented** in `INTEGRATION_GUIDE.md`
  - Example usage: `python analysis/evaluate_tcn_model.py --model models/weights/tcn_enhanced_best.pt`
- **Action:** ✅ KEEP (documented in integration guide)

### 5. `analysis/feature_importance.py` (3,142 bytes)
- **Status:** Standalone analysis script
- **Purpose:** Feature importance analysis (note: functionality now integrated into `train_tcn_enhanced.py`)
- **References:**
  - The comment in `train_tcn_enhanced.py:11` mentions: "Key improvements over separate feature_importance.py + train scripts"
  - This suggests the functionality was integrated and this is now legacy
- **Action:** ⚠️ CONSIDER REMOVING (functionality integrated into train_tcn_enhanced.py)
- **Recommendation:** If the standalone script is still useful for ad-hoc analysis, keep it. Otherwise, remove.

### 6. `examples/prop_firm_example.py` (8,273 bytes)
- **Status:** Example/documentation file
- **Purpose:** Shows how to configure prop firm trading
- **References:** Not imported (by design - it's an example)
- **Action:** ✅ KEEP (example/documentation)

---

## 📊 Detailed Analysis

### Analysis Method
1. Scanned all Python files in the project (excluding venv, __pycache__, .git)
2. Extracted all import statements (direct imports and relative imports)
3. Built module dependency graph
4. Identified files not imported anywhere
5. Manual verification using grep for references in documentation and comments

### Files Excluded from Analysis
- `__init__.py` files (package structure)
- Test files (`test_*.py`) - kept by default
- Entry point scripts (main.py, run_tests.py)
- Virtual environment files

### False Positives (Files marked unused but actually used)
- **None identified** - All truly unused files have been verified

---

## 🎯 Recommendations

### Immediate Actions

1. **Delete these 4 files:**
   ```bash
   rm tests/conftest_01.py
   rm tests/utils_conftest.py
   rm analyze_unused.py
   rm trend_detection/mtf_analyzer.py
   ```

2. **Verify and potentially delete:**
   ```bash
   # Check if feature_importance.py is still useful as standalone tool
   rm analysis/feature_importance.py  # if not needed
   ```

### Documentation Improvements

1. **Create analysis/README.md** to document the standalone analysis scripts:
   ```markdown
   # Analysis Tools

   ## Model Evaluation
   - `evaluate_tcn_model.py` - Evaluate TCN model performance
   - `evaluate_tcn_horizon.py` - Horizon-based TCN evaluation
   - `evaluate_model.py` - Generic model evaluation (legacy)
   - `evaluate_horizon.py` - Generic horizon evaluation (legacy)
   ```

2. **Update main README.md** to mention analysis tools

3. **Consider consolidating** `evaluate_model.py` and `evaluate_horizon.py` if they're truly legacy

### Future Maintenance

1. **Use type hints and imports** to make dependency tracking easier
2. **Document CLI scripts** in README files
3. **Mark deprecated files** with comments before removing
4. **Run this analysis periodically** to catch unused code early

---

## 📝 Notes

### About `mtf_analyzer.py` vs `mtf_analyzer_v2.py`
- The v2 version has 4.7x more code (493 vs 104 lines)
- v2 is actively imported by the main codebase
- v1 has no references except in this analysis
- **Safe to delete v1**

### About Analysis Scripts
- The `analysis/` folder contains standalone evaluation tools
- Two are documented (`evaluate_tcn_model.py`, `evaluate_tcn_horizon.py`)
- Two appear to be older versions (`evaluate_model.py`, `evaluate_horizon.py`)
- Consider consolidating or documenting all of them

### About Test Files
- `conftest_01.py` and `utils_conftest.py` are clearly unused
- Main test configuration is in `conftest.py`
- No references found in any test files

---

## 🔍 How to Use This Report

1. **Review the "SAFE TO REMOVE" section** - These can be deleted immediately
2. **Check the "KEEP" section** - Understand why these files should be kept
3. **Follow the recommendations** - Improve documentation and cleanup
4. **Re-run the analysis** - After making changes, verify no new unused files appear

---

## Appendix: Command to Reproduce Analysis

```bash
cd d:\myBot\pyForex-1
python analyze_unused.py
```

Or manually:
```bash
# Find all Python files
find . -name "*.py" -not -path "./*venv*" -not -path "./__pycache__/*"

# Search for imports of specific modules
grep -r "import module_name" --include="*.py"
grep -r "from.*module_name" --include="*.py"
```
