# ================================================================
# MH-TCN Full Training Pipeline — 13 Models
# ================================================================
# PART A: Risk Management MultiHeadTCN (9 models = 3 profiles × 3 TFs)
#   → 3TF cascade: HTF bias → MTF validation → LTF trigger
#   → SL/TP (quantile head), position sizing (outcome head), direction
# PART B: Research Variant MH-TCN (4 models)
#   → g_factor confidence modulation for 6-variant experiment
# ================================================================

& d:\myBot\.venv312\scripts\activate.ps1

# ── PART A: Risk Management (3 profiles × 3 TFs = 9 MultiHeadTCN models) ──
Write-Host "===== PART A: Risk Management MultiHeadTCN (9 models) =====" -ForegroundColor Cyan
# Uses scripts/train_all_models.py which iterates all 3 TFs per profile:
#   SCALP:    M5 (LTF), M15 (MTF), H1 (HTF)
#   INTRADAY: M15 (LTF), H1 (MTF), H4 (HTF)
#   SWING:    H1 (LTF), H4 (MTF), D1 (HTF)
python scripts/train_all_models.py --models tcn --profiles all --epochs 120 --device auto

# ── PART B: Research Variants (4 lightweight MH-TCN models) ──
Write-Host "===== PART B: Research Variant MH-TCN =====" -ForegroundColor Cyan
python -m research.train_mhtcn_variants `
    --data "E:\pyProject\data\raw\EURUSD_H1_latest.csv" `
    --output-dir "E:\pyProject\pyForex-assets\models\research" `
    --epochs 50 `
    --batch-size 64 `
    --lr 1e-3 `
    --label-horizon 20 `
    --persistence-threshold 0.0005 `
    --max-features 64 `
    --warmup 200 `
    --device auto

# ── PART B Validation: 6-variant experiment + negative controls ──
Write-Host "===== PART B Validation: 6-Variant Experiment =====" -ForegroundColor Cyan
python -m research.run_experiment `
    --data "E:\pyProject\data\raw\EURUSD_H1_latest.csv" `
    --output-dir "E:\pyProject\pyForex-assets\models\research" `
    --neg-controls --device auto
