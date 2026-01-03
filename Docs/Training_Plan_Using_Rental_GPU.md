
- **Single RTX 4090 = you should treat GPU training as a single-job queue.**  
  Running **two heavy GPU trainings at the same time** (YOLO + ViT / Fusion, etc.) usually:
  - slows both jobs down a lot
  - risks CUDA OOM / fragmentation
  - makes results less reproducible

What you *can* parallelize safely:
- **CPU-heavy jobs** while the GPU queue is running:
  - meta-labeling (`scripts/train_all_models.py --models meta ...`) — sklearn, uses CPU
  - trend classifier ([training/train_trend_classifier.py](cci:7://file:///e:/pyProject/pyForex-1/training/train_trend_classifier.py:0:0-0:0)) — CPU
  - packaging/syncing artifacts, computing hashes, compressing folders, etc.

So the optimal plan is:
- **1 GPU queue (sequential)**
- **1 CPU lane (parallel background tasks)**

---

# New full plan (parallelized) for your rental specs

## 0) One-time setup (tmux layout)
Open `tmux` and create 2 windows:
- **Window 1: `gpu`** (sequential GPU queue)
- **Window 2: `cpu`** (parallel CPU tasks)

---

## 1) Pre-flight checks (run once)
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
nvidia-smi
```

Verify required files exist:
- `data/raw/EURUSD_M5_latest.csv`
- `data/raw/EURUSD_M15_latest.csv`
- `data/raw/EURUSD_H1_latest.csv`
- `data/raw/EURUSD_H4_latest.csv`
- `data/raw/EURUSD_D1_latest.csv`
- `datasets/yolo_scalp/data.yaml`
- `datasets/yolo_intraday/data.yaml`
- `datasets/yolo_swing/data.yaml`

Create log folder:
```bash
mkdir -p logs
```

---

## 2) CPU lane (Window: `cpu`) — start these and leave them running
### 2.1 Meta-labeling (all profiles)
```bash
nohup python scripts/train_all_models.py --models meta --profiles all --data-rows 1000000 > logs/meta.log 2>&1 &
```

### 2.2 Trend classifier (optional but recommended)
```bash
nohup python training/train_trend_classifier.py --data data/raw/EURUSD_H1_latest.csv --output models/trend_classifier.joblib > logs/trend_classifier.log 2>&1 &
```

You can check:
```bash
tail -f logs/meta.log
```

---

## 3) GPU queue (Window: `gpu`) — run sequentially in this exact order

### 3.1 Enhanced TCN per profile+TF (Fusion expects these filenames)
This is the critical part for Fusion resolution: it needs `models/weights/{profile}_{tf}_best.pt` (lowercase).

```bash
# SCALP: M5, M15, H1
python training/train_tcn_enhanced.py --profile SCALP --data data/raw/EURUSD_M5_latest.csv  --save-dir models/weights --name scalp_m5  --seq-len 60  --epochs 50 --batch-size 64 | tee logs/tcn_scalp_m5.log
python training/train_tcn_enhanced.py --profile SCALP --data data/raw/EURUSD_M15_latest.csv --save-dir models/weights --name scalp_m15 --seq-len 60  --epochs 50 --batch-size 64 | tee logs/tcn_scalp_m15.log
python training/train_tcn_enhanced.py --profile SCALP --data data/raw/EURUSD_H1_latest.csv  --save-dir models/weights --name scalp_h1  --seq-len 60  --epochs 50 --batch-size 64 | tee logs/tcn_scalp_h1.log

# INTRADAY: M15, H1, H4
python training/train_tcn_enhanced.py --profile INTRADAY --data data/raw/EURUSD_M15_latest.csv --save-dir models/weights --name intraday_m15 --seq-len 60 --epochs 50 --batch-size 64 | tee logs/tcn_intraday_m15.log
python training/train_tcn_enhanced.py --profile INTRADAY --data data/raw/EURUSD_H1_latest.csv  --save-dir models/weights --name intraday_h1  --seq-len 60 --epochs 50 --batch-size 64 | tee logs/tcn_intraday_h1.log
python training/train_tcn_enhanced.py --profile INTRADAY --data data/raw/EURUSD_H4_latest.csv  --save-dir models/weights --name intraday_h4  --seq-len 90 --epochs 50 --batch-size 64 | tee logs/tcn_intraday_h4.log

# SWING: H1, H4, D1
python training/train_tcn_enhanced.py --profile SWING --data data/raw/EURUSD_H1_latest.csv --save-dir models/weights --name swing_h1 --seq-len 60  --epochs 60 --batch-size 64 | tee logs/tcn_swing_h1.log
python training/train_tcn_enhanced.py --profile SWING --data data/raw/EURUSD_H4_latest.csv --save-dir models/weights --name swing_h4 --seq-len 90  --epochs 60 --batch-size 64 | tee logs/tcn_swing_h4.log
python training/train_tcn_enhanced.py --profile SWING --data data/raw/EURUSD_D1_latest.csv --save-dir models/weights --name swing_d1 --seq-len 120 --epochs 80 --batch-size 64 | tee logs/tcn_swing_d1.log
```

### 3.2 ViT (ALL profiles) + dataset generation
```bash
python training/finetune_vit.py --profile ALL --device cuda --generate-dataset | tee logs/vit_all.log
```

### 3.3 YOLO (ALL profiles)
```bash
python training/train_yolo_profiles.py --profile ALL --device 0 | tee logs/yolo_all.log
```

### 3.4 Exit optimizer PPO (GPU/CPU depends on implementation, keep after YOLO/ViT)
```bash
python scripts/train_all_models.py --models exit --profiles all --data-rows 1000000 --device auto | tee logs/exit_all.log
```

### 3.5 Decision Fusion for every profile timeframe (9 runs)
```bash
# SCALP: M5, M15, H1
python training/train_decision_fusion.py --profile SCALP --timeframe M5  --device cuda --epochs 50 | tee logs/fusion_scalp_m5.log
python training/train_decision_fusion.py --profile SCALP --timeframe M15 --device cuda --epochs 50 | tee logs/fusion_scalp_m15.log
python training/train_decision_fusion.py --profile SCALP --timeframe H1  --device cuda --epochs 50 | tee logs/fusion_scalp_h1.log

# INTRADAY: M15, H1, H4
python training/train_decision_fusion.py --profile INTRADAY --timeframe M15 --device cuda --epochs 50 | tee logs/fusion_intraday_m15.log
python training/train_decision_fusion.py --profile INTRADAY --timeframe H1  --device cuda --epochs 50 | tee logs/fusion_intraday_h1.log
python training/train_decision_fusion.py --profile INTRADAY --timeframe H4  --device cuda --epochs 50 | tee logs/fusion_intraday_h4.log

# SWING: H1, H4, D1
python training/train_decision_fusion.py --profile SWING --timeframe H1 --device cuda --epochs 50 | tee logs/fusion_swing_h1.log
python training/train_decision_fusion.py --profile SWING --timeframe H4 --device cuda --epochs 50 | tee logs/fusion_swing_h4.log
python training/train_decision_fusion.py --profile SWING --timeframe D1 --device cuda --epochs 50 | tee logs/fusion_swing_d1.log
```

---

## 4) Monitoring (while running)
- **GPU utilization**
```bash
watch -n 2 nvidia-smi
```

- **Log tail**
```bash
tail -f logs/fusion_intraday_h1.log
```

---

## 5) End: package artifacts for download
```bash
tar -czf artifacts_models_and_checkpoints.tgz \
  models/weights models/vit models/yolo models/decision_fusion \
  checkpoints \
  logs \
  training_results.json 2>/dev/null || true
```

---

# Safety / performance notes for parallelization
- **Do not** run YOLO + ViT + Fusion simultaneously on one 4090 (not worth it).
- **CPU lane is safe** and will meaningfully reduce wall-clock time:
  - meta-labeling + trend classifier can finish while GPU is busy.
- If you ever hit CUDA OOM during Fusion:
  - re-run that specific command with smaller `--batch_size` (Fusion supports `--batch_size`; default is 32).

---

## Status
- Delivered a **new full plan** optimized for **1× RTX 4090**, with **safe parallelization** (GPU queue + concurrent CPU jobs) and updated runnable command blocks.

## Master Script
'''bash
#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MASTER TRAINING SCRIPT: Forex Multi-Model Suite
# Optimized for: RTX 4090 (24GB VRAM) + 32-Core CPU + 90GB RAM
# GPU strategy: single-job queue (no concurrent heavy GPU trainings)
# CPU strategy: run CPU-only trainings in background while GPU queue runs
# ==============================================================================

mkdir -p logs models/weights models/vit models/yolo models/decision_fusion checkpoints

START_TIME=$(date +%s)

# -----------------------------
# Config knobs (tweak if needed)
# -----------------------------
DATA_ROWS="${DATA_ROWS:-1000000}"

TCN_EPOCHS_FAST="${TCN_EPOCHS_FAST:-50}"
TCN_EPOCHS_SWING_H1_H4="${TCN_EPOCHS_SWING_H1_H4:-60}"
TCN_EPOCHS_SWING_D1="${TCN_EPOCHS_SWING_D1:-80}"

TCN_BATCH_SIZE="${TCN_BATCH_SIZE:-128}"      # safer default than 512
TCN_LR="${TCN_LR:-1e-3}"

FUSION_EPOCHS="${FUSION_EPOCHS:-50}"
FUSION_BATCH_SIZE="${FUSION_BATCH_SIZE:-32}" # lower to 16 if OOM

# If you want to reuse M15 between profiles to save time:
# REUSE_M15=1 will skip training scalp_m15 and copy intraday_m15 -> scalp_m15 (or vice versa).
REUSE_M15="${REUSE_M15:-0}"
REUSE_M15_SOURCE="${REUSE_M15_SOURCE:-intraday}"  # intraday|scalp

require_file () {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[FATAL] Missing required file: $p" >&2
    exit 1
  fi
}

seq_len_for_tf () {
  local tf="$1"
  case "$tf" in
    M5)  echo 30 ;;
    M15) echo 60 ;;
    H1)  echo 60 ;;
    H4)  echo 90 ;;
    D1)  echo 120 ;;
    *)   echo 60 ;;
  esac
}

echo "--- [1/6] Pre-flight Checks ---"
python - << 'PY'
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY

# Required CSVs
require_file "data/raw/EURUSD_M5_latest.csv"
require_file "data/raw/EURUSD_M15_latest.csv"
require_file "data/raw/EURUSD_H1_latest.csv"
require_file "data/raw/EURUSD_H4_latest.csv"
require_file "data/raw/EURUSD_D1_latest.csv"

# YOLO datasets (you said they will be available)
require_file "datasets/yolo_scalp/data.yaml"
require_file "datasets/yolo_intraday/data.yaml"
require_file "datasets/yolo_swing/data.yaml"

echo "--- [2/6] Launching CPU Lane (Background) ---"
# Make CPU lane play nicely with GPU jobs
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"

nohup python scripts/train_all_models.py --models meta --profiles all --data-rows "${DATA_ROWS}" \
  > logs/meta_background.log 2>&1 &
META_PID=$!

nohup python training/train_trend_classifier.py \
  --data data/raw/EURUSD_H1_latest.csv \
  --output models/trend_classifier.joblib \
  > logs/trend_classifier.log 2>&1 &
TREND_PID=$!

echo "CPU lane PIDs: meta=${META_PID}, trend=${TREND_PID}"
echo "Tail meta log: tail -f logs/meta_background.log"

echo "--- [3/6] Training Enhanced TCNs (GPU Queue) ---"
train_tcn () {
  local profile="$1"
  local tf="$2"
  local data="data/raw/EURUSD_${tf}_latest.csv"
  local name="${profile,,}_${tf,,}"
  local seq_len
  seq_len="$(seq_len_for_tf "$tf")"

  echo "[TCN] profile=${profile} tf=${tf} seq_len=${seq_len} batch=${TCN_BATCH_SIZE} name=${name}"
  python training/train_tcn_enhanced.py \
    --profile "${profile}" \
    --data "${data}" \
    --save-dir models/weights \
    --name "${name}" \
    --seq-len "${seq_len}" \
    --epochs "${TCN_EPOCHS_FAST}" \
    --batch-size "${TCN_BATCH_SIZE}" \
    --lr "${TCN_LR}" \
    | tee -a logs/tcn_training.log
}

# SCALP: M5, M15, H1
train_tcn "SCALP" "M5"
if [[ "${REUSE_M15}" != "1" ]]; then
  train_tcn "SCALP" "M15"
fi
train_tcn "SCALP" "H1"

# INTRADAY: M15, H1, H4
train_tcn "INTRADAY" "M15"
train_tcn "INTRADAY" "H1"
# For H4, keep epochs fast but seq_len=90 is already applied by seq_len_for_tf
train_tcn "INTRADAY" "H4"

# Optional reuse of M15
if [[ "${REUSE_M15}" == "1" ]]; then
  if [[ "${REUSE_M15_SOURCE}" == "intraday" ]]; then
    echo "[REUSE_M15] Copy intraday_m15_best.pt -> scalp_m15_best.pt"
    cp -f "models/weights/intraday_m15_best.pt" "models/weights/scalp_m15_best.pt"
  else
    echo "[REUSE_M15] Copy scalp_m15_best.pt -> intraday_m15_best.pt"
    cp -f "models/weights/scalp_m15_best.pt" "models/weights/intraday_m15_best.pt"
  fi
fi

# SWING: H1, H4, D1 (use longer epochs)
echo "[TCN] SWING H1/H4/D1 with longer epochs"
python training/train_tcn_enhanced.py --profile SWING --data data/raw/EURUSD_H1_latest.csv \
  --save-dir models/weights --name swing_h1 --seq-len "$(seq_len_for_tf H1)" \
  --epochs "${TCN_EPOCHS_SWING_H1_H4}" --batch-size "${TCN_BATCH_SIZE}" --lr "${TCN_LR}" \
  | tee -a logs/tcn_training.log

python training/train_tcn_enhanced.py --profile SWING --data data/raw/EURUSD_H4_latest.csv \
  --save-dir models/weights --name swing_h4 --seq-len "$(seq_len_for_tf H4)" \
  --epochs "${TCN_EPOCHS_SWING_H1_H4}" --batch-size "${TCN_BATCH_SIZE}" --lr "${TCN_LR}" \
  | tee -a logs/tcn_training.log

python training/train_tcn_enhanced.py --profile SWING --data data/raw/EURUSD_D1_latest.csv \
  --save-dir models/weights --name swing_d1 --seq-len "$(seq_len_for_tf D1)" \
  --epochs "${TCN_EPOCHS_SWING_D1}" --batch-size "${TCN_BATCH_SIZE}" --lr "${TCN_LR}" \
  | tee -a logs/tcn_training.log

echo "--- [4/6] Training Vision Models (ViT & YOLO) ---"
python training/finetune_vit.py --profile ALL --device cuda --generate-dataset | tee logs/vit_training.log
python training/train_yolo_profiles.py --profile ALL --device 0 | tee logs/yolo_training.log

echo "--- [5/6] Training Exit Optimizer (PPO) ---"
python scripts/train_all_models.py --models exit --profiles all --data-rows "${DATA_ROWS}" --device auto | tee logs/exit_ppo.log

echo "--- [6/6] Final Decision Fusion Matrix ---"
train_fusion () {
  local profile="$1"
  local tf="$2"
  echo "[FUSION] profile=${profile} tf=${tf} epochs=${FUSION_EPOCHS} batch=${FUSION_BATCH_SIZE}"
  python training/train_decision_fusion.py \
    --profile "${profile}" \
    --timeframe "${tf}" \
    --device cuda \
    --epochs "${FUSION_EPOCHS}" \
    --batch_size "${FUSION_BATCH_SIZE}" \
    | tee -a logs/fusion_matrix.log
}

# SCALP: M5, M15, H1
train_fusion "SCALP" "M5"
train_fusion "SCALP" "M15"
train_fusion "SCALP" "H1"

# INTRADAY: M15, H1, H4
train_fusion "INTRADAY" "M15"
train_fusion "INTRADAY" "H1"
train_fusion "INTRADAY" "H4"

# SWING: H1, H4, D1
train_fusion "SWING" "H1"
train_fusion "SWING" "H4"
train_fusion "SWING" "D1"

echo "--- Waiting for CPU lane to finish (meta + trend classifier) ---"
wait "${META_PID}" || true
wait "${TREND_PID}" || true

END_TIME=$(date +%s)
DURATION_MIN=$(( (END_TIME - START_TIME) / 60 ))
echo "Training Complete in ${DURATION_MIN} minutes."

# Include YOLO runs folder too (often contains best.pt, metrics, plots)
tar -czf final_artifacts.tar.gz \
  models/ checkpoints/ logs/ runs/ training_results.json 2>/dev/null || \
tar -czf final_artifacts.tar.gz models/ checkpoints/ logs/ training_results.json 2>/dev/null

echo "Artifacts saved to final_artifacts.tar.gz"
'''