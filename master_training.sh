#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MASTER TRAINING SCRIPT: Forex Multi-Model Suite
# Optimized for: RTX 4090 (24GB VRAM) + 32-Core CPU + 90GB RAM
# GPU strategy: single-job queue (no concurrent heavy GPU trainings)
# CPU strategy: run CPU-only trainings in background while GPU queue runs
# ==============================================================================

mkdir -p logs models/weights models/vit models/yolo models/decision_fusion checkpoints cache/decision_fusion

START_TIME=$(date +%s)

# -----------------------------
# Config knobs (tweak if needed)
# -----------------------------
DATA_ROWS="${DATA_ROWS:-1000000}"

TCN_EPOCHS_BASE_FAST="${TCN_EPOCHS_BASE_FAST:-50}"
TCN_EPOCHS_FINETUNE="${TCN_EPOCHS_FINETUNE:-15}"

TCN_EPOCHS_SWING_BASE_H4="${TCN_EPOCHS_SWING_BASE_H4:-60}"
TCN_EPOCHS_SWING_FINETUNE="${TCN_EPOCHS_SWING_FINETUNE:-25}"

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
  local epochs="$3"
  local init_from="${4:-}"

  local data="data/raw/EURUSD_${tf}_latest.csv"
  local name="${profile,,}_${tf,,}"
  local seq_len
  seq_len="$(seq_len_for_tf "$tf")"

  if [[ -n "$init_from" ]]; then
    require_file "$init_from"
  fi

  echo "[TCN] profile=${profile} tf=${tf} seq_len=${seq_len} batch=${TCN_BATCH_SIZE} epochs=${epochs} name=${name} init_from=${init_from:-none}"

  if [[ -n "$init_from" ]]; then
    python training/train_tcn_enhanced.py \
      --profile "${profile}" \
      --data "${data}" \
      --save-dir models/weights \
      --name "${name}" \
      --seq-len "${seq_len}" \
      --epochs "${epochs}" \
      --batch-size "${TCN_BATCH_SIZE}" \
      --lr "${TCN_LR}" \
      --init-from "${init_from}" \
      | tee -a logs/tcn_training.log
  else
    python training/train_tcn_enhanced.py \
      --profile "${profile}" \
      --data "${data}" \
      --save-dir models/weights \
      --name "${name}" \
      --seq-len "${seq_len}" \
      --epochs "${epochs}" \
      --batch-size "${TCN_BATCH_SIZE}" \
      --lr "${TCN_LR}" \
      | tee -a logs/tcn_training.log
  fi
}

# SCALP: base=H1, then warm-start M5/M15 from scalp_h1
train_tcn "SCALP" "H1" "${TCN_EPOCHS_BASE_FAST}"
train_tcn "SCALP" "M5" "${TCN_EPOCHS_FINETUNE}" "models/weights/scalp_h1_best.pt"
if [[ "${REUSE_M15}" != "1" ]]; then
  train_tcn "SCALP" "M15" "${TCN_EPOCHS_FINETUNE}" "models/weights/scalp_h1_best.pt"
fi

# INTRADAY: base=H1, then warm-start M15/H4 from intraday_h1
train_tcn "INTRADAY" "H1" "${TCN_EPOCHS_BASE_FAST}"
train_tcn "INTRADAY" "M15" "${TCN_EPOCHS_FINETUNE}" "models/weights/intraday_h1_best.pt"
train_tcn "INTRADAY" "H4" "${TCN_EPOCHS_FINETUNE}" "models/weights/intraday_h1_best.pt"

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

# SWING: base=H4, then warm-start H1/D1 from swing_h4
train_tcn "SWING" "H4" "${TCN_EPOCHS_SWING_BASE_H4}"
train_tcn "SWING" "H1" "${TCN_EPOCHS_SWING_FINETUNE}" "models/weights/swing_h4_best.pt"
train_tcn "SWING" "D1" "${TCN_EPOCHS_SWING_FINETUNE}" "models/weights/swing_h4_best.pt"

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
    --use_cache \
    --cache_dir cache/decision_fusion \
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
