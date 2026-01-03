#!/usr/bin/env bash
set -euo pipefail

if [[ "${OS:-}" == "Windows_NT" ]]; then
  echo "[WARN] Detected Windows OS. This script expects a POSIX bash environment (Git Bash / MSYS2 / WSL)." >&2
  echo "[WARN] For Windows-native setup, run: powershell -ExecutionPolicy Bypass -File .\\master_training.ps1" >&2
fi

# ==============================================================================
# MASTER TRAINING SCRIPT: Forex Multi-Model Suite
# Optimized for: RTX 4090 (24GB VRAM) + 32-Core CPU + 90GB RAM
# GPU strategy: single-job queue (no concurrent heavy GPU trainings)
# CPU strategy: run CPU-only trainings in background while GPU queue runs
# ==============================================================================

mkdir -p logs models/weights models/vit models/yolo models/decision_fusion checkpoints cache/decision_fusion

START_TIME=$(date +%s)

# -----------------------------
# Bootstrap dependencies (fast rental setup)
# -----------------------------
# Environment isolation (recommended on rentals)
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"

# 0 = skip installs entirely
BOOTSTRAP_DEPS="${BOOTSTRAP_DEPS:-1}"
# min = fastest (recommended for rentals); full = try to install everything in requirements.txt
RENTAL_REQS_PROFILE="${RENTAL_REQS_PROFILE:-min}"
# Optional heavy / platform-specific toggles
ENABLE_TENSORFLOW="${ENABLE_TENSORFLOW:-0}"
ENABLE_TORCH_GEOMETRIC="${ENABLE_TORCH_GEOMETRIC:-0}"
FORCE_REINSTALL_DEPS="${FORCE_REINSTALL_DEPS:-0}"

# If your rental image has a GPU, you usually want CUDA-enabled torch wheels.
# Set this to the matching CUDA index URL for your image/driver.
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-}"

select_python () {
  if [[ -n "${PYTHON_BIN}" ]] && command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "${PYTHON_BIN}"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo python
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo python3
    return 0
  fi
  echo "[FATAL] python/python3 not found on PATH" >&2
  exit 1
}

ensure_venv () {
  if [[ "${USE_VENV}" != "1" ]]; then
    return 0
  fi

  local py
  py="$(select_python)"

  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[ENV] Creating venv at: ${VENV_DIR} (using ${py})"
    "${py}" -m venv "${VENV_DIR}"
  fi

  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"

  python - << 'PY' >/dev/null 2>&1
import sys
print(sys.executable)
PY

  echo "[ENV] Using python: $(python -c 'import sys; print(sys.executable)')"
}

detect_cuda_version () {
  local v
  v=""

  if command -v nvidia-smi >/dev/null 2>&1; then
    # Example: "CUDA Version: 12.4"
    v="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)"
  fi

  if [[ -z "${v}" ]] && command -v nvcc >/dev/null 2>&1; then
    # Example: "release 12.4, V12.4.131"
    v="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | tail -n1)"
  fi

  echo "${v}"
}

auto_set_torch_cuda_index_url () {
  if [[ -n "${TORCH_CUDA_INDEX_URL}" ]]; then
    echo "[CUDA] TORCH_CUDA_INDEX_URL already set: ${TORCH_CUDA_INDEX_URL}"
    return 0
  fi

  local cuda_ver
  cuda_ver="$(detect_cuda_version)"
  if [[ -z "${cuda_ver}" ]]; then
    echo "[CUDA] CUDA version not detected (nvidia-smi/nvcc unavailable). Using default cu121 index URL."
    TORCH_CUDA_INDEX_URL="https://download.pytorch.org/whl/cu121"
    return 0
  fi

  echo "[CUDA] Detected CUDA version: ${cuda_ver}"

  local major minor
  major="${cuda_ver%%.*}"
  minor="${cuda_ver#*.}"

  # Choose the closest commonly available PyTorch wheel index.
  # - CUDA >= 12.4  -> cu124
  # - CUDA >= 12.1  -> cu121
  # - CUDA >= 11.8  -> cu118
  if [[ "${major}" -ge 13 ]]; then
    TORCH_CUDA_INDEX_URL="https://download.pytorch.org/whl/cu124"
  elif [[ "${major}" -eq 12 && "${minor}" -ge 4 ]]; then
    TORCH_CUDA_INDEX_URL="https://download.pytorch.org/whl/cu124"
  elif [[ "${major}" -ge 12 ]]; then
    TORCH_CUDA_INDEX_URL="https://download.pytorch.org/whl/cu121"
  elif [[ "${major}" -eq 11 && "${minor}" -ge 8 ]]; then
    TORCH_CUDA_INDEX_URL="https://download.pytorch.org/whl/cu118"
  else
    TORCH_CUDA_INDEX_URL="https://download.pytorch.org/whl/cu118"
  fi

  echo "[CUDA] Using TORCH_CUDA_INDEX_URL=${TORCH_CUDA_INDEX_URL}"
}

write_requirements_files () {
  # Full copy of repo requirements.txt (embedded for convenience)
  cat > requirements.rental.full.txt << 'REQ'
# --- Core Utilities & Config ---
numpy      #>=2.1.0          # 2.3.x is current stable in late 2025
pandas      #>=2.3.0         # Released Sep 2025
scipy      #>=1.16.0         # Released Oct 2025
joblib      #>=1.4.0
tqdm      #>=4.67.0
python-dotenv      #>=1.0.1
pyyaml     #>=6.0.2
pydantic     #>=2.41.0      # Released Nov 2025
pydantic-settings     #>=2.4.0

# --- Data Ingestion, Storage & Web ---
requests     #>=2.32.0
beautifulsoup4     #>=4.13.0
tweepy     #>=4.15.0
yfinance     #>=0.2.66      # Released Sep 2025
influxdb-client     #>=1.45.0
psycopg2-binary     #>=2.9.9
sqlalchemy     #>=2.0.35
redis     #>=5.1.0
# MetaTrader5 is Windows-only; build 5430 released Nov 2025
MetaTrader5     #>=5.0.5430; sys_platform == 'win32'

# --- Feature Engineering & Math ---
numba     #>=0.62.1         # Released Sep 2025
pandas_ta     #>=0.3.34     # Stable release
vaderSentiment     #>=3.3.2
nltk     #>=3.9.1
pyarrow #>22.0.0
polars
bottleneck

# --- Machine Learning (Tabular) ---
scikit-learn     #>=1.7.2   # Released Sep 2025
xgboost     #>=3.1.0        # Released Sep 2025
lightgbm     #>=4.6.0
catboost     #>=1.2.7
hf_xet    #>=1.2.0

# --- Deep Learning (Core Frameworks) ---
# Torch 2.9 is current as of Nov 2025
torch     #>=2.9.0,<3.0.0
torchvision     #>=0.24.0   # Released Nov 2025
pytorch-tabnet     #>=4.4.0 # Using maintained fork versioning
accelerate     #>=1.0.0
tensorflow>=2.18.0    # Optional

# --- Graph Neural Networks (GNN) ---
# Compatible with Torch 2.x series
torch_geometric     #>=2.7.0

# --- Computer Vision (YOLO & Processing) ---
opencv-python     #>=4.10.0
ultralytics     #>=8.3.0    # YOLO11 released Dec 2025
timm     #>=1.0.9

# --- NLP & Transformers ---
transformers     #>=4.57.0  # Released Nov 2025
sentence-transformers     #>=3.2.0

# --- Reinforcement Learning & Optimization ---
gymnasium     #>=1.2.2      # Released Nov 2025
stable-baselines3[extra]     #>=2.7.0 #
shimmy     #>=2.0.0
deap     #>=1.4.3           # Released May 2025
pyswarms     #>=1.3.0       # Stable
optuna     #>=4.6.0         # Released Nov 2025

# --- API, Execution & Monitoring ---
fastapi     #>=0.123.0      # Released Nov 2025
uvicorn     #>=0.32.0
python-telegram-bot     #>=22.5 # Released Sep 2025
prometheus-client     #>=0.21.0

# --- Visualization ---
matplotlib     #>=3.10.0
seaborn     #>=0.14.0
plotly     #>=5.24.0
mplfinance     #>=0.13.0

# --- Testing & Quality Assurance ---
pytest     #>=8.3.0
pytest-cov     #>=5.0.0
pytest-mock>=3.10.0
pytest-xdist>=3.0.0
pytest-html>=3.2.0
coverage>=7.0.0
REQ

  # Minimal install set for quickest bring-up on a fresh rental.
  # Includes only what is needed by the training pipeline used in this script.
  cat > requirements.rental.min.txt << 'REQMIN'
numpy
pandas
scipy
joblib
tqdm
python-dotenv
PyYAML
pydantic
pydantic-settings

requests

scikit-learn

torch
torchvision
accelerate

opencv-python
ultralytics
timm

transformers

gymnasium
stable-baselines3[extra]
shimmy
optuna

matplotlib
mplfinance
REQMIN

  if [[ "${RENTAL_REQS_PROFILE}" == "full" ]]; then
    cp -f requirements.rental.full.txt requirements.rental.install.txt
  else
    cp -f requirements.rental.min.txt requirements.rental.install.txt
  fi

  # MetaTrader5 is Windows-only; skip by default for Linux rentals.
  sed -i '/^MetaTrader5\b/d' requirements.rental.install.txt || true

  if [[ "${ENABLE_TENSORFLOW}" != "1" ]]; then
    sed -i '/^tensorflow\b/d' requirements.rental.install.txt || true
  fi

  if [[ "${ENABLE_TORCH_GEOMETRIC}" != "1" ]]; then
    sed -i '/^torch_geometric\b/d' requirements.rental.install.txt || true
  fi
}

deps_ok () {
  python - << 'PY' >/dev/null 2>&1
import importlib
mods = [
  'numpy','pandas','scipy','joblib','tqdm','yaml',
  'sklearn',
  'torch','torchvision',
  'timm',
  'cv2',
  'ultralytics',
  'gymnasium','stable_baselines3',
  'mplfinance',
]
for m in mods:
  importlib.import_module(m)
PY
}

bootstrap_deps () {
  if [[ "${BOOTSTRAP_DEPS}" != "1" ]]; then
    echo "[BOOTSTRAP] BOOTSTRAP_DEPS=0 (skipping dependency install)"
    return 0
  fi

  ensure_venv
  auto_set_torch_cuda_index_url

  if deps_ok; then
    echo "[BOOTSTRAP] Dependencies already satisfied (skipping pip install)"
    return 0
  fi

  echo "[BOOTSTRAP] Installing Python dependencies..."
  write_requirements_files

  python -m pip install -U pip setuptools wheel

  # If torch is missing or CPU-only, attempt to install CUDA-enabled wheels.
  if python - << 'PY' >/dev/null 2>&1
import torch
ok = torch.cuda.is_available() and (torch.version.cuda is not None)
raise SystemExit(0 if ok else 1)
PY
  then
    TORCH_OK=0
  else
    TORCH_OK=1
  fi

  if [[ "${TORCH_OK}" != "0" ]]; then
    echo "[BOOTSTRAP] Installing CUDA torch/torchvision from: ${TORCH_CUDA_INDEX_URL}"
    python -m pip install -U --index-url "${TORCH_CUDA_INDEX_URL}" torch torchvision
  fi

  if [[ "${FORCE_REINSTALL_DEPS}" == "1" ]]; then
    python -m pip install -U --no-cache-dir --force-reinstall -r requirements.rental.install.txt
  else
    python -m pip install -U -r requirements.rental.install.txt
  fi
}

ensure_venv
auto_set_torch_cuda_index_url

echo "[CUDA] Probe (nvidia-smi):"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not found"
fi

echo "[CUDA] Detected CUDA version: $(detect_cuda_version || true)"
echo "[CUDA] TORCH_CUDA_INDEX_URL=${TORCH_CUDA_INDEX_URL}"

bootstrap_deps

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
