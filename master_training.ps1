Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ============================================================================== 
# MASTER TRAINING SCRIPT (Windows 11 / PowerShell): Forex Multi-Model Suite
# GPU strategy: single-job queue (no concurrent heavy GPU trainings)
# CPU strategy: run CPU-only trainings in background while GPU queue runs
# ============================================================================== 

New-Item -ItemType Directory -Force -Path logs, models\weights, models\vit, models\yolo, models\decision_fusion, checkpoints, cache\decision_fusion | Out-Null

$StartTime = Get-Date

# -----------------------------
# Bootstrap dependencies (fast rental setup)
# -----------------------------
$USE_VENV = $env:USE_VENV; if (-not $USE_VENV) { $USE_VENV = '1' }
$VENV_DIR = $env:VENV_DIR; if (-not $VENV_DIR) { $VENV_DIR = '.venv' }
$PYTHON_EXE = $env:PYTHON_EXE; if (-not $PYTHON_EXE) { $PYTHON_EXE = 'python' }

$BOOTSTRAP_DEPS = $env:BOOTSTRAP_DEPS; if (-not $BOOTSTRAP_DEPS) { $BOOTSTRAP_DEPS = '1' }
$RENTAL_REQS_PROFILE = $env:RENTAL_REQS_PROFILE; if (-not $RENTAL_REQS_PROFILE) { $RENTAL_REQS_PROFILE = 'min' } # min|full
$ENABLE_TENSORFLOW = $env:ENABLE_TENSORFLOW; if (-not $ENABLE_TENSORFLOW) { $ENABLE_TENSORFLOW = '0' }
$ENABLE_TORCH_GEOMETRIC = $env:ENABLE_TORCH_GEOMETRIC; if (-not $ENABLE_TORCH_GEOMETRIC) { $ENABLE_TORCH_GEOMETRIC = '0' }
$FORCE_REINSTALL_DEPS = $env:FORCE_REINSTALL_DEPS; if (-not $FORCE_REINSTALL_DEPS) { $FORCE_REINSTALL_DEPS = '0' }

$TORCH_CUDA_INDEX_URL = $env:TORCH_CUDA_INDEX_URL # allow override

$FETCH_REQUIRED = $env:FETCH_REQUIRED; if (-not $FETCH_REQUIRED) { $FETCH_REQUIRED = '1' }
$DATA_SOURCE_DIR = $env:DATA_SOURCE_DIR; if (-not $DATA_SOURCE_DIR) { $DATA_SOURCE_DIR = '' }
$DATA_BASE_URL = $env:DATA_BASE_URL; if (-not $DATA_BASE_URL) { $DATA_BASE_URL = '' }
$REFRESH_DATA = $env:REFRESH_DATA; if (-not $REFRESH_DATA) { $REFRESH_DATA = '0' }

$GENERATE_DATASETS = $env:GENERATE_DATASETS; if (-not $GENERATE_DATASETS) { $GENERATE_DATASETS = '1' }
$REFRESH_DATASETS = $env:REFRESH_DATASETS; if (-not $REFRESH_DATASETS) { $REFRESH_DATASETS = '0' }
$DATASET_MAX_SAMPLES = $env:DATASET_MAX_SAMPLES; if (-not $DATASET_MAX_SAMPLES) { $DATASET_MAX_SAMPLES = '' }

function Require-File([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    throw "[FATAL] Missing required file: $Path"
  }
}

function Ensure-DirForFile([string]$Path) {
  $parent = Split-Path -Parent $Path
  if ($parent -and -not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
  }
}

function To-UrlPath([string]$Path) {
  return ($Path -replace '\\', '/')
}

function Fetch-File([string]$RelativePath) {
  if ($FETCH_REQUIRED -ne '1') {
    return
  }

  $dest = $RelativePath
  $need = (-not (Test-Path -LiteralPath $dest)) -or ($REFRESH_DATA -eq '1')
  if (-not $need) {
    return
  }

  Ensure-DirForFile $dest

  if ($DATA_SOURCE_DIR) {
    $src = Join-Path $DATA_SOURCE_DIR $RelativePath
    if (Test-Path -LiteralPath $src) {
      Write-Host "[FETCH] Copying: $src -> $dest"
      Copy-Item -Force -LiteralPath $src -Destination $dest
      return
    }
  }

  if ($DATA_BASE_URL) {
    $urlPath = To-UrlPath $RelativePath
    $url = ($DATA_BASE_URL.TrimEnd('/') + '/' + $urlPath)
    Write-Host "[FETCH] Downloading: $url -> $dest"
    try {
      Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
      return
    } catch {
      throw "[FATAL] Failed to download required file: $url"
    }
  }
}

function Ensure-RequiredInputs {
  $requiredCsv = @(
    'data\raw\EURUSD_M5_latest.csv',
    'data\raw\EURUSD_M15_latest.csv',
    'data\raw\EURUSD_H1_latest.csv',
    'data\raw\EURUSD_H4_latest.csv',
    'data\raw\EURUSD_D1_latest.csv'
  )

  $requiredYoloYaml = @(
    'datasets\yolo_scalp\data.yaml',
    'datasets\yolo_intraday\data.yaml',
    'datasets\yolo_swing\data.yaml'
  )

  foreach ($p in ($requiredCsv + $requiredYoloYaml)) {
    Fetch-File $p
  }

  foreach ($p in $requiredCsv) {
    if (-not (Test-Path -LiteralPath $p)) {
      $msg = "[FATAL] Missing required file: $p`n" +
             "To enable auto-fetch, set one of:`n" +
             "  - DATA_SOURCE_DIR to a folder containing the same relative paths (e.g. data\\raw\\..., datasets\\yolo_*\\...)`n" +
             "  - DATA_BASE_URL to a URL base where these files are hosted`n" +
             "Or place the files in the repo at the paths above."
      throw $msg
    }
  }

  if ($GENERATE_DATASETS -ne '1') {
    foreach ($p in $requiredYoloYaml) {
      if (-not (Test-Path -LiteralPath $p)) {
        $msg = "[FATAL] Missing required file: $p`n" +
               "You can either provide it (via DATA_SOURCE_DIR/DATA_BASE_URL) or set GENERATE_DATASETS=1 to generate datasets." 
        throw $msg
      }
    }
  }
}

function Invoke-PythonCode([string]$Code, [string]$LogPath) {
  if ($LogPath) {
    $Code | Out-Null
    & python -c $Code 2>&1 | Tee-Object -FilePath $LogPath -Append
  } else {
    & python -c $Code
  }
  if ($LASTEXITCODE -ne 0) {
    if ($LogPath) {
      throw "[FATAL] Python step failed. See log: $LogPath"
    }
    throw "[FATAL] Python step failed."
  }
}

function Generate-VisionDatasets {
  if ($GENERATE_DATASETS -ne '1') {
    Write-Host "[DATASET] GENERATE_DATASETS=0 (skipping dataset generation)"
    return
  }

  if ($REFRESH_DATASETS -eq '1') {
    $dirs = @('datasets\vit_scalp','datasets\vit_intraday','datasets\vit_swing','datasets\yolo_scalp','datasets\yolo_intraday','datasets\yolo_swing')
    foreach ($d in $dirs) {
      if (Test-Path -LiteralPath $d) {
        Write-Host "[DATASET] Refreshing: removing $d"
        Remove-Item -Recurse -Force $d
      }
    }
  }

  Write-Host "[DATASET] Generating YOLO datasets (if missing)..."
  $yoloPairs = @(
    @{ profile='scalp'; data='data\raw\EURUSD_M5_latest.csv'; out='datasets\yolo_scalp'; window=30; stride=10 },
    @{ profile='intraday'; data='data\raw\EURUSD_H1_latest.csv'; out='datasets\yolo_intraday'; window=60; stride=10 },
    @{ profile='swing'; data='data\raw\EURUSD_H4_latest.csv'; out='datasets\yolo_swing'; window=90; stride=10 }
  )

  foreach ($cfg in $yoloPairs) {
    $yaml = Join-Path $cfg.out 'data.yaml'
    $imgOk = (Test-Path -Path (Join-Path $cfg.out 'images\train\*.jpg')) -and (Test-Path -Path (Join-Path $cfg.out 'images\val\*.jpg'))
    $lblOk = (Test-Path -Path (Join-Path $cfg.out 'labels\train\*.txt')) -and (Test-Path -Path (Join-Path $cfg.out 'labels\val\*.txt'))
    $needsGen = (-not (Test-Path -LiteralPath $yaml)) -or (-not $imgOk) -or (-not $lblOk)
    if ($needsGen) {
      $code = @"
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from utils.yolo_dataset_generator import YOLODatasetGenerator

data_path = r'${($cfg.data)}'
out_dir = r'${($cfg.out)}'
gen = YOLODatasetGenerator(output_dir=out_dir, image_size=256, window_size=int(${($cfg.window)}), stride=int(${($cfg.stride)}), val_split=0.2)
gen.generate_from_csv(data_path, symbol='EURUSD_${($cfg.profile)}', max_samples=None)
print('OK')
"@
      Invoke-PythonCode $code 'logs\dataset_generation.log'
    }
  }

  Write-Host "[DATASET] Generating ViT datasets (if missing)..."
  $vitProfiles = @('SCALP','INTRADAY','SWING')
  foreach ($p in $vitProfiles) {
    $dir = "datasets\\vit_{0}" -f $p.ToLower()
    $trainDir = Join-Path $dir 'train'
    if (-not (Test-Path -LiteralPath $trainDir)) {
      $maxSamplesArg = 'None'
      if ($DATASET_MAX_SAMPLES) { $maxSamplesArg = $DATASET_MAX_SAMPLES }

      $code = @"
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path('.').resolve()))
vit_path = Path('training') / 'finetune_vit.py'
spec = importlib.util.spec_from_file_location('finetune_vit', vit_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ok = mod.generate_dataset('${p}', max_samples=${maxSamplesArg})
sys.exit(0 if ok else 1)
"@
      Invoke-PythonCode $code 'logs\dataset_generation.log'
    }
  }

  Require-File 'datasets\yolo_scalp\data.yaml'
  Require-File 'datasets\yolo_intraday\data.yaml'
  Require-File 'datasets\yolo_swing\data.yaml'
}

function Ensure-Venv {
  if ($USE_VENV -ne '1') { return }

  if (-not (Test-Path -LiteralPath $VENV_DIR)) {
    Write-Host "[ENV] Creating venv at: $VENV_DIR (using $PYTHON_EXE)"
    & $PYTHON_EXE -m venv $VENV_DIR
  }

  $activate = Join-Path $VENV_DIR 'Scripts\Activate.ps1'
  if (-not (Test-Path -LiteralPath $activate)) {
    throw "[FATAL] venv activation script not found: $activate"
  }

  . $activate

  $pyPath = ''
  try {
    $pyPath = (& python -c "import sys; print(sys.executable)")
  } catch {
    $pyPath = ''
  }
  Write-Host "[ENV] Using python: $pyPath"
}

function Detect-CudaVersion {
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $out = & nvidia-smi 2>$null
    $m = [regex]::Match($out, 'CUDA Version:\s*([0-9]+\.[0-9]+)')
    if ($m.Success) { return $m.Groups[1].Value }
  }

  if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    $out = & nvcc --version 2>$null
    $m = [regex]::Match($out, 'release\s*([0-9]+\.[0-9]+)')
    if ($m.Success) { return $m.Groups[1].Value }
  }

  return ''
}

function Auto-SetTorchCudaIndexUrl {
  if ($TORCH_CUDA_INDEX_URL) {
    Write-Host "[CUDA] TORCH_CUDA_INDEX_URL already set: $TORCH_CUDA_INDEX_URL"
    return
  }

  $cudaVer = Detect-CudaVersion
  if (-not $cudaVer) {
    Write-Host "[CUDA] CUDA version not detected. Defaulting TORCH_CUDA_INDEX_URL to cu121"
    $script:TORCH_CUDA_INDEX_URL = 'https://download.pytorch.org/whl/cu121'
    return
  }

  Write-Host "[CUDA] Detected CUDA version: $cudaVer"
  $parts = $cudaVer.Split('.')
  $major = [int]$parts[0]
  $minor = [int]$parts[1]

  if ($major -ge 13) {
    $script:TORCH_CUDA_INDEX_URL = 'https://download.pytorch.org/whl/cu124'
  } elseif ($major -eq 12 -and $minor -ge 4) {
    $script:TORCH_CUDA_INDEX_URL = 'https://download.pytorch.org/whl/cu124'
  } elseif ($major -ge 12) {
    $script:TORCH_CUDA_INDEX_URL = 'https://download.pytorch.org/whl/cu121'
  } elseif ($major -eq 11 -and $minor -ge 8) {
    $script:TORCH_CUDA_INDEX_URL = 'https://download.pytorch.org/whl/cu118'
  } else {
    $script:TORCH_CUDA_INDEX_URL = 'https://download.pytorch.org/whl/cu118'
  }

  Write-Host "[CUDA] Using TORCH_CUDA_INDEX_URL=$TORCH_CUDA_INDEX_URL"
}

function Write-RequirementsFiles {
  @'
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
'@ | Set-Content -Encoding UTF8 -Path 'requirements.rental.full.txt'

  @'
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
'@ | Set-Content -Encoding UTF8 -Path 'requirements.rental.min.txt'

  if ($RENTAL_REQS_PROFILE -eq 'full') {
    Copy-Item -Force requirements.rental.full.txt requirements.rental.install.txt
  } else {
    Copy-Item -Force requirements.rental.min.txt requirements.rental.install.txt
  }

  if ($ENABLE_TENSORFLOW -ne '1') {
    (Get-Content requirements.rental.install.txt) | Where-Object { $_ -notmatch '^tensorflow\b' } | Set-Content -Encoding UTF8 requirements.rental.install.txt
  }

  if ($ENABLE_TORCH_GEOMETRIC -ne '1') {
    (Get-Content requirements.rental.install.txt) | Where-Object { $_ -notmatch '^torch_geometric\b' } | Set-Content -Encoding UTF8 requirements.rental.install.txt
  }
}

function Deps-Ok {
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    $mods = "numpy,pandas,scipy,joblib,tqdm,yaml,sklearn,torch,torchvision,timm,cv2,ultralytics,gymnasium,stable_baselines3,mplfinance"
    & python -c "import importlib,sys; mods='$mods'.split(','); [importlib.import_module(m) for m in mods]; sys.exit(0)" 2>$null | Out-Null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  } finally {
    $ErrorActionPreference = $prev
  }
}

function Bootstrap-Deps {
  if ($BOOTSTRAP_DEPS -ne '1') {
    Write-Host "[BOOTSTRAP] BOOTSTRAP_DEPS=0 (skipping dependency install)"
    return
  }

  Ensure-Venv
  Auto-SetTorchCudaIndexUrl

  if (Deps-Ok) {
    Write-Host "[BOOTSTRAP] Dependencies already satisfied (skipping pip install)"
    return
  }

  Write-Host "[BOOTSTRAP] Installing Python dependencies..."
  Write-RequirementsFiles

  & python -m pip install -U pip setuptools wheel

  # If torch is missing or CPU-only, attempt to install CUDA-enabled wheels.
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    & python -c "import torch,sys; ok=torch.cuda.is_available() and (torch.version.cuda is not None); sys.exit(0 if ok else 1)" 2>$null | Out-Null
  } catch {
  } finally {
    $ErrorActionPreference = $prev
  }
  $torchOk = ($LASTEXITCODE -eq 0)

  if (-not $torchOk) {
    Write-Host "[BOOTSTRAP] Installing CUDA torch/torchvision from: $TORCH_CUDA_INDEX_URL"
    & python -m pip install -U --index-url $TORCH_CUDA_INDEX_URL torch torchvision
  }

  if ($FORCE_REINSTALL_DEPS -eq '1') {
    & python -m pip install -U --no-cache-dir --force-reinstall -r requirements.rental.install.txt
  } else {
    & python -m pip install -U -r requirements.rental.install.txt
  }
}

Ensure-Venv
Auto-SetTorchCudaIndexUrl

Write-Host "[CUDA] Probe (nvidia-smi):"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  & nvidia-smi
} else {
  Write-Host "nvidia-smi not found"
}
Write-Host "[CUDA] Detected CUDA version: $(Detect-CudaVersion)"
Write-Host "[CUDA] TORCH_CUDA_INDEX_URL=$TORCH_CUDA_INDEX_URL"

Bootstrap-Deps

# -----------------------------
# Config knobs (tweak if needed)
# -----------------------------
$DATA_ROWS = $env:DATA_ROWS; if (-not $DATA_ROWS) { $DATA_ROWS = '1000000' }

$TCN_EPOCHS_BASE_FAST = $env:TCN_EPOCHS_BASE_FAST; if (-not $TCN_EPOCHS_BASE_FAST) { $TCN_EPOCHS_BASE_FAST = '50' }
$TCN_EPOCHS_FINETUNE = $env:TCN_EPOCHS_FINETUNE; if (-not $TCN_EPOCHS_FINETUNE) { $TCN_EPOCHS_FINETUNE = '15' }

$TCN_EPOCHS_SWING_BASE_H4 = $env:TCN_EPOCHS_SWING_BASE_H4; if (-not $TCN_EPOCHS_SWING_BASE_H4) { $TCN_EPOCHS_SWING_BASE_H4 = '60' }
$TCN_EPOCHS_SWING_FINETUNE = $env:TCN_EPOCHS_SWING_FINETUNE; if (-not $TCN_EPOCHS_SWING_FINETUNE) { $TCN_EPOCHS_SWING_FINETUNE = '25' }

$TCN_BATCH_SIZE = $env:TCN_BATCH_SIZE; if (-not $TCN_BATCH_SIZE) { $TCN_BATCH_SIZE = '128' }
$TCN_LR = $env:TCN_LR; if (-not $TCN_LR) { $TCN_LR = '1e-3' }

$FUSION_EPOCHS = $env:FUSION_EPOCHS; if (-not $FUSION_EPOCHS) { $FUSION_EPOCHS = '50' }
$FUSION_BATCH_SIZE = $env:FUSION_BATCH_SIZE; if (-not $FUSION_BATCH_SIZE) { $FUSION_BATCH_SIZE = '32' }

$REUSE_M15 = $env:REUSE_M15; if (-not $REUSE_M15) { $REUSE_M15 = '0' }
$REUSE_M15_SOURCE = $env:REUSE_M15_SOURCE; if (-not $REUSE_M15_SOURCE) { $REUSE_M15_SOURCE = 'intraday' }

function SeqLen-ForTf([string]$tf) {
  switch ($tf.ToUpper()) {
    'M5' { return 30 }
    'M15' { return 60 }
    'H1' { return 60 }
    'H4' { return 90 }
    'D1' { return 120 }
    default { return 60 }
  }
}

Write-Host "--- [1/7] Pre-flight Checks ---"
& python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None"

Ensure-RequiredInputs
Generate-VisionDatasets

Require-File 'data\raw\EURUSD_M5_latest.csv'
Require-File 'data\raw\EURUSD_M15_latest.csv'
Require-File 'data\raw\EURUSD_H1_latest.csv'
Require-File 'data\raw\EURUSD_H4_latest.csv'
Require-File 'data\raw\EURUSD_D1_latest.csv'

Require-File 'datasets\yolo_scalp\data.yaml'
Require-File 'datasets\yolo_intraday\data.yaml'
Require-File 'datasets\yolo_swing\data.yaml'

Write-Host "--- [2/7] Launching CPU Lane (Background) ---"
$env:OMP_NUM_THREADS = $env:OMP_NUM_THREADS; if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = '32' }
$env:MKL_NUM_THREADS = $env:MKL_NUM_THREADS; if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = '32' }

$metaLog = 'logs\meta_background.log'
$trendLog = 'logs\trend_classifier.log'

$MetaProc = Start-Process -FilePath python -ArgumentList @('scripts\train_all_models.py','--models','meta','--profiles','all','--data-rows',$DATA_ROWS) -RedirectStandardOutput $metaLog -RedirectStandardError $metaLog -PassThru
$TrendProc = Start-Process -FilePath python -ArgumentList @('training\train_trend_classifier.py','--data','data\raw\EURUSD_H1_latest.csv','--output','models\trend_classifier.joblib') -RedirectStandardOutput $trendLog -RedirectStandardError $trendLog -PassThru

Write-Host "CPU lane PIDs: meta=$($MetaProc.Id), trend=$($TrendProc.Id)"
Write-Host "Tail meta log: Get-Content -Wait $metaLog"

Write-Host "--- [3/7] Training Enhanced TCNs (GPU Queue) ---"
function Train-Tcn([string]$profile, [string]$tf, [string]$epochs, [string]$initFrom='') {
  $data = "data\raw\EURUSD_${tf}_latest.csv"
  $name = "{0}_{1}" -f $profile.ToLower(), $tf.ToLower()
  $seqLen = SeqLen-ForTf $tf

  if ($initFrom) { Require-File $initFrom }

  $initFromDisplay = 'none'
  if ($initFrom) { $initFromDisplay = $initFrom }
  Write-Host "[TCN] profile=$profile tf=$tf seq_len=$seqLen batch=$TCN_BATCH_SIZE epochs=$epochs name=$name init_from=$initFromDisplay"

  $args = @('training\train_tcn_enhanced.py','--profile',$profile,'--data',$data,'--save-dir','models\weights','--name',$name,'--seq-len',$seqLen,'--epochs',$epochs,'--batch-size',$TCN_BATCH_SIZE,'--lr',$TCN_LR)
  if ($initFrom) { $args += @('--init-from',$initFrom) }

  & python @args 2>&1 | Tee-Object -FilePath 'logs\tcn_training.log' -Append
}

Train-Tcn 'SCALP' 'H1' $TCN_EPOCHS_BASE_FAST
Train-Tcn 'SCALP' 'M5' $TCN_EPOCHS_FINETUNE 'models\weights\scalp_h1_best.pt'
if ($REUSE_M15 -ne '1') {
  Train-Tcn 'SCALP' 'M15' $TCN_EPOCHS_FINETUNE 'models\weights\scalp_h1_best.pt'
}

Train-Tcn 'INTRADAY' 'H1' $TCN_EPOCHS_BASE_FAST
Train-Tcn 'INTRADAY' 'M15' $TCN_EPOCHS_FINETUNE 'models\weights\intraday_h1_best.pt'
Train-Tcn 'INTRADAY' 'H4' $TCN_EPOCHS_FINETUNE 'models\weights\intraday_h1_best.pt'

if ($REUSE_M15 -eq '1') {
  if ($REUSE_M15_SOURCE -eq 'intraday') {
    Copy-Item -Force 'models\weights\intraday_m15_best.pt' 'models\weights\scalp_m15_best.pt'
  } else {
    Copy-Item -Force 'models\weights\scalp_m15_best.pt' 'models\weights\intraday_m15_best.pt'
  }
}

Train-Tcn 'SWING' 'H4' $TCN_EPOCHS_SWING_BASE_H4
Train-Tcn 'SWING' 'H1' $TCN_EPOCHS_SWING_FINETUNE 'models\weights\swing_h4_best.pt'
Train-Tcn 'SWING' 'D1' $TCN_EPOCHS_SWING_FINETUNE 'models\weights\swing_h4_best.pt'

Write-Host "--- [4/7] Training Vision Models (ViT & YOLO) ---"
& python 'training\finetune_vit.py' --profile ALL --device cuda 2>&1 | Tee-Object -FilePath 'logs\vit_training.log'
& python 'training\train_yolo_profiles.py' --profile ALL --device 0 2>&1 | Tee-Object -FilePath 'logs\yolo_training.log'

Write-Host "--- [5/7] Training Exit Optimizer (PPO) ---"
& python 'scripts\train_all_models.py' --models exit --profiles all --data-rows $DATA_ROWS --device auto 2>&1 | Tee-Object -FilePath 'logs\exit_ppo.log'

Write-Host "--- [6/7] Final Decision Fusion Matrix ---"
function Train-Fusion([string]$profile, [string]$tf) {
  Write-Host "[FUSION] profile=$profile tf=$tf epochs=$FUSION_EPOCHS batch=$FUSION_BATCH_SIZE"
  & python 'training\train_decision_fusion.py' --profile $profile --timeframe $tf --device cuda --epochs $FUSION_EPOCHS --batch_size $FUSION_BATCH_SIZE --use_cache --cache_dir 'cache\decision_fusion' 2>&1 |
    Tee-Object -FilePath 'logs\fusion_matrix.log' -Append
}

Train-Fusion 'SCALP' 'M5'
Train-Fusion 'SCALP' 'M15'
Train-Fusion 'SCALP' 'H1'

Train-Fusion 'INTRADAY' 'M15'
Train-Fusion 'INTRADAY' 'H1'
Train-Fusion 'INTRADAY' 'H4'

Train-Fusion 'SWING' 'H1'
Train-Fusion 'SWING' 'H4'
Train-Fusion 'SWING' 'D1'

Write-Host "--- Waiting for CPU lane to finish (meta + trend classifier) ---"
try { Wait-Process -Id $MetaProc.Id -ErrorAction SilentlyContinue } catch {}
try { Wait-Process -Id $TrendProc.Id -ErrorAction SilentlyContinue } catch {}

$EndTime = Get-Date
$DurationMin = [int](($EndTime - $StartTime).TotalMinutes)
Write-Host "Training Complete in ${DurationMin} minutes."

# Artifacts packaging (zip on Windows)
$artifact = 'final_artifacts.zip'
if (Test-Path $artifact) { Remove-Item -Force $artifact }

$toZip = @('models','checkpoints','logs','runs','training_results.json') | Where-Object { Test-Path $_ }
if ($toZip.Count -gt 0) {
  Compress-Archive -Path $toZip -DestinationPath $artifact -Force
}
Write-Host "Artifacts saved to $artifact"
