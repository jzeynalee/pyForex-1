Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ============================================================================== 
# MASTER TRAINING SCRIPT (Windows 11 / PowerShell) - NO DEPS / NO VENV
# Assumes:
# - python is already available (and has required packages installed)
# - optional: required data files are already present OR can be fetched via env vars
# ============================================================================== 

New-Item -ItemType Directory -Force -Path logs, models\weights, models\vit, models\yolo, models\decision_fusion, checkpoints, cache\decision_fusion | Out-Null

$StartTime = Get-Date

$PYTHON_EXE = $env:PYTHON_EXE; if (-not $PYTHON_EXE) { $PYTHON_EXE = 'python' }

function Fetch-CsvFromMT5([string]$CsvPath) {
  if ($FETCH_CSV_FROM_MT5 -ne '1') {
    return
  }

  $m = [regex]::Match($CsvPath, '_(M5|M15|H1|H4|D1)_latest\.csv$', [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
  if (-not $m.Success) {
    return
  }

  $tf = $m.Groups[1].Value.ToUpper()
  Ensure-DirForFile $CsvPath

  $scriptContent = @'
import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--symbol', required=True)
    p.add_argument('--tf', required=True)
    p.add_argument('--bars', type=int, required=True)
    p.add_argument('--mt5-path', default='')
    args = p.parse_args()

    try:
        import MetaTrader5 as mt5
        import pandas as pd
    except Exception as e:
        print('MetaTrader5 import failed: {0}'.format(e), file=sys.stderr)
        return 2

    init_kwargs = {}
    if args.mt5_path:
        init_kwargs['path'] = args.mt5_path

    if not mt5.initialize(**init_kwargs):
        print('MT5 initialize failed: {0}'.format(mt5.last_error()), file=sys.stderr)
        return 3

    tf_map = {
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
    }

    tf = args.tf.upper()
    if tf not in tf_map:
        print('Unsupported timeframe: {0}'.format(tf), file=sys.stderr)
        mt5.shutdown()
        return 5

    rates = mt5.copy_rates_from_pos(args.symbol, tf_map[tf], 0, args.bars)
    if rates is None or len(rates) == 0:
        print('No rates returned for {0} {1}. Error: {2}'.format(args.symbol, tf, mt5.last_error()), file=sys.stderr)
        mt5.shutdown()
        return 4

    df = pd.DataFrame(rates)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'volume': 'tick_volume'}, inplace=True, errors='ignore')

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(csv_path), index=False)
    mt5.shutdown()
    print('Saved {0} rows to {1}'.format(len(df), csv_path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
'@

  $tmpDir = Join-Path $env:TEMP 'pyforex'
  if (-not (Test-Path -LiteralPath $tmpDir)) { New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null }
  $tmpScript = Join-Path $tmpDir ("fetch_mt5_{0}.py" -f ([guid]::NewGuid().ToString('N')))
  $scriptContent | Out-File -FilePath $tmpScript -Encoding utf8 -Force

  try {
    $fetchArgs = @($tmpScript,'--csv',$CsvPath,'--symbol',$MT5_SYMBOL,'--tf',$tf,'--bars',$MT5_BARS)
    if ($MT5_PATH) {
      $fetchArgs += @('--mt5-path',$MT5_PATH)
    }
    Invoke-PythonLogged $fetchArgs 'logs\fetch_data.log'
  } finally {
    if (Test-Path -LiteralPath $tmpScript) { Remove-Item -Force -LiteralPath $tmpScript }
  }

  if (-not (Test-Path -LiteralPath $CsvPath)) {
    Write-Host "[FETCH] MT5 fetch did not create: $CsvPath"
    Write-Host "[FETCH] Check logs\\fetch_data.log for details"
  }
}

$FETCH_REQUIRED = $env:FETCH_REQUIRED; if (-not $FETCH_REQUIRED) { $FETCH_REQUIRED = '1' }
$DATA_SOURCE_DIR = $env:DATA_SOURCE_DIR; if (-not $DATA_SOURCE_DIR) { $DATA_SOURCE_DIR = '' }
$DATA_BASE_URL = $env:DATA_BASE_URL; if (-not $DATA_BASE_URL) { $DATA_BASE_URL = '' }
$REFRESH_DATA = $env:REFRESH_DATA; if (-not $REFRESH_DATA) { $REFRESH_DATA = '0' }

$GENERATE_DATASETS = $env:GENERATE_DATASETS; if (-not $GENERATE_DATASETS) { $GENERATE_DATASETS = '1' }
$REFRESH_DATASETS = $env:REFRESH_DATASETS; if (-not $REFRESH_DATASETS) { $REFRESH_DATASETS = '0' }
$DATASET_MAX_SAMPLES = $env:DATASET_MAX_SAMPLES; if (-not $DATASET_MAX_SAMPLES) { $DATASET_MAX_SAMPLES = '' }

$FETCH_CSV_FROM_MT5 = $env:FETCH_CSV_FROM_MT5; if (-not $FETCH_CSV_FROM_MT5) { $FETCH_CSV_FROM_MT5 = '1' }
$MT5_PATH = $env:MT5_PATH; if (-not $MT5_PATH) { $MT5_PATH = '' }
$MT5_SYMBOL = $env:MT5_SYMBOL; if (-not $MT5_SYMBOL) { $MT5_SYMBOL = 'EURUSD' }
$MT5_BARS = $env:MT5_BARS; if (-not $MT5_BARS) { $MT5_BARS = '1000000' }

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

  if ($FETCH_CSV_FROM_MT5 -eq '1') {
    foreach ($p in $requiredCsv) {
      if (-not (Test-Path -LiteralPath $p)) {
        Write-Host "[FETCH] Missing CSV: $p"
        Write-Host "[FETCH] Attempting MT5 fetch (symbol=$MT5_SYMBOL bars=$MT5_BARS)..."
        Fetch-CsvFromMT5 $p
      }
    }
  }

  foreach ($p in $requiredCsv) {
    if (-not (Test-Path -LiteralPath $p)) {
      $msg = "[FATAL] Missing required file: $p`n" +
             "To enable auto-fetch, set one of:`n" +
             "  - DATA_SOURCE_DIR to a folder containing the same relative paths (e.g. data\\raw\\..., datasets\\yolo_*\\...)`n" +
             "  - DATA_BASE_URL to a URL base where these files are hosted`n" +
             "  - FETCH_CSV_FROM_MT5=1 and ensure MetaTrader5 is installed/configured (optional MT5_PATH)`n" +
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
    & $PYTHON_EXE -c $Code 2>&1 |
      ForEach-Object {
        $line = $_.ToString()
        $line
        $line | Out-File -FilePath $LogPath -Append -Encoding utf8
      }
  } else {
    & $PYTHON_EXE -c $Code
  }
  if ($LASTEXITCODE -ne 0) {
    if ($LogPath) {
      throw "[FATAL] Python step failed. See log: $LogPath"
    }
    throw "[FATAL] Python step failed."
  }
}

function Invoke-PythonLogged([string[]]$ArgList, [string]$LogPath) {
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    & $PYTHON_EXE @ArgList 2>&1 |
      ForEach-Object { $_.ToString() } |
      ForEach-Object {
        $line = $_
        $line
        $line | Out-File -FilePath $LogPath -Append -Encoding utf8
      }
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $prev
  }

  if ($exitCode -ne 0) {
    throw "[FATAL] Command failed (exit=$exitCode). See log: $LogPath"
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

try {
  Write-Host "[ENV] Using python: $PYTHON_EXE"
  Write-Host "[CUDA] Probe (nvidia-smi):"
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi
  } else {
    Write-Host "nvidia-smi not found"
  }
  Write-Host "[CUDA] Detected CUDA version: $(Detect-CudaVersion)"

  Write-Host "--- [1/7] Pre-flight Checks ---"
  & $PYTHON_EXE -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None"

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

$metaOutLog = 'logs\meta_background.out.log'
$metaErrLog = 'logs\meta_background.err.log'
$trendOutLog = 'logs\trend_classifier.out.log'
$trendErrLog = 'logs\trend_classifier.err.log'

$MetaProc = Start-Process -FilePath $PYTHON_EXE -ArgumentList @('scripts\train_all_models.py','--models','meta','--profiles','all','--data-rows',$DATA_ROWS) -RedirectStandardOutput $metaOutLog -RedirectStandardError $metaErrLog -PassThru
$TrendProc = Start-Process -FilePath $PYTHON_EXE -ArgumentList @('training\train_trend_classifier.py','--data','data\raw\EURUSD_H1_latest.csv','--output','models\trend_classifier.joblib') -RedirectStandardOutput $trendOutLog -RedirectStandardError $trendErrLog -PassThru

Write-Host "CPU lane PIDs: meta=$($MetaProc.Id), trend=$($TrendProc.Id)"
Write-Host "Tail meta stdout: Get-Content -Wait $metaOutLog"
Write-Host "Tail meta stderr: Get-Content -Wait $metaErrLog"

Write-Host "--- [3/7] Training Enhanced TCNs (GPU Queue) ---"
function Train-Tcn([string]$tcnProfile, [string]$tf, [string]$epochs, [string]$initFrom='') {
  $data = "data\raw\EURUSD_${tf}_latest.csv"
  $name = "{0}_{1}" -f $tcnProfile.ToLower(), $tf.ToLower()
  $seqLen = SeqLen-ForTf $tf

  if ($initFrom) { Require-File $initFrom }

  $initFromDisplay = 'none'
  if ($initFrom) { $initFromDisplay = $initFrom }
  Write-Host "[TCN] profile=$tcnProfile tf=$tf seq_len=$seqLen batch=$TCN_BATCH_SIZE epochs=$epochs name=$name init_from=$initFromDisplay"

  $tcnArgs = @('training\train_tcn_enhanced.py','--profile',$tcnProfile,'--data',$data,'--save-dir','models\weights','--name',$name,'--seq-len',$seqLen,'--epochs',$epochs,'--batch-size',$TCN_BATCH_SIZE,'--lr',$TCN_LR)
  if ($initFrom) { $tcnArgs += @('--init-from',$initFrom) }

  Invoke-PythonLogged $tcnArgs 'logs\tcn_training.log'
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
$vitArgs = @('training\finetune_vit.py','--profile','ALL','--device','cuda')
Invoke-PythonLogged $vitArgs 'logs\vit_training.log'

$yoloArgs = @('training\train_yolo_profiles.py','--profile','ALL','--device','0')
Invoke-PythonLogged $yoloArgs 'logs\yolo_training.log'

Write-Host "--- [5/7] Training Exit Optimizer (PPO) ---"
$exitArgs = @('scripts\train_all_models.py','--models','exit','--profiles','all','--data-rows',$DATA_ROWS,'--device','auto')
Invoke-PythonLogged $exitArgs 'logs\exit_ppo.log'

Write-Host "--- [6/7] Final Decision Fusion Matrix ---"
function Train-Fusion([string]$fusionProfile, [string]$tf) {
  Write-Host "[FUSION] profile=$fusionProfile tf=$tf epochs=$FUSION_EPOCHS batch=$FUSION_BATCH_SIZE"
  $fusionArgs = @('training\train_decision_fusion.py','--profile',$fusionProfile,'--timeframe',$tf,'--device','cuda','--epochs',$FUSION_EPOCHS,'--batch_size',$FUSION_BATCH_SIZE,'--use_cache','--cache_dir','cache\decision_fusion')
  Invoke-PythonLogged $fusionArgs 'logs\fusion_matrix.log'
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
  exit 0
} catch {
  $msg = $_.Exception.Message
  if (-not $msg) { $msg = $_.ToString() }
  Write-Host $msg
  Write-Host "[FATAL] Training script aborted."
  exit 1
}
