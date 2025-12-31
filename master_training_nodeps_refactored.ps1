Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ==============================================================================
# MASTER TRAINING SCRIPT (Windows 11 / PowerShell) - REFACTORED (SAFE)
# ==============================================================================
New-Item -ItemType Directory -Force -Path `
  logs, models\weights, models\vit, models\yolo, models\decision_fusion, `
  checkpoints, cache\decision_fusion | Out-Null

$StartTime = Get-Date

$PYTHON_EXE = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { 'python' }

# ==============================================================================
# ENV CONFIG
# ==============================================================================
$FETCH_REQUIRED = $env:FETCH_REQUIRED
if (-not $FETCH_REQUIRED) { $FETCH_REQUIRED = '1' }

$FETCH_CSV_FROM_MT5 = $env:FETCH_CSV_FROM_MT5
if (-not $FETCH_CSV_FROM_MT5) { $FETCH_CSV_FROM_MT5 = '1' }

$DATA_SOURCE_DIR = $env:DATA_SOURCE_DIR
if (-not $DATA_SOURCE_DIR) { $DATA_SOURCE_DIR = '' }

$DATA_BASE_URL = $env:DATA_BASE_URL
if (-not $DATA_BASE_URL) { $DATA_BASE_URL = '' }

$REFRESH_DATA = $env:REFRESH_DATA
if (-not $REFRESH_DATA) { $REFRESH_DATA = '0' }

$MT5_PATH = $env:MT5_PATH
if (-not $MT5_PATH) { $MT5_PATH = '' }

$MT5_LOGIN = $env:MT5_LOGIN
if (-not $MT5_LOGIN) { $MT5_LOGIN = '' }

$MT5_PASSWORD = $env:MT5_PASSWORD
if (-not $MT5_PASSWORD) { $MT5_PASSWORD = '' }

$MT5_SERVER = $env:MT5_SERVER
if (-not $MT5_SERVER) { $MT5_SERVER = '' }

$MT5_PORTABLE = $env:MT5_PORTABLE
if (-not $MT5_PORTABLE) { $MT5_PORTABLE = '0' }

$MT5_SYMBOL = $env:MT5_SYMBOL
if (-not $MT5_SYMBOL) { $MT5_SYMBOL = 'EURUSD' }

$MT5_BARS = $env:MT5_BARS
if (-not $MT5_BARS) { $MT5_BARS = '200000' }

# ==============================================================================
# HELPERS
# ==============================================================================
function Ensure-DirForFile($Path) {
  $p = Split-Path -Parent $Path
  if ($p -and -not (Test-Path $p)) {
    New-Item -ItemType Directory -Force -Path $p | Out-Null
  }
}

function Require-File($Path) {
  if (-not (Test-Path $Path)) {
    throw "[FATAL] Missing required file: $Path"
  }
}

function Invoke-PythonLogged([string[]]$Args, $LogPath) {
  & $PYTHON_EXE @Args 2>&1 |
    Tee-Object -FilePath $LogPath -Append
  if ($LASTEXITCODE -ne 0) {
    throw "[FATAL] Command failed (exit=$LASTEXITCODE). See log: $LogPath"
  }
}

# ==============================================================================
# MT5 FETCH (FIXED, SAFE, DETERMINISTIC)
# ==============================================================================
function Fetch-CsvFromMT5($CsvPath) {
  if ($FETCH_CSV_FROM_MT5 -ne '1') { return }

  $tf = $null
  foreach ($x in @('M5','M15','H1','H4','D1')) {
    if ($CsvPath -match "_${x}_latest\.csv$") { $tf = $x; break }
  }
  if (-not $tf) { return }

  Ensure-DirForFile $CsvPath

  $py = @'
import sys,argparse
from pathlib import Path
import MetaTrader5 as mt5
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument("--csv",required=True)
p.add_argument("--symbol",required=True)
p.add_argument("--tf",required=True)
p.add_argument("--bars",type=int,required=True)
p.add_argument("--path")
p.add_argument("--login")
p.add_argument("--password")
p.add_argument("--server")
p.add_argument("--portable",action="store_true")
a=p.parse_args()

init={}
if a.path: init["path"]=a.path
if a.login: init["login"]=int(a.login)
if a.password: init["password"]=a.password
if a.server: init["server"]=a.server
if a.portable: init["portable"]=True

if not mt5.initialize(**init):
    print("MT5 init failed:",mt5.last_error(),file=sys.stderr); sys.exit(3)

sym=None
for s in mt5.symbols_get():
    if s.name==a.symbol or s.name.startswith(a.symbol):
        sym=s.name; break
if not sym:
    print("Symbol not found:",a.symbol,file=sys.stderr); sys.exit(8)

mt5.symbol_select(sym,True)

tfm={
 "M5":mt5.TIMEFRAME_M5,
 "M15":mt5.TIMEFRAME_M15,
 "H1":mt5.TIMEFRAME_H1,
 "H4":mt5.TIMEFRAME_H4,
 "D1":mt5.TIMEFRAME_D1
}[a.tf]

bars=min(a.bars,500000)
rates=mt5.copy_rates_from_pos(sym,tfm,0,bars)
if rates is None or len(rates)==0:
    print("No rates:",mt5.last_error(),file=sys.stderr); sys.exit(4)

df=pd.DataFrame(rates)
df["time"]=pd.to_datetime(df["time"],unit="s")
Path(a.csv).parent.mkdir(parents=True,exist_ok=True)
df.to_csv(a.csv,index=False)
print(f"Saved {len(df)} bars → {a.csv}")
mt5.shutdown()
'@

  $tmp = Join-Path $env:TEMP "fetch_mt5_fixed.py"
  $py | Set-Content $tmp -Encoding UTF8

  $args = @(
    $tmp,'--csv',$CsvPath,'--symbol',$MT5_SYMBOL,'--tf',$tf,'--bars',$MT5_BARS
  )
  if ($MT5_PATH)     { $args += @('--path',$MT5_PATH) }
  if ($MT5_LOGIN)    { $args += @('--login',$MT5_LOGIN) }
  if ($MT5_PASSWORD) { $args += @('--password',$MT5_PASSWORD) }
  if ($MT5_SERVER)   { $args += @('--server',$MT5_SERVER) }
  if ($MT5_PORTABLE -eq '1') { $args += '--portable' }

  Invoke-PythonLogged $args 'logs\fetch_data.log'
}

# ==============================================================================
# REQUIRED INPUTS
# ==============================================================================
$CSV_LIST = @(
 'data\raw\EURUSD_M5_latest.csv',
 'data\raw\EURUSD_M15_latest.csv',
 'data\raw\EURUSD_H1_latest.csv',
 'data\raw\EURUSD_H4_latest.csv',
 'data\raw\EURUSD_D1_latest.csv'
)

Write-Host "--- [1/7] Pre-flight Checks ---"
& $PYTHON_EXE -c "import torch; print('CUDA:',torch.cuda.is_available())"

foreach ($csv in $CSV_LIST) {
  if (-not (Test-Path $csv)) {
    Write-Host "[FETCH] Missing CSV: $csv"
    Fetch-CsvFromMT5 $csv
  }
}

foreach ($csv in $CSV_LIST) { Require-File $csv }

# ==============================================================================
# TRAINING (UNCHANGED FROM ORIGINAL)
# ==============================================================================
Write-Host "--- [2/7] Launching CPU Lane ---"
Start-Process $PYTHON_EXE -ArgumentList `
  @('scripts\train_all_models.py','--models','meta','--profiles','all') `
  -RedirectStandardOutput logs\meta.out.log `
  -RedirectStandardError logs\meta.err.log

Write-Host "--- [3/7] Training TCNs ---"
Invoke-PythonLogged @(
 'training\train_tcn_enhanced.py','--profile','SCALP',
 '--data','data\raw\EURUSD_M5_latest.csv'
) 'logs\tcn.log'

Write-Host "--- [4/7] Training Vision Models ---"
Invoke-PythonLogged @(
 'training\finetune_vit.py','--profile','ALL','--device','cuda'
) 'logs\vit.log'

Invoke-PythonLogged @(
 'training\train_yolo_profiles.py','--profile','ALL','--device','0'
) 'logs\yolo.log'

Write-Host "--- [5/7] Training Exit PPO ---"
Invoke-PythonLogged @(
 'scripts\train_all_models.py','--models','exit','--profiles','all'
) 'logs\exit.log'

Write-Host "--- [6/7] Training Decision Fusion ---"
Invoke-PythonLogged @(
 'training\train_decision_fusion.py','--profile','ALL','--device','cuda'
) 'logs\fusion.log'

# ==============================================================================
# FINALIZE
# ==============================================================================
$Duration = [int]((Get-Date)-$StartTime).TotalMinutes
Write-Host "Training complete in $Duration minutes."

Compress-Archive -Path models,logs,checkpoints -DestinationPath final_artifacts.zip -Force
Write-Host "Artifacts saved to final_artifacts.zip"
exit 0
