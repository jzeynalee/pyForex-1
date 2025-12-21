# train_all_models.ps1
# Script to train TCN models for all profiles and timeframes
# Run from project root: .\scripts\train_all_models.ps1

# Activate virtual environment
& d:\myBot\.venv312\scripts\activate.ps1

$ErrorActionPreference = "Continue"

# Configuration
$PROFILES = @("SCALP", "INTRADAY", "SWING")
$DATA_FILES = @{
    "M5"  = "data/raw/EURUSD_M5_latest.csv"
    "M15" = "data/raw/EURUSD_M15_latest.csv"
    "H1"  = "data/raw/EURUSD_H1_latest.csv"
    "H4"  = "data/raw/EURUSD_H4_latest.csv"
    "D1"  = "data/raw/EURUSD_D1_latest.csv"
}

# Profile to timeframe mapping (recommended combinations)
$PROFILE_TF_MAP = @{
    "SCALP"    = @("M5", "M15")
    "INTRADAY" = @("M15", "H1", "H4")
    "SWING"    = @("H1", "H4", "D1")
}

$EPOCHS = 50
$BATCH_SIZE = 64
$HIDDEN_DIM = 64
$SEQ_LEN = 60

Write-Host "=" * 60
Write-Host "  TRAINING ALL TCN MODELS"
Write-Host "=" * 60
Write-Host ""

$startTime = Get-Date
$trainedModels = 0
$failedModels = 0

foreach ($profile in $PROFILES) {
    $timeframes = $PROFILE_TF_MAP[$profile]
    
    foreach ($tf in $timeframes) {
        $dataFile = $DATA_FILES[$tf]
        $saveName = "tcn_${profile}_${tf}"
        
        Write-Host ""
        Write-Host "=" * 60
        Write-Host "  Training: $profile profile on $tf timeframe"
        Write-Host "  Data: $dataFile"
        Write-Host "  Output: models/weights/${saveName}_best.pt"
        Write-Host "=" * 60
        
        try {
            python training/train_tcn.py `
                --data $dataFile `
                --profile $profile `
                --epochs $EPOCHS `
                --batch-size $BATCH_SIZE `
                --hidden-dim $HIDDEN_DIM `
                --seq-len $SEQ_LEN `
                --save-dir "models/weights"
            
            # Rename output to include profile and timeframe
            if (Test-Path "models/weights/tcn_best.pt") {
                Move-Item -Force "models/weights/tcn_best.pt" "models/weights/${saveName}_best.pt"
                Write-Host "[OK] Saved: models/weights/${saveName}_best.pt" -ForegroundColor Green
                $trainedModels++
            }
        }
        catch {
            Write-Host "[FAILED] Training failed for $profile on $tf" -ForegroundColor Red
            Write-Host $_.Exception.Message
            $failedModels++
        }
    }
}

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "=" * 60
Write-Host "  TRAINING COMPLETE"
Write-Host "=" * 60
Write-Host "  Models trained: $trainedModels"
Write-Host "  Failed: $failedModels"
Write-Host "  Duration: $($duration.ToString('hh\:mm\:ss'))"
Write-Host "=" * 60

# List all trained models
Write-Host ""
Write-Host "Trained model weights:"
Get-ChildItem "models/weights/tcn_*.pt" | ForEach-Object { Write-Host "  - $($_.Name)" }
