# (optional) activate venv
& .\.venv312\Scripts\Activate.ps1

$SYMBOL = "EURUSD"
$TFS = @("M5","M15","H1","H4","D1")

$RAW_DIR = "data\raw\mt5"                 # where your MT5 CSVs are
$OUT_BASE = "datasets"                   # output datasets root
$SAMPLES = 20000                         # cap windows/images per TF (tune)
$WINDOW = 60                             # candles per image (tune)
$YOLO_SIZE = 256
$VIT_SIZE  = 224

foreach ($tf in $TFS) {
  $csv = Join-Path $RAW_DIR "$SYMBOL`_$tf.csv"
  if (!(Test-Path $csv)) { throw "Missing CSV: $csv" }

  $out = Join-Path $OUT_BASE "$SYMBOL`_$tf"

  # IMPORTANT: module-style run so relative imports in utils/ work
  python -m utils.generate_dataset `
    --data $csv `
    --output $out `
    --samples $SAMPLES `
    --window $WINDOW `
    --yolo-size $YOLO_SIZE `
    --vit-size $VIT_SIZE
}