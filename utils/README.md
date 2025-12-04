# pyForex Dataset Generator

Generate training datasets for YOLO pattern detection and ViT trend classification.

## Quick Start

```bash
# Install dependencies
pip install pillow pyyaml tqdm pandas numpy

# Generate both datasets with synthetic data
python generate_dataset.py --synthetic --samples 5000

# Generate from your OHLCV CSV
python generate_dataset.py --data data/EURUSD_H1.csv --samples 10000

# YOLO only
python generate_dataset.py --yolo-only --synthetic --samples 3000

# ViT only  
python generate_dataset.py --vit-only --synthetic --samples 5000
```

## Output Structure

### YOLO Dataset
```
datasets/yolo/
├── images/
│   ├── train/*.jpg
│   └── val/*.jpg
├── labels/
│   ├── train/*.txt  (class x_center y_center width height)
│   └── val/*.txt
└── data.yaml
```

### ViT Dataset
```
datasets/vit/
├── train/
│   ├── BEARISH/
│   ├── SIDEWAYS/
│   └── BULLISH/
└── val/
    ├── BEARISH/
    ├── SIDEWAYS/
    └── BULLISH/
```

## Supported Patterns (YOLO)

| ID | Pattern | ID | Pattern |
|----|---------|----|---------| 
| 0 | Doji | 10 | Bearish Harami |
| 1 | Hammer | 11 | Shooting Star |
| 2 | Inverted Hammer | 12 | Hanging Man |
| 3 | Bullish Engulfing | 13 | Piercing Line |
| 4 | Bearish Engulfing | 14 | Dark Cloud Cover |
| 5 | Morning Star | 15 | Tweezer Top |
| 6 | Evening Star | 16 | Tweezer Bottom |
| 7 | Three White Soldiers | 17 | Spinning Top |
| 8 | Three Black Crows | 18 | Marubozu Bull |
| 9 | Bullish Harami | 19 | Marubozu Bear |

## ViT Labels

| Label | Description |
|-------|-------------|
| BEARISH | Future returns < -0.5% |
| SIDEWAYS | Returns within ±0.5% |
| BULLISH | Future returns > +0.5% |

## Command Line Options

```
--data PATH          OHLCV CSV file
--synthetic          Generate synthetic data
--output DIR         Output directory (default: datasets/)
--yolo-only          Generate only YOLO dataset
--vit-only           Generate only ViT dataset
--samples N          Number of samples (default: 5000)
--yolo-size SIZE     YOLO image size (default: 256)
--vit-size SIZE      ViT image size (default: 224)
--window N           Candles per image (default: 60)
```

## CSV Format

```csv
time,open,high,low,close,tick_volume
2024-01-01 00:00:00,1.10123,1.10234,1.10012,1.10189,1234
```

## Training

### YOLO
```bash
yolo detect train data=datasets/yolo/data.yaml model=yolov8n.pt epochs=80 imgsz=256
```

### ViT
```bash
python training/train_vit.py --data_dir datasets/vit --epochs 30
```

## Module Usage

### Candlestick Image Renderer
```python
from candle_to_image import candle_image, normalize_for_model

img = candle_image(df, target_size=224)  # HWC uint8
img_norm = normalize_for_model(img)       # CHW normalized for ViT
```

### Pattern Detector
```python
from pattern_detector import CandlestickPatternDetector

detector = CandlestickPatternDetector()
patterns = detector.detect_all_patterns(df)
for p in patterns:
    print(f"{p.pattern_name}: {p.start_idx}-{p.end_idx}")
```

### YOLO Generator
```python
from yolo_dataset_generator import YOLODatasetGenerator

generator = YOLODatasetGenerator(output_dir="yolo_data", image_size=256)
stats = generator.generate_from_csv("data.csv")
```

### ViT Generator
```python
from vit_dataset_generator import ViTDatasetGenerator, FuturePriceLabeler

labeler = FuturePriceLabeler(forward_bars=10, threshold_pct=0.5)
generator = ViTDatasetGenerator(output_dir="vit_data", labeler=labeler)
stats = generator.generate_from_csv("data.csv")
```

## Integration with pyForex

Copy these files to your `utils/` directory:
```bash
cp candle_to_image.py pattern_detector.py ../utils/
```

Then update your existing imports or use directly in training scripts.
