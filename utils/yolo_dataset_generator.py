# yolo_dataset_generator.py
"""
YOLO Dataset Generator for Candlestick Pattern Detection.

Creates dataset in YOLO format:
    dataset/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml
"""

import os
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm

from .candle_to_image import CandlestickRenderer
from .pattern_detector import CandlestickPatternDetector, PATTERN_NAMES


class YOLODatasetGenerator:
    """Generates YOLO-format dataset from OHLCV data."""
    
    def __init__(
        self,
        output_dir: str = "dataset_yolo",
        image_size: int = 256,
        window_size: int = 60,
        stride: int = 10,
        val_split: float = 0.2,
        min_patterns_per_image: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.window_size = window_size
        self.stride = stride
        self.val_split = val_split
        self.min_patterns_per_image = min_patterns_per_image
        
        self.renderer = CandlestickRenderer(image_size=(image_size, image_size))
        self.detector = CandlestickPatternDetector()
        
        self.pattern_names = PATTERN_NAMES
        self.num_classes = len(self.pattern_names)
    
    def generate_from_csv(
        self, 
        csv_path: str,
        symbol: str = "EURUSD",
        max_samples: Optional[int] = None,
    ) -> Dict:
        """Generate dataset from a single CSV file."""
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()
        return self.generate_from_dataframe(df, symbol, max_samples)
    
    def generate_from_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str = "DATA",
        max_samples: Optional[int] = None,
    ) -> Dict:
        """Generate dataset from DataFrame."""
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        self._setup_directories()
        starts = self._create_window_starts(df, max_samples)
        
        n_val = int(len(starts) * self.val_split)
        np.random.shuffle(starts)
        val_starts = starts[:n_val]
        train_starts = starts[n_val:]
        
        print(f"Generating {len(train_starts)} train + {len(val_starts)} val images...")
        
        stats = {'train': 0, 'val': 0, 'patterns_found': {}}
        
        for name, s_list in [('train', train_starts), ('val', val_starts)]:
            count, pattern_counts = self._generate_split(df, s_list, name, symbol)
            stats[name] = count
            for k, v in pattern_counts.items():
                stats['patterns_found'][k] = stats['patterns_found'].get(k, 0) + v
        
        self._create_yaml_config()
        
        print(f"\n✅ Dataset generated: {stats['train']} train, {stats['val']} val")
        print(f"   Total patterns: {sum(stats['patterns_found'].values())}")
        
        return stats
    
    def _setup_directories(self):
        """Create YOLO directory structure."""
        dirs = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def _create_windows(self, df: pd.DataFrame, max_samples: Optional[int]) -> List[pd.DataFrame]:
        """Create sliding windows of data."""
        windows = []
        for start in range(0, len(df) - self.window_size + 1, self.stride):
            window = df.iloc[start:start + self.window_size].copy().reset_index(drop=True)
            windows.append(window)
            if max_samples and len(windows) >= max_samples:
                break
        return windows

    def _create_window_starts(self, df: pd.DataFrame, max_samples: Optional[int]) -> np.ndarray:
        starts = np.arange(0, max(0, len(df) - self.window_size + 1), self.stride, dtype=np.int64)
        if max_samples is not None and max_samples > 0 and len(starts) > max_samples:
            starts = starts[:max_samples]
        return starts
    
    def _generate_split(self, df: pd.DataFrame, starts: np.ndarray, split: str, symbol: str) -> Tuple[int, Dict]:
        """Generate images and labels for a split."""
        img_dir = self.output_dir / "images" / split
        label_dir = self.output_dir / "labels" / split
        
        count = 0
        pattern_counts = {}
        
        for _, start in enumerate(tqdm(starts, desc=f"Generating {split}")):
            start_i = int(start)
            window = df.iloc[start_i:start_i + self.window_size]
            patterns = self.detector.detect_all_patterns(window)
            
            if len(patterns) < self.min_patterns_per_image:
                continue
            
            annotations = self.detector.to_yolo_annotations(patterns)
            img, bboxes = self.renderer.render_with_annotations(window, annotations)
            
            filename = f"{symbol}_{split}_{count:06d}"
            
            img_pil = Image.fromarray(img)
            img_pil.save(img_dir / f"{filename}.jpg", quality=95)
            
            label_path = label_dir / f"{filename}.txt"
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    line = f"{bbox['class_id']} {bbox['x_center']:.6f} {bbox['y_center']:.6f} {bbox['width']:.6f} {bbox['height']:.6f}\n"
                    f.write(line)
            
            count += 1
            for p in patterns:
                pattern_counts[p.pattern_name] = pattern_counts.get(p.pattern_name, 0) + 1
        
        return count, pattern_counts
    
    def _create_yaml_config(self):
        """Create YOLO data.yaml configuration file."""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: name for i, name in enumerate(self.pattern_names)},
            'nc': self.num_classes,
        }
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Created config: {yaml_path}")
    
    def generate_synthetic(self, n_samples: int = 5000, symbol: str = "SYNTHETIC") -> Dict:
        """Generate dataset with synthetic OHLCV data."""
        print(f"Generating {n_samples} synthetic samples...")
        total_candles = n_samples * self.stride + self.window_size
        df = self._generate_synthetic_ohlcv(total_candles)
        return self.generate_from_dataframe(df, symbol, n_samples)
    
    def _generate_synthetic_ohlcv(self, n: int) -> pd.DataFrame:
        """Generate synthetic OHLCV data with realistic patterns."""
        data = []
        base_price = 1.1000
        
        for i in range(n):
            volatility = 0.0010
            change = np.random.randn() * volatility
            pattern_prob = np.random.random()
            
            if pattern_prob < 0.02:  # Doji
                body_size = np.random.uniform(0.0001, 0.0002)
                shadow_size = np.random.uniform(0.0005, 0.0015)
                o = base_price
                c = o + np.random.choice([-1, 1]) * body_size
                h = max(o, c) + shadow_size
                l = min(o, c) - shadow_size
            elif pattern_prob < 0.04:  # Hammer
                body_size = np.random.uniform(0.0002, 0.0005)
                o = base_price
                c = o + body_size
                h = c + np.random.uniform(0.0001, 0.0003)
                l = o - body_size * 2.5
            elif pattern_prob < 0.06:  # Engulfing setup
                body_size = np.random.uniform(0.0003, 0.0008)
                direction = np.random.choice([-1, 1])
                o = base_price
                c = o + direction * body_size
                h = max(o, c) + np.random.uniform(0.0001, 0.0003)
                l = min(o, c) - np.random.uniform(0.0001, 0.0003)
            else:  # Normal candle
                o = base_price
                c = o + change
                h = max(o, c) + abs(np.random.randn() * volatility * 0.5)
                l = min(o, c) - abs(np.random.randn() * volatility * 0.5)
            
            data.append({
                'open': o, 'high': h, 'low': l, 'close': c,
                'tick_volume': np.random.randint(100, 5000),
            })
            base_price = c
        
        return pd.DataFrame(data)


def main():
    """Generate YOLO dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate YOLO dataset")
    parser.add_argument('--data', type=str, help='Path to OHLCV CSV')
    parser.add_argument('--output', type=str, default='dataset_yolo')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--val-split', type=float, default=0.2)
    
    args = parser.parse_args()
    
    generator = YOLODatasetGenerator(
        output_dir=args.output,
        image_size=args.image_size,
        window_size=args.window,
        stride=args.stride,
        val_split=args.val_split,
    )
    
    if args.synthetic or not args.data:
        generator.generate_synthetic(n_samples=args.samples)
    else:
        generator.generate_from_csv(args.data, max_samples=args.samples)


if __name__ == "__main__":
    main()
