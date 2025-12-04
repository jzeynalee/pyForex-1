# vit_dataset_generator.py
"""
ViT Dataset Generator for Chart Classification.

Creates dataset in ImageFolder format:
    dataset_vit/
    ├── train/
    │   ├── BEARISH/
    │   ├── SIDEWAYS/
    │   └── BULLISH/
    └── val/
        ├── BEARISH/
        ├── SIDEWAYS/
        └── BULLISH/
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from enum import IntEnum

from .candle_to_image import CandlestickRenderer


class TrendLabel(IntEnum):
    """Trend classification labels."""
    BEARISH = 0
    SIDEWAYS = 1
    BULLISH = 2


class LabelStrategy:
    """Base class for labeling strategies."""
    def get_label(self, df: pd.DataFrame) -> int:
        raise NotImplementedError


class FuturePriceLabeler(LabelStrategy):
    """Label based on future price movement."""
    
    def __init__(self, forward_bars: int = 10, threshold_pct: float = 0.5):
        self.forward_bars = forward_bars
        self.threshold_pct = threshold_pct
    
    def get_label(self, df: pd.DataFrame, future_df: Optional[pd.DataFrame] = None) -> int:
        if future_df is None or len(future_df) < self.forward_bars:
            return TrendLabel.SIDEWAYS
        
        current_price = df['close'].iloc[-1]
        future_price = future_df['close'].iloc[self.forward_bars - 1]
        pct_change = (future_price - current_price) / current_price * 100
        
        if pct_change >= self.threshold_pct:
            return TrendLabel.BULLISH
        elif pct_change <= -self.threshold_pct:
            return TrendLabel.BEARISH
        else:
            return TrendLabel.SIDEWAYS


class TrendStructureLabeler(LabelStrategy):
    """Label based on swing structure within window."""
    
    def __init__(self, min_swings: int = 3):
        self.min_swings = min_swings
    
    def get_label(self, df: pd.DataFrame, future_df: Optional[pd.DataFrame] = None) -> int:
        highs = df['high'].values
        lows = df['low'].values
        
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(df) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append((i, lows[i]))
        
        if len(swing_highs) < self.min_swings or len(swing_lows) < self.min_swings:
            return TrendLabel.SIDEWAYS
        
        hh_count = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i][1] > swing_highs[i-1][1])
        hl_count = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i][1] > swing_lows[i-1][1])
        lh_count = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i][1] < swing_highs[i-1][1])
        ll_count = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i][1] < swing_lows[i-1][1])
        
        bull_score = hh_count + hl_count
        bear_score = lh_count + ll_count
        
        if bull_score > bear_score + 1:
            return TrendLabel.BULLISH
        elif bear_score > bull_score + 1:
            return TrendLabel.BEARISH
        else:
            return TrendLabel.SIDEWAYS


class EMABasedLabeler(LabelStrategy):
    """Label based on EMA alignment."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def get_label(self, df: pd.DataFrame, future_df: Optional[pd.DataFrame] = None) -> int:
        if len(df) < self.slow_period:
            return TrendLabel.SIDEWAYS
        
        close = df['close']
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()
        
        fast_val = ema_fast.iloc[-1]
        slow_val = ema_slow.iloc[-1]
        price = close.iloc[-1]
        
        fast_slope = (ema_fast.iloc[-1] - ema_fast.iloc[-5]) / ema_fast.iloc[-5] * 100
        
        if fast_val > slow_val and price > fast_val and fast_slope > 0.1:
            return TrendLabel.BULLISH
        elif fast_val < slow_val and price < fast_val and fast_slope < -0.1:
            return TrendLabel.BEARISH
        else:
            return TrendLabel.SIDEWAYS


class ViTDatasetGenerator:
    """Generates ImageFolder dataset for ViT training."""
    
    CLASS_NAMES = ['BEARISH', 'SIDEWAYS', 'BULLISH']
    
    def __init__(
        self,
        output_dir: str = "dataset_vit",
        image_size: int = 224,
        window_size: int = 60,
        stride: int = 5,
        val_split: float = 0.2,
        labeler: Optional[LabelStrategy] = None,
        include_volume: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.window_size = window_size
        self.stride = stride
        self.val_split = val_split
        self.labeler = labeler or FuturePriceLabeler()
        self.include_volume = include_volume
        
        self.renderer = CandlestickRenderer(image_size=(image_size, image_size))
    
    def generate_from_csv(self, csv_path: str, symbol: str = "EURUSD", max_samples: Optional[int] = None) -> Dict:
        """Generate dataset from CSV file."""
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.lower()
        return self.generate_from_dataframe(df, symbol, max_samples)
    
    def generate_from_dataframe(self, df: pd.DataFrame, symbol: str = "DATA", max_samples: Optional[int] = None) -> Dict:
        """Generate dataset from DataFrame."""
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        self._setup_directories()
        samples = self._create_labeled_samples(df, max_samples)
        stats = self._split_and_save(samples, symbol)
        
        print(f"\n✅ Dataset generated at {self.output_dir}")
        return stats
    
    def _setup_directories(self):
        """Create ImageFolder directory structure."""
        for split in ['train', 'val']:
            for cls in self.CLASS_NAMES:
                (self.output_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    def _create_labeled_samples(self, df: pd.DataFrame, max_samples: Optional[int]) -> List[Tuple[pd.DataFrame, int]]:
        """Create windows with labels."""
        samples = []
        future_window = 20
        max_start = len(df) - self.window_size - future_window
        
        for start in range(0, max_start, self.stride):
            window = df.iloc[start:start + self.window_size].copy().reset_index(drop=True)
            future = df.iloc[start + self.window_size:start + self.window_size + future_window]
            
            if isinstance(self.labeler, FuturePriceLabeler):
                label = self.labeler.get_label(window, future)
            else:
                label = self.labeler.get_label(window)
            
            samples.append((window, label))
            
            if max_samples and len(samples) >= max_samples:
                break
        
        return samples
    
    def _split_and_save(self, samples: List[Tuple[pd.DataFrame, int]], symbol: str) -> Dict:
        """Split samples and save images."""
        np.random.shuffle(samples)
        
        n_val = int(len(samples) * self.val_split)
        val_samples = samples[:n_val]
        train_samples = samples[n_val:]
        
        stats = {
            'train': {'total': 0, 'BEARISH': 0, 'SIDEWAYS': 0, 'BULLISH': 0},
            'val': {'total': 0, 'BEARISH': 0, 'SIDEWAYS': 0, 'BULLISH': 0},
        }
        
        print(f"Saving {len(train_samples)} training images...")
        for i, (window, label) in enumerate(tqdm(train_samples)):
            self._save_sample(window, label, 'train', symbol, i)
            stats['train']['total'] += 1
            stats['train'][self.CLASS_NAMES[label]] += 1
        
        print(f"Saving {len(val_samples)} validation images...")
        for i, (window, label) in enumerate(tqdm(val_samples)):
            self._save_sample(window, label, 'val', symbol, i)
            stats['val']['total'] += 1
            stats['val'][self.CLASS_NAMES[label]] += 1
        
        print("\n📊 Class Distribution:")
        for split in ['train', 'val']:
            total = stats[split]['total']
            print(f"  {split.upper()}:")
            for cls in self.CLASS_NAMES:
                count = stats[split][cls]
                pct = count / total * 100 if total > 0 else 0
                print(f"    {cls}: {count} ({pct:.1f}%)")
        
        return stats
    
    def _save_sample(self, window: pd.DataFrame, label: int, split: str, symbol: str, idx: int):
        """Save a single sample."""
        img = self.renderer.render(window, include_volume=self.include_volume)
        class_name = self.CLASS_NAMES[label]
        filename = f"{symbol}_{idx:06d}.jpg"
        path = self.output_dir / split / class_name / filename
        Image.fromarray(img).save(path, quality=95)
    
    def generate_synthetic(
        self,
        n_samples: int = 10000,
        symbol: str = "SYNTHETIC",
        class_balance: Tuple[float, float, float] = (0.33, 0.34, 0.33),
    ) -> Dict:
        """Generate synthetic dataset with controlled class balance."""
        print(f"Generating {n_samples} synthetic samples...")
        
        samples = []
        n_bearish = int(n_samples * class_balance[0])
        n_sideways = int(n_samples * class_balance[1])
        n_bullish = n_samples - n_bearish - n_sideways
        
        print("  Generating BEARISH samples...")
        samples.extend(self._generate_trend_samples(n_bearish, TrendLabel.BEARISH))
        
        print("  Generating SIDEWAYS samples...")
        samples.extend(self._generate_trend_samples(n_sideways, TrendLabel.SIDEWAYS))
        
        print("  Generating BULLISH samples...")
        samples.extend(self._generate_trend_samples(n_bullish, TrendLabel.BULLISH))
        
        self._setup_directories()
        return self._split_and_save(samples, symbol)
    
    def _generate_trend_samples(self, n: int, trend: TrendLabel) -> List[Tuple[pd.DataFrame, int]]:
        """Generate synthetic samples for a specific trend."""
        samples = []
        for _ in range(n):
            df = self._generate_synthetic_trend(self.window_size, trend)
            samples.append((df, trend))
        return samples
    
    def _generate_synthetic_trend(self, n_candles: int, trend: TrendLabel) -> pd.DataFrame:
        """Generate synthetic OHLCV data with specified trend."""
        data = []
        base_price = 1.1000
        volatility = 0.0008
        
        if trend == TrendLabel.BULLISH:
            drift = np.random.uniform(0.0002, 0.0005)
        elif trend == TrendLabel.BEARISH:
            drift = np.random.uniform(-0.0005, -0.0002)
        else:
            drift = np.random.uniform(-0.0001, 0.0001)
        
        for i in range(n_candles):
            actual_drift = drift + np.random.randn() * drift * 0.5
            
            o = base_price
            c = o + actual_drift + np.random.randn() * volatility
            
            wick_factor = np.random.uniform(0.3, 0.8)
            h = max(o, c) + abs(np.random.randn() * volatility * wick_factor)
            l = min(o, c) - abs(np.random.randn() * volatility * wick_factor)
            
            data.append({
                'open': o, 'high': h, 'low': l, 'close': c,
                'tick_volume': np.random.randint(100, 3000),
            })
            base_price = c
        
        return pd.DataFrame(data)


def main():
    """Generate ViT dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ViT dataset")
    parser.add_argument('--data', type=str, help='Path to OHLCV CSV')
    parser.add_argument('--output', type=str, default='dataset_vit')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--labeler', type=str, default='future', choices=['future', 'structure', 'ema'])
    
    args = parser.parse_args()
    
    if args.labeler == 'future':
        labeler = FuturePriceLabeler(forward_bars=10, threshold_pct=0.5)
    elif args.labeler == 'structure':
        labeler = TrendStructureLabeler()
    else:
        labeler = EMABasedLabeler()
    
    generator = ViTDatasetGenerator(
        output_dir=args.output,
        image_size=args.image_size,
        window_size=args.window,
        stride=args.stride,
        val_split=args.val_split,
        labeler=labeler,
    )
    
    if args.synthetic or not args.data:
        generator.generate_synthetic(n_samples=args.samples)
    else:
        generator.generate_from_csv(args.data, max_samples=args.samples)


if __name__ == "__main__":
    main()
