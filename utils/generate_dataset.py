#!/usr/bin/env python3
# generate_dataset.py
"""
Main Dataset Generation Script for pyForex.

Generates training datasets for:
1. YOLO - Candlestick pattern detection

Usage:
    python generate_dataset.py --synthetic --samples 10000 --output datasets/
    python generate_dataset.py --data data/EURUSD_H1.csv --output datasets/
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

from .candle_to_image import CandlestickRenderer, candle_image
from .pattern_detector import CandlestickPatternDetector, PATTERN_NAMES
from .yolo_dataset_generator import YOLODatasetGenerator


def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║         pyForex Dataset Generator v1.0                        ║
║         Generate training data for YOLO models                ║
╚═══════════════════════════════════════════════════════════════╝
""")


def generate_yolo_dataset(data_path, output_dir, samples, synthetic, image_size=256, window_size=60):
    """Generate YOLO pattern detection dataset."""
    print("\n🎯 Generating YOLO Dataset...")
    print("-" * 50)
    
    generator = YOLODatasetGenerator(
        output_dir=output_dir,
        image_size=image_size,
        window_size=window_size,
        stride=10,
        val_split=0.2,
    )
    
    if synthetic:
        stats = generator.generate_synthetic(n_samples=samples, symbol="SYN")
    else:
        stats = generator.generate_from_csv(data_path, max_samples=samples)
    
    return stats


def save_stats(stats, output_path):
    """Save generation statistics to JSON."""
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"📄 Stats saved to {output_path}")


def main():
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="Generate training datasets for pyForex models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--data', type=str, help='Path to OHLCV CSV file')
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic data')
    parser.add_argument('--output', type=str, default='datasets', help='Output directory')
    parser.add_argument('--samples', type=int, default=5000, help='Number of samples')
    parser.add_argument('--yolo-size', type=int, default=256, help='YOLO image size')
    parser.add_argument('--window', type=int, default=60, help='Candles per image')
    
    args = parser.parse_args()
    
    if not args.synthetic and not args.data:
        print("⚠️  No data source specified. Using synthetic data...\n")
        args.synthetic = True
    
    if args.data and not args.synthetic:
        if not Path(args.data).exists():
            print(f"❌ Data file not found: {args.data}")
            sys.exit(1)
    
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    all_stats = {
        'generated_at': datetime.now().isoformat(),
        'source': 'synthetic' if args.synthetic else args.data,
        'samples_requested': args.samples,
    }
    
    yolo_output = output_base / "yolo"
    yolo_stats = generate_yolo_dataset(
            data_path=args.data,
            output_dir=str(yolo_output),
            samples=args.samples,
            synthetic=args.synthetic,
            image_size=args.yolo_size,
            window_size=args.window,
    )
    all_stats['yolo'] = yolo_stats
    
    stats_path = output_base / "generation_stats.json"
    save_stats(all_stats, str(stats_path))
    
    print("\n" + "=" * 60)
    print("✅ DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"📁 Output directory: {output_base.absolute()}")
    
    print(f"\n🎯 YOLO Dataset: {output_base / 'yolo'}")
    print(f"   Train: {all_stats['yolo'].get('train', 'N/A')} images")
    print(f"   Val: {all_stats['yolo'].get('val', 'N/A')} images")
    
    print("\n" + "=" * 60)
    print("\n📖 Next Steps:")
    print(f"""
   YOLO Training:
   $ yolo detect train data={output_base}/yolo/data.yaml model=yolov8n.pt epochs=80
""")


if __name__ == "__main__":
    main()
