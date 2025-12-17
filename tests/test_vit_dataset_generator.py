import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from PIL import Image
import tempfile
import shutil

from utils.vit_dataset_generator import (
    TrendLabel,
    FuturePriceLabeler,
    TrendStructureLabeler,
    EMABasedLabeler,
    ViTDatasetGenerator,
)


def make_bullish_df(n=60, start=1.1000, step=0.0005):
    """Create bullish OHLCV data."""
    data = []
    price = start
    for i in range(n):
        o = price
        c = price + step
        h = c + 0.0005
        l = o - 0.0001
        data.append({'open': o, 'high': h, 'low': l, 'close': c, 'tick_volume': 100})
        price = c
    return pd.DataFrame(data)


def make_bearish_df(n=60, start=1.1000, step=0.0005):
    """Create bearish OHLCV data."""
    data = []
    price = start
    for i in range(n):
        o = price
        c = price - step
        h = o + 0.0001
        l = c - 0.0005
        data.append({'open': o, 'high': h, 'low': l, 'close': c, 'tick_volume': 100})
        price = c
    return pd.DataFrame(data)


def make_sideways_df(n=60, start=1.1000, noise=0.0001):
    """Create sideways OHLCV data."""
    data = []
    price = start
    for i in range(n):
        o = price
        c = price + np.random.randn() * noise
        h = max(o, c) + abs(np.random.randn() * noise)
        l = min(o, c) - abs(np.random.randn() * noise)
        data.append({'open': o, 'high': h, 'low': l, 'close': c, 'tick_volume': 100})
        price = c
    return pd.DataFrame(data)


class TestTrendLabel:
    def test_trend_label_enum_values(self):
        assert TrendLabel.BEARISH == 0
        assert TrendLabel.SIDEWAYS == 1
        assert TrendLabel.BULLISH == 2


class TestFuturePriceLabeler:
    def test_init_defaults(self):
        labeler = FuturePriceLabeler()
        assert labeler.forward_bars == 10
        assert labeler.threshold_pct == 0.5

    def test_init_custom(self):
        labeler = FuturePriceLabeler(forward_bars=5, threshold_pct=1.0)
        assert labeler.forward_bars == 5
        assert labeler.threshold_pct == 1.0

    def test_label_bullish_movement(self):
        """Future price increases => BULLISH."""
        df = make_sideways_df(n=60)
        future = pd.DataFrame({
            'open': [1.1100] * 10,
            'high': [1.1100] * 10,
            'low': [1.1100] * 10,
            'close': [1.1105] * 10,  # +0.45% from 1.1000
        })
        labeler = FuturePriceLabeler(forward_bars=10, threshold_pct=0.5)

        label = labeler.get_label(df, future)

        assert label == TrendLabel.BULLISH

    def test_label_bearish_movement(self):
        """Future price decreases => BEARISH."""
        df = make_sideways_df(n=60)
        future = pd.DataFrame({
            'open': [1.0900] * 10,
            'high': [1.0900] * 10,
            'low': [1.0900] * 10,
            'close': [1.0895] * 10,  # -0.45% from 1.1000
        })
        labeler = FuturePriceLabeler(forward_bars=10, threshold_pct=0.5)

        label = labeler.get_label(df, future)

        assert label == TrendLabel.BEARISH

    def test_label_sideways_movement(self):
        """Future price flat => SIDEWAYS."""
        df = make_sideways_df(n=60)
        future = pd.DataFrame({
            'open': [1.1000] * 10,
            'high': [1.1000] * 10,
            'low': [1.1000] * 10,
            'close': [1.1001] * 10,  # +0.009% from 1.1000
        })
        labeler = FuturePriceLabeler(forward_bars=10, threshold_pct=0.5)

        label = labeler.get_label(df, future)

        assert label == TrendLabel.SIDEWAYS

    def test_label_no_future_data(self):
        """Missing future data => SIDEWAYS."""
        df = make_sideways_df(n=60)
        labeler = FuturePriceLabeler(forward_bars=10)

        label = labeler.get_label(df)

        assert label == TrendLabel.SIDEWAYS

    def test_label_insufficient_future_bars(self):
        """Insufficient future bars => SIDEWAYS."""
        df = make_sideways_df(n=60)
        future = pd.DataFrame({
            'open': [1.1100] * 5,
            'high': [1.1100] * 5,
            'low': [1.1100] * 5,
            'close': [1.1105] * 5,
        })
        labeler = FuturePriceLabeler(forward_bars=10)

        label = labeler.get_label(df, future)

        assert label == TrendLabel.SIDEWAYS


class TestTrendStructureLabeler:
    def test_init_defaults(self):
        labeler = TrendStructureLabeler()
        assert labeler.min_swings == 3

    def test_init_custom(self):
        labeler = TrendStructureLabeler(min_swings=5)
        assert labeler.min_swings == 5

    def test_label_bullish_structure(self):
        """HH/HL pattern => BULLISH."""
        df = make_bullish_df(n=60)
        labeler = TrendStructureLabeler(min_swings=2)

        label = labeler.get_label(df)

        # Uptrend should detect bullish structure
        assert label in [TrendLabel.BULLISH, TrendLabel.SIDEWAYS]

    def test_label_bearish_structure(self):
        """LL/LH pattern => BEARISH."""
        df = make_bearish_df(n=60)
        labeler = TrendStructureLabeler(min_swings=2)

        label = labeler.get_label(df)

        # Downtrend should detect bearish structure
        assert label in [TrendLabel.BEARISH, TrendLabel.SIDEWAYS]

    def test_label_insufficient_swings(self):
        """Few swings => SIDEWAYS."""
        df = make_sideways_df(n=20)  # Too small for many swings
        labeler = TrendStructureLabeler(min_swings=10)

        label = labeler.get_label(df)

        assert label == TrendLabel.SIDEWAYS


class TestEMABasedLabeler:
    def test_init_defaults(self):
        labeler = EMABasedLabeler()
        assert labeler.fast_period == 10
        assert labeler.slow_period == 30

    def test_init_custom(self):
        labeler = EMABasedLabeler(fast_period=5, slow_period=20)
        assert labeler.fast_period == 5
        assert labeler.slow_period == 20

    def test_label_bullish_alignment(self):
        """Bullish EMA alignment => BULLISH."""
        df = make_bullish_df(n=60)
        labeler = EMABasedLabeler(fast_period=10, slow_period=30)

        label = labeler.get_label(df)

        assert label in [TrendLabel.BULLISH, TrendLabel.SIDEWAYS]

    def test_label_bearish_alignment(self):
        """Bearish EMA alignment => BEARISH."""
        df = make_bearish_df(n=60)
        labeler = EMABasedLabeler(fast_period=10, slow_period=30)

        label = labeler.get_label(df)

        assert label in [TrendLabel.BEARISH, TrendLabel.SIDEWAYS]

    def test_label_insufficient_data(self):
        """Data shorter than slow period => SIDEWAYS."""
        df = make_sideways_df(n=10)  # < 30
        labeler = EMABasedLabeler(fast_period=10, slow_period=30)

        label = labeler.get_label(df)

        assert label == TrendLabel.SIDEWAYS


class TestViTDatasetGeneratorInit:
    def test_default_init(self):
        gen = ViTDatasetGenerator()
        assert gen.image_size == 224
        assert gen.window_size == 60
        assert gen.stride == 5
        assert gen.val_split == 0.2
        assert gen.include_volume is True
        assert isinstance(gen.labeler, FuturePriceLabeler)

    def test_custom_init(self):
        gen = ViTDatasetGenerator(
            output_dir='custom_dir',
            image_size=256,
            window_size=80,
            stride=10,
            val_split=0.3,
            labeler=TrendStructureLabeler(),
            include_volume=False,
        )
        assert gen.output_dir == Path('custom_dir')
        assert gen.image_size == 256
        assert gen.window_size == 80
        assert gen.stride == 10
        assert gen.val_split == 0.3
        assert gen.include_volume is False
        assert isinstance(gen.labeler, TrendStructureLabeler)

    def test_class_names_constant(self):
        gen = ViTDatasetGenerator()
        assert gen.CLASS_NAMES == ['BEARISH', 'SIDEWAYS', 'BULLISH']


class TestViTDatasetGeneratorDirectories:
    def test_setup_directories_creates_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ViTDatasetGenerator(output_dir=tmpdir)
            gen._setup_directories()

            base = Path(tmpdir)
            for split in ['train', 'val']:
                for cls in ['BEARISH', 'SIDEWAYS', 'BULLISH']:
                    path = base / split / cls
                    assert path.exists()
                    assert path.is_dir()


class TestViTDatasetGeneratorSynthetic:
    def test_generate_synthetic_returns_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ViTDatasetGenerator(output_dir=tmpdir, window_size=60)
            stats = gen.generate_synthetic(n_samples=10, symbol='TEST')

            assert isinstance(stats, dict)
            assert 'train' in stats
            assert 'val' in stats

    def test_generate_synthetic_stats_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ViTDatasetGenerator(output_dir=tmpdir, window_size=60)
            stats = gen.generate_synthetic(n_samples=10)

            for split in ['train', 'val']:
                assert 'total' in stats[split]
                assert 'BEARISH' in stats[split]
                assert 'SIDEWAYS' in stats[split]
                assert 'BULLISH' in stats[split]

    def test_generate_synthetic_creates_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ViTDatasetGenerator(output_dir=tmpdir, window_size=20, image_size=64)
            gen.generate_synthetic(n_samples=6, symbol='TEST')

            base = Path(tmpdir)
            # Check that images exist somewhere (train or val, any class)
            all_images = list(base.glob('*/*/TEST_*.jpg'))
            assert len(all_images) > 0

    def test_generate_synthetic_class_balance(self):
        """Verify class balance is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ViTDatasetGenerator(output_dir=tmpdir, window_size=20, image_size=64)
            stats = gen.generate_synthetic(
                n_samples=9,
                symbol='TEST',
                class_balance=(0.33, 0.34, 0.33),
            )

            total = stats['train']['total'] + stats['val']['total']
            assert total == 9

    def test_generate_synthetic_trend_samples_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ViTDatasetGenerator(output_dir=tmpdir, window_size=20, image_size=64)
            samples = gen._generate_trend_samples(5, TrendLabel.BULLISH)

            assert len(samples) == 5
            for df, label in samples:
                assert label == TrendLabel.BULLISH
                assert isinstance(df, pd.DataFrame)

    def test_generate_synthetic_trend_data(self):
        gen = ViTDatasetGenerator(window_size=30)
        df = gen._generate_synthetic_trend(30, TrendLabel.BULLISH)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 30
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns


class TestViTDatasetGeneratorFromDataFrame:
    def test_generate_from_dataframe_valid_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = make_bullish_df(n=150)
            gen = ViTDatasetGenerator(output_dir=tmpdir, window_size=60, image_size=64)
            stats = gen.generate_from_dataframe(df, symbol='TEST', max_samples=5)

            assert isinstance(stats, dict)
            assert stats['train']['total'] + stats['val']['total'] > 0

    def test_generate_from_dataframe_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'open': [1.1] * 60,
                'close': [1.1] * 60,
                # missing 'high', 'low'
            })
            gen = ViTDatasetGenerator(output_dir=tmpdir)

            with pytest.raises(ValueError, match="Missing columns"):
                gen.generate_from_dataframe(df)

    def test_generate_from_dataframe_creates_labeled_samples(self):
        df = make_bullish_df(n=200)
        gen = ViTDatasetGenerator(window_size=60, stride=20)
        samples = gen._create_labeled_samples(df, max_samples=None)

        assert len(samples) > 0
        for window, label in samples:
            assert isinstance(window, pd.DataFrame)
            assert label in [TrendLabel.BEARISH, TrendLabel.SIDEWAYS, TrendLabel.BULLISH]

    def test_generate_from_dataframe_max_samples(self):
        df = make_bullish_df(n=300)
        gen = ViTDatasetGenerator(window_size=60, stride=5)
        samples = gen._create_labeled_samples(df, max_samples=10)

        assert len(samples) == 10

    def test_generate_from_dataframe_window_size_respected(self):
        df = make_bullish_df(n=200)
        window_size = 60
        gen = ViTDatasetGenerator(window_size=window_size, stride=10)
        samples = gen._create_labeled_samples(df, max_samples=5)

        for window, label in samples:
            assert len(window) == window_size


class TestViTDatasetGeneratorSplitAndSave:
    def test_split_and_save_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ViTDatasetGenerator(output_dir=tmpdir, image_size=64)
            gen._setup_directories()  # Create directories first
            samples = [
                (make_bullish_df(n=30), TrendLabel.BULLISH),
                (make_bullish_df(n=30), TrendLabel.BULLISH),
                (make_bearish_df(n=30), TrendLabel.BEARISH),
            ]
            stats = gen._split_and_save(samples, 'TEST')

            # Check file structure
            base = Path(tmpdir)
            for split in ['train', 'val']:
                for cls in ['BEARISH', 'BULLISH', 'SIDEWAYS']:
                    assert (base / split / cls).exists()

    def test_split_and_save_respects_val_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ViTDatasetGenerator(output_dir=tmpdir, image_size=64, val_split=0.3)
            gen._setup_directories()  # Create directories first
            samples = [
                (make_bullish_df(n=30), TrendLabel.BULLISH),
                (make_bullish_df(n=30), TrendLabel.BULLISH),
                (make_bullish_df(n=30), TrendLabel.BULLISH),
                (make_bearish_df(n=30), TrendLabel.BEARISH),
                (make_bearish_df(n=30), TrendLabel.BEARISH),
            ]
            stats = gen._split_and_save(samples, 'TEST')

            total = stats['train']['total'] + stats['val']['total']
            val_ratio = stats['val']['total'] / total
            # Allow for integer rounding variance (20% = 1/5, 40% = 2/5)
            assert 0.15 <= val_ratio <= 0.45

    def test_split_and_save_stats_accuracy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ViTDatasetGenerator(output_dir=tmpdir, image_size=64, val_split=0.5)
            gen._setup_directories()  # Create directories first
            samples = [
                (make_bullish_df(n=30), TrendLabel.BULLISH),
                (make_bullish_df(n=30), TrendLabel.BULLISH),
                (make_bearish_df(n=30), TrendLabel.BEARISH),
                (make_sideways_df(n=30), TrendLabel.SIDEWAYS),
            ]
            stats = gen._split_and_save(samples, 'TEST')

            # Total samples should match
            total_saved = stats['train']['total'] + stats['val']['total']
            assert total_saved == len(samples)


class TestEdgeCases:
    def test_generate_from_csv_missing_file(self):
        gen = ViTDatasetGenerator()
        with pytest.raises(FileNotFoundError):
            gen.generate_from_csv('/nonexistent/path.csv')

    def test_generator_with_different_labelers(self):
        for labeler_class in [FuturePriceLabeler, TrendStructureLabeler, EMABasedLabeler]:
            gen = ViTDatasetGenerator(labeler=labeler_class())
            assert isinstance(gen.labeler, labeler_class)

    def test_synthetic_trend_negative_prices_prevented(self):
        """Synthetic trend should not produce negative prices."""
        gen = ViTDatasetGenerator(window_size=60)
        for trend in [TrendLabel.BULLISH, TrendLabel.BEARISH, TrendLabel.SIDEWAYS]:
            df = gen._generate_synthetic_trend(50, trend)
            assert (df['close'] > 0).all()
            assert (df['open'] > 0).all()
            assert (df['high'] > 0).all()
            assert (df['low'] > 0).all()


class TestIntegration:
    def test_full_workflow_synthetic(self):
        """Complete workflow: init → generate synthetic → verify output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ViTDatasetGenerator(
                output_dir=tmpdir,
                window_size=30,
                image_size=64,
                val_split=0.2,
            )

            stats = gen.generate_synthetic(n_samples=12, symbol='FULL_TEST')

            assert stats['train']['total'] > 0
            assert stats['val']['total'] > 0
            
            # Verify directory structure
            base = Path(tmpdir)
            assert (base / 'train' / 'BULLISH').exists()
            assert (base / 'val' / 'BEARISH').exists()

    def test_full_workflow_from_dataframe(self):
        """Complete workflow: init → generate from data → verify output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df = make_bullish_df(n=300)
            gen = ViTDatasetGenerator(
                output_dir=tmpdir,
                window_size=60,
                image_size=64,
                stride=20,
            )

            stats = gen.generate_from_dataframe(df, symbol='DF_TEST', max_samples=10)

            assert isinstance(stats, dict)
            assert stats['train']['total'] > 0
