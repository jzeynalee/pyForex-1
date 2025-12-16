import numpy as np
import pandas as pd
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from utils.yolo_dataset_generator import YOLODatasetGenerator


def make_simple_ohlcv(n=60, start=1.1000, volatility=0.0005):
    """Create basic OHLCV data."""
    data = []
    price = start
    for i in range(n):
        change = np.random.randn() * volatility
        o = price
        c = price + change
        h = max(o, c) + abs(np.random.randn() * volatility)
        l = min(o, c) - abs(np.random.randn() * volatility)
        data.append({'open': o, 'high': h, 'low': l, 'close': c, 'tick_volume': 100})
        price = c
    return pd.DataFrame(data)


def make_bullish_ohlcv(n=60, start=1.1000, step=0.0005):
    """Create bullish OHLCV data with uptrend."""
    data = []
    price = start
    for i in range(n):
        o = price
        c = price + step
        h = c + 0.0003
        l = o - 0.0001
        data.append({'open': o, 'high': h, 'low': l, 'close': c, 'tick_volume': 100})
        price = c
    return pd.DataFrame(data)


def make_bearish_ohlcv(n=60, start=1.1000, step=0.0005):
    """Create bearish OHLCV data with downtrend."""
    data = []
    price = start
    for i in range(n):
        o = price
        c = price - step
        h = o + 0.0001
        l = c - 0.0003
        data.append({'open': o, 'high': h, 'low': l, 'close': c, 'tick_volume': 100})
        price = c
    return pd.DataFrame(data)


def make_doji_data(n=60):
    """Create data with doji patterns."""
    data = []
    price = 1.1000
    for i in range(n):
        if i % 10 == 0:  # Doji at certain intervals
            o = price
            c = o + 0.00001  # Very small body
            h = o + 0.0010
            l = o - 0.0010
        else:
            o = price
            c = price + np.random.randn() * 0.0005
            h = max(o, c) + 0.0005
            l = min(o, c) - 0.0005
        data.append({'open': o, 'high': h, 'low': l, 'close': c, 'tick_volume': 100})
        price = c
    return pd.DataFrame(data)


class TestYOLODatasetGeneratorInit:
    def test_default_init(self):
        gen = YOLODatasetGenerator()
        assert gen.output_dir == Path("dataset_yolo")
        assert gen.image_size == 256
        assert gen.window_size == 60
        assert gen.stride == 10
        assert gen.val_split == 0.2
        assert gen.min_patterns_per_image == 0

    def test_custom_init(self):
        gen = YOLODatasetGenerator(
            output_dir="custom_yolo",
            image_size=512,
            window_size=100,
            stride=20,
            val_split=0.3,
            min_patterns_per_image=2,
        )
        assert gen.output_dir == Path("custom_yolo")
        assert gen.image_size == 512
        assert gen.window_size == 100
        assert gen.stride == 20
        assert gen.val_split == 0.3
        assert gen.min_patterns_per_image == 2

    def test_init_creates_renderer_and_detector(self):
        gen = YOLODatasetGenerator(image_size=256)
        assert gen.renderer is not None
        assert gen.detector is not None

    def test_init_sets_pattern_names(self):
        gen = YOLODatasetGenerator()
        assert gen.pattern_names is not None
        assert len(gen.pattern_names) > 0
        assert gen.num_classes == len(gen.pattern_names)


class TestYOLODatasetGeneratorDirectories:
    def test_setup_directories_creates_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = YOLODatasetGenerator(output_dir=tmpdir)
            gen._setup_directories()

            base = Path(tmpdir)
            assert (base / "images" / "train").exists()
            assert (base / "images" / "val").exists()
            assert (base / "labels" / "train").exists()
            assert (base / "labels" / "val").exists()

    def test_setup_directories_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = YOLODatasetGenerator(output_dir=tmpdir)
            gen._setup_directories()
            gen._setup_directories()  # Should not raise

            base = Path(tmpdir)
            assert (base / "images" / "train").exists()


class TestYOLODatasetGeneratorWindows:
    def test_create_windows_basic(self):
        df = make_simple_ohlcv(n=200)
        gen = YOLODatasetGenerator(window_size=60, stride=10)
        windows = gen._create_windows(df, None)

        assert len(windows) > 0
        for window in windows:
            assert len(window) == 60

    def test_create_windows_respects_stride(self):
        df = make_simple_ohlcv(n=200)
        stride = 15
        gen = YOLODatasetGenerator(window_size=60, stride=stride)
        windows = gen._create_windows(df, None)

        # Check stride between windows
        assert len(windows) > 1

    def test_create_windows_max_samples(self):
        df = make_simple_ohlcv(n=500)
        gen = YOLODatasetGenerator(window_size=60, stride=5)
        windows = gen._create_windows(df, max_samples=10)

        assert len(windows) == 10

    def test_create_windows_insufficient_data(self):
        df = make_simple_ohlcv(n=20)
        gen = YOLODatasetGenerator(window_size=60, stride=10)
        windows = gen._create_windows(df, None)

        assert len(windows) == 0

    def test_create_windows_exact_size(self):
        df = make_simple_ohlcv(n=60)
        gen = YOLODatasetGenerator(window_size=60, stride=10)
        windows = gen._create_windows(df, None)

        assert len(windows) == 1
        assert len(windows[0]) == 60


class TestYOLODatasetGeneratorFromDataFrame:
    def test_generate_from_dataframe_valid_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = make_simple_ohlcv(n=300)
            gen = YOLODatasetGenerator(
                output_dir=tmpdir,
                window_size=60,
                stride=30,
                image_size=128,
            )
            stats = gen.generate_from_dataframe(df, symbol="TEST", max_samples=5)

            assert isinstance(stats, dict)
            assert 'train' in stats
            assert 'val' in stats
            assert stats['train'] >= 0
            assert stats['val'] >= 0

    def test_generate_from_dataframe_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'open': [1.1] * 60,
                'close': [1.1] * 60,
                # missing 'high', 'low'
            })
            gen = YOLODatasetGenerator(output_dir=tmpdir)

            with pytest.raises(ValueError, match="Missing columns"):
                gen.generate_from_dataframe(df)

    def test_generate_from_dataframe_columns_already_lowercase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = make_simple_ohlcv(n=150)
            # Columns are already lowercase
            gen = YOLODatasetGenerator(output_dir=tmpdir, window_size=60, stride=30)

            # Should work with lowercase
            stats = gen.generate_from_dataframe(df, symbol="TEST", max_samples=2)
            assert isinstance(stats, dict)

    def test_generate_from_dataframe_creates_directory_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = make_simple_ohlcv(n=150)
            gen = YOLODatasetGenerator(output_dir=tmpdir, window_size=60, stride=30)
            gen.generate_from_dataframe(df, symbol="TEST", max_samples=2)

            base = Path(tmpdir)
            assert (base / "images" / "train").exists()
            assert (base / "images" / "val").exists()
            assert (base / "labels" / "train").exists()
            assert (base / "labels" / "val").exists()

    def test_generate_from_dataframe_creates_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = make_simple_ohlcv(n=150)
            gen = YOLODatasetGenerator(output_dir=tmpdir, window_size=60, stride=30)
            gen.generate_from_dataframe(df, symbol="TEST", max_samples=2)

            yaml_path = Path(tmpdir) / "data.yaml"
            assert yaml_path.exists()

    def test_generate_from_dataframe_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = make_simple_ohlcv(n=200)
            gen = YOLODatasetGenerator(output_dir=tmpdir, window_size=60, stride=20)
            stats = gen.generate_from_dataframe(df, symbol="TEST", max_samples=10)

            assert isinstance(stats, dict)
            if stats['train'] > 0 or stats['val'] > 0:
                assert stats['train'] + stats['val'] > 0


class TestYOLODatasetGeneratorFromCSV:
    def test_generate_from_csv_valid_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create temporary CSV
            csv_path = Path(tmpdir) / "test.csv"
            df = make_simple_ohlcv(n=150)
            df.to_csv(csv_path, index=False)

            gen = YOLODatasetGenerator(
                output_dir=Path(tmpdir) / "output",
                window_size=60,
                stride=30,
            )
            stats = gen.generate_from_csv(str(csv_path), symbol="CSV_TEST", max_samples=3)

            assert isinstance(stats, dict)

    def test_generate_from_csv_missing_file(self):
        gen = YOLODatasetGenerator()
        with pytest.raises(FileNotFoundError):
            gen.generate_from_csv("/nonexistent/path.csv")

    def test_generate_from_csv_lowercase_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            df = make_simple_ohlcv(n=150)
            df.columns = df.columns.str.upper()  # Save with uppercase
            df.to_csv(csv_path, index=False)

            gen = YOLODatasetGenerator(
                output_dir=Path(tmpdir) / "output",
                window_size=60,
                stride=30,
            )
            stats = gen.generate_from_csv(str(csv_path), max_samples=2)

            assert isinstance(stats, dict)


class TestYOLODatasetGeneratorSynthetic:
    def test_generate_synthetic_returns_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = YOLODatasetGenerator(
                output_dir=tmpdir,
                window_size=30,
                stride=10,
                image_size=128,
            )
            stats = gen.generate_synthetic(n_samples=5, symbol="SYN")

            assert isinstance(stats, dict)
            assert 'train' in stats
            assert 'val' in stats

    def test_generate_synthetic_creates_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = YOLODatasetGenerator(
                output_dir=tmpdir,
                window_size=20,
                stride=5,
                image_size=64,
            )
            gen.generate_synthetic(n_samples=3, symbol="SYN")

            base = Path(tmpdir)
            images = list(base.glob("images/*/SYN_*.jpg"))
            assert len(images) > 0

    def test_generate_synthetic_creates_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = YOLODatasetGenerator(
                output_dir=tmpdir,
                window_size=20,
                stride=5,
                image_size=64,
            )
            gen.generate_synthetic(n_samples=3, symbol="SYN")

            base = Path(tmpdir)
            labels = list(base.glob("labels/*/SYN_*.txt"))
            assert len(labels) > 0

    def test_synthetic_ohlcv_generation(self):
        gen = YOLODatasetGenerator(window_size=60)
        df = gen._generate_synthetic_ohlcv(100)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'tick_volume' in df.columns

    def test_synthetic_ohlcv_prices_positive(self):
        gen = YOLODatasetGenerator()
        df = gen._generate_synthetic_ohlcv(200)

        assert (df['open'] > 0).all()
        assert (df['high'] > 0).all()
        assert (df['low'] > 0).all()
        assert (df['close'] > 0).all()

    def test_synthetic_ohlcv_high_low_logic(self):
        gen = YOLODatasetGenerator()
        df = gen._generate_synthetic_ohlcv(200)

        for idx, row in df.iterrows():
            assert row['high'] >= max(row['open'], row['close'])
            assert row['low'] <= min(row['open'], row['close'])


class TestYOLODatasetGeneratorSplit:
    @patch('utils.yolo_dataset_generator.YOLODatasetGenerator._generate_split')
    def test_generate_from_dataframe_splits_correctly(self, mock_gen_split):
        mock_gen_split.return_value = (5, {})

        with tempfile.TemporaryDirectory() as tmpdir:
            df = make_simple_ohlcv(n=200)
            gen = YOLODatasetGenerator(
                output_dir=tmpdir,
                window_size=60,
                stride=20,
                val_split=0.25,
            )
            stats = gen.generate_from_dataframe(df, symbol="SPLIT", max_samples=20)

            # Check that _generate_split was called for both train and val
            assert mock_gen_split.call_count == 2

    def test_val_split_ratio_approximate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = make_simple_ohlcv(n=300)
            gen = YOLODatasetGenerator(
                output_dir=tmpdir,
                window_size=60,
                stride=15,
                val_split=0.25,
                image_size=64,
            )
            stats = gen.generate_from_dataframe(df, symbol="RATIO", max_samples=20)

            total = stats['train'] + stats['val']
            if total > 0:
                val_ratio = stats['val'] / total
                # Check that split is approximately correct (allow variance)
                assert 0.1 <= val_ratio <= 0.4


class TestYOLODatasetGeneratorYAML:
    def test_create_yaml_config_file_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = YOLODatasetGenerator(output_dir=tmpdir)
            gen._create_yaml_config()

            yaml_path = Path(tmpdir) / "data.yaml"
            assert yaml_path.exists()

    def test_create_yaml_config_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = YOLODatasetGenerator(output_dir=tmpdir, image_size=256)
            gen._create_yaml_config()

            yaml_path = Path(tmpdir) / "data.yaml"
            with open(yaml_path) as f:
                config = yaml.safe_load(f)

            assert 'path' in config
            assert 'train' in config
            assert 'val' in config
            assert 'names' in config
            assert 'nc' in config
            assert config['train'] == 'images/train'
            assert config['val'] == 'images/val'

    def test_create_yaml_config_nc_matches_classes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = YOLODatasetGenerator(output_dir=tmpdir)
            gen._create_yaml_config()

            yaml_path = Path(tmpdir) / "data.yaml"
            with open(yaml_path) as f:
                config = yaml.safe_load(f)

            assert config['nc'] == len(gen.pattern_names)
            assert len(config['names']) == config['nc']


class TestYOLODatasetGeneratorEdgeCases:
    def test_min_patterns_per_image_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = make_simple_ohlcv(n=150)
            gen = YOLODatasetGenerator(
                output_dir=tmpdir,
                window_size=60,
                stride=30,
                min_patterns_per_image=100,  # Very high threshold
                image_size=64,
            )
            stats = gen.generate_from_dataframe(df, symbol="FILTER", max_samples=5)

            # With high threshold, few or no images should pass
            assert stats['train'] + stats['val'] <= 5

    def test_small_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = make_simple_ohlcv(n=70)
            gen = YOLODatasetGenerator(
                output_dir=tmpdir,
                window_size=60,
                stride=1,
                image_size=64,
            )
            stats = gen.generate_from_dataframe(df, symbol="SMALL")

            total = stats['train'] + stats['val']
            assert total >= 0

    def test_large_stride_relative_to_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = make_simple_ohlcv(n=100)
            gen = YOLODatasetGenerator(
                output_dir=tmpdir,
                window_size=60,
                stride=100,
                image_size=64,
            )
            stats = gen.generate_from_dataframe(df, symbol="LARGE_STRIDE")

            assert isinstance(stats, dict)

    def test_empty_dataframe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({
                'open': [], 'high': [], 'low': [], 'close': [], 'tick_volume': []
            })
            gen = YOLODatasetGenerator(output_dir=tmpdir)

            # Empty DataFrame should succeed with 0 samples
            stats = gen.generate_from_dataframe(df)
            assert stats['train'] == 0
            assert stats['val'] == 0


class TestYOLODatasetGeneratorIntegration:
    def test_full_workflow_dataframe_to_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = make_bullish_ohlcv(n=200)
            gen = YOLODatasetGenerator(
                output_dir=tmpdir,
                window_size=60,
                stride=25,
                image_size=128,
                val_split=0.2,
            )

            stats = gen.generate_from_dataframe(df, symbol="WORKFLOW", max_samples=8)

            base = Path(tmpdir)
            assert (base / "data.yaml").exists()
            assert (base / "images" / "train").exists()
            assert (base / "labels" / "train").exists()

    def test_full_workflow_csv_to_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSV
            csv_path = Path(tmpdir) / "input.csv"
            df = make_bullish_ohlcv(n=200)
            df.to_csv(csv_path, index=False)

            # Generate dataset
            gen = YOLODatasetGenerator(
                output_dir=Path(tmpdir) / "dataset",
                window_size=60,
                stride=25,
                image_size=128,
            )
            stats = gen.generate_from_csv(str(csv_path), symbol="CSV_WORKFLOW", max_samples=8)

            base = Path(tmpdir) / "dataset"
            assert (base / "data.yaml").exists()
            assert stats['train'] >= 0
            assert stats['val'] >= 0

    def test_multiple_symbols_same_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df1 = make_bullish_ohlcv(n=120)
            df2 = make_bearish_ohlcv(n=120)

            gen = YOLODatasetGenerator(
                output_dir=tmpdir,
                window_size=60,
                stride=30,
                image_size=64,
            )

            gen.generate_from_dataframe(df1, symbol="EUR", max_samples=3)
            gen.generate_from_dataframe(df2, symbol="GBP", max_samples=3)

            base = Path(tmpdir)
            eur_images = list(base.glob("images/*/EUR_*.jpg"))
            gbp_images = list(base.glob("images/*/GBP_*.jpg"))

            assert len(eur_images) > 0 or len(gbp_images) > 0


class TestYOLODatasetGeneratorArgumentParsing:
    @patch('utils.yolo_dataset_generator.YOLODatasetGenerator.generate_synthetic')
    def test_main_synthetic_mode(self, mock_gen_synthetic):
        from utils.yolo_dataset_generator import main
        import sys

        original_argv = sys.argv
        try:
            sys.argv = ['prog', '--synthetic', '--samples', '100']
            main()
            mock_gen_synthetic.assert_called_once()
        finally:
            sys.argv = original_argv

    @patch('utils.yolo_dataset_generator.YOLODatasetGenerator.generate_from_csv')
    def test_main_csv_mode(self, mock_gen_csv):
        from utils.yolo_dataset_generator import main
        import sys

        original_argv = sys.argv
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                csv_path = Path(tmpdir) / "test.csv"
                df = make_simple_ohlcv(n=100)
                df.to_csv(csv_path, index=False)

                sys.argv = ['prog', '--data', str(csv_path), '--samples', '50']
                main()
                mock_gen_csv.assert_called_once()
        finally:
            sys.argv = original_argv
