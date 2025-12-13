#!/usr/bin/env python3
"""
Unit tests for pyForex main.py CLI entry point.

Run with: pytest tests/test_main.py -v
Coverage: pytest tests/test_main.py --cov=main --cov-report=html
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime

import pytest

# Ensure project root is in path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import module under test
from main import (
    AppConfig,
    CONFIG,
    setup_logging,
    SystemChecker,
    create_parser,
    cmd_multi_style,
    cmd_live,
    cmd_backtest,
    cmd_train,
    cmd_predict,
    cmd_generate,
    cmd_status,
    main,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_logging():
    """Clean up logging handlers after each test to prevent file lock issues on Windows."""
    yield
    # After each test, close and remove all handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        root_logger.removeHandler(handler)
    
    # Also clean up the pyforex logger
    pyforex_logger = logging.getLogger("pyforex")
    for handler in pyforex_logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        pyforex_logger.removeHandler(handler)


@pytest.fixture
def logger():
    """Create a test logger."""
    return logging.getLogger("test_pyforex")


@pytest.fixture
def system_checker(logger):
    """Create a SystemChecker instance."""
    return SystemChecker(logger)


@pytest.fixture
def parser():
    """Create argument parser."""
    return create_parser()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_csv_data(temp_dir):
    """Create a mock CSV data file."""
    csv_path = temp_dir / "test_data.csv"
    csv_content = """time,open,high,low,close,volume
2024-01-01 00:00:00,1.1000,1.1050,1.0950,1.1020,1000
2024-01-01 01:00:00,1.1020,1.1080,1.1000,1.1060,1200
2024-01-01 02:00:00,1.1060,1.1100,1.1040,1.1090,1100
2024-01-01 03:00:00,1.1090,1.1120,1.1070,1.1100,900
2024-01-01 04:00:00,1.1100,1.1130,1.1050,1.1080,1300
"""
    csv_path.write_text(csv_content)
    return csv_path


# ============================================================================
# TEST: AppConfig
# ============================================================================

class TestAppConfig:
    """Tests for AppConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AppConfig()
        
        assert config.default_symbol == "EURUSD"
        assert config.default_timeframe == "H1"
        assert config.default_sequence_length == 60
        assert config.default_image_size == 224
        assert config.log_level == "INFO"
        assert config.log_to_file is True

    def test_paths_are_path_objects(self):
        """Test that path attributes are Path objects."""
        config = AppConfig()
        
        assert isinstance(config.project_root, Path)
        assert isinstance(config.weights_dir, Path)
        assert isinstance(config.data_dir, Path)
        assert isinstance(config.logs_dir, Path)

    def test_weights_dir_structure(self):
        """Test weights directory path structure."""
        config = AppConfig()
        
        assert "models" in str(config.weights_dir)
        assert "weights" in str(config.weights_dir)

    def test_global_config_instance(self):
        """Test that global CONFIG instance exists."""
        assert CONFIG is not None
        assert isinstance(CONFIG, AppConfig)


# ============================================================================
# TEST: setup_logging
# ============================================================================

class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_logger(self):
        """Test that setup_logging returns a logger."""
        logger = setup_logging()
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "pyforex"

    def test_default_log_level(self):
        """Test default log level is INFO."""
        logger = setup_logging(level="INFO", verbose=False)
        
        # Logger should have INFO level (20)
        assert logging.getLogger().level == logging.INFO

    def test_verbose_enables_debug(self):
        """Test verbose flag enables DEBUG level."""
        logger = setup_logging(verbose=True)
        
        assert logging.getLogger().level == logging.DEBUG

    def test_log_file_creation(self, temp_dir):
        """Test log file is created when specified."""
        log_path = temp_dir / "test.log"
        
        logger = setup_logging(log_file=str(log_path))
        logger.info("Test message")
        
        # Force flush and close all handlers to release file lock (Windows compatibility)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.flush()
            handler.close()
            root_logger.removeHandler(handler)
        
        assert log_path.exists()

    def test_log_directory_created(self, temp_dir):
        """Test log directory is created if it doesn't exist."""
        nested_path = temp_dir / "subdir" / "logs" / "test.log"
        
        setup_logging(log_file=str(nested_path))
        
        # Close handlers to release file lock (Windows compatibility)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.flush()
            handler.close()
            root_logger.removeHandler(handler)
        
        assert nested_path.parent.exists()

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    def test_log_levels(self, level):
        """Test different log levels are respected."""
        logger = setup_logging(level=level)
        
        expected_level = getattr(logging, level)
        assert logging.getLogger().level == expected_level


# ============================================================================
# TEST: SystemChecker
# ============================================================================

class TestSystemChecker:
    """Tests for SystemChecker class."""

    def test_initialization(self, system_checker):
        """Test SystemChecker initializes correctly."""
        assert system_checker.issues == []
        assert system_checker.warnings == []

    def test_check_weights_missing_directory(self, system_checker):
        """Test check_weights when directory doesn't exist."""
        with patch.object(CONFIG, 'weights_dir', Path("/nonexistent/path")):
            result = system_checker.check_weights()
        
        assert result is False
        assert len(system_checker.issues) > 0

    def test_check_weights_missing_files(self, system_checker, temp_dir):
        """Test check_weights when specific files are missing."""
        with patch.object(CONFIG, 'weights_dir', temp_dir):
            result = system_checker.check_weights(required=["model.pt"])
        
        assert result is False
        assert len(system_checker.warnings) > 0

    def test_check_weights_files_exist(self, system_checker, temp_dir):
        """Test check_weights when files exist (TCN only after LSTM removal)."""
        # Create mock weight files (TCN-based now)
        (temp_dir / "tcn_best.pt").touch()
        (temp_dir / "fusion_best.pt").touch()

        with patch.object(CONFIG, 'weights_dir', temp_dir):
            result = system_checker.check_weights()

        assert result is True

    def test_check_mt5_not_installed(self, system_checker):
        """Test check_mt5 when MetaTrader5 is not installed."""
        with patch.dict(sys.modules, {'MetaTrader5': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                result = system_checker.check_mt5()
        
        assert result is False
        assert len(system_checker.warnings) > 0

    def test_check_mt5_import_success_init_fail(self, system_checker):
        """Test check_mt5 when MT5 installed but not initialized."""
        mock_mt5 = MagicMock()
        mock_mt5.initialize.return_value = False
        
        with patch.dict(sys.modules, {'MetaTrader5': mock_mt5}):
            result = system_checker.check_mt5()
        
        assert result is False

    def test_check_cuda_with_torch(self, system_checker):
        """Test check_cuda returns correct structure."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_name.return_value = "NVIDIA Test GPU"
        mock_torch.__version__ = "2.0.0"
        
        with patch.dict(sys.modules, {'torch': mock_torch}):
            result = system_checker.check_cuda()
        
        assert "cuda_available" in result
        assert result["cuda_available"] is True
        assert result["device_count"] == 1

    def test_check_cuda_not_available(self, system_checker):
        """Test check_cuda when CUDA is not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.__version__ = "2.0.0"
        
        with patch.dict(sys.modules, {'torch': mock_torch}):
            result = system_checker.check_cuda()
        
        assert result["cuda_available"] is False
        assert result["device_count"] == 0

    def test_check_dependencies_all_present(self, system_checker):
        """Test check_dependencies returns True when required deps are present."""
        # This test passes if required dependencies (torch, numpy, pandas, sklearn) are available
        # In the test environment, some may be missing, so we mock them
        mock_modules = {
            'torch': MagicMock(),
            'numpy': MagicMock(),
            'pandas': MagicMock(),
            'sklearn': MagicMock(),
        }
        
        with patch.dict(sys.modules, mock_modules):
            result = system_checker.check_dependencies()
        
        # Should return True when required deps are available
        assert result is True

    def test_check_data_file_exists(self, system_checker, mock_csv_data):
        """Test check_data with existing file."""
        result = system_checker.check_data(str(mock_csv_data))
        
        assert result is True

    def test_check_data_file_missing(self, system_checker):
        """Test check_data with missing file."""
        result = system_checker.check_data("/nonexistent/file.csv")
        
        assert result is False
        assert len(system_checker.issues) > 0

    def test_check_data_no_path(self, system_checker):
        """Test check_data with no path provided."""
        result = system_checker.check_data(None)
        
        assert result is True  # No path = nothing to check

    @pytest.mark.parametrize("mode", ["live", "backtest", "train", "predict"])
    def test_run_all_checks_modes(self, system_checker, mode):
        """Test run_all_checks for different modes."""
        # Mock all individual checks to return True
        with patch.object(system_checker, 'check_dependencies', return_value=True):
            with patch.object(system_checker, 'check_mt5', return_value=True):
                with patch.object(system_checker, 'check_weights', return_value=True):
                    with patch.object(system_checker, 'check_cuda', return_value={"cuda_available": True}):
                        result = system_checker.run_all_checks(mode)
        
        # Should complete without critical issues
        assert isinstance(result, bool)


# ============================================================================
# TEST: Argument Parser
# ============================================================================

class TestArgumentParser:
    """Tests for CLI argument parser."""

    def test_parser_creation(self, parser):
        """Test parser is created successfully."""
        assert parser is not None
        assert parser.prog == "pyforex"

    def test_no_command_shows_help(self, parser):
        """Test parsing with no command."""
        args = parser.parse_args([])
        
        assert args.command is None

    # --- Multi command ---
    def test_multi_command_defaults(self, parser):
        """Test multi command with defaults."""
        args = parser.parse_args(["multi"])
        
        assert args.command == "multi"
        assert args.symbol == "EURUSD"
        assert args.mock is False
        assert args.no_scalp is False
        assert args.no_intraday is False
        assert args.no_swing is False

    def test_multi_command_custom_symbol(self, parser):
        """Test multi command with custom symbol."""
        args = parser.parse_args(["multi", "--symbol", "GBPUSD"])
        
        assert args.symbol == "GBPUSD"

    def test_multi_command_disable_strategies(self, parser):
        """Test multi command with disabled strategies."""
        args = parser.parse_args(["multi", "--no-scalp", "--no-swing"])
        
        assert args.no_scalp is True
        assert args.no_intraday is False
        assert args.no_swing is True

    def test_multi_command_mock(self, parser):
        """Test multi command with mock flag."""
        args = parser.parse_args(["multi", "--mock"])
        
        assert args.mock is True

    # --- Live command ---
    def test_live_command_defaults(self, parser):
        """Test live command with defaults."""
        args = parser.parse_args(["live"])
        
        assert args.command == "live"
        assert args.symbol == "EURUSD"
        assert args.timeframe == "H1"
        assert args.strategy == "neural"
        assert args.interval == 10.0

    def test_live_command_all_options(self, parser):
        """Test live command with all options (TCN only after LSTM removal)."""
        args = parser.parse_args([
            "live",
            "--symbol", "USDJPY",
            "--timeframe", "M15",
            "--strategy", "neural",  # Changed from 'lstm' to 'neural' (TCN-based)
            "--interval", "5.0",
            "--mock"
        ])

        assert args.symbol == "USDJPY"
        assert args.timeframe == "M15"
        assert args.strategy == "neural"  # TCN-based neural strategy
        assert args.interval == 5.0
        assert args.mock is True

    # --- Backtest command ---
    def test_backtest_command_required_data(self, parser):
        """Test backtest command requires data argument."""
        with pytest.raises(SystemExit):
            parser.parse_args(["backtest"])  # Missing --data

    def test_backtest_command_with_data(self, parser):
        """Test backtest command with data."""
        args = parser.parse_args(["backtest", "--data", "test.csv"])
        
        assert args.command == "backtest"
        assert args.data == "test.csv"
        assert args.strategy == "neural"
        assert args.balance == 10000.0

    def test_backtest_command_all_options(self, parser):
        """Test backtest command with all options (TCN only after LSTM removal)."""
        args = parser.parse_args([
            "backtest",
            "--data", "data.csv",
            "--strategy", "neural",  # Changed from 'lstm' to 'neural' (TCN-based)
            "--balance", "50000",
            "--output", "results.json"
        ])

        assert args.strategy == "neural"  # TCN-based
        assert args.balance == 50000.0
        assert args.output == "results.json"

    # --- Train command ---
    def test_train_command_model_required(self, parser):
        """Test train command requires model argument."""
        with pytest.raises(SystemExit):
            parser.parse_args(["train"])  # Missing model

    @pytest.mark.parametrize("model", ["tcn", "vit", "vit-finetune", "fusion", "yolo", "trend"])
    def test_train_command_valid_models(self, parser, model):
        """Test train command accepts all valid models (TCN instead of LSTM)."""
        args = parser.parse_args(["train", model])

        assert args.command == "train"
        assert args.model == model

    def test_train_command_invalid_model(self, parser):
        """Test train command rejects invalid model."""
        with pytest.raises(SystemExit):
            parser.parse_args(["train", "invalid_model"])

    def test_train_command_options(self, parser):
        """Test train command with options (TCN instead of LSTM)."""
        args = parser.parse_args([
            "train", "tcn",  # Changed from 'lstm' to 'tcn'
            "--epochs", "100",
            "--batch-size", "32",
            "--lr", "0.0001",
            "--seq-len", "120"
        ])

        assert args.epochs == 100
        assert args.batch_size == 32
        assert args.lr == 0.0001
        assert args.seq_len == 120

    # --- Predict command ---
    def test_predict_command_defaults(self, parser):
        """Test predict command defaults."""
        args = parser.parse_args(["predict"])
        
        assert args.command == "predict"
        assert args.symbol == "EURUSD"
        assert args.simple is False

    def test_predict_command_options(self, parser):
        """Test predict command with options."""
        args = parser.parse_args([
            "predict",
            "--data", "test.csv",
            "--symbol", "AUDUSD",
            "--weights", "custom/weights",
            "--simple"
        ])
        
        assert args.data == "test.csv"
        assert args.symbol == "AUDUSD"
        assert args.weights == "custom/weights"
        assert args.simple is True

    # --- Generate command ---
    def test_generate_command_dataset_required(self, parser):
        """Test generate command requires dataset argument."""
        with pytest.raises(SystemExit):
            parser.parse_args(["generate"])

    @pytest.mark.parametrize("dataset", ["yolo", "vit", "both"])
    def test_generate_command_valid_datasets(self, parser, dataset):
        """Test generate command accepts valid dataset types."""
        args = parser.parse_args(["generate", dataset])
        
        assert args.command == "generate"
        assert args.dataset == dataset

    def test_generate_command_options(self, parser):
        """Test generate command with options."""
        args = parser.parse_args([
            "generate", "yolo",
            "--samples", "10000",
            "--output", "output/dir",
            "--synthetic",
            "--image-size", "256",
            "--window", "100",
            "--stride", "10"
        ])
        
        assert args.samples == 10000
        assert args.output == "output/dir"
        assert args.synthetic is True
        assert args.image_size == 256
        assert args.window == 100
        assert args.stride == 10

    # --- Status command ---
    def test_status_command(self, parser):
        """Test status command."""
        args = parser.parse_args(["status"])
        
        assert args.command == "status"

    def test_status_command_verbose(self, parser):
        """Test status command with verbose."""
        args = parser.parse_args(["status", "-v"])
        
        assert args.verbose is True

    # --- Global options ---
    def test_global_verbose_flag(self, parser):
        """Test global verbose flag - note: status subparser has its own -v."""
        # The status subparser has its own -v flag, so we test with a command that doesn't
        args = parser.parse_args(["-v", "multi"])
        
        assert args.verbose is True

    def test_global_log_file(self, parser):
        """Test global log-file option."""
        args = parser.parse_args(["--log-file", "test.log", "status"])
        
        assert args.log_file == "test.log"

    def test_global_dry_run(self, parser):
        """Test global dry-run flag."""
        args = parser.parse_args(["--dry-run", "multi"])
        
        assert args.dry_run is True


# ============================================================================
# TEST: Command Handlers
# ============================================================================

class TestCommandHandlers:
    """Tests for command handler functions."""

    # --- cmd_status ---
    def test_cmd_status_returns_zero(self, logger):
        """Test cmd_status returns 0."""
        args = argparse.Namespace(verbose=False)
        
        with patch.object(SystemChecker, 'print_status'):
            result = cmd_status(args, logger)
        
        assert result == 0

    def test_cmd_status_calls_print_status(self, logger):
        """Test cmd_status calls print_status with verbose flag."""
        args = argparse.Namespace(verbose=True)
        
        with patch.object(SystemChecker, 'print_status') as mock_print:
            cmd_status(args, logger)
        
        mock_print.assert_called_once_with(verbose=True)

    # --- cmd_multi_style ---
    def test_cmd_multi_style_dry_run(self, logger):
        """Test cmd_multi_style dry run mode."""
        args = argparse.Namespace(
            dry_run=True,
            symbol="EURUSD",
            no_scalp=False,
            no_intraday=False,
            no_swing=False,
            mock=False,
        )
        
        with patch.object(SystemChecker, 'run_all_checks', return_value=True):
            result = cmd_multi_style(args, logger)
        
        assert result == 0

    def test_cmd_multi_style_checks_fail(self, logger):
        """Test cmd_multi_style when checks fail."""
        args = argparse.Namespace(
            dry_run=False,
            symbol="EURUSD",
            no_scalp=False,
            no_intraday=False,
            no_swing=False,
            mock=False,
        )
        
        with patch.object(SystemChecker, 'run_all_checks', return_value=False):
            result = cmd_multi_style(args, logger)
        
        assert result == 1

    # --- cmd_live ---
    def test_cmd_live_dry_run(self, logger):
        """Test cmd_live dry run mode."""
        args = argparse.Namespace(
            dry_run=True,
            symbol="EURUSD",
            timeframe="H1",
            strategy="neural",
            interval=10.0,
            mock=False,
        )
        
        with patch.object(SystemChecker, 'run_all_checks', return_value=True):
            result = cmd_live(args, logger)
        
        assert result == 0

    def test_cmd_live_checks_fail(self, logger):
        """Test cmd_live when checks fail."""
        args = argparse.Namespace(
            dry_run=False,
            symbol="EURUSD",
            timeframe="H1",
            strategy="neural",
            interval=10.0,
            mock=False,
        )
        
        with patch.object(SystemChecker, 'run_all_checks', return_value=False):
            result = cmd_live(args, logger)
        
        assert result == 1

    # --- cmd_backtest ---
    def test_cmd_backtest_dry_run(self, logger, mock_csv_data):
        """Test cmd_backtest dry run mode."""
        args = argparse.Namespace(
            dry_run=True,
            data=str(mock_csv_data),
            strategy="neural",
            balance=10000.0,
            output=None,
        )
        
        with patch.object(SystemChecker, 'check_data', return_value=True):
            result = cmd_backtest(args, logger)
        
        assert result == 0

    def test_cmd_backtest_data_missing(self, logger):
        """Test cmd_backtest when data file is missing."""
        args = argparse.Namespace(
            dry_run=False,
            data="/nonexistent/data.csv",
            strategy="neural",
            balance=10000.0,
            output=None,
        )
        
        result = cmd_backtest(args, logger)
        
        assert result == 1

    # --- cmd_train ---
    def test_cmd_train_dry_run(self, logger):
        """Test cmd_train dry run mode (TCN instead of LSTM)."""
        args = argparse.Namespace(
            dry_run=True,
            model="tcn",  # Changed from 'lstm' to 'tcn'
            data="data.csv",
            data_dir=None,
            save_dir=None,
            epochs=50,
            batch_size=64,
            lr=0.001,
            seq_len=60,
            cache_path=None,
            synthetic=False,
        )

        with patch.object(SystemChecker, 'run_all_checks', return_value=True):
            result = cmd_train(args, logger)

        assert result == 0

    @pytest.mark.parametrize("model", ["tcn", "vit", "fusion", "trend"])
    def test_cmd_train_dry_run_all_models(self, logger, model):
        """Test cmd_train dry run for all models (TCN instead of LSTM)."""
        args = argparse.Namespace(
            dry_run=True,
            model=model,
            data="data.csv",
            data_dir="data/vit",
            save_dir=None,
            epochs=50,
            batch_size=64,
            lr=0.001,
            seq_len=60,
            cache_path=None,
            synthetic=False,
        )

        with patch.object(SystemChecker, 'run_all_checks', return_value=True):
            result = cmd_train(args, logger)

        assert result == 0

    # --- cmd_generate ---
    def test_cmd_generate_dry_run(self, logger):
        """Test cmd_generate dry run mode."""
        args = argparse.Namespace(
            dry_run=True,
            dataset="yolo",
            data=None,
            output="output/",
            samples=5000,
            synthetic=True,
            image_size=224,
            window=60,
            stride=5,
        )
        
        result = cmd_generate(args, logger)
        
        assert result == 0

    @pytest.mark.parametrize("dataset", ["yolo", "vit"])
    def test_cmd_generate_dry_run_all_datasets(self, logger, dataset):
        """Test cmd_generate dry run for all dataset types."""
        args = argparse.Namespace(
            dry_run=True,
            dataset=dataset,
            data=None,
            output="output/",
            samples=5000,
            synthetic=True,
            image_size=224,
            window=60,
            stride=5,
        )
        
        result = cmd_generate(args, logger)
        
        assert result == 0


# ============================================================================
# TEST: Main Function
# ============================================================================

class TestMainFunction:
    """Tests for main entry point function."""

    def test_main_no_args_shows_help(self):
        """Test main with no arguments shows help."""
        with patch('sys.argv', ['pyforex']):
            result = main()
        
        assert result == 0

    def test_main_status_command(self):
        """Test main dispatches status command."""
        with patch('sys.argv', ['pyforex', 'status']):
            with patch('main.cmd_status', return_value=0) as mock_cmd:
                result = main()
        
        mock_cmd.assert_called_once()
        assert result == 0

    def test_main_multi_command(self):
        """Test main dispatches multi command."""
        with patch('sys.argv', ['pyforex', '--dry-run', 'multi']):
            with patch('main.cmd_multi_style', return_value=0) as mock_cmd:
                result = main()
        
        mock_cmd.assert_called_once()
        assert result == 0

    def test_main_live_command(self):
        """Test main dispatches live command."""
        with patch('sys.argv', ['pyforex', '--dry-run', 'live']):
            with patch('main.cmd_live', return_value=0) as mock_cmd:
                result = main()
        
        mock_cmd.assert_called_once()
        assert result == 0

    def test_main_backtest_command(self):
        """Test main dispatches backtest command."""
        with patch('sys.argv', ['pyforex', '--dry-run', 'backtest', '--data', 'test.csv']):
            with patch('main.cmd_backtest', return_value=0) as mock_cmd:
                result = main()
        
        mock_cmd.assert_called_once()
        assert result == 0

    def test_main_train_command(self):
        """Test main dispatches train command (TCN instead of LSTM)."""
        with patch('sys.argv', ['pyforex', '--dry-run', 'train', 'tcn']):
            with patch('main.cmd_train', return_value=0) as mock_cmd:
                result = main()

        mock_cmd.assert_called_once()
        assert result == 0

    def test_main_predict_command(self):
        """Test main dispatches predict command."""
        with patch('sys.argv', ['pyforex', 'predict']):
            with patch('main.cmd_predict', return_value=0) as mock_cmd:
                result = main()
        
        mock_cmd.assert_called_once()
        assert result == 0

    def test_main_generate_command(self):
        """Test main dispatches generate command."""
        with patch('sys.argv', ['pyforex', '--dry-run', 'generate', 'yolo']):
            with patch('main.cmd_generate', return_value=0) as mock_cmd:
                result = main()
        
        mock_cmd.assert_called_once()
        assert result == 0

    def test_main_verbose_flag(self):
        """Test main with verbose flag."""
        with patch('sys.argv', ['pyforex', '-v', 'status']):
            with patch('main.cmd_status', return_value=0):
                result = main()
        
        assert result == 0

    def test_main_live_creates_log_file(self):
        """Test main creates log file for live command."""
        with patch('sys.argv', ['pyforex', '--dry-run', 'live']):
            with patch('main.cmd_live', return_value=0):
                with patch('main.setup_logging') as mock_setup:
                    main()
        
        # Verify log_file argument was passed (or at least setup_logging was called)
        mock_setup.assert_called_once()


# ============================================================================
# TEST: Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_cmd_multi_style_keyboard_interrupt(self, logger):
        """Test cmd_multi_style handles KeyboardInterrupt gracefully."""
        args = argparse.Namespace(
            dry_run=False,
            symbol="EURUSD",
            no_scalp=False,
            no_intraday=False,
            no_swing=False,
            mock=False,
        )
        
        # Create a mock module that raises KeyboardInterrupt
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_multi_style_bot = MagicMock(side_effect=KeyboardInterrupt)
        
        with patch.object(SystemChecker, 'run_all_checks', return_value=True):
            with patch.dict('sys.modules', {
                'trading': MagicMock(),
                'trading.multi_style_orchestrator': mock_orchestrator
            }):
                result = cmd_multi_style(args, logger)
        
        # Should return 0 on KeyboardInterrupt (graceful exit)
        assert result == 0

    def test_cmd_backtest_exception_handling(self, logger, mock_csv_data):
        """Test cmd_backtest handles exceptions."""
        args = argparse.Namespace(
            dry_run=False,
            data=str(mock_csv_data),
            strategy="neural",
            balance=10000.0,
            output=None,
        )
        
        # This will fail when trying to import BacktestBot
        result = cmd_backtest(args, logger)
        
        # Should return 1 on error
        assert result == 1

    def test_cmd_train_unknown_model(self, logger):
        """Test cmd_train handles unknown model gracefully."""
        args = argparse.Namespace(
            dry_run=False,
            model="unknown_model",
            data="data.csv",
            data_dir=None,
            save_dir=None,
            epochs=50,
            batch_size=64,
            lr=0.001,
            seq_len=60,
            cache_path=None,
            synthetic=False,
        )

        with patch.object(SystemChecker, 'run_all_checks', return_value=True):
            result = cmd_train(args, logger)

        assert result == 1

    def test_system_checker_exception_in_mt5(self, system_checker):
        """Test SystemChecker handles exceptions in check_mt5."""
        with patch('builtins.__import__', side_effect=Exception("Unexpected error")):
            result = system_checker.check_mt5()
        
        assert result is False
        assert len(system_checker.warnings) > 0

    def test_system_checker_torch_not_installed(self, system_checker):
        """Test check_cuda when torch is not installed."""
        # Save original modules
        original_torch = sys.modules.get('torch')
        
        try:
            # Remove torch from modules temporarily
            if 'torch' in sys.modules:
                del sys.modules['torch']
            
            # Mock the import to fail
            original_import = __builtins__['__import__'] if isinstance(__builtins__, dict) else __builtins__.__import__
            
            def mock_import(name, *args, **kwargs):
                if name == 'torch':
                    raise ImportError("No module named 'torch'")
                return original_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                result = system_checker.check_cuda()
            
            assert result["cuda_available"] is False
            assert "error" in result
        finally:
            # Restore torch if it was present
            if original_torch is not None:
                sys.modules['torch'] = original_torch


# ============================================================================
# TEST: Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for CLI workflow."""

    def test_full_status_check_workflow(self):
        """Test complete status check workflow."""
        with patch('sys.argv', ['pyforex', 'status', '-v']):
            with patch.object(SystemChecker, 'print_status') as mock_print:
                result = main()
        
        assert result == 0
        mock_print.assert_called_once_with(verbose=True)

    def test_full_dry_run_workflow(self):
        """Test complete dry-run workflow for multi command."""
        with patch('sys.argv', ['pyforex', '--dry-run', 'multi', '--symbol', 'GBPUSD', '--no-scalp']):
            with patch.object(SystemChecker, 'run_all_checks', return_value=True):
                result = main()
        
        assert result == 0

    def test_argument_flow_to_handler(self):
        """Test arguments flow correctly from parser to handler."""
        with patch('sys.argv', ['pyforex', 'multi', '--symbol', 'USDJPY', '--mock', '--no-swing']):
            with patch('main.cmd_multi_style') as mock_handler:
                mock_handler.return_value = 0
                main()
        
        # Verify arguments passed to handler
        call_args = mock_handler.call_args[0][0]  # First positional arg
        assert call_args.symbol == "USDJPY"
        assert call_args.mock is True
        assert call_args.no_swing is True


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])