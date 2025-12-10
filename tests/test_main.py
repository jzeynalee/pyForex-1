#!/usr/bin/env python3
"""
test_main.py - Comprehensive tests for main.py CLI entry point

Run with:
    pytest test_main.py -v
    pytest test_main.py -v --cov=main --cov-report=html
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import argparse
import sys
import os
import logging
from pathlib import Path
import tempfile
import json
from datetime import datetime

# Import the module to test
import main


class TestAppConfig(unittest.TestCase):
    """Test AppConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = main.AppConfig()
        
        self.assertEqual(config.default_symbol, "EURUSD")
        self.assertEqual(config.default_timeframe, "H1")
        self.assertEqual(config.default_sequence_length, 60)
        self.assertEqual(config.default_image_size, 224)
        self.assertEqual(config.log_level, "INFO")
        self.assertTrue(config.log_to_file)
        
        # Verify paths are Path objects
        self.assertIsInstance(config.project_root, Path)
        self.assertIsInstance(config.weights_dir, Path)
        self.assertIsInstance(config.data_dir, Path)
        self.assertIsInstance(config.logs_dir, Path)


class TestSystemChecker(unittest.TestCase):
    """Test SystemChecker class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger("test")
        self.checker = main.SystemChecker(self.logger)
    
    def test_initial_state(self):
        """Test initial state of SystemChecker."""
        self.assertEqual(self.checker.issues, [])
        self.assertEqual(self.checker.warnings, [])
        self.assertEqual(self.checker.logger, self.logger)
    
    @patch('main.CONFIG')
    @patch('pathlib.Path.exists')
    def test_check_weights_success(self, mock_exists, mock_config):
        """Test weight checking when weights exist."""
        mock_config.weights_dir = Path("/fake/weights")
        mock_exists.return_value = True
        
        result = self.checker.check_weights(["model1.pt", "model2.pt"])
        
        self.assertTrue(result)
        self.assertEqual(self.checker.issues, [])
        self.assertEqual(self.checker.warnings, [])
    
    @patch('main.CONFIG')
    @patch('pathlib.Path.exists')
    def test_check_weights_missing_dir(self, mock_exists, mock_config):
        """Test weight checking when weights directory doesn't exist."""
        mock_config.weights_dir = Path("/fake/weights")
        mock_exists.return_value = False
        
        result = self.checker.check_weights(["model1.pt"])
        
        self.assertFalse(result)
        self.assertEqual(len(self.checker.issues), 1)
        self.assertIn("Weights directory not found", self.checker.issues[0])
    
    @patch('main.CONFIG')
    @patch('pathlib.Path.exists')
    def test_check_weights_missing_files(self, mock_exists, mock_config):
        """Test weight checking when weight files are missing."""
        mock_config.weights_dir = Path("/fake/weights")
        # Directory exists but files don't
        def side_effect(path):
            return str(path) == "/fake/weights"
        mock_exists.side_effect = side_effect
        
        result = self.checker.check_weights(["model1.pt", "model2.pt"])
        
        self.assertFalse(result)
        self.assertEqual(len(self.checker.warnings), 1)
        self.assertIn("Missing weight files", self.checker.warnings[0])
    
    @patch('importlib.import_module')
    def test_check_mt5_success(self, mock_import):
        """Test MT5 check when MT5 is available."""
        mock_mt5 = MagicMock()
        mock_mt5.initialize.return_value = True
        mock_mt5.shutdown.return_value = None
        mock_import.return_value = mock_mt5
        
        result = self.checker.check_mt5()
        
        self.assertTrue(result)
        mock_mt5.initialize.assert_called_once()
        mock_mt5.shutdown.assert_called_once()
    
    @patch('importlib.import_module')
    def test_check_mt5_not_initialized(self, mock_import):
        """Test MT5 check when MT5 fails to initialize."""
        mock_mt5 = MagicMock()
        mock_mt5.initialize.return_value = False
        mock_import.return_value = mock_mt5
        
        result = self.checker.check_mt5()
        
        self.assertFalse(result)
        self.assertIn("MT5 not initialized", self.checker.warnings[0])
    
    @patch('importlib.import_module')
    def test_check_mt5_not_installed(self, mock_import):
        """Test MT5 check when MT5 library is not installed."""
        mock_import.side_effect = ImportError("No module named 'MetaTrader5'")
        
        result = self.checker.check_mt5()
        
        self.assertFalse(result)
        self.assertIn("MetaTrader5 library not installed", self.checker.warnings[0])
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_name')
    @patch('torch.__version__', '2.0.0')
    def test_check_cuda_available(self, mock_get_device_name, mock_device_count, mock_is_available):
        """Test CUDA check when GPU is available."""
        mock_is_available.return_value = True
        mock_device_count.return_value = 2
        mock_get_device_name.return_value = "NVIDIA RTX 4090"
        
        # Mock torch import
        with patch.dict('sys.modules', {'torch': MagicMock()}):
            result = self.checker.check_cuda()
        
        self.assertTrue(result["cuda_available"])
        self.assertEqual(result["device_count"], 2)
        self.assertEqual(result["device_name"], "NVIDIA RTX 4090")
        self.assertEqual(result["torch_version"], "2.0.0")
    
    def test_check_cuda_no_torch(self):
        """Test CUDA check when PyTorch is not installed."""
        # Remove torch from sys.modules if it exists
        original_torch = sys.modules.get('torch')
        if 'torch' in sys.modules:
            del sys.modules['torch']
        
        result = self.checker.check_cuda()
        
        self.assertFalse(result["cuda_available"])
        self.assertIn("error", result)
        
        # Restore torch if it was there
        if original_torch:
            sys.modules['torch'] = original_torch
    
    @patch('builtins.__import__')
    def test_check_dependencies_success(self, mock_import):
        """Test dependency check when all dependencies are available."""
        result = self.checker.check_dependencies()
        
        self.assertTrue(result)
        self.assertEqual(self.checker.issues, [])
        
        # Verify required modules were attempted to be imported
        required_calls = [
            call('torch'),
            call('numpy'),
            call('pandas'),
            call('sklearn'),
        ]
        
        # Check that at least the required modules were attempted
        for req_call in required_calls:
            self.assertIn(req_call, mock_import.call_args_list)
    
    @patch('builtins.__import__')
    def test_check_dependencies_missing_required(self, mock_import):
        """Test dependency check when required dependencies are missing."""
        def side_effect(name, *args, **kwargs):
            if name == 'torch':
                raise ImportError("No module named 'torch'")
            return MagicMock()
        mock_import.side_effect = side_effect
        
        result = self.checker.check_dependencies()
        
        self.assertFalse(result)
        self.assertIn("Missing required", self.checker.issues[0])
    
    @patch('builtins.__import__')
    def test_check_dependencies_missing_optional(self, mock_import):
        """Test dependency check when optional dependencies are missing."""
        def side_effect(name, *args, **kwargs):
            if name == 'timm':
                raise ImportError("No module named 'timm'")
            return MagicMock()
        mock_import.side_effect = side_effect
        
        result = self.checker.check_dependencies()
        
        self.assertTrue(result)  # Should still pass with optional missing
        self.assertIn("Missing optional", self.checker.warnings[0])
    
    @patch('pathlib.Path.exists')
    def test_check_data_exists(self, mock_exists):
        """Test data check when file exists."""
        mock_exists.return_value = True
        
        result = self.checker.check_data("/path/to/data.csv")
        
        self.assertTrue(result)
        mock_exists.assert_called_once()
    
    @patch('pathlib.Path.exists')
    def test_check_data_missing(self, mock_exists):
        """Test data check when file doesn't exist."""
        mock_exists.return_value = False
        
        result = self.checker.check_data("/path/to/data.csv")
        
        self.assertFalse(result)
        self.assertIn("Data file not found", self.checker.issues[0])
    
    def test_check_data_no_path(self):
        """Test data check when no path is provided."""
        result = self.checker.check_data(None)
        
        self.assertTrue(result)
        self.assertEqual(self.checker.issues, [])
    
    @patch.object(main.SystemChecker, 'check_dependencies')
    @patch.object(main.SystemChecker, 'check_mt5')
    @patch.object(main.SystemChecker, 'check_weights')
    @patch.object(main.SystemChecker, 'check_cuda')
    def test_run_all_checks_live(self, mock_cuda, mock_weights, mock_mt5, mock_deps):
        """Test running all checks for live trading mode."""
        mock_deps.return_value = True
        mock_mt5.return_value = True
        mock_weights.return_value = True
        mock_cuda.return_value = {"cuda_available": True}
        
        result = self.checker.run_all_checks("live")
        
        self.assertTrue(result)
        mock_deps.assert_called_once()
        mock_mt5.assert_called_once()
        mock_weights.assert_called_with(["lstm_best.pt", "fusion_best.pt"])
        mock_cuda.assert_not_called()  # Not called for live mode
    
    @patch.object(main.SystemChecker, 'check_dependencies')
    @patch.object(main.SystemChecker, 'check_mt5')
    @patch.object(main.SystemChecker, 'check_weights')
    def test_run_all_checks_live_failure(self, mock_weights, mock_mt5, mock_deps):
        """Test running all checks for live trading mode with failure."""
        mock_deps.return_value = True
        mock_mt5.return_value = False  # MT5 check fails
        mock_weights.return_value = True
        
        result = self.checker.run_all_checks("live")
        
        # Should still pass with warnings
        self.assertTrue(result)
        self.assertGreater(len(self.checker.warnings), 0)
    
    @patch.object(main.SystemChecker, 'check_dependencies')
    @patch.object(main.SystemChecker, 'check_cuda')
    def test_run_all_checks_train(self, mock_cuda, mock_deps):
        """Test running all checks for training mode."""
        mock_deps.return_value = True
        mock_cuda.return_value = {"cuda_available": False}
        
        result = self.checker.run_all_checks("train")
        
        self.assertTrue(result)
        mock_deps.assert_called_once()
        mock_cuda.assert_called_once()
        self.assertIn("CUDA not available", self.checker.warnings[0])


class TestCommandHandlers(unittest.TestCase):
    """Test command handler functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = MagicMock(spec=logging.Logger)
        self.mock_args = MagicMock(spec=argparse.Namespace)
        
        # Common mock args
        self.mock_args.dry_run = False
        self.mock_args.symbol = "EURUSD"
        self.mock_args.timeframe = "H1"
    
    @patch('main.SystemChecker')
    @patch('main.run_multi_style_bot')
    def test_cmd_multi_style_success(self, mock_run_bot, mock_checker_class):
        """Test multi-style trading command success."""
        mock_checker = MagicMock()
        mock_checker.run_all_checks.return_value = True
        mock_checker_class.return_value = mock_checker
        
        self.mock_args.no_scalp = False
        self.mock_args.no_intraday = False
        self.mock_args.no_swing = False
        self.mock_args.mock = False
        
        result = main.cmd_multi_style(self.mock_args, self.logger)
        
        self.assertEqual(result, 0)
        mock_checker.run_all_checks.assert_called_with("live")
        mock_run_bot.assert_called_once_with(
            symbol="EURUSD",
            mock=False,
            enable_scalp=True,
            enable_intraday=True,
            enable_swing=True,
        )
    
    @patch('main.SystemChecker')
    def test_cmd_multi_style_check_failed(self, mock_checker_class):
        """Test multi-style trading command when checks fail."""
        mock_checker = MagicMock()
        mock_checker.run_all_checks.return_value = False
        mock_checker_class.return_value = mock_checker
        
        result = main.cmd_multi_style(self.mock_args, self.logger)
        
        self.assertEqual(result, 1)
        self.logger.error.assert_called_with("Prerequisites not met. Aborting.")
    
    @patch('main.SystemChecker')
    def test_cmd_multi_style_dry_run(self, mock_checker_class):
        """Test multi-style trading command dry run."""
        mock_checker = MagicMock()
        mock_checker.run_all_checks.return_value = True
        mock_checker_class.return_value = mock_checker
        
        self.mock_args.dry_run = True
        self.mock_args.no_scalp = True
        self.mock_args.no_intraday = False
        self.mock_args.no_swing = True
        
        result = main.cmd_multi_style(self.mock_args, self.logger)
        
        self.assertEqual(result, 0)
        mock_checker.run_all_checks.assert_called_with("live")
        self.logger.info.assert_any_call("DRY RUN - would start multi-style trading with:")
    
    @patch('main.SystemChecker')
    @patch('main.TradingBot')
    @patch('main.NeuralHybridStrategy')
    def test_cmd_live_success(self, mock_strategy, mock_bot_class, mock_checker_class):
        """Test live trading command success."""
        mock_checker = MagicMock()
        mock_checker.run_all_checks.return_value = True
        mock_checker_class.return_value = mock_checker
        
        mock_bot = MagicMock()
        mock_bot_class.return_value = mock_bot
        
        self.mock_args.strategy = "neural"
        self.mock_args.interval = 10.0
        self.mock_args.mock = False
        
        result = main.cmd_live(self.mock_args, self.logger)
        
        self.assertEqual(result, 0)
        mock_checker.run_all_checks.assert_called_with("live")
        mock_bot_class.assert_called_once()
        mock_bot.run.assert_called_once()
    
    @patch('main.SystemChecker')
    @patch('main.pd.read_csv')
    @patch('main.BacktestBot')
    @patch('main.NeuralHybridStrategy')
    def test_cmd_backtest_success(self, mock_strategy, mock_bot_class, mock_read_csv, mock_checker_class):
        """Test backtest command success."""
        mock_checker = MagicMock()
        mock_checker.check_data.return_value = True
        mock_checker_class.return_value = mock_checker
        
        # Mock dataframe
        mock_df = MagicMock()
        mock_df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        mock_df.__len__.return_value = 1000
        mock_df.__getitem__.return_value = MagicMock()
        mock_df['time'].iloc = [0, -1]
        mock_read_csv.return_value = mock_df
        
        # Mock bot
        mock_bot = MagicMock()
        mock_bot.run.return_value = {
            'final_balance': 11000.0,
            'trades': [
                {'time': '2024-01-01', 'type': 'BUY', 'pnl': 100},
                {'time': '2024-01-02', 'type': 'SELL', 'pnl': -50},
            ]
        }
        mock_bot_class.return_value = mock_bot
        
        # Set up args
        self.mock_args.data = "data/EURUSD_H1.csv"
        self.mock_args.strategy = "neural"
        self.mock_args.balance = 10000.0
        self.mock_args.output = None
        
        result = main.cmd_backtest(self.mock_args, self.logger)
        
        self.assertEqual(result, 0)
        mock_checker.check_data.assert_called_with("data/EURUSD_H1.csv")
        mock_read_csv.assert_called_with("data/EURUSD_H1.csv")
        mock_bot_class.assert_called_once()
        mock_bot.run.assert_called_once()
    
    @patch('main.SystemChecker')
    @patch('main.train_lstm_model')
    def test_cmd_train_lstm(self, mock_train, mock_checker_class):
        """Test training command for LSTM."""
        mock_checker = MagicMock()
        mock_checker.run_all_checks.return_value = True
        mock_checker_class.return_value = mock_checker
        
        self.mock_args.model = "lstm"
        self.mock_args.data = "data/raw.csv"
        self.mock_args.save_dir = "models/weights"
        self.mock_args.epochs = 10
        self.mock_args.batch_size = 32
        self.mock_args.lr = 0.001
        self.mock_args.seq_len = 60
        self.mock_args.attention = True
        
        result = main.cmd_train(self.mock_args, self.logger)
        
        self.assertEqual(result, 0)
        mock_train.assert_called_once_with(
            data_path="data/raw.csv",
            save_dir="models/weights",
            epochs=10,
            batch_size=32,
            learning_rate=0.001,
            seq_len=60,
            use_attention=True,
        )
    
    @patch('main.SystemChecker')
    @patch('main.train_lstm_model')
    def test_cmd_train_unknown_model(self, mock_train, mock_checker_class):
        """Test training command with unknown model."""
        mock_checker = MagicMock()
        mock_checker.run_all_checks.return_value = True
        mock_checker_class.return_value = mock_checker
        
        self.mock_args.model = "unknown"
        
        result = main.cmd_train(self.mock_args, self.logger)
        
        self.assertEqual(result, 1)
        self.logger.error.assert_called_with("Unknown model: unknown")
    
    @patch('main.SystemChecker')
    @patch('main.pd.read_csv')
    @patch('main.HybridPredictor')
    def test_cmd_predict_with_csv(self, mock_predictor_class, mock_read_csv, mock_checker_class):
        """Test predict command with CSV data."""
        mock_checker = MagicMock()
        mock_checker.run_all_checks.return_value = True
        mock_checker_class.return_value = mock_checker
        
        # Mock dataframe
        mock_df = MagicMock()
        mock_df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        mock_df.__len__.return_value = 100
        mock_df['close'].iloc = [-1]
        mock_read_csv.return_value = mock_df
        
        # Mock predictor
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = MagicMock(
            predicted_class=0,
            confidence=0.85,
            probabilities=[0.7, 0.2, 0.1],
            gate_weights=[0.5, 0.3, 0.2]
        )
        mock_predictor_class.return_value = mock_predictor
        
        self.mock_args.data = "data/predict.csv"
        self.mock_args.symbol = "EURUSD"
        self.mock_args.weights = None
        self.mock_args.simple = False
        
        result = main.cmd_predict(self.mock_args, self.logger)
        
        self.assertEqual(result, 0)
        mock_read_csv.assert_called_with("data/predict.csv")
        mock_predictor_class.assert_called_once()
        mock_predictor.predict.assert_called_once_with(mock_df)
    
    @patch('main.SystemChecker')
    @patch('main.YOLODatasetGenerator')
    def test_cmd_generate_yolo(self, mock_generator_class, mock_checker_class):
        """Test dataset generation for YOLO."""
        mock_checker = MagicMock()
        mock_checker_class.return_value = mock_checker
        
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        
        self.mock_args.dataset = "yolo"
        self.mock_args.data = None
        self.mock_args.output = "datasets/yolo"
        self.mock_args.samples = 5000
        self.mock_args.synthetic = True
        self.mock_args.image_size = 256
        self.mock_args.window = 60
        self.mock_args.stride = 10
        
        result = main.cmd_generate(self.mock_args, self.logger)
        
        self.assertEqual(result, 0)
        mock_generator_class.assert_called_once_with(
            output_dir="datasets/yolo",
            image_size=256,
            window_size=60,
            stride=10,
        )
        mock_generator.generate_synthetic.assert_called_once_with(n_samples=5000)
    
    @patch('main.SystemChecker')
    def test_cmd_status(self, mock_checker_class):
        """Test status command."""
        mock_checker = MagicMock()
        mock_checker_class.return_value = mock_checker
        
        self.mock_args.verbose = False
        
        result = main.cmd_status(self.mock_args, self.logger)
        
        self.assertEqual(result, 0)
        mock_checker.print_status.assert_called_once_with(verbose=False)


class TestArgumentParser(unittest.TestCase):
    """Test argument parser creation."""
    
    def test_parser_creation(self):
        """Test that the argument parser is created correctly."""
        parser = main.create_parser()
        
        self.assertIsInstance(parser, argparse.ArgumentParser)
        self.assertEqual(parser.prog, "pyforex")
        self.assertIn("Multi-Modal Forex Trading System", parser.description)
    
    def test_global_arguments(self):
        """Test global arguments."""
        parser = main.create_parser()
        
        # Test verbose flag
        args = parser.parse_args(["--verbose", "status"])
        self.assertTrue(args.verbose)
        
        # Test dry-run flag
        args = parser.parse_args(["--dry-run", "live"])
        self.assertTrue(args.dry_run)
        
        # Test log-file
        args = parser.parse_args(["--log-file", "test.log", "status"])
        self.assertEqual(args.log_file, "test.log")
    
    def test_multi_command(self):
        """Test multi-style trading command arguments."""
        parser = main.create_parser()
        
        args = parser.parse_args(["multi", "--symbol", "GBPUSD", "--no-scalp", "--mock"])
        
        self.assertEqual(args.command, "multi")
        self.assertEqual(args.symbol, "GBPUSD")
        self.assertTrue(args.no_scalp)
        self.assertTrue(args.mock)
        self.assertFalse(args.no_intraday)
        self.assertFalse(args.no_swing)
    
    def test_live_command(self):
        """Test live trading command arguments."""
        parser = main.create_parser()
        
        args = parser.parse_args([
            "live", 
            "--symbol", "USDJPY", 
            "--timeframe", "M5", 
            "--strategy", "lstm",
            "--interval", "5.0",
            "--mock"
        ])
        
        self.assertEqual(args.command, "live")
        self.assertEqual(args.symbol, "USDJPY")
        self.assertEqual(args.timeframe, "M5")
        self.assertEqual(args.strategy, "lstm")
        self.assertEqual(args.interval, 5.0)
        self.assertTrue(args.mock)
    
    def test_backtest_command(self):
        """Test backtest command arguments."""
        parser = main.create_parser()
        
        args = parser.parse_args([
            "backtest",
            "--data", "data/test.csv",
            "--strategy", "neural",
            "--balance", "5000.0",
            "--output", "results.json"
        ])
        
        self.assertEqual(args.command, "backtest")
        self.assertEqual(args.data, "data/test.csv")
        self.assertEqual(args.strategy, "neural")
        self.assertEqual(args.balance, 5000.0)
        self.assertEqual(args.output, "results.json")
    
    def test_train_command(self):
        """Test train command arguments."""
        parser = main.create_parser()
        
        args = parser.parse_args([
            "train", "lstm",
            "--data", "data/train.csv",
            "--save-dir", "checkpoints",
            "--epochs", "100",
            "--batch-size", "128",
            "--lr", "0.0001",
            "--seq-len", "30",
            "--attention"
        ])
        
        self.assertEqual(args.command, "train")
        self.assertEqual(args.model, "lstm")
        self.assertEqual(args.data, "data/train.csv")
        self.assertEqual(args.save_dir, "checkpoints")
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.batch_size, 128)
        self.assertEqual(args.lr, 0.0001)
        self.assertEqual(args.seq_len, 30)
        self.assertTrue(args.attention)
    
    def test_predict_command(self):
        """Test predict command arguments."""
        parser = main.create_parser()
        
        args = parser.parse_args([
            "predict",
            "--data", "data/predict.csv",
            "--symbol", "AUDUSD",
            "--weights", "custom_weights",
            "--simple"
        ])
        
        self.assertEqual(args.command, "predict")
        self.assertEqual(args.data, "data/predict.csv")
        self.assertEqual(args.symbol, "AUDUSD")
        self.assertEqual(args.weights, "custom_weights")
        self.assertTrue(args.simple)
    
    def test_generate_command(self):
        """Test generate command arguments."""
        parser = main.create_parser()
        
        args = parser.parse_args([
            "generate", "vit",
            "--data", "data/source.csv",
            "--output", "datasets/vit_custom",
            "--samples", "10000",
            "--synthetic",
            "--image-size", "224",
            "--window", "100",
            "--stride", "2"
        ])
        
        self.assertEqual(args.command, "generate")
        self.assertEqual(args.dataset, "vit")
        self.assertEqual(args.data, "data/source.csv")
        self.assertEqual(args.output, "datasets/vit_custom")
        self.assertEqual(args.samples, 10000)
        self.assertTrue(args.synthetic)
        self.assertEqual(args.image_size, 224)
        self.assertEqual(args.window, 100)
        self.assertEqual(args.stride, 2)
    
    def test_status_command(self):
        """Test status command arguments."""
        parser = main.create_parser()
        
        args = parser.parse_args(["status", "--verbose"])
        
        self.assertEqual(args.command, "status")
        self.assertTrue(args.verbose)


class TestMainFunction(unittest.TestCase):
    """Test main function and entry point."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.original_argv = sys.argv
    
    def tearDown(self):
        """Clean up test fixtures."""
        sys.argv = self.original_argv
    
    @patch('main.setup_logging')
    @patch('main.cmd_status')
    def test_main_with_status_command(self, mock_cmd_status, mock_setup_logging):
        """Test main function with status command."""
        sys.argv = ["main.py", "status", "--verbose"]
        
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        mock_cmd_status.return_value = 0
        
        result = main.main()
        
        self.assertEqual(result, 0)
        mock_setup_logging.assert_called_once()
        mock_cmd_status.assert_called_once()
    
    @patch('main.setup_logging')
    def test_main_no_command(self, mock_setup_logging):
        """Test main function with no command (should print help)."""
        sys.argv = ["main.py"]
        
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        
        # Mock parser.print_help to avoid actual print
        with patch('argparse.ArgumentParser.print_help') as mock_print_help:
            result = main.main()
        
        self.assertEqual(result, 0)
        mock_print_help.assert_called_once()
    
    @patch('main.setup_logging')
    @patch('main.cmd_multi_style')
    def test_main_unknown_command(self, mock_cmd_multi, mock_setup_logging):
        """Test main function with unknown command."""
        # Create a parser that will have unknown command
        # Actually, we need to simulate what happens when command is not in command_handlers
        # But our create_parser will reject unknown commands before reaching that point
        
        # Instead, test the if handler: else: branch by mocking command_handlers
        with patch.dict('main.command_handlers', {}, clear=True):
            sys.argv = ["main.py", "unknown"]
            
            mock_logger = MagicMock()
            mock_setup_logging.return_value = mock_logger
            
            # Mock parser.print_help to avoid actual print
            with patch('argparse.ArgumentParser.print_help') as mock_print_help:
                result = main.main()
            
            self.assertEqual(result, 1)
            mock_print_help.assert_called_once()
    
    @patch('main.setup_logging')
    @patch('main.cmd_multi_style')
    def test_main_keyboard_interrupt(self, mock_cmd_multi, mock_setup_logging):
        """Test main function handling KeyboardInterrupt."""
        sys.argv = ["main.py", "multi"]
        
        mock_logger = MagicMock()
        mock_setup_logging.return_value = mock_logger
        mock_cmd_multi.side_effect = KeyboardInterrupt()
        
        result = main.main()
        
        self.assertEqual(result, 0)  # Should return 0 for KeyboardInterrupt
        mock_logger.info.assert_called_with("Interrupted by user")


class TestLoggingSetup(unittest.TestCase):
    """Test logging setup function."""
    
    @patch('logging.basicConfig')
    @patch('pathlib.Path.mkdir')
    @patch('logging.FileHandler')
    def test_setup_logging_with_file(self, mock_file_handler, mock_mkdir, mock_basic_config):
        """Test logging setup with log file."""
        mock_handler = MagicMock()
        mock_file_handler.return_value = mock_handler
        
        logger = main.setup_logging(
            level="DEBUG",
            log_file="/path/to/logfile.log",
            verbose=True
        )
        
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "pyforex")
        
        # Check basicConfig was called with correct parameters
        mock_basic_config.assert_called_once()
        call_kwargs = mock_basic_config.call_args[1]
        
        self.assertEqual(call_kwargs.get("level"), logging.DEBUG)
        self.assertIn("format", call_kwargs)
        self.assertIn("handlers", call_kwargs)
        
        # Should have 2 handlers: StreamHandler and FileHandler
        handlers = call_kwargs.get("handlers", [])
        self.assertEqual(len(handlers), 2)
        
        # Check mkdir was called
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @patch('logging.basicConfig')
    def test_setup_logging_without_file(self, mock_basic_config):
        """Test logging setup without log file."""
        logger = main.setup_logging(
            level="WARNING",
            log_file=None,
            verbose=False
        )
        
        self.assertIsInstance(logger, logging.Logger)
        
        # Check basicConfig was called
        mock_basic_config.assert_called_once()
        call_kwargs = mock_basic_config.call_args[1]
        
        self.assertEqual(call_kwargs.get("level"), logging.WARNING)
        
        # Should have only StreamHandler
        handlers = call_kwargs.get("handlers", [])
        self.assertEqual(len(handlers), 1)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios."""
    
    @patch('main.SystemChecker')
    @patch('main.TradingBot')
    @patch('main.NeuralHybridStrategy')
    def test_full_live_trading_flow(self, mock_strategy, mock_bot_class, mock_checker_class):
        """Test full live trading flow."""
        # Setup
        mock_checker = MagicMock()
        mock_checker.run_all_checks.return_value = True
        mock_checker_class.return_value = mock_checker
        
        mock_bot = MagicMock()
        mock_bot_class.return_value = mock_bot
        
        # Create args
        args = argparse.Namespace(
            symbol="EURUSD",
            timeframe="H1",
            strategy="neural",
            interval=10.0,
            mock=False,
            dry_run=False
        )
        
        logger = MagicMock()
        
        # Execute
        result = main.cmd_live(args, logger)
        
        # Verify
        self.assertEqual(result, 0)
        mock_checker.run_all_checks.assert_called_with("live")
        mock_bot_class.assert_called_once()
        mock_bot.run.assert_called_once()
    
    @patch('main.SystemChecker')
    @patch('main.train_lstm_model')
    def test_training_with_cpu_warning(self, mock_train, mock_checker_class):
        """Test training flow with CPU warning (no CUDA)."""
        mock_checker = MagicMock()
        mock_checker.run_all_checks.return_value = True
        mock_checker.warnings = ["CUDA not available - training will be slow on CPU"]
        mock_checker_class.return_value = mock_checker
        
        args = argparse.Namespace(
            model="lstm",
            data="data/train.csv",
            save_dir=None,
            epochs=10,
            batch_size=32,
            lr=0.001,
            seq_len=60,
            attention=False,
            dry_run=False
        )
        
        logger = MagicMock()
        
        result = main.cmd_train(args, logger)
        
        self.assertEqual(result, 0)
        mock_train.assert_called_once()


# Fixtures for pytest
@pytest.fixture
def mock_system_checker():
    """Create a mock SystemChecker for pytest tests."""
    return MagicMock(spec=main.SystemChecker)


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
    data = {
        'time': dates,
        'open': np.random.randn(100) + 1.1,
        'high': np.random.randn(100) + 1.15,
        'low': np.random.randn(100) + 1.05,
        'close': np.random.randn(100) + 1.12,
        'volume': np.random.randint(100, 1000, 100)
    }
    
    return pd.DataFrame(data)


# Pytest-specific tests
def test_command_handler_mapping():
    """Test that all commands have handlers."""
    parser = main.create_parser()
    
    # Get all subparsers
    subparsers_action = None
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            subparsers_action = action
            break
    
    assert subparsers_action is not None
    
    # Check all subcommands have handlers
    for cmd_name in subparsers_action.choices:
        assert cmd_name in main.command_handlers
        assert callable(main.command_handlers[cmd_name])


@pytest.mark.parametrize("level,expected", [
    ("DEBUG", logging.DEBUG),
    ("INFO", logging.INFO),
    ("WARNING", logging.WARNING),
    ("ERROR", logging.ERROR),
])
def test_setup_logging_levels(level, expected):
    """Test logging setup with different levels."""
    with patch('logging.basicConfig') as mock_basic_config:
        main.setup_logging(level=level, log_file=None, verbose=False)
        
        call_kwargs = mock_basic_config.call_args[1]
        assert call_kwargs.get("level") == expected


def test_system_checker_print_status_calls(mock_system_checker):
    """Test that print_status calls all check methods."""
    with patch('builtins.print'):
        mock_system_checker.check_cuda.return_value = {"cuda_available": True}
        mock_system_checker.check_mt5.return_value = True
        
        # Mock Path.exists for weights
        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.stat') as mock_stat:
                mock_stat.return_value.st_size = 1024 * 1024  # 1 MB
                
                mock_system_checker.print_status(verbose=True)
    
    # Verify all check methods were called
    mock_system_checker.check_cuda.assert_called_once()
    mock_system_checker.check_mt5.assert_called_once()


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)