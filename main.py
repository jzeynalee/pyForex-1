#!/usr/bin/env python3
# main.py
"""
pyForex - Multi-Modal Forex Trading System
==========================================

Unified CLI entry point for all operations:
    - Multi-style trading (scalp + intraday + swing simultaneously)
    - Live trading with MT5 (single strategy)
    - Backtesting on historical data
    - Model training (TCN, ViT, Fusion, YOLO, Trend Classifier)
    - Single predictions / inference
    - Dataset generation

Usage:
    python main.py multi [options]          # Multi-style trading (recommended)
    python main.py live [options]           # Single-style live trading
    python main.py backtest [options]       # Run backtest
    python main.py train <model> [options]  # Train a model
    python main.py predict [options]        # Run prediction
    python main.py generate <dataset>       # Generate training data
    python main.py status                   # Check system status

Examples:
    python main.py multi --symbol EURUSD                    # All 3 styles
    python main.py multi --symbol EURUSD --no-scalp         # Intraday + Swing only
    python main.py multi --mock                             # Test with mock connector
    python main.py live --symbol EURUSD --timeframe H1      # Single strategy
    python main.py backtest --data data/EURUSD_H1.csv --strategy neural
    python main.py train tcn --epochs 50 --data data/raw/eurusd.csv
    python main.py train vit --data-dir datasets/vit --epochs 30
    python main.py generate yolo --synthetic --samples 5000
    python main.py status --verbose
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AppConfig:
    """Application-wide configuration."""
    # Paths
    project_root: Path = PROJECT_ROOT
    weights_dir: Path = PROJECT_ROOT / "models" / "weights"
    data_dir: Path = PROJECT_ROOT / "data"
    logs_dir: Path = PROJECT_ROOT / "logs"
    
    # Defaults
    default_symbol: str = "EURUSD"
    default_timeframe: str = "H1"
    default_sequence_length: int = 60
    default_image_size: int = 224
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True


CONFIG = AppConfig()


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> logging.Logger:
    """Configure logging for the application."""
    
    log_level = logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO)
    
    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Format
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=fmt,
        datefmt=date_fmt,
        handlers=handlers,
        force=True,
    )
    
    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    return logging.getLogger("pyforex")


# ============================================================================
# HEALTH CHECKS & VALIDATION
# ============================================================================

class SystemChecker:
    """Validates system prerequisites before running operations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.issues = []
        self.warnings = []
    
    def check_weights(self, required: list[str] = None) -> bool:
        """Check if model weights exist."""
        weights_dir = CONFIG.weights_dir
        
        if not weights_dir.exists():
            self.issues.append(f"Weights directory not found: {weights_dir}")
            return False
        
        required = required or ["tcn_best.pt", "fusion_best.pt"]
        missing = []
        
        for weight_file in required:
            if not (weights_dir / weight_file).exists():
                missing.append(weight_file)
        
        if missing:
            self.warnings.append(f"Missing weight files: {missing}")
            return False
        
        return True
    
    def check_mt5(self) -> bool:
        """Check MT5 availability."""
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                self.warnings.append("MT5 not initialized (terminal may be closed)")
                return False
            mt5.shutdown()
            return True
        except ImportError:
            self.warnings.append("MetaTrader5 library not installed")
            return False
        except Exception as e:
            self.warnings.append(f"MT5 check failed: {e}")
            return False
    
    def check_cuda(self) -> Dict[str, Any]:
        """Check CUDA/GPU availability."""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            info = {
                "cuda_available": cuda_available,
                "device_count": torch.cuda.device_count() if cuda_available else 0,
                "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
                "torch_version": torch.__version__,
            }
            return info
        except ImportError:
            return {"cuda_available": False, "error": "PyTorch not installed"}
    
    def check_dependencies(self) -> bool:
        """Check critical dependencies."""
        required = [
            ("torch", "PyTorch"),
            ("numpy", "NumPy"),
            ("pandas", "Pandas"),
            ("sklearn", "Scikit-learn"),
        ]
        
        optional = [
            ("timm", "timm (for ViT)"),
            ("ultralytics", "Ultralytics (for YOLO)"),
            ("xgboost", "XGBoost"),
        ]
        
        missing_required = []
        missing_optional = []
        
        for module, name in required:
            try:
                __import__(module)
            except ImportError:
                missing_required.append(name)
        
        for module, name in optional:
            try:
                __import__(module)
            except ImportError:
                missing_optional.append(name)
        
        if missing_required:
            self.issues.append(f"Missing required: {missing_required}")
            return False
        
        if missing_optional:
            self.warnings.append(f"Missing optional: {missing_optional}")
        
        return True
    
    def check_data(self, path: Optional[str] = None) -> bool:
        """Check if data file exists."""
        if path:
            if not Path(path).exists():
                self.issues.append(f"Data file not found: {path}")
                return False
        return True
    
    def run_all_checks(self, mode: str = "live") -> bool:
        """Run all relevant checks for a given mode."""
        self.issues = []
        self.warnings = []
        
        # Always check dependencies
        self.check_dependencies()
        
        if mode == "live":
            self.check_mt5()
            self.check_weights(["tcn_best.pt", "fusion_best.pt"])

        elif mode == "backtest":
            self.check_weights(["tcn_best.pt"])
        
        elif mode == "train":
            # Just check CUDA info
            cuda_info = self.check_cuda()
            if not cuda_info.get("cuda_available"):
                self.warnings.append("CUDA not available - training will be slow on CPU")
        
        elif mode == "predict":
            self.check_weights()
        
        # Report
        if self.issues:
            self.logger.error("❌ Critical issues found:")
            for issue in self.issues:
                self.logger.error(f"   • {issue}")
            return False
        
        if self.warnings:
            self.logger.warning("⚠️  Warnings:")
            for warn in self.warnings:
                self.logger.warning(f"   • {warn}")
        
        return True
    
    def print_status(self, verbose: bool = False):
        """Print comprehensive system status."""
        print("\n" + "=" * 60)
        print("  pyForex System Status")
        print("=" * 60)
        
        # Dependencies
        print("\n📦 Dependencies:")
        deps = [
            ("torch", "PyTorch"), ("numpy", "NumPy"), ("pandas", "Pandas"),
            ("sklearn", "Scikit-learn"), ("timm", "timm"), 
            ("ultralytics", "YOLO"), ("xgboost", "XGBoost"),
            ("MetaTrader5", "MT5"),
        ]
        for module, name in deps:
            try:
                mod = __import__(module)
                version = getattr(mod, "__version__", "✓")
                print(f"   ✅ {name}: {version}")
            except ImportError:
                print(f"   ❌ {name}: not installed")
        
        # CUDA
        print("\n🖥️  GPU/CUDA:")
        cuda_info = self.check_cuda()
        if cuda_info.get("cuda_available"):
            print(f"   ✅ CUDA available")
            print(f"   • Device: {cuda_info.get('device_name')}")
            print(f"   • Count: {cuda_info.get('device_count')}")
        else:
            print("   ❌ CUDA not available")
        
        # Model Weights
        print("\n⚖️  Model Weights:")
        weights_dir = CONFIG.weights_dir
        weight_files = [
            "tcn_best.pt", "vit_best.pt", "fusion_best.pt",
            "yolo_best.pt", "trend_classifier.joblib",
        ]
        for wf in weight_files:
            path = weights_dir / wf
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {wf} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {wf} (not found)")
        
        # MT5
        print("\n📊 MetaTrader 5:")
        if self.check_mt5():
            print("   ✅ Connected")
        else:
            print("   ❌ Not available")
        
        # Directories
        if verbose:
            print("\n📁 Directories:")
            dirs = [
                ("Project Root", CONFIG.project_root),
                ("Weights", CONFIG.weights_dir),
                ("Data", CONFIG.data_dir),
                ("Logs", CONFIG.logs_dir),
            ]
            for name, path in dirs:
                exists = "✅" if path.exists() else "❌"
                print(f"   {exists} {name}: {path}")
        
        print("\n" + "=" * 60 + "\n")


# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def cmd_multi_style(args, logger: logging.Logger):
    """Handle multi-style trading command."""
    logger.info("🚀 Starting Multi-Style Trading Mode")
    
    # Determine which styles are enabled
    enabled_styles = []
    if not args.no_scalp:
        enabled_styles.append("SCALP")
    if not args.no_intraday:
        enabled_styles.append("INTRADAY")
    if not args.no_swing:
        enabled_styles.append("SWING")
    
    logger.info(f"🔍 Checking model weights for enabled styles: {enabled_styles}")
    
    # Check and train missing models before starting
    try:
        from training.auto_retrain import check_and_train_missing_models
        if not check_and_train_missing_models(enabled_styles, args.symbol):
            logger.error("❌ Model training failed. System not ready for trading.")
            return 1
    except Exception as e:
        logger.error(f"❌ Auto-training failed: {e}")
        logger.warning("⚠️ Continuing with basic system checks...")
    
    checker = SystemChecker(logger)
    if not checker.run_all_checks("live"):
        logger.error("Prerequisites not met. Aborting.")
        return 1
    
    if args.dry_run:
        logger.info("DRY RUN - would start multi-style trading with:")
        logger.info(f"  Symbol: {args.symbol}")
        logger.info(f"  Scalping: {'enabled' if not args.no_scalp else 'disabled'}")
        logger.info(f"  Intraday: {'enabled' if not args.no_intraday else 'disabled'}")
        logger.info(f"  Swing: {'enabled' if not args.no_swing else 'disabled'}")
        return 0
    
    try:
        from trading.multi_style_orchestrator import run_multi_style_bot
        
        run_multi_style_bot(
            symbol=args.symbol,
            mock=args.mock,
            enable_scalp=not args.no_scalp,
            enable_intraday=not args.no_intraday,
            enable_swing=not args.no_swing,
        )
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Multi-style trading error: {e}", exc_info=True)
        return 1


def cmd_live(args, logger: logging.Logger):
    """Handle live trading command (single style)."""
    logger.info("🚀 Starting Live Trading Mode (Single Style)")
    
    checker = SystemChecker(logger)
    if not checker.run_all_checks("live"):
        logger.error("Prerequisites not met. Aborting.")
        return 1
    
    if args.dry_run:
        logger.info("DRY RUN - would start live trading with:")
        logger.info(f"  Symbol: {args.symbol}")
        logger.info(f"  Timeframe: {args.timeframe}")
        logger.info(f"  Strategy: {args.strategy}")
        return 0
    
    try:
        from utils.config import settings
        from trading.bot import TradingBot, BotConfig
        from strategies.neural_hybrid import NeuralHybridStrategy
        
        # Override settings from CLI
        config = BotConfig(
            symbol=args.symbol,
            timeframe=args.timeframe,
            tick_interval=args.interval,
            use_mock=args.mock,
        )
        
        strategy_map = {
            "neural": NeuralHybridStrategy,
        }
        
        strategy_cls = strategy_map.get(args.strategy, NeuralHybridStrategy)
        
        bot = TradingBot(config=config, strategy_class=strategy_cls)
        bot.run()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Live trading error: {e}", exc_info=True)
        return 1


def cmd_backtest(args, logger: logging.Logger):
    """Handle backtest command."""
    logger.info("📊 Starting Backtest Mode")
    
    checker = SystemChecker(logger)
    if not checker.check_data(args.data):
        return 1
    
    if args.dry_run:
        logger.info("DRY RUN - would run backtest with:")
        logger.info(f"  Data: {args.data}")
        logger.info(f"  Strategy: {args.strategy}")
        logger.info(f"  Initial Balance: ${args.balance:,.2f}")
        return 0
    
    try:
        import pandas as pd
        from trading.bot import BacktestBot
        from strategies.neural_hybrid import NeuralHybridStrategy

        # Load data
        logger.info(f"Loading data from {args.data}")
        df = pd.read_csv(args.data)
        df.columns = df.columns.str.lower()

        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])

        logger.info(f"Loaded {len(df)} candles")

        # Select strategy (TCN-based neural hybrid)
        strategy_map = {
            "neural": NeuralHybridStrategy,
            "tcn": NeuralHybridStrategy,  # Alias for clarity
        }
        strategy_cls = strategy_map.get(args.strategy, NeuralHybridStrategy)
        
        # Run backtest
        bot = BacktestBot(
            data=df,
            strategy_class=strategy_cls,
            initial_balance=args.balance,
        )
        
        results = bot.run()
        
        # Print results
        print("\n" + "=" * 60)
        print("  BACKTEST RESULTS")
        print("=" * 60)
        print(f"\n  Strategy: {args.strategy}")
        print(f"  Period: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
        print(f"  Candles: {len(df)}")
        
        print(f"\n  Initial Balance: ${args.balance:,.2f}")
        print(f"  Final Balance:   ${results['final_balance']:,.2f}")
        
        pnl = results['final_balance'] - args.balance
        pnl_pct = (pnl / args.balance) * 100
        print(f"  P&L:             ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
        
        print(f"\n  Total Trades: {len(results['trades'])}")
        
        # Detailed metrics if available
        if results['trades']:
            wins = [t for t in results['trades'] if t.get('pnl', 0) > 0]
            losses = [t for t in results['trades'] if t.get('pnl', 0) < 0]
            
            win_rate = len(wins) / len(results['trades']) * 100 if results['trades'] else 0
            print(f"  Win Rate: {win_rate:.1f}%")
            print(f"  Wins: {len(wins)}, Losses: {len(losses)}")
        
        print("\n" + "=" * 60 + "\n")
        
        # Save results if requested
        if args.output:
            import json
            output_path = Path(args.output)
            
            # Convert trades to serializable format
            serializable_results = {
                "final_balance": results["final_balance"],
                "initial_balance": args.balance,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "total_trades": len(results["trades"]),
                "trades": [
                    {k: str(v) if isinstance(v, datetime) else v for k, v in t.items()}
                    for t in results["trades"]
                ],
            }
            
            with open(output_path, "w") as f:
                json.dump(serializable_results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        return 1


def cmd_train(args, logger: logging.Logger):
    """Handle training command."""
    model = args.model.lower()
    logger.info(f"🏋️ Starting Training: {model.upper()}")
    
    checker = SystemChecker(logger)
    checker.run_all_checks("train")
    
    if args.dry_run:
        logger.info("DRY RUN - would train with:")
        logger.info(f"  Model: {model}")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Data: {args.data or args.data_dir}")
        return 0
    
    try:
        if model == "vit":
            from training.train_vit import train_classifier, maybe_build_cache, get_args as get_vit_args
            
            # Build cache and train
            device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            cache = maybe_build_cache(args.data_dir, args.cache_path or "./dataset_cache.pt", device)
            
            # Create args-like object for training
            class VitArgs:
                pass
            
            vit_args = VitArgs()
            vit_args.data_dir = args.data_dir
            vit_args.batch_size = args.batch_size
            vit_args.num_epochs = args.epochs
            vit_args.lr = args.lr
            vit_args.weight_decay = 0.05
            vit_args.dropout = 0.2
            vit_args.label_smoothing = 0.1
            vit_args.mixup_alpha = 0.2
            vit_args.patience = 10
            vit_args.save_dir = args.save_dir or "./checkpoints_vit"
            vit_args.cache_path = args.cache_path or "./dataset_cache.pt"
            vit_args.device = device
            
            train_classifier(
                cache["train_x"], cache["train_y"],
                cache["val_x"], cache["val_y"],
                cache["num_classes"], vit_args
            )
        
        elif model == "vit-finetune":
            from training.finetune_vit import train
            
            class FTArgs:
                pass
            
            ft_args = FTArgs()
            ft_args.data_dir = args.data_dir
            ft_args.batch_size = args.batch_size
            ft_args.num_epochs = args.epochs
            ft_args.lr_head = args.lr
            ft_args.lr_backbone = args.lr / 100
            ft_args.weight_decay = 0.01
            ft_args.unfreeze_blocks = 4
            ft_args.dropout = 0.1
            ft_args.label_smoothing = 0.1
            ft_args.patience = 7
            ft_args.warmup_epochs = 2
            ft_args.save_dir = args.save_dir or "./checkpoints_vit_finetuned"
            ft_args.num_workers = 4
            ft_args.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            ft_args.resume = None
            
            train(ft_args)
        
        elif model == "fusion":
            from training.train_fusion import train_fusion_model
            
            train_fusion_model(
                data_path=args.data,
                weights_dir=args.save_dir or "models/weights",
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
            )
        
        elif model == "yolo":
            from training.train_yolo import model as yolo_model
            logger.info("YOLO training started via ultralytics...")
            # Note: train_yolo.py runs directly, this is informational
            logger.info("Run: python training/train_yolo.py directly for full control")
        
        elif model == "trend":
            from training.train_trend_classifier import main as train_trend_main
            
            # Construct sys.argv for the script
            sys_argv = ["train_trend_classifier.py"]
            if args.data:
                sys_argv.extend(["--data", args.data])
            if args.synthetic:
                sys_argv.append("--synthetic")
            sys_argv.extend(["--output", args.save_dir or "models/trend_classifier.joblib"])
            sys_argv.extend(["--n-estimators", str(args.epochs)])  # Reuse epochs for n_estimators
            
            sys.argv = sys_argv
            train_trend_main()
        
        else:
            logger.error(f"Unknown model: {model}")
            logger.info("Available models: tcn, vit, vit-finetune, fusion, yolo, trend")
            return 1
        
        logger.info("✅ Training complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        return 1


def cmd_predict(args, logger: logging.Logger):
    """Handle prediction command."""
    logger.info("🔮 Running Prediction")
    
    checker = SystemChecker(logger)
    if not checker.run_all_checks("predict"):
        return 1
    
    try:
        import pandas as pd
        import torch
        from inference.predictor import HybridPredictor, RiskAwareTCNPredictor

        # Load data
        if args.data:
            df = pd.read_csv(args.data)
            df.columns = df.columns.str.lower()
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
        else:
            # Try to fetch from MT5
            logger.info("Fetching live data from MT5...")
            from trading.mt5_connector import MT5Connector
            connector = MT5Connector(symbol=args.symbol)
            if not connector.connect():
                logger.error("Cannot connect to MT5")
                return 1
            df = connector.get_data(n=100)
            connector.disconnect()

        if df.empty:
            logger.error("No data available")
            return 1

        logger.info(f"Data: {len(df)} candles, last close: {df['close'].iloc[-1]:.5f}")

        # Run prediction
        if args.simple:
            predictor = RiskAwareTCNPredictor(
                weights_path=args.weights or "models/weights/tcn_best.pt"
            )
        else:
            predictor = HybridPredictor(
                weights_dir=args.weights or "models/weights"
            )

        result = predictor.predict(df)

        # Display results
        print("\n" + "=" * 50)
        print("  PREDICTION RESULT")
        print("=" * 50)

        class_names = ['BUY', 'SELL', 'HOLD']
        print(f"\n  Signal: {class_names[result.predicted_class]}")
        print(f"  Confidence: {result.confidence:.2%}")

        print(f"\n  Probabilities:")
        for i, name in enumerate(class_names):
            bar = "█" * int(result.probabilities[i] * 30)
            print(f"    {name:5}: {result.probabilities[i]:.2%} {bar}")

        if result.gate_weights is not None:
            print(f"\n  Gate Weights (modality importance):")
            modalities = ['TCN', 'ViT', 'YOLO']
            for i, name in enumerate(modalities):
                print(f"    {name}: {result.gate_weights[i]:.2%}")
        
        print("\n" + "=" * 50 + "\n")
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return 1


def cmd_generate(args, logger: logging.Logger):
    """Handle dataset generation command."""
    dataset_type = args.dataset.lower()
    logger.info(f"📦 Generating Dataset: {dataset_type.upper()}")
    
    if args.dry_run:
        logger.info("DRY RUN - would generate:")
        logger.info(f"  Type: {dataset_type}")
        logger.info(f"  Samples: {args.samples}")
        logger.info(f"  Output: {args.output}")
        logger.info(f"  Synthetic: {args.synthetic}")
        return 0
    
    try:
        output_dir = Path(args.output or f"datasets/{dataset_type}")
        
        if dataset_type == "yolo":
            from utils.yolo_dataset_generator import YOLODatasetGenerator
            
            generator = YOLODatasetGenerator(
                output_dir=str(output_dir),
                image_size=args.image_size,
                window_size=args.window,
                stride=args.stride,
            )
            
            if args.synthetic or not args.data:
                generator.generate_synthetic(n_samples=args.samples)
            else:
                generator.generate_from_csv(args.data, max_samples=args.samples)
        
        elif dataset_type == "vit":
            from utils.vit_dataset_generator import ViTDatasetGenerator, FuturePriceLabeler
            
            labeler = FuturePriceLabeler(forward_bars=10, threshold_pct=0.5)
            
            generator = ViTDatasetGenerator(
                output_dir=str(output_dir),
                image_size=args.image_size,
                window_size=args.window,
                stride=args.stride,
                labeler=labeler,
            )
            
            if args.synthetic or not args.data:
                generator.generate_synthetic(n_samples=args.samples)
            else:
                generator.generate_from_csv(args.data, max_samples=args.samples)
        
        elif dataset_type == "both":
            # Generate both datasets
            logger.info("Generating YOLO dataset...")
            cmd_generate_args = argparse.Namespace(
                dataset="yolo", data=args.data, output=str(output_dir / "yolo"),
                samples=args.samples, synthetic=args.synthetic,
                image_size=256, window=args.window, stride=10, dry_run=False,
            )
            cmd_generate(cmd_generate_args, logger)
            
            logger.info("Generating ViT dataset...")
            cmd_generate_args.dataset = "vit"
            cmd_generate_args.output = str(output_dir / "vit")
            cmd_generate_args.image_size = 224
            cmd_generate_args.stride = 5
            cmd_generate(cmd_generate_args, logger)
        
        else:
            logger.error(f"Unknown dataset type: {dataset_type}")
            logger.info("Available: yolo, vit, both")
            return 1
        
        logger.info(f"✅ Dataset generated at {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return 1


def cmd_status(args, logger: logging.Logger):
    """Handle status command."""
    checker = SystemChecker(logger)
    checker.print_status(verbose=args.verbose)
    return 0


# ============================================================================
# CLI ARGUMENT PARSER
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    
    parser = argparse.ArgumentParser(
        prog="pyforex",
        description="pyForex - Multi-Modal Forex Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s multi --symbol EURUSD                  # All 3 styles (scalp+intraday+swing)
  %(prog)s multi --no-scalp                       # Intraday + Swing only
  %(prog)s multi --mock                           # Test with mock connector
  %(prog)s live --symbol EURUSD --timeframe H1    # Single strategy mode
  %(prog)s backtest --data data/EURUSD_H1.csv
  %(prog)s train tcn --epochs 50
  %(prog)s predict --symbol EURUSD
  %(prog)s generate vit --synthetic --samples 10000
  %(prog)s status --verbose
        """,
    )
    
    # Global arguments
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log to file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # -------------------------------------------------------------------------
    # MULTI-STYLE TRADING
    # -------------------------------------------------------------------------
    multi_parser = subparsers.add_parser(
        "multi",
        help="Start multi-style trading (scalp + intraday + swing)",
        description="Run simultaneous scalping, intraday, and swing trading strategies",
    )
    multi_parser.add_argument("--symbol", default="EURUSD", help="Trading symbol")
    multi_parser.add_argument("--mock", action="store_true", help="Use mock connector (testing)")
    multi_parser.add_argument("--no-scalp", action="store_true", help="Disable scalping strategy")
    multi_parser.add_argument("--no-intraday", action="store_true", help="Disable intraday strategy")
    multi_parser.add_argument("--no-swing", action="store_true", help="Disable swing strategy")
    
    # -------------------------------------------------------------------------
    # LIVE TRADING (SINGLE STYLE)
    # -------------------------------------------------------------------------
    live_parser = subparsers.add_parser(
        "live",
        help="Start live trading (single style)",
        description="Run live trading with MT5 connection (single strategy)",
    )
    live_parser.add_argument("--symbol", default="EURUSD", help="Trading symbol")
    live_parser.add_argument("--timeframe", default="H1", help="Timeframe (M1,M5,M15,M30,H1,H4,D1)")
    live_parser.add_argument("--strategy", default="neural", help="Strategy (neural, tcn)")
    live_parser.add_argument("--interval", type=float, default=10.0, help="Check interval in seconds")
    live_parser.add_argument("--mock", action="store_true", help="Use mock connector (testing)")
    
    # -------------------------------------------------------------------------
    # BACKTEST
    # -------------------------------------------------------------------------
    bt_parser = subparsers.add_parser(
        "backtest",
        help="Run backtest on historical data",
        description="Backtest strategy on historical OHLCV data",
    )
    bt_parser.add_argument("--data", required=True, help="Path to OHLCV CSV file")
    bt_parser.add_argument("--strategy", default="neural", help="Strategy (neural, tcn)")
    bt_parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    bt_parser.add_argument("--output", type=str, help="Save results to JSON file")
    
    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model",
        description="Train ML models (TCN, ViT, Fusion, YOLO, Trend)",
    )
    train_parser.add_argument(
        "model",
        choices=["tcn", "vit", "vit-finetune", "fusion", "yolo", "trend"],
        help="Model to train",
    )
    train_parser.add_argument("--data", type=str, help="Path to training data CSV")
    train_parser.add_argument("--data-dir", type=str, help="Path to dataset directory (for ViT)")
    train_parser.add_argument("--save-dir", type=str, help="Directory to save weights")
    train_parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--seq-len", type=int, default=60, help="Sequence length (TCN)")
    train_parser.add_argument("--cache-path", type=str, help="Feature cache path (ViT)")
    train_parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (trend)")
    
    # -------------------------------------------------------------------------
    # PREDICT
    # -------------------------------------------------------------------------
    pred_parser = subparsers.add_parser(
        "predict",
        help="Run prediction",
        description="Generate prediction from data",
    )
    pred_parser.add_argument("--data", type=str, help="Path to OHLCV CSV (or fetch from MT5)")
    pred_parser.add_argument("--symbol", default="EURUSD", help="Symbol for MT5 fetch")
    pred_parser.add_argument("--weights", type=str, help="Path to weights directory/file")
    pred_parser.add_argument("--simple", action="store_true", help="Use simple TCN predictor")
    
    # -------------------------------------------------------------------------
    # GENERATE
    # -------------------------------------------------------------------------
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate training dataset",
        description="Generate YOLO or ViT training datasets",
    )
    gen_parser.add_argument(
        "dataset",
        choices=["yolo", "vit", "both"],
        help="Dataset type to generate",
    )
    gen_parser.add_argument("--data", type=str, help="Source OHLCV CSV file")
    gen_parser.add_argument("--output", type=str, help="Output directory")
    gen_parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    gen_parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    gen_parser.add_argument("--image-size", type=int, default=224, help="Image size")
    gen_parser.add_argument("--window", type=int, default=60, help="Candles per window")
    gen_parser.add_argument("--stride", type=int, default=5, help="Window stride")
    
    # -------------------------------------------------------------------------
    # STATUS
    # -------------------------------------------------------------------------
    status_parser = subparsers.add_parser(
        "status",
        help="Check system status",
        description="Display system status and prerequisites",
    )
    status_parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed info")
    
    return parser


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file
    if log_file is None and hasattr(args, 'command') and args.command == "live":
        log_file = f"logs/live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = setup_logging(
        level="DEBUG" if args.verbose else "INFO",
        log_file=log_file,
        verbose=args.verbose,
    )
    
    # Banner
    if args.command:
        logger.info("=" * 60)
        logger.info("  pyForex - Multi-Modal Forex Trading System")
        logger.info("=" * 60)
    
    # Dispatch command
    if args.command is None:
        parser.print_help()
        return 0
    
    command_handlers = {
        "multi": cmd_multi_style,
        "live": cmd_live,
        "backtest": cmd_backtest,
        "train": cmd_train,
        "predict": cmd_predict,
        "generate": cmd_generate,
        "status": cmd_status,
    }
    
    handler = command_handlers.get(args.command)
    if handler:
        return handler(args, logger)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
