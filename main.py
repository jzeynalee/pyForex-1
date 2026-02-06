#!/usr/bin/env python3
# main.py
"""
pyForex - Multi-Modal Forex Trading System (v2.0)
==================================================

Unified CLI entry point for all operations:
    - Multi-style trading (scalp + intraday + swing simultaneously)
    - Live trading with MT5 (single strategy)
    - Backtesting on historical data
    - Model training (MH-TCN with walk-forward validation)
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
    python main.py train walk-forward --data data/raw/eurusd.csv --profile INTRADAY
    python main.py train mhtcn --data data/raw/eurusd.csv --epochs 30
    python main.py status --verbose

Training Models (v2.0):
    walk-forward (wf)  - Walk-forward validated MH-TCN (RECOMMENDED)
    mhtcn / tcn        - Single-fold MH-TCN training
    trend              - Trend classifier (XGBoost)
    fusion             - Fusion network training
"""

import argparse
import logging
import sys
import os
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

try:
    from utils.mtf_config import get_profile
    HAS_MTF = True
except ImportError:
    get_profile = None
    HAS_MTF = False

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
        try:
            from utils.config import settings
            weights_dir = Path(getattr(settings, 'WEIGHTS_DIR', CONFIG.weights_dir))
        except Exception:
            weights_dir = CONFIG.weights_dir
        
        if not weights_dir.exists():
            self.issues.append(f"Weights directory not found: {weights_dir}")
            return False
        
        required = required or ["multihead_tcn_INTRADAY.pth"]
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
            self.check_weights(["multihead_tcn_INTRADAY.pth"])

        elif mode == "backtest":
            self.check_weights(["multihead_tcn_INTRADAY.pth"])
        
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
        try:
            from utils.config import settings
            weights_dir = Path(getattr(settings, 'WEIGHTS_DIR', CONFIG.weights_dir))
            trend_model = Path(getattr(settings, 'TREND_MODEL_PATH', weights_dir / 'trend_classifier.joblib'))
        except Exception:
            weights_dir = CONFIG.weights_dir
            trend_model = weights_dir / 'trend_classifier.joblib'
        weight_files = [
            "multihead_tcn_INTRADAY.pth",
            "multihead_tcn_INTRADAY_pa_v1.pth",
            "vit_INTRADAY.pth",
            "scaler.joblib",
        ]
        for wf in weight_files:
            path = weights_dir / wf
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {wf} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {wf} (not found)")

        if trend_model.exists():
            size_mb = trend_model.stat().st_size / (1024 * 1024)
            print(f"   ✅ trend_classifier.joblib ({size_mb:.1f} MB)")
        else:
            print("   ❌ trend_classifier.joblib (not found)")
        
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

    try:
        from utils.config import settings
        if bool(getattr(settings, 'ENFORCE_AUTHORITATIVE_PIPELINE', True)):
            logger.error("Authoritative pipeline enforced: multi-style orchestrator is disabled")
            return 1
    except Exception:
        pass
    
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


def cmd_alpha_backtest(args, logger: logging.Logger):
    """Run the Alpha Factory layered backtest on a CSV (separate from ML strategy backtest)."""
    logger.info("📊 Starting Alpha Factory Backtest Mode")

    try:
        import pandas as pd

        logger.info(f"Loading data from {args.data}")
        df = pd.read_csv(args.data)
        df.columns = df.columns.str.lower()

        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])

        if getattr(args, 'max_bars', None):
            df = df.tail(int(args.max_bars)).reset_index(drop=True)

        if 'spread' not in df.columns:
            df['spread'] = 1.0
        if 'tick_volume' not in df.columns:
            if 'volume' in df.columns:
                df['tick_volume'] = df['volume']
            else:
                df['tick_volume'] = 100
        if 'real_volume' not in df.columns:
            df['real_volume'] = df.get('tick_volume', 100) * 100

        engine = str(getattr(args, 'engine', 'decision') or 'decision').lower().strip()
        if engine not in {'decision', 'layered'}:
            engine = 'decision'

        if engine == 'layered':
            logger.error(
                "Layered alpha backtest engine has been removed. "
                "Use --engine decision to evaluate ProbabilisticAlphaFactory on a rolling window."
            )
            return 1

        else:
            from alpha_factory.features_engineering import FeatureEngineerOptimized
            from alpha_factory.market_data import MarketData
            from alpha_factory.probabilistic_alpha_factory import create_probabilistic_alpha_factory

            window = int(getattr(args, 'window', 300) or 300)
            _ = bool(getattr(args, 'disable_signal_quality', False))

            df = df.dropna().reset_index(drop=True)
            if window < 50:
                window = 50

            rejection_counts: Dict[str, int] = {}
            def _bump(key: str):
                rejection_counts[key] = int(rejection_counts.get(key, 0)) + 1

            feat_engineer = FeatureEngineerOptimized()
            engine = create_probabilistic_alpha_factory(profile='INTRADAY')

            buys = 0
            sells = 0
            holds = 0
            evaluated = 0

            for i in range(window - 1, len(df)):
                w = df.iloc[max(0, i - window + 1): i + 1].copy()

                if 'time' not in w.columns:
                    w['time'] = pd.NaT

                w.columns = [str(c).lower().strip() for c in w.columns]
                if 'volume' not in w.columns:
                    if 'tick_volume' in w.columns:
                        w['volume'] = w['tick_volume']
                    else:
                        w['volume'] = 0.0

                feats = feat_engineer.generate_features(w, batch_processing=False)

                swing_points = None
                try:
                    md = MarketData(w, handle_splits=False)
                    swing_points = md.extract_swings(lookback=5, strength_threshold=0.3)
                except Exception:
                    swing_points = None

                out = engine.evaluate(
                    df=w,
                    features=feats,
                    timeframe='H1',
                    swing_points=swing_points,
                    causality_results=None,
                    current_equity=float(getattr(args, 'balance', 10000.0) or 10000.0),
                    signal_id="alpha_backtest",
                )
                evaluated += 1

                if str(out.direction) == 'LONG':
                    buys += 1
                elif str(out.direction) == 'SHORT':
                    sells += 1
                else:
                    holds += 1
                    reasons = ' | '.join([str(r) for r in (getattr(out, 'reasoning', None) or [])])
                    reasons_l = reasons.lower()
                    if 'confidence gate' in reasons_l:
                        _bump('CONFIDENCE_GATE')
                    elif 'trade skipped' in reasons_l and 'regime' in reasons_l:
                        _bump('REGIME_SKIP')
                    elif 'threshold=' in reasons_l and 'p_long=' in reasons_l:
                        _bump('PROB_THRESHOLD')
                    else:
                        _bump('HOLD_OTHER')

            metrics = {
                'engine': 'decision',
                'bars': int(len(df)),
                'window': int(window),
                'evaluated': int(evaluated),
                'buy_signals': int(buys),
                'sell_signals': int(sells),
                'hold_signals': int(holds),
            }
            rejection_summary = dict(sorted(rejection_counts.items(), key=lambda kv: kv[1], reverse=True))

        print("\n" + "=" * 60)
        print("  ALPHA FACTORY BACKTEST RESULTS")
        print("=" * 60)
        if engine == 'layered':
            layer = str(getattr(args, 'layer', 'alpha_only') or 'alpha_only')
            print(f"\n  Engine: layered")
            print(f"  Layer: {layer}")
        else:
            print(f"\n  Engine: decision")
        if isinstance(metrics, dict) and 'error' in metrics:
            print(f"\n  Error: {metrics['error']}")
        else:
            for k, v in (metrics or {}).items():
                print(f"  {k}: {v}")
        if rejection_summary:
            print("\n  Rejection Breakdown:")
            for k, v in rejection_summary.items():
                print(f"  {k}: {v}")
        print("\n" + "=" * 60)

        return 0
    except Exception as e:
        logger.error(f"Alpha backtest error: {e}", exc_info=True)
        return 1


def cmd_fetch_data(args, logger: logging.Logger):
    """Fetch real historical OHLCV from MT5 and save outside the repo (external assets folder)."""
    logger.info("📥 Fetching Historical Data (MT5)")

    checker = SystemChecker(logger)
    if not checker.run_all_checks("live"):
        logger.error("Prerequisites not met. Aborting.")
        return 1

    try:
        import pandas as pd
        from datetime import datetime
        from utils.config import settings
        from trading.mt5_connector import MT5Connector

        symbol = str(getattr(args, 'symbol', None) or getattr(settings, 'SYMBOL', 'EURUSD')).upper()
        tfs = str(getattr(args, 'timeframes', 'H1') or 'H1')
        timeframes = [tf.strip().upper() for tf in tfs.split(',') if tf.strip()]
        n_bars = int(getattr(args, 'bars', 50000) or 50000)

        def _tf_to_minutes(tf: str) -> Optional[int]:
            tf = str(tf or '').upper().strip()
            if not tf:
                return None
            if tf.startswith('M') and tf[1:].isdigit():
                return int(tf[1:])
            if tf.startswith('H') and tf[1:].isdigit():
                return int(tf[1:]) * 60
            if tf in ('D1', '1D'):
                return 24 * 60
            if tf in ('W1', '1W'):
                return 7 * 24 * 60
            if tf in ('MN1', '1MN', 'MO1', '1MO'):
                return 30 * 24 * 60
            return None

        base_tf = timeframes[0] if timeframes else 'H1'
        base_minutes = _tf_to_minutes(base_tf)
        if base_minutes is None:
            logger.warning(f"Could not infer minutes for base timeframe {base_tf}; will fetch {n_bars} bars for each timeframe")

        out_root = (settings.ASSETS_DIR / 'data' / 'mt5' / symbol)
        out_root.mkdir(parents=True, exist_ok=True)

        connector = MT5Connector(
            account=int(getattr(settings, 'MT5_ACCOUNT', 0) or 0),
            password=str(getattr(settings, 'MT5_PASSWORD', '') or ''),
            server=str(getattr(settings, 'MT5_SERVER', '') or ''),
            path=str(getattr(settings, 'MT5_PATH', '') or ''),
            symbol=symbol,
            timeframe=timeframes[0] if timeframes else 'H1',
            magic_number=int(getattr(settings, 'MAGIC_NUMBER', 123456) or 123456),
        )
        if not connector.connect():
            logger.error("Failed to connect/login to MT5")
            return 1

        written = []
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for tf in timeframes:
            tf_minutes = _tf_to_minutes(tf)
            n_fetch = n_bars
            if base_minutes is not None and tf_minutes is not None and tf_minutes > 0:
                # Preserve aligned time-span: bars * minutes ~= constant across TFs.
                # Example: base=M5 (5m). For M15: N*5/15 = N/3. For H1: N*5/60 = N/12.
                n_fetch = int(max(1, math.ceil(n_bars * (base_minutes / float(tf_minutes)))))
            elif base_minutes is not None and tf_minutes is None:
                logger.warning(f"Could not infer minutes for timeframe {tf}; fetching {n_bars} bars")

            logger.info(f"Fetching {symbol} {tf}: {n_fetch} bars (base={base_tf}:{n_bars})")

            df = connector.get_data(n=n_fetch, timeframe=tf)
            if df is None or df.empty:
                logger.error(f"No data fetched for {symbol} {tf}")
                continue

            df = df.copy()
            df.columns = [c.lower().strip() for c in df.columns]
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])

            out_path = out_root / f"{symbol}_{tf}_{stamp}.csv"
            df.to_csv(out_path, index=False)
            written.append(str(out_path))
            logger.info(f"Wrote {len(df)} rows -> {out_path}")

        connector.disconnect()

        if not written:
            logger.error("No files written")
            return 1

        print("\n" + "=" * 60)
        print("  DATA FETCH COMPLETE")
        print("=" * 60)
        for p in written:
            print(f"  {p}")
        print("=" * 60 + "\n")

        return 0
    except Exception as e:
        logger.error(f"Fetch-data error: {e}", exc_info=True)
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
        strategy_map = {
            "neural": NeuralHybridStrategy,
            "tcn": NeuralHybridStrategy,
            "unified3tf": None,
        }

        try:
            from strategies.unified_3tf_strategy import Unified3TFStrategy
            strategy_map["unified3tf"] = Unified3TFStrategy
        except Exception as e:
            logger.warning(f"Could not import Unified3TFStrategy: {e}")
        
        # Override settings from CLI
        config = BotConfig(
            symbol=args.symbol,
            timeframe=args.timeframe,
            tick_interval=args.interval,
            use_mock=args.mock,
        )
        
        strategy_cls = strategy_map.get(str(getattr(args, 'strategy', '') or '').lower().strip(), NeuralHybridStrategy)

        if bool(getattr(settings, 'ENFORCE_AUTHORITATIVE_PIPELINE', True)):
            allow = False
            try:
                allow = (strategy_cls is NeuralHybridStrategy)
            except Exception:
                allow = False
            if not allow:
                try:
                    allow = (strategy_cls.__name__ == 'Unified3TFStrategy')
                except Exception:
                    allow = False
            if not allow:
                logger.error("Authoritative pipeline enforced: only NeuralHybridStrategy or Unified3TFStrategy is allowed")
                return 1
        
        if getattr(strategy_cls, '__name__', '') == 'Unified3TFStrategy':
            from trading.mt5_connector import MT5Connector
            from strategies.unified_3tf_strategy import Unified3TFConfig
            connector = MT5Connector(
                account=int(getattr(settings, 'MT5_ACCOUNT', 0) or 0),
                password=str(getattr(settings, 'MT5_PASSWORD', '') or ''),
                server=str(getattr(settings, 'MT5_SERVER', '') or ''),
                path=str(getattr(settings, 'MT5_PATH', '') or ''),
                symbol=str(args.symbol).upper(),
                timeframe=str(args.timeframe).upper(),
                magic_number=int(getattr(settings, 'MAGIC_NUMBER', 123456) or 123456),
            )
            class UnifiedProvider:
                def __init__(self, conn: MT5Connector):
                    self.conn = conn
                def get_ohlcv(self, symbol: str, timeframe: str = "", count: int = 200):
                    return self.conn.get_data(n=int(count), symbol=str(symbol).upper(), timeframe=str(timeframe).upper())
                def connect(self):
                    return self.conn.connect()
                def disconnect(self):
                    return self.conn.disconnect()
                def get_account_info(self):
                    return self.conn.get_account_info()
                def entry(self, signal: str, volume: float, sl: float, tp: float):
                    return self.conn.entry(signal, volume, sl, tp)
                @property
                def balance(self):
                    info = self.conn.get_account_info()
                    return float(getattr(info, 'balance', 10000.0) or 10000.0)
            provider = UnifiedProvider(connector)
            bot = TradingBot(config=config, strategy_class=lambda **kw: Unified3TFStrategy(
                config=Unified3TFConfig(profile=str(getattr(settings, 'PROFILE', 'INTRADAY') or 'INTRADAY').upper(), symbol=str(args.symbol).upper()),
                data_provider=provider,
                executor=provider,
            ), connector=connector)
        else:
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
    """Handle backtest command.

    Supports an optional bar cap (``--max-bars``) to run a quicker backtest over
    the most recent candles.

    For faster, pure time-series backtests, you can disable non-TCN modalities
    with ``--no-vision`` and/or ``--no-yolo``.

    Note:
        If your model produces low-confidence direction probabilities (common
        with older/legacy checkpoints), you may need to lower the confidence
        gate via ``--min-confidence`` to generate any trades.
    """
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
        from strategies.neural_hybrid import NeuralHybridStrategy, StrategyConfig
        from utils.config import settings

        # Reduce per-bar inference logging noise during backtest.
        logging.getLogger("inference.predictor").setLevel(logging.WARNING)
        logging.getLogger("strategies.neural_hybrid").setLevel(logging.INFO)
        logging.getLogger("risk_management.phase5_capital_protection.protection_rules").setLevel(logging.WARNING)

        def _load_csv(path: str) -> pd.DataFrame:
            df0 = pd.read_csv(path)
            df0.columns = df0.columns.str.lower()
            if 'time' in df0.columns:
                df0['time'] = pd.to_datetime(df0['time'])
            return df0

        logger.info(f"Loading data from {args.data}")
        df = _load_csv(args.data)

        try:
            if 'time' in df.columns and len(df) > 2:
                t0 = pd.to_datetime(df['time'].iloc[0])
                t1 = pd.to_datetime(df['time'].iloc[-1])
                span_days = (t1 - t0).days
                if span_days < 365:
                    logger.error(f"Backtest data span too short: {span_days} days (<365). Provide at least 1 year of data.")
                    return 1
        except Exception:
            pass

        mtf_df = None
        htf_df = None
        if getattr(args, 'data_mtf', None):
            logger.info(f"Loading MTF data from {args.data_mtf}")
            mtf_df = _load_csv(args.data_mtf)
        if getattr(args, 'data_htf', None):
            logger.info(f"Loading HTF data from {args.data_htf}")
            htf_df = _load_csv(args.data_htf)

        if getattr(args, 'max_bars', None):
            df = df.tail(int(args.max_bars)).reset_index(drop=True)

        logger.info(f"Loaded {len(df)} candles")

        # Select strategy (TCN-based neural hybrid)
        class BacktestNeuralStrategy(NeuralHybridStrategy):
            def __init__(self, data_provider=None, executor=None, **kwargs):
                min_mtf_alignment = getattr(args, 'min_mtf_alignment', None)
                if min_mtf_alignment is not None:
                    try:
                        min_mtf_alignment = float(min_mtf_alignment)
                    except Exception:
                        min_mtf_alignment = None
                cfg = StrategyConfig(
                    profile=str(getattr(args, 'profile', 'INTRADAY') or 'INTRADAY').upper(),
                    use_vision=not bool(getattr(args, 'no_vision', False)),
                    use_price_action=not bool(getattr(args, 'no_yolo', False)),
                    min_direction_confidence=float(getattr(args, 'min_confidence', 0.55)),
                    min_mtf_alignment=min_mtf_alignment if min_mtf_alignment is not None else StrategyConfig.min_mtf_alignment,
                )
                try:
                    if cfg.tcn_weights and not Path(str(cfg.tcn_weights)).exists():
                        cfg.tcn_weights = ''
                except Exception:
                    cfg.tcn_weights = ''
                if cfg.profile == 'SCALP':
                    cfg.avoid_rollover = False
                super().__init__(config=cfg, data_provider=data_provider, executor=executor, **kwargs)

        strategy_map = {
            "neural": BacktestNeuralStrategy,
            "tcn": BacktestNeuralStrategy,  # Alias for clarity
            "unified3tf": None,
        }

        try:
            from strategies.unified_3tf_strategy import Unified3TFStrategy
            strategy_map["unified3tf"] = Unified3TFStrategy
        except Exception as e:
            logger.warning(f"Could not import Unified3TFStrategy: {e}")

        try:
            from strategies.unified_3tf_strategy_riskmanaged import RiskManagedUnified3TFStrategy
            strategy_map["unified3tf"] = RiskManagedUnified3TFStrategy
        except Exception as e:
            logger.warning(f"Could not import RiskManagedUnified3TFStrategy: {e}")
        strategy_cls = strategy_map.get(args.strategy, NeuralHybridStrategy)

        if bool(getattr(settings, 'ENFORCE_AUTHORITATIVE_PIPELINE', True)):
            allow = False
            try:
                allow = issubclass(strategy_cls, NeuralHybridStrategy)
            except Exception:
                allow = False
            if not allow:
                try:
                    allow = (strategy_cls.__name__ in ('Unified3TFStrategy', 'RiskManagedUnified3TFStrategy'))
                except Exception:
                    allow = False
            if not allow:
                logger.error("Authoritative pipeline enforced: backtest strategy must be NeuralHybridStrategy or Unified3TFStrategy")
                return 1
        
        # Run backtest
        profile = str(getattr(args, 'profile', 'INTRADAY') or 'INTRADAY').upper()
        base_tf = str(getattr(args, 'base_tf', '') or '').upper()
        if not base_tf:
            try:
                if HAS_MTF:
                    base_tf = str(get_profile(profile).lower_tf.value)
            except Exception:
                base_tf = ''
        if not base_tf:
            if str(getattr(args, 'strategy', '') or '').lower().strip() == 'unified3tf':
                base_tf = {'SCALP': 'M5', 'INTRADAY': 'M15', 'SWING': 'H1'}.get(profile, 'M15')
            else:
                base_tf = 'M5' if profile == 'SCALP' else 'H1'

        # If user provides explicit MTF/HTF CSVs, run true 3TF backtest mode
        # without relying on BacktestBot resampling.
        if mtf_df is not None or htf_df is not None:
            from trading.backtest import BacktestExecutor, BacktestConfig

            try:
                prof = get_profile(profile) if HAS_MTF and get_profile else None
                if prof is None:
                    raise ValueError("MTF profiles not available")

                ltf_tf = str(prof.lower_tf.value).upper()
                primary_tf = str(prof.primary_tf.value).upper()
                higher_tf = str(prof.higher_tf.value).upper()

                def _prep_time_df(d0: pd.DataFrame) -> pd.DataFrame:
                    d0 = d0.copy()
                    d0.columns = [str(c).lower().strip() for c in d0.columns]
                    if 'time' in d0.columns:
                        d0['time'] = pd.to_datetime(d0['time'])
                        d0.sort_values('time', inplace=True)
                        try:
                            d0.drop_duplicates(subset=['time'], keep='last', inplace=True)
                        except Exception:
                            pass
                        d0.reset_index(drop=True, inplace=True)
                    return d0

                def _log_span(tag: str, d0: pd.DataFrame):
                    try:
                        if d0 is None or d0.empty or 'time' not in d0.columns:
                            return
                        t0 = pd.to_datetime(d0['time'].iloc[0])
                        t1 = pd.to_datetime(d0['time'].iloc[-1])
                        logger.info(f"{tag}: rows={len(d0)} span_days={(t1 - t0).days} start={t0} end={t1}")
                    except Exception:
                        pass

                df = _prep_time_df(df)
                if mtf_df is not None:
                    mtf_df = _prep_time_df(mtf_df)
                if htf_df is not None:
                    htf_df = _prep_time_df(htf_df)

                # Enforce strict 3TF alignment by trimming all TFs to the overlapping time window.
                dfs_for_window = [d for d in [df, mtf_df, htf_df] if d is not None and not d.empty and 'time' in d.columns]
                if len(dfs_for_window) >= 2:
                    starts = [pd.to_datetime(d['time'].iloc[0]) for d in dfs_for_window]
                    ends = [pd.to_datetime(d['time'].iloc[-1]) for d in dfs_for_window]
                    overlap_start = max(starts)
                    overlap_end = min(ends)

                    if overlap_end <= overlap_start:
                        logger.error(f"3TF alignment failed: no overlapping time window (start={overlap_start}, end={overlap_end})")
                        return 1

                    logger.info(
                        f"3TF alignment window: start={overlap_start} end={overlap_end} days={(overlap_end - overlap_start).days}"
                    )

                    def _trim(d0: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
                        if d0 is None or d0.empty or 'time' not in d0.columns:
                            return d0
                        mask = (d0['time'] >= overlap_start) & (d0['time'] <= overlap_end)
                        return d0.loc[mask].reset_index(drop=True)

                    df = _trim(df)
                    mtf_df = _trim(mtf_df)
                    htf_df = _trim(htf_df)

                    _log_span(f"Aligned {ltf_tf}", df)
                    if mtf_df is not None:
                        _log_span(f"Aligned {primary_tf}", mtf_df)
                    if htf_df is not None:
                        _log_span(f"Aligned {higher_tf}", htf_df)

                data_map = {
                    ltf_tf: df,
                }
                if mtf_df is not None:
                    data_map[primary_tf] = mtf_df
                if htf_df is not None:
                    data_map[higher_tf] = htf_df

                logger.info(f"3TF backtest enabled: LTF={ltf_tf}, MTF={primary_tf}, HTF={higher_tf}")

                class ThreeTFDataProvider:
                    def __init__(self, base_tf: str, data_by_tf: Dict[str, pd.DataFrame]):
                        self.base_tf = str(base_tf).upper()
                        self.data_by_tf = {str(k).upper(): v.copy() for k, v in data_by_tf.items() if v is not None}
                        self._time_index: Dict[str, Optional[pd.DatetimeIndex]] = {}
                        for k, d0 in list(self.data_by_tf.items()):
                            d0.columns = [str(c).lower().strip() for c in d0.columns]
                            if 'time' in d0.columns:
                                d0['time'] = pd.to_datetime(d0['time'])
                                d0.sort_values('time', inplace=True)
                                d0.reset_index(drop=True, inplace=True)
                                try:
                                    self._time_index[k] = pd.DatetimeIndex(d0['time'])
                                except Exception:
                                    self._time_index[k] = None
                            else:
                                self._time_index[k] = None
                            self.data_by_tf[k] = d0
                        self.current_idx = 0
                        self.current_time: Optional[datetime] = None

                    def get_ohlcv(self, symbol: str, timeframe: str = "", count: int = 200) -> pd.DataFrame:
                        tf = str(timeframe or self.base_tf).upper()
                        d0 = self.data_by_tf.get(tf)
                        if d0 is None or d0.empty:
                            return pd.DataFrame()

                        if self.current_time is not None and 'time' in d0.columns:
                            ti = self._time_index.get(tf)
                            if ti is not None:
                                try:
                                    end = int(ti.searchsorted(pd.Timestamp(self.current_time), side='right'))
                                    end = max(0, min(end, len(d0)))
                                    start = max(0, end - int(count))
                                    out = d0.iloc[start:end].copy()
                                except Exception:
                                    out = d0.tail(int(count)).copy()
                            else:
                                out = d0.tail(int(count)).copy()
                        else:
                            out = d0.tail(int(count)).copy()
                        out.columns = [str(c).lower().strip() for c in out.columns]
                        return out

                base_df = data_map.get(base_tf.upper(), df)
                provider = ThreeTFDataProvider(base_tf=base_tf, data_by_tf=data_map)
                executor = BacktestExecutor(config=BacktestConfig(initial_balance=args.balance))
                if getattr(strategy_cls, '__name__', '') == 'Unified3TFStrategy':
                    from strategies.unified_3tf_strategy import Unified3TFConfig
                    fast_bt = bool(getattr(args, 'fast_backtest', False))
                    if str(getattr(args, 'strategy', '') or '').lower().strip() == 'unified3tf':
                        fast_bt = True if fast_bt is False else fast_bt
                    strategy = strategy_cls(
                        config=Unified3TFConfig(profile=profile, symbol=str(getattr(args, 'symbol', 'EURUSD')), fast_backtest=fast_bt),
                        data_provider=provider,
                        executor=executor,
                    )
                elif getattr(strategy_cls, '__name__', '') == 'RiskManagedUnified3TFStrategy':
                    from strategies.unified_3tf_strategy import Unified3TFConfig
                    fast_bt = bool(getattr(args, 'fast_backtest', False))
                    if str(getattr(args, 'strategy', '') or '').lower().strip() == 'unified3tf':
                        fast_bt = True if fast_bt is False else fast_bt
                    strategy = strategy_cls(
                        config=Unified3TFConfig(profile=profile, symbol=str(getattr(args, 'symbol', 'EURUSD')), fast_backtest=fast_bt),
                        data_provider=provider,
                        executor=executor,
                        risk_percent=getattr(args, 'risk_percent', None),
                        cooldown_minutes=getattr(args, 'cooldown_minutes', None),
                    )
                else:
                    strategy = strategy_cls(data_provider=provider, executor=executor)
                if hasattr(strategy, 'initialize') and not getattr(strategy, '_initialized', False):
                    try:
                        strategy.initialize(starting_balance=float(args.balance))
                    except Exception:
                        pass

                window_size = 100
                signals_out = []
                rejection_counts: Dict[str, int] = {}
                rejection_examples: Dict[str, str] = {}

                def _bump_rej(stage: str, reason: str = ""):
                    k = str(stage or 'UNKNOWN')
                    rejection_counts[k] = int(rejection_counts.get(k, 0)) + 1
                    if reason and k not in rejection_examples:
                        rejection_examples[k] = str(reason)[:240]
                for i in range(window_size, len(base_df)):
                    provider.current_idx = i
                    try:
                        ts = None
                        if 'time' in base_df.columns:
                            ts = pd.to_datetime(base_df.iloc[i]['time']).to_pydatetime()
                        provider.current_time = ts
                    except Exception:
                        provider.current_time = None

                    ltf_window = provider.get_ohlcv(getattr(args, 'symbol', 'EURUSD'), timeframe=base_tf, count=window_size)
                    if ltf_window is None or ltf_window.empty:
                        continue
                    try:
                        exec_ts = ltf_window['time'].iloc[-1] if 'time' in ltf_window.columns else None
                        executor.update_price(float(ltf_window['close'].iloc[-1]), time=exec_ts)
                    except Exception:
                        executor.update_price(float(ltf_window['close'].iloc[-1]))

                    sig = strategy.on_bar(ltf_window)

                    if sig is None:
                        try:
                            stage = str(getattr(strategy, 'last_rejection_stage', '') or 'NO_SIGNAL')
                            reason = str(getattr(strategy, 'last_rejection_reason', '') or '')
                            _bump_rej(stage, reason)
                        except Exception:
                            _bump_rej('NO_SIGNAL', '')
                    try:
                        signals_out.append({'time': ltf_window['time'].iloc[-1], 'close': ltf_window['close'].iloc[-1], 'signal': sig})
                    except Exception:
                        signals_out.append({'time': None, 'close': None, 'signal': sig})

                try:
                    if hasattr(executor, 'close_all_positions'):
                        executor.close_all_positions()
                except Exception:
                    pass

                results = {
                    'trades': executor.get_trade_history(),
                    'final_balance': executor.balance,
                    'signals': signals_out,
                    'rejections': dict(sorted(rejection_counts.items(), key=lambda kv: kv[1], reverse=True)),
                    'rejection_examples': rejection_examples,
                }
            except Exception as e:
                logger.error(f"3TF backtest error: {e}", exc_info=True)
                return 1
        else:
            bot = BacktestBot(
                data=df,
                strategy_class=strategy_cls,
                initial_balance=args.balance,
                profile=profile,
                base_timeframe=base_tf,
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

        if isinstance(results, dict) and results.get('rejections'):
            print("\n  Rejection Breakdown:")
            for k, v in results['rejections'].items():
                print(f"  {k}: {v}")
            examples = results.get('rejection_examples') or {}
            if examples:
                print("\n  Rejection Examples:")
                for k, v in examples.items():
                    print(f"  {k}: {v}")
        
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

        if bool(getattr(args, 'export_csv', False)):
            try:
                from utils.config import settings
            except Exception:
                settings = None

            try:
                export_root = None
                if getattr(args, 'export_dir', None):
                    export_root = Path(str(args.export_dir))
                else:
                    export_root = Path(getattr(settings, 'ASSETS_DIR', Path('.'))) / 'backtests'
                export_root.mkdir(parents=True, exist_ok=True)

                stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                sym = str(getattr(args, 'symbol', 'EURUSD') or 'EURUSD').upper()
                prof = str(getattr(args, 'profile', 'INTRADAY') or 'INTRADAY').upper()
                strat = str(getattr(args, 'strategy', 'strategy') or 'strategy').lower().strip()

                trades = results.get('trades') if isinstance(results, dict) else None
                if trades:
                    df_trades = pd.DataFrame(trades)
                else:
                    df_trades = pd.DataFrame([])

                if not df_trades.empty:
                    for col in list(df_trades.columns):
                        try:
                            if df_trades[col].dtype == 'datetime64[ns]':
                                df_trades[col] = df_trades[col].astype(str)
                        except Exception:
                            pass

                out_trades = export_root / f"backtest_{strat}_{sym}_{prof}_trades_{stamp}.csv"
                df_trades.to_csv(out_trades, index=False)

                try:
                    period_start = df['time'].iloc[0] if isinstance(df, pd.DataFrame) and 'time' in df.columns and len(df) else None
                    period_end = df['time'].iloc[-1] if isinstance(df, pd.DataFrame) and 'time' in df.columns and len(df) else None
                except Exception:
                    period_start, period_end = None, None

                summary = {
                    'strategy': strat,
                    'symbol': sym,
                    'profile': prof,
                    'period_start': str(period_start) if period_start is not None else '',
                    'period_end': str(period_end) if period_end is not None else '',
                    'candles': int(len(df)) if isinstance(df, pd.DataFrame) else 0,
                    'initial_balance': float(args.balance),
                    'final_balance': float(results.get('final_balance')) if isinstance(results, dict) else float('nan'),
                    'pnl': float(pnl),
                    'pnl_pct': float(pnl_pct),
                    'total_trades': int(len(trades)) if trades else 0,
                }
                out_summary = export_root / f"backtest_{strat}_{sym}_{prof}_summary_{stamp}.csv"
                pd.DataFrame([summary]).to_csv(out_summary, index=False)

                logger.info(f"CSV export written: {out_trades}")
                logger.info(f"CSV export written: {out_summary}")
            except Exception as e:
                logger.warning(f"CSV export failed: {e}")
        
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
        if model == "walk-forward" or model == "wf":
            # Walk-forward training for MH-TCN
            from training.walk_forward_trainer import WalkForwardTrainer, WalkForwardConfig
            
            if not args.data:
                logger.error("Walk-forward training requires --data path to OHLCV CSV")
                return 1
            
            config = WalkForwardConfig(
                profile=str(getattr(args, 'profile', 'INTRADAY') or 'INTRADAY').upper(),
                epochs_per_fold=args.epochs,
                output_dir=args.save_dir or 'models/weights/walk_forward',
            )
            
            import pandas as pd
            df = pd.read_csv(args.data)
            
            trainer = WalkForwardTrainer(config)
            results = trainer.run(df)
            
            best_model = trainer.get_best_model_path()
            logger.info(f"Walk-forward training complete. Best model: {best_model}")
            logger.info(f"Average F1: {sum(r.test_f1 for r in results) / len(results):.4f}")
        
        elif model == "mhtcn" or model == "tcn":
            # Full MH-TCN training with all heads (direction, volatility, quantiles, outcomes)
            from training.train_mhtcn import train_mhtcn
            
            if not args.data:
                logger.error("MH-TCN training requires --data path to OHLCV CSV")
                return 1
            
            profile = str(getattr(args, 'profile', 'INTRADAY') or 'INTRADAY').upper()
            use_triple_barrier = not getattr(args, 'no_triple_barrier', False)
            
            results = train_mhtcn(
                data_path=args.data,
                profile=profile,
                epochs=args.epochs,
                batch_size=getattr(args, 'batch_size', 64) or 64,
                learning_rate=getattr(args, 'learning_rate', 1e-3) or 1e-3,
                output_dir=args.save_dir or 'models/weights',
                use_triple_barrier=use_triple_barrier,
            )
            
            logger.info(f"Training complete. Model saved to: {results['model_path']}")
            logger.info(f"Test accuracy: {results['test_metrics'].get('direction_accuracy', 0):.4f}")
        
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
            logger.info("Available models: walk-forward, wf, mhtcn, tcn, trend")
            logger.info("")
            logger.info("Recommended: python main.py train walk-forward --data <path> --profile INTRADAY")
            return 1
        
        logger.info("✅ Training complete!")
        return 0
        
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        return 1


def cmd_predict(args, logger: logging.Logger):
    """Handle prediction command.

    In addition to classic direction outputs (BUY/SELL/HOLD), this command
    also prints TP-before-SL probabilities (p_long/p_short) when available
    from the Phase-1 multi-head model.
    """
    logger.info("🔮 Running Prediction")
    
    checker = SystemChecker(logger)
    if not checker.run_all_checks("predict"):
        return 1
    
    try:
        import pandas as pd
        import torch
        from inference.predictor import HybridPredictor, RiskAwareTCNPredictor, PredictorConfig
        from utils.config import settings

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
            default_weights = str(Path(getattr(settings, 'WEIGHTS_DIR', 'models/weights')) / 'multihead_tcn_INTRADAY.pth')
            predictor = RiskAwareTCNPredictor(
                weights_path=args.weights or default_weights
            )
        else:
            weights_dir = Path(getattr(settings, 'WEIGHTS_DIR', 'models/weights'))
            profile = str(getattr(args, 'profile', 'INTRADAY') or 'INTRADAY').upper()
            cfg = PredictorConfig(
                profile=profile,
                use_vision=False,
                use_price_action=False,
            )
            tcn_weights = args.weights or str(weights_dir / 'multihead_tcn_INTRADAY.pth')
            predictor = HybridPredictor(config=cfg, tcn_weights=tcn_weights)

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

        if getattr(result, 'p_long', None) is not None and getattr(result, 'p_short', None) is not None:
            print(f"\n  TP-before-SL Probabilities:")
            print(f"    p_long : {result.p_long:.2%}")
            print(f"    p_short: {result.p_short:.2%}")
        
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
  %(prog)s fetch-data --symbol EURUSD --timeframes H1,H4 --bars 20000
  %(prog)s backtest --data data/EURUSD_H1.csv
  %(prog)s alpha-backtest --data data/EURUSD_H1.csv --layer alpha_only
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
    bt_parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log to file (can also be passed before the subcommand)",
    )
    bt_parser.add_argument("--data", required=True, help="Path to OHLCV CSV file")
    bt_parser.add_argument(
        "--data-mtf",
        type=str,
        default=None,
        help="Optional MTF CSV path (profile primary TF, e.g. M15 for SCALP, H1 for INTRADAY, H4 for SWING)",
    )
    bt_parser.add_argument(
        "--data-htf",
        type=str,
        default=None,
        help="Optional HTF CSV path (profile higher TF, e.g. H1 for SCALP, H4 for INTRADAY, D1 for SWING)",
    )
    bt_parser.add_argument(
        "--fast-backtest",
        action="store_true",
        help="Speed up backtest (unified3tf): cache HTF/MTF evals, use key features only, and skip swing extraction.",
    )
    bt_parser.add_argument(
        "--symbol",
        default="EURUSD",
        help="Trading symbol (used by strategies)",
    )
    bt_parser.add_argument("--strategy", default="neural", help="Strategy (neural, tcn)")
    bt_parser.add_argument("--balance", type=float, default=10000.0, help="Initial balance")
    bt_parser.add_argument(
        "--profile",
        type=str,
        default="INTRADAY",
        help="Trading profile (SCALP, INTRADAY, SWING) used for risk/MTF settings",
    )
    bt_parser.add_argument(
        "--base-tf",
        type=str,
        default="",
        help="Base timeframe of the input CSV (e.g. M5 for scalping); higher TFs are resampled",
    )
    bt_parser.add_argument(
        "--max-bars",
        type=int,
        help="Optional cap on number of candles (use most recent N) for faster backtests",
    )
    bt_parser.add_argument("--no-vision", action="store_true", help="Disable ViT vision model for backtest")
    bt_parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO model for backtest")
    bt_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Decision confidence threshold (lower to allow more trades)",
    )
    bt_parser.add_argument(
        "--min-mtf-alignment",
        type=float,
        default=None,
        help="Minimum multi-timeframe alignment (0-1). If omitted, uses strategy profile default.",
    )
    bt_parser.add_argument(
        "--risk-percent",
        type=float,
        default=None,
        help="Optional risk per trade (% of balance) override for unified3tf.",
    )
    bt_parser.add_argument(
        "--cooldown-minutes",
        type=float,
        default=None,
        help="Optional cooldown in minutes after entry/exit for unified3tf.",
    )
    bt_parser.add_argument("--output", type=str, help="Save results to JSON file")
    bt_parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export backtest results to CSV (trade history + summary)",
    )
    bt_parser.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="Optional directory for CSV export (defaults to settings.ASSETS_DIR/backtests)",
    )

    # -------------------------------------------------------------------------
    # ALPHA FACTORY BACKTEST (SEPARATE)
    # -------------------------------------------------------------------------
    ab_parser = subparsers.add_parser(
        "alpha-backtest",
        help="Run Alpha Factory backtest on historical data",
        description="Run Alpha Factory decision engine backtest separately from the ML strategy backtest",
    )
    ab_parser.add_argument("--data", required=True, help="Path to OHLCV CSV file")
    ab_parser.add_argument(
        "--engine",
        type=str,
        default="decision",
        help="Alpha backtest engine (decision)",
    )
    ab_parser.add_argument(
        "--layer",
        type=str,
        default="alpha_only",
        help="Layer to run (alpha_only, alpha_execution, alpha_risk, full_system)",
    )
    ab_parser.add_argument(
        "--window",
        type=int,
        default=300,
        help="Rolling window size (bars) for engine=decision",
    )
    ab_parser.add_argument(
        "--disable-signal-quality",
        action="store_true",
        help="Disable signal quality optimizer in engine=decision",
    )
    ab_parser.add_argument(
        "--max-bars",
        type=int,
        help="Optional cap on number of candles (use most recent N) for faster backtests",
    )

    # -------------------------------------------------------------------------
    # FETCH-DATA (MT5 -> external assets)
    # -------------------------------------------------------------------------
    fd_parser = subparsers.add_parser(
        "fetch-data",
        help="Fetch historical data from MT5 and save to external assets folder",
        description="Fetch OHLCV from MT5 and write CSVs under settings.ASSETS_DIR/data/mt5 (outside repo)",
    )
    fd_parser.add_argument("--symbol", default="EURUSD", help="Symbol to fetch")
    fd_parser.add_argument(
        "--timeframes",
        type=str,
        default="H1",
        help="Comma-separated timeframes (e.g. M15,H1,H4)",
    )
    fd_parser.add_argument("--bars", type=int, default=50000, help="Number of bars per timeframe")
    
    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model",
        description="Train ML models (MH-TCN with walk-forward validation, Trend classifier)",
    )
    train_parser.add_argument(
        "model",
        choices=["walk-forward", "wf", "mhtcn", "tcn", "trend"],
        help="Model to train (recommended: walk-forward for MH-TCN with proper validation)",
    )
    train_parser.add_argument("--data", type=str, help="Path to training data CSV")
    train_parser.add_argument("--data-dir", type=str, help="Path to dataset directory")
    train_parser.add_argument("--save-dir", type=str, help="Directory to save weights")
    train_parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--seq-len", type=int, default=60, help="Sequence length (TCN)")
    train_parser.add_argument("--cache-path", type=str, help="Feature cache path")
    train_parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (trend)")
    train_parser.add_argument("--profile", type=str, default="INTRADAY", 
                              choices=["SCALP", "INTRADAY", "SWING"],
                              help="Trading profile for MH-TCN training")
    train_parser.add_argument("--no-triple-barrier", action="store_true",
                              help="Disable triple-barrier outcome labels for MH-TCN training")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3,
                              help="Learning rate (alias for --lr)")
    
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
        "alpha-backtest": cmd_alpha_backtest,
        "fetch-data": cmd_fetch_data,
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
