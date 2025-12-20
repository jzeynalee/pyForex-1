# training/auto_retrain.py
"""
Enhanced Automatic Multi-Timeframe Retraining Commander.

Integrates with main.py and ml module to:
- Check for missing weights per timeframe
- Train models for specific timeframes  
- Support all trading styles (SCALP, INTRADAY, SWING)
- Trigger training before system starts when needed

Usage:
    from training.auto_retrain import check_and_train_missing_models
    check_and_train_missing_models(['SCALP', 'INTRADAY', 'SWING'])
"""
import sys
import os
from pathlib import Path
import logging

# Color-coded logging
try:
    from colorama import init, Fore, Style
    init()  # Initialize colorama
    COLORS = {
        'SUCCESS': Fore.GREEN,
        'WARNING': Fore.YELLOW, 
        'ERROR': Fore.RED,
        'INFO': Fore.CYAN,
        'TRAINING': Fore.MAGENTA,
        'RESET': Style.RESET_ALL
    }
    def color_log(msg: str, color: str = 'INFO') -> str:
        return f"{COLORS.get(color, '')}{msg}{COLORS['RESET']}"
except ImportError:
    # Fallback if colorama not available
    COLORS = {}
    def color_log(msg: str, color: str = 'INFO') -> str:
        return msg

# Try to import MetaTrader5, use mock if not available
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None
    logging.warning(color_log("MetaTrader5 not available - using mock mode for testing", 'WARNING'))

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.mt5_connector import MT5Connector
from training.train_tcn_enhanced import main as train_tcn_enhanced
from utils.mtf_config import PROFILES, get_profile

import warnings
# Silence all FutureWarnings (pandas updates, etc.)
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

# Enhanced MTF training functions
def check_missing_weights(styles: list[str], symbol: str = "EURUSD") -> dict:
    """
    Check which weights are missing for given trading styles.
    
    Args:
        styles: List of trading styles ['SCALP', 'INTRADAY', 'SWING']
        symbol: Trading symbol (default: EURUSD)
    
    Returns:
        Dict mapping style -> list of missing timeframes
    """
    weights_dir = Path("models/weights")
    missing = {}
    
    for style in styles:
        profile = get_profile(style)
        missing_tfs = []
        
        for tf in profile.timeframe_strings:
            # Check for style-specific weights: {style}_{tf}_best.pt
            weight_file = weights_dir / f"{style.lower()}_{tf.lower()}_best.pt"
            if not weight_file.exists():
                missing_tfs.append(tf)
        
        # Also check for generic weights as fallback
        generic_weight = weights_dir / f"tcn_best.pt"
        if not generic_weight.exists():
            if generic_weight not in missing_tfs:
                missing_tfs.append("GENERIC")
        
        if missing_tfs:
            missing[style] = missing_tfs
    
    return missing

def train_style_models(styles: list[str], symbol: str = "EURUSD", download_count: int = 1000000) -> bool:
    """
    Train models for specified trading styles and their timeframes.
    
    Args:
        styles: List of trading styles to train
        symbol: Trading symbol
        download_count: Number of candles to download per timeframe
    
    Returns:
        True if all training succeeded, False otherwise
    """
    connector = MT5Connector()
    if not connector.connect():
        logging.error("❌ Could not connect to MT5.")
        return False
    
    success = True
    
    for style in styles:
        profile = get_profile(style)
        logging.info(f"🎯 Training models for {style} style...")
        
        for tf in profile.timeframe_strings:
            # Skip if model already exists
            weight_file = Path(f"models/weights/{style.lower()}_{tf.lower()}_best.pt")
            if weight_file.exists():
                logging.info(f"✅ {style}_{tf} model already exists, skipping...")
                continue
            
            try:
                # Download data for this timeframe
                logging.info(f"⬇️ Downloading {download_count} {tf} candles for {symbol}...")
                df = connector.get_data(symbol=symbol, n=download_count, timeframe=tf)
                
                if df.empty:
                    logging.error(f"❌ No data received for {tf}.")
                    success = False
                    continue
                
                # Save data
                data_dir = Path("data/raw")
                data_dir.mkdir(parents=True, exist_ok=True)
                csv_path = data_dir / f"{symbol}_{tf}_latest.csv"
                df.to_csv(csv_path, index=False)
                logging.info(f"💾 Saved {len(df)} rows to {csv_path}")
                
                # Train model for this timeframe
                train_timeframe_model(csv_path, style, tf)
                
            except Exception as e:
                logging.error(f"❌ Failed to train {style}_{tf}: {e}")
                success = False
    
    connector.disconnect()
    return success

def train_timeframe_model(csv_path: Path, style: str, timeframe: str):
    """Train a TCN model for a specific timeframe."""
    try:
        logging.info(f"🧠 Training {style}_{timeframe} model...")
        
        # Create args for train_tcn_enhanced
        class Args:
            data = str(csv_path)
            epochs = 30  # Reduced for faster training
            batch_size = 64
            lr = 1e-3
            seq_len = 60
            save_dir = "models/weights"
            device = "auto"
            profile = style
            features = None
            skip_feature_selection = False
            n_features = 25
            hidden_dim = 64
            num_layers = 5
            dropout = 0.2
            threshold = 0.05
            patience = 8
            use_cosine = False
            no_onecycle = False
            name = f"{style.lower()}_{timeframe.lower()}_tcn"
        
        args = Args()
        train_tcn_enhanced(args)
        
        # Rename output file to match expected format
        original_name = f"models/weights/{args.name}_best.pt"
        expected_name = f"models/weights/{style.lower()}_{timeframe.lower()}_best.pt"
        
        if Path(original_name).exists():
            Path(original_name).rename(expected_name)
            logging.info(f"✅ {style}_{timeframe} model trained and saved.")
        else:
            logging.warning(f"⚠️ Expected output file not found: {original_name}")
            
    except Exception as e:
        logging.error(f"❌ Training failed for {style}_{timeframe}: {e}")
        import traceback
        traceback.print_exc()

def check_and_train_missing_models(styles: list[str] = None, symbol: str = "EURUSD") -> bool:
    """
    Main entry point: Check for missing weights and train if needed.
    
    Args:
        styles: List of styles to check/train (default: all styles)
        symbol: Trading symbol
    
    Returns:
        True if system is ready (all models exist), False otherwise
    """
    if styles is None:
        styles = ['SCALP', 'INTRADAY', 'SWING']
    
    logging.info("🔍 Checking for missing model weights...")
    missing = check_missing_weights(styles, symbol)
    
    if not missing:
        logging.info("✅ All required models found. System ready.")
        return True
    
    # Report what's missing with summary table
    print("\n" + "=" * 60)
    print("🚨 MISSING MODEL WEIGHTS DETECTED")
    print("=" * 60)
    
    # Create summary table
    all_timeframes = ['M5', 'M15', 'H1', 'H4', 'D1']
    
    print("\n📊 MODEL STATUS SUMMARY:")
    print(f"{'Style':<12} {'M5':<6} {'M15':<6} {'H1':<6} {'H4':<6} {'D1':<6}")
    print("-" * 48)
    
    for style in ['SCALP', 'INTRADAY', 'SWING']:
        if style not in missing:
            status = "✅"
        else:
            status = "❌"
        
        row = [f"{style}:"]
        for tf in all_timeframes:
            if style in missing and tf in missing[style]:
                row.append("❌")
            elif tf in get_profile(style).timeframe_strings:
                row.append("✅")
            else:
                row.append("—")
        
        print(f"{row[0]:<12} {row[1]:<6} {row[2]:<6} {row[3]:<6} {row[4]:<6} {row[5]:<6}")
    
    print(f"\n🔄 Starting automatic training for missing models...")
    print("=" * 60)
    
    # Train missing models
    success = train_style_models(styles, symbol)
    
    if success:
        print("\n✅ All missing models trained successfully!")
        print("🚀 System is now ready for trading.")
        return True
    else:
        print("\n❌ Some models failed to train.")
        print("⚠️ System may not be fully functional.")
        return False

def auto_retrain_job():
    print("=" * 70)
    print("🔄 STARTING TCN RETRAINING JOB (BIG DATA MODE)")
    print("=" * 70)

    # 1. Fetch Latest Data
    connector = MT5Connector()
    if not connector.connect():
        logging.error("❌ Could not connect to MT5.")
        return

    # SETTINGS: Maximize this based on your RAM and Broker limits
    # M15 * 100,000 = ~3 years of data
    # M5  * 100,000 = ~1 year of data
    # M1  * 100,000 = ~3 months of data
    DOWNLOAD_COUNT = 8000000
    TIMEFRAME = "M15"  # <--- CHANGE THIS to your desired timeframe
    SYMBOL = "EURUSD"

    logging.info(f"⬇️ Downloading latest {DOWNLOAD_COUNT} candles for {SYMBOL}...")

    # We pass the timeframe explicitly if your get_data supports it,
    # otherwise ensure MT5Connector defaults to the right one.
    df = connector.get_data(symbol=SYMBOL, n=DOWNLOAD_COUNT, timeframe=TIMEFRAME)

    if df.empty:
        logging.error("❌ No data received.")
        return

    # 2. Validation Check
    actual_count = len(df)
    logging.info(f"✅ Downloaded {actual_count} rows.")

    if actual_count < 10000:
        logging.warning("⚠️ WARNING: Dataset is very small (< 10k). Model may overfit.")
        logging.warning("   -> Check MT5 Terminal: Tools > Options > Charts > Max bars in chart")

    # 3. Save Data
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "eurusd_latest.csv"

    df.to_csv(csv_path, index=False)
    logging.info(f"💾 Saved to {csv_path}")

    # 4. Retrain TCN with Optimized Parameters
    try:
        logging.info("🧠 Starting TCN Training...")

        # Call train_tcn_enhanced with individual parameters as expected by tests
        train_tcn_enhanced(
            data=str(csv_path),
            epochs=50,
            batch_size=64,
            lr=1e-3,
            seq_len=60,
            save_dir="models/weights",
            device="auto",
            profile="INTRADAY",
            features=None,
            skip_feature_selection=False,
            n_features=25,
            hidden_dim=64,
            num_layers=5,
            dropout=0.2,
            threshold=0.05,
            patience=10,
            use_cosine=False,
            no_onecycle=False,
            name="tcn_enhanced"
        )
        logging.info("✅ Retraining Complete. TCN model updated.")

    except Exception as e:
        logging.error(f"❌ Training Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    auto_retrain_job()