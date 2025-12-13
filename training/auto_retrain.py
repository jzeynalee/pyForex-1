# training/auto_retrain.py
"""
Automatic retraining job for TCN models.
Note: This is a simplified version. For production use, consider using
the ml/retraining_pipeline.py which has drift detection and scheduling.
"""
import sys
import os
from pathlib import Path
import logging
import MetaTrader5 as mt5  # Needed for timeframe constants

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading.mt5_connector import MT5Connector
from training.train_tcn_enhanced import main as train_tcn_enhanced

import warnings
# Silence all FutureWarnings (pandas updates, etc.)
warnings.simplefilter(action='ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

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
    DOWNLOAD_COUNT = 100000
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

        # Create args for train_tcn_enhanced
        class Args:
            data = str(csv_path)
            epochs = 50
            batch_size = 64
            lr = 1e-3
            seq_len = 60
            save_dir = "models/weights"
            device = "auto"
            profile = "INTRADAY"

        args = Args()
        train_tcn_enhanced(args)
        logging.info("✅ Retraining Complete. TCN model updated.")

    except Exception as e:
        logging.error(f"❌ Training Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    auto_retrain_job()