#main.py

import logging
import sys
from trading.bot import TradingBot

# Configure Logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_session.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__ == "__main__":
    print("🚀 Starting PyForex Bot (Local Mode)...")
    try:
        bot = TradingBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user.")
    except Exception as e:
        logging.critical(f"🔥 Fatal Crash: {e}", exc_info=True)