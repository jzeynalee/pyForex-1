import time
import logging
from trading.mt5_connector import MT5Connector
from inference.predictor import HybridPredictor
from trading.risk_manager import RiskManager
from trading.signal_engine import generate_signal
from utils.config import settings
import MetaTrader5 as mt5

class TradingBot:
    def __init__(self):
        self.connector = MT5Connector()
        self.predictor = HybridPredictor()
        self.risk_manager = RiskManager()
        self.last_candle_time = None

    def run(self):
        if not self.connector.connect():
            return

        logging.info(f"Bot started on {settings.SYMBOL}")

        while True:
            try:
                # 1. Fetch Data
                df = self.connector.get_data(n=100)
                if df.empty:
                    time.sleep(5)
                    continue

                # 2. Check for New Candle
                current_time = df['time'].iloc[-1]
                if self.last_candle_time == current_time:
                    time.sleep(1)
                    continue
                
                self.last_candle_time = current_time
                logging.info(f"Analyzing candle: {current_time}")

                # 3. Get Prediction
                probs = self.predictor.predict(df)
                signal = generate_signal(probs, threshold=settings.CONFIDENCE_THRESHOLD)

                # 4. Execute Trade
                if signal != "NO_TRADE":
                    self._execute_trade(signal, df)

            except Exception as e:
                logging.error(f"Critical Error: {e}", exc_info=True)
                time.sleep(10)

    def _execute_trade(self, signal, df):
        # Calculate Risk
        balance = mt5.account_info().balance
        current_price = mt5.symbol_info_tick(settings.SYMBOL).ask # Simplified
        atr = self.risk_manager.calculate_volatility(df)
        
        vol, sl, tp = self.risk_manager.get_trade_params(balance, current_price, atr, signal)
        
        # Send Order
        result = self.connector.execute_order(signal, vol, sl, tp)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Trade Open: {signal} | Vol: {vol}")
        else:
            logging.error(f"Trade Failed: {result.comment}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    bot = TradingBot()
    bot.run()