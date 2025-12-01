# trading/bot.py
import time
import logging
from trading.mt5_connector import MT5Connector
from trading.risk_manager import RiskManager
from strategies.neural_hybrid import NeuralHybridStrategy
from utils.config import settings

class TradingBot:
    def __init__(self):
        self.connector = MT5Connector()
        # Initialize dependencies
        account_info = 10000 # Default/Placeholder, will be updated on connect
        self.risk_manager = RiskManager(account_balance=account_info) 
        
        # Inject dependencies into Strategy
        # Note: In a real scenario, you'd fetch balance after connect
        self.strategy = NeuralHybridStrategy(
            data_provider=self.connector, 
            executor=self.connector,
            risk_manager=self.risk_manager
        )

    def run(self):
        if not self.connector.connect():
            return

        logging.info(f"Bot started on {settings.SYMBOL} | Strategy: NeuralHybrid")
        
        # Update Risk Manager with actual balance
        import MetaTrader5 as mt5
        acc = mt5.account_info()
        if acc:
            self.risk_manager.starting_balance = acc.balance
            self.risk_manager.daily_start_balance = acc.balance

        while True:
            try:
                # 1. Fetch Data
                df = self.connector.get_data(n=100)
                if df.empty:
                    time.sleep(5)
                    continue

                # 2. Pass Data to Strategy
                # The strategy handles Prediction -> Signal -> Risk -> Execution internally
                self.strategy.on_bar(df)
                
                # Sleep to avoid spamming (approx check every 10s or align with candle close)
                time.sleep(10)

            except Exception as e:
                logging.error(f"Loop Error: {e}", exc_info=True)
                time.sleep(10)