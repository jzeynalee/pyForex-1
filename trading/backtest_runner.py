
import logging
import sys
import os
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from typing import Optional, List, Dict

from trading.bot import TradingBot, BotConfig
from trading.backtest_connector import BacktestConnector, BacktestConfig
from trading.data_loader import DataLoader
from strategies.neural_hybrid import NeuralHybridStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backtest_runner.log")
    ]
)
logger = logging.getLogger(__name__)

class BacktestRunner:
    def __init__(self, data_path: str, symbol: str = "EURUSD", initial_balance: float = 10000.0):
        self.data_path = data_path
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.loader = DataLoader()
        
    def run(self):
        logger.info("="*60)
        logger.info(f"STARTING BACKTEST: {self.symbol} | File: {self.data_path}")
        logger.info("="*60)
        
        # 1. Load Data
        try:
            df = self.loader.load_csv(self.data_path)
        except Exception as e:
            logger.warning(f"Could not load data: {e}. Generating synthetic data.")
            df = self.loader.generate_synthetic_data(n=2000)
            
        # 2. Validate Data
        val_res = self.loader.validate_data(df)
        if not val_res.passed:
            logger.warning("Data Validation Issues Found:")
            for issue in val_res.issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("Data Validation Passed")
            
        logger.info(f"Loaded {len(df)} bars. Range: {val_res.stats.get('start')} - {val_res.stats.get('end')}")
        
        # 3. Setup Connector
        bt_config = BacktestConfig(
            initial_balance=self.initial_balance,
            symbol=self.symbol,
            spread_pips=1.0,
            commission_per_lot=7.0
        )
        connector = BacktestConnector(df, bt_config)
        
        # 4. Setup Bot
        bot_config = BotConfig(
            symbol=self.symbol,
            timeframe="H1", # Should infer from data
            data_window=100,
            tick_interval=0 # No sleep in backtest
        )
        
        # Use NeuralHybridStrategy or simpler one for testing
        strategy_class = NeuralHybridStrategy
        
        bot = TradingBot(
            config=bot_config,
            strategy_class=strategy_class,
            connector=connector
        )
        
        # Explicitly initialize strategy for backtesting
        logger.info("Initializing strategy for backtest...")
        if hasattr(bot.strategy, 'initialize'):
            success = bot.strategy.initialize(starting_balance=self.initial_balance)
            if not success:
                logger.error("Failed to initialize strategy. Aborting.")
                return
        
        # 5. Run Loop
        logger.info("Starting simulation loop...")
        start_time = time.time()
        
        bars_processed = 0
        equity_curve = []
        
        while connector.next_bar():
            bot.step()
            
            # Record equity
            equity_curve.append({
                'time': connector.current_time,
                'equity': connector.equity,
                'balance': connector.balance
            })
            
            bars_processed += 1
            if bars_processed % 1000 == 0:
                logger.info(f"Processed {bars_processed} bars...")
                
        duration = time.time() - start_time
        logger.info(f"Simulation complete in {duration:.2f}s ({bars_processed/duration:.1f} bars/s)")
        
        # 6. Report
        self._generate_report(connector, equity_curve)

    def _generate_report(self, connector: BacktestConnector, equity_curve: List[Dict]):
        logger.info("\n" + "="*60)
        logger.info("BACKTEST REPORT")
        logger.info("="*60)
        
        account = connector.get_account_info()
        
        # Calculate Equity Metrics
        df_equity = pd.DataFrame(equity_curve)
        if not df_equity.empty:
            df_equity['returns'] = df_equity['equity'].pct_change()
            
            # Max Drawdown
            df_equity['peak'] = df_equity['equity'].cummax()
            df_equity['dd'] = (df_equity['equity'] - df_equity['peak']) / df_equity['peak']
            max_dd = df_equity['dd'].min()
            max_dd_amount = (df_equity['peak'] * df_equity['dd']).min()
            
            # Sharpe Ratio (assuming hourly data, annualized)
            risk_free_rate = 0.02
            mean_return = df_equity['returns'].mean() * 252 * 24 # Annualized hourly
            std_return = df_equity['returns'].std() * np.sqrt(252 * 24)
            sharpe = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0.0
            
            final_return_pct = (account.profit / self.initial_balance) * 100
        else:
            max_dd = 0.0
            sharpe = 0.0
            final_return_pct = 0.0
            
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Final Balance:   ${account.balance:,.2f}")
        logger.info(f"Final Equity:    ${account.equity:,.2f}")
        logger.info(f"Total Return:    ${account.profit:,.2f} ({final_return_pct:.2f}%)")
        logger.info(f"Max Drawdown:    {max_dd:.2%} (${max_dd_amount:,.2f})")
        logger.info(f"Sharpe Ratio:    {sharpe:.2f}")
        
        history = connector.history
        logger.info(f"Total Trades:    {len(history)}")
        
        if history:
            wins = [t for t in history if t['final_pnl'] > 0]
            losses = [t for t in history if t['final_pnl'] <= 0]
            win_rate = len(wins) / len(history) if history else 0
            
            logger.info(f"Win Rate:        {win_rate:.2%}")
            logger.info(f"Winning Trades:  {len(wins)}")
            logger.info(f"Losing Trades:   {len(losses)}")
            
            total_profit = sum(t['final_pnl'] for t in wins)
            total_loss = abs(sum(t['final_pnl'] for t in losses))
            pf = total_profit / total_loss if total_loss > 0 else float('inf')
            
            logger.info(f"Profit Factor:   {pf:.2f}")
            
        logger.info("="*60)


import time

if __name__ == "__main__":
    # Default to a sample file or generation
    import argparse
    parser = argparse.ArgumentParser(description='Run PyForex Backtest')
    parser.add_argument('--data', type=str, default='data/EURUSD_H1.csv', help='Path to CSV data file')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Symbol to trade')
    
    args = parser.parse_args()
    
    runner = BacktestRunner(args.data, args.symbol)
    runner.run()
