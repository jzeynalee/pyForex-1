
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading.backtest_runner import BacktestRunner
from trading.bot import BotConfig, TradingBot
from trading.backtest_connector import BacktestConnector, BacktestConfig
from trading.data_loader import DataLoader
from strategies.neural_hybrid import NeuralHybridStrategy, StrategyConfig
from config.prop_firm_config import get_prop_firm_config, PropFirm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("backtest_optimized.log")
    ]
)
logger = logging.getLogger(__name__)

class OptimizedBacktestRunner(BacktestRunner):
    def __init__(self, data_path: str, symbol: str = "EURUSD", profile: str = "SCALP"):
        super().__init__(data_path, symbol)
        self.profile = profile
        
    def run(self):
        logger.info("="*60)
        logger.info(f"STARTING OPTIMIZED BACKTEST: {self.symbol} | Profile: {self.profile}")
        logger.info("="*60)
        
        # 1. Load Data (M5 base data for SCALP)
        try:
            # Check if file exists
            if not os.path.exists(self.data_path):
                logger.error(f"Data file not found: {self.data_path}")
                return
                
            df = self.loader.load_csv(self.data_path)
            # Ensure index is datetime and sorted
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.set_index('time')
            df = df.sort_index()
            
            # Limit to recent data for speed
            if len(df) > 20000:
                logger.info(f"Limiting backtest to last 20,000 bars (from {len(df)})")
                df = df.iloc[-20000:]
            
        except Exception as e:
            logger.error(f"Could not load data: {e}", exc_info=True)
            return

        logger.info(f"Loaded {len(df)} bars. Range: {df.index[0]} - {df.index[-1]}")
        
        # Suppress high-frequency logs
        logging.getLogger("inference.predictor").setLevel(logging.WARNING)
        logging.getLogger("models.tcn").setLevel(logging.WARNING)
        
        # 2. Setup Prop Firm Config (Stable Profitability)
        # Using FTMO rules as a baseline for "stable" trading constraints
        prop_config = get_prop_firm_config('FTMO', account_size=100000, phase='evaluation', conservative=True)
        
        # 3. Setup Connector
        bt_config = BacktestConfig(
            initial_balance=prop_config.account_size,
            symbol=self.symbol,
            spread_pips=1.0, # Typical EURUSD spread
            commission_per_lot=7.0,
            slippage_pips=0.1, # Add some realism
        )
        connector = BacktestConnector(df, bt_config)
        
        # 4. Strategy Config
        # Pointing to the best weights found in the directory listing
        strategy_config = StrategyConfig(
            profile=self.profile,
            symbol=self.symbol,
            # Weights
            tcn_weights='models/weights/tcn_best.pt', # Or models/weights/scalp_m5_best.pt if available
            vit_weights=None, # Disable for speed unless essential, or point to models/vit/vit_SCALP.pth
            yolo_weights=None,
            fusion_weights=None, # 'models/weights/fusion_best.pt'
            
            # Risk Management (Strict)
            base_risk_percent=prop_config.max_risk_per_trade_pct, # 0.5%
            max_daily_loss_percent=prop_config.effective_daily_loss_pct,
            max_drawdown_percent=prop_config.effective_max_drawdown_pct,
            
            # SCALP specific
            min_mtf_alignment=0.7,
            max_open_trades=1, # Strict serialization for scalp
            use_vision=False, # Disable vision for faster initial backtest
            use_yolo=False,
        )
        
        # Check for profile specific weights
        if self.profile == 'SCALP':
            scalp_weights = 'models/weights/multihead_tcn_SCALP.pth'
            if os.path.exists(scalp_weights):
                strategy_config.tcn_weights = scalp_weights
                logger.info(f"Using SCALP specific TCN weights: {scalp_weights}")
            
            # Adjust sequence length if needed
            strategy_config.sequence_length = 60 # Match training default
            
        # 5. Initialize Bot with Strategy
        bot_config = BotConfig(
            symbol=self.symbol,
            timeframe="M5", # Base timeframe
            data_window=500, # Increased for indicators (SMA200 etc)
        )
        
        from trading.bot import BacktestBot
        # We use BacktestBot directly or inject into TradingBot? 
        # The codebase has a BacktestBot class in trading/bot.py, let's use that structure if possible
        # But BacktestRunner uses TradingBot. Let's stick to TradingBot with BacktestConnector as it seems designed for that.
        
        bot = TradingBot(
            config=bot_config,
            strategy_class=NeuralHybridStrategy,
            connector=connector
        )
        
        # Manually overwrite strategy config
        bot.strategy.config = strategy_config
        
        # Initialize strategy
        logger.info("Initializing strategy...")
        if not bot.strategy.initialize(starting_balance=bt_config.initial_balance):
            logger.error("Strategy initialization failed")
            return

        # 6. Run Simulation
        logger.info("Starting simulation...")
        import time
        start_time = time.time()
        
        equity_curve = []
        
        while connector.next_bar():
            bot.step()
            
            equity_curve.append({
                'time': connector.current_time,
                'equity': connector.equity,
                'balance': connector.balance
            })
            
            if len(equity_curve) % 1000 == 0:
                print(f"Processed {len(equity_curve)} bars...", end='\r')
                
        duration = time.time() - start_time
        logger.info(f"Simulation complete in {duration:.2f}s")
        
        # 7. Generate Report
        self._generate_report(connector, equity_curve)
        
        # Save equity curve
        pd.DataFrame(equity_curve).to_csv(f"backtest_results_{self.profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

if __name__ == "__main__":
    # Default to M5 data for SCALP
    data_file = "data/raw/EURUSD_M5_latest.csv"
    if not os.path.exists(data_file):
        # Fallback to creating synthetic or finding H1
        data_file = "data/raw/EURUSD_H1_latest.csv"
        print(f"Warning: M5 data not found, falling back to {data_file}")
        
    runner = OptimizedBacktestRunner(data_file, profile="SCALP")
    runner.run()
