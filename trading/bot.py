# trading/bot.py
"""
Main trading bot orchestration.
"""
import time
import logging
import signal as os_signal
import pandas as pd
import sys
from typing import Optional, Type
from datetime import datetime
from dataclasses import dataclass

from trading.mt5_connector import MT5Connector, MockMT5Connector
from trading.risk_manager import RiskManager, RiskConfig
from risk_management.risk_manager import RiskManager as RiskManagerV2, RiskManagerConfig
from strategies.base import Strategy
from strategies.neural_hybrid import NeuralHybridStrategy
from utils.config import settings

logger = logging.getLogger(__name__)


@dataclass
class BotConfig:
    """Bot configuration."""
    symbol: str = "EURUSD"
    timeframe: str = "H1"
    tick_interval: float = 10.0      # Seconds between checks
    data_window: int = 100           # Candles to fetch
    use_mock: bool = False           # Use mock connector for testing
    log_level: str = "INFO"


class TradingBot:
    """
    Main trading bot that orchestrates:
    - Market data fetching
    - Strategy execution
    - Risk management
    - Position tracking
    """
    
    def __init__(
        self,
        config: Optional[BotConfig] = None,
        strategy_class: Optional[Type[Strategy]] = None,
        connector: Optional[MT5Connector] = None,
    ):
        self.config = config or BotConfig(
            symbol=settings.SYMBOL,
            timeframe=settings.TIMEFRAME,
            tick_interval=settings.TICK_INTERVAL,
        )
        
        # Initialize components
        self.connector = connector
        if self.connector is None:
            self._init_connector()
            
        self._init_risk_manager()
        self._init_strategy(strategy_class)
        
        # State
        self.running = False
        self.iteration_count = 0
        self.last_bar_time: Optional[datetime] = None
        
        # Setup graceful shutdown
        self._setup_signal_handlers()
    
    def _init_connector(self):
        """Initialize market data and execution connector."""
        if self.config.use_mock:
            self.connector = MockMT5Connector(
                symbol=self.config.symbol,
            )
            logger.info("Using MOCK connector")
        else:
            self.connector = MT5Connector(
                account=settings.MT5_ACCOUNT,
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER,
                path=settings.MT5_PATH,
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                magic_number=settings.MAGIC_NUMBER,
            )
    
    def _init_risk_manager(self):
        """Initialize risk management."""
        # Get initial balance
        initial_balance = 10000.0  # Default fallback
        
        if self.connector.connect():
            account_info = self.connector.get_account_info()
            if account_info:
                initial_balance = account_info.balance
        
        # Create risk manager config
        config = RiskManagerConfig(
            profile=settings.TRADING_PROFILE if hasattr(settings, 'TRADING_PROFILE') else 'INTRADAY',
            input_features=settings.INPUT_FEATURES if hasattr(settings, 'INPUT_FEATURES') else 64,
        )
        
        self.risk_manager = RiskManagerV2(config=config)
        
        logger.info(f"Risk manager initialized with balance: {initial_balance:.2f}")
    
    def _init_strategy(self, strategy_class: Optional[Type[Strategy]]):
        """Initialize trading strategy."""
        strategy_cls = strategy_class or NeuralHybridStrategy
        
        self.strategy = strategy_cls(
            data_provider=self.connector,
            executor=self.connector,
            risk_manager=self.risk_manager,
        )
        
        logger.info(f"Strategy initialized: {self.strategy.name}")
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def shutdown_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.stop()
        
        try:
            os_signal.signal(os_signal.SIGINT, shutdown_handler)
            os_signal.signal(os_signal.SIGTERM, shutdown_handler)
        except Exception:
            pass  # May fail in some environments
    
    def run(self):
        """
        Main bot loop.
        
        Continuously:
        1. Fetches market data
        2. Passes to strategy
        3. Handles any errors
        4. Sleeps until next iteration
        """
        # Connect
        if not self.connector.connect():
            logger.error("Failed to connect. Exiting.")
            return
        
        # Update risk manager with real balance
        account_info = self.connector.get_account_info()
        if account_info:
            self.risk_manager.update_balance(account_info.balance)
        
        logger.info(
            f"🚀 Bot started | Symbol: {self.config.symbol} | "
            f"Timeframe: {self.config.timeframe} | Strategy: {self.strategy.name}"
        )
        
        self.running = True
        
        while self.running:
            try:
                self._run_iteration()
                time.sleep(self.config.tick_interval)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
                
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(self.config.tick_interval)
        
        self._shutdown()
    
    def step(self):
        """Single iteration step (public for backtesting)."""
        self._run_iteration()

    def _run_iteration(self):
        """Single iteration of the main loop."""
        self.iteration_count += 1
        
        # Fetch data
        df = self.connector.get_data(n=self.config.data_window)
        
        if df.empty:
            logger.warning("No data received")
            return
        
        # Check if new bar (avoid processing same bar multiple times)
        current_bar_time = df['time'].iloc[-1]
        if self.last_bar_time is not None and current_bar_time == self.last_bar_time:
            logger.debug(f"Same bar: {current_bar_time}")
            return
        
        self.last_bar_time = current_bar_time
        logger.debug(f"Processing bar: {current_bar_time}")
        
        # Run strategy
        signal = self.strategy.on_bar(df)
        
        if signal:
            logger.info(f"Iteration {self.iteration_count}: Signal = {signal}")
    
    def stop(self):
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        self.running = False
    
    def _shutdown(self):
        """Cleanup on shutdown."""
        logger.info("Shutting down...")
        
        # Log final stats
        if hasattr(self.strategy, 'get_stats'):
            stats = self.strategy.get_stats()
            logger.info(f"Strategy stats: {stats}")
        
        risk_status = self.risk_manager.get_status()
        logger.info(f"Risk status: {risk_status}")
        
        # Disconnect
        self.connector.disconnect()
        
        logger.info("✅ Bot shutdown complete")
    
    def get_status(self) -> dict:
        """Get current bot status."""
        return {
            'running': self.running,
            'iteration_count': self.iteration_count,
            'last_bar_time': str(self.last_bar_time) if self.last_bar_time else None,
            'strategy': self.strategy.get_stats() if hasattr(self.strategy, 'get_stats') else {},
            'risk': self.risk_manager.get_status() if hasattr(self.risk_manager, 'get_status') else {},
            'positions': self.connector.get_open_positions(),
        }


class BacktestBot:
    """
    Backtesting variant of the trading bot.
    Uses historical data instead of live feed.
    """
    
    def __init__(
        self,
        data: "pd.DataFrame",
        strategy_class: Type[Strategy],
        initial_balance: float = 10000.0,
    ):
        # FIX: Imported BacktestConfig here to fix the TypeError
        from trading.backtest import BacktestExecutor, BacktestConfig
        
        self.data = data
        
        # FIX: Create config object and pass it to executor
        config = BacktestConfig(initial_balance=initial_balance)
        self.executor = BacktestExecutor(config=config)
        
        # Create risk manager config
        risk_config = RiskManagerConfig(
            profile='INTRADAY',
            input_features=64,
        )
        self.risk_manager = RiskManagerV2(config=risk_config)
        
        self.strategy = strategy_class(
            data_provider=self,  # Bot acts as data provider
            executor=self.executor,
        )
        
        self.current_idx = 0
        self.window_size = 100
    
    def get_data(self, n: int = 100) -> "pd.DataFrame":
        """Provide data window for strategy."""
        import pandas as pd
        
        start_idx = max(0, self.current_idx - n + 1)
        return self.data.iloc[start_idx:self.current_idx + 1].copy()
    
    def run(self) -> dict:
        """Run backtest through all data."""
        results = []
        
        for i in range(self.window_size, len(self.data)):
            self.current_idx = i
            df = self.get_data(self.window_size)
            
            signal = self.strategy.on_bar(df)
            
            results.append({
                'time': df['time'].iloc[-1],
                'close': df['close'].iloc[-1],
                'signal': signal,
            })
            
            # Update executor with current price for P&L
            self.executor.update_price(df['close'].iloc[-1])
        
        return {
            'trades': self.executor.get_trade_history(),
            'final_balance': self.executor.balance,
            'signals': results,
        }