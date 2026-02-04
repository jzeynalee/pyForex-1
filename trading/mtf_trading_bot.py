# trading/mtf_trading_bot.py
"""
Multi-Timeframe Trading Bot

Complete trading bot with MTF analysis integration.
Supports both SCALP (M5/M15/H1) and SWING (M15/H1/H4) profiles.
"""

import time
import logging
import signal as os_signal
import pandas as pd
from typing import Optional, Type, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from unittest.mock import Mock

# Import at module level for test patching
from trading.mt5_connector import MT5Connector, MockMT5Connector
from trading.risk_manager import RiskManager, RiskConfig
from trading.mtf_data_provider import MTFDataProvider
from trend_detection.mtf_trend_detector import MTFTrendDetector
from utils.mtf_config import get_profile
from utils.config import settings

logger = logging.getLogger(__name__)


@dataclass
class MTFBotConfig:
    """Configuration for MTF Trading Bot."""
    # Trading settings
    symbol: str = "EURUSD"
    profile: str = "SWING"  # 'SCALP' or 'SWING'
    
    # Polling
    tick_interval: float = 10.0  # Seconds between checks
    
    # Execution
    use_mock: bool = False
    enable_trading: bool = True  # Set False for analysis-only mode
    
    # Risk
    max_daily_loss_pct: float = 0.03
    risk_per_trade_pct: float = 0.01
    
    # Logging
    log_level: str = "INFO"
    log_analysis: bool = True


class MTFTradingBot:
    """
    Multi-Timeframe Trading Bot.
    
    Features:
    - Configurable MTF profiles (SCALP or SWING)
    - Automatic multi-timeframe data fetching
    - Comprehensive trend analysis
    - Risk-managed position sizing
    - Support for live and mock trading
    
    Usage:
        bot = MTFTradingBot(config=MTFBotConfig(profile="SWING"))
        bot.run()
    """
    
    def __init__(
        self,
        config: Optional[MTFBotConfig] = None,
        connector: Optional[Any] = None,
        ml_model: Optional[Any] = None,
    ):
        if bool(getattr(settings, 'ENFORCE_AUTHORITATIVE_PIPELINE', True)):
            raise RuntimeError(
                "Authoritative pipeline enforced: MTFTradingBot is disabled (uses legacy trading.risk_manager path)"
            )
        self.config = config or MTFBotConfig()
        self.ml_model = ml_model
        
        # Initialize components
        self._init_connector(connector)
        self._init_mtf_engine()
        self._init_risk_manager()
        
        # State
        self.running = False
        self.iteration_count = 0
        self.last_bar_time: Dict[str, Optional[datetime]] = {}
        self.daily_trades = 0
        self.daily_pnl = 0.0
        
        # Setup shutdown handlers
        self._setup_signal_handlers()
    
    def _init_connector(self, connector: Optional[Any]):
        """Initialize market data and execution connector."""
        if connector is not None:
            self.connector = connector
            return
        
        if self.config.use_mock:
            self.connector = MockMT5Connector(symbol=self.config.symbol)
            logger.info("Using MOCK connector")
        else:
            try:
                from utils.config import settings
                
                self.connector = MT5Connector(
                    account=settings.MT5_ACCOUNT,
                    password=settings.MT5_PASSWORD,
                    server=settings.MT5_SERVER,
                    path=settings.MT5_PATH,
                    symbol=self.config.symbol,
                    timeframe="H1",  # Will be overridden by MTF
                    magic_number=settings.MAGIC_NUMBER,
                )
            except Exception as e:
                logger.warning(f"MT5 connection failed: {e}. Using mock.")
                self.connector = MockMT5Connector(symbol=self.config.symbol)
    
    def _init_mtf_engine(self):
        """Initialize MTF analysis engine."""
        # Get profile
        self.profile = get_profile(self.config.profile)
        
        # Data provider
        self.data_provider = MTFDataProvider(
            symbol=self.config.symbol,
            connector=self.connector,
            cache_enabled=True,
        )
        
        # Trend detector
        self.trend_detector = MTFTrendDetector(
            profile=self.profile,
            ml_model=self.ml_model,
        )
        
        logger.info(f"MTF Engine initialized: {self.profile.name} profile")
        logger.info(f"  Timeframes: {self.profile.timeframe_strings}")
        logger.info(f"  Weights: {self.profile.weights}")
    
    def _init_risk_manager(self):
        """Initialize risk management."""
        from trading.risk_manager import RiskManager, RiskConfig
        
        # Get initial balance
        initial_balance = 10000.0
        
        if self.connector.connect():
            account_info = self.connector.get_account_info()
            if account_info:
                initial_balance = account_info.balance
        
        # Initialize risk manager with mock model for testing
        self.risk_manager = RiskManager(
            model=Mock(),  # Mock model for testing
            feature_columns=['close', 'volume'],  # Basic features
            config=RiskConfig(
                risk_per_trade=self.config.risk_per_trade_pct if hasattr(self.config, 'risk_per_trade_pct') else 0.01,
            ),
        )
        
        logger.info(f"Risk manager initialized with balance: {initial_balance:.2f}")
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def shutdown_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.stop()
        
        try:
            os_signal.signal(os_signal.SIGINT, shutdown_handler)
            os_signal.signal(os_signal.SIGTERM, shutdown_handler)
        except Exception:
            pass
    
    def run(self):
        """
        Main bot loop.
        
        Continuously:
        1. Fetches MTF data
        2. Runs trend analysis
        3. Generates trading signals
        4. Manages positions
        """
        # Connect
        if not self.connector.connect():
            logger.error("Failed to connect. Exiting.")
            return
        
        # Update risk manager with real balance
        account_info = self.connector.get_account_info()
        if account_info:
            self.risk_manager.update_balance(account_info.balance)
        
        logger.info(f"🚀 MTF Bot started")
        logger.info(f"   Symbol: {self.config.symbol}")
        logger.info(f"   Profile: {self.config.profile}")
        logger.info(f"   Timeframes: {self.profile.timeframe_strings}")
        
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
    
    def _run_iteration(self):
        """Single iteration of the main loop."""
        self.iteration_count += 1
        
        # Fetch MTF data
        dfs_dict = self.data_provider.fetch_for_profile(self.profile)
        
        # Validate data
        is_valid, errors = self.data_provider.validate_data(
            dfs_dict,
            min_bars=self.profile.min_bars,
        )
        
        if not is_valid:
            logger.warning(f"Data validation issues: {errors}")
            return
        
        # Check if new bar on primary timeframe
        primary_tf = self.profile.primary_tf.value
        if primary_tf in dfs_dict and 'time' in dfs_dict[primary_tf].columns:
            current_time = dfs_dict[primary_tf]['time'].iloc[-1]
            last_time = self.last_bar_time.get(primary_tf)
            
            if last_time is not None and current_time == last_time:
                logger.debug(f"Same bar: {current_time}")
                return
            
            self.last_bar_time[primary_tf] = current_time
            logger.debug(f"Processing bar: {current_time}")
        
        # Run MTF analysis
        analysis = self.trend_detector.detect(dfs_dict, compute_ml_features=True)
        
        # Log analysis
        if self.config.log_analysis:
            self._log_analysis(analysis)
        
        # Handle signal
        if analysis.signal != 'NO_TRADE' and self.config.enable_trading:
            self._handle_signal(analysis, dfs_dict)
    
    def _log_analysis(self, analysis: "MTFTrendResult"):
        """Log analysis results."""
        logger.info(
            f"📊 Analysis | Trend: {analysis.trend_name} | "
            f"Direction: {analysis.direction} | Confidence: {analysis.confidence:.2f}"
        )
        logger.info(
            f"   MTF Score: {analysis.mtf_score:.2f} | "
            f"Alignment: {analysis.mtf_alignment:.2f} | "
            f"Higher TF: {'✓' if analysis.higher_tf_aligned else '✗'}"
        )
        logger.info(
            f"   Regime: {analysis.regime} | Signal: {analysis.signal} "
            f"({analysis.signal_confidence:.2f})"
        )
        
        # Per-timeframe details
        for tf, score in analysis.timeframe_scores.items():
            direction = analysis.timeframe_directions.get(tf, 0)
            dir_str = '↑' if direction > 0 else ('↓' if direction < 0 else '→')
            logger.debug(f"   {tf}: {score:.2f} {dir_str}")
    
    def _handle_signal(
        self,
        analysis: "MTFTrendResult",
        dfs_dict: Dict[str, pd.DataFrame],
    ):
        """Handle trading signal."""
        # Check risk limits
        account_info = self.connector.get_account_info()
        positions = self.connector.get_open_positions()
        
        can_trade, reason = self.risk_manager.check_risk_limits(
            current_balance=account_info.balance if account_info else 10000,
            current_equity=account_info.equity if account_info else 10000,
            open_positions=len(positions) if positions else 0,
        )
        
        if not can_trade:
            logger.warning(f"Risk limit reached: {reason}")
            return
        
        # Get primary TF data for ATR
        primary_tf = self.profile.primary_tf.value
        df = dfs_dict.get(primary_tf)
        
        if df is None or df.empty:
            logger.warning("No primary TF data for trade params")
            return
        
        # Calculate trade parameters
        params = self.risk_manager.get_params(
            df=df,
            signal=analysis.signal,
            current_balance=account_info.balance if account_info else None,
            symbol_info=self.connector.get_symbol_info(),
        )
        
        # Adjust for confidence
        volume = params.volume
        if analysis.confidence > 0.75:
            volume = min(volume * 1.2, 1.0)  # Cap at 1 lot
        elif analysis.confidence < 0.55:
            volume = volume * 0.8
        
        # Execute trade
        logger.info(
            f"📈 Executing {analysis.signal} | "
            f"Vol: {volume:.2f} | SL: {params.stop_loss:.5f} | TP: {params.take_profit:.5f}"
        )
        
        result = self.connector.entry(
            signal=analysis.signal,
            volume=volume,
            sl=params.stop_loss,
            tp=params.take_profit,
        )
        
        if result.success:
            logger.info(f"✅ Trade executed: Ticket {result.ticket} @ {result.price}")
            self.daily_trades += 1
            self.risk_manager.record_trade()
        else:
            logger.error(f"❌ Trade failed: {result.error}")
    
    def stop(self):
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        self.running = False
    
    def _shutdown(self):
        """Cleanup on shutdown."""
        logger.info("Shutting down...")
        
        # Log final stats
        risk_status = self.risk_manager.get_status()
        logger.info(f"Risk status: {risk_status}")
        logger.info(f"Daily trades: {self.daily_trades}")
        
        # Disconnect
        self.connector.disconnect()
        
        logger.info("✅ Bot shutdown complete")
    
    def get_status(self) -> dict:
        """Get current bot status."""
        return {
            'running': self.running,
            'iteration_count': self.iteration_count,
            'profile': self.config.profile,
            'timeframes': self.profile.timeframe_strings,
            'daily_trades': self.daily_trades,
            'risk': self.risk_manager.get_status(),
            'positions': self.connector.get_open_positions() if self.connector else [],
        }
    
    def analyze_once(self) -> "MTFTrendResult":
        """
        Run single analysis without trading.
        Useful for testing or manual analysis.
        
        Returns:
            MTFTrendResult
        """
        # Ensure connected
        if not self.connector.ensure_connected():
            self.connector.connect()
        
        # Fetch data
        dfs_dict = self.data_provider.fetch_for_profile(self.profile)
        
        # Run analysis
        return self.trend_detector.detect(dfs_dict)


class MTFBacktestBot:
    """
    Backtesting variant with MTF support.
    
    Uses pre-loaded historical data for all timeframes.
    """
    
    def __init__(
        self,
        historical_data: Dict[str, pd.DataFrame],
        profile_name: str = "SWING",
        initial_balance: float = 10000.0,
        ml_model: Optional[Any] = None,
    ):
        """
        Args:
            historical_data: Dict mapping timeframe to full historical DataFrame
            profile_name: MTF profile to use
            initial_balance: Starting balance
            ml_model: Optional ML model
        """
        from trading.mtf_data_provider import BacktestMTFDataProvider
        from trend_detection.mtf_trend_detector import MTFTrendDetector
        from trading.backtest import BacktestExecutor, BacktestConfig
        from trading.risk_manager import RiskManager
        from utils.mtf_config import get_profile
        
        # Get profile
        self.profile = get_profile(profile_name)
        
        # Validate historical data has all required timeframes
        for tf in self.profile.timeframe_strings:
            if tf not in historical_data:
                raise ValueError(f"Missing historical data for {tf}")
        
        # Initialize components
        self.data_provider = BacktestMTFDataProvider(
            historical_data=historical_data,
            symbol="BACKTEST",
        )
        
        self.trend_detector = MTFTrendDetector(
            profile=self.profile,
            ml_model=ml_model,
        )
        
        config = BacktestConfig(initial_balance=initial_balance)
        self.executor = BacktestExecutor(config=config)
        
        self.risk_manager = RiskManager(account_balance=initial_balance)
        
        # Results storage
        self.results: list = []
    
    def run(
        self,
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        """
        Run backtest through historical data.
        
        Args:
            progress_callback: Optional callback(current, total) for progress
        
        Returns:
            Dict with backtest results
        """
        primary_tf = self.profile.primary_tf.value
        total_bars = len(self.data_provider.historical_data.get(primary_tf, []))
        start_idx = self.profile.min_bars
        
        logger.info(f"Starting backtest: {total_bars - start_idx} bars")
        
        for i in range(start_idx, total_bars):
            # Set time position
            self.data_provider.current_idx[primary_tf] = i
            
            # Sync other timeframes
            current_time = self.data_provider.get_current_time(primary_tf)
            if current_time:
                self.data_provider.set_time(current_time)
            
            # Fetch data windows
            dfs_dict = self.data_provider.fetch_for_profile(self.profile)
            
            # Run analysis
            analysis = self.trend_detector.detect(dfs_dict, compute_ml_features=False)
            
            # Update executor price
            if primary_tf in dfs_dict and not dfs_dict[primary_tf].empty:
                current_price = dfs_dict[primary_tf]['close'].iloc[-1]
                self.executor.update_price(current_price, current_time)
            
            # Handle signal
            if analysis.signal != 'NO_TRADE':
                self._handle_backtest_signal(analysis, dfs_dict)
            
            # Store result
            self.results.append({
                'time': current_time,
                'trend': analysis.trend_name,
                'direction': analysis.direction,
                'signal': analysis.signal,
                'confidence': analysis.confidence,
                'mtf_score': analysis.mtf_score,
            })
            
            # Progress callback
            if progress_callback and i % 100 == 0:
                progress_callback(i - start_idx, total_bars - start_idx)
        
        # Close remaining positions
        self.executor.close_all_positions()
        
        # Get metrics
        metrics = self.executor.get_performance_metrics()
        
        return {
            'metrics': metrics,
            'trades': self.executor.get_trade_history(),
            'analysis_results': self.results,
            'final_balance': self.executor.balance,
        }
    
    def _handle_backtest_signal(
        self,
        analysis: "MTFTrendResult",
        dfs_dict: Dict[str, pd.DataFrame],
    ):
        """Handle signal in backtest."""
        primary_tf = self.profile.primary_tf.value
        df = dfs_dict.get(primary_tf)
        
        if df is None or df.empty:
            return
        
        # Check if we have open positions
        if len(self.executor.positions) > 0:
            return  # Simple: one position at a time
        
        # Get trade params
        params = self.risk_manager.get_params(
            df=df,
            signal=analysis.signal,
        )
        
        # Execute
        self.executor.entry(
            signal=analysis.signal,
            volume=params.volume,
            sl=params.stop_loss,
            tp=params.take_profit,
        )


def run_mtf_bot(
    profile: str = "SWING",
    mock: bool = False,
    analysis_only: bool = False,
):
    """
    Convenience function to run MTF bot.
    
    Args:
        profile: 'SCALP' or 'SWING'
        mock: Use mock connector
        analysis_only: Disable actual trading
    """
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('mtf_trading.log'),
        ],
    )
    
    config = MTFBotConfig(
        profile=profile,
        use_mock=mock,
        enable_trading=not analysis_only,
    )
    
    bot = MTFTradingBot(config=config)
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\nStopped by user")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MTF Trading Bot")
    parser.add_argument('--profile', type=str, default='SWING', choices=['SCALP', 'SWING'])
    parser.add_argument('--mock', action='store_true', help='Use mock connector')
    parser.add_argument('--analysis-only', action='store_true', help='Disable trading')
    
    args = parser.parse_args()
    
    run_mtf_bot(
        profile=args.profile,
        mock=args.mock,
        analysis_only=args.analysis_only,
    )


def add_get_state_method():
    """Add missing get_state method to MTFTradingBot class."""
    def get_state(self):
        """Get current bot state (alias for get_status)."""
        return self.get_status()
    
    # Add method to class
    from trading.mtf_trading_bot import MTFTradingBot
    MTFTradingBot.get_state = get_state


# Add missing get_state method to MTFTradingBot class
def get_state(self):
    """Get current bot state (alias for get_status)."""
    return self.get_status()

# Add the method to the class
MTFTradingBot.get_state = get_state
