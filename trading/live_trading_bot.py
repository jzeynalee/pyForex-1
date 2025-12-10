# trading/live_trading_bot.py
"""
Live Trading Bot with Risk Management Integration

Production-ready trading bot that:
- Runs continuous trading loop
- Integrates TCN predictions with risk management
- Manages positions with ML-based SL/TP
- Enforces hard rules and limits
- Provides real-time monitoring

Usage:
    bot = LiveTradingBot(config)
    bot.initialize()
    bot.run()  # Blocking
    # or
    bot.start()  # Non-blocking (thread)
"""

import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

# Strategy imports
from strategies.neural_hybrid import (
    NeuralHybridStrategy, StrategyConfig, create_strategy, Order
)
from trading.decision_engine import TradeDecision

# Risk management imports
try:
    from risk_management import (
        TradeGatekeeper, HardRulesConfig, TradingSession,
        generate_performance_report, PerformanceMetrics
    )
    HAS_RISK_MANAGEMENT = True
except ImportError:
    HAS_RISK_MANAGEMENT = False

logger = logging.getLogger(__name__)


class BotState(Enum):
    """Bot operational states."""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class BotConfig:
    """Configuration for live trading bot."""
    # Trading settings
    symbol: str = 'EURUSD'
    profile: str = 'INTRADAY'
    
    # Timing
    check_interval_seconds: int = 60     # How often to check for signals
    market_open_hour: int = 0            # UTC hour market opens (Sunday)
    market_close_hour: int = 22          # UTC hour market closes (Friday)
    
    # Model paths
    tcn_weights: str = 'models/weights/tcn_best.pt'
    vit_weights: Optional[str] = None
    yolo_weights: Optional[str] = None
    meta_model_path: Optional[str] = None
    
    # Risk settings
    base_risk_percent: float = 1.0
    max_daily_loss_percent: float = 3.0
    max_weekly_loss_percent: float = 6.0
    max_open_trades: int = 3
    
    # Notifications
    enable_notifications: bool = True
    telegram_config: Optional[Dict] = None
    
    # Logging
    log_trades: bool = True
    trades_log_file: str = 'logs/trades.json'
    performance_log_file: str = 'logs/performance.json'
    
    # Safety
    dry_run: bool = False               # Paper trading mode
    require_confirmation: bool = False   # Require manual confirmation
    max_consecutive_losses: int = 5      # Pause after N losses
    cooldown_minutes: int = 30           # Cooldown after loss streak


class LiveTradingBot:
    """
    Production live trading bot with full risk management.
    
    Features:
    - Continuous market monitoring
    - ML-based signal generation
    - Automated risk management
    - Position tracking
    - Performance monitoring
    - Notification system
    - Graceful shutdown handling
    
    Example:
        from trading.mt5_executor import MT5Executor
        from data.mt5_provider import MT5DataProvider
        
        bot = LiveTradingBot(
            config=BotConfig(symbol='EURUSD', profile='INTRADAY'),
            data_provider=MT5DataProvider(),
            executor=MT5Executor()
        )
        
        bot.initialize()
        bot.run()
    """
    
    def __init__(
        self,
        config: BotConfig,
        data_provider,
        executor,
        on_signal: Optional[Callable[[TradeDecision], None]] = None,
        on_trade: Optional[Callable[[Order, Dict], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None
    ):
        self.config = config
        self.data_provider = data_provider
        self.executor = executor
        
        # Callbacks
        self.on_signal = on_signal
        self.on_trade = on_trade
        self.on_error = on_error
        
        # State
        self.state = BotState.STOPPED
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Components
        self.strategy: Optional[NeuralHybridStrategy] = None
        self.gatekeeper: Optional[TradeGatekeeper] = None
        
        # Tracking
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._consecutive_losses = 0
        self._last_trade_time: Optional[datetime] = None
        self._cooldown_until: Optional[datetime] = None
        self._trade_count = 0
        
        # Performance
        self._session_start: Optional[datetime] = None
        self._signals_generated = 0
        self._trades_executed = 0
        self._trades_rejected = 0
        
        # Notifications
        self._notifier = None
        if config.enable_notifications:
            self._init_notifier()
    
    def initialize(self) -> bool:
        """
        Initialize all bot components.
        
        Returns:
            True if successful
        """
        self.state = BotState.INITIALIZING
        logger.info("Initializing trading bot...")
        
        try:
            # Create strategy config
            strategy_config = StrategyConfig(
                profile=self.config.profile,
                symbol=self.config.symbol,
                tcn_weights=self.config.tcn_weights,
                vit_weights=self.config.vit_weights,
                yolo_weights=self.config.yolo_weights,
                meta_model_path=self.config.meta_model_path,
                base_risk_percent=self.config.base_risk_percent,
                max_daily_loss_percent=self.config.max_daily_loss_percent,
                max_open_trades=self.config.max_open_trades
            )
            
            # Initialize strategy
            self.strategy = NeuralHybridStrategy(
                config=strategy_config,
                data_provider=self.data_provider,
                executor=self.executor
            )
            
            if not self.strategy.initialize():
                raise RuntimeError("Strategy initialization failed")
            
            # Initialize gatekeeper for additional rules
            if HAS_RISK_MANAGEMENT:
                self.gatekeeper = TradeGatekeeper(HardRulesConfig(
                    max_total_exposure=20.0,
                    max_single_pair_exposure=5.0
                ))
            
            # Verify connections
            if not self._verify_connections():
                raise RuntimeError("Connection verification failed")
            
            self._session_start = datetime.utcnow()
            self.state = BotState.STOPPED
            
            logger.info("Bot initialized successfully")
            return True
            
        except Exception as e:
            self.state = BotState.ERROR
            logger.error(f"Initialization failed: {e}", exc_info=True)
            if self.on_error:
                self.on_error(e)
            return False
    
    def start(self) -> bool:
        """
        Start bot in background thread.
        
        Returns:
            True if started successfully
        """
        if self.state == BotState.RUNNING:
            logger.warning("Bot already running")
            return False
        
        if self.strategy is None:
            logger.error("Bot not initialized")
            return False
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
        logger.info("Bot started in background")
        return True
    
    def run(self):
        """Run bot in main thread (blocking)."""
        if self.strategy is None:
            logger.error("Bot not initialized")
            return
        
        self._stop_event.clear()
        self._run_loop()
    
    def stop(self):
        """Stop the bot gracefully."""
        logger.info("Stopping bot...")
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)
        
        self.state = BotState.STOPPED
        self._save_performance_log()
        
        logger.info("Bot stopped")
    
    def pause(self):
        """Pause trading (keep running but don't execute)."""
        self.state = BotState.PAUSED
        logger.info("Bot paused")
    
    def resume(self):
        """Resume trading."""
        if self.state == BotState.PAUSED:
            self.state = BotState.RUNNING
            logger.info("Bot resumed")
    
    def _run_loop(self):
        """Main trading loop."""
        self.state = BotState.RUNNING
        logger.info(f"Trading loop started for {self.config.symbol}")
        
        while not self._stop_event.is_set():
            try:
                current_time = datetime.utcnow()
                
                # Check if market is open
                if not self._is_market_open(current_time):
                    logger.debug("Market closed, waiting...")
                    self._wait(60)
                    continue
                
                # Check cooldown
                if self._in_cooldown(current_time):
                    logger.debug(f"In cooldown until {self._cooldown_until}")
                    self._wait(60)
                    continue
                
                # Check daily/weekly limits
                if not self._check_limits():
                    self._wait(60)
                    continue
                
                # Evaluate market
                if self.state == BotState.RUNNING:
                    self._evaluate_and_trade(current_time)
                
                # Wait for next check
                self._wait(self.config.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                if self.on_error:
                    self.on_error(e)
                self._wait(60)  # Wait before retrying
        
        self.state = BotState.STOPPED
    
    def _evaluate_and_trade(self, current_time: datetime):
        """Evaluate market and execute trades if appropriate."""
        # Get trading decision
        decision = self.strategy.evaluate(current_time)
        self._signals_generated += 1
        
        if decision is None:
            return
        
        # Callback for signal
        if self.on_signal:
            self.on_signal(decision)
        
        if not decision.should_trade:
            self._trades_rejected += 1
            logger.debug(f"Signal rejected: {decision.rejection_reasons}")
            return
        
        # Additional gatekeeper check
        if self.gatekeeper:
            validation = self.gatekeeper.validate_trade(
                pair=self.config.symbol,
                direction=decision.direction,
                position_size=decision.position_size,
                entry_price=self.data_provider.get_price(self.config.symbol),
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                account_balance=self.executor.get_account_balance(),
                current_spread=self.data_provider.get_spread(self.config.symbol),
                current_time=current_time
            )
            
            if not validation['allowed']:
                self._trades_rejected += 1
                logger.info(f"Gatekeeper rejected: {[v['message'] for v in validation['violations']]}")
                return
        
        # Create and execute order
        order = self.strategy.create_order(decision)
        
        if order is None:
            return
        
        # Dry run mode
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would execute: {order.direction} {order.volume} {order.symbol}")
            self._trades_executed += 1
            return
        
        # Manual confirmation
        if self.config.require_confirmation:
            logger.info(f"[CONFIRM] Signal: {order.direction} {order.volume} {order.symbol}")
            # In a real implementation, wait for user confirmation
            return
        
        # Execute trade
        success = self.strategy.execute(order)
        
        if success:
            self._trades_executed += 1
            self._trade_count += 1
            self._last_trade_time = current_time
            
            # Log trade
            if self.config.log_trades:
                self._log_trade(order, decision)
            
            # Notify
            if self._notifier:
                self._send_trade_notification(order, decision)
            
            # Callback
            if self.on_trade:
                self.on_trade(order, decision.to_dict())
        else:
            logger.warning("Trade execution failed")
    
    def on_position_closed(self, ticket: str, pnl: float):
        """Handle position close event."""
        # Update strategy
        self.strategy.on_trade_closed(ticket, pnl)
        
        # Update tracking
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        
        # Track consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
            
            if self._consecutive_losses >= self.config.max_consecutive_losses:
                self._cooldown_until = datetime.utcnow() + timedelta(
                    minutes=self.config.cooldown_minutes
                )
                logger.warning(
                    f"Consecutive losses ({self._consecutive_losses}), "
                    f"entering cooldown until {self._cooldown_until}"
                )
                
                if self._notifier:
                    self._send_alert_notification(
                        f"⚠️ Entering cooldown after {self._consecutive_losses} losses"
                    )
        else:
            self._consecutive_losses = 0
        
        # Log
        logger.info(f"Position closed: {ticket} | PnL: {pnl:.2f} | Daily: {self._daily_pnl:.2f}")
    
    def reset_daily_stats(self):
        """Reset daily statistics."""
        self._daily_pnl = 0.0
        self.strategy.reset_daily_stats()
        logger.info("Daily stats reset")
    
    def reset_weekly_stats(self):
        """Reset weekly statistics."""
        self._weekly_pnl = 0.0
        logger.info("Weekly stats reset")
    
    def get_status(self) -> Dict:
        """Get current bot status."""
        return {
            'state': self.state.value,
            'symbol': self.config.symbol,
            'profile': self.config.profile,
            'session_start': self._session_start.isoformat() if self._session_start else None,
            'signals_generated': self._signals_generated,
            'trades_executed': self._trades_executed,
            'trades_rejected': self._trades_rejected,
            'daily_pnl': self._daily_pnl,
            'weekly_pnl': self._weekly_pnl,
            'consecutive_losses': self._consecutive_losses,
            'in_cooldown': self._in_cooldown(datetime.utcnow()),
            'cooldown_until': self._cooldown_until.isoformat() if self._cooldown_until else None,
            'open_positions': len(self.strategy._open_positions) if self.strategy else 0,
            'dry_run': self.config.dry_run
        }
    
    def get_performance(self) -> Dict:
        """Get performance statistics."""
        if self.strategy:
            return self.strategy.get_performance_stats()
        return {}
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    def _verify_connections(self) -> bool:
        """Verify data provider and executor connections."""
        try:
            # Check data provider
            data = self.data_provider.get_data(self.config.symbol, 'M1', 1)
            if data is None or len(data) == 0:
                logger.error("Data provider not returning data")
                return False
            
            # Check executor
            balance = self.executor.get_account_balance()
            if balance <= 0:
                logger.error("Invalid account balance")
                return False
            
            logger.info(f"Connections verified. Balance: {balance:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            return False
    
    def _is_market_open(self, current_time: datetime) -> bool:
        """Check if forex market is open."""
        weekday = current_time.weekday()
        hour = current_time.hour
        
        # Closed Saturday
        if weekday == 5:
            return False
        
        # Closed Sunday until open hour
        if weekday == 6 and hour < self.config.market_open_hour:
            return False
        
        # Closed Friday after close hour
        if weekday == 4 and hour >= self.config.market_close_hour:
            return False
        
        return True
    
    def _in_cooldown(self, current_time: datetime) -> bool:
        """Check if in cooldown period."""
        if self._cooldown_until is None:
            return False
        return current_time < self._cooldown_until
    
    def _check_limits(self) -> bool:
        """Check if daily/weekly loss limits exceeded."""
        balance = self.executor.get_account_balance()
        
        # Daily limit
        max_daily = balance * (self.config.max_daily_loss_percent / 100)
        if self._daily_pnl <= -max_daily:
            logger.warning(f"Daily loss limit reached: {self._daily_pnl:.2f}")
            return False
        
        # Weekly limit
        max_weekly = balance * (self.config.max_weekly_loss_percent / 100)
        if self._weekly_pnl <= -max_weekly:
            logger.warning(f"Weekly loss limit reached: {self._weekly_pnl:.2f}")
            return False
        
        return True
    
    def _wait(self, seconds: int):
        """Wait with early exit capability."""
        self._stop_event.wait(seconds)
    
    def _init_notifier(self):
        """Initialize notification system."""
        try:
            from notifications import SocialMediaNotifier
            self._notifier = SocialMediaNotifier(self.config.telegram_config)
        except ImportError:
            logger.debug("Notifications module not available")
    
    def _send_trade_notification(self, order: Order, decision: TradeDecision):
        """Send trade notification."""
        if self._notifier:
            try:
                self._notifier.send_trade(
                    symbol=order.symbol,
                    direction=order.direction,
                    volume=order.volume,
                    sl=order.stop_loss,
                    tp=order.take_profit,
                    confidence=order.confidence
                )
            except Exception as e:
                logger.warning(f"Notification failed: {e}")
    
    def _send_alert_notification(self, message: str):
        """Send alert notification."""
        if self._notifier:
            try:
                self._notifier.send_alert(message)
            except:
                pass
    
    def _log_trade(self, order: Order, decision: TradeDecision):
        """Log trade to file."""
        try:
            log_path = Path(self.config.trades_log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            trade_log = {
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': order.symbol,
                'direction': order.direction,
                'volume': order.volume,
                'stop_loss': order.stop_loss,
                'take_profit': order.take_profit,
                'confidence': order.confidence,
                'meta_score': order.meta_score,
                'risk_percent': order.risk_percent,
                'risk_reward': order.risk_reward,
                'decision': decision.to_dict()
            }
            
            # Append to log file
            with open(log_path, 'a') as f:
                f.write(json.dumps(trade_log) + '\n')
                
        except Exception as e:
            logger.warning(f"Trade logging failed: {e}")
    
    def _save_performance_log(self):
        """Save performance log on shutdown."""
        try:
            log_path = Path(self.config.performance_log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            performance = {
                'session_start': self._session_start.isoformat() if self._session_start else None,
                'session_end': datetime.utcnow().isoformat(),
                'signals_generated': self._signals_generated,
                'trades_executed': self._trades_executed,
                'trades_rejected': self._trades_rejected,
                'daily_pnl': self._daily_pnl,
                'performance': self.get_performance()
            }
            
            with open(log_path, 'w') as f:
                json.dump(performance, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Performance log save failed: {e}")


# Factory function
def create_bot(
    symbol: str = 'EURUSD',
    profile: str = 'INTRADAY',
    data_provider=None,
    executor=None,
    dry_run: bool = True,
    **kwargs
) -> LiveTradingBot:
    """
    Create configured trading bot.
    
    Args:
        symbol: Trading symbol
        profile: Trading profile
        data_provider: Data provider instance
        executor: Order executor instance
        dry_run: Paper trading mode
        **kwargs: Additional config options
    
    Returns:
        Configured LiveTradingBot
    """
    config = BotConfig(
        symbol=symbol,
        profile=profile,
        dry_run=dry_run,
        **kwargs
    )
    
    return LiveTradingBot(config, data_provider, executor)