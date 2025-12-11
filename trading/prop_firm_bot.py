# trading/prop_firm_bot.py
"""
Prop Firm Trading Bot

Specialized trading bot designed for prop firm challenges.
Integrates:
- Prop firm specific rules and limits
- Enhanced capital protection
- Signal publishing
- Challenge progress tracking
- Automated safety measures

Usage:
    from trading.prop_firm_bot import create_prop_firm_bot
    
    bot = create_prop_firm_bot(
        firm='FTMO',
        account_size=100000,
        phase='evaluation',
        symbol='EURUSD',
        data_provider=data_provider,
        executor=executor
    )
    
    bot.run()
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

from config.prop_firm_config import (
    get_prop_firm_config, PropFirmConfig, PropFirmMonitor,
    PropFirm, ChallengePhase
)
from trading.live_trading_bot import LiveTradingBot, BotConfig, BotState
from trading.decision_engine import EnhancedDecisionEngine, DecisionEngineConfig
from signals.signal_publisher import (
    SignalPublisher, PublisherConfig, TradingBotSignalAdapter
)
from strategies.neural_hybrid import NeuralHybridStrategy, StrategyConfig

logger = logging.getLogger(__name__)


class PropFirmBotState(Enum):
    """Extended states for prop firm bot."""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    PROTECTION_HALT = "protection_halt"
    DAILY_LIMIT_HIT = "daily_limit_hit"
    CHALLENGE_PASSED = "challenge_passed"
    CHALLENGE_FAILED = "challenge_failed"


@dataclass
class PropFirmBotConfig:
    """Configuration for prop firm bot."""
    # Prop firm settings
    firm: str = 'FTMO'
    account_size: float = 100000
    phase: str = 'evaluation'
    
    # Trading settings
    symbol: str = 'EURUSD'
    profile: str = 'INTRADAY'
    
    # Model paths
    tcn_weights: str = 'models/weights/tcn_best.pt'
    meta_model_path: Optional[str] = 'models/weights/meta_model.joblib'
    exit_model_path: Optional[str] = None
    
    # Timing
    check_interval_seconds: int = 60
    
    # Signal publishing
    enable_signals: bool = True
    publish_to_telegram: bool = True
    publish_to_twitter: bool = True
    publish_to_linkedin: bool = False
    
    # Safety
    conservative_mode: bool = True      # Extra safety margins
    auto_stop_on_limit: bool = True     # Stop when near limits
    close_before_weekend: bool = True   # Close Friday
    avoid_news: bool = True             # Skip high-impact news
    
    # Dry run
    dry_run: bool = True


class PropFirmTradingBot:
    """
    Trading bot specialized for prop firm challenges.
    
    Key Features:
    - Strict adherence to firm rules
    - Conservative risk management
    - Real-time limit monitoring
    - Automated safety stops
    - Signal publishing for track record
    - Challenge progress tracking
    
    Example:
        bot = PropFirmTradingBot(config, data_provider, executor)
        bot.set_notifiers(telegram=tg, twitter=tw)
        bot.initialize()
        bot.run()
    """
    
    def __init__(
        self,
        config: PropFirmBotConfig,
        data_provider,
        executor
    ):
        self.config = config
        self.data_provider = data_provider
        self.executor = executor
        
        # Get prop firm configuration
        self.prop_config = get_prop_firm_config(
            firm=config.firm,
            account_size=config.account_size,
            phase=config.phase,
            conservative=config.conservative_mode
        )
        
        # Create prop firm monitor
        self.monitor = PropFirmMonitor(self.prop_config)
        
        # Components (initialized later)
        self.strategy: Optional[NeuralHybridStrategy] = None
        self.publisher: Optional[SignalPublisher] = None
        self._signal_adapter: Optional[TradingBotSignalAdapter] = None
        self._base_bot: Optional[LiveTradingBot] = None
        
        # State
        self.state = PropFirmBotState.STOPPED
        self._initialized = False
        
        # Tracking
        self._challenge_start = None
        self._trading_days = 0
        self._last_trading_day = None
        
        # Callbacks
        self.on_limit_warning: Optional[Callable] = None
        self.on_challenge_passed: Optional[Callable] = None
        self.on_challenge_failed: Optional[Callable] = None
        
        logger.info(
            f"PropFirmTradingBot created: {self.prop_config.firm_name} "
            f"{self.prop_config.phase.value} ${config.account_size:,.0f}"
        )
    
    def set_notifiers(
        self,
        telegram=None,
        twitter=None,
        linkedin=None
    ):
        """Set notification clients for signal publishing."""
        if self.publisher is None:
            self.publisher = SignalPublisher(PublisherConfig(
                telegram_enabled=self.config.publish_to_telegram and telegram is not None,
                twitter_enabled=self.config.publish_to_twitter and twitter is not None,
                linkedin_enabled=self.config.publish_to_linkedin and linkedin is not None,
                min_confidence_to_publish=0.6,
                min_risk_reward_to_publish=1.5,
            ))
        
        if telegram:
            self.publisher.set_notifier('telegram', telegram)
        if twitter:
            self.publisher.set_notifier('twitter', twitter)
        if linkedin:
            self.publisher.set_notifier('linkedin', linkedin)
    
    def initialize(self) -> bool:
        """Initialize bot components."""
        self.state = PropFirmBotState.INITIALIZING
        
        try:
            # Initialize monitor
            starting_balance = self.config.account_size
            if self.executor and not self.config.dry_run:
                try:
                    starting_balance = self.executor.get_account_balance()
                except:
                    pass
            
            self.monitor.initialize(starting_balance)
            
            # Create strategy with prop firm settings
            strategy_config = StrategyConfig(
                profile=self.config.profile,
                symbol=self.config.symbol,
                tcn_weights=self.config.tcn_weights,
                meta_model_path=self.config.meta_model_path,
                exit_model_path=self.config.exit_model_path,
                # Apply prop firm limits
                base_risk_percent=self.prop_config.max_risk_per_trade_pct,
                min_risk_reward=self.prop_config.min_risk_reward,
                max_open_trades=min(
                    self.prop_config.rules.max_open_positions or 3,
                    3
                ),
                max_daily_trades=self.prop_config.max_trades_per_day,
                # Capital protection with prop firm rules
                enable_capital_protection=True,
                max_daily_loss_percent=self.prop_config.effective_daily_loss_pct,
                max_weekly_loss_percent=self.prop_config.effective_max_drawdown_pct * 0.7,
                max_drawdown_percent=self.prop_config.effective_max_drawdown_pct,
            )
            
            self.strategy = NeuralHybridStrategy(
                config=strategy_config,
                data_provider=self.data_provider,
                executor=self.executor
            )
            self.strategy.initialize(starting_balance)
            
            # Create base bot config
            bot_config = BotConfig(
                symbol=self.config.symbol,
                profile=self.config.profile,
                tcn_weights=self.config.tcn_weights,
                meta_model_path=self.config.meta_model_path,
                exit_model_path=self.config.exit_model_path,
                check_interval_seconds=self.config.check_interval_seconds,
                dry_run=self.config.dry_run,
                enable_exit_advisor=self.config.exit_model_path is not None,
                enable_capital_protection=True,
                **self.prop_config.to_bot_config()
            )
            
            self._base_bot = LiveTradingBot(
                config=bot_config,
                data_provider=self.data_provider,
                executor=self.executor,
                strategy=self.strategy
            )
            self._base_bot.initialize(starting_balance)
            
            # Connect signal publisher
            if self.config.enable_signals and self.publisher:
                self._signal_adapter = TradingBotSignalAdapter(
                    self._base_bot,
                    self.publisher
                )
                self._signal_adapter.connect()
            
            # Set callbacks
            self._base_bot.on_trade = self._on_trade
            self._base_bot.on_protection_event = self._on_protection_event
            
            self._challenge_start = datetime.utcnow()
            self._initialized = True
            self.state = PropFirmBotState.RUNNING
            
            # Publish start message
            if self.publisher:
                self.publisher.publish_alert(
                    f"🚀 Starting {self.prop_config.firm_name} "
                    f"{self.prop_config.phase.value} Challenge\n"
                    f"Account: ${self.config.account_size:,.0f}\n"
                    f"Target: ${self.prop_config.profit_target_amount:,.0f}",
                    priority="high"
                )
            
            logger.info("PropFirmTradingBot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            self.state = PropFirmBotState.ERROR
            return False
    
    def run(self):
        """Run the trading bot (blocking)."""
        if not self._initialized:
            raise RuntimeError("Bot must be initialized first")
        
        logger.info(f"Starting {self.prop_config.firm_name} challenge trading...")
        
        while self.state == PropFirmBotState.RUNNING:
            try:
                current_time = datetime.utcnow()
                
                # Pre-trade checks
                if not self._pre_trade_checks(current_time):
                    self._base_bot._wait(60)
                    continue
                
                # Update monitor with current balance
                self._update_monitor()
                
                # Check prop firm limits
                can_trade, reason = self.monitor.can_trade()
                if not can_trade:
                    if "limit reached" in reason.lower():
                        self.state = PropFirmBotState.DAILY_LIMIT_HIT
                        self._handle_limit_hit(reason)
                    self._base_bot._wait(60)
                    continue
                
                # Check challenge status
                if self.monitor.status.challenge_passed:
                    self.state = PropFirmBotState.CHALLENGE_PASSED
                    self._handle_challenge_passed()
                    break
                
                # Warn if approaching limits
                alerts = self.monitor.get_alerts()
                for alert in alerts:
                    if "WARNING" in alert or "⚠️" in alert:
                        if self.on_limit_warning:
                            self.on_limit_warning(alert)
                        if self.publisher:
                            self.publisher.publish_alert(alert, priority="high")
                
                # Execute single trading cycle
                self._base_bot._evaluate_and_trade(current_time)
                
                # Update trading days
                self._update_trading_days(current_time)
                
                # Wait
                self._base_bot._wait(self.config.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                self._base_bot._wait(60)
        
        logger.info(f"Trading stopped: {self.state.value}")
    
    def stop(self):
        """Stop the bot."""
        self.state = PropFirmBotState.STOPPED
        if self._base_bot:
            self._base_bot.stop()
        
        # Publish stop message
        if self.publisher:
            status = self.get_challenge_status()
            self.publisher.publish_alert(
                f"🛑 Trading Stopped\n{status['summary']}",
                priority="high"
            )
    
    def _pre_trade_checks(self, current_time: datetime) -> bool:
        """Perform pre-trade checks specific to prop firms."""
        # Check market open
        if not self._base_bot._is_market_open(current_time):
            return False
        
        # Check weekend holding rule
        if self.config.close_before_weekend:
            if not self.prop_config.rules.weekend_holding_allowed:
                # Friday after market hours - close all
                if current_time.weekday() == 4 and current_time.hour >= 20:
                    self._close_all_positions("Weekend rule")
                    return False
        
        # Check news blackout
        if self.config.avoid_news and not self.prop_config.rules.news_trading_allowed:
            if self._is_news_time(current_time):
                logger.debug("Skipping - news blackout")
                return False
        
        # Check trading days limit
        if self.prop_config.rules.max_trading_days > 0:
            days_elapsed = (current_time - self._challenge_start).days
            if days_elapsed >= self.prop_config.rules.max_trading_days:
                if not self.monitor.status.challenge_passed:
                    self.state = PropFirmBotState.CHALLENGE_FAILED
                    self._handle_challenge_failed("Time limit exceeded")
                    return False
        
        return True
    
    def _update_monitor(self):
        """Update prop firm monitor with current metrics."""
        # Get current balance
        if self.executor and not self.config.dry_run:
            try:
                balance = self.executor.get_account_balance()
            except:
                balance = self.strategy._current_balance
        else:
            balance = self.strategy._current_balance
        
        # Get P&L
        daily_pnl = self.strategy._daily_pnl
        total_pnl = balance - self.config.account_size
        
        self.monitor.update(
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            balance=balance
        )
    
    def _update_trading_days(self, current_time: datetime):
        """Track trading days."""
        current_date = current_time.date()
        if self._last_trading_day != current_date:
            self._trading_days += 1
            self._last_trading_day = current_date
            
            # Check minimum trading days
            if (self.prop_config.rules.min_trading_days > 0 and
                self._trading_days >= self.prop_config.rules.min_trading_days):
                if self.monitor.status.challenge_passed:
                    logger.info("Minimum trading days met and target reached!")
    
    def _is_news_time(self, current_time: datetime) -> bool:
        """Check if within news blackout period."""
        # Simplified - would need economic calendar integration
        # High impact news typically at:
        # - 8:30 ET (US data)
        # - 10:00 ET (more US data)
        # - 2:00 ET (Fed)
        
        blackout_minutes = self.prop_config.rules.news_blackout_minutes
        if blackout_minutes <= 0:
            return False
        
        # Check common news times (UTC)
        news_times = [
            (13, 30),  # 8:30 ET
            (15, 0),   # 10:00 ET
            (19, 0),   # 2:00 PM ET
        ]
        
        for hour, minute in news_times:
            news_time = current_time.replace(hour=hour, minute=minute, second=0)
            diff = abs((current_time - news_time).total_seconds() / 60)
            if diff <= blackout_minutes:
                return True
        
        return False
    
    def _close_all_positions(self, reason: str):
        """Close all open positions."""
        logger.info(f"Closing all positions: {reason}")
        if self._base_bot:
            self._base_bot.force_close_all(reason)
    
    def _on_trade(self, order, decision_dict):
        """Handle trade execution."""
        self.monitor.record_trade()
        
        # Adjust position size limit for next trade
        max_size = self.monitor.get_position_size_limit(
            sl_pips=decision_dict.get('sl_pips', 20)
        )
        logger.debug(f"Max position size for next trade: {max_size:.2f} lots")
    
    def _on_protection_event(self, event_type: str, data):
        """Handle capital protection events."""
        if event_type == 'kill_switch':
            self.state = PropFirmBotState.CHALLENGE_FAILED
            self._handle_challenge_failed("Kill switch activated")
        elif event_type == 'trade_blocked':
            logger.info(f"Trade blocked by protection: {data}")
    
    def _handle_limit_hit(self, reason: str):
        """Handle when daily/drawdown limit is hit."""
        logger.warning(f"Limit hit: {reason}")
        
        if self.publisher:
            self.publisher.publish_alert(
                f"⚠️ {self.prop_config.firm_name} LIMIT HIT\n{reason}\n"
                f"Trading paused for today.",
                priority="high"
            )
    
    def _handle_challenge_passed(self):
        """Handle challenge completion."""
        logger.info("🎉 CHALLENGE PASSED!")
        
        if self.on_challenge_passed:
            self.on_challenge_passed(self.get_challenge_status())
        
        if self.publisher:
            status = self.get_challenge_status()
            self.publisher.publish_milestone(
                f"🎉 {self.prop_config.firm_name} CHALLENGE PASSED!",
                f"Profit: ${status['total_pnl']:+,.2f}\n"
                f"Trading Days: {status['trading_days']}\n"
                f"Win Rate: {status['win_rate']:.1%}"
            )
    
    def _handle_challenge_failed(self, reason: str):
        """Handle challenge failure."""
        logger.error(f"Challenge failed: {reason}")
        
        if self.on_challenge_failed:
            self.on_challenge_failed(reason)
        
        if self.publisher:
            self.publisher.publish_alert(
                f"❌ {self.prop_config.firm_name} Challenge Failed\n{reason}",
                priority="high"
            )
    
    def get_challenge_status(self) -> Dict:
        """Get comprehensive challenge status."""
        status = self.monitor.status
        
        return {
            'firm': self.prop_config.firm_name,
            'phase': self.prop_config.phase.value,
            'state': self.state.value,
            'account_size': self.config.account_size,
            'current_balance': status.current_balance,
            'total_pnl': status.total_pnl,
            'daily_pnl': status.daily_pnl,
            'profit_progress_pct': status.profit_progress_pct,
            'profit_target': self.prop_config.profit_target_amount,
            'daily_loss_pct': status.daily_loss_pct,
            'max_daily_loss_pct': self.prop_config.effective_daily_loss_pct,
            'drawdown_pct': status.current_drawdown_pct,
            'max_drawdown_pct': self.prop_config.effective_max_drawdown_pct,
            'trading_days': self._trading_days,
            'min_trading_days': self.prop_config.rules.min_trading_days,
            'days_remaining': self.prop_config.rules.max_trading_days - (datetime.utcnow() - self._challenge_start).days if self.prop_config.rules.max_trading_days > 0 else 'unlimited',
            'win_rate': self.strategy._daily_wins / max(1, self.strategy._daily_wins + self.strategy._daily_losses),
            'challenge_passed': status.challenge_passed,
            'summary': status.get_status_message()
        }
    
    def get_status(self) -> Dict:
        """Get bot status."""
        base_status = self._base_bot.get_status() if self._base_bot else {}
        challenge_status = self.get_challenge_status()
        
        return {
            **base_status,
            'challenge': challenge_status,
            'prop_firm': self.prop_config.firm_name,
            'prop_phase': self.prop_config.phase.value
        }


def create_prop_firm_bot(
    firm: str,
    account_size: float,
    phase: str,
    symbol: str,
    data_provider,
    executor,
    profile: str = 'INTRADAY',
    tcn_weights: str = 'models/weights/tcn_best.pt',
    telegram_client=None,
    twitter_client=None,
    linkedin_client=None,
    conservative: bool = True,
    dry_run: bool = True,
    **kwargs
) -> PropFirmTradingBot:
    """
    Factory function to create prop firm trading bot.
    
    Args:
        firm: Prop firm name ('FTMO', 'MyForexFunds', etc.)
        account_size: Challenge account size
        phase: 'evaluation', 'verification', or 'funded'
        symbol: Trading pair
        data_provider: Data provider instance
        executor: Trade executor instance
        profile: Trading profile
        tcn_weights: Path to TCN model weights
        telegram_client: Telegram notifier (optional)
        twitter_client: Twitter notifier (optional)
        linkedin_client: LinkedIn notifier (optional)
        conservative: Use extra safety margins
        dry_run: Paper trading mode
    
    Returns:
        Configured PropFirmTradingBot
    
    Example:
        bot = create_prop_firm_bot(
            firm='FTMO',
            account_size=100000,
            phase='evaluation',
            symbol='EURUSD',
            data_provider=MT5DataProvider(),
            executor=MT5Executor(),
            telegram_client=my_telegram_bot,
            twitter_client=my_twitter_api,
            dry_run=False
        )
        
        bot.run()
    """
    config = PropFirmBotConfig(
        firm=firm,
        account_size=account_size,
        phase=phase,
        symbol=symbol,
        profile=profile,
        tcn_weights=tcn_weights,
        enable_signals=telegram_client or twitter_client or linkedin_client,
        publish_to_telegram=telegram_client is not None,
        publish_to_twitter=twitter_client is not None,
        publish_to_linkedin=linkedin_client is not None,
        conservative_mode=conservative,
        dry_run=dry_run,
        **kwargs
    )
    
    bot = PropFirmTradingBot(
        config=config,
        data_provider=data_provider,
        executor=executor
    )
    
    # Set notifiers
    if telegram_client or twitter_client or linkedin_client:
        bot.set_notifiers(
            telegram=telegram_client,
            twitter=twitter_client,
            linkedin=linkedin_client
        )
    
    bot.initialize()
    
    return bot
