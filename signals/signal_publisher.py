# signals/signal_publisher.py
"""
Signal Publishing Module for pyForex

Publishes trading signals to multiple platforms:
- Telegram (real-time alerts)
- Twitter/X (public signals)
- LinkedIn (professional updates)
- MQL5 Signals (copy trading)
- Webhook (custom integrations)
- Email (subscriber list)

Features:
- Configurable signal formats per platform
- Rate limiting and queuing
- Performance tracking
- Signal history
- Subscriber management

Usage:
    from signals.signal_publisher import SignalPublisher, SignalConfig
    
    publisher = SignalPublisher(config)
    publisher.publish_signal(trade_decision)
    publisher.publish_trade_closed(ticket, pnl)
    publisher.publish_daily_summary()
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Platform(Enum):
    """Supported publishing platforms."""
    TELEGRAM = "telegram"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    MQL5 = "mql5"
    WEBHOOK = "webhook"
    EMAIL = "email"


class SignalType(Enum):
    """Types of signals to publish."""
    ENTRY = "entry"
    EXIT = "exit"
    UPDATE = "update"           # SL/TP adjustment
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_SUMMARY = "weekly_summary"
    ALERT = "alert"            # Warnings, milestones


@dataclass
class Signal:
    """Trading signal data structure."""
    signal_id: str
    signal_type: SignalType
    timestamp: datetime
    
    # Trade details
    symbol: str = ""
    direction: str = ""          # BUY, SELL
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    
    # Risk info
    risk_percent: float = 0.0
    risk_reward_ratio: float = 0.0
    position_size: float = 0.0
    
    # ML confidence
    confidence: float = 0.0
    meta_score: float = 0.0
    
    # Exit details (for closed trades)
    exit_price: float = 0.0
    pnl: float = 0.0
    pnl_pips: float = 0.0
    
    # Context
    regime: str = ""
    notes: str = ""
    
    # Metadata
    published_to: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'signal_id': self.signal_id,
            'signal_type': self.signal_type.value,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'risk_percent': self.risk_percent,
            'risk_reward_ratio': self.risk_reward_ratio,
            'confidence': self.confidence,
            'pnl': self.pnl,
            'notes': self.notes
        }


@dataclass
class SignalPerformance:
    """Track signal performance."""
    total_signals: int = 0
    winning_signals: int = 0
    losing_signals: int = 0
    total_pnl: float = 0.0
    total_pips: float = 0.0
    
    @property
    def win_rate(self) -> float:
        closed = self.winning_signals + self.losing_signals
        return self.winning_signals / closed if closed > 0 else 0.0
    
    @property
    def average_pnl(self) -> float:
        closed = self.winning_signals + self.losing_signals
        return self.total_pnl / closed if closed > 0 else 0.0


@dataclass
class PublisherConfig:
    """Configuration for signal publisher."""
    # Enable/disable platforms
    telegram_enabled: bool = True
    twitter_enabled: bool = True
    linkedin_enabled: bool = False  # Less frequent
    mql5_enabled: bool = False
    webhook_enabled: bool = False
    email_enabled: bool = False
    
    # Platform credentials (from environment or config)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    twitter_api_key: str = ""
    twitter_api_secret: str = ""
    twitter_access_token: str = ""
    twitter_access_secret: str = ""
    linkedin_access_token: str = ""
    webhook_url: str = ""
    
    # Publishing rules
    publish_entries: bool = True
    publish_exits: bool = True
    publish_updates: bool = False   # SL/TP changes
    publish_daily_summary: bool = True
    
    # Content settings
    include_prices: bool = True     # Some hide exact prices
    include_sl_tp: bool = True
    include_confidence: bool = False  # ML details
    include_position_size: bool = False  # Privacy
    
    # Rate limits
    min_seconds_between_signals: int = 60
    max_signals_per_hour: int = 10
    max_signals_per_day: int = 50
    
    # Filtering
    min_confidence_to_publish: float = 0.6
    min_risk_reward_to_publish: float = 1.5


# =============================================================================
# SIGNAL FORMATTERS
# =============================================================================

class SignalFormatter(ABC):
    """Base class for platform-specific signal formatting."""
    
    @abstractmethod
    def format_entry(self, signal: Signal, config: PublisherConfig) -> str:
        pass
    
    @abstractmethod
    def format_exit(self, signal: Signal, config: PublisherConfig) -> str:
        pass
    
    @abstractmethod
    def format_daily_summary(
        self,
        performance: SignalPerformance,
        signals: List[Signal],
        config: PublisherConfig
    ) -> str:
        pass


class TelegramFormatter(SignalFormatter):
    """Telegram-specific formatting with emojis and markdown."""
    
    def format_entry(self, signal: Signal, config: PublisherConfig) -> str:
        direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
        
        msg = f"{direction_emoji} *{signal.direction} {signal.symbol}*\n"
        msg += f"━━━━━━━━━━━━━━━\n"
        
        if config.include_prices:
            msg += f"📍 Entry: `{signal.entry_price:.5f}`\n"
        
        if config.include_sl_tp:
            msg += f"🛑 SL: `{signal.stop_loss:.5f}`\n"
            msg += f"🎯 TP: `{signal.take_profit:.5f}`\n"
        
        msg += f"📊 R:R: `{signal.risk_reward_ratio:.2f}`\n"
        
        if config.include_confidence:
            msg += f"🤖 Confidence: `{signal.confidence:.1%}`\n"
        
        if config.include_position_size:
            msg += f"📐 Size: `{signal.position_size:.2f}` lots\n"
        
        msg += f"━━━━━━━━━━━━━━━\n"
        msg += f"⏰ {signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}"
        
        if signal.notes:
            msg += f"\n📝 {signal.notes}"
        
        return msg
    
    def format_exit(self, signal: Signal, config: PublisherConfig) -> str:
        pnl_emoji = "✅" if signal.pnl >= 0 else "❌"
        
        msg = f"{pnl_emoji} *{signal.symbol} CLOSED*\n"
        msg += f"━━━━━━━━━━━━━━━\n"
        
        if config.include_prices:
            msg += f"📍 Entry: `{signal.entry_price:.5f}`\n"
            msg += f"📍 Exit: `{signal.exit_price:.5f}`\n"
        
        msg += f"💰 P&L: `{signal.pnl:+.2f}` ({signal.pnl_pips:+.1f} pips)\n"
        msg += f"━━━━━━━━━━━━━━━\n"
        msg += f"⏰ {signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}"
        
        return msg
    
    def format_daily_summary(
        self,
        performance: SignalPerformance,
        signals: List[Signal],
        config: PublisherConfig
    ) -> str:
        pnl_emoji = "📈" if performance.total_pnl >= 0 else "📉"
        
        msg = f"{pnl_emoji} *DAILY SUMMARY*\n"
        msg += f"━━━━━━━━━━━━━━━\n"
        msg += f"📊 Signals: {performance.total_signals}\n"
        msg += f"✅ Wins: {performance.winning_signals}\n"
        msg += f"❌ Losses: {performance.losing_signals}\n"
        msg += f"🎯 Win Rate: `{performance.win_rate:.1%}`\n"
        msg += f"━━━━━━━━━━━━━━━\n"
        msg += f"💰 Total P&L: `{performance.total_pnl:+.2f}`\n"
        msg += f"📏 Total Pips: `{performance.total_pips:+.1f}`\n"
        msg += f"━━━━━━━━━━━━━━━\n"
        msg += f"📅 {datetime.utcnow().strftime('%Y-%m-%d')}"
        
        return msg


class TwitterFormatter(SignalFormatter):
    """Twitter-specific formatting (280 char limit)."""
    
    def format_entry(self, signal: Signal, config: PublisherConfig) -> str:
        direction_emoji = "🟢" if signal.direction == "BUY" else "🔴"
        
        # Concise format for Twitter
        msg = f"{direction_emoji} #{signal.symbol} {signal.direction}\n"
        
        if config.include_prices:
            msg += f"Entry: {signal.entry_price:.5f}\n"
        
        if config.include_sl_tp:
            sl_pips = abs(signal.entry_price - signal.stop_loss) * 10000
            tp_pips = abs(signal.take_profit - signal.entry_price) * 10000
            msg += f"SL: {sl_pips:.0f} pips | TP: {tp_pips:.0f} pips\n"
        
        msg += f"R:R {signal.risk_reward_ratio:.1f}\n"
        msg += f"#forex #trading #signals"
        
        return msg[:280]  # Twitter limit
    
    def format_exit(self, signal: Signal, config: PublisherConfig) -> str:
        pnl_emoji = "✅" if signal.pnl >= 0 else "❌"
        
        msg = f"{pnl_emoji} #{signal.symbol} Closed\n"
        msg += f"P&L: {signal.pnl_pips:+.0f} pips\n"
        msg += f"#forex #trading #results"
        
        return msg[:280]
    
    def format_daily_summary(
        self,
        performance: SignalPerformance,
        signals: List[Signal],
        config: PublisherConfig
    ) -> str:
        pnl_emoji = "📈" if performance.total_pnl >= 0 else "📉"
        
        msg = f"{pnl_emoji} Daily Results\n"
        msg += f"Trades: {performance.total_signals}\n"
        msg += f"Win Rate: {performance.win_rate:.0%}\n"
        msg += f"P&L: {performance.total_pips:+.0f} pips\n"
        msg += f"#forex #trading #performance"
        
        return msg[:280]


class LinkedInFormatter(SignalFormatter):
    """LinkedIn-specific formatting (professional tone)."""
    
    def format_entry(self, signal: Signal, config: PublisherConfig) -> str:
        # LinkedIn is for weekly/monthly summaries, not individual trades
        return ""
    
    def format_exit(self, signal: Signal, config: PublisherConfig) -> str:
        return ""
    
    def format_daily_summary(
        self,
        performance: SignalPerformance,
        signals: List[Signal],
        config: PublisherConfig
    ) -> str:
        msg = f"📊 Algorithmic Trading Performance Update\n\n"
        msg += f"Today's Results:\n"
        msg += f"• Total Signals: {performance.total_signals}\n"
        msg += f"• Win Rate: {performance.win_rate:.1%}\n"
        msg += f"• Net Result: {performance.total_pips:+.1f} pips\n\n"
        
        if performance.win_rate >= 0.5:
            msg += "Consistent execution with ML-driven risk management continues to deliver results.\n\n"
        else:
            msg += "Market conditions were challenging today. Risk management protected capital.\n\n"
        
        msg += "#AlgorithmicTrading #Forex #MachineLearning #Trading #FinTech"
        
        return msg


class WebhookFormatter(SignalFormatter):
    """JSON format for webhook integrations."""
    
    def format_entry(self, signal: Signal, config: PublisherConfig) -> str:
        return json.dumps(signal.to_dict())
    
    def format_exit(self, signal: Signal, config: PublisherConfig) -> str:
        return json.dumps(signal.to_dict())
    
    def format_daily_summary(
        self,
        performance: SignalPerformance,
        signals: List[Signal],
        config: PublisherConfig
    ) -> str:
        return json.dumps({
            'type': 'daily_summary',
            'date': datetime.utcnow().isoformat(),
            'total_signals': performance.total_signals,
            'wins': performance.winning_signals,
            'losses': performance.losing_signals,
            'win_rate': performance.win_rate,
            'total_pnl': performance.total_pnl,
            'total_pips': performance.total_pips,
            'signals': [s.to_dict() for s in signals]
        })


FORMATTERS = {
    Platform.TELEGRAM: TelegramFormatter(),
    Platform.TWITTER: TwitterFormatter(),
    Platform.LINKEDIN: LinkedInFormatter(),
    Platform.WEBHOOK: WebhookFormatter(),
}


# =============================================================================
# SIGNAL PUBLISHER
# =============================================================================

class SignalPublisher:
    """
    Multi-platform signal publishing system.
    
    Integrates with existing notification modules to publish trading signals
    across multiple platforms with appropriate formatting.
    
    Usage:
        # Initialize
        publisher = SignalPublisher(config)
        publisher.set_notifier(telegram=tg_client, twitter=tw_client)
        
        # Publish entry signal
        publisher.publish_entry(
            symbol='EURUSD',
            direction='BUY',
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            confidence=0.75
        )
        
        # Publish exit
        publisher.publish_exit(
            symbol='EURUSD',
            entry_price=1.1000,
            exit_price=1.1080,
            pnl=800.0,
            pnl_pips=80
        )
        
        # Daily summary
        publisher.publish_daily_summary()
    """
    
    def __init__(self, config: Optional[PublisherConfig] = None):
        self.config = config or PublisherConfig()
        
        # Notifier clients (set externally)
        self._notifiers: Dict[Platform, Any] = {}
        
        # Signal tracking
        self._signal_counter = 0
        self._signals_today: List[Signal] = []
        self._signal_history: deque = deque(maxlen=1000)
        self._performance = SignalPerformance()
        self._daily_performance = SignalPerformance()
        
        # Rate limiting
        self._last_publish_time: Dict[Platform, datetime] = {}
        self._hourly_count = 0
        self._daily_count = 0
        self._hour_start = datetime.utcnow()
        self._day_start = datetime.utcnow().date()
        
        # Async publishing
        self._publish_queue: deque = deque()
        self._publisher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        logger.info("SignalPublisher initialized")
    
    def set_notifier(self, platform: str, client: Any):
        """
        Set notifier client for a platform.
        
        Args:
            platform: 'telegram', 'twitter', 'linkedin', etc.
            client: Platform-specific client object
        """
        try:
            plat = Platform(platform.lower())
            self._notifiers[plat] = client
            logger.info(f"Notifier set for {platform}")
        except ValueError:
            logger.warning(f"Unknown platform: {platform}")
    
    def set_notifiers(self, **kwargs):
        """Set multiple notifiers at once."""
        for platform, client in kwargs.items():
            self.set_notifier(platform, client)
    
    def publish_entry(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float = 0.0,
        meta_score: float = 0.0,
        risk_percent: float = 0.0,
        position_size: float = 0.0,
        regime: str = "",
        notes: str = ""
    ) -> Optional[Signal]:
        """
        Publish entry signal.
        
        Returns:
            Signal object if published, None if filtered/rate-limited
        """
        if not self.config.publish_entries:
            return None
        
        # Check filters
        if confidence < self.config.min_confidence_to_publish:
            logger.debug(f"Signal filtered: low confidence {confidence:.2f}")
            return None
        
        # Calculate R:R
        if direction == "BUY":
            sl_distance = entry_price - stop_loss
            tp_distance = take_profit - entry_price
        else:
            sl_distance = stop_loss - entry_price
            tp_distance = entry_price - take_profit
        
        risk_reward = tp_distance / sl_distance if sl_distance > 0 else 0
        
        if risk_reward < self.config.min_risk_reward_to_publish:
            logger.debug(f"Signal filtered: low R:R {risk_reward:.2f}")
            return None
        
        # Check rate limits
        if not self._check_rate_limits():
            logger.debug("Signal rate-limited")
            return None
        
        # Create signal
        self._signal_counter += 1
        signal = Signal(
            signal_id=f"SIG-{datetime.utcnow().strftime('%Y%m%d')}-{self._signal_counter:04d}",
            signal_type=SignalType.ENTRY,
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_percent=risk_percent,
            risk_reward_ratio=risk_reward,
            position_size=position_size,
            confidence=confidence,
            meta_score=meta_score,
            regime=regime,
            notes=notes
        )
        
        # Publish to enabled platforms
        self._publish_signal(signal, SignalType.ENTRY)
        
        # Track
        self._signals_today.append(signal)
        self._signal_history.append(signal)
        self._performance.total_signals += 1
        self._daily_performance.total_signals += 1
        self._daily_count += 1
        self._hourly_count += 1
        
        logger.info(f"Entry signal published: {signal.signal_id} {direction} {symbol}")
        
        return signal
    
    def publish_exit(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pips: float,
        direction: str = "",
        notes: str = ""
    ) -> Optional[Signal]:
        """Publish exit/close signal."""
        if not self.config.publish_exits:
            return None
        
        self._signal_counter += 1
        signal = Signal(
            signal_id=f"EXIT-{datetime.utcnow().strftime('%Y%m%d')}-{self._signal_counter:04d}",
            signal_type=SignalType.EXIT,
            timestamp=datetime.utcnow(),
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pips=pnl_pips,
            notes=notes
        )
        
        # Update performance
        if pnl >= 0:
            self._performance.winning_signals += 1
            self._daily_performance.winning_signals += 1
        else:
            self._performance.losing_signals += 1
            self._daily_performance.losing_signals += 1
        
        self._performance.total_pnl += pnl
        self._performance.total_pips += pnl_pips
        self._daily_performance.total_pnl += pnl
        self._daily_performance.total_pips += pnl_pips
        
        # Publish
        self._publish_signal(signal, SignalType.EXIT)
        self._signal_history.append(signal)
        
        logger.info(f"Exit signal published: {symbol} P&L: {pnl:+.2f}")
        
        return signal
    
    def publish_daily_summary(self) -> bool:
        """Publish daily performance summary."""
        if not self.config.publish_daily_summary:
            return False
        
        if self._daily_performance.total_signals == 0:
            logger.debug("No signals today, skipping summary")
            return False
        
        # Publish to each platform
        for platform, notifier in self._notifiers.items():
            if not self._is_platform_enabled(platform):
                continue
            
            formatter = FORMATTERS.get(platform)
            if not formatter:
                continue
            
            try:
                message = formatter.format_daily_summary(
                    self._daily_performance,
                    self._signals_today,
                    self.config
                )
                
                if message:
                    self._send_to_platform(platform, notifier, message)
                    
            except Exception as e:
                logger.error(f"Failed to publish summary to {platform.value}: {e}")
        
        logger.info(f"Daily summary published: {self._daily_performance.total_signals} signals")
        
        return True
    
    def publish_alert(self, message: str, priority: str = "normal"):
        """Publish custom alert message."""
        for platform, notifier in self._notifiers.items():
            if not self._is_platform_enabled(platform):
                continue
            
            # Format alert
            if platform == Platform.TELEGRAM:
                if priority == "high":
                    formatted = f"🚨 *ALERT*\n\n{message}"
                else:
                    formatted = f"ℹ️ {message}"
            else:
                formatted = f"Alert: {message}"
            
            try:
                self._send_to_platform(platform, notifier, formatted)
            except Exception as e:
                logger.error(f"Failed to publish alert to {platform.value}: {e}")
    
    def publish_milestone(self, milestone: str, details: str = ""):
        """Publish milestone achievement."""
        message = f"🏆 MILESTONE: {milestone}"
        if details:
            message += f"\n{details}"
        
        self.publish_alert(message, priority="high")
    
    def reset_daily(self):
        """Reset daily counters (call at start of day)."""
        self._signals_today = []
        self._daily_performance = SignalPerformance()
        self._daily_count = 0
        self._day_start = datetime.utcnow().date()
        logger.info("Daily signal counters reset")
    
    def get_performance(self) -> SignalPerformance:
        """Get overall performance."""
        return self._performance
    
    def get_daily_performance(self) -> SignalPerformance:
        """Get today's performance."""
        return self._daily_performance
    
    def get_recent_signals(self, count: int = 10) -> List[Signal]:
        """Get recent signals."""
        return list(self._signal_history)[-count:]
    
    def _publish_signal(self, signal: Signal, signal_type: SignalType):
        """Publish signal to all enabled platforms."""
        for platform, notifier in self._notifiers.items():
            if not self._is_platform_enabled(platform):
                continue
            
            formatter = FORMATTERS.get(platform)
            if not formatter:
                continue
            
            try:
                if signal_type == SignalType.ENTRY:
                    message = formatter.format_entry(signal, self.config)
                elif signal_type == SignalType.EXIT:
                    message = formatter.format_exit(signal, self.config)
                else:
                    continue
                
                if message:
                    self._send_to_platform(platform, notifier, message)
                    signal.published_to.append(platform.value)
                    
            except Exception as e:
                logger.error(f"Failed to publish to {platform.value}: {e}")
    
    def _send_to_platform(self, platform: Platform, notifier: Any, message: str):
        """Send message to specific platform."""
        try:
            # Handle different notifier interfaces
            if hasattr(notifier, 'send_message'):
                notifier.send_message(message)
            elif hasattr(notifier, 'post'):
                notifier.post(message)
            elif hasattr(notifier, 'publish'):
                notifier.publish(message)
            elif callable(notifier):
                notifier(message)
            else:
                logger.warning(f"Unknown notifier interface for {platform.value}")
                
            self._last_publish_time[platform] = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Send to {platform.value} failed: {e}")
    
    def _is_platform_enabled(self, platform: Platform) -> bool:
        """Check if platform is enabled."""
        enabled_map = {
            Platform.TELEGRAM: self.config.telegram_enabled,
            Platform.TWITTER: self.config.twitter_enabled,
            Platform.LINKEDIN: self.config.linkedin_enabled,
            Platform.MQL5: self.config.mql5_enabled,
            Platform.WEBHOOK: self.config.webhook_enabled,
            Platform.EMAIL: self.config.email_enabled,
        }
        return enabled_map.get(platform, False)
    
    def _check_rate_limits(self) -> bool:
        """Check if within rate limits."""
        now = datetime.utcnow()

        # Global minimum spacing between any two signals
        if self.config.min_seconds_between_signals > 0 and self._last_publish_time:
            last_any = max(self._last_publish_time.values())
            if (now - last_any).total_seconds() < self.config.min_seconds_between_signals:
                return False
        
        # Reset hourly counter
        if (now - self._hour_start).total_seconds() >= 3600:
            self._hourly_count = 0
            self._hour_start = now
        
        # Reset daily counter
        if now.date() != self._day_start:
            self._daily_count = 0
            self._day_start = now.date()
        
        # Check limits
        if self._hourly_count >= self.config.max_signals_per_hour:
            return False
        if self._daily_count >= self.config.max_signals_per_day:
            return False
        
        return True


# =============================================================================
# INTEGRATION WITH TRADING BOT
# =============================================================================

def create_signal_publisher_for_bot(
    telegram_client=None,
    twitter_client=None,
    linkedin_client=None,
    config: Optional[PublisherConfig] = None
) -> SignalPublisher:
    """
    Factory function to create signal publisher with notifier clients.
    
    Args:
        telegram_client: Your existing Telegram notifier
        twitter_client: Your existing Twitter/X notifier
        linkedin_client: Your existing LinkedIn notifier
        config: Publisher configuration
    
    Returns:
        Configured SignalPublisher
    """
    publisher = SignalPublisher(config or PublisherConfig())
    
    if telegram_client:
        publisher.set_notifier('telegram', telegram_client)
    if twitter_client:
        publisher.set_notifier('twitter', twitter_client)
    if linkedin_client:
        publisher.set_notifier('linkedin', linkedin_client)
    
    return publisher


class TradingBotSignalAdapter:
    """
    Adapter to integrate SignalPublisher with LiveTradingBot.
    
    Automatically publishes signals when bot trades.
    
    Usage:
        bot = LiveTradingBot(config)
        publisher = SignalPublisher(pub_config)
        
        adapter = TradingBotSignalAdapter(bot, publisher)
        adapter.connect()
        
        bot.run()  # Signals auto-published
    """
    
    def __init__(
        self,
        bot,
        publisher: SignalPublisher,
        publish_entries: bool = True,
        publish_exits: bool = True
    ):
        self.bot = bot
        self.publisher = publisher
        self.publish_entries = publish_entries
        self.publish_exits = publish_exits
        
        # Store original callbacks
        self._original_on_trade = None
        self._original_on_exit = None
    
    def connect(self):
        """Connect adapter to bot callbacks."""
        # Store originals
        self._original_on_trade = self.bot.on_trade
        self._original_on_exit = getattr(self.bot, 'on_exit', None)
        
        # Set new callbacks
        self.bot.on_trade = self._on_trade_callback
        if hasattr(self.bot, 'on_exit'):
            self.bot.on_exit = self._on_exit_callback
    
    def disconnect(self):
        """Restore original callbacks."""
        if self._original_on_trade:
            self.bot.on_trade = self._original_on_trade
        if self._original_on_exit:
            self.bot.on_exit = self._original_on_exit
    
    def _on_trade_callback(self, order, decision_dict):
        """Callback for new trades."""
        if self.publish_entries:
            self.publisher.publish_entry(
                symbol=order.symbol,
                direction=order.direction,
                entry_price=order.price or decision_dict.get('entry_price', 0),
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                confidence=decision_dict.get('direction_confidence', 0),
                meta_score=decision_dict.get('meta_score', 0),
                risk_percent=decision_dict.get('risk_percent', 0),
                position_size=order.volume,
                regime=decision_dict.get('regime', ''),
                notes=f"R:R {decision_dict.get('risk_reward_ratio', 0):.2f}"
            )
        
        # Call original if exists
        if self._original_on_trade:
            self._original_on_trade(order, decision_dict)
    
    def _on_exit_callback(self, ticket, pnl, reason):
        """Callback for closed trades."""
        if self.publish_exits:
            # Get position info if available
            position = self.bot._open_positions.get(ticket) if hasattr(self.bot, '_open_positions') else None
            
            self.publisher.publish_exit(
                symbol=position.symbol if position else self.bot.config.symbol,
                entry_price=position.entry_price if position else 0,
                exit_price=0,  # Would need from executor
                pnl=pnl,
                pnl_pips=pnl / 10,  # Approximate
                direction='BUY' if position and position.direction == 1 else 'SELL',
                notes=f"Exit reason: {reason}"
            )
        
        # Call original if exists
        if self._original_on_exit:
            self._original_on_exit(ticket, pnl, reason)
