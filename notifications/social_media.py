"""
Social Media Notifications Module for pyForex Trading System.

Provides automated posting to Twitter/X, LinkedIn, and Telegram
for trade results, performance updates, and system alerts.

Usage:
    from notifications import SocialMediaNotifier, NotificationConfig
    
    config = NotificationConfig(
        telegram_bot_token="YOUR_BOT_TOKEN",
        telegram_chat_id="YOUR_CHAT_ID",
        twitter_api_key="YOUR_API_KEY",
        # ... other credentials
    )
    
    notifier = SocialMediaNotifier(config)
    notifier.post_trade_result(trade)
    notifier.post_performance_update(metrics)
"""

import os
import json
import logging
import requests
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib
import time

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class Platform(Enum):
    """Supported social media platforms."""
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    TELEGRAM = "telegram"


class PostType(Enum):
    """Types of posts."""
    TRADE_RESULT = "trade_result"
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_SUMMARY = "weekly_summary"
    PERFORMANCE_UPDATE = "performance_update"
    SYSTEM_ALERT = "system_alert"
    MODEL_UPDATE = "model_update"
    MILESTONE = "milestone"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TelegramConfig:
    """Telegram Bot configuration."""
    bot_token: str = ""
    chat_id: str = ""  # Can be channel (@channel) or chat ID
    enabled: bool = True
    
    # Posting rules
    post_trades: bool = True
    post_daily_summary: bool = True
    post_alerts: bool = True
    min_trade_pnl_to_post: float = 0.0  # Only post trades above this PnL
    
    # Rate limiting
    max_posts_per_hour: int = 30
    
    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)


@dataclass
class TwitterConfig:
    """Twitter/X API configuration (v2 API)."""
    api_key: str = ""
    api_secret: str = ""
    access_token: str = ""
    access_token_secret: str = ""
    bearer_token: str = ""
    enabled: bool = True
    
    # Posting rules
    post_trades: bool = False  # Individual trades can be noisy
    post_daily_summary: bool = True
    post_weekly_summary: bool = True
    post_milestones: bool = True
    min_trade_pnl_to_post: float = 50.0  # Only post significant trades
    
    # Content settings
    include_hashtags: bool = True
    default_hashtags: List[str] = field(
        default_factory=lambda: ["algotrading", "forex", "trading", "quant"]
    )
    
    # Rate limiting (Twitter has strict limits)
    max_posts_per_day: int = 50
    min_minutes_between_posts: int = 15
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret and 
                   self.access_token and self.access_token_secret)


@dataclass
class LinkedInConfig:
    """LinkedIn API configuration."""
    access_token: str = ""
    person_urn: str = ""  # Format: "urn:li:person:XXXXXXX"
    organization_urn: str = ""  # Optional: for company pages
    enabled: bool = True
    
    # Posting rules (LinkedIn = professional, less frequent)
    post_trades: bool = False
    post_daily_summary: bool = False
    post_weekly_summary: bool = True
    post_monthly_summary: bool = True
    post_milestones: bool = True
    
    # Content settings
    include_hashtags: bool = True
    default_hashtags: List[str] = field(
        default_factory=lambda: ["algorithmictrading", "fintech", "forex", "machinelearning"]
    )
    
    # Rate limiting
    max_posts_per_day: int = 3
    
    @property
    def is_configured(self) -> bool:
        return bool(self.access_token and (self.person_urn or self.organization_urn))


@dataclass
class NotificationConfig:
    """Master configuration for all social media notifications."""
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    twitter: TwitterConfig = field(default_factory=TwitterConfig)
    linkedin: LinkedInConfig = field(default_factory=LinkedInConfig)
    
    # Global settings
    dry_run: bool = False  # If True, don't actually post
    log_posts: bool = True
    posts_log_file: str = "logs/social_posts.json"
    
    # Branding
    bot_name: str = "pyForex"
    include_disclaimer: bool = True
    disclaimer_text: str = "Past performance ≠ future results. Not financial advice."
    
    # Image settings
    attach_charts: bool = True
    chart_directory: str = "./charts"
    
    @classmethod
    def from_env(cls) -> 'NotificationConfig':
        """Load configuration from environment variables."""
        return cls(
            telegram=TelegramConfig(
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
                chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            ),
            twitter=TwitterConfig(
                api_key=os.getenv("TWITTER_API_KEY", ""),
                api_secret=os.getenv("TWITTER_API_SECRET", ""),
                access_token=os.getenv("TWITTER_ACCESS_TOKEN", ""),
                access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET", ""),
                bearer_token=os.getenv("TWITTER_BEARER_TOKEN", ""),
            ),
            linkedin=LinkedInConfig(
                access_token=os.getenv("LINKEDIN_ACCESS_TOKEN", ""),
                person_urn=os.getenv("LINKEDIN_PERSON_URN", ""),
            ),
        )
    
    @classmethod
    def from_file(cls, filepath: str) -> 'NotificationConfig':
        """Load configuration from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        
        return cls(
            telegram=TelegramConfig(**data.get('telegram', {})),
            twitter=TwitterConfig(**data.get('twitter', {})),
            linkedin=LinkedInConfig(**data.get('linkedin', {})),
            dry_run=data.get('dry_run', False),
            bot_name=data.get('bot_name', 'pyForex'),
        )


# =============================================================================
# DATA CLASSES FOR CONTENT
# =============================================================================

@dataclass
class TradeData:
    """Trade data for posting."""
    trade_id: str
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    duration_minutes: int
    signal_confidence: Optional[float] = None
    
    @property
    def is_winner(self) -> bool:
        return self.pnl > 0
    
    @property
    def emoji(self) -> str:
        if self.pnl_pct >= 1.0:
            return "🚀"
        elif self.pnl > 0:
            return "✅"
        elif self.pnl_pct <= -1.0:
            return "💥"
        else:
            return "❌"


@dataclass
class PerformanceData:
    """Performance metrics for posting."""
    period: str  # "daily", "weekly", "monthly"
    start_date: datetime
    end_date: datetime
    
    # Core metrics
    total_trades: int
    winning_trades: int
    total_pnl: float
    total_pnl_pct: float
    
    # Advanced metrics
    win_rate: float
    profit_factor: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    avg_trade_pnl: Optional[float] = None
    best_trade_pnl: Optional[float] = None
    worst_trade_pnl: Optional[float] = None
    
    @property
    def is_profitable(self) -> bool:
        return self.total_pnl > 0
    
    @property
    def emoji(self) -> str:
        if self.total_pnl_pct >= 5.0:
            return "🔥"
        elif self.total_pnl_pct >= 1.0:
            return "📈"
        elif self.total_pnl_pct >= 0:
            return "➡️"
        elif self.total_pnl_pct >= -1.0:
            return "📉"
        else:
            return "⚠️"


# =============================================================================
# BASE PLATFORM CLASS
# =============================================================================

class BasePlatform(ABC):
    """Abstract base class for social media platforms."""
    
    def __init__(self, config: Any):
        self.config = config
        self.post_history: List[Dict] = []
        self.last_post_time: Optional[datetime] = None
    
    @property
    @abstractmethod
    def platform_name(self) -> str:
        """Return platform name."""
        pass
    
    @abstractmethod
    def post_text(self, text: str) -> bool:
        """Post text content."""
        pass
    
    @abstractmethod
    def post_with_image(self, text: str, image_path: str) -> bool:
        """Post text with image."""
        pass
    
    def can_post(self) -> bool:
        """Check if posting is allowed (rate limiting)."""
        return True
    
    def log_post(self, content: str, success: bool, post_type: PostType):
        """Log post for history."""
        self.post_history.append({
            'timestamp': datetime.now().isoformat(),
            'platform': self.platform_name,
            'content': content[:100] + "..." if len(content) > 100 else content,
            'success': success,
            'post_type': post_type.value,
        })
        if success:
            self.last_post_time = datetime.now()


# =============================================================================
# TELEGRAM IMPLEMENTATION
# =============================================================================

class TelegramPlatform(BasePlatform):
    """Telegram Bot implementation."""
    
    BASE_URL = "https://api.telegram.org/bot"
    
    def __init__(self, config: TelegramConfig):
        super().__init__(config)
        self.posts_this_hour = 0
        self.hour_start = datetime.now()
    
    @property
    def platform_name(self) -> str:
        return "telegram"
    
    def _get_url(self, method: str) -> str:
        return f"{self.BASE_URL}{self.config.bot_token}/{method}"
    
    def can_post(self) -> bool:
        """Check rate limits."""
        now = datetime.now()
        
        # Reset hourly counter
        if (now - self.hour_start).seconds >= 3600:
            self.posts_this_hour = 0
            self.hour_start = now
        
        return self.posts_this_hour < self.config.max_posts_per_hour
    
    def post_text(self, text: str, parse_mode: str = "HTML") -> bool:
        """Post text message to Telegram."""
        if not self.config.is_configured:
            logger.warning("Telegram not configured")
            return False
        
        if not self.can_post():
            logger.warning("Telegram rate limit reached")
            return False
        
        try:
            response = requests.post(
                self._get_url("sendMessage"),
                json={
                    "chat_id": self.config.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            
            success = response.status_code == 200
            if success:
                self.posts_this_hour += 1
                logger.info(f"Telegram post successful")
            else:
                logger.error(f"Telegram post failed: {response.text}")
            
            return success
            
        except Exception as e:
            logger.error(f"Telegram post error: {e}")
            return False
    
    def post_with_image(self, text: str, image_path: str) -> bool:
        """Post message with photo to Telegram."""
        if not self.config.is_configured:
            return False
        
        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            return self.post_text(text)
        
        try:
            with open(image_path, 'rb') as photo:
                response = requests.post(
                    self._get_url("sendPhoto"),
                    data={
                        "chat_id": self.config.chat_id,
                        "caption": text,
                        "parse_mode": "HTML",
                    },
                    files={"photo": photo},
                    timeout=30,
                )
            
            success = response.status_code == 200
            if success:
                self.posts_this_hour += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Telegram photo post error: {e}")
            return False
    
    def send_document(self, filepath: str, caption: str = "") -> bool:
        """Send document/file to Telegram."""
        if not self.config.is_configured:
            return False
        
        try:
            with open(filepath, 'rb') as doc:
                response = requests.post(
                    self._get_url("sendDocument"),
                    data={
                        "chat_id": self.config.chat_id,
                        "caption": caption,
                    },
                    files={"document": doc},
                    timeout=60,
                )
            
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Telegram document send error: {e}")
            return False


# =============================================================================
# TWITTER IMPLEMENTATION
# =============================================================================

class TwitterPlatform(BasePlatform):
    """Twitter/X API v2 implementation."""
    
    TWEET_URL = "https://api.twitter.com/2/tweets"
    MEDIA_UPLOAD_URL = "https://upload.twitter.com/1.1/media/upload.json"
    
    def __init__(self, config: TwitterConfig):
        super().__init__(config)
        self.posts_today = 0
        self.day_start = datetime.now().date()
    
    @property
    def platform_name(self) -> str:
        return "twitter"
    
    def _get_oauth_header(self) -> Dict[str, str]:
        """Generate OAuth 1.0a header for Twitter API."""
        # For simplicity, using requests-oauthlib
        try:
            from requests_oauthlib import OAuth1
            self._oauth = OAuth1(
                self.config.api_key,
                self.config.api_secret,
                self.config.access_token,
                self.config.access_token_secret,
            )
            return {}
        except ImportError:
            logger.error("requests-oauthlib required for Twitter. Install with: pip install requests-oauthlib")
            return {}
    
    def can_post(self) -> bool:
        """Check rate limits."""
        today = datetime.now().date()
        
        # Reset daily counter
        if today > self.day_start:
            self.posts_today = 0
            self.day_start = today
        
        # Check daily limit
        if self.posts_today >= self.config.max_posts_per_day:
            return False
        
        # Check minimum interval
        if self.last_post_time:
            minutes_since = (datetime.now() - self.last_post_time).seconds / 60
            if minutes_since < self.config.min_minutes_between_posts:
                return False
        
        return True
    
    def post_text(self, text: str) -> bool:
        """Post tweet."""
        if not self.config.is_configured:
            logger.warning("Twitter not configured")
            return False
        
        if not self.can_post():
            logger.warning("Twitter rate limit reached")
            return False
        
        # Truncate to Twitter limit
        if len(text) > 280:
            text = text[:277] + "..."
        
        try:
            from requests_oauthlib import OAuth1
            oauth = OAuth1(
                self.config.api_key,
                self.config.api_secret,
                self.config.access_token,
                self.config.access_token_secret,
            )
            
            response = requests.post(
                self.TWEET_URL,
                json={"text": text},
                auth=oauth,
                timeout=10,
            )
            
            success = response.status_code in [200, 201]
            if success:
                self.posts_today += 1
                logger.info("Twitter post successful")
            else:
                logger.error(f"Twitter post failed: {response.status_code} - {response.text}")
            
            return success
            
        except ImportError:
            logger.error("requests-oauthlib required for Twitter")
            return False
        except Exception as e:
            logger.error(f"Twitter post error: {e}")
            return False
    
    def post_with_image(self, text: str, image_path: str) -> bool:
        """Post tweet with image."""
        if not self.config.is_configured:
            return False
        
        # Twitter image upload is complex - for now, post text only
        # Full implementation would use chunked upload for media
        logger.warning("Twitter image upload not yet implemented, posting text only")
        return self.post_text(text)


# =============================================================================
# LINKEDIN IMPLEMENTATION
# =============================================================================

class LinkedInPlatform(BasePlatform):
    """LinkedIn API implementation."""
    
    SHARE_URL = "https://api.linkedin.com/v2/ugcPosts"
    
    def __init__(self, config: LinkedInConfig):
        super().__init__(config)
        self.posts_today = 0
        self.day_start = datetime.now().date()
    
    @property
    def platform_name(self) -> str:
        return "linkedin"
    
    def can_post(self) -> bool:
        """Check rate limits."""
        today = datetime.now().date()
        
        if today > self.day_start:
            self.posts_today = 0
            self.day_start = today
        
        return self.posts_today < self.config.max_posts_per_day
    
    def post_text(self, text: str) -> bool:
        """Post to LinkedIn."""
        if not self.config.is_configured:
            logger.warning("LinkedIn not configured")
            return False
        
        if not self.can_post():
            logger.warning("LinkedIn rate limit reached")
            return False
        
        author = self.config.organization_urn or self.config.person_urn
        
        payload = {
            "author": author,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": text
                    },
                    "shareMediaCategory": "NONE"
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
            }
        }
        
        try:
            response = requests.post(
                self.SHARE_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.config.access_token}",
                    "Content-Type": "application/json",
                    "X-Restli-Protocol-Version": "2.0.0",
                },
                timeout=10,
            )
            
            success = response.status_code in [200, 201]
            if success:
                self.posts_today += 1
                logger.info("LinkedIn post successful")
            else:
                logger.error(f"LinkedIn post failed: {response.status_code} - {response.text}")
            
            return success
            
        except Exception as e:
            logger.error(f"LinkedIn post error: {e}")
            return False
    
    def post_with_image(self, text: str, image_path: str) -> bool:
        """Post with image to LinkedIn."""
        # LinkedIn image upload requires registering upload first
        # For simplicity, posting text only
        logger.warning("LinkedIn image upload not yet implemented, posting text only")
        return self.post_text(text)


# =============================================================================
# MESSAGE FORMATTER
# =============================================================================

class MessageFormatter:
    """Formats messages for different platforms."""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
    
    def format_trade(
        self, 
        trade: TradeData, 
        platform: Platform
    ) -> str:
        """Format trade result for posting."""
        
        if platform == Platform.TELEGRAM:
            return self._format_trade_telegram(trade)
        elif platform == Platform.TWITTER:
            return self._format_trade_twitter(trade)
        elif platform == Platform.LINKEDIN:
            return self._format_trade_linkedin(trade)
        
        return self._format_trade_generic(trade)
    
    def _format_trade_telegram(self, trade: TradeData) -> str:
        """Telegram format with HTML."""
        emoji = trade.emoji
        direction_emoji = "🟢" if trade.direction == "LONG" else "🔴"
        
        msg = f"""
{emoji} <b>Trade Closed</b> {emoji}

{direction_emoji} {trade.direction} {trade.symbol}
━━━━━━━━━━━━━━━
📥 Entry: {trade.entry_price:.5f}
📤 Exit: {trade.exit_price:.5f}
⏱ Duration: {trade.duration_minutes} min

💰 <b>P&L: {trade.pnl:+.2f} ({trade.pnl_pct:+.2f}%)</b>
"""
        if trade.signal_confidence:
            msg += f"🎯 Signal Confidence: {trade.signal_confidence:.1%}\n"
        
        msg += f"\n🤖 <i>{self.config.bot_name}</i>"
        
        return msg.strip()
    
    def _format_trade_twitter(self, trade: TradeData) -> str:
        """Twitter format (280 char limit)."""
        emoji = trade.emoji
        
        msg = f"{emoji} {trade.direction} #{trade.symbol}\n"
        msg += f"P&L: {trade.pnl:+.2f} ({trade.pnl_pct:+.2f}%)\n"
        
        # Add hashtags if space permits
        if self.config.twitter.include_hashtags:
            tags = " ".join(f"#{t}" for t in self.config.twitter.default_hashtags[:3])
            if len(msg) + len(tags) < 275:
                msg += f"\n{tags}"
        
        return msg.strip()
    
    def _format_trade_linkedin(self, trade: TradeData) -> str:
        """LinkedIn format (professional)."""
        outcome = "profitable" if trade.is_winner else "closed at a loss"
        
        msg = f"""Trade Update: {trade.symbol}

Our algorithmic trading system closed a {trade.direction.lower()} position on {trade.symbol}, {outcome}.

Entry: {trade.entry_price:.5f}
Exit: {trade.exit_price:.5f}
Result: {trade.pnl_pct:+.2f}%

Trade duration: {trade.duration_minutes} minutes

"""
        if self.config.include_disclaimer:
            msg += f"\n{self.config.disclaimer_text}\n"
        
        if self.config.linkedin.include_hashtags:
            tags = " ".join(f"#{t}" for t in self.config.linkedin.default_hashtags)
            msg += f"\n{tags}"
        
        return msg.strip()
    
    def _format_trade_generic(self, trade: TradeData) -> str:
        """Generic format."""
        return f"{trade.emoji} {trade.direction} {trade.symbol}: {trade.pnl:+.2f} ({trade.pnl_pct:+.2f}%)"
    
    def format_performance(
        self, 
        perf: PerformanceData, 
        platform: Platform
    ) -> str:
        """Format performance summary for posting."""
        
        if platform == Platform.TELEGRAM:
            return self._format_perf_telegram(perf)
        elif platform == Platform.TWITTER:
            return self._format_perf_twitter(perf)
        elif platform == Platform.LINKEDIN:
            return self._format_perf_linkedin(perf)
        
        return self._format_perf_generic(perf)
    
    def _format_perf_telegram(self, perf: PerformanceData) -> str:
        """Telegram performance format."""
        emoji = perf.emoji
        period_title = perf.period.capitalize()
        
        msg = f"""
{emoji} <b>{period_title} Performance Report</b> {emoji}

📅 {perf.start_date.strftime('%b %d')} - {perf.end_date.strftime('%b %d, %Y')}
━━━━━━━━━━━━━━━━━

📊 <b>Results</b>
├ Trades: {perf.total_trades}
├ Wins: {perf.winning_trades} ({perf.win_rate:.1%})
└ P&L: <b>{perf.total_pnl:+.2f} ({perf.total_pnl_pct:+.2f}%)</b>

📈 <b>Metrics</b>
├ Profit Factor: {perf.profit_factor:.2f}
"""
        if perf.sharpe_ratio is not None:
            msg += f"├ Sharpe Ratio: {perf.sharpe_ratio:.2f}\n"
        if perf.max_drawdown is not None:
            msg += f"└ Max Drawdown: {perf.max_drawdown:.2%}\n"
        
        if perf.best_trade_pnl is not None:
            msg += f"""
🏆 Best Trade: {perf.best_trade_pnl:+.2f}
😔 Worst Trade: {perf.worst_trade_pnl:+.2f}
"""
        
        msg += f"\n🤖 <i>{self.config.bot_name}</i>"
        
        if self.config.include_disclaimer:
            msg += f"\n\n<i>{self.config.disclaimer_text}</i>"
        
        return msg.strip()
    
    def _format_perf_twitter(self, perf: PerformanceData) -> str:
        """Twitter performance format."""
        emoji = perf.emoji
        period = perf.period.capitalize()
        
        msg = f"{emoji} {period} Results\n\n"
        msg += f"📊 {perf.total_trades} trades | {perf.win_rate:.0%} win rate\n"
        msg += f"💰 {perf.total_pnl_pct:+.1f}% return\n"
        msg += f"📈 PF: {perf.profit_factor:.1f}"
        
        if perf.sharpe_ratio is not None:
            msg += f" | SR: {perf.sharpe_ratio:.1f}"
        
        if self.config.twitter.include_hashtags:
            tags = " ".join(f"#{t}" for t in self.config.twitter.default_hashtags[:2])
            msg += f"\n\n{tags}"
        
        return msg.strip()
    
    def _format_perf_linkedin(self, perf: PerformanceData) -> str:
        """LinkedIn performance format (professional)."""
        period = perf.period.capitalize()
        
        msg = f"""{period} Algorithmic Trading Performance Report

I'm pleased to share our trading system's {perf.period} performance metrics:

📊 Trading Activity
• Total Trades Executed: {perf.total_trades}
• Win Rate: {perf.win_rate:.1%}
• Profit Factor: {perf.profit_factor:.2f}

💹 Returns
• Period Return: {perf.total_pnl_pct:+.2f}%
"""
        
        if perf.sharpe_ratio is not None:
            msg += f"• Sharpe Ratio: {perf.sharpe_ratio:.2f}\n"
        if perf.max_drawdown is not None:
            msg += f"• Maximum Drawdown: {perf.max_drawdown:.2%}\n"
        
        msg += """
Our system uses machine learning models including Temporal Convolutional Networks and Vision Transformers to identify trading opportunities in the forex market.
"""
        
        if self.config.include_disclaimer:
            msg += f"\n{self.config.disclaimer_text}\n"
        
        if self.config.linkedin.include_hashtags:
            tags = " ".join(f"#{t}" for t in self.config.linkedin.default_hashtags)
            msg += f"\n{tags}"
        
        return msg.strip()
    
    def _format_perf_generic(self, perf: PerformanceData) -> str:
        """Generic performance format."""
        return f"{perf.emoji} {perf.period.capitalize()}: {perf.total_trades} trades, {perf.win_rate:.0%} WR, {perf.total_pnl_pct:+.1f}%"
    
    def format_alert(
        self, 
        alert_type: str, 
        message: str, 
        platform: Platform
    ) -> str:
        """Format system alert for posting."""
        
        if platform == Platform.TELEGRAM:
            return f"⚠️ <b>{alert_type}</b>\n\n{message}\n\n🤖 <i>{self.config.bot_name}</i>"
        elif platform == Platform.TWITTER:
            return f"⚠️ {alert_type}: {message[:200]}"
        else:
            return f"{alert_type}\n\n{message}"
    
    def format_milestone(
        self, 
        milestone_type: str, 
        details: str, 
        platform: Platform
    ) -> str:
        """Format milestone achievement for posting."""
        
        if platform == Platform.TELEGRAM:
            return f"""
🎉 <b>MILESTONE ACHIEVED!</b> 🎉

{milestone_type}
━━━━━━━━━━━━━━━

{details}

🤖 <i>{self.config.bot_name}</i>
""".strip()
        
        elif platform == Platform.TWITTER:
            msg = f"🎉 {milestone_type}!\n\n{details[:150]}"
            if self.config.twitter.include_hashtags:
                msg += f"\n\n#milestone #trading"
            return msg
        
        else:
            return f"🎉 Milestone: {milestone_type}\n\n{details}"


# =============================================================================
# MAIN NOTIFIER CLASS
# =============================================================================

class SocialMediaNotifier:
    """
    Main class for sending notifications to social media platforms.
    
    Usage:
        notifier = SocialMediaNotifier(config)
        
        # Post trade result
        notifier.post_trade_result(trade_data)
        
        # Post performance update
        notifier.post_performance_update(performance_data)
        
        # Post to specific platform
        notifier.post_to_platform(Platform.TELEGRAM, "Hello!")
    """
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.formatter = MessageFormatter(config)
        
        # Initialize platforms
        self.platforms: Dict[Platform, BasePlatform] = {}
        
        if config.telegram.enabled and config.telegram.is_configured:
            self.platforms[Platform.TELEGRAM] = TelegramPlatform(config.telegram)
        
        if config.twitter.enabled and config.twitter.is_configured:
            self.platforms[Platform.TWITTER] = TwitterPlatform(config.twitter)
        
        if config.linkedin.enabled and config.linkedin.is_configured:
            self.platforms[Platform.LINKEDIN] = LinkedInPlatform(config.linkedin)
        
        # Post history
        self.post_log: List[Dict] = []
        
        logger.info(f"SocialMediaNotifier initialized with platforms: {list(self.platforms.keys())}")
    
    def post_trade_result(
        self, 
        trade: TradeData,
        platforms: Optional[List[Platform]] = None
    ) -> Dict[Platform, bool]:
        """
        Post trade result to enabled platforms.
        
        Args:
            trade: Trade data to post
            platforms: Specific platforms (None = all enabled)
        
        Returns:
            Dict mapping platform to success status
        """
        results = {}
        target_platforms = platforms or list(self.platforms.keys())
        
        for platform in target_platforms:
            if platform not in self.platforms:
                continue
            
            # Check platform-specific rules
            if platform == Platform.TELEGRAM:
                if not self.config.telegram.post_trades:
                    continue
                if abs(trade.pnl) < self.config.telegram.min_trade_pnl_to_post:
                    continue
            
            elif platform == Platform.TWITTER:
                if not self.config.twitter.post_trades:
                    continue
                if abs(trade.pnl) < self.config.twitter.min_trade_pnl_to_post:
                    continue
            
            elif platform == Platform.LINKEDIN:
                if not self.config.linkedin.post_trades:
                    continue
            
            # Format and post
            message = self.formatter.format_trade(trade, platform)
            
            if self.config.dry_run:
                logger.info(f"[DRY RUN] Would post to {platform.value}: {message[:100]}...")
                results[platform] = True
            else:
                success = self.platforms[platform].post_text(message)
                results[platform] = success
                self._log_post(platform, PostType.TRADE_RESULT, message, success)
        
        return results
    
    def post_performance_update(
        self, 
        performance: PerformanceData,
        platforms: Optional[List[Platform]] = None,
        chart_path: Optional[str] = None
    ) -> Dict[Platform, bool]:
        """
        Post performance update to enabled platforms.
        
        Args:
            performance: Performance data to post
            platforms: Specific platforms (None = all enabled)
            chart_path: Optional path to equity curve chart
        
        Returns:
            Dict mapping platform to success status
        """
        results = {}
        target_platforms = platforms or list(self.platforms.keys())
        
        for platform in target_platforms:
            if platform not in self.platforms:
                continue
            
            # Check platform-specific rules based on period
            if performance.period == "daily":
                if platform == Platform.TELEGRAM and not self.config.telegram.post_daily_summary:
                    continue
                if platform == Platform.TWITTER and not self.config.twitter.post_daily_summary:
                    continue
                if platform == Platform.LINKEDIN and not self.config.linkedin.post_daily_summary:
                    continue
            
            elif performance.period == "weekly":
                if platform == Platform.TWITTER and not self.config.twitter.post_weekly_summary:
                    continue
                if platform == Platform.LINKEDIN and not self.config.linkedin.post_weekly_summary:
                    continue
            
            # Format message
            message = self.formatter.format_performance(performance, platform)
            
            if self.config.dry_run:
                logger.info(f"[DRY RUN] Would post to {platform.value}: {message[:100]}...")
                results[platform] = True
            else:
                # Post with chart if available
                if chart_path and self.config.attach_charts:
                    success = self.platforms[platform].post_with_image(message, chart_path)
                else:
                    success = self.platforms[platform].post_text(message)
                
                results[platform] = success
                
                post_type = PostType.DAILY_SUMMARY if performance.period == "daily" else PostType.WEEKLY_SUMMARY
                self._log_post(platform, post_type, message, success)
        
        return results
    
    def post_alert(
        self, 
        alert_type: str, 
        message: str,
        platforms: Optional[List[Platform]] = None
    ) -> Dict[Platform, bool]:
        """Post system alert."""
        results = {}
        target_platforms = platforms or [Platform.TELEGRAM]  # Alerts mainly to Telegram
        
        for platform in target_platforms:
            if platform not in self.platforms:
                continue
            
            if platform == Platform.TELEGRAM and not self.config.telegram.post_alerts:
                continue
            
            formatted = self.formatter.format_alert(alert_type, message, platform)
            
            if self.config.dry_run:
                logger.info(f"[DRY RUN] Alert to {platform.value}: {formatted[:100]}...")
                results[platform] = True
            else:
                success = self.platforms[platform].post_text(formatted)
                results[platform] = success
                self._log_post(platform, PostType.SYSTEM_ALERT, formatted, success)
        
        return results
    
    def post_milestone(
        self, 
        milestone_type: str, 
        details: str,
        platforms: Optional[List[Platform]] = None
    ) -> Dict[Platform, bool]:
        """Post milestone achievement."""
        results = {}
        target_platforms = platforms or list(self.platforms.keys())
        
        for platform in target_platforms:
            if platform not in self.platforms:
                continue
            
            if platform == Platform.TWITTER and not self.config.twitter.post_milestones:
                continue
            if platform == Platform.LINKEDIN and not self.config.linkedin.post_milestones:
                continue
            
            formatted = self.formatter.format_milestone(milestone_type, details, platform)
            
            if self.config.dry_run:
                logger.info(f"[DRY RUN] Milestone to {platform.value}")
                results[platform] = True
            else:
                success = self.platforms[platform].post_text(formatted)
                results[platform] = success
                self._log_post(platform, PostType.MILESTONE, formatted, success)
        
        return results
    
    def post_to_platform(
        self, 
        platform: Platform, 
        message: str,
        image_path: Optional[str] = None
    ) -> bool:
        """Post custom message to specific platform."""
        if platform not in self.platforms:
            logger.warning(f"Platform {platform.value} not configured")
            return False
        
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would post to {platform.value}: {message[:100]}...")
            return True
        
        if image_path:
            return self.platforms[platform].post_with_image(message, image_path)
        return self.platforms[platform].post_text(message)
    
    def _log_post(
        self, 
        platform: Platform, 
        post_type: PostType, 
        content: str, 
        success: bool
    ):
        """Log post to history."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'platform': platform.value,
            'post_type': post_type.value,
            'success': success,
            'content_preview': content[:100],
        }
        self.post_log.append(entry)
        
        # Save to file if enabled
        if self.config.log_posts:
            self._save_log()
    
    def _save_log(self):
        """Save post log to file."""
        try:
            log_path = Path(self.config.posts_log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_path, 'w') as f:
                json.dump(self.post_log[-1000:], f, indent=2)  # Keep last 1000
        except Exception as e:
            logger.error(f"Failed to save post log: {e}")
    
    def get_post_history(self, limit: int = 50) -> List[Dict]:
        """Get recent post history."""
        return self.post_log[-limit:]
    
    def get_platform_status(self) -> Dict[str, Any]:
        """Get status of all platforms."""
        status = {}
        
        for platform, client in self.platforms.items():
            status[platform.value] = {
                'enabled': True,
                'can_post': client.can_post(),
                'last_post': client.last_post_time.isoformat() if client.last_post_time else None,
                'posts_count': len(client.post_history),
            }
        
        return status