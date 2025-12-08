"""
pyForex Social Media Notifications Module

Provides automated posting to Twitter, LinkedIn, and Telegram
for trade results, performance updates, and system alerts.

Quick Start:
    from notifications import SocialMediaNotifier, NotificationConfig
    
    # Load config from environment
    config = NotificationConfig.from_env()
    
    # Create notifier
    notifier = SocialMediaNotifier(config)
    
    # Post trade result
    notifier.post_trade_result(trade_data)

Full Integration:
    from notifications import setup_social_notifications
    
    integration = setup_social_notifications(
        performance_monitor=monitor,
        retraining_scheduler=scheduler,
        start_scheduler=True,
    )

Environment Variables:
    TELEGRAM_BOT_TOKEN    - Telegram bot token from @BotFather
    TELEGRAM_CHAT_ID      - Target chat/channel ID
    TWITTER_API_KEY       - Twitter API key
    TWITTER_API_SECRET    - Twitter API secret
    TWITTER_ACCESS_TOKEN  - Twitter access token
    TWITTER_ACCESS_TOKEN_SECRET - Twitter access token secret
    LINKEDIN_ACCESS_TOKEN - LinkedIn OAuth access token
    LINKEDIN_PERSON_URN   - LinkedIn person URN
"""

# Core classes
from .social_media import (
    # Configuration
    NotificationConfig,
    TelegramConfig,
    TwitterConfig,
    LinkedInConfig,
    
    # Enums
    Platform,
    PostType,
    
    # Data classes
    TradeData,
    PerformanceData,
    
    # Main notifier
    SocialMediaNotifier,
    MessageFormatter,
    
    # Platform implementations
    TelegramPlatform,
    TwitterPlatform,
    LinkedInPlatform,
)

# Integration
from .integration import (
    SocialMediaIntegration,
    ChartGenerator,
    setup_social_notifications,
)

__version__ = "1.0.0"
__author__ = "pyForex Team"

__all__ = [
    # Configuration
    'NotificationConfig',
    'TelegramConfig',
    'TwitterConfig',
    'LinkedInConfig',
    
    # Enums
    'Platform',
    'PostType',
    
    # Data classes
    'TradeData',
    'PerformanceData',
    
    # Main classes
    'SocialMediaNotifier',
    'SocialMediaIntegration',
    
    # Utilities
    'MessageFormatter',
    'ChartGenerator',
    
    # Quick setup
    'setup_social_notifications',
]