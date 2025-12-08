#!/usr/bin/env python3
"""
Example: Social Media Integration for pyForex

This example shows how to:
1. Configure social media notifications
2. Connect to your trading system
3. Auto-post trade results
4. Schedule periodic summaries
5. Post custom updates

Run this example:
    python examples/social_media_example.py
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from notifications import (
    NotificationConfig,
    TelegramConfig,
    TwitterConfig,
    LinkedInConfig,
    SocialMediaNotifier,
    SocialMediaIntegration,
    TradeData,
    PerformanceData,
    Platform,
    setup_social_notifications,
)


def example_basic_usage():
    """Basic usage: Post a single trade result."""
    print("\n" + "="*60)
    print("Example 1: Basic Usage - Post Trade Result")
    print("="*60)
    
    # Configure (use your actual credentials)
    config = NotificationConfig(
        telegram=TelegramConfig(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            enabled=True,
        ),
        dry_run=True,  # Set to False for actual posting
        bot_name="pyForex Demo",
    )
    
    # Create notifier
    notifier = SocialMediaNotifier(config)
    
    # Create trade data
    trade = TradeData(
        trade_id="demo_001",
        symbol="EURUSD",
        direction="LONG",
        entry_price=1.08500,
        exit_price=1.08750,
        pnl=25.00,
        pnl_pct=0.23,
        entry_time=datetime.now() - timedelta(hours=2),
        exit_time=datetime.now(),
        duration_minutes=120,
        signal_confidence=0.82,
    )
    
    # Post to Telegram
    results = notifier.post_trade_result(trade, platforms=[Platform.TELEGRAM])
    
    print(f"Trade posted: {results}")
    print(f"Trade was: {'WIN' if trade.is_winner else 'LOSS'}")


def example_performance_update():
    """Post weekly performance summary."""
    print("\n" + "="*60)
    print("Example 2: Performance Summary")
    print("="*60)
    
    config = NotificationConfig(
        telegram=TelegramConfig(
            bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        ),
        twitter=TwitterConfig(
            api_key=os.getenv("TWITTER_API_KEY", ""),
            api_secret=os.getenv("TWITTER_API_SECRET", ""),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN", ""),
            access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET", ""),
        ),
        dry_run=True,
    )
    
    notifier = SocialMediaNotifier(config)
    
    # Create performance data
    perf = PerformanceData(
        period="weekly",
        start_date=datetime.now() - timedelta(days=7),
        end_date=datetime.now(),
        total_trades=47,
        winning_trades=28,
        total_pnl=342.50,
        total_pnl_pct=3.42,
        win_rate=0.596,
        profit_factor=1.87,
        sharpe_ratio=1.42,
        max_drawdown=0.045,
        avg_trade_pnl=7.29,
        best_trade_pnl=89.50,
        worst_trade_pnl=-34.20,
    )
    
    # Post to all configured platforms
    results = notifier.post_performance_update(perf)
    
    print(f"Performance posted: {results}")


def example_full_integration():
    """Full integration with trading system."""
    print("\n" + "="*60)
    print("Example 3: Full Integration")
    print("="*60)
    
    # In real usage, you would import your actual monitor and scheduler
    # from ml import PerformanceMonitor, RetrainingScheduler
    
    # For demo, we'll create a mock
    class MockPerformanceMonitor:
        def __init__(self):
            self.closed_trades = []
            self.equity_curve = [
                (datetime.now() - timedelta(days=i), 10000 + i*50)
                for i in range(30, 0, -1)
            ]
        
        def add_trade(self, trade):
            self.closed_trades.append(trade)
            return True
        
        def get_current_metrics(self):
            return {
                'win_rate': 0.58,
                'profit_factor': 1.65,
                'sharpe_ratio': 1.2,
                'max_drawdown': 0.08,
            }
        
        def get_performance_summary(self):
            return {
                'total_trades': len(self.closed_trades),
                'total_pnl': 500.0,
                'total_return_pct': 5.0,
            }
    
    # Create mock monitor
    monitor = MockPerformanceMonitor()
    
    # Setup integration
    integration = SocialMediaIntegration.from_env()
    integration.config.dry_run = True  # Demo mode
    
    # Connect to monitor (this wraps add_trade to auto-post)
    integration.connect_performance_monitor(monitor)
    
    # Now when trades are added, they auto-post
    print("Integration connected. Trades will auto-post when closed.")
    
    # Manual posting still works
    integration.post_daily_summary()
    
    # Get status
    status = integration.get_status()
    print(f"Integration status: {status}")


def example_custom_posting():
    """Post custom messages."""
    print("\n" + "="*60)
    print("Example 4: Custom Posting")
    print("="*60)
    
    config = NotificationConfig.from_env()
    config.dry_run = True
    
    notifier = SocialMediaNotifier(config)
    
    # Post alert
    results = notifier.post_alert(
        "Market Volatility Alert",
        "Significant volatility detected in EURUSD. Our system is adjusting position sizes accordingly.",
        platforms=[Platform.TELEGRAM]
    )
    print(f"Alert posted: {results}")
    
    # Post milestone
    results = notifier.post_milestone(
        "100 Trades Milestone 🎯",
        "Our algorithmic trading system has now executed 100 trades with a 58% win rate!",
    )
    print(f"Milestone posted: {results}")


def example_with_scheduler():
    """Setup with automatic scheduling."""
    print("\n" + "="*60)
    print("Example 5: Scheduled Posting")
    print("="*60)
    
    # Quick setup function
    integration = setup_social_notifications(
        performance_monitor=None,  # Would be your actual monitor
        retraining_scheduler=None,  # Would be your actual scheduler
        start_scheduler=False,  # Don't start for demo
        daily_hour=22,  # Post daily at 22:00 UTC
        weekly_day=6,   # Post weekly on Sunday
        weekly_hour=20, # At 20:00 UTC
    )
    
    print("Scheduler configured (not started for demo)")
    print(f"Status: {integration.get_status()}")
    
    # To actually start:
    # integration.start_scheduler()
    
    # To stop:
    # integration.stop_scheduler()


def example_environment_setup():
    """Show required environment variables."""
    print("\n" + "="*60)
    print("Environment Setup Guide")
    print("="*60)
    
    print("""
Required Environment Variables:

TELEGRAM:
    export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
    export TELEGRAM_CHAT_ID="@your_channel" or "-100123456789"

TWITTER (X):
    export TWITTER_API_KEY="your_api_key"
    export TWITTER_API_SECRET="your_api_secret"
    export TWITTER_ACCESS_TOKEN="your_access_token"
    export TWITTER_ACCESS_TOKEN_SECRET="your_access_token_secret"

LINKEDIN:
    export LINKEDIN_ACCESS_TOKEN="your_oauth_token"
    export LINKEDIN_PERSON_URN="urn:li:person:XXXXXXX"

How to get credentials:

1. TELEGRAM:
   - Message @BotFather on Telegram
   - Send /newbot and follow instructions
   - Copy the bot token
   - Add bot to your channel/group
   - Get chat ID from @userinfobot or API

2. TWITTER:
   - Go to developer.twitter.com
   - Create a project and app
   - Generate API keys and tokens
   - Ensure "Read and Write" permissions

3. LINKEDIN:
   - Go to linkedin.com/developers
   - Create an app
   - Request "Share on LinkedIn" permission
   - Use OAuth 2.0 flow to get access token
   - Get your person URN from /me endpoint
""")


if __name__ == "__main__":
    print("pyForex Social Media Integration Examples")
    print("=========================================")
    
    # Run examples
    example_basic_usage()
    example_performance_update()
    example_full_integration()
    example_custom_posting()
    example_with_scheduler()
    example_environment_setup()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)