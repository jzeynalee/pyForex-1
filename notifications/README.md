# pyForex Social Media Notifications

Automated posting to **Twitter/X**, **LinkedIn**, and **Telegram** for trade results, performance updates, and system alerts.

## Quick Start

```python
from notifications import SocialMediaNotifier, NotificationConfig

# Load config from environment variables
config = NotificationConfig.from_env()
notifier = SocialMediaNotifier(config)

# Post trade result
notifier.post_trade_result(trade_data)

# Post weekly summary
notifier.post_performance_update(performance_data)
```

## Installation

1. Copy the `notifications/` folder to your project
2. Install dependencies:
   ```bash
   pip install requests requests-oauthlib matplotlib
   ```
3. Set up environment variables (see below)

## Environment Variables

```bash
# Telegram (Required for Telegram)
export TELEGRAM_BOT_TOKEN="123456:ABC-DEF..."
export TELEGRAM_CHAT_ID="@your_channel"  # or "-100123456789"

# Twitter/X (Required for Twitter)
export TWITTER_API_KEY="your_api_key"
export TWITTER_API_SECRET="your_api_secret"
export TWITTER_ACCESS_TOKEN="your_access_token"
export TWITTER_ACCESS_TOKEN_SECRET="your_access_token_secret"

# LinkedIn (Required for LinkedIn)
export LINKEDIN_ACCESS_TOKEN="your_oauth_token"
export LINKEDIN_PERSON_URN="urn:li:person:XXXXXXX"
```

## Platform Setup Guides

### Telegram Bot Setup

1. Message **@BotFather** on Telegram
2. Send `/newbot` and follow the instructions
3. Copy the bot token
4. Add your bot to your channel as admin
5. Get chat ID:
   - For channels: `@channel_name`
   - For groups: Forward a message to @userinfobot

### Twitter/X Setup

1. Go to [developer.twitter.com](https://developer.twitter.com)
2. Create a project and app
3. Set permissions to "Read and Write"
4. Generate API keys and access tokens
5. Copy all 4 credentials

### LinkedIn Setup

1. Go to [linkedin.com/developers](https://www.linkedin.com/developers/)
2. Create an app
3. Request "Share on LinkedIn" permission
4. Complete OAuth 2.0 flow
5. Get your person URN from the `/me` API endpoint

## Full Integration with pyForex

```python
from notifications import setup_social_notifications
from ml import PerformanceMonitor, RetrainingScheduler

# Create your trading components
monitor = PerformanceMonitor()
scheduler = RetrainingScheduler(config)

# One-line setup - connects everything
integration = setup_social_notifications(
    performance_monitor=monitor,
    retraining_scheduler=scheduler,
    start_scheduler=True,
    daily_hour=22,   # Post daily summary at 22:00 UTC
    weekly_day=6,    # Post weekly on Sunday
    weekly_hour=20,  # At 20:00 UTC
)

# Now trades auto-post when closed!
# Daily and weekly summaries post automatically!
```

## What Gets Posted

### Telegram (Real-time)
- ✅ Every closed trade
- ✅ Daily performance summary
- ✅ System alerts (drift, retraining)
- ✅ Milestones

### Twitter/X (Curated)
- ❌ Individual trades (too noisy)
- ✅ Daily summary
- ✅ Weekly summary
- ✅ Milestones
- ✅ Significant trades (>$50 P&L)

### LinkedIn (Professional)
- ❌ Individual trades
- ❌ Daily summary
- ✅ Weekly summary
- ✅ Monthly summary
- ✅ Milestones

## Sample Posts

### Telegram Trade Alert
```
✅ Trade Closed ✅

🟢 LONG EURUSD
━━━━━━━━━━━━━━━
📥 Entry: 1.08500
📤 Exit: 1.08750
⏱ Duration: 120 min

💰 P&L: +$25.00 (+0.23%)
🎯 Signal Confidence: 82%

🤖 pyForex
```

### Twitter Weekly Summary
```
📈 Weekly Results

📊 47 trades | 60% win rate
💰 +3.4% return
📈 PF: 1.9 | SR: 1.4

#algotrading #forex #trading
```

## Configuration File

You can also use a JSON config file:

```python
config = NotificationConfig.from_file("config/notifications.json")
```

See `config.sample.json` for the full template.

## API Reference

### SocialMediaNotifier

```python
notifier = SocialMediaNotifier(config)

# Post trade
notifier.post_trade_result(trade_data, platforms=[Platform.TELEGRAM])

# Post performance
notifier.post_performance_update(perf_data, chart_path="chart.png")

# Post alert
notifier.post_alert("Alert Title", "Message", platforms=[Platform.TELEGRAM])

# Post milestone
notifier.post_milestone("100 Trades!", "Details...")

# Get status
notifier.get_platform_status()
```

### SocialMediaIntegration

```python
integration = SocialMediaIntegration.from_env()

# Connect to trading system
integration.connect_performance_monitor(monitor)
integration.connect_retraining_scheduler(scheduler)

# Start automatic posting
integration.start_scheduler(daily_hour=22, weekly_day=6, weekly_hour=20)

# Manual posts
integration.post_daily_summary()
integration.post_weekly_summary()
integration.post_custom("Custom message!")

# Status
integration.get_status()
```

## Dry Run Mode

Test without actually posting:

```python
config = NotificationConfig.from_env()
config.dry_run = True  # Logs posts instead of sending

notifier = SocialMediaNotifier(config)
notifier.post_trade_result(trade)  # Logged, not sent
```

## Files Structure

```
notifications/
├── __init__.py           # Package exports
├── social_media.py       # Core notification classes
├── integration.py        # Trading system integration
└── config.sample.json    # Sample configuration

examples/
└── social_media_example.py  # Usage examples
```

## Dependencies

- `requests` - HTTP client
- `requests-oauthlib` - Twitter OAuth (optional)
- `matplotlib` - Chart generation (optional)

## Tips for Social Media Growth

1. **Consistency**: Post daily summaries at the same time
2. **Transparency**: Show losses too, not just wins
3. **Engagement**: Respond to comments
4. **Hashtags**: Use relevant hashtags on Twitter
5. **Visuals**: Equity curve charts get more engagement
6. **Education**: Explain your strategy occasionally

## Changelog

### v1.0.0
- Initial release
- Telegram, Twitter, LinkedIn support
- Auto-posting from PerformanceMonitor
- Scheduled daily/weekly summaries
- Chart generation
- Milestone tracking