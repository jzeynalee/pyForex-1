# signals/__init__.py
"""
Signal Publishing Module for pyForex.

Publishes trading signals to multiple platforms:
- Telegram (real-time alerts)
- Twitter/X (public signals for track record)
- LinkedIn (professional updates)
- Webhooks (custom integrations)

Usage:
    from signals import SignalPublisher, PublisherConfig
    
    publisher = SignalPublisher(config)
    publisher.set_notifiers(telegram=tg_client, twitter=tw_client)
    publisher.publish_entry(symbol='EURUSD', direction='BUY', ...)
"""

from .signal_publisher import (
    Platform,
    SignalType,
    Signal,
    SignalPerformance,
    PublisherConfig,
    SignalPublisher,
    TradingBotSignalAdapter,
    create_signal_publisher_for_bot
)

__all__ = [
    'Platform',
    'SignalType',
    'Signal',
    'SignalPerformance',
    'PublisherConfig',
    'SignalPublisher',
    'TradingBotSignalAdapter',
    'create_signal_publisher_for_bot'
]
