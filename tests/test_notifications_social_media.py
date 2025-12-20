# tests/test_notifications_social_media.py
"""
Unit tests for notifications/social_media.py - Social media notifications module.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
from datetime import datetime
import json
import os
from notifications.social_media import (
    Platform, PostType, TelegramConfig, TwitterConfig, LinkedInConfig,
    NotificationConfig, SocialMediaNotifier
)


@pytest.mark.unit
class TestPlatform:
    """Test Platform enum."""

    def test_platform_values(self):
        """Test platform enum values."""
        assert Platform.TWITTER.value == "twitter"
        assert Platform.LINKEDIN.value == "linkedin"
        assert Platform.TELEGRAM.value == "telegram"


@pytest.mark.unit
class TestPostType:
    """Test PostType enum."""

    def test_post_type_values(self):
        """Test post type enum values."""
        assert PostType.TRADE_RESULT.value == "trade_result"
        assert PostType.DAILY_SUMMARY.value == "daily_summary"
        assert PostType.WEEKLY_SUMMARY.value == "weekly_summary"
        assert PostType.SYSTEM_ALERT.value == "system_alert"


@pytest.mark.unit
class TestTelegramConfig:
    """Test TelegramConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TelegramConfig()

        assert config.bot_token == ""
        assert config.chat_id == ""
        assert config.enabled is True
        assert config.post_trades is True
        assert config.max_posts_per_hour == 30

    def test_is_configured_true(self):
        """Test is_configured property when configured."""
        config = TelegramConfig(bot_token="token123", chat_id="chat123")

        assert config.is_configured is True

    def test_is_configured_false(self):
        """Test is_configured property when not configured."""
        config = TelegramConfig()

        assert config.is_configured is False


@pytest.mark.unit
class TestTwitterConfig:
    """Test TwitterConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TwitterConfig()

        assert config.api_key == ""
        assert config.enabled is True
        assert config.post_trades is False
        assert config.post_daily_summary is True
        assert config.max_posts_per_day == 50
        assert "algotrading" in config.default_hashtags

    def test_is_configured_true(self):
        """Test is_configured property when configured."""
        config = TwitterConfig(
            api_key="key",
            api_secret="secret",
            access_token="token",
            access_token_secret="secret2"
        )

        assert config.is_configured is True

    def test_is_configured_false(self):
        """Test is_configured property when not configured."""
        config = TwitterConfig()

        assert config.is_configured is False


@pytest.mark.unit
class TestLinkedInConfig:
    """Test LinkedInConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LinkedInConfig()

        assert config.access_token == ""
        assert config.enabled is True
        assert config.post_weekly_summary is True
        assert config.max_posts_per_day == 3

    def test_is_configured_true(self):
        """Test is_configured property when configured."""
        config = LinkedInConfig(access_token="token", person_urn="urn:li:person:123")

        assert config.is_configured is True


@pytest.mark.unit
class TestNotificationConfig:
    """Test NotificationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = NotificationConfig()

        assert config.dry_run is False
        assert config.bot_name == "pyForex"
        assert config.include_disclaimer is True
        assert isinstance(config.telegram, TelegramConfig)
        assert isinstance(config.twitter, TwitterConfig)

    def test_from_env(self):
        """Test loading configuration from environment."""
        with patch.dict(os.environ, {
            'TELEGRAM_BOT_TOKEN': 'test_token',
            'TELEGRAM_CHAT_ID': 'test_chat'
        }):
            config = NotificationConfig.from_env()

            assert config.telegram.bot_token == 'test_token'
            assert config.telegram.chat_id == 'test_chat'

    def test_from_file(self):
        """Test loading configuration from JSON file."""
        config_data = {
            'telegram': {
                'bot_token': 'file_token',
                'chat_id': 'file_chat'
            },
            'twitter': {
                'api_key': 'file_key'
            },
            'dry_run': True,
            'bot_name': 'TestBot'
        }

        with patch('builtins.open', mock_open(read_data=json.dumps(config_data))):
            config = NotificationConfig.from_file('test_config.json')

            assert config.telegram.bot_token == 'file_token'
            assert config.dry_run is True
            assert config.bot_name == 'TestBot'


@pytest.mark.unit
class TestSocialMediaNotifier:
    """Test SocialMediaNotifier class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock notification config."""
        config = NotificationConfig()
        config.telegram.bot_token = "test_token"
        config.telegram.chat_id = "test_chat"
        return config

    def test_init_default(self):
        """Test default initialization."""
        from notifications.social_media import SocialMediaConfig
        config = SocialMediaConfig()
        notifier = SocialMediaNotifier(config)

        assert notifier.config is not None
        assert isinstance(notifier.config, NotificationConfig)

    def test_init_with_config(self, mock_config):
        """Test initialization with config."""
        notifier = SocialMediaNotifier(config=mock_config)

        assert notifier.config == mock_config

    def test_post_trade_result_telegram(self, mock_config):
        """Test posting trade result to Telegram."""
        with patch('notifications.social_media.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'ok': True}
            mock_requests.post.return_value = mock_response

            notifier = SocialMediaNotifier(config=mock_config)
            
            trade_data = {
                'symbol': 'EURUSD',
                'direction': 'BUY',
                'pnl': 100.0
            }

            result = notifier.post_trade_result(trade_data)

            assert result is True
            mock_requests.post.assert_called()

    def test_post_trade_result_dry_run(self, mock_config):
        """Test posting trade result in dry run mode."""
        mock_config.dry_run = True
        notifier = SocialMediaNotifier(config=mock_config)

        trade_data = {'symbol': 'EURUSD', 'pnl': 100.0}
        result = notifier.post_trade_result(trade_data)

        # Should return True but not actually post
        assert result is True

    def test_post_performance_update(self, mock_config):
        """Test posting performance update."""
        with patch('notifications.social_media.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_requests.post.return_value = mock_response

            notifier = SocialMediaNotifier(config=mock_config)

            metrics = {
                'total_trades': 10,
                'win_rate': 0.7,
                'total_pnl': 500.0
            }

            result = notifier.post_performance_update(metrics)

            assert result is True

