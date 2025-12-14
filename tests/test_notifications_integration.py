# tests/test_notifications_integration.py
"""
Comprehensive unit tests for notifications/integration.py

Tests social media integration with trading system components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from notifications.integration import (
    ChartGenerator, SocialMediaIntegration, setup_social_notifications
)
from notifications.social_media import (
    NotificationConfig, TradeData, PerformanceData, Platform, PostType
)


@pytest.mark.unit
class TestChartGenerator:
    """Test ChartGenerator class."""

    @pytest.fixture
    def chart_generator(self, tmp_path):
        """Create ChartGenerator with temp directory."""
        return ChartGenerator(output_dir=str(tmp_path / "charts"))

    def test_init(self, tmp_path):
        """Test ChartGenerator initialization."""
        gen = ChartGenerator(output_dir=str(tmp_path / "charts"))
        assert gen.output_dir.exists()

    @patch('notifications.integration.plt')
    @patch('notifications.integration.matplotlib.use')
    def test_generate_equity_curve(self, mock_use, mock_plt, chart_generator):
        """Test equity curve generation."""
        equity_data = [
            (datetime(2024, 1, 1), 10000),
            (datetime(2024, 1, 2), 10100),
            (datetime(2024, 1, 3), 10200),
        ]

        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, MagicMock())

        result = chart_generator.generate_equity_curve(equity_data)

        # Should attempt to create chart
        mock_plt.subplots.assert_called()

    def test_generate_equity_curve_empty_data(self, chart_generator):
        """Test equity curve with empty data."""
        result = chart_generator.generate_equity_curve([])
        assert result is None

    def test_generate_equity_curve_single_point(self, chart_generator):
        """Test equity curve with single point."""
        equity_data = [(datetime(2024, 1, 1), 10000)]
        result = chart_generator.generate_equity_curve(equity_data)
        # Single point might return None or handle gracefully
        assert result is None or isinstance(result, str)

    @patch('notifications.integration.plt')
    def test_generate_performance_summary(self, mock_plt, chart_generator):
        """Test performance summary chart generation."""
        metrics = {
            'win_rate': 0.6,
            'profit_factor': 1.8,
            'sharpe_ratio': 1.5,
            'total_pnl': 500,
            'total_pnl_pct': 5.0,
            'total_trades': 50,
            'max_drawdown': 0.05
        }

        mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock()])
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, [MagicMock(), MagicMock()])

        result = chart_generator.generate_performance_summary(metrics)

        mock_plt.subplots.assert_called()


@pytest.mark.unit
class TestSocialMediaIntegration:
    """Test SocialMediaIntegration class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock notification config."""
        config = Mock(spec=NotificationConfig)
        config.telegram = Mock()
        config.telegram.is_configured = True
        config.twitter = Mock()
        config.twitter.is_configured = False
        config.linkedin = Mock()
        config.linkedin.is_configured = False
        return config

    @pytest.fixture
    def integration(self, mock_config):
        """Create SocialMediaIntegration instance."""
        with patch('notifications.integration.SocialMediaNotifier') as mock_notifier:
            mock_notifier.return_value = Mock()
            return SocialMediaIntegration(mock_config)

    def test_init(self, mock_config):
        """Test SocialMediaIntegration initialization."""
        with patch('notifications.integration.SocialMediaNotifier') as mock_notifier:
            mock_notifier.return_value = Mock()
            integration = SocialMediaIntegration(mock_config)

            assert integration.config == mock_config
            assert integration.notifier is not None

    @patch('notifications.integration.NotificationConfig')
    def test_from_env(self, mock_config_class):
        """Test creating integration from environment."""
        mock_config = Mock()
        mock_config_class.from_env.return_value = mock_config

        with patch('notifications.integration.SocialMediaNotifier') as mock_notifier:
            mock_notifier.return_value = Mock()
            integration = SocialMediaIntegration.from_env()

            assert integration.config == mock_config

    def test_connect_performance_monitor(self, integration):
        """Test connecting performance monitor."""
        mock_monitor = Mock()
        mock_monitor.add_callback = Mock()

        integration.connect_performance_monitor(mock_monitor)

        # Should have registered callbacks
        assert mock_monitor.add_callback.called

    def test_connect_retraining_scheduler(self, integration):
        """Test connecting retraining scheduler."""
        mock_scheduler = Mock()
        mock_scheduler.add_callback = Mock()

        integration.connect_retraining_scheduler(mock_scheduler)

        # Should have registered callbacks
        assert mock_scheduler.add_callback.called

    def test_post_trade_result(self, integration):
        """Test posting trade result."""
        trade_data = TradeData(
            trade_id="test_1",
            symbol="EURUSD",
            direction="LONG",
            entry_price=1.1000,
            exit_price=1.1050,
            pnl=50.0,
            pnl_pct=0.5,
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            duration_minutes=60
        )

        integration.notifier.post_trade_result = Mock(return_value=True)

        result = integration.post_trade_result(trade_data)

        assert integration.notifier.post_trade_result.called

    def test_post_daily_summary(self, integration):
        """Test posting daily summary."""
        mock_monitor = Mock()
        mock_monitor.get_daily_summary.return_value = {
            'total_trades': 10,
            'total_pnl': 100,
            'win_rate': 0.6
        }

        integration.performance_monitor = mock_monitor
        integration.notifier.post_performance_update = Mock(return_value=True)

        result = integration.post_daily_summary()

        # Should attempt to post
        assert mock_monitor.get_daily_summary.called or integration.notifier.post_performance_update.called

    def test_post_weekly_summary(self, integration):
        """Test posting weekly summary."""
        mock_monitor = Mock()
        mock_monitor.get_weekly_summary.return_value = {
            'total_trades': 50,
            'total_pnl': 500,
            'win_rate': 0.55
        }

        integration.performance_monitor = mock_monitor
        integration.notifier.post_performance_update = Mock(return_value=True)

        result = integration.post_weekly_summary()

        # Should attempt to post
        assert mock_monitor.get_weekly_summary.called or integration.notifier.post_performance_update.called

    def test_start_scheduler(self, integration):
        """Test starting scheduler."""
        integration.start_scheduler(daily_hour=22, weekly_day=6, weekly_hour=20)

        # Scheduler should be initialized
        assert integration.scheduler is not None

    def test_stop_scheduler(self, integration):
        """Test stopping scheduler."""
        integration.scheduler = Mock()
        integration.scheduler.stop = Mock()

        integration.stop_scheduler()

        if integration.scheduler:
            integration.scheduler.stop.assert_called()

    def test_get_status(self, integration):
        """Test getting integration status."""
        integration.performance_monitor = Mock()
        integration.retraining_scheduler = Mock()

        status = integration.get_status()

        assert isinstance(status, dict)
        assert 'enabled_platforms' in status or 'status' in status


@pytest.mark.unit
class TestSetupSocialNotifications:
    """Test setup_social_notifications function."""

    @patch('notifications.integration.SocialMediaIntegration')
    def test_setup_social_notifications(self, mock_integration_class):
        """Test setup function."""
        mock_monitor = Mock()
        mock_scheduler = Mock()
        mock_integration = Mock()
        mock_integration_class.from_env.return_value = mock_integration

        result = setup_social_notifications(
            performance_monitor=mock_monitor,
            retraining_scheduler=mock_scheduler,
            start_scheduler=True
        )

        assert result == mock_integration
        mock_integration.connect_performance_monitor.assert_called_with(mock_monitor)
        mock_integration.connect_retraining_scheduler.assert_called_with(mock_scheduler)

