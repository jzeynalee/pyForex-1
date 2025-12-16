# tests/test_signals_signal_publisher.py
"""
Unit tests for signals/signal_publisher.py - Signal publishing module.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import json
from signals.signal_publisher import (
    Platform, SignalType, Signal, SignalPerformance, PublisherConfig,
    SignalPublisher, SignalFormatter, TelegramFormatter, TwitterFormatter,
    LinkedInFormatter, WebhookFormatter,
    TradingBotSignalAdapter, create_signal_publisher_for_bot
)


@pytest.mark.unit
class TestPlatform:
    """Test Platform enum."""

    def test_platform_values(self):
        """Test platform enum values."""
        assert Platform.TELEGRAM.value == "telegram"
        assert Platform.TWITTER.value == "twitter"
        assert Platform.LINKEDIN.value == "linkedin"
        assert Platform.MQL5.value == "mql5"
        assert Platform.WEBHOOK.value == "webhook"
        assert Platform.EMAIL.value == "email"


@pytest.mark.unit
class TestSignalType:
    """Test SignalType enum."""

    def test_signal_type_values(self):
        """Test signal type enum values."""
        assert SignalType.ENTRY.value == "entry"
        assert SignalType.EXIT.value == "exit"
        assert SignalType.UPDATE.value == "update"
        assert SignalType.DAILY_SUMMARY.value == "daily_summary"
        assert SignalType.WEEKLY_SUMMARY.value == "weekly_summary"
        assert SignalType.ALERT.value == "alert"


@pytest.mark.unit
class TestSignal:
    """Test Signal dataclass."""

    def test_signal_creation(self):
        """Test creating Signal."""
        signal = Signal(
            signal_id="SIG-20240101-0001",
            signal_type=SignalType.ENTRY,
            timestamp=datetime.now()
        )

        assert signal.signal_id == "SIG-20240101-0001"
        assert signal.signal_type == SignalType.ENTRY
        assert signal.symbol == ""
        assert signal.published_to == []

    def test_signal_with_trade_details(self):
        """Test Signal with trade details."""
        signal = Signal(
            signal_id="SIG-20240101-0001",
            signal_type=SignalType.ENTRY,
            timestamp=datetime.now(),
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            risk_percent=1.0,
            risk_reward_ratio=2.0,
            confidence=0.75
        )

        assert signal.symbol == "EURUSD"
        assert signal.direction == "BUY"
        assert signal.entry_price == 1.1000
        assert signal.confidence == 0.75

    def test_signal_to_dict(self):
        """Test Signal to_dict method."""
        signal = Signal(
            signal_id="SIG-20240101-0001",
            signal_type=SignalType.ENTRY,
            timestamp=datetime(2024, 1, 1, 12, 0),
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            confidence=0.75
        )

        result = signal.to_dict()

        assert result['signal_id'] == "SIG-20240101-0001"
        assert result['signal_type'] == "entry"
        assert result['symbol'] == "EURUSD"
        assert result['direction'] == "BUY"
        assert result['entry_price'] == 1.1000
        assert 'timestamp' in result


@pytest.mark.unit
class TestSignalPerformance:
    """Test SignalPerformance dataclass."""

    def test_signal_performance_creation(self):
        """Test creating SignalPerformance."""
        performance = SignalPerformance()

        assert performance.total_signals == 0
        assert performance.winning_signals == 0
        assert performance.losing_signals == 0
        assert performance.total_pnl == 0.0

    def test_win_rate_property(self):
        """Test win_rate property calculation."""
        performance = SignalPerformance(
            winning_signals=3,
            losing_signals=2
        )

        assert performance.win_rate == 0.6

    def test_win_rate_no_closed(self):
        """Test win_rate when no closed signals."""
        performance = SignalPerformance()

        assert performance.win_rate == 0.0

    def test_average_pnl_property(self):
        """Test average_pnl property calculation."""
        performance = SignalPerformance(
            winning_signals=2,
            losing_signals=1,
            total_pnl=100.0
        )

        assert performance.average_pnl == pytest.approx(33.33, abs=0.1)

    def test_average_pnl_no_closed(self):
        """Test average_pnl when no closed signals."""
        performance = SignalPerformance()

        assert performance.average_pnl == 0.0


@pytest.mark.unit
class TestPublisherConfig:
    """Test PublisherConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PublisherConfig()

        assert config.telegram_enabled is True
        assert config.twitter_enabled is True
        assert config.linkedin_enabled is False
        assert config.publish_entries is True
        assert config.publish_exits is True
        assert config.min_confidence_to_publish == 0.6
        assert config.min_risk_reward_to_publish == 1.5

    def test_custom_values(self):
        """Test custom configuration."""
        config = PublisherConfig(
            telegram_enabled=False,
            linkedin_enabled=True,
            min_confidence_to_publish=0.75,
            max_signals_per_day=100
        )

        assert config.telegram_enabled is False
        assert config.linkedin_enabled is True
        assert config.min_confidence_to_publish == 0.75
        assert config.max_signals_per_day == 100


@pytest.mark.unit
class TestSignalFormatter:
    """Test SignalFormatter abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Test that SignalFormatter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SignalFormatter()


@pytest.mark.unit
class TestTelegramFormatter:
    """Test TelegramFormatter class."""

    @pytest.fixture
    def config(self):
        """Create default config."""
        return PublisherConfig()

    @pytest.fixture
    def entry_signal(self):
        """Create entry signal."""
        return Signal(
            signal_id="SIG-001",
            signal_type=SignalType.ENTRY,
            timestamp=datetime.now(),
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            risk_reward_ratio=2.0,
            confidence=0.75,
            position_size=0.1
        )

    def test_format_entry_buy(self, entry_signal, config):
        """Test formatting BUY entry signal."""
        formatter = TelegramFormatter()
        result = formatter.format_entry(entry_signal, config)

        assert "BUY EURUSD" in result
        assert "1.1000" in result  # Entry price
        assert "1.0950" in result  # SL
        assert "1.1100" in result  # TP
        assert "🟢" in result  # Buy emoji

    def test_format_entry_sell(self, config):
        """Test formatting SELL entry signal."""
        signal = Signal(
            signal_id="SIG-001",
            signal_type=SignalType.ENTRY,
            timestamp=datetime.now(),
            symbol="EURUSD",
            direction="SELL",
            entry_price=1.1000,
            stop_loss=1.1050,
            take_profit=1.0900
        )

        formatter = TelegramFormatter()
        result = formatter.format_entry(signal, config)

        assert "SELL" in result
        assert "🔴" in result  # Sell emoji

    def test_format_entry_with_confidence(self, entry_signal, config):
        """Test formatting with confidence included."""
        config.include_confidence = True
        formatter = TelegramFormatter()
        result = formatter.format_entry(entry_signal, config)

        assert "Confidence" in result
        assert "75" in result

    def test_format_exit_profit(self, config):
        """Test formatting exit with profit."""
        signal = Signal(
            signal_id="EXIT-001",
            signal_type=SignalType.EXIT,
            timestamp=datetime.now(),
            symbol="EURUSD",
            entry_price=1.1000,
            exit_price=1.1080,
            pnl=800.0,
            pnl_pips=80.0
        )

        formatter = TelegramFormatter()
        result = formatter.format_exit(signal, config)

        assert "CLOSED" in result
        assert "800" in result  # P&L
        assert "✅" in result  # Profit emoji

    def test_format_exit_loss(self, config):
        """Test formatting exit with loss."""
        signal = Signal(
            signal_id="EXIT-001",
            signal_type=SignalType.EXIT,
            timestamp=datetime.now(),
            symbol="EURUSD",
            entry_price=1.1000,
            exit_price=1.0950,
            pnl=-500.0,
            pnl_pips=-50.0
        )

        formatter = TelegramFormatter()
        result = formatter.format_exit(signal, config)

        assert "❌" in result  # Loss emoji

    def test_format_daily_summary(self, config):
        """Test formatting daily summary."""
        performance = SignalPerformance(
            total_signals=5,
            winning_signals=3,
            losing_signals=2,
            total_pnl=200.0,
            total_pips=20.0
        )

        formatter = TelegramFormatter()
        result = formatter.format_daily_summary(performance, [], config)

        assert "DAILY SUMMARY" in result
        assert "5" in result  # Total signals
        assert "3" in result  # Wins
        assert "2" in result  # Losses
        assert "200" in result  # P&L


@pytest.mark.unit
class TestTwitterFormatter:
    """Test TwitterFormatter class."""

    @pytest.fixture
    def config(self):
        """Create default config."""
        return PublisherConfig()

    def test_format_entry_truncated(self, config):
        """Test Twitter entry formatting (280 char limit)."""
        signal = Signal(
            signal_id="SIG-001",
            signal_type=SignalType.ENTRY,
            timestamp=datetime.now(),
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            risk_reward_ratio=2.0
        )

        formatter = TwitterFormatter()
        result = formatter.format_entry(signal, config)

        assert len(result) <= 280
        assert "#EURUSD" in result
        assert "BUY" in result

    def test_format_exit(self, config):
        """Test Twitter exit formatting."""
        signal = Signal(
            signal_id="EXIT-001",
            signal_type=SignalType.EXIT,
            timestamp=datetime.now(),
            symbol="EURUSD",
            pnl=100.0,
            pnl_pips=10.0
        )

        formatter = TwitterFormatter()
        result = formatter.format_exit(signal, config)

        assert len(result) <= 280
        assert "#EURUSD" in result
        assert "#forex" in result


@pytest.mark.unit
class TestLinkedInFormatter:
    """Test LinkedInFormatter class."""

    @pytest.fixture
    def config(self):
        """Create default config."""
        return PublisherConfig()

    def test_format_entry_empty(self, config):
        """Test LinkedIn entry formatting (empty for individual trades)."""
        signal = Signal(
            signal_id="SIG-001",
            signal_type=SignalType.ENTRY,
            timestamp=datetime.now(),
            symbol="EURUSD",
            direction="BUY"
        )

        formatter = LinkedInFormatter()
        result = formatter.format_entry(signal, config)

        assert result == ""

    def test_format_daily_summary(self, config):
        """Test LinkedIn daily summary formatting."""
        performance = SignalPerformance(
            total_signals=10,
            winning_signals=7,
            losing_signals=3,
            total_pips=50.0
        )

        formatter = LinkedInFormatter()
        result = formatter.format_daily_summary(performance, [], config)

        assert "Algorithmic Trading" in result
        assert "Win Rate" in result
        assert "#AlgorithmicTrading" in result


@pytest.mark.unit
class TestWebhookFormatter:
    """Test WebhookFormatter class."""

    @pytest.fixture
    def config(self):
        """Create default config."""
        return PublisherConfig()

    def test_format_entry_json(self, config):
        """Test webhook entry formatting (JSON)."""
        signal = Signal(
            signal_id="SIG-001",
            signal_type=SignalType.ENTRY,
            timestamp=datetime.now(),
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.1000
        )

        formatter = WebhookFormatter()
        result = formatter.format_entry(signal, config)

        # Should be valid JSON
        data = json.loads(result)
        assert data['signal_id'] == "SIG-001"
        assert data['symbol'] == "EURUSD"

    def test_format_daily_summary_json(self, config):
        """Test webhook daily summary formatting (JSON)."""
        performance = SignalPerformance(
            total_signals=5,
            winning_signals=3,
            losing_signals=2,
            total_pnl=100.0,
            total_pips=10.0
        )

        formatter = WebhookFormatter()
        result = formatter.format_daily_summary(performance, [], config)

        data = json.loads(result)
        assert data['type'] == 'daily_summary'
        assert data['total_signals'] == 5
        assert data['wins'] == 3


@pytest.mark.unit
class TestSignalPublisher:
    """Test SignalPublisher class."""

    @pytest.fixture
    def mock_notifier(self):
        """Create a mock notifier."""
        notifier = Mock()
        notifier.send_message = Mock()
        return notifier

    def test_init_default(self):
        """Test default initialization."""
        publisher = SignalPublisher()

        assert publisher.config.telegram_enabled is True
        assert len(publisher._notifiers) == 0
        assert publisher._signal_counter == 0
        assert len(publisher._signals_today) == 0

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = PublisherConfig(
            telegram_enabled=False,
            min_confidence_to_publish=0.75
        )
        publisher = SignalPublisher(config)

        assert publisher.config.telegram_enabled is False
        assert publisher.config.min_confidence_to_publish == 0.75

    def test_set_notifier(self, mock_notifier):
        """Test setting a notifier."""
        publisher = SignalPublisher()

        publisher.set_notifier('telegram', mock_notifier)

        assert Platform.TELEGRAM in publisher._notifiers
        assert publisher._notifiers[Platform.TELEGRAM] == mock_notifier

    def test_set_notifier_invalid_platform(self):
        """Test setting notifier with invalid platform."""
        publisher = SignalPublisher()

        # Should not raise, just log warning
        publisher.set_notifier('invalid', Mock())

        assert len(publisher._notifiers) == 0

    def test_set_notifiers_multiple(self, mock_notifier):
        """Test setting multiple notifiers at once."""
        publisher = SignalPublisher()
        twitter_notifier = Mock()

        publisher.set_notifiers(
            telegram=mock_notifier,
            twitter=twitter_notifier
        )

        assert Platform.TELEGRAM in publisher._notifiers
        assert Platform.TWITTER in publisher._notifiers

    def test_publish_entry_success(self, mock_notifier):
        """Test publishing entry signal successfully."""
        config = PublisherConfig(
            telegram_enabled=True,
            min_confidence_to_publish=0.5
        )
        publisher = SignalPublisher(config)
        publisher.set_notifier('telegram', mock_notifier)

        signal = publisher.publish_entry(
            symbol='EURUSD',
            direction='BUY',
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            confidence=0.75
        )

        assert signal is not None
        assert signal.signal_type == SignalType.ENTRY
        assert signal.symbol == 'EURUSD'
        assert len(publisher._signals_today) == 1

    def test_publish_entry_filtered_low_confidence(self):
        """Test entry filtered due to low confidence."""
        config = PublisherConfig(min_confidence_to_publish=0.8)
        publisher = SignalPublisher(config)

        signal = publisher.publish_entry(
            symbol='EURUSD',
            direction='BUY',
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            confidence=0.5  # Below threshold
        )

        assert signal is None

    def test_publish_entry_filtered_low_risk_reward(self):
        """Test entry filtered due to low risk-reward."""
        config = PublisherConfig(min_risk_reward_to_publish=2.0)
        publisher = SignalPublisher(config)

        signal = publisher.publish_entry(
            symbol='EURUSD',
            direction='BUY',
            entry_price=1.1000,
            stop_loss=1.0990,  # Tight SL
            take_profit=1.1010,  # Tight TP = 1:1 R:R
            confidence=0.75
        )

        assert signal is None

    def test_publish_entry_disabled(self):
        """Test entry not published when disabled."""
        config = PublisherConfig(publish_entries=False)
        publisher = SignalPublisher(config)

        signal = publisher.publish_entry(
            symbol='EURUSD',
            direction='BUY',
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            confidence=0.75
        )

        assert signal is None

    def test_publish_exit_success(self, mock_notifier):
        """Test publishing exit signal successfully."""
        config = PublisherConfig(publish_exits=True, telegram_enabled=True)
        publisher = SignalPublisher(config)
        publisher.set_notifier('telegram', mock_notifier)

        signal = publisher.publish_exit(
            symbol='EURUSD',
            entry_price=1.1000,
            exit_price=1.1080,
            pnl=800.0,
            pnl_pips=80.0
        )

        assert signal is not None
        assert signal.signal_type == SignalType.EXIT
        assert signal.pnl == 800.0
        assert publisher._performance.winning_signals == 1

    def test_publish_exit_loss(self, mock_notifier):
        """Test publishing exit with loss."""
        config = PublisherConfig(publish_exits=True, telegram_enabled=True)
        publisher = SignalPublisher(config)
        publisher.set_notifier('telegram', mock_notifier)

        signal = publisher.publish_exit(
            symbol='EURUSD',
            entry_price=1.1000,
            exit_price=1.0950,
            pnl=-500.0,
            pnl_pips=-50.0
        )

        assert signal is not None
        assert publisher._performance.losing_signals == 1
        assert publisher._performance.total_pnl == -500.0

    def test_publish_daily_summary(self, mock_notifier):
        """Test publishing daily summary."""
        config = PublisherConfig(publish_daily_summary=True, telegram_enabled=True)
        publisher = SignalPublisher(config)
        publisher.set_notifier('telegram', mock_notifier)

        # Add some signals
        publisher.publish_entry('EURUSD', 'BUY', 1.1000, 1.0950, 1.1100, confidence=0.75)
        publisher.publish_exit('EURUSD', 1.1000, 1.1080, 800.0, 80.0)

        result = publisher.publish_daily_summary()

        assert result is True

    def test_publish_daily_summary_no_signals(self):
        """Test daily summary with no signals."""
        config = PublisherConfig(publish_daily_summary=True)
        publisher = SignalPublisher(config)

        result = publisher.publish_daily_summary()

        assert result is False

    def test_publish_alert(self, mock_notifier):
        """Test publishing alert."""
        config = PublisherConfig(telegram_enabled=True)
        publisher = SignalPublisher(config)
        publisher.set_notifier('telegram', mock_notifier)

        publisher.publish_alert("Test alert message")

        # Should call send_message on notifier
        assert mock_notifier.send_message.called

    def test_publish_milestone(self, mock_notifier):
        """Test publishing milestone."""
        config = PublisherConfig(telegram_enabled=True)
        publisher = SignalPublisher(config)
        publisher.set_notifier('telegram', mock_notifier)

        publisher.publish_milestone("100 trades", "Reached 100 successful trades")

        assert mock_notifier.send_message.called

    def test_reset_daily(self):
        """Test resetting daily counters."""
        publisher = SignalPublisher()

        # Add some signals
        publisher._signals_today.append(Mock())
        publisher._daily_performance.total_signals = 5
        publisher._daily_count = 5

        publisher.reset_daily()

        assert len(publisher._signals_today) == 0
        assert publisher._daily_performance.total_signals == 0
        assert publisher._daily_count == 0

    def test_get_performance(self):
        """Test getting performance metrics."""
        publisher = SignalPublisher()
        publisher._performance.total_signals = 10
        publisher._performance.winning_signals = 7

        performance = publisher.get_performance()

        assert performance.total_signals == 10
        assert performance.winning_signals == 7

    def test_get_daily_performance(self):
        """Test getting daily performance."""
        publisher = SignalPublisher()
        publisher._daily_performance.total_signals = 5

        daily_perf = publisher.get_daily_performance()

        assert daily_perf.total_signals == 5

    def test_get_recent_signals(self):
        """Test getting recent signals."""
        publisher = SignalPublisher()

        # Add signals
        signal1 = Signal(
            signal_id="SIG-001",
            signal_type=SignalType.ENTRY,
            timestamp=datetime.now()
        )
        signal2 = Signal(
            signal_id="SIG-002",
            signal_type=SignalType.ENTRY,
            timestamp=datetime.now()
        )

        publisher._signal_history.append(signal1)
        publisher._signal_history.append(signal2)

        recent = publisher.get_recent_signals(count=1)

        assert len(recent) == 1
        assert recent[0].signal_id == "SIG-002"

    def test_rate_limiting_hourly(self):
        """Test hourly rate limiting."""
        config = PublisherConfig(max_signals_per_hour=2)
        publisher = SignalPublisher(config)

        # First two should pass
        assert publisher._check_rate_limits() is True
        publisher._hourly_count = 1
        assert publisher._check_rate_limits() is True

        # Third should fail
        publisher._hourly_count = 2
        assert publisher._check_rate_limits() is False

    def test_rate_limiting_daily(self):
        """Test daily rate limiting."""
        config = PublisherConfig(max_signals_per_day=5)
        publisher = SignalPublisher(config)

        # First 5 should pass
        publisher._daily_count = 4
        assert publisher._check_rate_limits() is True

        # 6th should fail
        publisher._daily_count = 5
        assert publisher._check_rate_limits() is False

    def test_send_to_platform_send_message(self, mock_notifier):
        """Test sending to platform with send_message method."""
        publisher = SignalPublisher()
        publisher._send_to_platform(Platform.TELEGRAM, mock_notifier, "Test message")

        mock_notifier.send_message.assert_called_once_with("Test message")

    def test_send_to_platform_post(self):
        """Test sending to platform with post method."""
        notifier = Mock()
        notifier.post = Mock()
        publisher = SignalPublisher()

        publisher._send_to_platform(Platform.TWITTER, notifier, "Test message")

        notifier.post.assert_called_once_with("Test message")

    def test_send_to_platform_callable(self):
        """Test sending to platform with callable notifier."""
        notifier = Mock()
        publisher = SignalPublisher()

        publisher._send_to_platform(Platform.WEBHOOK, notifier, "Test message")

        notifier.assert_called_once_with("Test message")


@pytest.mark.unit
class TestTradingBotSignalAdapter:
    """Test TradingBotSignalAdapter class."""

    @pytest.fixture
    def mock_bot(self):
        """Create a mock trading bot."""
        bot = Mock()
        bot.on_trade = Mock()
        bot.on_exit = Mock()
        bot.config = Mock()
        bot.config.symbol = 'EURUSD'
        bot._open_positions = {}
        return bot

    @pytest.fixture
    def mock_publisher(self):
        """Create a mock signal publisher."""
        publisher = Mock()
        publisher.publish_entry = Mock(return_value=Mock())
        publisher.publish_exit = Mock(return_value=Mock())
        return publisher

    def test_init(self, mock_bot, mock_publisher):
        """Test adapter initialization."""
        adapter = TradingBotSignalAdapter(mock_bot, mock_publisher)

        assert adapter.bot == mock_bot
        assert adapter.publisher == mock_publisher
        assert adapter.publish_entries is True
        assert adapter.publish_exits is True

    def test_init_custom_flags(self, mock_bot, mock_publisher):
        """Test adapter initialization with custom flags."""
        adapter = TradingBotSignalAdapter(
            mock_bot, mock_publisher,
            publish_entries=False,
            publish_exits=False
        )

        assert adapter.publish_entries is False
        assert adapter.publish_exits is False

    def test_connect(self, mock_bot, mock_publisher):
        """Test connecting adapter to bot."""
        adapter = TradingBotSignalAdapter(mock_bot, mock_publisher)

        adapter.connect()

        # Bot callbacks should be replaced
        assert mock_bot.on_trade == adapter._on_trade_callback
        assert mock_bot.on_exit == adapter._on_exit_callback

    def test_disconnect(self, mock_bot, mock_publisher):
        """Test disconnecting adapter."""
        adapter = TradingBotSignalAdapter(mock_bot, mock_publisher)
        original_on_trade = mock_bot.on_trade

        adapter.connect()
        adapter.disconnect()

        # Should restore original
        assert mock_bot.on_trade == original_on_trade

    def test_on_trade_callback(self, mock_bot, mock_publisher):
        """Test trade callback."""
        adapter = TradingBotSignalAdapter(mock_bot, mock_publisher)
        adapter.connect()

        order = Mock()
        order.symbol = 'EURUSD'
        order.direction = 'BUY'
        order.price = 1.1000
        order.stop_loss = 1.0950
        order.take_profit = 1.1100
        order.volume = 0.1

        decision_dict = {
            'direction_confidence': 0.75,
            'meta_score': 0.8,
            'risk_percent': 1.0,
            'risk_reward_ratio': 2.0,
            'regime': 'TRENDING'
        }

        adapter._on_trade_callback(order, decision_dict)

        mock_publisher.publish_entry.assert_called_once()
        call_kwargs = mock_publisher.publish_entry.call_args[1]
        assert call_kwargs['symbol'] == 'EURUSD'
        assert call_kwargs['direction'] == 'BUY'
        assert call_kwargs['confidence'] == 0.75

    def test_on_exit_callback(self, mock_bot, mock_publisher):
        """Test exit callback."""
        adapter = TradingBotSignalAdapter(mock_bot, mock_publisher)
        adapter.connect()

        # Mock position
        mock_position = Mock()
        mock_position.symbol = 'EURUSD'
        mock_position.entry_price = 1.1000
        mock_position.direction = 1  # BUY
        mock_bot._open_positions['12345'] = mock_position

        adapter._on_exit_callback('12345', 800.0, 'TP_HIT')

        mock_publisher.publish_exit.assert_called_once()
        call_kwargs = mock_publisher.publish_exit.call_args[1]
        assert call_kwargs['symbol'] == 'EURUSD'
        assert call_kwargs['pnl'] == 800.0


@pytest.mark.unit
class TestCreateSignalPublisherForBot:
    """Test create_signal_publisher_for_bot factory function."""

    def test_create_with_no_clients(self):
        """Test creating publisher with no clients."""
        publisher = create_signal_publisher_for_bot()

        assert isinstance(publisher, SignalPublisher)
        assert len(publisher._notifiers) == 0

    def test_create_with_telegram(self):
        """Test creating publisher with Telegram client."""
        telegram_client = Mock()

        publisher = create_signal_publisher_for_bot(telegram_client=telegram_client)

        assert Platform.TELEGRAM in publisher._notifiers

    def test_create_with_multiple_clients(self):
        """Test creating publisher with multiple clients."""
        telegram_client = Mock()
        twitter_client = Mock()
        linkedin_client = Mock()

        publisher = create_signal_publisher_for_bot(
            telegram_client=telegram_client,
            twitter_client=twitter_client,
            linkedin_client=linkedin_client
        )

        assert Platform.TELEGRAM in publisher._notifiers
        assert Platform.TWITTER in publisher._notifiers
        assert Platform.LINKEDIN in publisher._notifiers

    def test_create_with_custom_config(self):
        """Test creating publisher with custom config."""
        config = PublisherConfig(min_confidence_to_publish=0.8)
        telegram_client = Mock()

        publisher = create_signal_publisher_for_bot(
            telegram_client=telegram_client,
            config=config
        )

        assert publisher.config.min_confidence_to_publish == 0.8

