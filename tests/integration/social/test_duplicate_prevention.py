import pytest

from signals.signal_publisher import SignalPublisher, PublisherConfig


@pytest.mark.integration
def test_publisher_duplicate_prevention(monkeypatch):
    sent = []

    class Notifier:
        def send_message(self, msg):
            sent.append(msg)

    cfg = PublisherConfig(
        telegram_enabled=True,
        twitter_enabled=False,
        min_seconds_between_signals=3600,
        min_confidence_to_publish=0.0,
        min_risk_reward_to_publish=0.0,
    )
    pub = SignalPublisher(cfg)
    pub.set_notifier("telegram", Notifier())

    s1 = pub.publish_entry(
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
        confidence=0.9,
    )
    s2 = pub.publish_entry(
        symbol="EURUSD",
        direction="BUY",
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=1.1100,
        confidence=0.9,
    )

    assert s1 is not None
    assert s2 is None
    assert len(sent) == 1
