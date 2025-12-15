"""Unit tests for `notifications/social_media_example.py`.

Test Summary:
    Verifies the example module is importable and its public example functions
    can be invoked safely in a controlled/test environment.

Test Breakdown:
    - Module import
        - importing `notifications.social_media_example` succeeds
    - Example functions
        - each `example_*` function can be called without raising
          (with notifier/network side-effects mocked)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestSocialMediaExample(unittest.TestCase):
    def test_importable(self):
        import notifications.social_media_example as ex  # noqa: F401

    def test_example_functions_callable(self):
        import notifications.social_media_example as ex

        # Silence prints.
        with patch("builtins.print"):
            # Ensure any posting calls are mocked so the example never reaches network.
            with patch("notifications.social_media_example.SocialMediaNotifier") as mock_notifier_cls:
                mock_notifier = mock_notifier_cls.return_value
                mock_notifier.post_trade_result.return_value = {"telegram": True}
                mock_notifier.post_performance_update.return_value = {"telegram": True}
                mock_notifier.post_alert.return_value = {"telegram": True}
                mock_notifier.post_milestone.return_value = {"telegram": True}

                # These should not raise.
                ex.example_basic_usage()
                ex.example_performance_update()
                ex.example_custom_posting()

            # Full integration uses SocialMediaIntegration.from_env(); patch it.
            with patch("notifications.social_media_example.SocialMediaIntegration") as mock_integration_cls:
                mock_integration = mock_integration_cls.from_env.return_value
                mock_integration.config.dry_run = True
                mock_integration.connect_performance_monitor.return_value = None
                mock_integration.post_daily_summary.return_value = {}
                mock_integration.get_status.return_value = {"ok": True}
                ex.example_full_integration()

            with patch("notifications.social_media_example.setup_social_notifications") as mock_setup:
                mock_setup.return_value.get_status.return_value = {"ok": True}
                ex.example_with_scheduler()

            ex.example_environment_setup()


if __name__ == "__main__":
    unittest.main()
