"""Unit tests for `notifications/__init__.py`.

Test Summary:
    Verifies the `notifications` package exports the expected public API via
    direct attributes and `__all__`.

Test Breakdown:
    - Package import
        - importing `notifications` succeeds
    - Public API
        - core names (config classes, enums, data classes, and integration helpers)
          are available on the package
        - `__all__` contains expected entries
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestNotificationsInit(unittest.TestCase):
    def test_package_exports_expected_names(self):
        import notifications

        expected_attrs = [
            # Configuration
            "NotificationConfig",
            "TelegramConfig",
            "TwitterConfig",
            "LinkedInConfig",
            # Enums
            "Platform",
            "PostType",
            # Data classes
            "TradeData",
            "PerformanceData",
            # Main classes
            "SocialMediaNotifier",
            "SocialMediaIntegration",
            # Utilities
            "MessageFormatter",
            "ChartGenerator",
            # Setup
            "setup_social_notifications",
        ]

        for name in expected_attrs:
            self.assertTrue(hasattr(notifications, name), f"notifications missing {name}")

        self.assertTrue(hasattr(notifications, "__all__"))
        for name in [
            "NotificationConfig",
            "Platform",
            "SocialMediaNotifier",
            "setup_social_notifications",
        ]:
            self.assertIn(name, notifications.__all__)


if __name__ == "__main__":
    unittest.main()
