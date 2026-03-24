from __future__ import annotations

import unittest

from security import is_loopback_bind, is_origin_allowed


class WorkerSecurityTests(unittest.TestCase):
    def test_loopback_bind_detection(self) -> None:
        self.assertTrue(is_loopback_bind("127.0.0.1"))
        self.assertTrue(is_loopback_bind("::1"))
        self.assertTrue(is_loopback_bind("localhost"))
        self.assertFalse(is_loopback_bind("0.0.0.0"))
        self.assertFalse(is_loopback_bind("::"))
        self.assertFalse(is_loopback_bind(""))

    def test_origin_allowed_for_true_loopback_only(self) -> None:
        self.assertTrue(
            is_origin_allowed(
                "http://127.0.0.1:3000",
                host_header="127.0.0.1:8080",
                loopback_bind=True,
                allowed_origins=(),
            )
        )
        self.assertFalse(
            is_origin_allowed(
                "http://evil.example",
                host_header="0.0.0.0:8080",
                loopback_bind=False,
                allowed_origins=(),
            )
        )


if __name__ == "__main__":
    unittest.main()
