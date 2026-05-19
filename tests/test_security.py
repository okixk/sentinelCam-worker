from __future__ import annotations

import unittest

from security import is_loopback_bind, is_loopback_peer, is_origin_allowed


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

    def test_origin_null_is_not_treated_as_loopback(self) -> None:
        # file:// pages and sandboxed iframes send Origin: null. Even when the
        # worker is bound to loopback we must not whitelist them.
        self.assertFalse(
            is_origin_allowed(
                "null",
                host_header="127.0.0.1:8080",
                loopback_bind=True,
                allowed_origins=(),
            )
        )

    def test_loopback_peer_detection(self) -> None:
        self.assertTrue(is_loopback_peer("127.0.0.1"))
        self.assertTrue(is_loopback_peer("::1"))
        self.assertFalse(is_loopback_peer("10.0.0.1"))
        self.assertFalse(is_loopback_peer(""))
        self.assertFalse(is_loopback_peer("not-an-ip"))


if __name__ == "__main__":
    unittest.main()
