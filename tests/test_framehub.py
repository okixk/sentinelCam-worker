from __future__ import annotations

import unittest

import numpy as np

from stream_server import FrameHub


class FrameHubTests(unittest.TestCase):
    def test_lazy_jpeg_encoding_is_generated_on_first_read_and_cached(self) -> None:
        hub = FrameHub(jpeg_quality=80, eager_jpeg=False)
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        frame[:, :] = (12, 34, 56)

        hub.update(frame)
        frame[:, :] = (200, 210, 220)

        raw_frame, ts = hub.latest_frame()
        self.assertIsNotNone(raw_frame)
        self.assertGreater(ts, 0.0)
        self.assertEqual(raw_frame[0, 0].tolist(), [12, 34, 56])

        jpeg_first, first_ts = hub.latest()
        jpeg_second, second_ts = hub.latest()
        self.assertIsNotNone(jpeg_first)
        self.assertEqual(first_ts, second_ts)
        self.assertEqual(jpeg_first, jpeg_second)


if __name__ == "__main__":
    unittest.main()
