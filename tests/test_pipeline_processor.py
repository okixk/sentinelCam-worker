"""Smoke tests for the web-pipeline frame processor wiring.

We can't easily exercise a real WebSocket loop in unit tests, so these
checks focus on the contracts both sides of the loop rely on:

* the stub overlay returns a :class:`ProcessedFrame` with the same shape
  as the input,
* a custom processor returning a plain ndarray is still accepted by the
  binary-frame handler (legacy/backwards compat),
* a processor returning detections actually triggers a JSON detection
  message on the WebSocket, but only once per camera per second.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from typing import List

import numpy as np

from web_pipeline import client
from web_pipeline.client import (
    _Stats,
    _handle_binary_frame,
    _maybe_emit_detection,
    stub_overlay,
)
from web_pipeline.inference import ProcessedFrame
from web_pipeline.protocol import MSG_RAW_FRAME, encode


class _FakeWS:
    def __init__(self) -> None:
        self.sent_bytes: List[bytes] = []
        self.sent_json: List[dict] = []

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def send_json(self, payload: dict) -> None:
        self.sent_json.append(payload)


def _make_raw_envelope(cam_id: int = 1) -> bytes:
    frame = np.full((32, 32, 3), 64, dtype=np.uint8)
    import cv2  # type: ignore
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok, "could not build test JPEG"
    return encode(MSG_RAW_FRAME, cam_id, 1234, buf.tobytes())


class StubOverlayTests(unittest.TestCase):
    def test_stub_overlay_returns_processed_frame_same_shape(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        result = asyncio.run(stub_overlay(frame, 0))
        self.assertIsInstance(result, ProcessedFrame)
        self.assertEqual(result.overlay.shape, frame.shape)
        self.assertEqual(result.classes, [])


class BinaryFrameHandlerTests(unittest.TestCase):
    def test_legacy_processor_returning_ndarray_is_accepted(self) -> None:
        async def legacy_processor(frame, _capture_ms):
            return frame  # plain ndarray, the pre-1.0 contract

        ws = _FakeWS()
        stats = _Stats()
        envelope = _make_raw_envelope()
        asyncio.run(
            _handle_binary_frame(ws, envelope, legacy_processor, stats, [1, 90], {})
        )
        self.assertEqual(stats.frames_out, 1)
        self.assertEqual(stats.detections_sent, 0)
        self.assertEqual(ws.sent_json, [])
        self.assertEqual(len(ws.sent_bytes), 1)

    def test_processor_with_classes_emits_detection_message(self) -> None:
        async def yolo_like(frame, _capture_ms):
            return ProcessedFrame(overlay=frame, classes=["person", "dog"])

        ws = _FakeWS()
        stats = _Stats()
        last_sent: dict[int, float] = {}
        envelope = _make_raw_envelope(cam_id=7)
        asyncio.run(_handle_binary_frame(ws, envelope, yolo_like, stats, [1, 90], last_sent))

        self.assertEqual(stats.detections_sent, 1)
        self.assertEqual(len(ws.sent_json), 1)
        payload = ws.sent_json[0]
        self.assertEqual(payload["type"], "detection")
        self.assertEqual(payload["camera_id"], 7)
        self.assertEqual(payload["classes"], ["person", "dog"])
        self.assertIn(7, last_sent)


class DetectionThrottleTests(unittest.TestCase):
    def test_throttle_skips_second_call_within_window(self) -> None:
        ws = _FakeWS()
        stats = _Stats()
        last_sent: dict[int, float] = {}
        async def run() -> None:
            await _maybe_emit_detection(ws, 1, ["person"], stats, last_sent)
            await _maybe_emit_detection(ws, 1, ["person"], stats, last_sent)
        asyncio.run(run())
        self.assertEqual(stats.detections_sent, 1)
        self.assertEqual(len(ws.sent_json), 1)


if __name__ == "__main__":
    unittest.main()
