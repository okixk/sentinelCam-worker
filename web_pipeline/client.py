"""WebSocket client that dials sentinelCam-web and runs the inference loop.

Skeleton implementation:
- Connect outbound to wss://<web>/api/worker/connect with a Bearer token.
- Receive raw JPEG frames via the binary envelope, run :func:`process_frame`
  (currently a stub that overlays a "PROCESSING" banner), encode back, push
  the result through the same socket.
- Send a JSON heartbeat every five seconds with basic worker stats so the
  web server's admin status panel knows we are alive.

YOLO + NVENC + pose are intentionally left for a follow-up commit. They
plug in by replacing :func:`process_frame` with the real inference path.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import aiohttp
import cv2  # type: ignore
import numpy as np  # type: ignore

from web_pipeline.protocol import (
    MSG_PROCESSED_FRAME,
    MSG_RAW_FRAME,
    Frame,
    decode,
    encode,
    now_ms,
)


log = logging.getLogger("sentinelCam.worker.client")


# A function that takes (raw_bgr_frame, capture_ms) and returns the processed
# BGR frame (with overlay). Replace this when YOLO + pose land.
FrameProcessor = Callable[[np.ndarray, int], Awaitable[np.ndarray]]


@dataclass
class WorkerConfig:
    web_url: str
    token: str
    name: str = "worker-default"
    jpeg_quality: int = 88
    heartbeat_interval: float = 5.0
    reconnect_initial: float = 1.0
    reconnect_max: float = 30.0

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        url = (os.environ.get("WEB_URL") or "").strip()
        token = (os.environ.get("WEB_TOKEN") or "").strip()
        if not url or not token:
            raise SystemExit("WEB_URL and WEB_TOKEN must be set in the environment")
        return cls(
            web_url=url.rstrip("/"),
            token=token,
            name=(os.environ.get("WORKER_NAME") or "worker-default").strip() or "worker-default",
            jpeg_quality=int(os.environ.get("JPEG_QUALITY", "88")),
        )


@dataclass
class _Stats:
    started_at: float = field(default_factory=time.time)
    frames_in: int = 0
    frames_out: int = 0
    last_in_at: float = 0.0
    last_out_at: float = 0.0
    last_processing_ms: float = 0.0

    def snapshot(self) -> dict[str, object]:
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "frames_in": self.frames_in,
            "frames_out": self.frames_out,
            "last_in_at": self.last_in_at or None,
            "last_out_at": self.last_out_at or None,
            "last_processing_ms": round(self.last_processing_ms, 2),
        }


async def stub_overlay(frame_bgr: np.ndarray, capture_ms: int) -> np.ndarray:
    """Placeholder overlay so the pipeline is end-to-end testable.

    Real inference replaces this — see ``README_PIPELINE.md`` for the plan.
    """
    out = frame_bgr.copy()
    cv2.putText(
        out,
        f"PROCESSING (no model loaded)  t={capture_ms}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


async def run_worker(config: WorkerConfig, processor: Optional[FrameProcessor] = None) -> None:
    processor = processor or stub_overlay
    stats = _Stats()
    backoff = config.reconnect_initial
    stop = asyncio.Event()

    def _stop_handler(*_args: object) -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _stop_handler)

    while not stop.is_set():
        try:
            log.info("connecting to %s", config.web_url)
            async with aiohttp.ClientSession() as session:
                ws_url = config.web_url.replace("https://", "wss://").replace("http://", "ws://")
                async with session.ws_connect(
                    f"{ws_url}/api/worker/connect",
                    headers={"Authorization": f"Bearer {config.token}"},
                    heartbeat=20,
                    max_msg_size=8 * 1024 * 1024,
                ) as ws:
                    backoff = config.reconnect_initial
                    log.info("worker connected; awaiting frames")

                    heartbeat_task = asyncio.create_task(_heartbeat_loop(ws, stats, config.heartbeat_interval, stop))
                    try:
                        await _consume_frames(ws, processor, stats, config, stop)
                    finally:
                        heartbeat_task.cancel()
                        with contextlib.suppress(Exception):
                            await heartbeat_task
        except aiohttp.ClientConnectionError as exc:
            log.warning("connection lost: %s", exc)
        except Exception:
            log.exception("worker loop crashed")

        if stop.is_set():
            break
        log.info("reconnecting in %.1fs", backoff)
        try:
            await asyncio.wait_for(stop.wait(), timeout=backoff)
        except asyncio.TimeoutError:
            pass
        backoff = min(backoff * 2, config.reconnect_max)


async def _heartbeat_loop(ws: aiohttp.ClientWebSocketResponse, stats: _Stats, interval: float, stop: asyncio.Event) -> None:
    while not stop.is_set() and not ws.closed:
        try:
            await ws.send_json({"type": "heartbeat", "status": stats.snapshot()})
        except Exception:
            log.debug("heartbeat send failed", exc_info=True)
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


async def _consume_frames(
    ws: aiohttp.ClientWebSocketResponse,
    processor: FrameProcessor,
    stats: _Stats,
    config: WorkerConfig,
    stop: asyncio.Event,
) -> None:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(config.jpeg_quality)]
    async for msg in ws:
        if stop.is_set():
            break
        if msg.type == aiohttp.WSMsgType.BINARY:
            await _handle_binary_frame(ws, msg.data, processor, stats, encode_params)
        elif msg.type == aiohttp.WSMsgType.TEXT:
            try:
                payload = json.loads(msg.data)
            except Exception:
                continue
            log.debug("server message: %s", payload)
        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
            log.info("server closed the channel: %s", msg)
            return


async def _handle_binary_frame(
    ws: aiohttp.ClientWebSocketResponse,
    data: bytes,
    processor: FrameProcessor,
    stats: _Stats,
    encode_params: list[int],
) -> None:
    try:
        frame = decode(data)
    except ValueError:
        return
    if frame.msg_type != MSG_RAW_FRAME:
        return
    stats.frames_in += 1
    stats.last_in_at = time.time()

    started = time.perf_counter()
    try:
        decoded = cv2.imdecode(np.frombuffer(frame.payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            return
        processed = await processor(decoded, frame.capture_ms)
        ok, buf = cv2.imencode(".jpg", processed, encode_params)
        if not ok:
            return
        out_bytes = buf.tobytes()
    except Exception:
        log.exception("frame processing failed")
        return

    stats.last_processing_ms = (time.perf_counter() - started) * 1000.0
    envelope = encode(MSG_PROCESSED_FRAME, frame.camera_id, frame.capture_ms, out_bytes)
    try:
        await ws.send_bytes(envelope)
    except Exception:
        log.exception("failed to send processed frame back")
        return
    stats.frames_out += 1
    stats.last_out_at = time.time()
