"""WebSocket client that dials sentinelCam-web and runs the inference loop.

Behaviour:

- Dial outbound to ``wss://<web>/api/worker/connect`` with a Bearer token.
- For each incoming raw JPEG frame (binary envelope, type ``MSG_RAW_FRAME``)
  decode it and run the configured processor. The annotated result is shipped
  back in two lanes:
    * ``MSG_PROCESSED_H264`` — one or more H.264 Annex-B access units per frame,
      GPU-encoded, used for the low-latency live view (the web relays them to
      browsers as fragmented-MP4 without re-encoding). One encoder per camera.
    * ``MSG_PROCESSED_FRAME`` — a JPEG of the same frame for snapshots, clips
      and the MJPEG fallback (optionally rate-limited via WORKER_PROCESSED_JPEG_FPS).
- Honour ``MSG_KEYFRAME_REQ`` from the web by forcing an IDR on the next
  encoded frame for that camera (so a freshly-joined viewer starts at a keyframe).
- Emit ``heartbeat`` and ``detection`` JSON text frames carrying ``proto``.

If PyAV / a working H.264 encoder is unavailable, the H.264 lane is disabled
and the worker degrades to JPEG-only (the web then serves MJPEG).
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
from typing import Awaitable, Callable, Dict, Optional, Union

import aiohttp
import cv2  # type: ignore
import numpy as np  # type: ignore

from web_pipeline.inference import ProcessedFrame
from web_pipeline.protocol import (
    PROTOCOL_VERSION,
    MSG_KEYFRAME_REQ,
    MSG_PROCESSED_FRAME,
    MSG_PROCESSED_H264,
    MSG_RAW_FRAME,
    Frame,
    decode,
    encode,
    now_ms,
)


log = logging.getLogger("sentinelCam.worker.client")


# Processor signature. May return either:
#   - a BGR ndarray (legacy: no detections will be emitted)
#   - a ProcessedFrame (overlay + list of detected class labels)
FrameProcessor = Callable[[np.ndarray, int], Awaitable[Union[np.ndarray, ProcessedFrame]]]


# Minimum gap between detection frames per camera.
_DETECTION_MIN_INTERVAL_S = 1.0


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, "")).strip() or default)
    except ValueError:
        return default


@dataclass
class WorkerConfig:
    web_url: str
    token: str
    name: str = "worker-default"
    jpeg_quality: int = 88
    heartbeat_interval: float = 5.0
    reconnect_initial: float = 1.0
    reconnect_max: float = 30.0
    tls_verify: bool = True
    # H.264 live lane
    h264_enabled: bool = True
    video_fps: int = 30
    h264_bitrate_kbps: int = 4000
    h264_gop_seconds: float = 2.0
    # JPEG lane (snapshots / clips / MJPEG fallback); 0 == every frame
    processed_jpeg_fps: float = 0.0

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        url = (os.environ.get("WEB_URL") or "").strip()
        token = (os.environ.get("WEB_TOKEN") or "").strip()
        if not url or not token:
            raise SystemExit("WEB_URL and WEB_TOKEN must be set in the environment")
        tls_verify_raw = (os.environ.get("WEB_TLS_VERIFY") or "").strip().lower()
        tls_verify = tls_verify_raw not in ("0", "false", "no", "off")
        h264_raw = (os.environ.get("WORKER_H264") or "1").strip().lower()
        return cls(
            web_url=url.rstrip("/"),
            token=token,
            name=(os.environ.get("WORKER_NAME") or "worker-default").strip() or "worker-default",
            jpeg_quality=_env_int("JPEG_QUALITY", 88),
            tls_verify=tls_verify,
            h264_enabled=h264_raw not in ("0", "false", "no", "off"),
            video_fps=_env_int("WORKER_VIDEO_FPS", 30),
            h264_bitrate_kbps=_env_int("WORKER_H264_BITRATE_KBPS", 4000),
            h264_gop_seconds=_env_float("WORKER_H264_GOP_SECONDS", 2.0),
            processed_jpeg_fps=_env_float("WORKER_PROCESSED_JPEG_FPS", 0.0),
        )


@dataclass
class _Stats:
    started_at: float = field(default_factory=time.time)
    frames_in: int = 0
    frames_out: int = 0
    h264_units_out: int = 0
    detections_sent: int = 0
    last_in_at: float = 0.0
    last_out_at: float = 0.0
    last_processing_ms: float = 0.0
    encoder: str = "off"

    def snapshot(self) -> dict[str, object]:
        return {
            "uptime_s": round(time.time() - self.started_at, 1),
            "frames_in": self.frames_in,
            "frames_out": self.frames_out,
            "h264_units_out": self.h264_units_out,
            "detections_sent": self.detections_sent,
            "last_in_at": self.last_in_at or None,
            "last_out_at": self.last_out_at or None,
            "last_processing_ms": round(self.last_processing_ms, 2),
            "encoder": self.encoder,
        }


class _EncoderPool:
    """Lazily creates one H.264 encoder per camera id.

    Falls back to JPEG-only (encoders disabled) if PyAV or a working encoder is
    unavailable. Tracks a per-camera "force keyframe" flag the web can set via
    MSG_KEYFRAME_REQ.
    """

    def __init__(self, config: WorkerConfig) -> None:
        self._config = config
        self._encoders: Dict[int, object] = {}
        self._keyframe_pending: set[int] = set()
        self._disabled = not config.h264_enabled
        self.label = "off"
        if not config.h264_enabled:
            log.info("H.264 live lane disabled by config (WORKER_H264=0)")

    def request_keyframe(self, camera_id: int) -> None:
        self._keyframe_pending.add(int(camera_id))

    def _get(self, camera_id: int):
        if self._disabled:
            return None
        enc = self._encoders.get(camera_id)
        if enc is not None:
            return enc
        try:
            from web_pipeline.encoder import H264Encoder

            enc = H264Encoder(
                bitrate_bps=self._config.h264_bitrate_kbps * 1000,
                fps=self._config.video_fps,
                gop_seconds=self._config.h264_gop_seconds,
            )
            self._encoders[camera_id] = enc
            self.label = getattr(enc, "codec_label", "h264")
            # First frame of a fresh encoder is always an IDR.
            self._keyframe_pending.add(camera_id)
            log.info("H.264 encoder created for camera %d (%s)", camera_id, self.label)
            return enc
        except Exception:
            log.exception("could not initialise H.264 encoder; disabling H.264 lane")
            self._disabled = True
            return None

    def encode(self, camera_id: int, overlay_bgr: np.ndarray) -> list[bytes]:
        enc = self._get(camera_id)
        if enc is None:
            return []
        force = camera_id in self._keyframe_pending
        if force:
            self._keyframe_pending.discard(camera_id)
        try:
            return enc.encode(overlay_bgr, force_keyframe=force)  # type: ignore[attr-defined]
        except Exception:
            log.exception("H.264 encode failed for camera %d; dropping encoder", camera_id)
            self._encoders.pop(camera_id, None)
            return []


async def stub_overlay(frame_bgr: np.ndarray, capture_ms: int) -> ProcessedFrame:
    """Placeholder overlay so the pipeline is end-to-end testable without YOLO."""
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
    return ProcessedFrame(overlay=out, classes=[])


async def run_worker(config: WorkerConfig, processor: Optional[FrameProcessor] = None) -> None:
    processor = processor or stub_overlay
    stats = _Stats()
    last_detection_sent: dict[int, float] = {}
    last_jpeg_sent: dict[int, float] = {}
    encoders = _EncoderPool(config)
    backoff = config.reconnect_initial
    stop = asyncio.Event()

    def _stop_handler(*_args: object) -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _stop_handler)

    ssl_param: object = True
    if not config.tls_verify:
        log.warning(
            "WEB_TLS_VERIFY=0 — TLS certificate verification disabled. "
            "Connection is encrypted but vulnerable to MITM. Use only as a "
            "temporary workaround until a trusted certificate is in place."
        )
        ssl_param = False

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
                    ssl=ssl_param,
                ) as ws:
                    backoff = config.reconnect_initial
                    log.info("worker connected; awaiting frames")

                    heartbeat_task = asyncio.create_task(
                        _heartbeat_loop(ws, stats, encoders, config.heartbeat_interval, stop)
                    )
                    try:
                        await _consume_frames(
                            ws, processor, stats, config, stop,
                            last_detection_sent, last_jpeg_sent, encoders,
                        )
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


async def _heartbeat_loop(
    ws: aiohttp.ClientWebSocketResponse,
    stats: _Stats,
    encoders: _EncoderPool,
    interval: float,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set() and not ws.closed:
        try:
            stats.encoder = encoders.label
            await ws.send_json({"type": "heartbeat", "proto": PROTOCOL_VERSION, "status": stats.snapshot()})
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
    last_detection_sent: dict[int, float],
    last_jpeg_sent: dict[int, float],
    encoders: _EncoderPool,
) -> None:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(config.jpeg_quality)]
    async for msg in ws:
        if stop.is_set():
            break
        if msg.type == aiohttp.WSMsgType.BINARY:
            await _handle_binary(
                ws, msg.data, processor, stats, config,
                encode_params, last_detection_sent, last_jpeg_sent, encoders,
            )
        elif msg.type == aiohttp.WSMsgType.TEXT:
            _handle_text(msg.data)
        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
            log.info("server closed the channel: %s", msg)
            return


def _handle_text(data: str) -> None:
    try:
        payload = json.loads(data)
    except Exception:
        return
    if isinstance(payload, dict) and payload.get("type") == "hello":
        peer_proto = payload.get("proto")
        if peer_proto is not None and int(peer_proto) != PROTOCOL_VERSION:
            log.warning(
                "protocol version mismatch: web=%s worker=%s — frames may mis-decode",
                peer_proto, PROTOCOL_VERSION,
            )
    log.debug("server message: %s", data[:200])


async def _handle_binary(
    ws: aiohttp.ClientWebSocketResponse,
    data: bytes,
    processor: FrameProcessor,
    stats: _Stats,
    config: WorkerConfig,
    encode_params: list[int],
    last_detection_sent: dict[int, float],
    last_jpeg_sent: dict[int, float],
    encoders: _EncoderPool,
) -> None:
    try:
        frame = decode(data)
    except ValueError:
        return

    if frame.msg_type == MSG_KEYFRAME_REQ:
        encoders.request_keyframe(frame.camera_id)
        log.debug("keyframe requested for camera %d", frame.camera_id)
        return
    if frame.msg_type != MSG_RAW_FRAME:
        return

    stats.frames_in += 1
    stats.last_in_at = time.time()

    started = time.perf_counter()
    try:
        decoded = await asyncio.to_thread(
            cv2.imdecode, np.frombuffer(frame.payload, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if decoded is None:
            return
        result = await processor(decoded, frame.capture_ms)
        if isinstance(result, ProcessedFrame):
            overlay = result.overlay
            classes = result.classes
            det_boxes = getattr(result, "boxes", [])
        else:
            overlay = result
            classes = []
            det_boxes = []
    except Exception:
        log.exception("frame processing failed")
        return

    # Encode both lanes off the event loop in one hop.
    want_jpeg = _jpeg_due(frame.camera_id, config, last_jpeg_sent)
    try:
        nals, jpeg_bytes = await asyncio.to_thread(
            _encode_lanes, encoders, frame.camera_id, overlay, encode_params, want_jpeg
        )
    except Exception:
        log.exception("encode failed")
        return

    stats.last_processing_ms = (time.perf_counter() - started) * 1000.0

    # Live H.264 lane (one ws message per access unit).
    for nal in nals:
        try:
            await ws.send_bytes(encode(MSG_PROCESSED_H264, frame.camera_id, frame.capture_ms, nal))
            stats.h264_units_out += 1
        except Exception:
            log.exception("failed to send H.264 unit")
            return

    # JPEG lane (snapshots / clips / MJPEG fallback).
    if jpeg_bytes is not None:
        try:
            await ws.send_bytes(encode(MSG_PROCESSED_FRAME, frame.camera_id, frame.capture_ms, jpeg_bytes))
            stats.frames_out += 1
            stats.last_out_at = time.time()
        except Exception:
            log.exception("failed to send processed JPEG")
            return

    if classes:
        await _maybe_emit_detection(ws, frame.camera_id, classes, det_boxes, stats, last_detection_sent)


def _jpeg_due(camera_id: int, config: WorkerConfig, last_jpeg_sent: dict[int, float]) -> bool:
    fps = config.processed_jpeg_fps
    if fps <= 0:
        return True
    now = time.monotonic()
    gap = 1.0 / fps
    if now - last_jpeg_sent.get(int(camera_id), 0.0) >= gap:
        last_jpeg_sent[int(camera_id)] = now
        return True
    return False


def _encode_lanes(
    encoders: _EncoderPool,
    camera_id: int,
    overlay: np.ndarray,
    encode_params: list[int],
    want_jpeg: bool,
) -> tuple[list[bytes], Optional[bytes]]:
    nals = encoders.encode(camera_id, overlay)
    jpeg_bytes: Optional[bytes] = None
    if want_jpeg:
        ok, buf = cv2.imencode(".jpg", overlay, encode_params)
        if ok:
            jpeg_bytes = buf.tobytes()
    return nals, jpeg_bytes


async def _maybe_emit_detection(
    ws: aiohttp.ClientWebSocketResponse,
    camera_id: int,
    classes: list[str],
    boxes: list[dict],
    stats: _Stats,
    last_detection_sent: dict[int, float],
) -> None:
    """Send at most one detection JSON frame per camera per ~1s."""
    now = time.monotonic()
    last = last_detection_sent.get(int(camera_id), 0.0)
    if now - last < _DETECTION_MIN_INTERVAL_S:
        return
    last_detection_sent[int(camera_id)] = now
    try:
        await ws.send_json({
            "type": "detection",
            "proto": PROTOCOL_VERSION,
            "camera_id": int(camera_id),
            "classes": list(classes),
            # Normalized {x,y,w,h,label,conf} so the browser can draw the
            # overlay client-side (edge-H.264 streams bypass our re-encode).
            "boxes": list(boxes or []),
            "ts": now_ms(),
        })
        stats.detections_sent += 1
    except Exception:
        log.debug("detection send failed", exc_info=True)
