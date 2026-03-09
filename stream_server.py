#!/usr/bin/env python3
"""
stream_server.py (worker-side)

Minimal MJPEG + JSON control API, *no* HTML UI.

Endpoints:
  GET  /stream.mjpg   -> multipart/x-mixed-replace MJPEG stream
  GET  /frame.jpg     -> latest JPEG frame
  GET  /api/state     -> JSON state (CORS enabled)
  POST /api/cmd       -> JSON command (CORS enabled)
  GET  /health        -> JSON health

Everything else: 404.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlparse

import cv2  # type: ignore
import numpy as np  # type: ignore


@dataclass
class ControlAPI:
    get_state: Callable[[], Dict[str, Any]]
    command: Callable[[Any], None]


@dataclass(frozen=True)
class SecurityConfig:
    auth_token: str = ""
    allowed_origins: Tuple[str, ...] = ()
    max_cmd_bytes: int = 8192
    allow_unauthenticated_health: bool = True


class FrameHub:
    """Holds the latest encoded JPEG. Producer calls update(frame_bgr)."""

    def __init__(self, jpeg_quality: int = 80):
        self.jpeg_quality = int(max(10, min(95, jpeg_quality)))
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._jpeg: Optional[bytes] = None
        self._ts: float = 0.0

    def update(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None:
            return
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return
        data = buf.tobytes()
        with self._cond:
            self._jpeg = data
            self._ts = time.time()
            self._cond.notify_all()

    def latest(self) -> Tuple[Optional[bytes], float]:
        with self._lock:
            return self._jpeg, self._ts

    def wait_newer(self, last_ts: float, timeout: float = 1.0) -> Tuple[Optional[bytes], float]:
        end = time.time() + timeout
        with self._cond:
            while True:
                if self._jpeg is not None and self._ts > last_ts:
                    return self._jpeg, self._ts
                remaining = end - time.time()
                if remaining <= 0:
                    return self._jpeg, self._ts
                self._cond.wait(timeout=remaining)


def run_mjpeg_server(
    hub: FrameHub,
    host: str = "127.0.0.1",
    port: int = 8080,
    control: Optional[ControlAPI] = None,
    stop_event: Optional[threading.Event] = None,
    security: Optional[SecurityConfig] = None,
) -> None:
    boundary = "frame"
    security = security or SecurityConfig()

    class Handler(BaseHTTPRequestHandler):
        server_version = "sentinelCam-worker"
        sys_version = ""

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep it quiet by default
            return

        def end_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            super().end_headers()

        def _request_origin(self) -> str:
            origin = (self.headers.get("Origin", "") or "").strip()
            if origin:
                return origin.rstrip("/")
            referer = (self.headers.get("Referer", "") or "").strip()
            if not referer:
                return ""
            parsed = urlparse(referer)
            if not parsed.scheme or not parsed.netloc:
                return ""
            return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")

        def _origin_allowed(self, origin: str) -> bool:
            if not origin:
                return False
            return origin in security.allowed_origins

        def _add_cors_headers(self) -> bool:
            origin = self._request_origin()
            if not self._origin_allowed(origin):
                return False
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-SentinelCam-Token")
            self.send_header("Access-Control-Max-Age", "600")
            return True

        def _reject(self, status: int, message: str, *, content_type: str = "text/plain; charset=utf-8") -> None:
            body = message.encode("utf-8", "replace")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _reject_json(self, status: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self._add_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _is_authenticated(self) -> bool:
            expected = security.auth_token
            if not expected:
                return True

            authz = (self.headers.get("Authorization", "") or "").strip()
            if authz.lower().startswith("bearer "):
                token = authz[7:].strip()
                if token and secrets.compare_digest(token, expected):
                    return True

            header_token = (self.headers.get("X-SentinelCam-Token", "") or "").strip()
            if header_token and secrets.compare_digest(header_token, expected):
                return True

            return False

        def _require_auth(self) -> bool:
            if self._is_authenticated():
                return True
            self.send_response(401)
            self._add_cors_headers()
            self.send_header("WWW-Authenticate", 'Bearer realm="sentinelCam-worker"')
            self.send_header("Content-Type", "application/json")
            body = json.dumps({"ok": False, "error": "authentication required"}).encode("utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return False

        def _require_allowed_origin(self) -> bool:
            origin = self._request_origin()
            if not origin:
                return True
            if self._origin_allowed(origin):
                return True
            self._reject_json(403, {"ok": False, "error": "origin not allowed"})
            return False

        def do_OPTIONS(self) -> None:  # noqa: N802
            p = urlparse(self.path).path
            if p not in (
                "/stream.mjpg",
                "/mjpeg",
                "/video",
                "/video_feed",
                "/frame.jpg",
                "/snapshot.jpg",
                "/api/state",
                "/api/cmd",
                "/health",
                "/api/health",
            ):
                self._reject(404, "Not Found\n")
                return
            if not self._origin_allowed(self._request_origin()):
                self._reject_json(403, {"ok": False, "error": "origin not allowed"})
                return
            self.send_response(204)
            self._add_cors_headers()
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            p = urlparse(self.path).path

            if p in ("/health", "/api/health"):
                self._handle_health()
                return

            if not self._require_allowed_origin():
                return
            if not self._require_auth():
                return

            if p in ("/stream.mjpg", "/mjpeg", "/video", "/video_feed"):
                self._handle_stream()
                return

            if p in ("/frame.jpg", "/snapshot.jpg"):
                self._handle_frame()
                return

            if p in ("/api/state",):
                self._handle_state()
                return

            # No website/UI here.
            self._reject(404, "Not Found\n")

        def do_POST(self) -> None:  # noqa: N802
            p = urlparse(self.path).path

            if p != "/api/cmd":
                self._reject(404, "Not Found\n")
                return
            if not self._require_allowed_origin():
                return
            if not self._require_auth():
                return

            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except Exception:
                self._reject_json(400, {"ok": False, "error": "invalid content length"})
                return
            if length < 0:
                self._reject_json(400, {"ok": False, "error": "invalid content length"})
                return
            if length > int(security.max_cmd_bytes):
                self._reject_json(413, {"ok": False, "error": "request body too large"})
                return
            raw = self.rfile.read(length) if length > 0 else b""
            payload: Any = None
            content_type = (self.headers.get("Content-Type", "") or "").split(";", 1)[0].strip().lower()
            try:
                if content_type == "application/json":
                    payload = json.loads(raw.decode("utf-8") or "{}")
                else:
                    payload = {"cmd": raw.decode("utf-8", "ignore")}
            except Exception:
                self._reject_json(400, {"ok": False, "error": "invalid request body"})
                return

            if control is not None:
                try:
                    control.command(payload)
                except Exception:
                    pass

            self.send_response(200)
            self._add_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))

        def _handle_health(self) -> None:
            if not security.allow_unauthenticated_health:
                if not self._require_allowed_origin():
                    return
                if not self._require_auth():
                    return
            self.send_response(200)
            self._add_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))

        def _handle_state(self) -> None:
            st: Dict[str, Any] = {}
            if control is not None:
                try:
                    st = control.get_state()
                except Exception:
                    st = {}
            self.send_response(200)
            self._add_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(st).encode("utf-8"))

        def _handle_frame(self) -> None:
            jpeg, _ = hub.latest()
            if not jpeg:
                self.send_response(503)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"No frame yet\n")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Length", str(len(jpeg)))
            self._add_cors_headers()
            self.end_headers()
            self.wfile.write(jpeg)

        def _handle_stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self._add_cors_headers()
            self.end_headers()

            last_ts = 0.0
            try:
                while True:
                    if stop_event is not None and stop_event.is_set():
                        break
                    jpeg, last_ts = hub.wait_newer(last_ts, timeout=1.0)
                    if not jpeg:
                        continue
                    self.wfile.write(f"--{boundary}\r\n".encode("utf-8"))
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("utf-8"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                return

    httpd = ThreadingHTTPServer((host, int(port)), Handler)
    httpd.daemon_threads = True

    stopper: Optional[threading.Thread] = None
    if stop_event is not None:
        def _watch():
            stop_event.wait()
            try:
                httpd.shutdown()
            except Exception:
                pass
        stopper = threading.Thread(target=_watch, name="sentinelcam-httpd-stop", daemon=True)
        stopper.start()

    try:
        httpd.serve_forever(poll_interval=0.2)
    finally:
        try:
            httpd.server_close()
        except Exception:
            pass
