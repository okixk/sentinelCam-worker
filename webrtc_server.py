#!/usr/bin/env python3
"""
webrtc_server.py (worker-side)

Worker-side WebRTC video sender with HTTP signaling and the same JSON
state / control endpoints used by the web UI.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import time
from fractions import Fraction
from typing import Any, Dict, Optional, Set
from urllib.parse import urlparse

from aiohttp import web
from aiortc import RTCPeerConnection, RTCRtpSender, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from stream_server import ControlAPI, FrameHub, SecurityConfig


TEST_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>sentinelCam WebRTC Test</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f2efe8;
      --panel: #fffaf2;
      --ink: #1f1a17;
      --accent: #c65f2d;
      --muted: #6f655d;
      --border: #d8cbbd;
    }
    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      background: radial-gradient(circle at top, #fff7ea, var(--bg));
      color: var(--ink);
      min-height: 100vh;
    }
    main {
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08);
      overflow: hidden;
    }
    .head {
      padding: 16px 20px;
      border-bottom: 1px solid var(--border);
    }
    h1 {
      font-size: 24px;
      margin: 0 0 8px;
    }
    p {
      margin: 0;
      color: var(--muted);
    }
    video {
      display: block;
      width: 100%;
      background: #000;
      aspect-ratio: 16 / 9;
    }
    .body {
      padding: 16px 20px 20px;
    }
    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-bottom: 14px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      background: var(--accent);
      color: white;
      cursor: pointer;
      font: inherit;
    }
    button.secondary {
      background: #8a8178;
    }
    code, pre {
      font-family: Consolas, "Courier New", monospace;
      font-size: 13px;
    }
    .status {
      font-weight: 600;
    }
    pre {
      margin: 0;
      padding: 14px;
      border-radius: 12px;
      background: #201a17;
      color: #f7efe8;
      overflow: auto;
      max-height: 220px;
    }
  </style>
</head>
<body>
  <main>
    <div class="panel">
      <div class="head">
        <h1>sentinelCam WebRTC Test</h1>
        <p>This page talks to the worker on the same origin and helps verify whether WebRTC works before debugging your external HTML server.</p>
      </div>
      <video id="stream" autoplay playsinline muted></video>
      <div class="body">
        <div class="row">
          <button id="connectBtn" type="button">Connect</button>
          <button id="disconnectBtn" class="secondary" type="button">Disconnect</button>
          <span class="status" id="status">Idle</span>
        </div>
        <pre id="log"></pre>
      </div>
    </div>
  </main>
  <script>
    let pc = null;
    const video = document.getElementById("stream");
    const statusEl = document.getElementById("status");
    const logEl = document.getElementById("log");
    const connectBtn = document.getElementById("connectBtn");
    const disconnectBtn = document.getElementById("disconnectBtn");

    function log(message) {
      const ts = new Date().toLocaleTimeString();
      logEl.textContent += `[${ts}] ${message}\\n`;
      logEl.scrollTop = logEl.scrollHeight;
    }

    function setStatus(message) {
      statusEl.textContent = message;
      log(message);
    }

    async function waitForIceComplete(peer) {
      await new Promise((resolve) => {
        if (peer.iceGatheringState === "complete") {
          resolve();
          return;
        }
        const onStateChange = () => {
          if (peer.iceGatheringState === "complete") {
            peer.removeEventListener("icegatheringstatechange", onStateChange);
            resolve();
          }
        };
        peer.addEventListener("icegatheringstatechange", onStateChange);
      });
    }

    async function disconnect() {
      if (pc) {
        try {
          pc.ontrack = null;
          pc.close();
        } catch (err) {
          log(`close error: ${err}`);
        }
        pc = null;
      }
      video.srcObject = null;
      setStatus("Disconnected");
    }

    async function connect() {
      await disconnect();
      setStatus("Creating peer connection");
      pc = new RTCPeerConnection();
      pc.addTransceiver("video", { direction: "recvonly" });

      pc.ontrack = (event) => {
        const stream = event.streams && event.streams[0];
        if (stream) {
          video.srcObject = stream;
          setStatus("Receiving video");
        }
      };

      pc.onconnectionstatechange = () => {
        log(`connectionState=${pc.connectionState}`);
      };

      pc.oniceconnectionstatechange = () => {
        log(`iceConnectionState=${pc.iceConnectionState}`);
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await waitForIceComplete(pc);

      setStatus("Sending offer");
      const response = await fetch("/api/webrtc/offer", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type
        })
      });

      if (!response.ok) {
        const body = await response.text();
        throw new Error(`offer failed: ${response.status} ${body}`);
      }

      const answer = await response.json();
      await pc.setRemoteDescription(answer);
      setStatus("Connected, waiting for video");
    }

    connectBtn.addEventListener("click", () => {
      connect().catch((err) => {
        setStatus(`Failed: ${err}`);
      });
    });

    disconnectBtn.addEventListener("click", () => {
      disconnect().catch((err) => {
        setStatus(`Disconnect failed: ${err}`);
      });
    });

    window.addEventListener("beforeunload", () => {
      if (pc) {
        pc.close();
      }
    });
  </script>
</body>
</html>
"""


def _request_origin(request: web.Request) -> str:
    origin = (request.headers.get("Origin", "") or "").strip()
    if origin:
        return origin.rstrip("/")
    referer = (request.headers.get("Referer", "") or "").strip()
    if not referer:
        return ""
    parsed = urlparse(referer)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def _origin_allowed(origin: str, security: SecurityConfig) -> bool:
    if not origin:
        return False
    return origin in security.allowed_origins


def _apply_common_headers(response: web.StreamResponse) -> None:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"


def _apply_cors_headers(response: web.StreamResponse, request: web.Request, security: SecurityConfig) -> bool:
    origin = _request_origin(request)
    if not _origin_allowed(origin, security):
        return False
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-SentinelCam-Token"
    response.headers["Access-Control-Max-Age"] = "600"
    return True


def _json_response(
    request: web.Request,
    security: SecurityConfig,
    payload: Dict[str, Any],
    *,
    status: int = 200,
) -> web.Response:
    response = web.json_response(payload, status=status)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    _apply_common_headers(response)
    _apply_cors_headers(response, request, security)
    return response


def _plain_response(
    request: web.Request,
    security: SecurityConfig,
    text: str,
    *,
    status: int,
) -> web.Response:
    response = web.Response(text=text, status=status, content_type="text/plain", charset="utf-8")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    _apply_common_headers(response)
    _apply_cors_headers(response, request, security)
    return response


def _is_authenticated(request: web.Request, security: SecurityConfig) -> bool:
    expected = security.auth_token
    if not expected:
        return True

    authz = (request.headers.get("Authorization", "") or "").strip()
    if authz.lower().startswith("bearer "):
        token = authz[7:].strip()
        if token and secrets.compare_digest(token, expected):
            return True

    header_token = (request.headers.get("X-SentinelCam-Token", "") or "").strip()
    if header_token and secrets.compare_digest(header_token, expected):
        return True

    return False


def _check_origin_and_auth(
    request: web.Request,
    security: SecurityConfig,
    *,
    require_auth: bool = True,
) -> Optional[web.Response]:
    origin = _request_origin(request)
    if origin and not _origin_allowed(origin, security):
        return _json_response(request, security, {"ok": False, "error": "origin not allowed"}, status=403)
    if require_auth and not _is_authenticated(request, security):
        response = _json_response(request, security, {"ok": False, "error": "authentication required"}, status=401)
        response.headers["WWW-Authenticate"] = 'Bearer realm="sentinelCam-worker"'
        return response
    return None


class HubVideoStreamTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, hub: FrameHub):
        super().__init__()
        self._hub = hub
        self._last_ts = 0.0
        self._last_frame = None
        self._clock_start = None
        self._last_pts = 0

    async def recv(self) -> VideoFrame:
        while True:
            frame_bgr, ts = await asyncio.to_thread(self._hub.wait_newer_frame, self._last_ts, 1.0)
            if frame_bgr is None:
                await asyncio.sleep(0.02)
                continue

            if ts > self._last_ts or self._last_frame is None:
                self._last_ts = ts
                self._last_frame = frame_bgr
            else:
                frame_bgr = self._last_frame.copy()

            now = time.time()
            if self._clock_start is None:
                self._clock_start = now
                pts = 0
            else:
                pts = max(int((now - self._clock_start) * 90000), self._last_pts + 1)
            self._last_pts = pts
            frame = VideoFrame.from_ndarray(frame_bgr, format="bgr24")
            frame.pts = pts
            frame.time_base = Fraction(1, 90000)
            return frame


def _apply_codec_preference(transceiver: Any, preference: str) -> None:
    pref = (preference or "auto").strip().lower()
    if pref in ("", "auto"):
        return

    mime_map = {
        "h264": "video/H264",
        "vp8": "video/VP8",
        "vp9": "video/VP9",
        "av1": "video/AV1",
    }
    preferred_mime = mime_map.get(pref)
    if not preferred_mime:
        return

    capabilities = RTCRtpSender.getCapabilities("video")
    codecs = list(getattr(capabilities, "codecs", []) or [])
    if not codecs:
        return

    preferred = [codec for codec in codecs if getattr(codec, "mimeType", "") == preferred_mime]
    if not preferred:
        return
    others = [codec for codec in codecs if getattr(codec, "mimeType", "") != preferred_mime]
    transceiver.setCodecPreferences(preferred + others)


async def _run_webrtc_server(
    hub: FrameHub,
    host: str,
    port: int,
    control: Optional[ControlAPI],
    stop_event: Optional[Any],
    security: SecurityConfig,
    codec_preference: str,
) -> None:
    pcs: Set[RTCPeerConnection] = set()

    async def handle_options(request: web.Request) -> web.Response:
        if request.path not in (
            "/offer",
            "/api/webrtc/offer",
            "/api/state",
            "/api/cmd",
            "/health",
            "/api/health",
        ):
            return _plain_response(request, security, "Not Found\n", status=404)
        origin = _request_origin(request)
        if not _origin_allowed(origin, security):
            return _json_response(request, security, {"ok": False, "error": "origin not allowed"}, status=403)
        response = web.Response(status=204)
        _apply_common_headers(response)
        _apply_cors_headers(response, request, security)
        return response

    async def handle_health(request: web.Request) -> web.Response:
        error = None
        if not security.allow_unauthenticated_health:
            error = _check_origin_and_auth(request, security, require_auth=True)
        if error is not None:
            return error
        return _json_response(request, security, {"ok": True})

    async def handle_state(request: web.Request) -> web.Response:
        error = _check_origin_and_auth(request, security, require_auth=True)
        if error is not None:
            return error
        state: Dict[str, Any] = {}
        if control is not None:
            try:
                state = control.get_state()
            except Exception:
                state = {}
        return _json_response(request, security, state)

    async def handle_cmd(request: web.Request) -> web.Response:
        error = _check_origin_and_auth(request, security, require_auth=True)
        if error is not None:
            return error

        if request.content_length is not None and request.content_length > int(security.max_cmd_bytes):
            return _json_response(request, security, {"ok": False, "error": "request body too large"}, status=413)

        try:
            raw = await request.read()
        except Exception:
            return _json_response(request, security, {"ok": False, "error": "could not read request body"}, status=400)

        if len(raw) > int(security.max_cmd_bytes):
            return _json_response(request, security, {"ok": False, "error": "request body too large"}, status=413)

        content_type = (request.content_type or "").strip().lower()
        try:
            if content_type == "application/json":
                payload: Any = json.loads(raw.decode("utf-8") or "{}")
            else:
                payload = {"cmd": raw.decode("utf-8", "ignore")}
        except Exception:
            return _json_response(request, security, {"ok": False, "error": "invalid request body"}, status=400)

        if control is not None:
            try:
                control.command(payload)
            except Exception:
                pass

        return _json_response(request, security, {"ok": True})

    async def handle_offer(request: web.Request) -> web.Response:
        error = _check_origin_and_auth(request, security, require_auth=True)
        if error is not None:
            return error

        if request.content_length is not None and request.content_length > int(security.max_cmd_bytes):
            return _json_response(request, security, {"ok": False, "error": "request body too large"}, status=413)

        try:
            params = await request.json()
        except Exception:
            return _json_response(request, security, {"ok": False, "error": "invalid WebRTC offer"}, status=400)

        sdp = str(params.get("sdp", "") or "")
        offer_type = str(params.get("type", "") or "")
        if not sdp or offer_type != "offer":
            return _json_response(request, security, {"ok": False, "error": "missing offer.sdp/type"}, status=400)

        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if pc.connectionState in ("failed", "closed", "disconnected"):
                pcs.discard(pc)
                await pc.close()

        track = HubVideoStreamTrack(hub)
        transceiver = pc.addTransceiver(track, direction="sendonly")
        _apply_codec_preference(transceiver, codec_preference)

        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=offer_type))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
        except Exception as exc:
            pcs.discard(pc)
            try:
                await pc.close()
            except Exception:
                pass
            return _json_response(
                request,
                security,
                {"ok": False, "error": f"WebRTC negotiation failed: {type(exc).__name__}: {exc}"},
                status=400,
            )

        return _json_response(
            request,
            security,
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        )

    async def handle_offer_info(request: web.Request) -> web.Response:
        error = _check_origin_and_auth(request, security, require_auth=False)
        if error is not None:
            return error
        return _json_response(
            request,
            security,
            {
                "ok": True,
                "endpoint": "/api/webrtc/offer",
                "method": "POST",
                "message": "This is a WebRTC signaling endpoint, not a direct video URL. Open /webrtc-test or use your HTML page to POST an SDP offer.",
            },
        )

    async def handle_test_page(request: web.Request) -> web.Response:
        error = _check_origin_and_auth(request, security, require_auth=False)
        if error is not None:
            return error
        response = web.Response(text=TEST_PAGE_HTML, content_type="text/html", charset="utf-8")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        _apply_common_headers(response)
        _apply_cors_headers(response, request, security)
        return response

    app = web.Application()
    app.router.add_route("OPTIONS", "/{tail:.*}", handle_options)
    app.router.add_get("/", handle_test_page)
    app.router.add_get("/webrtc-test", handle_test_page)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/state", handle_state)
    app.router.add_get("/offer", handle_offer_info)
    app.router.add_get("/api/webrtc/offer", handle_offer_info)
    app.router.add_post("/api/cmd", handle_cmd)
    app.router.add_post("/offer", handle_offer)
    app.router.add_post("/api/webrtc/offer", handle_offer)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=int(port))
    await site.start()

    try:
        if stop_event is None:
            await asyncio.Future()
        else:
            await asyncio.to_thread(stop_event.wait)
    finally:
        close_tasks = [pc.close() for pc in list(pcs)]
        pcs.clear()
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        await runner.cleanup()


def run_webrtc_server(
    hub: FrameHub,
    host: str = "127.0.0.1",
    port: int = 8080,
    control: Optional[ControlAPI] = None,
    stop_event: Optional[Any] = None,
    security: Optional[SecurityConfig] = None,
    codec_preference: str = "auto",
) -> None:
    security = security or SecurityConfig()
    asyncio.run(
        _run_webrtc_server(
            hub=hub,
            host=host,
            port=int(port),
            control=control,
            stop_event=stop_event,
            security=security,
            codec_preference=codec_preference,
        )
    )
