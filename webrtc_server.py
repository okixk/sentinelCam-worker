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
import socket
import time
from fractions import Fraction
from typing import Any, Dict, Optional, Set
from urllib.parse import urlparse

import fractions
import logging
import threading
from typing import List, Tuple

from aiohttp import web
from aiortc import RTCPeerConnection, RTCRtpSender, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import av

from stream_server import ControlAPI, FrameHub, SecurityConfig

_logger = logging.getLogger("sentinelCam.webrtc")

# ---------------------------------------------------------------------------
#  GPU / HW encoder auto-detection
# ---------------------------------------------------------------------------

# Ordered preference: GPU first, CPU fallback last.
_H264_ENCODER_CANDIDATES: List[Tuple[str, dict, str, str]] = [
    # (codec_name, low-latency options, label, pix_fmt)
    ("h264_nvenc", {"preset": "p4", "tune": "ull", "zerolatency": "1", "rc": "cbr"}, "NVIDIA NVENC", "yuv420p"),
    ("h264_amf",   {"usage": "ultralowlatency", "quality": "speed"},               "AMD AMF",      "nv12"),
    ("h264_qsv",   {"preset": "veryfast", "low_power": "0"},                       "Intel QSV",    "nv12"),
    ("libx264",    {"tune": "zerolatency"},                                         "CPU (libx264)", "yuv420p"),
]


def _detect_best_h264_encoder() -> Tuple[str, dict, str, str]:
    """Probe available H.264 encoders and return (codec_name, options, label, pix_fmt)."""
    import numpy as np

    for codec_name, opts, label, pix_fmt in _H264_ENCODER_CANDIDATES:
        if codec_name not in av.codecs_available:
            continue
        try:
            c = av.CodecContext.create(codec_name, "w")
            c.width = 64
            c.height = 64
            c.bit_rate = 500_000
            c.pix_fmt = pix_fmt
            c.framerate = fractions.Fraction(30, 1)
            c.time_base = fractions.Fraction(1, 30)
            c.options = dict(opts)
            if codec_name == "libx264":
                c.profile = "Baseline"
            frame = av.VideoFrame.from_ndarray(
                np.zeros((64, 64, 3), dtype=np.uint8), format="bgr24"
            )
            frame.pts = 0
            frame.time_base = fractions.Fraction(1, 30)
            list(c.encode(frame))
            list(c.encode(None))  # flush
            _logger.info("H.264 encoder detected: %s (%s)", codec_name, label)
            return codec_name, opts, label, pix_fmt
        except Exception:
            continue

    # Should not happen since libx264 is always bundled, but be safe
    return "libx264", {"tune": "zerolatency"}, "CPU (libx264)", "yuv420p"


def _install_gpu_encoder(codec_name: str, codec_options: dict, pix_fmt: str = "yuv420p") -> None:
    """Monkey-patch aiortc's H264Encoder to use the given codec."""
    try:
        from aiortc.codecs import h264 as _h264_mod
    except ImportError:
        return

    _OrigEncoder = _h264_mod.H264Encoder

    _orig_encode_frame = _OrigEncoder._encode_frame

    def _patched_encode_frame(self, frame, force_keyframe):
        # On first call (or resolution/bitrate change), create codec with our chosen encoder
        if self.codec and (
            frame.width != self.codec.width
            or frame.height != self.codec.height
            or abs(self.target_bitrate - self.codec.bit_rate) / max(self.codec.bit_rate, 1) > 0.1
        ):
            self.buffer_data = b""
            self.buffer_pts = None
            self.codec = None

        if force_keyframe:
            frame.pict_type = av.video.frame.PictureType.I
        else:
            frame.pict_type = av.video.frame.PictureType.NONE

        if self.codec is None:
            self.codec = av.CodecContext.create(codec_name, "w")
            self.codec.width = frame.width
            self.codec.height = frame.height
            self.codec.bit_rate = self.target_bitrate
            self.codec.pix_fmt = pix_fmt
            self.codec.framerate = fractions.Fraction(_h264_mod.MAX_FRAME_RATE, 1)
            self.codec.time_base = fractions.Fraction(1, _h264_mod.MAX_FRAME_RATE)
            opts = dict(codec_options)
            if codec_name == "libx264":
                opts.setdefault("level", "31")
                self.codec.profile = "Baseline"
            self.codec.options = opts

        data_to_send = b""
        for package in self.codec.encode(frame):
            data_to_send += bytes(package)
        if data_to_send:
            yield from self._split_bitstream(data_to_send)

    _OrigEncoder._encode_frame = _patched_encode_frame


# ---------------------------------------------------------------------------
#  Frame-sharing: encode once, distribute to all peer connections
# ---------------------------------------------------------------------------

class SharedEncoder:
    """Encodes frames once via the best available H.264 encoder.

    Subscribers (one per peer connection) read the latest encoded packets.
    """

    def __init__(self, hub: FrameHub, codec_name: str, codec_options: dict, target_bitrate_bps: int, pix_fmt: str = "yuv420p"):
        self._hub = hub
        self._codec_name = codec_name
        self._codec_options = codec_options
        self._pix_fmt = pix_fmt
        self._target_bitrate = max(target_bitrate_bps, 500_000)
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._packets: List[bytes] = []
        self._seq: int = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._encode_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _encode_loop(self) -> None:
        codec = None
        last_ts = 0.0
        while self._running:
            frame_bgr, ts = self._hub.wait_newer_frame(last_ts, 1.0)
            if frame_bgr is None:
                continue
            last_ts = ts

            vf = VideoFrame.from_ndarray(frame_bgr, format="bgr24")

            if codec is None or vf.width != codec.width or vf.height != codec.height:
                codec = av.CodecContext.create(self._codec_name, "w")
                codec.width = vf.width
                codec.height = vf.height
                codec.bit_rate = self._target_bitrate
                codec.pix_fmt = self._pix_fmt
                codec.framerate = fractions.Fraction(30, 1)
                codec.time_base = fractions.Fraction(1, 30)
                opts = dict(self._codec_options)
                if self._codec_name == "libx264":
                    opts.setdefault("level", "31")
                    codec.profile = "Baseline"
                codec.options = opts

            vf.pts = int(ts * 90000)
            vf.time_base = Fraction(1, 90000)
            vf.pict_type = av.video.frame.PictureType.NONE

            try:
                raw = b""
                for pkt in codec.encode(vf):
                    raw += bytes(pkt)
                if not raw:
                    continue
            except Exception:
                codec = None
                continue

            with self._cond:
                self._packets = self._split_bitstream(raw)
                self._seq += 1
                self._cond.notify_all()

    @staticmethod
    def _split_bitstream(buf: bytes) -> List[bytes]:
        """Split H.264 Annex-B bitstream into NAL units."""
        nals: List[bytes] = []
        i = 0
        while True:
            i = buf.find(b"\x00\x00\x01", i)
            if i == -1:
                break
            i += 3
            end = buf.find(b"\x00\x00\x01", i)
            if end == -1:
                nals.append(buf[i:])
                break
            elif end > 0 and buf[end - 1] == 0:
                nals.append(buf[i:end - 1])
            else:
                nals.append(buf[i:end])
        return nals

    def wait_newer(self, last_seq: int, timeout: float = 1.0) -> Tuple[List[bytes], int]:
        """Block until a new encoded frame is available. Returns (nal_list, seq)."""
        end = time.time() + timeout
        with self._cond:
            while self._seq <= last_seq:
                remaining = end - time.time()
                if remaining <= 0:
                    return self._packets, self._seq
                self._cond.wait(timeout=remaining)
            return list(self._packets), self._seq


class SharedVideoTrack(VideoStreamTrack):
    """A track that reads pre-encoded NALs from SharedEncoder and returns them
    as av.Packet so aiortc packetizes without re-encoding."""

    kind = "video"

    def __init__(self, shared: SharedEncoder):
        super().__init__()
        self._shared = shared
        self._last_seq = 0
        self._clock_start: float | None = None
        self._last_pts = 0

    async def recv(self) -> av.Packet:
        nals, seq = await asyncio.to_thread(self._shared.wait_newer, self._last_seq, 1.0)
        while not nals:
            await asyncio.sleep(0.02)
            nals, seq = await asyncio.to_thread(self._shared.wait_newer, self._last_seq, 1.0)
        self._last_seq = seq

        now = time.time()
        if self._clock_start is None:
            self._clock_start = now
            pts = 0
        else:
            pts = max(int((now - self._clock_start) * 90000), self._last_pts + 1)
        self._last_pts = pts

        # Reassemble NALs into Annex-B bitstream
        data = b""
        for nal in nals:
            data += b"\x00\x00\x00\x01" + nal

        packet = av.Packet(data)
        packet.pts = pts
        packet.dts = pts
        packet.time_base = Fraction(1, 90000)
        return packet


def _configure_ice_port_range(port_min: int, port_max: int) -> None:
    """Restrict aioice ICE UDP candidate ports to [port_min, port_max]."""
    try:
        import aioice.ice
    except ImportError:
        return

    _orig_gather = aioice.ice.Connection.gather_candidates

    async def _gather_restricted(self):
        loop = asyncio.get_event_loop()
        _real_create = loop.create_datagram_endpoint
        _allocated: set = set()

        async def _create_in_range(protocol_factory, *, local_addr=None, **kw):
            if local_addr is not None and len(local_addr) == 2 and local_addr[1] == 0:
                host = local_addr[0]
                for port in range(port_min, port_max + 1):
                    if port in _allocated:
                        continue
                    try:
                        result = await _real_create(
                            protocol_factory, local_addr=(host, port), **kw
                        )
                        _allocated.add(port)
                        return result
                    except OSError:
                        continue
                raise OSError(
                    f"No available UDP port in range {port_min}-{port_max}"
                )
            return await _real_create(
                protocol_factory, local_addr=local_addr, **kw
            )

        loop.create_datagram_endpoint = _create_in_range
        try:
            await _orig_gather(self)
        finally:
            loop.create_datagram_endpoint = _real_create

    aioice.ice.Connection.gather_candidates = _gather_restricted


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


def _same_origin(request: web.Request, origin: str) -> bool:
    if not origin:
        return False
    host = (request.headers.get("Host", "") or request.host or "").strip()
    if not host:
        return False
    forwarded_proto = (request.headers.get("X-Forwarded-Proto", "") or "").split(",", 1)[0].strip().lower()
    schemes = [forwarded_proto] if forwarded_proto in ("http", "https") else [request.scheme or "http"]
    for scheme in schemes:
        if origin == f"{scheme}://{host}":
            return True
    return False


def _local_origin_allowed(origin: str, loopback_bind: bool) -> bool:
    if not loopback_bind:
        return False
    if origin == "null":
        return True
    parsed = urlparse(origin)
    host = (parsed.hostname or "").strip().lower()
    return host in ("127.0.0.1", "localhost", "::1")


def _origin_allowed(request: web.Request, origin: str, security: SecurityConfig, loopback_bind: bool) -> bool:
    if not origin:
        return False
    if _same_origin(request, origin):
        return True
    if _local_origin_allowed(origin, loopback_bind):
        return True
    return origin in security.allowed_origins


def _apply_common_headers(response: web.StreamResponse) -> None:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"


def _apply_cors_headers(
    response: web.StreamResponse,
    request: web.Request,
    security: SecurityConfig,
    loopback_bind: bool,
) -> bool:
    origin = _request_origin(request)
    if not _origin_allowed(request, origin, security, loopback_bind):
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
    loopback_bind: bool,
    payload: Dict[str, Any],
    *,
    status: int = 200,
) -> web.Response:
    response = web.json_response(payload, status=status)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    _apply_common_headers(response)
    _apply_cors_headers(response, request, security, loopback_bind)
    return response


def _plain_response(
    request: web.Request,
    security: SecurityConfig,
    loopback_bind: bool,
    text: str,
    *,
    status: int,
) -> web.Response:
    response = web.Response(text=text, status=status, content_type="text/plain", charset="utf-8")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    _apply_common_headers(response)
    _apply_cors_headers(response, request, security, loopback_bind)
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
    loopback_bind: bool,
    *,
    require_auth: bool = True,
) -> Optional[web.Response]:
    origin = _request_origin(request)
    if origin and not _origin_allowed(request, origin, security, loopback_bind):
        return _json_response(request, security, loopback_bind, {"ok": False, "error": "origin not allowed"}, status=403)
    if require_auth and not _is_authenticated(request, security):
        response = _json_response(
            request,
            security,
            loopback_bind,
            {"ok": False, "error": "authentication required"},
            status=401,
        )
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


async def _apply_target_bitrate(sender: Any, target_bitrate_bps: int) -> None:
    if int(target_bitrate_bps) <= 0:
        return

    # aiortc currently applies encoder bitrate through the encoder instance
    # rather than a public sender parameter API.
    for _ in range(60):
        encoder = getattr(sender, "_RTCRtpSender__encoder", None)
        if encoder is not None and hasattr(encoder, "target_bitrate"):
            try:
                encoder.target_bitrate = int(target_bitrate_bps)
            except Exception:
                pass
            return
        await asyncio.sleep(0.05)


async def _run_webrtc_server(
    hub: FrameHub,
    host: str,
    port: int,
    control: Optional[ControlAPI],
    stop_event: Optional[Any],
    security: SecurityConfig,
    codec_preference: str,
    target_bitrate_kbps: int,
    ice_port_min: int,
    ice_port_max: int,
    hw_encoder_name: str,
    hw_encoder_opts: dict,
    hw_encoder_label: str,
    hw_pix_fmt: str,
    use_frame_sharing: bool,
) -> None:
    pcs: Set[RTCPeerConnection] = set()
    boundary = "frame"
    bind_host = (host or "").strip().lower()
    loopback_bind = bind_host in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "::", "")

    if int(ice_port_min) > 0 and int(ice_port_max) >= int(ice_port_min):
        _configure_ice_port_range(int(ice_port_min), int(ice_port_max))

    # Install GPU / HW encoder patch (applies even for frame-sharing's fallback per-client path)
    if hw_encoder_name != "libx264":
        _install_gpu_encoder(hw_encoder_name, hw_encoder_opts, hw_pix_fmt)

    # Frame sharing: single encoder for all clients
    shared_encoder: Optional[SharedEncoder] = None
    if use_frame_sharing:
        bitrate_bps = int(target_bitrate_kbps) * 1000 if int(target_bitrate_kbps) > 0 else 2_500_000
        shared_encoder = SharedEncoder(hub, hw_encoder_name, hw_encoder_opts, bitrate_bps, hw_pix_fmt)
        shared_encoder.start()

    async def handle_options(request: web.Request) -> web.Response:
        if request.path not in (
            "/offer",
            "/api/webrtc/offer",
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
            return _plain_response(request, security, loopback_bind, "Not Found\n", status=404)
        origin = _request_origin(request)
        if not _origin_allowed(request, origin, security, loopback_bind):
            return _json_response(request, security, loopback_bind, {"ok": False, "error": "origin not allowed"}, status=403)
        response = web.Response(status=204)
        _apply_common_headers(response)
        _apply_cors_headers(response, request, security, loopback_bind)
        return response

    async def handle_health(request: web.Request) -> web.Response:
        error = None
        if not security.allow_unauthenticated_health:
            error = _check_origin_and_auth(request, security, loopback_bind, require_auth=True)
        if error is not None:
            return error
        return _json_response(request, security, loopback_bind, {"ok": True})

    async def handle_state(request: web.Request) -> web.Response:
        error = _check_origin_and_auth(request, security, loopback_bind, require_auth=True)
        if error is not None:
            return error
        state: Dict[str, Any] = {}
        if control is not None:
            try:
                state = control.get_state()
            except Exception:
                state = {}
        state = dict(state or {})
        state["webrtc_available"] = True
        state["mjpeg_available"] = True
        state["stream_backend"] = "webrtc"
        return _json_response(request, security, loopback_bind, state)

    async def handle_cmd(request: web.Request) -> web.Response:
        error = _check_origin_and_auth(request, security, loopback_bind, require_auth=True)
        if error is not None:
            return error

        if request.content_length is not None and request.content_length > int(security.max_cmd_bytes):
            return _json_response(request, security, loopback_bind, {"ok": False, "error": "request body too large"}, status=413)

        try:
            raw = await request.read()
        except Exception:
            return _json_response(request, security, loopback_bind, {"ok": False, "error": "could not read request body"}, status=400)

        if len(raw) > int(security.max_cmd_bytes):
            return _json_response(request, security, loopback_bind, {"ok": False, "error": "request body too large"}, status=413)

        content_type = (request.content_type or "").strip().lower()
        try:
            if content_type == "application/json":
                payload: Any = json.loads(raw.decode("utf-8") or "{}")
            else:
                payload = {"cmd": raw.decode("utf-8", "ignore")}
        except Exception:
            return _json_response(request, security, loopback_bind, {"ok": False, "error": "invalid request body"}, status=400)

        if control is not None:
            try:
                control.command(payload)
            except Exception:
                pass

        return _json_response(request, security, loopback_bind, {"ok": True})

    async def handle_offer(request: web.Request) -> web.Response:
        error = _check_origin_and_auth(request, security, loopback_bind, require_auth=True)
        if error is not None:
            return error

        if request.content_length is not None and request.content_length > int(security.max_cmd_bytes):
            return _json_response(request, security, loopback_bind, {"ok": False, "error": "request body too large"}, status=413)

        try:
            params = await request.json()
        except Exception:
            return _json_response(request, security, loopback_bind, {"ok": False, "error": "invalid WebRTC offer"}, status=400)

        sdp = str(params.get("sdp", "") or "")
        offer_type = str(params.get("type", "") or "")
        if not sdp or offer_type != "offer":
            return _json_response(request, security, loopback_bind, {"ok": False, "error": "missing offer.sdp/type"}, status=400)

        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            if pc.connectionState in ("failed", "closed", "disconnected"):
                pcs.discard(pc)
                await pc.close()

        if shared_encoder is not None:
            track = SharedVideoTrack(shared_encoder)
        else:
            track = HubVideoStreamTrack(hub)
        transceiver = pc.addTransceiver(track, direction="sendonly")
        _apply_codec_preference(transceiver, codec_preference)

        try:
            await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=offer_type))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            if int(target_bitrate_kbps) > 0:
                asyncio.create_task(_apply_target_bitrate(transceiver.sender, int(target_bitrate_kbps) * 1000))
        except Exception as exc:
            pcs.discard(pc)
            try:
                await pc.close()
            except Exception:
                pass
            return _json_response(
                request,
                security,
                loopback_bind,
                {"ok": False, "error": f"WebRTC negotiation failed: {type(exc).__name__}: {exc}"},
                status=400,
            )

        return _json_response(
            request,
            security,
            loopback_bind,
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        )

    async def handle_offer_info(request: web.Request) -> web.Response:
        error = _check_origin_and_auth(request, security, loopback_bind, require_auth=False)
        if error is not None:
            return error
        return _json_response(
            request,
            security,
            loopback_bind,
            {
                "ok": True,
                "endpoint": "/api/webrtc/offer",
                "method": "POST",
                "message": "This is a WebRTC signaling endpoint, not a direct video URL. Open /webrtc-test or use your HTML page to POST an SDP offer.",
            },
        )

    async def handle_frame(request: web.Request) -> web.Response:
        error = _check_origin_and_auth(request, security, loopback_bind, require_auth=True)
        if error is not None:
            return error

        jpeg, _ = await asyncio.to_thread(hub.latest)
        if not jpeg:
            return _plain_response(request, security, loopback_bind, "No frame yet\n", status=503)

        response = web.Response(body=jpeg, content_type="image/jpeg")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        _apply_common_headers(response)
        _apply_cors_headers(response, request, security, loopback_bind)
        return response

    async def handle_mjpeg_stream(request: web.Request) -> web.StreamResponse:
        error = _check_origin_and_auth(request, security, loopback_bind, require_auth=True)
        if error is not None:
            return error

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": f"multipart/x-mixed-replace; boundary={boundary}",
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        _apply_common_headers(response)
        _apply_cors_headers(response, request, security, loopback_bind)
        await response.prepare(request)

        last_ts = 0.0
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                jpeg, last_ts = await asyncio.to_thread(hub.wait_newer, last_ts, 1.0)
                if not jpeg:
                    continue
                await response.write(f"--{boundary}\r\n".encode("utf-8"))
                await response.write(b"Content-Type: image/jpeg\r\n")
                await response.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("utf-8"))
                await response.write(jpeg)
                await response.write(b"\r\n")
        except (asyncio.CancelledError, ConnectionResetError, RuntimeError):
            pass
        finally:
            try:
                await response.write_eof()
            except Exception:
                pass

        return response

    async def handle_test_page(request: web.Request) -> web.Response:
        error = _check_origin_and_auth(request, security, loopback_bind, require_auth=False)
        if error is not None:
            return error
        response = web.Response(text=TEST_PAGE_HTML, content_type="text/html", charset="utf-8")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        _apply_common_headers(response)
        _apply_cors_headers(response, request, security, loopback_bind)
        return response

    app = web.Application()
    app.router.add_route("OPTIONS", "/{tail:.*}", handle_options)
    app.router.add_get("/", handle_test_page)
    app.router.add_get("/webrtc-test", handle_test_page)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/stream.mjpg", handle_mjpeg_stream)
    app.router.add_get("/mjpeg", handle_mjpeg_stream)
    app.router.add_get("/video", handle_mjpeg_stream)
    app.router.add_get("/video_feed", handle_mjpeg_stream)
    app.router.add_get("/frame.jpg", handle_frame)
    app.router.add_get("/snapshot.jpg", handle_frame)
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
        if shared_encoder is not None:
            shared_encoder.stop()
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
    target_bitrate_kbps: int = 2500,
    ice_port_min: int = 0,
    ice_port_max: int = 0,
    use_gpu: bool = True,
    use_frame_sharing: bool = True,
) -> None:
    security = security or SecurityConfig()

    # Detect best encoder
    if use_gpu:
        enc_name, enc_opts, enc_label, enc_pix_fmt = _detect_best_h264_encoder()
    else:
        enc_name, enc_opts, enc_label, enc_pix_fmt = "libx264", {"tune": "zerolatency"}, "CPU (libx264)", "yuv420p"
    print(f"WebRTC H.264 encoder: {enc_label}")
    if use_frame_sharing:
        print("WebRTC frame sharing: enabled (single encoder for all clients)")

    asyncio.run(
        _run_webrtc_server(
            hub=hub,
            host=host,
            port=int(port),
            control=control,
            stop_event=stop_event,
            security=security,
            codec_preference=codec_preference,
            target_bitrate_kbps=int(target_bitrate_kbps),
            ice_port_min=int(ice_port_min),
            ice_port_max=int(ice_port_max),
            hw_encoder_name=enc_name,
            hw_encoder_opts=enc_opts,
            hw_encoder_label=enc_label,
            hw_pix_fmt=enc_pix_fmt,
            use_frame_sharing=bool(use_frame_sharing),
        )
    )
