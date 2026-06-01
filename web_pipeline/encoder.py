"""GPU-accelerated H.264 encoder for the web-streaming pipeline.

The pipeline worker uses this to turn each annotated BGR frame into one or
more H.264 Annex-B access units. The web server relays those access units to
browsers as fragmented-MP4 (``ffmpeg -c:v copy``) without ever re-encoding, so
all the encode cost stays on the GPU box that runs YOLO.

Encoder preference is GPU first (NVIDIA NVENC, AMD AMF, Intel QSV, Apple
VideoToolbox), falling back to CPU libx264. Options are tuned for low latency
(zerolatency / ultra-low-latency, short GOP, no B-frames).

This module imports :mod:`av` (PyAV) lazily so the rest of the pipeline still
imports on hosts without PyAV; :func:`H264Encoder` raises a clear error if it
is missing.
"""
from __future__ import annotations

import fractions
import logging
from typing import List, Optional, Tuple

import numpy as np  # type: ignore


log = logging.getLogger("sentinelCam.worker.encoder")


# Ordered preference: GPU encoders first, CPU libx264 last.
# (codec_name, low-latency options, human label, pixel format)
_H264_ENCODER_CANDIDATES: List[Tuple[str, dict, str, str]] = [
    ("h264_nvenc", {"preset": "p4", "tune": "ull", "zerolatency": "1", "rc": "cbr", "bf": "0"}, "NVIDIA NVENC", "yuv420p"),
    ("h264_amf", {"usage": "ultralowlatency", "quality": "speed", "bf": "0"}, "AMD AMF", "nv12"),
    ("h264_qsv", {"preset": "veryfast", "async_depth": "1", "low_power": "1", "look_ahead": "0", "bf": "0"}, "Intel QSV", "nv12"),
    ("h264_videotoolbox", {"realtime": "1"}, "Apple VideoToolbox", "nv12"),
    ("libx264", {"preset": "veryfast", "tune": "zerolatency", "bf": "0"}, "CPU (libx264)", "yuv420p"),
]


def _normalize_fps(value: int) -> int:
    return max(1, min(int(value or 30), 120))


def detect_best_h264_encoder() -> Tuple[str, dict, str, str]:
    """Probe available H.264 encoders, returning (codec, options, label, pix_fmt)."""
    import av  # type: ignore

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
            frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), dtype=np.uint8), format="bgr24")
            frame.pts = 0
            frame.time_base = fractions.Fraction(1, 30)
            list(c.encode(frame))
            list(c.encode(None))  # flush
            log.info("H.264 encoder selected: %s (%s)", codec_name, label)
            return codec_name, dict(opts), label, pix_fmt
        except Exception as exc:  # pragma: no cover - hardware dependent
            log.debug("H.264 encoder probe failed for %s (%s): %s", codec_name, label, exc)
            continue
    # libx264 is bundled with PyAV's ffmpeg, so this is the guaranteed fallback.
    return "libx264", {"preset": "veryfast", "tune": "zerolatency", "bf": "0"}, "CPU (libx264)", "yuv420p"


# ---------------------------------------------------------------------------
#  Annex-B NAL helpers (shared shape with the web side's keyframe detection)
# ---------------------------------------------------------------------------

def iter_nal_units(buf: bytes):
    """Yield raw NAL unit payloads (without start codes) from an Annex-B buffer."""
    i = 0
    n = len(buf)
    while i < n:
        # find next start code (00 00 01 or 00 00 00 01)
        start = buf.find(b"\x00\x00\x01", i)
        if start == -1:
            break
        nal_start = start + 3
        nxt = buf.find(b"\x00\x00\x01", nal_start)
        if nxt == -1:
            yield buf[nal_start:]
            break
        end = nxt
        # a 4-byte start code (00 00 00 01) has an extra leading zero to trim
        if end > 0 and buf[end - 1] == 0:
            end -= 1
        yield buf[nal_start:end]
        i = nxt


def is_keyframe(buf: bytes) -> bool:
    """True if the Annex-B access unit contains an IDR slice or parameter set."""
    for nal in iter_nal_units(buf):
        if not nal:
            continue
        nal_type = nal[0] & 0x1F
        # 5 = IDR slice, 7 = SPS, 8 = PPS  -> treat as a keyframe boundary
        if nal_type in (5, 7, 8):
            return True
    return False


class H264Encoder:
    """Stateful H.264 encoder producing Annex-B access units from BGR frames."""

    def __init__(
        self,
        *,
        bitrate_bps: int = 4_000_000,
        fps: int = 30,
        gop_seconds: float = 2.0,
        codec_name: Optional[str] = None,
        codec_options: Optional[dict] = None,
        pix_fmt: Optional[str] = None,
    ) -> None:
        try:
            import av  # type: ignore  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyAV (av) is required for H.264 encoding. Install with: pip install av"
            ) from exc

        if codec_name is None:
            codec_name, codec_options, label, pix_fmt = detect_best_h264_encoder()
            self.label = label
        else:
            self.label = codec_name
        self._codec_name = codec_name
        self._codec_options = dict(codec_options or {})
        self._pix_fmt = pix_fmt or "yuv420p"
        self._bitrate = max(int(bitrate_bps), 500_000)
        self._fps = _normalize_fps(fps)
        self._gop = max(1, int(self._fps * max(0.2, gop_seconds)))
        self._codec = None  # created lazily once we know width/height
        self._pts = 0
        self._width = 0
        self._height = 0

    @property
    def codec_label(self) -> str:
        return self.label

    def _ensure_codec(self, width: int, height: int) -> None:
        import av  # type: ignore

        if self._codec is not None and width == self._width and height == self._height:
            return
        codec = av.CodecContext.create(self._codec_name, "w")
        codec.width = width
        codec.height = height
        codec.bit_rate = self._bitrate
        codec.pix_fmt = self._pix_fmt
        codec.framerate = fractions.Fraction(self._fps, 1)
        codec.time_base = fractions.Fraction(1, self._fps)
        try:
            codec.gop_size = self._gop
        except Exception:
            self._codec_options.setdefault("g", str(self._gop))
        opts = dict(self._codec_options)
        if self._codec_name == "libx264":
            opts.setdefault("preset", "veryfast")
            opts.setdefault("tune", "zerolatency")
        codec.options = opts
        self._codec = codec
        self._width = width
        self._height = height
        self._pts = 0
        log.info(
            "H.264 codec ready: %s %dx%d @%dfps %dkbps gop=%d",
            self._codec_name, width, height, self._fps, self._bitrate // 1000, self._gop,
        )

    def encode(self, frame_bgr: np.ndarray, *, force_keyframe: bool = False) -> List[bytes]:
        """Encode one BGR frame, returning a list of Annex-B access-unit byte strings."""
        import av  # type: ignore

        if frame_bgr is None or frame_bgr.size == 0:
            return []
        height, width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        self._ensure_codec(width, height)
        assert self._codec is not None

        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame_bgr), format="bgr24")
        if force_keyframe:
            vf.pict_type = av.video.frame.PictureType.I
        else:
            vf.pict_type = av.video.frame.PictureType.NONE
        vf.pts = self._pts
        vf.time_base = fractions.Fraction(1, self._fps)
        self._pts += 1

        out: List[bytes] = []
        try:
            for pkt in self._codec.encode(vf):
                data = bytes(pkt)
                if data:
                    out.append(data)
        except Exception:
            log.exception("H.264 encode failed; resetting codec")
            self._codec = None
            return []
        return out

    def flush(self) -> List[bytes]:
        if self._codec is None:
            return []
        out: List[bytes] = []
        try:
            for pkt in self._codec.encode(None):
                data = bytes(pkt)
                if data:
                    out.append(data)
        except Exception:
            log.debug("H.264 flush failed", exc_info=True)
        return out

    def close(self) -> None:
        self._codec = None
