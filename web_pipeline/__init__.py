"""Web-streaming pipeline mode for sentinelCam-worker.

In this mode the worker does NOT publish its own HTTP / WebRTC server.
Instead it dials outbound to a sentinelCam-web instance over a single
authenticated WebSocket, receives raw JPEG frames from the camera/edge-side
``FrameHub``, processes each frame (YOLO + pose + overlay), and pushes the
result back over the same socket in two lanes:

- H.264 access units (``MSG_PROCESSED_H264``) for the low-latency live view,
  which the web server relays to browsers as fragmented-MP4 without re-encoding;
- JPEG frames (``MSG_PROCESSED_FRAME``) for snapshots, clips and the MJPEG
  fallback.

Entry point: ``python -m web_pipeline.run``. See README in this directory.
"""
from web_pipeline.protocol import (
    HEADER_LEN,
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

__all__ = [
    "HEADER_LEN",
    "PROTOCOL_VERSION",
    "MSG_KEYFRAME_REQ",
    "MSG_PROCESSED_FRAME",
    "MSG_PROCESSED_H264",
    "MSG_RAW_FRAME",
    "Frame",
    "decode",
    "encode",
    "now_ms",
]
