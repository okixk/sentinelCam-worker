"""Web-streaming pipeline mode for sentinelCam-worker.

In this mode the worker does NOT publish its own HTTP / WebRTC server.
Instead it dials outbound to a sentinelCam-web instance over a single
authenticated WebSocket, receives raw JPEG frames from the Pi-side
``FrameHub``, processes each frame (YOLO + pose + overlay), and pushes the
result back over the same socket.

Entry point: ``python -m web_pipeline.run``. See README in this directory.
"""
from web_pipeline.protocol import (
    HEADER_LEN,
    MSG_KEYFRAME_REQ,
    MSG_PROCESSED_FRAME,
    MSG_RAW_FRAME,
    Frame,
    decode,
    encode,
    now_ms,
)

__all__ = [
    "HEADER_LEN",
    "MSG_KEYFRAME_REQ",
    "MSG_PROCESSED_FRAME",
    "MSG_RAW_FRAME",
    "Frame",
    "decode",
    "encode",
    "now_ms",
]
