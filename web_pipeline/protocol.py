"""Mirror of the web-side ``app.streaming.protocol`` envelope.

Kept identical on purpose; if you change one, change both. The web server
and the worker must agree byte-for-byte on the frame header.
"""
from __future__ import annotations

import struct
import time
from typing import Final, NamedTuple

_HEADER = struct.Struct(">BQQ")
HEADER_LEN: Final[int] = _HEADER.size  # 17

MSG_RAW_FRAME: Final[int] = 0x01
MSG_PROCESSED_FRAME: Final[int] = 0x02
MSG_KEYFRAME_REQ: Final[int] = 0x03


class Frame(NamedTuple):
    msg_type: int
    camera_id: int
    capture_ms: int
    payload: bytes


def now_ms() -> int:
    return int(time.time() * 1000)


def encode(msg_type: int, camera_id: int, capture_ms: int, payload: bytes) -> bytes:
    return _HEADER.pack(int(msg_type) & 0xFF, int(camera_id), int(capture_ms)) + payload


def decode(data: bytes) -> Frame:
    if len(data) < HEADER_LEN:
        raise ValueError(f"frame too short ({len(data)} bytes, need {HEADER_LEN})")
    msg_type, camera_id, capture_ms = _HEADER.unpack(data[:HEADER_LEN])
    return Frame(msg_type=msg_type, camera_id=camera_id, capture_ms=capture_ms, payload=data[HEADER_LEN:])
