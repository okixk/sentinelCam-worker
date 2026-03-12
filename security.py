#!/usr/bin/env python3
"""
security.py

Shared security helpers for sentinelCam worker servers (MJPEG and WebRTC).

Provides framework-agnostic origin checking, CORS validation, and bearer-token
authentication so that both stream_server.py and webrtc_server.py use
identical security logic without duplication.
"""
from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
from urllib.parse import urlparse


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


# ---------------------------------------------------------------------------
#  Framework-agnostic helpers (operate on plain strings)
# ---------------------------------------------------------------------------

def parse_origin(origin_header: str, referer_header: str) -> str:
    """Extract the origin string from Origin or Referer headers."""
    origin = (origin_header or "").strip()
    if origin:
        return origin.rstrip("/")
    referer = (referer_header or "").strip()
    if not referer:
        return ""
    parsed = urlparse(referer)
    if not parsed.scheme or not parsed.netloc:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def is_same_origin(
    origin: str,
    host_header: str,
    forwarded_proto: str = "",
    default_scheme: str = "http",
) -> bool:
    """Check whether *origin* matches the Host header (same-origin request)."""
    if not origin or not host_header:
        return False
    host = host_header.strip()
    proto = (forwarded_proto or "").split(",", 1)[0].strip().lower()
    schemes = [proto] if proto in ("http", "https") else [default_scheme]
    for scheme in schemes:
        if origin == f"{scheme}://{host}":
            return True
    return False


def is_local_origin(origin: str, loopback_bind: bool) -> bool:
    """Allow requests from loopback origins when the server is bound locally."""
    if not loopback_bind:
        return False
    if origin == "null":
        return True
    parsed = urlparse(origin)
    host = (parsed.hostname or "").strip().lower()
    return host in ("127.0.0.1", "localhost", "::1")


def is_origin_allowed(
    origin: str,
    host_header: str,
    loopback_bind: bool,
    allowed_origins: Sequence[str],
    forwarded_proto: str = "",
    default_scheme: str = "http",
) -> bool:
    """Master check: is *origin* permitted to access this server?"""
    if not origin:
        return False
    if is_same_origin(origin, host_header, forwarded_proto, default_scheme):
        return True
    if is_local_origin(origin, loopback_bind):
        return True
    return origin in allowed_origins


def check_bearer_token(
    auth_header: str,
    custom_token_header: str,
    expected: str,
) -> bool:
    """Validate a request's bearer token or X-SentinelCam-Token header."""
    if not expected:
        return True

    authz = (auth_header or "").strip()
    if authz.lower().startswith("bearer "):
        token = authz[7:].strip()
        if token and secrets.compare_digest(token, expected):
            return True

    header_token = (custom_token_header or "").strip()
    if header_token and secrets.compare_digest(header_token, expected):
        return True

    return False


def is_loopback_bind(host: str) -> bool:
    """Return True if *host* is a loopback or wildcard bind address."""
    return (host or "").strip().lower() in (
        "127.0.0.1", "localhost", "::1", "0.0.0.0", "::", "",
    )
