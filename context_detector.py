#!/usr/bin/env python3
"""Background Ollama vision context detection for sentinelCam-worker."""
from __future__ import annotations

import base64
import ipaddress
import json
import logging
import queue
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Iterable, Optional
from urllib.parse import urlparse

import cv2  # type: ignore
import numpy as np  # type: ignore


_log = logging.getLogger("sentinelCam.context")


def _resolve_host_addresses(hostname: str) -> list[ipaddress._BaseAddress]:
    try:
        infos = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return []
    addrs: list[ipaddress._BaseAddress] = []
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        try:
            addrs.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    return addrs


def _validate_ollama_host(host: str) -> str:
    """Reject hosts that point to cloud-metadata or otherwise dangerous targets.

    We still allow private/loopback ranges by default because Ollama almost
    always runs locally; we only reject categories that have no legitimate
    Ollama deployment story (link-local + reserved/multicast/cloud-metadata).
    """
    candidate = (host or "").strip()
    if not candidate:
        raise ValueError("Ollama host must not be empty")
    parsed = urlparse(candidate if "://" in candidate else f"http://{candidate}")
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Ollama host scheme {scheme!r} is not supported (use http/https)")
    hostname = (parsed.hostname or "").strip()
    if not hostname:
        raise ValueError("Ollama host must include a hostname")

    # Reject cloud-metadata IPs explicitly — these are the well-known
    # SSRF-pivot targets that should never be a legitimate Ollama endpoint.
    blocked_literal = {"169.254.169.254", "fd00:ec2::254"}
    if hostname in blocked_literal:
        raise ValueError(f"Refusing to connect to cloud-metadata address {hostname!r}")

    addrs: list[ipaddress._BaseAddress] = []
    try:
        addrs = [ipaddress.ip_address(hostname)]
    except ValueError:
        addrs = _resolve_host_addresses(hostname)

    for addr in addrs:
        if str(addr) in blocked_literal:
            raise ValueError(f"Refusing to connect to cloud-metadata address {addr!s}")
        if addr.is_link_local or addr.is_multicast or addr.is_reserved or addr.is_unspecified:
            raise ValueError(f"Refusing to connect to dangerous address {addr!s}")
    return parsed.geturl().rstrip("/")


DEFAULT_CONTEXT_PROMPT = (
    "Describe what is currently happening in this camera frame for a security camera UI. "
    "Use one short sentence. Mention visible people, motion, posture, notable objects, "
    "and anything unusual. Do not invent details that are not visible."
)


class OllamaContextDetector:
    """Samples frames and captions them with a local Ollama vision model."""

    def __init__(
        self,
        *,
        model: str,
        host: str = "http://127.0.0.1:11434",
        prompt: str = DEFAULT_CONTEXT_PROMPT,
        interval: float = 8.0,
        timeout: float = 45.0,
        image_width: int = 640,
        jpeg_quality: int = 82,
        enabled: bool = False,
        state_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.model = (model or "moondream").strip()
        raw_host = (host or "http://127.0.0.1:11434").rstrip("/")
        try:
            self.host = _validate_ollama_host(raw_host)
        except ValueError as exc:
            _log.warning("Refusing Ollama host %r: %s — falling back to 127.0.0.1:11434", raw_host, exc)
            self.host = "http://127.0.0.1:11434"
        self.prompt = prompt or DEFAULT_CONTEXT_PROMPT
        self.interval = max(1.0, float(interval))
        self.timeout = max(1.0, float(timeout))
        self.image_width = max(160, int(image_width))
        self.jpeg_quality = max(10, min(95, int(jpeg_quality)))
        self._enabled = bool(enabled)
        self._state_callback = state_callback
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._generation = 0
        self._busy = False
        self._models: list[str] = []
        self._models_error: Optional[str] = None
        self._models_last_refresh = 0.0
        self._models_refreshing = False
        self._latest: Dict[str, Any] = {
            "enabled": self._enabled,
            "model": self.model,
            "trigger": "interval",
            "cooldown": self.interval,
            "summary": None,
            "detail": None,
            "objects": [],
            "updated_at": None,
            "latency_ms": None,
            "error": None,
            "busy": False,
        }
        self._last_submit = 0.0
        self._thread = threading.Thread(target=self._run, name="sentinelcam-context-ollama", daemon=True)
        self._thread.start()

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)
            self._latest["enabled"] = self._enabled
            self._generation += 1
        self._publish()

    def enabled(self) -> bool:
        with self._lock:
            return bool(self._enabled)

    def latest(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def busy(self) -> bool:
        with self._lock:
            return bool(self._busy)

    def models(self) -> list[str]:
        with self._lock:
            return list(self._models)

    def models_error(self) -> Optional[str]:
        with self._lock:
            return self._models_error

    def refresh_models_async(self, *, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            if self._models_refreshing:
                return
            if not force and now - self._models_last_refresh < 30.0:
                return
            self._models_refreshing = True
        thread = threading.Thread(target=self._refresh_models_worker, name="sentinelcam-ollama-models", daemon=True)
        thread.start()

    def configure(
        self,
        *,
        enabled: Optional[bool] = None,
        model: Optional[str] = None,
        interval: Optional[float] = None,
        trigger: Optional[str] = None,
        cooldown: Optional[float] = None,
    ) -> None:
        with self._lock:
            if enabled is not None:
                self._enabled = bool(enabled)
                self._latest["enabled"] = self._enabled
                self._generation += 1
            if model:
                self.model = str(model).strip()
                self._latest["model"] = self.model
            if interval is not None:
                self.interval = max(1.0, float(interval))
            if cooldown is None:
                cooldown = self.interval
            self._latest["trigger"] = str(trigger or self._latest.get("trigger") or "interval")
            self._latest["cooldown"] = max(1.0, float(cooldown))
        self._publish()

    def emergency_stop(self, reason: str = "AI context detection stopped") -> None:
        with self._lock:
            self._enabled = False
            self._generation += 1
            self._busy = False
            self._latest.update(
                {
                    "enabled": False,
                    "summary": None,
                    "detail": None,
                    "objects": [],
                    "updated_at": time.time(),
                    "latency_ms": None,
                    "error": reason,
                    "busy": False,
                }
            )
        self._drain_queue()
        self._publish()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait({})
        except queue.Full:
            pass
        self._thread.join(timeout=3.0)

    def submit(
        self,
        frame_bgr: np.ndarray,
        objects: Iterable[Dict[str, Any]],
        *,
        force: bool = False,
        bypass_enabled: bool = False,
    ) -> bool:
        if frame_bgr is None:
            return False
        now = time.time()
        with self._lock:
            if not self._enabled and not bypass_enabled:
                return False
            if not force and now - self._last_submit < self.interval:
                return False
            self._last_submit = now
            generation = self._generation

        item = {
            "frame": frame_bgr.copy(),
            "objects": list(objects or [])[:12],
            "submitted_at": now,
            "generation": generation,
            "manual": bool(bypass_enabled),
        }
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            if not force:
                return False
            self._drain_queue()
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                return False

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            except Exception:
                return

    def _publish(self) -> None:
        if self._state_callback is None:
            return
        try:
            self._state_callback(self.latest())
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._stop.is_set():
                break
            frame = item.get("frame")
            if frame is None:
                continue
            objects = item.get("objects") or []
            generation = int(item.get("generation") or 0)
            started = time.time()
            with self._lock:
                self._busy = True
                self._latest["busy"] = True
            self._publish()
            try:
                summary = self._ask_ollama(frame, objects)
                latency_ms = int(round((time.time() - started) * 1000))
                result = {
                    "enabled": self.enabled(),
                    "model": self.model,
                    "trigger": self.latest().get("trigger", "interval"),
                    "cooldown": self.latest().get("cooldown", self.interval),
                    "summary": _one_line(summary, 180),
                    "detail": summary,
                    "objects": objects,
                    "updated_at": time.time(),
                    "latency_ms": latency_ms,
                    "error": None,
                    "busy": False,
                }
            except Exception as exc:
                result = {
                    "enabled": self.enabled(),
                    "model": self.model,
                    "trigger": self.latest().get("trigger", "interval"),
                    "cooldown": self.latest().get("cooldown", self.interval),
                    "summary": None,
                    "detail": None,
                    "objects": objects,
                    "updated_at": time.time(),
                    "latency_ms": int(round((time.time() - started) * 1000)),
                    "error": f"{type(exc).__name__}: {exc}",
                    "busy": False,
                }
            with self._lock:
                self._busy = False
                if generation != self._generation:
                    self._latest["busy"] = False
                    continue
                self._latest.update(result)
            self._publish()

    def _refresh_models_worker(self) -> None:
        names: list[str] = []
        error: Optional[str] = None
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=min(5.0, self.timeout)) as resp:
                body = resp.read()
            parsed = json.loads(body.decode("utf-8", "replace"))
            for item in parsed.get("models", []) or []:
                name = str((item or {}).get("name", "") or "").strip()
                if name:
                    names.append(name)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._models = sorted(set(names), key=str.lower)
            self._models_error = error
            self._models_last_refresh = time.time()
            self._models_refreshing = False
        self._publish()

    def _ask_ollama(self, frame_bgr: np.ndarray, objects: Iterable[Dict[str, Any]]) -> str:
        image_b64 = self._encode_image(frame_bgr)
        prompt = self._build_prompt(objects)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 96,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        parsed = json.loads(body.decode("utf-8", "replace"))
        response = str(parsed.get("response", "") or "").strip()
        if not response:
            raise RuntimeError("Ollama returned an empty response")
        return response

    def _build_prompt(self, objects: Iterable[Dict[str, Any]]) -> str:
        object_lines = []
        for obj in objects:
            label = str(obj.get("label", "") or "").strip()
            if not label:
                continue
            conf = obj.get("conf")
            action = str(obj.get("action", "") or "").strip()
            tid = obj.get("track_id")
            bits = [label]
            if tid not in (None, ""):
                bits.append(f"id={tid}")
            if action:
                bits.append(f"action={action}")
            if isinstance(conf, (int, float)):
                bits.append(f"conf={float(conf):.2f}")
            object_lines.append("- " + ", ".join(bits))
        object_hint = "\nKnown detector objects:\n" + "\n".join(object_lines) if object_lines else ""
        return self.prompt + object_hint

    def _encode_image(self, frame_bgr: np.ndarray) -> str:
        frame = frame_bgr
        h, w = frame.shape[:2]
        if w > self.image_width:
            scale = self.image_width / float(w)
            frame = cv2.resize(frame, (self.image_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            raise RuntimeError("failed to encode context frame")
        return base64.b64encode(buf.tobytes()).decode("ascii")


def _one_line(text: str, limit: int) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "..."
