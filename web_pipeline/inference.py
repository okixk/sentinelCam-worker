"""Minimal YOLO inference for the web-streaming pipeline mode.

Runs detection on each incoming raw frame and draws bounding boxes +
labels as an overlay. Optionally runs YOLOv8-pose on the same frame and
draws keypoints — useful for the standard "is someone here?" use case.

This is intentionally *much* smaller than ``webcam.py``: no tracking,
no sitting/face heuristics, no GUI, no recording. Those features belong
to the standalone mode. The pipeline mode just needs:

1. accept a JPEG-decoded BGR frame,
2. run inference,
3. return ``(overlay_bgr, detected_classes)`` so the WebSocket client
   can ship the rendered frame back and emit ``detection`` JSON
   messages to the web server.

The class is intentionally tolerant of a missing ``ultralytics``
package — that way the rest of the pipeline still imports on hosts
without the heavy ML deps. Use :func:`load_yolo_processor` to wire one
up; if the import fails the function raises a clear RuntimeError so the
launcher can fall back to the stub.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2  # type: ignore
import numpy as np  # type: ignore


log = logging.getLogger("sentinelCam.worker.inference")


_PALETTE: tuple[tuple[int, int, int], ...] = (
    (66, 135, 245),  # orange-ish blue (BGR)
    (60, 179, 113),
    (255, 99, 132),
    (255, 206, 86),
    (153, 102, 255),
    (75, 192, 192),
    (255, 159, 64),
    (0, 200, 200),
)


# COCO-17 pose skeleton (matches the YOLOv8-pose keypoint order).
_POSE_PAIRS: tuple[tuple[int, int], ...] = (
    (5, 7), (7, 9),       # left arm
    (6, 8), (8, 10),      # right arm
    (11, 13), (13, 15),   # left leg
    (12, 14), (14, 16),   # right leg
    (5, 6),               # shoulders
    (11, 12),             # hips
    (5, 11), (6, 12),     # torso
    (0, 1), (0, 2),       # nose -> eyes
    (1, 3), (2, 4),       # eyes -> ears
)


@dataclass
class ProcessedFrame:
    """Return type for a frame processor.

    ``overlay`` is the BGR image to ship back to the web server. ``classes``
    is the set of distinct class labels detected on this frame; the client
    forwards them to the web via a ``{"type":"detection",...}`` JSON message
    so auto-recording can fire.
    """

    overlay: np.ndarray
    classes: list[str] = field(default_factory=list)


class YoloInference:
    """Wraps a YOLO detection model and (optionally) a pose model."""

    def __init__(
        self,
        model_path: str,
        *,
        pose_model_path: Optional[str] = None,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        device: Optional[str] = None,
        half: bool = False,
    ) -> None:
        from ultralytics import YOLO  # type: ignore  # imported here so the
        # module can still be imported on hosts without ultralytics installed
        # (see load_yolo_processor for fallback).

        self._model = YOLO(model_path)
        self._pose_model = YOLO(pose_model_path) if pose_model_path else None
        self._conf = float(conf)
        self._iou = float(iou)
        self._imgsz = int(imgsz)
        self._device = device or _autodetect_device()
        self._half = bool(half) and self._device.startswith("cuda")

        log.info(
            "YOLO ready: model=%s pose=%s device=%s half=%s conf=%.2f",
            model_path, pose_model_path or "off", self._device, self._half, conf,
        )

    async def __call__(self, frame_bgr: np.ndarray, capture_ms: int) -> ProcessedFrame:
        # YOLO + cv2.drawing are CPU-bound, so push them off the event loop.
        return await asyncio.to_thread(self._run, frame_bgr, capture_ms)

    def _run(self, frame_bgr: np.ndarray, _capture_ms: int) -> ProcessedFrame:
        try:
            results = self._model.predict(
                frame_bgr,
                imgsz=self._imgsz,
                conf=self._conf,
                iou=self._iou,
                device=self._device,
                half=self._half,
                verbose=False,
            )
        except Exception:
            log.exception("YOLO predict failed; returning raw frame")
            return ProcessedFrame(overlay=frame_bgr, classes=[])

        result = results[0]
        boxes = result.boxes
        names = result.names if hasattr(result, "names") else self._model.names
        overlay = frame_bgr.copy()
        seen: set[str] = set()

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
                label = self._class_label(names, int(cls_id))
                seen.add(label)
                self._draw_box(overlay, int(x1), int(y1), int(x2), int(y2),
                               label=f"{label} {conf:.2f}", color_seed=int(cls_id))

        # Optional pose overlay — only fires when a person is in the frame
        # to keep us off the GPU when nobody is around.
        if self._pose_model is not None and "person" in seen:
            try:
                pres = self._pose_model.predict(
                    frame_bgr,
                    imgsz=self._imgsz,
                    conf=max(self._conf, 0.15),
                    device=self._device,
                    half=self._half,
                    verbose=False,
                )[0]
            except Exception:
                log.exception("YOLO pose predict failed; skipping skeleton")
            else:
                kpts = getattr(pres, "keypoints", None)
                if kpts is not None and getattr(kpts, "xy", None) is not None:
                    xy = kpts.xy.cpu().numpy()         # [N, 17, 2]
                    confs = kpts.conf.cpu().numpy() if getattr(kpts, "conf", None) is not None else None
                    for i in range(len(xy)):
                        self._draw_skeleton(
                            overlay, xy[i],
                            confs[i] if confs is not None else None,
                        )

        return ProcessedFrame(overlay=overlay, classes=sorted(seen))

    @staticmethod
    def _class_label(names, cls_id: int) -> str:
        if isinstance(names, dict):
            value = names.get(cls_id) or names.get(str(cls_id))
            return str(value) if value is not None else str(cls_id)
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return str(names[cls_id])
        return str(cls_id)

    @staticmethod
    def _draw_box(
        img: np.ndarray, x1: int, y1: int, x2: int, y2: int, *,
        label: str, color_seed: int,
    ) -> None:
        color = _PALETTE[color_seed % len(_PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        bg_y1 = max(0, y1 - text_h - baseline - 4)
        cv2.rectangle(img, (x1, bg_y1), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(
            img, label, (x1 + 3, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    @staticmethod
    def _draw_skeleton(img: np.ndarray, kpts_xy: np.ndarray, kpts_conf: Optional[np.ndarray]) -> None:
        threshold = 0.35
        for a, b in _POSE_PAIRS:
            if kpts_conf is not None and (kpts_conf[a] < threshold or kpts_conf[b] < threshold):
                continue
            xa, ya = int(kpts_xy[a, 0]), int(kpts_xy[a, 1])
            xb, yb = int(kpts_xy[b, 0]), int(kpts_xy[b, 1])
            if xa == 0 and ya == 0:
                continue
            if xb == 0 and yb == 0:
                continue
            cv2.line(img, (xa, ya), (xb, yb), (0, 255, 255), 2, cv2.LINE_AA)
        for idx, (x, y) in enumerate(kpts_xy):
            if kpts_conf is not None and kpts_conf[idx] < threshold:
                continue
            if x == 0 and y == 0:
                continue
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), -1, cv2.LINE_AA)


def _autodetect_device() -> str:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_yolo_processor() -> Optional[YoloInference]:
    """Build a YOLO processor from environment variables.

    Returns ``None`` if ``WORKER_YOLO_MODEL`` is not set so the launcher
    can fall back to the stub overlay. Raises :class:`RuntimeError` only
    when the env asked for YOLO but the dependencies are missing — that
    signals a misconfiguration the operator wants to see.
    """
    model_path = (os.environ.get("WORKER_YOLO_MODEL") or "").strip()
    if not model_path:
        return None
    pose_path = (os.environ.get("WORKER_YOLO_POSE_MODEL") or "").strip() or None

    # Ultralytics writes auto-downloaded weights to the current working
    # directory. Containers with a read-only root FS need that to be a
    # writable, ideally persistent, location so the download survives
    # restarts and doesn't crash on read-only /app.
    cache_dir = (os.environ.get("WORKER_YOLO_CACHE_DIR") or "").strip()
    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/yolo-models")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        os.chdir(cache_dir)
    except OSError as exc:
        log.warning("could not chdir to cache dir %s: %s", cache_dir, exc)

    try:
        return YoloInference(
            model_path,
            pose_model_path=pose_path,
            conf=float(os.environ.get("WORKER_YOLO_CONF", "0.35")),
            iou=float(os.environ.get("WORKER_YOLO_IOU", "0.5")),
            imgsz=int(os.environ.get("WORKER_YOLO_IMGSZ", "640")),
            device=(os.environ.get("WORKER_YOLO_DEVICE") or "").strip() or None,
            half=os.environ.get("WORKER_YOLO_HALF", "").strip().lower() in ("1", "true", "yes"),
        )
    except ImportError as exc:
        raise RuntimeError(
            "WORKER_YOLO_MODEL is set but ultralytics is not installed. "
            "Install with: pip install ultralytics torch torchvision"
        ) from exc
